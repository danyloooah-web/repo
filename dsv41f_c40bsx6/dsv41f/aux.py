"""Lever AUX (levers.AUX, gen_aux.py; kernel in the CUDA unit eg_gather: auxm::mean_kernel): the DSPARK draft's
per-target-layer aux mean (stock's `aux.mean(dim=1)` over the 4 mHC streams of the residual [rows, 4, hidden] bf16,
layers 37 / 38 / 39, on lever EGQ's aux side stream) as a small fixed-grid kernel (levers.AUX_LAUNCH: 16..32 CTAs)
instead of torch's reduce_kernel<MeanOps> (~1 thread an output chunk: 480..1440 CTAs that flood every SM at the layer
start and delay the main chain's mHC stats partial and wq_b; c29 profiles, L38 vs L35).

Exactness: torch 2.14's CUDA mean of a bf16 tensor over a strided 4-element dim is mean_kernel_impl<BFloat16, float>:
one thread reduces each output (values per thread 4 < 16: no block / grid split), thread_reduce_impl with vt0 = 4
accumulators started at ident 0 and combined in order, then MeanOps::project (acc * factor, factor = n_out / numel =
0.25 exactly) and c10::BFloat16(float) (cvt.rn.bf16.f32 on sm_80+):
    out = bf16_rn((((0 + a0) + (0 + a1)) + (0 + a2)) + (0 + a3)) * 0.25f)
The unit does exactly these IEEE fp32 operations as inline PTX without FTZ (the bundle builds with --use_fast_math,
whose -ftz would flush the subnormal inputs bf16 shares with fp32) and cvt.rn.bf16x2.f32 per pair. prepare() compares
the unit with stock's own call byte for byte at every served row count 1..AUX_MAX_ROWS (realistic, exponent-spread
2^-133..2^127, order-sensitive narrow-window, +-0 / inf / NaN rows; contiguous and row-padded layouts; twice for
determinism), agreed over the TP ranks; the first eager use per row count compares again against stock's call on the
real input (mismatch: stock's bytes returned and the lever off for good). Fail closed: any doubt keeps stock's call.

HONESTY: every decision is a function of static facts (levers, the rank-agreed prepare result, the row count, the
tensor layout), the same in eager and CUDA-graph forwards on both ranks (the eager first-use check only compares;
the residual is TP-replicated, so both ranks see the same input). Deterministic and batch-invariant. Nothing is
patched on a served module or class: lever EGQ's eg.aux_mean calls mean().
"""
import torch

from dsv41f import levers

HC = 4
_UNIT: list = []
_STATE = {"detail": "not prepared", "on": False, "cols": 0, "checked": set(), "calls": 0}


def unit():
    """The bundle's CUDA unit eg_gather with its aux_mean entry, or None."""
    if not _UNIT:
        mod = None
        try:
            import eg_gather as mod
        except ImportError as err:
            levers.announce("aux-missing", f"bundle CUDA unit eg_gather not importable ({err}): lever AUX off")
        if mod is not None and not callable(getattr(mod, "aux_mean", None)):
            levers.announce("aux-missing", "eg_gather has no aux_mean entry: lever AUX off")
            mod = None
        _UNIT.append(mod)
    return _UNIT[0]


def _agree(values):
    """MIN over the TP CPU group (gloo), elementwise (lever RE5's helper)."""
    from sglang.srt.distributed import get_tp_group

    t = torch.tensor(values, dtype=torch.int64)
    group = get_tp_group()
    if group.world_size > 1:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group)
    return t.tolist()


def grid(rows: int) -> int:
    for max_rows, g in levers.AUX_LAUNCH:
        if rows <= max_rows:
            return int(g)
    return int(levers.AUX_LAUNCH[-1][1])


def eligible(x, cols=None) -> bool:
    """The tested layouts only: [rows, 4, cols] bf16 on CUDA, unit column stride, the 4 streams back to back in a row
    (stride(1) == cols), row stride >= 4 * cols, 16-byte aligned rows / streams, 1..AUX_MAX_ROWS rows."""
    cols = _STATE["cols"] if cols is None else cols
    return (isinstance(x, torch.Tensor) and x.is_cuda and x.dim() == 3 and x.dtype == torch.bfloat16
            and x.shape[1] == HC and x.shape[2] == cols and cols > 0 and cols % 8 == 0
            and 1 <= x.shape[0] <= int(levers.AUX_MAX_ROWS) and x.stride(2) == 1 and x.stride(1) == cols
            and x.stride(0) >= HC * cols and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0)


def _launch(x, out, g=None):
    rows = int(x.shape[0])
    unit().aux_mean(x.data_ptr(), x.stride(0), x.stride(1), out.data_ptr(), rows, int(x.shape[2]),
                    grid(rows) if g is None else int(g), torch.cuda.current_stream(x.device).cuda_stream)


def run(x, g=None):
    """The unit's mean of an eligible x into a fresh [rows, cols] bf16 tensor on the current stream."""
    out = torch.empty(x.shape[0], x.shape[2], dtype=torch.bfloat16, device=x.device)
    _launch(x, out, g)
    return out


def _bits_spread(shape, g, dev, lo_exp=0, hi_exp=254):
    """bf16 bit patterns: random sign / mantissa, biased exponent uniform in [lo_exp, hi_exp] (0 = subnormal)."""
    n = 1
    for s in shape:
        n *= s
    sign = torch.randint(0, 2, (n,), generator=g, device=dev, dtype=torch.int32) << 15
    exp = torch.randint(lo_exp, hi_exp + 1, (n,), generator=g, device=dev, dtype=torch.int32) << 7
    man = torch.randint(0, 128, (n,), generator=g, device=dev, dtype=torch.int32)
    v = (sign | exp | man).to(torch.int32)
    v = torch.where(v >= 32768, v - 65536, v).to(torch.int16)
    return v.view(torch.bfloat16).view(*shape)


def make_input(rows, cols, fam, g, dev, pad=0):
    """Self-test residuals [rows, 4, cols] (row stride 4 * cols + pad): fam 0 realistic heavy-tailed, 1 exponent spread
    over the full range incl. subnormals, 2 order-sensitive (exponents in a narrow random window, cancellations),
    3 specials (+-0, all -0 groups, +-inf, inf - inf, NaN, max / min values) on a realistic base."""
    full = torch.empty(rows, HC * cols + pad, dtype=torch.bfloat16, device=dev)
    full.view(torch.int16).fill_(0x7FC1)   # padding a NaN pattern: a read of it would show
    x = full[:, :HC * cols].view(rows, HC, cols)
    if fam == 0:
        base = torch.randn(rows, HC, cols, generator=g, device=dev)
        colscale = torch.exp(torch.randn(cols, generator=g, device=dev) * 2.0)
        x.copy_((base * colscale).to(torch.bfloat16))
    elif fam == 1:
        x.copy_(_bits_spread((rows, HC, cols), g, dev))
    elif fam == 2:
        c = int(torch.randint(10, 240, (1,), generator=g, device=dev).item())
        v = _bits_spread((rows, HC, cols), g, dev, max(0, c - 20), min(254, c + 20)).float()
        flip = torch.rand(rows, cols, generator=g, device=dev) < 0.3
        v[:, 1] = torch.where(flip, -v[:, 0] * (1 + torch.rand(rows, cols, generator=g, device=dev) * 2 ** -8),
                              v[:, 1])
        x.copy_(v.to(torch.bfloat16))
    else:
        x.copy_(torch.randn(rows, HC, cols, generator=g, device=dev).to(torch.bfloat16))
        sp = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 3.3895e38, -3.3895e38, 9.2e-41,
                           -9.2e-41, 1.1755e-38, 2 ** -133], device=dev).to(torch.bfloat16)
        flat = x.reshape(-1) if x.is_contiguous() else None
        if flat is None:
            m = torch.randint(0, sp.numel(), (rows, HC, cols), generator=g, device=dev)
            keep = torch.rand(rows, HC, cols, generator=g, device=dev) < 0.9
            x.copy_(torch.where(keep, x, sp[m]))
        else:
            idx = torch.randint(0, flat.numel(), (max(16, flat.numel() // 5),), generator=g, device=dev)
            flat[idx] = sp[torch.randint(0, sp.numel(), idx.shape, generator=g, device=dev)]
        x[0].fill_(-0.0)
        if rows > 1:
            x[rows - 1, :, : cols // 2].fill_(0.0)
            x[rows - 1, 0, cols // 2:].fill_(-0.0)
            x[rows - 1, 1:, cols // 2:].fill_(0.0)
    return x


def same(x, g=None):
    """'' when the unit equals stock's x.mean(dim=1) byte for byte (and twice the same), else what differs."""
    want = x.mean(dim=1)
    got = [run(x, g), run(x, g)]
    torch.cuda.synchronize(x.device)
    if not torch.equal(got[0].view(torch.int16), got[1].view(torch.int16)):
        return "not deterministic"
    if not torch.equal(want.view(torch.int16), got[0].view(torch.int16)):
        bad = int((want.view(torch.int16) != got[0].view(torch.int16)).sum())
        return f"{bad} of {want.numel()} outputs differ"
    return ""


def _self_test(dev, cols):
    g = torch.Generator(device=dev).manual_seed(int(levers.AUX_SEED))
    n = 0
    for rows in range(1, int(levers.AUX_MAX_ROWS) + 1):
        fams = range(4) if rows in (1, 2, 8, 48, 49, 96, 127, 144, 384) else (rows % 4,)
        for fam in fams:
            pad = 8 * (rows % 3 == 0) if fam != 3 else 0
            why = same(make_input(rows, cols, fam, g, dev, pad))
            n += 1
            if why:
                return False, f"MISMATCH at {rows} rows fam {fam} pad {pad}: {why}"
    return n > 0, f"{n} cases at rows 1..{levers.AUX_MAX_ROWS}: bit-identical to stock's mean and deterministic"


def prepare(model=None) -> str:
    """Once, outside any capture, on every rank (the agreement is collective)."""
    cfg = getattr(model, "config", None)
    cols = int(getattr(cfg, "hidden_size", 0) or 0)
    ok, detail = False, "off"
    if levers.AUX:
        try:
            if unit() is None:
                detail = "CUDA unit eg_gather / aux_mean unavailable"
            elif not torch.cuda.is_available() or cols <= 0 or cols % 8:
                detail = f"no CUDA device / hidden size {cols}"
            else:
                ok, detail = _self_test(torch.device("cuda", torch.cuda.current_device()), cols)
        except Exception as err:  # noqa: BLE001 - any failure keeps stock's call everywhere
            ok, detail = False, f"{type(err).__name__}: {str(err)[:160]}"
    agreed = bool(_agree([int(ok)])[0])
    _STATE["on"] = agreed
    _STATE["cols"] = cols if agreed else 0
    _STATE["detail"] = (f"{'ON' if agreed else 'OFF'} (prepare on this rank: {detail}; rank-agreed); DSPARK aux means "
                        f"of 1..{levers.AUX_MAX_ROWS} rows x {cols} on eg_gather's aux_mean at grids "
                        f"{levers.AUX_LAUNCH} (stock: reduce_kernel<MeanOps>, ~1 CTA a 128 output chunks); "
                        f"first eager use per row count checked vs stock; exact")
    return _STATE["detail"]


def mean(x):
    """Stock's `x.mean(dim=1)` bytes: the unit's small-grid mean where armed and eligible, stock's call otherwise."""
    if not _STATE["on"] or not eligible(x):
        return x.mean(dim=1)
    rows = int(x.shape[0])
    out = run(x)
    if rows not in _STATE["checked"] and not torch.cuda.is_current_stream_capturing():
        want = x.mean(dim=1)
        torch.cuda.current_stream(x.device).synchronize()
        _STATE["checked"].add(rows)
        if not torch.equal(want.view(torch.int16), out.view(torch.int16)):
            _STATE["on"] = False
            levers.announce("aux-mismatch", f"lever AUX: first eager use at {rows} rows differs from stock: OFF")
            return want
        if levers.first("aux-on"):
            levers.announce("aux-on", f"lever AUX: first eager use at {rows} rows bit-identical to stock (grid "
                            f"{grid(rows)})")
    _STATE["calls"] += 1
    return out


def describe() -> str:
    return _STATE["detail"]
