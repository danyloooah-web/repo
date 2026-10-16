"""Lever MHC (levers.MHC, CUDA unit mhc_unit, gen_mhc.py): the mHC stats PARTIAL of our layer replicas' sublayers at
levers.MHC_MIN_ROWS..MHC_MAX_ROWS rows (lever SK's small path, <= 127 rows) on the unit -- 80..160 CTAs of a few us
with all loads of a warp in flight -- instead of stock's Triton _hc_mix_stats_partial_kernel (one stage: 240..640 CTAs
for 10..31 us in-engine); lever SK's reduce + sinkhorn (sk_unit, lever SKW's skw_unit in graphs: the same bytes) then
runs unchanged on the partials, which keep stock's layout ([80 slices, m, 24] + [80, m]).

Why: the stats run on the mHC stats stream beside the sublayer, and B300 runs one kernel's CTAs on an SM at a time:
while the 240..640-CTA partial is resident, a main-chain kernel that needs most SMs (wq_b's 128 CTAs, fmha's 144 CTAs
at 48 / 72 rows, the router, FC1's waves) waits for it (c27 profiles; tests/mhc_micro.py D: a 128-CTA stand-in
launched 1 us after the stats takes +1.4..+12.7 us with stock's partial). The unit's partial is wide but short.

Two kernels (levers.MHC_CONFIG):
  "x<MT>" EXACT (the default): stock's own arithmetic -- Triton 3.8's tf32x3 dot (three tf32 MMA passes with its
         kWidth = 2 operand layout and NaN select) and its row-sum-of-squares tree -- re-issued with more memory-level
         parallelism (mhc_unit.cu xpartial_kernel); bit-identical partials (tests/mhc_micro.py X). Served like lever
         SK's other exact work: inside sk._ours, so lever SK's first eager call per row count compares the whole
         triplet against stock bit for bit (mismatch: stock for good), and the prepare self-test compares the partials
         bit for bit on real weights; any difference turns the lever off on both ranks.
  "p<P>t<MT>" TOLERANCE (kept for comparison, not served by default): bf16x3 tensor-core dot products against the
         fork's split_bf16_hc_weight parts in a fixed order -- triplet within fp32 rounding of stock, not bitwise; it
         must never go through a bitwise check, so lever SK's _wrapped calls it directly (same kernel eager and in
         CUDA graphs; the decision is static: row count, layout, replica weight, prepare result).

HONESTY: every decision is a function of static facts (levers, the rank-agreed prepare result, the row count, the
tensor layout, which replica weight the call passes), identical in eager (prefill / extend, the eager live audit) and
CUDA-graph forwards on both ranks; the kernels are deterministic and batch-invariant. The weight lookup is built
once at prepare; nothing is patched on a served module or class (lever SK's bundle module calls serves() /
partial()).
"""
import torch

from dsv41f import levers

K, MIX, SLICES = 20480, 24, 80
_UNIT: list = []
_STATE = {"detail": "not prepared", "on": False, "cfg": -1, "exact": True, "parts": {}, "calls": 0}


def unit():
    if not _UNIT:
        mod = None
        try:
            import mhc_unit as mod
        except ImportError as err:
            levers.announce("mhc-missing", f"bundle CUDA unit mhc_unit not importable ({err}): lever MHC off")
        _UNIT.append(mod)
    return _UNIT[0]


def exact() -> bool:
    return str(levers.MHC_CONFIG).startswith("x")


def _agree(values):
    """MIN over the TP CPU group (gloo), elementwise (lever RE5's helper)."""
    from sglang.srt.distributed import get_tp_group

    t = torch.tensor(values, dtype=torch.int64)
    group = get_tp_group()
    if group.world_size > 1:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group)
    return t.tolist()


def _weights(hc):
    """(w0, w1, w2) for the unit: the fp32 mixing weight three times (exact kernel) or its bf16x3 parts (stock's
    HcSubLayer.bf16_parts, split with stock's formula if absent; tolerance kernel); None when the weight is unusable."""
    fn = getattr(hc, "fn", None)
    if (not isinstance(fn, torch.Tensor) or fn.dtype != torch.float32 or tuple(fn.shape) != (MIX, K) or not fn.is_cuda
            or not fn.is_contiguous() or fn.data_ptr() % 16):
        return None
    if exact():
        return (fn.data, fn.data, fn.data)
    parts = getattr(hc, "bf16_parts", None)
    if not parts or len(parts) != 3:
        from sglang.kernels.ops.layernorm.mhc import split_bf16_hc_weight

        parts = split_bf16_hc_weight(fn.data.contiguous())
    parts = tuple(p.contiguous() for p in parts)
    for p in parts:
        if p.dtype != torch.bfloat16 or tuple(p.shape) != (MIX, K) or p.data_ptr() % 16:
            return None
    return parts


def _stock_partial(x, fn):
    """Stock's Triton partial at stock's launch shape for these rows (the bits every shape gives)."""
    from sglang.kernels.ops.layernorm import mhc as smhc

    m = x.shape[0]
    pm = torch.empty((SLICES, m, MIX), dtype=torch.float32, device=x.device)
    ps = torch.empty((SLICES, m), dtype=torch.float32, device=x.device)
    block_m = smhc._block_m_for(m)
    grid_m = smhc.triton.cdiv(m, block_m)
    spc, warps = smhc._stats_launch_shape(m, grid_m)
    smhc._hc_mix_stats_partial_kernel[(grid_m, SLICES // spc)](
        x, fn, pm, ps, m, K, x.stride(0), fn.stride(0), MIX=MIX, MIX_PAD=32, NUM_SLICES=SLICES, BLOCK_M=block_m,
        BLOCK_K=smhc._HC_MIX_BLOCK_K, DOT_PRECISION=smhc._HC_MIX_DOT_PRECISION, SLICES_PER_CTA=spc, num_warps=warps,
        num_stages=smhc._HC_MIX_NUM_STAGES)
    return pm, ps


def _self_test(dev, fn, w):
    """Real weights of one sublayer, synthetic residual rows (normal, outlier columns, zero / -0 rows, extreme and
    subnormal values): the unit's partials are deterministic and, for the exact kernel, bit-identical to stock's
    Triton partial at every tested row count (tolerance kernel: within levers.MHC_SELFTEST_ATOL relative)."""
    g = torch.Generator(device=dev).manual_seed(int(levers.MHC_SEED))
    ok, n, worst = True, 0, 0.0
    for m in (1, 5, 8, 9, 17, 48, 64, 65, 96, 127):
        if not (levers.MHC_MIN_ROWS <= m <= levers.MHC_MAX_ROWS):
            continue
        for fam in range(3):
            xf = torch.randn(m, K, device=dev, generator=g)
            if fam == 1:
                xf[:, torch.randint(0, K, (24,), device=dev, generator=g)] *= 60.0
            elif fam == 2:
                flat = xf.view(-1)
                idx = torch.randint(0, flat.numel(), (max(8, flat.numel() // 307),), device=dev, generator=g)
                vals = torch.tensor([0.0, -0.0, 1e-38, -3e-39, 6e4, -6e4, 3e38], device=dev)
                flat[idx] = vals[torch.randint(0, vals.numel(), idx.shape, device=dev, generator=g)]
                xf[0] = 0.0
                if m > 1:
                    xf[m - 1] = -0.0
            x = xf.to(torch.bfloat16).contiguous()
            got = []
            for _ in range(2):
                pm = torch.full((SLICES, m, MIX), float("nan"), dtype=torch.float32, device=dev)
                ps = torch.full((SLICES, m), float("nan"), dtype=torch.float32, device=dev)
                partial(x, w, pm, ps)
                got.append((pm, ps))
            want = _stock_partial(x, fn)
            torch.cuda.synchronize(dev)
            ok &= all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(got[0], got[1]))
            if exact():
                ok &= all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(got[0], want))
            else:
                d = float(((got[0][0] - want[0]).abs() / (want[0].abs() + 1.0)).max())
                worst = max(worst, d if d == d else float("inf"))
            n += 1
    if not exact():
        ok &= worst <= float(levers.MHC_SELFTEST_ATOL)
    return ok and n > 0, (f"{n} cases, partials {'bit-identical to stock' if exact() else f'max rel d {worst:.2e}'}"
                          f" and deterministic: {ok and n > 0}")


def prepare(layers=()) -> str:
    """Once, outside any capture, on every rank (the agreement is collective): the weight lookup of every replica
    sublayer (keyed by the tensor its small-path stats call passes: hc.fn), the unit's init and a self-test, agreed
    over the TP CPU group (one rank failing turns it off on all)."""
    parts = {}
    first = None
    for layer in layers:
        for hc in (getattr(layer, "attn_hc", None), getattr(layer, "ffn_hc", None)):
            if hc is None:
                continue
            w = _weights(hc)
            if w is None:
                continue
            parts[int(hc.fn.data_ptr())] = w
            if first is None:
                first = (hc.fn.data, w)
    ok, detail = False, "off"
    if levers.MHC:
        try:
            mod = unit()
            if mod is None:
                detail = "CUDA unit mhc_unit unavailable"
            elif not torch.cuda.is_available() or first is None:
                detail = "no CUDA device / no replica mixing weights"
            else:
                mod.init()
                _STATE["cfg"] = int(mod.config_index(str(levers.MHC_CONFIG)))
                _STATE["exact"] = exact()
                if _STATE["cfg"] < 0:
                    detail = f"unknown config {levers.MHC_CONFIG}"
                else:
                    ok, detail = _self_test(torch.device("cuda", torch.cuda.current_device()), *first)
        except Exception as err:  # noqa: BLE001 - any failure keeps stock's path everywhere
            ok, detail = False, f"{type(err).__name__}: {str(err)[:160]}"
    agreed = bool(_agree([int(ok)])[0])
    _STATE["parts"] = parts if agreed else {}
    _STATE["on"] = agreed
    kind = ("EXACT (stock's tf32x3 partial bits; lever SK's first-call bitwise check per row count applies)"
            if exact() else "TOLERANCE (bf16x3 tensor-core dot products in a fixed order; never bitwise-checked)")
    _STATE["detail"] = (f"{'ON' if agreed else 'OFF'} (prepare on this rank: {detail}; rank-agreed); rows "
                        f"{levers.MHC_MIN_ROWS}..{levers.MHC_MAX_ROWS}, config {levers.MHC_CONFIG} {kind}, "
                        f"{len(parts)} weight keys; the stats partial of small-path calls in the band on the unit, "
                        f"eager and CUDA graph alike, then lever SK's reduce + sinkhorn unchanged")
    return _STATE["detail"]


def partial(x, w, part_mix, part_sq):
    """The unit's partial into stock's layout (part_mix [80, m, 24], part_sq [80, m]) on the current stream."""
    unit().run(_STATE["cfg"], x.data_ptr(), x.stride(0), w[0].data_ptr(), w[1].data_ptr(), w[2].data_ptr(),
               part_mix.data_ptr(), part_sq.data_ptr(), int(x.shape[0]), torch.cuda.current_stream(x.device).cuda_stream)


def serves(x_flat, key):
    """The replica weight's unit operands when this call takes the unit's partial, else None. Static facts only."""
    if not _STATE["on"] or not isinstance(key, torch.Tensor):
        return None
    m = int(x_flat.shape[0])
    if (m < int(levers.MHC_MIN_ROWS) or m > int(levers.MHC_MAX_ROWS) or x_flat.dim() != 2 or x_flat.shape[1] != K
            or x_flat.dtype != torch.bfloat16 or not x_flat.is_cuda or x_flat.stride(1) != 1
            or x_flat.stride(0) % 8 or x_flat.data_ptr() % 16):
        return None
    w = _STATE["parts"].get(int(key.data_ptr()))
    if w is None or w[0].device != x_flat.device:
        return None
    _STATE["calls"] += 1
    return w
