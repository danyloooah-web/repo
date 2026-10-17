"""Lever PWN on the fork (levers.PWS + levers.PWN; = lever PWS (kits/pws.py) with its MXFP8 quantize NATIVE): our layer
replicas' dense MXFP8 linears (wo_b, wq_b, the indexer's wq_b) at decode / verify sizes launch their GEMM as
DbwDenseGemmKernel (pws_gemm.py -- FlashInfer 0.7.1's CuTe-DSL persistent block-scaled GEMM, compiled through
FlashInfer's own installed _compile_block_scaled_gemm, whose TMA warp issues the STATIC WEIGHT operand's (A + SFA under
swap-AB) first pipeline stages before griddepcontrol.wait) after an MXFP8 quantize whose PDL trigger is at its entry, on
the tactic stock / lever DT would launch.

The quantize (lever PWN): pwn::quant_kernel in the bundle copy of the CUDA unit gq_unit (gq_unit.pwn) -- FlashInfer
0.7.1's MXFP8QuantizeSwizzledKernel (bf16, 4 threads a 32-column scale block, 128x4 swizzled UE8M0 scales, padded rows'
scale bytes zero) op for op, at FlashInfer's own launch (its threads a CTA / rows a CTA iteration, grid min(ceil(padded
rows / rows), 4 x SMs), PDL launch), with griddepcontrol.launch_dependents at its entry and griddepcontrol.wait before
its first global access. Lever PWS ran FlashInfer's CuTe-DSL module with that one edit (pws_quant.py), which the
validator's intake check refuses (bundle cute.compile with a string-literal option); PWN has no bundle cute.compile.

Why: FlashInfer's GEMM waits on griddepcontrol.wait before its TMA producer issues anything, and its predecessors
release their dependents at their end, so every dense GEMM of a layer starts its weight stream cold from HBM only
after its predecessor's grid completed. The weight is a model parameter no kernel of the forward writes; its first
stages can be in flight while the predecessor runs. On c29+ wq_b's predecessor (lever QN, lever SCH's early trigger in
decode / verify capture) already triggers early; wo_b's is the quantize here.

Exactness: DbwDenseGemmKernel's stages hold the same bytes as stock's (same TMA atoms / partitions / smem stages /
mbarriers; only the issue time of the weight loads changes, and the activation loads stay after the wait), and its MMA
warp and epilogue are stock's code unchanged -> stock's bits for any split-1 tactic. The quantize's trigger point cannot
change a byte (a dependent's own griddepcontrol.wait waits for the full completion and memory flush of this grid), and
its arithmetic is FlashInfer's. Enforced, fail closed: (1) the pinned FlashInfer / SGLang sources (lever GQ's pins +
_compute_optimal_warps) and FlashInfer's own launch shape for each K (MXFP8QuantizeSwizzledKernel's warps / rows) must
equal the unit's, and a self-test (realistic, extreme, +-0, subnormal, inf, NaN, strided rows at 1..384 rows) must give
stock's bytes, else the native quantize stays off (FlashInfer's own, stock trigger); (2) the first EAGER call of every
(rows, K, row stride) compares the unit's q and every scale byte with stock's flashinfer_mxfp8_quantize; a difference
turns that key off for the process; an unchecked key inside CUDA-graph capture runs FlashInfer's own; (3) as PWS: the
first EAGER call of every (N, K, tactic, call form) runs the wrapped linear (lever DT's / stock's) and PWN on the same
input and compares every output byte; a difference (or an exception) turns that key off for the process; unchecked
keys run the wrapped linear inside CUDA-graph capture (sglang's warmup forwards before each capture are eager).

Scope: rows levers.PWS_ROWS of shapes levers.PWS_SHAPES, no bias / extra kwargs, contiguous bf16 or (fp8 + 1-D u8
scale) input; never inside lever FL's eager-extend scope. The served model is not modified: PWN replaces
method.w8a8_mxfp8_linear on the replica Fp8LinearMethod copies lever DT made.
"""
import sys

import torch

from dsv41f import levers

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
U8 = torch.uint8
QK = (4096, 1280)           # the unit's widths: wo_b's input, wq_b's / the indexer wq_b's input
_GEMM_CACHE: dict = {}      # (device, compile key) -> (compiled kernel, max active clusters)
_KEY_OK: dict = {}          # (n, k, tactic, form) -> True (bytes == wrapped call) / False (off for the process)
_Q_OK: dict = {}            # (rows, k, row stride) -> True (unit bytes == stock quantize) / False
_Q: dict = {}               # "state": None (not tried) / True / False; "detail"; k -> (threads, rows a CTA); fns
_STATS = {"bound": 0, "checked": 0, "off": 0, "q_checked": 0, "q_off": 0, "q_unit": 0}
# lever GQ's pins (the FlashInfer / SGLang sources the quantize's bytes rely on) + the launch shape helper
_EXTRA_PINS = {
    ("flashinfer.quantization.kernels.mxfp8_quantize", "_compute_optimal_warps"): "9b1a647b5d24ead2",
}


# ---------------------------------------------------------------------------------------------------- quantize
def _stock_quant(x):
    from sglang.srt.layers.quantization import fp8_utils

    return fp8_utils.flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)


def _fi_quant(x):
    """FlashInfer's own mxfp8_quantize_cute_dsl (stock trigger), as lever PWS's non-early path."""
    from flashinfer.quantization.kernels import mxfp8_quantize as q

    return q.mxfp8_quantize_cute_dsl(x, is_sf_swizzled_layout=True, alignment=32, enable_pdl=None,
                                     is_sf_8x4_layout=False)


def _launch(x, m, k, sx):
    """The unit's q [m, k] e4m3 + scales [padded_m * k / 32] u8 of bf16 rows x (row stride sx elements)."""
    rows = _Q[k][1]
    padded = (m + 127) // 128 * 128
    grid = min((padded + rows - 1) // rows, _Q["sms"][x.get_device()] * 4)
    q = torch.empty((m, k), dtype=U8, device=x.device)
    s = torch.empty((padded * (k // 32),), dtype=U8, device=x.device)
    _Q["run"](x.data_ptr(), sx, q.data_ptr(), s.data_ptr(), m, k, grid, _Q["raw"](x.get_device()))
    return q.view(FP8), s


def _diff(x, m, k, sx):
    """'' when the unit's q / scales equal stock's flashinfer_mxfp8_quantize bytes for x, else what differs."""
    q_ref, s_ref = _stock_quant(x if x.is_contiguous() else x.contiguous())
    q, s = _launch(x, m, k, sx)
    if q_ref.dtype is not FP8 or tuple(q_ref.shape) != (m, k) or q_ref.stride() != q.stride():
        return f"q layout {q_ref.dtype} {tuple(q_ref.shape)} {q_ref.stride()}"
    if not torch.equal(q.view(U8), q_ref.view(U8)):
        return "q bytes"
    if s_ref.dtype is not U8 or s_ref.dim() != 1 or s_ref.numel() != s.numel() or not torch.equal(s, s_ref):
        return f"scales ({s_ref.dtype} {tuple(s_ref.shape)} vs {tuple(s.shape)})"
    return ""


def selftest_inputs(k: int, seed: int = 20261016):
    """Self-test rows: realistic (normal, heavy-tailed), extreme magnitudes, zero / subnormal / +-0 / inf / NaN rows,
    amax at the subnormal-scale edge (2^-118..2^-117), strided rows; row counts across the 1..384 band."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + k)
    out = []
    for m in (1, 7, 17, 64, 127, 128, 129, 200, 384):
        x = torch.randn((m, k), generator=g, device="cuda", dtype=torch.float32)
        x = x * torch.exp(torch.randn((m, 1), generator=g, device="cuda") * 2.0)
        x[0, 3] = x[0, 3] * 1e-30
        if m > 6:
            x[1, ::7] = 0.0
            x[2, :32] = -0.0
            x[3, 64:96] = 3.0e38
            x[4, 5] = 1e-41
            x[5, ::33] = 448.0 * 2.0 ** 100
            x[6] = torch.randint(-3, 4, (k,), generator=g, device="cuda").float() * 2.0 ** -126
        if m > 16:
            x[7] = torch.sign(torch.randn((k,), generator=g, device="cuda")) * torch.exp2(
                -133 + 6 * torch.rand((k,), generator=g, device="cuda"))
            x[8] = torch.exp2(-118 + torch.rand((k,), generator=g, device="cuda")) * torch.sign(
                torch.randn((k,), generator=g, device="cuda"))
            x[9, 0] = float("inf")
            x[10, 40] = float("-inf")
            x[11, 70] = float("nan")
            x[12, :32] = float("nan")
            x[13] = -0.0
            x[14, 100] = float("nan")
            x[14, 101] = float("inf")
            x[15] = 0.0
        out.append(x.to(BF16))
    wide = torch.randn((72, k + 64), generator=g, device="cuda").to(BF16)[:, 32:32 + k]   # strided rows
    out.append(wide)
    return out


def _ready(dev_index: int) -> bool:
    """Pins, FlashInfer's launch shape per K and the self-test, once per process (fail closed)."""
    st = _Q.get("state")
    if st is not None:
        return st
    try:
        from dsv41f import gq, hq
        from flashinfer.quantization.kernels import mxfp8_quantize as fq
        import cutlass

        mods = gq._modules()
        cache: dict = {}
        pins = dict(gq.PINNED)
        pins.update(_EXTRA_PINS)
        bad = [f"{mod}.{qual}" for (mod, qual), want in pins.items()
               if hq.source_hash(mods[mod], qual, cache) != want]
        if bad:
            raise RuntimeError("stock source differs from the pinned read: " + ", ".join(bad))
        import gq_unit

        raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        _Q.update(run=gq_unit.pwn, raw=raw if raw is not None else (lambda idx: torch.cuda.current_stream(idx).cuda_stream),
                  sms={})
        for d in range(torch.cuda.device_count()):
            _Q["sms"][d] = torch.cuda.get_device_properties(d).multi_processor_count
        for k in QK:
            threads, rows = (int(v) for v in gq_unit.pwn_cfg(k))
            ko = fq.MXFP8QuantizeSwizzledKernel(cutlass.BFloat16, k, True, False, sf_layout=fq.SF_LAYOUT_128x4,
                                                small_blocks=False)
            if (ko.warps_per_block * 32, ko.rows_per_block, ko.needs_col_loop) != (threads, rows, False):
                raise RuntimeError(f"K {k}: FlashInfer's launch {ko.warps_per_block * 32} threads / "
                                   f"{ko.rows_per_block} rows != the unit's {threads} / {rows}")
            _Q[k] = (threads, rows)
        n = 0
        for k in QK:
            for x in selftest_inputs(k):
                what = _diff(x, x.shape[0], k, x.stride(0))
                n += 1
                if what:
                    raise RuntimeError(f"self-test K {k} at {x.shape[0]} rows (stride {x.stride(0)}): {what} != stock")
        torch.cuda.synchronize()
        _Q["state"] = True
        _Q["detail"] = (f"on: native quantize (gq_unit.pwn, entry trigger) at K {QK}, launches "
                        f"{[_Q[k] for k in QK]} (threads, rows a CTA) == FlashInfer's; self-test {n} inputs == stock")
    except Exception as e:  # noqa: BLE001 - any missing / changed internal: FlashInfer's own quantize
        _Q["state"] = False
        _Q["detail"] = f"off ({type(e).__name__}: {str(e)[:300]})"
    levers.announce("pwn-quant", f"lever PWN quantize: {_Q['detail']}")
    return _Q["state"]


def _unit_quant(x):
    """The unit's (q, scales) of x (stock's bytes, entry trigger), or None (FlashInfer's own quantize)."""
    if not getattr(levers, "PWN", False) or x.dtype is not BF16 or x.dim() != 2 or not x.is_cuda:
        return None
    m, k = x.shape
    if k not in QK or m < 1 or m * (k // 32) >= 65536 or x.stride(1) != 1 or x.stride(0) % 8 or x.stride(0) < k \
            or x.data_ptr() % 16:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    st = _Q.get("state")
    if st is None:
        if capturing:
            return None
        st = _ready(x.get_device())
    if not st:
        return None
    key = (m, k, x.stride(0))
    ok = _Q_OK.get(key)
    if ok is None:
        if capturing:
            return None
        try:
            what = _diff(x, m, k, x.stride(0))
        except Exception as e:  # noqa: BLE001
            what = f"{type(e).__name__}: {str(e)[:160]}"
        ok = not what
        _Q_OK[key] = ok
        _STATS["q_checked"] += 1
        _STATS["q_off"] += 0 if ok else 1
        if not ok:
            levers.announce(f"pwn-q-{m}x{k}-{x.stride(0)}",
                            f"lever PWN quantize {m} x {k} (stride {x.stride(0)}): bytes != stock: OFF ({what})")
        return None   # the checking call returns FlashInfer's own (the caller's next calls take the unit)
    if not ok:
        return None
    _STATS["q_unit"] += 1
    return _launch(x, m, k, x.stride(0))


def quant(x, early: bool = True):
    """sglang's flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32) (backend cute-dsl) bytes: the
    unit's entry-trigger quantize when `early` (and checked), else FlashInfer's own mxfp8_quantize_cute_dsl."""
    if early:
        got = _unit_quant(x)
        if got is not None:
            return got
    return _fi_quant(x)


# ---------------------------------------------------------------------------------------------------- kernels
def runner_tactic(m: int, n: int, tactic):
    """The tactic CuteDSLMxfp8GemmRunner.forward launches for a split-1 `tactic` (its narrow-tile re-check)."""
    from flashinfer.gemm import gemm_base
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 import Sm100BlockScaledPersistentDenseGemmKernel as P
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk import Sm100BlockScaledSplitKGemmKernel as S

    if tactic is None or tactic == -1:
        tactic = fallback_tactic(m)
    tiler, cluster, swap, pf, split = tactic
    if split != 1:
        return None
    if not P.narrow_tile_ok(tiler[1], m if swap else n):
        tactic = fallback_tactic(m)
    _ = gemm_base
    return (tuple(tactic[0]), tuple(tactic[1]), bool(tactic[2]), bool(tactic[3]), int(tactic[4]))


def fallback_tactic(m: int):
    """CuteDSLMxfp8GemmRunner.forward's untuned default (contiguous output)."""
    from flashinfer.gemm.gemm_base import _SM100_DEFAULT_CLUSTER_SHAPE_MN, _SM100_DEFAULT_MMA_TILER_MN
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk import Sm100BlockScaledSplitKGemmKernel as S

    if S.supports_m(m):
        return (tuple(S.mma_tiler_mn_for_m(m)), (1, 1), True, False, 1)
    return (tuple(_SM100_DEFAULT_MMA_TILER_MN), tuple(_SM100_DEFAULT_CLUSTER_SHAPE_MN), False, False, 1)


def gemm(a, b, a_scale, b_scale, out_dtype, tactic, early_trigger: bool = False, w_first: bool = True,
         pre_stages: int = -1, l2_blocks: int = 0):
    """_cute_dsl_mm_mxfp8_with_tactic(a, b, a_scale, b_scale, out_dtype, tactic) for a split-1 tactic, on
    DbwDenseGemmKernel (`w_first`: the weight's first `pre_stages` stages (-1 = all) loaded and `l2_blocks` further
    k-blocks L2-prefetched before the wait; False = stock's load order, for tests). a: (m, k) fp8 activation,
    b: weight.t() (k, n), scales as the runner takes them. Returns (m, n) out_dtype."""
    import cutlass
    from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _compile_block_scaled_gemm, _prepare_alpha_for_launch

    from dsv41f.pws_gemm import DbwDenseGemmKernel

    m, real_k = a.shape
    n = b.shape[1]
    tiler, cluster, swap, pf, split = tactic
    if split != 1:
        raise ValueError(f"lever PWN: split-K tactic {tactic}")
    if swap:
        kernel_m, kernel_n = n, m
        ka, kb, ka_sf, kb_sf = b.T, a, b_scale, a_scale
        w_op = "a"
    else:
        kernel_m, kernel_n = m, n
        ka, kb, ka_sf, kb_sf = a, b.T, a_scale, b_scale
        w_op = "b"
    if not w_first:
        w_op, pre_stages, l2_blocks = "none", -1, 0
    sf_m = (kernel_m + 127) // 128
    sf_n = (kernel_n + 127) // 128
    sf_k = (real_k // 32 + 3) // 4
    key = ("pws", 32, tuple(tiler), tuple(cluster), bool(swap), bool(pf), True, out_dtype, 1, w_op,
           bool(early_trigger), int(pre_stages), int(l2_blocks))
    c_dtype = cutlass.BFloat16 if out_dtype == BF16 else cutlass.Float16
    compiled, _ = _compile_block_scaled_gemm(
        _GEMM_CACHE, key,
        lambda: DbwDenseGemmKernel(32, tuple(tiler), tuple(cluster), bool(pf), True, w_operand=w_op,
                                   early_trigger=bool(early_trigger), w_pre_stages=int(pre_stages),
                                   w_l2_blocks=int(l2_blocks)),
        ab_cutlass_dtype=cutlass.Float8E4M3FN, sf_dtype=cutlass.Float8E8M0FNU, c_cutlass_dtype=c_dtype,
        ab_assumed_align=16, cluster_shape_mn=tuple(cluster), swap_ab=bool(swap), sf_m=sf_m, sf_n=sf_n, sf_k=sf_k,
        batch_size=1, cluster_shape_k=1)
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    launch_out = out.as_strided(out.shape, (1, out.shape[0])) if swap else out
    compiled(ka, kb, launch_out, sf_m, sf_n, sf_k, ka_sf.data_ptr(), kb_sf.data_ptr(),
             _prepare_alpha_for_launch(None, a.device))
    return out


# ---------------------------------------------------------------------------------------------------- the linear
def served_tactic(m: int, n: int, k: int, dev):
    """The tactic the wrapped linear launches at m rows: lever DT's band tactic (band on or unchecked), else the fork
    pin (fp8_utils._mxfp8_decode_tactic), else the runner's default; after the runner's narrow-tile re-check. None =
    split-K or nothing PWS launches."""
    from sglang.srt.layers.quantization.fp8_utils import _mxfp8_decode_tactic

    from dsv41f import dt

    if levers.DT and levers.DT_TACTICS:
        band, tac = dt._band(m, n, k)
        if band is not None and dt._KEY_OK.get((n, k, band)) is not False:
            return runner_tactic(m, n, tuple(tac))
    pin = _mxfp8_decode_tactic(m, n, dev)
    return runner_tactic(m, n, tuple(pin) if pin is not None else None)


def _fl_scope() -> bool:
    from dsv41f import fl

    return fl._ON.get() is not None


def _same(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.element_size() == 2:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a, b)


def _forms(n: int, k: int):
    """The call forms of shape (n, k) that take PWS's GEMM (levers.PWS_GEMM_FORMS: {(N, K): forms})."""
    return levers.PWS_GEMM_FORMS.get((n, k), ())


def _run(input, weight, weight_scale, input_scale, tactic, form):
    input_2d = input.view(-1, input.shape[-1])
    if input_scale is None:
        q_input, x_scale = quant(input_2d, early=levers.PWS_EARLY_QUANT)
    else:
        q_input, x_scale = input_2d, input_scale
    out_dtype = input_2d.dtype if input_2d.dtype in (torch.float16, BF16, torch.float32) else BF16
    w_scale = weight_scale.t() if weight_scale.ndim == 2 else weight_scale
    if form in _forms(weight.shape[0], weight.shape[1]):
        out = gemm(q_input, weight.t(), x_scale, w_scale, out_dtype, tactic,
                   early_trigger=levers.PWS_EARLY_GEMM_TRIGGER, pre_stages=levers.PWS_PRE_STAGES,
                   l2_blocks=levers.PWS_L2_BLOCKS)
    else:   # stock's GEMM launch (the call form's quantize is PWS's)
        from sglang.srt.layers.quantization.fp8_utils import _cute_dsl_mm_mxfp8_with_tactic

        out = _cute_dsl_mm_mxfp8_with_tactic(q_input, weight.t(), x_scale, w_scale, out_dtype, tuple(tactic))
    return out.to(dtype=out_dtype).view(*input.shape[:-1], weight.shape[0])


def _make_linear(inner):
    lo, hi = levers.PWS_ROWS
    shapes = set(tuple(s) for s in levers.PWS_SHAPES)

    def linear(input, weight, weight_scale, input_scale=None, bias=None, **kwargs):
        if kwargs or bias is not None or not levers.PWS:
            return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias,
                         **kwargs)
        n, k = weight.shape
        m = input.numel() // input.shape[-1]
        if not (lo <= m <= hi) or (n, k) not in shapes or _fl_scope():
            return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias)
        if input_scale is None:
            if input.dtype is not BF16 or not input.is_contiguous():
                return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=None, bias=None)
            form = "quant"
        else:
            if (input.dtype is not FP8 or input_scale.dtype is not U8 or input_scale.dim() != 1
                    or not input.is_contiguous()):
                return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                             bias=None)
            form = "pre"
            if form not in _forms(n, k):   # nothing of PWS's in this call form: the wrapped call
                return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                             bias=None)
        if form == "quant" and not levers.PWS_EARLY_QUANT and form not in _forms(n, k):
            return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=None, bias=None)
        tactic = served_tactic(m, n, k, weight.device)
        if tactic is None:
            return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=None)
        key = (n, k, tactic, form)
        ok = _KEY_OK.get(key)
        if ok is None:
            if torch.cuda.is_current_stream_capturing():
                return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                             bias=None)
            ref = inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=None)
            try:
                got = _run(input, weight, weight_scale, input_scale, tactic, form)
                ok = _same(ref, got)
                err = ""
            except Exception as e:  # noqa: BLE001 - a launch PWS cannot make: the wrapped call from now on
                ok, err = False, f" ({type(e).__name__}: {str(e)[:160]})"
            _KEY_OK[key] = ok
            _STATS["checked"] += 1
            _STATS["off"] += 0 if ok else 1
            levers.announce(f"pwn-{n}x{k}-{tactic}-{form}",
                            f"lever PWN N={n} K={k} tactic {tactic} {form} first at {m} rows: bytes "
                            f"{'== wrapped: on' if ok else '!= wrapped: OFF'}{err}")
            return ref
        if not ok:
            return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=None)
        return _run(input, weight, weight_scale, input_scale, tactic, form)

    linear.inner = inner   # lever FG's _served_partial walks .stock / .inner down to stock's partial
    return linear


def bind(replica) -> int:
    """Wrap the w8a8_mxfp8_linear of every replica linear lever DT made private (dsv4f_dt) in PWN's; returns how many."""
    if not levers.PWS:
        return 0
    shapes = set(tuple(s) for s in levers.PWS_SHAPES)
    found = []

    def walk(module):
        for name, child in module._modules.items():
            # `_next_layer` is the following decoder layer (lever DT's walk skips it too): not this layer's linears
            if child is None or name == "_next_layer" or type(child).__name__ == "DeepseekV4DecoderLayer":
                continue
            if getattr(child, "dsv4f_dt", False) is True:      # lever DT's private replica linear
                found.append(child)
            else:
                walk(child)

    walk(replica)
    count = 0
    for mod in found:
        method = getattr(mod, "quant_method", None)
        w = mod._parameters.get("weight")
        if getattr(mod, "dsv4f_pwn", False) or method is None or w is None or tuple(w.shape) not in shapes:
            continue
        method.w8a8_mxfp8_linear = _make_linear(method.w8a8_mxfp8_linear)
        mod.dsv4f_pwn = True
        count += 1
    _STATS["bound"] += count
    return count


def describe(layers) -> str:
    n = sum(int(getattr(r, "dsv4f_pwn", 0) or 0) for r in layers)
    return (f"{'on' if levers.PWS else 'off'}: {n} dense MXFP8 linears on {sum(bool(getattr(r, 'dsv4f_pwn', 0)) for r in layers)} "
            f"layer replicas, rows {levers.PWS_ROWS}, shapes {sorted(levers.PWS_SHAPES)}, early quantize "
            f"{levers.PWS_EARLY_QUANT}, PWS GEMM call forms {levers.PWS_GEMM_FORMS} (weight stages before the wait "
            f"{levers.PWS_PRE_STAGES}, L2 blocks {levers.PWS_L2_BLOCKS}, early GEMM trigger "
            f"{levers.PWS_EARLY_GEMM_TRIGGER}) (bytes checked per key at first eager use); native entry-trigger "
            f"quantize {'on' if getattr(levers, 'PWN', False) else 'off'} at K {QK} (gq_unit.pwn; pins + FlashInfer's "
            f"launch + self-test at the first eager call, bytes checked per (rows, K, stride) at first eager use)")
