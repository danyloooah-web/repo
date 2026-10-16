"""Lever PWS on the fork (levers.PWS; = lever DB's kits/db.py re-based on c29): our layer replicas' dense MXFP8 linears
(wo_b, wq_b, the indexer's wq_b) at decode / verify sizes launch their GEMM as DbwDenseGemmKernel (pws_gemm.py --
FlashInfer 0.7.1's CuTe-DSL persistent block-scaled GEMM whose TMA warp issues the STATIC WEIGHT operand's (A + SFA
under swap-AB) first pipeline stages before griddepcontrol.wait) after FlashInfer's MXFP8 quantize with its PDL
trigger at its entry (pws_quant.py), on the tactic stock / lever DT would launch.

Why: FlashInfer's kernel waits on griddepcontrol.wait before its TMA producer issues anything, and its predecessors
release their dependents at their end, so every dense GEMM of a layer starts its weight stream cold from HBM only
after its predecessor's grid completed. The weight is a model parameter no kernel of the forward writes; its first
stages can be in flight while the predecessor runs. On c29 wq_b's predecessor (lever QN, lever SCH's early trigger in
decode / verify capture) already triggers early; wo_b's is the quantize here.

Exactness: DbwDenseGemmKernel's stages hold the same bytes as stock's (same TMA atoms / partitions / smem stages /
mbarriers; only the issue time of the weight loads changes, and the activation loads stay after the wait), and its MMA
warp and epilogue are stock's code unchanged -> stock's bits for any split-1 tactic (one fp32 accumulator, K
ascending). The quantize's trigger point cannot change a byte (a dependent's own griddepcontrol.wait waits for the full
completion and memory flush of this grid). Enforced: the first EAGER call of every (N, K, tactic, call form) runs the
wrapped linear (lever DT's / stock's) and PWS on the same input and compares every output byte; a difference (or an
exception) turns that key off for the process; unchecked keys run the wrapped linear inside CUDA-graph capture
(sglang's warmup forwards before each capture are eager, so captured keys are checked).

Scope: rows levers.PWS_ROWS of shapes levers.PWS_SHAPES, no bias / extra kwargs, contiguous bf16 or (fp8 + 1-D u8
scale) input; never inside lever FL's eager-extend scope. The served model is not modified: PWS replaces
method.w8a8_mxfp8_linear on the replica Fp8LinearMethod copies lever DT made.
"""
import torch

from dsv41f import levers

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
U8 = torch.uint8
_GEMM_CACHE: dict = {}      # (device, compile key) -> (compiled kernel, max active clusters)
_KEY_OK: dict = {}          # (n, k, tactic, form) -> True (bytes == wrapped call) / False (off for the process)
_STATS = {"bound": 0, "checked": 0, "off": 0}


# ---------------------------------------------------------------------------------------------------- kernels
def quant(x, early: bool = True):
    """sglang's flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32) (backend cute-dsl): FlashInfer's
    mxfp8_quantize_cute_dsl, from pws_quant (trigger at its entry) when `early`, else FlashInfer's own."""
    if early:
        from dsv41f import pws_quant as q
    else:
        from flashinfer.quantization.kernels import mxfp8_quantize as q
    return q.mxfp8_quantize_cute_dsl(x, is_sf_swizzled_layout=True, alignment=32, enable_pdl=None,
                                     is_sf_8x4_layout=False)


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
        raise ValueError(f"lever PWS: split-K tactic {tactic}")
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
            levers.announce(f"pws-{n}x{k}-{tactic}-{form}",
                            f"lever PWS N={n} K={k} tactic {tactic} {form} first at {m} rows: bytes "
                            f"{'== wrapped: on' if ok else '!= wrapped: OFF'}{err}")
            return ref
        if not ok:
            return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=None)
        return _run(input, weight, weight_scale, input_scale, tactic, form)

    linear.inner = inner   # lever FG's _served_partial walks .stock / .inner down to stock's partial
    return linear


def bind(replica) -> int:
    """Wrap the w8a8_mxfp8_linear of every replica linear lever DT made private (dsv4f_dt) in PWS's; returns how many."""
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
        if getattr(mod, "dsv4f_pws", False) or method is None or w is None or tuple(w.shape) not in shapes:
            continue
        method.w8a8_mxfp8_linear = _make_linear(method.w8a8_mxfp8_linear)
        mod.dsv4f_pws = True
        count += 1
    _STATS["bound"] += count
    return count


def describe(layers) -> str:
    n = sum(int(getattr(r, "dsv4f_pws", 0) or 0) for r in layers)
    return (f"{'on' if levers.PWS else 'off'}: {n} dense MXFP8 linears on {sum(bool(getattr(r, 'dsv4f_pws', 0)) for r in layers)} "
            f"layer replicas, rows {levers.PWS_ROWS}, shapes {sorted(levers.PWS_SHAPES)}, early quantize "
            f"{levers.PWS_EARLY_QUANT}, PWS GEMM call forms {levers.PWS_GEMM_FORMS} (weight stages before the wait "
            f"{levers.PWS_PRE_STAGES}, L2 blocks {levers.PWS_L2_BLOCKS}, early GEMM trigger "
            f"{levers.PWS_EARLY_GEMM_TRIGGER}) (bytes checked per key at first eager use)")
