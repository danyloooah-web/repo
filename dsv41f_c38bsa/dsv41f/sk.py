"""Lever SK: the mHC stats reduce + sinkhorn of our layer replicas' decode / verify sublayers on the bundle CUDA unit
sk_unit (bit-identical to the fork's Triton _hc_mix_reduce_sinkhorn_kernel, partials fetched in one round trip).

install() wraps sglang.kernels.ops.layernorm.mhc.hc_mix_stats_sinkhorn (the 1..127-row Blackwell path of
deepseek_v4_mhc._mix_stats_impl, which imports it at call time) and hc_mix_stats_sinkhorn_deepgemm (128+ rows: two
deep_gemm tf32 prenorm GEMMs, then the reduce with the low part as residual); the wrappers run stock's own partial
launches and then the unit, only inside our model forward (forward() below sets the context) and only for row counts whose first
eager call (a warmup forward, outside graph capture) matched stock's triplet bit for bit; everything else is stock.

LEVER P1 (levers.SK_PARTIAL_SHAPES, gen_p1.py): at the row counts of a (lo, hi, slices a CTA, warps, stages) band the
small path launches stock's own _hc_mix_stats_partial_kernel (the same JIT function, the same constexpr arguments but
SLICES_PER_CTA) at that launch shape instead of stock's _stats_launch_shape / _HC_MIX_NUM_STAGES. The slice loop is
the kernel's runtime loop and every slice keeps its own accumulators and its own partial rows, so a shape changes the
grid / issue order only; whether a shape keeps stock's bits is measured (fork/tests/p1_test.py) and, in the engine,
checked again by the first eager call of every row count against stock's triplet (mismatch: stock for good).

LEVER SKW (gen_skw.py, skw.py): both sk_unit launches go through skw.run: inside CUDA-graph captures, at
row counts of levers.SKW_BANDS whose first eager call matched sk_unit byte for byte, the same reduce +
sinkhorn runs on skw_unit with a capped grid (B300 runs one kernel's CTAs on an SM at a time: the stats
stream no longer takes the SMs fmha / FC1 need at launch); everywhere else this module's sk_unit launch.

LEVER MHC (gen_mhc.py, mhc.py; EXACT config): _ours computes the stats partial on mhc_unit at
levers.MHC_MIN_ROWS..MHC_MAX_ROWS rows with a replica's mixing weight -- stock's Triton tf32x3 partial bit
for bit, on a short wide grid -- so lever SK's first-call bitwise check per row count covers it as before.
"""
import contextvars
import sys

import torch

from dsv41f import levers
from dsv41f import skw   # lever SKW: the reduce + sinkhorn on a capped grid in graphs
from dsv41f import mhc   # lever MHC: the mHC stats partial on mhc_unit

_ON = contextvars.ContextVar("dsv41f_sk_on", default=False)
_UNIT: list = []
_OK: dict = {}            # rows -> True (bitwise vs stock) / False (mismatch: stock for good)
_STATE = {"detail": "not installed", "stock": None, "mod": None, "calls": 0}   # + "stock_dg" once wrapped
_NEEDED = ("_hc_mix_stats_partial_kernel", "_num_slices_for", "_block_m_for", "_stats_launch_shape",
           "_HC_MIX_BLOCK_K", "_HC_MIX_DOT_PRECISION", "_HC_MIX_NUM_STAGES", "hc_mix_stats_sinkhorn", "triton")


def unit():
    if not _UNIT:
        mod = None
        try:
            import sk_unit as mod
        except ImportError as err:
            levers.announce("sk-missing", f"bundle CUDA unit sk_unit not importable ({err}): lever SK off")
        _UNIT.append(mod)
    return _UNIT[0]


def install() -> str:
    if not levers.SK:
        _STATE["detail"] = "off"
        return _STATE["detail"]
    if _STATE["stock"] is not None:
        return _STATE["detail"]
    try:
        from sglang.kernels.ops.layernorm import mhc as mod
        for name in _NEEDED:
            if not hasattr(mod, name):
                raise AttributeError(f"sglang.kernels.ops.layernorm.mhc.{name}")
    except Exception as e:  # noqa: BLE001 - a stack without this path keeps stock stats
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:160]})"
        return _STATE["detail"]
    if unit() is None:
        _STATE["detail"] = "off (unit sk_unit missing)"
        return _STATE["detail"]
    _STATE["stock"] = mod.hc_mix_stats_sinkhorn
    _STATE["mod"] = mod
    mod.hc_mix_stats_sinkhorn = _wrapped
    if levers.SK_DG and hasattr(mod, "hc_mix_stats_sinkhorn_deepgemm") and hasattr(mod, "_HC_MIX_COMPENSATED_SLICES"):
        _STATE["stock_dg"] = mod.hc_mix_stats_sinkhorn_deepgemm
        mod.hc_mix_stats_sinkhorn_deepgemm = _wrapped_dg
    _STATE["detail"] = (f"on, 1..{levers.SK_MAX_ROWS} rows" + (f" + deep_gemm path to {levers.SK_DG_MAX_ROWS} rows"
                        if "stock_dg" in _STATE else "") + ", first eager call per row count checked bitwise vs stock")
    return _STATE["detail"]


def forward(fn, *args, **kwargs):
    """Run our model forward with lever SK's wrapper active."""
    token = _ON.set(True)
    try:
        return fn(*args, **kwargs)
    finally:
        _ON.reset(token)


def partial_shape(mod, m: int, grid_m: int, num_slices: int):
    """(slices a CTA, warps, stages) of the stats partial launch: the first lever-P1 band holding m, else stock's."""
    for lo, hi, spc, warps, stages in getattr(levers, "SK_PARTIAL_SHAPES", ()):
        if int(lo) <= m <= int(hi) and int(spc) >= 1 and num_slices % int(spc) == 0:
            return int(spc), int(warps), int(stages)
    spc, warps = mod._stats_launch_shape(m, grid_m)
    return int(spc), int(warps), int(mod._HC_MIX_NUM_STAGES)


def _ours(x_flat, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps):
    """Stock hc_mix_stats_sinkhorn (CUDA path) with the reduce + sinkhorn launch on the unit."""
    mod = _STATE["mod"]
    m, k = x_flat.shape
    mix = hc_fn.shape[0]
    dev = x_flat.device
    pre = torch.empty(m, hc_mult, dtype=torch.float32, device=dev)
    post = torch.empty(m, hc_mult, dtype=torch.float32, device=dev)
    comb = torch.empty(m, hc_mult, hc_mult, dtype=torch.float32, device=dev)
    num_slices = mod._num_slices_for(k)
    mix_pad = max(16, mod.triton.next_power_of_2(mix))
    part_mix = torch.empty((num_slices, m, mix), dtype=torch.float32, device=dev)
    part_sq = torch.empty((num_slices, m), dtype=torch.float32, device=dev)
    block_m = mod._block_m_for(m)
    grid_m = mod.triton.cdiv(m, block_m)
    slices_per_cta, num_warps, num_stages = partial_shape(mod, m, grid_m, num_slices)   # lever P1 band, else stock's
    parts = mhc.serves(x_flat, hc_fn)   # lever MHC (exact): stock's partial bits on mhc_unit, short and wide
    if parts is not None and num_slices == mhc.SLICES and mix == mhc.MIX:
        mhc.partial(x_flat, parts, part_mix, part_sq)
    else:
        mod._hc_mix_stats_partial_kernel[(grid_m, num_slices // slices_per_cta)](
            x_flat,
            hc_fn,
            part_mix,
            part_sq,
            m,
            k,
            x_flat.stride(0),
            hc_fn.stride(0),
            MIX=mix,
            MIX_PAD=mix_pad,
            NUM_SLICES=num_slices,
            BLOCK_M=block_m,
            BLOCK_K=mod._HC_MIX_BLOCK_K,
            DOT_PRECISION=mod._HC_MIX_DOT_PRECISION,
            SLICES_PER_CTA=slices_per_cta,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    scale = hc_scale.float().contiguous()
    base = hc_base.float().contiguous()
    # lever SKW: skw_unit on a capped grid inside CUDA-graph captures at checked keys, this sk_unit launch otherwise
    skw.run(unit(), part_mix, None, part_sq, scale, base, pre, post, comb, m, 1.0 / k, float(rms_eps), num_slices,
            int(sinkhorn_iters), float(hc_eps), 1, torch.cuda.current_stream(dev).cuda_stream, tag=hc_fn)
    return pre, post, comb


def _ours_dg(x_flat, weight_parts, hc_scale, hc_base, sinkhorn_iters, rms_eps, hc_eps, hc_mult=4):
    """Stock hc_mix_stats_sinkhorn_deepgemm with the reduce + sinkhorn launch on the unit (residual = low part)."""
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

    mod = _STATE["mod"]
    m, k = x_flat.shape
    mix = (2 + hc_mult) * hc_mult
    slices = mod._HC_MIX_COMPENSATED_SLICES
    high, low = weight_parts
    dev = x_flat.device
    pre = torch.empty((m, hc_mult), dtype=torch.float32, device=dev)
    post = torch.empty_like(pre)
    comb = torch.empty((m, hc_mult, hc_mult), dtype=torch.float32, device=dev)
    mix_hi = torch.empty((slices, m, mix), dtype=torch.float32, device=dev)
    mix_lo = torch.empty_like(mix_hi)
    sq = torch.empty((slices, m), dtype=torch.float32, device=dev)
    tf32_hc_prenorm_gemm(x_flat, high, mix_hi, sq, slices)
    tf32_hc_prenorm_gemm(x_flat, low, mix_lo, sq, slices)
    # lever SKW: skw_unit on a capped grid inside CUDA-graph captures at checked keys, this sk_unit launch otherwise
    skw.run(unit(), mix_hi, mix_lo, sq, hc_scale, hc_base, pre, post, comb, m, 1.0 / k, float(rms_eps), slices,
            int(sinkhorn_iters), float(hc_eps), int(levers.SK_DG_ROWS_PER_CTA),
            torch.cuda.current_stream(dev).cuda_stream, tag=high)
    return pre, post, comb


def _note(key) -> str:
    if not isinstance(key, int) or not _STATE["mod"]:
        return ""
    mod = _STATE["mod"]
    try:
        grid_m = mod.triton.cdiv(key, mod._block_m_for(key))
        spc, warps, stages = partial_shape(mod, key, grid_m, 80)
    except Exception:  # noqa: BLE001 - the note is informational
        return ""
    return f" (partial {spc} slice(s) a CTA, {warps} warps, {stages} stage(s))"


def _checked(key, stock_call, ours_call):
    """Stock until this key's first eager call matched stock bit for bit, ours after; mismatch: stock for good."""
    ok = _OK.get(key)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return stock_call()
        want = stock_call()
        try:
            got = ours_call()
            ok = all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(want, got))
        except Exception as e:  # noqa: BLE001 - any unit failure keeps stock for this key
            print(f"CACHEON_DSV4F lever SK: {key} failed ({type(e).__name__}: {str(e)[:120]}), stock",
                  file=sys.stderr, flush=True)
            ok = False
        _OK[key] = ok
        levers.announce(f"sk{key}", f"lever SK: {key} {'bitwise vs stock: on' if ok else 'MISMATCH: stock'}{_note(key)}")
        return want
    if not ok:
        return stock_call()
    _STATE["calls"] += 1
    return ours_call()


def _eligible(x_flat, hc_fn, hc_mult, num_slices_ok=True) -> bool:
    m = x_flat.shape[0]
    return (1 <= m <= levers.SK_MAX_ROWS and hc_mult == 4 and hc_fn.dim() == 2 and hc_fn.shape[0] == 24
            and hc_fn.dtype == torch.float32 and x_flat.is_cuda and x_flat.dim() == 2 and x_flat.stride(1) == 1
            and hc_fn.stride(1) == 1 and num_slices_ok)


def _wrapped(x_flat, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps):
    stock = _STATE["stock"]
    args = (x_flat, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps)
    if not _ON.get() or not _eligible(x_flat, hc_fn, hc_mult, _STATE["mod"]._num_slices_for(x_flat.shape[1]) <= 80):
        return stock(*args)
    return _checked(int(x_flat.shape[0]), lambda: stock(*args), lambda: _ours(*args))


def _wrapped_dg(x_flat, weight_parts, hc_scale, hc_base, sinkhorn_iters, rms_eps, hc_eps, hc_mult=4):
    stock = _STATE["stock_dg"]
    args = (x_flat, weight_parts, hc_scale, hc_base, sinkhorn_iters, rms_eps, hc_eps, hc_mult)
    m = int(x_flat.shape[0])
    if (not _ON.get() or not levers.SK_DG or m < 1 or m > levers.SK_DG_MAX_ROWS or hc_mult != 4
            or x_flat.dtype != torch.bfloat16 or not x_flat.is_contiguous() or len(weight_parts) != 2
            or hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32
            or not hc_scale.is_contiguous() or not hc_base.is_contiguous()):
        return stock(*args)
    return _checked(f"dg{m}", lambda: stock(*args), lambda: _ours_dg(*args))
