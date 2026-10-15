"""Lever SK: the mHC stats reduce + sinkhorn of our layer replicas' decode / verify sublayers on the bundle CUDA unit
sk_unit (bit-identical to the fork's Triton _hc_mix_reduce_sinkhorn_kernel, partials fetched in one round trip).

install() wraps sglang.kernels.ops.layernorm.mhc.hc_mix_stats_sinkhorn (the 1..127-row Blackwell path of
deepseek_v4_mhc._mix_stats_impl, which imports it at call time) and hc_mix_stats_sinkhorn_deepgemm (128+ rows: two
deep_gemm tf32 prenorm GEMMs, then the reduce with the low part as residual); the wrappers run stock's own partial
launches and then the unit, only inside our model forward (forward() below sets the context) and only for row counts whose first
eager call (a warmup forward, outside graph capture) matched stock's triplet bit for bit; everything else is stock.
"""
import contextvars
import sys

import torch

from dsv41f import levers

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
    slices_per_cta, num_warps = mod._stats_launch_shape(m, grid_m)
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
        num_stages=mod._HC_MIX_NUM_STAGES,
    )
    scale = hc_scale.float().contiguous()
    base = hc_base.float().contiguous()
    unit().run(part_mix.data_ptr(), 0, part_sq.data_ptr(), scale.data_ptr(), base.data_ptr(), pre.data_ptr(),
               post.data_ptr(), comb.data_ptr(), m, 1.0 / k, float(rms_eps), num_slices, int(sinkhorn_iters),
               float(hc_eps), 1, torch.cuda.current_stream(dev).cuda_stream)
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
    unit().run(mix_hi.data_ptr(), mix_lo.data_ptr(), sq.data_ptr(), hc_scale.data_ptr(), hc_base.data_ptr(),
               pre.data_ptr(), post.data_ptr(), comb.data_ptr(), m, 1.0 / k, float(rms_eps), slices,
               int(sinkhorn_iters), float(hc_eps), int(levers.SK_DG_ROWS_PER_CTA),
               torch.cuda.current_stream(dev).cuda_stream)
    return pre, post, comb


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
        levers.announce(f"sk{key}", f"lever SK: {key} {'bitwise vs stock: on' if ok else 'MISMATCH: stock'}")
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
