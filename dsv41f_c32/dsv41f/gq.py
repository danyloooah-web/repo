"""Lever GQ (levers.GQ, CUDA unit gq_unit, gen_gq.py): the shared expert's gate_up input quantize on a SMALL grid in
lever STP's split ("tail" / "late" modes, stp.shared) at levers.GQ_MIN_ROWS..GQ_MAX_ROWS rows: the unit's e4m3 values
+ UE8M0 scales (FlashInfer's 128x4 swizzled layout, every scale byte of the padded rows zero) are stock's bytes for the
call's row count, handed to gate_up_proj as the Mxfp8SwizzledInput stock's own fused producers hand MXFP8 linears.

Why (c29sp1 profile, k8 / 48 rows): stock's flashinfer_mxfp8_quantize (FlashInfer 0.7.1 CuTe-DSL
MXFP8QuantizeSwizzledKernel, 128 CTAs x 640 threads) runs on the MoE's alt stream, forked before the router; when it is
still spread over ~128 SMs as the router GEMM (RTR) ends, RE5's 12-CTA cluster (a whole free GPC) waits behind the
side chain: route_e5 > 12 us in 27 % of such layers vs 1.6 % otherwise, FC1 start - RTR end 11.3 vs 8.2 us. The unit
runs ceil(rows / GQ_ROWS_PER_CTA) CTAs (6 at 48 rows, 15 at 120).

Path: Fp8LinearMethod.apply takes an Mxfp8SwizzledInput (a tuple) on the MXFP8 / CuTe-DSL view and calls the linear's
w8a8_mxfp8_linear(input=q, input_scale=scales) -- stock's flashinfer_mxfp8_blockscaled_linear (or lever DT's forced
tactic copy of its body), which skips only its own flashinfer_mxfp8_quantize call; output dtype (bf16), tactic
(_mxfp8_decode_tactic of q.shape[0] = the same M) and GEMM launch are the ones stock's bf16 input takes.

Exactness: the unit's arithmetic is FlashInfer's MXFP8 quantize (lev_HQ / lev_QN mode 0, pinned FlashInfer sources);
prepare self-tests it byte for byte against stock's flashinfer_mxfp8_quantize on realistic, extreme and special rows at
several row counts (agreement over the TP ranks is not needed: a rank whose test fails runs stock's bytes, which equal
the unit's wherever it passes). In the engine the first eager call of every (rows, row stride) key (the decode / verify
graph warmups) runs stock's quantize and the unit on the same input and compares q and every scale byte; a difference
turns GQ off for the process (stock from then on); inside a capture an unchecked key runs stock. Lever STP's own
first-call check then compares the whole shared output (GQ's gate_up included) with stock's _forward_shared_experts.
Gated off: other dtypes / widths / strides / alignment, a sequence-parallel forward, a gate_up linear that is not an
MXFP8 CuTe-DSL Fp8LinearMethod accepting the swizzled input contract, hooks, a stock source that differs from the
pinned one.
"""
import sys

import torch

from dsv41f import levers

H = 5120
SF_TILE = 128 * (H // 32)
BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
U8 = torch.uint8
_FN: dict = {}            # run, quant, mxi, raw, fwd
_OK: dict = {}            # (rows, row stride) -> True (bytes == stock) / False
_LIN: dict = {}           # id(replica gate_up linear) -> the linear (prepare)
_STATE = {"detail": "not installed", "ok": False, "fast": 0, "checks": 0}
# sha256[:16] of the source of the stock functions the unit's bytes / the handoff rely on (values as lever HQ pins them)
PINNED = {
    ("sglang.srt.layers.quantization.mxfp8_input", "Mxfp8SwizzledInput"): "54c48c829cc789f1",
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_quantize"): "01a12afdc8ca1435",
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_blockscaled_linear"): "99da4a08ba42d542",
    ("sglang.srt.layers.quantization.fp8", "Fp8LinearMethod.apply"): "24ab0721ac46abe0",
    ("flashinfer.quantization.fp8_quantization", "mxfp8_quantize"): "ccfdbec3c2435309",
    ("flashinfer.quantization.kernels.mxfp8_quantize", "MXFP8QuantizeSwizzledKernel"): "38a19792f2c0d41f",
    ("flashinfer.quantization.kernels.mxfp8_quantize", "mxfp8_quantize_cute_dsl"): "e9cc7dad53e84360",
    ("flashinfer.quantization.kernels.mxfp8_quantize", "_get_compiled_kernel_mxfp8_swizzled"): "4e50f2ed723fbaaa",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "float_to_ue8m0_fast"): "b9f92736cce2e9b1",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "ue8m0_to_inv_scale_fast"): "0d4be586bdf7821a",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "bfloat2_to_fp8x2_scaled"): "b3ea10404f5068f9",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "bfloat2_hmax_reduce_to_f32"): "4060f8d935ef457e",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "bfloat2_max_abs_4"): "3b96731a268157ae",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "bfloat2_max_abs_8"): "97ed493579f08abe",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "bfloat2x4_to_fp8x8_packed"): "628f9bc278de0113",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "reduce_max_2threads"): "8742672d9d38788f",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "reduce_max_4threads"): "f788a1c51d76a4e3",
    ("flashinfer.quantization.quantization_cute_dsl_utils", "compute_sf_index_swizzled_128x4_gpu"): "a7d51edbe135c092",
}


def _modules():
    import flashinfer.quantization.fp8_quantization  # noqa: F401
    import flashinfer.quantization.kernels.mxfp8_quantize  # noqa: F401
    import flashinfer.quantization.quantization_cute_dsl_utils  # noqa: F401
    import sglang.srt.layers.quantization.fp8  # noqa: F401
    import sglang.srt.layers.quantization.fp8_utils  # noqa: F401
    import sglang.srt.layers.quantization.mxfp8_input  # noqa: F401
    return {name: sys.modules[name] for name, _ in PINNED}


def _rows_for_test():
    """Self-test inputs: realistic (normal, heavy-tailed), extreme magnitudes, zero / subnormal / +-0 / inf / NaN rows."""
    g = torch.Generator(device="cuda")
    g.manual_seed(int(levers.GQ_SEED))
    out = []
    for m in (9, 48, 72, 96, 120, 127):
        x = torch.randn((m, H), generator=g, device="cuda", dtype=torch.float32)
        x = x * torch.exp(torch.randn((m, 1), generator=g, device="cuda") * 2.0)
        x[0] = x[0] * 1e-30
        x[1, ::7] = 0.0
        x[2, :32] = -0.0
        x[3, 64:96] = 3.0e38
        x[4, 5] = 1e-41
        if m > 6:
            x[5, ::33] = 448.0 * 2.0 ** 100
            x[6] = torch.randint(-3, 4, (H,), generator=g, device="cuda").float() * 2.0 ** -126
        out.append(x.to(BF16))
    sp = torch.randn((48, H), generator=g, device="cuda").to(BF16)
    sp[0, 10] = float("inf")
    sp[1, 40] = float("-inf")
    sp[2, 70] = float("nan")
    sp[3, :32] = float("nan")
    out.append(sp)
    wide = torch.randn((72, H + 64), generator=g, device="cuda").to(BF16)[:, 32:32 + H]   # strided rows
    out.append(wide)
    return out


def _launch(x, m, sx):
    q = torch.empty((m, H), dtype=U8, device=x.device)
    s = torch.empty((((m + 127) >> 7) * SF_TILE,), dtype=U8, device=x.device)
    _FN["run"](x.data_ptr(), sx, q.data_ptr(), s.data_ptr(), m, int(levers.GQ_ROWS_PER_CTA), int(levers.GQ_PDL),
               _FN["raw"](x.get_device()))
    return q.view(FP8), s


def _same(x, m, sx):
    """'' when the unit's q / scales equal stock's flashinfer_mxfp8_quantize bytes for x, else what differs."""
    q_ref, s_ref = _FN["quant"](x.view(-1, H) if x.is_contiguous() else x, is_sf_swizzled_layout=True, alignment=32)
    q, s = _launch(x, m, sx)
    if q_ref.dtype is not FP8 or tuple(q_ref.shape) != (m, H) or q_ref.stride() != q.stride():
        return f"q layout {q_ref.dtype} {tuple(q_ref.shape)} {q_ref.stride()}", None
    if not torch.equal(q.view(U8), q_ref.view(U8)):
        return "q bytes", None
    if s_ref.dtype is not U8 or s_ref.dim() != 1 or s_ref.numel() != s.numel() or not torch.equal(s, s_ref):
        return f"scales ({s_ref.dtype} {tuple(s_ref.shape)} vs {tuple(s.shape)})", None
    return "", (q, s)


def _eligible(lin) -> str:
    from sglang.srt.layers.quantization.mxfp8_input import accepts_mxfp8_swizzled_input

    method = getattr(lin, "quant_method", None)
    if method is None or type(method).__name__ != "Fp8LinearMethod":
        return "gate_up not Fp8LinearMethod"
    backend = getattr(method, "mxfp8_dense_backend", None)
    if backend is None or not backend.is_flashinfer_cutedsl() or not accepts_mxfp8_swizzled_input(lin):
        return "gate_up not an MXFP8 CuTe-DSL linear taking Mxfp8SwizzledInput"
    w = lin._parameters.get("weight")
    if w is None or w.dtype is not FP8 or w.dim() != 2 or w.shape[1] != H:
        return "gate_up weight not e4m3 [N, 5120]"
    if getattr(lin, "bias", None) is not None:
        return "gate_up has a bias"
    return ""


def prepare(layers) -> str:
    """Pin the stock sources, load + self-test the unit and register the STP-armed replicas' gate_up linears
    (model_node.prepare, after lever STP's prepare)."""
    if not getattr(levers, "GQ", False):
        _STATE["detail"] = "off (levers.GQ)"
        return _STATE["detail"]
    try:
        from dsv41f import hq
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
        from sglang.srt.runtime_context import get_forward

        mods = _modules()
        cache: dict = {}
        bad = [f"{mod}.{qual}" for (mod, qual), want in PINNED.items()
               if hq.source_hash(mods[mod], qual, cache) != want]
        if bad:
            raise RuntimeError("stock source differs from the pinned read: " + ", ".join(bad))
        import gq_unit

        raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        _FN.update(run=gq_unit.run, quant=fp8_utils.flashinfer_mxfp8_quantize, mxi=Mxfp8SwizzledInput,
                   fwd=get_forward,
                   raw=raw if raw is not None else (lambda idx: torch.cuda.current_stream(idx).cuda_stream))
        why, n = {}, 0
        for r in layers:
            moe = r._modules.get("mlp")
            if moe is None or not getattr(moe, "dsv4f_stp", False):
                continue
            lin = moe._modules["shared_experts"]._modules.get("gate_up_proj")
            reason = _eligible(lin) if lin is not None else "no gate_up"
            if reason:
                why[reason] = why.get(reason, 0) + 1
                continue
            _LIN[id(lin)] = lin
            n += 1
        if not n:
            raise RuntimeError(f"no STP-armed shared gate_up eligible ({why})")
        for x in _rows_for_test():
            what, _ = _same(x, x.shape[0], x.stride(0))
            if what:
                raise RuntimeError(f"self-test at {x.shape[0]} rows (stride {x.stride(0)}): {what} != stock")
        torch.cuda.synchronize()
        _STATE["ok"] = True
        _STATE["detail"] = (f"on: {n} shared gate_up linears, rows {levers.GQ_MIN_ROWS}..{levers.GQ_MAX_ROWS}, "
                            f"{levers.GQ_ROWS_PER_CTA} rows a CTA, pdl {levers.GQ_PDL}; self-test bytes == stock"
                            + (f"; not armed {why}" if why else ""))
    except Exception as e:  # noqa: BLE001 - any missing / changed internal keeps stock
        _STATE["ok"] = False
        _LIN.clear()
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:300]})"
    return _STATE["detail"]


def gate_up_input(lin, x):
    """What lever STP's split hands gate_up_proj: x itself (stock: the linear quantizes it), or the unit's
    Mxfp8SwizzledInput of x (stock's bytes) for an eligible call."""
    if not _STATE["ok"] or _LIN.get(id(lin)) is not lin:
        return x
    if type(x) is not torch.Tensor or x.dtype is not BF16 or x.dim() != 2 or x.shape[1] != H or x.stride(1) != 1:
        return x
    m = int(x.shape[0])
    if m < levers.GQ_MIN_ROWS or m > levers.GQ_MAX_ROWS:
        return x
    sx = int(x.stride(0))
    if sx % 8 or sx < H or x.data_ptr() & 15 or not x.is_cuda:
        return x
    if lin._forward_hooks or lin._forward_pre_hooks or getattr(_FN["fwd"](), "sp_active", True):
        return x
    key = (m, sx)
    ok = _OK.get(key)
    if ok:
        q, s = _launch(x, m, sx)
        _STATE["fast"] += 1
        return _FN["mxi"](q, s)
    if ok is not None or torch.cuda.is_current_stream_capturing():
        return x
    try:
        what, got = _same(x, m, sx)
    except Exception as e:  # noqa: BLE001 - a failed launch / check: stock from now on
        what, got = f"{type(e).__name__}: {str(e)[:200]}", None
    _STATE["checks"] += 1
    _OK[key] = not what
    if what:
        _STATE["ok"] = False
        print(f"CACHEON_DSV4F lever GQ: {key}: {what} differs from stock: lever GQ OFF", file=sys.stderr, flush=True)
        return x
    levers.announce(f"gq-{m}-{sx}", f"lever GQ: first at {m} rows (stride {sx}): q / scales == stock: on")
    _STATE["fast"] += 1
    return _FN["mxi"](*got)


def summary() -> str:
    return f"{_STATE['fast']} fast calls, {_STATE['checks']} checks, {'on' if _STATE['ok'] else 'off'}"
