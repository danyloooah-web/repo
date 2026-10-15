"""Lever DT on the fork (levers.DT_TACTICS): small-row calls of the replicas' dense MXFP8 linears (the checkpoint's 32x32
ue8m0 block-FP8 weights served as MXFP8 on FlashInfer's CuTe-DSL mm_mxfp8 runner: wqkv_a / wq_a / wkv, wq_b, wo_b, the
indexer's wq_b, the shared expert, Engram's wkv) launch at a measured tactic for their (N, K, row band) instead of the
runner's untuned low-M default (the fork pins its own tactic only above 32 rows: fp8_utils._mxfp8_decode_tactic).

Why: at decode / verify sizes these GEMMs stream a weight that the default narrow swap-AB tile spreads over N / 128
CTAs only (14-40 of 148 SMs for this model's shapes), so each launch is bound by per-SM pull rate, not HBM.

Mechanism: every MXFP8 linear under a layer replica gets a replica linear (shallow copy) holding a replica quant method
whose w8a8_mxfp8_linear is this module's linear(): stock's flashinfer_mxfp8_blockscaled_linear body (quantize, scale
view, CuTe-DSL runner, bias, view) with the band's tactic; calls outside a band, with extra kwargs (the prefill
autotune pin), or on a key whose bit check failed reach stock's function unchanged. The served model is not modified.

Exactness: tactics of split 1 differ in tile / cluster / prefetch only, which leave each output's K reduction order
alone; the first eager call of every (N, K, band) runs stock and the forced launch on the same input and compares
every output byte -- a difference turns that key off for the process. Split-K tactics (split > 1) change the fp32
summation order: they are only used when levers.DT_SPLIT_K allows it (PRECISION), never bit-checked.
"""
import copy

import torch

from dsv41f import levers

_KEY_OK: dict = {}       # (n, k, band index) -> True (bits equal / split-K allowed) / False (off for the process)


def _band(m: int, n: int, k: int):
    for i, (lo, hi, tactic) in enumerate(levers.DT_TACTICS.get((n, k), ())):
        if lo <= m <= hi:
            return i, tactic
    return None, None


def _forced(input, weight, weight_scale, input_scale, bias, tactic):
    """flashinfer_mxfp8_blockscaled_linear(backend='cute-dsl') with the launch tactic given (stock's body otherwise)."""
    from sglang.srt.layers.quantization.fp8_utils import _cute_dsl_mm_mxfp8_with_tactic, flashinfer_mxfp8_quantize

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    if input_scale is None:
        q_input, x_scale_u8 = flashinfer_mxfp8_quantize(input_2d, is_sf_swizzled_layout=True, alignment=32)
    else:
        q_input, x_scale_u8 = input_2d, input_scale
    if input_2d.dtype in (torch.float16, torch.bfloat16, torch.float32):
        output_dtype = input_2d.dtype
    else:
        output_dtype = torch.bfloat16
    weight_scale_t = weight_scale.t() if weight_scale.ndim == 2 else weight_scale
    output = _cute_dsl_mm_mxfp8_with_tactic(q_input, weight.t(), x_scale_u8, weight_scale_t, output_dtype,
                                            tuple(tactic))
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)


def _make_linear(stock):
    def linear(input, weight, weight_scale, input_scale=None, bias=None, **kwargs):
        if kwargs:
            return stock(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias,
                         **kwargs)
        m = input.numel() // input.shape[-1]
        n, k = weight.shape
        band, tactic = _band(m, n, k)
        if band is None:
            return stock(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias)
        key = (n, k, band)
        ok = _KEY_OK.get(key)
        if ok is None:
            split = tactic[4] if len(tactic) > 4 else 1
            if split > 1:
                ok = _KEY_OK[key] = bool(levers.DT_SPLIT_K)
                levers.announce(f"dt-{n}x{k}-{band}", f"lever DT N={n} K={k} band {band} tactic {tactic} first at "
                                f"{m} rows: split-K {'on (PRECISION)' if ok else 'not allowed: OFF'}")
            elif torch.cuda.is_current_stream_capturing():
                return stock(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                             bias=bias)
            else:
                ref = stock(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias)
                try:
                    got = _forced(input, weight, weight_scale, input_scale, bias, tactic)
                    ok = ref.shape == got.shape and ref.dtype == got.dtype and torch.equal(
                        ref.view(torch.int16) if ref.element_size() == 2 else ref,
                        got.view(torch.int16) if got.element_size() == 2 else got)
                except Exception as e:  # noqa: BLE001 - a tactic the kernel rejects at this shape: stock from now on
                    ok = False
                    levers.announce(f"dt-err-{n}x{k}-{band}", f"lever DT N={n} K={k} tactic {tactic}: "
                                    f"{type(e).__name__}: {str(e)[:160]}")
                _KEY_OK[key] = ok
                levers.announce(f"dt-{n}x{k}-{band}", f"lever DT N={n} K={k} band {band} tactic {tactic} first at "
                                f"{m} rows: forced bits {'== stock: on' if ok else '!= stock: OFF'}")
                return got if ok else ref
        if not ok:
            return stock(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias)
        return _forced(input, weight, weight_scale, input_scale, bias, tactic)

    return linear


def _eligible(module) -> bool:
    method = getattr(module, "quant_method", None)
    if method is None or type(method).__name__ != "Fp8LinearMethod":
        return False
    if getattr(method, "w8a8_mxfp8_linear", None) is None:
        return False
    backend = getattr(method, "mxfp8_dense_backend", None)
    if backend is None or not backend.is_flashinfer_cutedsl():
        return False
    weight = module._parameters.get("weight")
    return weight is not None and weight.dtype == torch.float8_e4m3fn and (tuple(weight.shape) in levers.DT_TACTICS)


def _cow(parent, name):
    """The child `name` of `parent` as a private shallow copy (own `_modules`), installed on parent."""
    child = parent._modules[name]
    if getattr(child, "dsv4f_dt_copy", False):
        return child
    twin = copy.copy(child)
    twin._modules = dict(child._modules)
    twin.dsv4f_dt_copy = True
    parent._modules[name] = twin
    return twin


def bind(replica) -> int:
    """Swap every eligible MXFP8 linear under the layer replica for a DT replica linear; returns how many."""
    if not levers.DT or not levers.DT_TACTICS:
        return 0
    paths = []

    def walk(module, path):
        for name, child in module._modules.items():
            # `_next_layer` is the following decoder layer (read for its norm / mHC weights): not this layer's linears
            if child is None or name == "_next_layer" or type(child).__name__ == "DeepseekV4DecoderLayer":
                continue
            if _eligible(child):
                paths.append(path + (name,))
            else:
                walk(child, path + (name,))

    walk(replica, ())
    for path in paths:
        parent = replica
        for name in path[:-1]:
            parent = _cow(parent, name)
        linear = copy.copy(parent._modules[path[-1]])
        method = copy.copy(linear.quant_method)
        method.w8a8_mxfp8_linear = _make_linear(method.w8a8_mxfp8_linear)
        linear.quant_method = method
        linear.dsv4f_dt = True
        parent._modules[path[-1]] = linear
    return len(paths)
