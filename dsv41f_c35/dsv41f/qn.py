"""Lever QN (CUDA unit qn_unit): MQALayer._normalize_q_lora inside OUR model forward -- the q-path RMSNorm
(q_norm over q_lora_rank K = 1280) and the MXFP8 quantize of wq_b's input -- as ONE launch whose y (bf16), q (e4m3) and
scales (UE8M0, FlashInfer's 128x4 swizzled layout, padded rows zero) equal stock's bytes for the call's row count.

Stock (fork deepseek_v4.py): 1..64 rows -> the Triton rmsnorm_mxfp8 (M is a tl.constexpr: one compile per row count, a
single 128-row scale tile) handing wq_b an Mxfp8SwizzledInput; above 64 rows -> self.q_norm (sgl_kernel.rmsnorm ->
FlashInfer 0.7.1's CuTe-DSL RMSNorm) and, inside wq_b's linear, flashinfer_mxfp8_quantize (CuTe-DSL, 128x4). The two
norms round differently (other fp32 reduction trees), so the unit has one mode per regime (qn_unit.cu header): mode 1 =
the Triton kernel's arithmetic as Triton 3.8.0 compiles it for sm_103 (per-thread squares summed as an f32x2 pairwise
tree), mode 0 = FlashInfer's norm (cutlass-dsl 4.8.0: one fma chain per thread) + quantize; both take M at run time and
any number of 128-row scale tiles. tests/qn_test.py extracts both references' reduction trees from their sm_103 PTX. Above 64 rows wq_b then gets the Mxfp8SwizzledInput: the input-scale path of the same linear (stock's own
<= 64-row contract), i.e. the same GEMM launch on the same bytes.

Why: eager EXTEND forwards are host-bound on TP-1; above 64 rows stock pays the RMSNorm module dispatch, the
sgl_kernel / FlashInfer custom-op wrappers and, in wq_b, the quantize call chain (two kernels); here one pybind launch.

Binding: install() sets `_normalize_q_lora` as an instance attribute of OUR private attention replicas (the copies
model_node's _replica makes; the lifted attention forward's stock _forward_prepare calls self._normalize_q_lora), never
on MQALayer or a served module: cacheon's node adapter seals every callable of the served node's modules and of their
classes at bind time and refuses an audited call once one was rebound ("a method inside node ... was rebound after
binding"), so a class-level wrapper would fail the audit launch.

Scope: only inside our model forward (forward() sets the context): EXTEND forwards outside CUDA-graph capture, the
breakable prefill graph and torch.compile at every row count, and -- with levers.QN_GRAPH_MAX_ROWS -- decode / verify
calls of 65..QN_GRAPH_MAX_ROWS rows (capture-safe launch; a key is never checked under capture). Stock's method runs for
everything else: another mode, deterministic / batch-invariant inference, a forced fused-op backend, a layer outside
stock's fused-path conditions (CuTe-DSL MXFP8 wq_b, bf16 1280-wide q_norm on its plain CUDA path), a call of another
dtype / width / stride or a misaligned row.

Exactness: the unit test sweeps every M = 1..4096 and samples to 16384 against stock's functions byte for byte. In the
engine the first eager call of every check key (mode, rows mod 4, whole 128-row tiles, more than one tile; each decode /
verify row count) runs stock and the unit on the same input and compares y, q and every scale byte: one difference
turns the lever off for the process. install() pins the source (sha256 of the AST segment) of every stock function the
unit or this module mirrors or relies on, and the FlashInfer / CuTe-DSL / Triton versions whose code generation the
unit's arithmetic was taken from; any difference keeps stock.
"""
import ast
import contextvars
import functools
import hashlib
import inspect
import sys
import types

import torch

from dsv41f import levers

K = 1280
BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
U8 = torch.uint8
_ON = contextvars.ContextVar("dsv41f_qn", default=None)
_STATE = {"detail": "not installed", "ok": False, "fast": [0, 0], "checks": 0, "stock": None}
_LAYER: dict = {}         # id(attention) -> (attention, q_norm weight, eps, device, device index) or (attention, None)
_OK: dict = {}            # check key -> True (bytes == stock) / False
_FN: dict = {}            # run, mxi, quant, raw_stream, fm, in_bcg, det, fused_backend

VERSIONS = {"flashinfer": "0.7.1", "cutlass": "4.8.0", "triton": "3.8.0"}

# sha256[:16] of the source segment of every function the unit mirrors or this module relies on, read from the fork's
# SGLang (5e72ce5) and FlashInfer 0.7.1 (0b46115d); any difference: lever QN off
PINNED = {
    ("sglang.srt.models.deepseek_v4", "MQALayer._normalize_q_lora"): "f74e4175035f8bce",
    ("sglang.kernels.ops.layernorm.mxfp8_epilogue", "ue8m0_scale"): "4d2537c046eb29a8",
    ("sglang.kernels.ops.layernorm.mxfp8_epilogue", "mxfp8_epilogue"): "6fcd45ea13ee6fce",
    ("sglang.kernels.ops.layernorm.mxfp8_epilogue", "_rmsnorm_mxfp8_kernel"): "488395f6ae5b98b0",
    ("sglang.kernels.ops.layernorm.mxfp8_epilogue", "rmsnorm_mxfp8"): "43fe69f9e0218a74",
    ("sglang.srt.layers.layernorm", "RMSNorm.forward_cuda"): "bae8ae6279fa203d",
    ("sgl_kernel.elementwise", "rmsnorm"): "6719458ecd215842",
    ("flashinfer.norm", "rmsnorm"): "e38b8b8ab290ca6e",
    ("flashinfer.norm", "_rmsnorm_impl"): "fac46e118f4370d2",
    ("flashinfer.norm", "_use_cuda_norm"): "2720fee53a515025",
    ("flashinfer.norm.kernels.rmsnorm", "RMSNormKernel"): "d67c06a3d52fb609",
    ("flashinfer.norm.kernels.rmsnorm", "rmsnorm_cute"): "764118b09d160608",
    ("flashinfer.norm.kernels.rmsnorm", "_get_compiled_rmsnorm_kernel"): "3e291c8444c74527",
    ("flashinfer.norm.utils", "row_reduce_sum_multirow"): "5f11756d90994acd",
    ("flashinfer.norm.utils", "warp_reduce"): "a96a7feb8267a875",
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
    ("flashinfer.cute_dsl.fp4_common", "bfloat2_habs2"): "2ac5f89ad4334d3b",
    ("flashinfer.cute_dsl.fp4_common", "bfloat2_hmax2"): "cdfa3326e734345d",
    ("flashinfer.cute_dsl.fp4_common", "fmax_f32"): "ebbb9d5ad71dca46",
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_quantize"): "01a12afdc8ca1435",
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_blockscaled_linear"): "99da4a08ba42d542",
    ("sglang.srt.layers.quantization.fp8", "Fp8LinearMethod.apply"): "24ab0721ac46abe0",
    ("sglang.srt.layers.linear", "ColumnParallelLinear.forward"): "b259e0c52e38cfe8",
}


def source_hash(module, qualname: str, cache=None):
    """sha256[:16] of the exact source text of `qualname` ("f" or "Class.f") in `module` (None: not found)."""
    hit = cache.get(module.__name__) if cache is not None else None
    if hit is None:
        src = inspect.getsource(module)
        hit = (src, ast.parse(src))
        if cache is not None:
            cache[module.__name__] = hit
    src, tree = hit
    node = tree
    for part in qualname.split("."):
        found = None
        for child in ast.walk(node):
            if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part
                    and child is not node):
                found = child
                break
        if found is None:
            return None
        node = found
    seg = ast.get_source_segment(src, node)
    return hashlib.sha256(seg.encode()).hexdigest()[:16] if seg is not None else None


def _modules():
    """The modules PINNED names, imported (raises on a missing one)."""
    import flashinfer.cute_dsl.fp4_common  # noqa: F401
    import flashinfer.norm
    import flashinfer.norm.kernels.rmsnorm  # noqa: F401
    import flashinfer.norm.utils  # noqa: F401
    import flashinfer.quantization.fp8_quantization  # noqa: F401
    import flashinfer.quantization.kernels.mxfp8_quantize  # noqa: F401
    import flashinfer.quantization.quantization_cute_dsl_utils  # noqa: F401
    import sgl_kernel.elementwise  # noqa: F401
    import sglang.kernels.ops.layernorm.mxfp8_epilogue  # noqa: F401
    import sglang.srt.layers.layernorm  # noqa: F401
    import sglang.srt.layers.linear  # noqa: F401
    import sglang.srt.layers.quantization.fp8  # noqa: F401
    import sglang.srt.layers.quantization.fp8_utils  # noqa: F401
    import sglang.srt.models.deepseek_v4  # noqa: F401
    return {name: sys.modules[name] for name, _ in PINNED}


def unit():
    if "run" not in _FN:
        try:
            import qn_unit
        except ImportError as err:
            levers.announce("qn-missing", f"bundle CUDA unit qn_unit not importable ({err}): lever QN off")
            return None
        _FN["run"] = qn_unit.run
    return _FN["run"]


def install(model, layers=None) -> str:
    """Pin the stock internals, check the served model's wq_b linears and bind the wrapped _normalize_q_lora on the
    private attention replicas of `layers` (model_node.prepare's replicas, aligned with model.layers)."""
    if _STATE["stock"] is not None or _STATE["detail"].startswith("off"):
        return _STATE["detail"]
    if not getattr(levers, "QN", False):
        _STATE["detail"] = "off (levers.QN)"
        return _STATE["detail"]
    try:
        import cutlass
        import flashinfer
        import triton
        from sglang.kernels.fused_op import get_fused_op_backend
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.runtime_context import get_exec

        have = {"flashinfer": flashinfer.__version__, "cutlass": cutlass.__version__, "triton": triton.__version__}
        if have != VERSIONS:
            raise RuntimeError(f"versions {have} (unit matched against {VERSIONS})")
        mods = _modules()
        cache: dict = {}
        bad = [f"{mod}.{name} {got}" for (mod, name), want in PINNED.items()
               if (got := source_hash(mods[mod], name, cache)) != want]
        if bad:
            raise RuntimeError("source differs from the mirror: " + ", ".join(bad))
        dsv4 = mods["sglang.srt.models.deepseek_v4"]
        fin = mods["flashinfer.norm"]
        fp8_utils = mods["sglang.srt.layers.quantization.fp8_utils"]
        # the runtime bindings the source pins cannot see: RMSNorm.forward_cuda's module-level `rmsnorm` is
        # sgl_kernel.elementwise.rmsnorm, and that takes its flashinfer.norm branch (no ImportError fallback to
        # sgl_kernel's own CUDA rmsnorm, whose fp32 reduction tree is not mode 0's)
        sgl_el = mods["sgl_kernel.elementwise"]
        if (getattr(sgl_el, "_has_flashinfer", None) is not True or getattr(sgl_el, "_flashinfer_norm", None) is not fin
                or BF16 not in getattr(sgl_el, "_FLASHINFER_NORM_SUPPORTED_DTYPES", ())
                or getattr(mods["sglang.srt.layers.layernorm"], "rmsnorm", None) is not sgl_el.rmsnorm):
            raise RuntimeError("q_norm's rmsnorm does not resolve to sgl_kernel -> flashinfer.norm.rmsnorm")
        layer_cls = dsv4.MQALayer
        stock = layer_cls._normalize_q_lora
        if fin._use_cuda_norm(torch.device("cuda", torch.cuda.current_device())):
            raise RuntimeError("FlashInfer selects its CUDA-JIT norm (the mode 0 reference is the CuTe-DSL norm)")
        if dsv4._is_hip or not dsv4._is_cuda or not dsv4.get_platform().is_blackwell:
            raise RuntimeError("not CUDA Blackwell")
        why = _served_linears(model, fp8_utils.flashinfer_mxfp8_blockscaled_linear)
        if why:
            raise RuntimeError(why)
        targets = _replica_attns(model, layers, layer_cls, stock)
        if not targets:
            raise RuntimeError("no private MQALayer replica to bind on (lever QN never wraps a served module)")
        if unit() is None:
            raise RuntimeError("unit qn_unit missing")
        raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        _FN.update(mxi=Mxfp8SwizzledInput, quant=fp8_utils.flashinfer_mxfp8_quantize, fm=ForwardMode,
                   in_bcg=is_in_breakable_cuda_graph, bi=is_batch_invariant_mode_enabled, get_exec=get_exec,
                   fused_backend=get_fused_op_backend, rmsnorm_cls=mods["sglang.srt.layers.layernorm"].RMSNorm,
                   rmsnorm_cuda=mods["sglang.srt.layers.layernorm"].RMSNorm.forward_cuda,
                   cutedsl=dsv4.Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
                   raw=raw if raw is not None else (lambda idx: torch.cuda.current_stream(idx).cuda_stream))
    except Exception as e:  # noqa: BLE001 - any missing / changed internal keeps stock
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:400]})"
        return _STATE["detail"]
    _STATE["stock"] = stock
    _STATE["ok"] = True
    for attn in targets:
        attn._normalize_q_lora = types.MethodType(_wrapped, attn)   # the replica's own attribute; MQALayer untouched
    graph = int(levers.QN_GRAPH_MAX_ROWS)
    lo = max(1, int(getattr(levers, "QN_MIN_ROWS", 1)))
    _STATE["detail"] = (f"on: {len(targets)} attention replicas, eager EXTEND at {lo}..{levers.QN_MAX_ROWS} rows"
                        + (f" + decode / verify at 65..{graph} rows" if graph >= 65 else "")
                        + f", PDL {'on' if levers.QN_PDL else 'off'}; first eager call per key checked vs stock")
    return _STATE["detail"]


def _replica_attns(model, layers, layer_cls, stock) -> list:
    """The private MQALayer replicas of `layers` (never the served attention a replica layer may still share): each must
    resolve _normalize_q_lora to stock's MQALayer method, so binding the wrapper on it shadows exactly that method."""
    out = []
    for r, s in zip(layers or (), getattr(model, "layers", ())):
        attn = r._modules.get("self_attn") if hasattr(r, "_modules") else None
        served = s._modules.get("self_attn") if hasattr(s, "_modules") else None
        if (attn is None or attn is served or not isinstance(attn, layer_cls)
                or getattr(type(attn), "_normalize_q_lora", None) is not stock):
            continue
        out.append(attn)
    return out


def _served_linears(model, linear_fn) -> str:
    """'' when every served MQA attention's wq_b runs stock's partial(flashinfer_mxfp8_blockscaled_linear,
    backend='cute-dsl'): above 64 rows the unit hands that linear the quantized input its own quantize would make."""
    seen = 0
    for layer in getattr(model, "layers", ()):
        attn = layer._modules.get("self_attn") if hasattr(layer, "_modules") else None
        wq_b = attn._modules.get("wq_b") if attn is not None and hasattr(attn, "_modules") else None
        if wq_b is None:
            continue
        fn = getattr(getattr(wq_b, "quant_method", None), "w8a8_mxfp8_linear", None)
        if (not isinstance(fn, functools.partial) or fn.func is not linear_fn or fn.args
                or fn.keywords.get("backend") != "cute-dsl"):
            return f"served wq_b linear {fn!r:.120} is not partial(flashinfer_mxfp8_blockscaled_linear, cute-dsl)"
        seen += 1
    return "" if seen else "no served attention wq_b found"


def forward(fn, forward_batch, *args, **kwargs):
    """Run our model forward with lever QN's context set when the forward is eligible."""
    st = _gate(forward_batch)
    if st is None:
        return fn(*args, **kwargs)
    token = _ON.set(st)
    try:
        return fn(*args, **kwargs)
    finally:
        _ON.reset(token)


def _gate(forward_batch):
    """(lowest rows, highest rows, phase, PDL) for an eligible forward, else None."""
    if not _STATE["ok"]:
        return None
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or torch.compiler.is_compiling() or _FN["in_bcg"]():
        return None
    pdl = 1 if levers.QN_PDL else 0
    if mode == _FN["fm"].EXTEND:
        if torch.cuda.is_current_stream_capturing():
            return None
        st = (max(1, int(getattr(levers, "QN_MIN_ROWS", 1))), int(levers.QN_MAX_ROWS), "e", pdl)
    elif int(levers.QN_GRAPH_MAX_ROWS) >= 65 and (mode.is_decode() or mode.is_target_verify()):
        st = (65, int(levers.QN_GRAPH_MAX_ROWS), "g", pdl)
    else:
        return None
    if _FN["bi"]() or _FN["get_exec"]().deterministic.enable_deterministic_inference or _FN["fused_backend"]() is not None:
        return None
    return st


def _wrapped(self, q):
    """A replica's _normalize_q_lora (bound by install): the unit inside an eligible forward of ours, stock's method
    otherwise."""
    st = _ON.get()
    if st is not None and _STATE["ok"]:
        out = _fast(st, self, q)
        if out is not None:
            return out
    return _STATE["stock"](self, q)


def _layer_info(self):
    """Stock's fused-path conditions that do not depend on the call (deepseek_v4 _normalize_q_lora) plus q_norm's plain
    CUDA path above 64 rows: (attention, weight, eps, device, index), or (attention, None) for stock."""
    off = (self, None)
    try:
        if not getattr(self, "is_dsv41", False):
            return off
        norm = self.q_norm
        w = norm.weight
        if (type(norm) is not _FN["rmsnorm_cls"] or not isinstance(w, torch.Tensor) or w.dtype is not BF16
                or tuple(w.shape) != (K,) or not w.is_cuda or not w.is_contiguous() or w.data_ptr() & 15):
            return off
        if (norm.variance_size_override is not None or norm.cast_x_before_out_mul or not norm.has_weight
                or getattr(norm, "_quant_linear", None) is not None):
            return off
        bound = getattr(norm, "_forward_method", None)
        if bound is not None and getattr(bound, "__func__", None) is not _FN["rmsnorm_cuda"]:
            return off
        method = self.wq_b.quant_method
        if getattr(method, "mxfp8_dense_backend", None) != _FN["cutedsl"] or getattr(method, "use_marlin", False):
            return off
        # Fp8LinearMethod.apply's own mxfp8_view test: above 64 rows wq_b must take the MXFP8 branch for stock's bf16
        # input too (the same w8a8_mxfp8_linear, quantizing itself), never the block-FP8 kernel or the
        # Mxfp8SwizzledInput ValueError (stricter than stock's <= 64-row condition: such a layer simply keeps stock)
        if not (getattr(method, "use_mxfp8", False) or (getattr(method, "block_fp8_as_mxfp8", False)
                                                         and getattr(self.wq_b, "block_fp8_mxfp8_ready", False))):
            return off
        return (self, w, float(norm.variance_epsilon), w.device, w.get_device())
    except Exception:  # noqa: BLE001 - an unexpected layer keeps stock
        return off


def _fast(st, self, q):
    info = _LAYER.get(id(self))
    if info is None or info[0] is not self:
        info = _LAYER[id(self)] = _layer_info(self)
    if info[1] is None or q.dtype is not BF16 or q.dim() != 2:
        return None
    m, k = q.shape
    if k != K or m < st[0] or m > st[1] or q.stride(1) != 1:
        return None
    sx = q.stride(0)
    ptr = q.data_ptr()
    if sx < K or sx & 7 or ptr & 15 or q.get_device() != info[4]:
        return None
    mode = 1 if m <= 64 else 0
    key = (mode, m & 3, not m & 127, m > 128) if st[2] == "e" else ("g", m)
    ok = _OK.get(key)
    if ok:
        try:
            return _launch(info, ptr, sx, m, mode, st[3])
        except Exception as e:  # noqa: BLE001 - a failed launch: stock from now on, loudly
            _STATE["ok"] = False
            print(f"CACHEON_DSV4F lever QN: launch at {m} rows failed ({type(e).__name__}: {str(e)[:200]}): "
                  f"lever QN OFF", file=sys.stderr, flush=True)
            return None
    if ok is None and not torch.cuda.is_current_stream_capturing():
        return _check(self, q, info, ptr, sx, m, mode, key, st[3])
    return None


def _launch(info, ptr, sx, m, mode, pdl):
    _attn, w, eps, dev, idx = info
    pm = (m + 127) & -128
    y = torch.empty_strided((m, K), (K, 1), dtype=BF16, device=dev)
    q8 = torch.empty_strided((m, K), (K, 1), dtype=FP8, device=dev)
    sf = torch.empty_strided((pm * (K // 32),), (1,), dtype=U8, device=dev)
    _FN["run"](ptr, sx, w.data_ptr(), y.data_ptr(), q8.data_ptr(), sf.data_ptr(), m, pm, eps, mode, pdl,
               _FN["raw"](idx))
    _STATE["fast"][mode] += 1
    return y, _FN["mxi"](q8, sf)


def _check(self, q, info, ptr, sx, m, mode, key, pdl):
    """Stock and the unit on the same input; the unit's result if y, q and every scale byte equal stock's (key on),
    stock's otherwise (lever off for the process)."""
    want = _STATE["stock"](self, q)
    why = ""
    try:
        got = _launch(info, ptr, sx, m, mode, pdl)
        y_ref = want[0]
        if mode == 0:
            # stock above 64 rows: (y, y), wq_b quantizes y itself (flashinfer_mxfp8_blockscaled_linear)
            if want[1] is not y_ref:
                raise RuntimeError(f"stock returned {type(want[1]).__name__} for wq_b")
            q_ref, s_ref = _FN["quant"](y_ref.view(-1, K), is_sf_swizzled_layout=True, alignment=32)
        else:
            if not isinstance(want[1], _FN["mxi"]):
                raise RuntimeError(f"stock returned {type(want[1]).__name__} for wq_b")
            q_ref, s_ref = want[1]
        y, (q8, sf) = got
        if (y_ref.dtype is not BF16 or tuple(y_ref.shape) != (m, K) or y_ref.stride() != y.stride()
                or not torch.equal(y.view(torch.int16), y_ref.view(torch.int16))):
            why = "y"
        elif q_ref.dtype is not FP8 or tuple(q_ref.shape) != (m, K) or not torch.equal(q8.view(U8), q_ref.view(U8)):
            why = "q"
        elif s_ref.dtype is not U8 or s_ref.numel() != sf.numel() or not torch.equal(sf, s_ref.reshape(-1)):
            why = "scales"
    except Exception as e:  # noqa: BLE001 - any failure keeps stock
        why = f"{type(e).__name__}: {str(e)[:200]}"
    _STATE["checks"] += 1
    ok = not why
    _OK[key] = ok
    if not ok:
        _STATE["ok"] = False
        print(f"CACHEON_DSV4F lever QN: {key} at {m} rows: {why} differs from stock: lever QN OFF", file=sys.stderr,
              flush=True)
        return want
    levers.announce(f"qn-{key}", f"lever QN: key {key} first at {m} rows: y / q / scales == stock: on")
    return got


def summary() -> str:
    return (f"{_STATE['fast'][0]} mode-0 + {_STATE['fast'][1]} mode-1 fast calls, {_STATE['checks']} checks, "
            f"{'on' if _STATE['ok'] else 'off'}")
