"""Lever HQ (CUDA unit hq_unit + generated dsv41f/hq_layer.py): inside OUR model forward, the attention
hyper-connection's combine of calls above 128 rows -- stock's hc_combine_norm (four-stream combine + RMSNorm, bf16 rows)
followed by wqkv_a's own MXFP8 quantize of those rows -- as ONE launch whose y (bf16), q (e4m3) and scales (UE8M0,
FlashInfer's 128x4 swizzled layout, every byte of the padded rows zero) equal stock's bytes for the call's row count;
wqkv_a takes them as the Mxfp8SwizzledInput stock's own fused path hands it at <= 128 rows.

Stock (fork 5e72ce5, deepseek_v4.py / deepseek_v4_mhc.py): DeepseekV4DecoderLayer.forward_hc_pre_from_prev ->
run_attn_hc -> mhc.combine(attn_hc, state, quantized) -> _combine: at <= 128 rows hc_combine_norm_mxfp8 (one Triton
kernel; `quantized` gets its Mxfp8SwizzledInput), above that hc_combine_norm (Triton _hc_combine_norm, 8 warps, below 4096
rows; _hc_combine_norm_prefill, 4 warps, from 4096) and the attention's wqkv_a (Fp8LinearMethod.apply ->
flashinfer_mxfp8_blockscaled_linear, or lever DT's / FL's launch of it) quantizes the bf16 rows itself
(flashinfer_mxfp8_quantize: FlashInfer 0.7.1's CuTe-DSL MXFP8QuantizeSwizzledKernel). The two Triton kernels reduce in
different fp32 trees; the unit has one mode per kernel (hq_unit.cu header: the trees as Triton 3.8.0 compiles them for
sm_103, read back from the references' PTX by tests/hq_test.py) and FlashInfer's quantize arithmetic.

Why: decode / verify batches of 129..384 rows (22..64 requests x 6 verify rows) run the pair inside their CUDA graphs on
the critical path of every layer (_hc_combine_norm, then mxfp8_quantize, then the wqkv_a GEMM); eager EXTEND forwards
(host-bound) pay a Triton launch plus the quantize call chain. One kernel launch replaces both.

Binding: install() binds _layer_forward as the forward_hc_pre_from_prev instance attribute of OUR private layer
replicas (model_node._replica's copies; the lifted layer loop calls layers[i].forward_hc_pre_from_prev), never on
DeepseekV4DecoderLayer or a served module: cacheon's node adapter seals every callable of the served node's modules and
of their classes at bind time and refuses an audited call once one was rebound. _layer_forward runs stock's method
unless lever HQ is active in this forward; then the lifted copy hq_layer.forward_hc_pre_from_prev (generated from the
fork's source by gen_hq.py with two asserted edits: its attention combine calls combine() below, and
can_fuse_all_reduce is looked up in deepseek_v4 at call time -- where stock's method finds it, lever AR's gate
included). No sglang function or class is patched. Lever FG binds its own lifted forward_hc_pre_from_prev (fg_layer.py)
on the same replicas at the end of prepare, after this install: gen_hq run on a bundle with FG gives fg_layer.py the
same combine edit, so those replicas reach combine() below through FG's forward; a FG generated after HQ without that
edit shadows HQ -- forward() reads the bindings once and announces any replica whose forward does not call hq.combine.
combine() runs stock's mhc.combine for every call it does not take, installed or not.

Scope: only inside our model forward (forward() sets the context): decode / verify forwards (CUDA graphs) with
attention combines of 129..HQ_MAX_ROWS rows, and with HQ_EAGER eager EXTEND forwards (no capture, not inside the
breakable prefill graph or torch.compile) of 129..HQ_EAGER_MAX_ROWS rows (in this arena's eager prefill only the
Engram layers' combines and calls of <= 384 rows reach _combine: the mega-mHC posts hand the other layers normed rows).
Stock's code runs for everything else: other modes, deterministic / batch-invariant inference, a state already
carrying the next input (stock's shortcut), the first layer (no pre), a combine stock would not quantize, an
hyper-connection outside stock's fused-path conditions (4 streams of 5120, bf16 contiguous norm weight, plain norm), a
call of another dtype / stride / alignment, a pre that is not fp32, a served wqkv_a whose linear is not stock's
partial(flashinfer_mxfp8_blockscaled_linear, backend='cute-dsl') (the consumer whose own quantize the unit stands in for).

Exactness: tests/hq_test.py (B300, fork image) sweeps every M = 129..4096 and samples to 16384 against stock's pair byte
for byte (y, q, every scale byte including the padding the consumer GEMM reads), on realistic and association-sensitive
rows (with a control build of another tree that must differ), and checks the wqkv_a GEMM output from the unit's input
equals stock's (pinned, lever-DT and prefill-autotuned paths). In the engine the first eager call of every check key
(graph phase: each decode / verify row count; eager: kernel mode, whole 128-row tiles, strides, eps) runs stock and the
unit on the same input and compares y, q and every scale byte: one difference turns the lever off for the process.
install() pins the source (sha256 of the AST segment) of every stock function the unit or this module mirrors or relies
on, and the FlashInfer / CuTe-DSL / Triton versions whose code generation the unit's arithmetic was taken from; any
difference keeps stock.

Measured (tests/hq_test.py, B300 box1, fork image; fork/results/hq/): 4253 hq.combine calls over every M 129..4096 + 29
samples to 16384 (realistic, extreme, -0 and strided rows) and 857k + 53k association-sensitive rows byte-identical to
stock (y, q, every scale byte including the padding; the chain-order control build differs in ~1.7k rows); the stock
kernels' PTX trees equal the unit's; wqkv_a's GEMM output from the unit's input == stock's (pinned, DT-forced and
prefill-autotuned paths); the bound replica forward == stock's method end to end. GPU us per call (CUDA graph, 20 calls)
stock pair -> unit: 4.31 -> 2.96 at 144 rows, 5.36 -> 3.51 at 192, 6.05 -> 3.53 at 288, 6.58 -> 4.68 at 384; with
wqkv_a's GEMM 11.32 -> 10.29, 12.35 -> 10.38, 12.85 -> 11.44, 13.64 -> 11.81; eager host us per call 75 -> 24.
"""
import ast
import contextvars
import functools
import hashlib
import inspect
import sys
import textwrap
import types

import torch

from dsv41f import levers

H = 5120
STREAMS = 4
SF_TILE = 128 * (H // 32)       # bytes of one 128-row scale tile
BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
U8 = torch.uint8
F32 = torch.float32
_ON = contextvars.ContextVar("dsv41f_hq", default=None)
_STATE = {"detail": "not installed", "ok": False, "fast": [0, 0], "checks": 0, "stock": None, "targets": [],
          "shadow": None}
_HC: dict = {}            # id(attn_hc) -> (attn_hc, norm weight, eps, device index) or (attn_hc, None, None, None)
_OK: dict = {}            # check key -> True (bytes == stock) / False
_FN: dict = {}            # run, mhc, quant, mxi, fm, in_bcg, bi, get_exec, lifted, raw

VERSIONS = {"flashinfer": "0.7.1", "cutlass": "4.8.0", "triton": "3.8.0"}

LAYER_PIN = ("sglang.srt.models.deepseek_v4", "DeepseekV4DecoderLayer.forward_hc_pre_from_prev")
# sha256[:16] of the source segment of every function the unit mirrors or this module relies on, read from the fork's
# SGLang (5e72ce5) and FlashInfer 0.7.1 (0b46115d); any difference: lever HQ off
PINNED = {
    LAYER_PIN: "5314f75a71b7221a",
    ("sglang.srt.models.deepseek_v4", "MQALayer.accepts_mxfp8_swizzled_input"): "c2d54c194c68c34b",
    ("sglang.srt.models.deepseek_v4", "MQALayer._forward_prepare_multi_stream"): "5976621b797b4cb6",
    ("sglang.srt.models.deepseek_v4", "MQALayer._forward_prepare_low_ratio_multi_stream"): "0a53b96e3e670175",
    ("sglang.srt.models.deepseek_v4", "MQALayer._forward_prepare"): "a909d336b2e5c295",
    ("sglang.srt.models.deepseek_v4_mhc", "combine"): "db09783e2a3b863c",
    ("sglang.srt.models.deepseek_v4_mhc", "_combine"): "9a26d30ee6bf878d",
    ("sglang.srt.models.deepseek_v4_mhc", "HcState"): "143e3166f04af59f",
    ("sglang.srt.models.deepseek_v4_mhc", "HcSubLayer"): "6b020373726eca20",
    ("sglang.srt.models.deepseek_v4_mhc", "HcConfig"): "9a0c8a8b2036767d",
    ("sglang.srt.models.deepseek_v4_mhc", "HcNormed"): "cc032aff6e8b11fa",
    ("sglang.srt.models.deepseek_v4_mhc", "HcQuantized"): "9a01a0883a7b9a42",
    ("sglang.kernels.ops.layernorm.hc_combine_norm", "_hc_combine_norm"): "1fa82de1c3d35309",
    ("sglang.kernels.ops.layernorm.hc_combine_norm", "_hc_combine_norm_prefill"): "24e5da900b033d08",
    ("sglang.kernels.ops.layernorm.hc_combine_norm", "hc_combine_norm"): "f422051e10fa54bd",
    ("sglang.kernels.ops.layernorm.hc_combine_norm", "hc_combine_norm_mxfp8"): "96baeaf8035231ec",
    ("sglang.kernels.ops.layernorm.mxfp8_epilogue", "ue8m0_scale"): "4d2537c046eb29a8",
    ("sglang.srt.layers.quantization.mxfp8_input", "Mxfp8SwizzledInput"): "54c48c829cc789f1",
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_quantize"): "01a12afdc8ca1435",
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_blockscaled_linear"): "99da4a08ba42d542",
    ("sglang.srt.layers.quantization.fp8_utils", "_mxfp8_decode_tactic"): "bdb231795cc1f6f8",
    ("sglang.srt.layers.quantization.fp8_utils", "_cute_dsl_mm_mxfp8_with_tactic"): "ced323827fb15816",
    ("sglang.srt.layers.quantization.fp8", "Fp8LinearMethod.apply"): "24ab0721ac46abe0",
    ("sglang.srt.layers.linear", "ReplicatedLinear.forward"): "0c5ebd6bd25e3a93",
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
    import flashinfer.quantization.fp8_quantization  # noqa: F401
    import flashinfer.quantization.kernels.mxfp8_quantize  # noqa: F401
    import flashinfer.quantization.quantization_cute_dsl_utils  # noqa: F401
    import sglang.kernels.ops.layernorm.hc_combine_norm  # noqa: F401
    import sglang.kernels.ops.layernorm.mxfp8_epilogue  # noqa: F401
    import sglang.srt.layers.linear  # noqa: F401
    import sglang.srt.layers.quantization.fp8  # noqa: F401
    import sglang.srt.layers.quantization.fp8_utils  # noqa: F401
    import sglang.srt.layers.quantization.mxfp8_input  # noqa: F401
    import sglang.srt.models.deepseek_v4  # noqa: F401
    import sglang.srt.models.deepseek_v4_mhc  # noqa: F401
    return {name: sys.modules[name] for name, _ in PINNED}


def unit():
    if "run" not in _FN:
        try:
            import hq_unit
        except ImportError as err:
            levers.announce("hq-missing", f"bundle CUDA unit hq_unit not importable ({err}): lever HQ off")
            return None
        _FN["run"] = hq_unit.run
    return _FN["run"]


def install(model, layers=None) -> str:
    """Pin the stock internals, check the replicas' wqkv_a linears and bind _layer_forward on the private layer replicas
    of `layers` (model_node.prepare's replicas, aligned with model.layers)."""
    if _STATE["stock"] is not None or _STATE["detail"].startswith("off"):
        return _STATE["detail"]
    if not getattr(levers, "HQ", False):
        _STATE["detail"] = "off (levers.HQ)"
        return _STATE["detail"]
    try:
        import cutlass
        import flashinfer
        import triton
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.runtime_context import get_exec

        from dsv41f import hq_layer

        have = {"flashinfer": flashinfer.__version__, "cutlass": cutlass.__version__, "triton": triton.__version__}
        if have != VERSIONS:
            raise RuntimeError(f"versions {have} (unit matched against {VERSIONS})")
        mods = _modules()
        cache: dict = {}
        bad = [f"{mod}.{name} {got}" for (mod, name), want in PINNED.items()
               if (got := source_hash(mods[mod], name, cache)) != want]
        if bad:
            raise RuntimeError("source differs from the mirror: " + ", ".join(bad))
        if getattr(hq_layer, "STOCK_HASH", None) != PINNED[LAYER_PIN]:
            raise RuntimeError(f"hq_layer was lifted from another forward_hc_pre_from_prev "
                               f"({getattr(hq_layer, 'STOCK_HASH', None)})")
        dsv4 = mods["sglang.srt.models.deepseek_v4"]
        mhc = mods["sglang.srt.models.deepseek_v4_mhc"]
        fp8_utils = mods["sglang.srt.layers.quantization.fp8_utils"]
        if getattr(dsv4, "mhc", None) is not mhc or hq_layer.mhc is not mhc or hq_layer._dsv4 is not dsv4:
            raise RuntimeError("deepseek_v4.mhc / hq_layer's modules are not the pinned deepseek_v4_mhc / deepseek_v4")
        if dsv4._is_hip or mhc._is_hip or not dsv4._is_cuda or not dsv4.get_platform().is_blackwell:
            raise RuntimeError("not CUDA Blackwell")
        combine_fn = mhc.combine
        if getattr(combine_fn, "__module__", None) != mhc.__name__ or getattr(combine_fn, "__name__", None) != "combine":
            raise RuntimeError(f"deepseek_v4_mhc.combine is rebound ({combine_fn!r:.120})")
        layer_cls = dsv4.DeepseekV4DecoderLayer
        stock = layer_cls.forward_hc_pre_from_prev
        targets = _replica_layers(model, layers, layer_cls, stock)
        if not targets:
            raise RuntimeError("no private DeepseekV4DecoderLayer replica to bind on (lever HQ never wraps a served "
                               "module)")
        why = (_replica_wqkv(targets, dsv4.Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL)
               or _served_wqkv(model, layers, targets, fp8_utils.flashinfer_mxfp8_blockscaled_linear))
        if why:
            raise RuntimeError(why)
        if unit() is None:
            raise RuntimeError("unit hq_unit missing")
        raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        _FN.update(mhc=mhc, quant=fp8_utils.flashinfer_mxfp8_quantize, mxi=Mxfp8SwizzledInput, fm=ForwardMode,
                   in_bcg=is_in_breakable_cuda_graph, bi=is_batch_invariant_mode_enabled, get_exec=get_exec,
                   lifted=hq_layer.forward_hc_pre_from_prev,
                   raw=raw if raw is not None else (lambda idx: torch.cuda.current_stream(idx).cuda_stream))
    except Exception as e:  # noqa: BLE001 - any missing / changed internal keeps stock
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:400]})"
        return _STATE["detail"]
    _STATE["stock"] = stock
    _STATE["ok"] = True
    _STATE["targets"] = list(targets)
    for layer in targets:
        layer.forward_hc_pre_from_prev = types.MethodType(_layer_forward, layer)   # the replica's own attribute
    graph = int(levers.HQ_MAX_ROWS)
    pdl = {0: "off", 1: "on, dependents released after the wait"}.get(levers.HQ_PDL, "on, released after the stores")
    _STATE["detail"] = (f"on: {len(targets)} layer replicas, attention combines of decode / verify at 129..{graph} rows"
                        + (f" + eager EXTEND at 129..{levers.HQ_EAGER_MAX_ROWS} rows" if levers.HQ_EAGER else "")
                        + f", PDL {pdl}; first eager call per key checked vs stock")
    return _STATE["detail"]


def _replica_layers(model, layers, layer_cls, stock) -> list:
    """The private DeepseekV4DecoderLayer replicas of `layers` (never a served layer): each must resolve
    forward_hc_pre_from_prev to stock's method, so the bound attribute shadows exactly that method."""
    out = []
    for r, s in zip(layers or (), getattr(model, "layers", ())):
        if r is s or not isinstance(r, layer_cls) or getattr(type(r), "forward_hc_pre_from_prev", None) is not stock:
            continue
        bound = getattr(r, "forward_hc_pre_from_prev", None)
        if getattr(bound, "__func__", None) is not stock or getattr(bound, "__self__", None) is not r:
            continue
        out.append(r)
    return out


def _replica_wqkv(targets, cutedsl) -> str:
    """'' when every replica attention's wqkv_a is an MXFP8 Fp8LinearMethod on FlashInfer's CuTe-DSL backend (stock's
    accepts_mxfp8_swizzled_input conditions, backend narrowed to the one the GEMM equivalence was checked on)."""
    for layer in targets:
        attn = layer._modules.get("self_attn") if hasattr(layer, "_modules") else None
        lin = attn._modules.get("wqkv_a") if attn is not None and hasattr(attn, "_modules") else None
        if lin is None:
            return f"layer {getattr(layer, 'layer_id', '?')}: attention without a fused wqkv_a"
        method = getattr(lin, "quant_method", None)
        if (type(method).__name__ != "Fp8LinearMethod" or getattr(method, "mxfp8_dense_backend", None) != cutedsl
                or getattr(method, "use_marlin", False)
                or not (getattr(method, "use_mxfp8", False) or (getattr(method, "block_fp8_as_mxfp8", False)
                                                                and getattr(lin, "block_fp8_mxfp8_ready", False)))):
            return f"layer {getattr(layer, 'layer_id', '?')}: wqkv_a is not a CuTe-DSL MXFP8 linear ({method!r:.80})"
    return ""


def _served_wqkv(model, layers, targets, linear_fn) -> str:
    """'' when the served wqkv_a of every target's layer runs stock's partial(flashinfer_mxfp8_blockscaled_linear,
    backend='cute-dsl'): the consumer whose own quantize (flashinfer_mxfp8_quantize) the unit's q / scales and the
    first-call checks stand in for; the replicas' wqkv_a (levers DT / FL) wrap exactly that callable."""
    want = {id(t) for t in targets}
    seen = 0
    for r, s in zip(layers or (), getattr(model, "layers", ())):
        if id(r) not in want:
            continue
        attn = s._modules.get("self_attn") if hasattr(s, "_modules") else None
        lin = attn._modules.get("wqkv_a") if attn is not None and hasattr(attn, "_modules") else None
        fn = getattr(getattr(lin, "quant_method", None), "w8a8_mxfp8_linear", None)
        if (not isinstance(fn, functools.partial) or fn.func is not linear_fn or fn.args
                or fn.keywords.get("backend") != "cute-dsl"):
            return (f"served layer {getattr(s, 'layer_id', '?')}: wqkv_a linear {fn!r:.120} is not "
                    f"partial(flashinfer_mxfp8_blockscaled_linear, backend='cute-dsl')")
        seen += 1
    return "" if seen == len(want) else f"{len(want) - seen} replicas without a served layer"


def _reaches_combine(fn) -> bool:
    """Whether a replica's bound forward_hc_pre_from_prev runs combine() below for the attention combine: HQ's
    _layer_forward, or a lifted forward whose module binds this module as `hq` and whose source calls
    hq.combine(self.attn_hc, ...) and never mhc.combine(self.attn_hc, ...) -- gen_hq's hq_layer.py, or lever FG's
    fg_layer.py given HQ's edit by gen_hq (or by hand)."""
    if fn is _layer_forward:
        return True
    mod = sys.modules.get(getattr(fn, "__module__", None) or "")
    if getattr(getattr(mod, "hq", None), "combine", None) is not combine:
        return False
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001 - no source to read: reported as not reaching HQ
        return False
    calls = [n.func.value.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "combine"
             and isinstance(n.func.value, ast.Name) and n.args and isinstance(n.args[0], ast.Attribute)
             and n.args[0].attr == "attn_hc"]
    return "hq" in calls and "mhc" not in calls


def _shadow_check() -> None:
    """Once, at the first forward after a successful install: a lever that binds its own lifted
    forward_hc_pre_from_prev on the same replicas after install() (lever FG's fg_layer.py, generated after gen_hq
    without HQ's edit) silently shadows HQ there. Announced only: such a replica never reaches combine(), so nothing
    else changes."""
    lost = {}
    seen = {}
    for layer in _STATE["targets"]:
        fn = getattr(getattr(layer, "forward_hc_pre_from_prev", None), "__func__", None)
        ok = seen.get(id(fn))
        if ok is None:
            ok = seen[id(fn)] = _reaches_combine(fn)
        if not ok:
            name = getattr(fn, "__module__", None) or "?"
            lost[name] = lost.get(name, 0) + 1
    _STATE["shadow"] = sum(lost.values())
    if lost:
        levers.announce("hq-shadowed", f"lever HQ: {_STATE['shadow']} of {len(_STATE['targets'])} layer replicas run "
                        f"another lever's forward_hc_pre_from_prev, which does not call hq.combine ({lost}): no HQ "
                        f"there (run gen_hq after that lever's generator)")


def forward(fn, forward_batch, *args, **kwargs):
    """Run our model forward with lever HQ's context set when the forward is eligible."""
    if _STATE["shadow"] is None and _STATE["ok"]:
        _shadow_check()
    st = _gate(forward_batch)
    if st is None:
        return fn(*args, **kwargs)
    token = _ON.set(st)
    try:
        return fn(*args, **kwargs)
    finally:
        _ON.reset(token)


def _gate(forward_batch):
    """(phase "g" / "e", highest rows, PDL) for an eligible forward, else None."""
    if not _STATE["ok"]:
        return None
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or torch.compiler.is_compiling() or _FN["in_bcg"]():
        return None
    pdl = int(levers.HQ_PDL)
    pdl = pdl if pdl in (0, 1, 2) else 2
    if mode == _FN["fm"].EXTEND:
        if not levers.HQ_EAGER or torch.cuda.is_current_stream_capturing():
            return None
        st = ("e", int(levers.HQ_EAGER_MAX_ROWS), pdl)
    elif int(levers.HQ_MAX_ROWS) >= 129 and (mode.is_decode() or mode.is_target_verify()):
        st = ("g", int(levers.HQ_MAX_ROWS), pdl)
    else:
        return None
    if _FN["bi"]() or _FN["get_exec"]().deterministic.enable_deterministic_inference:
        return None
    return st


def _layer_forward(self, *args, **kwargs):
    """A replica's forward_hc_pre_from_prev (bound by install): the lifted copy inside an eligible forward of ours,
    stock's method otherwise."""
    if _ON.get() is None:
        return _STATE["stock"](self, *args, **kwargs)
    return _FN["lifted"](self, *args, **kwargs)


def combine(hc, state, quantized):
    """The lifted forward_hc_pre_from_prev's attention combine: the unit for an eligible call, stock's mhc.combine
    otherwise (the same arguments; its result and `quantized` exactly as stock leaves them). Also reached from a lifted
    forward HQ did not bind (lever FG's fg_layer, generated with HQ's edit) in every forward, before or without a
    successful install: stock's deepseek_v4_mhc.combine then."""
    st = _ON.get()
    if st is not None and type(quantized) is list and not quantized and state.input is None and _STATE["ok"]:
        out = _fast(st, hc, state, quantized)
        if out is not None:
            return out
    mhc = _FN.get("mhc")
    if mhc is None:
        from sglang.srt.models import deepseek_v4_mhc as mhc
    return mhc.combine(hc, state, quantized)


def _hc_info(hc):
    """Stock's fused-path conditions of the hyper-connection that do not depend on the call (_combine):
    (hc, norm weight, eps, device index), or (hc, None, None, None) for stock."""
    off = (hc, None, None, None)
    try:
        cfg, norm = hc.cfg, hc.norm
        if cfg.mult != STREAMS or cfg.hidden != H:
            return off
        if norm.cast_x_before_out_mul or norm.variance_size_override is not None:
            return off
        w = norm.weight
        if (not isinstance(w, torch.Tensor) or w.dtype is not BF16 or tuple(w.shape) != (H,) or not w.is_cuda
                or not w.is_contiguous() or w.data_ptr() & 15):
            return off
        eps = norm.variance_epsilon
        if type(eps) is not float:
            return off
        return (hc, w, eps, w.get_device())
    except Exception:  # noqa: BLE001 - an unexpected hyper-connection keeps stock
        return off


def _fast(st, hc, state, quantized):
    x, pre = state.streams, state.pre
    if type(x) is not torch.Tensor or type(pre) is not torch.Tensor:
        return None
    info = _HC.get(id(hc))
    if info is None or info[0] is not hc or (info[1] is not None and hc.norm.weight is not info[1]):
        info = _HC[id(hc)] = _hc_info(hc)
    if info[1] is None or x.dtype is not BF16 or x.dim() != 3:
        return None
    m, mult, hid = x.shape
    if mult != STREAMS or hid != H or m < 129 or m > st[1]:
        return None
    x_flat = x.flatten(1)
    if (x_flat.stride(1) != 1 or pre.dtype is not F32 or pre.dim() != 2 or pre.shape[0] != m or pre.shape[1] != STREAMS
            or pre.stride(1) != 1):
        return None
    sx, sp = x_flat.stride(0), pre.stride(0)
    if (sx & 7 or sx < STREAMS * H or sp < STREAMS or (x_flat.data_ptr() | pre.data_ptr()) & 15
            or x_flat.get_device() != info[3] or pre.get_device() != info[3]):
        return None
    warps = 4 if m >= 4096 else 8
    key = (("g", m) if st[0] == "g" else ("e", warps, not m & 127)) + (sx, sp, info[2])
    ok = _OK.get(key)
    if ok:
        try:
            y, q, s = _launch(info, x_flat, pre, m, warps, st[2])
        except Exception as e:  # noqa: BLE001 - a failed launch: stock from now on, loudly
            _STATE["ok"] = False
            print(f"CACHEON_DSV4F lever HQ: launch at {m} rows failed ({type(e).__name__}: {str(e)[:200]}): "
                  f"lever HQ OFF", file=sys.stderr, flush=True)
            return None
        quantized.append(_FN["mxi"](q, s))
        _STATE["fast"][warps == 4] += 1
        return y
    if ok is None and not torch.cuda.is_current_stream_capturing():
        return _check(hc, state, quantized, info, x_flat, pre, m, warps, st[2], key)
    return None


def _launch(info, x_flat, pre, m, warps, pdl):
    _hc, w, eps, idx = info
    dev = x_flat.device
    y = torch.empty((m, H), dtype=BF16, device=dev)
    q = torch.empty((m, H), dtype=FP8, device=dev)
    s = torch.empty((((m + 127) >> 7) * SF_TILE,), dtype=U8, device=dev)
    _FN["run"](x_flat.data_ptr(), x_flat.stride(0), pre.data_ptr(), pre.stride(0), w.data_ptr(), y.data_ptr(),
               q.data_ptr(), s.data_ptr(), m, eps, warps, pdl, _FN["raw"](idx))
    return y, q, s


def _check(hc, state, quantized, info, x_flat, pre, m, warps, pdl, key):
    """Stock and the unit on the same input; the unit's result if y, q and every scale byte equal stock's (key on),
    stock's otherwise (lever off for the process)."""
    ref_list = []
    y_ref = _FN["mhc"].combine(hc, state, ref_list)     # stock's combine: bf16 rows above 128 (nothing appended)
    why = ""
    got = None
    try:
        if ref_list:
            raise RuntimeError(f"stock handed wqkv_a {type(ref_list[0]).__name__} at {m} rows")
        # what the attention's wqkv_a computes from those rows: flashinfer_mxfp8_blockscaled_linear's own quantize
        q_ref, s_ref = _FN["quant"](y_ref.view(-1, H), is_sf_swizzled_layout=True, alignment=32)
        got = _launch(info, x_flat, pre, m, warps, pdl)
        y, q, s = got
        if (y_ref.dtype is not BF16 or tuple(y_ref.shape) != (m, H) or y_ref.stride() != y.stride()
                or not torch.equal(y.view(torch.int16), y_ref.view(torch.int16))):
            why = "y"
        elif (q_ref.dtype is not FP8 or tuple(q_ref.shape) != (m, H) or q_ref.stride() != q.stride()
              or not torch.equal(q.view(U8), q_ref.view(U8))):
            why = "q"
        elif s_ref.dtype is not U8 or s_ref.numel() != s.numel() or not torch.equal(s, s_ref.reshape(-1)):
            why = "scales"
    except Exception as e:  # noqa: BLE001 - any failure keeps stock
        why = f"{type(e).__name__}: {str(e)[:200]}"
    _STATE["checks"] += 1
    ok = not why
    _OK[key] = ok
    if not ok:
        _STATE["ok"] = False
        print(f"CACHEON_DSV4F lever HQ: {key} at {m} rows: {why} differs from stock: lever HQ OFF", file=sys.stderr,
              flush=True)
        return y_ref
    levers.announce(f"hq-{key}", f"lever HQ: key {key} first at {m} rows: y / q / scales == stock: on")
    quantized.append(_FN["mxi"](got[1], got[2]))
    _STATE["fast"][warps == 4] += 1
    return got[0]


def summary() -> str:
    return (f"{_STATE['fast'][0]} 8-warp + {_STATE['fast'][1]} 4-warp fast calls, {_STATE['checks']} checks, "
            f"{'on' if _STATE['ok'] else 'off'}")
