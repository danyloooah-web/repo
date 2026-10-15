"""Lever FL on the fork (levers.FL): the host path of our layer replicas' dense MXFP8 linears in EAGER EXTEND forwards
(the checkpoint's 32x32 ue8m0 block-FP8 weights served as MXFP8 on FlashInfer 0.7.1's CuTe-DSL backend: wqkv_a, wq_b,
wo_b, the shared expert's gate_up / down, the indexer's wq_b, Engram's wkv).

Why: an eager prefill of up to ~11k rows is host-bound, and each of these linears costs ~270 us of Python before its two
kernels launch: SGLang's torch custom op sglang::flashinfer_mxfp8_quantize -> flashinfer.mxfp8_quantize ->
mxfp8_quantize_cute_dsl, then sglang::flashinfer_mm_mxfp8 -> Mxfp8DispatchCache -> mm_mxfp8 (backend checks, runner
factory, AutoTuner.choose_one) -> CuteDSLMxfp8GemmRunner.forward, or the pinned variants (_mxfp8_decode_tactic, an
autotune(False, skip_ops) context per call).

What runs instead is exactly what stock launches (FlashInfer 0.7.1 / the fork's SGLang, pinned by source hash below):
  quantize  the compiled kernel _get_compiled_kernel_mxfp8_swizzled returns for stock's arguments (bf16, K, PDL, 2T/SF
            for this M, 128x4 layout, no SM107 small blocks) on stock's (m, padded_m, num_blocks), into stock's buffers.
  GEMM      the compiled kernel CuteDSLMxfp8GemmRunner.forward takes from _CUTE_DSL_MM_MXFP8_KERNEL_CACHE for the tactic
            stock's call resolves, with the runner's operand order, sf_m / sf_n / sf_k, scale pointers and alpha, into
            mm_mxfp8's output allocation.
Plans are learned from stock, never guessed. A plan key is (pin_tactic, decode class (pinned) or AutoTuner bucket (tuned),
2T/SF of the quantize), per weight shape (N, K) and shared by all layers: everything stock decides from M is constant
inside a key for M > 32 (split-K / narrow tiles only exist below 33 rows), and nothing it decides depends on the weight
values or pointers. The first call of every (shape, key) runs stock with FlashInfer's two kernel getters wrapped by recorders
(for that one call), builds the plan from the recorded launches, runs the fast path once through recording proxies and
requires the same kernel objects, the same scalar arguments, the same tensor roles / shapes / strides / dtypes and
byte-identical outputs; anything else turns the key off for the process (stock from then on). The learn call returns
stock's output.

Scope: only inside our model forward (forward() sets the context) and only for forwards whose mode is in levers.FL_MODES,
outside CUDA-graph capture, outside the breakable prefill graph, outside torch.compile and while FlashInfer's AutoTuner
is not tuning (no autotune_v2 store attached, mm_mxfp8's tuning config the one read at install); a change of the
AutoTuner's caches drops the tuned-bucket plans. Calls of fewer than FL_MIN_ROWS rows,
calls lever DT takes (its bands, no kwargs), calls with a bias, other weights, unexpected keyword arguments, layouts or
dtypes go to the previous w8a8_mxfp8_linear (lever DT's, or stock's) unchanged.

Prefill autotune pin: the replicas' Fp8LinearMethod copies (DT's, FL's) are made at prepare, before the model runner's
startup prefill autotune sets mxfp8_prefill_autotune_min_tokens on the served methods. FL's own copies (of the linears
the replica shared with the served model: wq_b, the indexer's wq_b, the shared expert's down, Engram's wkv) stand in
for the served objects, so forward() copies the served value onto them at the start of EVERY forward (breakable-graph
captures and decode / verify included; attribute copies only): without it the >= 4096-token breakable prefill graphs,
captured before the first eligible forward, would run those linears pinned where f47a / stock run the tuned tactic.
Lever DT's copies take the served value at the start of each eligible-mode eager forward only, so >= min_tokens-row
eager calls carry the same pin_tactic as the served model's while graphs captured before that keep what f47a captured.
"""
import ast
import contextvars
import copy
import functools
import hashlib
import inspect

import torch

from dsv41f import dt, levers

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
U8 = torch.uint8
_ON = contextvars.ContextVar("dsv41f_fl", default=None)
_STATE = {"detail": "not installed", "fi": None, "token": None, "own": [], "dt": [], "linears": [],
          "own_synced": False, "dt_synced": False, "learned": 0, "refused": 0}
_MIN_ROWS = [1 << 62]
# (N, K, device index) -> {plan key: plan | False}, shared by every replica linear of that weight shape: stock's choices
# depend on the shapes, dtypes and the AutoTuner state only (never on the weight values or pointers), so a key learned
# and checked on one layer's linear holds for every layer's linear of the same shape
_PLANS: dict = {}

# sha256[:16] of the source of every function whose behaviour the fast path mirrors or relies on (module, qualified
# name), read from FlashInfer 0.7.1 (commit 0b46115d) and the fork's SGLang (5e72ce5); any difference: lever FL off
PINNED = {
    ("flashinfer.gemm.gemm_base", "mm_mxfp8"): "98745e2977e434ba",
    ("flashinfer.gemm.gemm_base", "_cute_dsl_gemm_mxfp8_runner"): "12f24e74f5a111f1",
    ("flashinfer.gemm.gemm_mm_fp4_cute_dsl", "_compile_block_scaled_gemm"): "5c7badfe114e1d35",
    ("flashinfer.gemm.gemm_mm_fp4_cute_dsl", "_prepare_alpha_for_launch"): "3580151641e55000",
    ("flashinfer.quantization.kernels.mxfp8_quantize", "mxfp8_quantize_cute_dsl"): "e9cc7dad53e84360",
    ("flashinfer.quantization.kernels.mxfp8_quantize", "_get_compiled_kernel_mxfp8_swizzled"): "4e50f2ed723fbaaa",
    ("flashinfer.quantization.fp8_quantization", "mxfp8_quantize"): "ccfdbec3c2435309",
    ("flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100",
     "Sm100BlockScaledPersistentDenseGemmKernel.narrow_tile_ok"): "17030a9c48ef2aff",
    ("flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk",
     "Sm100BlockScaledSplitKGemmKernel.supports_m"): "c59f4389a6d63c5d",
    ("flashinfer.autotuner.autotuner", "AutoTuner.choose_one"): "ecf4c10829cfbfe4",
    # the tuned lookup a plan key stands for: its result depends on the bucketed profile + tuner state only
    ("flashinfer.autotuner.autotuner", "AutoTuner.search_cache"): "9fff387d4f797060",
    ("flashinfer.autotuner.autotuner", "AutoTuner._get_cache_key"): "14c737eb0ab51a40",
    ("flashinfer.autotuner.autotuner", "AutoTuner._find_nearest_profile_cached"): "3bea11710c38dcea",
    ("flashinfer.autotuner.autotuner", "AutoTuner._tactic_still_valid"): "68b09ac16d5f6568",
    # fp8_utils._flashinfer_mm_mxfp8_impl, between sglang::flashinfer_mm_mxfp8 and mm_mxfp8 (bypassed by the fast path)
    ("sglang.srt.layers.quantization.mxfp8_dispatch_cache", "Mxfp8DispatchCache.__call__"): "ba11b1ef8a81fca0",
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_blockscaled_linear"): "99da4a08ba42d542",
    ("sglang.srt.layers.quantization.fp8_utils", "_mxfp8_decode_tactic"): "bdb231795cc1f6f8",
    ("sglang.srt.layers.quantization.fp8_utils", "_cute_dsl_mm_mxfp8_with_tactic"): "ced323827fb15816",
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_quantize"): "01a12afdc8ca1435",
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mm_mxfp8"): "ca28f597cfd33639",
    ("sglang.srt.layers.quantization.fp8", "Fp8LinearMethod.apply"): "24ab0721ac46abe0",
}


_PARSED: dict = {}        # module name -> (source, ast) while install() runs


def source_hash(module, qualname: str):
    """sha256[:16] of the exact source text of `qualname` ("f" or "Class.f") in `module` (None: not found)."""
    hit = _PARSED.get(module.__name__)
    if hit is None:
        src = inspect.getsource(module)
        hit = _PARSED[module.__name__] = (src, ast.parse(src))
    src, tree = hit
    parts = qualname.split(".")
    nodes = [tree]
    for part in parts:
        found = None
        for top in nodes:
            for node in ast.walk(top):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == part:
                    found = node
                    break
            if found is not None:
                break
        if found is None:
            return None
        nodes = [found]
    seg = ast.get_source_segment(src, nodes[0])
    return hashlib.sha256(seg.encode()).hexdigest()[:16] if seg is not None else None


class _Fi:
    """The FlashInfer / SGLang objects stock's call chain reaches, imported once (install)."""

    def __init__(self):
        import flashinfer
        from flashinfer.autotuner import AutoTuner
        from flashinfer.gemm import gemm_base as gb
        from flashinfer.gemm import gemm_mm_fp4_cute_dsl as g4
        from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm100 as dk
        from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm100_splitk as sk
        from flashinfer.quantization import fp8_quantization as fq
        from flashinfer.quantization.kernels import mxfp8_quantize as mq
        from flashinfer.autotuner import autotuner as at
        from sglang.srt.layers.quantization import fp8 as fp8_mod
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.layers.quantization import mxfp8_dispatch_cache as mdc
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import \
            is_in_breakable_cuda_graph

        self.version = flashinfer.__version__
        self.modules = {"flashinfer.gemm.gemm_base": gb, "flashinfer.gemm.gemm_mm_fp4_cute_dsl": g4,
                        "flashinfer.quantization.kernels.mxfp8_quantize": mq,
                        "flashinfer.quantization.fp8_quantization": fq,
                        "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100": dk,
                        "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk": sk,
                        "flashinfer.autotuner.autotuner": at,
                        "sglang.srt.layers.quantization.fp8_utils": fp8_utils,
                        "sglang.srt.layers.quantization.fp8": fp8_mod,
                        "sglang.srt.layers.quantization.mxfp8_dispatch_cache": mdc}
        self.AutoTuner = AutoTuner
        self.gb = gb
        self.mq = mq
        self.kcache = gb._CUTE_DSL_MM_MXFP8_KERNEL_CACHE
        self.compile = gb._compile_block_scaled_gemm
        self.getter = mq._get_compiled_kernel_mxfp8_swizzled
        self.tuning_config = gb._MM_MXFP8_CUTE_DSL_TUNING_CONFIG     # mm_mxfp8 reads the global per call
        self.bucket = self.tuning_config.dynamic_tensor_specs[0].map_to_tuning_buckets
        self.two_t = int(mq.MXFP8_2T_SF_THRESHOLD)
        self.blocks_per_sm = int(mq._BLOCKS_PER_SM)
        self.sf128 = mq.SF_LAYOUT_128x4
        self.num_sm = mq.get_num_sm
        self.split_max_m = int(sk.Sm100BlockScaledSplitKGemmKernel.MAX_M)
        self.narrow_max = int(dk.Sm100BlockScaledPersistentDenseGemmKernel.NARROW_TILE_MAX_N)
        self.linear_fn = fp8_utils.flashinfer_mxfp8_blockscaled_linear
        self.Fp8LinearMethod = fp8_mod.Fp8LinearMethod
        self.in_bcg = is_in_breakable_cuda_graph
        self.modes = tuple(ForwardMode[name] for name in levers.FL_MODES)


def install() -> str:
    """Import + pin the handles once (model_node.prepare, before the replicas bind)."""
    if _STATE["fi"] is not None or _STATE["detail"].startswith("off"):
        return _STATE["detail"]
    if not getattr(levers, "FL", False):
        _STATE["detail"] = "off (levers.FL)"
        return _STATE["detail"]
    try:
        fi = _Fi()
        if fi.version != "0.7.1":
            raise RuntimeError(f"flashinfer {fi.version} (mirror read from 0.7.1)")
        if torch.cuda.get_device_capability() == (10, 7):
            raise RuntimeError("SM107 quantize paths are not mirrored")
        bad = []
        for (mod, name), want in PINNED.items():
            got = source_hash(fi.modules[mod], name)
            if got != want:
                bad.append(f"{mod}.{name} {got}")
        _PARSED.clear()
        if bad:
            raise RuntimeError("source differs from the mirror: " + ", ".join(bad))
        if fi.split_max_m >= int(levers.FL_MIN_ROWS) or fi.narrow_max >= int(levers.FL_MIN_ROWS):
            raise RuntimeError(f"FL_MIN_ROWS {levers.FL_MIN_ROWS} inside FlashInfer's M-dependent tactic range")
    except Exception as e:  # noqa: BLE001 - any missing / changed internal keeps stock
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:300]})"
        return _STATE["detail"]
    _STATE["fi"] = fi
    _MIN_ROWS[0] = int(levers.FL_MIN_ROWS)
    _STATE["detail"] = (f"on from {levers.FL_MIN_ROWS} rows in {', '.join(levers.FL_MODES)} forwards (eager, no capture "
                        f"/ breakable graph / tuning); plans learned from stock per key, launches + bytes checked")
    return _STATE["detail"]


# ---------------------------------------------------------------------------------------------------- binding
def _swizzled_len(rows: int, k: int) -> int:
    return (rows + 127) // 128 * 128 * ((k + 127) // 128 * 4)


def _why_not(fi, lin, served) -> str:
    """'' when `lin` (the replica's linear) / `served` (the served model's at the same path) can take lever FL."""
    if served is None:
        return "no served twin"
    method = getattr(lin, "quant_method", None)
    smethod = getattr(served, "quant_method", None)
    if not isinstance(method, fi.Fp8LinearMethod) or not isinstance(smethod, fi.Fp8LinearMethod):
        return "not Fp8LinearMethod"
    stock = getattr(smethod, "w8a8_mxfp8_linear", None)
    if (not isinstance(stock, functools.partial) or stock.func is not fi.linear_fn or stock.args
            or not set(stock.keywords) <= {"backend", "pin_tactic"} or stock.keywords.get("backend") != "cute-dsl"):
        return "stock linear is not partial(flashinfer_mxfp8_blockscaled_linear, backend='cute-dsl')"
    if getattr(method, "w8a8_mxfp8_linear", None) is None or getattr(method, "use_marlin", False):
        return "no MXFP8 linear"
    backend = getattr(method, "mxfp8_dense_backend", None)
    if backend is None or not backend.is_flashinfer_cutedsl():
        return "backend is not FlashInfer CuTe-DSL"
    if not (getattr(method, "use_mxfp8", False)
            or (getattr(method, "block_fp8_as_mxfp8", False) and getattr(lin, "block_fp8_mxfp8_ready", False))):
        return "no MXFP8 view"
    if not hasattr(method, "mxfp8_prefill_autotune_min_tokens") or not hasattr(smethod,
                                                                              "mxfp8_prefill_autotune_min_tokens"):
        return "no prefill autotune attribute"
    weight = lin._parameters.get("weight")
    scale = getattr(lin, "weight_scale_inv_swizzled", None)
    if served._parameters.get("weight") is not weight or getattr(served, "weight_scale_inv_swizzled", None) is not scale:
        return "weights differ from the served twin"
    if (not isinstance(weight, torch.Tensor) or weight.dtype != FP8 or weight.dim() != 2 or not weight.is_cuda
            or not weight.is_contiguous() or weight.shape[0] < 128 or weight.shape[1] < 128
            or weight.shape[0] % 8 or weight.shape[1] % 128):
        return f"weight {getattr(weight, 'dtype', None)} {tuple(getattr(weight, 'shape', ()))}"
    if (not isinstance(scale, torch.Tensor) or scale.dtype != U8 or scale.dim() != 1 or scale.device != weight.device
            or scale.shape[0] != _swizzled_len(weight.shape[0], weight.shape[1])):
        return "weight scale is not the 1D 128x4 swizzled layout"
    return ""


def bind(replica, served_layer) -> int:
    """Wrap every eligible MXFP8 linear under the layer replica (after lever DT's bind); returns how many."""
    fi = _STATE["fi"]
    if fi is None:
        return 0
    paths = []

    def walk(module, path):
        for name, child in module._modules.items():
            # `_next_layer` is the following decoder layer (read for its norm / mHC weights): not this layer's linears
            if child is None or name == "_next_layer" or type(child).__name__ == "DeepseekV4DecoderLayer":
                continue
            if isinstance(getattr(child, "quant_method", None), fi.Fp8LinearMethod):
                paths.append(path + (name,))
            else:
                walk(child, path + (name,))

    walk(replica, ())
    count = 0
    for path in paths:
        served = served_layer
        for name in path:
            served = served._modules.get(name) if served is not None else None
        node = replica
        for name in path:
            node = node._modules[name]
        why = _why_not(fi, node, served)
        if why:
            levers.announce(f"fl-skip-{type(node).__name__}-{why[:40]}", f"lever FL: {'.'.join(path)} stays on its "
                            f"current path ({why})")
            continue
        parent = replica
        for name in path[:-1]:
            parent = dt._cow(parent, name)
        lin = parent._modules[path[-1]]
        own = lin is served               # the replica ran the served linear itself: FL's copy stands in for it
        if not getattr(lin, "dsv4f_dt", False) and not getattr(lin, "dsv4f_fl", False):
            lin = copy.copy(lin)          # private linear + method (lever DT's are private already)
            lin.quant_method = copy.copy(lin.quant_method)
            parent._modules[path[-1]] = lin
        method = lin.quant_method
        n, k = lin.weight.shape
        dt_hi = 0
        if getattr(lin, "dsv4f_dt", False) and levers.DT:
            dt_hi = max((hi for _lo, hi, _t in levers.DT_TACTICS.get((n, k), ())), default=0)
        wrapper = FlLinear(method.w8a8_mxfp8_linear, served.quant_method.w8a8_mxfp8_linear, lin.weight,
                           lin.weight_scale_inv_swizzled, dt_hi, ".".join(path))
        method.w8a8_mxfp8_linear = wrapper
        lin.dsv4f_fl = True
        _STATE["own" if own else "dt"].append((method, served.quant_method))
        _STATE["linears"].append(wrapper)
        count += 1
    return count


# ---------------------------------------------------------------------------------------------------- forward scope
def _tuner_token(tuner):
    return (tuner, tuner._dirty_seq, len(tuner.profiling_cache), len(tuner._file_configs), tuner._managed_cache,
            len(tuner._winner_partitions))


def _sync_min_tokens(which: str) -> None:
    """The served methods' startup prefill-autotune bound onto the replicas' method copies in _STATE[which]: "own"
    (the copies bind() made of linears the replica shared with the served model: they stand in for the served objects,
    so they follow the served value in every forward -- breakable-graph captures included) or "dt" (linears that were
    private before FL bound them -- lever DT's copies, made at prepare as well: eligible eager forwards only)."""
    pairs = _STATE[which]
    if not pairs:
        return
    flag = which + "_synced"
    if _STATE[flag]:
        meth, served = pairs[0]
        if meth.mxfp8_prefill_autotune_min_tokens == served.mxfp8_prefill_autotune_min_tokens:
            return
    done = True
    for meth, served in pairs:
        value = served.mxfp8_prefill_autotune_min_tokens
        if meth.mxfp8_prefill_autotune_min_tokens != value:
            meth.mxfp8_prefill_autotune_min_tokens = value
        if value is None:
            done = False
    if done and not _STATE[flag]:
        what = "FL's copies, every forward" if which == "own" else "lever DT's copies, eligible forwards"
        levers.announce(f"fl-sync-{which}", f"lever FL: prefill autotune bound "
                        f"{pairs[0][1].mxfp8_prefill_autotune_min_tokens} copied onto {len(pairs)} replica MXFP8 "
                        f"methods ({what})")
    _STATE[flag] = done


def _gate(forward_batch):
    """The context for an eligible forward (eager, mode in FL_MODES, no capture / BCG / compile / tuning), else None."""
    fi = _STATE["fi"]
    if fi is None or torch.cuda.is_current_stream_capturing():
        return None
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or mode not in fi.modes or fi.in_bcg() or torch.compiler.is_compiling():
        return None
    _sync_min_tokens("dt")        # lever DT's copies: eager forwards of FL's modes only
    tuner = fi.AutoTuner.get()
    if (tuner.is_tuning_mode or tuner._active_tuning_contexts or tuner._get_skip_ops_stack()
            or tuner._get_override_stack() or tuner._in_v2_context or tuner._effective_measure_policy is not None):
        return None
    if tuner._managed_cache is not None or fi.gb._MM_MXFP8_CUTE_DSL_TUNING_CONFIG is not fi.tuning_config:
        # an autotune_v2 store can gain entries on disk without a change FL tracks; a replaced tuning config would
        # change stock's buckets under FL's plan keys: stock for such forwards
        levers.announce("fl-tuner-off", "lever FL: an autotune_v2 managed store is attached or mm_mxfp8's tuning "
                        "config was replaced: stock for those forwards")
        return None
    token = _tuner_token(tuner)
    if token != _STATE["token"]:
        if _STATE["token"] is not None:
            levers.announce("fl-retune", "lever FL: AutoTuner caches changed: tuned-bucket plans re-learned")
        for plans in _PLANS.values():
            for key in [key for key in plans if not key[0]]:
                del plans[key]
        _STATE["token"] = token
    return fi


def forward(fn, forward_batch, *args, **kwargs):
    """Run our model forward with lever FL's fast path active when the forward is eligible."""
    _sync_min_tokens("own")       # every forward (attribute copies only, capture-safe): as the served methods
    st = _gate(forward_batch)
    if st is None:
        return fn(*args, **kwargs)
    token = _ON.set(st)
    try:
        return fn(*args, **kwargs)
    finally:
        _ON.reset(token)


# ---------------------------------------------------------------------------------------------------- learn
class _Rec:
    """Forwards to one compiled kernel with identical arguments, recording the call."""

    def __init__(self, real, kind, info, box):
        self.real = real
        self.kind = kind
        self.info = info
        self.box = box

    def __call__(self, *args):
        self.box.append((self.kind, self.real, self.info, args))
        return self.real(*args)


def _record_stock(fi, stock, call, box):
    """stock(**call) with FlashInfer's GEMM compile helper and swizzled-quantize getter wrapped by recorders. The
    recorders wrap (and the finally restores) whatever is installed at this call, not the install-time objects: a
    wrapper another lever installs after fl.install() is part of stock's call and stays installed."""
    gb, mq, kcache = fi.gb, fi.mq, fi.kcache
    real_compile, real_getter = gb._compile_block_scaled_gemm, mq._get_compiled_kernel_mxfp8_swizzled

    def compile_rec(cache, cache_key, *a, **kw):
        res = real_compile(cache, cache_key, *a, **kw)
        if cache is not kcache:
            return res
        info = (cache_key, kw.get("swap_ab"), kw.get("sf_m"), kw.get("sf_n"), kw.get("sf_k"),
                kw.get("cluster_shape_k", 1), kw.get("device_index"))
        return (_Rec(res[0], "gemm", info, box),) + tuple(res[1:])

    def getter_rec(*a, **kw):
        res = real_getter(*a, **kw)
        return (_Rec(res[0], "quant", (a, tuple(sorted(kw.items())), res[1]), box),) + tuple(res[1:])

    gb._compile_block_scaled_gemm = compile_rec
    mq._get_compiled_kernel_mxfp8_swizzled = getter_rec
    try:
        return stock(**call)
    finally:
        gb._compile_block_scaled_gemm = real_compile
        mq._get_compiled_kernel_mxfp8_swizzled = real_getter


def _norm(args, roles):
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(("T", roles.get(a.data_ptr(), "?"), tuple(a.shape), tuple(a.stride()), a.dtype, a.device))
        elif isinstance(a, int) and not isinstance(a, bool) and a in roles:
            out.append(("P", roles[a]))
        else:
            out.append(a)
    return tuple(out)


def _launches(box, roles):
    return [(kind, id(real), _norm(args, roles)) for kind, real, _info, args in box]


class FlLinear:
    """`w8a8_mxfp8_linear` of a replica's Fp8LinearMethod copy: stock's launches on learned plans in eligible forwards,
    the previous callable (`inner`: lever DT's or stock's) otherwise."""

    def __init__(self, inner, stock, weight, scale, dt_hi, path):
        self.inner = inner
        self.stock = stock
        self.pin0 = bool(stock.keywords.get("pin_tactic", False))
        self.weight = weight
        self.scale = scale
        self.n, self.k = int(weight.shape[0]), int(weight.shape[1])
        self.dev = weight.device
        self.dt_hi = int(dt_hi)
        self.path = path
        self.sfb = self.k // 32                    # SF blocks per row (K % 128 == 0: no K padding)
        self.sf_k = (self.sfb + 3) // 4            # the runner's sf_k
        self.sf_cols = self.sf_k * 4               # the quantizer's padded SF columns
        self.ntiles = (self.n + 127) // 128
        self.sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        self.two_t = _STATE["fi"].two_t            # mxfp8_quantize_cute_dsl's 2T/SF threshold (SF blocks)
        self.bucket = _STATE["fi"].bucket          # the AutoTuner's M -> bucket map for mm_mxfp8 (cute-dsl config)
        self.plans = _PLANS.setdefault((self.n, self.k, self.dev.index), {})   # shared per weight shape
        self.fast = 0
        self.fallback = 0

    def __call__(self, input, weight, weight_scale, input_scale=None, bias=None, **kwargs):
        if _ON.get() is None or bias is not None or weight is not self.weight or weight_scale is not self.scale:
            return self.inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                              bias=bias, **kwargs)
        if kwargs:
            pin = kwargs.get("pin_tactic")
            if len(kwargs) != 1 or not isinstance(pin, bool):
                return self.inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                                  bias=bias, **kwargs)
        else:
            pin = self.pin0
        nd = input.dim()
        x = input if nd == 2 else input.view(-1, input.shape[-1])
        m = x.shape[0]
        k = self.k
        if m < _MIN_ROWS[0] or (not kwargs and m <= self.dt_hi) or x.shape[1] != k or not x.is_contiguous():
            return self.inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                              bias=bias, **kwargs)
        if input_scale is None:
            if x.dtype is not BF16 or x.data_ptr() & 15:
                self.fallback += 1
                return self.inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                                  bias=bias, **kwargs)
            two = m * self.sfb >= self.two_t
        else:
            if (x.dtype is not FP8 or input_scale.dtype is not U8 or input_scale.dim() != 1
                    or input_scale.shape[0] != (m + 127) // 128 * 128 * self.sf_cols):
                self.fallback += 1
                return self.inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                                  bias=bias, **kwargs)
            two = None
        # stock's tactic source: _mxfp8_decode_tactic's CTA test (pinned; m > 32 here) or the AutoTuner bucket (tuned)
        key = (pin, self.ntiles * ((m + 63) // 64) <= self.sms if pin else self.bucket(m), two)
        plan = self.plans.get(key)
        if plan is None:
            return self._learn(key, input, x, input_scale, m, kwargs)
        if plan is False:
            return self.inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale,
                              bias=bias, **kwargs)
        self.fast += 1
        out = self._fast(plan, x, input_scale, m)
        return out if nd == 2 else out.view(*input.shape[:-1], self.n)

    def _fast(self, plan, x, input_scale, m):
        """mxfp8_quantize_cute_dsl (swizzled 128x4, padded_k == k) + CuteDSLMxfp8GemmRunner.forward's launch."""
        qk, rpb, grid, gk, swap, alpha = plan
        n = self.n
        dev = self.dev
        if qk is not None:
            pm = (m + 127) // 128 * 128
            q = torch.empty(m, self.k, dtype=U8, device=dev)
            sf = torch.empty(pm * self.sf_cols, dtype=U8, device=dev)
            nb = (pm + rpb - 1) // rpb
            qk(x, q, sf, m, pm, nb if nb < grid else grid)
            a = q.view(FP8)
        else:
            a, sf = x, input_scale
        out = torch.empty((m, n), device=dev, dtype=BF16)
        if swap:
            gk(self.weight, a, out.as_strided((m, n), (1, m)), self.ntiles, (m + 127) // 128, self.sf_k,
               self.scale.data_ptr(), sf.data_ptr(), alpha)
        else:
            gk(a, self.weight, out, (m + 127) // 128, self.ntiles, self.sf_k, sf.data_ptr(), self.scale.data_ptr(),
               alpha)
        return out

    def _plan_from(self, fi, box, x, input_scale, m, key):
        """(plan, '') from stock's recorded launches, or (None, why)."""
        want_q = input_scale is None
        kinds = [r[0] for r in box]
        if kinds != (["quant", "gemm"] if want_q else ["gemm"]):
            return None, f"stock launched {kinds}"
        qk = rpb = grid = None
        if want_q:
            _kind, qreal, (gargs, gkw, rpb), qargs = box[0]
            if (gkw or len(gargs) != 6 or gargs[0] != "bf16" or gargs[1] != self.k or gargs[3] is not key[2]
                    or gargs[4] != fi.sf128 or gargs[5] is not False):
                return None, f"quantize kernel arguments {gargs}"
            if fi.mq._get_compiled_kernel_mxfp8_swizzled(*gargs)[0] is not qreal:    # the getter stock calls
                return None, "quantize kernel is not FlashInfer's cached object"
            grid = fi.num_sm(x.device) * fi.blocks_per_sm
            rpb = int(rpb)
        _kind, greal, (cache_key, swap, _sf_m, _sf_n, _sf_k, split, dev_index), gargs = box[-1]
        if split != 1 or cache_key[-1] != 1 or dev_index is not None:
            return None, f"GEMM launch {cache_key} split {split} device {dev_index}"
        entry = fi.kcache.get((torch.cuda.current_device(), cache_key))
        if entry is None or entry[0] is not greal or torch.cuda.current_device() != self.dev.index:
            return None, "GEMM kernel is not FlashInfer's cached object for this device"
        alpha = gargs[8] if len(gargs) == 9 else None
        if not isinstance(alpha, torch.Tensor) or alpha.numel() != 1 or alpha.dtype != torch.float32:
            return None, "alpha"
        return (qk if not want_q else qreal, rpb, grid, greal, bool(swap), alpha), ""

    def _learn(self, key, input, x, input_scale, m, kwargs):
        """Stock for this call (returned); the key's plan from its recorded launches, checked against our launch."""
        fi = _STATE["fi"]
        call = {"input": input, "weight": self.weight, "weight_scale": self.scale, "input_scale": input_scale,
                "bias": None}
        call.update(kwargs)
        if torch.cuda.is_current_stream_capturing():
            return self.stock(**call)
        sbox = []
        ref = _record_stock(fi, self.stock, call, sbox)
        plan, why, got = None, "", None
        try:
            plan, why = self._plan_from(fi, sbox, x, input_scale, m, key)
            if plan is not None:
                obox = []
                qk, rpb, grid, gk, swap, alpha = plan
                proxied = (_Rec(qk, "quant", None, obox) if qk is not None else None, rpb, grid,
                           _Rec(gk, "gemm", None, obox), swap, alpha)
                got = self._fast(proxied, x, input_scale, m)
                if input.dim() != 2:
                    got = got.view(*input.shape[:-1], self.n)
                common = {self.weight.data_ptr(): "W", self.scale.data_ptr(): "WSF", alpha.data_ptr(): "ALPHA",
                          x.data_ptr(): "X" if input_scale is None else "XQ"}
                sroles, oroles = dict(common), dict(common)
                if input_scale is None:
                    sroles[sbox[0][3][1].data_ptr()] = "XQ"
                    sroles[sbox[0][3][2].data_ptr()] = "XSF"
                    oroles[obox[0][3][1].data_ptr()] = "XQ"
                    oroles[obox[0][3][2].data_ptr()] = "XSF"
                else:
                    sroles[input_scale.data_ptr()] = "XSF"
                    oroles[input_scale.data_ptr()] = "XSF"
                sroles[ref.data_ptr()] = "OUT"
                oroles[got.data_ptr()] = "OUT"
                s_l, o_l = _launches(sbox, sroles), _launches(obox, oroles)
                if s_l != o_l:
                    why = f"launches differ: stock {s_l} ours {o_l}"
                elif (ref.shape != got.shape or ref.stride() != got.stride() or ref.dtype != got.dtype
                      or not torch.equal(ref.view(torch.int16), got.view(torch.int16))):
                    why = "output bytes differ"
        except Exception as e:  # noqa: BLE001 - a plan we cannot build or launch keeps this key on stock
            why = f"{type(e).__name__}: {str(e)[:200]}"
        ok = plan is not None and not why
        self.plans[key] = plan if ok else False
        _STATE["learned" if ok else "refused"] += 1
        tag = f"fl-{self.n}x{self.k}-{'on' if ok else key}"
        if ok:
            levers.announce(tag, f"lever FL {self.path} N={self.n} K={self.k}: first plan at {m} rows key {key} "
                            f"(GEMM swap {plan[4]}, {'quantize + ' if plan[0] is not None else ''}GEMM launches == "
                            f"stock, bytes == stock): on")
        else:
            levers.announce(tag, f"lever FL {self.path} N={self.n} K={self.k} key {key} at {m} rows: OFF ({why[:400]})")
        return ref


def summary() -> str:
    lins = _STATE["linears"]
    return (f"{len(lins)} linears, {sum(lin.fast for lin in lins)} fast calls, {_STATE['learned']} plans learned, "
            f"{_STATE['refused']} refused, {sum(lin.fallback for lin in lins)} input fallbacks")
