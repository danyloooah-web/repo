"""Lever AR: the tensor-parallel all-reduces of OUR eager EXTEND forwards (two a layer -- wo_b's and the MoE's
post-experts sum -- plus the embedding's and the Engram layers') issue the collective stock picks for that tensor
directly, without stock's per-call dispatch (GroupCoordinator.all_reduce's method selection, sglang's custom-op
dispatcher hop, torch.distributed.all_reduce's Python wrapper).

What stock runs (fork 5e72ce5, eager, CUDA, TP2, this arena's engine config): GroupCoordinator.all_reduce(x) picks
  * x up to the custom all-reduce v2 eager ceiling (16 MiB): outplace_all_reduce(x, "ca") ->
    CustomAllReduceV2.custom_all_reduce -> kernels.ops.communication.all_reduce.custom_all_reduce(comm, x, algo) ->
    the JIT module's all_reduce(comm, x, algo_name, None, use_multicast), wrapped by torch.from_dlpack;
  * anything larger: inplace_all_reduce(x) -> _all_reduce_in_place (pynccl is disabled outside graph capture) ->
    torch.distributed.all_reduce(x, group=device_group) -> device_group.allreduce([x], opts(SUM, sync)).wait().
A plan is that choice for one (shape, dtype, device), made once by stock's own predicates (should_custom_ar,
_can_use_flashinfer_allreduce, _resolve_outplace_all_reduce_method, _pick_config) and then replayed: the same final
call with the same arguments on the current stream -> the same collective on the same communicator, bit for bit, in
the same order on every rank. Anything a plan does not cover (other communicators, symmetric memory, FlashInfer
all-reduce, graph mode, pynccl enabled, non-contiguous inputs, torch-function modes, coalescing) stays stock.

Chunked custom all-reduce (levers.AR_CHUNK, TP2 bf16 only): where stock would take the ProcessGroupNCCL path only
because x is larger than the custom all-reduce workspace (16 MiB: the SM100 world-2 eager table prefers the custom
kernel up to 128 MB, the workspace clips it), the plan runs the custom all-reduce v2 kernel on <= workspace-sized
contiguous chunks of x (stock's own _pick_config per chunk) and copies each chunk's sum back into x (stock reduces in
place). With two ranks every output element is the single sum a + b of the two ranks' bf16 values, rounded once (an
fp32 sum of two bf16 values is exact unless their exponents differ by more than 15, and then the smaller one is far
below half an ulp of the larger, so both roundings agree): the bytes equal NCCL's. The first call per size class is
byte-checked against stock like every plan. Why: ProcessGroupNCCL's allreduce blocks the calling thread until its
kernel ends (measured in the engine: ~150 us a call at 2000 rows, the GPU queue drained at every all-reduce of a
host-bound eager pass); the custom kernel is a plain stream launch.

Two exact trims on the way to those all-reduces, same context:
  * MoE entry: our lifted MoE forwards call stock's post_experts_all_reduce (moe_fwd's global); its config-static half
    (EP / MoE-TP widths, whether the combine already summed) is evaluated once a forward with stock's predicates, its
    per-call half (should_skip_mlp_all_reduce) every call, then the MoE-TP group's all_reduce as _post_experts_sum does;
  * fused-AR gate: forward_hc_pre_from_prev asks deepseek_v4.can_fuse_all_reduce twice a layer whether the fused
    all-reduce + mHC epilogue applies; stock's answer above 384 rows is always False, given without its registry lookup.

Scope: only inside our model forward for EXTEND / MIXED batches outside any CUDA-graph capture and outside the
breakable / piecewise prefill graphs (forward() below sets the context); decode / verify (captured or eager) and every
other caller -- the served model's own forward (the node audit's twin), the draft, the logits -- run stock unchanged.
Pinned: the stock functions whose logic a plan reproduces are hashed at install (their source text, ast-extracted)
against the fork 5e72ce5 / torch 2.14.1 source (levers.AR_PINS); any difference leaves the lever off.
Checked: the first call of every (plan, dtype, size class) up to AR_CHECK_MAX_BYTES runs stock on a copy and the plan
on the tensor (two collectives on every rank, same order) and compares every output byte; a difference (or a plan call
that fails before its launch) turns that whole (plan kind, algo, dtype) family off for good on that group -- every
size, including classes that went direct earlier. Larger classes go direct once the same plan kind passed below them
and none failed.
"""
import contextvars
import sys

import torch

from dsv41f import levers

_ON = contextvars.ContextVar("dsv41f_ar", default=None)   # per eager EXTEND forward: {"moe": MoE entry state}
_GROUPS: list = []        # (group, stock bound all_reduce, per-group plans)
_CHECKED: dict = {}       # group name -> {(plan kind, algo, dtype, size class): True (bytes equal) / False}
_DEAD: dict = {}          # group name -> {(plan kind, algo, dtype)}: families off for good (a failed check)
_STATE = {"detail": "not installed", "ca": 0, "cac": 0, "c10d": 0, "stock": 0, "checks": 0, "fail": 0, "modes": None,
          "gates": None}

# stock functions a plan's logic depends on: (module, qualified name); levers.AR_PINS holds their source hashes
PIN_TARGETS = (
    ("sglang.srt.distributed.parallel_state", "GroupCoordinator.all_reduce"),
    ("sglang.srt.distributed.parallel_state", "GroupCoordinator._resolve_outplace_all_reduce_method"),
    ("sglang.srt.distributed.parallel_state", "GroupCoordinator._can_use_flashinfer_allreduce"),
    ("sglang.srt.distributed.parallel_state", "GroupCoordinator._all_reduce_out_place"),
    ("sglang.srt.distributed.parallel_state", "GroupCoordinator._all_reduce_in_place"),
    ("sglang.srt.distributed.parallel_state", "inplace_all_reduce"),
    ("sglang.srt.distributed.parallel_state", "outplace_all_reduce"),
    ("sglang.srt.distributed.device_communicators.custom_all_reduce_v2", "CustomAllReduceV2.should_custom_ar"),
    ("sglang.srt.distributed.device_communicators.custom_all_reduce_v2", "CustomAllReduceV2.custom_all_reduce"),
    ("sglang.srt.distributed.device_communicators.custom_all_reduce_v2", "CustomAllReduceV2._pick_config"),
    ("sglang.srt.distributed.device_communicators.custom_all_reduce_v2", "CustomAllReduceV2._can_use_graph"),
    ("sglang.kernels.ops.communication.all_reduce", "custom_all_reduce"),
    ("sglang.kernels.ops.communication.all_reduce", "get_all_reduce_module"),
    ("sglang.srt.utils.custom_op", "CustomOpWrapper.real_impl"),
    ("torch.distributed.distributed_c10d", "all_reduce"),
)
# the fused all-reduce + mHC epilogue gate whose > 384-row answer (False) the gate shortcut returns directly
PIN_TARGETS_FUSE = (
    ("sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe", "can_fuse_all_reduce"),
)
# the MoE's all-reduce entry (moe_fwd's post_experts_all_reduce) whose decision the MoE entry fast path reproduces
PIN_TARGETS_MOE = (
    ("sglang.srt.layers.moe.utils", "post_experts_all_reduce"),
    ("sglang.srt.layers.moe.utils", "_post_experts_sum"),
    ("sglang.srt.layers.moe.utils", "should_skip_post_experts_all_reduce"),
    ("sglang.srt.layers.moe.utils", "should_skip_mlp_all_reduce"),
    ("sglang.srt.distributed.communication_op", "moe_tensor_model_parallel_all_reduce"),
)


def source_hash(text: str, qualname: str):
    """sha256 (16 hex) of a def's source text (decorators included, dedented) in a module's source TEXT, found by ast;
    None if absent. The LAST def of that name wins (typing.overload stubs come first). Text, not a path: the
    validator's bundle scanner bans builtin open() and importlib, so in the bundle the text comes from
    inspect.getsource(module) (_module_source) and offline (gen_ar.py) from the same file read as text."""
    import ast
    import hashlib
    import textwrap

    if not text.endswith("\n"):
        text += "\n"                          # linecache (inspect.getsource) ends the last line with a newline
    body = ast.parse(text).body
    node = None
    for part in qualname.split("."):
        node = None
        for n in body:
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part:
                node = n
        if node is None:
            return None
        body = node.body
    lines = text.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    seg = textwrap.dedent("".join(lines[start - 1:node.end_lineno]))
    return hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]


def _module_source(name: str) -> str:
    """The source text of a pinned module, by its dotted name: each module imported statically here (the validator's
    scanner bans importlib and builtin open() in a bundle), its text from inspect.getsource (the module's own file,
    through linecache). Every PIN_TARGETS* module needs a branch (gen_ar.py checks it)."""
    import inspect

    if name == "sglang.srt.distributed.parallel_state":
        from sglang.srt.distributed import parallel_state as mod
    elif name == "sglang.srt.distributed.device_communicators.custom_all_reduce_v2":
        from sglang.srt.distributed.device_communicators import custom_all_reduce_v2 as mod
    elif name == "sglang.kernels.ops.communication.all_reduce":
        from sglang.kernels.ops.communication import all_reduce as mod
    elif name == "sglang.srt.utils.custom_op":
        from sglang.srt.utils import custom_op as mod
    elif name == "torch.distributed.distributed_c10d":
        from torch.distributed import distributed_c10d as mod
    elif name == "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe":
        from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as mod
    elif name == "sglang.srt.layers.moe.utils":
        from sglang.srt.layers.moe import utils as mod
    elif name == "sglang.srt.distributed.communication_op":
        from sglang.srt.distributed import communication_op as mod
    else:
        raise KeyError(f"no static import for pinned module {name}")
    if mod.__name__ != name:
        raise ImportError(f"{name} resolved to {mod.__name__}")
    return inspect.getsource(mod)


def _check_pins(targets=PIN_TARGETS) -> str:
    bad = []
    for mod_name, qual in targets:
        want = levers.AR_PINS.get(f"{mod_name}:{qual}")
        try:
            got = source_hash(_module_source(mod_name), qual)
        except Exception as e:  # noqa: BLE001
            got = type(e).__name__
        if want is None or got != want:
            bad.append(f"{qual} {got} != {want}")
    return "; ".join(bad)


def _bits(t: torch.Tensor) -> torch.Tensor:
    n = t.element_size()
    return t.view(torch.int16 if n == 2 else torch.int32 if n == 4 else torch.uint8)


def install() -> str:
    """Wrap the TP group coordinators' all_reduce (model_node.prepare, before any capture). Off on any surprise."""
    if not levers.AR:
        _STATE["detail"] = "off"
        return _STATE["detail"]
    if _GROUPS:
        return _STATE["detail"]
    try:
        from torch.distributed import ReduceOp
        from torch.distributed.distributed_c10d import AllreduceOptions, _world

        from sglang.kernels.ops.communication import all_reduce as ar_ops
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
        from sglang.srt.distributed.parallel_state import GroupCoordinator
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
            is_in_tc_piecewise_cuda_graph,
        )
        from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
        from sglang.srt.runtime_context import get_parallel

        coalesce = _world.pg_coalesce_state
        if not isinstance(coalesce, dict) or not hasattr(ar_ops, "get_all_reduce_module"):
            raise AttributeError("torch _world.pg_coalesce_state / all_reduce.get_all_reduce_module")
        dispatch_modes = torch._C._len_torch_dispatch_stack
        if not (hasattr(CustomAllReduceV2, "_pick_config") and hasattr(CustomAllReduceV2, "_can_use_graph")
                and hasattr(CustomAllReduceV2, "should_custom_ar") and hasattr(CustomAllReduceV2, "custom_all_reduce")):
            raise AttributeError("CustomAllReduceV2 _pick_config / _can_use_graph / should_custom_ar / custom_all_reduce")
        if not (hasattr(GroupCoordinator, "all_reduce") and hasattr(GroupCoordinator, "_resolve_outplace_all_reduce_method")
                and hasattr(GroupCoordinator, "_can_use_flashinfer_allreduce")
                and hasattr(GroupCoordinator, "_all_reduce_in_place") and hasattr(GroupCoordinator, "_all_reduce_out_place")):
            raise AttributeError("GroupCoordinator all_reduce / _resolve_outplace_all_reduce_method / "
                                 "_can_use_flashinfer_allreduce / _all_reduce_in_place / _all_reduce_out_place")
        pins = _check_pins() if levers.AR_PIN else ""
        if pins:
            _STATE["detail"] = f"off (stock changed: {pins[:400]})"
            return _STATE["detail"]
        parallel = get_parallel()
        cands = []
        for label in ("tp", "attn_tp", "moe_tp"):
            try:
                g = (parallel.tp_group if label == "tp" else parallel.attn_tp_group if label == "attn_tp"
                     else parallel.moe_tp_group)
            except Exception:  # noqa: BLE001 - a group the layout leaves unset
                continue
            if isinstance(g, GroupCoordinator) and g.world_size > 1 and all(g is not c for c, _ in cands):
                cands.append((g, label))
    except Exception as e:  # noqa: BLE001 - another stack keeps stock's all-reduce
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:200]})"
        return _STATE["detail"]
    if not cands:
        _STATE["detail"] = "off (no TP group of 2+ ranks)"
        return _STATE["detail"]
    _STATE["modes"] = (ForwardMode.EXTEND, ForwardMode.MIXED)
    _STATE["gates"] = (is_in_breakable_cuda_graph, is_in_tc_piecewise_cuda_graph, get_is_capture_mode)
    ctx = {"ReduceOp": ReduceOp, "AllreduceOptions": AllreduceOptions, "coalesce": coalesce,
           "CustomAllReduceV2": CustomAllReduceV2, "ar_ops": ar_ops, "dispatch_modes": dispatch_modes}
    names = []
    for g, label in cands:
        stock = g.all_reduce
        plans: dict = {}
        _GROUPS.append((g, stock, plans))
        g.all_reduce = _wrapper(g, stock, plans, ctx)
        names.append(f"{label}={g.unique_name}")
    _STATE["detail"] = (f"on for {', '.join(names)}: eager EXTEND / MIXED all-reduces issue stock's chosen collective "
                        f"directly; first call per (plan, dtype, size class) <= {levers.AR_CHECK_MAX_BYTES >> 20} MiB "
                        f"{'byte-checked vs stock' if levers.AR_CHECK else 'NOT checked'}"
                        f"{'' if levers.AR_PIN else ' (source pins skipped)'}; {_install_moe_entry()}; "
                        f"{_install_fuse_gate()}")
    return _STATE["detail"]


def _install_fuse_gate() -> str:
    """The layers' fused all-reduce + mHC epilogue gate (deepseek_v4.can_fuse_all_reduce, asked twice a layer by
    forward_hc_pre_from_prev): stock answers False for every num_tokens > 384 (its registry lookup only decides
    smaller batches), so inside our eager EXTEND forwards that answer is returned without the lookup."""
    if not levers.AR_FUSE_GATE:
        return "fused-AR gate off"
    try:
        from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as fm
        from sglang.srt.models import deepseek_v4 as dv4
    except Exception as e:  # noqa: BLE001
        return f"fused-AR gate off ({type(e).__name__}: {str(e)[:120]})"
    stock_fn = getattr(dv4, "can_fuse_all_reduce", None)
    if stock_fn is None or stock_fn is not getattr(fm, "can_fuse_all_reduce", None):
        return "fused-AR gate off (deepseek_v4.can_fuse_all_reduce is not stock's)"
    pins = _check_pins(PIN_TARGETS_FUSE) if levers.AR_PIN else ""
    if pins:
        return f"fused-AR gate off (stock changed: {pins[:200]})"

    def can_fuse_all_reduce(num_tokens, hidden_dim):
        if num_tokens > 384 and _ON.get() is not None:
            return False                          # stock: `num_tokens <= 384 and ...` (or no registered comm)
        return stock_fn(num_tokens, hidden_dim)

    dv4.can_fuse_all_reduce = can_fuse_all_reduce
    _STATE["fuse_gate"] = (dv4, stock_fn)
    return "fused-AR gate on (deepseek_v4.can_fuse_all_reduce: > 384 rows answered False directly)"


def _install_moe_entry() -> str:
    """The MoE replicas' all-reduce entry (moe_fwd's post_experts_all_reduce, stock's function imported by the lifted
    forwards): inside our eager EXTEND forwards its config-static half (EP / MoE-TP widths, whether the combine already
    summed the output) is evaluated once per forward with stock's own predicates; its per-call half
    (should_skip_mlp_all_reduce) runs every call; then the MoE-TP group's all_reduce (lever AR's wrapper) as stock's
    _post_experts_sum would. An EP sum (not this arena's layout) keeps stock's function."""
    if not levers.AR_MOE_ENTRY:
        return "MoE entry off"
    try:
        from dsv41f import moe_fwd
        from sglang.srt.layers.moe import utils as mu
        from sglang.srt.runtime_context import get_parallel
    except Exception as e:  # noqa: BLE001
        return f"MoE entry off ({type(e).__name__}: {str(e)[:120]})"
    stock_fn = getattr(moe_fwd, "post_experts_all_reduce", None)
    if stock_fn is None or stock_fn is not getattr(mu, "post_experts_all_reduce", None):
        return "MoE entry off (moe_fwd's post_experts_all_reduce is not stock's)"
    for fn in (getattr(mu, "should_skip_mlp_all_reduce", None), getattr(mu, "post_experts_output_is_complete", None)):
        if fn is None:
            return "MoE entry off (moe.utils predicates missing)"
    pins = _check_pins(PIN_TARGETS_MOE) if levers.AR_PIN else ""
    if pins:
        return f"MoE entry off (stock changed: {pins[:300]})"
    skip_mlp = mu.should_skip_mlp_all_reduce
    complete = mu.post_experts_output_is_complete

    def static() -> str:
        """Stock post_experts_all_reduce's config-static half: "stock" (an EP sum: stock's function), "none" (no
        MoE-TP sum owed: stock returns the tensor), "tp" (the MoE-TP sum unless should_skip_mlp_all_reduce)."""
        parallel = get_parallel()
        if parallel.moe_ep_size > 1 and not complete(is_tp_path=False):
            return "stock"
        return "tp" if parallel.moe_tp_size > 1 and not complete(is_tp_path=True) else "none"

    def post_experts_all_reduce(hidden_states):
        st = _ON.get()
        if st is None:
            return stock_fn(hidden_states)
        how = st["moe"]
        if how is None:
            try:
                how = static()
            except Exception:  # noqa: BLE001 - stock decides
                how = "stock"
            st["moe"] = how
        if how == "tp":
            if skip_mlp():
                return hidden_states              # stock: reduce_tp False, reduce_ep False -> the tensor itself
            return get_parallel().moe_tp_group.all_reduce(hidden_states)   # stock: moe_tensor_model_parallel_all_reduce
        if how == "none":
            return hidden_states
        return stock_fn(hidden_states)

    moe_fwd.post_experts_all_reduce = post_experts_all_reduce
    _STATE["moe_entry"] = (moe_fwd, stock_fn)
    return "MoE entry on (moe_fwd.post_experts_all_reduce)"


def _plan(g, x, ctx):
    """Stock GroupCoordinator.all_reduce's eager CUDA choice for x as a plan, or None (stock keeps the call):
    ("ca", module.all_reduce, comm, algo_name, use_multicast, ca_comm) / ("c10d", device_group.allreduce, opts,
    pynccl_comm)."""
    if g.world_size == 1 or not x.is_cuda or not x.is_contiguous() or x.is_complex():
        return None
    if g.hpu_communicator is not None and not g.hpu_communicator.disabled:
        return None
    if g.xpu_communicator is not None and not g.xpu_communicator.disabled:
        return None
    if g.npu_communicator is not None and not g.npu_communicator.disabled:
        return None
    if g.pymscclpp_comm is not None:
        return None
    ca = g.ca_comm
    use_ca = ca is not None and not ca.disabled and ca.should_custom_ar(x)
    if g.pynccl_comm is not None and g.is_symmetric_memory_enabled() and not use_ca:
        return None                                   # stock: symmetric-memory pynccl in place
    if g._can_use_flashinfer_allreduce(x):
        return None                                   # stock: FlashInfer kAllReduce
    method = g._resolve_outplace_all_reduce_method(input_=x, should_use_pymscclpp_allreduce=False)
    if method == "ca":
        if (not use_ca or not isinstance(ca, ctx["CustomAllReduceV2"]) or ca.override_algo is not None
                or ca._graph_mode_allowed):
            return None
        cfg = ca._pick_config(x.numel() * x.element_size(), ca._can_use_graph())
        if cfg is None or cfg.use_graph:
            return None
        module = ctx["ar_ops"].get_all_reduce_module(x.dtype, ca.obj.world_size)
        return ("ca", module.all_reduce, ca.obj, cfg.algo.algo_name, bool(cfg.use_multicast), ca)
    if method is not None:
        return None                                   # pcie_ipc / qr / pymscclpp / torch_symm_mem / pynccl
    p = g.pynccl_comm
    if p is not None and not p.disabled:
        return None                                   # stock: pynccl in place
    t = g.torch_symm_mem_comm
    if t is not None and not t.disabled and t.should_torch_symm_mem_allreduce(x):
        return None
    if g.device_group is None:
        return None
    cac = _chunk_plan(g, x, ctx)
    if cac is not None:
        return cac
    opts = ctx["AllreduceOptions"]()
    opts.reduceOp = ctx["ReduceOp"].SUM
    opts.asyncOp = False
    return ("c10d", g.device_group.allreduce, opts, p)


def _chunk_plan(g, x, ctx):
    """Lever AR_CHUNK: where stock takes ProcessGroupNCCL for a TP2 bf16 tensor, the custom all-reduce v2 kernel on
    <= workspace-sized contiguous chunks (stock's _pick_config per chunk; every chunk must pass stock's
    should_custom_ar), or None (stock's c10d plan)."""
    if not levers.AR_CHUNK or g.world_size != 2 or x.dtype != torch.bfloat16:
        return None
    ca = g.ca_comm
    if (ca is None or ca.disabled or not isinstance(ca, ctx["CustomAllReduceV2"]) or ca.override_algo is not None
            or ca._graph_mode_allowed or ca.obj.world_size != 2):
        return None
    numel = x.numel()
    nbytes = numel * x.element_size()
    cap = int(ca.max_size)
    if levers.AR_CHUNK_MAX_BYTES:
        cap = min(cap, int(levers.AR_CHUNK_MAX_BYTES))
    if nbytes % 16 or cap < (1 << 20) or nbytes <= cap:
        return None
    flat = x.view(-1)
    for n in range(-(-nbytes // cap), -(-nbytes // cap) + 4):
        per = -(-numel // n)
        per = -(-per // 8) * 8                       # 16-byte aligned chunk starts (bf16)
        chunks = []
        for s in range(0, numel, per):
            e = min(s + per, numel)
            seg = flat[s:e]
            cfg = ca._pick_config((e - s) * 2, False)
            if cfg is None or cfg.use_graph or not ca.should_custom_ar(seg):
                chunks = None
                break
            chunks.append((s, e, cfg.algo.algo_name, bool(cfg.use_multicast)))
        if chunks:
            module = ctx["ar_ops"].get_all_reduce_module(x.dtype, ca.obj.world_size)
            return ("cac", module.all_reduce, ca.obj, tuple(chunks), ca)
    return None


def _run_chunks(plan, x, no_rewrap):
    """A cac plan: every chunk's custom all-reduce first (out of place: x untouched until all launched), then each
    chunk's sum copied back into x; returns x (stock's in-place contract)."""
    flat = x.view(-1)
    fn, comm = plan[1], plan[2]
    outs = [fn(comm, flat[s:e], algo, None, mc) for s, e, algo, mc in plan[3]]
    for (s, e, _, _), r in zip(plan[3], outs):
        flat[s:e].copy_(r if no_rewrap and type(r) is torch.Tensor else torch.from_dlpack(r))
    return x


def _run(plan, x):
    """The plan's final call: exactly stock's last call for this tensor (see _rewrap for the ca result)."""
    if plan[0] == "ca":
        return _rewrap(plan[1](plan[2], x, plan[3], None, plan[4]))
    if plan[0] == "cac":
        return _run_chunks(plan, x, bool(levers.AR_CA_NO_REWRAP))
    work = plan[1]([x], plan[2])
    if work is not None:
        work.wait()
    return x


def _rewrap(res):
    """Stock wraps the JIT module's result in torch.from_dlpack. tvm_ffi already returns a torch.Tensor when its
    arguments carried torch's DLPack exchange API (the alias from_dlpack makes is the same storage, shape, strides and
    dtype, with no stream work: producer and consumer stream are both the current stream), so with
    levers.AR_CA_NO_REWRAP that tensor is returned as is; anything else goes through torch.from_dlpack as in stock."""
    if levers.AR_CA_NO_REWRAP and type(res) is torch.Tensor:
        return res
    return torch.from_dlpack(res)


def _kind(plan, dtype):
    """A plan's check family: (plan kind, custom AR algo / chunked algos or "nccl", dtype)."""
    if plan[0] == "cac":
        return (plan[0], "+".join(sorted({c[2] for c in plan[3]})), dtype)
    return (plan[0], plan[3] if plan[0] == "ca" else "nccl", dtype)


def _wrapper(g, stock, plans, ctx):
    coalesce = ctx["coalesce"]
    capturing = torch.cuda.is_current_stream_capturing
    has_tf = torch.overrides.has_torch_function
    dispatch_modes = ctx["dispatch_modes"]
    no_rewrap = bool(levers.AR_CA_NO_REWRAP)
    tensor_type = torch.Tensor
    name = g.unique_name
    checked = _CHECKED.setdefault(name, {})   # this group's (plan kind, algo, dtype, size class) -> bytes == stock
    dead = _DEAD.setdefault(name, set())      # this group's plan families turned off for good

    def kill(kind):
        """A family's check failed (bytes or a pre-launch error): every size of it is stock from now on, cached
        plans included (classes that went direct earlier too)."""
        dead.add(kind)
        for k, p in list(plans.items()):
            if p and _kind(p, k[1]) == kind:
                plans[k] = False

    def first(x, key):
        """A shape's first call (eager, outside capture): its plan from stock's predicates, then its class check."""
        try:
            plan = _plan(g, x, ctx)
        except Exception as e:  # noqa: BLE001 - stock keeps this shape
            levers.announce(f"ar-plan-err-{type(e).__name__}", f"lever AR: plan failed ({type(e).__name__}: "
                            f"{str(e)[:160]}): stock for that shape")
            plan = None
        if plan is None:
            plans[key] = False
            _STATE["stock"] += 1
            return stock(x)
        nbytes = x.numel() * x.element_size()
        kind = _kind(plan, x.dtype)
        if kind in dead:
            plans[key] = False
            _STATE["stock"] += 1
            return stock(x)
        if plan[0] in ("c10d", "cac") and coalesce:
            _STATE["stock"] += 1
            return stock(x)                       # a coalescing manager is open: stock queues the collective (not cached)
        ckey = kind + (nbytes.bit_length(),)
        tag = f"ar-{name}-{ckey}"
        ok = checked.get(ckey)
        if ok is None and (not levers.AR_CHECK or nbytes > levers.AR_CHECK_MAX_BYTES):
            # too big to copy for a check (or checks off): direct once this plan kind passed below and never failed
            below = [v for k, v in checked.items() if k[:3] == kind]
            if not levers.AR_CHECK or (below and all(below)):
                ok = checked[ckey] = True
                levers.announce(tag, f"lever AR: {kind[0]} {kind[1]} {str(x.dtype)[6:]} first at "
                                f"{tuple(x.shape)} ({nbytes} B): direct ({'unchecked' if not levers.AR_CHECK else 'kind verified at ' + str(len(below)) + ' smaller classes'})")
            else:
                _STATE["stock"] += 1
                return stock(x)                   # not cached: a later call may go direct once a smaller one passed
        if ok is None:
            ref = stock(x.clone())                # collective 1 on every rank: stock on a copy
            try:
                got = _run(plan, x)               # collective 2 on every rank: the plan on the tensor
            except Exception as e:  # noqa: BLE001 - raised before its launch (same on every rank): stock's instead
                levers.announce(f"ar-run-err-{type(e).__name__}", f"lever AR: {kind[0]} call failed ({type(e).__name__}: "
                                f"{str(e)[:160]}): stock for this kind")
                got = None
            ok = (got is not None and ref.shape == got.shape and ref.dtype == got.dtype
                  and torch.equal(_bits(ref), _bits(got)))
            if got is None:
                got = stock(x)                    # collective 2 on every rank, stock's
            checked[ckey] = ok
            _STATE["checks"] += 1
            if not ok:
                _STATE["fail"] += 1
                kill(kind)
            levers.announce(tag, f"lever AR: {kind[0]} {kind[1]} {str(x.dtype)[6:]} first at "
                            f"{tuple(x.shape)} ({nbytes} B): "
                            f"{'bytes == stock: on' if ok else 'bytes != stock: OFF for this kind (every size)'}")
            plans[key] = plan if ok else False
            if ok:
                return got
            if plan[0] in ("c10d", "cac"):
                x.copy_(ref)                      # stock reduces in place: the input itself holds stock's sum
                return x
            return ref
        plans[key] = plan if ok else False
        if not ok:
            _STATE["stock"] += 1
            return stock(x)
        return _run(plan, x)

    def all_reduce(input_):
        # stock reaches its collective through torch.ops (sglang custom ops): any torch-function / dispatch mode or
        # tensor subclass sees that path, so those calls stay stock; so does anything under graph capture
        if _ON.get() is None or capturing() or dispatch_modes() or has_tf((input_,)):
            return stock(input_)
        key = (input_.shape, input_.dtype, input_.get_device())
        plan = plans.get(key)
        if not plan:
            if plan is None:
                return first(input_, key)
            _STATE["stock"] += 1
            return stock(input_)
        if not input_.is_contiguous():
            _STATE["stock"] += 1
            return stock(input_)
        if plan[0] == "ca":
            if plan[5]._graph_mode_allowed:
                _STATE["stock"] += 1
                return stock(input_)
            try:
                res = plan[1](plan[2], input_, plan[3], None, plan[4])
            except Exception as e:  # noqa: BLE001 - raised before its launch (same on every rank)
                return failed(key, plan, e, input_)
            _STATE["ca"] += 1
            return res if no_rewrap and type(res) is tensor_type else torch.from_dlpack(res)
        if plan[0] == "cac":
            if plan[4]._graph_mode_allowed or coalesce:
                _STATE["stock"] += 1
                return stock(input_)
            try:
                _run_chunks(plan, input_, no_rewrap)
            except Exception as e:  # noqa: BLE001 - raised before a launch (same on every rank): input untouched
                return failed(key, plan, e, input_)
            _STATE["cac"] += 1
            return input_
        pyn = plan[3]
        if (pyn is not None and not pyn.disabled) or coalesce:
            _STATE["stock"] += 1
            return stock(input_)
        try:
            work = plan[1]([input_], plan[2])
        except Exception as e:  # noqa: BLE001 - raised before its launch (same on every rank)
            return failed(key, plan, e, input_)
        _STATE["c10d"] += 1
        if work is not None:
            work.wait()
        return input_

    def failed(key, plan, e, x):
        plans[key] = False
        _STATE["fail"] += 1
        kill(_kind(plan, x.dtype))
        levers.announce(f"ar-direct-err-{type(e).__name__}", f"lever AR: direct call failed ({type(e).__name__}: "
                        f"{str(e)[:160]}): stock for this kind")
        return stock(x)

    all_reduce.dsv4f_ar = True
    return all_reduce


def forward(fn, forward_batch, *args, **kwargs):
    """Run our model forward with lever AR on for eager EXTEND / MIXED batches only."""
    if not _GROUPS or not _eager_extend(forward_batch):
        return fn(*args, **kwargs)
    token = _ON.set({"moe": None})
    try:
        return fn(*args, **kwargs)
    finally:
        _ON.reset(token)


def _eager_extend(forward_batch) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or mode not in _STATE["modes"]:
        return False
    bcg, pcg, capture_mode = _STATE["gates"]
    return not (torch.cuda.is_current_stream_capturing() or bcg() or pcg() or capture_mode()
                or torch.compiler.is_compiling())


def active():
    """Context for tests: lever AR on in this scope (as inside an eager EXTEND forward)."""
    import contextlib

    @contextlib.contextmanager
    def scope():
        token = _ON.set({"moe": None})
        try:
            yield
        finally:
            _ON.reset(token)

    return scope()


def stats() -> dict:
    return {k: _STATE[k] for k in ("ca", "c10d", "stock", "checks", "fail")}


def describe() -> str:
    return _STATE["detail"]
