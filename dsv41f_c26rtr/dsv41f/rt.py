"""Lever RT on the fork (levers.RT_ROUTER / RT_GATE / RT_WOA): the host path of three steps of our layer replicas' eager
EXTEND forwards, cut down to the launches stock makes.

Why: a live warm turn's prefill (~1.9k new rows on a ~110k cached context) runs eager (the fork's breakable prefill
graphs cover contexts <= 16384 tokens only) and is host-bound on the slower rank; every Python microsecond per layer is
40 microseconds per pass. B300 profile of bundle f47a (one 2000-row pass, 40 layers, TP-1): the MoE router chain
(MoEGate.forward -> linear_bf16_fp32 -> torch.mm, then route_e5.route -> vision_topk -> moe_fused_gate ->
JITFunction.run -> CudaLauncher -> launch) and the attention's WO-A einsum (torch.einsum -> aten::bmm through torch
2.14's Python eager router, then a permute copy at wo_b's flatten) are pure host overhead around one or two kernels.

ROUTER (RT_ROUTER). Stock, on an eager extend batch (E5 prune off: deepseek_v4_prune._on is empty, so route_e5.route is
stock vision_topk): moe_fused_gate's Triton _router_triton_kernel through JITFunction.run (binder, cache key, launch
metadata, CudaLauncher.__call__) + _finish_vision_topk (the identity for this model: no fused shared experts, no
MegaMoE, padded rows already masked by the kernel) -> StandardTopKOutput(weights, ids, logits).
Ours: the same compiled kernel (the CompiledKernel stock's own JIT cache holds) launched through its CudaLauncher's C
launch function with stock's argument tuple, grid, stream, PDL flag and packed metadata; fresh outputs allocated as
stock allocates them.
  learn   key = (the layer's static routing signature, rows, logits dtype / width / strides / 16-byte alignment, the
          input-ids and padding-count tensors' presence / dtype / alignment / stride, current device): a key's first
          call runs stock whole while a temporary pre-run hook on the JIT function records the argument list stock
          launched with (exactly one router launch per call, its tensors the ones we hold, else the key stays stock);
          the binder on that list names the cached kernel. A pass's first MoE layer learns its row count, the other 39
          take the direct launch.
  verify  the first direct launch of every (compiled kernel, row-count class = bit length of rows) is followed by
          stock on the same logits and every output byte is compared (weights as int32 bits, ids); a difference turns
          the router fast path off for the process.
  gate    _ON (our forward, ForwardMode.EXTEND, no capture, not in a breakable / piecewise prefill graph); per call:
          E5 prune inactive, stock's functions not rebound (deepseek_v2.vision_topk, moe_fused_gate.moe_fused_gate /
          _router_triton_kernel, vl_routing._finish_vision_topk), the layer's bias tensors the bound ones, no Triton
          launch hooks / pre-run hooks / stage hook, the same Triton driver, debug and instrumentation knobs as when
          learned, the kernel's used globals unchanged (stock's own check).
GATE (RT_GATE). Stock (> tiny-GEMM rows, bf16 weight, cuBLAS algo): nn.Module.__call__ -> MoEGate.forward ->
linear_bf16_fp32 -> _linear_bf16_fp32_cublas -> torch.mm(x, w.t(), out_dtype=torch.float32). Ours: that torch.mm on a
cached w.t() view (same storage, shape, strides) when the module has no hooks and every branch stock takes is the
cuBLAS one; the first call per row-count class (bit length of rows) is compared with stock's logits bit for bit.
WO-A (RT_WOA). Stock _apply_wo_a_bf16_matmul runs torch.einsum("tgd,grd->tgr") for extend calls below 4096 rows and
the strided torch.bmm into the token-major output from 4096 rows (TP2 shape: 4 local groups, R 1024, D 4096, bf16;
fork/tests/woa_test.py: the two are bit-identical at every row count 1..4096). Ours: that strided bmm for eager
extend calls of 1..4095 rows (one aten::bmm.out, no permute copy at wo_b's flatten); the first call per row-count class
is compared with stock's result bit for bit (a difference: stock for good).

Exactness: the router and the gate launch the kernels stock launches with the arguments stock passes; WO-A launches the
fork's own >= 4096-row form, checked against stock in the engine. Fail-closed: anything unexpected takes stock's path,
each reason announced once (CACHEON_DSV4F lines).
"""
import contextvars

import torch

from dsv41f import levers, route_e5

TRITON_VERSION = "3.8.0"            # the source the launch mirror was read from
_ON = contextvars.ContextVar("dsv41f_rt_on", default=None)
_H: dict = {}                       # handles pinned at install
_PLANS: dict = {}                   # router: call key -> _Plan | False (stock for this key)
_KOK: dict = {}                     # router: ((device, Triton cache key), rows bit length) -> True (== stock) / False
_GOK: dict = {}                     # gate: rows bit length -> True / False
_WOK: dict = {}                     # WO-A: (rows bit length, o row stride) -> True / False
_MAX_PLANS = 512
_ROUTER_ARGS = ("scores_ptr", "bias_ptr", "bias_alt_ptr", "input_ids_ptr", "num_token_non_padded_ptr",
                "out_weights_ptr", "out_indices_ptr", "out_packed_ptr", "M", "routed_scaling_factor", "moe_softcapping")
_ST = {"router": "off (not installed)", "gate": "off (not installed)", "woa": "off (not installed)",
       "router_on": False, "gate_on": False, "woa_on": False, "any_on": False,
       "r_fast": 0, "r_learn": 0, "r_stock": 0, "g_fast": 0, "g_stock": 0, "w_fast": 0, "w_stock": 0}
WOA_MAX_ROWS = 4095                 # stock's strided bmm from 4096 rows (is_prefill, Blackwell)
WOA_SHAPE = (4, 1024, 4096)         # TP2: local groups, o_lora_rank, group width (woa_test.py's bit-identity shape)


def _off(tag, why):
    levers.announce(tag, why)
    return False


def _bits_equal(a, b) -> bool:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.float32, torch.int32):
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a, b)


# ---------------------------------------------------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------------------------------------------------


def install(layers=None) -> str:
    """model_node.prepare: pin the internals each fast path mirrors (any missing piece: that path stays stock)."""
    why = _install_common()
    if why:
        _ST["router"] = _ST["gate"] = _ST["woa"] = f"off ({why})"
    else:
        _ST["router"] = _install_router() if getattr(levers, "RT_ROUTER", False) else "off (levers)"
        _ST["gate"] = _install_gate() if getattr(levers, "RT_GATE", False) else "off (levers)"
        _ST["woa"] = _install_woa() if getattr(levers, "RT_WOA", False) else "off (levers)"
    _ST["router_on"] = _ST["router"].startswith("on")
    _ST["gate_on"] = _ST["gate"].startswith("on")
    _ST["woa_on"] = _ST["woa"].startswith("on")
    _ST["any_on"] = _ST["router_on"] or _ST["gate_on"] or _ST["woa_on"]
    where = ""
    if layers is not None:
        from dsv41f import fork_attn, moe_fwd

        moe = [r._modules.get("mlp") for r in layers]
        lifted = sum(getattr(getattr(m, "forward_normal", None), "__func__", None) is moe_fwd.forward_normal
                     for m in moe if m is not None)
        attn = [r._modules.get("self_attn") for r in layers]
        lifted_attn = sum(getattr(getattr(a, "forward", None), "__func__", None) is fork_attn.attn_forward
                          for a in attn if a is not None)
        where = f"; {lifted} MoE replicas run the lifted forward_normal, {lifted_attn} attention replicas the lifted forward"
    return f"router {_ST['router']}; gate {_ST['gate']}; WO-A {_ST['woa']}{where}"


def _install_common() -> str:
    if "EXTEND" in _H:
        return ""
    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
            is_in_tc_piecewise_cuda_graph,
        )
    except Exception as e:  # noqa: BLE001 - another stack keeps stock everywhere
        return f"{type(e).__name__}: {str(e)[:160]}"
    _H.update(EXTEND=ForwardMode.EXTEND, in_bcg=is_in_breakable_cuda_graph, in_pcg=is_in_tc_piecewise_cuda_graph)
    return ""


def _install_router() -> str:
    try:
        import triton
        from triton import knobs
        from triton.backends.nvidia.driver import CudaLauncher
        from triton.compiler.compiler import CompiledKernel
        from triton.runtime.driver import driver as tdriver   # the DriverConfig jit.py launches through
        from triton.runtime.jit import JITFunction, compute_cache_key

        from sglang.kernels.ops.moe import moe_fused_gate as mfg
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.models import deepseek_v2 as dv2
        from sglang.srt.models import deepseek_v4_prune as prune
        from sglang.srt.multimodal.dsv41 import vl_routing as vl
    except Exception as e:  # noqa: BLE001
        return f"off ({type(e).__name__}: {str(e)[:160]})"
    if triton.__version__ != TRITON_VERSION:
        return f"off (Triton {triton.__version__}, mirror read from {TRITON_VERSION})"
    fn = getattr(mfg, "_router_triton_kernel", None)
    for mod, names in ((mfg, ("moe_fused_gate", "_dummy_i32")), (vl, ("vision_topk", "_finish_vision_topk")),
                       (dv2, ("vision_topk",)), (prune, ("_on",)), (route_e5, ("route", "after_topk", "_LAST")),
                       (CudaLauncher, ("__call__",)), (CompiledKernel, ("run", "launch_metadata"))):
        for n in names:
            if not hasattr(mod, n):
                return f"off (missing {getattr(mod, '__name__', mod)}.{n})"
    if not isinstance(fn, JITFunction):
        return "off (moe_fused_gate._router_triton_kernel is not a Triton JITFunction)"
    if getattr(mfg.moe_fused_gate, "__wrapped__", None) is not None:
        return "off (moe_fused_gate is wrapped: SGLANG_KERNEL_API_LOGLEVEL logging is on)"
    if dv2.vision_topk is not vl.vision_topk or getattr(vl, "StandardTopKOutput", None) is not StandardTopKOutput:
        return "off (deepseek_v2.vision_topk / StandardTopKOutput are not vl_routing's)"
    if tuple(fn.arg_names[:len(_ROUTER_ARGS)]) != _ROUTER_ARGS:
        return f"off (_router_triton_kernel arguments {fn.arg_names[:len(_ROUTER_ARGS)]})"
    for n in ("pre_run_hooks", "device_caches", "used_global_vals", "debug"):
        if not hasattr(fn, n):
            return f"off (JITFunction.{n} missing)"
    rt_knobs, comp_knobs = knobs.runtime, knobs.compilation
    enter, leave = rt_knobs.launch_enter_hook, rt_knobs.launch_exit_hook
    if not (isinstance(getattr(enter, "calls", None), list) and isinstance(getattr(leave, "calls", None), list)):
        return "off (Triton launch hooks are not hook chains)"
    for n in ("debug", "add_stages_inspection_hook"):
        if not hasattr(rt_knobs, n):
            return f"off (triton.knobs.runtime.{n} missing)"
    if not hasattr(comp_knobs, "instrumentation_mode"):
        return "off (triton.knobs.compilation.instrumentation_mode missing)"
    drv = tdriver.active
    raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    get_dev = getattr(torch._C, "_cuda_getDevice", None)
    if drv.get_current_stream is not raw_stream or drv.get_current_device is not torch.cuda.current_device:
        return "off (Triton's driver does not take torch's current device / raw stream)"
    if getattr(drv, "launcher_cls", None) is not CudaLauncher or get_dev is None:
        return "off (Triton's launcher is not the CUDA launcher this mirrors)"
    if torch.cuda.is_initialized() and get_dev() != torch.cuda.current_device():
        return "off (torch._C._cuda_getDevice disagrees with torch.cuda.current_device)"
    _H.update(fn=fn, mfg=mfg, mfg_fn=mfg.moe_fused_gate, vl=vl, vision_topk=vl.vision_topk,
              finish=vl._finish_vision_topk, dv2=dv2, prune=prune, STKO=StandardTopKOutput, knobs_rt=rt_knobs,
              knobs_comp=comp_knobs, enter=enter, leave=leave, drvcfg=tdriver, drv=drv, raw_stream=raw_stream,
              get_dev=get_dev, compute_cache_key=compute_cache_key, CompiledKernel=CompiledKernel,
              CudaLauncher=CudaLauncher)
    return "on (direct launch of stock's compiled _router_triton_kernel; learned per call key, checked per kernel and row class)"


def _install_gate() -> str:
    try:
        from sglang.kernels.ops.gemm import bf16_fp32 as bf
        from sglang.srt.models import deepseek_v2 as dv2
        from torch.nn.modules import module as nn_module
    except Exception as e:  # noqa: BLE001
        return f"off ({type(e).__name__}: {str(e)[:160]})"
    for mod, names in ((bf, ("linear_bf16_fp32", "_linear_bf16_fp32_cublas", "_linear_bf16_fp32_algo")),
                       (dv2, ("MoEGate", "_is_cuda", "_use_aiter", "get_exec", "use_intel_amx_backend")),
                       (nn_module, ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
                                    "_global_backward_pre_hooks"))):
        for n in names:
            if not hasattr(mod, n):
                return f"off (missing {getattr(mod, '__name__', mod)}.{n})"
    if not dv2._is_cuda or dv2._use_aiter:
        return "off (not the CUDA / non-aiter gate branch)"
    if bf._linear_bf16_fp32_algo != "cublas":
        return f"off (SGLANG_OPT_BF16_FP32_GEMM_ALGO={bf._linear_bf16_fp32_algo!r}, not the cuBLAS branch)"
    tracing = getattr(torch._C, "_get_tracing_state", None)
    if tracing is None or not hasattr(torch.nn.Module, "_compiled_call_impl"):
        return "off (torch.nn.Module call internals differ)"
    _H.update(bf=bf, lin=bf.linear_bf16_fp32, cub=bf._linear_bf16_fp32_cublas, gate_cls=dv2.MoEGate,
              gate_forward=dv2.MoEGate.forward, nnm=nn_module, dv2_gate=dv2, tracing=tracing)
    return "on (torch.mm on a cached weight view above the tiny-GEMM rows; first call per row class checked)"


def _install_woa() -> str:
    try:
        from dsv41f import fork_attn
        from sglang.srt.models import deepseek_v4 as dv4
    except Exception as e:  # noqa: BLE001
        return f"off ({type(e).__name__}: {str(e)[:160]})"
    stock = getattr(dv4, "_apply_wo_a_bf16_matmul", None)
    if stock is None or getattr(fork_attn, "_apply_wo_a_bf16_matmul", None) is not stock:
        return "off (the lifted attention does not call stock's _apply_wo_a_bf16_matmul)"
    if getattr(fork_attn, "_is_hip", True) or getattr(dv4, "_is_hip", True):
        return "off (ROCm build: WO-A takes the HIP routes)"
    _H.update(fa=fork_attn, woa_stock=stock)
    return f"on (strided bmm for eager extend calls of 1..{WOA_MAX_ROWS} rows at {WOA_SHAPE}; first call per row class checked)"


# ---------------------------------------------------------------------------------------------------------------------
# the forward gate
# ---------------------------------------------------------------------------------------------------------------------


def forward(fn, prepared, args, kwargs):
    """model_node.forward: run fn(prepared, args, kwargs) with lever RT's fast paths active on eager EXTEND forwards."""
    if not _ST["any_on"]:
        return fn(prepared, args, kwargs)
    fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    mode = getattr(fb, "forward_mode", None)
    if (mode is None or mode != _H["EXTEND"] or torch.cuda.is_current_stream_capturing() or _H["in_bcg"]()
            or _H["in_pcg"]()):
        return fn(prepared, args, kwargs)
    token = _ON.set(True)
    try:
        return fn(prepared, args, kwargs)
    finally:
        _ON.reset(token)


# ---------------------------------------------------------------------------------------------------------------------
# per-layer static state (on the MoE replica: private to our forward)
# ---------------------------------------------------------------------------------------------------------------------


class _Static:
    __slots__ = ("gate", "w", "wptr", "wt", "tiny", "det", "bias", "alt", "skey", "k", "gate_ok", "route_ok")


def _bind(moe):
    """The replica's static routing state (moe.dsv4f_rt), or False when this layer is not the vision_topk shape."""
    st = _Static()
    st.gate_ok = st.route_ok = st.det = False
    gate = moe._modules.get("gate")
    why = None
    if gate is None or getattr(moe, "is_hash", False) or getattr(moe, "is_nextn", False):
        why = "no gate / hash-routed / MTP layer"
    st.gate = gate
    if why is None:
        p = gate._parameters
        w, bias, alt = p.get("weight"), p.get("e_score_correction_bias"), p.get("e_score_correction_bias_vl")
        cfg = getattr(getattr(moe, "topk", None), "topk_config", None)
        image = getattr(getattr(moe, "config", None), "image_token_id", None)
        if not (isinstance(w, torch.Tensor) and w.dim() == 2 and w.is_cuda and w.is_contiguous()):
            why = "gate weight not a 2-D CUDA tensor"
        elif not all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.dim() == 1 and t.is_cuda
                     and t.shape[0] == w.shape[0] and t.device == w.device for t in (bias, alt)):
            why = "correction biases not fp32 [experts] CUDA tensors (the vision_topk branch)"
        elif cfg is None or int(cfg.num_fused_shared_experts) != 0 or not isinstance(image, int):
            why = "top-k config (fused shared experts) / image token"
        else:
            from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend

            if get_moe_runner_backend().is_flashinfer_mxfp4() or get_moe_a2a_backend().is_megamoe():
                why = "packed top-k / MegaMoE"
    if why is not None:
        _off(f"rt-bind-{why[:24]}", f"lever RT: layer {getattr(moe, 'layer_id', '?')}: {why}: stock router")
        moe.dsv4f_rt = False
        return False
    st.w, st.wptr, st.wt = w, w.data_ptr(), w.t()
    st.tiny = int(getattr(gate, "tiny_router_gemm_max_tokens", -1))
    st.bias, st.alt = bias, alt
    st.k = int(cfg.top_k)
    scale = cfg.routed_scaling_factor
    st.skey = (w.device.index, int(w.shape[0]), st.k, bool(cfg.renormalize), None if scale is None else float(scale),
               bool(cfg.apply_routed_scaling_factor_on_output), image, bias.data_ptr() & 15, bias.stride(0),
               alt.data_ptr() & 15, alt.stride(0))
    if _ST["gate_on"]:
        H = _H
        if type(gate) is not H["gate_cls"] or type(gate).forward is not H["gate_forward"]:
            _off("rt-gate-class", "lever RT: the router gate is not stock's MoEGate: stock gate")
        elif w.dtype != torch.bfloat16 or H["dv2_gate"].use_intel_amx_backend(gate):
            _off("rt-gate-dtype", "lever RT: the router weight is not bf16 (or AMX): stock gate")
        else:
            st.det = bool(H["dv2_gate"].get_exec().deterministic.enable_deterministic_inference)
            st.gate_ok = True
    st.route_ok = _ST["router_on"]
    moe.dsv4f_rt = st
    return st


# ---------------------------------------------------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------------------------------------------------


def gate(moe, hidden_states, gemm_output_zero_allocator=None):
    """The lifted forward_normal's self.gate(...) call: on eager EXTEND forwards, MoEGate.forward's cuBLAS torch.mm
    without the module / wrapper layers."""
    if _ON.get() is None or not _ST["gate_on"]:
        return moe.gate(hidden_states, gemm_output_zero_allocator)
    st = getattr(moe, "dsv4f_rt", None)
    if st is None:
        st = _bind(moe)
    x = hidden_states
    if not st or not st.gate_ok or moe._modules.get("gate") is not st.gate:
        _ST["g_stock"] += 1
        return moe.gate(hidden_states, gemm_output_zero_allocator)
    g = st.gate
    rows = x.shape[0]
    w = g._parameters.get("weight")
    H = _H
    nnm = H["nnm"]
    bf = H["bf"]
    # nn.Module.__call__'s hook-free fast path (no compiled call, no tracing, no hooks) reaching MoEGate.forward's
    # cuBLAS branch (bf16 x / weight, above the tiny-GEMM rows or deterministic, stock linear_bf16_fp32 on 'cublas')
    if (not (rows > st.tiny or st.det) or w is not st.w or w.data_ptr() != st.wptr or x.dtype != torch.bfloat16
            or not x.is_cuda or x.dim() != 2 or torch.is_grad_enabled() or g._compiled_call_impl is not None
            or g._forward_hooks
            or g._forward_pre_hooks or g._backward_hooks or g._backward_pre_hooks or nnm._global_forward_hooks
            or nnm._global_forward_pre_hooks or nnm._global_backward_hooks or nnm._global_backward_pre_hooks
            or getattr(g.forward, "__func__", None) is not H["gate_forward"] or H["tracing"]()
            or bf.linear_bf16_fp32 is not H["lin"] or bf._linear_bf16_fp32_cublas is not H["cub"]
            or bf._linear_bf16_fp32_algo != "cublas"):
        _ST["g_stock"] += 1
        return moe.gate(hidden_states, gemm_output_zero_allocator)
    cls = rows.bit_length()
    ok = _GOK.get(cls)
    if ok is False:
        _ST["g_stock"] += 1
        return moe.gate(hidden_states, gemm_output_zero_allocator)
    logits = torch.mm(x, st.wt, out_dtype=torch.float32)
    if ok is None:
        ref = moe.gate(hidden_states, gemm_output_zero_allocator)
        same = _bits_equal(logits, ref)
        _GOK[cls] = same
        levers.announce(f"rt-gate{cls}", f"lever RT: router gate, first at {rows} rows (class {cls}): "
                        f"{'== stock: on' if same else '!= stock: OFF for this class'}")
        if not same:
            return ref
    _ST["g_fast"] += 1
    return logits


# ---------------------------------------------------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------------------------------------------------


class _Plan:
    __slots__ = ("ckey", "kernel", "launcher", "launch", "function", "packed", "coop", "pdl", "ann", "sig", "mid",
                 "tail", "block_m", "debug", "imode", "drv", "globals", "k")


def _stock_route(moe, router_logits, input_ids, num_token_non_padded):
    _ST["r_stock"] += 1
    return route_e5.route(moe, router_logits, input_ids, num_token_non_padded)


def route(moe, router_logits, input_ids, num_token_non_padded):
    """The lifted forward_normal's route_e5.route call: on eager EXTEND forwards (no E5 prune in flight, so stock is
    vision_topk), stock's compiled router kernel launched directly with stock's arguments."""
    if _ON.get() is None or not _ST["router_on"]:
        return route_e5.route(moe, router_logits, input_ids, num_token_non_padded)
    st = getattr(moe, "dsv4f_rt", None)
    if st is None:
        st = _bind(moe)
    H = _H
    if (not st or not st.route_ok or H["prune"]._on or H["dv2"].vision_topk is not H["vision_topk"]
            or H["mfg"].moe_fused_gate is not H["mfg_fn"] or H["mfg"]._router_triton_kernel is not H["fn"]
            or H["vl"]._finish_vision_topk is not H["finish"]):
        return _stock_route(moe, router_logits, input_ids, num_token_non_padded)
    p = st.gate._parameters
    bias, alt = p.get("e_score_correction_bias"), p.get("e_score_correction_bias_vl")
    if bias is not st.bias or alt is not st.alt or moe._modules.get("gate") is not st.gate \
            or not isinstance(router_logits, torch.Tensor) or router_logits.dim() != 2:
        return _stock_route(moe, router_logits, input_ids, num_token_non_padded)
    m = router_logits.shape[0]
    if m <= 0:
        return _stock_route(moe, router_logits, input_ids, num_token_non_padded)
    ids_sig = None if input_ids is None else (input_ids.dtype, input_ids.data_ptr() & 15, tuple(input_ids.shape),
                                              input_ids.stride(0) if input_ids.dim() else None)
    nt = num_token_non_padded
    nt_sig = None if nt is None else (nt.dtype, nt.data_ptr() & 15, tuple(nt.shape))
    dev = H["get_dev"]()
    key = (st.skey, m, router_logits.dtype, router_logits.shape[1], router_logits.stride(0), router_logits.stride(1),
           router_logits.data_ptr() & 15, ids_sig, nt_sig, dev)
    plan = _PLANS.get(key)
    if plan is None:
        return _learn(moe, key, router_logits, input_ids, nt, m, dev)
    kr = H["knobs_rt"]
    if (plan is False or kr.launch_enter_hook is not H["enter"] or kr.launch_exit_hook is not H["leave"]
            or H["enter"].calls or H["leave"].calls or H["fn"].pre_run_hooks
            or (H["fn"].debug or kr.debug) != plan.debug
            or H["knobs_comp"].instrumentation_mode != plan.imode or kr.add_stages_inspection_hook is not None
            or H["drvcfg"].active is not plan.drv or _KOK.get((plan.ckey, m.bit_length())) is False):
        return _stock_route(moe, router_logits, input_ids, num_token_non_padded)
    for name, gdict, val in plan.globals:      # stock's check that the kernel's globals did not change
        if gdict.get(name, _MISSING) != val:
            return _stock_route(moe, router_logits, input_ids, num_token_non_padded)
    weights = router_logits.new_empty((m, plan.k), dtype=torch.float32)
    ids = router_logits.new_empty((m, plan.k), dtype=torch.int32)
    if (weights.data_ptr() | ids.data_ptr()) & 15:
        return _stock_route(moe, router_logits, input_ids, num_token_non_padded)
    args = (router_logits, bias, alt, input_ids, nt, weights, ids) + plan.mid + (m,) + plan.tail
    plan.launch((m + plan.block_m - 1) // plan.block_m, 1, 1, H["raw_stream"](dev), plan.function, plan.coop,
                plan.pdl, plan.packed, None, None, None, None, None, plan.ann, plan.sig, args)
    out = H["STKO"](weights, ids, router_logits)
    vkey = (plan.ckey, m.bit_length())
    if vkey not in _KOK:
        ref = route_e5.route(moe, router_logits, input_ids, num_token_non_padded)
        same = (type(ref) is H["STKO"] and ref.router_logits is router_logits
                and _bits_equal(weights, ref.topk_weights) and _bits_equal(ids, ref.topk_ids))
        _KOK[vkey] = same
        if not same:
            _ST["router_on"] = False
        levers.announce(f"rt-route-k{len(_KOK)}", f"lever RT: router direct launch #{len(_KOK)} (first at {m} rows, "
                        f"layer {getattr(moe, 'layer_id', '?')}): "
                        f"{'== stock bit for bit: on' if same else '!= stock: router fast path OFF'}")
        if not same:
            return ref
    _ST["r_fast"] += 1
    return out


_MISSING = object()


def _learn(moe, key, router_logits, input_ids, nt, m, dev):
    """A key's first call: stock whole, its router launch recorded (pre-run hook) and turned into a direct-launch plan."""
    H = _H
    fn = H["fn"]
    seen = []

    def hook(*args, **kwargs):
        seen.append((args, dict(kwargs)))

    _ST["r_learn"] += 1
    if fn.pre_run_hooks:                         # someone else's hooks: stock (learned later, once they are gone)
        return route_e5.route(moe, router_logits, input_ids, nt)
    fn.pre_run_hooks.append(hook)
    try:
        out = route_e5.route(moe, router_logits, input_ids, nt)
    finally:
        fn.pre_run_hooks.remove(hook)
    try:
        plan = _make_plan(moe, out, seen, router_logits, input_ids, nt, m, dev)
    except Exception as e:  # noqa: BLE001 - a key we cannot mirror stays stock
        plan = _off(f"rt-plan-{type(e).__name__}", f"lever RT: router plan failed ({type(e).__name__}: "
                    f"{str(e)[:160]}): stock for that key")
    if len(_PLANS) >= _MAX_PLANS:
        _PLANS.clear()
    _PLANS[key] = plan
    return out


def _make_plan(moe, out, seen, router_logits, input_ids, nt, m, dev):
    H = _H
    st = moe.dsv4f_rt
    if len(seen) != 1:
        return _off(f"rt-plan-n{len(seen)}", f"lever RT: {len(seen)} router launches in one stock call: stock")
    args, kwargs = seen[0]
    if type(out) is not H["STKO"] or out.router_logits is not router_logits:
        return _off("rt-plan-out", f"lever RT: stock router output {type(out).__name__}: stock")
    w, i = out.topk_weights, out.topk_ids
    if not (len(args) == len(_ROUTER_ARGS) and args[0] is router_logits and args[1] is st.bias
            and args[2] is st.alt and args[3] is input_ids and args[4] is nt and args[5] is w and args[6] is i
            and args[8] == m and isinstance(args[7], torch.Tensor) and args[7] is not w and args[7] is not i):
        return _off("rt-plan-args", "lever RT: the recorded router launch is not on this call's tensors: stock")
    if kwargs.get("HAS_PACKED") is not False or kwargs.get("N_GROUP") != 1 or kwargs.get("BLOCK_M") is None:
        return _off("rt-plan-kind", f"lever RT: router launch kind (packed {kwargs.get('HAS_PACKED')}, groups "
                    f"{kwargs.get('N_GROUP')}): stock")
    if not (w.dtype == torch.float32 and i.dtype == torch.int32 and w.dim() == 2 and tuple(w.shape) == tuple(i.shape)
            and w.shape[0] == m and w.is_contiguous() and i.is_contiguous() and w.shape[1] == st.k):
        return _off("rt-plan-outs", "lever RT: stock router outputs not fresh contiguous [rows, top_k]: stock")
    kr = H["knobs_rt"]
    if kr.add_stages_inspection_hook is not None:
        return _off("rt-plan-stages", "lever RT: a Triton stage-inspection hook is set: stock")
    fn = H["fn"]
    if dev != torch.cuda.current_device():
        return _off("rt-plan-dev", "lever RT: device index mismatch: stock")
    kernel_cache, kernel_key_cache, _target, _backend, binder = fn.device_caches[dev]
    bound, specialization, options = binder(*args, **kwargs)
    ckey = H["compute_cache_key"](kernel_key_cache, specialization, options)
    kernel = kernel_cache.get(ckey)
    if not isinstance(kernel, H["CompiledKernel"]):
        return _off("rt-plan-kernel", "lever RT: stock's router kernel is not in Triton's cache: stock")
    launcher = kernel.run
    if (type(launcher) is not H["CudaLauncher"] or launcher.global_scratch_size or launcher.profile_scratch_size
            or getattr(launcher, "gsan_enabled", True) or kernel.function is None):
        return _off("rt-plan-launcher", "lever RT: router launcher needs scratch / sanitizer state: stock")
    names = list(bound.keys())
    values = list(bound.values())
    if tuple(names[:len(_ROUTER_ARGS)]) != _ROUTER_ARGS or any(a is not b for a, b in zip(values, args)):
        return _off("rt-plan-bind", "lever RT: Triton's bound arguments differ from the recorded call: stock")
    plan = _Plan()
    plan.ckey = (dev, ckey)
    plan.kernel = kernel                 # held: its module (and so the function handle below) stays loaded
    plan.launcher = launcher
    plan.launch = launcher.launch
    plan.function = kernel.function
    plan.packed = kernel.packed_metadata
    plan.coop = launcher.launch_cooperative_grid
    plan.pdl = launcher.launch_pdl
    plan.ann = launcher.arg_annotations
    plan.sig = launcher.kernel_signature
    plan.mid = (values[7],)
    plan.tail = tuple(values[9:])
    plan.block_m = int(bound["BLOCK_M"])
    plan.debug = kwargs.get("debug")
    plan.imode = kwargs.get("instrumentation_mode")
    plan.drv = H["drvcfg"].active
    plan.globals = tuple((name, gdict, val) for (name, _), (val, gdict) in fn.used_global_vals.items())
    plan.k = int(w.shape[1])
    if plan.block_m < 1 or (m + plan.block_m - 1) // plan.block_m != -(-m // plan.block_m):
        return _off("rt-plan-grid", "lever RT: router grid: stock")
    if levers.first("rt-plan"):
        levers.announce("rt-plan", f"lever RT: router plan learned at {m} rows (layer {getattr(moe, 'layer_id', '?')}, "
                        f"PDL {bool(plan.pdl)}, block_m {plan.block_m})")
    return plan


def after_topk(moe, topk_output):
    """The lifted forward_normal's route_e5.after_topk call: with no decode / verify batch in flight (E5's _on empty)
    stock's after_topk returns its input unchanged (RR's small path, SV's prune and prune_topk all return at their
    first check) and clears route_e5._LAST; here directly."""
    if _ON.get() is None or not _ST["router_on"] or _H["prune"]._on:
        return route_e5.after_topk(moe, topk_output)
    route_e5._LAST[0] = None
    return topk_output


# ---------------------------------------------------------------------------------------------------------------------
# WO-A
# ---------------------------------------------------------------------------------------------------------------------


def wo_a(o, wo_a_w, attn, fuse_mxfp8_quant):
    """The lifted attention's WO-A on eager EXTEND calls of 1..WOA_MAX_ROWS rows: stock's >= 4096-row strided bmm into
    the token-major output (stock below 4096: the einsum + a permute copy at wo_b's flatten). None: take stock."""
    if _ON.get() is None or not _ST["woa_on"]:
        return None
    t = o.shape[0]
    if not (0 < t <= WOA_MAX_ROWS) or _H["fa"]._apply_wo_a_bf16_matmul is not _H["woa_stock"]:
        _ST["w_stock"] += 1
        return None
    if (o.dim() != 3 or tuple(wo_a_w.shape) != WOA_SHAPE or o.shape[1] != WOA_SHAPE[0] or o.shape[2] != WOA_SHAPE[2]
            or o.dtype != torch.bfloat16 or wo_a_w.dtype != torch.bfloat16 or o.stride(2) != 1
            or o.stride(1) != WOA_SHAPE[2] or o.stride(0) < WOA_SHAPE[0] * WOA_SHAPE[2] or not wo_a_w.is_contiguous()
            or not o.is_cuda or torch.is_grad_enabled()):
        _ST["w_stock"] += 1
        return None
    cls = (t.bit_length(), o.stride(0))      # row class x operand layout (cuBLAS picks its kernel per both)
    ok = _WOK.get(cls)
    if ok is False:
        _ST["w_stock"] += 1
        return None
    result = o.new_empty((t, WOA_SHAPE[0], WOA_SHAPE[1]))   # stock: torch.empty(..., dtype=o.dtype, device=o.device)
    torch.bmm(o.transpose(0, 1), wo_a_w.transpose(1, 2), out=result.transpose(0, 1))
    if ok is None:
        ref = _H["woa_stock"](o, wo_a_w, is_decode=False, is_target_verify=False, is_prefill=True,
                              fast_path=attn.is_dsv41, fuse_mxfp8_quant=fuse_mxfp8_quant, fp8_grid=False,
                              emit_fp8=False)
        same = _bits_equal(result, ref)
        _WOK[cls] = same
        levers.announce(f"rt-woa{cls[0]}-{cls[1]}", f"lever RT: WO-A strided bmm, first at {t} rows (class {cls[0]}, "
                        f"row stride {cls[1]}): "
                        f"{'== stock einsum: on' if same else '!= stock: OFF for this class'}")
        if not same:
            return ref
    _ST["w_fast"] += 1
    return result


def stats() -> dict:
    """Counters (tests / logs)."""
    return {k: v for k, v in _ST.items() if isinstance(v, int) and not isinstance(v, bool)}
