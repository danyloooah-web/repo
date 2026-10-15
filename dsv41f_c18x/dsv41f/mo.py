"""Lever MO on the fork (levers.MO_MOE / MO_QUANT / MO_MHC): the host path of two ops of our layer replicas' eager
EXTEND forwards, cut down to the launches stock makes -- the routed NVFP4 MoE op (FlashInfer 0.7.1's trtllm-gen fused
MoE behind SGLang's ModelOptNvFp4FusedMoEMethod / flashinfer_trtllm runner) and the mega-mHC sublayer posts (deep_gemm
mega_mhc, SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1).

Why: an eager prefill pass of a few thousand rows on a long cached context (the live warm turns; the fork's breakable
prefill graphs cover contexts <= 16384 tokens only) is host-bound. B300 k4 trace of bundle f47a (632-2357-row passes,
40 layers, median over 80 layers): the MoE op takes 765-775 us of host per layer from the top-k launch to the finalize
launch (top-k -> NVFP4 quantize 187-199 us, quantize -> routing kernel 472-483, routing -> FC1 67-80, FC1 -> finalize
23), and a mega-mHC post 274-290 us from the previous launch to the mega_mhc launch (91 of them inside deep_gemm's C++
entry: 14 pointer checks, a memset).

MoE (MO_MOE). Stock: FusedMoE.forward -> forward_impl -> StandardDispatcher.dispatch -> ModelOptNvFp4FusedMoEMethod.apply
(FlashInferTrtllmFp4MoeQuantInfo) -> MoeRunner.run -> fused_experts_none_to_flashinfer_trtllm(_routed) ->
fused_experts_none_to_flashinfer_trtllm_fp4 (quantize_hidden_states_fp4, _get_fp4_moe_output_buffer, the fork's decode
pin) -> flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe (lever RR's wrapper) -> trtllm_fp4_block_scale_moe_op
(TrtllmMoERunner, tuning config, AutoTuner.choose_one through lever FMT's and the fork's pins) -> run_selected_tactic ->
moe_op.trtllm_fp4_block_scale_moe (native launcher: routing kernel, FC1, FC2, finalize).
Ours (the experts replica's forward): stock's quantize_hidden_states_fp4 (or, with MO_QUANT, the CuTe-DSL kernel object
FlashInfer's own cache holds for stock's call, launched with stock's arguments), stock's _get_fp4_moe_output_buffer,
then moe_op.trtllm_fp4_block_scale_moe with run_selected_tactic's argument list at the tactic stock's choose_one
returned for this call's key.
  learn   key = (rows, top_k, weights dtype, hidden, the layer's static op signature, lever FMT's pin, the fork's
          decode pin, AutoTuner state): the first call of a key runs stock whole under an outermost choose_one hook
          that records the tactic stock launched (exactly one choose_one of the op per call, else the key stays
          stock; no fast path or learning while the hook is not the outermost choose_one, or while the tuner tunes,
          has a thread override / skip context or serves an autotune_v2 store). A pass's first MoE layer learns its
          row count, the other layers of the pass take ours.
  verify  first fast call per (tile, config, PDL): ours first (output cloned), then stock on the same input (its
          tactic must be the learned one), every output byte compared; a difference or an exception turns that
          tactic off for the process. MO_QUANT: the first fast quantize per (K, PDL, fast-math, grid capped) is
          compared with stock's quantize the same way.
  gate    _ON (our forward, ForwardMode.EXTEND, no capture, not inside a breakable / piecewise prefill graph) and per
          call: no pre-quantized input, a standard unpacked top-k output, a bf16 [rows, hidden] CUDA input, no deferred
          finalize context, no armed lever-RR plan, a standard dispatcher that never had a dispatch / combine hook
          (single-batch-overlap's shared expert rides the combine), the AutoTuner state above; static (bind): the
          stock classes and methods named above without class- or instance-level overrides (lever NQ's pass-through
          dispatch excepted), the flashinfer_trtllm(_routed) runner without LoRA, no per-token activation scale /
          hidden padding / MegaMoE / DWDP / ascend fuse-EP / EP mapping, FusedMoE.reduce_results off (or a TP1 / EP1
          layer), FlashInfer's dist-aware MoE autotune off, FlashInfer 0.7.1.

mHC (MO_MHC). Stock: forward_hc_pre_from_prev -> mhc.run_attn_post / run_moe_post -> _compute_triplet (returns the
precomputed triplet) -> _post_fusion -> mhc_mega_boundary (5 allocations) -> deep_gemm.mega_mhc -> _C.mega_mhc.
Ours (both module functions wrapped, acting only under _ON): a plain-tensor sublayer output with a precomputed triplet
at a seam whose next hyper-connection takes the mega kernel (stock's own _post_fusion conditions, the static part
evaluated once per (hc, next) pair) -> the same 5 allocations and _C.mega_mhc with deep_gemm.mega_mhc's positional
argument list -> stock's HcState. First call per post kind (attn / ffn): stock and ours on the same inputs, every output
byte compared (a difference: stock for good).

Exactness: the same kernels with the same arguments (same tensors, scalars, tactic, PDL flag); only Python between the
launches is gone. Fail-closed: anything unexpected takes stock's path, each reason announced once (CACHEON_DSV4F lines).

Measured (fork/tests/mo_test.py + mo_micro.py, B300 box1, real FusedMoE at the per-rank shape, 64..16384 rows): every
output byte equal to stock at every size, the same 5 (6 from 12288 rows) kernels in the same order; MoE call host
554 -> 230 us at 64 rows and 690-703 -> 238-246 us at 600-16384 rows (-450 us, -65 %; box2's faster host: 362 -> 126 us),
of which MO_QUANT 57 -> 39 us. mHC post: stock 103-106 us vs ours 104-106 us (deep_gemm's C++ entry 31 us and the five
allocations 22 us dominate; the wrappers' Python is about what any wrapper costs), so levers.MO_MHC ships off.
Review (box2 t_morev1, box1 t_morev2): every GPU launch of a fast call equals stock's in name, grid, block, shared
memory and registers (5 launches, 6 from 12288 rows); mo._install_moe binds lever NQ's experts replica in place and a
private twin of shared experts (served FusedMoE untouched), both bitwise == stock; a second layer takes a key learned on
the first with its own tensors; a combine hook on the dispatcher sends the call to stock. Host at 600-16384 rows:
698-720 -> 240-246 us (box2 loaded), 382-396 -> 127-132 us (box1 quieter).
"""
import contextvars
import operator
import os
import types

import torch

from dsv41f import levers

OP = "flashinfer::trtllm_fp4_block_scale_moe"
FLASHINFER_VERSION = "0.7.1"       # the source the MoE / quantize mirrors were read from
DEEP_GEMM_VERSION = "0.2.2"        # deep_gemm.mega_mhc's argument list
_ON = contextvars.ContextVar("dsv41f_mo_on", default=None)
_LEARN = contextvars.ContextVar("dsv41f_mo_learn", default=None)
_H: dict = {}            # handles imported once at install
_TAC: dict = {}          # learn key -> tactic list as run_selected_tactic passes it | False (stock for this key)
_OK: dict = {}           # (tile, config, pdl) -> True (ours == stock bit for bit) / False (stock for good)
_QOK: dict = {}          # (k, pdl, fast-math, capped) -> True / False (MO_QUANT)
_QPLAN: dict = {}        # (k, device index) -> quantize plan | False (MO_QUANT)
_GSV: dict = {}          # id(global scale) -> (scale, data_ptr, kernel argument view)
_MHC_PLAN: dict = {}     # (id(hc), id(next)) -> (hc, next) when the seam takes the mega kernel | False
_MHC_OK: dict = {}       # "attn" / "ffn" -> True / False
_STATE = {"moe": "off (not installed)", "mhc": "off (not installed)", "fast": 0, "learn": 0, "stock": 0,
          "quant_fast": 0, "mhc_fast": 0, "mhc_stock": 0}
_FP8 = torch.float8_e4m3fn


def _off(tag, why):
    levers.announce(tag, why)
    return False


# ---------------------------------------------------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------------------------------------------------


def install(layers, served_layers) -> str:
    """model_node.prepare, after every other lever (so the choose_one hook is outermost and the MoE replicas exist)."""
    _STATE["moe"] = _install_moe(layers, served_layers) if getattr(levers, "MO_MOE", False) else "MoE off (levers)"
    _STATE["mhc"] = _install_mhc() if getattr(levers, "MO_MHC", False) else "mHC off (levers)"
    return f"{_STATE['moe']}; {_STATE['mhc']}"


def _common() -> str:
    """The forward gate's handles ('' = imported)."""
    if "ForwardMode" in _H:
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
    _H.update(ForwardMode=ForwardMode, in_bcg=is_in_breakable_cuda_graph, in_pcg=is_in_tc_piecewise_cuda_graph)
    return ""


def _install_moe(layers, served_layers) -> str:
    why = _common()
    if why:
        return f"MoE off ({why})"
    try:
        import flashinfer
        from flashinfer.autotuner import AutoTuner
        from flashinfer.fused_moe import core
        from flashinfer.fused_moe.da_config import is_trtllm_da_enabled
        from flashinfer.utils import device_support_pdl, get_compute_capability
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as ft
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.moe.topk import TopKOutputChecker
        from sglang.srt.layers.moe.utils import should_use_flashinfer_cutlass_moe_fp4_allgather
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod

        for mod, names in (
                (core, ("get_trtllm_moe_sm100_module", "RoutingInputMode", "_device_support_moe_pdl")),
                (ft, ("quantize_hidden_states_fp4", "_get_fp4_moe_output_buffer", "trtllm_moe_enable_pdl",
                      "get_activation_type", "_deferred_finalize_enabled", "_expert_tactic_pin",
                      "fused_experts_none_to_flashinfer_trtllm", "fused_experts_none_to_flashinfer_trtllm_routed")),
                (FusedMoE, ("forward", "forward_impl", "run_moe_core", "_dispatch_with_pre_quant")),
                (TopKOutputChecker, ("format_is_standard", "format_is_packed"))):
            for n in names:
                if not hasattr(mod, n):
                    raise AttributeError(f"{getattr(mod, '__name__', mod)}.{n}")
        tuner = AutoTuner.get()
        for n in ("_dirty_seq", "profiling_cache", "_file_configs", "_namespaced_records", "is_tuning_mode",
                  "_skip_ops_local", "_override_local", "_instance", "_managed_cache", "_v2_local"):
            if not hasattr(tuner, n):
                raise AttributeError(f"AutoTuner.{n}")
        # dispatch / combine hooks swap the dispatcher's bound methods through these (BaseDispatcher); the instance
        # side (_original_dispatch_func / _original_combine_func) is checked in _moe_static and per call
        for n in ("_override_dispatch_func", "_override_combine_func"):
            if not hasattr(StandardDispatcher, n):
                raise AttributeError(f"StandardDispatcher.{n}")
        if AutoTuner._instance is not tuner:
            raise AttributeError("AutoTuner singleton")
        native = core.get_trtllm_moe_sm100_module().moe_op.trtllm_fp4_block_scale_moe
    except Exception as e:  # noqa: BLE001 - a stack without these internals keeps stock's MoE
        return f"MoE off ({type(e).__name__}: {str(e)[:160]})"
    if getattr(flashinfer, "__version__", "?") != FLASHINFER_VERSION:
        return f"MoE off (FlashInfer {getattr(flashinfer, '__version__', '?')}; mirror read from {FLASHINFER_VERSION})"
    if is_trtllm_da_enabled():
        return "MoE off (FlashInfer dist-aware MoE autotune on: stock's DA path)"
    if should_use_flashinfer_cutlass_moe_fp4_allgather():
        return "MoE off (FP4 all-gather dispatch)"
    from sglang.srt.layers.moe import topk as topk_mod

    std = tuple(c for c in (getattr(topk_mod, "StandardTopKOutput", None),
                            getattr(topk_mod, "StandardTopKOutputPacked", None)) if c is not None)
    if not std or getattr(topk_mod, "PackedTopKOutput", None) in std:
        return "MoE off (top-k output classes)"
    _H.update(core=core, ft=ft, AutoTuner=AutoTuner, native=native, FusedMoE=FusedMoE, MoeRunner=MoeRunner,
              StandardDispatcher=StandardDispatcher, NvFp4=ModelOptNvFp4FusedMoEMethod, checker=TopKOutputChecker,
              unpacked=int(core.RoutingInputMode.UnpackedPrecomputed), moe_pdl=core._device_support_moe_pdl,
              device_support_pdl=device_support_pdl, cc=get_compute_capability, tuner=tuner, std=frozenset(std),
              min_rows=max(1, int(getattr(levers, "MO_MOE_MIN_ROWS", 1))))
    try:
        from dsv41f import fmt as _fmt
        _H["fmt_pin"] = getattr(_fmt, "_PIN", None)
    except ImportError:
        _H["fmt_pin"] = None
    try:
        from dsv41f import rr as _rr
        _H["rr_plan"] = getattr(_rr, "_PLAN", None)
    except ImportError:
        _H["rr_plan"] = None
    quant = _install_quant() if getattr(levers, "MO_QUANT", False) else "quantize: stock's (levers)"
    if "hook" not in _H:
        # the fork's decode pin wraps choose_one lazily at its first use; installed here first, ours stays outermost
        try:
            ft._install_expert_tactic_pin()
        except Exception as e:  # noqa: BLE001 - without it a pinned call records nothing: that key stays stock
            levers.announce("mo-pin", f"lever MO: the fork's decode pin not pre-installed ({type(e).__name__})")
        wrapped = AutoTuner.choose_one

        def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
            out = wrapped(self, custom_op, runners, tuning_config, inputs, **kwargs)
            box = _LEARN.get()
            if box is not None and custom_op == OP:
                box.append(out[1])
            return out

        AutoTuner.choose_one = choose_one
        _H["hook"] = True
        _H["hook_fn"] = choose_one      # _tuner_state: stock whenever this is not the outermost choose_one
    bound, why = 0, {}
    for replica, served in zip(layers, served_layers):
        moe = replica._modules.get("mlp")
        served_moe = served._modules.get("mlp")
        experts = moe._modules.get("experts") if moe is not None else None
        if experts is None:
            continue
        if moe is served_moe:
            why["MoE module not a private replica"] = why.get("MoE module not a private replica", 0) + 1
            continue
        reason = _moe_static(experts)
        if reason:
            why[reason] = why.get(reason, 0) + 1
            continue
        if experts is served_moe._modules.get("experts"):
            import copy

            twin = copy.copy(experts)
            twin._modules = dict(experts._modules)
            moe._modules["experts"] = twin
            experts = twin
        experts.dsv4f_mo_stock = experts.forward        # whatever forward the replica resolved to (stock's)
        experts.dsv4f_mo_act = _H["ft"].get_activation_type(experts.quant_method.runner.config.activation,
                                                            is_gated=experts.quant_method.runner.config.is_gated)
        experts.dsv4f_mo_statics = None                  # the native op's per-layer tensor arguments, cached
        experts.forward = types.MethodType(_experts_forward, experts)
        bound += 1
    skipped = "; ".join(f"{n} x {k}" for k, n in why.items())
    return (f"MoE on {bound} experts replicas (eager EXTEND, learned tactic per row count, first use per tactic "
            f"checked bitwise){'; skipped: ' + skipped if skipped else ''}; {quant}")


def _moe_static(exp) -> str:
    """Why the experts module cannot take the fast path ('' = it can)."""
    H = _H
    if not isinstance(exp, H["FusedMoE"]):
        return f"{type(exp).__name__} is not a FusedMoE"
    cls, fm = type(exp), H["FusedMoE"]
    if (cls.forward is not fm.forward or cls.forward_impl is not fm.forward_impl or cls.run_moe_core is not fm.run_moe_core
            or cls._dispatch_with_pre_quant is not fm._dispatch_with_pre_quant):
        return f"{cls.__name__} overrides forward / forward_impl / run_moe_core / _dispatch_with_pre_quant"
    if (getattr(exp.forward, "__func__", None) is not fm.forward
            or getattr(exp.forward_impl, "__func__", None) is not fm.forward_impl
            or getattr(exp.run_moe_core, "__func__", None) is not fm.run_moe_core):
        return "instance-level forward / forward_impl / run_moe_core"
    dwp = getattr(exp._dispatch_with_pre_quant, "__func__", None)
    if dwp is not fm._dispatch_with_pre_quant and not _nq_dispatch(dwp):
        return "instance-level _dispatch_with_pre_quant"
    qm = getattr(exp, "quant_method", None)
    if type(qm) is not H["NvFp4"] or getattr(qm.apply, "__func__", None) is not H["NvFp4"].apply:
        return f"quant method {type(qm).__name__}"
    backend = getattr(qm, "_moe_runner_backend", None)
    if backend is None or not (backend.is_flashinfer_trtllm() or backend.is_flashinfer_trtllm_routed()):
        return f"runner backend {backend}"
    if getattr(getattr(qm, "quant_config", None), "use_per_token_activation", True):
        return "per-token activation scales"
    runner = getattr(qm, "runner", None)
    ft = H["ft"]
    if (type(runner) is not H["MoeRunner"] or getattr(runner, "lora_enabled", True)
            or getattr(runner.run, "__func__", None) is not H["MoeRunner"].run
            or runner.fused_func not in (ft.fused_experts_none_to_flashinfer_trtllm,
                                         ft.fused_experts_none_to_flashinfer_trtllm_routed)):
        return "MoE runner (fused func / LoRA)"
    cfg = runner.config
    if cfg.activation not in ("silu", "relu2", "gelu", "situ") or getattr(qm, "moe_runner_config", None) is None:
        return f"activation {cfg.activation}"
    if (getattr(exp, "_mega_moe_nvfp4", True) or getattr(exp, "trtllm_padded_hidden_size", 0) is not None
            or getattr(exp, "_use_ascend_fuseep", True) or getattr(exp, "_dwdp_bound", True)):
        return "MegaMoE / hidden padding / ascend fuse-EP / DWDP"
    if exp.reduce_results and (exp.moe_tp_size > 1 or exp.moe_ep_size > 1):
        return "FusedMoE reduces its own output"
    disp = getattr(exp, "dispatcher", None)
    sd = H["StandardDispatcher"]
    if (type(disp) is not sd or getattr(disp.dispatch, "__func__", None) is not sd.dispatch
            or getattr(disp.combine, "__func__", None) is not sd.combine):
        return f"dispatcher {type(disp).__name__}"
    if (not hasattr(disp, "_original_dispatch_func") or not hasattr(disp, "_original_combine_func")
            or disp._original_dispatch_func is not None or disp._original_combine_func is not None):
        return "dispatcher with dispatch / combine hooks"
    if getattr(disp, "local_expert_mapping", None) is not None or (disp.moe_ep_size > 1
                                                                     and not disp.skip_local_expert_mapping):
        return "EP expert mapping"
    tensors = (getattr(exp, "w13_weight", None), getattr(exp, "w2_weight", None),
               getattr(exp, "w13_weight_scale", None), getattr(exp, "w2_weight_scale", None),
               getattr(exp, "g1_scale_c", None), getattr(exp, "g1_alphas", None), getattr(exp, "g2_alphas", None),
               getattr(exp, "w13_input_scale_quant", None))
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return "missing NVFP4 weight / scale tensors"
    if exp.w13_weight.dtype != torch.uint8 or exp.w2_weight.dtype != torch.uint8:
        return "weights not packed FP4"
    return ""


def _nq_dispatch(fn) -> bool:
    """Lever NQ's experts-replica _dispatch_with_pre_quant (nq.bind_experts): stock's for every input except its own
    NvFp4PreQuant -- and ours runs only without a pre-quantized input, so skipping it skips stock's."""
    try:
        from dsv41f import nq as _nq
    except ImportError:
        return False
    return (fn is not None and getattr(fn, "__module__", None) == _nq.__name__
            and getattr(fn, "__qualname__", None) == "bind_experts.<locals>.dispatch_with_pre_quant")


# ---------------------------------------------------------------------------------------------------------------------
# the forward gate
# ---------------------------------------------------------------------------------------------------------------------


def forward(fn, forward_batch, *args, **kwargs):
    """Run our model forward with lever MO's wrappers active on eager EXTEND forwards only."""
    if "ForwardMode" not in _H or _STATE["moe"].startswith("MoE off") and _STATE["mhc"].startswith("mHC off"):
        return fn(*args, **kwargs)
    mode = getattr(forward_batch, "forward_mode", None)
    if (mode is None or mode != _H["ForwardMode"].EXTEND or torch.cuda.is_current_stream_capturing()
            or _H["in_bcg"]() or _H["in_pcg"]()):
        return fn(*args, **kwargs)
    token = _ON.set(True)
    try:
        return fn(*args, **kwargs)
    finally:
        _ON.reset(token)


# ---------------------------------------------------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------------------------------------------------


def _tuner_state():
    """The AutoTuner state a cached tactic depends on, or None while it tunes / has an override or skip context on
    this thread / serves an autotune_v2 store or context (winner partitions and disk lookups the tuple below does not
    see), or when the singleton was replaced or our choose_one hook is no longer the outermost (a later wrapper could
    change the tactic after ours records it)."""
    tuner = _H["tuner"]
    AT = _H["AutoTuner"]
    if (AT._instance is not tuner or AT.choose_one is not _H["hook_fn"] or tuner.is_tuning_mode
            or tuner._managed_cache is not None or tuner._v2_local.active
            or getattr(tuner._skip_ops_local, "stack", None) or getattr(tuner._override_local, "stack", None)):
        return None
    return (tuner._dirty_seq, len(tuner.profiling_cache), len(tuner._file_configs), id(tuner._namespaced_records))


def _native_tactic(t):
    """run_selected_tactic's tactic argument for choose_one's answer, as a list of two ints (None: not a form ours
    reproduces). Besides lists / tuples this takes any 2-element sequence of integral values -- the TVM-FFI array the
    op's own valid-tactic list holds, which lever FMT's pin returns: stock hands that object to the native op, ours the
    same two int64 values as a list (one Array<int64> either way)."""
    if isinstance(t, bool) or t is None or isinstance(t, (str, bytes)):
        return None
    if isinstance(t, int):
        return [-1, -1] if t == -1 else None
    try:
        vals = list(t)
    except TypeError:
        return None
    if len(vals) != 2 or any(isinstance(v, bool) for v in vals):
        return None
    try:
        return [operator.index(vals[0]), operator.index(vals[1])]
    except TypeError:
        return None


def _experts_forward(self, hidden_states, topk_output, pre_quant_input=None):
    """The experts replica's forward: stock's FusedMoE.forward, or the native op on stock's arguments."""
    stock = self.dsv4f_mo_stock
    if _ON.get() is None or pre_quant_input is not None:
        return stock(hidden_states, topk_output, pre_quant_input)
    H = _H
    hs = hidden_states
    disp = self.dispatcher
    # a dispatch / combine hook (e.g. the single-batch-overlap shared expert that forward_normal registers on the
    # dispatcher right before this call) runs inside stock's dispatch / combine, which ours skips: stock
    if (type(topk_output) not in H["std"] or type(hs) is not torch.Tensor or hs.dtype is not torch.bfloat16
            or hs.dim() != 2 or not hs.is_cuda or hs.shape[0] < H["min_rows"] or self._dwdp_bound
            or type(disp) is not H["StandardDispatcher"] or disp._original_dispatch_func is not None
            or disp._original_combine_func is not None or H["ft"]._deferred_finalize_enabled.get()
            or (H["rr_plan"] is not None and H["rr_plan"].get() is not None)):
        _STATE["stock"] += 1
        return stock(hidden_states, topk_output, pre_quant_input)
    ids, weights = topk_output.topk_ids, topk_output.topk_weights
    rows = hs.shape[0]
    if ids.dtype is not torch.int32 or ids.dim() != 2 or ids.shape[0] != rows or weights.shape != ids.shape:
        _STATE["stock"] += 1
        return stock(hidden_states, topk_output, pre_quant_input)
    state = _tuner_state()
    if state is None:
        _STATE["stock"] += 1
        return stock(hidden_states, topk_output, pre_quant_input)
    try:
        statics, sig = _statics(self)
    except Exception as e:  # noqa: BLE001 - a layer whose tensors stock reads differently: stock
        levers.announce("mo-statics", f"lever MO: static MoE arguments ({type(e).__name__}: {str(e)[:120]}): stock")
        _STATE["stock"] += 1
        return stock(hidden_states, topk_output, pre_quant_input)
    pin = H["fmt_pin"].get() if H["fmt_pin"] is not None else None
    dpin = H["ft"]._expert_tactic_pin.get()
    # sig: the layer's static op shape (experts, intermediate, weight shapes / dtypes, activation), which the
    # tuner's cache key (TrtllmMoERunner.get_cache_key_extras) and the fork's decode pin also depend on
    key = (rows, ids.shape[1], weights.dtype, hs.shape[1], hs.device, sig,
           None if pin is None else (tuple(pin[0]), pin[1]), None if dpin is None else tuple(dpin), state)
    entry = _TAC.get(key)
    if entry is None:
        return _learn(stock, key, hidden_states, topk_output)
    if entry is False:
        _STATE["stock"] += 1
        return stock(hidden_states, topk_output, pre_quant_input)
    tac, pdl, vkey = entry
    ok = _OK.get(vkey)
    if ok is None:
        return _verify(self, stock, key, vkey, tac, pdl, hidden_states, topk_output)
    if not ok:
        _STATE["stock"] += 1
        return stock(hidden_states, topk_output, pre_quant_input)
    try:
        out = _fast(self, hs, ids, weights, tac, pdl, statics)
    except Exception as e:  # noqa: BLE001 - never seen after the bitwise check; stock for this tactic from now on
        _OK[vkey] = False
        levers.announce(f"mo-err-{vkey}", f"lever MO: MoE tactic {list(tac)} failed at {rows} rows "
                        f"({type(e).__name__}: {str(e)[:120]}): stock for good")
        return stock(hidden_states, topk_output, pre_quant_input)
    _STATE["fast"] += 1
    return out


def _learn(stock, key, hidden_states, topk_output):
    box = []
    token = _LEARN.set(box)
    try:
        out = stock(hidden_states, topk_output, None)
    finally:
        _LEARN.reset(token)
    _STATE["learn"] += 1
    if len(_TAC) >= 4096:
        _TAC.clear()            # one entry per (row count, pins, tuner state): relearn rather than grow
    tac = _native_tactic(box[0]) if len(box) == 1 else None
    if tac is None:
        _TAC[key] = False
        levers.announce(f"mo-learn-{len(box)}", f"lever MO: a {key[0]}-row MoE call made {len(box)} choose_one "
                        f"calls ({box[:2]}): that key stays stock")
    else:
        # stock's enable_pdl for this row count (trtllm_moe_enable_pdl, then the op's _device_support_moe_pdl gate)
        pdl = bool(_H["ft"].trtllm_moe_enable_pdl(key[0])) and bool(_H["moe_pdl"](hidden_states.device))
        _TAC[key] = (tac, pdl, (tac[0], tac[1], pdl))
        if levers.first("mo-learn"):
            levers.announce("mo-learn", f"lever MO: first learned MoE tactic {tac} at {key[0]} rows")
    return out


def _verify(exp, stock, key, vkey, tac, pdl, hidden_states, topk_output):
    """Ours first (cloned), then stock on the same input; their bytes decide the tactic for the process."""
    why = ""
    got = None
    try:
        got = _fast(exp, hidden_states, topk_output.topk_ids, topk_output.topk_weights, tac, pdl).clone()
    except Exception as e:  # noqa: BLE001 - any failure keeps stock for this tactic
        why = f"{type(e).__name__}: {str(e)[:160]}"
    box = []
    token = _LEARN.set(box)
    try:
        ref = stock(hidden_states, topk_output, None)
    finally:
        _LEARN.reset(token)
    if not why and (len(box) != 1 or _native_tactic(box[0]) != tac):
        # stock moved off the learned tactic for this key: the key stays stock; the tactic itself stays unverified
        # (another key may still verify it), so this is announced under its own tag
        _TAC[key] = False
        levers.announce(f"mo-drift-{vkey}", f"lever MO: at {key[0]} rows stock's tactic {box[:2]} != learned "
                        f"{tac}: that key stays stock")
        return ref
    if not why:
        if not (isinstance(ref, torch.Tensor) and ref.shape == got.shape and ref.dtype == got.dtype
                and ref.dtype == torch.bfloat16 and torch.equal(ref.view(torch.int16), got.view(torch.int16))):
            why = "output bytes differ"
    ok = not why
    _OK[vkey] = ok
    levers.announce(f"mo-verify-{vkey}", f"lever MO: MoE tactic {list(tac)} PDL {pdl} first at {key[0]} rows: "
                    f"{'ours == stock bit for bit: on' if ok else 'OFF: ' + why}")
    return ref


def _quantize(exp, hs):
    """quantize_hidden_states_fp4(hs, exp.w13_input_scale_quant): stock's, or the same kernel launched directly."""
    if "nvq" in _H:
        out = _fast_quantize(exp, hs)
        if out is not None:
            return out
    return _H["ft"].quantize_hidden_states_fp4(hs, exp.w13_input_scale_quant)


_STATIC = ("w13_weight", "w13_weight_scale", "gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "w2_weight",
           "w2_weight_scale", "g1_scale_c", "g1_alphas", "g2_alphas")


_SIG: dict = {}          # a layer's static op shape -> small int (part of the learn key)


def _statics(exp):
    """(the native op's per-layer tensor arguments exactly as stock derives them on every call -- ModelOptNvFp4FusedMoE
    Method.apply's .data, fused_experts_none_to_flashinfer_trtllm_fp4's FP8 scale views --, the layer's static op
    signature), cached on the replica while each source is the same parameter object on the same storage (SGLang
    keeps parameter identity and storage stable after loading -- copy_or_rebind_param copies in place or rebinds new
    storage; the decode CUDA graphs bake the same pointers)."""
    params = exp._parameters
    cache = getattr(exp, "dsv4f_mo_statics", None)
    if cache is not None:
        srcs, ptrs, args, sig = cache
        for name, src, ptr in zip(_STATIC, srcs, ptrs):
            p = params.get(name)
            if p is not src or (p is not None and p.data_ptr() != ptr):
                break
        else:
            return args, sig
    srcs = (getattr(exp, "w13_weight", None), getattr(exp, "w13_weight_scale", None),
            getattr(exp, "gemm1_alpha", None), getattr(exp, "gemm1_beta", None),
            getattr(exp, "gemm1_clamp_limit", None), getattr(exp, "w2_weight", None),
            getattr(exp, "w2_weight_scale", None), getattr(exp, "g1_scale_c", None), getattr(exp, "g1_alphas", None),
            getattr(exp, "g2_alphas", None))
    args = (srcs[0].data, srcs[1].data.view(_FP8), srcs[2].data if srcs[2] is not None else None,
            srcs[3].data if srcs[3] is not None else None, srcs[4].data if srcs[4] is not None else None,
            srcs[5].data, srcs[6].data.view(_FP8), srcs[7].data, srcs[8].data, srcs[9].data)
    # what the tuner's choice and the fork's decode pin depend on beyond the call's rows / top-k / hidden
    shape = (int(exp.num_experts), int(exp.num_local_experts), int(exp.moe_ep_rank),
             int(exp.intermediate_size_per_partition), int(exp.dsv4f_mo_act),
             tuple(None if t is None else (tuple(t.shape), t.dtype) for t in args))
    sig = _SIG.setdefault(shape, len(_SIG))
    cacheable = all(params.get(name) is src for name, src in zip(_STATIC, srcs))
    exp.dsv4f_mo_statics = ((srcs, tuple(t.data_ptr() if t is not None else None for t in srcs), args, sig)
                            if cacheable else None)
    return args, sig


def _fast(exp, hs, ids, weights, tac, pdl, statics=None):
    """fused_experts_none_to_flashinfer_trtllm_fp4 + trtllm_fp4_block_scale_routed_moe + trtllm_fp4_block_scale_moe_op
    for this call, launching only what they launch: the quantize, the output allocation, the native op."""
    H = _H
    rows = hs.shape[0]
    hs_fp4, hs_scale_linear = _quantize(exp, hs)
    hs_scale = hs_scale_linear.view(_FP8).reshape(*hs_scale_linear.shape[:-1], -1)
    hidden = hs_fp4.shape[-1] * 2 if hs_fp4.dtype == torch.uint8 else hs_fp4.shape[-1]
    out = H["ft"]._get_fp4_moe_output_buffer(num_tokens=rows, hidden_size=hidden, dtype=hs.dtype,
                                             device=hs_fp4.device, symmetric=True)
    ids = ids.contiguous()
    weights = weights.contiguous()
    w13, w13s, g1_alpha, g1_beta, g1_clamp, w2, w2s, g1c, g1a, g2a = statics if statics is not None else \
        _statics(exp)[0]
    H["native"](
        H["unpacked"], None, ids, weights, None, hs_fp4, hs_scale, w13, w13s, None, None, g1_alpha, g1_beta, g1_clamp,
        w2, w2s, None, g1c, g1a, g2a, None, exp.num_experts, ids.shape[1], 0, 0, 0, exp.intermediate_size_per_partition,
        exp.moe_ep_rank * exp.num_local_experts, exp.num_local_experts, None, 1, True, pdl, exp.dsv4f_mo_act, out,
        list(tac), True, None, [], [], False, None, None)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# MO_QUANT: the routed input's NVFP4 quantize (fp4_quantize backend cute-dsl, linear scale layout)
# ---------------------------------------------------------------------------------------------------------------------


def _install_quant() -> str:
    try:
        from flashinfer.cute_dsl.utils import get_num_sm
        from flashinfer.quantization import fp4_quantization as fq
        from flashinfer.quantization.kernels import nvfp4_quantize as nvq
        from flashinfer.quantization.nvfp4_quantization_utils import resolve_nvfp4_4over6
        from sglang.srt.layers.quantization import fp4_utils

        for n in ("_get_compiled_kernel_nvfp4", "_should_use_tma", "_env_flag_enabled", "SF_LAYOUT_LINEAR",
                  "_BLOCKS_PER_SM", "NVFP4_SF_VEC_SIZE", "nvfp4_quantize_cute_dsl"):
            if not hasattr(nvq, n):
                raise AttributeError(f"nvfp4_quantize.{n}")
        if getattr(fp4_utils, "_flashinfer_fp4_quantize_backend", None) != "cute-dsl":
            return "quantize: stock's (SGLang's fp4_quantize backend is not cute-dsl)"
        # stock's chain is the one mirrored here: flashinfer_trtllm.fp4_quantize = fp4_utils' custom op over
        # flashinfer.fp4_quantize (cute-dsl) -> nvfp4_quantize_cute_dsl, 16-element SF blocks
        if (getattr(_H["ft"], "fp4_quantize", None) is not getattr(fp4_utils, "fp4_quantize", 0)
                or getattr(fp4_utils, "_flashinfer_fp4_quantize", None) is not fq.fp4_quantize
                or nvq.NVFP4_SF_VEC_SIZE != 16):
            return "quantize: stock's (fp4_quantize chain / SF block size not the one mirrored)"
        if not hasattr(nvq._get_compiled_kernel_nvfp4, "cache_info"):
            return "quantize: stock's (kernel getter not functools-cached)"
    except Exception as e:  # noqa: BLE001
        return f"quantize: stock's ({type(e).__name__}: {str(e)[:120]})"
    _H.update(nvq=nvq, fq=fq, num_sm=get_num_sm, resolve_4over6=resolve_nvfp4_4over6)
    return "quantize: FlashInfer's cached NVFP4 kernel launched directly (first use per shape class checked)"


def _quant_plan(hs):
    """(kernel, SF blocks per CTA, grid cap, (open key, capped key)) of stock's nvfp4_quantize_cute_dsl branch for bf16
    inputs of this K on this device, or False where stock takes another branch. Built once per (K, device): the
    environment switches stock reads per call (4over6 recipe, TMA override, fast-math) are process-constant."""
    key = (hs.shape[1], hs.get_device())
    plan = _QPLAN.get(key)
    if plan is not None:
        return plan
    H = _H
    nvq = H["nvq"]
    k = hs.shape[1]
    plan = False
    if (k % nvq.NVFP4_SF_VEC_SIZE == 0 and H["resolve_4over6"]() is None and H["cc"](hs.device) != (10, 7)
            and "FLASHINFER_NVFP4_QUANTIZE_USE_TMA" not in os.environ):
        # not SM107 and no TMA override: _should_use_tma is False at every M; the SM107-only tile / FP8-pair /
        # blocks-per-SM branches do not apply
        enable_pdl = bool(H["device_support_pdl"](hs.device))
        fast_math_off = bool(nvq._env_flag_enabled("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"))
        getter = nvq._get_compiled_kernel_nvfp4
        misses = getter.cache_info().misses
        # stock's call, argument for argument (functools.cache keys on the positional / keyword split and order)
        kernel_fn, block_unit = getter("bfloat16", k, nvq.SF_LAYOUT_LINEAR, enable_pdl, fast_math_off, None,
                                       global_scale_is_tensor=True, pair_fp8_blocks=False)
        if getter.cache_info().misses != misses:
            levers.announce("mo-quant-compile", "lever MO: the NVFP4 quantize kernel was not in FlashInfer's cache "
                            "(compiled now)")
        plan = (kernel_fn, int(block_unit), H["num_sm"](hs.device) * nvq._BLOCKS_PER_SM,
                ((k, enable_pdl, fast_math_off, False), (k, enable_pdl, fast_math_off, True)))
    _QPLAN[key] = plan
    return plan


def _gs_view(gs, device):
    """nvfp4_quantize_cute_dsl's global scale argument (fp4_quantize's .to(device, float32), then its own
    .float().reshape(1).contiguous().to(device)), kept per scale tensor while its storage is the same; None when the
    scale is not a one-element tensor (stock's call is then left to stock)."""
    entry = _GSV.get(id(gs))
    if entry is not None and entry[0] is gs and entry[1] == gs.data_ptr() and gs.dtype is torch.float32:
        return entry[2]
    if not isinstance(gs, torch.Tensor) or gs.numel() != 1:
        return None
    view = gs.to(device=device, dtype=torch.float32).float().reshape(1).contiguous().to(device)
    if view.data_ptr() == gs.data_ptr() and view.device == gs.device:
        # an alias of the scale's own storage: every launch reads its current bytes, as stock's fresh view does.
        # A converted copy (another dtype / device) is never cached: an in-place update of the scale would leave
        # a cached copy stale, so it is rebuilt per call like stock's.
        _GSV[id(gs)] = (gs, gs.data_ptr(), view)
    return view


def _fast_quantize(exp, hs):
    """(hs_fp4, hs_sf) with quantize_hidden_states_fp4's bytes and geometry; None = stock's quantize."""
    if hs.dtype is not torch.bfloat16 or hs.dim() != 2:
        return None                 # the plan below is stock's bf16 [M, K] branch only
    gs = exp.w13_input_scale_quant
    try:
        plan = _quant_plan(hs)
        gsv = _gs_view(gs, hs.device) if plan is not False else None
    except Exception as e:  # noqa: BLE001
        _QPLAN[(hs.shape[1], hs.get_device())] = False
        levers.announce("mo-quant-plan", f"lever MO: quantize plan failed ({type(e).__name__}: {str(e)[:120]}): "
                        f"stock's")
        return None
    if plan is False or gsv is None:
        if levers.first("mo-quant-branch"):
            levers.announce("mo-quant-branch", f"lever MO: a {tuple(hs.shape)} quantize takes another FlashInfer "
                            f"branch: stock's")
        return None
    kernel_fn, block_unit, target, keys = plan
    m, k = hs.shape
    total = m * (k // 16)
    blocks = (total + block_unit - 1) // block_unit
    capped = blocks >= target
    if capped:
        blocks = target
    qkey = keys[capped]
    ok = _QOK.get(qkey)
    if ok is False:
        return None
    x = hs.contiguous()
    fp4 = torch.empty(m, k // 2, dtype=torch.uint8, device=hs.device)
    sf = torch.empty(total, dtype=torch.uint8, device=hs.device)
    kernel_fn(x, fp4, sf, m, total, blocks, gsv)
    hs_sf = sf.view(_FP8).view(m, k // 16)
    if ok is None:
        ref_q, ref_sf = _H["ft"].quantize_hidden_states_fp4(hs, gs)
        same = (ref_q.shape == fp4.shape and ref_sf.shape == hs_sf.shape and ref_q.dtype == fp4.dtype
                and ref_sf.dtype == hs_sf.dtype and ref_q.stride() == fp4.stride()
                and ref_sf.stride() == hs_sf.stride() and torch.equal(ref_q, fp4)
                and torch.equal(ref_sf.view(torch.uint8), hs_sf.view(torch.uint8)))
        _QOK[qkey] = same
        levers.announce(f"mo-quant-{qkey}", f"lever MO: NVFP4 quantize K={k} PDL {qkey[1]} grid "
                        f"{'capped' if capped else 'open'} first at {m} rows: "
                        f"{'== stock bit for bit: on' if same else 'MISMATCH: stock for good'}")
        return ref_q, ref_sf
    _STATE["quant_fast"] += 1
    return fp4, hs_sf


# ---------------------------------------------------------------------------------------------------------------------
# mHC
# ---------------------------------------------------------------------------------------------------------------------


def _install_mhc() -> str:
    why = _common()
    if why:
        return f"mHC off ({why})"
    try:
        import deep_gemm
        from sglang.kernels.ops.layernorm import mhc_mega
        from sglang.srt.models import deepseek_v4_mhc as m

        for n in ("run_attn_post", "run_moe_post", "_post_fusion", "_can_use_mega_mhc", "mhc_mega_boundary",
                  "HcState", "HcNormed", "AttnOutput", "MoEOutput"):
            if not hasattr(m, n):
                raise AttributeError(f"deepseek_v4_mhc.{n}")
        if m.mhc_mega_boundary is not mhc_mega.mhc_mega_boundary:
            raise AttributeError("deepseek_v4_mhc.mhc_mega_boundary rebound")
        if getattr(deep_gemm, "__version__", "?") != DEEP_GEMM_VERSION:
            raise AttributeError(f"deep_gemm {getattr(deep_gemm, '__version__', '?')} (argument list read from "
                                 f"{DEEP_GEMM_VERSION})")
        launch = deep_gemm._C.mega_mhc
    except Exception as e:  # noqa: BLE001 - stock's posts
        return f"mHC off ({type(e).__name__}: {str(e)[:160]})"
    if "mhc_stock" in _H:
        return "mHC on (already wrapped)"
    _H.update(mhc=m, mega=launch, mhc_stock={"attn": m.run_attn_post, "ffn": m.run_moe_post})
    stock_attn, stock_ffn = m.run_attn_post, m.run_moe_post

    def run_attn_post(hc, out, residual, *, stats_stream, next, world_size, precomputed=None):
        if _ON.get() is not None:
            got = _post("attn", stock_attn, hc, out, residual, stats_stream, next, world_size, precomputed)
            if got is not None:
                return got
        return stock_attn(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                          precomputed=precomputed)

    def run_moe_post(hc, out, residual, *, stats_stream, next, world_size, precomputed=None):
        if _ON.get() is not None:
            got = _post("ffn", stock_ffn, hc, out, residual, stats_stream, next, world_size, precomputed)
            if got is not None:
                return got
        return stock_ffn(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                         precomputed=precomputed)

    m.run_attn_post = run_attn_post
    m.run_moe_post = run_moe_post
    return "mHC on (mega-mHC posts of eager EXTEND forwards: deep_gemm's launch without the wrappers' Python)"


def _mega_seam(hc, nxt) -> bool:
    """Whether stock's _post_fusion takes mhc_mega_boundary at this seam (its static conditions; the row count is
    checked per call). Cached per (hc, next) object pair; the entry holds both objects, so their ids stay theirs."""
    key = (id(hc), id(nxt))
    entry = _MHC_PLAN.get(key)
    if entry is None or entry[0] is not hc or entry[1] is not nxt:
        m = _H["mhc"]
        try:
            mega = bool(nxt is not None and nxt.norm_fusable and nxt.hc is not None and m._can_use_mega_mhc(hc.cfg))
        except Exception:  # noqa: BLE001
            mega = False
        entry = _MHC_PLAN[key] = (hc, nxt, mega)
    return entry[2]


def _post(kind, stock, hc, out, residual, stats_stream, nxt, world_size, precomputed):
    """Ours for a plain-tensor output with a precomputed triplet at a mega seam; None = stock's."""
    if (precomputed is None or type(out) is not torch.Tensor or out.dim() != 2 or out.shape[0] == 0
            or not isinstance(residual, torch.Tensor) or residual.dim() != 3 or residual.shape[0] != out.shape[0]):
        _STATE["mhc_stock"] += 1
        return None
    if not _mega_seam(hc, nxt):
        _STATE["mhc_stock"] += 1
        return None
    ok = _MHC_OK.get(kind)
    if ok is False:
        return None
    if ok is None:
        try:
            got = _mega(out, residual, precomputed, nxt)
        except Exception as e:  # noqa: BLE001
            _MHC_OK[kind] = False
            levers.announce(f"mo-mhc-{kind}", f"lever MO: mega-mHC {kind} post failed ({type(e).__name__}: "
                            f"{str(e)[:120]}): stock for good")
            return None
        ref = stock(hc, out, residual, stats_stream=stats_stream, next=nxt, world_size=world_size,
                    precomputed=precomputed)
        why = _same_state(ref, got)
        _MHC_OK[kind] = not why
        levers.announce(f"mo-mhc-{kind}", f"lever MO: mega-mHC {kind} post first at {out.shape[0]} rows: "
                        f"{'ours == stock bit for bit: on' if not why else 'OFF: ' + why}")
        return ref
    try:
        got = _mega(out, residual, precomputed, nxt)
    except Exception as e:  # noqa: BLE001 - stock for this post kind from now on
        _MHC_OK[kind] = False
        levers.announce(f"mo-mhc-err-{kind}", f"lever MO: mega-mHC {kind} post failed ({type(e).__name__}: "
                        f"{str(e)[:120]}): stock for good")
        return None
    _STATE["mhc_fast"] += 1
    return got


def _mega(y, residual, coefficients, nxt):
    """_post_fusion's mega branch + mhc_mega_boundary + deep_gemm.mega_mhc for this call."""
    m = _H["mhc"]
    pre, post_mix, comb = coefficients
    nh = nxt.hc
    ncfg = nh.cfg
    rows = y.shape[0]
    hc_mult = residual.shape[1]
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(y)
    next_pre = torch.empty((rows, hc_mult, 1), device=y.device, dtype=torch.float32)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty_like(comb)
    _H["mega"](y, residual, pre.unsqueeze(-1), post_mix.unsqueeze(-1), comb, nh.fn, nh.scale, nh.base, hc_mult,
               ncfg.rms_eps, ncfg.eps, 2.0, ncfg.eps, ncfg.sinkhorn_iters, nh.norm.weight, nh.norm.variance_epsilon,
               1.0, updated, next_pre, next_post, next_comb, normalized, None, None, None, None, 0, None)
    return m.HcState(updated, pre, m.HcNormed(normalized), (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb))


def _same_state(a, b) -> str:
    """'' when two HcStates hold the same tensors' bytes in the same structure."""
    m = _H["mhc"]
    if type(a) is not m.HcState or type(b) is not m.HcState:
        return f"state types {type(a).__name__} / {type(b).__name__}"
    pairs = [(a.streams, b.streams), (a.pre, b.pre)]
    if type(a.input) is not type(b.input):
        return "input kinds differ"
    if isinstance(a.input, m.HcNormed):
        pairs.append((a.input.rows, b.input.rows))
    if (a.stats is None) != (b.stats is None):
        return "stats presence differs"
    if a.stats is not None:
        pairs.extend(zip(a.stats, b.stats))
    for x, y in pairs:
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            if x is not y:
                return "non-tensor fields differ"
            continue
        if x.shape != y.shape or x.dtype != y.dtype or x.stride() != y.stride():
            return f"tensor geometry differs ({tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype})"
        if not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)):
            return f"bytes differ ({tuple(x.shape)} {x.dtype})"
    return ""
