"""Lever NQ on the fork (levers.NQ): the routed NVFP4 MoE's input quantization of CUDA-graph decode / verify calls runs on
a side stream at the start of the MoE forward, beside the router GEMM / top-k / prune, instead of on the main stream
between the prune and the routed MoE (B300 k=1 trace: quantize 2.5 us a layer on the critical path).

Exactness: the side stream runs stock's own quantize_hidden_states_fp4 on the same bf16 input tensor with the same
w13_input_scale_quant -- the call the routed runner (fused_experts_none_to_flashinfer_trtllm_fp4) would make -- and the
experts replica's dispatch hands the result to the runner as an already-quantized FP4 input (the runner's
hidden_states_scale branch, which passes it through unchanged), after the main stream waits for the side stream's
event. Same bytes into the same routed MoE call.

Gate (issue): levers.NQ, the MoE replica armed (an NVFP4 layer on the flashinfer_trtllm(_routed) runner, no per-token
activation scale, no hidden-dim padding, standard dispatcher), a non-empty bf16 input. Anything else returns None and
the stock path runs.
"""
from typing import NamedTuple, Optional

import torch

from dsv41f import levers

_SIDE: dict = {}       # device index -> the quantize stream


class NvFp4PreQuant(NamedTuple):
    x_q: torch.Tensor                  # [T, hidden / 2] uint8 (stock quantize_hidden_states_fp4)
    x_sf: torch.Tensor                 # its linear scale bytes
    ready: Optional[torch.cuda.Event]  # recorded on the side stream


def eligible(moe) -> str:
    """Why lever NQ cannot serve this MoE replica ('' = it can)."""
    experts = moe._modules.get("experts")
    method = getattr(experts, "quant_method", None)
    if experts is None or type(method).__name__ != "ModelOptNvFp4FusedMoEMethod":
        return "not an NVFP4 ModelOpt MoE"
    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    backend = getattr(method, "_moe_runner_backend", None) or get_moe_runner_backend()   # (stock apply's own lookup)
    if not (backend.is_flashinfer_trtllm() or backend.is_flashinfer_trtllm_routed()):
        return "not the flashinfer trtllm runner"
    if getattr(method.quant_config, "use_per_token_activation", False):
        return "per-token activation scales"
    if getattr(experts, "trtllm_padded_hidden_size", None) is not None:
        return "hidden-dim padding"
    if getattr(experts, "w13_input_scale_quant", None) is None:
        return "no w13_input_scale_quant"
    if type(getattr(experts, "dispatcher", None)).__name__ != "StandardDispatcher":
        return "not the standard dispatcher"
    return ""


def issue(moe, hidden_states) -> Optional[NvFp4PreQuant]:
    """The MoE forward's start (lifted forward_normal_dual_stream): issue the routed input's quantize on the side stream."""
    if not getattr(moe, "dsv4f_nq", False) or hidden_states.shape[0] == 0 or hidden_states.dtype != torch.bfloat16:
        return None
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import quantize_hidden_states_fp4

    device = hidden_states.device
    side = _SIDE.get(device.index)
    if side is None:
        side = _SIDE[device.index] = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        x_q, x_sf = quantize_hidden_states_fp4(hidden_states, moe._modules["experts"].w13_input_scale_quant)
        ready = torch.cuda.Event()
        ready.record(side)
    if levers.first("nq-on"):
        levers.announce("nq-on", f"lever NQ: routed NVFP4 input quantize on a side stream, first at "
                        f"{hidden_states.shape[0]} rows, layer {moe.layer_id}")
    return NvFp4PreQuant(x_q, x_sf, ready)


def bind_experts(moe) -> bool:
    """Give the MoE replica an experts replica whose dispatch accepts an NvFp4PreQuant (stock's otherwise)."""
    import copy
    import types

    why = eligible(moe)
    if not levers.NQ or why:
        return False
    experts = moe._modules["experts"]
    replica = copy.copy(experts)
    replica._modules = dict(experts._modules)
    stock = replica._dispatch_with_pre_quant

    def dispatch_with_pre_quant(self, hidden_states, topk_output, pre_quant_input):
        if isinstance(pre_quant_input, NvFp4PreQuant):
            out = self.dispatcher.dispatch(hidden_states=hidden_states, topk_output=topk_output)
            torch.cuda.current_stream(hidden_states.device).wait_event(pre_quant_input.ready)
            if out.format.is_standard() and out.hidden_states_scale is None:
                return out._replace(hidden_states=pre_quant_input.x_q, hidden_states_scale=pre_quant_input.x_sf)
            return out
        return stock(hidden_states, topk_output, pre_quant_input)

    replica._dispatch_with_pre_quant = types.MethodType(dispatch_with_pre_quant, replica)
    moe._modules["experts"] = replica
    moe.dsv4f_nq = True
    return True
