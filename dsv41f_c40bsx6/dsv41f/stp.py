"""Lever STP on the fork (levers.STP): the shared expert's tail on a top-priority stream in CUDA-graph decode / verify
MoE calls of STP_MIN_ROWS..STP_MAX_ROWS rows (the lifted forward_normal_dual_stream, moe_fwd.py).

Why (c19x k24 profile, 144 rows): stock issues the whole shared expert (MXFP8 quantize, gate_up, silu, quantize, down)
on the MoE's alt stream at the default (lowest) priority, forked before the router. The chain (~28 us) outlasts the
route window (~20 us); its down GEMM becomes ready just as FC1 (7182 CTAs, main stream at lever PM's top priority)
launches, so it waits until every FC1 CTA has been dispatched, runs ~77 us beside FC2, and the MoE finalize waits
2.3-5 us a layer for it. Stream priority only orders PENDING CTAs: a later grid at a higher priority is dispatched
ahead of FC1's remaining CTAs as SMs free up; at equal or lower priority it is not.

Modes (levers.STP_MODE):
  "tail"  gate_up (with its input quantize) stays on the alt stream where stock forks it; silu + the down projection
          (its MXFP8 quantize + GEMM) run on the tail stream, which waits for the alt stream's gate_up;
  "late"  as "tail", and the tail also waits for the main stream right after the router (RE5): silu / quantize / down
          never sit in the route window, they start beside FC1;
  "all"   the whole shared expert (stock's own _forward_shared_experts call) on the tail stream, forked right after
          the router, nothing of it in the route window.
The main stream joins the tail stream where stock joins the alt stream (before the finalize; the alt join stays).
The tail stream is ONE persistent stream per device, created at prepare (never inside a capture) at the highest
priority; lever PM's main stream sits one level below it (levers.STP_PM_BELOW) so the tail's pending CTAs win over
FC1's.

Exactness: scheduling only. "all" is stock's call on another stream. "tail" / "late" run DeepseekV2MLP.forward's own
CUDA path in two pieces on the same modules, inputs and functions (gate_up_proj; silu_and_mul_clamp -- the name stock
resolves -- or act_fn; down_proj), gated by: the stock source of DeepseekV2MLP.forward and
DeepseekV2MoE._forward_shared_experts hashing to the values read from the fork (5e72ce5), the replica's bound methods
being those, and the instance taking none of forward's other branches (NVFP4 swiglu fusion, the fp8 fused-down path,
the zero-allocator gate_up output, HIP clamp routes, quant-once pre-quant, hooks). The first eager call of every row
count (the decode-graph warmups) runs the split and then stock's own _forward_shared_experts on the same input and
compares every output byte; a difference turns STP off for the process (stock from then on). Inside a capture an
unchecked row count runs stock. Every cross-stream tensor is ordered by the joins (the next layer's alt / tail work
starts after the main stream's wait on this layer's tail), as stock's alt-stream shared output is.

LEVER GQ (gen_gq.py, gq.py): the split's gate_up takes gq.gate_up_input(gate_up, hidden_states) -- the
small-grid unit's Mxfp8SwizzledInput of the rows (stock's quantize bytes, checked per key) at
levers.GQ_MIN_ROWS..GQ_MAX_ROWS rows, the rows themselves otherwise.
"""
import torch

from dsv41f import gq   # lever GQ: the split's gate_up input quantize on the small-grid unit
from dsv41f import levers

_TAIL: dict = {}          # device index -> the top-priority tail stream (prepare)
_CHECKED: dict = {}       # rows -> True (split == stock bits) / False (STP off)
_STATE = {"detail": "not installed", "off": False}
# sha256[:16] of the exact source text (fl.source_hash) of the stock functions the split reproduces, fork 5e72ce5
PINNED = {
    "DeepseekV2MLP.forward": "f30d9a8fc00cf8fc",
    "DeepseekV2MoE._forward_shared_experts": "4d0e42df7f8a5041",
}
MODES = ("tail", "late", "all")


def pm_priority(low: int, high: int) -> int:
    """Lever PM's main-stream priority: one level below the tail stream's (the top) when STP is on."""
    if levers.STP and levers.STP_PM_BELOW and _TAIL and high < low:
        return high + 1
    return high


def _why_not(moe, dv2) -> str:
    """Why the split cannot serve this MoE replica ('' = it can)."""
    if getattr(dv2, "_is_hip", True) or getattr(dv2, "_is_npu", True) or not getattr(dv2, "_is_cuda", False):
        return "not the CUDA path"
    if getattr(moe, "num_fused_shared_experts", 1) != 0 or getattr(moe, "_shared_expert_tp1", True):
        return "fused or replicated (TP1) shared expert"
    if getattr(moe, "alt_stream", None) is None:
        return "no alt stream"
    if getattr(moe._forward_shared_experts, "__func__", None) is not dv2.DeepseekV2MoE._forward_shared_experts:
        return "_forward_shared_experts is not stock's"
    shared = moe._modules.get("shared_experts")
    if shared is None or type(shared) is not dv2.DeepseekV2MLP:
        return "shared expert is not a DeepseekV2MLP"
    if getattr(shared.forward, "__func__", None) is not dv2.DeepseekV2MLP.forward:
        return "shared expert forward is not stock's"
    if getattr(shared, "_enable_nvfp4_gemm_swiglu_fusion", False):
        return "NVFP4 gemm + swiglu fusion"
    if getattr(shared, "use_fused_clamp_act_mul", True):
        return "fused clamp act (HIP)"
    gu = shared._modules.get("gate_up_proj")
    dn = shared._modules.get("down_proj")
    if gu is None or dn is None:
        return "no gate_up / down"
    gw = getattr(gu, "weight", None)
    dw = getattr(dn, "weight", None)
    if gw is None or dw is None or gw.dtype == torch.uint8:
        return "gate_up weight absent or uint8 (zero-allocator path)"
    if (shared.swiglu_limit is not None and shared.allow_fused_down and dw.dtype == torch.uint8
            and hasattr(dn, "weight_scale_inv")):
        return "fp8 fused-down path"
    return ""


def prepare(layers) -> str:
    """Create the tail stream (top priority, this process's device) and arm the lifted MoE replicas (model_node.prepare)."""
    if not levers.STP:
        _STATE["detail"] = "off (levers.STP)"
        return _STATE["detail"]
    try:
        import sglang.srt.models.deepseek_v2 as dv2
        from dsv41f import fl

        if levers.STP_MODE not in MODES:
            raise RuntimeError(f"STP_MODE {levers.STP_MODE!r} not in {MODES}")
        bad = []
        for qual, want in PINNED.items():
            got = fl.source_hash(dv2, qual)
            if got != want:
                bad.append(f"{qual} {got}")
        fl._PARSED.pop(dv2.__name__, None)
        if bad:
            raise RuntimeError("stock source differs from the fork read: " + ", ".join(bad))
        armed, why, dev = 0, {}, None
        for r in layers:
            moe = r._modules.get("mlp")
            if moe is None or not getattr(moe, "dsv4f_dt_copy", False) or not hasattr(moe, "forward_normal_dual_stream"):
                continue
            reason = _why_not(moe, dv2)
            if reason:
                why[reason] = why.get(reason, 0) + 1
                continue
            moe.dsv4f_stp = True
            armed += 1
            if dev is None:
                dev = moe._modules["shared_experts"]._modules["gate_up_proj"].weight.device
        if not armed:
            raise RuntimeError(f"no MoE replica eligible ({why})")
        low, high = torch.cuda.Stream.priority_range()
        tail = _TAIL[dev.index] = torch.cuda.Stream(device=dev, priority=high)
        _STATE["detail"] = (f"on: mode {levers.STP_MODE}, {armed} MoE replicas, rows {levers.STP_MIN_ROWS}.."
                            f"{levers.STP_MAX_ROWS}, tail stream priority {tail.priority} (range {low}..{high}), "
                            f"PM main {pm_priority(low, high) if levers.PM else 'off'}"
                            + (f"; not armed {why}" if why else ""))
    except Exception as e:  # noqa: BLE001 - any missing / changed internal keeps stock
        _STATE["off"] = True
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:300]})"
    return _STATE["detail"]


def _mode(moe, hidden_states, pre_quant_input):
    """This call's STP mode, or None for stock's alt-stream shared expert."""
    if _STATE["off"] or not getattr(moe, "dsv4f_stp", False) or pre_quant_input is not None:
        return None
    rows = int(hidden_states.shape[0])
    if rows < levers.STP_MIN_ROWS or rows > levers.STP_MAX_ROWS or hidden_states.dtype != torch.bfloat16:
        return None
    ok = _CHECKED.get(rows)
    if ok is False or (ok is None and torch.cuda.is_current_stream_capturing()):
        return None
    shared = moe._modules["shared_experts"]
    if shared._forward_hooks or shared._forward_pre_hooks:
        return None
    if _TAIL.get(hidden_states.device.index) is None:
        return None
    return levers.STP_MODE


def mark(moe, hidden_states, pre_quant_input, current_stream):
    """Right after the router (lifted forward, after lever SD): this call's mode; "late" / "all" fork the tail here."""
    mode = _mode(moe, hidden_states, pre_quant_input)
    if mode in ("late", "all"):
        _TAIL[hidden_states.device.index].wait_stream(current_stream)
    return mode


def _tail_part(shared, gate_up):
    """DeepseekV2MLP.forward from its gate_up output on (the CUDA path _why_not admits): silu (+ clamp) and down_proj."""
    if shared.swiglu_limit is not None:
        from sglang.srt.models.deepseek_v2 import silu_and_mul_clamp

        M, N = gate_up.shape
        x = gate_up.new_empty((M, N // 2))
        silu_and_mul_clamp(gate_up, x, float(shared.swiglu_limit))
    else:
        x = shared.act_fn(gate_up)
    x, _ = shared.down_proj(x)
    return x


def _same(a, b) -> bool:
    if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.element_size() == 2:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a, b)


def shared(moe, hidden_states, gemm_output_zero_allocator, pre_quant_input, current_stream, mode):
    """The lifted forward's shared-expert block: stock's alt-stream call (mode None) or STP's tail-stream schedule,
    joined into the main stream at stock's join point (the caller keeps stock's alt join after this)."""
    if mode is None:
        with torch.cuda.stream(moe.alt_stream):
            return moe._forward_shared_experts(hidden_states, gemm_output_zero_allocator, pre_quant_input=pre_quant_input)
    tail = _TAIL[hidden_states.device.index]
    if mode == "all":
        with torch.cuda.stream(tail):
            out = moe._forward_shared_experts(hidden_states, gemm_output_zero_allocator, pre_quant_input=pre_quant_input)
    else:
        sh = moe._modules["shared_experts"]
        with torch.cuda.stream(moe.alt_stream):
            # lever GQ: the unit's MXFP8 quantize of the input (stock's bytes) where eligible, else the linear's own
            gate_up, _ = sh.gate_up_proj(gq.gate_up_input(sh._modules["gate_up_proj"], hidden_states))
        tail.wait_stream(moe.alt_stream)
        with torch.cuda.stream(tail):
            out = _tail_part(sh, gate_up)
    rows = int(hidden_states.shape[0])
    if _CHECKED.get(rows) is None:
        # first eager call at this row count (a capture never gets here: _mode ran stock): stock's own call on the
        # same input, on the tail stream after the split, every byte compared
        with torch.cuda.stream(tail):
            ref = moe._forward_shared_experts(hidden_states, gemm_output_zero_allocator, pre_quant_input=pre_quant_input)
            ok = _same(ref, out)
        _CHECKED[rows] = ok
        if not ok:
            _STATE["off"] = True
            out = ref
        levers.announce(f"stp-{rows}", f"lever STP mode {mode} first at {rows} rows (layer {moe.layer_id}): shared "
                        f"output {'== stock bit for bit: on' if ok else '!= stock: OFF for the process'}")
    current_stream.wait_stream(tail)
    return out
