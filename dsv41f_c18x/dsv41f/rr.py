"""Lever RR on the fork (levers.RR): lever RE5's kernel also writes FlashInfer's trtllm-gen routing record for the call's
final (pruned) ids, and the NVFP4 routed MoE of decode / verify calls runs FlashInfer 0.7.1's prepared body on it (the
native trtllm_fp4_block_scale_moe's da_routing_metadata / da_body_workspace entry) -- the routingIndices{DynBlock,
Cluster}Kernel launch between the prune and GEMM1 is gone; GEMM1 / GEMM2 are the same kernels at the same tactic.

Chain of a pruned decode / verify MoE layer: router -> top-k (stock, or RE5's fused kernel from ROUTE_E5_MIN_ROWS) ->
prune (stock Triton E5 with lever SV's parameters, or RE5) -> routed MoE = FlashInfer routing kernel -> GEMM1 -> GEMM2.
With RR: router -> top-k -> RE5 (prune-only below ROUTE_E5_MIN_ROWS, fused above) writing ids, weights AND the record
-> GEMM1 -> GEMM2.

Mechanics:
  learn     the routed call's tactic per row count is the one the stock call takes (AutoTuner.choose_one on its op, the
            fork's decode pin and lever FMT's decode bands included): an outermost choose_one hook records it during
            stock calls (the first warmup forward of a CUDA-graph batch size), and the record buffers (FlashInfer's own
            trtllm_moe_allocate_routing_metadata_multi_tile for those rows and the tactic's tile) and the body
            workspace (the native op's preparation call) are allocated right there, outside any capture.
  arm       route_e5.route / after_topk pass the record to the RE5 launch when a tactic is known for the rows and the
            shape is not failed, and leave a Plan for the routed call (identity of the ids tensor).
  run       the wrapper around flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe: the planned call on a verified
            (rows, tactic) shape runs the prepared body on the record; anything else takes the stock call.
  verify    once per (rows, tactic), on a warmup forward's real inputs (never inside a capture): the record vs
            FlashInfer's own record of the same ids (populate_trtllm_moe_routing_metadata_: CTA arrays and counts
            equal; maps equal up to the order of an expert's rows), and the body's GEMM2 rows per (token, slot) vs the
            stock call's, bit for bit. A mismatch or an exception turns that shape off (stock from then on).
Exactness: the same GEMM kernels, tactic and per-(token, slot) rows; only the permuted row order inside an expert may
differ, and every GEMM row is computed from its own token alone.
"""
import contextvars
import dataclasses

import torch

from dsv41f import levers

OP = "flashinfer::trtllm_fp4_block_scale_moe"
_PLAN = contextvars.ContextVar("dsv41f_rr_plan", default=None)
_LEARN = contextvars.ContextVar("dsv41f_rr_learn", default=None)
_STATE = {"installed": False, "failed": False, "detail": "not installed"}
_TACTIC: dict = {}        # rows -> [tile, config] the stock routed call takes
_SLOT: dict = {}          # (rows, tile) -> FlashInfer TrtllmMoERoutingMetadataSlot (graph-stable record buffers)
_WORK: dict = {}          # (rows, tile, config) -> prepared body workspace (7 tensors)
_OK: dict = {}            # (rows, tile, config) -> True (verified) / False (off)
_STOCK: list = []         # [stock trtllm_fp4_block_scale_routed_moe]


@dataclasses.dataclass
class Plan:
    ids: torch.Tensor
    weights: torch.Tensor
    rows: int
    tactic: tuple
    slot: object


def _core():
    from flashinfer.fused_moe import core

    return core


def install() -> str:
    """Wrap the routed NVFP4 entry and AutoTuner.choose_one once (model_node.prepare, outside any capture)."""
    if _STATE["installed"]:
        return _STATE["detail"]
    if not getattr(levers, "RR", False):
        _STATE["detail"] = "off (levers.RR)"
        return _STATE["detail"]
    try:
        import flashinfer.fused_moe as fm
        from flashinfer.autotuner import AutoTuner
        from flashinfer.fused_moe import core

        stock = fm.trtllm_fp4_block_scale_routed_moe
        for name in ("trtllm_moe_allocate_routing_metadata_multi_tile", "populate_trtllm_moe_routing_metadata_",
                     "get_trtllm_moe_sm100_module", "_unpack_trtllm_moe_output", "_device_support_moe_pdl",
                     "_torch_view_of_ffi_tensor", "device_support_pdl", "RoutingInputMode"):
            if not hasattr(core, name):
                raise AttributeError(f"flashinfer.fused_moe.core.{name}")
    except Exception as e:  # noqa: BLE001 - a FlashInfer without the prepared-body API keeps stock routing
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:160]})"
        return _STATE["detail"]
    _STOCK.append(stock)
    wrapped_choose = AutoTuner.choose_one

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        out = wrapped_choose(self, custom_op, runners, tuning_config, inputs, **kwargs)
        box = _LEARN.get()
        if box is not None and custom_op == OP:
            box.append(out[1])
        return out

    AutoTuner.choose_one = choose_one

    def routed(*args, **kwargs):
        return _routed(*args, **kwargs)

    fm.trtllm_fp4_block_scale_routed_moe = routed
    _STATE["installed"] = True
    _STATE["detail"] = (f"armed: rows {levers.RR_MIN_ROWS}..{levers.RR_MAX_ROWS}, record by RE5, prepared body "
                        f"(FlashInfer da_routing_metadata)")
    return _STATE["detail"]


def tactic_for(rows: int):
    """The learned (tile, config) of the routed call at this row count, when lever RR may arm it."""
    if not _STATE["installed"] or _STATE["failed"] or not levers.RR_MIN_ROWS <= rows <= levers.RR_MAX_ROWS:
        return None
    tac = _TACTIC.get(rows)
    if tac is None or _OK.get((rows,) + tac) is False:
        return None
    if (rows, tac[0]) not in _SLOT or (rows,) + tac not in _WORK:
        return None
    return tac


def record_args(rows: int, tac):
    """[map, tok, batch, limit, padded, nonexit, tile, offset, num_local] for route_e5.run."""
    slot = _SLOT[(rows, tac[0])]
    off, num_local = _LOCAL[rows]
    return slot, [slot.expanded_idx_to_permuted_idx.data_ptr(), slot.permuted_idx_to_token_idx.data_ptr(),
                  slot.cta_idx_xy_to_batch_idx.data_ptr(), slot.cta_idx_xy_to_mn_limit.data_ptr(),
                  slot.total_num_padded_tokens.data_ptr(), slot.num_non_exiting_ctas.data_ptr(), int(tac[0]),
                  int(off), int(num_local)]


_LOCAL: dict = {}         # rows -> (local_expert_offset, local_num_experts) of the learned call


def arm(ids, weights, rows, tac, slot) -> None:
    _PLAN.set(Plan(ids, weights, rows, tuple(tac), slot))


def _native_args(kw, ids, weights, slot, work, prepare: bool, tactic):
    core = _core()
    hs = kw["hidden_states"]
    enable_pdl = kw.get("enable_pdl")
    if enable_pdl is None:
        enable_pdl = core.device_support_pdl(hs.device)
    if not core._device_support_moe_pdl(hs.device):
        enable_pdl = False
    weights_slot = dataclasses.replace(slot, expert_weights=weights)
    return (int(core.RoutingInputMode.UnpackedPrecomputed), None, ids, weights, kw.get("routing_bias"), hs,
            kw.get("hidden_states_scale"), kw["gemm1_weights"], kw["gemm1_weights_scale"], kw.get("gemm1_bias"), None,
            kw.get("gemm1_alpha"), kw.get("gemm1_beta"), kw.get("gemm1_clamp_limit"), kw["gemm2_weights"],
            kw["gemm2_weights_scale"], kw.get("gemm2_bias"), kw.get("output1_scale_scalar"),
            kw.get("output1_scale_gate_scalar"), kw.get("output2_scale_scalar"), kw.get("per_token_scale"),
            int(kw["num_experts"]), int(kw["top_k"]), 0, kw.get("n_group"), kw.get("topk_group"),
            int(kw["intermediate_size"]), int(kw["local_expert_offset"]), int(kw["local_num_experts"]),
            kw.get("routed_scaling_factor"), int(kw.get("routing_method_type", 1)), bool(kw.get("do_finalize", True)),
            bool(enable_pdl), int(kw.get("activation_type")), kw.get("output"), [int(tactic[0]), int(tactic[1])],
            True, None, list(weights_slot.tensors()), list(work) if work is not None else [], bool(prepare), None,
            None)


def _body(kw, plan):
    core = _core()
    op = core.get_trtllm_moe_sm100_module().moe_op
    work = _WORK[(plan.rows,) + plan.tactic]
    out = op.trtllm_fp4_block_scale_moe(*_native_args(kw, plan.ids, plan.weights, plan.slot, work, False,
                                                      plan.tactic))
    return core._unpack_trtllm_moe_output(out, kw.get("output"), bool(kw.get("do_finalize", True)), None,
                                          plan.weights)


def _supported(kw) -> bool:
    routing = kw.get("topk_ids")
    return (isinstance(routing, tuple) and len(routing) == 2 and kw.get("gemm1_lora_delta") is None
            and kw.get("valid_hidden_size") is None and kw.get("valid_intermediate_size") is None
            and not kw.get("num_fused_shared_experts") and torch.is_tensor(kw.get("output")))


def _prepare_shape(kw, rows, tac) -> None:
    """Record buffers for (rows, tile) and the body workspace for (rows, tactic): outside any capture."""
    core = _core()
    ids, weights = kw["topk_ids"]
    key_s = (rows, int(tac[0]))
    if key_s not in _SLOT:
        meta = core.trtllm_moe_allocate_routing_metadata_multi_tile(
            ids, num_experts=int(kw["num_experts"]), top_k=int(kw["top_k"]),
            local_expert_offset=int(kw["local_expert_offset"]), num_local_experts=int(kw["local_num_experts"]),
            tile_ns=(int(tac[0]),), routing_input_mode=core.RoutingInputMode.UnpackedPrecomputed,
            topk_weights=weights)
        _SLOT[key_s] = meta.slots[0]
    _LOCAL[rows] = (int(kw["local_expert_offset"]), int(kw["local_num_experts"]))
    key_w = (rows,) + tuple(tac)
    if key_w not in _WORK:
        op = core.get_trtllm_moe_sm100_module().moe_op
        work = op.trtllm_fp4_block_scale_moe(*_native_args(kw, ids, weights, _SLOT[key_s], None, True, tac))
        _WORK[key_w] = tuple(core._torch_view_of_ffi_tensor(t) for t in work)


def _rows_of(out, gemm2_idx=0):
    return out


def _compare(kw, plan, ref, got) -> str:
    """'' when the body's result equals the stock call's for every (token, slot); else why not."""
    do_finalize = bool(kw.get("do_finalize", True))
    if do_finalize:
        a, b = (ref[0] if isinstance(ref, (list, tuple)) else ref), (got[0] if isinstance(got, (list, tuple)) else got)
        return "" if torch.equal(a.view(torch.int16), b.view(torch.int16)) else "finalized output differs"
    g_ref, w_ref, m_ref = ref[0], ref[1], ref[2]
    g_got, w_got, m_got = got[0], got[1], got[2]
    m_ref = m_ref.reshape(-1).long()
    m_got = m_got.reshape(-1).long()
    if not torch.equal(m_ref < 0, m_got < 0):
        return "dropped / remote slots differ"
    live = m_ref >= 0
    rows_ref = g_ref.reshape(-1, g_ref.shape[-1])[m_ref[live]]
    rows_got = g_got.reshape(-1, g_got.shape[-1])[m_got[live]]
    if not torch.equal(rows_ref.view(torch.int16), rows_got.view(torch.int16)):
        return "GEMM2 rows differ"
    if not torch.equal(w_ref.float(), w_got.float()):
        return "weights differ"
    return ""


def _compare_record(kw, plan) -> str:
    """The kernel's record vs FlashInfer's own record of the same ids ('' = equal up to an expert's row order)."""
    core = _core()
    ids, weights = plan.ids, plan.weights
    meta = core.trtllm_moe_allocate_routing_metadata_multi_tile(
        ids, num_experts=int(kw["num_experts"]), top_k=int(kw["top_k"]),
        local_expert_offset=int(kw["local_expert_offset"]), num_local_experts=int(kw["local_num_experts"]),
        tile_ns=(int(plan.tactic[0]),), routing_input_mode=core.RoutingInputMode.UnpackedPrecomputed,
        topk_weights=weights)
    core.populate_trtllm_moe_routing_metadata_(meta, ids, weights)
    torch.cuda.synchronize(ids.device)
    fi, ours = meta.slots[0], plan.slot
    n = int(fi.num_non_exiting_ctas.item())
    if int(ours.num_non_exiting_ctas.item()) != n or int(ours.total_num_padded_tokens.item()) != \
            int(fi.total_num_padded_tokens.item()):
        return "CTA count / padded size"
    if not torch.equal(ours.cta_idx_xy_to_batch_idx[:n], fi.cta_idx_xy_to_batch_idx[:n]) or \
            not torch.equal(ours.cta_idx_xy_to_mn_limit[:n], fi.cta_idx_xy_to_mn_limit[:n]):
        return "CTA arrays"
    k = int(kw["top_k"])
    m_fi = fi.expanded_idx_to_permuted_idx[: plan.rows * k].long()
    m_ours = ours.expanded_idx_to_permuted_idx[: plan.rows * k].long()
    if not torch.equal(m_fi < 0, m_ours < 0):
        return "map holes"
    # the same permuted range per expert: permuted index -> CTA -> expert must agree slot for slot
    tile = int(plan.tactic[0])
    live = m_fi >= 0
    e_fi = fi.cta_idx_xy_to_batch_idx[(m_fi[live] // tile)]
    e_ours = ours.cta_idx_xy_to_batch_idx[(m_ours[live] // tile)]
    if not torch.equal(e_fi, e_ours):
        return "map experts"
    tok_ours = ours.permuted_idx_to_token_idx[m_ours[live]].long()
    want = torch.arange(plan.rows, device=ids.device).repeat_interleave(k)[live]
    if not torch.equal(tok_ours, want):
        return "tok map"
    return ""


def _routed(*args, **kw):
    stock = _STOCK[0]
    plan = _PLAN.get()
    routing = kw.get("topk_ids")
    if args or not _supported(kw):
        return stock(*args, **kw)
    rows = int(routing[0].shape[0])
    if plan is None or routing[0] is not plan.ids or routing[1] is not plan.weights:
        if torch.cuda.is_current_stream_capturing() or _STATE["failed"] or \
                not levers.RR_MIN_ROWS <= rows <= levers.RR_MAX_ROWS:
            return stock(**kw)
        box = []
        token = _LEARN.set(box)
        try:
            out = stock(**kw)
        finally:
            _LEARN.reset(token)
        if box and isinstance(box[-1], (list, tuple)) and len(box[-1]) == 2 and int(box[-1][0]) > 0:
            tac = (int(box[-1][0]), int(box[-1][1]))
            if _TACTIC.get(rows) not in (None, tac):
                _OK[(rows,) + _TACTIC[rows]] = False
            _TACTIC[rows] = tac
            try:
                _prepare_shape(kw, rows, tac)
            except Exception as e:  # noqa: BLE001 - a preparation failure keeps this shape on stock
                _OK[(rows,) + tac] = False
                levers.announce(f"rr-prep-{rows}", f"lever RR: {rows}-row preparation failed "
                                f"({type(e).__name__}: {str(e)[:160]}): stock")
        return out
    _PLAN.set(None)
    key = (plan.rows,) + plan.tactic
    ok = _OK.get(key)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return stock(**kw)
        box = []
        token = _LEARN.set(box)
        try:
            ref = stock(**kw)
        finally:
            _LEARN.reset(token)
        why = ""
        if not box or (int(box[-1][0]), int(box[-1][1])) != plan.tactic:
            why = f"tactic changed ({box[-1] if box else None} vs {plan.tactic})"
        if not why:
            try:
                why = _compare_record(kw, plan)
                if not why:
                    got = _body(kw, plan)
                    torch.cuda.synchronize(plan.ids.device)
                    why = _compare(kw, plan, ref, got)
            except Exception as e:  # noqa: BLE001 - any failure keeps this shape on stock
                why = f"{type(e).__name__}: {str(e)[:160]}"
        _OK[key] = not why
        levers.announce(f"rr-{plan.rows}-{plan.tactic}", f"lever RR {plan.rows} rows tactic {list(plan.tactic)}: "
                        f"{'verified (record + GEMM2 rows == stock): on' if not why else 'OFF: ' + why}")
        return ref
    if not ok:
        return stock(**kw)
    if levers.first("rr-on"):
        levers.announce("rr-on", f"lever RR: prepared body on the RE5 record, first at {plan.rows} rows")
    return _body(kw, plan)
