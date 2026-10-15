"""Lever SV on the fork (levers.SV_*): the fork's own E5 routed-expert prune (deepseek_v4_prune._prune_kernel, launched
exactly as stock prune_ids launches it) with a wider share vote and budget -- a PRECISION change, like the arena line's
v9p (audit-like PASS 0.9586 there).

Stock E5 (PRUNE_MAX_COUNT = 1, PRUNE_BUDGET = 0.4): only experts picked by ONE row of the call may be dropped. E5's
kernel already implements a vote for larger counts: an expert picked by <= max_count rows is eligible, and it is dropped
only when EVERY row that picked it chose it as a candidate (its pick is within that row's budget, lightest first). SV
runs that same kernel with max_count = SV_MAX_COUNT on decode / verify calls of >= SV_MIN_ROWS rows (where several rows
sharing a light expert are common and each dropped expert saves a whole weight stream) and budget SV_BUDGET at every
pruned size. Everything else -- which calls are pruned (stock's decode_rows / hash / MTP / row-limit rules), the
launch shape, the in-place update of ids / weights (and the packed form) -- is stock prune_ids'.
"""
import torch
import triton

from dsv41f import levers


def _stock():
    from sglang.srt.models import deepseek_v4_prune

    return deepseek_v4_prune


def prune(moe, topk_output):
    """The MoE forwards' prune_topk call: stock prune_topk, with SV's budget / share vote where it applies."""
    p = _stock()
    if not levers.SV:
        return p.prune_topk(moe, topk_output)
    if not p._on or moe.is_hash or moe.is_nextn or p.PRUNE_BUDGET <= 0.0 or p.PRUNE_MAX_COUNT <= 0:
        return topk_output
    ids = topk_output.topk_ids
    rows = ids.shape[0]
    if rows == 0 or rows > p.PRUNE_MAX_ROWS:
        return topk_output
    weights = topk_output.topk_weights
    packed = getattr(topk_output, "packed_topk_ids", None)
    k = ids.shape[1]
    if not (ids.dtype == torch.int32 and weights.dtype == torch.float32 and weights.shape == ids.shape
            and ids.stride(1) == 1 and weights.stride(1) == 1 and k <= p._KP and 0 < rows <= p._BR):
        return p.prune_topk(moe, topk_output)
    if packed is not None and not (packed.dtype == torch.int32 and packed.shape == ids.shape and packed.stride(1) == 1):
        return p.prune_topk(moe, topk_output)
    max_count = int(levers.SV_MAX_COUNT) if rows >= int(levers.SV_MIN_ROWS) else int(p.PRUNE_MAX_COUNT)
    budget = float(levers.SV_BUDGET)
    scratch = p._scratch.get(ids.device)
    if scratch is None:
        scratch = p._scratch[ids.device] = torch.empty(2, p._NB, dtype=torch.int32, device=ids.device)
    p._prune_kernel[(1,)](ids, weights, packed if packed is not None else ids, rows, ids.stride(0), weights.stride(0),
                          packed.stride(0) if packed is not None else 0, scratch[0], scratch[1], budget, max_count,
                          K=k, KP=p._KP, BR=max(8, triton.next_power_of_2(rows)), NB=p._NB,
                          PACKED=packed is not None, num_warps=4 if rows <= 32 else 8)
    if levers.first("sv-on"):
        levers.announce("sv-on", f"lever SV: E5 budget {budget}, share vote up to {levers.SV_MAX_COUNT} rows from "
                        f"{levers.SV_MIN_ROWS} rows (first at {rows} rows, layer {moe.layer_id})")
    return topk_output
