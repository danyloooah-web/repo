"""Lever RE5 on the fork (levers.ROUTE_E5): the decode / target-verify MoE top-k (stock vision_topk -> moe_fused_gate's
Triton router) and the fork's E5 routed-expert prune (deepseek_v4_prune.prune_ids, one Triton CTA) as ONE kernel per
MoE layer (bundle CUDA unit route_e5) -- the arena line's levers RP / RF, re-targeted at the fork's stock prune.

Stock chain on a pruned decode / verify MoE layer: router (gate) -> moe_fused_gate launch -> prune_ids launch -> the
routed MoE. Here: router -> ONE launch that computes stock's top-k rows bit for bit (route_row: the RP / RF code) into
one CTA's shared memory and runs E5 there with thread = row before a single store. The result is the
StandardTopKOutput(weights, ids, logits) vision_topk + prune_topk produce, bit for bit.

E5's two fp32 sums (a row's `mass`, a pick's `run`) follow the reduction tree Triton's layout gives the stock kernel's
tile (BR = max(8, next_pow2(rows)) rows, 4 / 8 warps); levers.E5_TREES names the tree per BR (fitted from the stock
kernel's sums by fork/tests/fit_e5.py). The kernel takes the trees as runtime arguments.

Where: the MoE replicas' forward_normal_dual_stream (CUDA-graph decode / verify) and forward_normal (eager: the audit)
are the fork's own methods lifted with two edits (moe_fwd.py): vision_topk(...) -> route(...) and prune_topk(...) ->
after_topk(...). route() returns the fused, pruned output only when stock prune_topk would prune this call (its own
module state and constants: a decode / verify batch under decode_rows, not a hash / MTP layer, 0 < rows <=
PRUNE_MAX_ROWS) and the inputs fit the kernel's contract; otherwise it returns stock vision_topk's output and
after_topk runs stock prune_topk on it.

Fail-closed: ready() calibrates the top-k's routed-sum tree against the installed stock router, then compares the fused
kernel with stock moe_fused_gate + prune_ids bit for bit on random batches at every launch bucket, and E5 alone (the
kernel's prune-only mode vs stock prune_ids) on rows whose lightest eligible pick sits on the budget boundary (where a
wrong summation order flips the decision); the outcome is agreed over the TP CPU group (gloo MIN).

Lever RE5NC (levers.ROUTE_E5_FUSED_MAX_ROWS, ROUTE_E5_NC_MODE): the fused kernel's 2..16-CTA cluster needs a whole GPC
and waits behind the shared-expert / mHC side kernels that spread over every GPC (B300). Above ROUTE_E5_FUSED_MAX_ROWS
rows no cluster is launched: mode 1 = route() returns stock vision_topk's output (stock's Triton router, plain grid)
and after_topk runs the prune-only mode (+ lever RR's record) on it (_rr_small); mode 2 = route() launches the top-k
alone on a plain multi-CTA grid (route_e5.run_grid: route_row verbatim, ROUTE_E5_NC_SHAPE) and the prune-only mode
(+ record) as its PDL dependent on those rows in place; mode 3 = ONE plain-grid launch (route_e5.run_nc): every CTA
routes its rows and stores them, the CTA drawing the grid's last ticket (a per-device counter, reset by that CTA)
loads them back through L2 and runs E5 (+ record) as the fused kernel's leader does. All halves are the code the fused
kernel and the self-test already use; ready() additionally compares the configured RE5NC chain with stock top-k + prune_ids at every bucket
above ROUTE_E5_FUSED_MAX_ROWS. The choice depends on the row count and constants only (identical on every rank and
baked into each captured decode graph by its static row count).
"""
import itertools

import numpy as np
import torch

from dsv41f import levers, rr

TOP_K = 6
EXPERTS = 384
MAX_ROWS = 256
EPS = 1e-20                      # srt/layers/moe/topk.py _RENORMALIZE_SUM_EPSILON
BUCKETS = (8, 16, 32, 64, 128, 256)
SELF_TEST_ROWS = (1, 5, 6, 12, 16, 24, 31, 48, 49, 64, 72, 96, 108, 120, 127, 128, 144, 192, 256)   # + lever P1's buckets

_NATIVE: list = []
_READY: list = []                # [(ok, sum candidate index, max_cluster)], rank-agreed
_TOPK_CLS: list = []
_LAST: list = [None]


# ------------------------------------------------------------------------------------------------ reduction trees
def tree_of(s, t, w, within="seq", lanes="desc", order="k"):
    """Triton's reduce of 8 slots under a blocked layout (s elements a thread, t lanes, w warps along the axis): a
    thread sums its elements in slot order (or pairwise), then shuffle-xor butterflies over lanes, then warps. Nested
    pairs of slot indices (fork/tests/fit_e5.py's model)."""
    L = 8
    s_eff = min(s, L)
    t_eff = max(1, min(t, L // s_eff))
    w_eff = max(1, min(w, L // (s_eff * t_eff)))
    cover = s_eff * t_eff * w_eff
    reps = L // cover
    groups = {}
    for wi in range(w_eff):
        for li in range(t_eff):
            idx = []
            for rep in range(reps):
                base = rep * cover + (wi * t_eff + li) * s_eff
                idx.extend(range(base, base + s_eff))
            if order == "e":
                idx = sorted(idx, key=lambda x: ((x % s_eff), x))
            if within == "seq":
                node = idx[0]
                for x in idx[1:]:
                    node = (node, x)
            else:
                nodes = list(idx)
                while len(nodes) > 1:
                    nodes = [(nodes[i], nodes[i + 1]) for i in range(0, len(nodes), 2)]
                node = nodes[0]
            groups[(wi, li)] = node

    def butterfly(vals):
        n = len(vals)
        cur = list(vals)
        offs = []
        off = n // 2
        while off >= 1:
            offs.append(off)
            off //= 2
        if lanes == "asc":
            offs = offs[::-1]
        for off in offs:
            cur = [(cur[i], cur[i ^ off]) if (i & off) == 0 else (cur[i ^ off], cur[i]) for i in range(n)]
        return cur[0]

    return butterfly([butterfly([groups[(wi, li)] for li in range(t_eff)]) for wi in range(w_eff)])


def steps_of(node):
    """The tree as the kernel's 7 add steps: step k writes value 8 + k from two earlier values; the root is value 14."""
    steps = []

    def walk(n):
        if isinstance(n, int):
            return n
        a, b = walk(n[0]), walk(n[1])
        steps.append((a, b))
        return 7 + len(steps)

    root = walk(node)
    if len(steps) != 7 or root != 14:
        raise ValueError(f"not an 8-leaf binary tree: {node}")
    return [v for st in steps for v in st]


def eval_tree(node, v):
    """fp32 evaluation of the tree over v[..., 8] (numpy float32)."""
    if isinstance(node, int):
        return v[..., node]
    return (eval_tree(node[0], v) + eval_tree(node[1], v)).astype(np.float32)


def _tree_layouts():
    """The distinct trees of the layout family (s, t, w, within, lanes, order), in a fixed order."""
    seen, out = set(), []
    for lay in itertools.product((1, 2, 4, 8), (1, 2, 4, 8, 16, 32), (1, 2, 4, 8), ("seq", "tree"), ("desc", "asc"),
                                 ("k", "e")):
        key = repr(tree_of(*lay))
        if key not in seen:
            seen.add(key)
            out.append(lay)
    return out


SUM_CANDIDATES = [(m, None) for m in range(5)] + [(5, lay) for lay in _tree_layouts()]   # (sum_mode, ktree layout)


def trees_for(rows: int):
    """(BR, mass tree, run tree) of the stock launch for `rows` rows; trees None when levers.E5_TREES has no fitted
    entry for that tile (the call then stays on the stock path)."""
    br = max(8, 1 << max(0, (rows - 1).bit_length()))
    entry = levers.E5_TREES.get(br)
    if entry is None:
        return br, None, None
    return br, tree_of(*entry[0]), tree_of(*entry[1])


# ------------------------------------------------------------------------------------------------ native unit
def native():
    if not _NATIVE:
        mod = None
        try:
            import route_e5 as mod
        except ImportError as err:
            levers.announce("re5-native-missing", f"bundle CUDA unit route_e5 not importable ({err}): lever RE5 off")
        if mod is not None and (mod.MAX_ROWS, mod.TOP_K, mod.EXPERTS) != (MAX_ROWS, TOP_K, EXPERTS):
            levers.announce("re5-native-geometry", "route_e5 geometry mismatch: lever RE5 off")
            mod = None
        _NATIVE.append(mod)
    return _NATIVE[0]


def _standard_topk_cls():
    if not _TOPK_CLS:
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        _TOPK_CLS.append(StandardTopKOutput)
    return _TOPK_CLS[0]


def _prune_mod():
    from sglang.srt.models import deepseek_v4_prune

    return deepseek_v4_prune


def shape(rows: int, max_cluster: int):
    """(threads, cluster): the first ROUTE_E5_SHAPES entry (max_rows, threads, rows_per_cta) with rows <= max_rows."""
    threads, per = 256, 0
    for max_rows, t, p in levers.ROUTE_E5_SHAPES:
        if rows <= int(max_rows):
            threads, per = int(t), int(p)
            break
    threads = max(threads, next(b for b in (32, 64, 128, 256) if b >= rows))
    if per <= 0 or max_cluster < 2 or rows < 2:
        return threads, 1
    return threads, max(2, min(max_cluster, rows, (rows + per - 1) // per))


def fused_max() -> int:
    """Lever RE5NC: the most rows route() serves with the fused (cluster) kernel; above it the RE5NC chain."""
    return int(getattr(levers, "ROUTE_E5_FUSED_MAX_ROWS", MAX_ROWS))


def nc_mode() -> int:
    """Lever RE5NC above fused_max(): 1 = stock top-k + prune-only (after_topk), 2 = run_grid top-k + prune-only (route();
    needs the unit's run_grid, else mode 1), 3 = run_nc: top-k + E5 in one plain-grid launch, E5 in its last CTA (needs
    run_nc, else mode 2)."""
    mode = int(getattr(levers, "ROUTE_E5_NC_MODE", 1))
    if mode == 3 and (native() is None or getattr(native(), "run_nc", None) is None):
        mode = 2
    if mode == 2 and (native() is None or getattr(native(), "run_grid", None) is None):
        mode = 1
    return mode


_TICKETS: dict = {}       # (device, layer) -> mode 3's arrival counter (zeroed int32; allocated outside any capture)


def _ticket(device, layer=-1):
    """Mode 3's counter of one MoE layer (-1: the self-test's): one a layer, so two launches can share a counter only
    when they are the same layer's -- and a layer's route calls are stream-ordered."""
    t = _TICKETS.get((device, layer))
    if t is None and not torch.cuda.is_current_stream_capturing():
        t = _TICKETS[(device, layer)] = torch.zeros(1, dtype=torch.int32, device=device)
    return t


def nc_pdl(rows: int) -> bool:
    """Lever RE5NC: PDL on the mode 2 / 3 launches of `rows` rows (ROUTE_E5_NC_PDL: True / False, or an int = PDL up to
    that many rows; default ROUTE_E5_PDL). Off: the launch waits for the router's split-K reduce to complete and gets
    every SM it frees at once -- an early (PDL) launch at 144 rows left CTAs pending behind the shared-expert / mHC side
    kernels (the 22-26 us mode); at 48 rows the early launch is the faster one."""
    v = getattr(levers, "ROUTE_E5_NC_PDL", levers.ROUTE_E5_PDL)
    if isinstance(v, bool):
        return v
    return rows <= int(v)


def nc_shape():
    """(threads, rows a CTA) of mode 2's top-k grid."""
    threads, per = getattr(levers, "ROUTE_E5_NC_SHAPE", (256, 8))
    return int(threads), max(1, int(per))


def prune_params(rows: int):
    """(budget, max_count) of the E5 prune this call takes: stock's, or lever SV's budget / share vote when SV is on."""
    p = _prune_mod()
    if getattr(levers, "SV", False):
        count = int(levers.SV_MAX_COUNT) if rows >= int(levers.SV_MIN_ROWS) else int(p.PRUNE_MAX_COUNT)
        return float(levers.SV_BUDGET), count
    return float(p.PRUNE_BUDGET), int(p.PRUNE_MAX_COUNT)


def _ref_prune(ids, weights, budget: float, max_count: int):
    """The stock E5 kernel on (ids, weights) in place, launched exactly as prune_ids / sv.prune launch it."""
    import triton

    p = _prune_mod()
    rows, k = ids.shape
    scratch = torch.empty(2, p._NB, dtype=torch.int32, device=ids.device)
    p._prune_kernel[(1,)](ids, weights, ids, rows, ids.stride(0), weights.stride(0), 0, scratch[0], scratch[1],
                          float(budget), int(max_count), K=k, KP=p._KP, BR=max(8, triton.next_power_of_2(rows)),
                          NB=p._NB, PACKED=False, num_warps=4 if rows <= 32 else 8)


# route_e5.cu's compile-time trees (fixed_step): steps -> id; any other tree runs from the runtime steps (id -1)
_FIXED_TREES = {(0, 4, 2, 6, 8, 9, 1, 5, 3, 7, 11, 12, 10, 13): 0, (0, 2, 1, 3, 8, 9, 4, 6, 5, 7, 11, 12, 10, 13): 1,
                (0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 11, 12, 10, 13): 2}


def tree_ids(*steps):
    return [_FIXED_TREES.get(tuple(int(x) for x in s), -1) for s in steps]


def _launch(logits, bias, alt, input_ids, ntnp, weights, ids, image_token, scale, apply_scale, cand, prune, rows,
            cluster, threads, pdl, ids_in=None, w_in=None, record=()):
    """cand: index into SUM_CANDIDATES (the top-k's routed-sum tree)."""
    p = _prune_mod()
    budget, max_count = prune_params(rows)
    sum_mode, klay = SUM_CANDIDATES[cand]
    ktree = steps_of(tree_of(*klay)) if klay is not None else steps_of(tree_of(8, 1, 1))
    _, mtree, rtree = trees_for(rows)
    if mtree is None:                    # top-k only (calibration) on a tile without fitted trees
        mtree = rtree = tree_of(8, 1, 1)
    stream = torch.cuda.current_stream(weights.device).cuda_stream
    native().run(
        logits.data_ptr() if logits is not None else 0, logits.stride(0) if logits is not None else EXPERTS, rows,
        bias.data_ptr(), alt.data_ptr() if alt is not None else 0,
        input_ids.data_ptr() if input_ids is not None else 0, input_ids is not None and input_ids.dtype == torch.int64,
        ntnp.data_ptr() if ntnp is not None else 0, ntnp is not None and ntnp.dtype == torch.int64,
        weights.data_ptr(), ids.data_ptr(), image_token, scale, EPS, apply_scale, sum_mode, prune,
        budget, max_count, steps_of(mtree), steps_of(rtree), cluster, threads, pdl,
        int(levers.ROUTE_E5_TRIGGER), 0, stream,
        ids_in.data_ptr() if ids_in is not None else 0, w_in.data_ptr() if w_in is not None else 0, ktree,
        list(record), tree_ids(steps_of(mtree), steps_of(rtree), ktree))


def _launch_grid(logits, bias, alt, input_ids, ntnp, weights, ids, image_token, scale, apply_scale, cand, rows, pdl):
    """Lever RE5NC mode 2: the top-k alone (no E5) on run_grid's plain grid into weights / ids."""
    sum_mode, klay = SUM_CANDIDATES[cand]
    ktree = steps_of(tree_of(*klay)) if klay is not None else steps_of(tree_of(8, 1, 1))
    threads, per = nc_shape()
    stream = torch.cuda.current_stream(weights.device).cuda_stream
    native().run_grid(
        logits.data_ptr(), logits.stride(0), rows, bias.data_ptr(), alt.data_ptr() if alt is not None else 0,
        input_ids.data_ptr() if input_ids is not None else 0, input_ids is not None and input_ids.dtype == torch.int64,
        ntnp.data_ptr() if ntnp is not None else 0, ntnp is not None and ntnp.dtype == torch.int64,
        weights.data_ptr(), ids.data_ptr(), image_token, scale, EPS, apply_scale, sum_mode, ktree, tree_ids(ktree)[0],
        per, threads, pdl, 1, stream)


def _launch_nc(logits, bias, alt, input_ids, ntnp, weights, ids, image_token, scale, apply_scale, cand, rows, pdl,
               record=()):
    """Lever RE5NC mode 2: run_grid's top-k into weights / ids, then the prune-only mode (+ record) on them in place
    as its PDL dependent (one CTA: every row is loaded into shared memory before the first store)."""
    _launch_grid(logits, bias, alt, input_ids, ntnp, weights, ids, image_token, scale, apply_scale, cand, rows, pdl)
    threads, _ = shape(rows, 1)
    _launch(None, bias, None, None, None, weights, ids, 0, 1.0, False, cand, True, rows, 1, threads, pdl,
            ids_in=ids, w_in=weights, record=record)


def _launch_nc3(logits, bias, alt, input_ids, ntnp, weights, ids, image_token, scale, apply_scale, cand, rows, pdl,
                record=(), layer=-1):
    """Lever RE5NC mode 3: run_nc -- the fused kernel's top-k + E5 (+ record) on a plain grid (ROUTE_E5_NC_SHAPE's rows a
    CTA), E5 in the grid's last CTA."""
    budget, max_count = prune_params(rows)
    sum_mode, klay = SUM_CANDIDATES[cand]
    ktree = steps_of(tree_of(*klay)) if klay is not None else steps_of(tree_of(8, 1, 1))
    _, mtree, rtree = trees_for(rows)
    threads, _ = shape(rows, 1)
    _, per = nc_shape()
    stream = torch.cuda.current_stream(weights.device).cuda_stream
    native().run_nc(
        logits.data_ptr(), logits.stride(0), rows, bias.data_ptr(), alt.data_ptr() if alt is not None else 0,
        input_ids.data_ptr() if input_ids is not None else 0, input_ids is not None and input_ids.dtype == torch.int64,
        ntnp.data_ptr() if ntnp is not None else 0, ntnp is not None and ntnp.dtype == torch.int64,
        weights.data_ptr(), ids.data_ptr(), image_token, scale, EPS, apply_scale, sum_mode, budget, max_count,
        steps_of(mtree), steps_of(rtree), threads, pdl, stream, ktree, list(record),
        tree_ids(steps_of(mtree), steps_of(rtree), ktree), per, _ticket(weights.device, layer).data_ptr())


# ------------------------------------------------------------------------------------------------ the hooks
def _ok_int(t, rows, device):
    return t is None or (torch.is_tensor(t) and t.is_cuda and t.device == device and t.dim() == 1
                         and t.shape[0] == rows and t.is_contiguous() and t.dtype in (torch.int32, torch.int64))


def _ok_ntnp(t, device):
    return t is None or (torch.is_tensor(t) and t.is_cuda and t.device == device and t.numel() >= 1
                         and t.dtype in (torch.int32, torch.int64))


def _ok_logits(t, rows):
    return (torch.is_tensor(t) and t.dtype == torch.float32 and t.dim() == 2 and tuple(t.shape) == (rows, EXPERTS)
            and t.stride(1) == 1 and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0)


def _stock_would_prune(moe, rows) -> bool:
    p = _prune_mod()
    return (bool(p._on) and not moe.is_hash and not moe.is_nextn and p.PRUNE_BUDGET > 0.0 and p.PRUNE_MAX_COUNT > 0
            and int(getattr(levers, "ROUTE_E5_MIN_ROWS", 1)) <= rows <= min(int(p.PRUNE_MAX_ROWS), MAX_ROWS)
            and trees_for(rows)[1] is not None)


def route(moe, router_logits, input_ids, num_token_non_padded):
    """The MoE forwards' vision_topk call: the fused, pruned top-k when stock prune_topk would prune this call and
    everything fits the kernel; otherwise stock vision_topk's (unpruned) output."""
    from sglang.srt.models.deepseek_v2 import vision_topk

    rows = router_logits.shape[0]
    if not (levers.ROUTE_E5 and getattr(moe, "dsv4f_re5", False) and _READY and _READY[0][0]
            and _stock_would_prune(moe, rows)):
        return vision_topk(moe, router_logits, input_ids, num_token_non_padded=num_token_non_padded)
    fused = rows <= fused_max()
    mode = 0 if fused else nc_mode()
    if mode == 3 and _ticket(router_logits.device, moe.layer_id) is None:
        mode = 2           # (no counter allocated before this capture: mode 2, same bits)
    if mode == 1:      # lever RE5NC mode 1: stock's top-k (no cluster); after_topk runs the prune-only mode on it
        if levers.first("re5nc-on"):
            levers.announce("re5nc-on", f"lever RE5NC: above {fused_max()} rows stock top-k + RE5 prune-only (no "
                            f"cluster), first at {rows} rows, layer {moe.layer_id}")
        return vision_topk(moe, router_logits, input_ids, num_token_non_padded=num_token_non_padded)
    device = router_logits.device
    if not (_ok_logits(router_logits, rows) and _ok_int(input_ids, rows, device)
            and _ok_ntnp(num_token_non_padded, device)):
        if levers.first(f"re5-contract-{rows}"):
            levers.announce(f"re5-contract-{rows}", f"lever RE5: a {rows}-row call outside the kernel's contract: stock")
        return vision_topk(moe, router_logits, input_ids, num_token_non_padded=num_token_non_padded)
    _, cand, max_cluster = _READY[0]
    image_token, scale, apply_scale = moe.dsv4f_re5_args
    gate = moe.gate
    weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=device)
    ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
    tac = rr.tactic_for(rows) if getattr(levers, "RR", False) else None     # lever RR: the routing record too
    slot, record = rr.record_args(rows, tac) if tac is not None else (None, ())
    if mode in (2, 3):  # lever RE5NC: no cluster -- mode 2 top-k grid -> prune-only (+ record) PDL dependent;
        # mode 3 top-k + E5 (+ record) in one grid, E5 in the last CTA
        args = (router_logits, gate.e_score_correction_bias, gate.e_score_correction_bias_vl,
                input_ids if input_ids is not None else None, num_token_non_padded, weights, ids, image_token, scale,
                apply_scale, cand, rows, nc_pdl(rows))
        if mode == 3:
            _launch_nc3(*args, record=record, layer=moe.layer_id)
        else:
            _launch_nc(*args, record=record)
        if levers.first(f"re5nc-on-{mode}"):
            threads, per = nc_shape()
            levers.announce(f"re5nc-on-{mode}", f"lever RE5NC mode {mode}: above {fused_max()} rows " + (
                f"top-k grid ({threads} threads x {per} rows a CTA, no cluster) + RE5 prune-only" if mode == 2 else
                f"top-k + E5 on a plain grid ({per} rows a CTA, E5 in the last CTA, no cluster)") +
                f", first at {rows} rows, layer {moe.layer_id}")
    else:
        threads, cluster = shape(rows, max_cluster)
        _launch(router_logits, gate.e_score_correction_bias, gate.e_score_correction_bias_vl,
                input_ids if input_ids is not None else None, num_token_non_padded, weights, ids, image_token, scale,
                apply_scale, cand, True, rows, cluster, threads, bool(levers.ROUTE_E5_PDL), record=record)
        if levers.first("re5-on"):
            levers.announce("re5-on", f"lever RE5: fused top-k + E5, first at {rows} rows, layer {moe.layer_id}, "
                            f"{threads} threads x {cluster} CTA(s)")
    if tac is not None:
        rr.arm(ids, weights, rows, tac, slot)
    out = _standard_topk_cls()(weights, ids, router_logits)
    _LAST[0] = out
    return out


def _rr_small(moe, topk_output):
    """Lever RR below ROUTE_E5_MIN_ROWS and lever RE5NC mode 1 above ROUTE_E5_FUSED_MAX_ROWS (wherever route() returned
    stock's top-k): RE5's prune-only mode on stock's top-k rows (sv.prune's E5, bit for bit) also writing the routing
    record; None when sv.prune would not prune this call or the record is not armed (stock prune then)."""
    p = _prune_mod()
    if not (getattr(levers, "RR", False) and levers.ROUTE_E5 and getattr(moe, "dsv4f_re5", False) and _READY
            and _READY[0][0]) or topk_output is None:
        return None
    if not p._on or moe.is_hash or moe.is_nextn or p.PRUNE_BUDGET <= 0.0 or p.PRUNE_MAX_COUNT <= 0:
        return None
    ids_in = getattr(topk_output, "topk_ids", None)
    w_in = getattr(topk_output, "topk_weights", None)
    if ids_in is None or w_in is None or getattr(topk_output, "packed_topk_ids", None) is not None:
        return None
    rows = ids_in.shape[0]
    if not (0 < rows <= min(int(p.PRUNE_MAX_ROWS), MAX_ROWS) and trees_for(rows)[1] is not None):
        return None
    if not (ids_in.dtype == torch.int32 and w_in.dtype == torch.float32 and tuple(ids_in.shape) == (rows, TOP_K)
            and tuple(w_in.shape) == (rows, TOP_K) and ids_in.is_contiguous() and w_in.is_contiguous()):
        return None
    tac = rr.tactic_for(rows)
    if tac is None:
        return None
    slot, record = rr.record_args(rows, tac)
    weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=ids_in.device)
    ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=ids_in.device)
    _, cand, _ = _READY[0]
    gate = moe.gate
    threads, _ = shape(rows, 1)
    _launch(None, gate.e_score_correction_bias, None, None, None, weights, ids, 0, 1.0, False, cand, True, rows, 1,
            threads, bool(levers.ROUTE_E5_PDL), ids_in=ids_in, w_in=w_in, record=record)
    rr.arm(ids, weights, rows, tac, slot)
    if levers.first("rr-small-on"):
        levers.announce("rr-small-on", f"lever RR: prune-only RE5 + record on stock's top-k, first at {rows} rows, "
                        f"layer {moe.layer_id}")
    return _standard_topk_cls()(weights, ids, topk_output.router_logits)


def after_topk(moe, topk_output):
    """The MoE forwards' prune_topk call: stock prune_topk, except on the output route() pruned already."""
    if topk_output is not None and topk_output is _LAST[0]:
        _LAST[0] = None
        return topk_output
    _LAST[0] = None
    small = _rr_small(moe, topk_output)
    if small is not None:
        return small
    if getattr(levers, "SV", False):
        from dsv41f import sv

        return sv.prune(moe, topk_output)
    return _prune_mod().prune_topk(moe, topk_output)


# ------------------------------------------------------------------------------------------------ binding
def layer_reason(moe):
    if getattr(moe, "is_hash", False) or getattr(moe, "is_nextn", False):
        return "hash-routed / MTP layer"
    gate = moe._modules.get("gate")
    if gate is None:
        return "no gate"
    for name, t in (("e_score_correction_bias", getattr(gate, "e_score_correction_bias", None)),
                    ("e_score_correction_bias_vl", getattr(gate, "e_score_correction_bias_vl", None))):
        if t is None or not torch.is_tensor(t) or t.dtype != torch.float32 or t.numel() != EXPERTS \
                or not t.is_contiguous() or t.data_ptr() % 16 != 0:
            return f"{name} not an fp32 [384] tensor (the vision_topk branch)"
    cfg = getattr(getattr(moe, "topk", None), "topk_config", None)
    if cfg is None or int(cfg.top_k) != TOP_K or int(cfg.num_fused_shared_experts) != 0 or not cfg.renormalize:
        return "top-k config (top-6, renormalized, no fused shared experts)"
    if not isinstance(getattr(moe.config, "image_token_id", None), int):
        return "no integer image_token_id"
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend

    if get_moe_runner_backend().is_flashinfer_mxfp4() or get_moe_a2a_backend().is_megamoe():
        return "packed top-k / MegaMoE"
    return None


def _args(moe):
    cfg = moe.topk.topk_config
    scale = float(cfg.routed_scaling_factor) if cfg.routed_scaling_factor is not None else 1.0
    return int(moe.config.image_token_id), scale, bool(cfg.apply_routed_scaling_factor_on_output)


def bind(moe) -> bool:
    """Mark the MoE replica for lever RE5 when it can serve it and the rank-agreed self-test passed."""
    if not levers.ROUTE_E5:
        return False
    why = layer_reason(moe)
    if why is not None:
        return False
    if not ready(moe):
        return False
    moe.dsv4f_re5_args = _args(moe)
    if nc_mode() == 3:              # lever RE5NC mode 3: this layer's counter, outside any capture
        _ticket(moe.gate.e_score_correction_bias.device, moe.layer_id)
    moe.dsv4f_re5 = True
    return True


# ------------------------------------------------------------------------------------------------ readiness
def _agree(values):
    """MIN over the TP CPU group (gloo), elementwise."""
    from sglang.srt.distributed import get_tp_group

    t = torch.tensor(values, dtype=torch.int64)
    group = get_tp_group()
    if group.world_size > 1:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group)
    return t.tolist()


def ready(moe=None) -> bool:
    if not _READY:
        if torch.cuda.is_current_stream_capturing() or moe is None:
            return False
        ok, mode, max_cluster, detail = False, -1, 1, "CUDA unit unavailable"
        if native() is not None:
            try:
                ok, mode, max_cluster, detail = self_test(moe)
            except Exception as err:   # noqa: BLE001 - any failure keeps the stock path
                ok, detail = False, f"{type(err).__name__}: {err}"
        a = _agree([int(ok), mode, -mode, max_cluster])
        agreed = bool(a[0]) and a[1] == -a[2] and a[1] >= 0
        levers.announce("re5-ready" if agreed else "re5-selftest-fail",
                        f"lever RE5 self-test on this rank: {detail}; rank-agreed: {'ON' if agreed else 'OFF'}")
        _READY.append((agreed, a[1], max(1, a[3])))
    return _READY[0][0]


def _max_cluster():
    mod = native()
    for size in (16, 8, 4, 2):
        try:
            if mod.cluster_fits(size, 256) >= 1:
                return size
        except Exception:  # noqa: BLE001
            pass
    return 1


def _stock_topk(logits, bias, alt, input_ids, ntnp, image_token, scale, apply_scale):
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    return moe_fused_gate(
        logits, bias, topk=TOP_K, scoring_func="sqrtsoftplus", num_fused_shared_experts=0, bias_alt=alt,
        input_ids=input_ids, bias_alt_token_id=image_token, renormalize=True, renormalize_epsilon=EPS,
        routed_scaling_factor=scale, apply_routed_scaling_factor_on_output=apply_scale,
        num_token_non_padded=ntnp, packed_out=None, sqrtsoftplus_log1p=True)


def _same(a, b) -> bool:
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def _probe_batch(gen, rows, trial, image_token, device):
    x = torch.randn(rows, EXPERTS, generator=gen, device=device) * (1.0 + trial % 3)
    if trial % 2:
        x = torch.round(x * 4.0) / 4.0
    if rows > 2:
        x[rows - 1] = x[0]
    ids = torch.randint(image_token - 1, image_token + 2, (rows,), generator=gen, device=device)
    ntnp = torch.tensor([max(1, rows - (trial % 2) * (rows // 3))], dtype=torch.int32, device=device)
    return x.contiguous(), ids, ntnp


def _boundary_rows(rng, rows, budget, mtree, rtree):
    """[rows, 6] ids / fp32 weights where every row's lightest pick is a unique expert whose run sits at budget x mass
    (exactly, or one ulp to either side) under the given trees: a different summation order flips such decisions."""
    ids = np.zeros((rows, TOP_K), np.int32)
    w = np.zeros((rows, TOP_K), np.float32)
    perm = rng.permutation(EXPERTS)
    unique_rows = min(rows, 60)              # 60 rows x 6 picks of distinct experts; the rest share 24 experts
    for r in range(rows):
        if r < unique_rows:
            ids[r] = perm[6 * r + np.arange(6)]
        else:
            ids[r] = perm[360 + rng.choice(24, 6, replace=False)]
        base = (rng.random(6) * rng.random() * 0.5 + 0.05).astype(np.float32)
        light = int(rng.integers(0, 6))
        v8 = np.zeros(8, np.float32)
        v8[:6] = base
        for _ in range(4):
            mass = eval_tree(mtree, v8)
            target = np.float32(np.float32(budget) * mass)
            v8[light] = target
        nudge = int(rng.integers(-1, 2))
        v8[light] = np.nextafter(v8[light], np.float32(np.inf if nudge > 0 else -np.inf)) if nudge else v8[light]
        for k in range(6):          # the boundary pick must be the lightest: the others above it
            if k != light and v8[k] <= v8[light]:
                v8[k] = np.float32(v8[light] * np.float32(1.5) + np.float32(0.01))
        w[r] = v8[:6]
    return ids, w


def self_test(moe):
    """(ok, sum_mode, max_cluster, detail)."""
    mod = native()
    p = _prune_mod()
    gate = moe.gate
    bias, alt = gate.e_score_correction_bias, gate.e_score_correction_bias_vl
    device = bias.device
    image_token, scale, apply_scale = _args(moe)
    max_cluster = _max_cluster()
    gen = torch.Generator(device=device)
    gen.manual_seed(20261014)
    # (1) routed-sum tree of the top-k: route_prune's five fixed trees, then the layout family (mode 5)
    alive = list(range(len(SUM_CANDIDATES)))
    for rows in (1, 6, 24, 48, 144, 256):
        for trial in range(3):
            x, ids, ntnp = _probe_batch(gen, rows, trial, image_token, device)
            sw, si = _stock_topk(x, bias, alt, ids, ntnp, image_token, scale, apply_scale)
            for cand in list(alive):
                w = torch.empty(rows, TOP_K, dtype=torch.float32, device=device)
                i = torch.empty(rows, TOP_K, dtype=torch.int32, device=device)
                _launch(x, bias, alt, ids, ntnp, w, i, image_token, scale, apply_scale, cand, False, rows, 1, 256,
                        False)
                if not (_same(w, sw) and _same(i, si)):
                    alive.remove(cand)
    if not alive:
        return False, -1, max_cluster, "no routed-sum tree reproduces the installed stock router"
    mode = alive[0]
    # (2) fused top-k + E5 vs stock moe_fused_gate + prune_ids, every bucket and launch shape
    cases = mism = drops = 0
    tested = [r for r in SELF_TEST_ROWS if trees_for(r)[1] is not None]
    if not tested:
        return False, mode, max_cluster, "no E5 tree fitted for any tile"
    for rows in tested:
        for trial in range(3):
            x, ids, ntnp = _probe_batch(gen, rows, trial, image_token, device)
            sw, si = _stock_topk(x, bias, alt, ids, ntnp, image_token, scale, apply_scale)
            n0 = int((si < 0).sum())
            _ref_prune(si, sw, *prune_params(rows))
            drops += int((si < 0).sum()) - n0
            for threads, cluster in sorted({(256, 1), shape(rows, max_cluster)}):
                w = torch.empty(rows, TOP_K, dtype=torch.float32, device=device)
                i = torch.empty(rows, TOP_K, dtype=torch.int32, device=device)
                _launch(x, bias, alt, ids, ntnp, w, i, image_token, scale, apply_scale, mode, True, rows, cluster,
                        threads, False)
                cases += 1
                mism += 0 if (_same(w, sw) and _same(i, si)) else 1
    # (3) E5 alone on budget-boundary rows (prune-only mode vs stock prune_ids)
    rng = np.random.default_rng(20261014)
    bcases = bmism = bdrops = 0
    for rows in tested:
        _, mtree, rtree = trees_for(rows)
        for trial in range(4):
            budget, max_count = prune_params(rows)
            bi, bw = _boundary_rows(rng, rows, budget, mtree, rtree)
            si = torch.from_numpy(bi).to(device)
            sw = torch.from_numpy(bw).to(device)
            ri, rw = si.clone(), sw.clone()
            _ref_prune(ri, rw, budget, max_count)
            w = torch.empty(rows, TOP_K, dtype=torch.float32, device=device)
            i = torch.empty(rows, TOP_K, dtype=torch.int32, device=device)
            threads = max(32, next(b for b in (32, 64, 128, 256) if b >= rows))
            _launch(None, bias, None, None, None, w, i, image_token, scale, apply_scale, mode, True, rows, 1,
                    threads, False, ids_in=si, w_in=sw)
            bcases += 1
            bdrops += int((ri < 0).sum())
            bmism += 0 if (_same(w, rw) and _same(i, ri)) else 1
    # (4) lever RE5NC: the chain served above fused_max() rows vs stock top-k + E5 (mode 2: its top-k grid alone too)
    ncm = nc_mode()
    if ncm == 3 and _ticket(device) is None:
        ncm = 2
    ncases = nmism = 0
    for rows in [r for r in tested if r > fused_max()]:
        for trial in range(3):
            x, ids, ntnp = _probe_batch(gen, rows, trial, image_token, device)
            sw, si = _stock_topk(x, bias, alt, ids, ntnp, image_token, scale, apply_scale)
            rw, ri = sw.clone(), si.clone()
            _ref_prune(ri, rw, *prune_params(rows))
            w = torch.empty(rows, TOP_K, dtype=torch.float32, device=device)
            i = torch.empty(rows, TOP_K, dtype=torch.int32, device=device)
            threads, _ = shape(rows, 1)
            if ncm == 3:      # (consecutive launches of different grids: the counter must reset every time)
                _launch_nc3(x, bias, alt, ids, ntnp, w, i, image_token, scale, apply_scale, mode, rows, nc_pdl(rows))
            elif ncm == 2:
                _launch_grid(x, bias, alt, ids, ntnp, w, i, image_token, scale, apply_scale, mode, rows, False)
                ncases += 1
                nmism += 0 if (_same(w, sw) and _same(i, si)) else 1
                w = torch.empty(rows, TOP_K, dtype=torch.float32, device=device)
                i = torch.empty(rows, TOP_K, dtype=torch.int32, device=device)
                _launch_nc(x, bias, alt, ids, ntnp, w, i, image_token, scale, apply_scale, mode, rows, nc_pdl(rows))
            else:
                _launch(None, bias, None, None, None, w, i, 0, 1.0, False, mode, True, rows, 1, threads,
                        bool(levers.ROUTE_E5_PDL), ids_in=si, w_in=sw)      # as _rr_small launches it
            ncases += 1
            nmism += 0 if (_same(w, rw) and _same(i, ri)) else 1
    torch.cuda.synchronize(device)
    detail = (f"routed-sum candidate {mode} = {SUM_CANDIDATES[mode]} (matching {alive[:6]}); {cases} fused batches vs stock top-k + E5: {mism} "
              f"mismatches ({drops} drops); {bcases} budget-boundary batches: {bmism} mismatches ({bdrops} drops); "
              f"RE5NC mode {ncm} above {fused_max()} rows: {ncases} batches vs stock: {nmism} mismatches; "
              f"clusters up to {max_cluster}; tiles {sorted({trees_for(r)[0] for r in tested})}")
    return mism == 0 and bmism == 0 and nmism == 0, mode, max_cluster, detail
