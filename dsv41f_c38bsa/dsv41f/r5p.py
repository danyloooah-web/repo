"""Lever R5P on the fork (levers.R5P, CUDA unit r5p_unit): lever RE5's fused top-k + E5 (+ lever RR's routing record) of
CUDA-graph decode / verify MoE calls at R5P_MIN_ROWS..R5P_MAX_ROWS rows on r5p_unit's cluster kernel -- the same bytes
as route_e5.cu's route_e5_kernel (ids, weights, record), with E5 spread over the routing warps.

RE5 at 49..127 rows (lever P1 shape, 256 threads x 16 rows a CTA, a 4..8-CTA cluster) spends, in its leader's clock
(tests/r5p/r5p_phase.py, c26rtr, 96 rows, isolated): routing 3.2 us (two rows a warp), join 0.5, then the leader alone:
E5 1.5 (thread = row), record 2.1 (row masks, popc counts, scan, CTA arrays, padding, map) -- code that runs once per
launch from a cold instruction cache.
r5p_unit: a cluster of ceil(rows / R) CTAs of R5P_THREADS threads, one row a warp (R = ceil(rows / R5P_CLUSTER) with
R5P_CLUSTER > 0, else R5P_ROWS_PER_CTA or threads / 32); each warp pushes its picks into the leader's histogram (and
row masks), and after a cluster barrier runs E5 for its own row (lane = pick) on the complete histogram, stores its
final ids / weights and hands the final ids / dropped experts to the leader, which then builds the record only
(r5p_unit.cu header: the exactness argument).

Gates (any miss: RE5's own launch, unchanged):
  prepare  levers.R5P; RE5 ready (rank-agreed) with its routed-sum candidate; the unit imports and its geometry matches;
           a self-test on this rank: r5p vs stock top-k + prune_ids (ids / weights bits) and vs RE5's fused kernel
           (ids / weights / record bytes, untouched bytes included) at every served row count x the probe batches x
           record layouts, plus E5 alone on budget-boundary rows (prune-only mode vs stock prune_ids, max_count 1 / 2
           / 4); the verdict is agreed over the TP CPU group (gloo MIN), so both ranks serve the same row counts.
  call     route_e5.route's fused branch (rows <= ROUTE_E5_FUSED_MAX_ROWS) with ROUTE_E5_TRIGGER 0: rows in the band,
           the record (when armed) has <= R5P_THREADS local experts; INSIDE a CUDA-graph capture only, and only for a
           (rows, layer, record) whose first eager use matched: route_e5.route then launches r5p instead of RE5. Eager
           calls (warmups, the eager audit) always run RE5; the first eager call of each (rows, layer, record) also
           runs r5p into scratch buffers and compares every byte with RE5's output (a difference turns that row count
           off on this rank; the call's own output is RE5's either way).
Exact: the unit reproduces RE5's bytes by construction (r5p_unit.cu header) and by the checks above. Nothing is patched
on a served module or class; decisions depend on the row count, the layer and static facts (both ranks capture the same
kernels unless a first-use check fails on one of them, which changes no bytes).
"""
import torch

from dsv41f import levers

TOP_K = 6
EXPERTS = 384
_NATIVE: list = []
_STATE = {"on": False, "detail": "not prepared"}
_CHECKED: dict = {}       # (rows, layer, record armed) -> True (first eager use == RE5) / False
_OFF: set = set()         # row counts turned off on this rank by a first-use mismatch


def native():
    if not _NATIVE:
        mod = None
        try:
            import route_e5 as _base   # lever RTP's fold: R5P's kernel is part of the route_e5 unit (12 units)
            import types

            mod = types.SimpleNamespace(run=_base.r5p_run, cluster_fits=_base.r5p_cluster_fits,
                                        MAX_ROWS=_base.R5P_MAX_ROWS, TOP_K=_base.TOP_K, EXPERTS=_base.EXPERTS)
        except ImportError as err:
            levers.announce("r5p-native-missing", f"bundle CUDA unit r5p_unit not importable ({err}): lever R5P off")
        if mod is not None and (mod.TOP_K, mod.EXPERTS) != (TOP_K, EXPERTS):
            levers.announce("r5p-native-geometry", "r5p_unit geometry mismatch: lever R5P off")
            mod = None
        _NATIVE.append(mod)
    return _NATIVE[0]


def config(rows: int):
    """(threads, rows a CTA, PDL) of r5p at `rows` rows: rows a CTA = ceil(rows / R5P_CLUSTER) when R5P_CLUSTER > 0 (the
    fewest rows an SM within that cluster size), else R5P_ROWS_PER_CTA (0: threads / 32); threads = R5P_THREADS, or
    (R5P_THREADS 0) 384 up to 12 rows a CTA and 512 above; RE5's ROUTE_E5_PDL."""
    nthreads = int(getattr(levers, "R5P_THREADS", 512))
    rpc = int(getattr(levers, "R5P_ROWS_PER_CTA", 0))
    target = int(getattr(levers, "R5P_CLUSTER", 0))
    if target > 0:
        rpc = min((nthreads or 512) // 32, -(-rows // target))
    if nthreads == 0:
        nthreads = 384 if 0 < rpc <= 12 else 512
    return nthreads, rpc or nthreads // 32, bool(levers.ROUTE_E5_PDL)


def band():
    return int(getattr(levers, "R5P_MIN_ROWS", 49)), int(getattr(levers, "R5P_MAX_ROWS", 127))


def in_band(rows: int) -> bool:
    from dsv41f import route_e5 as re5

    lo, hi = band()
    _, rpc, _ = config(rows)
    return (lo <= rows <= hi and rows <= re5.fused_max() and rows <= int(native().MAX_ROWS)
            and -(-rows // rpc) <= 16 and int(levers.ROUTE_E5_TRIGGER) == 0)


def _record_ok(record, nthreads) -> bool:
    return not record or int(record[8]) <= nthreads


def _run(moe, logits, input_ids, ntnp, weights, ids, cand, record, rows, pdl, ids_in=None, w_in=None,
         budget=None, max_count=None, clocks=0, args=None, nthreads=512, rpc=0):
    """r5p_unit.run with the arguments route_e5._launch passes RE5's fused kernel for this call (nthreads / rpc:
    threads / rows a CTA, 0 = one a warp)."""
    from dsv41f import route_e5 as re5

    b0, m0 = re5.prune_params(rows)
    budget = b0 if budget is None else budget
    max_count = m0 if max_count is None else max_count
    sum_mode, klay = re5.SUM_CANDIDATES[cand]
    ktree = re5.steps_of(re5.tree_of(*klay)) if klay is not None else re5.steps_of(re5.tree_of(8, 1, 1))
    _, mtree, rtree = re5.trees_for(rows)
    image_token, scale, apply_scale = moe.dsv4f_re5_args if args is None else args
    bias, alt = moe.gate.e_score_correction_bias, moe.gate.e_score_correction_bias_vl
    stream = torch.cuda.current_stream(weights.device).cuda_stream
    return native().run(
        logits.data_ptr() if logits is not None else 0, logits.stride(0) if logits is not None else EXPERTS, rows,
        bias.data_ptr(), alt.data_ptr() if alt is not None else 0,
        input_ids.data_ptr() if input_ids is not None else 0, input_ids is not None and input_ids.dtype == torch.int64,
        ntnp.data_ptr() if ntnp is not None else 0, ntnp is not None and ntnp.dtype == torch.int64,
        weights.data_ptr(), ids.data_ptr(), image_token, scale, re5.EPS, apply_scale, sum_mode, budget, max_count,
        re5.steps_of(mtree), re5.steps_of(rtree), nthreads, pdl, clocks, stream, ktree, list(record),
        re5.tree_ids(re5.steps_of(mtree), re5.steps_of(rtree), ktree),
        ids_in.data_ptr() if ids_in is not None else 0, w_in.data_ptr() if w_in is not None else 0, rpc)


def launch(moe, logits, input_ids, ntnp, weights, ids, cand, record, rows) -> bool:
    """route_e5.route's fused branch: True when r5p ran this call (a CUDA-graph capture of a first-use-checked (rows,
    layer, record))."""
    if not (_STATE["on"] and torch.cuda.is_current_stream_capturing()):
        return False
    if rows in _OFF or not _CHECKED.get((rows, moe.layer_id, bool(record)), False):
        return False
    nthreads, rpc, pdl = config(rows)
    if not (in_band(rows) and _record_ok(record, nthreads)):
        return False
    n = _run(moe, logits, input_ids, ntnp, weights, ids, cand, record, rows, pdl, nthreads=nthreads, rpc=rpc)
    if levers.first(f"r5p-on-{rows}"):
        levers.announce(f"r5p-on-{rows}", f"lever R5P: {rows} rows captured on r5p_unit ({nthreads} threads x {n} "
                        f"CTA(s), {rpc} rows a CTA, PDL {'on' if pdl else 'off'}, record {'on' if record else 'off'}), "
                        f"first at layer {moe.layer_id}")
    return True


def _scratch(slot):
    """The slot's six record tensors (map, tok, batch, limit, padded, nonexit) and same-shaped scratch copies (-7)."""
    real = [slot.expanded_idx_to_permuted_idx, slot.permuted_idx_to_token_idx, slot.cta_idx_xy_to_batch_idx,
            slot.cta_idx_xy_to_mn_limit, slot.total_num_padded_tokens, slot.num_non_exiting_ctas]
    if any(not (torch.is_tensor(t) and t.dtype == torch.int32 and t.is_contiguous()) for t in real):
        return None, None
    return real, [torch.full_like(t, -7) for t in real]


def _same_bits(a, b) -> bool:
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def _record_same(real, mine, rows) -> str:
    """'' when r5p's scratch record equals RE5's real one on every written range and r5p wrote nothing else."""
    n = int(real[5][0].item())
    p = int(real[4][0].item())
    if int(mine[5][0].item()) != n or int(mine[4][0].item()) != p:
        return "counts"
    for k, (a, b, size) in enumerate(((real[0], mine[0], rows * TOP_K), (real[1], mine[1], p), (real[2], mine[2], n),
                                      (real[3], mine[3], n))):
        if size > a.numel() or not torch.equal(a[:size], b[:size]):
            return f"array {k}"
        if not bool((b[size:] == -7).all()):
            return f"array {k} past its range"
    return ""


def check(moe, logits, input_ids, ntnp, weights, ids, cand, record, slot, rows) -> None:
    """After RE5's fused launch of an eager call: on the first eager use of (rows, layer, record), run r5p into scratch
    outputs and compare every byte with RE5's output (RE5's output stays the call's output)."""
    if not _STATE["on"] or torch.cuda.is_current_stream_capturing():
        return
    key = (rows, moe.layer_id, bool(record))
    nthreads, rpc, _ = config(rows)
    if key in _CHECKED or rows in _OFF or not (in_band(rows) and _record_ok(record, nthreads)):
        return
    why = ""
    try:
        w2 = torch.empty_like(weights)
        i2 = torch.empty_like(ids)
        rec2 = ()
        real = mine = None
        if record:
            real, mine = _scratch(slot)
            if real is None:
                why = "record slot layout"
            else:
                rec2 = [t.data_ptr() for t in mine] + list(record[6:])
        if not why:
            _run(moe, logits, input_ids, ntnp, w2, i2, cand, rec2, rows, False, nthreads=nthreads, rpc=rpc)
            if not (_same_bits(w2, weights) and _same_bits(i2, ids)):
                why = "ids / weights"
            elif record:
                why = _record_same(real, mine, rows)
    except Exception as err:  # noqa: BLE001 - any failure keeps RE5
        why = f"{type(err).__name__}: {str(err)[:160]}"
    _CHECKED[key] = not why
    if why:
        _OFF.add(rows)
        levers.announce(f"r5p-off-{rows}", f"lever R5P: {rows} rows layer {moe.layer_id} first eager use differs "
                        f"from RE5 ({why}): RE5 at {rows} rows on this rank")
    elif levers.first(f"r5p-chk-{rows}-{int(bool(record))}"):
        levers.announce(f"r5p-chk-{rows}-{int(bool(record))}", f"lever R5P first at {rows} rows (layer "
                        f"{moe.layer_id}, record {'on' if record else 'off'}): == RE5 bit for bit: on")


# ------------------------------------------------------------------------------------------------ self-test
def _buffers(rows, tile, device):
    n = rows * TOP_K
    return [torch.full((n,), -7, dtype=torch.int32, device=device),
            torch.full((n * tile + 64,), -7, dtype=torch.int32, device=device),
            torch.full((n + 8,), -7, dtype=torch.int32, device=device),
            torch.full((n + 8,), -7, dtype=torch.int32, device=device),
            torch.full((4,), -7, dtype=torch.int32, device=device),
            torch.full((4,), -7, dtype=torch.int32, device=device)]


def self_test(moe):
    """(ok, detail) on this rank: r5p vs stock top-k + prune_ids and vs RE5's fused kernel, ids / weights / record
    bytes, at every served row count; E5 alone on budget-boundary rows."""
    import numpy as np

    from dsv41f import route_e5 as re5

    _, cand, max_cluster = re5._READY[0]
    gate = moe.gate
    bias, alt = gate.e_score_correction_bias, gate.e_score_correction_bias_vl
    device = bias.device
    image_token, scale, apply_scale = moe.dsv4f_re5_args
    gen = torch.Generator(device=device)
    gen.manual_seed(int(getattr(levers, "R5P_SEED", 20261016)))
    rng = np.random.default_rng(int(getattr(levers, "R5P_SEED", 20261016)))
    lo, hi = band()
    rows_list = sorted({r for r in (lo, hi, 48, 49, 54, 60, 64, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 127)
                        if in_band(r)})
    if not rows_list:
        return False, "no served row count"
    layouts = [(32, 0, EXPERTS), (16, 0, 192), (32, 192, 192), (8, 0, EXPERTS)]
    cases = bad = drops = bcases = bbad = 0
    first = ""
    shapes = []
    for rows in rows_list:
        nthreads, rpc, _ = config(rows)
        shapes.append(f"{rows}:{nthreads}x{rpc}")
        threads_re5, cluster = re5.shape(rows, max_cluster)
        for trial in range(int(getattr(levers, "R5P_SELFTEST", 4))):
            x, tok, ntnp = re5._probe_batch(gen, rows, trial, image_token, device)
            sw, si = re5._stock_topk(x, bias, alt, tok, ntnp, image_token, scale, apply_scale)
            rw, ri = sw.clone(), si.clone()
            re5._ref_prune(ri, rw, *re5.prune_params(rows))
            drops += int((ri < 0).sum()) - int((si < 0).sum())
            tile, off, nl = layouts[trial % len(layouts)]
            for with_rec in (False, True):
                bufs_a = _buffers(rows, tile, device)
                bufs_b = _buffers(rows, tile, device)
                rec_a = [t.data_ptr() for t in bufs_a] + [tile, off, nl] if with_rec else []
                rec_b = [t.data_ptr() for t in bufs_b] + [tile, off, nl] if with_rec else []
                wa = torch.full((rows, TOP_K), -3.0, device=device)
                ia = torch.full((rows, TOP_K), -5, dtype=torch.int32, device=device)
                wb, ib = wa.clone(), ia.clone()
                re5._launch(x, bias, alt, tok, ntnp, wa, ia, image_token, scale, apply_scale, cand, True, rows,
                            cluster, threads_re5, False, record=rec_a)
                _run(moe, x, tok, ntnp, wb, ib, cand, rec_b, rows, False, nthreads=nthreads, rpc=rpc)
                cases += 1
                ok = (_same_bits(wb, rw) and _same_bits(ib, ri) and _same_bits(wb, wa) and _same_bits(ib, ia)
                      and all(torch.equal(p, q) for p, q in zip(bufs_a, bufs_b)))
                if not ok:
                    bad += 1
                    first = first or f"rows {rows} trial {trial} record {with_rec}"
        # E5 alone on budget-boundary rows (prune-only mode vs stock prune_ids), stock's and two other max_counts
        _, mtree, rtree = re5.trees_for(rows)
        for trial in range(4):
            budget, max_count = re5.prune_params(rows)
            if trial >= 2:
                max_count = 2 if trial == 2 else 4
            bi, bw = re5._boundary_rows(rng, rows, budget, mtree, rtree)
            si = torch.from_numpy(bi).to(device)
            sw = torch.from_numpy(bw).to(device)
            ri, rw = si.clone(), sw.clone()
            re5._ref_prune(ri, rw, budget, max_count)
            wb = torch.empty(rows, TOP_K, dtype=torch.float32, device=device)
            ib = torch.empty(rows, TOP_K, dtype=torch.int32, device=device)
            _run(moe, None, None, None, wb, ib, cand, (), rows, False, ids_in=si, w_in=sw, budget=budget,
                 max_count=max_count, nthreads=nthreads, rpc=rpc)
            bcases += 1
            if not (_same_bits(wb, rw) and _same_bits(ib, ri)):
                bbad += 1
                first = first or f"boundary rows {rows} trial {trial}"
    torch.cuda.synchronize(device)
    detail = (f"{cases} batches vs stock top-k + E5 and RE5 (+ record bytes): {bad} mismatches ({drops} drops); "
              f"{bcases} budget-boundary batches vs stock prune_ids: {bbad} mismatches; shapes (rows:threads x rows a "
              f"CTA) {' '.join(shapes)}" + (f"; first: {first}" if first else ""))
    return bad == 0 and bbad == 0, detail


def prepare(layers) -> str:
    """model_node.prepare (both ranks, after RE5's binding): unit, self-test, rank agreement."""
    if not getattr(levers, "R5P", False):
        _STATE["detail"] = "off (levers.R5P)"
        return _STATE["detail"]
    from dsv41f import route_e5 as re5

    moe = next((r._modules.get("mlp") for r in layers if getattr(r._modules.get("mlp"), "dsv4f_re5", False)), None)
    ok, detail = False, "no RE5-bound MoE replica"
    if moe is not None and re5._READY and re5._READY[0][0]:
        if native() is None:
            detail = "unit r5p_unit unavailable"
        else:
            try:
                ok, detail = self_test(moe)
            except Exception as err:  # noqa: BLE001 - any failure keeps RE5
                ok, detail = False, f"{type(err).__name__}: {str(err)[:200]}"
    elif moe is not None:
        detail = "RE5 not ready"
    agreed = bool(re5._agree([int(ok)])[0])
    _STATE["on"] = agreed
    lo, hi = band()
    _STATE["detail"] = (f"{'ON' if agreed else 'OFF'} (rank-agreed): cluster kernel at {lo}..{hi} rows, PDL "
                        f"{config(lo)[2]}, inside CUDA-graph captures of first-use-checked (rows, layer); self-test "
                        f"on this rank: {detail}")
    return _STATE["detail"]
