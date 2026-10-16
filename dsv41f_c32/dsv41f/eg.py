"""Lever EG on the fork (levers.ENGRAM_FAST_GATHER): the Engram layers' per-rank host-table lookup gathers through the
bundle CUDA unit eg_gather instead of stock's Triton _engram_gather_kernel, and (EG_EARLY) is issued right after
hashing on a side stream.

The engine keeps both Engram tables (layers 1 and 14, fp8 rows of 256 bytes + e8m0 scales) in host memory, one shard
per TP rank (SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank, registered with cudaHostRegister; the fork's
EngramEmbedding._table_ptrs() gives the device-visible pointers). Every rank gathers the rows it owns, zero for the
rest, and an all-reduce assembles them (EngramEmbedding._lookup). The gather is bound by host reads (~11-12 GB/s);
on a prefill chunk it sits on the critical path of layers 1 and 14, the slower rank setting the all-reduce.

Exactness: out = bf16(fp32(fp8) * 2**(e - 127)) rounded once, 2**-127 for e == 0, zero for unowned ids -- stock's
arithmetic, the scale an exact power of two. prepare() compares eg_gather with stock's engram_gather byte for byte on
the served tables before it arms a layer. The all-reduce, wkv and gate after the gather are stock's calls on the same
tensor, so every output byte is stock's.

Early gather: the hash ids exist at the forward start, so every armed layer's gather is issued right after hashing on a
side stream into a buffer allocated on the main stream, and the main stream waits for it right before that layer's
all-reduce; the slow host reads leave the critical path (prefill: behind layer 0's compute; decode / verify graphs:
the ~18 us latency-bound gathers behind layer 0). Only where a fork and its join live in one stream capture or none:
eager calls and whole-forward graph captures, never a breakable prefill graph (its segments end at eager breaks, so a
fork could not be joined inside its segment); there the layer gathers inline with eg_gather. Same kernel, same ids,
same bytes either way.

LEVER EGQ (levers.EGQ, gen_egq.py): at decode / verify sizes (1..EGQ_MAX_ROWS rows) the early fork carries only the
FIRST armed table's gather, at a measured narrow launch (EGQ_LAUNCH: a few CTAs -- an SM runs one kernel's CTAs at a
time, and the 64-CTA grid-stride gather held 64 SMs for its whole ~100 us of host reads, starving layer 0's main-stream
kernels 3-20x); the later tables' gathers are forked just before layer EGQ_DEFER_LAYER (EGQ_DEFER_LAUNCH, 1-2 CTAs;
layer 3: after layer 2's 148-CTA persistent indexer kernel, done long before layer 8's), behind the main stream's
position there, and joined before their own layer's all-reduce as before. Exact: the same eg_gather
kernel on the same ids into a buffer of the same shape -- grid, unroll and stream only change which warp moves which
row and when; every output row is the same per-row dequantize (prepare()'s self-test also runs the EGQ launches byte
for byte against stock's engram_gather). Rank-safe: the schedule depends only on the row count and the switches, so
both ranks capture the same graph, and the all-reduce sees the same bytes as stock's.

EGQ aux means (levers.EGQ_AUX, fork_loop.py): the DSPARK draft's per-target-layer means (`aux.mean(dim=1)`, stock's
call) run on their own side stream (default priority) forked at the call and joined at the end of the layer loop;
stock's call on the same tensor gives the same bytes; the input and the output are recorded on the streams that use
them, so the caching allocator cannot hand either block out early. Same fork / join rule as the early gather (eager or
a whole-forward capture, never a breakable prefill graph). Only at decode / verify sizes (aux rows <= EGQ_MAX_ROWS),
where every residual is a fresh allocation: above that, lever FG's eager EXTEND replays hand the next layer its
residual as a VIEW of FG's single static next-residual buffer (fg.py B.upd), which that same layer's FFN graph then
overwrites in place -- a side-stream reader of it would have no ordering edge with that write (record_stream guards
the allocator, not in-place writers), so the mean runs inline there, as stock. prepare() arms the aux stream only
while FG_MIN_ROWS and AG_MIN_ROWS (their static-buffer replays) exceed EGQ_MAX_ROWS (385 > 384).

LEVER AUX (aux.py, gen_aux.py): the side-stream mean is aux.mean(aux) -- eg_gather's small-grid aux_mean kernel (stock's
bytes, rank-agreed prepare self-test + first eager use per row count checked) where armed, stock's call otherwise.
"""

import torch

from dsv41f import aux as auxk   # lever AUX: the aux means on eg_gather's small-grid kernel
from dsv41f import levers

_UNIT: list = []
_STATE = {"detail": "not prepared"}
_ARMED: list = []      # [(layer id, hash index, embed)] of the armed Engram layers, layer order
_SIDE: dict = {}       # device index -> the early-gather stream
_PEND: dict = {}       # hash index -> (event, out, rows) of this forward's fork, not yet waited for
_DEFER: dict = {}      # layer id -> [(hash index, embed, ids view, rows, cols)] of this forward's deferred forks (EGQ)
_AUX_SIDE: dict = {}   # device index -> the aux-mean stream (EGQ_AUX)
_AUX_PEND: list = []   # [stream] forked by aux_mean and not yet joined


def unit():
    """The bundle's CUDA unit `eg_gather`, or None."""
    if not _UNIT:
        mod = None
        try:
            import eg_gather as mod
        except ImportError as err:
            levers.announce("eg-missing", f"bundle CUDA unit eg_gather not importable ({err}): lever EG off")
        _UNIT.append(mod)
    return _UNIT[0]


def _config(n: int, early: bool = False, bands=None):
    """(grid, unroll) of the measured eg_gather launch at n lookups."""
    if bands is None:
        bands = levers.EG_EARLY_LAUNCH if early else levers.EG_LAUNCH
    for max_n, grid, unroll in bands:
        if n <= max_n:
            return grid, unroll
    return bands[-1][1], bands[-1][2]


def _blk() -> int:
    from sglang.kernels.ops.attention.dsv4.torch_quant import FP8_BLOCK_SIZE

    return int(FP8_BLOCK_SIZE)


def _ptrs(embed):
    return embed._table_ptrs()


def _gather(w, s, ids, out, lo, hi, grid, unroll):
    """eg_gather.gather on tensors: ids [rows, cols] int32 / int64 with unit column stride, out [rows * cols, 256] bf16
    contiguous, on the current stream (the unit takes device pointers: no torch headers in it).

    From levers.EG_SORT_MIN lookups (levers.EG_SORT) the lookups run in ascending table-row order and each row is then
    copied to its own output position: random rows of the ~100 GiB per-rank host table cost the GPU one TLB miss (and page
    walk) each, which also slows every kernel running beside the gather; in address order consecutive lookups share their
    2 MiB pages. Every output row is the same per-row dequantize of the same table row (only the order of work changes),
    and index_copy_ moves bytes, so the output is byte-identical to the unsorted gather."""
    if not (ids.is_cuda and out.is_cuda and ids.device == out.device and ids.dim() == 2 and ids.stride(1) == 1
            and ids.dtype in (torch.int32, torch.int64) and out.dtype == torch.bfloat16 and out.is_contiguous()
            and out.numel() == ids.numel() * 256):
        raise RuntimeError("eg_gather: ids [rows, cols] int32 / int64 (unit column stride) and out bf16 [n, 256]")
    stream = torch.cuda.current_stream(ids.device).cuda_stream
    n = ids.numel()
    if levers.EG_SORT and n >= levers.EG_SORT_MIN:
        sid, perm = torch.sort(ids.reshape(-1))
        tmp = torch.empty_like(out)
        unit().gather(w, s, sid.data_ptr(), sid.dtype == torch.int64, 1, 1, tmp.data_ptr(), n, lo, hi, grid, unroll,
                      _blk(), stream)
        out.index_copy_(0, perm, tmp)
        _STATE["sorted"] = _STATE.get("sorted", 0) + 1
        return
    unit().gather(w, s, ids.data_ptr(), ids.dtype == torch.int64, ids.stride(0), ids.shape[1], out.data_ptr(),
                  n, lo, hi, grid, unroll, _blk(), stream)


def _eligible(embed) -> str:
    table = getattr(embed, "host_table", None)
    if table is None or getattr(table, "layout", None) != "per_rank":
        return f"table layout {getattr(table, 'layout', 'device')} (EG serves the per-rank host table only)"
    blk = _blk()
    if embed.dim != 256 or blk not in (32, 128):
        return f"geometry dim {embed.dim} block {blk}"
    w, s = getattr(embed, "weight", None), getattr(embed, "scale", None)
    if not (isinstance(w, torch.Tensor) and isinstance(s, torch.Tensor) and w.dtype == torch.float8_e4m3fn
            and s.dtype == torch.float8_e8m0fnu and w.shape == (embed.rows, 256)
            and s.shape == (embed.rows, 256 // blk) and w.is_contiguous() and s.is_contiguous()):
        return "table tensors"
    if embed.rows <= 0:
        return "empty shard"
    if not callable(getattr(embed, "_table_ptrs", None)):
        return "no _table_ptrs"
    return ""


def _self_test(embed, device) -> str:
    """eg_gather == stock engram_gather byte for byte on the served table: owned, unowned and boundary ids, strided
    int64 columns, several launch shapes."""
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather

    w, s = _ptrs(embed)
    lo, hi = embed.row_start, embed.row_start + embed.rows
    g = torch.Generator(device=device).manual_seed(20261014 + lo % 997)
    span = embed.rows
    for rows, cols in ((6, 24), (144, 24), (4096, 24)):
        ids3 = torch.randint(max(0, lo - span), hi + span, (rows, 2, cols), generator=g, device=device,
                             dtype=torch.int64)
        ids3[0, 0, :4] = torch.tensor([lo, hi - 1, max(lo - 1, 0), hi], device=device)
        view = ids3[:, 1]
        ref = torch.empty(rows * cols, 256, dtype=torch.bfloat16, device=device)
        engram_gather(w, s, view.reshape(-1).contiguous(), ref, 256, _blk(), row_lo=lo, row_hi=hi)
        shapes = {_config(rows * cols), _config(rows * cols, early=True), (148, 1), (16, 4)}
        if levers.EGQ:   # lever EGQ's decode launches, byte for byte too
            shapes |= {_config(rows * cols, bands=levers.EGQ_LAUNCH), _config(rows * cols, bands=levers.EGQ_DEFER_LAUNCH)}
        for grid, unroll in sorted(shapes):
            got = torch.full_like(ref, 3.0)
            _gather(w, s, view, got, lo, hi, grid, unroll)
            torch.cuda.synchronize(device)
            if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                return f"MISMATCH at {rows}x{cols} grid {grid} unroll {unroll}"
    return ""


def prepare(model) -> str:
    """Arm every eligible Engram layer (unit loaded, self-test passed on its served table); once, outside any capture."""
    if not levers.ENGRAM_FAST_GATHER:
        _STATE["detail"] = "off (ENGRAM_FAST_GATHER=False)"
        return _STATE["detail"]
    if unit() is None:
        _STATE["detail"] = "unit missing"
        return _STATE["detail"]
    armed, notes = [], []
    dev = torch.device("cuda", torch.cuda.current_device())
    for i, layer in enumerate(model.layers):
        engram = getattr(layer, "engram", None)
        if engram is None:
            continue
        embed = engram.embed
        why = _eligible(embed)
        if not why:
            try:
                why = _self_test(embed, dev)
            except Exception as err:  # noqa: BLE001
                why = f"self-test raised {type(err).__name__}: {err}"
        if why:
            notes.append(f"L{i}: {why}")
            continue
        embed.dsv4f_eg = True
        armed.append(i)
        _ARMED.append((i, engram.layer_hash_index, embed))
    early = "off"
    if levers.EG_EARLY and _ARMED:
        _SIDE[dev.index] = torch.cuda.Stream(device=dev)
        early = f"on (side stream, up to {levers.EG_EARLY_MAX_ROWS} rows, bands {levers.EG_EARLY_LAUNCH})"
        if levers.EGQ:
            later = [i for i, _, _ in _ARMED[1:]]
            early += (f"; lever EGQ at 1..{levers.EGQ_MAX_ROWS} rows: L{_ARMED[0][0]} early at {levers.EGQ_LAUNCH}"
                      + (f", L{later} forked before layer {levers.EGQ_DEFER_LAYER} at {levers.EGQ_DEFER_LAUNCH}"
                         if later else ""))
    if levers.EGQ_AUX:
        why = _aux_unsafe()
        if why:
            early += f"; EGQ aux means OFF ({why})"
        else:
            _AUX_SIDE[dev.index] = torch.cuda.Stream(device=dev)
            early += f"; EGQ aux means on a side stream at 1..{levers.EGQ_MAX_ROWS} rows"
    _STATE["detail"] = (f"armed on Engram layers {armed} (bands {levers.EG_LAUNCH}); early gather {early}"
                        + (f"; not armed: {notes}" if notes else ""))
    return _STATE["detail"]


def _aux_unsafe() -> str:
    """Why a side-stream aux mean at 1..EGQ_MAX_ROWS rows could read a static buffer a later main-stream replay
    overwrites in place (lever FG's next residual, lever AG's attention buffers), or '' (static lever facts: the same on
    every rank)."""
    for name, on, lo in (("FG", getattr(levers, "FG", False), getattr(levers, "FG_MIN_ROWS", 0)),
                         ("AG", getattr(levers, "AG", False), getattr(levers, "AG_MIN_ROWS", 0))):
        if on and lo <= levers.EGQ_MAX_ROWS:
            return f"lever {name} replays static buffers from {lo} rows <= EGQ_MAX_ROWS {levers.EGQ_MAX_ROWS}"
    return ""


def _drop(idx) -> None:
    """A fork this layer will not use: the main stream still waits for it, so its buffer is never freed under a running
    gather (and, inside a capture, the side stream is joined)."""
    p = _PEND.pop(idx, None)
    if p is not None:
        torch.cuda.current_stream().wait_event(p[0])


def clear() -> None:
    """A forward that forks nothing: drop any state a previous forward left (never consumed by this one)."""
    _PEND.clear()
    _DEFER.clear()


def _fork(side, main, embed, idx, ids, rows, cols, grid, unroll):
    """One gather on `side` behind main's current position, into a buffer allocated on main; _PEND[idx] = its join."""
    out = torch.empty(rows, cols, embed.dim, dtype=torch.bfloat16, device=ids.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        w, s = _ptrs(embed)
        _gather(w, s, ids, out.view(-1, 256), embed.row_start, embed.row_start + embed.rows, grid, unroll)
        ev = torch.cuda.Event()
        ev.record(side)
    _PEND[idx] = (ev, out, rows)


def issue_early(hash_ids) -> None:
    """Right after hashing: every armed layer's gather on the side stream (hash_ids [T, engram layers, cols]); with
    lever EGQ at decode / verify sizes only the first armed table's, the others deferred to issue_deferred."""
    _PEND.clear()
    _DEFER.clear()
    if not (levers.EG_EARLY and _ARMED) or hash_ids is None or hash_ids.dim() != 3:
        return
    rows, cols = hash_ids.shape[0], hash_ids.shape[2]
    if not 0 < rows <= levers.EG_EARLY_MAX_ROWS or hash_ids.stride(2) != 1:
        return
    from sglang.srt.models.deepseek_v4 import is_in_breakable_cuda_graph

    if is_in_breakable_cuda_graph():
        return
    side = _SIDE.get(hash_ids.device.index)
    if side is None:
        return
    main = torch.cuda.current_stream()
    if levers.EGQ and rows <= levers.EGQ_MAX_ROWS:
        # lever EGQ: the first table now at the narrow decode launch, the others before layer EGQ_DEFER_LAYER
        i0, idx0, embed0 = _ARMED[0]
        grid, unroll = _config(rows * cols, bands=levers.EGQ_LAUNCH)
        _fork(side, main, embed0, idx0, hash_ids[:, idx0], rows, cols, grid, unroll)
        for i, idx, embed in _ARMED[1:]:
            if levers.EGQ_DEFER_LAYER < i:   # forked strictly before the layer that joins it
                _DEFER.setdefault(levers.EGQ_DEFER_LAYER, []).append((idx, embed, hash_ids[:, idx], rows, cols))
            else:
                _fork(side, main, embed, idx, hash_ids[:, idx], rows, cols, grid, unroll)
        if levers.first("egq-on"):
            levers.announce("egq-on", f"lever EGQ: first at {rows} rows: L{i0} gather early (grid {grid}, unroll "
                            f"{unroll}), deferred {sorted((L, [e[0] for e in v]) for L, v in _DEFER.items())} at "
                            f"{_config(rows * cols, bands=levers.EGQ_DEFER_LAUNCH)}")
        return
    grid, unroll = _config(rows * cols, early=True)
    outs = [torch.empty(rows, cols, embed.dim, dtype=torch.bfloat16, device=hash_ids.device)
            for _, _, embed in _ARMED]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        for (_, idx, embed), out in zip(_ARMED, outs):
            w, s = _ptrs(embed)
            _gather(w, s, hash_ids[:, idx], out.view(-1, 256), embed.row_start, embed.row_start + embed.rows,
                    grid, unroll)
            ev = torch.cuda.Event()
            ev.record(side)
            _PEND[idx] = (ev, out, rows)
    if levers.first("eg-early-on"):
        levers.announce("eg-early-on", f"Engram early gather forked, first at {rows} rows (grid {grid}, unroll {unroll})")


def issue_deferred(layer_id: int) -> None:
    """Lever EGQ: right before layer `layer_id`'s forward, the gathers deferred to it, on the side stream behind the
    main stream's position (joined by engram_forward before their layer's all-reduce, as the early fork)."""
    todo = _DEFER.pop(layer_id, None) if _DEFER else None
    if not todo:
        return
    main = torch.cuda.current_stream()
    for idx, embed, ids, rows, cols in todo:
        side = _SIDE.get(ids.device.index)
        if side is None:
            continue   # never armed on this device: engram_forward gathers inline (stock's bytes)
        grid, unroll = _config(rows * cols, bands=levers.EGQ_DEFER_LAUNCH)
        _fork(side, main, embed, idx, ids, rows, cols, grid, unroll)


def aux_mean(aux):
    """EGQ aux: stock's `aux.mean(dim=1)` on the aux side stream behind main's current position (joined by aux_join);
    stock's call inline where no side stream is armed, above EGQ_MAX_ROWS rows or inside a breakable prefill graph."""
    side = (_AUX_SIDE.get(aux.device.index)
            if (levers.EGQ_AUX and aux.is_cuda and aux.dim() == 3 and aux.shape[0] <= levers.EGQ_MAX_ROWS) else None)
    if side is None:   # also every eager EXTEND above EGQ_MAX_ROWS rows: aux may be lever FG's static buffer view
        return aux.mean(dim=1)
    from sglang.srt.models.deepseek_v4 import is_in_breakable_cuda_graph

    if is_in_breakable_cuda_graph():
        return aux.mean(dim=1)
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        out = auxk.mean(aux)   # lever AUX: stock's bytes on a small grid where armed, else stock's call
    aux.record_stream(side)    # its block is not handed out again before the side stream has read it
    out.record_stream(main)    # nor the mean's before main's readers (after aux_join)
    if side not in _AUX_PEND:
        _AUX_PEND.append(side)
    return out


def aux_join() -> None:
    """EGQ aux: the main stream waits for every aux mean forked by this forward (end of the layer loop)."""
    if _AUX_PEND:
        main = torch.cuda.current_stream()
        for side in _AUX_PEND:
            main.wait_stream(side)
        _AUX_PEND.clear()


def describe() -> str:
    return _STATE["detail"]


def engram_forward(engram, x, ids, forward_batch, cp_all_tokens):
    """Stock Engram.forward(x, ids, forward_batch, cp_all_tokens=...) with EngramEmbedding._lookup's gather through
    eg_gather (or this forward's early fork); the all-reduce, wkv and gate are stock's calls."""
    embed = engram.embed
    for v in _DEFER.values():   # lever EGQ: a fork still deferred for this table is never issued after its layer
        v[:] = [e for e in v if e[0] != engram.layer_hash_index]
    if not getattr(embed, "dsv4f_eg", False) or cp_all_tokens or ids.dim() != 2 or ids.shape[0] == 0 \
            or ids.stride(1) != 1:
        _drop(engram.layer_hash_index)
        return engram(x, ids, forward_batch, cp_all_tokens=cp_all_tokens)
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.engram import engram_gate
    from sglang.srt.runtime_context import get_parallel

    if embed.tp_size > 1 and get_parallel().attn_dp_size > 1:
        _drop(engram.layer_hash_index)
        return engram(x, ids, forward_batch, cp_all_tokens=cp_all_tokens)
    rows, cols = ids.shape
    p = _PEND.pop(engram.layer_hash_index, None)
    if p is not None:
        torch.cuda.current_stream().wait_event(p[0])   # used or not: never free its buffer under a running gather
    if p is not None and p[2] == rows and p[1].shape[1] == cols:
        out = p[1]
    else:
        out = torch.empty(rows, cols, embed.dim, dtype=torch.bfloat16, device=ids.device)
        grid, unroll = _config(rows * cols)
        w, s = _ptrs(embed)
        _gather(w, s, ids, out.view(-1, embed.dim), embed.row_start, embed.row_start + embed.rows, grid, unroll)
        if levers.first("eg-on"):
            levers.announce("eg-on", f"Engram gather through eg_gather, first at {rows} rows (grid {grid}, unroll "
                            f"{unroll})")
    emb = tensor_model_parallel_all_reduce(out) if embed.tp_size > 1 else out
    if x.shape[0] == 0:
        return x
    kv, _ = engram.wkv(emb.flatten(-2))
    return engram_gate(x, kv, engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value)
