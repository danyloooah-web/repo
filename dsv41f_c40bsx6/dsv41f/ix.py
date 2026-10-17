"""Lever IX: the layer-invariant host work of the ratio-1 / ratio-2 index layers, done once per eager EXTEND forward of
our model node instead of once per index layer (exact: a reused result is stock's own tensor, computed by stock's
statements at its first use in the forward and only read afterwards).

What repeats. An eager EXTEND forward (contexts past the fork's 16k prefill-graph envelope) runs
DeepseekV4AttnBackend.forward_low_ratio_sources on the 8 index layers -- 2, 8, 14 (ratio 2, full top-k), 20 (ratio 1,
candidate source), 24, 28, 32, 36 (ratio 1, candidate consumers) -- and every call rebuilds from the batch alone:
  1. IX_REQ: token_req_indices(forward_batch, num_tokens=rows), the per-row request pool index (a dtype cast + a
     repeat_interleave: 5 launches), the same tensor in all 8 layers (stock rebuilds it 8 times);
  2. IX_DG: the DeepGEMM prefill indexer's get_deep_gemm_prefill_data (full top-k, publish and consume all start with
     it): besides the layer's own queries (_index_q_and_weights: wq_b, head weights, rope + fp4 pack) it rebuilds the
     flattened-K column slots, the per-row request starts and the per-row compressed lengths from the batch's
     lengths, request pool rows, req_to_token, extend lengths and positions -- two pinned H2D copies + 22 launches,
     the same tensors in every layer of one ratio (3 ratio-2 + 5 ratio-1 calls, 2 distinct results).
No host sync is involved (stock passes output_size to every repeat_interleave and stages its H2D copies in pinned
memory), so the saving is pure host time on the host-bound rank.

How. install() rebinds two module globals, both acting only inside our forward: forward() below sets a ContextVar for
eager EXTEND / MIXED forwards outside any CUDA graph capture or breakable prefill graph; the served model's own forward
(the node audit's stock twin), decode / verify / draft forwards and every other caller reach stock unchanged.
  - deepseek_v4_backend.token_req_indices (forward_low_ratio_sources' name for dsv41_sparse.token_req_indices): a
    per-forward memo keyed on (mode, req_pool_indices, extend_seq_lens, num_tokens) for EXTEND / MIXED batches;
  - full_topk / sparse_table.get_deep_gemm_prefill_data (their names for scoring.get_deep_gemm_prefill_data): the
    generated copy ix_dg.get_deep_gemm_prefill_data (gen_ix.py: stock's function with asserted edits), which runs
    stock's statements on a miss and takes the three tensors from this forward's memo on a hit, keyed on (ratio, the
    per-request compressed lengths, the per-request rows, positions, req_pool_indices, rows_per_request_device,
    req_to_token). The layer's queries are computed in every call exactly as stock (stock's _index_q_and_weights).
Keys hold each tensor's identity (pointer, shape, strides, dtype, device) and a reference to it, so no key tensor can
be freed and its address reused while the forward runs; their contents are fixed during a forward (batch metadata and
req_to_token are written by the scheduler before it; the memoized tensors reach kernels as const arguments only). The
memo lives in the forward's ContextVar state and is dropped when the forward returns.

Checks. In the first IX_CHECK_FORWARDS eligible forwards that reuse anything, the first hit of every memo entry
recomputes stock's result, compares it with the memo byte for byte (dtype, shape, strides, values) and returns the
fresh one; any difference turns that memo off for the process (stock from then on) and prints a line. Source pins: the
generated ix_dg carries the sha256 of the stock sources it was lifted from / relies on (gen_ix.py); install() compares
them with the running stack's sources (inspect + ast, as lever FL does) and leaves a memo off on any difference.

A/B (diagnostic, levers.IX_AB = N > 0): eligible forwards alternate memo on / off; the host time spent inside
forward_low_ratio_sources (timed around the class attribute, i.e. around lever CT's wrapper) and in the whole forward
is summed per arm, and every N eligible forwards each rank prints one line
  CACHEON_DSV4F lever IX A/B cuda:<dev>: memo on <n> fwd (<rows> rows), index sources <us> us/fwd over <calls> calls
  (<us> us/call), forward host <ms> ms | memo off ... | saved <us> us/fwd in the index sources; ...
"""
import ast
import contextvars
import hashlib
import inspect
import sys
import time

import torch

from dsv41f import levers

_ON = contextvars.ContextVar("dsv41f_ix", default=None)
_STATE = {"detail": "not installed", "installed": False, "req_stock": None, "dg_stock": None, "dg_fn": None,
          "req_ok": False, "dg_ok": False, "check_forwards": 0, "checks": {"req": 0, "dg": 0},
          "hits": {"req": 0, "dg": 0}, "miss": {"req": 0, "dg": 0}, "forwards": 0, "modes": (), "in_bcg": None,
          "sources_orig": None, "ab_n": 0, "ab": {True: [0, 0.0, 0, 0.0, 0], False: [0, 0.0, 0, 0.0, 0]}}

_SPARSE = "sglang.srt.layers.attention.dsv4.dsv41_sparse"
_IDX = "sglang.srt.layers.attention.dsv4.v41_indexer"
_SCORING = f"{_IDX}.scoring"


def _source_hash(module, name: str, parsed: dict):
    """sha256[:16] of the exact source text of the module-level function / class `name` (None: not found); `parsed`
    caches each module's (source, ast) for one install."""
    hit = parsed.get(id(module))
    if hit is None:
        src = inspect.getsource(module)
        hit = parsed[id(module)] = (src, ast.parse(src))
    src, tree = hit
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            seg = ast.get_source_segment(src, node)
            return hashlib.sha256(seg.encode()).hexdigest()[:16] if seg is not None else None
    return None


def install() -> str:
    """Pin + rebind once (model_node.prepare, after lever CT, before any capture)."""
    if _STATE["installed"]:
        return _STATE["detail"]
    if not getattr(levers, "IX", False):
        _STATE["detail"] = "off"
        return _STATE["detail"]
    try:
        from sglang.srt.layers.attention import deepseek_v4_backend as bk
        from sglang.srt.layers.attention.dsv4 import dsv41_sparse as sp
        from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk, scoring, sparse_table
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )

        from dsv41f import ix_dg
        mods = {_SPARSE: sp, _SCORING: scoring, f"{_IDX}.full_topk": full_topk, f"{_IDX}.sparse_table": sparse_table}
        bad, parsed = [], {}
        for key, want in ix_dg.PINS.items():
            label, name = key.split(":")
            if label not in mods or _source_hash(mods[label], name, parsed) != want:
                bad.append(key)
        modes = (ForwardMode.EXTEND, ForwardMode.MIXED)
    except Exception as e:  # noqa: BLE001 - another stack keeps stock's index layers
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:160]})"
        return _STATE["detail"]
    _STATE["modes"] = modes
    _STATE["in_bcg"] = is_in_breakable_cuda_graph
    parts = []
    if not getattr(levers, "IX_REQ", False):
        parts.append("req off")
    elif bk.token_req_indices is not sp.token_req_indices:
        parts.append("req off (the backend's token_req_indices is not stock's)")
    elif f"{_SPARSE}:token_req_indices" in bad:
        parts.append("req off (token_req_indices differs from the pinned source)")
    else:
        _STATE["req_stock"] = sp.token_req_indices
        bk.token_req_indices = _req
        _STATE["req_ok"] = True
        parts.append("req on")
    stock = scoring.get_deep_gemm_prefill_data
    dg_bad = [b for b in bad if not b.startswith(_SPARSE)]
    if not getattr(levers, "IX_DG", False):
        parts.append("dg off")
    elif full_topk.get_deep_gemm_prefill_data is not stock or sparse_table.get_deep_gemm_prefill_data is not stock:
        parts.append("dg off (the indexer modules' get_deep_gemm_prefill_data is not stock's)")
    elif dg_bad:
        parts.append(f"dg off (differs from the pinned sources: {dg_bad})")
    else:
        _STATE["dg_stock"] = stock
        _STATE["dg_fn"] = ix_dg.get_deep_gemm_prefill_data
        full_topk.get_deep_gemm_prefill_data = _dg
        sparse_table.get_deep_gemm_prefill_data = _dg
        _STATE["dg_ok"] = True
        parts.append("dg on")
    if _STATE["req_ok"] or _STATE["dg_ok"]:
        parts.append(f"first hit of every memo entry checked against stock in the first {levers.IX_CHECK_FORWARDS} "
                     f"reusing forwards")
    ab = int(getattr(levers, "IX_AB", 0) or 0)
    if ab > 0 and (_STATE["req_ok"] or _STATE["dg_ok"]):
        orig = bk.DeepseekV4AttnBackend.forward_low_ratio_sources
        _STATE["sources_orig"] = orig

        def timed_sources(self, *args, **kwargs):
            st = _ON.get()
            if st is None:
                return orig(self, *args, **kwargs)
            t0 = time.perf_counter()
            try:
                return orig(self, *args, **kwargs)
            finally:
                st["t_src"] += time.perf_counter() - t0
                st["n_src"] += 1

        bk.DeepseekV4AttnBackend.forward_low_ratio_sources = timed_sources
        parts.append(f"A/B ON: the memo alternates per eligible forward, host-time summary every {ab} forwards")
    _STATE["installed"] = True
    _STATE["detail"] = "; ".join(parts)
    return _STATE["detail"]


def forward(fn, prepared, *args, **kwargs):
    """Run the node forward `fn` inside the memo scope for eager EXTEND / MIXED forwards (stock reach otherwise)."""
    if not (_STATE["req_ok"] or _STATE["dg_ok"]):
        return fn(prepared, *args, **kwargs)
    fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    mode = getattr(fb, "forward_mode", None)
    if mode not in _STATE["modes"] or torch.cuda.is_current_stream_capturing() or _STATE["in_bcg"]():
        return fn(prepared, *args, **kwargs)
    st = {"memo": True, "check": _STATE["check_forwards"] < levers.IX_CHECK_FORWARDS, "checked": 0, "hits": 0,
          "req": {}, "dg": {}, "t_src": 0.0, "n_src": 0}
    ab = int(levers.IX_AB or 0)
    if ab > 0:
        st["memo"] = _STATE["ab_n"] % 2 == 0
        _STATE["ab_n"] += 1
    _STATE["forwards"] += 1
    t0 = time.perf_counter()
    token = _ON.set(st)
    try:
        return fn(prepared, *args, **kwargs)
    finally:
        _ON.reset(token)
        _end(st)
        if ab > 0:
            try:
                _ab_record(st, time.perf_counter() - t0, args, kwargs, ab)
            except Exception as e:  # noqa: BLE001 - a diagnostic never fails the forward
                levers.announce("ix-ab-error", f"lever IX A/B record failed ({type(e).__name__}: {str(e)[:120]})")


def _counts() -> str:
    h, m = _STATE["hits"], _STATE["miss"]
    return f"req hits {h['req']} miss {m['req']}, dg hits {h['dg']} miss {m['dg']}"


def _end(st) -> None:
    if st["hits"] and levers.first("ix-active"):
        levers.announce("ix-active", f"lever IX: first reusing eager EXTEND forward: {st['hits']} memo hits, "
                        f"{st['checked']} checked against stock ({_counts()})")
    if st["checked"]:
        _STATE["check_forwards"] += 1
        if _STATE["check_forwards"] == levers.IX_CHECK_FORWARDS and (_STATE["req_ok"] or _STATE["dg_ok"]):
            c = _STATE["checks"]
            levers.announce("ix-checked", f"lever IX: {c['req']} req + {c['dg']} dg memo hits of "
                            f"{levers.IX_CHECK_FORWARDS} forwards matched stock byte for byte: req memo "
                            f"{'on' if _STATE['req_ok'] else 'OFF'}, dg memo {'on' if _STATE['dg_ok'] else 'OFF'} "
                            f"({_counts()})")


def _ab_record(st, t_fwd, args, kwargs, every) -> None:
    pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
    acc = _STATE["ab"][st["memo"]]
    acc[0] += 1
    acc[1] += st["t_src"]
    acc[2] += st["n_src"]
    acc[3] += t_fwd
    acc[4] += int(pos.shape[0]) if isinstance(pos, torch.Tensor) else 0
    if _STATE["ab_n"] % every:
        return

    def arm(a):
        n = max(a[0], 1)
        return (f"{a[0]} fwd ({a[4] / n:.0f} rows), index sources {a[1] / n * 1e6:.0f} us/fwd over {a[2] / n:.1f} "
                f"calls ({a[1] / max(a[2], 1) * 1e6:.0f} us/call), forward host {a[3] / n * 1e3:.2f} ms")

    on, off = _STATE["ab"][True], _STATE["ab"][False]
    saved = (off[1] / max(off[0], 1) - on[1] / max(on[0], 1)) * 1e6
    dev = pos.device if isinstance(pos, torch.Tensor) else "?"
    print(f"CACHEON_DSV4F lever IX A/B {dev}: memo on {arm(on)} | memo off {arm(off)} | "
          f"saved {saved:.0f} us/fwd in the index sources; {_counts()}", file=sys.stderr, flush=True)


def _sig(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


def _equal(a, b) -> bool:
    return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
            and tuple(a.shape) == tuple(b.shape) and tuple(a.stride()) == tuple(b.stride()) and a.device == b.device
            and torch.equal(a, b))


def _off(kind: str, why: str) -> None:
    _STATE[f"{kind}_ok"] = False
    print(f"CACHEON_DSV4F lever IX: a {kind} memo hit differs from stock ({why}): {kind} memo OFF (stock from now on)",
          file=sys.stderr, flush=True)


def _hit(st, kind: str, entry) -> bool:
    """A hit on `entry` ([refs, value, checked]): True = verify it against stock now (the entry's first hit in one of
    the first IX_CHECK_FORWARDS reusing forwards), False = reuse it."""
    if st["check"] and not entry[2]:
        entry[2] = True
        st["checked"] += 1
        _STATE["checks"][kind] += 1
        return True
    st["hits"] += 1
    _STATE["hits"][kind] += 1
    return False


# ---------------------------------------------------------------------------------------------------- IX_REQ

def _req(forward_batch, *, num_tokens=None):
    """deepseek_v4_backend.token_req_indices inside our eager EXTEND forward: the forward's first result, reused."""
    stock = _STATE["req_stock"]
    st = _ON.get()
    if st is None or not st["memo"] or not _STATE["req_ok"]:
        return stock(forward_batch, num_tokens=num_tokens)
    mode = getattr(forward_batch, "forward_mode", None)
    pool = getattr(forward_batch, "req_pool_indices", None)
    lens = getattr(forward_batch, "extend_seq_lens", None)
    if (mode not in _STATE["modes"] or not isinstance(pool, torch.Tensor) or not isinstance(lens, torch.Tensor)
            or not (num_tokens is None or isinstance(num_tokens, int))):
        return stock(forward_batch, num_tokens=num_tokens)
    key = (mode, _sig(pool), _sig(lens), num_tokens)
    entry = st["req"].get(key)
    if entry is None:
        out = stock(forward_batch, num_tokens=num_tokens)
        st["req"][key] = [(pool, lens), out, False]
        _STATE["miss"]["req"] += 1
        return out
    if _hit(st, "req", entry):
        fresh = stock(forward_batch, num_tokens=num_tokens)
        if not _equal(fresh, entry[1]):
            _off("req", f"{num_tokens} rows")
        return fresh
    return entry[1]


# ---------------------------------------------------------------------------------------------------- IX_DG

def _dg(inputs, req_to_token):
    """full_topk / sparse_table.get_deep_gemm_prefill_data: the memo copy inside our eager EXTEND forward."""
    st = _ON.get()
    if st is None or not st["memo"] or not _STATE["dg_ok"]:
        return _STATE["dg_stock"](inputs, req_to_token)
    return _STATE["dg_fn"](inputs, req_to_token)


def dg_lookup(inputs, req_to_token, lens_per_request):
    """ix_dg: (the memoized (k_slots, request_starts, compress_lens), None) on a hit, else (None, memo): the caller runs
    stock's statements and hands their results to dg_store(memo, ...) (a miss, a checked hit, or no memo: memo None)."""
    st = _ON.get()
    pos, pool, rows_dev = inputs.positions, inputs.req_pool_indices, inputs.rows_per_request_device
    if (st is None or not _STATE["dg_ok"] or not isinstance(pos, torch.Tensor) or not isinstance(pool, torch.Tensor)
            or not isinstance(rows_dev, torch.Tensor) or not isinstance(req_to_token, torch.Tensor)
            or inputs.rows_per_request is None):
        return None, None
    key = (int(inputs.compress_ratio), tuple(int(v) for v in lens_per_request),
           tuple(int(v) for v in inputs.rows_per_request), _sig(pos), _sig(pool), _sig(rows_dev), _sig(req_to_token))
    entry = st["dg"].get(key)
    if entry is None:
        return None, (st, key, (pos, pool, rows_dev, req_to_token), None)
    if _hit(st, "dg", entry):
        return None, (st, key, None, entry[1])
    return entry[1], None


def dg_store(memo, k_slots, request_starts, compress_lens):
    """ix_dg: stock's three tensors of a miss (kept for the rest of the forward) or of a checked hit (compared with the
    memo: any byte different turns the memo off). Returns stock's fresh tensors."""
    fresh = (k_slots, request_starts, compress_lens)
    if memo is None:
        return fresh
    st, key, refs, cached = memo
    if cached is None:
        st["dg"][key] = [refs, fresh, False]
        _STATE["miss"]["dg"] += 1
    elif not all(_equal(a, b) for a, b in zip(fresh, cached)):
        _off("dg", f"ratio {key[0]}, {len(key[1])} requests, {key[3][1][0]} rows")
    return fresh
