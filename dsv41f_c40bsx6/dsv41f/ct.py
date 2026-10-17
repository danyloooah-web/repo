"""Lever CT: two exact trims of the fork's trtllm-gen decode attention glue, active only inside our model forward
(forward() below sets the context; the served model's own forward -- the node audit's stock twin -- is untouched).

1. Counter buffer. The fork installs one persistent multi-CTA KV counter buffer for every trtllm-gen launch
   (deepseek_v4_trtllm_backend._install_persistent_trtllm_semaphores) by rebinding
   flashinfer.mla._core._get_trtllm_gen_multi_ctas_kv_counter_buffer, but this FlashInfer's DSv4 launcher resolves its
   buffer through _resolve_trtllm_gen_multi_ctas_kv_counter_buffer, which calls flashinfer.utils' own copy: every
   attention launch still allocates and zero-fills a fresh buffer (a FillFunctor<uint8> kernel before each fmha). Inside
   our DECODE / VERIFY forwards the resolver gets the fork's intended buffer instead: allocated once outside graph
   capture, sized for the fork's query-row capacity, zero at allocation; the decode kernel resets its counters at the
   end of every launch. Extend forwards keep stock's fresh buffer: with one shared buffer the VarSeq prefill launches
   of a cold 24 x 60k prefill hung (bench n47a: a wo_b all-reduce timed out on both ranks behind a stuck attention).
2. Sparse tail. Ratio-1 / ratio-2 layers share one decode table (metadata trtllm_c4_indices / trtllm_c4_lens) and the
   layers between two index source layers read their source's top-k, so each re-packs the same bytes into the same table
   (pack_sparse_tail). A decode / verify pack is skipped when its operands (tensors, shapes, strides) equal the previous
   pack's and no layer with an indexer or compressor ran since (its table already holds those bytes); the first eager
   would-be skips run the pack anyway and compare the table bytes before / after (any change: the skip is off for good).
3. Prefill (lever CT_PREFILL; eager EXTEND forwards outside any graph): the backend's _forward_trtllm_prefill, lifted as
   ct_pf.forward_trtllm_prefill (generated, two asserted edits), rebuilds the layer-invariant SWA index table of the chunk
   in every layer and gives every ratio-1 / ratio-2 layer a fresh lens buffer + a tail pack, though the layers between
   two index source layers pack the same bytes into the same per-chunk table. Here the chunk's SWA table is built once
   and a consumer layer reuses the previous pack's lens buffer (same rule as 2.); the first CT_PREFILL_CHECKS hits of
   each kind recompute stock's result and compare it byte for byte (any difference: stock for good).
"""
import contextvars
import sys

import torch

from dsv41f import levers

_ON = contextvars.ContextVar("dsv41f_ct", default=None)
_STATE = {"detail": "not installed", "buf": None, "old": [], "calls": 0, "fallbacks": 0, "packs": 0, "skips": 0,
          "checks": 0, "tail_ok": True, "orig_resolve": None, "orig_pack": None, "orig_sources": None,
          "pf_ok": True, "pf_checks": 0, "pf_skips": 0, "pf_swa_hits": 0, "orig_pf": None, "pack_fn": None}


def install(layers=None, attn_forward=None) -> str:
    """Wrap the counter resolver and the tail pack (model_node.prepare, before any capture)."""
    parts = []
    parts.append(_install_counter() if levers.CT_COUNTER else "counter off")
    parts.append(_install_tail(layers, attn_forward) if levers.CT_TAIL else "tail off")
    parts.append(_install_prefill() if getattr(levers, "CT_PREFILL", False) else "prefill off")
    _STATE["detail"] = "; ".join(parts)
    return _STATE["detail"]


def _install_counter() -> str:
    if _STATE["orig_resolve"] is not None:
        return "counter on"
    try:
        import flashinfer.mla._core as fic
        from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes
        from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as tb
        orig = fic._resolve_trtllm_gen_multi_ctas_kv_counter_buffer
    except Exception as e:  # noqa: BLE001 - another stack keeps stock's per-launch buffer
        return f"counter off ({type(e).__name__}: {str(e)[:120]})"
    _STATE["orig_resolve"] = orig

    def resolve(counter_buffer, batch_size, num_qo_heads, sm_count, device):
        st = _ON.get()
        if counter_buffer is not None or st is None or not st["dec"]:
            return orig(counter_buffer, batch_size, num_qo_heads, sm_count, device)
        need = get_trtllm_gen_multi_ctas_kv_counter_bytes(batch_size, num_qo_heads, sm_count)
        buf = _STATE["buf"]
        if buf is None or buf.device != device or buf.numel() < need:
            if torch.cuda.is_current_stream_capturing():
                _STATE["fallbacks"] += 1
                return orig(None, batch_size, num_qo_heads, sm_count, device)    # stock: a fresh zeroed buffer
            rows = max(int(getattr(tb, "_trtllm_semaphore_rows", 0) or 0), int(batch_size))
            nbytes = max(need, get_trtllm_gen_multi_ctas_kv_counter_bytes(rows, num_qo_heads, sm_count))
            if buf is not None:
                _STATE["old"].append(buf)     # graphs captured with it keep their address valid
            buf = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            _STATE["buf"] = buf
            levers.announce(f"ct-buf{nbytes}", f"lever CT: persistent counter buffer {nbytes} B ({rows} query rows, "
                            f"{num_qo_heads} heads) on {device}")
        _STATE["calls"] += 1
        return orig(buf, batch_size, num_qo_heads, sm_count, device)

    fic._resolve_trtllm_gen_multi_ctas_kv_counter_buffer = resolve
    return "counter on"


def _install_tail(layers, attn_forward) -> str:
    if _STATE["orig_pack"] is not None:
        return "tail on"
    try:
        from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as tb
        from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
        orig_pack = tb.pack_sparse_tail
        orig_sources = DeepseekV4AttnBackend.forward_low_ratio_sources
    except Exception as e:  # noqa: BLE001
        return f"tail off ({type(e).__name__}: {str(e)[:120]})"
    if layers is not None:
        attn = [r._modules.get("self_attn") for r in layers]
        mqa = [a for a in attn if a is not None and type(a).__name__ == "MQALayer"]
        bound = sum(getattr(getattr(a, "forward", None), "__func__", None) is attn_forward for a in mqa)
        if not mqa or bound != len(mqa):
            return f"tail off ({bound} of {len(mqa)} attention replicas run the lifted forward)"
    _STATE["orig_pack"] = orig_pack
    _STATE["orig_sources"] = orig_sources

    def pack(indices, lengths, table, total_lengths, swa_width=128):
        st = _ON.get()
        if st is None or not st["dec"] or not levers.CT_TAIL or not _STATE["tail_ok"]:
            return orig_pack(indices, lengths, table, total_lengths, swa_width)
        key = (st["gen"], _sig(indices), _sig(lengths), _sig(table), _sig(total_lengths), int(swa_width))
        if key != st["last"]:
            st["last"] = key
            _STATE["packs"] += 1
            return orig_pack(indices, lengths, table, total_lengths, swa_width)
        if _STATE["checks"] < levers.CT_TAIL_CHECKS:
            if torch.cuda.is_current_stream_capturing():
                return orig_pack(indices, lengths, table, total_lengths, swa_width)   # unchecked yet: stock
            t0, l0 = table.clone(), total_lengths.clone()
            orig_pack(indices, lengths, table, total_lengths, swa_width)
            same = torch.equal(t0, table) and torch.equal(l0, total_lengths)
            _STATE["checks"] += 1
            if not same:
                _STATE["tail_ok"] = False
                print(f"CACHEON_DSV4F lever CT: a would-be skipped tail pack changed the table (rows "
                      f"{indices.shape[0]}): tail skip OFF", file=sys.stderr, flush=True)
            elif _STATE["checks"] == levers.CT_TAIL_CHECKS:
                levers.announce("ct-tail-ok", f"lever CT: {_STATE['checks']} would-be skipped tail packs left the "
                                f"table unchanged: tail skip on")
            return None
        _STATE["skips"] += 1
        return None

    def sources(self, *args, **kwargs):
        st = _ON.get()
        if st is not None:
            st["gen"] += 1            # a compressor / indexer ran: the next pack is fresh
        return orig_sources(self, *args, **kwargs)

    tb.pack_sparse_tail = pack
    DeepseekV4AttnBackend.forward_low_ratio_sources = sources
    return f"tail on (checked on the first {levers.CT_TAIL_CHECKS} eager would-be skips)"


def _install_prefill() -> str:
    if _STATE["orig_pf"] is not None:
        return "prefill on"
    if _STATE["orig_sources"] is None:
        return "prefill off (needs the tail's indexer / compressor hook)"
    try:
        from sglang.kernels.ops.attention.dsv4.trtllm_metadata import pack_sparse_tail
        from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as tb
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )

        from dsv41f import ct_pf
        cls = tb.DeepseekV4TrtllmAttnBackend
        orig = cls._forward_trtllm_prefill
    except Exception as e:  # noqa: BLE001
        return f"prefill off ({type(e).__name__}: {str(e)[:120]})"
    _STATE["orig_pf"] = orig
    _STATE["pack_fn"] = pack_sparse_tail

    def prefill(self, **kwargs):
        st = _ON.get()
        if (st is None or st["dec"] or not _STATE["pf_ok"] or is_in_breakable_cuda_graph()
                or torch.cuda.is_current_stream_capturing()):
            return orig(self, **kwargs)
        return ct_pf.forward_trtllm_prefill(self, **kwargs)

    cls._forward_trtllm_prefill = prefill
    return f"prefill on (first {levers.CT_PREFILL_CHECKS} hits of each kind checked against stock)"


def _pf_fail(what: str) -> None:
    _STATE["pf_ok"] = False
    print(f"CACHEON_DSV4F lever CT: prefill {what} differs from stock: prefill memo OFF", file=sys.stderr, flush=True)


def pf_swa(core, swa_page_indices, sum_q, make):
    """ct_pf: the chunk's SWA index table, built once per forward and chunk (stock rebuilds it every layer)."""
    st = _ON.get()
    key = (id(core), _sig(swa_page_indices), int(sum_q))
    hit = st.get("pf_swa")
    if hit is not None and hit[0] == key and _STATE["pf_ok"]:
        if _STATE["pf_checks"] < levers.CT_PREFILL_CHECKS:
            fresh = make()
            _STATE["pf_checks"] += 1
            if not torch.equal(fresh, hit[1]):
                _pf_fail("SWA table")
            return fresh
        _STATE["pf_swa_hits"] += 1
        return hit[1]
    fresh = make()
    st["pf_swa"] = (key, fresh)
    return fresh


def pf_tail(core, make_lens, indices, lengths, table):
    """ct_pf: a ratio-1 / ratio-2 layer's lens buffer; the previous pack's when its operands are unchanged and no layer
    with an indexer or compressor ran since (the table already holds the bytes), else a fresh buffer and stock's pack."""
    st = _ON.get()
    key = (id(core), st["gen"], _sig(indices), _sig(lengths), _sig(table))
    hit = st.get("pf_tail")
    pack = _STATE["pack_fn"]
    if hit is not None and hit[0] == key and _STATE["pf_ok"]:
        if _STATE["pf_checks"] < levers.CT_PREFILL_CHECKS:
            before = table.clone()
            fresh = make_lens()
            pack(indices, lengths, table, fresh)
            _STATE["pf_checks"] += 1
            if not (torch.equal(fresh, hit[1]) and torch.equal(before, table)):
                _pf_fail("tail pack")
            elif _STATE["pf_checks"] == levers.CT_PREFILL_CHECKS:
                levers.announce("ct-pf-ok", f"lever CT: {_STATE['pf_checks']} prefill memo hits matched stock: "
                                f"prefill memo on")
            return fresh
        _STATE["pf_skips"] += 1
        return hit[1]
    fresh = make_lens()
    pack(indices, lengths, table, fresh)
    st["pf_tail"] = (key, fresh)
    return fresh


def _sig(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def layer(attn) -> None:
    """Lifted attention forward, after the prepare arm: a layer with an indexer or compressor makes the next pack
    fresh (the backend wrapper bumps too; either suffices)."""
    st = _ON.get()
    if st is not None and (getattr(attn, "indexer", None) is not None or getattr(attn, "compressor", None) is not None):
        st["gen"] += 1


def forward(fn, forward_batch, *args, **kwargs):
    """Run our model forward with lever CT's wrappers active (both act on decode / verify forwards only)."""
    mode = getattr(forward_batch, "forward_mode", None)
    dec = bool(mode is not None and (mode.is_decode() or mode.is_target_verify()))
    token = _ON.set({"dec": dec, "gen": 0, "last": None})
    try:
        return fn(*args, **kwargs)
    finally:
        _ON.reset(token)
