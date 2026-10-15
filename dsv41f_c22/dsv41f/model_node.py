"""DeepSeek-V4.1-Flash `model` node for the cacheon/dsv41-tp2-upstream fork (nightly 684fea8a): the fork's own
DeepseekV4Model.forward and layer loop (fork_loop.py, lifted verbatim) on private layer replicas.

Why `model`: the layer loop calls DeepseekV4DecoderLayer.forward_hc_pre_from_prev, never the layers' `forward`, and the
mHC mixing and every layer's all-reduces live in that loop. A replica is a shallow module copy whose `_modules` dict
is its own, so a lever can swap one child (the MoE, the attention) on the replica while the served model -- the node
audit's stock reference -- is never modified. f0 swaps nothing: every replica runs stock's method on stock's children.
f1 adds lever EG (eg.py): the Engram host-table gather on the eg_gather unit, issued early.
f2 adds lever FMT (fmt.py): the MoE replicas pin the routed expert op's prefill tactic by row band.
f4 adds lever SV (sv.py): the fork's E5 prune with a wider budget / share vote (PRECISION change).
f6 adds lever NQ (nq.py): the routed NVFP4 input quantize on a side stream (bitwise); f5's lever L2 window B
(WO-A + wo_b warmed into L2 during the attention) measured slower at 1-2 requests on the fork and is gone.
f8 adds lever RE5 (route_e5.py): top-k + E5 (SV parameters) as one kernel from ROUTE_E5_MIN_ROWS rows,
lever FMT's decode / verify bands and lever DT (dt.py): small-row dense MXFP8 linears at a measured tactic.
f10 adds lever L2PF (l2pf.py): L2 prefetch hints for WO-A / wo_b during the attention (hints only).
f15 adds lever RR (rr.py): RE5 writes FlashInfer's routing record; the routed MoE runs the prepared body.
f20 adds lever HP (high-priority mHC stats stream) and lever SD (MoE side work after the router).
f21 adds lever SK (sk.py): the mHC stats reduce + sinkhorn on a bit-identical CUDA unit.
f47 adds lever CT (ct.py): the persistent trtllm-gen counter buffer + no repeated sparse-tail packs.
fg adds lever FG (fg.py + fg_layer.py): eager-EXTEND FFN blocks replayed from CUDA graphs per row bucket (gate GEMM + all-reduce eager).
fgp adds lever FGP (fgp.py): FG's bucket admission + warmups + captures once at the first eligible eager EXTEND forward (synthetic contents), first real use still checked bitwise.
qn adds lever QN (qn.py + qn_unit.cu): the q-path RMSNorm + wq_b's MXFP8 quantize as one kernel, stock's bytes per row count.
qr adds lever QR (qr.py + qr_unit.cu): the FP8 query rope (fused_q_norm_rope, no norm) with the hardware e4m3 conversion, stock's bytes.
hq adds lever HQ (hq.py + hq_unit.cu + hq_layer.py): the attention combine + RMSNorm + wqkv_a's MXFP8 quantize above 128 rows as one kernel, stock's bytes.
ix adds lever IX (ix.py + ix_dg.py): eager-EXTEND index layers' layer-invariant host work once per forward.
rt adds lever RT (rt.py): eager-EXTEND router launch / gate GEMM / WO-A strided bmm without the wrapper Python.
mo adds lever MO (mo.py): eager-EXTEND routed MoE op / mega-mHC posts on stock's launches, no wrapper Python.
ar adds lever AR (ar.py): eager EXTEND all-reduces issue stock's chosen collective directly.
fl adds lever FL (fl.py): eager-extend MXFP8 linears launch stock's kernels directly (plans learned from stock).
ro adds lever RO (ro.py + ro_unit.cu): the attention output's inverse RoPE (fork_attn) as one kernel with 16-byte loads, stock's bytes.
ag adds lever AG (ag.py): eager-EXTEND post-attention segments (RO, WO-A, wo_b) replayed from CUDA graphs per row bucket, wo_b's all-reduce eager.
aq adds lever AQ (ag.py): eager-EXTEND q paths (wqkv_a, QN, wq_b, QR) replayed from CUDA graphs per AG bucket, KV store / indexer eager.
ago adds lever AGO (ag.py): AG / AQ's per-forward gate, zero-copy q graphs on lever FG's static buffer, and every AG bucket pre-captured at the first eligible eager EXTEND forward.
fgb adds lever FGB (fg.py): FG's graph 1 also holds the router gate GEMM (cuBLAS sweep per bucket) and, after an FG layer, the attention post's mega boundary.
p1 adds lever P1 (sk.py partial_shape + RE5 shapes): mHC stats partial at 49..127 rows: 1 slice(s) a CTA / 2 warps / 1 stages; RE5 shapes ((8, 256, 0), (48, 256, 4), (127, 256, 16), (256, 256, 4)).
c21eg adds lever EGQ (eg.py): decode Engram gathers narrow + the later table's forked late, an Engram wkv DT band, DSPARK aux means on a side stream.
"""
import copy
import sys
import types

import torch

import dsv41f.ix as ix  # lever IX (own line: other kits edit the next one)
from dsv41f import ar, ct, dt, eg, fmt, fork_attn, fork_loop, l2pf, levers, mo, moe_fwd, nq, route_e5, rr, sk
from dsv41f import fl  # lever FL
import dsv41f.qn as qn  # lever QN
import dsv41f.qr as qr  # lever QR
import dsv41f.hq as hq  # lever HQ
import dsv41f.rt as rt   # lever RT: eager-EXTEND router / gate / WO-A host fast paths


import dsv41f.mf as mf  # lever MF
import dsv41f.fg as fg   # lever FG: eager-EXTEND FFN-block CUDA graphs
import dsv41f.fgp as fgp   # lever FGP: FG's bucket captures at the first eligible forward
import dsv41f.ro as ro  # lever RO
import dsv41f.ag as ag   # lever AG: eager-EXTEND post-attention CUDA graphs


class Prepared:
    """The served model and its private layer replicas, aligned with model.layers indices."""

    def __init__(self, model, layers, streams=(None, None)):
        self.model = model
        self.layers = layers
        self.streams = streams                    # lever HP: (stock's stats stream, the top-priority one)
        self.pm_stream = None                     # lever PM: the top-priority main stream


def _hp_stream(model):
    """Lever HP: one stats stream for all layer replicas, at the highest priority, on stock's device (None when
    stock runs the stats inline)."""
    stock = getattr(model.layers[model.start_layer], "hc_stats_stream", None) if len(model.layers) else None
    if not levers.HP or stock is None:
        return stock
    _, high = torch.cuda.Stream.priority_range()
    return torch.cuda.Stream(device=stock.device, priority=high)


def _replica(layer, stats_stream=None):
    replica = copy.copy(layer)
    if stats_stream is not None:
        replica.hc_stats_stream = stats_stream    # lever HP: the replica's own (high-priority) stats stream
    replica._modules = dict(layer._modules)
    mlp = layer._modules.get("mlp")
    if mlp is not None and hasattr(mlp, "forward_normal_dual_stream"):
        moe = copy.copy(mlp)
        moe._modules = dict(mlp._modules)
        changed = fmt.armed() and fmt.bind(moe)  # lever FMT: the replica's forward pins the prefill tactic
        nq_on = nq.bind_experts(moe)             # lever NQ: an experts replica taking the side-stream FP4 input
        re5_on = route_e5.bind(moe)              # lever RE5: fused top-k + E5 (rank-agreed self-test first)
        if levers.SV or nq_on or re5_on:          # levers SV / NQ / RE5: the lifted forwards
            moe.forward_normal_dual_stream = types.MethodType(moe_fwd.forward_normal_dual_stream, moe)
            moe.forward_normal = types.MethodType(moe_fwd.forward_normal, moe)
            moe.dsv4f_sv = bool(levers.SV)
            changed = True
        if changed:
            moe.dsv4f_stats_stream = stats_stream     # lever SD: joins the stats stream after the router
            moe.dsv4f_dt_copy = True              # private already: lever DT swaps its children in place
            replica._modules["mlp"] = moe
    attn = layer._modules.get("self_attn")
    if levers.L2PF and attn is not None and type(attn).__name__ == "MQALayer":
        attn = copy.copy(attn)
        attn._modules = dict(layer._modules["self_attn"]._modules)
        attn.forward = types.MethodType(fork_attn.attn_forward, attn)   # lever L2PF: the lifted forward
        attn.dsv4f_dt_copy = True                 # private already: lever DT swaps its linears in place
        replica._modules["self_attn"] = attn
    replica.dsv4f_dt = dt.bind(replica)          # lever DT: the replica's dense MXFP8 linears at measured tactics
    replica.dsv4f_fl = fl.bind(replica, layer)   # lever FL: eager-extend fast launch of those linears (after DT)
    return replica


def prepare(model):
    fmt_state = fmt.prepare()                     # lever FMT: the choose_one wrapper, once, outside any capture
    rr_state = rr.install()                       # lever RR: the routed-call wrapper + tactic hook (outermost)
    sk_state = sk.install()                       # lever SK: the mHC stats wrapper (active inside forward only)
    fl_state = fl.install()                       # lever FL: pinned FlashInfer / SGLang internals (before binding)
    hp = _hp_stream(model)
    stock_stream = getattr(model.layers[model.start_layer], "hc_stats_stream", None) if len(model.layers) else None
    layers = [_replica(layer, stock_stream) for layer in model.layers]
    print(f"CACHEON_DSV4F model node (dsv41f-c22): {len(layers)} layer replicas, fork loop "
          f"(start {model.start_layer}, end {model.end_layer})", file=sys.stderr, flush=True)
    # lever EG: unit + a byte-for-byte self-test against stock's gather on each served table, once, here
    print(f"CACHEON_DSV4F lever EG: {eg.prepare(model)}", file=sys.stderr, flush=True)
    moe = sum(1 for r, s in zip(layers, model.layers) if r._modules.get("mlp") is not s._modules.get("mlp"))
    print(f"CACHEON_DSV4F lever FMT: {fmt_state}; MoE replicas {moe}", file=sys.stderr, flush=True)
    nq_on = sum(bool(getattr(r._modules.get("mlp"), "dsv4f_nq", False)) for r in layers)
    print(f"CACHEON_DSV4F lever NQ: side-stream routed NVFP4 input quantize on {nq_on} MoE replicas",
          file=sys.stderr, flush=True)
    re5 = sum(bool(getattr(r._modules.get("mlp"), "dsv4f_re5", False)) for r in layers)
    print(f"CACHEON_DSV4F lever RE5: fused top-k + E5 on {re5} MoE replicas from {levers.ROUTE_E5_MIN_ROWS} rows "
          f"(tiles {sorted(levers.E5_TREES)})", file=sys.stderr, flush=True)
    dt_on = sum(r.dsv4f_dt for r in layers)
    print(f"CACHEON_DSV4F lever DT: {dt_on} dense MXFP8 linears on {sum(bool(r.dsv4f_dt) for r in layers)} layer "
          f"replicas, shapes {sorted(levers.DT_TACTICS)}, split-K {'allowed (PRECISION)' if levers.DT_SPLIT_K else 'off'}",
          file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever FL: {fl_state}; {sum(r.dsv4f_fl for r in layers)} MXFP8 linears on "
          f"{sum(bool(r.dsv4f_fl) for r in layers)} layer replicas", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever L2PF: {l2pf.prepare(layers, fork_attn.attn_forward)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever QN: {qn.install(model, layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever QR: {qr.install(model, layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever HQ: {hq.install(model, layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever RO: {ro.install(model, layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever CT: {ct.install(layers, fork_attn.attn_forward)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever MF: {mf.install()}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever IX: {ix.install()}", file=sys.stderr, flush=True)   # after CT (A/B timer wraps CT's)
    print(f"CACHEON_DSV4F lever AR: {ar.install()}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever RR: {rr_state}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever SK: {sk_state}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever RT: {rt.install(layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever HP: stats stream {'priority ' + str(hp.priority) + ' at <= ' + str(levers.HP_MAX_ROWS) + ' rows' if hp is not None else 'none'}; "
          f"lever SD: side work after the router from {levers.SD_MIN_ROWS} rows", file=sys.stderr, flush=True)
    sv_on = sum(bool(getattr(r._modules.get("mlp"), "dsv4f_sv", False)) for r in layers)
    print(f"CACHEON_DSV4F lever SV: E5 budget {levers.SV_BUDGET}, share vote <= {levers.SV_MAX_COUNT} rows from "
          f"{levers.SV_MIN_ROWS} rows, on {sv_on} MoE replicas (PRECISION)", file=sys.stderr, flush=True)
    # lever MO: last, so its choose_one hook is the outermost and the MoE replicas exist
    print(f"CACHEON_DSV4F lever MO: {mo.install(layers, model.layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever FG: {fg.install(model, layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever FGP: {fgp.install(layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever AG: {ag.install(model, layers)}", file=sys.stderr, flush=True)
    prepared = Prepared(model, layers, (stock_stream, hp))
    if levers.PM and hp is not None:
        _, high = torch.cuda.Stream.priority_range()
        prepared.pm_stream = torch.cuda.Stream(device=hp.device, priority=high)
    print(f"CACHEON_DSV4F lever PM: {'main chain at priority ' + str(prepared.pm_stream.priority) + ' above ' + str(levers.HP_MAX_ROWS) + ' decode / verify rows' if prepared.pm_stream is not None else 'off'}",
          file=sys.stderr, flush=True)
    return prepared


def _pick_stats_stream(prepared, args, kwargs):
    """Lever HP: the top-priority stats stream for forwards of <= HP_MAX_ROWS tokens, stock's otherwise."""
    stock_stream, hp = prepared.streams
    if hp is None or stock_stream is None:
        return
    pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
    rows = int(pos.shape[0]) if isinstance(pos, torch.Tensor) else 1 << 30
    want = hp if rows <= levers.HP_MAX_ROWS else stock_stream
    for r in prepared.layers:
        r.hc_stats_stream = want
        mlp = r._modules.get("mlp")
        if mlp is not None and getattr(mlp, "dsv4f_dt_copy", False):
            mlp.dsv4f_stats_stream = want


def _pm_stream(prepared, args, kwargs):
    """Lever PM: the top-priority main stream for decode / verify forwards above HP_MAX_ROWS tokens, else None."""
    if prepared.pm_stream is None:
        return None
    fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
    mode = getattr(fb, "forward_mode", None)
    if mode is None or not (mode.is_decode() or mode.is_target_verify()) or not isinstance(pos, torch.Tensor):
        return None
    return prepared.pm_stream if int(pos.shape[0]) > levers.HP_MAX_ROWS else None


def _run(prepared, args, kwargs):
    """Lever HQ: the one-kernel attention combine + norm + quantize context around the rest of the node forward."""
    fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    return hq.forward(_run_hq, fb, prepared, args, kwargs)


def _run_hq(prepared, args, kwargs):
    """Lever QN: the one-kernel q-path norm + quantize context around the rest of the node forward."""
    fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    return qn.forward(_run_qn, fb, prepared, args, kwargs)


def _run_qn(prepared, args, kwargs):
    """Lever FL: the fast MXFP8-linear launch context around the rest of the node forward."""
    fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    return fl.forward(_run_inner, fb, prepared, args, kwargs)


def _run_inner(prepared, args, kwargs):
    """Lever FG: the FFN-block graph context around the rest of the node forward (eager EXTEND forwards only)."""
    fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    return ag.forward(fg.forward, fb, _run_inner_fg, fb, prepared, args, kwargs)   # lever AG around lever FG


def _run_inner_fg(prepared, args, kwargs):
    fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    out = ar.forward(ct.forward, fb, sk.forward, fb, mo.forward, fork_loop.model_forward, fb, prepared.model,
                     prepared.layers, *args, **kwargs)  # levers AR + CT + SK + MO
    l2pf.join_all()                               # lever L2PF: the prefetch stream ends joined (graph capture)
    return out


def _forward(prepared, *args, **kwargs):
    _pick_stats_stream(prepared, args, kwargs)    # lever HP
    main = _pm_stream(prepared, args, kwargs)     # lever PM
    if main is None:
        return rt.forward(_run, prepared, args, kwargs)   # lever RT: eager EXTEND fast paths
    caller = torch.cuda.current_stream()
    main.wait_stream(caller)
    with torch.cuda.stream(main):
        out = _run(prepared, args, kwargs)
    caller.wait_stream(main)
    if isinstance(out, torch.Tensor) and not torch.cuda.is_current_stream_capturing():
        out.record_stream(caller)                 # eager: the caller's stream reads it after the join
    return out


def forward(prepared, *args, **kwargs):
    """Lever IX: the node forward inside the index-layer memo scope (eager EXTEND / MIXED forwards only)."""
    return ix.forward(_forward, prepared, *args, **kwargs)
