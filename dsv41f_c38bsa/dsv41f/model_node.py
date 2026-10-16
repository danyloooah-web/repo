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
re5nc adds lever RE5NC (route_e5.py): no RE5 cluster, above 127 rows mode 3: RE5 top-k + E5 on one plain grid of 8 rows a CTA, E5 in the last CTA, launched without PDL.
c23stp adds lever STP (stp.py): the shared expert's tail on a top-priority stream (mode tail), PM one level below, a <= 60-CTA shared-down DT band.
rtr adds lever RTR (rtr.py + rtr_unit.cu): the decode / verify router gate GEMM as one kernel, stock's cuBLAS split-K + splitKreduce bits, S learned per row count.
sch adds lever SCH (model_node.py + qn_unit.cu + l2pf.py, graph captures only): IXP index-layer side streams at top priority, QN's early PDL trigger, L2PF's lazy join at the forward end.
skw adds lever SKW (skw.py + skw_unit.cu): lever SK's mHC reduce + sinkhorn on a capped grid in CUDA-graph captures (the stats stream holds <= 4 SMs), sk_unit's bytes.
moet adds lever MOET (moet.py): decode / verify routed NVFP4 MoE calls of a row band at a measured trtllm-gen tactic (bit-identical; mode + rows + shape only).
mhc adds lever MHC (mhc.py + mhc_unit.cu): the mHC stats partial of small-path sublayer calls on a short wide tensor-core kernel (exact: stock's tf32x3 bits).
gq adds lever GQ (gq.py + gq_unit.cu): lever STP's split hands the shared gate_up a small-grid MXFP8 quantize of its input (stock's bytes; exact).
aux adds lever AUX (aux.py + eg_gather.cu aux_mean): the DSPARK aux means on a small fixed grid, stock's bytes (exact).
fin adds lever FIN (fin.py, one JIT module): the replicas' fused attention / MoE posts on our copy of stock's finalize + all-reduce + mHC kernel (pre-poll epilogue loads, no dead store, in-kernel identity routing; exact).
r5p adds lever R5P (r5p.py + r5p_unit.cu): RE5 at 49..127 decode / verify rows in CUDA graphs on one row a warp, E5 in the routing warps, RE5's bytes.
"""
import copy
import sys
import types

import torch

import dsv41f.ix as ix  # lever IX (own line: other kits edit the next one)
from dsv41f import ar, ct, dt, eg, fmt, fork_attn, fork_loop, l2pf, levers, mo, moe_fwd, nq, route_e5, rr, sk
from dsv41f import fl  # lever FL
import dsv41f.stp as stp   # lever STP: the shared expert's tail on a top-priority stream
import dsv41f.gq as gq   # lever GQ: the STP split's gate_up input quantize on a small grid
import dsv41f.rtr as rtr   # lever RTR: the router gate GEMM as one exact kernel
import dsv41f.pwn as pwn   # lever PWN: dense MXFP8 GEMMs pull their weight before the PDL wait (native entry-trigger quantize)
import dsv41f.r5p as r5p   # lever R5P: RE5 at 49..127 rows, one row a warp, E5 in the warps
import dsv41f.moet as moet   # lever MOET: decode / verify routed-MoE tactic band
import dsv41f.skw as skw   # lever SKW: the mHC sinkhorn on a capped grid in graphs
import dsv41f.mhc as mhc   # lever MHC: the mHC stats partial on the unit
import dsv41f.aux as auxk   # lever AUX: the DSPARK aux means on a small grid
import dsv41f.fin as fin   # lever FIN: the fused posts on our finalize + all-reduce + mHC kernel
import dsv41f.qn as qn  # lever QN
import dsv41f.qr as qr  # lever QR
import dsv41f.hq as hq  # lever HQ
import dsv41f.rt as rt   # lever RT: eager-EXTEND router / gate / WO-A host fast paths


import dsv41f.mf as mf  # lever MF
import dsv41f.fg as fg   # lever FG: eager-EXTEND FFN-block CUDA graphs
import dsv41f.fgp as fgp   # lever FGP: FG's bucket captures at the first eligible forward
import dsv41f.ro as ro  # lever RO
import dsv41f.bsa as bsa  # lever BSA (exact, bitwise attention)
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
        changed = moet.bind(moe) or changed   # lever MOET: decode / verify tactic band (wraps the forward)
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
    replica.dsv4f_pwn = pwn.bind(replica)        # lever PWN: those linears' GEMM pulls its weight before the wait
    replica.dsv4f_fl = fl.bind(replica, layer)   # lever FL: eager-extend fast launch of those linears (after DT)
    return replica


def prepare(model):
    fmt_state = fmt.prepare()                     # lever FMT: the choose_one wrapper, once, outside any capture
    moet_state = moet.prepare()                   # lever MOET: its choose_one wrapper (inside RR's hook)
    rr_state = rr.install()                       # lever RR: the routed-call wrapper + tactic hook (outermost)
    sk_state = sk.install()                       # lever SK: the mHC stats wrapper (active inside forward only)
    fl_state = fl.install()                       # lever FL: pinned FlashInfer / SGLang internals (before binding)
    hp = _hp_stream(model)
    stock_stream = getattr(model.layers[model.start_layer], "hc_stats_stream", None) if len(model.layers) else None
    layers = [_replica(layer, stock_stream) for layer in model.layers]
    print(f"CACHEON_DSV4F model node (dsv41f-c36pwn): {len(layers)} layer replicas, fork loop "
          f"(start {model.start_layer}, end {model.end_layer})", file=sys.stderr, flush=True)
    # lever EG: unit + a byte-for-byte self-test against stock's gather on each served table, once, here
    print(f"CACHEON_DSV4F lever EG: {eg.prepare(model)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever AUX: {auxk.prepare(model)}", file=sys.stderr, flush=True)
    moe = sum(1 for r, s in zip(layers, model.layers) if r._modules.get("mlp") is not s._modules.get("mlp"))
    print(f"CACHEON_DSV4F lever FMT: {fmt_state}; MoE replicas {moe}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever MOET: {moet_state}; replicas {sum(bool(getattr(r._modules.get('mlp'), 'dsv4f_moet', False)) for r in layers)}", file=sys.stderr, flush=True)
    nq_on = sum(bool(getattr(r._modules.get("mlp"), "dsv4f_nq", False)) for r in layers)
    print(f"CACHEON_DSV4F lever NQ: side-stream routed NVFP4 input quantize on {nq_on} MoE replicas",
          file=sys.stderr, flush=True)
    re5 = sum(bool(getattr(r._modules.get("mlp"), "dsv4f_re5", False)) for r in layers)
    print(f"CACHEON_DSV4F lever RE5: fused top-k + E5 on {re5} MoE replicas from {levers.ROUTE_E5_MIN_ROWS} rows "
          f"to {route_e5.fused_max()} rows, above: lever RE5NC mode {route_e5.nc_mode()} (no cluster) "
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
    print(f"CACHEON_DSV4F lever BSA: {bsa.install(model, layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever CT: {ct.install(layers, fork_attn.attn_forward)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever MF: {mf.install()}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever IX: {ix.install()}", file=sys.stderr, flush=True)   # after CT (A/B timer wraps CT's)
    print(f"CACHEON_DSV4F lever AR: {ar.install()}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever FIN: {fin.install()}", file=sys.stderr, flush=True)
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
    print(f"CACHEON_DSV4F lever STP: {stp.prepare(layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever GQ: {gq.prepare(layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever RTR: {rtr.prepare(layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever PWN: {pwn.describe(layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever R5P: {r5p.prepare(layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever SKW: {skw.prepare((stock_stream, hp), layers)}", file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever MHC: {mhc.prepare(layers)}", file=sys.stderr, flush=True)
    prepared = Prepared(model, layers, (stock_stream, hp))
    if levers.PM and hp is not None:
        low, high = torch.cuda.Stream.priority_range()
        # lever STP: one level below the tail stream (the top) when STP is on
        prepared.pm_stream = torch.cuda.Stream(device=hp.device, priority=stp.pm_priority(low, high))
    print(f"CACHEON_DSV4F lever PM: {'main chain at priority ' + str(prepared.pm_stream.priority) + ' above ' + str(levers.HP_MAX_ROWS) + ' decode / verify rows' if prepared.pm_stream is not None else 'off'}",
          file=sys.stderr, flush=True)
    print(f"CACHEON_DSV4F lever SCH: {_sch_prepare(model, layers, prepared)}", file=sys.stderr, flush=True)
    return prepared


# lever SCH (gen_sch.py): scheduling only, CUDA-graph captures only (stock / SRC behaviour for every eager forward)
SCH_IXP = True                    # index layers' compressor + indexer side streams at top priority (decode / verify)
SCH_IXP_PRIORITY = 'greatest'       # "greatest" = the device's greatest stream priority, or an explicit priority
SCH_IXP_MIN_ROWS = 1
SCH_IXP_MAX_ROWS = 384
SCH_QN = 'mixed'                  # QN launch_dependents right after griddepcontrol.wait: "off" | "early" | "mixed"
_SCH: dict = {"armed": [], "on": False, "streams": (), "detail": "not prepared", "mod": None, "qn": False,
              "qn_on": False}


def _sch_unit():
    """The bundle unit qn_unit when it carries lever SCH's host functions, else None."""
    try:
        import qn_unit as mod
    except ImportError:
        return None
    ok = (hasattr(mod, "set_trigger") and hasattr(mod, "trigger_state") and hasattr(mod, "priority_range")
          and hasattr(mod, "stream_create") and hasattr(mod, "stream_priority") and hasattr(mod, "stream_flags")
          and hasattr(mod, "set_phase"))
    return mod if ok else None


def _sch_known_streams(prepared, layers) -> dict:
    """cuda_stream handle -> name of every stream this process's model / levers capture work on (log + collision check)."""
    found = {}

    def add(name, s):
        if isinstance(s, torch.cuda.Stream):
            found.setdefault(int(s.cuda_stream), name)

    add("current", torch.cuda.current_stream())
    add("pm", prepared.pm_stream)
    for s in prepared.streams:
        add("stats", s)
    for replica in layers:
        for m in replica.modules():
            add("hc_stats_stream", getattr(m, "hc_stats_stream", None))
            add("alt_stream", getattr(m, "alt_stream", None))
            add("routed_quant_stream", getattr(m, "routed_quant_stream", None))
            add("dsv4f_stats_stream", getattr(m, "dsv4f_stats_stream", None))
            for name, seq in (("alt_streams", getattr(m, "alt_streams", None)),
                              ("alt_streams_indexer", getattr(m, "alt_streams_indexer", None))):
                if isinstance(seq, (list, tuple)):
                    for j, s in enumerate(seq):
                        add(f"{name}[{j}]", s)
    for name, table in (("l2pf", getattr(l2pf, "_STREAMS", None)), ("stp tail", getattr(stp, "_TAIL", None)),
                        ("eg side", getattr(eg, "_SIDE", None)), ("eg aux", getattr(eg, "_AUX_SIDE", None)),
                        ("nq side", getattr(nq, "_SIDE", None))):
        for s in (table or {}).values():
            add(name, s)
    return found


def _sch_prepare(model, layers, prepared) -> str:
    """Lever SCH: arm IXP on the index-layer attention replicas (two raw top-priority streams) and set QN's trigger
    policy in the unit. Before any capture; any failure leaves SRC's behaviour (fail closed)."""
    mod = _sch_unit()
    if mod is None:
        _SCH["detail"] = "off (unit qn_unit without the SCH functions)"
        return _SCH["detail"]
    parts = []
    armed = []
    if SCH_IXP:
        try:
            dev = torch.device("cuda", torch.cuda.current_device())
            least, greatest = (int(v) for v in mod.priority_range(dev.index))
            prio = greatest if SCH_IXP_PRIORITY == "greatest" else int(SCH_IXP_PRIORITY)
            if not greatest <= prio <= least:
                raise RuntimeError(f"priority {prio} outside the device range {least}..{greatest}")
            cands = []
            for r, s in zip(layers, model.layers):
                attn = r._modules.get("self_attn")
                served = s._modules.get("self_attn")
                if attn is None or attn is served or type(attn).__name__ != "MQALayer":
                    continue
                if getattr(attn, "indexer", None) is None or getattr(attn, "compress_ratio", None) not in (1, 2):
                    continue
                alt = getattr(attn, "alt_streams", None)
                if not (isinstance(alt, list) and len(alt) == 3 and all(isinstance(x, torch.cuda.Stream) for x in alt)
                        and len({int(x.cuda_stream) for x in alt}) == 3 and alt is getattr(served, "alt_streams", None)):
                    raise RuntimeError(f"layer {getattr(attn, 'layer_id', '?')}: alt_streams is not stock's 3-stream list")
                if getattr(attn, "dsv4f_sch_ixp", None) is not None:
                    raise RuntimeError("an attention replica is armed twice")
                cands.append(attn)
            if not cands:
                raise RuntimeError("no private index-layer attention replica")
            known = _sch_known_streams(prepared, layers)
            comp = torch.cuda.ExternalStream(int(mod.stream_create(dev.index, prio)), device=dev)
            src = torch.cuda.ExternalStream(int(mod.stream_create(dev.index, prio)), device=dev)
            hits = [known[int(s.cuda_stream)] for s in (comp, src) if int(s.cuda_stream) in known]
            if hits or comp.cuda_stream == src.cuda_stream:
                raise RuntimeError(f"raw stream collides with {hits or 'itself'}")
            got = [int(mod.stream_priority(int(s.cuda_stream))) for s in (comp, src)]
            flags = [int(mod.stream_flags(int(s.cuda_stream))) for s in (comp, src)]
            if got != [prio, prio] or flags != [1, 1]:
                raise RuntimeError(f"raw streams priority {got} / flags {flags} (want {prio} / non-blocking)")
            for attn in cands:
                stock = attn.alt_streams
                attn.dsv4f_sch_ixp = (stock, [stock[0], comp, src])   # the replica's own lists; the served list untouched
            _SCH["streams"] = (comp, src)
            armed = cands
            parts.append(f"IXP armed on {len(cands)} index-layer replicas (layers "
                         f"{[getattr(a, 'layer_id', -1) for a in cands]}): compressor + indexer side streams = 2 raw "
                         f"non-blocking streams at priority {prio} (device range {least}..{greatest}; outside torch's "
                         f"pools, distinct from {len(known)} known streams) in decode / verify graph captures of "
                         f"{SCH_IXP_MIN_ROWS}..{SCH_IXP_MAX_ROWS} rows")
        except Exception as err:  # noqa: BLE001 - any surprise keeps stock's streams
            for r in layers:
                attn = r._modules.get("self_attn")
                if attn is not None and getattr(attn, "dsv4f_sch_ixp", None) is not None:
                    attn.dsv4f_sch_ixp = None
            armed = []
            parts.append(f"IXP off ({type(err).__name__}: {err})")
    else:
        parts.append("IXP off (SCH_IXP)")
    _SCH["armed"] = armed
    _SCH["mod"] = mod
    _SCH["qn"] = False
    _SCH["qn_on"] = False
    try:
        mod.set_phase(0)                  # no forward under capture yet: _sch_select sets it per node forward
        if SCH_QN == "off":
            mod.set_trigger(0, [])
        elif SCH_QN in ("early", "mixed"):
            late = []
            if SCH_QN == "mixed":
                for attn in armed:
                    w = getattr(getattr(attn, "q_norm", None), "weight", None)
                    if not isinstance(w, torch.Tensor):
                        raise RuntimeError(f"layer {getattr(attn, 'layer_id', '?')}: no q_norm weight")
                    late.append(int(w.data_ptr()))
            mod.set_trigger(1, late)
        else:
            raise RuntimeError(f"SCH_QN {SCH_QN!r}")
        early, n_late, phase = (int(v) for v in mod.trigger_state())
        if phase != 0:
            raise RuntimeError(f"unit phase {phase} after set_phase(0)")
        _SCH["qn"] = bool(early)
        parts.append(f"QN trigger {'right after griddepcontrol.wait inside decode / verify graph captures (EXTEND / AQ graphs: after the stores)' if early else 'after the stores (SRC)'}"
                     + (f", {n_late} layers keep it after the stores (IXP's index layers)" if early else ""))
    except Exception as err:  # noqa: BLE001
        _SCH["qn"] = False
        try:
            mod.set_trigger(0, [])
        except Exception:  # noqa: BLE001
            pass
        parts.append(f"QN trigger after the stores (SRC; {type(err).__name__}: {err})")
    join_end = getattr(l2pf, "L2PF_JOIN_END", False)
    _SCH["detail"] = "; ".join(parts) + f"; L2PF join {'at the forward end inside captures' if join_end else 'per layer (SRC)'}"
    return _SCH["detail"]


def _sch_select(args, kwargs) -> None:
    """Lever SCH, at every node forward's entry: for decode / verify forwards recorded into a CUDA-graph capture the
    armed index-layer replicas take the top-priority side streams (IXP) and the unit's phase allows QN's early trigger;
    every other forward (eager decode / verify of the audit, EXTEND with lever AQ's own q-path graphs, warmups) gets
    stock's alt_streams lists and QN's trigger after the stores (SRC)."""
    armed = _SCH["armed"]
    if not armed and not _SCH["qn"]:
        return
    dv, rows = False, 0
    if torch.cuda.is_current_stream_capturing():
        fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
        pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
        mode = getattr(fb, "forward_mode", None)
        if mode is not None and (mode.is_decode() or mode.is_target_verify()) and isinstance(pos, torch.Tensor):
            dv, rows = True, int(pos.shape[0])
    if _SCH["qn"] and dv != _SCH["qn_on"]:
        try:
            _SCH["mod"].set_phase(1 if dv else 0)
            _SCH["qn_on"] = dv
        except Exception as err:  # noqa: BLE001 - fail closed: QN's trigger after the stores from now on
            _SCH["qn"] = False
            try:
                _SCH["mod"].set_trigger(0, [])
            except Exception:  # noqa: BLE001
                pass
            print(f"CACHEON_DSV4F lever SCH: QN phase failed ({type(err).__name__}: {err}): trigger after the stores",
                  file=sys.stderr, flush=True)
    want = dv and SCH_IXP_MIN_ROWS <= rows <= SCH_IXP_MAX_ROWS
    if not armed or want == _SCH["on"]:
        return
    for attn in armed:
        stock, ixp = attn.dsv4f_sch_ixp
        attn.alt_streams = ixp if want else stock
    _SCH["on"] = want
    if want and levers.first("sch-ixp-used"):
        pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
        levers.announce("sch-ixp-used", f"lever SCH IXP: top-priority index-layer side streams first captured at "
                        f"{int(pos.shape[0])} rows ({len(armed)} layers)")


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
    _sch_select(args, kwargs)                     # lever SCH: IXP's streams in decode / verify captures
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
