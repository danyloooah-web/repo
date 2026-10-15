"""Lever L2PF on the fork (levers.L2PF): during a decode / verify layer's attention, a short-lived side-stream grid
(CUDA unit l2pf) issues L2 prefetch hints for the bytes the layer streams right after it -- the BF16 WO-A weight
(33.5 MB a TP2 rank), wo_b's MXFP8 weight + swizzled scales (21.6 MB) and, with L2PF_ROUTER, the router weight
(3.9 MB) -- instead of window B's resident fill blocks (fork k=1 profile, f5 vs f4: WO-A -90 us and wo_b -67 us a cycle,
but the fill's 28 resident blocks cost the attention +46 us: B300 runs one kernel's CTAs per SM at a time).

Placement: forked right after the attention's prepare arm (q path, KV store, indexer / compressor joined) -- the same
point as window B -- and never joined before WO-A: the prefetch grid computes nothing anyone waits for, so its stream is
joined lazily (at the next layer's fork, and by join_all() at the end of the model forward, which a CUDA-graph capture
needs).

Exactness: hints only (no stores, nothing returned to a register that is used); every consumer reads the same bytes.
"""
import torch

from dsv41f import levers

_UNIT: list = []
_STREAMS: dict = {}       # device index -> the prefetch stream
_PEND: dict = {}          # device index -> True while the stream has unjoined work
_STATE = {"armed": 0, "detail": "not prepared", "failed": False}


def unit():
    if not _UNIT:
        mod = None
        try:
            import l2pf_unit as mod
        except ImportError as err:
            levers.announce("l2pf-missing", f"bundle CUDA unit l2pf_unit not importable ({err}): lever L2PF off")
        _UNIT.append(mod)
    return _UNIT[0]


def _bytes(t) -> int:
    return t.numel() * t.element_size()


def _usable(t, device) -> bool:
    return (isinstance(t, torch.Tensor) and t.is_cuda and t.device == device and t.is_contiguous()
            and _bytes(t) > 0 and _bytes(t) % 16 == 0 and t.data_ptr() % 16 == 0)


def regions(attn, mlp):
    """[WO-A weight, wo_b weight + swizzled scales when wo_b is the FlashInfer MXFP8 linear, router weight] or None."""
    wo_a = getattr(getattr(attn, "wo_a", None), "weight", None)
    if not (isinstance(wo_a, torch.Tensor) and wo_a.dtype == torch.bfloat16 and _usable(wo_a, wo_a.device)):
        return None
    out = [wo_a]
    wo_b = getattr(attn, "wo_b", None)
    method = getattr(wo_b, "quant_method", None)
    backend = getattr(method, "mxfp8_dense_backend", None)
    mxfp8 = bool(getattr(method, "use_mxfp8", False) or getattr(wo_b, "block_fp8_mxfp8_ready", False))
    weight = getattr(wo_b, "weight", None)
    scale = getattr(wo_b, "weight_scale_inv_swizzled", None)
    if (mxfp8 and backend is not None and (backend.is_flashinfer_cutedsl() or backend.is_flashinfer_cutlass())
            and isinstance(weight, torch.Tensor) and weight.dtype == torch.float8_e4m3fn
            and _usable(weight, wo_a.device) and _usable(scale, wo_a.device)):
        out += [weight, scale]
    gate = getattr(getattr(mlp, "gate", None), "weight", None) if mlp is not None else None
    if levers.L2PF_ROUTER and _usable(gate, wo_a.device):
        out.append(gate)
    return out


def _launch(views, grid, block, mode, stream) -> int:
    return unit().prefetch([t.data_ptr() for t in views], [_bytes(t) for t in views], int(grid), int(block),
                           int(mode), int(levers.L2PF_CHUNK), stream.cuda_stream)


def prepare(layers, attn_forward) -> str:
    """Arm the attention replicas running attn_forward (model_node.prepare, before any capture)."""
    if not levers.L2PF:
        _STATE["detail"] = "off (levers.L2PF)"
        return _STATE["detail"]
    if unit() is None:
        _STATE["detail"] = "off (unit l2pf_unit missing)"
        return _STATE["detail"]
    armed, first, skipped = 0, None, 0
    for replica in layers:
        attn = replica._modules.get("self_attn")
        bound = getattr(getattr(attn, "forward", None), "__func__", None) is attn_forward
        if not bound or getattr(attn, "layer_id", -1) < levers.L2PF_MIN_LAYER:
            continue
        views = regions(attn, replica._modules.get("mlp"))
        if views is None:
            skipped += 1
            continue
        attn.dsv4f_l2pf = tuple(views)
        armed += 1
        first = first or views
    if first is None:
        _STATE["detail"] = f"off (no armed attention replica; {skipped} without usable regions)"
        return _STATE["detail"]
    device = first[0].device
    try:
        stream = _STREAMS.get(device.index)
        if stream is None:
            stream = _STREAMS[device.index] = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _, grid, block, mode in levers.L2PF_BANDS:
                covered = _launch(first, grid, block, mode, stream)
        torch.cuda.synchronize(device)
        if covered != sum(_bytes(t) for t in first):
            raise RuntimeError(f"covered {covered} of {sum(_bytes(t) for t in first)} bytes")
    except Exception as err:  # noqa: BLE001 - a load-time failure keeps stock's path
        for replica in layers:
            attn = replica._modules.get("self_attn")
            if getattr(attn, "dsv4f_l2pf", None) is not None:
                attn.dsv4f_l2pf = None
        _STATE["detail"] = f"off (self-test: {type(err).__name__}: {err})"
        return _STATE["detail"]
    _STATE["armed"] = armed
    mb = sum(_bytes(t) for t in first) / 1e6
    _STATE["detail"] = (f"armed on {armed} attention replicas (layers >= {levers.L2PF_MIN_LAYER}; {len(first)} regions, "
                        f"{mb:.1f} MB a layer; bands (max rows, grid, block, mode) {levers.L2PF_BANDS}, chunk "
                        f"{levers.L2PF_CHUNK}); {skipped} without usable regions")
    return _STATE["detail"]


def issue(attn, forward_batch, rows: int) -> None:
    """attn_forward, right after the prepare arm: issue this layer's prefetch hints on the side stream."""
    views = getattr(attn, "dsv4f_l2pf", None)
    if views is None or forward_batch is None or _STATE["failed"]:
        return
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return
    for max_rows, grid, block, pmode in levers.L2PF_BANDS:
        if rows <= max_rows:
            break
    else:
        return
    device = views[0].device
    stream = _STREAMS[device.index]
    current = torch.cuda.current_stream(device)
    if _PEND.get(device.index):
        current.wait_stream(stream)          # the previous layer's hints (long done): keep the graph joined
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            _launch(views, grid, block, pmode, stream)
    except RuntimeError as err:
        _STATE["failed"] = True
        levers.announce("l2pf-failed", f"prefetch launch failed ({err}): lever L2PF off from here on")
    _PEND[device.index] = True
    if levers.first("l2pf-used"):
        levers.announce("l2pf-used", f"lever L2PF: prefetch first at {rows} rows (layer {attn.layer_id}, grid {grid} x "
                        f"{block}, mode {pmode}, {sum(_bytes(t) for t in views) / 1e6:.1f} MB)")


def join_all() -> None:
    """End of the model forward: join every prefetch stream with unjoined work (a capture must end joined)."""
    for index in list(_PEND):
        if _PEND.pop(index):
            torch.cuda.current_stream(torch.device("cuda", index)).wait_stream(_STREAMS[index])


def _linear_views(linear, device):
    """[weight, swizzled scales] of a FlashInfer MXFP8 dense linear, or []."""
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scale_inv_swizzled", None)
    if not (_usable(weight, device) and weight.dtype == torch.float8_e4m3fn):
        return []
    return [weight] + ([scale] if _usable(scale, device) else [])


def prepare_next(layers) -> str:
    """Lever L2PF next: each MoE replica gets the next layer's attention-prologue weights (wqkv_a or wq_a + wkv, wq_b)
    to hint at the start of its MoE forward (the router chain reads little from HBM). Returns the log detail."""
    if not (levers.L2PF and levers.L2PF_NEXT) or unit() is None or not _STREAMS:
        return "off"
    armed = 0
    mb = 0.0
    for i in range(len(layers) - 1):
        moe = layers[i]._modules.get("mlp")
        attn = layers[i + 1]._modules.get("self_attn")
        if moe is None or attn is None or not getattr(moe, "dsv4f_dt_copy", False):   # private MoE replicas only
            continue
        device = next(iter(_STREAMS))
        dev = torch.device("cuda", device)
        views = []
        for name in ("wqkv_a", "wq_a", "wkv", "wq_b"):
            sub = attn._modules.get(name)
            if sub is not None:
                views += _linear_views(sub, dev)
        if levers.L2PF_NEXT_SOURCES:
            # ratio-1/2 layers: the compressor's BF16 projections and the indexer's weights (its side chain is exposed)
            for owner, names in (("compressor", ("wkv_gate", "wkv", "wgate")), ("indexer", ("wq_b", "weights_proj", "wk"))):
                mod = attn._modules.get(owner)
                for name in names:
                    sub = mod._modules.get(name) if mod is not None else None
                    if sub is None:
                        continue
                    w = getattr(sub, "weight", None)
                    if _usable(w, dev) and w.dtype == torch.float8_e4m3fn:
                        views += _linear_views(sub, dev)
                    elif _usable(w, dev):
                        views.append(w)
        if not views or len(views) > 16:
            continue
        moe.dsv4f_l2pf_next = tuple(views)
        armed += 1
        mb = sum(_bytes(t) for t in views) / 1e6
    return (f"next-layer prologue hints on {armed} MoE replicas ({mb:.1f} MB the last layer, <= {levers.L2PF_NEXT_MAX_ROWS} "
            f"rows, sources {levers.L2PF_NEXT_SOURCES})")


def issue_next(moe, rows: int) -> None:
    """Lifted MoE forward start: hint the next layer's attention-prologue weights on the prefetch stream."""
    views = getattr(moe, "dsv4f_l2pf_next", None)
    if views is None or rows > levers.L2PF_NEXT_MAX_ROWS or rows <= 0 or _STATE["failed"]:
        return
    device = views[0].device
    stream = _STREAMS[device.index]
    current = torch.cuda.current_stream(device)
    if _PEND.get(device.index):
        current.wait_stream(stream)
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            _launch(views, levers.L2PF_NEXT_GRID, levers.L2PF_NEXT_BLOCK, levers.L2PF_NEXT_MODE, stream)
    except RuntimeError as err:
        _STATE["failed"] = True
        levers.announce("l2pf-failed", f"prefetch launch failed ({err}): lever L2PF off from here on")
    _PEND[device.index] = True
    if levers.first("l2pf-next-used"):
        levers.announce("l2pf-next-used", f"lever L2PF next: prologue hints first at {rows} rows "
                        f"({sum(_bytes(t) for t in views) / 1e6:.1f} MB)")
