"""Lever SKW on the fork (levers.SKW, CUDA unit skw_unit): lever SK's mHC stats reduce + sinkhorn launch (sk.py _ours /
_ours_dg -> sk_unit.run) of CUDA-graph decode / verify forwards on a CAPPED grid, so the low-priority mHC stats stream
never holds more than levers.SKW_BANDS' grid cap of SMs.

Why (c26rtr k8..k24 profiles, both TP ranks aligned by the all-reduce kernels): B300 runs one kernel's CTAs on an SM at a
time. sk_unit takes one CTA a row (48..144 CTAs); where its launch races the main chain it takes the SMs that the
one-CTA-a-row persistent sparse-attention kernel (fmha) or the routed FC1 needs at launch: k20 fmha 17.8 -> 24.6 us in
34 % of the attention halves, k24 FFN half +7.4 us in 28 % (FC1 starts ~3 us late, runs ~2 us longer), k16 / k20 the
sinkhorn sometimes ends after FC2 (+7..+22 us, the MoE all-reduce waits). The race is decided per rank, so it is also
the largest single source of the TP cross-rank wait at the all-reduces (e.g. k24 FFN halves: segment difference sd 1.86
us when both ranks placed the sinkhorn alike, 4.06 when not). Served at levers.SKW_BANDS (9..136 rows, <= 8 CTAs of
one 16-row / 32-row block, a grid-stride loop beyond): fmha's one CTA a row keeps >= 12 free SMs, FC1 / FC2 lose <= 8
SMs for ~10 us. Not above 136 rows: beside fmha's 144 CTAs only 4 SMs are free, so a CTA would walk two blocks (~20 us
in-engine) and the k24 attention half / index layers measured +0.2 / +0.9 us a layer. Measured (c27skw vs five c26rtr
reads, box2): k8 wq_b 8.35 -> 6.91 us, attention half -1.5 us a layer; k20 fmha slow mode 17 % -> 1 % of the halves,
attention half -1.3 us; stack minus FC kernels k8 -38 / k12 -8 / k16 -15 / k20 -51 us a cycle (k24 not served).

Gates (any miss: sk_unit.run exactly as lever SK calls it):
  prepare  levers.SKW; the unit imports and init() opts its kernels into their shared memory; a self-test on this rank
           (lever SK's sk_unit vs skw_unit on synthetic partials incl. +-0 / inf / NaN / subnormal / huge values, every
           served row count of both paths) agreed over the TP CPU group (gloo MIN) -- one rank failing turns it off on all.
  call     a CUDA-graph capture (eager forwards -- prefill / extend and the eager live audit -- always run sk_unit's
           kernel) on an mHC stats stream (decode / verify: prefill graphs run the stats inline on the main stream,
           where the stats are on the critical path and sk_unit's wide grid is faster); the row count in a
           levers.SKW_BANDS band; slices within the path's cap; the key (rows, path, slices)
           checked: its first EAGER call (a warmup forward, before any capture of that batch) runs sk_unit into the real
           outputs AND skw_unit into scratch outputs (pre-filled with a NaN pattern) and compares every byte; a difference
           or an error keeps sk_unit for that key for good. Lever SK compares the real outputs against stock in the same
           call (sk._checked), so a served key is stock's bytes twice over.
Exact: same per-row arithmetic as sk_unit (skw_unit.cu header); the grid only changes which CTA / warp computes a row.
Decisions depend on the row count, static facts and deterministic byte checks, so both TP ranks capture the same graph.
Nothing is patched on a served module or class (sk.py, a bundle module, calls run() in place of sk_unit.run).
"""
import sys

import torch

from dsv41f import levers

SMALL_SLICES, DG_SLICES = 80, 16
_UNIT: list = []
_STATE = {"detail": "not prepared", "on": False, "graph": 0, "sk": 0, "streams": (), "attn": frozenset(),
          "ffn": frozenset()}
_OK: dict = {}            # (rows, residual path, slices) -> first eager call == sk_unit byte for byte
_POISON = 0x7FBADBAD      # a NaN payload neither kernel writes: unwritten scratch outputs differ


def unit():
    if not _UNIT:
        mod = None
        try:
            import skw_unit as mod
        except ImportError as err:
            levers.announce("skw-missing", f"bundle CUDA unit skw_unit not importable ({err}): lever SKW off")
        _UNIT.append(mod)
    return _UNIT[0]


def _cfg(m: int, residual: bool, slices: int):
    """(rows a block, grid cap) of the first levers.SKW_BANDS band holding m, else None."""
    if slices < 1 or slices > (DG_SLICES if residual else SMALL_SLICES):
        return None
    rpb = int(levers.SKW_ROWS_PER_BLOCK[1 if residual else 0])
    for lo, hi, cap in levers.SKW_BANDS:
        if int(lo) <= m <= int(hi) and int(cap) >= 1:
            return rpb, int(cap)
    return None


def _bits_equal(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int32), b.view(torch.int32))


def _agree(values):
    """MIN over the TP CPU group (gloo), elementwise (lever RE5's helper)."""
    from sglang.srt.distributed import get_tp_group

    t = torch.tensor(values, dtype=torch.int64)
    group = get_tp_group()
    if group.world_size > 1:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group)
    return t.tolist()


def _rows_served():
    out = []
    for lo, hi, cap in levers.SKW_BANDS:
        out.extend(range(int(lo), int(hi) + 1))
    return sorted(set(out))


def _partials(g, m, slices, dev, special):
    mix = torch.randn(slices, m, 24, device=dev, generator=g) * 3.0
    sq = torch.rand(slices, m, device=dev, generator=g) * 400.0
    if special:
        flat, sflat = mix.view(-1), sq.view(-1)
        vals = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1e-40, -1e-41, 3e38, -3e38,
                             1e-30], device=dev)
        idx = torch.randint(0, flat.numel(), (max(4, flat.numel() // 97),), device=dev, generator=g)
        flat[idx] = vals[torch.randint(0, vals.numel(), idx.shape, device=dev, generator=g)]
        sidx = torch.randint(0, sflat.numel(), (max(2, sflat.numel() // 211),), device=dev, generator=g)
        sflat[sidx] = vals[torch.randint(0, vals.numel(), sidx.shape, device=dev, generator=g)].abs()
        if m > 2:
            mix[:, 1, :] = 0.0          # an all-zero row (rsqrt of eps only)
            sq[:, 1] = 0.0
            mix[:, 2, :] = -0.0
    return mix.contiguous(), sq.contiguous()


def _self_test(sk_mod, dev):
    """sk_unit vs skw_unit at every served row count of both paths on synthetic partials; returns (ok, detail)."""
    mod = unit()
    g = torch.Generator(device=dev).manual_seed(int(levers.SKW_SEED))
    scale = (torch.rand(3, device=dev, generator=g) * 2.0).contiguous()
    base = torch.randn(24, device=dev, generator=g).contiguous()
    stream = torch.cuda.current_stream(dev).cuda_stream
    bad, cases = [], 0
    for m in _rows_served():
        for residual in (False, True):
            if residual and not (levers.SK_DG and 128 <= m <= levers.SK_DG_MAX_ROWS):
                continue
            if not residual and m > levers.SK_MAX_ROWS:
                continue
            slices = DG_SLICES if residual else SMALL_SLICES
            cfg = _cfg(m, residual, slices)
            if cfg is None:
                continue
            for special in (False, True):
                mix, sq = _partials(g, m, slices, dev, special)
                lo = (torch.randn(slices, m, 24, device=dev, generator=g) * 1e-3).contiguous() if residual else None
                outs = []
                for which in (0, 1):
                    pre = torch.empty(m, 4, dtype=torch.float32, device=dev)
                    post = torch.empty_like(pre)
                    comb = torch.empty(m, 4, 4, dtype=torch.float32, device=dev)
                    for t in (pre, post, comb):
                        t.view(torch.int32).fill_(_POISON)
                    args = (mix.data_ptr(), 0 if lo is None else lo.data_ptr(), sq.data_ptr(), scale.data_ptr(),
                            base.data_ptr(), pre.data_ptr(), post.data_ptr(), comb.data_ptr(), m, 1.0 / 20480,
                            1e-6 if special else 1e-20, slices, 20, 1e-6)
                    if which == 0:
                        sk_mod.run(*args, int(levers.SK_DG_ROWS_PER_CTA) if residual else 1, stream)
                    else:
                        mod.run(*args, cfg[0], cfg[1], int(levers.SKW_ASYNC), stream)
                    outs.append((pre, post, comb))
                cases += 1
                if not all(_bits_equal(a, b) for a, b in zip(*outs)):
                    bad.append((m, residual, special))
    torch.cuda.synchronize(dev)
    return not bad, f"{cases} cases, {len(bad)} differing{(' e.g. ' + str(bad[:3])) if bad else ''}"


def _sublayer_tags(layers):
    """data_ptr()s of the mixing weights each sublayer's stats call passes (hc.fn on the small path, hc.tf32_parts[0]
    on the deep_gemm path), per sublayer kind: they tell an attention sublayer's call from an FFN sublayer's."""
    tags = {"attn": set(), "ffn": set()}
    for layer in layers:
        for kind, hc in (("attn", getattr(layer, "attn_hc", None)), ("ffn", getattr(layer, "ffn_hc", None))):
            if hc is None:
                continue
            fn = getattr(hc, "fn", None)
            parts = getattr(hc, "tf32_parts", None)
            if isinstance(fn, torch.Tensor):
                tags[kind].add(int(fn.data_ptr()))
            if parts and isinstance(parts[0], torch.Tensor):
                tags[kind].add(int(parts[0].data_ptr()))
    return frozenset(tags["attn"] - tags["ffn"]), frozenset(tags["ffn"] - tags["attn"])


def prepare(stats_streams=(), layers=()) -> str:
    """Once, outside any capture, on every rank (the agreement is collective). stats_streams: the mHC stats streams the
    replicas use (stock's, lever HP's): only launches on them are served (decode / verify; prefill and extend forwards run
    the stats inline on the main stream, where sk_unit's one-CTA-a-row grid is the faster one). layers: the replicas,
    for levers.SKW_SUBLAYERS (which sublayers' stats are served; a call whose weight is in neither set is served only
    when both kinds are)."""
    _STATE["streams"] = tuple(sorted({int(s.cuda_stream) for s in stats_streams if s is not None}))
    try:
        _STATE["attn"], _STATE["ffn"] = _sublayer_tags(layers)
    except Exception:  # noqa: BLE001 - no tags: only the both-kinds setting serves
        _STATE["attn"], _STATE["ffn"] = frozenset(), frozenset()
    if not levers.SKW:
        _STATE["detail"] = "off"
        return _STATE["detail"]
    ok, detail = False, "CUDA unit skw_unit unavailable"
    try:
        from dsv41f import sk

        mod = unit()
        sk_mod = sk.unit() if sk._STATE.get("stock") is not None else None
        if mod is None or sk_mod is None:
            detail = "skw_unit or lever SK's sk_unit unavailable"
        elif not torch.cuda.is_available():
            detail = "no CUDA device"
        else:
            mod.init()
            ok, detail = _self_test(sk_mod, torch.device("cuda", torch.cuda.current_device()))
    except Exception as err:   # noqa: BLE001 - any failure keeps sk_unit everywhere
        ok, detail = False, f"{type(err).__name__}: {str(err)[:160]}"
    if not _STATE["streams"]:
        ok, detail = False, f"no mHC stats stream ({detail})"
    agreed = bool(_agree([int(ok)])[0])
    _STATE["on"] = agreed
    _STATE["detail"] = (f"{'ON' if agreed else 'OFF'} (self-test on this rank: {detail}; rank-agreed); bands "
                        f"{levers.SKW_BANDS}, rows a block {levers.SKW_ROWS_PER_BLOCK} (small, deep_gemm path), "
                        f"{'cp.async' if levers.SKW_ASYNC else 'register'} loads; sublayers {levers.SKW_SUBLAYERS} "
                        f"({len(_STATE['attn'])} / {len(_STATE['ffn'])} attn / ffn weight tags); CUDA-graph captures on "
                        f"the mHC stats stream only ({len(_STATE['streams'])} stream(s)), first eager call per key "
                        f"checked vs sk_unit")
    return _STATE["detail"]


def _check(key, ptrs, outs, tail, cfg, stream) -> bool:
    try:
        scratch = [torch.empty_like(t) for t in outs]
        for t in scratch:
            t.view(torch.int32).fill_(_POISON)
        unit().run(*ptrs, *(t.data_ptr() for t in scratch), *tail, cfg[0], cfg[1], int(levers.SKW_ASYNC), stream)
        ok = all(_bits_equal(a, b) for a, b in zip(outs, scratch))
    except Exception as e:  # noqa: BLE001 - any failure keeps sk_unit for this key
        print(f"CACHEON_DSV4F lever SKW: {key} failed ({type(e).__name__}: {str(e)[:120]}), sk_unit",
              file=sys.stderr, flush=True)
        ok = False
    levers.announce(f"skw{key}", f"lever SKW: rows {key[0]} {'deep_gemm' if key[1] else 'small'} path, {key[2]} "
                    f"slices: {'== sk_unit byte for byte: graphs on' if ok else 'DIFFERS: sk_unit'} (grid <= {cfg[1]}, "
                    f"{cfg[0]} rows a block)")
    return ok


def _sublayer_ok(tag) -> bool:
    want = tuple(levers.SKW_SUBLAYERS)
    if "attn" in want and "ffn" in want:
        return True
    ptr = int(tag.data_ptr()) if isinstance(tag, torch.Tensor) else None
    kind = "attn" if ptr in _STATE["attn"] else "ffn" if ptr in _STATE["ffn"] else None
    return kind is not None and kind in want


def run(sk_mod, part_mix, residual, part_sq, scale, base, pre, post, comb, m, inv_k, rms_eps, slices, iters, eps,
        rows_per_cta, stream, tag=None):
    """lever SK's sk_unit.run(...) call with tensors for the pointers: skw_unit inside a CUDA-graph capture at a checked
    key, sk_unit otherwise (eager: also the key's first-call check). tag: the call's mixing weight (sublayer kind)."""
    ptrs = (part_mix.data_ptr(), 0 if residual is None else residual.data_ptr(), part_sq.data_ptr(), scale.data_ptr(),
            base.data_ptr())
    outp = (pre.data_ptr(), post.data_ptr(), comb.data_ptr())
    tail = (m, inv_k, rms_eps, slices, iters, eps)
    cfg = (_cfg(int(m), residual is not None, int(slices))
           if _STATE["on"] and int(stream) in _STATE["streams"] and _sublayer_ok(tag) else None)
    if cfg is not None:
        key = (int(m), residual is not None, int(slices))
        if torch.cuda.is_current_stream_capturing():
            if _OK.get(key) is True:
                unit().run(*ptrs, *outp, *tail, cfg[0], cfg[1], int(levers.SKW_ASYNC), stream)
                _STATE["graph"] += 1
                return
        else:
            sk_mod.run(*ptrs, *outp, *tail, rows_per_cta, stream)
            if key not in _OK:
                _OK[key] = _check(key, ptrs, (pre, post, comb), tail, cfg, stream)
            _STATE["sk"] += 1
            return
    sk_mod.run(*ptrs, *outp, *tail, rows_per_cta, stream)
    _STATE["sk"] += 1
