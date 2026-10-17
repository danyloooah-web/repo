"""Lever BSX = lever BSA2's module calling lever BSX's kernel (bsx_run in the bundle copy of ro_unit.cu: BSA2's
kernel re-scheduled -- a converged MMA-issuer warp, TMA gather4 KV tiles, a one-round-trip prologue, S^T two tiles
ahead, a one-round-trip split merge -- with every float op and MMA of BSA2, so the bytes are BSA2's = stock's) with
the two pools' slot counts (its TMA maps' row dims); the split always runs in real CTAs (BSX has no one-CTA replay).
Everything else (pins, fail-closed first-eager byte check vs stock, rank agreement, scratch) is lever BSA2's.

Lever BSA2 = lever BSA with stock's MultiCtasKv split in REAL CTAs (levers.BSA_CTAS): at <= 74 rows the kernel runs
min(3, SMs / rows) CTAs a row, one per part (stock's partition), each storing bf16(O rcp 448) and (max, den) into a
persistent scratch allocated at install (part_o [444, 512, 32] bf16, part_s [444, 64] fp32, counters [148] int32
zeroed once); the CTA that completes a row's arrival count merges every part in part order with fmhaReduction's
arithmetic (the same ops as BSA's one-CTA replay, so the bytes are unchanged) and resets the count. Launches are
stream-ordered (each BSA launch reads the scratch only after griddepcontrol.wait; its data predecessors complete
first), so one scratch serves every layer.

Lever BSA (kernel bsa_run in the bundle copy of the CUDA unit ro_unit): our own sparse MLA decode attention for the
indexed layers (compress ratio 1 / 2 / 4) of decode / target-verify calls of levers.BSA_ROWS rows, BIT-IDENTICAL to
stock's trtllm-gen kernels (EXACT): lever SPM's kernel with stock's PV accumulation (O rescaled and accumulated in
TMEM) and stock's denominator layout / summation order, read from the cubin SASS (ro_unit.cu's BSA header); lever RO's
inverse RoPE of the output fused into its epilogue (RO's per-pair arithmetic).

Fail closed, per key (rows, compress ratio): the FIRST EAGER call of a key runs stock (the backend's _forward_trtllm +
lever RO's inverse RoPE call, what fork_attn runs) and ours (rope fused), compares every output byte and agrees the
verdict over the TP group (MIN); that call itself returns None (stock runs as usual). Later calls of a verified key run
ours, eager and CUDA-graph alike; a key that failed, or was never checked eagerly before a capture, stays stock.

Call site: the lifted MQALayer.forward (dsv41f/fork_attn.py, bound only on OUR private attention replicas; the served
modules and classes keep stock's code -- node seal) asks attn() before stock's `attn_backend.forward(...)` in its
non-unified branch; None = stock's call runs unchanged. When attn() returns the output, the replica's inverse RoPE call
(ro.inv_rope) is skipped (gen_bsa.py's second edit), because attn() only takes calls where that call would run, and
applies the same per-pair arithmetic in the kernel.

What attn() reproduces of stock (fork 5e72ce5, pinned by source hash): DeepseekV4AttnBackend.forward ->
_forward_attention (DSV4AttnMetadata branch, save_kv_cache False: no store) -> DeepseekV4TrtllmAttnBackend._forward_trtllm
(V4.1: the first tp_q_head_num heads and sinks) -> _forward_trtllm_decode for compress ratio 1 / 2 / 4 with 128 SWA
columns, (rows, 512) top-k tables, as many metadata rows as query rows: the same tables (core metadata swa_page_indices,
sparse_page_indices(ratio), sparse_topk_lengths(ratio)), the same pools (_trtllm_kv_cache_views), the same softmax
scale (_get_trtllm_bmm_scales) and sinks; then forward()'s request-window commit. Everything else (DSpark windows,
draft extend, padded metadata, other ratios, other dtypes / layouts, capture-time oddities) -> None (stock).

Semantics (levers.BSA_ZERO 2 = trtllm-gen's): an SWA column at or past the row's causal length (seq_lens_casual) is
ignored whatever it holds; an SWA column below it holding -1, and a top-k column below the row's length holding -1, is
a zero K/V row with logit 0; top-k columns at or past the length are ignored. Stock's MultiCtasKv split (min(3, SMs /
rows) CTAs a row at <= 74 rows) is replayed by vsplit = _vsplit(rows) inside one CTA. Decisions depend only on the
call's mode, shapes, dtypes, strides, row count and the rank-agreed first-use verdicts: identical on both ranks, in
eager and in CUDA-graph replays (the kernel is deterministic). install() also runs a synthetic self-test (fp32 torch
reference, rel L2 per row <= BSA_SELFTEST_TOL); a failure keeps stock.
"""
import ast
import hashlib
import inspect
import sys

import torch

from dsv41f import levers

H, D, SWA, TOPK = 32, 512, 128, 512
WORLD = 2
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
I32 = torch.int32
I64 = torch.int64
C64 = torch.complex64
_STATE = {"detail": "not installed", "ok": False, "calls": 0, "checks": 0}
_FN: dict = {}
_TAB: dict = {}
_CLS: dict = {}
_CHECK: dict = {}   # (rows, ratio) -> True (first eager call == stock, rank-agreed) / False

# sha256[:16] of the source segment of every stock function attn() reproduces or relies on (fork 5e72ce5)
PINNED = {
    ("sglang.srt.layers.attention.deepseek_v4_backend", "DeepseekV4AttnBackend.forward"): "f00554dae292ca7b",
    ("sglang.srt.layers.attention.deepseek_v4_backend", "DeepseekV4AttnBackend._forward_attention"): "ce5abbceee4825ee",
    ("sglang.srt.layers.attention.deepseek_v4_backend", "DSV4AttnMetadata.sparse_page_indices"): "b8b1984a02989dc1",
    ("sglang.srt.layers.attention.deepseek_v4_backend", "DSV4AttnMetadata.sparse_topk_lengths"): "f44ab9d4894d1510",
    ("sglang.srt.layers.attention.deepseek_v4_trtllm_backend", "DeepseekV4TrtllmAttnBackend._forward_trtllm"):
        "8f2a3584ee8ac2dd",
    ("sglang.srt.layers.attention.deepseek_v4_trtllm_backend", "DeepseekV4TrtllmAttnBackend._forward_trtllm_decode"):
        "b4506c7e8b830587",
    ("sglang.srt.layers.attention.deepseek_v4_trtllm_backend", "DeepseekV4TrtllmAttnBackend._trtllm_kv_cache_views"):
        "174ba6a2786f99d6",
    ("sglang.srt.layers.attention.deepseek_v4_trtllm_backend", "DeepseekV4TrtllmAttnBackend._get_trtllm_bmm_scales"):
        "3d9f261fe80f4a6f",
    ("sglang.srt.models.deepseek_v4", "MQALayer.forward"): "b1423a431a7aaca7",
}


def source_hash(module, qualname: str, cache=None):
    """sha256[:16] of the exact source text of `qualname` ("f" or "Class.f") in `module` (None: not found)."""
    hit = cache.get(module.__name__) if cache is not None else None
    if hit is None:
        src = inspect.getsource(module)
        hit = (src, ast.parse(src))
        if cache is not None:
            cache[module.__name__] = hit
    src, node = hit
    for part in qualname.split("."):
        node = next((c for c in ast.walk(node) if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and c.name == part and c is not node), None)
        if node is None:
            return None
    seg = ast.get_source_segment(src, node)
    return hashlib.sha256(seg.encode()).hexdigest()[:16] if seg is not None else None


def _modules():
    import sglang.srt.layers.attention.deepseek_v4_backend  # noqa: F401
    import sglang.srt.layers.attention.deepseek_v4_trtllm_backend  # noqa: F401
    import sglang.srt.models.deepseek_v4  # noqa: F401
    return {name: sys.modules[name] for name, _ in PINNED}


def _calls_bsa(fn) -> bool:
    """The lifted attention forward asks bsa.attn( before stock's backend call and gates lever RO on bsa_o."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return "bsa.attn(" in src and "bsa_o is None" in src


def install(model=None, layers=None) -> str:
    """Pin the stock internals, check fork_attn's edits, load the kernel and self-test it (model_node.prepare)."""
    if _STATE["ok"] or _STATE["detail"].startswith("off"):
        return _STATE["detail"]
    if not getattr(levers, "BSA", False):
        _STATE["detail"] = "off (levers.BSA)"
        return _STATE["detail"]
    try:
        mods = _modules()
        cache: dict = {}
        bad = [f"{mod}.{name} {got}" for (mod, name), want in PINNED.items()
               if (got := source_hash(mods[mod], name, cache)) != want]
        if bad:
            raise RuntimeError("source differs from the mirror: " + ", ".join(bad))
        tb = mods["sglang.srt.layers.attention.deepseek_v4_trtllm_backend"]
        bk = mods["sglang.srt.layers.attention.deepseek_v4_backend"]
        dsv4 = mods["sglang.srt.models.deepseek_v4"]
        if not dsv4._is_cuda or dsv4._is_hip or dsv4._is_npu:
            raise RuntimeError("not the CUDA path")
        from dsv41f import fork_attn
        from dsv41f import ro

        if not _calls_bsa(fork_attn.attn_forward):
            raise RuntimeError("fork_attn.attn_forward lacks lever BSA's edits")
        try:
            import ro_unit
        except ImportError as err:
            raise RuntimeError(f"CUDA unit ro_unit not importable ({err})")
        run = getattr(ro_unit, "bsx_run", None)
        if run is None:
            raise RuntimeError("ro_unit has no bsx_run (not the BSX build of the unit)")
        raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        _FN.update(run=run, trt=tb.DeepseekV4TrtllmAttnBackend, meta=bk.DSV4AttnMetadata,
                   fused_max=dsv4._FUSED_WO_A_MAX_TOKENS, invrope=dsv4._wo_a_fp8_mxscale_fused_invrope,
                   ro=ro.inv_rope,
                   raw=raw if raw is not None else (lambda idx: torch.cuda.current_stream(idx).cuda_stream))
        lo, hi = (int(x) for x in levers.BSA_ROWS)
        if not 1 <= lo <= hi <= 148:
            raise RuntimeError(f"levers.BSA_ROWS {levers.BSA_ROWS!r}")
        _FN["rows"] = (lo, hi)
        _FN["skip"] = frozenset(int(r) for r in getattr(levers, "BSA_SKIP_ROWS", ()))
        _FN["sms"] = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
        _FN["ctas"] = bool(getattr(levers, "BSA_CTAS", True))
        if not _FN["ctas"]:
            raise RuntimeError("levers.BSA_CTAS False (lever BSX runs the split in real CTAs only)")
        dev0 = torch.device("cuda", torch.cuda.current_device())
        _FN["po"] = torch.zeros(148 * 3, D, H, dtype=BF16, device=dev0)
        _FN["ps"] = torch.zeros(148 * 3, 64, dtype=torch.float32, device=dev0)
        _FN["cnt"] = torch.zeros(148, dtype=I32, device=dev0)
        zero = int(getattr(levers, "BSA_ZERO", 2))
        if zero != 2:
            raise RuntimeError(f"levers.BSA_ZERO {zero!r} (only trtllm-gen's semantics 2 is bit-identical)")
        _FN["zero"] = zero
        err = _selftest()
        if err:
            raise RuntimeError(f"self-test: {err}")
        replicas = 0
        for r in layers or ():
            attn = r._modules.get("self_attn") if hasattr(r, "_modules") else None
            replicas += getattr(getattr(attn, "forward", None), "__func__", None) is fork_attn.attn_forward
    except Exception as e:  # noqa: BLE001 - any missing / changed internal keeps stock
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:400]})"
        return _STATE["detail"]
    _STATE["ok"] = True
    _STATE["detail"] = (f"on (EXACT, bitwise vs stock, first eager call per (rows, ratio) byte-checked, rank-agreed; "
                        f"lever BSX kernel, split in real CTAs): "
                        f"sparse MLA decode attention + inverse RoPE on {replicas} attention replicas, decode / verify, "
                        f"ratio 1 / 2 / 4, {lo}..{hi} rows; self-test {_STATE.get('self', '')}")
    return _STATE["detail"]


def _selftest() -> str:
    """Synthetic tables at 1 / 6 / 48 / 72 / 100 rows (mixed lengths, empty SWA slots, short top-k) vs an fp32
    reference; '' = pass (the bitwise check vs stock runs at each key's first eager call)."""
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device="cpu").manual_seed(int(getattr(levers, "BSA_SEED", 20261016)))
    tol = float(getattr(levers, "BSA_SELFTEST_TOL", 0.03))
    worst = 0.0
    for rows in (1, 6, 48, 72, 100):
        n_swa, n_c = 4096, 8192
        swa = (torch.randn(n_swa, D, generator=g) * 1.0).to(FP8).to(dev)
        ckv = (torch.randn(n_c, D, generator=g) * 1.0).to(FP8).to(dev)
        q = (torch.randn(rows, H, D, generator=g) * 2.0).to(FP8).to(dev)
        swa_idx = torch.randint(0, n_swa, (rows, SWA), generator=g, dtype=I32)
        ctx = torch.randint(1, 200, (rows,), generator=g)
        seq = torch.full((rows,), SWA, dtype=I32)
        swa_idx[torch.arange(SWA)[None, :] >= ctx[:, None]] = -1
        topk = torch.randint(0, n_c, (rows, TOPK), generator=g, dtype=I32)
        lens = torch.randint(0, TOPK + 1, (rows,), generator=g, dtype=I32)
        topk[torch.arange(TOPK)[None, :] >= lens[:, None]] = -1
        sinks = torch.randn(H, generator=g).to(dev)
        swa_idx, topk, lens, seq = swa_idx.to(dev), topk.to(dev), lens.to(dev), seq.to(dev)
        out = torch.empty(rows, H, D, dtype=BF16, device=dev)
        scale = D ** -0.5
        _FN["run"](q.data_ptr(), H * D, swa.data_ptr(), ckv.data_ptr(), swa_idx.data_ptr(), SWA, topk.data_ptr(), TOPK,
                   lens.data_ptr(), 0, sinks.data_ptr(), 0, 0, 0, out.data_ptr(), rows, scale, 0, 0, 2, seq.data_ptr(),
                   *_split_args(rows), _FN["raw"](dev.index), n_swa, n_c)
        torch.cuda.synchronize(dev)
        qf, sf, cf = q.float(), swa.float(), ckv.float()
        for r in range(rows):
            si = swa_idx[r]
            nz = int((si < 0).sum())
            sv = si[si >= 0].long()
            tv = topk[r, :int(lens[r])]
            tv = tv[tv >= 0].long()
            k = torch.cat([sf[sv], cf[tv]], 0)
            s = torch.cat([(qf[r] @ k.T) * scale, torch.zeros(H, nz, device=dev), sinks[:, None]], 1)
            ref = torch.softmax(s, 1)[:, :k.shape[0]] @ k
            e = float((out[r].float() - ref).norm() / ref.norm().clamp_min(1e-30))
            if not e == e:
                return f"NaN at rows {rows} row {r}"
            worst = max(worst, e)
    _STATE["self"] = f"worst rel L2 {worst:.2e} (tol {tol})"
    return "" if worst <= tol else f"rel L2 {worst:.3e} > {tol}"


def _vsplit(rows: int) -> int:
    """stock's CTAs a row (computeCtaAndClusterConfig: min(3 = ceil(640 / 256), SMs / rows); 1 = its single-CTA
    Persistent kernel)."""
    k = min(3, max(1, _FN["sms"] // rows))
    return k if k > 1 else 1


def _split_args(rows: int):
    """(vsplit, ctas, part_o, part_s, counters) of bsx_run: one CTA a part (ctas == vsplit)."""
    vs = _vsplit(rows)
    ctas = vs
    return (vs, ctas, _FN["po"].data_ptr(), _FN["ps"].data_ptr(), _FN["cnt"].data_ptr())


def _freqs(freqs_cis):
    ent = _TAB.get(id(freqs_cis))
    if ent is None or ent[0] is not freqs_cis:
        f = None
        if (type(freqs_cis) is torch.Tensor and freqs_cis.dtype is C64 and freqs_cis.dim() == 2
                and freqs_cis.shape[1] == 32 and freqs_cis.is_cuda and freqs_cis.is_contiguous()
                and not freqs_cis.is_conj()):
            f = torch.view_as_real(freqs_cis).flatten(-2)
        ent = (freqs_cis, f)
        _TAB[id(freqs_cis)] = ent
    f = ent[1]
    if f is None or f.data_ptr() & 15:
        return None
    return f


def _backend_ok(cls) -> bool:
    """The trtllm-gen DSV4 backend class, or a subclass that keeps every method attn() reproduces (cached per class)."""
    ok = _CLS.get(cls)
    if ok is None:
        trt = _FN["trt"]
        ok = cls is trt or (isinstance(cls, type) and issubclass(cls, trt)
                            and cls.forward is trt.forward and cls._forward_attention is trt._forward_attention
                            and cls._forward_trtllm is trt._forward_trtllm
                            and cls._forward_trtllm_decode is trt._forward_trtllm_decode
                            and cls._trtllm_kv_cache_views is trt._trtllm_kv_cache_views
                            and cls._get_trtllm_bmm_scales is trt._get_trtllm_bmm_scales)
        _CLS[cls] = ok
        if not ok:
            levers.announce(f"bsa-backend-{cls.__name__}", f"lever BSA: backend {cls.__name__} is not the pinned "
                                                         f"trtllm-gen DSV4 backend: stock attention")
    return ok


def _rows(x, bs):
    """stock's match_num_queries for a table that needs no padding (None: it would pad)."""
    if x is None or x.shape[0] < bs:
        return None
    return x if x.shape[0] == bs else x[:bs]


def _agree(ok: bool) -> bool:
    """The verdict MIN over the TP group (both ranks reach a key's first eager call in the same layer call)."""
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group.device_group
    if dist.get_world_size(group) != WORLD:
        return False
    t = torch.tensor([1 if ok else 0], device=torch.cuda.current_device(), dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def attn(layer, backend, q, forward_batch, attn_sink, positions, agp):
    """fork_attn's attention call when lever BSA takes it: the bf16 [rows, 32, 512] output with the inverse RoPE of
    the last 64 dims of every head applied (lever RO's bytes on stock's attention values); None = stock's call."""
    if not _STATE["ok"] or agp is not None:
        return None
    try:
        args = _args(layer, backend, q, forward_batch, attn_sink, positions)
    except Exception as e:  # noqa: BLE001 - an unexpected object keeps stock for this call
        levers.announce("bsa-error", f"lever BSA: {type(e).__name__}: {str(e)[:300]} (stock for this call)")
        return None
    if args is None:
        return None
    key = (args["bs"], args["ratio"])
    ok = _CHECK.get(key)
    if ok is False:
        return None
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return None   # never checked eagerly: stock is captured for this key
        _first(key, layer, backend, q, forward_batch, attn_sink, positions, args)
        return None       # the checking call itself runs stock
    out = torch.empty((args["bs"], H, D), dtype=BF16, device=q.device)
    _launch(args, q, out, rope=1)
    pool = backend.token_to_kv_pool   # stock forward()'s tail: the request window commit
    window = pool.request_window
    if window is not None and not backend.is_dspark_draft and not forward_batch.forward_mode.is_idle():
        window.commit(pool._swa_local_layer_id(layer.attn_mqa.layer_id))
    _STATE["calls"] += 1
    return out


def _launch(a, q, out, rope):
    _FN["run"](q.data_ptr(), q.stride(0), a["swa_kv"].data_ptr(), a["c_kv"].data_ptr(), a["swa"].data_ptr(),
               a["swa"].stride(0), a["extra"].data_ptr(), a["extra"].stride(0), a["lens"].data_ptr(),
               1 if a["lens"].dtype is I64 else 0, a["sink"].data_ptr(), a["f"].data_ptr() if rope else 0,
               a["positions"].data_ptr() if rope else 0, a["p64"], out.data_ptr(), a["bs"], float(a["scale"]), rope,
               1, _FN["zero"], a["seq"].data_ptr(), *_split_args(a["bs"]), _FN["raw"](a["dev"]),
               a["swa_kv"].numel() // D, a["c_kv"].numel() // D)


def _first(key, layer, backend, q, forward_batch, attn_sink, positions, a):
    """Stock (the backend's _forward_trtllm + lever RO's inverse RoPE call) vs ours (rope fused), every byte."""
    why = ""
    _STATE["checks"] += 1
    try:
        attn_layer = layer.attn_mqa
        ref = backend._forward_trtllm(q=q, layer=attn_layer, compress_ratio=a["ratio"], core_attn_metadata=a["core"],
                                      forward_batch=forward_batch, attn_sink=attn_sink, swa_page_indices=a["swa"],
                                      extra_indices=a["extra"], extra_topk_lengths=a["lens"])
        ref = ref.reshape(a["bs"], H, D).clone()
        _FN["ro"](ref[..., -layer.qk_rope_head_dim:], None, layer.freqs_cis, positions=positions, inverse=True)
        got = torch.empty((a["bs"], H, D), dtype=BF16, device=q.device)
        _launch(a, q, got, rope=1)
        if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
            n = int((ref.view(torch.int16) != got.view(torch.int16)).flatten(1).any(1).sum())
            why = f"{n} of {a['bs']} rows differ"
    except Exception as e:  # noqa: BLE001 - any failure keeps stock for this key
        why = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        ok = _agree(not why)
    except Exception as e:  # noqa: BLE001
        ok = False
        why = why or f"agree {type(e).__name__}: {str(e)[:120]}"
    _CHECK[key] = ok
    levers.announce(f"bsa-{key[0]}-{key[1]}",
                    f"lever BSA: {key[0]} rows ratio {key[1]} ({'verify' if forward_batch.forward_mode.is_target_verify() else 'decode'}, "
                    f"vsplit {_vsplit(key[0])}) first eager call: " +
                    ("ours == stock byte for byte: on" if ok else f"OFF ({why or 'peer rank differed'}): stock"))


def _args(layer, backend, q, forward_batch, attn_sink, positions):
    if type(q) is not torch.Tensor or q.dtype is not FP8 or q.dim() != 3 or torch.compiler.is_compiling():
        return None
    bs, nh, hd = q.shape
    lo, hi = _FN["rows"]
    if not lo <= bs <= hi or nh != H or hd != D or bs in _FN["skip"]:
        return None
    ratio = layer.compress_ratio
    if ratio not in (1, 2, 4):
        return None
    fm = forward_batch.forward_mode
    if not (fm.is_decode() or fm.is_target_verify()) or fm.is_idle():
        return None
    if not _backend_ok(type(backend)) or not getattr(backend, "trtllm_attn", False) or not backend.is_dsv41:
        return None
    # RO's call in fork_attn runs for this output: CUDA, no fused WO-A rope, no HIP / NPU branch
    if layer.wo_a_fp8 and _FN["invrope"] is not None:
        return None
    if layer.use_fused_wo_a and 0 < bs <= _FN["fused_max"]:
        return None
    attn_layer = layer.attn_mqa
    if attn_layer.tp_q_head_num != H:
        return None
    meta = backend.forward_metadata
    core = getattr(meta, "core_attn_metadata", None)
    if not isinstance(core, _FN["meta"]):
        return None
    if core.seq_lens_casual.shape[0] < bs:
        return None
    swa = _rows(core.swa_page_indices, bs)
    extra = _rows(core.sparse_page_indices(ratio), bs)
    lens = _rows(core.sparse_topk_lengths(ratio), bs)
    if swa is None or extra is None or lens is None:
        return None
    if (swa.dim() != 2 or swa.shape[1] != SWA or swa.dtype is not I32 or swa.stride(1) != 1
            or extra.dim() != 2 or extra.shape[1] != TOPK or extra.dtype is not I32 or extra.stride(1) != 1
            or lens.dim() != 1 or lens.dtype not in (I32, I64) or (bs > 1 and lens.stride(0) != 1)):
        return None
    if q.stride(2) != 1 or q.stride(1) != D or q.stride(0) & 15 or q.data_ptr() & 15:
        return None
    sink = attn_sink[:H]
    if sink.dtype is not torch.float32 or not sink.is_contiguous() or sink.shape[0] != H:
        return None
    swa_kv, c_kv = backend._trtllm_kv_cache_views(attn_layer.layer_id, ratio)
    if (swa_kv.dim() != 4 or swa_kv.shape[-1] != D or not swa_kv.is_contiguous() or swa_kv.element_size() != 1
            or c_kv.dim() != 4 or c_kv.shape[-1] != D or not c_kv.is_contiguous() or c_kv.element_size() != 1):
        return None
    scale = backend._get_trtllm_bmm_scales(attn_layer)[0]
    f = _freqs(layer.freqs_cis)
    if f is None or type(positions) is not torch.Tensor or positions.dim() != 1 or positions.shape[0] < bs:
        return None
    if positions.dtype is I64:
        p64 = 1
    elif positions.dtype is I32:
        p64 = 0
    else:
        return None
    if bs > 1 and positions.stride(0) != 1:
        return None
    seq = core.seq_lens_casual
    if seq.dtype is not I32 or (bs > 1 and seq.stride(0) != 1):
        return None
    return dict(bs=bs, ratio=ratio, core=core, swa=swa, extra=extra, lens=lens, sink=sink, swa_kv=swa_kv, c_kv=c_kv,
                scale=scale, f=f, positions=positions, p64=p64, seq=seq, dev=q.get_device())


def summary() -> str:
    on = sorted(k for k, v in _CHECK.items() if v)
    off = sorted(k for k, v in _CHECK.items() if not v)
    return (f"lever BSA: {_STATE['detail']}; calls {_STATE['calls']}; first-use checks {_STATE['checks']}: "
            f"{len(on)} keys on, {len(off)} off {off[:12]}")
