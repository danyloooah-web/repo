"""Lever FGP on the fork (levers.FGP): lever FG's per-bucket CUDA-graph captures done ahead, at the engine's first
eligible eager EXTEND forward, instead of at each bucket's first real use.

WHY. Lever FG (fg.py) captures a row bucket lazily, inside the bucket's first real forward: admission (rank agreement,
FMT band verified, free memory), warmups at S rows until the levers' learn / verify state settles, two graph captures
per FG layer, then the first replay checked against the eager path. That costs 16-520 ms per bucket (40 layers) and
lands inside the TIMED replay windows of the live evaluation: its untimed warmup (the 24 sessions' first prompts, cold
prefixes chunked at 16384 rows) only runs eager extends above FG_MAX_ROWS, so every FG bucket is first met in a timed
window, where the stall costs speed and the service-attainment gate.

WHAT RUNS. Once per process, at the first fg.plan() call of the first eligible eager EXTEND forward (any row count:
fg.py's hook, inside every lever context of our node forward, before that forward's first layer runs; this lands in
the live replay's untimed warmup), on both TP ranks:
  1. plan: the FG layers (the replicas fgp.install found carrying FG's graph segment, their mHC boundaries resolved as
     their own forward resolves them), FG's bucket table, the buckets still new, their FMT bands; one agreement that
     every rank holds the same plan (else nothing more happens: FG's lazy path everywhere); then, rank-local, the ops
     of FG's first-use byte check on scratch tensors (_warm_check_ops: their first kernel load would otherwise stall the
     first timed forward that checks a bucket);
  2. lever FMT's band check for every band of those buckets not checked yet, through FMT's own wrapper on the first FG
     layer's MoE replica (stock's tactic and the pinned one on the same synthetic rows, every output byte compared,
     result in fmt._BAND_OK) with the post-experts all-reduce left out by stock's own flag
     (get_forward().scoped(mlp_reduce_scatter=True), what the MoE honours when a later step reduces): no collective;
  3. per bucket, largest first (the first capture reserves FG's graph pool at the largest bucket): a budget agreement
     (FGP_BUDGET_S of wall clock since step 1 began, FMT checks included; SGLang's watchdog ends a forward after 300 s),
     FG's own admission (fg._admit: band verified on every rank, free memory for the static
     buffers + the pool reserve, FG_MEM_MARGIN; refusal -> the bucket stays new, memory -> off, FG's rules), then on
     deterministic synthetic static-buffer contents (_fill) FG's own fg._capture for every FG layer whose launch
     choices hold at S rows (_same_choices / _not_fused at m = S), then one agreement: every rank captured without an
     error -> the graphs wait in _PRE and the bucket stays FG's "capturing"; otherwise they are released on every
     rank and the bucket goes back to FG's lazy path (state new).
  Synthetic contents (FGP_SEED): normalized FFN input and residual streams N(0, 1) bf16, the FFN triplet positive
  (pre = sigmoid(N), post = 2 sigmoid(N)) with comb = Sinkhorn-normalized exp(N) (rows and columns sum to ~1), router
  logits = the first FG layer's own gate GEMM of that input (stock's cuBLAS torch.mm, fp32), text token ids in vocabulary
  range (never the image token), padding bound nt = S (every row real), so the levers' learn / verify steps the
  warmups trigger (FMT bands, MO / RT / FL / DT keys at S rows) see finite, non-degenerate data.
FIRST REAL USE (FG's exactness gate, unchanged). A pre-captured bucket is still "capturing": its first eligible forward
takes FG's capture path, where fg._ffn_capture takes the pre-captured graphs (take) instead of capturing, replays them
on the real rows, runs the eager path at M rows (what that call returns), compares every output byte on the GPU, and
fg._resolve agrees over the ranks before any graph result is used (mismatch / failure: the bucket off for good, FG's
rule). Graphs no layer took are released once their bucket leaves "capturing" (resolved).
RANKS. Every collective FGP adds is a gloo MIN over the TP CPU group (fg._agree) at a call site both ranks reach in the
same order: the plan agreement, one for the bands, and per bucket one budget + FG's admission + one capture agreement;
branches follow agreed values or agreed FG bucket states only. Everything between them is rank-local and collective-free
(FMT's check without the all-reduce; FG's warmups and captures hold no collective) and wrapped (_safe), so a rank-local
error is answered as a 0 in the next agreement, never a skipped collective, never a hang.
MEMORY. FG's: one set of static buffers (FG_MAX_ROWS rows) and one private pool reserved at the largest bucket, every
bucket admitted through FG's free-memory gate; pre-captured graphs count as live graphs of the pool (fg._live_graphs).
"""
import time
import types

import torch

from dsv41f import fg, levers

_REPS: list = []          # the layer replicas carrying FG's graph segment (fgp.install)
_PRE: dict = {}           # bucket S -> {id(_Layer): (_Layer, (graph 1, graph 2))} pre-captured, not taken yet
_P = {"armed": False, "done": False, "filled": False, "detail": "not installed", "taken": 0, "dropped": 0,
      "captured": 0, "graphs": 0, "agree": 0, "ms": 0.0, "errors": [], "bucket_ms": {}, "bands": {}, "lazy": [],
      "off": [], "layers": 0, "skipped": 0, "unresolved": 0, "fmt_ms": 0.0, "ops_ms": 0.0}


def _set(detail: str) -> str:
    _P["detail"] = detail
    return detail


def install(layers) -> str:
    """model_node.prepare, right after fg.install: remember FG's layer replicas and arm the hook in fg.plan."""
    if not getattr(levers, "FGP", False):
        return _set("off (levers.FGP)")
    if not fg._ST.get("on"):
        return _set(f"off (lever FG is off: {str(fg._ST.get('detail', ''))[:160]})")
    reps = []
    for rep in layers:
        mods = getattr(rep, "_modules", None)
        moe = mods.get("mlp") if isinstance(mods, dict) else None
        if moe is not None and getattr(moe, "dsv4f_fg_seg1", None) is not None:
            reps.append(rep)
    if not reps:
        return _set("off (no layer replica carries FG's graph segment)")
    _REPS[:] = reps
    fg._FGP[:] = [types.SimpleNamespace(due=due, precapture=precapture, take=take, resolved=resolved, live=live)]
    _P["armed"] = True
    budget = float(getattr(levers, "FGP_BUDGET_S", 120.0))
    return _set(f"armed: {len(reps)} FG layer replicas; every bucket of FG's table admitted + warmed + captured at the "
                f"engine's first eligible eager EXTEND forward (synthetic static-buffer contents, budget {budget:.0f} s); "
                f"first real use per bucket still checked bitwise vs eager")


# ------------------------------------------------------------------------------------------------------------- hooks
def due() -> bool:
    """fg.plan: the pre-capture has not run yet in this process."""
    return _P["armed"] and not _P["done"]


def live() -> int:
    """fg._live_graphs: pre-captured graphs hold FG's pool too."""
    return sum(2 * len(d) for d in _PRE.values())


def take(L, s):
    """fg._ffn_capture: ((graph 1, graph 2), 0 warmups) pre-captured for (layer, bucket s), removed from the cache;
    None: FG captures lazily as before."""
    d = _PRE.get(s)
    if not d:
        return None
    hit = d.pop(id(L), None)
    if not d:
        _PRE.pop(s, None)
    if hit is None:
        return None
    if hit[0] is not L:
        _reset(hit[1])
        return None
    _P["taken"] += 1
    return hit[1], 0


def resolved(fwd) -> None:
    """The end of fg._resolve: buckets no longer "capturing" (on or off now) release the pre-captured graphs no layer
    took (rank-local bookkeeping, no collective)."""
    n = 0
    for s in list(_PRE):
        if fg._BSTATE.get(s) != "capturing":
            for _L, graphs in _PRE.pop(s).values():
                _reset(graphs)
                n += 1
    if n:
        _P["dropped"] += n
        fg._pool_check()


def stats() -> dict:
    return {k: v for k, v in _P.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


# ----------------------------------------------------------------------------------------------------- the pre-capture
def precapture(fwd) -> None:
    """fg.plan's hook (once): pre-capture every new bucket for every FG layer. Never raises an Exception into the
    forward; FG's per-forward prep cache is cleared after (the static padding count / ids were rewritten)."""
    if not due():
        return
    _P["done"] = True
    try:
        _run(fwd)
    except Exception as e:  # noqa: BLE001 - an agreement itself failed (gloo): FG's lazy path from here
        _P["errors"].append(f"run: {type(e).__name__}: {str(e)[:200]}")
        levers.announce("fgp", f"lever FGP: stopped ({type(e).__name__}: {str(e)[:200]}); captured "
                        f"{_P['captured']} buckets, the rest stays on FG's lazy path")
    finally:
        fwd.nt_m, fwd.ids_src, fwd.ids_m = -1, None, -1


def _agree(site, values):
    """Every FGP agreement (gloo MIN over the TP CPU group, FG's form); `site` names the call site."""
    _P["agree"] += 1
    return fg._agree(values)


def _safe(what, fn, *args):
    """(True, fn(*args)) or (False, None): a rank-local error, answered as a 0 in the next agreement."""
    try:
        return True, fn(*args)
    except Exception as e:  # noqa: BLE001
        _P["errors"].append(f"{what}: {type(e).__name__}: {str(e)[:200]}")
        return False, None


def _run(fwd) -> None:
    """Both ranks make the same agreement calls in the same order: everything rank-local between two of them runs
    inside _safe (arguments evaluated inside too), and every branch follows agreed values or agreed FG states."""
    t0 = time.perf_counter()
    budget = float(getattr(levers, "FGP_BUDGET_S", 120.0))
    ok, plan = _safe("plan", _plan)
    good = bool(ok and plan is not None)
    if good:
        good, sig = _safe("signature", _signature, plan)
    if not good:
        sig = (0, 0, 0, 0)
    a = _agree("plan", [int(good)] + [v for x in sig for v in (x, -x)])
    same = all(a[1 + 2 * i] == -a[2 + 2 * i] for i in range(len(sig)))
    if not a[0] or not same:
        why = "the FG layers / bucket table differ between ranks" if a[0] else "no FG layer / bucket table on some rank"
        _finish(t0, fwd, 0, why)
        return
    Ls, t, todo, bands = plan
    _P["layers"] = len(Ls)
    to = time.perf_counter()
    _safe("check ops", _warm_check_ops, Ls[0])
    _P["ops_ms"] = round((time.perf_counter() - to) * 1e3, 1)
    # 1. lever FMT's band check (FG's admission needs fmt._BAND_OK True on every rank), rank-local, no collective
    tf = time.perf_counter()
    if bands:
        fmt = fg._H["fmt"]
        need = _agree("bands", [int(fmt._BAND_OK.get(b) is None) for b in bands])
        for i, b in enumerate(bands):
            if need[i]:
                _safe(f"FMT band {b}", _verify_band, Ls[0], t, b, fwd)
            _P["bands"][b] = fmt._BAND_OK.get(b)
    _P["fmt_ms"] = round((time.perf_counter() - tf) * 1e3, 1)
    # 2. every new bucket, largest first
    for e in todo:
        if fg._BSTATE.get(e) is not None:            # agreed FG state only
            continue
        go = _agree(("go", e), [int(time.perf_counter() - t0 < budget)])
        if not go[0]:
            _P["errors"].append(f"budget {budget:.0f} s reached before bucket {e}")
            break
        if not fg._admit(Ls[0], t, e, e, fwd):       # FG's agreed admission (refused: stays new; memory: off)
            continue
        tb = time.perf_counter()
        ok, got = _safe(f"bucket {e}", _capture_bucket, Ls, t, e)
        got = got if ok else []
        a = _agree(("bucket", e), [int(ok), len(got)])
        if a[0] and a[1] > 0:
            _PRE[e] = {id(L): (L, graphs) for L, graphs in got}
            _P["captured"] += 1
            _P["graphs"] += 2 * len(got)
        else:
            for _L, graphs in got:
                _reset(graphs)
            fg._BSTATE.pop(e, None)                   # FG's lazy path (new) on every rank
            fg._pool_check()
        _P["bucket_ms"][e] = round((time.perf_counter() - tb) * 1e3, 1)
    _finish(t0, fwd, len(todo), "")


def _plan():
    """(FG layers, FG's table, new bucket ends largest first, their FMT bands) or None. Deterministic on every rank:
    load-time facts and agreed FG bucket states only."""
    Ls = []
    for rep in _REPS:
        if getattr(rep, "local_boundary", None) is None:
            # resolved exactly as the replica's own forward resolves them (stock's _init_boundaries, idempotent,
            # load-time facts only); fg._layer must never see a layer before that (it caches the verdict)
            init = getattr(rep, "_init_boundaries", None)
            if callable(init):
                init()
            if getattr(rep, "local_boundary", None) is None:
                _P["unresolved"] += 1
                continue
        L = fg._layer(rep)
        if L:
            Ls.append(L)
    if not Ls:
        return None
    t = fg._table(Ls[0])
    if t is None:
        return None
    Ls = [L for L in Ls if fg._table(L) is t]
    todo = sorted((e for _s, e in t.buckets if fg._BSTATE.get(e) is None), reverse=True)
    if not todo:
        return None
    bands = sorted({t.key[e][0] for e in todo})
    return Ls, t, todo, bands


def _signature(plan):
    Ls, _t, todo, bands = plan
    h = 0
    for i, e in enumerate(todo):
        h = (h * 1000003 + (i + 1) * int(e)) % (1 << 40)
    for b in bands:
        h = (h * 1000003 + 7919 * (int(b) + 1)) % (1 << 40)
    return len(Ls), len(todo), len(bands), h


def _ids_into(out, moe, g):
    """Text token ids in vocabulary range (never the image token: vision routing stays off)."""
    cfg = getattr(moe, "config", None)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    if vocab < 2:
        vocab = 32000
    image = getattr(cfg, "image_token_id", None)
    out.random_(0, vocab, generator=g)
    if isinstance(image, int) and 0 <= image < vocab:
        out.masked_fill_(out == image, (image + 1) % vocab)
    return out


def _generator(dev, salt):
    g = torch.Generator(device=dev)
    g.manual_seed(int(getattr(levers, "FGP_SEED", 20261015)) + int(salt))
    return g


class _BatchView:
    """What the MoE replica's forward reads off its forward batch, for FMT's band check on synthetic rows: this
    forward's own mode (FMT's extend / decode test) and every row real (moe_num_token_non_padded = the row count)."""

    __slots__ = ("forward_mode", "_nt")

    def __init__(self, mode, nt):
        self.forward_mode = mode
        self._nt = nt

    def moe_num_token_non_padded(self):
        return self._nt


def _verify_band(L0, t, band, fwd) -> None:
    """Lever FMT's own first-call check of `band` (fmt.bind's wrapper on the MoE replica: stock's forward and the
    pinned one on the same input, every output byte compared, verdict in fmt._BAND_OK) at the band's smallest bucket
    row count, on synthetic rows. The post-experts all-reduce is left out (stock's mlp_reduce_scatter flag), so the
    check is rank-local and holds no collective."""
    rows = min(s for s, e in t.buckets if t.key[e][0] == band)
    dev = L0.gate_w.device
    g = _generator(dev, 7 + band)
    x = torch.empty((rows, L0.hidden), dtype=torch.bfloat16, device=dev).normal_(generator=g)
    ids_t = getattr(fwd.fb, "input_ids", None)
    ids = torch.empty(rows, dtype=ids_t.dtype if isinstance(ids_t, torch.Tensor) else torch.int64, device=dev)
    _ids_into(ids, L0.moe, g)
    nt = torch.full((), rows, dtype=torch.int32, device=dev)
    view = _BatchView(fwd.fb.forward_mode, nt)
    with fg._H["get_forward"]().scoped(mlp_reduce_scatter=True):
        L0.moe(x, view, input_ids=ids, input_ids_global=ids, skip_shared_experts=False, return_moe_output=False)


def _warm_check_ops(L0) -> None:
    """The ops of FG's first-use byte check that nothing else runs (fg._neq on 2- and 4-byte views, .any(), a 0-d bool
    flag and torch.logical_or into it), once on scratch tensors: their kernels' first load in a fresh process (measured
    1.5-2.8 s on B300, the first checked bucket's forward otherwise) happens here, not in the first timed forward that
    checks a bucket. Touches no FG state."""
    dev = L0.gate_w.device
    bad = torch.zeros((), dtype=torch.bool, device=dev)
    for dt in (torch.bfloat16, torch.float32, torch.int64):
        a = torch.zeros(8, dtype=dt, device=dev)
        d = fg._neq(a, a.clone())
        if d is not None:
            torch.logical_or(bad, d, out=bad)
    if bool(bad.item()):
        raise RuntimeError("check ops: equal tensors compared unequal")


def _fill(B, L0) -> None:
    """Deterministic synthetic contents of every row of FG's static FFN inputs (see the module docstring)."""
    g = _generator(B.x.device, 0)
    B.x.normal_(generator=g)
    B.res.normal_(generator=g)
    B.pre.normal_(generator=g)
    torch.sigmoid(B.pre, out=B.pre)
    B.post.normal_(generator=g)
    torch.sigmoid(B.post, out=B.post)
    B.post.mul_(2.0)
    B.comb.normal_(generator=g)
    B.comb.exp_()
    for _ in range(10):
        B.comb.div_(B.comb.sum(dim=2, keepdim=True))
        B.comb.div_(B.comb.sum(dim=1, keepdim=True))
    torch.mm(B.x, L0.wt, out_dtype=torch.float32, out=B.logits)
    _ids_into(B.ids, L0.moe, g)


def _capture_bucket(Ls, t, e):
    """Rank-local, no collective: FG's warmups + captures of bucket e for every FG layer whose launch choices hold at e
    rows, on the synthetic contents. [(layer, graphs)]; an error releases what this bucket captured and re-raises."""
    B = fg._BUF[0]
    if not _P["filled"]:
        _fill(B, Ls[0])
        _P["filled"] = True
    B.nt.fill_(e)
    got = []
    try:
        for L in Ls:
            if not (fg._same_choices(L, t, e, e) and fg._not_fused(L.layer, e, L)):
                _P["skipped"] += 1
                continue
            graphs, _nwarm = fg._capture(L, e)
            got.append((L, graphs))
    except BaseException:
        torch.cuda.current_stream().wait_stream(fg._stream())
        for _L, graphs in got:
            _reset(graphs)
        raise
    return got


def _reset(graphs) -> None:
    for g in graphs:
        try:
            g.reset()
        except Exception:  # noqa: BLE001
            pass


def _finish(t0, fwd, n_new, why) -> None:
    try:
        torch.cuda.current_stream().synchronize()     # the pre-capture's GPU work inside the reported time
    except Exception:  # noqa: BLE001
        pass
    dt = time.perf_counter() - t0
    _P["ms"] = round(dt * 1e3, 1)
    lazy = sorted(s for _lo, s in (fg._TABLE[0].buckets if fg._TABLE and fg._TABLE[0] else ())
                  if fg._BSTATE.get(s) is None)
    off = sorted(s for s, v in fg._BSTATE.items() if v == "dead")
    _P["lazy"], _P["off"] = lazy, off
    ids = getattr(fwd.fb, "input_ids", None)
    rows = int(ids.shape[0]) if isinstance(ids, torch.Tensor) and ids.dim() >= 1 else -1
    errs = f"; rank-local errors: {' | '.join(_P['errors'])[:400]}" if _P["errors"] else ""
    detail = (f"lever FGP: {_P['captured']}/{n_new} buckets pre-captured ({_P['layers']} FG layers, {_P['graphs']} "
              f"graphs, {_P['skipped']} layer-buckets left to FG's lazy path, {_P['unresolved']} replicas unresolved) "
              f"in {dt:.2f} s at the first eligible eager EXTEND forward ({rows} rows; FMT band checks "
              f"{_P['fmt_ms'] / 1e3:.2f} s, check ops {_P['ops_ms'] / 1e3:.2f} s); FMT bands {_P['bands']}; "
              f"lazy {lazy}; off {off}{'; ' + why if why else ''}{errs}")
    _set(detail)
    levers.announce("fgp", detail)
