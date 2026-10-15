"""Lever AG on the fork (levers.AG): the post-attention segment of our eager EXTEND attention replayed from one CUDA
graph per (layer, row bucket) -- lever FG's design (fg.py) for the attention half.

WHY. With lever FG the FFN half of an eager EXTEND layer is two graph replays; the attention half now dominates the
host of a warm-turn prefill (contexts above 16384 tokens run eager, host-bound on TP-1). CPU + GPU profile of c16 at
2000 rows on 60k cached (box1 prof_fork_c16cp, profiler-inflated ~1.7x): the lifted attention forward ~2.4 ms a layer,
of which the tail after the sparse-MLA kernel -- RO's inverse RoPE (0.05 ms), RT's WO-A strided bmm (0.13), wo_b
(RowParallelLinear + Fp8LinearMethod.apply + lever FL: MXFP8 quantize + CuTe-DSL GEMM, 0.13 + 0.04 of wrappers), the
tail's branch tests and _project_wo_b (~0.05) -- is ~0.4 ms of host for four kernels.

WHAT RUNS (per eligible attention call, M real rows, bucket S >= M; fork_attn.attn_forward carries two guarded edits,
ct_pf.forward_trtllm_prefill one, all from gen_ag.py):
  plan        fork_attn, right before the attention kernel: ag.plan(...) -> this call's plan, or None (stock's path,
              unchanged).
  attention   eager at M rows as stock (lever MF's sparse-MLA launch), its output written straight into AG's static
              input buffer [:M]: ct_pf passes ag.fmha_out(...) as stock's own `out` argument (the form stock's padded
              path uses); every other argument is stock's.
  graph (S)   stock's tail for this call's branch (fork_attn: no fused WO-A, no HIP / NPU route, bf16 WO-A) on the
              static buffers: lever RO's inverse RoPE in place (positions from AG's static copy), lever RT's WO-A strided
              bmm, wo_b = the replica's RowParallelLinear.forward with skip_all_reduce=True (lever FL / DT: MXFP8
              quantize + CuTe-DSL GEMM at the band's tactic), copied into AG's static output.
  all-reduce  eager on the real rows [:M]: RowParallelLinear.forward's own reduce branch (the same group call, lever
              AR's entry) and _project_wo_b's return; that is what attn_forward returns.
  Collectives never run inside a graph. A graph is captured on AG's capture stream after warmups at S until the
  levers' learn / verify state (RT / RO / DT / FL) stops changing, so the capture holds the launches the eager path
  makes at S rows.

EXACTNESS. Every output byte of the M real rows equals the eager path at M rows:
  * outside the graph everything runs at M rows as stock does (the attention kernel writes the same bytes into another
    buffer of the same layout: same strides, 256-byte aligned base; the all-reduce runs on the real rows);
  * inside the graph every kernel is row-independent: the inverse RoPE is per (token, head, pair) with the token's own
    position; the WO-A bmm is cuBLAS's strided batched GEMM over tokens; the MXFP8 quantize is per row (32-element
    blocks; the 128x4-swizzled scale layout puts a row's scales at a position independent of the row count); the
    CuTe-DSL GEMM runs one fixed split-1 tactic; padded rows (M..S-1, stale or poisoned) only reach padded output rows;
  * every launch decision the eager path takes from the row count is equal for M and S: buckets never cross an edge of
    lever DT's band for wo_b (= lever FL's plan key with its 2T/SF choice), mxfp8_quantize_cute_dsl's 2T/SF threshold,
    lever RO's tile band, the fused-WO-A bound (32 rows), the prefill autotune bound (no bucket reaches it: wo_b's
    call carries no pin_tactic) and RT's strided-bmm range (<= 4095 rows);
  * cuBLAS picks the WO-A kernel per row count, so a bucket is admitted only after a synthetic sweep on this rank
    (deterministic random inputs, the real WO-A weight, the eager call's exact operand layout): bmm at every row count
    m of the bucket == the first m rows of bmm at S, byte for byte (B300, cuBLAS of torch 2.14.1+cu130: one kernel
    class over 385..8192 rows, results/ag/agp1); and a replay needs RT's own first-call check (strided bmm == stock's
    einsum) passed for both M's and S's row class on this rank;
  * the first replay of every (bucket, layer) is compared with the eager path at M rows byte for byte on every rank
    (the eager result is what that call returns) and the bucket goes live only if every layer passed on every rank;
    any difference turns the bucket off for good;
  * proven in tests/ag_test.py over many M per bucket (edges, random, poisoned padded rows) against stock's tail (the
    lifted fork_attn source, run as is) and in the engine by the first-use check.
SCOPE. Our node's eager EXTEND forwards only (no capture, no breakable / piecewise prefill graph, not under SGLang's
capture mode, not compiling), never when the engine runs with CUDA graphs disabled (the node audit's eager twin stays
plain eager), AG_MIN_ROWS..AG_MAX_ROWS rows, our private attention replicas (fork_attn.attn_forward bound) whose call
takes the uniform-FP8 trtllm path with an unpadded query (q_out FP8, no head padding), no unified KV, no multi-stream
prepare, no deferred all-reduce, no sequence parallelism, no symmetric memory, attention TP == TP (no DP / CP
attention: every TP rank must attend the same rows for the agreements below).
RANKS. Every decision from rank-local state (free memory, the WO-A sweep, capture success, the first-replay check,
lever RT / DT verification states) is agreed over the TP CPU group (gloo MIN) at two call sites both ranks reach under
the same deterministic conditions (rows, static layer facts, agreed bucket states): a new bucket's admission (two
agreements) and the end of its capture forward (one). Between them every path of a rank (capture, replay, stock) makes
the same single wo_b all-reduce, so a rank-local difference can neither hang nor diverge the ranks.
MEMORY. One set of static buffers (the attention output [AG max bucket, heads, 512] bf16, wo_b's output, positions;
shared by every bucket and layer: layers run one after the other on one stream) and one private graph pool shared by
every AG graph (no graph output lives in the pool; the first capture reserves it at the largest bucket), admitted per
bucket by a free-memory gate.
MEASURED (2026-10-15, box1 B300, fork image 5e72ce5, results/ag/):
  tests/ag_woa_probe.py (agp1): the strided WO-A bmm at every row count 385..8192 == the first rows of the bmm at 8192
  (one cuBLAS signature class, 7808 / 7808 prefixes equal).
  tests/ag_test.py (ag2 / ag3, TP2 per-rank shapes, TP1 math, real MQALayer objects + RowParallelLinear MXFP8 wo_b,
  both RoPE tables on two layers): 590 / 590 AG calls == stock's tail (the lifted fork_attn tail run as is) over 467
  row counts in 12 buckets (every row count of 385..511 and 2049..2303; 18 first-use captures + checks, 572 replays,
  191 with NaN / inf / 3e38 in every padded input row, 52 also == pure stock without levers); WO-A sweep of all 3711
  row counts consistent (16 s at install, ~4.4 ms a row count: cuBLAS's per-shape heuristics); a forced capture
  failure, a corrupted first replay, simulated other-rank failures (sweep / capture / captured nothing / memory) and
  8 empty capture forwards each turn the bucket off with exact results; a rank-local RT state declines the replay on
  that rank only. Host per attention tail 219-220 us eager -> 96-98 us AG (incl. the test's input copy and forward
  context); GPU span 256 -> 157 us (700 rows), 259 -> 182 (2000), 281 -> 283 (4000). Static buffers 168 MiB (4095
  rows), graph pool 90 MiB for all buckets.
  In engine (box1 dev lane, TP2, unprofiled + seal probe, single warm turns on 59904 cached tokens): install-time sweep
  1.4 s on each rank (12 / 12 buckets consistent); every bucket's capture forward: 40 layers' graphs, first replay ==
  eager bit for bit on every rank (buckets 895 / 2048 / 2303 / 4095); seal clean after prepare and forwards 1..300.
  ag18a (c18a, pf 2000 = 2096 rows): capture forward 0.214 s once, then 0.088-0.092 s (median 0.0916; FG-era cached
  reads 0.092-0.099). In-engine A/B (levers.AG_AB = 2): 796 rows AG on 0.0561 / 0.0564 / 0.0575 / 0.0777 vs off
  0.0604 / 0.0608 / 0.0615 / 0.0613 s (median -6.8 %); 3996 rows on 0.1035-0.1046 vs off 0.1021-0.1041 s (median
  +0.9 %, GPU-bound: neutral within noise). A bucket's first use costs +0.12..0.15 s once (lazy capture).
"""
import contextvars
import time

import torch

from dsv41f import levers

_ON = contextvars.ContextVar("dsv41f_ag_on", default=None)   # per eligible forward: _Fwd
_CUR: list = [None]       # the plan of the attention call in flight (ct_pf's fmha_out reads it; post clears it)
_H: dict = {}            # handles imported at install
_ST = {"detail": "not installed", "on": False, "dead": False, "replay": 0, "capture": 0, "check": 0, "eager": 0,
       "fail": 0, "admit": 0, "refuse": 0, "warm": 0, "cap_ms": 0.0, "fwd": 0, "sweep": 0, "sweep_ms": 0.0,
       "local_eager": 0, "ab": 0}
_LAYERS: dict = {}       # id(attention replica) -> (replica, _Layer | False)
_BSTATE: dict = {}       # bucket S -> "capturing" | "active" | "dead" (absent: new)
_WARM: dict = {}         # (bucket S, freqs table) -> the eligible forward (_Fwd.seq) whose capture warmed the levers up
_RETRY: dict = {}        # bucket S -> capture forwards that ended with no layer captured on any rank (agreed count)
_SWEPT: dict = {}        # bucket S -> True / False: the WO-A sweep on this rank
_TABLE: list = []        # [_Table] once built
_BUF: list = []          # [_Buf] once allocated
_RES: dict = {}          # "pool" / "stream" / "reserve" once created
_KEYS: dict = {}         # rows -> decision key (False: unreadable)

# sha256[:16] of the stock functions AG mirrors or relies on (fork 5e72ce5, FlashInfer 0.7.1); levers.AG_PINS
PIN_TARGETS = (
    ("sglang.srt.models.deepseek_v4", "MQALayer._project_wo_b"),
    ("sglang.srt.models.deepseek_v4", "MqaAttentionBase.use_flashinfer_mxfp8_wo_b"),
    ("sglang.srt.models.deepseek_v4", "_apply_wo_a_bf16_matmul"),
    ("sglang.srt.layers.linear", "RowParallelLinear.forward"),
    ("sglang.srt.layers.quantization.fp8", "Fp8LinearMethod.apply"),
    ("sglang.srt.distributed.device_communicators.pynccl_allocator", "use_symmetric_memory"),
    ("sglang.srt.layers.attention.deepseek_v4_backend", "DeepseekV4AttnBackend.forward"),
    ("sglang.srt.layers.attention.deepseek_v4_backend", "DeepseekV4AttnBackend._forward_attention"),
    ("sglang.srt.layers.attention.deepseek_v4_trtllm_backend", "DeepseekV4TrtllmAttnBackend._forward_trtllm"),
    ("sglang.srt.layers.attention.deepseek_v4_trtllm_backend", "DeepseekV4TrtllmAttnBackend._forward_trtllm_prefill"),
    ("flashinfer.quantization.kernels.mxfp8_quantize", "mxfp8_quantize_cute_dsl"),
)


def _set(detail: str) -> str:
    _ST["detail"] = detail
    return detail


def source_hash(text: str, qualname: str):
    """sha256 (16 hex) of a def's source text (decorators included, dedented) in a module's source TEXT, found by ast
    (the last def of that name wins); None if absent. Same algorithm as fg.py / gen_ag.py."""
    import ast
    import hashlib
    import textwrap

    if not text.endswith("\n"):
        text += "\n"
    body = ast.parse(text).body
    node = None
    for part in qualname.split("."):
        node = None
        for n in body:
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part:
                node = n
        if node is None:
            return None
        body = node.body
    lines = text.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    seg = textwrap.dedent("".join(lines[start - 1:node.end_lineno]))
    return hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]


def _module_source(name: str) -> str:
    """A pinned module's source text by dotted name: static imports only (the bundle scanner bans importlib / open)."""
    import inspect

    if name == "sglang.srt.models.deepseek_v4":
        from sglang.srt.models import deepseek_v4 as mod
    elif name == "sglang.srt.layers.linear":
        from sglang.srt.layers import linear as mod
    elif name == "sglang.srt.layers.quantization.fp8":
        from sglang.srt.layers.quantization import fp8 as mod
    elif name == "sglang.srt.distributed.device_communicators.pynccl_allocator":
        from sglang.srt.distributed.device_communicators import pynccl_allocator as mod
    elif name == "sglang.srt.layers.attention.deepseek_v4_backend":
        from sglang.srt.layers.attention import deepseek_v4_backend as mod
    elif name == "sglang.srt.layers.attention.deepseek_v4_trtllm_backend":
        from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as mod
    elif name == "flashinfer.quantization.kernels.mxfp8_quantize":
        from flashinfer.quantization.kernels import mxfp8_quantize as mod
    else:
        raise KeyError(f"no static import for pinned module {name}")
    if mod.__name__ != name:
        raise ImportError(f"{name} resolved to {mod.__name__}")
    return inspect.getsource(mod)


def check_pins() -> str:
    """'' when every pinned stock function's source hash equals levers.AG_PINS, else what differs."""
    bad = []
    texts: dict = {}
    for mod_name, qual in PIN_TARGETS:
        want = levers.AG_PINS.get(f"{mod_name}:{qual}")
        try:
            if mod_name not in texts:
                texts[mod_name] = _module_source(mod_name)
            got = source_hash(texts[mod_name], qual)
        except Exception as e:  # noqa: BLE001
            got = type(e).__name__
        if want is None or got != want:
            bad.append(f"{qual} {got} != {want}")
    return "; ".join(bad)


# ------------------------------------------------------------------------------------------------------------- install
def _import_handles() -> None:
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators import pynccl_allocator
    from sglang.srt.layers import linear
    from sglang.srt.model_executor import cuda_graph_config as cgc
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
        is_in_tc_piecewise_cuda_graph,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
    from sglang.srt.models import deepseek_v4 as dv4
    from sglang.srt.runtime_context import get_forward

    import flashinfer
    from flashinfer.quantization.kernels import mxfp8_quantize as mq

    from dsv41f import dt, fl, fork_attn, ro, rt

    if flashinfer.__version__ != "0.7.1":
        raise RuntimeError(f"flashinfer {flashinfer.__version__} (mirror read from 0.7.1)")
    for mod, names in ((linear, ("RowParallelLinear", "get_group_rank_size", "should_skip_mlp_all_reduce",
                                 "get_parallel", "tensor_model_parallel_all_reduce")),
                       (pynccl_allocator, ("is_symmetric_memory_enabled",)), (mq, ("MXFP8_2T_SF_THRESHOLD",)),
                       (dv4, ("MQALayer", "_FUSED_WO_A_MAX_TOKENS", "_apply_wo_a_bf16_matmul")),
                       (rt, ("wo_a", "_WOK", "_ON", "_ST", "WOA_MAX_ROWS", "WOA_SHAPE")),
                       (ro, ("inv_rope", "_OK", "_SELF")), (dt, ("_band", "_KEY_OK")), (fl, ("_STATE", "_PLANS")),
                       (fork_attn, ("attn_forward", "_apply_wo_a_bf16_matmul")),
                       (cgc, ("check_cuda_graph_backend", "cuda_graph_fully_disabled"))):
        for n in names:
            if not hasattr(mod, n):
                raise AttributeError(f"{getattr(mod, '__name__', mod)}.{n}")
    if fork_attn._apply_wo_a_bf16_matmul is not dv4._apply_wo_a_bf16_matmul:
        raise RuntimeError("fork_attn does not call stock's _apply_wo_a_bf16_matmul")
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

    _H.update(tp_group=get_tp_group, symm=pynccl_allocator.is_symmetric_memory_enabled, linear=linear, cgc=cgc,
              Phase=Phase, Backend=Backend, EXTEND=ForwardMode.EXTEND, in_bcg=is_in_breakable_cuda_graph,
              in_pcg=is_in_tc_piecewise_cuda_graph, capture_mode=get_is_capture_mode, dv4=dv4,
              get_forward=get_forward, two_t=int(mq.MXFP8_2T_SF_THRESHOLD), fused_max=int(dv4._FUSED_WO_A_MAX_TOKENS),
              fa=fork_attn, ro=ro, rt=rt, dt=dt, fl=fl, Row=linear.RowParallelLinear)


def _graphs_disabled() -> str:
    """Why the engine runs without CUDA graphs ('' when it does not): AG stays off then, so cacheon's eager twin
    (engine_worker: disable_cuda_graph=True for the audit role) runs plain eager (fg._graphs_disabled's test)."""
    cgc, Phase, Backend = _H["cgc"], _H["Phase"], _H["Backend"]
    try:
        if cgc.cuda_graph_fully_disabled():
            return "CUDA graphs disabled on every phase"
        for phase in (Phase.DECODE, Phase.PREFILL):
            if cgc.check_cuda_graph_backend(phase, Backend.DISABLED):
                return f"CUDA graphs disabled for {phase}"
        if not any(cgc.check_cuda_graph_backend(Phase.PREFILL, b) for b in (Backend.BREAKABLE, Backend.FULL)):
            return "prefill CUDA-graph backend not published as breakable / full"
    except Exception as e:  # noqa: BLE001 - unknown: assume disabled
        return f"CUDA-graph config unreadable ({type(e).__name__})"
    return ""


def install(model, layers) -> str:
    """model_node.prepare: AG binds nothing (its call sites are fork_attn's / ct_pf's, bound on our private replicas
    only); this pins the stock functions it mirrors and counts the attention replicas that can take it."""
    if not getattr(levers, "AG", False):
        return _set("off (levers.AG)")
    if _ST["on"]:
        return _ST["detail"]
    try:
        _import_handles()
    except Exception as e:  # noqa: BLE001 - another stack: stock everywhere
        return _set(f"off ({type(e).__name__}: {str(e)[:200]})")
    why = _graphs_disabled()
    if why:
        return _set(f"off ({why}: the eager twin stays plain eager)")
    if getattr(levers, "AG_PIN", True):
        bad = check_pins()
        if bad:
            return _set(f"off (stock changed: {bad[:400]})")
    if not _H["rt"]._ST.get("woa_on"):
        return _set("off (lever RT's WO-A strided bmm is off: the graph holds that form)")
    n = 0
    attns = []
    for rep in layers:
        attn = rep._modules.get("self_attn") if hasattr(rep, "_modules") else None
        if attn is not None and getattr(getattr(attn, "forward", None), "__func__", None) is _H["fa"].attn_forward:
            n += 1
            attns.append(attn)
    _RES["attns"] = tuple(attns)
    _ST["on"] = n > 0
    pre = _presweep() if _ST["on"] else ""
    return _set(f"{'on' if _ST['on'] else 'off'}: {n} attention replicas run the lifted forward; {pre}; eager EXTEND calls "
                f"of {levers.AG_MIN_ROWS}..{levers.AG_MAX_ROWS} rows replay the post-attention segment (RO inverse "
                f"RoPE, RT WO-A, wo_b MXFP8 quantize + GEMM) from one graph per (layer, bucket <= "
                f"{levers.AG_PAD:.0%} padding inside constant-choice regions), wo_b's all-reduce eager; buckets "
                f"admitted after a WO-A cuBLAS row-consistency sweep, captured lazily, first replay per (bucket, "
                f"layer) checked bitwise vs eager{'' if levers.AG_CHECK else ' (CHECK OFF)'}")


# -------------------------------------------------------------------------------------------------------- forward gate
class _Fwd:
    """One eligible forward: what AG already set up for it."""

    __slots__ = ("fb", "refused", "pos_src", "pos_m", "capturing", "pending", "fail", "bad", "rows", "seq")

    def __init__(self, fb):
        self.fb = fb
        _ST["fwd"] += 1
        self.seq = _ST["fwd"]     # this eligible forward's number (a bucket's warmup is redone in each capture forward)
        self.refused = set()      # buckets whose admission was refused in this forward
        self.pos_src = None       # the positions tensor copied into the static positions
        self.pos_m = -1
        self.capturing = set()    # buckets in their capture forward (state "capturing", agreed): resolved at the end
        self.pending = {}         # bucket -> [(layer state, graph)] captured + replayed in this forward
        self.fail = {}            # bucket -> why a capture / check step failed on this rank
        self.bad = {}             # bucket -> 0-d GPU bool: some graph byte != the eager byte
        self.rows = {}            # bucket -> the real row count it was captured at


def forward(fn, forward_batch, *args, **kwargs):
    """Run the rest of our model forward with lever AG active on eager EXTEND forwards. Every test here is the same on
    both TP ranks (mode, capture state, the batch, the engine config); a rank-local failure (_ST["dead"]) does not end
    the context early: the other rank may be heading for an agreement, which this rank then answers with False."""
    if not _ST["on"]:
        return fn(*args, **kwargs)
    mode = getattr(forward_batch, "forward_mode", None)
    if (mode is None or mode != _H["EXTEND"] or torch.cuda.is_current_stream_capturing() or _H["in_bcg"]()
            or _H["in_pcg"]() or _H["capture_mode"]() or torch.compiler.is_compiling()):
        return fn(*args, **kwargs)
    if getattr(levers, "AG_AB", 0) and not _ab_on(forward_batch):
        return fn(*args, **kwargs)
    fwd = _Fwd(forward_batch)
    token = _ON.set(fwd)
    try:
        out = fn(*args, **kwargs)
    finally:
        _ON.reset(token)
        _CUR[0] = None
    if fwd.capturing:
        _resolve(fwd)              # both ranks: the bucket was "capturing" (an agreed state) when this forward met it
    return out


def _presweep():
    """The WO-A sweep (_sweep) of every bucket of the table, once, at install (model_node.prepare: engine startup, the
    weights loaded, before serving) within AG_PRESWEEP_BUDGET_S, on a temporary input buffer freed afterwards (the KV
    pool is sized later, from free memory). cuBLAS's per-shape heuristics make a sweep ~1-4 ms a row count (3711 row
    counts: 4..15 s), which would otherwise fall into a timed request at each bucket's admission. Rank-local work (no
    collective): the results are agreed at each bucket's admission; a bucket past the budget is swept there."""
    _RES["presweep"] = ()
    if not getattr(levers, "AG_PRESWEEP", True) or not getattr(levers, "AG_WOA_SWEEP", True):
        return "presweep off"
    t0 = time.perf_counter()
    try:
        L = next((x for x in (_layer(a) for a in _RES.get("attns", ())) if x), None)
        if L is None:
            return "presweep: no eligible attention replica"
        t = _table(L)
        if t is None or not t.buckets:
            return "presweep: no bucket table"
        need = t.max_s * L.heads * L.hdim * 2 + 2 * t.max_s * L.groups * L.rank * 2
        free, _total = torch.cuda.mem_get_info()
        if free - need < int(levers.AG_MEM_MARGIN):
            return f"presweep skipped (free memory {free / 2**30:.1f} GiB)"
        xbuf = torch.empty(t.max_s, L.heads, L.hdim, dtype=torch.bfloat16, device=L.woa_w.device)
        done = rows = 0
        for lo, s_ in t.buckets:
            if time.perf_counter() - t0 > float(levers.AG_PRESWEEP_BUDGET_S):
                break
            _sweep(L, t, s_, xbuf)
            done += 1
            rows += s_ - lo + 1
        del xbuf
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        ms = (time.perf_counter() - t0) * 1e3
        _RES["presweep"] = (done, len(t.buckets), round(ms))
        return (f"WO-A sweeps of {done}/{len(t.buckets)} buckets ({rows} row counts) in {ms:.0f} ms "
                f"({ms * 1e3 / max(1, rows):.0f} us a row count): {sum(1 for v in _SWEPT.values() if v)} consistent, "
                f"{sum(1 for v in _SWEPT.values() if not v)} not")
    except Exception as e:  # noqa: BLE001 - nothing lost: admission sweeps lazily
        return f"presweep failed ({type(e).__name__}: {str(e)[:160]})"


def _ab_on(forward_batch):
    """Diagnostic A/B inside one engine (levers.AG_AB = N > 0, never in a shipped bundle): of the eager EXTEND forwards
    with AG_MIN_ROWS..AG_MAX_ROWS rows, only every N-th runs with AG (the others take the eager path), announced per
    forward. The count is the same on both ranks (same batches, same tests), so both ranks take the same branch."""
    ids = getattr(forward_batch, "input_ids", None)
    rows = -1 if ids is None else int(ids.shape[0])
    if not levers.AG_MIN_ROWS <= rows <= levers.AG_MAX_ROWS:
        return True
    _ST["ab"] += 1
    on = _ST["ab"] % int(levers.AG_AB) == 0
    levers.announce(f"ag-ab{_ST['ab']}", f"lever AG A/B: eligible forward {_ST['ab']} ({rows} rows): AG "
                    f"{'on' if on else 'off'}")
    return on


# --------------------------------------------------------------------------------------------- per-layer static state
class _Layer:
    __slots__ = ("attn", "radix", "wo_b", "woa_w", "groups", "rank", "heads", "hdim", "rope", "fuse_q", "fused_woa",
                 "n_out", "k_in", "sfb", "wkey", "graphs", "sig")


def _layer(attn):
    hit = _LAYERS.get(id(attn))
    if hit is not None and hit[0] is attn:
        return hit[1]
    try:
        L, why = _bind_layer(attn)
    except Exception as e:  # noqa: BLE001 - deterministic on every rank (same module structure)
        L, why = False, f"{type(e).__name__}: {str(e)[:160]}"
    if not L:
        levers.announce(f"ag-layer-{why[:40]}", f"lever AG: attention layer {getattr(attn, 'layer_id', '?')} stays "
                        f"eager ({why})")
    _LAYERS[id(attn)] = (attn, L)
    return L


def _bind_layer(attn):
    """(_Layer, '') when this attention replica's tail takes the graphs, else (False, why). Static (load-time) facts."""
    dv4, rt = _H["dv4"], _H["rt"]
    if not isinstance(attn, dv4.MQALayer) or getattr(getattr(attn, "forward", None), "__func__", None) is not \
            _H["fa"].attn_forward:
        return False, "not one of our attention replicas (fork_attn.attn_forward)"
    if not attn.is_dsv41 or attn.q_head_norm or attn.wo_a_fp8 or getattr(attn, "use_npu_arch35_mxfp8_wo_a", True):
        return False, "attention tail is not the DSV4.1 bf16 WO-A path"
    if getattr(_H["fa"], "_is_hip", True) or getattr(_H["fa"], "_is_npu", True) or not getattr(_H["fa"], "_is_cuda",
                                                                                              False):
        return False, "not the CUDA build"
    w = attn.wo_a._parameters.get("weight") if hasattr(attn.wo_a, "_parameters") else None
    g, r = int(attn.n_local_groups), int(attn.o_lora_rank)
    if not (isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_contiguous()
            and w.shape[0] == g * r and (g, r, int(w.shape[1])) == tuple(rt.WOA_SHAPE)):
        return False, "WO-A weight is not the bf16 TP2 shape lever RT takes"
    heads, hdim, rope = int(attn.n_local_heads), int(attn.head_dim), int(attn.qk_rope_head_dim)
    if hdim != 512 or heads * hdim != g * int(w.shape[1]) or rope != 64:
        return False, "head layout (heads x 512 = groups x group width, rope 64)"
    wo_b = attn._modules.get("wo_b")
    if not isinstance(wo_b, _H["Row"]) or type(wo_b).forward is not _H["Row"].forward or wo_b._forward_hooks \
            or wo_b._forward_pre_hooks:
        return False, "wo_b is not a hook-free RowParallelLinear"
    meth = getattr(wo_b, "quant_method", None)
    wb = wo_b._parameters.get("weight")
    if (type(meth).__name__ != "Fp8LinearMethod" or getattr(meth, "w8a8_mxfp8_linear", None) is None
            or not isinstance(wb, torch.Tensor) or wb.dtype != torch.float8_e4m3fn or wb.dim() != 2
            or getattr(wo_b, "bias", None) is not None or not wo_b.input_is_parallel or not getattr(wo_b, "dsv4f_dt",
                                                                                                    False)):
        return False, "wo_b is not lever DT's MXFP8 linear (no bias, parallel input)"
    n, k = int(wb.shape[0]), int(wb.shape[1])
    if k != g * r or not levers.DT_TACTICS.get((n, k)):
        return False, f"wo_b ({n}, {k}) without lever DT bands"
    fq = bool(attn.use_flashinfer_mxfp8_wo_b)
    radix = attn._modules.get("attn_mqa")
    if radix is None:
        return False, "no attn_mqa"
    L = _Layer()
    L.attn, L.radix, L.wo_b = attn, radix, wo_b
    L.woa_w = w.view(g, r, -1)
    L.groups, L.rank, L.heads, L.hdim, L.rope = g, r, heads, hdim, rope
    L.fuse_q, L.fused_woa = fq, bool(attn.use_fused_wo_a)
    L.n_out, L.k_in, L.sfb = n, k, k // 32
    fc = attn.freqs_cis
    L.wkey = (fc.data_ptr(), tuple(fc.shape)) if isinstance(fc, torch.Tensor) else None
    L.graphs = {}
    L.sig = (g, r, heads, hdim, n, k, fq, L.fused_woa, str(w.device))
    if L.wkey is None:
        return False, "no freqs table"
    return L, ""


# ---------------------------------------------------------------------------------------------- the decision key / table
def _ro_tile(m, heads):
    """lever RO's tile for m rows (ro._tile's formula on levers.RO_TILES: first band holding m, CTA capped at 512
    threads)."""
    tiles = sorted(tuple(int(v) for v in t) for t in levers.RO_TILES)
    tile = tiles[-1]
    for t in tiles:
        if m <= t[0]:
            tile = t
            break
    _, it, tok, fx = tile
    return it, max(1, min(tok, 512 // (8 * heads // it))), fx


def _key(L, m):
    """Every launch decision the eager tail takes from the row count: lever DT's wo_b band (FL's plan key), the 2T/SF
    quantize choice, lever RO's tile, the fused-WO-A bound."""
    band = _H["dt"]._band(m, L.n_out, L.k_in)[0]
    return (band, m * L.sfb >= _H["two_t"], _ro_tile(m, L.heads), L.fused_woa and 0 < m <= _H["fused_max"])


class _Table:
    __slots__ = ("sig", "buckets", "los", "key", "min_tok", "max_s")


def _table(L):
    """The bucket table (built once, at the first eligible call: every input is a load-time / startup fact)."""
    if _TABLE:
        t = _TABLE[0]
        return t if t is not None and t.sig == L.sig else None
    try:
        t = _build_table(L)
    except Exception as e:  # noqa: BLE001 - deterministic on every rank
        levers.announce("ag-table", f"lever AG: bucket table failed ({type(e).__name__}: {str(e)[:160]}): off")
        t = None
    _TABLE.append(t)
    return t


def _build_table(L):
    lo = int(levers.AG_MIN_ROWS)
    min_tok = L.wo_b.quant_method.mxfp8_prefill_autotune_min_tokens
    hi = min(int(levers.AG_MAX_ROWS), int(_H["rt"].WOA_MAX_ROWS))
    if min_tok is not None:
        hi = min(hi, int(min_tok) - 1)       # from the bound on wo_b's call carries pin_tactic (another launch path)
    pad, align = float(levers.AG_PAD), max(1, int(levers.AG_ALIGN))
    keys = [_key(L, m) for m in range(lo, hi + 1)] if hi >= lo else []
    buckets = []
    a = lo
    while a <= hi:
        b = a
        while b + 1 <= hi and keys[b + 1 - lo] == keys[a - lo]:
            b += 1
        if keys[a - lo][0] is not None and not keys[a - lo][3]:     # outside every DT band / fused WO-A: eager
            s = a
            while s <= b:
                want = -(-int(s * (1.0 + pad)) // align) * align
                end = min(b, max(s, want))
                buckets.append((s, end))
                s = end + 1
        a = b + 1
    t = _Table()
    t.sig = L.sig
    t.buckets = buckets
    t.los = [x[0] for x in buckets]
    t.key = {e: keys[e - lo] for _s, e in buckets}
    t.min_tok = min_tok
    t.max_s = max((e for _s, e in buckets), default=0)
    regions = sum(1 for i in range(1, len(keys)) if keys[i] != keys[i - 1]) + 1 if keys else 0
    levers.announce("ag-table", f"lever AG: {len(buckets)} buckets over {lo}..{hi} rows in {regions} constant-choice "
                    f"regions (edges: DT wo_b bands, 2T/SF at {_H['two_t']} SF blocks, RO tiles, prefill autotune "
                    f"bound {min_tok}); bucket ends {[e for _s, e in buckets]}")
    return t


def _bucket(t, m):
    import bisect

    i = bisect.bisect_right(t.los, m) - 1
    if i < 0:
        return None
    s, e = t.buckets[i]
    return e if s <= m <= e else None


def _key_m(L, m):
    hit = _KEYS.get(m)
    if hit is None:
        try:
            hit = _key(L, m)
        except Exception:  # noqa: BLE001 - never a graph at m rows
            hit = False
        if len(_KEYS) >= 65536:
            _KEYS.clear()
        _KEYS[m] = hit
    return hit if hit is not False else None


# --------------------------------------------------------------------------------------------------------- the plan
class _Plan:
    __slots__ = ("L", "rows", "S", "fwd", "capture", "given")

    def __init__(self, L, rows, S, fwd, capture):
        self.L, self.rows, self.S, self.fwd, self.capture = L, rows, S, fwd, capture
        self.given = False


def _woa_cls(m, L):
    """lever RT's check class of an m-row WO-A call on the [m, groups, group width] view of the attention output."""
    return (int(m).bit_length(), L.heads * L.hdim)


def _local_ok(L, t, m, s, replay):
    """This rank's lever states allow the graph for m rows of bucket s (a rank-local answer: it only picks this
    rank's path -- capture, replay or stock -- which all make the same one all-reduce)."""
    if _ST["dead"] or _key_m(L, m) != t.key.get(s) or _SWEPT.get(s) is not True:
        return False
    rt, dt = _H["rt"], _H["dt"]
    if rt._ON.get() is None or not rt._ST.get("woa_on"):
        return False
    mt = L.wo_b.quant_method.mxfp8_prefill_autotune_min_tokens
    if mt is not None and s >= mt:
        return False                             # from the prefill autotune bound wo_b's call carries pin_tactic
    band = t.key[s][0]
    state = dt._KEY_OK.get((L.n_out, L.k_in, band))
    if state is False:
        return False
    if replay:
        # the graph holds RT's strided bmm and DT's band tactic: the eager path at m rows takes them only once RT's
        # first-call check passed for m's row class (else stock's einsum) and DT's band check passed
        if state is not True or rt._WOK.get(_woa_cls(m, L)) is not True or rt._WOK.get(_woa_cls(s, L)) is not True:
            return False
    return True


def plan(attn, x, positions, forward_batch, defer_all_reduce, unified, multi_stream, q_padded, q_out, inv_rope):
    """fork_attn.attn_forward, right before the attention kernel: this call's AG plan, or None (stock's path). The two
    agreement sites (a new bucket's admission here, a capture forward's resolution in ag.forward) are reached by both
    ranks under the same deterministic conditions (rows, the call's branch flags, static layer facts, agreed bucket
    states); rank-local facts only decide this rank's path or enter as the values agreed on."""
    _CUR[0] = None
    fwd = _ON.get()
    if fwd is None or fwd.fb is not forward_batch:
        return None
    if defer_all_reduce or unified or multi_stream or q_padded is not None or inv_rope is not None:
        return None
    if type(x) is not torch.Tensor or x.dim() != 2:
        return None
    m = int(x.shape[0])
    if m < levers.AG_MIN_ROWS or m > levers.AG_MAX_ROWS:
        return None
    if (type(positions) is not torch.Tensor or positions.dim() != 1 or positions.shape[0] != m
            or positions.dtype != torch.int64 or not positions.is_contiguous()):
        return None
    L = _layer(attn)
    if not L:
        return None
    if (type(q_out) is not torch.Tensor or q_out.dtype != torch.float8_e4m3fn or q_out.dim() != 3
            or tuple(q_out.shape) != (m, L.heads, L.hdim)):
        return None
    if (int(attn.n_local_heads) != L.heads or int(attn.n_local_groups) != L.groups or attn._modules.get("wo_b")
            is not L.wo_b or L.wo_b.use_decode_attn_tp or _H["symm"]() or _H["get_forward"]().sp_active):
        return None
    lm = _H["linear"]
    par = lm.get_parallel()
    if attn.attn_tp_size != par.tp_size or par.attn_dp_size != 1 or par.attn_cp_size != 1:
        # attention TP < TP: _project_wo_b's attn-TP all-reduce, and with DP / CP attention each TP rank attends its own
        # rows, so the agreements below would not be reached by every rank under the same conditions (hang / mismatch)
        return None
    t = _table(L)
    if t is None:
        return None
    s = _bucket(t, m)
    if s is None:
        return None
    bs = _BSTATE.get(s)
    if bs == "dead" or s in fwd.refused:
        return None
    if bs is None:
        if not _admit(L, t, s, m, fwd):
            fwd.refused.add(s)
            return None
        bs = _BSTATE[s]
    if bs == "capturing":
        fwd.capturing.add(s)      # deterministic: both ranks resolve this bucket at the end of this forward
    graph = L.graphs.get(s)
    if graph is None:
        # the capture path holds no collective but the all-reduce the eager path makes too, so a rank-local skip is safe;
        # the bucket's outcome is agreed once, at the end of this forward (_resolve)
        if bs != "capturing" or s in fwd.fail or not _local_ok(L, t, m, s, False):
            _ST["local_eager"] += bs == "capturing"
            return None
        p = _Plan(L, m, s, fwd, True)
    else:
        if not _local_ok(L, t, m, s, True):
            _ST["local_eager"] += 1
            return None
        p = _Plan(L, m, s, fwd, False)
    _CUR[0] = p
    return p


def fmha_out(layer, q, num_qo_padded, sum_q):
    """ct_pf (the lifted _forward_trtllm_prefill): the `out` argument of the sparse-MLA launch -- AG's static input
    buffer [:sum_q] when this attention call holds an AG plan and the query is unpadded, else None (stock's)."""
    p = _CUR[0]
    if p is None or p.given or layer is not p.L.radix or _ON.get() is not p.fwd or not _BUF:
        return None
    B = _BUF[0]
    if (num_qo_padded != sum_q or sum_q != p.rows or type(q) is not torch.Tensor or q.dim() != 3
            or q.shape[1] != B.o.shape[1] or q.shape[2] != B.o.shape[2]):
        return None
    p.given = True
    return B.o[:sum_q]


def _admit(L, t, s, m, fwd) -> bool:
    """A new bucket: rank-agreed admission (free memory for the buffers + the graph pool; the WO-A sweep)."""
    if levers.AG_POOL_RESERVE and "reserve" not in _RES:
        need = t.max_s * int(levers.AG_POOL_BYTES_PER_ROW)
    elif _RES.get("reserve"):
        need = int(levers.AG_POOL_SLACK)
    else:
        need = s * int(levers.AG_POOL_BYTES_PER_ROW)
    need += (0 if _BUF else _buf_bytes(L, t.max_s)) + 2 * t.max_s * L.groups * L.rank * 2   # + the sweep's scratch
    try:
        free, _total = torch.cuda.mem_get_info()
        avail = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    except Exception:  # noqa: BLE001
        avail = 0
    mem_ok = avail - need >= int(levers.AG_MEM_MARGIN)
    a = _agree([int(mem_ok), int(not _ST["dead"])])
    if not a[1]:
        _BSTATE[s] = "dead"
        _ST["refuse"] += 1
        levers.announce(f"ag-never{s}", f"lever AG: bucket {s} OFF (AG failed on some rank)")
        return False
    if not a[0]:
        _BSTATE[s] = "dead"
        _ST["refuse"] += 1
        levers.announce(f"ag-mem{s}", f"lever AG: bucket {s} OFF (free memory: {avail / 2**30:.1f} GiB available, "
                        f"{need / 2**30:.2f} GiB needed + {int(levers.AG_MEM_MARGIN) / 2**30:.0f} GiB margin)")
        return False
    ok = True
    if not _BUF:
        try:
            _BUF.append(_Buf(L, t.max_s))
        except Exception as e:  # noqa: BLE001
            ok = False
            levers.announce("ag-buf", f"lever AG: static buffers failed ({type(e).__name__}: {str(e)[:120]})")
    if ok:
        ok = _sweep(L, t, s)
    if not _agree_bool(ok):
        _BSTATE[s] = "dead"
        _ST["refuse"] += 1
        levers.announce(f"ag-adm{s}", f"lever AG: bucket {s} OFF (static buffers or the WO-A sweep failed on some "
                        f"rank; this rank: {'ok' if ok else 'failed'})")
        return False
    _BSTATE[s] = "capturing"
    _ST["admit"] += 1
    if levers.first("ag-admit"):
        b = _BUF[0]
        levers.announce("ag-admit", f"lever AG: first bucket {s} admitted at {m} rows; static buffers {b.rows} rows "
                        f"({b.nbytes / 2**20:.0f} MiB), {avail / 2**30:.1f} GiB available")
    return True


def _bmm(o, w):
    """lever RT's WO-A launch (rt.wo_a): the strided bmm into the token-major output."""
    result = o.new_empty((o.shape[0], w.shape[0], w.shape[1]))
    torch.bmm(o.transpose(0, 1), w.transpose(1, 2), out=result.transpose(0, 1))
    return result


def _sweep(L, t, s, xbuf=None) -> bool:
    """cuBLAS picks the WO-A kernel per row count: on deterministic random inputs in the static attention-output buffer
    (the eager call's operand layout: [m, groups, group width] views with the heads x 512 row stride) and the layer's
    real WO-A weight, the bmm at every row count m of bucket s must give the first m rows of the bmm at s byte for byte.
    Once per bucket and rank (the weight's shape / layout is the same in every layer)."""
    hit = _SWEPT.get(s)
    if hit is not None:
        return hit
    if not getattr(levers, "AG_WOA_SWEEP", True):
        _SWEPT[s] = True
        return True
    t0 = time.perf_counter()
    lo = next((a for a, e in t.buckets if e == s), s)
    xbuf = _BUF[0].o if xbuf is None else xbuf
    ok = True
    try:
        with torch.no_grad():
            g = torch.Generator(device=xbuf.device)
            g.manual_seed(int(levers.AG_SEED) + s)
            x = xbuf[:s]
            x.normal_(generator=g)
            ys = _bmm(x.view(s, L.groups, -1), L.woa_w)
            bad = torch.zeros((), dtype=torch.bool, device=xbuf.device)
            for mm in range(lo, s):
                ym = _bmm(x[:mm].view(mm, L.groups, -1), L.woa_w)
                torch.logical_or(bad, _neq(ym, ys[:mm]), out=bad)
            ok = not bool(bad.item())
    except Exception as e:  # noqa: BLE001
        ok = False
        levers.announce(f"ag-sweep-err{s}", f"lever AG: WO-A sweep at bucket {s} failed ({type(e).__name__}: "
                        f"{str(e)[:160]})")
    _SWEPT[s] = ok
    _ST["sweep"] += 1
    _ST["sweep_ms"] += (time.perf_counter() - t0) * 1e3
    levers.announce(f"ag-sweep{s}", f"lever AG: WO-A cuBLAS bmm at every row count {lo}..{s} == the first rows at {s}: "
                    f"{'yes' if ok else 'NO (bucket off)'} ({_ST['sweep_ms']:.0f} ms of sweeps so far)")
    return ok


# ------------------------------------------------------------------------------------------------------- static buffers
def _buf_bytes(L, rows):
    """_Buf's size at `rows` rows: the attention output (heads x 512 bf16), wo_b's output (bf16), positions (int64)."""
    return rows * (L.heads * L.hdim * 2 + L.n_out * 2 + 8) + 4096


class _Buf:
    """AG's static tensors at `rows` rows, shared by every bucket (views [:S]) and layer (one after the other)."""

    def __init__(self, L, rows):
        dev = L.woa_w.device
        self.rows = rows
        self.o = torch.zeros(rows, L.heads, L.hdim, dtype=torch.bfloat16, device=dev)   # attention output (fmha out)
        self.y = torch.zeros(rows, L.n_out, dtype=torch.bfloat16, device=dev)            # wo_b output (pre all-reduce)
        self.pos = torch.zeros(rows, dtype=torch.int64, device=dev)                       # positions (valid indices)
        self.nbytes = sum(t.numel() * t.element_size() for t in (self.o, self.y, self.pos))


# ------------------------------------------------------------------------------------------------- the tail (eager / graph)
def _tail(L, o, positions):
    """Stock's tail for the plan's branch (fork_attn.attn_forward after the attention kernel, levers RO / RT / FL / DT
    as the eager path calls them) on `o` (rotated in place, as stock does): wo_b's output before its all-reduce."""
    attn = L.attn
    _H["ro"].inv_rope(o[..., -attn.qk_rope_head_dim:], None, attn.freqs_cis, positions=positions, inverse=True)
    v = o.view(o.shape[0], attn.n_local_groups, -1)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    r = _H["rt"].wo_a(v, wo_a, attn, L.fuse_q)
    if r is None:
        r = _H["fa"]._apply_wo_a_bf16_matmul(v, wo_a, is_decode=False, is_target_verify=False, is_prefill=True,
                                             fast_path=attn.is_dsv41, fuse_mxfp8_quant=L.fuse_q, fp8_grid=False,
                                             emit_fp8=False)
    y, _ = attn.wo_b(r.flatten(1) if isinstance(r, torch.Tensor) else r, skip_all_reduce=True)
    return y


def _seg(L, s):
    """The graph at s rows: the tail on the static buffers (RT's strided bmm required), into the static output."""
    B = _BUF[0]
    attn = L.attn
    o = B.o[:s]
    _H["ro"].inv_rope(o[..., -L.rope:], None, attn.freqs_cis, positions=B.pos[:s], inverse=True)
    r = _H["rt"].wo_a(o.view(s, L.groups, -1), L.woa_w, attn, L.fuse_q)
    if type(r) is not torch.Tensor:
        raise RuntimeError("lever RT's WO-A did not take the call")
    y, _ = attn.wo_b(r.flatten(1), skip_all_reduce=True)
    if type(y) is not torch.Tensor or tuple(y.shape) != (s, L.n_out) or y.dtype != B.y.dtype:
        raise RuntimeError("wo_b's output is not [rows, N] bf16")
    B.y[:s].copy_(y)


def _reduce(L, y):
    """RowParallelLinear.forward's reduce branch for wo_b (skip_all_reduce False, no forward batch: no quantized
    communication), then _project_wo_b's return (no deferred all-reduce; attention TP == TP)."""
    wo_b, lm = L.wo_b, _H["linear"]
    _tp_rank, tp_size = lm.get_group_rank_size(wo_b.tp_group)
    if ((wo_b.reduce_results and tp_size > 1) or wo_b.use_decode_attn_tp) and not lm.should_skip_mlp_all_reduce():
        if wo_b.use_dp_attention_reduce:
            return lm.get_parallel().attn_tp_group.all_reduce(y)
        if wo_b.parallel_group == "tp":
            return lm.get_parallel().tp_group.all_reduce(y)
        return lm.tensor_model_parallel_all_reduce(y)
    return y


def _positions(p, positions):
    """The static positions [:M] (copied once a forward: every layer gets the same tensor)."""
    fwd, m = p.fwd, p.rows
    if fwd.pos_src is not positions or fwd.pos_m != m:
        _BUF[0].pos[:m].copy_(positions)
        fwd.pos_src, fwd.pos_m = positions, m


def post(p, attn, o, positions):
    """fork_attn, right after the attention kernel: the post-attention segment from this bucket's graph + the eager
    all-reduce (the value attn_forward returns), or None: stock's tail on `o` (untouched)."""
    _CUR[0] = None
    L, m = p.L, p.rows
    B = _BUF[0] if _BUF else None
    if (B is None or attn is not L.attn or not p.given or type(o) is not torch.Tensor or o.dim() != 3
            or tuple(o.shape) != (m, L.heads, L.hdim) or o.data_ptr() != B.o.data_ptr() or o.stride() != B.o.stride()
            or type(positions) is not torch.Tensor or positions.dtype != B.pos.dtype):
        if p.capture:
            p.fwd.fail.setdefault(p.S, "the attention output is not AG's static buffer")
        _ST["eager"] += 1
        return None
    if p.capture:
        return _post_capture(p, o, positions)
    g = L.graphs.get(p.S)
    if g is None:
        _ST["eager"] += 1
        return None
    try:
        _positions(p, positions)
        g.replay()
    except Exception as e:  # noqa: BLE001 - nothing launched: stock's tail on the untouched attention output
        _local_fail("replay", e)
        return None
    _ST["replay"] += 1
    if levers.first("ag-replay"):
        levers.announce("ag-replay", f"lever AG: first graph replay at {m} rows (bucket {p.S}, layer "
                        f"{getattr(attn, 'layer_id', '?')})")
    return _reduce(L, B.y[:m])


# ----------------------------------------------------------------------------------------------- capture + first check
def _epoch():
    """The levers' learn / verify state the tail reads: a graph is captured only once a warmup leaves it unchanged."""
    out = []
    try:
        rt = _H["rt"]
        out += [len(rt._WOK), bool(rt._ST.get("woa_on"))]
    except Exception:  # noqa: BLE001
        out.append(None)
    try:
        ro = _H["ro"]
        out += [len(ro._OK), len(ro._SELF)]
    except Exception:  # noqa: BLE001
        out.append(None)
    try:
        out.append(len(_H["dt"]._KEY_OK))
    except Exception:  # noqa: BLE001
        out.append(None)
    try:
        fl = _H["fl"]
        out += [fl._STATE.get("learned", 0), fl._STATE.get("refused", 0), sum(len(v) for v in fl._PLANS.values())]
    except Exception:  # noqa: BLE001
        out.append(None)
    return tuple(out)


def _pool():
    if "pool" not in _RES:
        _RES["pool"] = torch.cuda.graph_pool_handle()
    return _RES["pool"]


def _stream():
    if "stream" not in _RES:
        _RES["stream"] = torch.cuda.Stream()
    return _RES["stream"]


def _graph(fn, pool):
    """One capture into the shared pool (thread-local capture mode: other engine threads' CUDA calls are theirs)."""
    g = torch.cuda.CUDAGraph()
    g.capture_begin(pool=pool, capture_error_mode="thread_local")
    try:
        fn()
    except BaseException:
        try:
            g.capture_end()
        except Exception:  # noqa: BLE001 - an invalidated capture: the original error is the one raised
            pass
        try:
            g.reset()
        except Exception:  # noqa: BLE001
            pass
        raise
    g.capture_end()
    return g


def _seq() -> int:
    fwd = _ON.get()
    return fwd.seq if fwd is not None else -1


def _warm(L, s):
    """Eager runs of the segment at s rows on the capture stream until the levers' learn / verify state stops changing
    (RT checks the s-row class, RO the s-row key of this layer's freqs table, DT / FL the band; cuBLAS / CuTe-DSL get
    their per-stream state). Once per (bucket, freqs table, capture forward)."""
    n = 0
    for _ in range(int(levers.AG_MAX_WARMUPS)):
        e0 = _epoch()
        _seg(L, s)
        n += 1
        if _epoch() == e0:
            _WARM[(s, L.wkey)] = _seq()
            return n
    raise RuntimeError(f"the levers' state did not settle in {n} warmups at {s} rows")


def _reserve(L, pool):
    """The pool's footprint at the largest bucket, once: the segment captured at t.max_s rows into the shared pool
    (never replayed, kept alive so the pool keeps its segments); later captures carve their intermediates out of them.
    Tried once: a failure leaves no reserve (and no half-captured graph holding the pool)."""
    smax = _TABLE[0].max_s
    _RES["reserve"] = ()
    n = _warm(L, smax) if _WARM.get((smax, L.wkey)) != _seq() else 0
    _RES["reserve"] = (_graph(lambda: _seg(L, smax), pool),)
    return n


def _live_graphs() -> int:
    n = len(_RES.get("reserve") or ())
    for _attn, L in _LAYERS.values():
        if L:
            n += len(L.graphs)
    return n


def _pool_check():
    """After a failed capture or a dropped bucket: when no live graph holds the shared pool any more, the next capture
    takes a fresh pool (PyTorch refuses to share a pool whose use count dropped to zero)."""
    if "pool" in _RES and _live_graphs() == 0:
        del _RES["pool"]


def _capture(L, s):
    """Warm up (the bucket's first capture of this freqs table in this forward) and capture the graph of (layer,
    bucket s) on AG's capture stream."""
    cap, pool = _stream(), _pool()
    cur = torch.cuda.current_stream()
    cap.wait_stream(cur)
    t0 = time.perf_counter()
    nwarm = 0
    with torch.cuda.stream(cap):
        if "reserve" not in _RES and levers.AG_POOL_RESERVE:
            nwarm += _reserve(L, pool)
        if _WARM.get((s, L.wkey)) != _seq():
            nwarm += _warm(L, s)
        e0 = _epoch()
        g = _graph(lambda: _seg(L, s), pool)
        if _epoch() != e0:
            g.reset()
            raise RuntimeError("a lever learned / verified during the capture")
    cur.wait_stream(cap)
    _ST["warm"] += nwarm
    _ST["cap_ms"] += (time.perf_counter() - t0) * 1e3
    return g, nwarm


def _neq(a, b):
    """0-d GPU bool: a's bytes differ from b's (same shape / dtype asserted on the host), None on another geometry."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return None
    v = torch.int16 if a.element_size() == 2 else torch.int32 if a.element_size() == 4 else torch.int64
    if a.element_size() == 1:
        v = torch.uint8
    return (a.contiguous().view(v) != b.contiguous().view(v)).any()


def _post_capture(p, o, positions):
    """The bucket's capture forward, this layer: capture, replay once on the real rows, and return the eager path's
    result. The graph output is compared with it on the GPU (no host sync here); _resolve reads the flag once at the
    end of the forward and agrees the bucket's outcome over the ranks before any graph result is used. The warmups and
    the capture rotate the static attention output in place, so the real rows are kept aside first and restored before
    the replay; the eager path runs on that copy. Nothing here is a collective but the all-reduce the eager path makes
    (exactly once, on the eager rows), so a rank-local failure only marks the bucket."""
    L, m, s, fwd = p.L, p.rows, p.S, p.fwd
    B = _BUF[0]
    snap = o.clone()                         # the attention kernel's real rows (eager input)
    try:
        _positions(p, positions)
        g, nwarm = _capture(L, s)
    except Exception as e:  # noqa: BLE001 - nothing captured: the eager tail on the kept rows (its own all-reduce)
        torch.cuda.current_stream().wait_stream(_stream())
        fwd.fail.setdefault(s, f"capture: {type(e).__name__}: {str(e)[:200]}")
        return _reduce(L, _tail(L, snap, positions))
    _ST["capture"] += 1
    if levers.first(f"ag-cap{s}"):
        levers.announce(f"ag-cap{s}", f"lever AG: bucket {s} first captured at {m} rows (layer "
                        f"{getattr(L.attn, 'layer_id', '?')}, {nwarm} warmups, {_ST['cap_ms']:.0f} ms capture total, "
                        f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB)")
    bad = fwd.bad.get(s)
    if bad is None:
        bad = fwd.bad[s] = torch.zeros((), dtype=torch.bool, device=o.device)
    ok = True
    try:
        B.o[:m].copy_(snap)
        _positions(p, positions)
        g.replay()
    except Exception as e:  # noqa: BLE001
        ok = False
        fwd.fail.setdefault(s, f"replay: {type(e).__name__}: {str(e)[:120]}")
    y_ref = _tail(L, snap, positions)          # the eager path at m rows (what this call returns)
    if ok:
        d = _neq(B.y[:m], y_ref) if type(y_ref) is torch.Tensor else None
        if d is None:
            ok = False
        else:
            torch.logical_or(bad, d, out=bad)
    if ok:
        fwd.pending.setdefault(s, []).append((L, g))
        fwd.rows.setdefault(s, m)
    else:
        fwd.fail.setdefault(s, "the graph output had another geometry")
        g.reset()
        _pool_check()
    return _reduce(L, y_ref)


def _resolve(fwd):
    """The end of a capture forward (both ranks reach it: the bucket's "capturing" state is agreed): read this rank's
    flags once, agree (gloo MIN) on [every capture / replay / byte check passed, this rank captured some layer, this
    rank captured none], then on every rank: activate the pending graphs (and the bucket), or turn the bucket off for
    good (a failure, or some ranks captured and others not), or keep it capturing (no rank captured: retried at its
    next forward, at most AG_MAX_CAPTURE_FORWARDS times)."""
    for s in sorted(fwd.capturing):
        pend = fwd.pending.pop(s, [])
        why = fwd.fail.get(s, "")
        ok = not why
        if ok and pend and levers.AG_CHECK:
            try:
                if bool(fwd.bad[s].item()):
                    ok, why = False, "graph bytes != eager bytes"
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"check: {type(e).__name__}"
        a = _agree([int(ok), int(bool(pend)), int(not pend)])
        _ST["check"] += len(pend)
        if not a[0] or not (a[1] or a[2]):
            for _L, g in pend:
                g.reset()
            _kill_bucket(s, f"capture forward at {fwd.rows.get(s, '?')} rows: "
                            f"{why or ('failed on another rank' if not a[0] else 'captured on some ranks only')}")
        elif a[1]:
            for L, g in pend:
                L.graphs[s] = g
            _BSTATE[s] = "active"
            levers.announce(f"ag-chk{s}", f"lever AG: bucket {s} on: {len(pend)} layers' graphs captured at "
                            f"{fwd.rows.get(s, '?')} rows, first replay == eager bit for bit on every rank")
        else:
            _RETRY[s] = _RETRY.get(s, 0) + 1
            if _RETRY[s] >= int(getattr(levers, "AG_MAX_CAPTURE_FORWARDS", 8)):
                _kill_bucket(s, f"{_RETRY[s]} capture forwards without a captured layer")
                continue
            for _L, g in pend:
                g.reset()
            _pool_check()


# ------------------------------------------------------------------------------------------------------- bookkeeping
def _agree(values):
    """MIN over the TP CPU group (gloo), elementwise (fg._agree's form)."""
    t = torch.tensor([int(v) for v in values], dtype=torch.int64)
    g = _H["tp_group"]()
    if g.world_size > 1:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=g.cpu_group)
    return t.tolist()


def _agree_bool(ok) -> bool:
    return bool(_agree([int(bool(ok))])[0])


def _kill_bucket(s, why):
    _BSTATE[s] = "dead"
    for _attn, L in list(_LAYERS.values()):
        if L:
            g = L.graphs.pop(s, None)
            if g is not None:
                g.reset()
    _pool_check()
    _ST["fail"] += 1
    levers.announce(f"ag-dead{s}", f"lever AG: bucket {s} OFF for good ({why[:300]}): eager for those rows")


def _local_fail(where, e):
    """An unexpected error on this rank: AG's replays stop here for good; plan() still reaches (and answers False to)
    every agreement the other rank reaches, so the ranks never wait on each other."""
    _ST["dead"] = True
    _ST["fail"] += 1
    levers.announce(f"ag-err-{where}", f"lever AG: {where} failed ({type(e).__name__}: {str(e)[:200]}): replays off "
                    f"for good on this rank (stock path)")


def stats() -> dict:
    return {k: v for k, v in _ST.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def summary() -> str:
    live = sorted(s for s, v in _BSTATE.items() if v == "active")
    dead = sorted(s for s, v in _BSTATE.items() if v == "dead")
    return (f"buckets active {live}, dead {dead}; replays {_ST['replay']}, captures {_ST['capture']}, checks "
            f"{_ST['check']}, warmups {_ST['warm']}, capture {_ST['cap_ms']:.0f} ms, sweeps {_ST['sweep']} "
            f"({_ST['sweep_ms']:.0f} ms), eager fallbacks {_ST['eager']}, local eager {_ST['local_eager']}, "
            f"failures {_ST['fail']}")

