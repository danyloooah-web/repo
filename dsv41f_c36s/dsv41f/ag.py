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

LEVER AQ (levers.AQ, same kit, same buckets / buffers / agreements): the PRE-attention q path of the same eager EXTEND
calls -- wqkv_a (lever FL / DT: MXFP8 quantize of the attention-norm output + CuTe-DSL GEMM), lever QN's q_norm +
quantize unit, wq_b (FL / DT GEMM on QN's quantized rows), lever QR's FP8 query rope -- replayed from a second graph per
(layer, bucket). fork_attn.attn_forward's prepare call (the plain _forward_prepare branch: no multi-stream prepare)
first asks qprep(): the call's x is copied into AQ's static input [:M] (positions: AG's static copy), the graph writes
wqkv_a's output, QN's normalized q_lora (bf16, what the indexer / compressor read) and the FP8 query (QR's q_out:
what the attention kernel reads) into static buffers, and the eager remainder -- stock's _forward_prepare after the q
path for this branch (the KV store _compute_kv_to_cache on wqkv_a's kv half, the indexer / compressor or the low-ratio
sources, every one a stock call at M rows) -- runs on views [:M] of them. Scope (else stock's _forward_prepare runs):
x a contiguous 16-byte aligned bf16 [M, 5120] (no HQ quantized input), no unified KV / CP / padded heads, fused
wqa+wkv, no fused qk-norm-rope (CUDA), wqkv_a a hook-free ReplicatedLinear and wq_b a hook-free ColumnParallelLinear
without output gather (no collective in the q path), both lever DT's MXFP8 linears, QN's and QR's wrappers bound.
Exactness: the table's key adds DT's band for wqkv_a (1792, 5120) and wq_b (16384, 1280) and wqkv_a's 2T/SF choice
(wq_b's input is QN's quantized rows: no quantize), so no bucket crosses an edge of a q-path launch decision (keyed on
the static switch levers.AQ, never on a rank's QN / QR install state: every rank builds the same table); every
q-path kernel is row-independent (per-row MXFP8 quantize with row-count-independent 128x4 scale positions, CuTe-DSL
split-1 GEMMs with a fixed K order, QN's per-row norm + quantize, QR's per (token, head) rope with the token's own
position); a replay needs this rank's QN check of M's key, QR's check of M's row class and DT's checks of both bands
passed and QN taking this layer's call (else the eager path, which may still be checking, runs); the first replay of
every (bucket, layer) is compared byte for byte with the eager q path at M rows (wqkv_a's output, the normalized
q_lora, q_out) and agreed over the ranks at the same end-of-forward agreement as the post graphs (a constant-length
vector on every rank). A q-path failure
turns AQ off for that bucket only (AG's post graphs stay; a capture / check failure on every rank, a replay error on
the rank that raised it); levers.AQ_AB = N > 0 (diagnostic) replays the q graphs on every N-th eligible forward only.

LEVER AGO (levers.AGO / AGO_ZC / AGP, gen_ago.py, same kit): AG / AQ's own host overhead trimmed, their captures done
ahead of the timed forwards. Exactness and rank rules as above; every graph result is still used only after its
first real-row replay was compared byte for byte with the eager path.
  (1) PER-FORWARD GATE (levers.AGO). The tests of _gate whose inputs are fixed for one forward and one positions
      tensor -- the row count's range, the positions tensor's form, the engine's symmetric-memory / SP / parallel
      state, the row count's bucket -- are evaluated once per (forward, positions tensor, rows) (_Fwd.gc); the call's
      branch flags, x / q_out forms and this replica's facts per call, as before. plan() reuses the gate qprep()
      computed for the same attention call (the same replica, x and q_out objects; plan's own flags all falsy, the
      gate's condition), qprep's forward-batch tests (CP, mode) run once per forward, and the replay-time lever checks
      (_q_ok_fast / _l_ok_fast) build their row-count keys once per (forward, rows, bucket) while every lever verdict,
      global switch and context is still read per call. Same answers, same agreement sites in the same order.
  (2) ZERO-COPY Q INPUT (levers.AGO_ZC). Lever FG's graph 2 writes the next layer's attention input into FG's static
      buffer (fg._BUF[0].nrm), so in a steady eager forward a layer's x is that buffer's first M rows. A ZC graph is a
      second q graph of the same (layer, bucket), captured by the pre-capture (3) with that buffer as its input (AQ
      holds a reference to it, so the address stays FG's buffer for the process), for the layers whose previous
      replica is an FG layer and which have no Engram (the only ones whose x can be FG's rows). It replays only when x
      is exactly the buffer's first M rows (same data pointer, [M, 5120] bf16 contiguous: x's bytes are what the graph
      reads); any other x is copied into AQ's own input as before. A ZC graph replaces the copy path only after its
      first real-row replay on this rank was compared byte for byte with the eager q path at M rows (wqkv_a's output,
      the normed q_lora, q_out; the eager values are what that call returns), read once at the end of the forward
      (_resolve_z). Rank-local: the q path holds no collective, so a ZC state only picks this rank's path.
  (3) PRE-CAPTURE (levers.AGP). At the engine's first eligible eager EXTEND forward (its first prepare call: layer 0,
      after lever FGP's pre-capture at fg.plan; any row count, so the live replay's untimed 16384-row warmup), on both
      ranks: one agreement that both hold the same plan (AG layers, table, new buckets), then per new bucket, largest
      first: a budget agreement (AGP_BUDGET_S), AG's own agreed admission (_admit: memory gate, static buffers, the
      WO-A sweep; refused -> off, AG's rule), then rank-locally on deterministic synthetic static contents (AGP_SEED:
      attention output and x N(0, 1) bf16, positions inside every layer's RoPE table) the warmups + captures AG / AQ
      make at S rows for every layer whose lever states allow them (post graph, q graph, ZC graph), then one agreement
      [no error on any rank, some post graph]: the graphs wait in _PRE and the bucket stays "capturing", else they are
      released on every rank and the bucket goes back to the lazy path (new, its warm marks dropped). FIRST REAL USE
      UNCHANGED: a pre-captured bucket's first eligible forward takes AG's capture path, which takes the pre-captured
      graph instead of capturing, replays it on the real rows, runs the eager path at M rows (what that call returns),
      compares every byte on the GPU, and _resolve agrees over the ranks at the end of the forward before any graph
      result is used; graphs no layer took are released once the bucket leaves "capturing". Levers checked during the
      warmups at S rows (RO / RT / DT / FL / QN / QR keys of S) run their own first-use checks on the synthetic rows
      (each compares its launch with stock's on the same input); the graphs themselves are checked on real rows.
  levers.AGO_AB = N > 0 (diagnostic): (1) + (2) on every N-th eligible forward only (the others take AG / AQ's c19x
  paths, same bytes), announced per forward.
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
_QSTATE: dict = {}       # bucket S -> "active" | "dead" for lever AQ's q-path graphs (absent: not decided yet)
_QWARM: dict = {}        # (bucket S, freqs table) -> the eligible forward whose q capture warmed the levers up
_AQ = {"on": False, "detail": "not installed", "replay": 0, "capture": 0, "check": 0, "fail": 0, "local_eager": 0,
       "ab": 0, "decline": 0}
# lever AGO
_GO = {"on": False, "detail": "not installed", "ab": 0, "fast": 0, "reuse": 0}
_ZC = {"on": False, "replay": 0, "check": 0, "pass": 0, "fail": 0, "drop": 0, "layers": 0}
_PRE: dict = {}          # lever AGP: bucket S -> {id(_Layer): [layer state, post graph, q graph, ZC graph]} pre-captured
_P = {"armed": False, "done": False, "filled": False, "detail": "not armed", "captured": 0, "graphs": 0, "taken": 0,
      "dropped": 0, "agree": 0, "ms": 0.0, "errors": [], "bucket_ms": {}, "skipped": 0, "layers": 0, "zlayers": 0}
_BF16, _FP8 = torch.bfloat16, torch.float8_e4m3fn
_SIGS: dict = {}         # lever AGO: layer sig -> small int (_Layer.sk)
_QKV_NK = (1792, 5120)   # wqkv_a's MXFP8 weight (q_lora_rank 1280 + head_dim 512, hidden 5120)
_QB_NK = (16384, 1280)   # wq_b's per-rank MXFP8 weight (32 heads x 512, q_lora_rank 1280)
AQ_PIN_TARGETS = (
    ("sglang.srt.models.deepseek_v4", "MQALayer._forward_prepare"),
    ("sglang.srt.layers.linear", "ReplicatedLinear.forward"),
    ("sglang.srt.layers.linear", "ColumnParallelLinear.forward"),
)

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


def check_aq_pins() -> str:
    """'' when every function lever AQ mirrors (stock's _forward_prepare: the q path + this branch's remainder) or
    relies on (the q linears' forwards) hashes to levers.AQ_PINS, else what differs."""
    bad = []
    texts: dict = {}
    pins = getattr(levers, "AQ_PINS", {})
    for mod_name, qual in AQ_PIN_TARGETS:
        want = pins.get(f"{mod_name}:{qual}")
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

    from dsv41f import dt, fl, fork_attn, qn, qr, ro, rt

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
              fa=fork_attn, ro=ro, rt=rt, dt=dt, fl=fl, Row=linear.RowParallelLinear, qn=qn, qr=qr,
              Rep=linear.ReplicatedLinear, Col=linear.ColumnParallelLinear, cp=dv4.is_cp_active, fg=None)
    try:
        from dsv41f import fg   # lever AGO's ZC graphs read lever FG's static next-attention input (read only)

        for n in ("_BUF", "_ST", "_LAYERS"):
            if not hasattr(fg, n):
                raise AttributeError(f"fg.{n}")
        _H["fg"] = fg
    except Exception:  # noqa: BLE001 - no FG: no ZC graphs (the copy path)
        _H["fg"] = None


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
    _RES["chain"] = tuple(layers)        # lever AGO: the layer replicas in order (ZC: which layer follows an FG layer)
    _ST["on"] = n > 0
    if _ST["on"]:
        _aq_install()
    pre = _presweep() if _ST["on"] else ""
    _ago_install()
    return _set(f"{'on' if _ST['on'] else 'off'}: {n} attention replicas run the lifted forward; {pre}; eager EXTEND calls "
                f"of {levers.AG_MIN_ROWS}..{levers.AG_MAX_ROWS} rows replay the post-attention segment (RO inverse "
                f"RoPE, RT WO-A, wo_b MXFP8 quantize + GEMM) from one graph per (layer, bucket <= "
                f"{levers.AG_PAD:.0%} padding inside constant-choice regions), wo_b's all-reduce eager; buckets "
                f"admitted after a WO-A cuBLAS row-consistency sweep, captured lazily, first replay per (bucket, "
                f"layer) checked bitwise vs eager{'' if levers.AG_CHECK else ' (CHECK OFF)'}; lever AQ: "
                f"{_AQ['detail']}; lever AGO: {_GO['detail']}")


def _ago_install() -> str:
    """Lever AGO's install-time state (rank-local switches: the per-forward gate and the ZC graphs only pick this
    rank's host path; the pre-capture's agreements are reached by every rank whose AG is on, the same condition)."""
    _GO["on"] = bool(_ST["on"] and getattr(levers, "AGO", False))
    _ZC["on"] = bool(_GO["on"] and _AQ["on"] and getattr(levers, "AGO_ZC", False) and _H.get("fg") is not None)
    # pre-captured graphs go live only through AG's real-row first-use check
    _P["armed"] = bool(_ST["on"] and getattr(levers, "AGP", False) and levers.AG_CHECK)
    _P["detail"] = "armed" if _P["armed"] else ("off (levers.AG_CHECK off: no real-row first-use check)"
                                                 if getattr(levers, "AGP", False) else "off (levers.AGP)")
    ab = int(getattr(levers, "AGO_AB", 0))
    _GO["detail"] = (f"per-forward gate {'on' if _GO['on'] else 'off'}, zero-copy q input from lever FG's static "
                     f"buffer {'on' if _ZC['on'] else 'off'}, pre-capture (AGP) {_P['detail']}"
                     f"{f' (budget {float(levers.AGP_BUDGET_S):.0f} s)' if _P['armed'] else ''}"
                     f"{f'; A/B: gate + zero-copy on every {ab}-th eligible forward' if ab else ''}")
    return _GO["detail"]


def _aq_install() -> str:
    """Lever AQ's install-time state on this rank (rank-local, never agreed: it only picks this rank's paths; the bucket
    table is keyed on the static switch levers.AQ, so it is the same on every rank)."""
    _AQ["on"] = False
    if not getattr(levers, "AQ", False):
        _AQ["detail"] = "off (levers.AQ)"
        return _AQ["detail"]
    if getattr(levers, "AG_PIN", True):
        bad = check_aq_pins()
        if bad:
            _AQ["detail"] = f"off (stock changed: {bad[:300]})"
            return _AQ["detail"]
    qn, qr = _H["qn"], _H["qr"]
    if not qn._STATE.get("ok") or not qr._STATE.get("ok"):
        _AQ["detail"] = f"off (lever QN {'on' if qn._STATE.get('ok') else 'off'}, QR {'on' if qr._STATE.get('ok') else 'off'})"
        return _AQ["detail"]
    _AQ["on"] = True
    ab = int(getattr(levers, "AQ_AB", 0))
    _AQ["detail"] = (f"on: the q path (wqkv_a, QN, wq_b, QR) of the same calls replayed from a second graph per (layer, "
                     f"bucket), x copied into the static input, KV store / indexer / compressor eager on the static "
                     f"outputs{f'; A/B: q graphs replayed on every {ab}-th eligible forward' if ab else ''}")
    return _AQ["detail"]


# -------------------------------------------------------------------------------------------------------- forward gate
class _Fwd:
    """One eligible forward: what AG already set up for it."""

    __slots__ = ("fb", "refused", "pos_src", "pos_m", "capturing", "pending", "fail", "bad", "rows", "seq", "aq",
                 "qpending", "qfail", "qbad", "ago", "gc", "st", "qfb", "lk", "qk", "zpend", "zbad", "zcap")

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
        self.aq = True            # lever AQ's q graphs may replay in this forward (AQ_AB: every N-th)
        self.qpending = {}        # bucket -> [(layer state, q graph)] captured + replayed in this forward
        self.qfail = {}           # bucket -> why a q capture / check step failed on this rank
        self.qbad = {}            # bucket -> 0-d GPU bool: some q-graph byte != the eager byte
        self.ago = False          # lever AGO's per-forward gate + ZC graphs in this forward (AGO_AB: every N-th)
        self.gc = None            # AGO: [positions, rows, tp size or None (not eligible), table, bucket] of this forward
        self.st = None            # AGO: (replica, x, q_out, gate) of the attention call qprep gated, for its plan()
        self.qfb = None           # AGO: qprep's forward-batch tests (no CP, an extend, not verify) for this forward
        self.lk = {}              # AGO: (rows, bucket, layer sig) -> the post replay check's keys (or False)
        self.qk = {}              # AGO: (rows, bucket, heads) -> the q replay check's keys (or False)
        self.zpend = []           # AGO: [(layer state, bucket, ZC entry, GPU flags)] whose first real-row replay ran here
        self.zbad = None          # (unused: each ZC check carries its own flags)
        self.zcap = {}            # AGP: bucket -> [(layer state, ZC entry)] taken in this capture forward


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
    if _AQ["on"] and getattr(levers, "AQ_AB", 0):
        fwd.aq = _aq_ab_on(forward_batch)
    if _GO["on"]:
        fwd.ago = _ago_ab_on(forward_batch) if getattr(levers, "AGO_AB", 0) else True
    token = _ON.set(fwd)
    try:
        out = fn(*args, **kwargs)
    finally:
        _ON.reset(token)
        _CUR[0] = None
        fwd.st = None
    if fwd.capturing:
        _resolve(fwd)              # both ranks: the bucket was "capturing" (an agreed state) when this forward met it
    if fwd.zpend:
        _resolve_z(fwd)            # rank-local: the ZC graphs whose first real-row replay ran in this forward
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


def _aq_ab_on(forward_batch):
    """Diagnostic A/B of lever AQ inside one engine (levers.AQ_AB = N > 0, never in a shipped bundle): of the AG
    forwards with AG_MIN_ROWS..AG_MAX_ROWS rows, the q graphs replay on every N-th only (capture forwards capture them
    regardless: only replays are gated). The count is the same on both ranks, and a q replay is a rank-local path."""
    ids = getattr(forward_batch, "input_ids", None)
    rows = -1 if ids is None else int(ids.shape[0])
    if not levers.AG_MIN_ROWS <= rows <= levers.AG_MAX_ROWS:
        return True
    _AQ["ab"] += 1
    on = _AQ["ab"] % int(levers.AQ_AB) == 0
    levers.announce(f"aq-ab{_AQ['ab']}", f"lever AQ A/B: eligible forward {_AQ['ab']} ({rows} rows): AQ "
                    f"{'on' if on else 'off'}")
    return on


def _ago_ab_on(forward_batch):
    """Diagnostic A/B of lever AGO inside one engine (levers.AGO_AB = N > 0, never in a shipped bundle): of the AG
    forwards with AG_MIN_ROWS..AG_MAX_ROWS rows, every N-th runs with AGO's per-forward gate + ZC graphs, the others
    with AG / AQ's per-call gate + x copy (the same bytes, the same agreement sites). The count is the same on both
    ranks; the pre-capture (AGP) is not gated."""
    ids = getattr(forward_batch, "input_ids", None)
    rows = -1 if ids is None else int(ids.shape[0])
    if not levers.AG_MIN_ROWS <= rows <= levers.AG_MAX_ROWS:
        return True
    _GO["ab"] += 1
    on = _GO["ab"] % int(levers.AGO_AB) == 0
    levers.announce(f"ago-ab{_GO['ab']}", f"lever AGO A/B: eligible forward {_GO['ab']} ({rows} rows): AGO "
                    f"{'on' if on else 'off'}")
    return on


# --------------------------------------------------------------------------------------------- per-layer static state
class _Layer:
    __slots__ = ("attn", "radix", "wo_b", "woa_w", "groups", "rank", "heads", "hdim", "rope", "fuse_q", "fused_woa",
                 "n_out", "k_in", "sfb", "wkey", "graphs", "sig", "aq", "aq_why", "wqkv", "wqb", "qlr", "qgraphs", "zg", "sk")


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
    L.qgraphs = {}
    L.zg = {}                # lever AGO: bucket -> [ZC q graph, "pending" | "checking" | "active"]
    L.sig = (g, r, heads, hdim, n, k, fq, L.fused_woa, str(w.device))
    L.sk = _SIGS.setdefault(L.sig, len(_SIGS))   # lever AGO: a small key per distinct layer shape
    if L.wkey is None:
        return False, "no freqs table"
    L.aq, L.aq_why = False, "lever AQ off"
    L.wqkv = L.wqb = None
    L.qlr = 0
    if _AQ["on"]:
        try:
            L.aq_why = _bind_q(attn, L)
        except Exception as e:  # noqa: BLE001 - deterministic (static facts): this layer's q path stays eager
            L.aq_why = f"{type(e).__name__}: {str(e)[:120]}"
        L.aq = not L.aq_why
        if not L.aq:
            levers.announce(f"aq-layer-{L.aq_why[:40]}", f"lever AQ: attention layer {getattr(attn, 'layer_id', '?')}'s "
                            f"q path stays eager ({L.aq_why})")
    return L, ""


def _q_linear(lin, cls, nk):
    """'' when `lin` is a hook-free `cls` running lever DT's MXFP8 linear of shape nk without bias, else why not."""
    if not isinstance(lin, cls) or type(lin).forward is not cls.forward or lin._forward_hooks or lin._forward_pre_hooks:
        return f"not a hook-free {cls.__name__}"
    meth = getattr(lin, "quant_method", None)
    w = lin._parameters.get("weight")
    if (type(meth).__name__ != "Fp8LinearMethod" or getattr(meth, "w8a8_mxfp8_linear", None) is None
            or getattr(meth, "use_marlin", False) or not isinstance(w, torch.Tensor) or w.dtype != torch.float8_e4m3fn
            or w.dim() != 2 or tuple(w.shape) != nk or getattr(lin, "bias", None) is not None
            or not getattr(lin, "dsv4f_dt", False) or not levers.DT_TACTICS.get(nk)):
        return f"not lever DT's MXFP8 linear {nk} without bias"
    return ""


def _bind_q(attn, L):
    """'' when this replica's q path (stock's plain _forward_prepare branch on CUDA) takes lever AQ's graphs, else
    why not. Static (load-time) facts."""
    if not getattr(attn, "fuse_wqa_wkv", False) or getattr(attn, "use_fused_qk_norm_rope", True):
        return "not the fused wqa+wkv path without fused qk-norm-rope"
    if int(attn.q_lora_rank) != _QB_NK[1] or int(attn.head_dim) != 512 or attn.q_head_norm:
        return "q layout (q_lora_rank 1280, head 512, no q head norm)"
    wqkv, wqb = attn._modules.get("wqkv_a"), attn._modules.get("wq_b")
    why = _q_linear(wqkv, _H["Rep"], _QKV_NK)
    if why:
        return f"wqkv_a {why}"
    why = _q_linear(wqb, _H["Col"], (L.heads * L.hdim, _QB_NK[1]))
    if why or (L.heads * L.hdim, _QB_NK[1]) != _QB_NK:
        return f"wq_b {why or 'shape'}"
    if wqb.gather_output or int(wqkv.weight.shape[1]) != int(attn.hidden_size):
        return "wq_b gathers its output (a collective) / hidden size"
    qn, qr = _H["qn"], _H["qr"]
    bn, bq = getattr(attn, "_normalize_q_lora", None), getattr(attn, "_compute_q_b", None)
    if getattr(bn, "__func__", None) is not qn._wrapped or getattr(bn, "__self__", None) is not attn:
        return "lever QN's _normalize_q_lora not bound on this replica"
    if getattr(bq, "__func__", None) is not qr._wrapped or getattr(bq, "__self__", None) is not attn:
        return "lever QR's _compute_q_b not bound on this replica"
    L.wqkv, L.wqb, L.qlr = wqkv, wqb, int(attn.q_lora_rank)
    return ""


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
    quantize choice, lever RO's tile, the fused-WO-A bound. Lever AQ's fields follow the static switch levers.AQ,
    never this rank's QN / QR install state (_AQ["on"]): every rank builds the same table, so the admission
    agreements keyed by its buckets pair up whatever AQ's state on each rank (that state only picks a rank's path)."""
    band = _H["dt"]._band(m, L.n_out, L.k_in)[0]
    key = (band, m * L.sfb >= _H["two_t"], _ro_tile(m, L.heads), L.fused_woa and 0 < m <= _H["fused_max"])
    if getattr(levers, "AQ", False):
        # lever AQ: DT's bands of wqkv_a and wq_b, wqkv_a's 2T/SF quantize choice (wq_b reads QN's quantized rows)
        dt = _H["dt"]
        key += (dt._band(m, *_QKV_NK)[0], m * (_QKV_NK[1] // 32) >= _H["two_t"], dt._band(m, *_QB_NK)[0])
    return key


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
    if getattr(levers, "AQ", False):
        # the q linears' pin_tactic bound (lever AQ): static facts of the replica, whatever this rank's AQ state
        for name in ("wqkv_a", "wq_b"):
            meth = getattr(L.attn._modules.get(name), "quant_method", None)
            mt = getattr(meth, "mxfp8_prefill_autotune_min_tokens", None)
            if mt is not None:
                hi = min(hi, int(mt) - 1)
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
    st, fwd.st = fwd.st, None
    if (st is not None and st[0] is attn and st[1] is x and st[2] is q_out and st[3] is positions
            and not defer_all_reduce and not unified and not multi_stream and q_padded is None and inv_rope is None):
        # lever AGO: qprep gated this very attention call (same replica, x, q_out, positions; every flag plan's gate
        # tests is falsy, as the gate requires): the same answer
        g = st[4]
        _GO["reuse"] += 1
    elif fwd.ago:
        g = _gate_fast(fwd, attn, x, positions, defer_all_reduce, unified, multi_stream, q_padded, q_out, inv_rope)
    else:
        g = _gate(attn, x, positions, defer_all_reduce, unified, multi_stream, q_padded, q_out, inv_rope)
    if g is None:
        return None
    L, t, m, s = g
    bs = _state(L, t, s, m, fwd)
    if bs is None:
        return None
    graph = L.graphs.get(s)
    if graph is None:
        # the capture path holds no collective but the all-reduce the eager path makes too, so a rank-local skip is safe;
        # the bucket's outcome is agreed once, at the end of this forward (_resolve)
        if bs != "capturing" or s in fwd.fail or not _local_ok(L, t, m, s, False):
            _ST["local_eager"] += bs == "capturing"
            return None
        p = _Plan(L, m, s, fwd, True)
    else:
        if not (_l_ok_fast(L, t, m, s, fwd) if fwd.ago else _local_ok(L, t, m, s, True)):
            _ST["local_eager"] += 1
            return None
        p = _Plan(L, m, s, fwd, False)
    _CUR[0] = p
    return p


def _gate(attn, x, positions, defer_all_reduce, unified, multi_stream, q_padded, q_out, inv_rope):
    """(layer state, table, rows, bucket) when this attention call can take AG's (and AQ's) graphs, else None. Every
    test is the same on both TP ranks (the call's branch flags, rows, static layer facts, the engine config): plan()
    and qprep() reach the admission agreement (_state) only through it, with the same arguments."""
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
    return L, t, m, s


def _gate_fwd(positions, m):
    """Lever AGO: _gate's tests that do not depend on the layer or the call beyond (forward, positions tensor, rows):
    [positions, rows, attention TP size or None (no call of this forward with these rows / positions takes AG), table,
    bucket] (table / bucket filled at the first layer call). Evaluated in _gate's order and with _gate's tests."""
    c = [positions, m, None, None, None]
    if m < levers.AG_MIN_ROWS or m > levers.AG_MAX_ROWS:
        return c
    if (type(positions) is not torch.Tensor or positions.dim() != 1 or positions.shape[0] != m
            or positions.dtype != torch.int64 or not positions.is_contiguous()):
        return c
    if _H["symm"]() or _H["get_forward"]().sp_active:
        return c
    par = _H["linear"].get_parallel()
    if par.attn_dp_size != 1 or par.attn_cp_size != 1:
        return c
    c[2] = par.tp_size
    return c


def _gate_fast(fwd, attn, x, positions, defer_all_reduce, unified, multi_stream, q_padded, q_out, inv_rope):
    """Lever AGO: _gate with its per-forward tests from _Fwd.gc (once per forward, positions tensor and row count: a
    late-layer tail's narrower rows / positions start a new entry); the call's flags, x / q_out and this replica's
    facts per call. The same answer as _gate (both ranks: the same deterministic inputs)."""
    if defer_all_reduce or unified or multi_stream or q_padded is not None or inv_rope is not None:
        return None
    if type(x) is not torch.Tensor or x.dim() != 2:
        return None
    m = int(x.shape[0])
    c = fwd.gc
    if c is None or c[0] is not positions or c[1] != m:
        c = fwd.gc = _gate_fwd(positions, m)
    if c[2] is None:
        return None
    L = _layer(attn)
    if not L:
        return None
    if (type(q_out) is not torch.Tensor or q_out.dtype != _FP8 or q_out.dim() != 3
            or tuple(q_out.shape) != (m, L.heads, L.hdim)):
        return None
    if (int(attn.n_local_heads) != L.heads or int(attn.n_local_groups) != L.groups or attn._modules.get("wo_b")
            is not L.wo_b or L.wo_b.use_decode_attn_tp or attn.attn_tp_size != c[2]):
        return None
    t = _table(L)
    if t is None:
        return None
    if c[3] is not t:
        c[3], c[4] = t, _bucket(t, m)
    s = c[4]
    if s is None:
        return None
    _GO["fast"] += 1
    return L, t, m, s


def _l_ok_fast(L, t, m, s, fwd):
    """Lever AGO: _local_ok(L, t, m, s, replay=True) with the row-count keys built once per (forward, rows, bucket,
    layer shape): the decision key's match, the bucket's WO-A sweep (final once admitted), DT's band key and RT's two
    row-class keys; AG's dead flag, RT's context / switch, the autotune bound and every verdict read per call."""
    k = (m, s, L.sk)
    c = fwd.lk.get(k)
    if c is None:
        c = False
        if t.key.get(s) is not None and _key_m(L, m) == t.key.get(s) and _SWEPT.get(s) is True:
            c = ((L.n_out, L.k_in, t.key[s][0]), _woa_cls(m, L), _woa_cls(s, L))
        fwd.lk[k] = c
    if c is False or _ST["dead"]:
        return False
    rt = _H["rt"]
    if rt._ON.get() is None or not rt._ST.get("woa_on"):
        return False
    mt = L.wo_b.quant_method.mxfp8_prefill_autotune_min_tokens
    if mt is not None and s >= mt:
        return False
    return _H["dt"]._KEY_OK.get(c[0]) is True and rt._WOK.get(c[1]) is True and rt._WOK.get(c[2]) is True


def _q_ok_fast(L, t, m, s, fwd):
    """Lever AGO: _q_local_ok(L, t, m, s, replay=True) with the row-count keys built once per (forward, rows, bucket,
    heads): the decision key's match, DT's two band keys, QN's and QR's check keys; AG's dead flag, AQ's / QN's / QR's
    switches, FL's / QN's contexts, the q linears' autotune bounds, every verdict and QN's layer facts read per call."""
    k = (m, s, L.heads)
    c = fwd.qk.get(k)
    if c is None:
        c = False
        key = t.key.get(s)
        if (key is not None and _key_m(L, m) == key and m > 64 and len(key) >= 7 and key[4] is not None
                and key[6] is not None):
            c = ((_QKV_NK[0], _QKV_NK[1], key[4]), (_QB_NK[0], _QB_NK[1], key[6]), _qn_key(m), _qr_key(m, L))
        fwd.qk[k] = c
    if c is False or _ST["dead"] or not _AQ["on"]:
        return False
    qn, qr, dt = _H["qn"], _H["qr"], _H["dt"]
    if not qn._STATE.get("ok") or not qr._STATE.get("ok") or _H["fl"]._ON.get() is None:
        return False
    for lin in (L.wqkv, L.wqb):
        mt = lin.quant_method.mxfp8_prefill_autotune_min_tokens
        if mt is not None and s >= mt:
            return False
    if (dt._KEY_OK.get(c[0]) is not True or dt._KEY_OK.get(c[1]) is not True or qn._OK.get(c[2]) is not True
            or qr._OK.get(c[3]) is not True or qn._ON.get() is None):
        return False
    info = qn._LAYER.get(id(L.attn))
    return info is not None and info[0] is L.attn and info[1] is not None


def _state(L, t, s, m, fwd):
    """The bucket's agreed state for this call ("capturing" / "active"), admitting a new bucket (the rank-agreed
    admission: the first call of this forward that meets it, qprep's or plan's), or None (dead / refused)."""
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
    return bs


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
    if _AQ["on"] and not _RES.get("qreserve"):
        need += t.max_s * int(getattr(levers, "AQ_POOL_BYTES_PER_ROW", 0))      # lever AQ's q graphs in the pool
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
    q = (_QKV_NK[1] * 2 + _QKV_NK[0] * 2 + _QB_NK[1] * 2 + L.heads * L.hdim) if _AQ["on"] else 0   # lever AQ
    return rows * (L.heads * L.hdim * 2 + L.n_out * 2 + 8 + q) + 4096


class _Buf:
    """AG's static tensors at `rows` rows, shared by every bucket (views [:S]) and layer (one after the other)."""

    def __init__(self, L, rows):
        dev = L.woa_w.device
        self.rows = rows
        self.o = torch.zeros(rows, L.heads, L.hdim, dtype=torch.bfloat16, device=dev)   # attention output (fmha out)
        self.y = torch.zeros(rows, L.n_out, dtype=torch.bfloat16, device=dev)            # wo_b output (pre all-reduce)
        self.pos = torch.zeros(rows, dtype=torch.int64, device=dev)                       # positions (valid indices)
        bufs = [self.o, self.y, self.pos]
        self.x = self.qkv = self.ql = self.q = None
        if _AQ["on"]:   # lever AQ: the q path's input (x) and outputs (wqkv_a's, the normed q_lora, the FP8 query)
            self.x = torch.zeros(rows, _QKV_NK[1], dtype=torch.bfloat16, device=dev)
            self.qkv = torch.zeros(rows, _QKV_NK[0], dtype=torch.bfloat16, device=dev)
            self.ql = torch.zeros(rows, _QB_NK[1], dtype=torch.bfloat16, device=dev)
            self.q = torch.zeros(rows, L.heads, L.hdim, dtype=torch.float8_e4m3fn, device=dev)
            bufs += [self.x, self.qkv, self.ql, self.q]
        self.nbytes = sum(t.numel() * t.element_size() for t in bufs)


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


# ------------------------------------------------------------------------------------------------ lever AQ: the q path
def qprep(attn, x, positions, forward_batch, attn_backend, q_out, x_quant, q_rope, k_nope, k_rope, defer_all_reduce,
          unified, q_padded):
    """fork_attn.attn_forward's plain prepare branch (no multi-stream prepare): (q, kv) = stock's _forward_prepare
    result for this call with the q path (wqkv_a, QN, wq_b, QR) from this bucket's q graph and stock's remainder
    eager on its static outputs, or None (nothing launched: stock's _forward_prepare runs). The admission agreement is
    made here (before any rank-local test) with plan()'s own gate, so both ranks reach it under the same conditions
    whatever their rank-local lever states; everything after it only picks this rank's path (the q path holds no
    collective)."""
    if not _ST["on"]:
        return None
    fwd = _ON.get()
    if fwd is None or fwd.fb is not forward_batch:
        return None
    if _P["armed"] and not _P["done"] and not _fgp_due():
        # lever AGP: once per process, at the first eligible forward's first prepare call after lever FGP's
        # pre-capture (fg.plan: lever FG's static buffers exist then, for the ZC graphs), on both ranks
        _precapture(fwd)
    fwd.st = None
    if fwd.ago:
        g = _gate_fast(fwd, attn, x, positions, defer_all_reduce, unified, False, q_padded, q_out, None)
        if g is not None:
            fwd.st = (attn, x, q_out, positions, g)   # lever AGO: this attention call's plan() reuses the gate
    else:
        g = _gate(attn, x, positions, defer_all_reduce, unified, False, q_padded, q_out, None)
    if g is None:
        return None
    L, t, m, s = g
    bs = _state(L, t, s, m, fwd)
    if bs is None or not _AQ["on"] or not L.aq or _QSTATE.get(s) == "dead":
        return None
    # rank-local from here: this call's q-path form (stock's branch) and this rank's lever states
    if (x_quant is not None or q_rope is not None or k_nope is not None or k_rope is not None or x.dtype != torch.bfloat16
            or tuple(x.shape) != (m, _QKV_NK[1]) or not x.is_contiguous() or x.data_ptr() & 15 or not _BUF
            or attn._modules.get("wqkv_a") is not L.wqkv or attn._modules.get("wq_b") is not L.wqb
            or not _fb_ok(fwd, forward_batch)):
        _AQ["decline"] += 1
        return None
    if bs == "capturing":
        if L.qgraphs.get(s) is not None or s in fwd.qfail or not _q_local_ok(L, t, m, s, False):
            _AQ["local_eager"] += 1
            return None
        return _qprep_capture(L, m, s, fwd, x, positions, forward_batch, attn_backend, q_out)
    gq = L.qgraphs.get(s)
    if (gq is None or _QSTATE.get(s) != "active" or not fwd.aq
            or not (_q_ok_fast(L, t, m, s, fwd) if fwd.ago else _q_local_ok(L, t, m, s, True))):
        _AQ["local_eager"] += gq is not None and fwd.aq
        return None
    B = _BUF[0]
    z = L.zg.get(s) if fwd.ago else None
    if z is not None and _is_zsrc(x):
        # lever AGO: x is the first m rows of lever FG's static buffer, which this layer's ZC graph reads directly
        if z[1] == "pending":
            return _zcheck(L, m, s, fwd, z, x, positions, forward_batch, attn_backend, q_out)
        if z[1] == "active":
            try:
                _pos(fwd, positions, m)
                z[0].replay()
            except Exception as e:  # noqa: BLE001 - nothing read yet: stock's _forward_prepare recomputes everything
                _z_drop(L, s, f"replay: {type(e).__name__}: {str(e)[:160]}")
                _q_local_fail(s, e)
                return None
            _ZC["replay"] += 1
            _AQ["replay"] += 1
            if levers.first("ago-zc-replay"):
                levers.announce("ago-zc-replay", f"lever AGO: first zero-copy q-graph replay at {m} rows (bucket {s}, "
                                f"layer {getattr(attn, 'layer_id', '?')})")
            return _rest(attn, x, positions, forward_batch, attn_backend, B.qkv[:m], B.ql[:m], B.q[:m])
    try:
        B.x[:m].copy_(x)
        _pos(fwd, positions, m)
        gq.replay()
    except Exception as e:  # noqa: BLE001 - nothing read yet: stock's _forward_prepare recomputes everything
        _q_local_fail(s, e)
        return None
    _AQ["replay"] += 1
    if levers.first("aq-replay"):
        levers.announce("aq-replay", f"lever AQ: first q-path graph replay at {m} rows (bucket {s}, layer "
                        f"{getattr(attn, 'layer_id', '?')})")
    return _rest(attn, x, positions, forward_batch, attn_backend, B.qkv[:m], B.ql[:m], B.q[:m])


def _rest(attn, x, positions, forward_batch, attn_backend, qkv_a, q_lora, q):
    """Stock's _forward_prepare after the q path for this branch (CUDA, not unified, no CP, no fused qk-norm-rope, no
    HQ input: x_linear is x; outside the breakable prefill graph), verbatim calls at the call's rows:
    _compute_kv_to_cache on wqkv_a's output (its kv half), then the low-ratio sources or the indexer / compressor."""
    attn._compute_kv_to_cache(x, positions, forward_batch, attn_backend, qkv_a=qkv_a, q_rope=None)
    kv = None
    if attn.compress_ratio in (1, 2) and (attn.compressor is not None or attn.indexer is not None):
        attn_backend.forward_low_ratio_sources(layer=attn, x=x, q_lora=q_lora, positions=positions,
                                               forward_batch=forward_batch)
    else:
        if attn.indexer is not None:
            attn.indexer(x=x, q_lora=q_lora, forward_batch=forward_batch, attn_backend=attn_backend)
        if attn.compressor is not None:
            attn_backend.forward_core_compressor(x, forward_batch, attn.layer_id, attn.compressor)
    return q, kv


def _fb_test(fb):
    """qprep's forward-batch tests (stock's plain branch here): no context parallelism, an extend, not a verify."""
    return not _H["cp"](fb) and fb.forward_mode.is_extend() and not fb.forward_mode.is_target_verify()


def _fb_ok(fwd, fb):
    """_fb_test, once per forward under lever AGO (fwd.fb is fb: qprep's first test)."""
    if not fwd.ago:
        return _fb_test(fb)
    v = fwd.qfb
    if v is None:
        v = fwd.qfb = bool(_fb_test(fb))
    return v


# ------------------------------------------------------------------------------- lever AGO: zero-copy (ZC) q graphs
def _is_zsrc(x):
    """x (qprep checked: a contiguous [m, 5120] bf16) is the first m rows of the ZC graphs' input (lever FG's static
    next-attention-input buffer, held in _RES["zsrc"]): the bytes a ZC graph reads for the real rows are x's."""
    z = _RES.get("zsrc")
    return z is not None and x.data_ptr() == z.data_ptr()


def _zcheck(L, m, s, fwd, z, x, positions, forward_batch, attn_backend, q_out):
    """A ZC graph's first use on real rows (rank-local, no collective): the eager q path at m rows (stock's calls of
    this branch: what this call returns), then the ZC graph replayed on the same rows (x is its input buffer) and every
    byte of its outputs compared with the eager values on the GPU (read once at the end of the forward, _resolve_z)."""
    attn = L.attn
    qkv_e, _ = attn.wqkv_a(x)
    ql_e = qkv_e[..., : L.qlr]
    ql_e, qfw = attn._normalize_q_lora(ql_e)
    q_e = attn._compute_q_b(qfw, positions, q_out)
    _zcmp(L, m, s, fwd, z, positions, qkv_e, ql_e, q_e)
    return _rest(attn, x, positions, forward_batch, attn_backend, qkv_e, ql_e, q_e)


def _zcmp(L, m, s, fwd, z, positions, qkv_e, ql_e, q_e, x=None):
    """Replay ZC graph z at m rows and compare its outputs with the eager q path's on the GPU, into this entry's own
    flags [input bytes differ, output bytes differ] (read once at the end of the forward, _resolve_z); z becomes
    "checking" until then (any error: z dropped on this rank). With `x` (a capture forward whose x is not the buffer
    itself: lever FG's capture forward returns its eager rows, its graph 2 having written the same rows into the
    buffer), the buffer's first m rows are compared with x too: only equal input bytes make the check count (else z
    stays pending for a later forward), so a passed check always means the ZC graph on x's bytes == the eager path."""
    B = _BUF[0]
    try:
        f = torch.zeros(2, dtype=torch.bool, device=B.q.device)
        if x is not None:
            src = _RES.get("zsrc")
            d = _neq(src[:m], x) if src is not None and src.shape[0] >= m else None
            if d is None:
                return                                # no comparable input: z stays pending
            torch.logical_or(f[0], d, out=f[0])
        _pos(fwd, positions, m)
        z[0].replay()
        for got, want in ((B.qkv[:m], qkv_e), (B.ql[:m], ql_e), (B.q[:m], q_e)):
            d = _neq(got, want) if type(want) is torch.Tensor else None
            if d is None:
                raise RuntimeError("the ZC graph's outputs had another geometry")
            torch.logical_or(f[1], d, out=f[1])
    except Exception as e:  # noqa: BLE001 - the eager values are returned anyway
        _z_drop(L, s, f"first replay: {type(e).__name__}: {str(e)[:160]}", z)
        return
    z[1] = "checking"
    _ZC["check"] += 1
    fwd.zpend.append((L, s, z, f))


def _resolve_z(fwd):
    """End of a forward with ZC first replays (rank-local, one read of every entry's flags): equal inputs and every
    output byte equal -> that ZC graph replaces the copy path; output bytes differ -> dropped (this rank keeps the copy
    path for that layer); inputs differed (a capture forward's buffer did not hold x's bytes) -> pending again. Never
    raises into the forward."""
    pend, fwd.zpend = fwd.zpend, []
    try:
        vals = torch.stack([e[3] for e in pend]).tolist() if pend else []
    except Exception:  # noqa: BLE001
        vals = [[True, True]] * len(pend)
    n, buckets = 0, set()
    for (L, s, z, _f), (din, dout) in zip(pend, vals):
        if L.zg.get(s) is not z or z[1] != "checking":
            continue                              # dropped meanwhile (bucket / q graphs off)
        if din:
            z[1] = "pending"
            continue
        if dout:
            _z_drop(L, s, "ZC graph bytes != eager bytes")
            continue
        z[1] = "active"
        _ZC["pass"] += 1
        n += 1
        buckets.add(s)
    if n:
        b = sorted(buckets)
        levers.announce(f"ago-zc-{b[0]}", f"lever AGO: buckets {b}: {n} zero-copy q graphs' first replay == eager bit "
                        f"for bit on this rank (x read from lever FG's static buffer, no copy)")


def _z_drop(L, s, why, z=None):
    """ZC graph z (default: the one installed for (layer, bucket s)) off on this rank: released, the copy path for
    that layer."""
    if z is None:
        z = L.zg.get(s)
    if z is not None and L.zg.get(s) is z:
        L.zg.pop(s, None)
    if z is not None and z[0] is not None:
        z[1] = "dead"
        try:
            z[0].reset()
        except Exception:  # noqa: BLE001
            pass
        z[0] = None
        _ZC["drop"] += 1
        if why.startswith(("first", "replay", "ZC graph bytes")):
            _ZC["fail"] += 1
        levers.announce(f"ago-zc-drop{s}", f"lever AGO: bucket {s} layer {getattr(L.attn, 'layer_id', '?')}: zero-copy q "
                        f"graph off on this rank ({why[:200]}): x copied as before")
    _pool_check()


def _z_release(s):
    """Every ZC graph of bucket s (its q graphs went off): released on this rank."""
    for _attn, L in list(_LAYERS.values()):
        if L:
            z = L.zg.pop(s, None)
            if z is not None and z[0] is not None:
                z[1] = "dead"
                try:
                    z[0].reset()
                except Exception:  # noqa: BLE001
                    pass
                z[0] = None


def _qn_key(m):
    """lever QN's check key of an eager EXTEND call of m > 64 rows (qn._fast)."""
    return (0, m & 3, not m & 127, m > 128)


def _qr_key(m, L):
    """lever QR's fine check key of the eager call (qr._key: int64 positions, contiguous wq_b view and q_out)."""
    return (8, L.heads, True, True, min(int(m).bit_length(), 15))


def _q_local_ok(L, t, m, s, replay):
    """This rank's lever states allow the q graph for m rows of bucket s (rank-local: only this rank's path)."""
    if _ST["dead"] or not _AQ["on"] or _key_m(L, m) != t.key.get(s) or m <= 64:
        return False
    key = t.key[s]
    if len(key) < 7 or key[4] is None or key[6] is None:
        return False                             # outside DT's bands of wqkv_a / wq_b
    qn, qr, dt, fl = _H["qn"], _H["qr"], _H["dt"], _H["fl"]
    if not qn._STATE.get("ok") or not qr._STATE.get("ok") or fl._ON.get() is None:
        return False
    for lin in (L.wqkv, L.wqb):
        mt = lin.quant_method.mxfp8_prefill_autotune_min_tokens
        if mt is not None and s >= mt:
            return False                         # from the q linears' prefill autotune bound (pin_tactic path)
    sq = dt._KEY_OK.get((_QKV_NK[0], _QKV_NK[1], key[4]))
    sb = dt._KEY_OK.get((_QB_NK[0], _QB_NK[1], key[6]))
    if sq is False or sb is False:
        return False
    if replay:
        # the graph holds QN's unit, QR's unit and DT's band tactics: the eager path at m rows takes them only once
        # QN's key, QR's row class and DT's bands are checked on this rank, and QN takes this layer's call (QN's
        # context set, its per-replica layer facts not stock's: else stock's q_norm and wq_b's own quantize run)
        info = qn._LAYER.get(id(L.attn))
        if (sq is not True or sb is not True or qn._OK.get(_qn_key(m)) is not True
                or qr._OK.get(_qr_key(m, L)) is not True or qn._ON.get() is None or info is None
                or info[0] is not L.attn or info[1] is None):
            return False
    return True


def _pos(fwd, positions, m):
    """The static positions [:M] (copied once a forward; AG's post graphs read the same copy)."""
    if fwd.pos_src is not positions or fwd.pos_m != m:
        _BUF[0].pos[:m].copy_(positions)
        fwd.pos_src, fwd.pos_m = positions, m


def _qseg(L, s, src=None):
    """The q graph at s rows: stock's q path calls of this branch on the static input (x; lever AGO's ZC graph: `src`,
    lever FG's static buffer) and positions, the FP8 query straight into the static q_out, wqkv_a's output and the
    normalized q_lora copied into their static buffers."""
    B = _BUF[0]
    attn = L.attn
    qkv_a, _ = attn.wqkv_a((B.x if src is None else src)[:s])
    if type(qkv_a) is not torch.Tensor or tuple(qkv_a.shape) != (s, _QKV_NK[0]) or qkv_a.dtype != torch.bfloat16:
        raise RuntimeError("wqkv_a's output is not [rows, 1792] bf16")
    q_lora = qkv_a[..., : L.qlr]
    q_lora, q_for_wqb = attn._normalize_q_lora(q_lora)
    qo = B.q[:s]
    q = attn._compute_q_b(q_for_wqb, B.pos[:s], qo)
    if (type(q) is not torch.Tensor or q.data_ptr() != qo.data_ptr() or type(q_lora) is not torch.Tensor
            or tuple(q_lora.shape) != (s, L.qlr) or q_lora.dtype != torch.bfloat16):
        raise RuntimeError("the q path did not write the static q_out / return a [rows, 1280] bf16 q_lora")
    B.qkv[:s].copy_(qkv_a)
    B.ql[:s].copy_(q_lora)


def _qwarm(L, s, src=None):
    """Eager runs of the q segment at s rows on the capture stream until the levers' learn / verify state stops
    changing (DT / FL the bands of wqkv_a and wq_b, QN the s-row key, QR the s-row class). A ZC graph's input (`src`)
    is warmed separately (its own mark: any lever key of that input's layout is learned before its capture)."""
    n = 0
    wk = (s, L.wkey) if src is None else (s, L.wkey, "zc")
    for _ in range(int(levers.AG_MAX_WARMUPS)):
        e0 = _epoch()
        _qseg(L, s, src)
        n += 1
        if _epoch() == e0:
            _QWARM[wk] = _seq()
            return n
    raise RuntimeError(f"the levers' state did not settle in {n} q warmups at {s} rows")


def _qcapture(L, s, src=None):
    cap, pool = _stream(), _pool()
    cur = torch.cuda.current_stream()
    cap.wait_stream(cur)
    t0 = time.perf_counter()
    nwarm = 0
    with torch.cuda.stream(cap):
        if "qreserve" not in _RES and levers.AG_POOL_RESERVE:
            smax = _TABLE[0].max_s
            _RES["qreserve"] = ()
            if _QWARM.get((smax, L.wkey)) != _seq():
                nwarm += _qwarm(L, smax)
            _RES["qreserve"] = (_graph(lambda: _qseg(L, smax), pool),)
        if _QWARM.get((s, L.wkey) if src is None else (s, L.wkey, "zc")) != _seq():
            nwarm += _qwarm(L, s, src)
        if not _q_local_ok(L, _TABLE[0], s, s, True):
            raise RuntimeError("the levers' checks at the bucket's rows did not pass (QN / QR / DT)")
        e0 = _epoch()
        g = _graph(lambda: _qseg(L, s, src), pool)
        if _epoch() != e0:
            g.reset()
            raise RuntimeError("a lever learned / verified during the q capture")
    cur.wait_stream(cap)
    _ST["warm"] += nwarm
    _ST["cap_ms"] += (time.perf_counter() - t0) * 1e3
    return g, nwarm


def _qprep_capture(L, m, s, fwd, x, positions, forward_batch, attn_backend, q_out):
    """The bucket's capture forward, this layer's q path: the eager q path at m rows (what this call returns: stock's
    calls of this branch), then capture + replay once and compare the static outputs [:m] with it on the GPU (read once
    in _resolve). No collective anywhere here."""
    attn = L.attn
    qkv_e, _ = attn.wqkv_a(x)
    ql_e = qkv_e[..., : L.qlr]
    ql_e, qfw = attn._normalize_q_lora(ql_e)
    q_e = attn._compute_q_b(qfw, positions, q_out)
    B = _BUF[0]
    pre = _take(L, s, 2)                       # lever AGP: the q graph pre-captured for (layer, bucket)
    try:
        B.x[:m].copy_(x)
        _pos(fwd, positions, m)
        g, nwarm = pre if pre is not None else _qcapture(L, s)
    except Exception as e:  # noqa: BLE001 - nothing captured: the eager results
        torch.cuda.current_stream().wait_stream(_stream())
        fwd.qfail.setdefault(s, f"q capture: {type(e).__name__}: {str(e)[:200]}")
        _pool_check()
        return _rest(attn, x, positions, forward_batch, attn_backend, qkv_e, ql_e, q_e)
    _AQ["capture"] += pre is None
    if pre is None and levers.first(f"aq-cap{s}"):
        levers.announce(f"aq-cap{s}", f"lever AQ: bucket {s} q path first captured at {m} rows (layer "
                        f"{getattr(attn, 'layer_id', '?')}, {nwarm} warmups)")
    bad = fwd.qbad.get(s)
    if bad is None:
        bad = fwd.qbad[s] = torch.zeros((), dtype=torch.bool, device=x.device)
    ok = True
    try:
        B.x[:m].copy_(x)
        _pos(fwd, positions, m)
        g.replay()
    except Exception as e:  # noqa: BLE001
        ok = False
        fwd.qfail.setdefault(s, f"q replay: {type(e).__name__}: {str(e)[:120]}")
    if ok:
        for got, want in ((B.qkv[:m], qkv_e), (B.ql[:m], ql_e), (B.q[:m], q_e)):
            d = _neq(got, want) if type(want) is torch.Tensor else None
            if d is None:
                ok = False
                break
            torch.logical_or(bad, d, out=bad)
    if ok:
        fwd.qpending.setdefault(s, []).append((L, g))
    else:
        fwd.qfail.setdefault(s, "the q graph's outputs had another geometry")
        g.reset()
        _pool_check()
    zt = _take(L, s, 3)                        # lever AGO: this layer's pre-captured ZC graph goes with the q graph
    if zt is not None:
        z = [zt[0], "pending"]
        fwd.zcap.setdefault(s, []).append((L, z))
        if ok and fwd.ago:
            # checked here too: x is FG's buffer itself, or (FG's own capture forward) a tensor whose bytes FG's graph
            # 2 also wrote into the buffer, compared byte for byte as part of the check
            _zcmp(L, m, s, fwd, z, positions, qkv_e, ql_e, q_e, None if _is_zsrc(x) else x)
    return _rest(attn, x, positions, forward_batch, attn_backend, qkv_e, ql_e, q_e)


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
    try:
        qn, qr = _H["qn"], _H["qr"]
        out += [len(qn._OK), bool(qn._STATE.get("ok")), len(qr._OK), bool(qr._STATE.get("ok"))]
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
    n = len(_RES.get("reserve") or ()) + len(_RES.get("qreserve") or ())
    for _attn, L in _LAYERS.values():
        if L:
            n += len(L.graphs) + len(L.qgraphs) + len(L.zg)
    for d in _PRE.values():                  # lever AGP: pre-captured graphs hold the pool too
        for ent in d.values():
            n += sum(1 for g in ent[1:] if g is not None)
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
    pre = _take(L, s, 1)                     # lever AGP: the post graph pre-captured for (layer, bucket)
    try:
        _positions(p, positions)
        g, nwarm = pre if pre is not None else _capture(L, s)
    except Exception as e:  # noqa: BLE001 - nothing captured: the eager tail on the kept rows (its own all-reduce)
        torch.cuda.current_stream().wait_stream(_stream())
        fwd.fail.setdefault(s, f"capture: {type(e).__name__}: {str(e)[:200]}")
        return _reduce(L, _tail(L, snap, positions))
    _ST["capture"] += pre is None
    if pre is None and levers.first(f"ag-cap{s}"):
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
        qpend = fwd.qpending.pop(s, [])          # lever AQ (empty when AQ is off on this rank)
        qwhy = fwd.qfail.get(s, "")
        qok = not qwhy
        if qok and qpend and levers.AG_CHECK:
            try:
                if bool(fwd.qbad[s].item()):
                    qok, qwhy = False, "q-graph bytes != eager bytes"
            except Exception as e:  # noqa: BLE001
                qok, qwhy = False, f"q check: {type(e).__name__}"
        # one constant-length vector on every rank (AQ on or off): [post ok, post captured some, post captured none,
        # q ok, q captured some, q captured none]
        a = _agree([int(ok), int(bool(pend)), int(not pend), int(qok), int(bool(qpend)), int(not qpend)])
        _ST["check"] += len(pend)
        _AQ["check"] += len(qpend)
        if not a[0] or not (a[1] or a[2]):
            for _L, g in pend + qpend:
                g.reset()
            _kill_bucket(s, f"capture forward at {fwd.rows.get(s, '?')} rows: "
                            f"{why or ('failed on another rank' if not a[0] else 'captured on some ranks only')}")
        elif a[1]:
            for L, g in pend:
                L.graphs[s] = g
            _BSTATE[s] = "active"
            levers.announce(f"ag-chk{s}", f"lever AG: bucket {s} on: {len(pend)} layers' graphs captured at "
                            f"{fwd.rows.get(s, '?')} rows, first replay == eager bit for bit on every rank")
            _resolve_q(s, qpend, qwhy, a)
        else:
            for _L, g in qpend:
                g.reset()
            _RETRY[s] = _RETRY.get(s, 0) + 1
            if _RETRY[s] >= int(getattr(levers, "AG_MAX_CAPTURE_FORWARDS", 8)):
                _kill_bucket(s, f"{_RETRY[s]} capture forwards without a captured layer")
            else:
                for _L, g in pend:
                    g.reset()
                _pool_check()
        _resolve_zcap(fwd, s)
        if _BSTATE.get(s) != "capturing":
            _pre_release(s)                     # lever AGP: pre-captured graphs no layer took


def _resolve_zcap(fwd, s):
    """Lever AGO: the ZC graphs taken in bucket s's capture forward go with their layer's live q graph (state
    "pending" or "checking": _resolve_z decides the ones checked in this forward), else they are released."""
    for L, z in fwd.zcap.pop(s, []):
        if (z[0] is not None and z[1] != "dead" and _BSTATE.get(s) == "active" and _QSTATE.get(s) == "active"
                and L.qgraphs.get(s) is not None and L.zg.get(s) is None):
            L.zg[s] = z
            _ZC["layers"] += 1
        else:
            _z_drop(L, s, "its q graph did not go live", z)


def _pre_release(s):
    """Lever AGP: release bucket s's pre-captured graphs that no layer took (rank-local bookkeeping)."""
    d = _PRE.pop(s, None)
    n = 0
    for ent in (d or {}).values():
        for i in (1, 2, 3):
            if ent[i] is not None:
                try:
                    ent[i].reset()
                except Exception:  # noqa: BLE001
                    pass
                ent[i] = None
                n += 1
    if n:
        _P["dropped"] += n
    _pool_check()


def _take(L, s, i):
    """Lever AGP: (graph, 0 warmups) pre-captured for (layer, bucket s) -- i = 1 post, 2 q, 3 ZC -- removed from the
    cache; None: the lazy capture as before."""
    d = _PRE.get(s)
    ent = d.get(id(L)) if d else None
    if ent is None or ent[0] is not L or ent[i] is None:
        return None
    g, ent[i] = ent[i], None
    _P["taken"] += 1
    return g, 0


def _resolve_q(s, qpend, qwhy, a):
    """Lever AQ's outcome for bucket s, once its post graphs went live (agreed values a[3:6]): the q graphs live too
    (every rank's captures / first replays passed and some rank captured), or AQ off for this bucket for good (a
    failure on some rank, captured on some ranks only, or captured on none: its layers' q paths stay eager)."""
    if not _AQ["on"] and not qpend:
        return
    if not a[3] or not a[4]:
        for _L, g in qpend:
            g.reset()
        _QSTATE[s] = "dead"
        _z_release(s)
        _pool_check()
        if not a[3] or not a[5]:
            _AQ["fail"] += 1
        why = qwhy or ("failed on another rank" if not a[3] else "captured on some ranks only" if not a[5] else
                       "no layer's q path captured on any rank")
        levers.announce(f"aq-dead{s}", f"lever AQ: bucket {s} q graphs OFF for good ({why[:300]}): the q path stays "
                        f"eager for those rows")
        return
    for L, g in qpend:
        L.qgraphs[s] = g
    _QSTATE[s] = "active"
    levers.announce(f"aq-chk{s}", f"lever AQ: bucket {s} on: {len(qpend)} layers' q-path graphs, first replay == eager "
                    f"bit for bit on every rank")


# ------------------------------------------------------------------------------------------- lever AGP: pre-capture
def _precapture(fwd) -> None:
    """qprep's hook (once per process, both ranks): pre-capture every new bucket for every AG layer. Never raises an
    Exception into the forward; the static positions copy of this forward is invalidated after (rewritten)."""
    if not _P["armed"] or _P["done"]:
        return
    _P["done"] = True
    try:
        _prun(fwd)
    except Exception as e:  # noqa: BLE001 - an agreement itself failed (gloo): AG's lazy path from here
        _perr(f"run: {type(e).__name__}: {str(e)[:200]}")
        try:
            levers.announce("agp", f"lever AGP: stopped ({type(e).__name__}: {str(e)[:200]}); pre-captured "
                            f"{_P['captured']} buckets, the rest stays on AG's lazy path")
        except Exception:  # noqa: BLE001 - a failed log line must not reach the forward
            pass
    finally:
        fwd.pos_src, fwd.pos_m = None, -1


def _perr(what) -> None:
    if len(_P["errors"]) < 8:
        _P["errors"].append(what)


def _fgp_due() -> bool:
    """Lever FGP armed and not run yet (fg._FGP's hook: an agreed, rank-symmetric state): the pre-capture waits for it
    (at most to a later layer of the same forward: FGP runs at the first fg.plan of an eligible forward), so lever FG's
    static buffers exist for the ZC graphs. Read only; any doubt: not due."""
    try:
        fgm = _H.get("fg")
        hooks = getattr(fgm, "_FGP", None) if fgm is not None else None
        return bool(hooks) and bool(hooks[0].due())
    except Exception:  # noqa: BLE001
        return False


def _psafe(what, fn, *args):
    """(True, fn(*args)) or (False, None): a rank-local error, answered as a 0 in the next agreement."""
    try:
        return True, fn(*args)
    except Exception as e:  # noqa: BLE001
        _perr(f"{what}: {type(e).__name__}: {str(e)[:200]}")
        return False, None


def _pagree(values):
    _P["agree"] += 1
    return _agree(values)


def _prun(fwd) -> None:
    """Both ranks make the same agreement calls in the same order (the plan, then per bucket: budget, AG's admission
    -- its own two agreements --, the capture outcome); everything rank-local between two of them runs inside _psafe
    and every branch follows agreed values or agreed bucket states."""
    t0 = time.perf_counter()
    budget = float(getattr(levers, "AGP_BUDGET_S", 90.0))
    ok, plan = _psafe("plan", _pplan)
    good = bool(ok and plan is not None)
    sig = (0, 0, 0)
    if good:
        good, sig = _psafe("signature", _psig, plan)
        if not good:
            sig = (0, 0, 0)
    a = _pagree([int(good)] + [v for x in sig for v in (x, -x)])
    if not a[0] or not all(a[1 + 2 * i] == -a[2 + 2 * i] for i in range(len(sig))):
        _pfinish(t0, fwd, 0, "the AG layers / bucket table / new buckets differ between ranks" if a[0] else
                 "no AG layer / table / new bucket on some rank")
        return
    Ls, t, todo = plan
    _P["layers"] = len(Ls)
    _psafe("check ops", _pcheck_ops, Ls[0])
    zsrc, zids = _zc_plan(Ls, t) if _ZC["on"] else (None, set())
    _P["zlayers"] = len(zids)
    for e in todo:
        if _BSTATE.get(e) is not None:            # agreed state only
            continue
        go = _pagree([int(time.perf_counter() - t0 < budget)])
        if not go[0]:
            _perr(f"budget {budget:.0f} s reached before bucket {e}")
            break
        if not _admit(Ls[0], t, e, e, fwd):       # AG's agreed admission (refused: off for good, AG's rule)
            continue
        tb = time.perf_counter()
        ok, got = _psafe(f"bucket {e}", _pbucket, Ls, t, e, zsrc, zids)
        got = got if ok else {}
        npost = sum(1 for ent in got.values() if ent[1] is not None)
        a = _pagree([int(ok), npost])
        if a[0] and a[1] > 0:
            _PRE[e] = got
            _P["captured"] += 1
            _P["graphs"] += sum(1 for ent in got.values() for g in ent[1:] if g is not None)
        else:
            for ent in got.values():
                for i in (1, 2, 3):
                    if ent[i] is not None:
                        try:
                            ent[i].reset()
                        except Exception:  # noqa: BLE001
                            pass
                        ent[i] = None
            _BSTATE.pop(e, None)                  # AG's lazy path (new) on every rank
            _psafe("warm marks", _forget, e)      # its lazy capture warms up again
            _pool_check()
        _P["bucket_ms"][e] = round((time.perf_counter() - tb) * 1e3, 1)
    _pfinish(t0, fwd, len(todo), "")


def _pplan():
    """(AG layers, AG's table, new bucket ends largest first) or None. Deterministic on every rank: load-time facts
    and agreed bucket states only (the layers' AQ eligibility is rank-local and stays out of it)."""
    Ls = [L for L in (_layer(a) for a in _RES.get("attns", ())) if L]
    if not Ls:
        return None
    t = _table(Ls[0])
    if t is None:
        return None
    Ls = [L for L in Ls if _table(L) is t]
    todo = sorted((e for _s, e in t.buckets if _BSTATE.get(e) is None), reverse=True)
    if not todo:
        return None
    return Ls, t, todo


def _psig(plan):
    Ls, _t, todo = plan
    h = 0
    for i, e in enumerate(todo):
        h = (h * 1000003 + (i + 1) * int(e)) % (1 << 40)
    return len(Ls), len(todo), h


def _zc_plan(Ls, t):
    """(lever FG's static next-attention-input buffer, ids of the AQ layers whose x can be its rows) for the ZC
    graphs, or (None, set()): the layers whose previous replica is a lever-FG layer (fg._LAYERS: FG's own verdict,
    read only) and which have no Engram (an Engram rewrites the residual: the combine computes x)."""
    fgm = _H.get("fg")
    try:
        if fgm is None or not fgm._ST.get("on") or not fgm._BUF:
            return None, set()
        nrm = getattr(fgm._BUF[0], "nrm", None)
        B = _BUF[0] if _BUF else None
        if (type(nrm) is not torch.Tensor or nrm.dtype != _BF16 or nrm.dim() != 2 or nrm.shape[1] != _QKV_NK[1]
                or not nrm.is_contiguous() or nrm.shape[0] < t.max_s or nrm.data_ptr() & 255
                or (B is not None and B.x is not None and nrm.device != B.x.device)):
            return None, set()
        chain = _RES.get("chain", ())
        pos = {}
        for i, rep in enumerate(chain):
            mods = getattr(rep, "_modules", None)
            a = mods.get("self_attn") if isinstance(mods, dict) else None
            if a is not None:
                pos[id(a)] = i
        ids = set()
        for L in Ls:
            i = pos.get(id(L.attn))
            if not L.aq or not i:
                continue
            rep, prev = chain[i], chain[i - 1]
            if getattr(rep, "engram", None) is not None:
                continue
            hit = fgm._LAYERS.get(id(prev))
            if hit is not None and hit[0] is prev and hit[1]:
                ids.add(id(L))
    except Exception as e:  # noqa: BLE001 - no ZC graphs (the copy path)
        _perr(f"zc plan: {type(e).__name__}: {str(e)[:160]}")
        return None, set()
    if not ids:
        return None, set()
    _RES["zsrc"] = nrm          # held: the ZC graphs' input address stays this buffer's for the process
    return nrm, ids


def _pfill(B, Ls) -> None:
    """Deterministic synthetic contents of AG's static inputs (AGP_SEED): the attention output and x N(0, 1) bf16,
    positions inside every layer's RoPE table (warmups at S rows run the levers on finite, non-degenerate rows)."""
    g = torch.Generator(device=B.o.device)
    g.manual_seed(int(getattr(levers, "AGP_SEED", 20261015)))
    B.o.normal_(generator=g)
    if B.x is not None:
        B.x.normal_(generator=g)
    rows = min(int(L.attn.freqs_cis.shape[0]) for L in Ls)
    B.pos.random_(0, max(1, min(rows, 1 << 17)), generator=g)


def _pbucket(Ls, t, e, zsrc, zids):
    """Rank-local, no collective: AG's / AQ's warmups + captures of bucket e for every layer whose lever states allow
    them, on the synthetic contents: {id(L): [L, post graph, q graph, ZC graph]}; an error releases what this bucket
    captured and re-raises."""
    B = _BUF[0]
    if not _P["filled"]:
        _pfill(B, Ls)
        _P["filled"] = True
    got = {}
    try:
        for L in Ls:
            ent = [L, None, None, None]
            got[id(L)] = ent
            if _local_ok(L, t, e, e, False):
                ent[1] = _capture(L, e)[0]
            else:
                _P["skipped"] += 1
            if _AQ["on"] and L.aq and _q_local_ok(L, t, e, e, False):
                # a q capture error leaves this layer's q path to the lazy capture (whose own error turns AQ off for
                # the bucket at its first use, AQ's rule); the post graphs stay pre-captured
                for i, src in ((2, None), (3, zsrc)):
                    if i == 3 and (zsrc is None or id(L) not in zids or ent[2] is None):
                        break
                    try:
                        ent[i] = _qcapture(L, e, src)[0]
                    except Exception as ex:  # noqa: BLE001
                        torch.cuda.current_stream().wait_stream(_stream())
                        _perr(f"bucket {e} layer {getattr(L.attn, 'layer_id', '?')} {'ZC' if src is not None else 'q'} "
                              f"capture: {type(ex).__name__}: {str(ex)[:160]}")
                        break
    except BaseException:
        torch.cuda.current_stream().wait_stream(_stream())
        for ent in got.values():
            for i in (1, 2, 3):
                if ent[i] is not None:
                    try:
                        ent[i].reset()
                    except Exception:  # noqa: BLE001
                        pass
                    ent[i] = None
        raise
    return {k: ent for k, ent in got.items() if any(g is not None for g in ent[1:])}


def _pcheck_ops(L0) -> None:
    """The ops of AG / AQ's first-use byte checks (_neq on 2- and 1-byte views, .any(), a 0-d bool flag, logical_or
    into it, the flag's read), once on scratch tensors: their kernels' first load in a fresh process happens here, not
    in the first timed forward that checks a bucket (lever FGP's _warm_check_ops for FG's dtypes). Touches no AG
    state."""
    dev = L0.woa_w.device
    bad = torch.zeros((), dtype=torch.bool, device=dev)
    for dt in (_BF16, _FP8):
        a = torch.zeros(8, dtype=dt, device=dev)
        d = _neq(a, a.clone())
        if d is None:
            raise RuntimeError("check ops: no comparison")
        torch.logical_or(bad, d, out=bad)
    if bool(bad.item()):
        raise RuntimeError("check ops: equal tensors compared unequal")


def _forget(e) -> None:
    for d in (_WARM, _QWARM):
        for k in [k for k in d if k[0] == e]:
            d.pop(k, None)


def _pfinish(t0, fwd, n_new, why) -> None:
    try:
        torch.cuda.current_stream().synchronize()     # the pre-capture's GPU work inside the reported time
    except Exception:  # noqa: BLE001
        pass
    dt = time.perf_counter() - t0
    _P["ms"] = round(dt * 1e3, 1)
    lazy = sorted(e for _s, e in (_TABLE[0].buckets if _TABLE and _TABLE[0] else ()) if _BSTATE.get(e) is None)
    off = sorted(e for e, v in _BSTATE.items() if v == "dead")
    ids = getattr(fwd.fb, "input_ids", None)
    rows = int(ids.shape[0]) if isinstance(ids, torch.Tensor) and ids.dim() >= 1 else -1
    errs = f"; rank-local errors: {' | '.join(_P['errors'])[:400]}" if _P["errors"] else ""
    detail = (f"lever AGP: {_P['captured']}/{n_new} buckets pre-captured ({_P['layers']} AG layers, {_P['graphs']} "
              f"graphs incl. ZC graphs for {_P['zlayers']} layers, {_P['skipped']} layer-buckets left lazy) in "
              f"{dt:.2f} s at the first eligible eager EXTEND forward ({rows} rows); lazy {lazy}; off {off}"
              f"{'; ' + why if why else ''}{errs}")
    _P["detail"] = detail
    levers.announce("agp", detail)


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
    _QSTATE[s] = "dead"
    for _attn, L in list(_LAYERS.values()):
        if L:
            for gs in (L.graphs, L.qgraphs):
                g = gs.pop(s, None)
                if g is not None:
                    g.reset()
    _z_release(s)          # lever AGO
    _pre_release(s)        # lever AGP
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


def _q_local_fail(s, e):
    """A q-graph replay error on this rank: lever AQ's q graphs of bucket s stop here for good on this rank (stock's
    _forward_prepare for those rows); AG's post graphs stay. Rank-local: _QSTATE only picks this rank's path after the
    agreements (the q path holds no collective), so the other rank needs no answer."""
    _QSTATE[s] = "dead"
    _z_release(s)          # lever AGO: its ZC graphs too
    _AQ["fail"] += 1
    levers.announce(f"aq-err{s}", f"lever AQ: bucket {s} q replay failed ({type(e).__name__}: {str(e)[:200]}): q graphs "
                    f"off for good for those rows on this rank (stock q path; AG's post graphs stay)")


def stats() -> dict:
    out = {k: v for k, v in _ST.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out.update({f"aq_{k}": v for k, v in _AQ.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for pre, d in (("ago_", _GO), ("zc_", _ZC), ("agp_", _P)):
        out.update({f"{pre}{k}": v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return out


def summary() -> str:
    live = sorted(s for s, v in _BSTATE.items() if v == "active")
    dead = sorted(s for s, v in _BSTATE.items() if v == "dead")
    return (f"buckets active {live}, dead {dead}; replays {_ST['replay']}, captures {_ST['capture']}, checks "
            f"{_ST['check']}, warmups {_ST['warm']}, capture {_ST['cap_ms']:.0f} ms, sweeps {_ST['sweep']} "
            f"({_ST['sweep_ms']:.0f} ms), eager fallbacks {_ST['eager']}, local eager {_ST['local_eager']}, "
            f"failures {_ST['fail']}; AQ q graphs active {sorted(s for s, v in _QSTATE.items() if v == 'active')}, "
            f"dead {sorted(s for s, v in _QSTATE.items() if v == 'dead')}, replays {_AQ['replay']}, captures "
            f"{_AQ['capture']}, checks {_AQ['check']}, local eager {_AQ['local_eager']}, failures {_AQ['fail']}; AGO "
            f"fast gates {_GO['fast']}, plan reuses {_GO['reuse']}, ZC layers {_ZC['layers']} (checks {_ZC['check']}, "
            f"pass {_ZC['pass']}, dropped {_ZC['drop']}, replays {_ZC['replay']}); AGP {_P['captured']} buckets "
            f"pre-captured ({_P['graphs']} graphs, {_P['taken']} taken, {_P['dropped']} released, {_P['ms']:.0f} ms)")

