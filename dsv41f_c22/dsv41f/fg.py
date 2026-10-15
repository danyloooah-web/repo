"""Lever FG on the fork (levers.FG): FFN-block CUDA graphs for our eager EXTEND forwards.

WHY. Live warm turns prefill ~1.9k new tokens on contexts above 16384, where the fork's breakable prefill graphs do not
apply, so the pass runs eager and is host-bound on TP-1 (c6 profile, 2000 rows on 60k cached: GPU idle ~70 %). The FFN
half of a layer (run_ffn_hc) is ~40 host launches: router gate GEMM, router, NVFP4 quantize + trtllm-gen MoE, the shared
expert's two MXFP8 linears + activation, the shared add, the TP all-reduce and the mega-mHC post (~1.1 ms of host a layer
unprofiled). FG replays that block from two CUDA graphs per layer and row bucket instead.

WHAT RUNS (per eligible layer call, M real rows, bucket S >= M; `fg_layer.forward_hc_pre_from_prev` is the fork's
DeepseekV4DecoderLayer.forward_hc_pre_from_prev lifted with three guarded edits, stock otherwise):
  attention post  eager at M rows, stock's _post_fusion mega branch (deep_gemm.mega_mhc), but writing its five outputs
                  (updated residual, normalized FFN input, the FFN triplet pre / post / comb) straight into FG's static
                  input buffers [:M] (fg_layer.mhc_mega_boundary_out = stock mhc_mega_boundary with output arguments).
  router gate     eager at M rows: stock's cuBLAS torch.mm(x, W^T, out_dtype=fp32) written into the static logits [:M]
                  (cuBLAS picks its kernel per M, so it stays outside the graph).
  graph 1 (S)     the MoE replica's own forward_normal from the router on (moe_fwd.forward_normal_fg: forward_normal
                  with two asserted edits -- the logits come in, the all-reduce is left out): router (stock Triton
                  kernel, padded rows masked by num_token_non_padded = M: weight 0 / id -1, the form stock's own padded
                  and E5-pruned rows take, which FlashInfer's routed MoE and finalize skip), NVFP4 quantize, trtllm-gen
                  MoE (lever FMT's band pin, written into the static MoE output through stock's moe_output_buffer_ctx),
                  shared expert, shared add.
  all-reduce      eager on the real rows [:M] through the same moe_fwd.post_experts_all_reduce stock calls (lever AR's
                  entry); an out-of-place result (custom AR) is copied back into the static MoE output.
  graph 2 (S)     the FFN post: stock's mega boundary on the static buffers, into the static outputs (next residual,
                  next attention input, next triplet), which the next layer's eager attention reads as views [:M].
  Collectives never run inside a graph. Graphs are captured with every eager lever active (the launches the eager
  forward makes at S rows), after warmups at S until the levers' learn / verify state stops changing.

EXACTNESS. Every output byte of the M real rows equals the eager stock path at M rows:
  * the only M-dependent work outside the graphs (gate GEMM, all-reduce) runs at M rows as stock does;
  * inside the graphs every kernel is row-independent (GEMMs over tokens with a fixed K order, per-row router /
    quantize / norm / sinkhorn / elementwise ops, the finalize's per-token sum), and padded rows are masked out of the
    MoE (they add no expert work and change no per-expert token count);
  * every launch decision stock makes from the row count is equal for M and S: buckets never cross an edge of the
    decision key (lever FMT's band, the MoE PDL bound, the MoE tile centre, each shared MXFP8 linear's (pinned tactic or
    AutoTuner bucket, 2T/SF quantize) choice, deep_gemm mega_mhc's split-K count), and the FMT tactic must be valid at
    both M and S (FlashInfer's own trtllm_get_valid_moe_configs, per new M);
  * proven in tests/fg_test.py over many M per bucket (edges, random, poisoned padded rows) against the eager stock
    path, and in the engine: the first replay of every (bucket, layer) is compared with the eager path at M rows byte
    for byte (the eager result is what that call returns); any difference turns the bucket off for good.
SCOPE. Our node's eager EXTEND forwards only (no capture, no breakable / piecewise prefill graph, not under
SGLang's capture mode, text-only batches, the noop expert-distribution recorder), rows in [FG_MIN_ROWS, FG_MAX_ROWS],
and never when the engine runs with CUDA graphs disabled (the node audit's eager twin stays plain eager).
RANKS. Every decision taken from rank-local state (free memory, capture success, the first-replay check, FMT's band
check) is agreed over the TP CPU group (gloo MIN) at two call sites both ranks reach under the same deterministic
conditions: a new bucket's admission, and the end of its capture forward (one agreement per bucket: every layer's
capture, replay and byte check passed on every rank -> its graphs go live; some ranks captured and others not -> off;
no rank captured -> retried at most FG_MAX_CAPTURE_FORWARDS forwards). Between them every rank's path (capture,
replay, stock) makes the same single all-reduce per FFN block, so a rank-local difference can neither hang nor diverge
the ranks; the replay path makes no extra collective. A bucket FMT's band check turned off (or on a rank whose FG
failed) is off at its admission, so no forward pays an agreement for a bucket that can never go live.
TUNED BUCKETS. Where a shared-expert MXFP8 linear takes FlashInfer's AutoTuner tactic (>= the prefill autotune bound,
4096 rows), the graphs hold the tactic the tuner gave at capture: such a bucket replays only while the AutoTuner state
(lever FL's token) is the one it went live under, and stays eager when that state cannot be tracked (an autotune_v2
managed store, as lever FL).
MEMORY. One set of static buffers (FG_MAX_ROWS rows, shared by every bucket and layer: layers run one after the other
on one stream) and one private graph pool shared by every graph (no graph output lives in the pool; the first capture
reserves it at the largest bucket so later buckets reuse its segments), admitted per bucket by a free-memory gate.
MEASURED (2026-10-14 / 15, box2 B300, fork image 5e72ce5, results/fg/):
  tests/fg_test.py (TP2 per-rank shapes, TP1 math, real SGLang DeepseekV2MoE: NVFP4 trtllm-gen experts, MXFP8 CuTe-DSL
  shared expert, mega mHC): 178/178 FG calls byte-identical to the eager path over 131 row counts in 21 buckets x 2
  layers (each bucket's edges + random rows; 42 first-use captures + checks, 136 replays, 45 with NaN / inf / 3e38 in
  every padded input row, 19 also vs pure stock = the served MoE without levers); split-K mirror == deep_gemm's printed
  count at 17 row counts; a forced capture failure turns that bucket off and leaves the pool usable. Host per layer
  (attention post + FFN block) 626-653 us eager -> 202-207 us FG (600..8000 rows); GPU span -21 % (600 rows) .. -7 %
  (4500), +0.1 / +2.7 % at 6000 / 8000 rows (padding). Memory: static buffers 894 MiB (8192 rows) + graph pool 1054 MiB
  for all buckets (7.6 GiB without the reserve).
  In engine (dev lane, TP2, unprofiled, single warm turns on 59904 cached tokens), steady TTFT, bundle_c8 -> bundle_fg:
  2096 rows, three engine pairs 0.122 -> 0.095 s (-22 %), 0.133 -> 0.078 s (-42 %), 0.122 -> 0.094 s (-23 %); 796
  rows 0.116 -> 0.087 s (-25 %); inside ONE engine (levers.FG_AB = 2: FG on every other eligible forward) 2096 rows
  0.130 -> 0.101 s (-23 %, 6 + 6 turns), 712..952 rows 0.124 -> 0.106 s (-15 %). A bucket's capture forward costs
  +0.08..0.14 s once (37 layers: warmups, capture, replay, eager reference, byte check); the engine's first capture
  also reserves the pool (+0.37 s once; it falls in the cold opening). Both ranks: every first replay == eager bit for
  bit, 0 tracebacks; pf output tokens == bundle_c8's (24 / 24 turns over four read pairs).
REVIEW (2026-10-15, adversarial review on box2, results/fg/review/):
  * EVERY replay checked in the engine (tests/fg_chkall_gen.py diagnostic bundle, never shipped: each replay followed by
    the eager path at M rows on the same inputs, every byte of the gate logits, graph 1 before the all-reduce and graph 2
    compared; no extra collective): reads fgchk1 (pre-review kit; full suite k1..24 + pf 6000), fgchk2 (pre-EP-guard
    review kit; full suite + pf 2000), fgchk3 (this kit; suite k1..8 with 900 new tokens a request + pf 3000): >= 48
    replay forwards, >= 1776 layer replays on both ranks (single and multi-request batches, 616..7704 rows, pinned and
    tuned MXFP8 buckets, real AutoTuner cache) == eager bit for bit, 0 mismatches; 27 bucket first-use checks pass.
  * tests/fg_test.py (this kit): 178 / 178 as before, plus review_tests: every row count of 513..639, 897..1023 and
    4097..4736 (894 replays) == eager; the tuned-bucket AutoTuner guard; a capture forward captured on one rank only ->
    bucket off; empty capture forwards -> retried 8x then off; FMT band off -> off at admission; recapture after the
    levers' key caches dropped the bucket's keys warms up again (the pre-review kit learned inside that capture and lost
    the bucket); an unreadable decision key keeps its rows on stock.
  * torch 2.14.1: a thread_local capture invalidated by a host sync (capture_end raises cudaErrorStreamCaptureInvalidated)
    leaves no allocator routing behind (freed record_stream blocks recycle, the pool takes new captures).
  * tests/fg_evict_repro.py: after the levers' key caches dropped a bucket's keys, the pre-review kit lost the bucket on
    its next capture forward (no warmup -> a key learned inside the capture -> off for good); this kit recaptures it
    (2 warmups, == eager). Seal probe (DSV4_DEVBIND_SEAL=1) of c9 + this kit: clean after prepare and forwards 1..300.
    pf 2000 steady TTFT back to back, pre-review kit 0.095-0.102 s vs this kit 0.096-0.098 s (no host cost added).

LEVER FGB (levers.FGB, same kit, same buckets / buffers / agreements): the two eager launches FG left in every layer
of an eager EXTEND pass -- the attention post's mega boundary (deep_gemm mega_mhc, ~130 us of host a call inside
deep_gemm's C++ launcher, profiled) and the router gate GEMM (cuBLAS torch.mm, ~68 us) -- move into graph 1:
  "m" (a layer whose previous layer is an FG layer: its attention post reads FG's static outputs B.upd / B.npre /
      B.npost / B.ncomb as residual + triplet): graph 1 = the mega boundary on FGB's static inputs (the REDUCED
      attention output copied into B.ao [:M]; the attention all-reduce stays eager, inside the attention) into FG's
      static FFN inputs, then the gate GEMM (torch.mm(B.x[:S], W^T, out_dtype=fp32, out=B.logits[:S])), then FG's
      segment. The attention post only copies the attention output and returns the static views (deferred: graph 1
      writes them first thing in the FFN half; every exit of fg.ffn that does not replay graph 1 runs the eager mega
      boundary on the static inputs first -- stock's kernel at M rows on the same bytes).
  "g" (the first FG layer of a run of FG layers): graph 1 = the gate GEMM + FG's segment (FG's eager attention post).
  "p" (FGB off, or the bucket's gate sweep / agreement failed): FG as before.
Exactness. The mega boundary is the same kernel as graph 2's (shifted mHC, bf16 output, hidden 5120, mult 4): every
row-count-dependent launch decision is its split-K count (deep_gemm get_num_splits(ceil(M / 64), hidden, SMs): already
an edge of FG's decision key) and the persistent grid (num SMs), every op per row (the split-K partials of a row are
reduced in a fixed order, Sinkhorn / RMSNorm per row): rows [:M] at S == stock at M (FG's graph 2 has proven this form
since c16). cuBLAS picks the gate GEMM's kernel per row count (B300, cuBLAS of torch 2.14.1+cu130, results/fgb/fgbp1:
split-K kernels at 385..400, 409..480, 496, 505..520, 785..792 rows change the summation order; every non-split-K
kernel over 385..8192 rows gives the same bytes), so a bucket is admitted with the gate in its graph only after a
sweep on this rank (deterministic random rows, the layer's real gate weight, stock's own call form): stock's
torch.mm at every row count of the bucket == the first rows of the graph's out= form at S, byte for byte; the verdict
is agreed (gloo MIN) inside FG's admission agreement (one more value), so both ranks capture the same variant. The
first replay of every (bucket, layer) is compared with the eager path at M rows (stock's attention post -- the triplet,
then _post_fusion -- fresh, then stock's gate / MoE / all-reduce / post): the mega's five outputs, the gate logits
(and the graph's gate at S rows against an eager torch.mm at S on the graph's own input, every row), the MoE output and
graph 2's outputs; any difference turns the bucket off for good (FG's rule). Padded rows [M:S] of the static inputs
reach only padded rows. Rank safety: FGB adds no collective (the variant is a static layer fact + the agreed gate
verdict; the deferred / copy / fallback paths are rank-local, every path makes FG's one all-reduce). levers.FGB_AB = N
> 0 (diagnostic): FG's plain graph 1 is captured beside (checked at the capture forward too) and only every N-th
eligible forward replays FGB's graph 1, the others run FG's c19x path (eager mega + eager gate + plain graph 1).
levers.FGB_MAX_ROWS: buckets above it keep FG's plain graph 1 (a static switch; graph 1 at S rows pays the padding of
the mega + gate on the GPU, and a copy of the attention output, against ~90-110 us of host a layer).
Cost: one static buffer more (B.ao, FG_MAX_ROWS x hidden bf16, 80 MiB); the gate sweep ~0.3 ms a row count, all
buckets ~1.4 s a rank in the engine (inside FGP's pre-capture, the untimed cold opening).
MEASURED (2026-10-15, box1 B300, fork image 5e72ce5, results/fgb/):
  tests/fgb_gate_probe.py (fgbp1): split-K gate kernels only at 385..400, 409..480, 496, 505..520, 785..792 rows; every
  other row count == the first rows at 8192.
  tests/fgb_test.py (fgb1, chains of 3 real layers): gate sweeps of 19/19 buckets == the probe's classes (14 consistent:
  every bucket from 833 rows on); 327 + 5097 layer outputs == the eager chain bit for bit (capture forwards "g" / "m" /
  "p", replays, poisoned padded rows of every static input incl. ao / upd / npre / npost / ncomb, pure stock on a
  sample; exhaustive 833..1023, 1821..2048, 4097..5376); fallbacks (fresh state copied into place, a declined FFN half
  materialized, no precomputed triplet declined, an injected replay-1 failure materialized) == eager; FGB_AB twin
  (3 graphs a layer, on / off alternating) == eager; FGP pre-capture of 19/19 buckets x 3 layers with FGB variants,
  first uses on the pre-captured graphs == eager. Host per layer (3-layer chain, unit harness): -86..-93 us at
  1100..6000 rows. Regressions: tests/fg_test.py PASS (158/158 + review tests), tests/fgp_test.py PASS (fgbfgp2).
  In engine (c20fgb_ab, FGB_AB 2, unprofiled, 60k cached; FGP 19/19 buckets x 37 layers in 3.1 s incl. the sweeps,
  every bucket's first use == eager on both ranks, 0 tracebacks): pf 1000 (1096 rows, bucket 1344) on 0.063-0.066
  (median 0.064) vs off 0.068-0.073 (median 0.070): -8.6 %; pf 2000 (2096 rows, bucket 2688, 28 % padding) on
  0.081-0.087 (median 0.084) vs off 0.083-0.085 (median 0.084): +0.0 % (GPU-bound there: the host saved is absorbed,
  the graph's padding + copy cost ~0.8 ms of GPU a pass). Medians from the reads' JSON TTFTs (5 on / 5 off each):
  1096 rows -8.1 %, 1596 rows -0.2 %, 2096 rows +0.4 %, 3096 rows (bucket 4032) -0.4 %, 6096 rows (bucket 7040) +1.2 %
  (on 130.2-132.2 vs off 129.8-131.0 ms: GPU-bound, the padding + copy cost shows) -> FGB_MAX_ROWS 4096 (buckets above
  keep FG's plain graph 1). CPU+GPU profile at 2000 rows (fgb20cp vs c19cp): deep_gemm mega_mhc's launcher (4.8 ms
  self a pass) and aten::mm (3.7 ms) gone from the host top list, the GPU gap after the attention all-reduce + before the
  gate GEMM (14.4 ms a pass profiled) down to ~6 ms, profiled pass window 143.0 -> 133.7 ms, GPU busy 59.0 -> 59.3 ms;
  unprofiled the attention post's host work at >= ~1600 rows hides under the sparse-MLA kernel (~160 us a layer at
  2096 rows), so the saving shows only at smaller row counts. In-engine first-event lines: every read 'm' / 'g' replays,
  0 copies (the previous FG replay's outputs always found in place), 0 declines, 0 materializations. Seal probe run
  (c20fgb, FGB_MAX_ROWS 8192: bench suite unprofiled): 18x clean, 0 tracebacks, every lever check == stock.
"""
import contextvars
import time
import types

import torch

from dsv41f import levers

_ON = contextvars.ContextVar("dsv41f_fg_on", default=None)   # per eligible forward: _Fwd
_H: dict = {}            # handles imported at install
_ST = {"detail": "not installed", "on": False, "dead": False, "replay": 0, "capture": 0, "check": 0, "eager": 0,
       "fail": 0, "admit": 0, "refuse": 0, "warm": 0, "cap_ms": 0.0, "ab": 0, "fwd": 0, "stale": 0}
_LAYERS: dict = {}       # id(layer replica) -> (replica, _Layer | False)
_BSTATE: dict = {}       # bucket S -> "capturing" | "active" | "dead" (absent: new)
_WARM: dict = {}         # bucket S -> the eligible forward (_Fwd.seq) whose capture warmed the levers up at S rows
_RETRY: dict = {}        # bucket S -> capture forwards that ended with no layer captured on any rank (agreed count)
_TOK: dict = {}          # bucket S -> FlashInfer AutoTuner state when its graphs went live (tuned MXFP8 buckets)
_VALID: dict = {}        # (rows, tactic) -> the MoE tactic is valid at rows (FlashInfer's own valid-config list)
_TABLE: list = []        # [_Table] once built
_BUF: list = []          # [_Buf] once allocated
_RES: dict = {}          # "pool" / "stream" once created
_MOE_TMPL: list = []     # [(moe_op, query key with num_tokens = None)] once found
_FGP: list = []          # lever FGP (fgp.py, gen_fgp.py): [its hooks] once fgp.install armed it
_TILES = (8, 16, 32, 64, 128, 256)   # trtllm-gen MoE tile candidates (the launcher's supported tile list)
_PREV: dict = {}         # lever FGB: id(layer replica) -> the previous replica in the model's layer order (or None)
_GATE: dict = {}         # lever FGB: bucket S -> the agreed verdict "the gate GEMM goes into graph 1" (admission)
_GSWEEP: dict = {}       # lever FGB: bucket S -> this rank's gate row-consistency sweep (True: consistent)
_VRANK = {"p": 0, "g": 1, "m": 2}   # lever FGB: graph-1 variants, each one's launches a superset of the previous one's
_FGB = {"on": False, "ab": 0, "defer": 0, "copy": 0, "mat": 0, "replay_m": 0, "replay_g": 0, "replay_off": 0,
        "sweep": 0, "sweep_ms": 0.0, "gate_on": 0, "gate_off": 0, "decline": 0}

# sha256[:16] of the stock functions FG lifts, mirrors or relies on (fork 5e72ce5, FlashInfer 0.7.1); levers.FG_PINS
PIN_TARGETS = (
    ("sglang.srt.models.deepseek_v4", "DeepseekV4DecoderLayer.forward_hc_pre_from_prev"),
    ("sglang.srt.models.deepseek_v4", "DeepseekV4DecoderLayer._run_moe_ffn_dp_sync"),
    ("sglang.srt.models.deepseek_v4", "DeepseekV4DecoderLayer._can_fuse_ffn_mhc"),
    ("sglang.srt.models.deepseek_v4_mhc", "run_attn_post"),
    ("sglang.srt.models.deepseek_v4_mhc", "run_moe_post"),
    ("sglang.srt.models.deepseek_v4_mhc", "_post_fusion"),
    ("sglang.srt.models.deepseek_v4_mhc", "_compute_triplet"),
    ("sglang.srt.models.deepseek_v4_mhc", "combine"),
    ("sglang.srt.models.deepseek_v4_mhc", "_can_use_mega_mhc"),
    ("sglang.kernels.ops.layernorm.mhc_mega", "mhc_mega_boundary"),
    ("sglang.srt.models.deepseek_v2", "DeepseekV2MoE.forward"),
    ("sglang.srt.models.deepseek_v2", "DeepseekV2MoE.forward_normal"),
    ("sglang.srt.models.deepseek_v2", "MoEGate.forward"),
    ("sglang.srt.models.deepseek_v2", "DeepseekV2MLP.forward"),
    ("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm", "_get_fp4_moe_output_buffer"),
    ("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm", "trtllm_moe_enable_pdl"),
    ("sglang.srt.layers.moe.moe_runner.base", "moe_output_buffer_ctx"),
    ("sglang.kernels.ops.gemm.bf16_fp32", "_linear_bf16_fp32_cublas"),
    ("sglang.kernels.ops.moe.moe_fused_gate", "moe_fused_gate"),
    ("sglang.kernels.ops.moe.moe_fused_gate", "_router_triton_kernel"),
    ("sglang.srt.multimodal.dsv41.vl_routing", "vision_topk"),
    ("sglang.srt.multimodal.dsv41.vl_routing", "_finish_vision_topk"),
    ("sglang.srt.layers.quantization.fp8_utils", "_mxfp8_decode_tactic"),
    ("sglang.srt.layers.quantization.fp8_utils", "flashinfer_mxfp8_blockscaled_linear"),
    ("sglang.srt.layers.quantization.fp8", "Fp8LinearMethod.apply"),
    ("flashinfer.quantization.kernels.mxfp8_quantize", "mxfp8_quantize_cute_dsl"),
    ("flashinfer.fused_moe.backends.trtllm.sm100_runner", "MoERunner.get_valid_tactics"),
    ("flashinfer.fused_moe.utils", "map_to_hybrid_bucket_uncapped"),
    # stock's run_ffn_hc wraps the MoE in decode_rows (E5 pruning on for decode / verify batches only); the graphs run
    # without it, which is exact while decode_rows / prune_topk leave EXTEND alone (pinned)
    ("sglang.srt.models.deepseek_v4_prune", "decode_rows"),
    ("sglang.srt.models.deepseek_v4_prune", "prune_topk"),
)
DEEP_GEMM_VERSION = "0.2.2"     # mega_mhc's split-K heuristic (csrc/apis/mega_mhc.hpp get_num_splits) mirrored below


def _set(detail: str) -> str:
    _ST["detail"] = detail
    return detail


def source_hash(text: str, qualname: str):
    """sha256 (16 hex) of a def's source text (decorators included, dedented) in a module's source TEXT, found by ast
    (the last def of that name wins); None if absent. Same algorithm as gen_fg.py (and kits/ar.py)."""
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
    elif name == "sglang.srt.models.deepseek_v4_mhc":
        from sglang.srt.models import deepseek_v4_mhc as mod
    elif name == "sglang.srt.models.deepseek_v4_prune":
        from sglang.srt.models import deepseek_v4_prune as mod
    elif name == "sglang.kernels.ops.layernorm.mhc_mega":
        from sglang.kernels.ops.layernorm import mhc_mega as mod
    elif name == "sglang.srt.models.deepseek_v2":
        from sglang.srt.models import deepseek_v2 as mod
    elif name == "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm":
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as mod
    elif name == "sglang.srt.layers.moe.moe_runner.base":
        from sglang.srt.layers.moe.moe_runner import base as mod
    elif name == "sglang.kernels.ops.gemm.bf16_fp32":
        from sglang.kernels.ops.gemm import bf16_fp32 as mod
    elif name == "sglang.kernels.ops.moe.moe_fused_gate":
        from sglang.kernels.ops.moe import moe_fused_gate as mod
    elif name == "sglang.srt.multimodal.dsv41.vl_routing":
        from sglang.srt.multimodal.dsv41 import vl_routing as mod
    elif name == "sglang.srt.layers.quantization.fp8_utils":
        from sglang.srt.layers.quantization import fp8_utils as mod
    elif name == "sglang.srt.layers.quantization.fp8":
        from sglang.srt.layers.quantization import fp8 as mod
    elif name == "flashinfer.quantization.kernels.mxfp8_quantize":
        from flashinfer.quantization.kernels import mxfp8_quantize as mod
    elif name == "flashinfer.fused_moe.backends.trtllm.sm100_runner":
        from flashinfer.fused_moe.backends.trtllm import sm100_runner as mod
    elif name == "flashinfer.fused_moe.utils":
        from flashinfer.fused_moe import utils as mod
    else:
        raise KeyError(f"no static import for pinned module {name}")
    if mod.__name__ != name:
        raise ImportError(f"{name} resolved to {mod.__name__}")
    return inspect.getsource(mod)


def check_pins() -> str:
    """'' when every pinned stock function's source hash equals levers.FG_PINS, else what differs."""
    bad = []
    texts: dict = {}
    for mod_name, qual in PIN_TARGETS:
        want = levers.FG_PINS.get(f"{mod_name}:{qual}")
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
    import deep_gemm
    from sglang.kernels.ops.layernorm import mhc_mega
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.eplb.expert_distribution import (
        _ExpertDistributionRecorderNoop,
        get_global_expert_distribution_recorder,
    )
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as ft
    from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.model_executor import cuda_graph_config as cgc
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
        is_in_tc_piecewise_cuda_graph,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
    from sglang.srt.models import deepseek_v2 as dv2
    from sglang.srt.models import deepseek_v4 as dv4
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.runtime_context import get_forward

    import flashinfer
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe import core as fcore
    from flashinfer.fused_moe.backends.trtllm import MoERunner as TrtllmMoERunner
    from flashinfer.gemm import gemm_base as gb
    from flashinfer.quantization.kernels import mxfp8_quantize as mq

    from dsv41f import fmt, moe_fwd

    if flashinfer.__version__ != "0.7.1":
        raise RuntimeError(f"flashinfer {flashinfer.__version__} (mirror read from 0.7.1)")
    if getattr(deep_gemm, "__version__", "?") != DEEP_GEMM_VERSION:
        raise RuntimeError(f"deep_gemm {getattr(deep_gemm, '__version__', '?')} (split-K mirror read from "
                           f"{DEEP_GEMM_VERSION})")
    for mod, names in ((ft, ("_get_fp4_moe_output_buffer", "trtllm_moe_enable_pdl")),
                       (fp8_utils, ("_mxfp8_decode_tactic",)),
                       (mhc, ("HcState", "HcNormed", "AttnOutput", "_compute_triplet", "_can_use_mega_mhc",
                              "run_attn_post", "run_moe_post")),
                       (mhc_mega, ("mhc_mega_boundary",)), (fmt, ("_PIN", "_BAND_OK")),
                       (moe_fwd, ("forward_normal", "forward_normal_fg", "fg_gate", "post_experts_all_reduce")),
                       (gb, ("_MM_MXFP8_CUTE_DSL_TUNING_CONFIG",)), (mq, ("MXFP8_2T_SF_THRESHOLD",)),
                       (TrtllmMoERunner, ("valid_tactics_dict",)), (fcore, ("get_trtllm_moe_sm100_module",)),
                       (deep_gemm, ("get_num_sms", "get_deterministic_algorithms", "mega_mhc")),
                       (cgc, ("check_cuda_graph_backend", "cuda_graph_fully_disabled"))):
        for n in names:
            if not hasattr(mod, n):
                raise AttributeError(f"{getattr(mod, '__name__', mod)}.{n}")
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

    _H.update(dg=deep_gemm, mega=mhc_mega.mhc_mega_boundary, tp_group=get_tp_group,
              rec=get_global_expert_distribution_recorder, rec_noop=_ExpertDistributionRecorderNoop, ft=ft,
              buf_ctx=moe_output_buffer_ctx, fp8_utils=fp8_utils, Fp8=Fp8LinearMethod, cgc=cgc, Phase=Phase,
              Backend=Backend, EXTEND=ForwardMode.EXTEND, in_bcg=is_in_breakable_cuda_graph,
              in_pcg=is_in_tc_piecewise_cuda_graph, capture_mode=get_is_capture_mode, dv2=dv2, dv4=dv4, mhc=mhc,
              get_forward=get_forward, fcore=fcore, MoERunner=TrtllmMoERunner, gb=gb,
              two_t=int(mq.MXFP8_2T_SF_THRESHOLD), fmt=fmt, moe_fwd=moe_fwd, tuner=AutoTuner)


def _graphs_disabled() -> str:
    """Why the engine runs without CUDA graphs ('' when it does not): FG stays off then, so cacheon's eager twin
    (engine_worker: disable_cuda_graph=True for the audit role -> both phases' backend DISABLED) runs plain eager. A
    config that is not published (check_cuda_graph_backend answers False for every backend) counts as disabled."""
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
    """model_node.prepare: bind the lifted layer forward and the MoE graph segment on our private replicas."""
    if not getattr(levers, "FG", False):
        return _set("off (levers.FG)")
    if _ST["on"]:
        return _ST["detail"]
    try:
        _import_handles()
        from dsv41f import fg_layer
    except Exception as e:  # noqa: BLE001 - another stack: stock everywhere
        return _set(f"off ({type(e).__name__}: {str(e)[:200]})")
    why = _graphs_disabled()
    if why:
        return _set(f"off ({why}: the eager twin stays plain eager)")
    if getattr(levers, "FG_PIN", True):
        bad = check_pins()
        if bad:
            return _set(f"off (stock changed: {bad[:400]})")
    served = list(getattr(model, "layers", ()))
    dv4 = _H["dv4"]
    stock_fhc = dv4.DeepseekV4DecoderLayer.forward_hc_pre_from_prev
    moe_fwd = _H["moe_fwd"]
    n_layer = n_moe = 0
    layers = list(layers)
    _PREV.clear()
    for i, rep in enumerate(layers):   # lever FGB: the model's layer order (the fork loop runs layers[i] after [i - 1])
        _PREV[id(rep)] = layers[i - 1] if i > 0 else None
    _FGB["on"] = bool(getattr(levers, "FGB", False))
    for i, rep in enumerate(layers):
        src = served[i] if i < len(served) else None
        if (rep is src or not isinstance(rep, dv4.DeepseekV4DecoderLayer)
                or type(rep).forward_hc_pre_from_prev is not stock_fhc or not getattr(rep, "hc_pre_from_prev_sublayer",
                                                                                       False)):
            continue
        rep.forward_hc_pre_from_prev = types.MethodType(fg_layer.forward_hc_pre_from_prev, rep)
        n_layer += 1
        moe = rep._modules.get("mlp")
        smoe = src._modules.get("mlp") if src is not None else None
        if (moe is not None and moe is not smoe
                and getattr(getattr(moe, "forward_normal", None), "__func__", None) is moe_fwd.forward_normal):
            moe.dsv4f_fg_seg1 = types.MethodType(moe_fwd.forward_normal_fg, moe)
            moe.dsv4f_fg_gate = types.MethodType(moe_fwd.fg_gate, moe)
            n_moe += 1
    _ST["on"] = n_moe > 0
    lo, hi = int(levers.FG_MIN_ROWS), int(levers.FG_MAX_ROWS)
    return _set(f"{'on' if _ST['on'] else 'off'}: {n_layer} layer replicas run the lifted layer forward, {n_moe} MoE "
                f"replicas carry the graph segment; eager EXTEND forwards of {lo}..{hi} rows (buckets <= "
                f"{levers.FG_PAD:.0%} padding inside constant-choice regions), graphs captured lazily per bucket, first "
                f"replay per (bucket, layer) checked bitwise vs eager{'' if levers.FG_CHECK else ' (CHECK OFF)'}"
                f"{_fgb_detail()}")


def _fgb_detail() -> str:
    if not _FGB["on"]:
        return "; lever FGB off"
    ab = int(getattr(levers, "FGB_AB", 0) or 0)
    return (f"; lever FGB on: graph 1 also holds the router gate GEMM (buckets whose cuBLAS sweep is consistent, agreed)"
            f"{' and, after an FG layer, the attention post mega boundary' if getattr(levers, 'FGB_MEGA', True) else ''}"
            f"{f' (A/B: FGB on every {ab}-th eligible forward)' if ab else ''}")


# -------------------------------------------------------------------------------------------------------- forward gate
class _Fwd:
    """One eligible forward: what FG already set up for it."""

    __slots__ = ("fb", "refused", "nt_m", "ids_src", "ids_m", "capturing", "pending", "fail", "bad", "rows", "seq",
                 "tok", "fgb", "so")

    def __init__(self, fb):
        self.fb = fb
        _ST["fwd"] += 1
        self.seq = _ST["fwd"]     # this eligible forward's number (a bucket's warmup is redone in each capture forward)
        self.tok = _tuner_token()  # FlashInfer's AutoTuner state in this forward (tuned buckets replay only under theirs)
        self.refused = set()      # buckets whose admission was refused in this forward
        self.nt_m = -1            # the row count the static padding count holds
        self.ids_src = None       # the input-ids tensor copied into the static ids
        self.ids_m = -1
        self.capturing = set()    # buckets in their capture forward (state "capturing", agreed): resolved at the end
        self.pending = {}         # bucket -> [(layer state, graphs)] captured + replayed in this forward
        self.fail = {}            # bucket -> why a capture / check step failed on this rank
        self.bad = {}             # bucket -> 0-d GPU bool: some graph byte != the eager byte
        self.rows = {}            # bucket -> the real row count it was captured at
        self.fgb = False          # lever FGB: this forward replays FGB's graph 1 (False: FG's plain graph 1 / FGB off)
        self.so = None            # lever FGB: (residual, triplet, rows) the last FG replay returned (FG's static views)


def forward(fn, forward_batch, *args, **kwargs):
    """Run our model forward with lever FG active on eager EXTEND, text-only forwards. Every test here is the same on
    both TP ranks (mode, capture state, the batch, the engine config); a rank-local failure (_ST["dead"]) must NOT end
    the context early: the other rank may be heading for an agreement in plan() / ffn(), which this rank then answers
    with False instead of leaving it hanging."""
    if not _ST["on"]:
        return fn(*args, **kwargs)
    mode = getattr(forward_batch, "forward_mode", None)
    if (mode is None or mode != _H["EXTEND"] or torch.cuda.is_current_stream_capturing() or _H["in_bcg"]()
            or _H["in_pcg"]() or _H["capture_mode"]() or torch.compiler.is_compiling()):
        return fn(*args, **kwargs)
    try:
        mm = bool(forward_batch.contains_mm_inputs())
        rec_ok = type(_H["rec"]()) is _H["rec_noop"]
    except Exception:  # noqa: BLE001
        mm, rec_ok = True, False
    if mm or not rec_ok:
        return fn(*args, **kwargs)
    if levers.FG_AB and not _ab_on(forward_batch):
        return fn(*args, **kwargs)
    fwd = _Fwd(forward_batch)
    if _FGB["on"]:
        fwd.fgb = _fgb_ab_on(forward_batch) if getattr(levers, "FGB_AB", 0) else True
    token = _ON.set(fwd)
    try:
        out = fn(*args, **kwargs)
    finally:
        _ON.reset(token)
    if fwd.capturing:
        _resolve(fwd)              # both ranks: the bucket was "capturing" (an agreed state) when this forward met it
    return out


def _tuner_token():
    """FlashInfer's AutoTuner state, as lever FL reads it (fl._tuner_token): it changes when a tuning run or a loaded
    config can give a tuned MXFP8 bucket another tactic than the one a graph captured. None (tuned buckets stay eager):
    unreadable, or an autotune_v2 managed store is attached (it can gain entries on disk without a change here; lever
    FL leaves such forwards to stock too)."""
    try:
        tuner = _H["tuner"].get()
        if tuner._managed_cache is not None:
            return None
        return (tuner, tuner._dirty_seq, len(tuner.profiling_cache), len(tuner._file_configs),
                len(tuner._winner_partitions))
    except Exception:  # noqa: BLE001
        return None


def _ab_on(forward_batch):
    """Diagnostic A/B inside one engine (levers.FG_AB = N > 0, never in a shipped bundle): of the eligible forwards with
    FG_MIN_ROWS..FG_MAX_ROWS rows, only every N-th runs with FG (the others take the plain eager path), announced per
    forward. The count is the same on both ranks (same batches, same tests), so both ranks take the same branch."""
    ids = forward_batch.input_ids
    rows = -1 if ids is None else int(ids.shape[0])
    if not levers.FG_MIN_ROWS <= rows <= levers.FG_MAX_ROWS:
        return True
    _ST["ab"] += 1
    on = _ST["ab"] % levers.FG_AB == 0
    levers.announce(f"fg-ab{_ST['ab']}", f"lever FG A/B: eligible forward {_ST['ab']} ({rows} rows): "
                    f"FG {'on' if on else 'off'}")
    return on


def _fgb_ab_on(forward_batch):
    """Lever FGB's diagnostic A/B (levers.FGB_AB = N > 0, never in a shipped bundle): of the FG-eligible forwards with
    FG_MIN_ROWS..FG_MAX_ROWS rows only every N-th replays FGB's graph 1, the others FG's plain path (c19x). Same count
    on both ranks."""
    ids = forward_batch.input_ids
    rows = -1 if ids is None else int(ids.shape[0])
    if not levers.FG_MIN_ROWS <= rows <= levers.FG_MAX_ROWS:
        return True
    _FGB["ab"] += 1
    on = _FGB["ab"] % int(levers.FGB_AB) == 0
    levers.announce(f"fgb-ab{_FGB['ab']}", f"lever FGB A/B: eligible forward {_FGB['ab']} ({rows} rows): "
                    f"FGB {'on' if on else 'off'}")
    return on


# --------------------------------------------------------------------------------------------- per-layer static state
class _Layer:
    __slots__ = ("layer", "moe", "ffn_hc", "attn_hc", "local_b", "next_b", "next_hc", "hidden", "mult", "experts",
                 "top_k", "gate_w", "wt", "lin_gu", "lin_dn", "vision", "graphs", "sig", "member", "var")


def _layer(layer):
    hit = _LAYERS.get(id(layer))
    if hit is not None and hit[0] is layer:
        return hit[1]
    try:
        L, why = _bind_layer(layer)
    except Exception as e:  # noqa: BLE001 - deterministic on every rank (same module structure)
        L, why = False, f"{type(e).__name__}: {str(e)[:160]}"
    if not L:
        levers.announce(f"fg-layer-{why[:40]}", f"lever FG: layer {getattr(layer, 'layer_id', '?')} stays eager "
                        f"({why})")
    _LAYERS[id(layer)] = (layer, L)
    return L


def _bind_layer(layer):
    """(_Layer, '') when this layer's FFN block takes the graphs, else (False, why). Static (load-time) facts only; the
    boundaries exist once _init_boundaries ran (the lifted forward calls plan() after it)."""
    mhc = _H["mhc"]
    cfg = getattr(layer, "hc_cfg", None)
    lb, nb = getattr(layer, "local_boundary", None), getattr(layer, "next_boundary", None)
    if cfg is None or not cfg.pre_from_prev or cfg.cp_prefill:
        return False, "not a pre-from-prev mHC layer"
    if lb is None or nb is None or lb.hc is None or nb.hc is None or not lb.norm_fusable or not nb.norm_fusable:
        return False, "no mega seam on both posts (last layer / before an Engram layer)"
    if lb.hc is not layer.ffn_hc or not mhc._can_use_mega_mhc(cfg) or not mhc._can_use_mega_mhc(lb.hc.cfg):
        return False, "the posts do not take stock's mega kernel"
    moe = layer._modules.get("mlp")
    if moe is None or getattr(moe, "dsv4f_fg_seg1", None) is None:
        return False, "MoE replica without FG's graph segment"
    exp = moe._modules.get("experts")
    se = moe._modules.get("shared_experts")
    gate = moe._modules.get("gate")
    if exp is None or se is None or gate is None:
        return False, "MoE without experts / shared experts / gate"
    dv2 = _H["dv2"]
    if (getattr(moe, "is_hash", True) or getattr(moe, "is_nextn", True) or moe._enable_a2a_moe
            or moe.num_fused_shared_experts != 0 or moe._shared_expert_tp1 or not moe.reduce_results
            or moe._fuse_shared_experts_inside_sbo):
        return False, "MoE layout (hash / MTP / a2a / fused shared / TP1 shared / no reduce / SBO)"
    if moe._moe_quant_once_enabled():
        return False, "quant-once MoE input"
    if getattr(dv2, "_is_hip", False) or not getattr(dv2, "_is_cuda", False):
        return False, "not the CUDA build"
    from sglang.srt.layers.moe.mega_moe import _use_amd_flydsl_mega_moe
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    if get_moe_a2a_backend().is_megamoe() or not get_moe_a2a_backend().is_none() or _use_amd_flydsl_mega_moe():
        return False, "MegaMoE / a2a backend"
    from sglang.srt.runtime_context import get_parallel

    par = get_parallel()
    if par.attn_dp_size != 1 or par.attn_cp_size != 1 or (par.attn_tp_size > 1 and not get_moe_a2a_backend().is_none()):
        return False, "DP / CP attention (the MoE input is gathered)"
    if int(getattr(par, "moe_ep_size", 1)) != 1:
        # the decision key (MoE tile centre / tactic validity) was read for the TP-sharded MoE of this arena (ep 1)
        return False, "expert-parallel MoE"
    if gate.e_score_correction_bias_vl is None:
        return False, "no vision-routing bias (the router is not vision_topk)"
    w = gate._parameters.get("weight")
    if not (isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_contiguous()):
        return False, "router gate weight not bf16 [experts, hidden]"
    from sglang.kernels.ops.gemm import bf16_fp32

    if (type(gate).forward is not dv2.MoEGate.forward or gate._forward_hooks or gate._forward_pre_hooks
            or bf16_fp32._linear_bf16_fp32_algo != "cublas" or gate.tiny_router_gemm_max_tokens >= levers.FG_MIN_ROWS):
        return False, "router gate is not stock's cuBLAS torch.mm at these rows"
    if (getattr(se, "_enable_nvfp4_gemm_swiglu_fusion", False) or se.swiglu_limit is None or se.is_replicated
            or se.down_proj.weight.dtype == torch.uint8 or getattr(se.down_proj, "reduce_results", False)):
        # (a down_proj that all-reduces would put a collective inside graph 1)
        return False, "shared expert path (NVFP4 swiglu fusion / no clamp / fused down / reducing down_proj)"
    lins = []
    for lin in (se.gate_up_proj, se.down_proj):
        meth = getattr(lin, "quant_method", None)
        if not isinstance(meth, _H["Fp8"]) or getattr(lin, "bias", None) is not None:
            return False, "shared expert linear not Fp8LinearMethod"
        backend = getattr(meth, "mxfp8_dense_backend", None)
        if (backend is None or not backend.is_flashinfer_cutedsl()
                or not (meth.use_mxfp8 or (meth.block_fp8_as_mxfp8 and getattr(lin, "block_fp8_mxfp8_ready", False)))):
            return False, "shared expert linear not MXFP8 on CuTe-DSL"
        part = _served_partial(meth.w8a8_mxfp8_linear)
        if (part is None or part.func is not _H["fp8_utils"].flashinfer_mxfp8_blockscaled_linear
                or part.keywords.get("backend") != "cute-dsl"):
            return False, "shared expert linear is not stock's partial(flashinfer_mxfp8_blockscaled_linear, cute-dsl)"
        lins.append(lin)
    cfgk = moe.topk.topk_config
    L = _Layer()
    L.layer, L.moe = layer, moe
    L.ffn_hc, L.attn_hc, L.local_b, L.next_b, L.next_hc = layer.ffn_hc, layer.attn_hc, lb, nb, nb.hc
    L.hidden, L.mult = int(cfg.hidden), int(cfg.mult)
    L.experts, L.top_k = int(exp.num_local_experts), int(cfgk.top_k)
    L.gate_w, L.wt = w, w.t()
    L.lin_gu, L.lin_dn = lins
    L.vision = True
    L.graphs = {}
    L.member = None   # lever FGB: the previous layer is an FG layer (decided at first need, _variant)
    L.var = {}        # lever FGB: bucket S -> the graph-1 variant its graphs were captured as
    L.sig = (L.hidden, L.mult, int(w.shape[0]), L.experts, L.top_k, int(exp.intermediate_size_per_partition),
             tuple(lins[0].weight.shape), tuple(lins[1].weight.shape), str(w.device))
    if int(w.shape[0]) != int(exp.num_experts) or int(w.shape[1]) != L.hidden:
        return False, "gate / experts shapes"
    return L, ""


# ---------------------------------------------------------------------------------------------- the decision key / table
def _served_partial(fn):
    """The served method's partial(flashinfer_mxfp8_blockscaled_linear, ...) under the levers' wrappers (FL keeps it
    as .stock, DT's / FL's previous callable as .inner), or None."""
    import functools

    for _ in range(6):
        if isinstance(fn, functools.partial):
            return fn
        nxt = getattr(fn, "stock", None)
        fn = nxt if nxt is not None else getattr(fn, "inner", None)
        if fn is None:
            return None
    return None


def _fl_key(m, lin):
    """Stock's launch decisions for an MXFP8 linear at m rows (Fp8LinearMethod.apply -> flashinfer_mxfp8_blockscaled_
    linear): pinned (the decode tactic for m, None = FlashInfer's default) or tuned (the AutoTuner bucket of m), plus
    mxfp8_quantize_cute_dsl's 2T/SF choice."""
    meth = lin.quant_method
    n, k = int(lin.weight.shape[0]), int(lin.weight.shape[1])
    stock = _served_partial(meth.w8a8_mxfp8_linear)
    pin0 = bool(stock.keywords.get("pin_tactic", False))
    min_tok = meth.mxfp8_prefill_autotune_min_tokens
    pin = pin0
    if min_tok is not None and m >= min_tok:
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
        from sglang.srt.runtime_context import get_exec

        pin = bool(is_batch_invariant_mode_enabled() or get_exec().deterministic.enable_deterministic_inference)
    if pin:
        choice = _H["fp8_utils"]._mxfp8_decode_tactic(m, n, lin.weight.device)
        choice = None if choice is None else tuple(tuple(c) if isinstance(c, (list, tuple)) else c for c in choice)
    else:
        choice = int(_H["gb"]._MM_MXFP8_CUTE_DSL_TUNING_CONFIG.dynamic_tensor_specs[0].map_to_tuning_buckets(m))
    return (pin, choice, m * (k // 32) >= _H["two_t"])


def _splits(m, hidden, sms, det):
    """deep_gemm 0.2.2 mega_mhc's split-K count (csrc/apis/mega_mhc.hpp get_num_splits; BLOCK_M / BLOCK_K 64,
    16 default / 64 max splits)."""
    if det:
        return 16
    nkb = hidden // 64
    nmb = -(-m // 64)
    mx = min(64, max(16, sms // nmb))
    return -(-nkb // -(-nkb // mx))


def _band(m):
    for i, (lo, hi, _t) in enumerate(levers.FMT_TACTICS):
        if lo <= m <= hi:
            return i
    return None


def _centre(m, experts, top_k):
    avg = m * top_k / experts
    for t in _TILES:
        if t >= avg:
            return t
    return _TILES[-1]


class _Table:
    __slots__ = ("sig", "buckets", "los", "key", "min_tok", "sms", "det", "max_s", "tuned")


def _key(L, m, sms, det):
    ft = _H["ft"]
    return (_band(m), bool(ft.trtllm_moe_enable_pdl(m)), _centre(m, L.experts, L.top_k), _fl_key(m, L.lin_gu),
            _fl_key(m, L.lin_dn), _splits(m, L.hidden, sms, det))


def _table(L):
    """The bucket table (built once, at the first eligible call: every input is a load-time / startup fact)."""
    if _TABLE:
        t = _TABLE[0]
        return t if t is not None and t.sig == L.sig else None
    try:
        t = _build_table(L)
    except Exception as e:  # noqa: BLE001 - deterministic on every rank
        levers.announce("fg-table", f"lever FG: bucket table failed ({type(e).__name__}: {str(e)[:160]}): off")
        t = None
    _TABLE.append(t)
    return t


def _build_table(L):
    lo, hi = int(levers.FG_MIN_ROWS), int(levers.FG_MAX_ROWS)
    pad, align = float(levers.FG_PAD), max(1, int(levers.FG_ALIGN))
    dg = _H["dg"]
    sms, det = int(dg.get_num_sms()), bool(dg.get_deterministic_algorithms())
    keys = [_key(L, m, sms, det) for m in range(lo, hi + 1)]
    buckets = []
    a = lo
    while a <= hi:
        b = a
        while b + 1 <= hi and keys[b + 1 - lo] == keys[a - lo]:
            b += 1
        if keys[a - lo][0] is not None:               # rows outside every FMT band stay eager
            s = a
            while s <= b:
                want = -(-int(s * (1.0 + pad)) // align) * align
                end = min(b, max(s, want))
                buckets.append((s, end))
                s = end + 1
        a = b + 1
    t = _Table()
    t.sig, t.sms, t.det = L.sig, sms, det
    t.buckets = buckets
    t.los = [x[0] for x in buckets]
    t.key = {e: keys[e - lo] for _s, e in buckets}
    # buckets whose shared-expert linears take the AutoTuner's tactic (not pinned): their graphs hold the tactic the
    # tuner gave at capture time, so they replay only while the tuner state is the one they went live under
    t.tuned = {e: not (keys[e - lo][3][0] and keys[e - lo][4][0]) for _s, e in buckets}
    t.min_tok = (L.lin_gu.quant_method.mxfp8_prefill_autotune_min_tokens,
                 L.lin_dn.quant_method.mxfp8_prefill_autotune_min_tokens)
    t.max_s = max((e for _s, e in buckets), default=0)
    regions = sum(1 for i in range(1, len(keys)) if keys[i] != keys[i - 1]) + 1
    levers.announce("fg-table", f"lever FG: {len(buckets)} buckets over {lo}..{hi} rows in {regions} constant-choice "
                    f"regions (edges: FMT bands, MoE PDL / tile centre, shared MXFP8 tactic + 2T/SF, mega_mhc splits; "
                    f"{sms} SMs, prefill autotune bound {t.min_tok[0]}); bucket sizes "
                    f"{[e for _s, e in buckets][:12]}{'...' if len(buckets) > 12 else ''}")
    return t


def _bucket(t, m):
    import bisect

    i = bisect.bisect_right(t.los, m) - 1
    if i < 0:
        return None
    s, e = t.buckets[i]
    return e if s <= m <= e else None


# ------------------------------------------------------------------------------------- MoE tactic validity at a row count
def _moe_template(L):
    """(moe_op, query key) of FlashInfer's TrtllmMoERunner.get_valid_tactics for this layer's routed op, taken from the
    runner's own valid-tactics cache (filled by lever FMT's validity checks); None until one exists."""
    if _MOE_TMPL:
        return _MOE_TMPL[0]
    exp = L.moe._modules["experts"]
    want = (L.top_k, L.hidden, int(exp.intermediate_size_per_partition), L.experts)
    try:
        moe_op = _H["fcore"].get_trtllm_moe_sm100_module().moe_op
    except Exception:  # noqa: BLE001
        return None
    found = set()
    for key in list(_H["MoERunner"].valid_tactics_dict.keys()):
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == id(moe_op)):
            continue
        qk = key[1]
        if not (isinstance(qk, tuple) and len(qk) == 14):
            continue
        if (int(qk[3]), int(qk[4]), int(qk[6]), int(qk[7])) != want or qk[13]:
            continue
        found.add(qk[:12] + (None,) + qk[13:])
    if len(found) != 1:
        return None
    _MOE_TMPL.append((moe_op, found.pop()))
    return _MOE_TMPL[0]


def _tactic_valid(L, rows, tactic):
    key = (rows, tactic)
    hit = _VALID.get(key)
    if hit is not None:
        return hit
    tmpl = _moe_template(L)
    if tmpl is None:
        return False
    moe_op, qk = tmpl
    try:
        valid = moe_op.trtllm_get_valid_moe_configs(*(qk[:12] + (rows,) + qk[13:]))
        ok = any(tuple(int(x) for x in t) == tactic for t in valid)
    except Exception:  # noqa: BLE001
        ok = False
    if len(_VALID) >= 32768:
        _VALID.clear()
    _VALID[key] = ok
    return ok


def _same_choices(L, t, m, s):
    """Every launch decision stock makes from the row count is equal for m and s (deterministic on every rank)."""
    if t.key.get(s) is None or _key_m(L, t, m) != t.key[s]:
        return False
    band = t.key[s][0]
    fmt = _H["fmt"]
    if fmt._BAND_OK.get(band) is not True:
        return False
    tactic = tuple(int(x) for x in levers.FMT_TACTICS[band][2])
    if not (_tactic_valid(L, m, tactic) and _tactic_valid(L, s, tactic)):
        return False
    dg = _H["dg"]
    if int(dg.get_num_sms()) != t.sms or bool(dg.get_deterministic_algorithms()) != t.det:
        return False
    if (L.lin_gu.quant_method.mxfp8_prefill_autotune_min_tokens,
            L.lin_dn.quant_method.mxfp8_prefill_autotune_min_tokens) != t.min_tok:
        return False
    return True


_KEYS: dict = {}


def _key_m(L, t, m):
    """The decision key at m rows (cached per m); None when a launch decision cannot be read (stock at m rows)."""
    hit = _KEYS.get(m)
    if hit is None:
        try:
            hit = _key(L, m, t.sms, t.det)
        except Exception:  # noqa: BLE001 - e.g. a served MXFP8 method rebound since load: never a graph at m rows
            hit = False
        _KEYS[m] = hit
    return hit if hit is not False else None


# --------------------------------------------------------------------------------------------------------- the plan
class _Plan:
    __slots__ = ("L", "rows", "S", "fwd", "capture", "broken", "v", "stash", "deferred")

    def __init__(self, L, rows, S, fwd, capture, v="p"):
        self.L, self.rows, self.S, self.fwd, self.capture = L, rows, S, fwd, capture
        self.broken = False
        self.v = v                # lever FGB: graph 1's variant for this call ("m" / "g" / "p")
        self.stash = None         # lever FGB capture forward: the attention post's inputs (out, residual, triplet)
        self.deferred = False     # lever FGB "m" replay: the attention post's mega boundary waits for graph 1


def plan(layer, state, forward_batch, stats_stream):
    """forward_hc_pre_from_prev (lifted), right after the boundaries are resolved: this call's FG plan, or None (stock).
    The two agreement sites (a new bucket's admission here, a capture forward's resolution in fg.forward) are reached by
    both ranks under the same deterministic conditions (rows, static layer facts, agreed bucket states); rank-local
    facts (FMT's band check, a local failure) only decide this rank's path -- capture, replay or stock, all of them
    exact and all making the same one all-reduce -- or enter as the values agreed on."""
    fwd = _ON.get()
    if _FGP and fwd is not None and fwd.fb is forward_batch and _FGP[0].due():
        _FGP[0].precapture(fwd)   # lever FGP: every bucket admitted + warmed + captured once, here (both ranks)
    if fwd is None or stats_stream is not None or fwd.fb is not forward_batch:
        return None
    res = state.streams
    if type(res) is not torch.Tensor or res.dim() != 3:
        return None
    m = int(res.shape[0])
    if m < levers.FG_MIN_ROWS or m > levers.FG_MAX_ROWS:
        return None
    L = _layer(layer)
    if (not L or layer.local_boundary is not L.local_b or layer.next_boundary is not L.next_b
            or L.moe.gate._parameters.get("weight") is not L.gate_w):
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
    local = not _ST["dead"] and _same_choices(L, t, m, s) and _not_fused(layer, m, L)
    graphs = L.graphs.get(s)
    if local and t.tuned.get(s) and (fwd.tok is None or (graphs is not None and _TOK.get(s) != fwd.tok)):
        # a tuned-MXFP8 bucket under an untrackable AutoTuner state, or whose tuner state moved since its graphs went
        # live (stock's tactic may differ from the captured one): eager for these rows on this rank (every path makes
        # the same one all-reduce)
        local = False
        _ST["stale"] += 1
        if levers.first(f"fg-stale{s}"):
            levers.announce(f"fg-stale{s}", f"lever FG: bucket {s} (tuned MXFP8 tactics): AutoTuner state "
                            f"{'untrackable' if fwd.tok is None else 'changed since its graphs went live'}: eager for "
                            f"those rows")
    if graphs is None:
        # the capture path holds no collective but the all-reduce the eager path makes too, so a rank-local skip is safe;
        # the bucket's outcome is agreed once, at the end of this forward (_resolve)
        if bs != "capturing" or not local or s in fwd.fail:
            return None
        return _Plan(L, m, s, fwd, True, _variant(L, s))
    if not local:
        return None
    v = L.var.get(s, "p")
    if v != "p" and not fwd.fgb:
        # lever FGB A/B "off" forward: FG's plain graph 1 (captured beside FGB's), FG's eager attention post + gate
        if len(graphs) < 3:
            return None
        v = "p"
        _FGB["replay_off"] += 1
    return _Plan(L, m, s, fwd, False, v)


def _variant(L, s) -> str:
    """Lever FGB: what graph 1 of (layer L, bucket s) holds -- "m": the attention post's mega boundary + the router
    gate GEMM + FG's segment (the previous layer is an FG layer: this layer's attention post reads FG's static
    outputs); "g": the gate GEMM + FG's segment (the first FG layer of a run); "p": FG's segment only (FGB off, or
    bucket s's gate sweep / agreement failed). Deterministic on every rank: static layer facts (the layer order,
    which replicas bind) and the bucket's agreed gate verdict."""
    if not _fgb_bucket(s) or _GATE.get(s) is not True:
        return "p"
    if L.member is None:
        prev = _PREV.get(id(L.layer))
        hit = _LAYERS.get(id(prev)) if prev is not None else None
        L.member = bool(hit is not None and hit[0] is prev and hit[1] and getattr(levers, "FGB_MEGA", True))
    return "m" if L.member else "g"


def _fgb_bucket(s) -> bool:
    """Lever FGB applies to bucket s (FGB on, s <= FGB_MAX_ROWS: a static switch, the same on every rank)."""
    return bool(_FGB["on"]) and s <= int(getattr(levers, "FGB_MAX_ROWS", 1 << 30))


def _wvar(s) -> str:
    """Lever FGB: the warmup variant of bucket s (the superset of every layer's graph-1 launches at s rows)."""
    if not _fgb_bucket(s) or _GATE.get(s) is not True:
        return "p"
    return "m" if getattr(levers, "FGB_MEGA", True) else "g"


def _not_fused(layer, m, L):
    """Stock's FFN-side fused all-reduce + mHC epilogue does not apply at m rows (run_ffn_hc's own test)."""
    try:
        return not (layer._can_fuse_ffn_mhc and _H["dv4"].can_fuse_all_reduce(m, L.hidden))
    except Exception:  # noqa: BLE001
        return False


def _admit(L, t, s, m, fwd) -> bool:
    """A new bucket: rank-agreed admission (FMT band verified, free memory for the buffers + graph pool)."""
    band = t.key[s][0]
    band_state = _H["fmt"]._BAND_OK.get(band)        # True: verified == stock; False: FMT turned the band off
    band_ok = band_state is True
    if levers.FG_POOL_RESERVE and "reserve" not in _RES:
        need = t.max_s * int(levers.FG_POOL_BYTES_PER_ROW)     # the reserve capture at the largest bucket
    elif _RES.get("reserve"):
        need = int(levers.FG_POOL_SLACK)                       # carved out of the reserve's segments
    else:
        need = s * int(levers.FG_POOL_BYTES_PER_ROW)           # no reserve (off or failed): this bucket's own growth
    need += 0 if _BUF else _buf_bytes(L, t.max_s)
    # lever FGB: this rank's gate row-consistency sweep (rank-local, collective-free; cached per bucket), agreed below
    gate_ok = bool(_fgb_bucket(s) and band_ok and _gate_sweep(L, t, s))
    try:
        free, _total = torch.cuda.mem_get_info()
        avail = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    except Exception:  # noqa: BLE001
        avail = 0
    mem_ok = avail - need >= int(levers.FG_MEM_MARGIN)
    a = _agree([int(band_ok), int(mem_ok), int(band_state is not False and not _ST["dead"]), int(gate_ok)])
    if not a[2]:
        # FMT's band off on some rank, or FG failed on some rank: this bucket can never go live -- off for good
        # instead of a new admission (one more agreement) in every forward of these rows
        _BSTATE[s] = "dead"
        _ST["refuse"] += 1
        levers.announce(f"fg-never{s}", f"lever FG: bucket {s} OFF (FMT band {band} off or FG failed on some rank)")
        return False
    if not a[0]:
        _ST["refuse"] += 1
        levers.announce(f"fg-wait{s}", f"lever FG: bucket {s} waits (FMT band {band} not verified on every rank yet)")
        return False
    if not a[1]:
        _BSTATE[s] = "dead"
        levers.announce(f"fg-mem{s}", f"lever FG: bucket {s} OFF (free memory: {avail / 2**30:.1f} GiB available, "
                        f"{need / 2**30:.2f} GiB needed + {int(levers.FG_MEM_MARGIN) / 2**30:.0f} GiB margin)")
        return False
    ok = True
    if not _BUF:
        try:
            ids = getattr(fwd.fb, "input_ids", None)
            _BUF.append(_Buf(L, t.max_s, ids.dtype if isinstance(ids, torch.Tensor) else torch.int64))
        except Exception as e:  # noqa: BLE001
            ok = False
            levers.announce("fg-buf", f"lever FG: static buffers failed ({type(e).__name__}: {str(e)[:120]})")
    if not _agree_bool(ok):
        _BSTATE[s] = "dead"
        return False
    _BSTATE[s] = "capturing"
    _GATE[s] = bool(a[3])                             # lever FGB: agreed (every rank swept the bucket consistent)
    if _FGB["on"]:
        _FGB["gate_on" if a[3] else "gate_off"] += 1
    _ST["admit"] += 1
    if levers.first("fg-admit"):
        b = _BUF[0]
        levers.announce("fg-admit", f"lever FG: first bucket {s} admitted at {m} rows; static buffers {b.rows} rows "
                        f"({b.nbytes / 2**20:.0f} MiB), {avail / 2**30:.1f} GiB available")
    return True


# ------------------------------------------------------------------------------------------------------- static buffers
def _buf_bytes(L, rows):
    """_Buf's size at `rows` rows: x, out, nrm (h bf16), res, upd (c x h bf16), two triplets (2c + c^2 fp32), logits
    (experts fp32), ids (int64)."""
    h, c, e = L.hidden, L.mult, int(L.gate_w.shape[0])
    per = 3 * h * 2 + 2 * c * h * 2 + 2 * (2 * c + c * c) * 4 + e * 4 + 8
    per += h * 2 if _FGB["on"] else 0          # lever FGB: the reduced attention output (ao, bf16)
    return rows * per + 4096


class _Buf:
    """FG's static tensors at `rows` rows, shared by every bucket (views [:S]) and layer (one after the other)."""

    def __init__(self, L, rows, ids_dtype=torch.int64):
        dev = L.gate_w.device
        h, c, e = L.hidden, L.mult, int(L.gate_w.shape[0])
        bf, f32 = torch.bfloat16, torch.float32
        self.rows = rows
        self.x = torch.zeros(rows, h, dtype=bf, device=dev)          # FFN input (normalized)
        self.res = torch.zeros(rows, c, h, dtype=bf, device=dev)     # FFN residual
        self.pre = torch.zeros(rows, c, 1, dtype=f32, device=dev)    # FFN triplet
        self.post = torch.zeros(rows, c, 1, dtype=f32, device=dev)
        self.comb = torch.zeros(rows, c, c, dtype=f32, device=dev)
        self.logits = torch.zeros(rows, e, dtype=f32, device=dev)    # router gate output
        self.ids = torch.zeros(rows, dtype=ids_dtype, device=dev)    # router input ids (the batch's id dtype)
        self.nt = torch.zeros((), dtype=torch.int32, device=dev)     # router padding bound (= real rows)
        self.out = torch.zeros(rows, h, dtype=bf, device=dev)        # MoE output (+ shared): the all-reduce operand
        self.upd = torch.zeros(rows, c, h, dtype=bf, device=dev)     # next residual
        self.nrm = torch.zeros(rows, h, dtype=bf, device=dev)        # next attention input
        self.npre = torch.zeros(rows, c, 1, dtype=f32, device=dev)   # next triplet
        self.npost = torch.zeros(rows, c, 1, dtype=f32, device=dev)
        self.ncomb = torch.zeros(rows, c, c, dtype=f32, device=dev)
        # lever FGB: the reduced attention output graph 1 ("m") reads (with upd / npre / npost / ncomb: the previous
        # FG layer's outputs = this layer's attention-post residual + triplet)
        self.ao = torch.zeros(rows, h, dtype=bf, device=dev) if _FGB["on"] else None
        self.nbytes = sum(t.numel() * t.element_size() for t in self.tensors())

    def tensors(self):
        return [t for t in (self.x, self.res, self.pre, self.post, self.comb, self.logits, self.ids, self.out, self.upd,
                            self.nrm, self.npre, self.npost, self.ncomb, self.ao) if t is not None]


# ---------------------------------------------------------------------------------------------------- attention post
def attn_post(fgp, hc, out, residual, *, stats_stream, next, world_size, precomputed=None):  # noqa: A002
    """run_attn_post for an FG layer: stock's own path (non-fused post: the triplet, then _post_fusion's mega branch)
    with the mega boundary's outputs written into FG's static FFN inputs [:M]."""
    mhc = _H["mhc"]
    m = fgp.rows
    if (isinstance(out, mhc.AttnOutput) or type(out) is not torch.Tensor or out.dim() != 2 or out.shape[0] != m
            or next is not fgp.L.local_b or not _BUF or stats_stream is not None):
        fgp.broken = True
        return mhc.run_attn_post(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                                 precomputed=precomputed)
    if fgp.v != "p":
        if fgp.capture:
            return _attn_post_stock(fgp, hc, out, residual, stats_stream, next, precomputed)
        if fgp.v == "m":
            st = _attn_post_defer(fgp, out, residual, precomputed)
            if st is not None:
                return st
            # graph 1 needs the deferred form: FG's eager post below (materialized), the FFN half then on stock
            fgp.broken = True
            _FGB["decline"] += 1
            if levers.first("fgb-decline"):
                levers.announce("fgb-decline", f"lever FGB: an 'm' attention post declined (layer "
                                f"{getattr(fgp.L.layer, 'layer_id', '?')}, {m} rows): FG's eager post, stock FFN half")
    try:
        coefficients = mhc._compute_triplet(hc, residual, stats_stream, precomputed)
        pre, post_mix, comb = coefficients
        B = _BUF[0]
        nh = next.hc
        from dsv41f import fg_layer

        upd, nrm, stats = fg_layer.mhc_mega_boundary_out(
            out, residual, pre, post_mix, comb, nh.fn, nh.scale, nh.base, nh.norm.weight, nh.cfg.rms_eps, nh.cfg.eps,
            nh.norm.variance_epsilon, nh.cfg.sinkhorn_iters, B.res[:m], B.x[:m], B.pre[:m], B.post[:m], B.comb[:m])
        return mhc.HcState(upd, pre, mhc.HcNormed(nrm), stats)
    except Exception as e:  # noqa: BLE001 - nothing launched or the launch failed: stock's post (same inputs)
        fgp.broken = True
        _local_fail("attn-post", e)
        return mhc.run_attn_post(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                                 precomputed=precomputed)


def _attn_post_stock(fgp, hc, out, residual, stats_stream, next, precomputed):  # noqa: A002
    """Lever FGB, a bucket's capture forward: stock's attention post for a plain tensor (run_attn_post's body: the
    triplet, then _post_fusion -- stock's mega boundary, fresh outputs), which this call returns; its inputs are kept
    for the capture check (cloned where they live in FG's static buffers, which the capture's warmups rewrite)."""
    mhc = _H["mhc"]
    coefficients = mhc._compute_triplet(hc, residual, stats_stream, precomputed)
    st = mhc._post_fusion(hc, out, residual, coefficients, next)
    try:
        fgp.stash = tuple(_own(x) for x in (out, residual) + tuple(coefficients))
        if len(fgp.stash) != 5:
            raise ValueError(f"triplet of {len(fgp.stash) - 2}")
    except Exception as e:  # noqa: BLE001 - no check possible: the capture forward fails this bucket (eager result)
        fgp.stash = None
        fgp.fwd.fail.setdefault(fgp.S, f"FGB stash: {type(e).__name__}: {str(e)[:120]}")
    return st


def _fg_storage(t) -> bool:
    """t lives in one of FG's static buffers."""
    p = t.untyped_storage().data_ptr()
    return any(b.untyped_storage().data_ptr() == p for b in _BUF[0].tensors())


def _own(t):
    if type(t) is not torch.Tensor:
        raise TypeError(type(t).__name__)
    return t.clone() if _fg_storage(t) else t


def _same_view(a, b) -> bool:
    return (a.data_ptr() == b.data_ptr() and a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and a.stride() == b.stride())


def _attn_post_defer(fgp, out, residual, precomputed):
    """Lever FGB, an "m" replay: the attention post's mega boundary runs inside graph 1. Here only its inputs are put
    where graph 1 reads them -- the reduced attention output copied into B.ao [:M]; the residual and triplet are the
    previous FG layer's static outputs (B.upd / B.npre / B.npost / B.ncomb [:M], untouched since its replay) or are
    copied there (bytes) -- and the state is returned as FG's static views, written by graph 1 first thing in the FFN
    half (fg.ffn materializes them with the eager kernel on every exit that does not replay graph 1). None: not this
    form (FG's eager post runs instead)."""
    mhc = _H["mhc"]
    B, L, m = _BUF[0], fgp.L, fgp.rows
    tri = precomputed
    if B.ao is None or out.dtype != torch.bfloat16 or tuple(out.shape) != (m, L.hidden) or not isinstance(tri, tuple) \
            or len(tri) != 3:
        return None
    pre, post_mix, comb = tri
    c, h = L.mult, L.hidden
    so = fgp.fwd.so
    try:
        if not (so is not None and so[2] == m and residual is so[0] and tri is so[1]):
            # not the previous FG replay's own outputs: check the form, copy what is not already in place
            f32, bf = torch.float32, torch.bfloat16
            for t, shape, dt in ((residual, (m, c, h), bf), (pre, (m, c), f32), (post_mix, (m, c), f32),
                                 (comb, (m, c, c), f32)):
                if type(t) is not torch.Tensor or tuple(t.shape) != shape or t.dtype != dt or t.device != B.x.device:
                    return None
            pairs = ((residual, B.upd[:m]), (pre, B.npre[:m, :, 0]), (post_mix, B.npost[:m, :, 0]), (comb, B.ncomb[:m]))
            todo = [(src, dst) for src, dst in pairs if not _same_view(src, dst)]
            if any(_fg_storage(src) for src, _dst in todo) or _fg_storage(out):
                return None      # another view of FG's buffers (never produced): the eager post
            for src, dst in todo:
                dst.copy_(src)
            _FGB["copy"] += len(todo)
            if todo and levers.first("fgb-copy"):
                levers.announce("fgb-copy", f"lever FGB: first attention post whose residual / triplet were not the "
                                f"previous FG replay's (layer {getattr(L.layer, 'layer_id', '?')}, {m} rows): "
                                f"{len(todo)} copied into place")
        B.ao[:m].copy_(out)
    except Exception as e:  # noqa: BLE001 - an unexpected error on this rank: FG off here (stock from now on)
        _local_fail("fgb-defer", e)
        return None
    fgp.deferred = True
    _FGB["defer"] += 1
    return mhc.HcState(B.res[:m], pre, mhc.HcNormed(B.x[:m]),
                       (B.pre[:m].squeeze(-1), B.post[:m].squeeze(-1), B.comb[:m]))


def _materialize(fgp) -> None:
    """Lever FGB: a deferred attention post that graph 1 did not run -- stock's mega boundary kernel at M rows on the
    same static inputs into the same static outputs (FG's eager attention post, byte for byte)."""
    if not fgp.deferred:
        return
    fgp.deferred = False
    from dsv41f import fg_layer

    B, nh, m = _BUF[0], fgp.L.local_b.hc, fgp.rows
    _FGB["mat"] += 1
    if levers.first("fgb-mat"):
        levers.announce("fgb-mat", f"lever FGB: a deferred attention post materialized eagerly (layer "
                        f"{getattr(fgp.L.layer, 'layer_id', '?')}, {m} rows: the FFN half did not replay graph 1)")
    fg_layer.mhc_mega_boundary_out(
        B.ao[:m], B.upd[:m], B.npre[:m, :, 0], B.npost[:m, :, 0], B.ncomb[:m], nh.fn, nh.scale, nh.base,
        nh.norm.weight, nh.cfg.rms_eps, nh.cfg.eps, nh.norm.variance_epsilon, nh.cfg.sinkhorn_iters, B.res[:m],
        B.x[:m], B.pre[:m], B.post[:m], B.comb[:m])


# ------------------------------------------------------------------------------------------------------------- FFN
def _static_state(state, m):
    """The FFN state attn_post built: (residual, precomputed) as FG's static views [:m], else None."""
    mhc = _H["mhc"]
    B = _BUF[0]
    inp = state.input
    res = state.streams
    st = state.stats
    if (type(inp) is not mhc.HcNormed or type(res) is not torch.Tensor or st is None or len(st) != 3
            or inp.rows.data_ptr() != B.x.data_ptr() or inp.rows.shape[0] != m or res.data_ptr() != B.res.data_ptr()
            or res.shape[0] != m or st[0].data_ptr() != B.pre.data_ptr() or st[1].data_ptr() != B.post.data_ptr()
            or st[2].data_ptr() != B.comb.data_ptr() or st[0].shape[0] != m):
        return None
    return res, st


def ffn(fgp, layer, state, forward_batch, input_ids, input_ids_global):
    """run_ffn_hc for an FG layer; None = stock's run_ffn_hc on the untouched state."""
    if fgp.broken or not _BUF:
        if fgp.capture:
            fgp.fwd.fail.setdefault(fgp.S, "the attention post did not write FG's buffers")
        _materialize(fgp)
        return None
    got = _static_state(state, fgp.rows)
    ids_ok = (isinstance(input_ids_global, torch.Tensor) and input_ids_global.dim() == 1
              and input_ids_global.shape[0] == fgp.rows and input_ids_global.dtype == _BUF[0].ids.dtype)
    if fgp.capture:
        if fgp.v != "p":   # lever FGB: stock's attention post ran (fresh state), its inputs in fgp.stash
            return _ffn_capture_b(fgp, state, input_ids, input_ids_global if ids_ok else None)
        return _ffn_capture(fgp, state, got if ids_ok else None, input_ids, input_ids_global)
    if got is None or not ids_ok:
        _ST["eager"] += 1
        _materialize(fgp)   # lever FGB: a deferred post is written before stock's FFN half reads the state
        return None
    return _ffn_replay(fgp, state, got, input_ids_global)


def _prep(fgp, x, ids):
    """The eager pieces before graph 1: the gate GEMM into the static logits, the router's ids and padding bound."""
    B, L, m, fwd = _BUF[0], fgp.L, fgp.rows, fgp.fwd
    if x is not None:      # lever FGB: None when graph 1 holds the gate GEMM
        torch.mm(x, L.wt, out_dtype=torch.float32, out=B.logits[:m])
    if fwd.nt_m != m:
        B.nt.fill_(m)
        fwd.nt_m = m
    if fwd.ids_src is not ids or fwd.ids_m != m:
        B.ids[:m].copy_(ids)
        fwd.ids_src, fwd.ids_m = ids, m


def _all_reduce(y):
    """stock: post_experts_all_reduce inside _run_moe_ffn_dp_sync's scoped(mlp_reduce_scatter=False)."""
    with _H["get_forward"]().scoped(mlp_reduce_scatter=False):
        return _H["moe_fwd"].post_experts_all_reduce(y)


def _ffn_replay(fgp, state, got, ids):
    mhc = _H["mhc"]
    B, L, m, s = _BUF[0], fgp.L, fgp.rows, fgp.S
    graphs = L.graphs[s]
    v = fgp.v
    # lever FGB: graph 1 = graphs[0] as captured (L.var[s]); an A/B "off" forward of an FGB bucket: FG's plain graph 1
    g1 = graphs[2] if (v == "p" and L.var.get(s, "p") != "p") else graphs[0]
    g2 = graphs[1]
    _res, stats = got
    try:
        _prep(fgp, B.x[:m] if v == "p" else None, ids)
        g1.replay()
    except Exception as e:  # noqa: BLE001 - before the all-reduce: stock's FFN half instead (its own all-reduce)
        _local_fail("replay-1", e)
        _materialize(fgp)
        return None
    fgp.deferred = False
    if v != "p":
        _FGB["replay_m" if v == "m" else "replay_g"] += 1
        if levers.first(f"fgb-replay-{v}"):
            levers.announce(f"fgb-replay-{v}", f"lever FGB: first graph-1 '{v}' replay at {m} rows (bucket {s}, layer "
                            f"{getattr(L.layer, 'layer_id', '?')}; deferred posts {_FGB['defer']}, copies "
                            f"{_FGB['copy']})")
    y = B.out[:m]
    r = _all_reduce(y)
    if r.data_ptr() != y.data_ptr():
        y.copy_(r)
    try:
        g2.replay()
    except Exception as e:  # noqa: BLE001 - after the all-reduce: stock's mega post on the reduced rows
        _local_fail("replay-2", e)
        state.release()
        upd, nrm, nst = _H["mega"](y, B.res[:m], stats[0], stats[1], stats[2], L.next_hc.fn, L.next_hc.scale,
                                   L.next_hc.base, L.next_hc.norm.weight, L.next_hc.cfg.rms_eps, L.next_hc.cfg.eps,
                                   L.next_hc.norm.variance_epsilon, L.next_hc.cfg.sinkhorn_iters)
        return mhc.HcState(upd, stats[0], mhc.HcNormed(nrm), nst)
    state.release()
    _ST["replay"] += 1
    if levers.first("fg-replay"):
        levers.announce("fg-replay", f"lever FG: first graph replay at {m} rows (bucket {s}, layer "
                        f"{getattr(L.layer, 'layer_id', '?')})")
    upd = B.upd[:m]
    tri = (B.npre[:m].squeeze(-1), B.npost[:m].squeeze(-1), B.ncomb[:m])
    fgp.fwd.so = (upd, tri, m)   # lever FGB: the next layer's attention post may find these in place (identity)
    return mhc.HcState(upd, stats[0], mhc.HcNormed(B.nrm[:m]), tri)


# ----------------------------------------------------------------------------------------------- capture + first check
def _seg1(L, s):
    """Graph 1 at s rows: the MoE replica's forward_normal from the router on (fg_moe edits), lever FMT's pin for an
    s-row extend call, the routed output written into the static MoE output."""
    B = _BUF[0]
    fmt = _H["fmt"]
    band = _TABLE[0].key[s][0]
    tok = fmt._PIN.set((levers.FMT_TACTICS[band][2], s))
    try:
        with _H["buf_ctx"](B.out[:s]):
            y = L.moe.dsv4f_fg_seg1(B.logits[:s], B.x[:s], None, None, input_ids_global=B.ids[:s],
                                    skip_shared_experts=False, num_token_non_padded=B.nt, use_vision_topk=L.vision,
                                    return_moe_output=False)
    finally:
        fmt._PIN.reset(tok)
    if type(y) is not torch.Tensor or y.data_ptr() != B.out.data_ptr() or tuple(y.shape) != (s, L.hidden):
        raise RuntimeError("the routed MoE output is not FG's static buffer")


def _seg2(L, s):
    """Graph 2 at s rows: stock's mega boundary on the static buffers, into the static outputs."""
    from dsv41f import fg_layer

    B, nh = _BUF[0], L.next_hc
    fg_layer.mhc_mega_boundary_out(
        B.out[:s], B.res[:s], B.pre[:s, :, 0], B.post[:s, :, 0], B.comb[:s], nh.fn, nh.scale, nh.base, nh.norm.weight,
        nh.cfg.rms_eps, nh.cfg.eps, nh.norm.variance_epsilon, nh.cfg.sinkhorn_iters, B.upd[:s], B.nrm[:s], B.npre[:s],
        B.npost[:s], B.ncomb[:s])


def _seg0(L, s):
    """Lever FGB, the head of an "m" graph 1 at s rows: the attention post's mega boundary (stock's kernel, FG's
    lifted output form) on FGB's static inputs -- the reduced attention output B.ao, the previous FG layer's residual
    B.upd and triplet B.npre / B.npost / B.ncomb -- into FG's static FFN inputs (B.res / B.x / B.pre / B.post / B.comb),
    with this layer's FFN hyper-connection (local boundary) as stock's _post_fusion takes it."""
    from dsv41f import fg_layer

    B, nh = _BUF[0], L.local_b.hc
    fg_layer.mhc_mega_boundary_out(
        B.ao[:s], B.upd[:s], B.npre[:s, :, 0], B.npost[:s, :, 0], B.ncomb[:s], nh.fn, nh.scale, nh.base, nh.norm.weight,
        nh.cfg.rms_eps, nh.cfg.eps, nh.norm.variance_epsilon, nh.cfg.sinkhorn_iters, B.res[:s], B.x[:s], B.pre[:s],
        B.post[:s], B.comb[:s])


def _segg(L, s):
    """Lever FGB: the router gate GEMM at s rows in graph 1 (stock's cuBLAS torch.mm, fp32 out, into the static
    logits; admitted per bucket by the row-consistency sweep)."""
    B = _BUF[0]
    torch.mm(B.x[:s], L.wt, out_dtype=torch.float32, out=B.logits[:s])


def _seg1v(L, s, v):
    """Graph 1 of variant v ("m": mega boundary + gate + segment, "g": gate + segment, "p": FG's segment)."""
    if v == "m":
        _seg0(L, s)
    if v != "p":
        _segg(L, s)
    _seg1(L, s)


def _gate_sweep(L, t, s) -> bool:
    """Lever FGB: cuBLAS picks the gate GEMM's kernel per row count (split-K at some small row counts changes the
    summation order). On deterministic random rows ([s, hidden] bf16, a fresh 512-byte aligned buffer: the layout of
    the static input) and the layer's real gate weight: stock's own call (torch.mm(x, W.t(), out_dtype=fp32)) at every
    row count m of bucket s must give the first m rows of the graph's form (out= into an [s, experts] fp32 buffer) at
    s, byte for byte. Once per bucket and rank (every FG layer's gate has the same shape / layout); collective-free."""
    hit = _GSWEEP.get(s)
    if hit is not None:
        return hit
    if not getattr(levers, "FGB_GATE_SWEEP", True):
        _GSWEEP[s] = True
        return True
    t0 = time.perf_counter()
    lo = next((a for a, e in t.buckets if e == s), s)
    ok = True
    try:
        with torch.no_grad():
            dev = L.gate_w.device
            g = torch.Generator(device=dev)
            g.manual_seed(int(getattr(levers, "FGB_SEED", 20261015)) + s)
            x = torch.empty((s, L.hidden), dtype=torch.bfloat16, device=dev).normal_(generator=g)
            ref = torch.empty((s, int(L.gate_w.shape[0])), dtype=torch.float32, device=dev)
            torch.mm(x, L.wt, out_dtype=torch.float32, out=ref)
            bad = torch.zeros((), dtype=torch.bool, device=dev)
            for mm in range(lo, s + 1):
                d = _neq(torch.mm(x[:mm], L.gate_w.t(), out_dtype=torch.float32), ref[:mm])
                if d is None:
                    raise RuntimeError("gate output geometry")
                torch.logical_or(bad, d, out=bad)
            ok = not bool(bad.item())
    except Exception as e:  # noqa: BLE001
        ok = False
        levers.announce(f"fgb-sweep-err{s}", f"lever FGB: gate sweep at bucket {s} failed ({type(e).__name__}: "
                        f"{str(e)[:160]})")
    _GSWEEP[s] = ok
    _FGB["sweep"] += 1
    _FGB["sweep_ms"] += (time.perf_counter() - t0) * 1e3
    levers.announce(f"fgb-sweep{s}", f"lever FGB: router gate cuBLAS torch.mm at every row count {lo}..{s} == the "
                    f"first rows at {s}: {'yes' if ok else 'NO (gate stays eager in this bucket)'} "
                    f"({_FGB['sweep_ms']:.0f} ms of sweeps so far)")
    return ok


def _epoch():
    """The levers' learn / verify state: a graph is captured only once a warmup leaves it unchanged."""
    out = []
    try:
        from dsv41f import mo

        out += [len(mo._TAC), len(mo._OK), len(mo._QOK), len(mo._QPLAN), mo._STATE.get("learn", 0)]
    except Exception:  # noqa: BLE001
        out.append(None)
    try:
        from dsv41f import rt

        out += [len(rt._PLANS), len(rt._KOK), len(rt._GOK), rt._ST.get("r_learn", 0)]
    except Exception:  # noqa: BLE001
        out.append(None)
    try:
        from dsv41f import fl

        out += [fl._STATE.get("learned", 0), fl._STATE.get("refused", 0), sum(len(v) for v in fl._PLANS.values())]
    except Exception:  # noqa: BLE001
        out.append(None)
    fmt = _H["fmt"]
    out += [len(fmt._BAND_OK), len(fmt._VALID)]
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
    """The current eligible forward's number (-1 outside one)."""
    fwd = _ON.get()
    return fwd.seq if fwd is not None else -1


def _warm(L, s, v="p"):
    """Eager runs of both segments at s rows on the capture stream until the levers' learn / verify state stops changing
    (lever MO / RT learn the s-row keys, then verify; deep_gemm's per-stream split barriers get allocated). Once per
    (bucket, capture forward): a later capture forward of the bucket (a retry) warms up again, since the levers' key
    caches may have dropped the s-row keys in between (lever RT clears its plans at 512 keys, MO at 4096) and a key
    learned during a capture would void it. Lever FGB: graph 1's launches of variant v (_wvar: the bucket's superset,
    so the gate GEMM's first cuBLAS call and the attention mega's first launch on the capture stream come before any
    capture)."""
    n = 0
    for _ in range(int(levers.FG_MAX_WARMUPS)):
        e0 = _epoch()
        _seg1v(L, s, v)
        _seg2(L, s)
        n += 1
        if _epoch() == e0:
            _WARM[s] = (_seq(), _VRANK[v])
            return n
    raise RuntimeError(f"the levers' state did not settle in {n} warmups at {s} rows")


def _warmed(s, v) -> bool:
    """Bucket s was warmed up in this forward, with graph-1 launches covering variant v."""
    w = _WARM.get(s)
    return isinstance(w, tuple) and len(w) == 2 and w[0] == _seq() and w[1] >= _VRANK[v]


def _reserve(L, pool, fgb_flow=False):
    """The pool's footprint at the largest bucket, once: both segments captured at t.max_s rows into the shared pool
    (never replayed, kept alive so the pool keeps its segments). Every later capture -- smaller or equal -- then carves
    its intermediates out of these segments instead of adding new ones (buckets arrive in any order; ascending order
    grew the pool by each bucket's footprint: 7.6 GiB for 22 buckets in fg_test). Tried once: a failure leaves no
    reserve (and no half-captured graph holding the pool). Lever FGB: in a capture of FGB's flow graph 1 is captured in
    the largest bucket's superset variant; in FG's plain flow FG's own segment only (the plain capture forward keeps
    the eager attention post's results in the static FFN inputs, which FGB's mega / gate warmups would rewrite)."""
    smax = _TABLE[0].max_s
    _RES["reserve"] = ()
    rv = _wvar(smax) if fgb_flow else "p"
    n = 0 if _warmed(smax, rv) else _warm(L, smax, rv)
    r1 = _graph(lambda: _seg1v(L, smax, rv), pool)
    try:
        r2 = _graph(lambda: _seg2(L, smax), pool)
    except BaseException:
        r1.reset()
        raise
    _RES["reserve"] = (r1, r2)
    return n


def _live_graphs() -> int:
    n = len(_RES.get("reserve") or ())
    n += _FGP[0].live() if _FGP else 0   # lever FGP: pre-captured graphs hold the pool too
    for _layer_obj, L in _LAYERS.values():
        if L:
            n += sum(len(gs) for gs in L.graphs.values())   # 2 a (layer, bucket); 3 with lever FGB's A/B plain graph 1
    return n


def _pool_check():
    """After a failed capture or a dropped bucket: when no live graph holds the shared pool any more, the next capture
    takes a fresh pool (PyTorch refuses to share a pool whose use count dropped to zero)."""
    if "pool" in _RES and _live_graphs() == 0:
        del _RES["pool"]


def _capture(L, s):
    """Warm up (the bucket's first capture in this forward) and capture both graphs of (layer, bucket s) on FG's
    capture stream. Lever FGB: graph 1 in this layer's variant (_variant: recorded in L.var[s]); with levers.FGB_AB
    FG's plain graph 1 too (the A/B "off" forwards replay it)."""
    v = _variant(L, s)
    ab = v != "p" and bool(getattr(levers, "FGB_AB", 0))
    cap, pool = _stream(), _pool()
    cur = torch.cuda.current_stream()
    cap.wait_stream(cur)
    fwd = _ON.get()
    if fwd is not None:
        fwd.so = None             # lever FGB: the warmups rewrite FG's static outputs
    t0 = time.perf_counter()
    nwarm = 0
    wv = _wvar(s) if v != "p" else "p"
    with torch.cuda.stream(cap):
        if "reserve" not in _RES and levers.FG_POOL_RESERVE:
            nwarm += _reserve(L, pool, v != "p")
        if not _warmed(s, wv):
            nwarm += _warm(L, s, wv)
        e0 = _epoch()
        gs = [_graph(lambda: _seg1v(L, s, v), pool)]
        try:
            gs.append(_graph(lambda: _seg2(L, s), pool))
            if ab:
                gs.append(_graph(lambda: _seg1(L, s), pool))
        except BaseException:
            for g in gs:
                g.reset()
            raise
        if _epoch() != e0:
            for g in gs:
                g.reset()
            raise RuntimeError("a lever learned / verified during the capture")
    cur.wait_stream(cap)
    L.var[s] = v
    _ST["warm"] += nwarm
    _ST["cap_ms"] += (time.perf_counter() - t0) * 1e3
    return tuple(gs), nwarm


def _neq(a, b):
    """0-d GPU bool: a's bytes differ from b's (same shape / dtype / strides asserted on the host)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return None
    v = torch.int16 if a.element_size() == 2 else torch.int32 if a.element_size() == 4 else torch.int64
    return (a.contiguous().view(v) != b.contiguous().view(v)).any()


def _ffn_capture(fgp, state, got, input_ids, ids):
    """The bucket's capture forward, this layer: capture, replay once, and return the eager path's result. The graph
    outputs are compared with it on the GPU (no host sync here); _resolve reads the flags once at the end of the
    forward and agrees the bucket's outcome over the ranks before any graph is used. Nothing here is a collective but
    the all-reduce the eager path makes (exactly once, on the eager rows), so a rank-local failure only marks the bucket."""
    mhc = _H["mhc"]
    B, L, m, s, fwd = _BUF[0], fgp.L, fgp.rows, fgp.S, fgp.fwd
    if got is None:
        fwd.fail.setdefault(s, "state / input ids not FG's static form")
        return None
    pre = None
    try:
        _prep(fgp, B.x[:m], ids)
        pre = _FGP[0].take(L, s) if _FGP else None   # lever FGP: graphs pre-captured for (layer, bucket)
        graphs, nwarm = pre if pre is not None else _capture(L, s)
    except Exception as e:  # noqa: BLE001 - nothing captured: stock's FFN half (its own all-reduce)
        torch.cuda.current_stream().wait_stream(_stream())
        fwd.fail.setdefault(s, f"capture: {type(e).__name__}: {str(e)[:200]}")
        return None
    _ST["capture"] += 1
    if pre is None and levers.first(f"fg-cap{s}"):
        levers.announce(f"fg-cap{s}", f"lever FG: bucket {s} first captured at {m} rows (layer "
                        f"{getattr(L.layer, 'layer_id', '?')}, {nwarm} warmups, {_ST['cap_ms']:.0f} ms capture total, "
                        f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB)")
    res, stats = got
    x = B.x[:m]
    g1, g2 = graphs
    bad = fwd.bad.get(s)
    if bad is None:
        bad = fwd.bad[s] = torch.zeros((), dtype=torch.bool, device=x.device)
    ok = True

    def acc(a, b):
        nonlocal ok
        d = _neq(a, b)
        if d is None:
            ok = False
        else:
            torch.logical_or(bad, d, out=bad)

    try:
        g1.replay()
    except Exception as e:  # noqa: BLE001
        ok = False
        fwd.fail.setdefault(s, f"replay 1: {type(e).__name__}: {str(e)[:120]}")
    # the eager path at m rows (what this call returns): stock's gate, the MoE replica's forward_normal (lever FMT's pin
    # for an m-row extend call, stock's padding count), then the all-reduce and stock's mega post
    fmt = _H["fmt"]
    band = _TABLE[0].key[s][0]
    logits = L.moe.dsv4f_fg_gate(x, None)
    tok = fmt._PIN.set((levers.FMT_TACTICS[band][2], m))
    try:
        y_ref = L.moe.dsv4f_fg_seg1(logits, x, None, input_ids, input_ids_global=ids, skip_shared_experts=False,
                                    num_token_non_padded=fwd.fb.moe_num_token_non_padded(),
                                    use_vision_topk=L.vision, return_moe_output=False)
    finally:
        fmt._PIN.reset(tok)
    if ok:
        acc(B.logits[:m], logits)
        acc(B.out[:m], y_ref)
    y = _all_reduce(y_ref)
    if ok:
        try:
            B.out[:m].copy_(y)
            g2.replay()
        except Exception as e:  # noqa: BLE001
            ok = False
            fwd.fail.setdefault(s, f"replay 2: {type(e).__name__}: {str(e)[:120]}")
    state.release()
    nh = L.next_hc
    upd, nrm, nst = _H["mega"](y, res, stats[0], stats[1], stats[2], nh.fn, nh.scale, nh.base, nh.norm.weight,
                               nh.cfg.rms_eps, nh.cfg.eps, nh.norm.variance_epsilon, nh.cfg.sinkhorn_iters)
    if ok:
        acc(B.upd[:m], upd)
        acc(B.nrm[:m], nrm)
        acc(B.npre[:m].squeeze(-1), nst[0])
        acc(B.npost[:m].squeeze(-1), nst[1])
        acc(B.ncomb[:m], nst[2])
    if ok:
        fwd.pending.setdefault(s, []).append((L, graphs))
        fwd.rows.setdefault(s, m)
    else:
        fwd.fail.setdefault(s, "a graph output had another geometry")
        for g in graphs:
            g.reset()
    return mhc.HcState(upd, stats[0], mhc.HcNormed(nrm), nst)


def _ffn_capture_b(fgp, state, input_ids, ids):
    """Lever FGB, the bucket's capture forward, this layer (graph 1 "m": attention mega + gate + segment, or "g": gate
    + segment): capture (or take FGP's), load the eager call's inputs into the static inputs graph 1 reads, replay,
    and compare every output with the eager path at M rows on the GPU (_resolve reads the flag): the mega's five
    outputs vs stock's attention post (which ran in attn_post and is what this call continues from), the gate logits
    vs stock's gate at M rows and the graph's gate at S rows vs an eager torch.mm at S on its own input, the MoE output,
    graph 2's outputs vs stock's post; with levers.FGB_AB FG's plain graph 1 as well. Returns the eager result; None
    (only before the all-reduce): stock's FFN half on the eager state."""
    mhc = _H["mhc"]
    B, L, m, s, fwd, v = _BUF[0], fgp.L, fgp.rows, fgp.S, fgp.fwd, fgp.v
    st = fgp.stash
    inp, res1, stats1 = state.input, state.streams, state.stats
    if (st is None or ids is None or type(inp) is not mhc.HcNormed or type(res1) is not torch.Tensor or stats1 is None
            or len(stats1) != 3 or (v == "m" and B.ao is None)):
        fwd.fail.setdefault(s, "FGB: attention-post state / inputs / ids not in the expected form")
        return None
    x1 = inp.rows
    pre = None
    try:
        _prep(fgp, None, ids)
        pre = _FGP[0].take(L, s) if _FGP else None   # lever FGP: graphs pre-captured for (layer, bucket)
        graphs, nwarm = pre if pre is not None else _capture(L, s)
    except Exception as e:  # noqa: BLE001 - nothing captured: stock's FFN half (its own all-reduce)
        torch.cuda.current_stream().wait_stream(_stream())
        fwd.fail.setdefault(s, f"capture: {type(e).__name__}: {str(e)[:200]}")
        return None
    _ST["capture"] += 1
    if pre is None and levers.first(f"fgb-cap{s}"):
        levers.announce(f"fgb-cap{s}", f"lever FG: bucket {s} first captured at {m} rows (lever FGB graph 1 '{v}', "
                        f"layer {getattr(L.layer, 'layer_id', '?')}, {nwarm} warmups, {_ST['cap_ms']:.0f} ms capture "
                        f"total, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB)")
    want = 3 if getattr(levers, "FGB_AB", 0) else 2
    if L.var.get(s) != v or len(graphs) != want:
        fwd.fail.setdefault(s, f"FGB: graphs of variant {L.var.get(s)} x{len(graphs)} for a '{v}' call")
        for g in graphs:
            g.reset()
        return None
    bad = fwd.bad.get(s)
    if bad is None:
        bad = fwd.bad[s] = torch.zeros((), dtype=torch.bool, device=x1.device)
    ok = True

    def acc(a, b):
        nonlocal ok
        d = _neq(a, b)
        if d is None:
            ok = False
        else:
            torch.logical_or(bad, d, out=bad)

    try:
        if v == "m":       # the attention post's inputs (graph 1 runs the mega boundary itself)
            out_a, res_a, pre_a, post_a, comb_a = st
            B.ao[:m].copy_(out_a)
            B.upd[:m].copy_(res_a)
            B.npre[:m, :, 0].copy_(pre_a)
            B.npost[:m, :, 0].copy_(post_a)
            B.ncomb[:m].copy_(comb_a)
        else:              # "g": FG's static FFN inputs as FG's eager attention post writes them
            B.x[:m].copy_(x1)
            B.res[:m].copy_(res1)
            B.pre[:m, :, 0].copy_(stats1[0])
            B.post[:m, :, 0].copy_(stats1[1])
            B.comb[:m].copy_(stats1[2])
        graphs[0].replay()
    except Exception as e:  # noqa: BLE001
        ok = False
        fwd.fail.setdefault(s, f"FGB replay 1: {type(e).__name__}: {str(e)[:120]}")
    # the eager path at m rows (what this call returns): stock's gate, the MoE replica's forward_normal (lever FMT's pin
    # for an m-row extend call, stock's padding count), then the all-reduce and stock's mega post
    fmt = _H["fmt"]
    band = _TABLE[0].key[s][0]
    logits = L.moe.dsv4f_fg_gate(x1, None)
    tok = fmt._PIN.set((levers.FMT_TACTICS[band][2], m))
    try:
        y_ref = L.moe.dsv4f_fg_seg1(logits, x1, None, input_ids, input_ids_global=ids, skip_shared_experts=False,
                                    num_token_non_padded=fwd.fb.moe_num_token_non_padded(),
                                    use_vision_topk=L.vision, return_moe_output=False)
    finally:
        fmt._PIN.reset(tok)
    if ok:
        if v == "m":
            acc(B.res[:m], res1)
            acc(B.x[:m], x1)
            acc(B.pre[:m].squeeze(-1), stats1[0])
            acc(B.post[:m].squeeze(-1), stats1[1])
            acc(B.comb[:m], stats1[2])
        acc(B.logits[:m], logits)
        acc(B.out[:m], y_ref)
        acc(B.logits[:s], torch.mm(B.x[:s], L.wt, out_dtype=torch.float32))   # graph's gate == eager, every row at S
        if len(graphs) > 2:   # FGB_AB: FG's plain graph 1 as an "off" forward replays it (eager gate in _prep's form)
            try:
                B.x[:m].copy_(x1)
                torch.mm(B.x[:m], L.wt, out_dtype=torch.float32, out=B.logits[:m])
                acc(B.logits[:m], logits)
                graphs[2].replay()
                acc(B.out[:m], y_ref)
            except Exception as e:  # noqa: BLE001
                ok = False
                fwd.fail.setdefault(s, f"FGB plain graph 1: {type(e).__name__}: {str(e)[:120]}")
    y = _all_reduce(y_ref)
    if ok:
        try:
            B.out[:m].copy_(y)
            graphs[1].replay()
        except Exception as e:  # noqa: BLE001
            ok = False
            fwd.fail.setdefault(s, f"replay 2: {type(e).__name__}: {str(e)[:120]}")
    state.release()
    nh = L.next_hc
    upd, nrm, nst = _H["mega"](y, res1, stats1[0], stats1[1], stats1[2], nh.fn, nh.scale, nh.base, nh.norm.weight,
                               nh.cfg.rms_eps, nh.cfg.eps, nh.norm.variance_epsilon, nh.cfg.sinkhorn_iters)
    if ok:
        acc(B.upd[:m], upd)
        acc(B.nrm[:m], nrm)
        acc(B.npre[:m].squeeze(-1), nst[0])
        acc(B.npost[:m].squeeze(-1), nst[1])
        acc(B.ncomb[:m], nst[2])
    if ok:
        fwd.pending.setdefault(s, []).append((L, graphs))
        fwd.rows.setdefault(s, m)
    else:
        fwd.fail.setdefault(s, "a graph output had another geometry")
        for g in graphs:
            g.reset()
    return mhc.HcState(upd, stats1[0], mhc.HcNormed(nrm), nst)


def _resolve(fwd):
    """The end of a capture forward (both ranks reach it: the bucket's "capturing" state is agreed): read this rank's
    flags once, agree (gloo MIN) on [every capture / replay / byte check passed, every rank captured some layer, no
    rank captured any], then on every rank: activate the pending graphs (and the bucket), or turn the bucket off for
    good (a failure, or some ranks captured and others not -- retrying that would recapture on the capturing ranks in
    every forward of these rows), or keep it capturing (no rank captured: retry at its next forward, at most
    FG_MAX_CAPTURE_FORWARDS times)."""
    for s in sorted(fwd.capturing):
        pend = fwd.pending.pop(s, [])
        why = fwd.fail.get(s, "")
        ok = not why
        if ok and pend and levers.FG_CHECK:
            try:
                if bool(fwd.bad[s].item()):
                    ok, why = False, "graph bytes != eager bytes"
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"check: {type(e).__name__}"
        a = _agree([int(ok), int(bool(pend)), int(not pend)])
        _ST["check"] += len(pend)
        if not a[0] or not (a[1] or a[2]):
            for _L, graphs in pend:
                for g in graphs:
                    g.reset()
            _kill_bucket(s, f"capture forward at {fwd.rows.get(s, '?')} rows: "
                            f"{why or ('failed on another rank' if not a[0] else 'captured on some ranks only')}")
        elif a[1]:
            for L, graphs in pend:
                L.graphs[s] = graphs
            _BSTATE[s] = "active"
            _TOK[s] = fwd.tok
            levers.announce(f"fg-chk{s}", f"lever FG: bucket {s} on: {len(pend)} layers' graphs captured at "
                            f"{fwd.rows.get(s, '?')} rows, first replay == eager bit for bit on every rank")
        else:
            _RETRY[s] = _RETRY.get(s, 0) + 1
            if _RETRY[s] >= int(getattr(levers, "FG_MAX_CAPTURE_FORWARDS", 8)):
                _kill_bucket(s, f"{_RETRY[s]} capture forwards without a captured layer")
                continue
            for _L, graphs in pend:
                for g in graphs:
                    g.reset()
            _pool_check()
    if _FGP:
        _FGP[0].resolved(fwd)   # lever FGP: graphs of buckets no longer capturing that no layer took


# ------------------------------------------------------------------------------------------------------- bookkeeping
def _agree(values):
    """MIN over the TP CPU group (gloo), elementwise (route_e5._agree's form)."""
    t = torch.tensor([int(v) for v in values], dtype=torch.int64)
    g = _H["tp_group"]()
    if g.world_size > 1:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=g.cpu_group)
    return t.tolist()


def _agree_bool(ok) -> bool:
    return bool(_agree([int(bool(ok))])[0])


def _kill_bucket(s, why):
    _BSTATE[s] = "dead"
    for _layer_obj, L in list(_LAYERS.values()):
        if L:
            for g in L.graphs.pop(s, None) or ():
                g.reset()
    _pool_check()
    _ST["fail"] += 1
    levers.announce(f"fg-dead{s}", f"lever FG: bucket {s} OFF for good ({why[:300]}): eager for those rows")


def _local_fail(where, e):
    """An unexpected error on this rank: FG's replays stop here for good; plan() / ffn() still reach (and answer False
    to) every agreement the other rank reaches, so the ranks never wait on each other."""
    _ST["dead"] = True
    _ST["fail"] += 1
    levers.announce(f"fg-err-{where}", f"lever FG: {where} failed ({type(e).__name__}: {str(e)[:200]}): replays off "
                    f"for good on this rank (stock path)")


def stats() -> dict:
    out = {k: v for k, v in _ST.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if _FGB["on"]:
        out.update({f"fgb_{k}": v for k, v in _FGB.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return out


def summary() -> str:
    live = sorted(s for s, v in _BSTATE.items() if v == "active")
    dead = sorted(s for s, v in _BSTATE.items() if v == "dead")
    return (f"buckets active {live}, dead {dead}; replays {_ST['replay']}, captures {_ST['capture']}, checks "
            f"{_ST['check']}, warmups {_ST['warm']}, capture {_ST['cap_ms']:.0f} ms, eager fallbacks {_ST['eager']}, "
            f"failures {_ST['fail']}{_fgb_summary()}")


def _fgb_summary() -> str:
    if not _FGB["on"]:
        return ""
    gon = sorted(s for s, v in _GATE.items() if v)
    var = {}
    for _layer_obj, L in _LAYERS.values():
        if L:
            for s, v in L.var.items():
                var[v] = var.get(v, 0) + 1
    return (f"; lever FGB: gate in graph 1 at buckets {gon}, graph-1 variants {var}, replays m {_FGB['replay_m']} / "
            f"g {_FGB['replay_g']} / A/B off {_FGB['replay_off']}, deferred posts {_FGB['defer']} (copies "
            f"{_FGB['copy']}, materialized {_FGB['mat']}, declined {_FGB['decline']}), gate sweeps {_FGB['sweep']} in "
            f"{_FGB['sweep_ms']:.0f} ms")
