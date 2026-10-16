"""Switches of the fork-line `model` node (dsv41f, cacheon/dsv41-tp2-upstream fork). Every lever is behind its own
switch; with all of them False the node runs the fork's stock code on its replicas (bundle f0)."""

# lever EG (eg.py, CUDA unit eg_gather): the Engram layers' per-rank host-table gather through eg_gather (stock
# engram_gather's bits, one warp a row), armed per layer by a byte-for-byte self-test on the served table.
ENGRAM_FAST_GATHER = True
EG_LAUNCH = ((256, 64, 1), (4096, 296, 1), (1 << 62, 592, 2))   # (max lookups, grid, unroll), B300 lev_EG micro
# EG early: every armed layer's gather forked on a side stream right after hashing (eager calls and whole-forward
# decode / verify graph captures; never inside a breakable prefill graph, whose segments cannot span a fork / join),
# waited for right before that layer's all-reduce.
EG_EARLY = True
EG_SORT = True                   # gather lookups in ascending table-row order from EG_SORT_MIN lookups (TLB locality)
EG_SORT_MIN = 16384
EG_EARLY_MAX_ROWS = 16384        # the fork's prefill chunk
EG_EARLY_LAUNCH = ((4096, 64, 1), (1 << 62, 16, 4))   # side-stream bands: few CTAs at prefill sizes
# lever EGQ (eg.py "LEVER EGQ", gen_egq.py): decode / verify forwards of 1..EGQ_MAX_ROWS rows fork only the FIRST
# armed Engram table's gather early, at EGQ_LAUNCH ((max lookups, grid, unroll): few CTAs = few SMs held beside layer
# 0's main-stream kernels), the later tables' just before layer EGQ_DEFER_LAYER at EGQ_DEFER_LAUNCH (1-2 CTAs; not
# beside a 148-CTA persistent kernel), each joined right before its own layer's all-reduce; same kernel, ids and bytes
EGQ = True
EGQ_MAX_ROWS = 384
EGQ_LAUNCH = ((4608, 4, 4), (4611686018427387904, 8, 4))
EGQ_DEFER_LAYER = 3
EGQ_DEFER_LAUNCH = ((4611686018427387904, 2, 4),)
# EGQ aux: the DSPARK draft's per-target-layer means (stock's aux.mean(dim=1)) on a side stream at 1..EGQ_MAX_ROWS
# rows, joined at the end of the layer loop (eager and whole-forward captures; inline inside a breakable prefill graph
# and above EGQ_MAX_ROWS, where lever FG's static next-residual buffer is overwritten in place by the same layer)
EGQ_AUX = True

# lever FMT (fmt.py): extend MoE calls of rows in [lo, hi] run the routed NVFP4 expert op at the measured tactic
# [tile, config] (fork/tests/fork_mt.py on the fork image; every tactic bit-identical, first use per band bit-checked)
FMT_TACTICS = ((256, 639, [32, 218]), (640, 1023, [64, 58]), (1024, 3071, [32, 218]), (3072, 10239, [128, 5]), (10240, 13311, [128, 21]), (13312, 16384, [256, 0]))
# decode / verify MoE calls of rows in [lo, hi]: the measured tactic instead of the fork's pin (fork_mt.py fmt3 on SV-pruned
# routing, every tactic bit-identical; first eager use per band bit-checked)
FMT_DECODE_TACTICS = ()

# lever SV (sv.py): the fork's E5 prune with budget SV_BUDGET at every pruned decode / verify call and a share vote
# (an expert picked by <= SV_MAX_COUNT rows is dropped when every holder drops it) from SV_MIN_ROWS rows; PRECISION
SV = False   # EXACT: stock's own E5 pruning (budget 0.4, max_count 1); SV pruning failed the live audit (f23 10-15)
SV_BUDGET = 0.9            # stock E5: 0.4
SV_MIN_ROWS = 13            # share vote from this many rows (stock: max_count 1 at every size)
SV_MAX_COUNT = 4

# lever NQ (nq.py): the routed NVFP4 input quantize of CUDA-graph decode / verify MoE calls on a side stream at the MoE
# forward's start (stock's quantize on the same tensor; bitwise)
NQ = True

# lever RE5 (route_e5.py, CUDA unit route_e5): decode / verify top-k + E5 prune (with lever SV's budget / share vote)
# as ONE kernel from ROUTE_E5_MIN_ROWS rows (bit for bit stock's top-k + SV's E5 launch)
ROUTE_E5 = True
ROUTE_E5_MIN_ROWS = 1          # with lever RR the fused top-k + E5 + record beats stock top-k + prune + routing at every size (rr_unit)
# lever P1 (gen_p1.py): 49..127 verify rows on smaller RE5 clusters (more rows a CTA) that fit beside the mHC
# stats grid; the routing / E5 arithmetic is cluster-independent (route_e5.self_test covers every size)
ROUTE_E5_SHAPES = ((8, 256, 0), (48, 256, 4), (127, 256, 16), (256, 256, 4))   # (max rows, block threads, rows a CTA; 0 = one CTA): first match
ROUTE_E5_PDL = True
ROUTE_E5_TRIGGER = 0
# lever RE5NC (route_e5.py, gen_re5nc.py): decode / verify calls of more than ROUTE_E5_FUSED_MAX_ROWS rows launch no
# RE5 cluster (a 2..16-CTA cluster needs a whole GPC and waits behind side kernels spread over every GPC): mode 1 =
# stock vision_topk (Triton router) + RE5's one-CTA prune-only mode (+ RR record); mode 2 = RE5's top-k on a plain
# grid (ROUTE_E5_NC_SHAPE = (block threads, rows a CTA)) + the prune-only mode as its PDL dependent; mode 3 = top-k +
# E5 in one plain-grid launch (E5 in the CTA drawing the last ticket; NC_SHAPE's rows a CTA); exact (the same code as
# the fused kernel's halves; self-tested vs stock top-k + prune_ids per bucket, rank-agreed)
ROUTE_E5_FUSED_MAX_ROWS = 127
ROUTE_E5_NC_MODE = 3
ROUTE_E5_NC_SHAPE = (256, 8)
# PDL on the mode 2 / 3 launches (True / False, or N = PDL up to N rows): off = launch at the router reduce's completion,
# every freed SM at once (an early PDL launch at 144 rows left CTAs pending behind the shared-expert / mHC side kernels:
# 22-26 us in ~40 % of the calls; at 48 rows the early launch is faster)
ROUTE_E5_NC_PDL = False
# E5's fp32 reduction trees per stock tile BR (fork/tests/fit_e5.py on the stock kernel's sums, Triton 3.8.0)
E5_TREES = {8: ((1, 1, 4, 'seq', 'desc', 'k'), (1, 4, 2, 'seq', 'desc', 'k')), 16: ((1, 1, 4, 'seq', 'desc', 'k'), (1, 1, 4, 'seq', 'desc', 'k')), 32: ((1, 2, 4, 'seq', 'desc', 'k'), (1, 2, 4, 'seq', 'desc', 'k')), 64: ((1, 2, 4, 'seq', 'desc', 'k'), (1, 2, 4, 'seq', 'desc', 'k')), 128: ((1, 4, 2, 'seq', 'desc', 'k'), (1, 4, 2, 'seq', 'desc', 'k')), 256: ((1, 4, 2, 'seq', 'desc', 'k'), (1, 4, 2, 'seq', 'desc', 'k'))}

# lever DT (dt.py): dense MXFP8 linears of weight (N, K) at rows in [lo, hi] launch the CuTe-DSL tactic
# (mma_tiler_mn, cluster_mn, swap_ab, prefetch, split_k) measured by fork/tests/dense_mt.py; split 1 tactics are
# bit-checked at first eager use, split-K ones need DT_SPLIT_K (fp32 summation order: PRECISION)
DT = True
DT_SPLIT_K = False
# lever EGQ (gen_egq.py): the Engram wkv GEMM (25600, 6144) at decode / verify rows, a bit-identical split-1 tactic (dense_mt.py egq sweep)
# lever STP (gen_stp.py): the shared down projection (5120, 1152) at decode / verify rows, split-1 tactics of <= 60 CTAs (beside FC1)
DT_TACTICS = {(1792, 5120): ((17, 32, ((128, 16), (1, 1), True, False, 1)), (33, 256, ((256, 64), (2, 1), True, False, 1)), (257, 384, ((256, 64), (2, 1), True, False, 1)), (385, 640, ((256, 64), (2, 1), True, False, 1)), (641, 1279, ((256, 128), (2, 1), True, False, 1)), (1280, 1791, ((256, 192), (2, 1), False, False, 1)), (1792, 2815, ((256, 256), (2, 1), False, False, 1)), (2816, 3583, ((256, 192), (2, 2), False, False, 1)), (3584, 4095, ((256, 256), (2, 2), False, False, 1))), (1280, 5120): ((17, 32, ((128, 16), (1, 1), True, False, 1)),), (2304, 5120): ((17, 32, ((128, 16), (1, 1), True, False, 1)), (33, 256, ((256, 64), (2, 1), True, False, 1)), (257, 384, ((256, 64), (2, 1), True, False, 1)), (385, 640, ((256, 64), (2, 1), True, False, 1)), (641, 1279, ((256, 128), (2, 1), True, False, 1)), (1280, 1791, ((256, 192), (2, 1), False, False, 1)), (1792, 2303, ((256, 256), (2, 1), False, False, 1)), (2304, 2815, ((256, 192), (2, 4), False, False, 1)), (2816, 3583, ((256, 192), (2, 1), True, False, 1)), (3584, 4095, ((256, 256), (2, 1), False, False, 1))), (5120, 4096): ((17, 32, ((128, 16), (1, 1), True, False, 1)), (193, 384, ((256, 128), (2, 1), True, False, 1)), (385, 640, ((256, 192), (2, 1), False, False, 1)), (641, 895, ((256, 256), (2, 1), False, False, 1)), (896, 1279, ((256, 192), (2, 4), False, False, 1)), (1280, 1791, ((256, 256), (2, 4), False, False, 1)), (1792, 2303, ((256, 192), (2, 1), False, False, 1)), (2304, 2815, ((256, 256), (2, 1), False, False, 1)), (2816, 4095, ((256, 192), (2, 1), False, False, 1))), (16384, 1280): ((129, 256, ((128, 192), (1, 1), True, False, 1)), (257, 384, ((256, 192), (2, 2), True, False, 1)), (385, 895, ((256, 192), (2, 2), False, False, 1)), (896, 1791, ((256, 192), (2, 1), False, False, 1)), (1792, 4095, ((256, 256), (2, 1), False, False, 1))), (4096, 1280): ((640, 895, ((128, 192), (1, 1), True, False, 1)), (896, 1279, ((256, 128), (2, 1), False, False, 1)), (1280, 1791, ((256, 192), (2, 1), False, False, 1)), (1792, 2303, ((256, 128), (2, 2), False, False, 1)), (2304, 2815, ((256, 192), (2, 1), False, False, 1)), (2816, 3583, ((256, 192), (4, 1), True, False, 1)), (3584, 4095, ((256, 192), (2, 1), True, False, 1))), (5120, 1152): ((65, 128, ((128, 128), (1, 1), True, False, 1)), (129, 192, ((128, 192), (1, 1), True, False, 1)), (193, 256, ((128, 192), (1, 1), False, False, 1)), (257, 384, ((128, 256), (1, 1), False, False, 1)), (385, 640, ((256, 192), (2, 1), False, False, 1)), (641, 895, ((256, 128), (2, 1), False, False, 1)), (896, 1279, ((256, 192), (2, 1), False, False, 1)), (1280, 1791, ((256, 128), (2, 2), False, False, 1)), (1792, 2303, ((256, 192), (2, 1), True, False, 1)), (2304, 2815, ((256, 192), (2, 1), False, False, 1)), (2816, 3583, ((256, 192), (4, 1), True, False, 1)), (3584, 4095, ((256, 192), (2, 1), False, False, 1))), (25600, 6144): ((33, 128, ((128, 192), (1, 1), False, False, 1)), (129, 256, ((256, 192), (2, 1), False, False, 1)), (257, 384, ((128, 192), (1, 2), True, False, 1)))}

# lever L2PF (l2pf.py, CUDA unit l2pf): decode / verify layers >= L2PF_MIN_LAYER issue L2 prefetch hints for their
# WO-A + wo_b (+ router) weights during the attention on a short-lived side-stream grid (no resident fill blocks)
L2PF = True
L2PF_MIN_LAYER = 0
L2PF_ROUTER = True
L2PF_BANDS = ((12, 16, 256, 0),)   # (max rows, grid, block, mode 0 = line prefetch / 1 = bulk prefetch); above the last: off
L2PF_CHUNK = 65536

# lever RR (rr.py; RE5's kernel writes FlashInfer's trtllm-gen routing record): decode / verify routed MoE calls of
# RR_MIN_ROWS..RR_MAX_ROWS rows run FlashInfer 0.7.1's prepared body on the record (no routingIndices kernel); every
# (rows, tactic) shape is verified on a warmup forward (record vs FlashInfer's, GEMM2 rows vs the stock call) first
RR = True
RR_MIN_ROWS = 1
RR_MAX_ROWS = 256

# lever HP: the layer replicas' mHC stats stream (mix stats + sinkhorn beside each sublayer) at the highest stream
# priority, so its few CTAs get SMs ahead of the routed GEMMs' waves (scheduling only)
HP = False
PM = True                      # lever PM: main chain on a top-priority stream above HP_MAX_ROWS
HP_MAX_ROWS = 8            # HP only for forwards of <= HP_MAX_ROWS tokens
# lever SD: from SD_MIN_ROWS decode / verify rows the shared expert (alt stream) and the mHC stats start after the
# router (RE5) instead of beside it, so the one-CTA-cluster router runs alone (scheduling only); 0 = off
SD_MIN_ROWS = 0
# lever STP (stp.py, gen_stp.py): CUDA-graph decode / verify MoE calls of STP_MIN_ROWS..STP_MAX_ROWS rows run the
# shared expert's tail on one top-priority stream (mode "tail": silu + down after the alt stream's gate_up; "late": also
# forked after the router; "all": the whole shared expert forked after the router), joined where stock joins the alt
# stream; lever PM's main stream one level below it (STP_PM_BELOW); first eager call per row count checked vs stock
STP = True
STP_MODE = 'tail'
STP_MIN_ROWS = 9
STP_MAX_ROWS = 384
STP_PM_BELOW = True

# lever SK (sk.py + sk_unit.cu): the mHC stats reduce + sinkhorn of our replicas' decode / verify sublayers of
# 1..SK_MAX_ROWS rows on a CUDA unit (partials in one round trip; stock's epilogue instruction for instruction); each
# row count's first eager call is compared with stock bit for bit (mismatch: stock for that row count)
SK = True
SK_MAX_ROWS = 127
# and the deep_gemm path (128+ rows: two tf32 prenorm GEMMs, reduce with the low part as residual) up to SK_DG_MAX_ROWS
SK_DG = True
SK_DG_MAX_ROWS = 512
SK_DG_ROWS_PER_CTA = 1          # 1 or 4 (the small path always takes 1 row a CTA)
# lever P1 (sk.py partial_shape, gen_p1.py): small-path row counts in [lo, hi] launch stock's own
# _hc_mix_stats_partial_kernel at (slices a CTA, warps, stages) instead of stock's shape (grid / issue order only;
# bits measured by fork/tests/p1_test.py and re-checked per row count at first eager use); () = stock's shape
SK_PARTIAL_SHAPES = ((49, 127, 1, 2, 1),)

# lever CT (ct.py): inside our forward the trtllm-gen launches take the fork's persistent multi-CTA counter buffer
# (the fork's install misses this FlashInfer's resolver: a fresh zero-filled buffer every launch), and decode / verify
# sparse-tail packs that would rewrite the same bytes into the shared table are skipped (the first CT_TAIL_CHECKS
# would-be skips run the pack anyway and compare the table bytes); both exact
CT_COUNTER = True
CT_TAIL = True
CT_TAIL_CHECKS = 30
# lever CT_PREFILL (ct.py + generated ct_pf.py): eager EXTEND attention builds the chunk's SWA index table once and
# reuses the previous tail pack's lens buffer in consumer layers (first CT_PREFILL_CHECKS hits checked vs stock)
CT_PREFILL = True
CT_PREFILL_CHECKS = 8

# lever HF: the final hc_combine reads pre_hc_head in bf16 (the kernel converts on load: bit-identical, one
# elementwise kernel and its fp32 copy fewer a forward)
HF = True

# lever FL (fl.py): in eager forwards of the modes FL_MODES (no capture / breakable graph / compile / AutoTuner tuning)
# our replicas' dense MXFP8 linears of >= FL_MIN_ROWS rows launch the kernels stock's call launches (FlashInfer 0.7.1
# CuTe-DSL mxfp8_quantize + mm_mxfp8 GEMM at stock's tactic) directly; each plan key's first call runs stock with
# launch recorders, and the fast path is kept only if its kernels, arguments and output bytes equal stock's (exact)
FL = True
FL_MIN_ROWS = 33
# FL launches DT's band tactics on plans learned from DT's call (else DT's own path)
FL_DT = True
FL_MODES = ('EXTEND',)

# lever AR (ar.py): inside our eager EXTEND / MIXED forwards the TP all-reduces (wo_b, MoE post-experts, embedding,
# Engram) issue the collective stock picks for the tensor directly (custom all-reduce v2 kernel up to its eager ceiling,
# else ProcessGroupNCCL allreduce on the current stream): the same final call with the same arguments, chosen once per
# shape by stock's own predicates; the first call of every (plan, dtype, size class) up to AR_CHECK_MAX_BYTES also runs
# stock on a copy and compares the bytes (two collectives on every rank); a failed check turns that whole (plan kind,
# algo, dtype) family off for good. AR_PINS = source hashes of the stock functions a plan reproduces (fork 5e72ce5,
# torch 2.14.1; read through inspect.getsource in the engine): any difference leaves the lever off
AR = True
AR_CHECK = True
AR_CHECK_MAX_BYTES = 64 << 20
AR_CA_NO_REWRAP = True      # the custom AR's torch.Tensor result as is (stock re-wraps it: torch.from_dlpack alias)
AR_MOE_ENTRY = True         # the MoE replicas' post_experts_all_reduce: config half once a forward, then the MoE-TP sum
AR_FUSE_GATE = True         # deepseek_v4.can_fuse_all_reduce: > 384 rows answered False (stock's answer) without the lookup
AR_CHUNK = False             # TP2 bf16 tensors stock sends to ProcessGroupNCCL (larger than the custom AR workspace): the custom
                                 # AR v2 kernel on workspace-sized chunks, sums copied back in place (bit-identical at 2 ranks; checked)
AR_CHUNK_MAX_BYTES = 0           # chunk cap (0: the custom AR's own max_size)
AR_PIN = True
AR_PINS = {
    'sglang.srt.distributed.parallel_state:GroupCoordinator.all_reduce': '69fe5dcb304ce86b',
    'sglang.srt.distributed.parallel_state:GroupCoordinator._resolve_outplace_all_reduce_method': '7187faf3cbdaa812',
    'sglang.srt.distributed.parallel_state:GroupCoordinator._can_use_flashinfer_allreduce': '226ce24c4ade6e15',
    'sglang.srt.distributed.parallel_state:GroupCoordinator._all_reduce_out_place': 'aa21a3b4d454bcbf',
    'sglang.srt.distributed.parallel_state:GroupCoordinator._all_reduce_in_place': '1ad693335b2e54ef',
    'sglang.srt.distributed.parallel_state:inplace_all_reduce': '8d4aaba6d8cdcec5',
    'sglang.srt.distributed.parallel_state:outplace_all_reduce': '27f3edd4c482e33b',
    'sglang.srt.distributed.device_communicators.custom_all_reduce_v2:CustomAllReduceV2.should_custom_ar': '4164fd44bb94b68f',
    'sglang.srt.distributed.device_communicators.custom_all_reduce_v2:CustomAllReduceV2.custom_all_reduce': 'b4e6aebc49436815',
    'sglang.srt.distributed.device_communicators.custom_all_reduce_v2:CustomAllReduceV2._pick_config': '1629f5ba05e14d58',
    'sglang.srt.distributed.device_communicators.custom_all_reduce_v2:CustomAllReduceV2._can_use_graph': 'd004068f94e8a127',
    'sglang.kernels.ops.communication.all_reduce:custom_all_reduce': '609098ab3ab49dbd',
    'sglang.kernels.ops.communication.all_reduce:get_all_reduce_module': '0c8a897d6018be9f',
    'sglang.srt.utils.custom_op:CustomOpWrapper.real_impl': '6fd3d03253250900',
    'torch.distributed.distributed_c10d:all_reduce': 'd4fb262e12e01f83',
    'sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe:can_fuse_all_reduce': '5f781becc2d683e1',
    'sglang.srt.layers.moe.utils:post_experts_all_reduce': 'e20fba73c5fec493',
    'sglang.srt.layers.moe.utils:_post_experts_sum': '7b6aac8e351e13ed',
    'sglang.srt.layers.moe.utils:should_skip_post_experts_all_reduce': 'ee293bfe0643d1d8',
    'sglang.srt.layers.moe.utils:should_skip_mlp_all_reduce': '0f28950f738dc02b',
    'sglang.srt.distributed.communication_op:moe_tensor_model_parallel_all_reduce': 'd0fec3f9918c3030',
}

# lever MO (mo.py): eager EXTEND forwards only (no capture, not in a breakable / piecewise prefill graph). MO_MOE: the
# routed NVFP4 MoE op as FlashInfer 0.7.1's native trtllm-gen op on stock's own arguments, at the tactic stock's
# autotuner chose for the call's key (learned from the key's first, stock call; first use per tactic checked bitwise);
# MO_QUANT: its NVFP4 input quantize as FlashInfer's cached CuTe-DSL kernel launched directly (first use per shape class
# checked bitwise); MO_MHC: mega-mHC posts as deep_gemm's launch without the wrappers' Python (first use per post kind
# checked bitwise; measured no host saving, off by default). All exact.
MO_MOE = True
MO_MOE_MIN_ROWS = 1
MO_QUANT = True
MO_MHC = False

# lever RT (rt.py): eager EXTEND forwards only (no capture, not in a breakable / piecewise prefill graph). RT_ROUTER:
# the MoE router's moe_fused_gate launch as stock's compiled Triton kernel launched directly with the argument list
# stock's own call recorded (learned per call key; first direct launch per compiled kernel checked bitwise); RT_GATE:
# the router gate's cuBLAS torch.mm without the module / wrapper layers (first call per row class checked bitwise);
# RT_WOA: WO-A of 1..4095 rows as the fork's own >= 4096-row strided bmm (first call per row class checked bitwise).
RT_ROUTER = True
RT_GATE = True
RT_WOA = True

# lever IX (ix.py + generated ix_dg.py): in eager EXTEND / MIXED forwards of our node (no capture, no breakable
# prefill graph) the index layers' layer-invariant host work runs once per forward: token_req_indices (IX_REQ) and the
# DeepGEMM prefill indexer's column slots / request starts / compressed lengths per ratio (IX_DG); in the first
# IX_CHECK_FORWARDS reusing forwards the first hit of every memo entry is recomputed and compared with stock byte for
# byte (any difference: that memo off for good); exact. IX_AB > 0 (diagnostic): the memo alternates per eligible
# forward and a host-time A/B line is printed every IX_AB forwards
IX = True
IX_REQ = True
IX_DG = True
IX_CHECK_FORWARDS = 4
IX_AB = 0

# lever QN (qn.py + CUDA unit qn_unit): inside our model forward, MQALayer._normalize_q_lora of eager EXTEND forwards
# (QN_MIN_ROWS..QN_MAX_ROWS rows) runs as one kernel: the q_norm RMSNorm + wq_b's MXFP8 input quantize, bit for bit stock's path of
# the row count (FlashInfer CuTe-DSL norm + quantize above 64 rows, the Triton fused kernel at 1..64; wq_b takes the
# quantized input); with QN_GRAPH_MAX_ROWS >= 65 also decode / verify calls of 65..QN_GRAPH_MAX_ROWS rows (CUDA graphs).
# The first eager call of each check key runs stock too and compares every output byte (mismatch: off for the process)
QN = True
QN_MIN_ROWS = 1               # 65: keep stock's Triton fused kernel for 1..64-row extends
QN_MAX_ROWS = 1 << 20
QN_GRAPH_MAX_ROWS = 384
QN_PDL = True                  # programmatic dependent launch (weights loaded before the wait); scheduling only

# lever MF (mf.py): eager EXTEND sparse-MLA launches (ct_pf.py) go straight to stock's trtllm-gen launcher with the
# arguments stock's wrapper derives; first call per key: stock's launcher call recorded and ours compared (arguments, output
# bytes); MF_PINS = sha256[:16] of the pinned FlashInfer 0.7.1 sources (any difference: off)
MF = True
MF_PINS = {
    'trtllm_batch_decode_sparse_mla_dsv4': 'ab6fa466e365fd1a',
    '_check_dsv4_sparse_mla_inputs': '472892f1d3b8948a',
    '_normalize_dsv4_sparse_mla_kv_cache': '8d2b4f1eaf981117',
    '_normalize_dsv4_topk_lens': '3edee82c42cb4cc4',
    '_resolve_dsv4_sparse_mla_backend': '293bc5ad3e46d5d6',
    '_validate_dsv4_sync_checks': '5cb4882ab6dfac5b',
}

# lever FG (fg.py + generated fg_layer.py / moe_fwd.forward_normal_fg): in our eager EXTEND forwards (no capture, no
# breakable / piecewise prefill graph, text-only, CUDA graphs not disabled) each eligible layer's FFN block of
# FG_MIN_ROWS..FG_MAX_ROWS rows replays two CUDA graphs captured at a row bucket S >= M (router..shared add; the mega-mHC
# post), with the router gate GEMM and the TP all-reduce eager at the real rows between them; buckets never cross an
# edge of stock's row-count-dependent launch choices (padding <= FG_PAD inside a region, sizes aligned to FG_ALIGN); a
# bucket is admitted when FG_MEM_MARGIN bytes stay free beyond the static buffers + the graph pool (FG_POOL_RESERVE: the
# first capture reserves the pool at the largest bucket, FG_POOL_BYTES_PER_ROW x rows; later buckets FG_POOL_SLACK); the
# first replay of every (bucket, layer) is compared with the eager path byte for byte (FG_CHECK; mismatch: that bucket
# off for good); FG_PINS = source hashes of the stock functions FG lifts / mirrors (any difference: FG off); FG_AB = N > 0
# is a diagnostic only (FG on every N-th eligible forward, an A/B inside one engine), 0 in a shipped bundle;
# FG_MAX_CAPTURE_FORWARDS: a bucket whose capture forwards end with no layer captured on any rank is retried at most
# this many forwards, then off (one agreement per such forward otherwise, forever)
FG = True
FG_MIN_ROWS = 385
FG_MAX_ROWS = 8192
FG_PAD = 0.3
FG_ALIGN = 64
FG_CHECK = True
FG_PIN = True
FG_MEM_MARGIN = 6 << 30
FG_POOL_BYTES_PER_ROW = 120 << 10
FG_POOL_RESERVE = True
FG_POOL_SLACK = 256 << 20
FG_MAX_WARMUPS = 4
FG_MAX_CAPTURE_FORWARDS = 8
FG_AB = 0
FG_PINS = {
    'sglang.srt.models.deepseek_v4:DeepseekV4DecoderLayer.forward_hc_pre_from_prev': '5dd33fadb392594e',
    'sglang.srt.models.deepseek_v4:DeepseekV4DecoderLayer._run_moe_ffn_dp_sync': 'db995f3eeabaaf56',
    'sglang.srt.models.deepseek_v4:DeepseekV4DecoderLayer._can_fuse_ffn_mhc': '78f13a1ff6daffcc',
    'sglang.srt.models.deepseek_v4_mhc:run_attn_post': '77de828299d2aada',
    'sglang.srt.models.deepseek_v4_mhc:run_moe_post': '62431f4cc899ea63',
    'sglang.srt.models.deepseek_v4_mhc:_post_fusion': '0c0bf72e6d438547',
    'sglang.srt.models.deepseek_v4_mhc:_compute_triplet': 'b3e5a53247eff8b6',
    'sglang.srt.models.deepseek_v4_mhc:combine': '4c1e50d8489abfae',
    'sglang.srt.models.deepseek_v4_mhc:_can_use_mega_mhc': '9684c0e84847b222',
    'sglang.kernels.ops.layernorm.mhc_mega:mhc_mega_boundary': '9d234078e63f577c',
    'sglang.srt.models.deepseek_v2:DeepseekV2MoE.forward': '9671a4e7e192c785',
    'sglang.srt.models.deepseek_v2:DeepseekV2MoE.forward_normal': 'd0d489bac3964958',
    'sglang.srt.models.deepseek_v2:MoEGate.forward': '9ed33b8ec8173096',
    'sglang.srt.models.deepseek_v2:DeepseekV2MLP.forward': '5cfc97f1bfe201d6',
    'sglang.srt.layers.moe.moe_runner.flashinfer_trtllm:_get_fp4_moe_output_buffer': 'e5e8f1b470154b0a',
    'sglang.srt.layers.moe.moe_runner.flashinfer_trtllm:trtllm_moe_enable_pdl': '991295498440298f',
    'sglang.srt.layers.moe.moe_runner.base:moe_output_buffer_ctx': '93a536f263cac096',
    'sglang.kernels.ops.gemm.bf16_fp32:_linear_bf16_fp32_cublas': 'cd2c011371491088',
    'sglang.kernels.ops.moe.moe_fused_gate:moe_fused_gate': 'd58bf7342179f035',
    'sglang.kernels.ops.moe.moe_fused_gate:_router_triton_kernel': '76b8ea362d5d6d11',
    'sglang.srt.multimodal.dsv41.vl_routing:vision_topk': '4817cd8f406c0041',
    'sglang.srt.multimodal.dsv41.vl_routing:_finish_vision_topk': '24ce9365ce1d7024',
    'sglang.srt.layers.quantization.fp8_utils:_mxfp8_decode_tactic': '838310aa01dfd458',
    'sglang.srt.layers.quantization.fp8_utils:flashinfer_mxfp8_blockscaled_linear': '6fb544f3c043ec0a',
    'sglang.srt.layers.quantization.fp8:Fp8LinearMethod.apply': '45cb0b2d8b56ae2c',
    'flashinfer.quantization.kernels.mxfp8_quantize:mxfp8_quantize_cute_dsl': '995eb9350d599b1b',
    'flashinfer.fused_moe.backends.trtllm.sm100_runner:MoERunner.get_valid_tactics': '5575d0951a00ab0b',
    'flashinfer.fused_moe.utils:map_to_hybrid_bucket_uncapped': '8e3fbc4bb909bddb',
    'sglang.srt.models.deepseek_v4_prune:decode_rows': '3095f5d52b16e68c',
    'sglang.srt.models.deepseek_v4_prune:prune_topk': '87cfdd1703392733',
}

# lever HQ (hq.py + CUDA unit hq_unit + generated hq_layer.py): inside our model forward the attention hyper-connection's
# combine of calls above 128 rows (stock: hc_combine_norm's bf16 rows, then wqkv_a's own flashinfer_mxfp8_quantize) runs
# as one kernel writing the bf16 rows, e4m3 values and 128x4-swizzled UE8M0 scales (padded rows zero) stock's pair writes;
# wqkv_a takes them as stock's <= 128-row Mxfp8SwizzledInput. Decode / verify calls of 129..HQ_MAX_ROWS rows (CUDA
# graphs; 0 = off) and, with HQ_EAGER, eager EXTEND calls of 129..HQ_EAGER_MAX_ROWS rows. The first eager call of each
# check key runs stock too and compares every output byte (mismatch: off for the process)
HQ = True
HQ_MAX_ROWS = 384
HQ_EAGER = True
HQ_EAGER_MAX_ROWS = 1 << 20
HQ_PDL = 1                     # PDL (weight loads before the wait); 1: dependents released right after the wait (B300:
                               # best ahead of wqkv_a's GEMM), 2: after the stores, 0: plain launch; scheduling only

# lever QR (qr.py + CUDA unit qr_unit): MQALayer._compute_q_b on our attention replicas: the FP8 query rope of the
# no-q-head-norm path (stock fused_q_norm_rope<bf16, 512, 64, PosT, kUsePDL, false, true>) on the unit's kernel: same
# arithmetic, hardware e4m3x2 conversion + NaN fix-up instead of the per-element emulation; bytes == stock. Install
# self-test + the first eager call of each check key run stock too and compare every byte (mismatch: off for the process);
# CUDA-graph captures (decode / verify, breakable prefill) use keys verified eagerly only
QR = True
QR_MIN_ROWS = 1
QR_MAX_ROWS = 1 << 20

# lever RO (ro.py + CUDA unit ro_unit): our attention replicas' inverse RoPE of the attention output (fork_attn.py:
# stock's fused_rope_inplace(o[..., -64:], None, freqs_cis, positions, inverse=True), every layer) as one launch with
# stock's per-pair arithmetic (its sm_103a FMUL / FFMA order, bf16 RNE pack), 16-byte loads and a thread's freq pairs
# loaded once for its heads, stock's PDL placement; prefill (eager EXTEND) and decode / verify (CUDA graphs). The first
# eager call of each check key (table, heads, position dtype, strides, tile, rows mod tokens a CTA) runs stock on a
# copy and compares every byte (mismatch: stock's bytes, off for the process); before the first check of a (table,
# heads, position dtype) a synthetic self-test (random / cancelling / subnormal / bit-pattern rows at random positions,
# every tile) must equal stock too (graph warmups run at positions 0 = the identity rotation); a capture takes the unit
# only for keys verified eagerly. RO_TILES: (max rows, heads a thread, tokens a CTA, fixed 512 head stride), first match
# (tests/ro_test.py sweep, B300 cold-L2 CUDA graphs)
RO = True
RO_TILES = ((640, 1, 2, 0), (2048, 2, 4, 0), (1073741824, 8, 8, 0))

# lever FGP (fgp.py, gen_fgp.py): at the engine's first eligible eager EXTEND forward (any row count; the live replay's
# untimed warmup) lever FG's admission + warmups + both graph captures for every new bucket of its table and every FG
# layer, on deterministic synthetic static-buffer contents (FGP_SEED; FMT bands checked first through FMT's own
# check, without the all-reduce), largest bucket first, rank-agreed per bucket; buckets past the rank-agreed wall-clock
# budget FGP_BUDGET_S, waiting for a band check on some rank or failing their capture stay on FG's lazy path (FG's
# admission turns off the ones that can never go live -- FMT band off -- or lack memory). A pre-captured bucket's first
# real use still replays, runs the eager path at the real rows and compares every byte before FG's rank agreement
# turns it on
FGP = True
FGP_BUDGET_S = 120.0
FGP_SEED = 20261015

# lever AG (ag.py, gen_ag.py): in our eager EXTEND forwards (no capture, no breakable / piecewise prefill graph, CUDA
# graphs not disabled) each attention replica's post-attention segment of AG_MIN_ROWS..AG_MAX_ROWS rows (lever RO's
# inverse RoPE, lever RT's WO-A strided bmm, wo_b's MXFP8 quantize + GEMM via levers FL / DT) replays one CUDA graph
# captured at a row bucket S >= M, the attention kernel writing straight into AG's static input buffer and wo_b's
# all-reduce eager at the real rows; buckets never cross an edge of stock's row-count-dependent launch choices (DT's
# wo_b band, 2T/SF, RO's tile; padding <= AG_PAD, sizes aligned to AG_ALIGN); a bucket is admitted after the WO-A
# cuBLAS row-consistency sweep (AG_WOA_SWEEP, seed AG_SEED; AG_PRESWEEP: every bucket's sweep once at install = engine
# startup, within AG_PRESWEEP_BUDGET_S, the rest at admission) when AG_MEM_MARGIN bytes stay free beyond the static
# buffers + the graph pool (AG_POOL_RESERVE: the first capture reserves it at the largest bucket,
# AG_POOL_BYTES_PER_ROW x rows; later buckets AG_POOL_SLACK); the first replay of every (bucket, layer) is compared with
# the eager path byte for byte (AG_CHECK; mismatch: that bucket off for good) and agreed over the TP ranks; AG_PINS =
# source hashes of the stock functions AG mirrors (any difference: AG off); a bucket whose capture forwards end with no
# layer captured on any rank is retried at most AG_MAX_CAPTURE_FORWARDS forwards; AG_AB = N > 0 is a diagnostic only (AG on
# every N-th eligible forward, an A/B inside one engine), 0 in a shipped bundle
AG = True
AG_MIN_ROWS = 385
AG_MAX_ROWS = 4095
AG_PAD = 0.3
AG_ALIGN = 64
AG_CHECK = True
AG_PIN = True
AG_WOA_SWEEP = True
AG_PRESWEEP = True
AG_PRESWEEP_BUDGET_S = 60.0
AG_SEED = 20261015
AG_MEM_MARGIN = 6 << 30
AG_POOL_BYTES_PER_ROW = 32 << 10
AG_POOL_RESERVE = True
AG_POOL_SLACK = 64 << 20
AG_MAX_WARMUPS = 4
AG_MAX_CAPTURE_FORWARDS = 8
AG_AB = 0
AG_PINS = {
    'sglang.srt.models.deepseek_v4:MQALayer._project_wo_b': 'd5fc3144850433a7',
    'sglang.srt.models.deepseek_v4:MqaAttentionBase.use_flashinfer_mxfp8_wo_b': 'f022da8617af0e49',
    'sglang.srt.models.deepseek_v4:_apply_wo_a_bf16_matmul': '4e0caf34374ac7e1',
    'sglang.srt.layers.linear:RowParallelLinear.forward': 'd994d7ab3245a371',
    'sglang.srt.layers.quantization.fp8:Fp8LinearMethod.apply': '45cb0b2d8b56ae2c',
    'sglang.srt.distributed.device_communicators.pynccl_allocator:use_symmetric_memory': 'c033c22c721fd015',
    'sglang.srt.layers.attention.deepseek_v4_backend:DeepseekV4AttnBackend.forward': 'd7572f13d4319cd6',
    'sglang.srt.layers.attention.deepseek_v4_backend:DeepseekV4AttnBackend._forward_attention': '1a8c6445c043fe4b',
    'sglang.srt.layers.attention.deepseek_v4_trtllm_backend:DeepseekV4TrtllmAttnBackend._forward_trtllm': '373646325a39608a',
    'sglang.srt.layers.attention.deepseek_v4_trtllm_backend:DeepseekV4TrtllmAttnBackend._forward_trtllm_prefill': 'fb7eaeb80dca2188',
    'flashinfer.quantization.kernels.mxfp8_quantize:mxfp8_quantize_cute_dsl': '995eb9350d599b1b',
}

# lever AQ (ag.py "LEVER AQ", gen_aq.py): with lever AG on, the PRE-attention q path of the same eager EXTEND calls
# (wqkv_a via FL / DT, lever QN's norm + quantize, wq_b via FL / DT, lever QR's FP8 query rope) replays a second CUDA
# graph per (layer, AG bucket), x copied into a static input, stock's _forward_prepare remainder (KV store, indexer /
# compressor / low-ratio sources) eager on the static outputs; the table's key adds DT's wqkv_a / wq_b bands and
# wqkv_a's 2T/SF; first replay per (bucket, layer) checked bitwise vs the eager q path and agreed over the ranks;
# AQ_POOL_BYTES_PER_ROW sizes the graph pool's q-path share for AG's memory gate; AQ_PINS = source hashes of the stock
# functions AQ mirrors (any difference: AQ off, AG unchanged); AQ_AB = N > 0 is a diagnostic only (q graphs replayed on
# every N-th eligible forward, an A/B inside one engine), 0 in a shipped bundle
AQ = True
AQ_AB = 0
AQ_POOL_BYTES_PER_ROW = 64 << 10
AQ_PINS = {
    'sglang.srt.models.deepseek_v4:MQALayer._forward_prepare': '73be4f0c994810bf',
    'sglang.srt.layers.linear:ReplicatedLinear.forward': '3e09ca711135dffb',
    'sglang.srt.layers.linear:ColumnParallelLinear.forward': 'ec920699788e5208',
}

# lever AGO (ag.py "LEVER AGO", gen_ago.py): with levers AG + AQ on, AGO = the gate's forward-constant tests once per
# (forward, positions, rows), plan() reusing qprep()'s gate of the same attention call, the replay checks' row-count
# keys once per forward (every lever verdict still read per call); AGO_ZC = zero-copy q graphs (a second q graph per
# (layer, bucket) reading lever FG's static next-attention input, replayed without the x copy when x is that buffer's
# first M rows, its first real-row replay checked bitwise vs the eager q path on this rank); AGP = every new AG bucket
# admitted + warmed + captured (post, q and ZC graphs) at the engine's first eligible eager EXTEND forward on synthetic
# static contents (AGP_SEED) within AGP_BUDGET_S, agreed per bucket, the first real use still checked bitwise vs eager
# and agreed; AGO_AB = N > 0 is a diagnostic only (AGO + AGO_ZC on every N-th eligible forward), 0 in a shipped bundle
AGO = True
AGO_ZC = True
AGO_AB = 0
AGP = True
AGP_BUDGET_S = 90.0
AGP_SEED = 20261015

# lever FGB (fg.py "LEVER FGB", gen_fgb.py): with lever FG on, graph 1 of an FG layer's FFN block also holds the
# router gate GEMM (buckets whose cuBLAS sweep -- stock's torch.mm at every row count of the bucket == the first rows at
# S, byte for byte, FGB_SEED rows, FGB_GATE_SWEEP -- passed on every rank, agreed at FG's admission) and, in a layer
# that follows an FG layer (FGB_MEGA), the attention post's mega boundary on a static copy of the reduced attention
# output + the previous FG layer's static residual / triplet; first replay per (bucket, layer) checked bitwise vs
# eager (the mega's outputs, the gate logits at M and S rows, the MoE output, graph 2) and agreed over the ranks;
# buckets above FGB_MAX_ROWS keep FG's plain graph 1 (GPU-bound there: graph 1's padding + copy cost ~+1 % at
# 6096 rows; measured -8 % at 1096, neutral 1596..3096 rows);
# FGB_AB = N > 0 is a diagnostic only (FGB's graph 1 on every N-th eligible forward, FG's plain path otherwise, both
# captured; an A/B inside one engine), 0 in a shipped bundle
FGB = True
FGB_MEGA = True
FGB_GATE_SWEEP = True
FGB_SEED = 20261015
FGB_MAX_ROWS = 4096
FGB_AB = 0

# lever RTR (rtr.py + CUDA unit rtr_unit, gen_rtr.py): CUDA-graph decode / verify MoE calls (the lifted
# forward_normal_dual_stream) of RTR_ROWS rows run the router gate GEMM (x [rows, 5120] bf16 . W [384, 5120]^T -> fp32)
# as ONE kernel, bit for bit stock's cuBLAS nvjet split-K GEMM + splitKreduce (S = 2 / 3) or no-split GEMM (S = 1):
# S learned from stock per row count on each rank at prepare, then RTR_SELFTEST byte checks per row count (seed
# RTR_SEED); first eager use per (row count, layer) checked vs stock; any difference -> stock. RTR_PDL: PDL launch (the
# static weight's first stages loaded before griddepcontrol.wait); RTR_TILES: (lo rows, hi rows, S, tile config) bands
# overriding the unit's default tile (rtr_unit.cu kConfigs; a tile sets the grid, so how many SMs stay free for the
# side streams; bits are tile-independent)
RTR = True
RTR_ROWS = (18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126)
RTR_TILES = ()
RTR_PDL = True
RTR_SELFTEST = 6
RTR_SEED = 20261015

# lever SKW (skw.py + CUDA unit skw_unit, gen_skw.py): CUDA-graph captures of lever SK's reduce + sinkhorn (sk.py _ours /
# _ours_dg) at row counts of a SKW_BANDS (lo rows, hi rows, grid cap) band run on skw_unit: rows a block
# SKW_ROWS_PER_BLOCK (small path 8 / 16, deep_gemm path 16 / 32), at most `cap` CTAs walking the row blocks, loads
# per SKW_ASYNC, sk_unit's per-row arithmetic (bytes == sk_unit == stock); prepare self-test (seed SKW_SEED) agreed
# over the TP ranks, first eager call per (rows, path, slices) byte-checked vs sk_unit; eager forwards run sk_unit
SKW = True
SKW_BANDS = ((9, 136, 8),)
SKW_ROWS_PER_BLOCK = (16, 32)
SKW_ASYNC = 1            # 1: partials land in shared memory by cp.async (no register staging), 0: register loads
SKW_SUBLAYERS = ('attn', 'ffn')   # whose stats are served: the attention sublayers' ("attn") / the FFN sublayers' ("ffn")
SKW_SEED = 20261016

# lever MOET (moet.py, gen_moet.py): decode / verify MoE calls of rows in [lo, hi] run the routed NVFP4 expert op
# (trtllm-gen FC1 + FC2) at tactic [tile, config] instead of the fork's decode pin; every tactic bit-identical
# (tests/moet_micro.py on real routing); decided by forward mode + rows + shape only (eager == graphs)
MOET = True
MOET_TACTICS = ((43, 150, [16, 312]),)

# lever MHC (mhc.py + CUDA unit mhc_unit, gen_mhc.py): inside our model forward, small-path mHC stats calls of
# MHC_MIN_ROWS..MHC_MAX_ROWS rows with a replica sublayer's mixing weight compute the stats PARTIAL (stock's layout)
# on the unit, short and wide, instead of stock's 240..640-CTA Triton partial; lever SK's reduce + sinkhorn unchanged.
# MHC_CONFIG "x<MT>": EXACT -- stock's tf32x3 partial bit for bit (inside sk._ours: lever SK's first-call bitwise
# check per row count applies; prepare self-test bitwise on real weights, agreed over the TP ranks). "p<P>t<MT>":
# TOLERANCE bf16x3 kernel (not bitwise; called from sk._wrapped without the check, eager == graph).
MHC = True
MHC_MIN_ROWS = 1
MHC_MAX_ROWS = 127
MHC_CONFIG = 'x2'   # x<16-row m-tiles a CTA>: exact | p<weight parts>t<m-tiles>: tolerance (mhc_unit.cu kConfigs)
MHC_SELFTEST_ATOL = 1e-4   # tolerance configs only (relative, partials)
MHC_SEED = 20261016

# lever GQ (gq.py + CUDA unit gq_unit, gen_gq.py): in lever STP's split, the shared expert's gate_up input MXFP8
# quantize at GQ_MIN_ROWS..GQ_MAX_ROWS rows on ceil(rows / GQ_ROWS_PER_CTA) CTAs (stock: FlashInfer's 128-CTA kernel,
# which keeps RE5's cluster from a free GPC when it overlaps the router end); stock's bytes (prepare self-test + first
# eager call per (rows, stride) checked); GQ_PDL 0 plain / 1 dependents after the wait / 2 after the stores (stock-like)
GQ = True
GQ_MIN_ROWS = 9
GQ_MAX_ROWS = 48
GQ_ROWS_PER_CTA = 8
GQ_PDL = 2
GQ_SEED = 20261016

# lever AUX (aux.py + eg_gather.cu auxm::mean_kernel, gen_aux.py): lever EGQ's side-stream DSPARK aux means
# (stock's aux.mean(dim=1), layers 37-39) at 1..AUX_MAX_ROWS rows on a small fixed grid instead of torch's
# ~1000-CTA reduce_kernel<MeanOps>; AUX_LAUNCH ((max rows, CTAs), ...); stock's bytes (prepare self-test at every row
# count, rank-agreed, + first eager use per row count checked)
AUX = True
AUX_MAX_ROWS = 384
AUX_LAUNCH = ((64, 24), (112, 32), (384, 48))
AUX_SEED = 20261016

# lever FIN (fin.py, gen_fin.py; no CUDA unit: one JIT module built in prepare): the replicas' fused attention post
# (identity-routed all-reduce + mHC + norm) and MoE post (finalize + all-reduce + mHC [+ norm + MXFP8 quant]) on our copy
# of stock's kernel: epilogue inputs loaded before the peer poll, no store of the discarded reduced row, identity routing
# in the kernel; same FP ops / order / push protocol; first eager call per key checked vs stock, rank-agreed
FIN = True
FIN_EARLY_PDL = False
FIN_MAILBOX = False
FIN_MIN_ROWS = 1
FIN_MAX_ROWS = 384
FIN_TOP_K = 6
# which epilogue loads move before the peer poll: bit 0 the residual vectors, bit 1 post / comb / pre + norm weight
FIN_PREPOLL = 3

# lever R5P (r5p.py + CUDA unit r5p_unit, gen_r5p.py): RE5's fused top-k + E5 (+ lever RR's record) at R5P_MIN_ROWS..
# R5P_MAX_ROWS decode / verify rows inside CUDA-graph captures runs on r5p_unit: a cluster of ceil(rows / R) CTAs of
# R5P_THREADS threads (0: 384 up to 12 rows a CTA, else 512), one row a warp, R = ceil(rows / R5P_CLUSTER)
# (R5P_CLUSTER > 0; else R5P_ROWS_PER_CTA, 0 = threads / 32); E5 by the routing warps on the pushed histogram, the leader builds the record only; RE5's bytes
# (self-test vs stock + RE5 at prepare, gloo-agreed; first eager use per (rows, layer) vs RE5; eager calls always RE5)
R5P = True
R5P_MIN_ROWS = 49
R5P_MAX_ROWS = 127
R5P_THREADS = 512
R5P_CLUSTER = 0
R5P_ROWS_PER_CTA = 0
R5P_SELFTEST = 4
R5P_SEED = 20261016

_announced: set = set()


def announce(tag: str, detail: str) -> None:
    """One stderr line per process and outcome, so a TP2 run that falls back to stock is visible in the engine log."""
    if tag not in _announced:
        import sys

        _announced.add(tag)
        print(f"CACHEON_DSV4F {tag} {detail}", file=sys.stderr, flush=True)


def first(tag: str) -> bool:
    """Not announced yet: callers on hot paths build the detail string only then."""
    return tag not in _announced
