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
EG_EARLY_MAX_ROWS = 16384        # the fork's prefill chunk
EG_EARLY_LAUNCH = ((4096, 64, 1), (1 << 62, 16, 4))   # side-stream bands: few CTAs at prefill sizes

# lever FMT (fmt.py): extend MoE calls of rows in [lo, hi] run the routed NVFP4 expert op at the measured tactic
# [tile, config] (fork/tests/fork_mt.py on the fork image; every tactic bit-identical, first use per band bit-checked)
FMT_TACTICS = ((256, 639, [32, 218]), (640, 1023, [64, 58]), (1024, 3071, [32, 218]), (3072, 10239, [128, 5]), (10240, 13311, [128, 21]), (13312, 16384, [256, 0]))
# decode / verify MoE calls of rows in [lo, hi]: the measured tactic instead of the fork's pin (fork_mt.py fmt3 on SV-pruned
# routing, every tactic bit-identical; first eager use per band bit-checked)
FMT_DECODE_TACTICS = ()

# lever SV (sv.py): the fork's E5 prune with budget SV_BUDGET at every pruned decode / verify call and a share vote
# (an expert picked by <= SV_MAX_COUNT rows is dropped when every holder drops it) from SV_MIN_ROWS rows; PRECISION
SV = True
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
ROUTE_E5_SHAPES = ((8, 256, 0), (256, 256, 4))   # (max rows, block threads, rows a CTA; 0 = one CTA): first match
ROUTE_E5_PDL = True
ROUTE_E5_TRIGGER = 0
# E5's fp32 reduction trees per stock tile BR (fork/tests/fit_e5.py on the stock kernel's sums, Triton 3.8.0)
E5_TREES = {8: ((1, 1, 4, 'seq', 'desc', 'k'), (1, 4, 2, 'seq', 'desc', 'k')), 16: ((1, 1, 4, 'seq', 'desc', 'k'), (1, 1, 4, 'seq', 'desc', 'k')), 32: ((1, 2, 4, 'seq', 'desc', 'k'), (1, 2, 4, 'seq', 'desc', 'k')), 64: ((1, 2, 4, 'seq', 'desc', 'k'), (1, 2, 4, 'seq', 'desc', 'k')), 128: ((1, 4, 2, 'seq', 'desc', 'k'), (1, 4, 2, 'seq', 'desc', 'k')), 256: ((1, 4, 2, 'seq', 'desc', 'k'), (1, 4, 2, 'seq', 'desc', 'k'))}

# lever DT (dt.py): dense MXFP8 linears of weight (N, K) at rows in [lo, hi] launch the CuTe-DSL tactic
# (mma_tiler_mn, cluster_mn, swap_ab, prefetch, split_k) measured by fork/tests/dense_mt.py; split 1 tactics are
# bit-checked at first eager use, split-K ones need DT_SPLIT_K (fp32 summation order: PRECISION)
DT = True
DT_SPLIT_K = False
DT_TACTICS = {(1792, 5120): ((17, 32, ((128, 16), (1, 1), True, False, 1)), (33, 256, ((256, 64), (2, 1), True, False, 1)), (257, 384, ((256, 64), (2, 1), True, False, 1)), (385, 640, ((256, 64), (2, 1), True, False, 1)), (641, 1279, ((256, 128), (2, 1), True, False, 1)), (1280, 1791, ((256, 192), (2, 1), False, False, 1)), (1792, 2815, ((256, 256), (2, 1), False, False, 1)), (2816, 3583, ((256, 192), (2, 2), False, False, 1)), (3584, 4095, ((256, 256), (2, 2), False, False, 1))), (1280, 5120): ((17, 32, ((128, 16), (1, 1), True, False, 1)),), (2304, 5120): ((17, 32, ((128, 16), (1, 1), True, False, 1)), (33, 256, ((256, 64), (2, 1), True, False, 1)), (257, 384, ((256, 64), (2, 1), True, False, 1)), (385, 640, ((256, 64), (2, 1), True, False, 1)), (641, 1279, ((256, 128), (2, 1), True, False, 1)), (1280, 1791, ((256, 192), (2, 1), False, False, 1)), (1792, 2303, ((256, 256), (2, 1), False, False, 1)), (2304, 2815, ((256, 192), (2, 4), False, False, 1)), (2816, 3583, ((256, 192), (2, 1), True, False, 1)), (3584, 4095, ((256, 256), (2, 1), False, False, 1))), (5120, 4096): ((17, 32, ((128, 16), (1, 1), True, False, 1)), (193, 384, ((256, 128), (2, 1), True, False, 1)), (385, 640, ((256, 192), (2, 1), False, False, 1)), (641, 895, ((256, 256), (2, 1), False, False, 1)), (896, 1279, ((256, 192), (2, 4), False, False, 1)), (1280, 1791, ((256, 256), (2, 4), False, False, 1)), (1792, 2303, ((256, 192), (2, 1), False, False, 1)), (2304, 2815, ((256, 256), (2, 1), False, False, 1)), (2816, 4095, ((256, 192), (2, 1), False, False, 1))), (16384, 1280): ((129, 256, ((128, 192), (1, 1), True, False, 1)), (257, 384, ((256, 192), (2, 2), True, False, 1)), (385, 895, ((256, 192), (2, 2), False, False, 1)), (896, 1791, ((256, 192), (2, 1), False, False, 1)), (1792, 4095, ((256, 256), (2, 1), False, False, 1))), (4096, 1280): ((640, 895, ((128, 192), (1, 1), True, False, 1)), (896, 1279, ((256, 128), (2, 1), False, False, 1)), (1280, 1791, ((256, 192), (2, 1), False, False, 1)), (1792, 2303, ((256, 128), (2, 2), False, False, 1)), (2304, 2815, ((256, 192), (2, 1), False, False, 1)), (2816, 3583, ((256, 192), (4, 1), True, False, 1)), (3584, 4095, ((256, 192), (2, 1), True, False, 1))), (5120, 1152): ((385, 640, ((256, 192), (2, 1), False, False, 1)), (641, 895, ((256, 128), (2, 1), False, False, 1)), (896, 1279, ((256, 192), (2, 1), False, False, 1)), (1280, 1791, ((256, 128), (2, 2), False, False, 1)), (1792, 2303, ((256, 192), (2, 1), True, False, 1)), (2304, 2815, ((256, 192), (2, 1), False, False, 1)), (2816, 3583, ((256, 192), (4, 1), True, False, 1)), (3584, 4095, ((256, 192), (2, 1), False, False, 1)))}

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

# lever SK (sk.py + sk_unit.cu): the mHC stats reduce + sinkhorn of our replicas' decode / verify sublayers of
# 1..SK_MAX_ROWS rows on a CUDA unit (partials in one round trip; stock's epilogue instruction for instruction); each
# row count's first eager call is compared with stock bit for bit (mismatch: stock for that row count)
SK = True
SK_MAX_ROWS = 127
# and the deep_gemm path (128+ rows: two tf32 prenorm GEMMs, reduce with the low part as residual) up to SK_DG_MAX_ROWS
SK_DG = True
SK_DG_MAX_ROWS = 512
SK_DG_ROWS_PER_CTA = 1          # 1 or 4 (the small path always takes 1 row a CTA)

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
# is a diagnostic only (FG on every N-th eligible forward, an A/B inside one engine), 0 in a shipped bundle
FG = True
FG_MIN_ROWS = 512
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
# copy and compares every byte (mismatch: stock's bytes, off for the process); a capture takes the unit only for keys
# verified eagerly. RO_TILES: (max rows, heads a thread, tokens a CTA, fixed 512 head stride), first match
# (tests/ro_test.py sweep, B300 cold-L2 CUDA graphs)
RO = True
RO_TILES = ((640, 1, 2, 0), (2048, 2, 4, 0), (1073741824, 8, 8, 0))

# lever FGP (fgp.py, gen_fgp.py): at the engine's first eligible eager EXTEND forward (any row count; the live replay's
# untimed warmup) lever FG's admission + warmups + both graph captures for every new bucket of its table and every FG
# layer, on deterministic synthetic static-buffer contents (FGP_SEED; FMT bands checked first through FMT's own
# check, without the all-reduce), largest bucket first, rank-agreed per bucket; buckets past the rank-agreed wall-clock
# budget FGP_BUDGET_S, refused or failing ones stay on FG's lazy path. A pre-captured bucket's first real use still
# replays, runs the eager path at the real rows and compares every byte before FG's rank agreement turns it on
FGP = True
FGP_BUDGET_S = 120.0
FGP_SEED = 20261015

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
