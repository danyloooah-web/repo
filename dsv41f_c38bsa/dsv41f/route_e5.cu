// route_e5 (fork-line `model` bundle dsv41f, cacheon/dsv41-tp2-upstream fork): the decode / target-verify MoE
// top-k of stock vision_topk AND the fork's E5 routed-expert prune as ONE kernel per MoE layer.
//
// Stock (fork) per decode / verify MoE layer: vision_topk -> moe_fused_gate's Triton router (one warp a row:
// sqrtsoftplus via log1p, + e_score_correction_bias (the image-token bias where input_ids == the image token), top-6 by
// the biased score, weights = unbiased scores / (sum + 1e-20) [x routed_scaling_factor], rows >= num_token_non_padded
// -> id -1 / weight 0), then deepseek_v4_prune.prune_ids (E5, one Triton CTA):
//   1. an expert picked by <= max_count rows of the call is eligible;
//   2. per row: mass = sum of its valid weights; a pick's run = sum of the eligible weights before it in (weight, slot)
//      order + its own weight; it is a candidate when eligible and run <= budget x mass;
//   3. an expert is dropped only when every row that picked it chose it as a candidate: id -1 / weight 0, the kept
//      weights are not renormalized.
// This unit: the top-k half is the arena line's lever RP / RF code verbatim (route_row, its exact PTX operations and the
// calibrated routed-sum tree); the rows land in ONE CTA's shared memory (its own or the cluster leader's) and that
// CTA runs E5 with thread = row: a shared-memory histogram of the picks, mass / run as fp32 add.rn along the reduction
// trees Triton's layout for the stock kernel's tile gives (runtime arguments: route_e5.py fits and verifies them at
// load against the stock kernel), the budget product as mul.rn, the comparisons as setp; then one store of ids / weights.
// Launch shapes, PDL and trigger semantics as lever RP's.
// Lever RE5NC (route_e5.py): above ROUTE_E5_FUSED_MAX_ROWS rows no cluster launch at all -- the top-k runs as stock's
// Triton router (mode 1) or as run_grid's plain multi-CTA grid of route_row (mode 2), then the one-CTA prune-only mode.
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// No torch headers: torch 2.14's C++ API needs C++20 and cacheon's build_cuda_ext compiles C++17. Python passes raw
// device pointers, sizes and the current stream (route_e5.py checks shapes / dtypes / contiguity there).

namespace route_e5 {

constexpr int kTopK = 6;
constexpr int kExperts = 384;               // DeepSeek-V4.1 routed experts
constexpr int kPerLane = 12;                // 384 / 32: lane l holds experts 12 l .. 12 l + 11
constexpr int kMaxRows = 256;               // E5's PRUNE_MAX_ROWS / one-program tile
constexpr int kMaxCluster = 16;
constexpr int kSumModes = 6;                // routed-sum trees: 0..4 routed_sum6, 5 = the runtime tree ktree
constexpr int kMaxLocal = kExperts;         // lever RR: local experts a record covers at most (host: <= kExperts)
constexpr int kBins = 512;                  // E5's histogram bins: expert id + 1 (bin 0 = no pick)
constexpr int kTreeSteps = 7;               // an 8-input fp32 reduction tree: 7 adds

#define RP_DEVICE __forceinline__ __device__

// ---- exact PTX ops ------------------------------------------------------------------------------------------------
RP_DEVICE float fma_rn_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
RP_DEVICE float fma_rm_ftz(float a, float b, float c) {
  float d;
  asm("fma.rm.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
RP_DEVICE float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
RP_DEVICE float add_rz_ftz(float a, float b) {
  float d;
  asm("add.rz.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
RP_DEVICE float mul_rn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
RP_DEVICE float neg_f32(float a) {
  float d;
  asm("neg.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
RP_DEVICE float sat_ftz(float a) {
  float d;
  asm("cvt.ftz.sat.f32.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
RP_DEVICE float ex2_approx_ftz(float a) {
  float d;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
RP_DEVICE float sqrt_approx_ftz(float a) {
  float d;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
RP_DEVICE float div_full(float a, float b) {
  float d;
  asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
RP_DEVICE float i2f_rn(int a) {
  float d;
  asm("cvt.rn.f32.s32 %0, %1;" : "=f"(d) : "r"(a));
  return d;
}
RP_DEVICE bool gt_f32(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.gt.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}
RP_DEVICE bool eq_f32(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.eq.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}
RP_DEVICE bool is_num(float a) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.num.f32 q, %1, %1;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a));
  return p != 0;
}
RP_DEVICE float bits(uint32_t b) { return __uint_as_float(b); }
// selp.f32: a branch-free select keeps the 12 per-lane score chains interleaved
RP_DEVICE float sel_f32(bool c, float a, float b) {
  float d;
  asm("{\n .reg .pred q;\n setp.ne.u32 q, %3, 0;\n selp.f32 %0, %1, %2, q;\n}"
      : "=f"(d) : "f"(a), "f"(b), "r"(uint32_t(c)));
  return d;
}
RP_DEVICE uint32_t warp_max_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}

RP_DEVICE void grid_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
RP_DEVICE void grid_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// ---- cluster primitives -------------------------------------------------------------------------------------------
RP_DEVICE uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
// start-up phase: arrive early (relaxed), wait right before the first distributed shared memory store -- every CTA of
// the cluster must have started before DSMEM is touched
RP_DEVICE void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
RP_DEVICE void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
RP_DEVICE void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
RP_DEVICE uint32_t mapa_leader(const void* local_smem_ptr) {
  const uint32_t local = static_cast<uint32_t>(__cvta_generic_to_shared(local_smem_ptr));
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(0u));
  return remote;
}
// predicated store (a plain `if` around the DSMEM store made ptxas fail register allocation in lev_R's cluster kernel)
RP_DEVICE void st_cluster_if(uint32_t addr, uint32_t value, uint32_t pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q st.shared::cluster.u32 [%0], %1;\n}"
               ::"r"(addr), "r"(value), "r"(pred) : "memory");
}

// ---- the stock router's per-expert math ----------------------------------------------------------------------------
// libdevice __nv_expf (nvvm-reflect-ftz = 1), instruction for instruction as Triton emits it.
RP_DEVICE float expf_tri(float x) {
  const float t = sat_ftz(fma_rn_ftz(x, bits(0x3BBB989Du), bits(0x3F000000u)));
  const float r = fma_rm_ftz(t, bits(0x437C0000u), bits(0x4B400001u));
  const float s = neg_f32(add_rn(r, bits(0xCB40007Fu)));
  float z = fma_rn_ftz(x, bits(0x3FB8AA3Bu), s);
  z = fma_rn_ftz(x, bits(0x32A57060u), z);
  const float scale = __uint_as_float(__float_as_uint(r) << 23);
  return mul_rn(ex2_approx_ftz(z), scale);
}

// libdevice __nv_log1pf (nvvm-reflect-ftz = 1), instruction for instruction as Triton emits it.
RP_DEVICE float log1pf_tri(float x) {
  const uint32_t xb = __float_as_uint(x);
  const float u = add_rz_ftz(x, 1.0f);
  const int32_t t = int32_t(__float_as_uint(u)) + int32_t(-1061158912);
  const int32_t e = t & int32_t(-8388608);
  const int32_t m = int32_t(xb) - e;
  const int32_t s = int32_t(1082130432) - e;
  const float a = fma_rn_ftz(bits(0x3E800000u), __int_as_float(s), bits(0xBF800000u));
  const float fr = add_rn(a, __int_as_float(m));
  const float ef = mul_rn(i2f_rn(e), bits(0x34000000u));
  float p = fma_rn_ftz(bits(0xBD39BF78u), fr, bits(0x3DD80012u));
  p = fma_rn_ftz(p, fr, bits(0xBE0778E0u));
  p = fma_rn_ftz(p, fr, bits(0x3E146475u));
  p = fma_rn_ftz(p, fr, bits(0xBE2A68DDu));
  p = fma_rn_ftz(p, fr, bits(0x3E4CAF9Eu));
  p = fma_rn_ftz(p, fr, bits(0xBE800042u));
  p = fma_rn_ftz(p, fr, bits(0x3EAAAAE6u));
  p = fma_rn_ftz(p, fr, bits(0xBF000000u));
  const float q = mul_rn(fr, p);
  const float rr = fma_rn_ftz(q, fr, fr);
  const float res = fma_rn_ftz(ef, bits(0x3F317218u), rr);
  // libdevice's special-value tail (inf / nan / negative inputs), as selects
  const bool special = !(xb < 2139095040u);
  const float big = fma_rn_ftz(x, bits(0x7F800000u), bits(0x7F800000u));
  float out = sel_f32(special && !(int32_t(xb) < int32_t(-1082130431)), big, res);
  out = sel_f32(special && eq_f32(x, 0.0f), bits(0x80000000u), out);
  return out;
}

// sqrtsoftplus (DeepSeek-V4.1 numerics): sqrt(x > 20 ? x : log1p(exp(x)))
RP_DEVICE float act_of(float x) {
  const float sp = sel_f32(gt_f32(x, 20.0f), x, log1pf_tri(expf_tri(x)));
  return sqrt_approx_ftz(sp);
}

// Order-preserving key of a biased score (NaN -> +inf as stock ranks it; -0 folded onto +0; key 0 never occurs, so 0
// marks a removed expert).
RP_DEVICE uint32_t key_of(float b) {
  b = sel_f32(is_num(b), b, bits(0x7F800000u));
  uint32_t u = __float_as_uint(b);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The routed-weight sum over the Triton kernel's 8 slots (6 weights + 2 zeros; + 0 is exact for these non-negative
// weights) in its reduction order -- in-thread sums in slot order, then a shuffle butterfly over the lanes:
//   0: 4 slots a lane                   (((w0 + w1) + w2) + w3) + (w4 + w5)
//   1: 8 slots a lane                   ((((w0 + w1) + w2) + w3) + w4) + w5
//   2: 1 or 2 a lane, xor 1 first       ((w0 + w1) + (w2 + w3)) + (w4 + w5)
//   3: 1 a lane, xor 4 first            ((w0 + w4) + w2) + ((w1 + w5) + w3)
//   4: 2 a lane, xor 2 first            ((w0 + w1) + (w4 + w5)) + (w2 + w3)
RP_DEVICE float routed_sum6(const float (&w)[kTopK], int mode) {
  if (mode == 3) return add_rn(add_rn(add_rn(w[0], w[4]), w[2]), add_rn(add_rn(w[1], w[5]), w[3]));
  if (mode == 4) return add_rn(add_rn(add_rn(w[0], w[1]), add_rn(w[4], w[5])), add_rn(w[2], w[3]));
  if (mode == 1) {
    float s = add_rn(w[0], w[1]);
    s = add_rn(s, w[2]);
    s = add_rn(s, w[3]);
    s = add_rn(s, w[4]);
    return add_rn(s, w[5]);
  }
  if (mode == 2) return add_rn(add_rn(add_rn(w[0], w[1]), add_rn(w[2], w[3])), add_rn(w[4], w[5]));
  float s = add_rn(w[0], w[1]);
  s = add_rn(s, w[2]);
  s = add_rn(s, w[3]);
  return add_rn(s, add_rn(w[4], w[5]));
}

struct Args {
  const float* logits;          // [M, 384] fp32, row stride `stride` (the gate output)
  int64_t stride;
  const float* bias;            // [384] fp32 e_score_correction_bias
  const float* bias_alt;        // [384] fp32 image-token bias (null: bias)
  const void* input_ids;        // [M] int32 / int64, or null (no token bias)
  const void* ntnp;             // [1] int32 / int64 num_token_non_padded, or null
  float* weights;               // [M, 6] fp32 out
  int32_t* ids;                 // [M, 6] int32 out
  long long* clocks;            // [8] debug: leader thread 0's clock64 at the phase boundaries, or null
  const int32_t* ids_in;        // prune-only mode (load-time self-test): [M, 6] given top-k ids / weights, no routing
  const float* w_in;
  int rows;
  int rows_per_cta;             // cluster launch: rows of CTA r are [r * rows_per_cta, min(rows, (r + 1) * rows_per_cta))
  int image_token;
  int ids64;
  int ntnp64;
  int sum_mode;
  int apply_scale;
  int prune;                    // 0: top-k only (calibration)
  int trigger;                  // PDL: 0 at grid exit, 1 after the stores + fence, 2 before the stores
  float scale;                  // routed_scaling_factor
  float eps;                    // renormalization epsilon (stock 1e-20)
  float budget;                 // E5 PRUNE_BUDGET (fp32, as the stock launch passes it)
  int max_count;                // E5 PRUNE_MAX_COUNT
  // lever RR: FlashInfer's trtllm-gen routing record of the final ids (record_build), or rr_on = 0
  int32_t* rr_map;              // [rows * 6] expanded -> permuted (-1: dropped / remote pick)
  int32_t* rr_tok;              // [>= max padded + 1] permuted -> token (-1: tile padding)
  int32_t* rr_batch;            // [>= max CTAs] CTA -> local expert
  int32_t* rr_limit;            // [>= max CTAs] CTA -> mn limit
  int32_t* rr_padded;           // [1] total_num_padded_tokens
  int32_t* rr_nonexit;          // [1] num_non_exiting_ctas
  int rr_on;
  int rr_tile_log2;             // body tile N = 1 << rr_tile_log2
  int rr_offset;                // first local expert (global id)
  int rr_num_local;             // local experts
  uint8_t mtree[2 * kTreeSteps];   // `mass` reduction tree: step s writes r[8 + s] = r[a] + r[b]; result r[14]
  uint8_t rtree[2 * kTreeSteps];   // `run` reduction tree (over i of the before-masked weights)
  uint8_t ktree[2 * kTreeSteps];   // top-k routed-sum tree over the 8 padded slots (sum_mode 5)
  int8_t mtree_id;              // fixed-tree ids of mtree / rtree / ktree (-1: the runtime steps)
  int8_t rtree_id;
  int8_t ktree_id;
  // (appended last: the fields above keep their offsets)
  unsigned int* ticket;         // lever RE5NC mode 3: the grid's arrival counter (0 between launches), or null
  // lever RTP (route_rtp_kernel): the layer's exchange table [2][kMaxRows][2] epoch-tagged 64-bit words and its epoch
  unsigned long long* rtp_table;
  unsigned int* rtp_epoch;
  unsigned long long* rtp_clocks;   // debug: CTA 0 thread 0's %globaltimer at the phase boundaries, or null
};

// An 8-input fp32 sum along a reduction tree (each step one add.rn of two earlier values).
// The reduction trees in use are three fixed step tables (fork/tests/fit_e5.py's layouts (1,1,4), (1,4,2), (1,2,4) and
// the calibrated top-k tree); route_e5.py passes their ids. A fixed tree is 7 register adds with compile-time operands
// (no select network, no indexed array: this one-CTA kernel runs from a cold instruction cache on every launch, so
// code size is what costs). Any other tree takes tree8_rt, out of line. The same add.rn ops in the same order.
constexpr int kFixedTrees = 3;
RP_DEVICE constexpr int fixed_step(int id, int k) {
  constexpr uint8_t t0[14] = {0, 4, 2, 6, 8, 9, 1, 5, 3, 7, 11, 12, 10, 13};   // (1, 1, 4, seq, desc, k)
  constexpr uint8_t t1[14] = {0, 2, 1, 3, 8, 9, 4, 6, 5, 7, 11, 12, 10, 13};   // (1, 4, 2, seq, desc, k)
  constexpr uint8_t t2[14] = {0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 11, 12, 10, 13};   // (1, 2, 4, seq, desc, k)
  return id == 0 ? t0[k] : id == 1 ? t1[k] : t2[k];
}
template <int ID>
RP_DEVICE float tree8_fixed(const float (&v)[8]) {
  float r[8 + kTreeSteps];
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = v[i];
#pragma unroll
  for (int s = 0; s < kTreeSteps; ++s) r[8 + s] = add_rn(r[fixed_step(ID, 2 * s)], r[fixed_step(ID, 2 * s + 1)]);
  return r[8 + kTreeSteps - 1];
}
// The runtime-steps fallback takes its 8 inputs by value: passing the caller's array by address put every tree8
// caller's array in local memory (8 STL + LDL per call on the fixed path too: ~400 cycles a tree, 1.6 us of E5).
__device__ __noinline__ float tree8_rt(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
                                       const uint8_t* steps) {
  float r[8 + kTreeSteps] = {v0, v1, v2, v3, v4, v5, v6, v7};
  for (int s = 0; s < kTreeSteps; ++s) r[8 + s] = add_rn(r[steps[2 * s]], r[steps[2 * s + 1]]);
  return r[8 + kTreeSteps - 1];
}
RP_DEVICE float tree8(const float (&v)[8], const uint8_t* steps, int id) {
  switch (id) {
    case 0: return tree8_fixed<0>(v);
    case 1: return tree8_fixed<1>(v);
    case 2: return tree8_fixed<2>(v);
    default: return tree8_rt(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], steps);
  }
}

// The top-k's routed sum: one of route_prune's five fixed trees, or (mode 5) the calibrated runtime tree over the
// Triton router's [1, 8] slot tensor (6 weights + 2 zero pads).
RP_DEVICE float routed_sum(const float (&w)[kTopK], const Args& a) {
  if (a.sum_mode == 5) {
    const float v[8] = {w[0], w[1], w[2], w[3], w[4], w[5], 0.0f, 0.0f};
    return tree8(v, a.ktree, a.ktree_id);
  }
  return routed_sum6(w, a.sum_mode);
}

// Lever RF: route_row's per-warp scratch (one row's lane maxima and top-6 candidates; its winners in slot order).
struct alignas(16) RowScratch {
  uint32_t lmax[32];            // lane maxima of the row's keys
  uint32_t ckey[32];            // candidates (key >= the 6th largest lane maximum): key, expert id, activation
  int32_t cid[32];
  float cact[32];
  float cx[32];                 // (activation filter) the candidates' logits and biases
  float cbias[32];
  float win_w[8];               // the six winners in slot order
  int32_t win_id[8];
};

RP_DEVICE long long clock_now() {
  long long c;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)::"memory");
  return c;
}
RP_DEVICE void stamp(const Args& a, int i) {
  if (a.clocks != nullptr && threadIdx.x == 0 && cluster_ctarank() == 0) a.clocks[i] = clock_now();   // lever RF: leader only
}

RP_DEVICE int64_t live_rows(const Args& a) {
  int64_t live = a.rows;
  if (a.ntnp != nullptr) {
    const int64_t n = a.ntnp64 ? *static_cast<const int64_t*>(a.ntnp) : int64_t(*static_cast<const int32_t*>(a.ntnp));
    live = n < live ? n : live;
  }
  return live;
}

RP_DEVICE bool row_alt(const Args& a, int row) {
  if (a.input_ids == nullptr) return false;
  const int64_t tok = a.ids64 ? static_cast<const int64_t*>(a.input_ids)[row]
                              : int64_t(static_cast<const int32_t*>(a.input_ids)[row]);
  return tok == int64_t(a.image_token);
}

RP_DEVICE void load_bias(const Args& a, int lane, float (&b)[kPerLane], float (&alt)[kPerLane]) {
#pragma unroll
  for (int q = 0; q < 3; ++q) {
    const float4 v = *reinterpret_cast<const float4*>(a.bias + kPerLane * lane + 4 * q);
    b[4 * q] = v.x;
    b[4 * q + 1] = v.y;
    b[4 * q + 2] = v.z;
    b[4 * q + 3] = v.w;
    const float4 u = a.bias_alt != nullptr ? *reinterpret_cast<const float4*>(a.bias_alt + kPerLane * lane + 4 * q) : v;
    alt[4 * q] = u.x;
    alt[4 * q + 1] = u.y;
    alt[4 * q + 2] = u.z;
    alt[4 * q + 3] = u.w;
  }
}

// Lever RF: predicated stores (a plain `if` around a store becomes a branch, ~20 cycles of a single warp's latency)
// and fp32 ops the compiler must not sink into branches (the same instructions as prune_core's f32_add / f32_gt).
// (no "memory" clobber: the candidate / winner lists are read only after a __syncwarp, which orders them; the clobber
// made each of route_row's 36 list stores a full compiler barrier)
RP_DEVICE void st_shared_if(const void* p, uint32_t v, bool c) {
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q st.shared.u32 [%0], %1;\n}"
               ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))), "r"(v), "r"(uint32_t(c)));
}
RP_DEVICE void st_shared64_if(const void* p, uint64_t v, bool c) {
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q st.shared.u64 [%0], %1;\n}"
               ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))), "l"(v), "r"(uint32_t(c)) : "memory");
}
RP_DEVICE void st_global_if(const void* p, uint32_t v, bool c) {
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q st.global.u32 [%0], %1;\n}"
               ::"l"(p), "r"(v), "r"(uint32_t(c)) : "memory");
}
RP_DEVICE float add_rn_v(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
RP_DEVICE bool gt_f32_v(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.gt.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}

RP_DEVICE float add_rm(float a, float b) {   // (no .ftz: a directed bound must not flush)
  float d;
  asm("add.rm.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
RP_DEVICE float add_rp(float a, float b) {
  float d;
  asm("add.rp.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
RP_DEVICE bool lt_f32(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.lt.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}

// Lever RF: the activation filter. act_fast(x) is act_of(x) to within kActDelta for EVERY fp32 x -- NaN exactly where
// act_of is NaN, +inf exactly where it is +inf, and x > 20 takes act_of's own path (equal) -- checked on B300 over all
// 2^32 inputs (tests/b300_rf.py (e)). ~10 instructions instead of act_of's ~45.
constexpr float kActDelta = 6.103515625e-05f;   // 2^-14
RP_DEVICE float act_fast(float x) {
  float e2, l2;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e2) : "f"(mul_rn(x, bits(0x3FB8AA3Bu))));        // 2^(x log2 e)
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(l2) : "f"(add_rn(e2, 1.0f)));
  const float sp = sel_f32(lt_f32(e2, bits(0x3A800000u)), e2, mul_rn(l2, bits(0x3F317218u)));   // e2 < 2^-10: ~e2
  return sqrt_approx_ftz(sel_f32(gt_f32(x, 20.0f), x, sp));
}
// The order-preserving image of a bound (NaN -> +inf's image as key_of; no -0 folding: a lower bound of -0 only lowers
// the threshold, an upper bound is never -0 -- add.rp of a positive and anything is +0 at zero).
RP_DEVICE uint32_t bound_key(float v) {
  const uint32_t u = __float_as_uint(sel_f32(is_num(v), v, bits(0x7F800000u)));
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Lever RP's selection, kept as lever RF's exact fallback: six rounds of local max (lower index on equal keys), warp
// max, lowest lane holding it (= lowest expert id); w / sel in slot (= rank) order.
RP_DEVICE void select_rounds(int lane, const uint32_t (&key)[kPerLane], const float (&act)[kPerLane],
                             float (&w)[kTopK], int32_t (&sel)[kTopK]) {
  uint32_t removed = 0;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    // (the winner's activation rides along the tree: a select chain over act[] by index would be turned into a
    // local-memory array access)
    uint32_t tk[kPerLane];
    int ti[kPerLane];
    float ta[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      tk[i] = ((removed >> i) & 1u) ? 0u : key[i];
      ti[i] = i;
      ta[i] = act[i];
    }
#pragma unroll
    for (int span = 1; span < kPerLane; span <<= 1) {
#pragma unroll
      for (int i = 0; i + span < kPerLane; i += 2 * span) {
        const bool right = tk[i + span] > tk[i];
        tk[i] = right ? tk[i + span] : tk[i];
        ti[i] = right ? ti[i + span] : ti[i];
        ta[i] = sel_f32(right, ta[i + span], ta[i]);
      }
    }
    const uint32_t mx = warp_max_u32(tk[0]);
    const uint32_t ballot = __ballot_sync(0xffffffffu, tk[0] == mx);
    const int owner = __ffs(ballot) - 1;
    w[k] = __shfl_sync(0xffffffffu, ta[0], owner);
    sel[k] = int32_t(kPerLane * owner + __shfl_sync(0xffffffffu, ti[0], owner));
    if (lane == owner) removed |= 1u << ti[0];
  }
}

// Lever RF: the exact fallback (every expert's act_of + key, lever RP's six rounds) as a separate, never inlined
// function: on a cold instruction stream (code evicted from L2 by the layer's GEMMs) a taken branch over ~16 KB of code
// costs ~1,100 cycles on B300, so rare paths live out of line.
__device__ __noinline__ void select_exact_ool(int lane, const float* x, const float* bias, float* w, int32_t* sel) {
  uint32_t k[kPerLane];
  float ac[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    ac[i] = act_of(x[i]);
    k[i] = key_of(add_rn(ac[i], bias[i]));
  }
  float wl[kTopK];
  int32_t sl[kTopK];
  select_rounds(lane, k, ac, wl, sl);
#pragma unroll
  for (int i = 0; i < kTopK; ++i) {
    w[i] = wl[i];
    sel[i] = sl[i];
  }
}

// One warp, one row: the stock top-6 + weights. Lanes 0..5 return slot `lane` (id, weight); padded rows -1 / 0.
// Lever RF: a padded row (row >= num_token_non_padded: warp-uniform) is not computed (stock's -1 / 0 either way). Each
// expert gets bounds lo <= key <= hi of its exact key key_of(add_rn(act_of(x), bias)) from act_fast (|act_of - act_fast|
// <= kActDelta; the bias added rounding down / up: lo = rd(rd(a - d) + b), hi = ru(ru(a + d) + b)). lim = the 6th
// largest lane maximum of lo: >= 6 distinct experts have key >= lo >= lim, so the 6th largest key is >= lim, and every
// expert with key >= lim -- every winner, and every expert ranked above one -- has hi >= lim: the candidates. They are
// listed through 12 ballots (local index, then lane), one a lane, get their exact act_of / key, and a candidate's slot
// is the number of candidates with a larger key or an equal key and a lower expert id (lever RP's order). More than 32
// candidates (ties / NaN rows): select_exact_ool. The same ids and activations in the same slots, the same sum.
RP_DEVICE void route_row(const Args& a, int row, int64_t live, int lane, const float (&b)[kPerLane],
                         const float (&alt)[kPerLane], RowScratch& sc, int32_t& out_id, float& out_w) {
  const float* src = a.logits + int64_t(row) * a.stride + kPerLane * lane;
  float x[kPerLane];
#pragma unroll
  for (int q = 0; q < 3; ++q) {
    const float4 v = *reinterpret_cast<const float4*>(src + 4 * q);
    x[4 * q] = v.x;
    x[4 * q + 1] = v.y;
    x[4 * q + 2] = v.z;
    x[4 * q + 3] = v.w;
  }
  const bool use_alt = row_alt(a, row);   // (loaded with the logits, before the padded-row test)
  if (int64_t(row) >= live) {
    out_id = -1;
    out_w = 0.0f;
    return;
  }
  float bs[kPerLane];
  uint32_t hi[kPerLane];
  uint32_t m = 0;   // lane maximum of the lower bounds
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    bs[i] = use_alt ? alt[i] : b[i];
    const float af = act_fast(x[i]);
    m = max(m, bound_key(add_rm(add_rm(af, -kActDelta), bs[i])));
    hi[i] = bound_key(add_rp(add_rp(af, kActDelta), bs[i]));
  }
  sc.lmax[lane] = m;
  __syncwarp();
  int above = 0;   // lanes with a larger maximum
#pragma unroll
  for (int j = 0; j < 32; j += 4) {
    const uint4 q = *reinterpret_cast<const uint4*>(&sc.lmax[j]);
    above += (q.x > m ? 1 : 0) + (q.y > m ? 1 : 0) + (q.z > m ? 1 : 0) + (q.w > m ? 1 : 0);
  }
  const uint32_t lim = __reduce_min_sync(0xffffffffu, above < kTopK ? m : 0xffffffffu);
  const uint32_t lt = (1u << lane) - 1u;
  int total = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    const bool c = hi[i] >= lim;
    const uint32_t bal = __ballot_sync(0xffffffffu, c);
    const int p = total + __popc(bal & lt);
    const bool put = c & (p < 32);
    st_shared_if(&sc.cx[p & 31], __float_as_uint(x[i]), put);
    st_shared_if(&sc.cbias[p & 31], __float_as_uint(bs[i]), put);
    st_shared_if(&sc.cid[p & 31], uint32_t(kPerLane * lane + i), put);
    total += __popc(bal);
  }
  float w[kTopK];
  int32_t ik;
  if (total <= 32) {
    __syncwarp();
    // candidate `lane`: its exact activation and key (lanes past the list compute garbage nobody reads)
    const float ca = act_of(sc.cx[lane]);
    const uint32_t mine = key_of(add_rn(ca, sc.cbias[lane]));
    const int me = sc.cid[lane];
    sc.ckey[lane] = mine;
    __syncwarp();
    int r = 0;
    for (int j = 0; j < total; j += 8) {   // (typically one pass: <= 8 candidates)
#pragma unroll
      for (int h = 0; h < 8; h += 4) {
        const uint4 k4 = *reinterpret_cast<const uint4*>(&sc.ckey[j + h]);
        const int4 i4 = *reinterpret_cast<const int4*>(&sc.cid[j + h]);
        // (bitwise & / |: no short-circuit branches)
        r += int(j + h < total) & (int(k4.x > mine) | (int(k4.x == mine) & int(i4.x < me)));
        r += int(j + h + 1 < total) & (int(k4.y > mine) | (int(k4.y == mine) & int(i4.y < me)));
        r += int(j + h + 2 < total) & (int(k4.z > mine) | (int(k4.z == mine) & int(i4.z < me)));
        r += int(j + h + 3 < total) & (int(k4.w > mine) | (int(k4.w == mine) & int(i4.w < me)));
      }
    }
    const bool win = (lane < total) & (r < kTopK);
    st_shared_if(&sc.win_w[r & 7], __float_as_uint(ca), win);
    st_shared_if(&sc.win_id[r & 7], uint32_t(me), win);
    __syncwarp();
    const float4 wa = *reinterpret_cast<const float4*>(&sc.win_w[0]);
    const float2 wb = *reinterpret_cast<const float2*>(&sc.win_w[4]);
    w[0] = wa.x;
    w[1] = wa.y;
    w[2] = wa.z;
    w[3] = wa.w;
    w[4] = wb.x;
    w[5] = wb.y;
    ik = sc.win_id[lane < kTopK ? lane : 0];
    __syncwarp();   // the warp's next row reuses the scratch
  } else {   // (out of line)
    float xc[kPerLane];
    float bc[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      xc[i] = x[i];
      bc[i] = bs[i];
    }
    float wc[kTopK];
    int32_t sc6[kTopK];
    select_exact_ool(lane, xc, bc, wc, sc6);
    ik = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      w[k] = wc[k];
      ik = (k == lane) ? sc6[k] : ik;
    }
  }
  const float denom = add_rn(routed_sum(w, a), a.eps);   // route_e5: + the runtime tree
  float wk = 0.0f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) wk = (k == lane) ? w[k] : wk;
  float out = div_full(wk, denom);
  if (a.apply_scale) out = mul_rn(a.scale, out);
  out_w = out;
  out_id = ik;
}


template <int kThreads>
struct Shared {
  int32_t id[kThreads * kTopK];
  float w[kThreads * kTopK];
  int32_t cnt[kBins];           // E5 step 1: picks per expert (bin id + 1)
  int32_t cnd[kBins];           // E5 step 3: candidate picks per expert
  RowScratch rs[kThreads / 32]; // lever RF: one a warp
  int32_t wsum[kThreads / 32];  // lever RR: the record's block scan
  uint32_t rmask[kMaxLocal * ((kThreads + 31) / 32)];   // lever RR: per local expert, its rows' bits (zeroed at start)
};

RP_DEVICE bool lt_f32_e(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.lt.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}
RP_DEVICE bool le_f32_e(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.le.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}
RP_DEVICE bool eq_f32_e(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.eq.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}

// Lever RR: FlashInfer 0.7.1's trtllm-gen routing record for the call's final ids (thread t = row t, every thread of
// the block calls it once E5 has decided), the arrays routingIndicesDynBlockKernel writes for UnpackedPrecomputed ids
// (fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cuh) at tile N = 1 << tile_log2:
//   count[le] = rows holding a pick of local expert le; ncta[le] = ceil(count / N); off[le] = exclusive prefix of ncta
//   over the local experts in id order; batch[off + j] = le; limit[off + j] = min((off + j + 1) N, off N + count);
//   tok[limit .. (off + j + 1) N) = -1 (initRoutingTilePadding); padded = N sum(ncta); nonexit = sum(ncta);
//   map[t * 6 + k] = off[le] N + rank of row t among le's rows (rows ascending), or -1 (pruned / remote pick);
//   tok[map[t * 6 + k]] = t.
// The routed weights are the caller's (unpacked mode: FlashInfer's record borrows them). Shared memory: the rows'
// ids / weights (dead here: every thread holds its row in registers) carry the per-expert row masks, cnt / cnd the
// per-expert counts / CTA offsets, wsum the scan.
template <int kThreads>
RP_DEVICE void record_build(const Args& a, Shared<kThreads>& s, const int32_t (&id)[kTopK]) {
  constexpr int kWarps = kThreads / 32;
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  const int rows = a.rows;
  const int words = (rows + 31) >> 5;
  const int sh = a.rr_tile_log2;
  const int nl = a.rr_num_local;
  const bool live = t < rows;
  uint32_t* mask = s.rmask;     // zeroed at kernel start (nl * words words, before the PDL wait)
  int32_t* cnt = s.cnt;
  int32_t* off = s.cnd;
  if (live) {
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const unsigned le = unsigned(id[k] - a.rr_offset);
      if (id[k] >= 0 && le < unsigned(nl)) atomicOr(&mask[le * words + warp], 1u << lane);
    }
  }
  __syncthreads();
  const int per = (nl + kThreads - 1) / kThreads;
  const int e0 = min(t * per, nl);
  const int e1 = min(e0 + per, nl);
  int mine = 0;
#pragma unroll 1
  for (int le = e0; le < e1; ++le) {
    int c = 0;
#pragma unroll 1
    for (int w = 0; w < words; ++w) c += __popc(mask[le * words + w]);
    cnt[le] = c;
    mine += (c + (1 << sh) - 1) >> sh;
  }
  stamp(a, 6);
  int incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  int run = incl - mine;
  int total = __shfl_sync(0xffffffffu, incl, 31);
  if constexpr (kWarps > 1) {
    if (lane == 31) s.wsum[warp] = incl;
    __syncthreads();
    int before = 0;
    total = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const int x = s.wsum[w];
      before += w < warp ? x : 0;
      total += x;
    }
    run += before;
  }
#pragma unroll 1
  for (int le = e0; le < e1; ++le) {
    const int c = cnt[le];
    const int n = (c + (1 << sh) - 1) >> sh;
    off[le] = run;
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
      a.rr_batch[run + q] = le;
      a.rr_limit[run + q] = min((run + q + 1) << sh, (run << sh) + c);
    }
    run += n;
  }
  if (t == 0) {
    a.rr_padded[0] = total << sh;
    a.rr_nonexit[0] = total;
  }
  // tile padding: every permuted slot of [0, padded) starts at -1 (coalesced, block-wide); the map phase below writes
  // the live slots' tokens after the barrier, leaving -1 exactly on [limit, tile end) of each CTA (initRoutingTilePadding)
#pragma unroll 1
  for (int r = t; r < (total << sh); r += kThreads) a.rr_tok[r] = -1;
  stamp(a, 7);
  __syncthreads();
  if (live) {
    const uint32_t below = (1u << lane) - 1u;
    int out[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const unsigned le = unsigned(id[k] - a.rr_offset);
      out[k] = -1;
      if (id[k] >= 0 && le < unsigned(nl)) {
        const uint32_t* m = mask + le * words;
        int rank = __popc(m[warp] & below);
#pragma unroll 1
        for (int w = 0; w < warp; ++w) rank += __popc(m[w]);
        out[k] = (off[le] << sh) + rank;
        a.rr_tok[out[k]] = t;
      }
    }
    int2* mp = reinterpret_cast<int2*>(a.rr_map + t * kTopK);
#pragma unroll
    for (int k = 0; k < kTopK / 2; ++k) mp[k] = make_int2(out[2 * k], out[2 * k + 1]);
  }
}

// E5 (thread = row, rows in shared memory) and the single store of the results.
template <int kThreads>
RP_DEVICE void e5_and_store(const Args& a, Shared<kThreads>& s) {
  const int t = threadIdx.x;
  const bool mine = t < a.rows;
  int id[kTopK];
  float w[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    id[k] = mine ? s.id[t * kTopK + k] : -1;
    w[k] = mine ? s.w[t * kTopK + k] : 0.0f;
  }
  if (a.prune) {
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
      if (id[k] >= 0) atomicAdd(&s.cnt[id[k] + 1], 1);
    __syncthreads();
    bool elig[kTopK];
    int cnt[kTopK];
    float mv[8];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const bool valid = id[k] >= 0;
      cnt[k] = valid ? s.cnt[id[k] + 1] : 0;
      elig[k] = valid && cnt[k] <= a.max_count;
      mv[k] = valid ? w[k] : 0.0f;
    }
    mv[6] = 0.0f;
    mv[7] = 0.0f;
    const float mass = tree8(mv, a.mtree, a.mtree_id);
    float thr;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(thr) : "f"(a.budget), "f"(mass));
    bool cand[kTopK];
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
      float tv[8];
#pragma unroll
      for (int i = 0; i < kTopK; ++i) {
        const bool before = elig[i] && (lt_f32_e(w[i], w[j]) || (eq_f32_e(w[i], w[j]) && i < j));
        tv[i] = before ? w[i] : 0.0f;
      }
      tv[6] = 0.0f;
      tv[7] = 0.0f;
      const float run = add_rn_v(tree8(tv, a.rtree, a.rtree_id), w[j]);
      cand[j] = elig[j] && le_f32_e(run, thr);
      if (cand[j]) atomicAdd(&s.cnd[id[j] + 1], 1);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
      if (cand[j] && s.cnd[id[j] + 1] == cnt[j]) {
        id[j] = -1;
        w[j] = 0.0f;
      }
    }
  }
  if (a.rr_on) record_build<kThreads>(a, s, id);   // lever RR (uniform: every thread calls it)
  stamp(a, 4);
  if (a.trigger == 2) grid_launch_dependents();
  if (mine) {
    int2* ip = reinterpret_cast<int2*>(a.ids + t * kTopK);
    float2* wp = reinterpret_cast<float2*>(a.weights + t * kTopK);
#pragma unroll
    for (int k = 0; k < kTopK / 2; ++k) {
      ip[k] = make_int2(id[2 * k], id[2 * k + 1]);
      wp[k] = make_float2(w[2 * k], w[2 * k + 1]);
    }
  }
  if (a.trigger == 1) {
    __threadfence();
    __syncthreads();
    grid_launch_dependents();
  }
  stamp(a, 5);
}

// kCluster = false: one CTA, warp w routes rows w, w + W, ... (W = kThreads / 32); kCluster = true: CTA r of the
// cluster routes its rows_per_cta rows into the leader's (rank 0) shared memory, and the leader runs E5.
template <int kThreads, bool kCluster>
__global__ void __launch_bounds__(kThreads, 1) route_e5_kernel(const Args a) {
  constexpr int kWarps = kThreads / 32;
  __shared__ __align__(16) Shared<kThreads> s;
  stamp(a, 0);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  uint32_t crank = 0;
  if constexpr (kCluster) crank = cluster_ctarank();
  if (crank == 0 && a.prune) {             // before the wait: input independent
    for (int e = threadIdx.x; e < kBins; e += kThreads) {
      s.cnt[e] = 0;
      s.cnd[e] = 0;
    }
  }
  if (crank == 0 && a.rr_on) {             // lever RR: the record's masks, before the wait too
    const int n = a.rr_num_local * ((a.rows + 31) >> 5);
    for (int i = threadIdx.x; i < n; i += kThreads) s.rmask[i] = 0u;
  }
  float b[kPerLane], alt[kPerLane];
  load_bias(a, lane, b, alt);                                          // parameters: safe before the wait
  if constexpr (kCluster) cluster_arrive_relaxed();
  grid_wait();
  stamp(a, 1);
  const int64_t live = live_rows(a);
  const int first = kCluster ? int(crank) * a.rows_per_cta : 0;
  const int end = kCluster ? min(first + a.rows_per_cta, a.rows) : a.rows;
  if (a.ids_in != nullptr) {             // prune-only mode (one CTA): the given rows instead of the top-k
    for (int i = threadIdx.x; i < a.rows * kTopK; i += kThreads) {
      s.id[i] = a.ids_in[i];
      s.w[i] = a.w_in[i];
    }
    __syncthreads();
    e5_and_store<kThreads>(a, s);
    return;
  }
  if constexpr (kCluster) {
    const uint32_t id_base = mapa_leader(&s.id[0]);
    const uint32_t w_base = mapa_leader(&s.w[0]);
    cluster_wait();             // every CTA of the cluster has started: the leader's shared memory may be written
    for (int row = first + warp; row < end; row += kWarps) {
      int32_t oid;
      float ow;
      route_row(a, row, live, lane, b, alt, s.rs[warp], oid, ow);
      const uint32_t slot = uint32_t(row * kTopK + (lane < kTopK ? lane : 0)) * 4u;
      const uint32_t keep = lane < kTopK ? 1u : 0u;
      st_cluster_if(id_base + slot, uint32_t(oid), keep);
      st_cluster_if(w_base + slot, __float_as_uint(ow), keep);
    }
    stamp(a, 2);
    cluster_sync_all();         // release / acquire: every row is in the leader's shared memory
    if (crank != 0) return;
    __syncthreads();
    stamp(a, 3);
  } else {
    for (int row = warp; row < end; row += kWarps) {
      int32_t oid;
      float ow;
      route_row(a, row, live, lane, b, alt, s.rs[warp], oid, ow);
      if (lane < kTopK) {
        s.id[row * kTopK + lane] = oid;
        s.w[row * kTopK + lane] = ow;
      }
    }
    stamp(a, 2);
    __syncthreads();
    stamp(a, 3);
  }
  e5_and_store<kThreads>(a, s);
}

// Lever RE5NC (step 2): the top-k alone on a plain multi-CTA grid -- no cluster, so its CTAs need single free SMs, not a
// whole GPC beside the shared-expert / mHC side kernels. CTA b routes rows [b * rows_per_cta, ...) (warp w: rows
// first + w, first + w + W, ...) with route_row verbatim and stores its lanes 0..5 straight to ids / weights; RE5's
// prune-only mode (one CTA) is its PDL dependent and reads those rows with plain (coherent) loads after
// griddepcontrol.wait. trigger 1: launch_dependents after the stores (dependents never read before the grid completes:
// griddepcontrol.wait waits for completion and the memory flush either way).
template <int kThreads>
__global__ void __launch_bounds__(kThreads, 1) route_topk_grid_kernel(const Args a) {
  constexpr int kWarps = kThreads / 32;
  __shared__ __align__(16) RowScratch rs[kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float b[kPerLane], alt[kPerLane];
  load_bias(a, lane, b, alt);                                          // parameters: safe before the wait
  grid_wait();
  const int64_t live = live_rows(a);
  const int first = int(blockIdx.x) * a.rows_per_cta;
  const int end = min(first + a.rows_per_cta, a.rows);
  for (int row = first + warp; row < end; row += kWarps) {
    int32_t oid;
    float ow;
    route_row(a, row, live, lane, b, alt, rs[warp], oid, ow);
    const bool keep = lane < kTopK;
    const int slot = row * kTopK + (keep ? lane : 0);
    st_global_if(a.ids + slot, uint32_t(oid), keep);
    st_global_if(a.weights + slot, __float_as_uint(ow), keep);
  }
  if (a.trigger == 1) {
    __threadfence();
    __syncthreads();
    grid_launch_dependents();
  }
}

// Lever RE5NC mode 3: run_grid's top-k AND E5 (+ record) in ONE plain-grid launch -- every CTA routes its rows
// (route_row verbatim) and stores them to ids / weights, then takes a ticket (CTA-flag pattern: barrier, thread 0
// __threadfence + atomicAdd); the CTA drawing the last ticket fences, loads every row through L2 (ld.global.cg:
// coherent with the other CTAs' stores), resets the counter for the next launch and runs E5 + the single store exactly
// as the fused kernel's leader (e5_and_store). No cluster and no second launch.
template <int kThreads>
__global__ void __launch_bounds__(kThreads, 1) route_nc_last_kernel(const Args a) {
  constexpr int kWarps = kThreads / 32;
  __shared__ __align__(16) Shared<kThreads> s;
  __shared__ int is_last;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int e = threadIdx.x; e < kBins; e += kThreads) {    // before the wait: input independent (any CTA may be last)
    s.cnt[e] = 0;
    s.cnd[e] = 0;
  }
  if (a.rr_on) {
    const int n = a.rr_num_local * ((a.rows + 31) >> 5);
    for (int i = threadIdx.x; i < n; i += kThreads) s.rmask[i] = 0u;
  }
  float b[kPerLane], alt[kPerLane];
  load_bias(a, lane, b, alt);                                          // parameters: safe before the wait
  grid_wait();
  const int64_t live = live_rows(a);
  const int first = int(blockIdx.x) * a.rows_per_cta;
  const int end = min(first + a.rows_per_cta, a.rows);
  for (int row = first + warp; row < end; row += kWarps) {
    int32_t oid;
    float ow;
    route_row(a, row, live, lane, b, alt, s.rs[warp], oid, ow);
    const bool keep = lane < kTopK;
    const int slot = row * kTopK + (keep ? lane : 0);
    st_global_if(a.ids + slot, uint32_t(oid), keep);
    st_global_if(a.weights + slot, __float_as_uint(ow), keep);
  }
  // the CTA-flag pattern (CUTLASS semaphore): barrier -> thread 0 fence (release, cumulative over the CTA's stores) ->
  // ticket; the last CTA's thread 0: ticket -> fence (acquire) -> barrier -> every thread's loads
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const int last = atomicAdd(a.ticket, 1u) == gridDim.x - 1u ? 1 : 0;
    if (last) __threadfence();
    is_last = last;
  }
  __syncthreads();
  if (!is_last) return;
  for (int i = threadIdx.x; i < a.rows * kTopK; i += kThreads) {
    s.id[i] = __ldcg(a.ids + i);
    s.w[i] = __ldcg(a.weights + i);
  }
  if (threadIdx.x == 0) *a.ticket = 0u;   // every CTA has drawn: the next launch (stream-ordered) starts from 0
  __syncthreads();
  e5_and_store<kThreads>(a, s);
}

// ---- lever RTP: the route window as ONE plain grid, one row a warp, rows exchanged through an epoch table --------
// The same arithmetic as route_e5_kernel / route_nc_last_kernel (route_row verbatim; e5_and_store's E5 operations;
// record_build's record), only the decomposition differs:
//   1. warp w of CTA b routes row r = b * W + w (route_row, or the given ids / weights in the prune-only test mode);
//   2. the row's 6 ids are published as two 64-bit words (tag << 32 | 3 x 9-bit (id + 1)) in table 1 -- tag = the
//      layer's epoch + 1, so a word of an earlier launch never matches; every CTA rescans table 1 until every word of
//      the call carries the tag (single-copy atomic relaxed 64-bit accesses: the data travels inside the word) and
//      builds the complete pick histogram in its shared memory (integer atomics: order independent);
//   3. E5 for the warp's own row, lane j = pick j: eligibility from the complete histogram, mass / run along the same
//      fp32 trees (tree8 / add.rn / mul.rn / non-ftz setp) as e5_and_store; with max_count <= 1 (host-checked) an
//      eligible expert has exactly one picker, so e5_and_store's drop test (cnd == cnt) is cand itself;
//   4. the final ids are published the same way in table 2; every CTA rescans it and builds the per-local-expert
//      row masks of the final picks (atomicOr), then record_build's counts / exclusive scan in local expert order;
//      CTA b owns a slice of the local experts and their permuted slots: -1 over the slots, a barrier, then the
//      CTA arrays and the live tokens (an expert's rows in ascending order, from its row bits) -- record_build's
//      -1 fill of [0, padded) followed by the live tokens, per slice; warp w writes its row's map entries (rank =
//      the expert's row bits below the row): the same bytes;
//   5. CTA 0 stores epoch = tag after its table-2 scan (every CTA has read the epoch before publishing); the next
//      launch on this table is stream-ordered after this grid's completion.
// No CTA waits on anything but the call's own CTAs, which are all launched (no dependent can start before the grid
// completes: launch_dependents at exit only), so partially placed grids only spin until the rest is placed.
RP_DEVICE unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
RP_DEVICE void st_relaxed_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
RP_DEVICE unsigned int ld_relaxed_u32(const unsigned int* p) {
  unsigned int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
RP_DEVICE void st_relaxed_u32(unsigned int* p, unsigned int v) {
  asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
RP_DEVICE void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
RP_DEVICE unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
RP_DEVICE void rtp_stamp(const Args& a, int i) {
  if (a.rtp_clocks != nullptr && threadIdx.x == 0 && blockIdx.x == 0) a.rtp_clocks[i] = gtimer();
}

template <int kThreads>
struct RtpShared {
  int32_t cnt[kBins];                              // pre-prune picks per expert (bin id + 1)
  uint32_t rmask[kMaxLocal * (kMaxRows / 32)];     // final picks: per local expert, its rows' bits
  int32_t num[kMaxLocal];                          // final picks per local expert
  int32_t off[kMaxLocal];                          // first record CTA of a local expert
  int32_t wsum[kThreads / 32];
  RowScratch rs[kThreads / 32];
};

// lanes 0..5 hold the row's (id) codes; lanes 0 / 1 return word 0 / 1 (ids 0..2 / 3..5)
RP_DEVICE unsigned long long rtp_word(uint32_t tag, int32_t id, int lane) {
  const uint32_t code = uint32_t(id + 1) & 511u;
  const int base = (lane & 1) * 3;
  const uint32_t c0 = __shfl_sync(0xffffffffu, code, base);
  const uint32_t c1 = __shfl_sync(0xffffffffu, code, base + 1);
  const uint32_t c2 = __shfl_sync(0xffffffffu, code, base + 2);
  return (static_cast<unsigned long long>(tag) << 32) | (c0 | (c1 << 9) | (c2 << 18));
}

template <int kThreads>
__global__ void __launch_bounds__(kThreads, 1) route_rtp_kernel(const Args a) {
  constexpr int kWarps = kThreads / 32;
  __shared__ __align__(16) RtpShared<kThreads> s;
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  const int rows = a.rows;
  const int words = (rows + 31) >> 5;
  const int nl = a.rr_on ? a.rr_num_local : 0;
  rtp_stamp(a, 0);
  for (int e = t; e < kBins; e += kThreads) s.cnt[e] = 0;            // before the wait: input independent
  for (int i = t; i < nl * words; i += kThreads) s.rmask[i] = 0u;
  float b[kPerLane], alt[kPerLane];
  load_bias(a, lane, b, alt);                                          // parameters: safe before the wait
  // this launch's tag: the layer's previous launch on this table stored its own tag as the epoch
  unsigned long long* t1 = a.rtp_table;
  unsigned long long* t2 = a.rtp_table + 2 * kMaxRows;
  {   // the table's lines + the epoch into L2 before the wait (a hint, no load: the table was last touched a whole
      // cycle ago and is cold; an exchange poll that misses to HBM costs ~1 us twice a launch)
    const int lines = (2 * rows + 15) >> 4;
    if (t < lines) prefetch_l2(t1 + 16 * t);
    else if (t < 2 * lines) prefetch_l2(t2 + 16 * (t - lines));
    else if (t == 2 * lines) prefetch_l2(a.rtp_epoch);
  }
  __syncthreads();
  grid_wait();
  // (read after the wait: ordered after every earlier launch on this table; its latency hides under route_row)
  const uint32_t tag = ld_relaxed_u32(a.rtp_epoch) + 1u;
  rtp_stamp(a, 1);
  const int64_t live = live_rows(a);
  const int row = int(blockIdx.x) * kWarps + warp;
  const bool mine = row < rows;                                        // warp-uniform
  int32_t oid = -1;
  float ow = 0.0f;
  if (mine) {
    if (a.ids_in != nullptr) {          // prune-only test mode: the given rows instead of the top-k
      if (lane < kTopK) {
        oid = a.ids_in[row * kTopK + lane];
        ow = a.w_in[row * kTopK + lane];
      }
    } else {
      route_row(a, row, live, lane, b, alt, s.rs[warp], oid, ow);
    }
    const unsigned long long w1 = rtp_word(tag, oid, lane);
    if (lane < 2) st_relaxed_u64(t1 + 2 * row + lane, w1);
  }
  rtp_stamp(a, 2);
  // table 1: every row's picks -> the complete histogram
  for (int i = t; i < 2 * rows; i += kThreads) {
    unsigned long long v;
    do {
      v = ld_relaxed_u64(t1 + i);
    } while (uint32_t(v >> 32) != tag);
#pragma unroll
    for (int q = 0; q < 3; ++q) {
      const uint32_t c = uint32_t(v >> (9 * q)) & 511u;
      if (c != 0u) atomicAdd(&s.cnt[c], 1);
    }
  }
  __syncthreads();
  rtp_stamp(a, 3);
  // E5 on the warp's row (lane j = pick j): e5_and_store's operations
  int32_t fid = oid;
  float fw = ow;
  if (mine) {
    int id[kTopK];
    float w[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      id[k] = __shfl_sync(0xffffffffu, oid, k);
      w[k] = __shfl_sync(0xffffffffu, ow, k);
    }
    bool elig[kTopK];
    float mv[8];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const bool valid = id[k] >= 0;
      const int c = valid ? s.cnt[id[k] + 1] : 0;
      elig[k] = valid && c <= a.max_count;
      mv[k] = valid ? w[k] : 0.0f;
    }
    mv[6] = 0.0f;
    mv[7] = 0.0f;
    const float mass = tree8(mv, a.mtree, a.mtree_id);
    float thr;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(thr) : "f"(a.budget), "f"(mass));
    const int j = lane < kTopK ? lane : 0;
    float wj = w[0];
    bool ej = elig[0];
#pragma unroll
    for (int k = 1; k < kTopK; ++k) {
      wj = (k == j) ? w[k] : wj;
      ej = (k == j) ? elig[k] : ej;
    }
    float tv[8];
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
      const bool before = elig[i] && (lt_f32_e(w[i], wj) || (eq_f32_e(w[i], wj) && i < j));
      tv[i] = before ? w[i] : 0.0f;
    }
    tv[6] = 0.0f;
    tv[7] = 0.0f;
    const float run = add_rn_v(tree8(tv, a.rtree, a.rtree_id), wj);
    const bool cand = ej && le_f32_e(run, thr);
    if (a.prune && cand && lane < kTopK) {   // max_count <= 1: the expert's only holder chose it (cnd == cnt == 1)
      fid = -1;
      fw = 0.0f;
    }
    if (lane < kTopK) {
      a.ids[row * kTopK + lane] = fid;
      a.weights[row * kTopK + lane] = fw;
    }
  }
  rtp_stamp(a, 4);
  if (!a.rr_on) {
    if (blockIdx.x == 0) {   // (no record: the epoch still advances once every row has been read)
      __syncthreads();
      if (t == 0) st_relaxed_u32(a.rtp_epoch, tag);
    }
    return;
  }
  if (mine) {
    const unsigned long long w2 = rtp_word(tag, fid, lane);
    if (lane < 2) st_relaxed_u64(t2 + 2 * row + lane, w2);
  }
  // table 2: every row's final picks -> per local expert row masks
  for (int i = t; i < 2 * rows; i += kThreads) {
    unsigned long long v;
    do {
      v = ld_relaxed_u64(t2 + i);
    } while (uint32_t(v >> 32) != tag);
    const int r = i >> 1;
#pragma unroll
    for (int q = 0; q < 3; ++q) {
      const int id = int(uint32_t(v >> (9 * q)) & 511u) - 1;
      const unsigned le = unsigned(id - a.rr_offset);
      if (id >= 0 && le < unsigned(nl)) atomicOr(&s.rmask[le * words + (r >> 5)], 1u << (r & 31));
    }
  }
  __syncthreads();
  if (blockIdx.x == 0 && t == 0) st_relaxed_u32(a.rtp_epoch, tag);   // every CTA has published table 2
  rtp_stamp(a, 5);
  // record_build's counts and exclusive scan over the local experts (thread t: experts e0 .. e1)
  const int sh = a.rr_tile_log2;
  const int per = (nl + kThreads - 1) / kThreads;
  const int e0 = min(t * per, nl);
  const int e1 = min(e0 + per, nl);
  int mcta = 0;
#pragma unroll 1
  for (int le = e0; le < e1; ++le) {
    int c = 0;
#pragma unroll 1
    for (int w = 0; w < words; ++w) c += __popc(s.rmask[le * words + w]);
    s.num[le] = c;
    mcta += (c + (1 << sh) - 1) >> sh;
  }
  int incl = mcta;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  int run = incl - mcta;
  int total = __shfl_sync(0xffffffffu, incl, 31);
  if constexpr (kWarps > 1) {
    if (lane == 31) s.wsum[warp] = incl;
    __syncthreads();
    int before = 0;
    total = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const int x = s.wsum[w];
      before += w < warp ? x : 0;
      total += x;
    }
    run += before;
  }
#pragma unroll 1
  for (int le = e0; le < e1; ++le) {
    s.off[le] = run;
    run += (s.num[le] + (1 << sh) - 1) >> sh;
  }
  if (blockIdx.x == 0 && t == 0) {
    a.rr_padded[0] = total << sh;
    a.rr_nonexit[0] = total;
  }
  __syncthreads();
  // this CTA's slice of the local experts [x0, x1) owns the permuted slots [lo N, hi N): -1 over all of them
  // (coalesced), then (after the barrier: same CTA) the live tokens and the CTA arrays, one expert a thread
  const int slice = (nl + int(gridDim.x) - 1) / int(gridDim.x);
  const int x0 = min(int(blockIdx.x) * slice, nl);
  const int x1 = min(x0 + slice, nl);
  const int lo = x0 < nl ? s.off[x0] : total;
  const int hi = x1 < nl ? s.off[x1] : total;
  for (int p = (lo << sh) + t; p < (hi << sh); p += kThreads) a.rr_tok[p] = -1;
  __syncthreads();
#pragma unroll 1
  for (int le = x0 + t; le < x1; le += kThreads) {
    const int c = s.num[le];
    const int o = s.off[le];
    const int n = (c + (1 << sh) - 1) >> sh;
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
      a.rr_batch[o + q] = le;
      a.rr_limit[o + q] = min((o + q + 1) << sh, (o << sh) + c);
    }
    int rank = o << sh;
#pragma unroll 1
    for (int w = 0; w < words; ++w) {
      uint32_t m = s.rmask[le * words + w];
      while (m != 0u) {
        a.rr_tok[rank++] = 32 * w + (__ffs(m) - 1);
        m &= m - 1u;
      }
    }
  }
  // the warp's row: map (rank = the expert's row bits below the row)
  if (mine && lane < kTopK) {
    int out = -1;
    const unsigned le = unsigned(fid - a.rr_offset);
    if (fid >= 0 && le < unsigned(nl)) {
      const uint32_t* m = s.rmask + le * words;
      int rank = __popc(m[row >> 5] & ((1u << (row & 31)) - 1u));
#pragma unroll 1
      for (int w = 0; w < (row >> 5); ++w) rank += __popc(m[w]);
      out = (s.off[le] << sh) + rank;
    }
    a.rr_map[row * kTopK + lane] = out;
  }
  rtp_stamp(a, 6);
  if (a.trigger == 1) {
    __threadfence();
    __syncthreads();
    grid_launch_dependents();
  }
}

// ---- host ----------------------------------------------------------------------------------------------------------
static void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("route_e5: " + what);
}

static void check_runtime(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    cudaGetLastError();
    require(false, std::string(what) + ": " + cudaGetErrorString(err));
  }
}

static void allow_big_clusters() {
  static bool done[64] = {};
  int dev = 0;
  check_runtime(cudaGetDevice(&dev), "device");
  if (dev >= 0 && dev < 64 && !done[dev]) {
    const cudaFuncAttribute attr = cudaFuncAttributeNonPortableClusterSizeAllowed;
    check_runtime(cudaFuncSetAttribute(route_e5_kernel<32, true>, attr, 1), "cluster attribute");
    check_runtime(cudaFuncSetAttribute(route_e5_kernel<64, true>, attr, 1), "cluster attribute");
    check_runtime(cudaFuncSetAttribute(route_e5_kernel<128, true>, attr, 1), "cluster attribute");
    check_runtime(cudaFuncSetAttribute(route_e5_kernel<256, true>, attr, 1), "cluster attribute");
    done[dev] = true;
  }
}

template <int kThreads>
static void launch_shape(cudaLaunchConfig_t& cfg, cudaLaunchAttribute* attrs, int64_t cluster, const Args& a) {
  cfg.blockDim = dim3(kThreads, 1, 1);
  if (cluster > 1) {
    allow_big_clusters();
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = static_cast<unsigned>(cluster);
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;
    cfg.numAttrs = 2;
    check_runtime(cudaLaunchKernelEx(&cfg, route_e5_kernel<kThreads, true>, a), "cluster launch");
  } else {
    check_runtime(cudaLaunchKernelEx(&cfg, route_e5_kernel<kThreads, false>, a), "launch");
  }
}

static void put_tree(uint8_t* dst, const std::vector<int64_t>& steps, const char* what) {
  require(static_cast<int>(steps.size()) == 2 * kTreeSteps, std::string(what) + " needs 14 step indices");
  for (int s = 0; s < kTreeSteps; ++s) {
    for (int h = 0; h < 2; ++h) {
      const int64_t v = steps[2 * s + h];
      require(v >= 0 && v < 8 + s, std::string(what) + ": a step reads an unwritten value");
      dst[2 * s + h] = static_cast<uint8_t>(v);
    }
  }
}

// One call = the stock top-k of `logits` [rows, 384] fp32 (row stride `stride` elements) (+ E5 when `prune`) into
// weights / ids [rows, 6] (fp32 / int32, contiguous). Pointers are device addresses (0 = absent for bias_alt,
// input_ids, ntnp, clocks); `stream` is the cudaStream_t of the caller's current stream. ids_in / w_in (0: off):
// prune-only mode -- E5 on the given [rows, 6] ids / weights instead of the top-k (the load-time self-test).
// run()'s argument checks and Args (shared with run_nc: lever RE5NC mode 3).
static Args make_args(int64_t logits, int64_t stride, int64_t rows, int64_t bias, int64_t bias_alt, int64_t input_ids, bool ids64,
                       int64_t ntnp, bool ntnp64, int64_t weights, int64_t ids, int64_t image_token, double scale, double eps,
                       bool apply_scale, int64_t sum_mode, bool prune, double budget, int64_t max_count, std::vector<int64_t> mtree,
                       std::vector<int64_t> rtree, int64_t cluster, int64_t threads, bool pdl, int64_t trigger, int64_t clocks,
                       int64_t stream, int64_t ids_in, int64_t w_in, std::vector<int64_t> ktree, std::vector<int64_t> record, std::vector<int64_t> tree_ids) {
  require(rows >= 1 && rows <= kMaxRows, "1..256 rows");
  require(logits % 16 == 0 && stride % 4 == 0 && stride >= kExperts, "logits rows must be 16-byte aligned");
  require(bias % 16 == 0 && (bias_alt == 0 || bias_alt % 16 == 0), "biases must be 16-byte aligned");
  require(weights % 8 == 0 && ids % 8 == 0, "weights / ids must be 8-byte aligned");
  require(input_ids == 0 || bias_alt != 0, "input_ids need bias_alt");
  require(sum_mode >= 0 && sum_mode < kSumModes, "sum_mode 0..5");
  require(cluster >= 1 && cluster <= kMaxCluster && cluster <= rows, "cluster 1..16 and <= rows");
  require(trigger >= 0 && trigger <= 2, "trigger 0..2");
  require((threads == 32 || threads == 64 || threads == 128 || threads == 256) && rows <= threads,
          "threads 32 / 64 / 128 / 256 and >= rows");
  require(max_count >= 0 && max_count <= kMaxRows, "max_count");
  Args a = {};
  a.logits = reinterpret_cast<const float*>(logits);
  a.stride = stride;
  a.bias = reinterpret_cast<const float*>(bias);
  a.bias_alt = bias_alt ? reinterpret_cast<const float*>(bias_alt) : nullptr;
  a.input_ids = input_ids ? reinterpret_cast<const void*>(input_ids) : nullptr;
  a.ids64 = ids64 ? 1 : 0;
  a.ntnp = ntnp ? reinterpret_cast<const void*>(ntnp) : nullptr;
  a.ntnp64 = ntnp64 ? 1 : 0;
  a.weights = reinterpret_cast<float*>(weights);
  a.ids = reinterpret_cast<int32_t*>(ids);
  a.clocks = clocks ? reinterpret_cast<long long*>(clocks) : nullptr;
  require((ids_in == 0) == (w_in == 0) && (ids_in == 0 || cluster == 1), "prune-only mode: both inputs, one CTA");
  a.ids_in = ids_in ? reinterpret_cast<const int32_t*>(ids_in) : nullptr;
  a.w_in = w_in ? reinterpret_cast<const float*>(w_in) : nullptr;
  a.rows = static_cast<int>(rows);
  a.rows_per_cta = static_cast<int>((rows + cluster - 1) / cluster);
  a.image_token = static_cast<int>(image_token);
  a.sum_mode = static_cast<int>(sum_mode);
  a.apply_scale = apply_scale ? 1 : 0;
  a.prune = prune ? 1 : 0;
  a.trigger = static_cast<int>(trigger);
  a.scale = static_cast<float>(scale);
  a.eps = static_cast<float>(eps);
  a.budget = static_cast<float>(budget);
  a.max_count = static_cast<int>(max_count);
  put_tree(a.mtree, mtree, "mtree");
  put_tree(a.rtree, rtree, "rtree");
  put_tree(a.ktree, ktree, "ktree");
  require(tree_ids.empty() || tree_ids.size() == 3, "tree_ids: [] or [mtree, rtree, ktree] ids");
  a.mtree_id = a.rtree_id = a.ktree_id = -1;
  if (!tree_ids.empty()) {
    for (int i = 0; i < 3; ++i) require(tree_ids[i] >= -1 && tree_ids[i] < kFixedTrees, "tree id -1..2");
    a.mtree_id = static_cast<int8_t>(tree_ids[0]);
    a.rtree_id = static_cast<int8_t>(tree_ids[1]);
    a.ktree_id = static_cast<int8_t>(tree_ids[2]);
  }
  // lever RR: record = [] (off) or [map, tok, batch, limit, padded, nonexit, tile, offset, num_local] (device pointers
  // to int32 buffers sized by FlashInfer's trtllm_moe_allocate_routing_metadata_multi_tile for these rows and tile)
  require(record.empty() || record.size() == 9, "record: [] or 9 values");
  if (!record.empty()) {
    for (int i = 0; i < 6; ++i) require(record[i] != 0 && record[i] % 4 == 0, "record buffers: int32 device pointers");
    require(record[0] % 8 == 0, "record map: 8-byte aligned");
    int log2 = -1;
    for (int b = 0; b <= 8; ++b)
      if ((int64_t(1) << b) == record[6]) log2 = b;
    require(log2 >= 0, "record tile: a power of two <= 256");
    require(record[8] >= 1 && record[8] <= kExperts && record[7] >= 0 && record[7] + record[8] <= kExperts,
            "record local expert range");
    a.rr_map = reinterpret_cast<int32_t*>(record[0]);
    a.rr_tok = reinterpret_cast<int32_t*>(record[1]);
    a.rr_batch = reinterpret_cast<int32_t*>(record[2]);
    a.rr_limit = reinterpret_cast<int32_t*>(record[3]);
    a.rr_padded = reinterpret_cast<int32_t*>(record[4]);
    a.rr_nonexit = reinterpret_cast<int32_t*>(record[5]);
    a.rr_tile_log2 = log2;
    a.rr_offset = static_cast<int>(record[7]);
    a.rr_num_local = static_cast<int>(record[8]);
    a.rr_on = 1;
    require(prune, "record: only with E5 (the final ids)");
  }
  return a;
}

void run(int64_t logits, int64_t stride, int64_t rows, int64_t bias, int64_t bias_alt, int64_t input_ids, bool ids64,
         int64_t ntnp, bool ntnp64, int64_t weights, int64_t ids, int64_t image_token, double scale, double eps,
         bool apply_scale, int64_t sum_mode, bool prune, double budget, int64_t max_count, std::vector<int64_t> mtree,
         std::vector<int64_t> rtree, int64_t cluster, int64_t threads, bool pdl, int64_t trigger, int64_t clocks,
         int64_t stream, int64_t ids_in, int64_t w_in, std::vector<int64_t> ktree, std::vector<int64_t> record, std::vector<int64_t> tree_ids) {
  const Args a = make_args(logits, stride, rows, bias, bias_alt, input_ids, ids64, ntnp, ntnp64, weights, ids, image_token, scale, eps,
                  apply_scale, sum_mode, prune, budget, max_count, mtree, rtree, cluster, threads, pdl, trigger,
                  clocks, stream, ids_in, w_in, ktree, record, tree_ids);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(cluster), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  switch (threads) {
    case 32: launch_shape<32>(cfg, attrs, cluster, a); break;
    case 64: launch_shape<64>(cfg, attrs, cluster, a); break;
    case 128: launch_shape<128>(cfg, attrs, cluster, a); break;
    default: launch_shape<256>(cfg, attrs, cluster, a); break;
  }
}

// Lever RE5NC (step 2): the stock top-k of `logits` [rows, 384] into weights / ids [rows, 6] on a plain grid of
// ceil(rows / rows_per_cta) CTAs of `threads` threads (no cluster, no E5: RE5's prune-only mode follows as a PDL
// dependent). Arguments as run()'s top-k half; ktree_id: the fixed-tree id of ktree (-1: the runtime steps).
void run_grid(int64_t logits, int64_t stride, int64_t rows, int64_t bias, int64_t bias_alt, int64_t input_ids,
              bool ids64, int64_t ntnp, bool ntnp64, int64_t weights, int64_t ids, int64_t image_token, double scale,
              double eps, bool apply_scale, int64_t sum_mode, std::vector<int64_t> ktree, int64_t ktree_id,
              int64_t rows_per_cta, int64_t threads, bool pdl, int64_t trigger, int64_t stream) {
  require(rows >= 1 && rows <= kMaxRows, "1..256 rows");
  require(logits != 0 && logits % 16 == 0 && stride % 4 == 0 && stride >= kExperts,
          "logits rows must be 16-byte aligned");
  require(bias != 0 && bias % 16 == 0 && (bias_alt == 0 || bias_alt % 16 == 0), "biases must be 16-byte aligned");
  require(weights != 0 && ids != 0 && weights % 4 == 0 && ids % 4 == 0, "weights / ids must be 4-byte aligned");
  require(input_ids == 0 || bias_alt != 0, "input_ids need bias_alt");
  require(sum_mode >= 0 && sum_mode < kSumModes, "sum_mode 0..5");
  require(trigger == 0 || trigger == 1, "trigger 0..1");
  require(threads == 32 || threads == 64 || threads == 128 || threads == 256, "threads 32 / 64 / 128 / 256");
  require(rows_per_cta >= 1 && rows_per_cta <= kMaxRows, "rows_per_cta 1..256");
  require(ktree_id >= -1 && ktree_id < kFixedTrees, "ktree id -1..2");
  Args a = {};
  a.logits = reinterpret_cast<const float*>(logits);
  a.stride = stride;
  a.bias = reinterpret_cast<const float*>(bias);
  a.bias_alt = bias_alt ? reinterpret_cast<const float*>(bias_alt) : nullptr;
  a.input_ids = input_ids ? reinterpret_cast<const void*>(input_ids) : nullptr;
  a.ids64 = ids64 ? 1 : 0;
  a.ntnp = ntnp ? reinterpret_cast<const void*>(ntnp) : nullptr;
  a.ntnp64 = ntnp64 ? 1 : 0;
  a.weights = reinterpret_cast<float*>(weights);
  a.ids = reinterpret_cast<int32_t*>(ids);
  a.rows = static_cast<int>(rows);
  a.rows_per_cta = static_cast<int>(rows_per_cta);
  a.image_token = static_cast<int>(image_token);
  a.sum_mode = static_cast<int>(sum_mode);
  a.apply_scale = apply_scale ? 1 : 0;
  a.trigger = static_cast<int>(trigger);
  a.scale = static_cast<float>(scale);
  a.eps = static_cast<float>(eps);
  put_tree(a.ktree, ktree, "ktree");
  a.ktree_id = static_cast<int8_t>(ktree_id);
  a.mtree_id = a.rtree_id = -1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>((rows + rows_per_cta - 1) / rows_per_cta), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  switch (threads) {
    case 32: check_runtime(cudaLaunchKernelEx(&cfg, route_topk_grid_kernel<32>, a), "grid launch"); break;
    case 64: check_runtime(cudaLaunchKernelEx(&cfg, route_topk_grid_kernel<64>, a), "grid launch"); break;
    case 128: check_runtime(cudaLaunchKernelEx(&cfg, route_topk_grid_kernel<128>, a), "grid launch"); break;
    default: check_runtime(cudaLaunchKernelEx(&cfg, route_topk_grid_kernel<256>, a), "grid launch"); break;
  }
}

// Lever RE5NC mode 3: run()'s fused top-k + E5 (+ record) on a plain grid of ceil(rows / rows_per_cta) CTAs (no
// cluster) whose last CTA runs E5; `ticket`: a zeroed device uint32 owned by the caller's device, never used by two
// launches at once (route calls are stream-ordered); threads >= rows (E5: thread = row). Other arguments as run()'s
// with cluster 1 and no prune-only inputs.
void run_nc(int64_t logits, int64_t stride, int64_t rows, int64_t bias, int64_t bias_alt, int64_t input_ids,
            bool ids64, int64_t ntnp, bool ntnp64, int64_t weights, int64_t ids, int64_t image_token, double scale,
            double eps, bool apply_scale, int64_t sum_mode, double budget, int64_t max_count,
            std::vector<int64_t> mtree, std::vector<int64_t> rtree, int64_t threads, bool pdl, int64_t stream,
            std::vector<int64_t> ktree, std::vector<int64_t> record, std::vector<int64_t> tree_ids,
            int64_t rows_per_cta, int64_t ticket) {
  require(logits != 0, "logits");
  require(rows_per_cta >= 1 && rows_per_cta <= kMaxRows, "rows_per_cta 1..256");
  require(ticket != 0 && ticket % 4 == 0, "ticket: a uint32 device pointer");
  Args a = make_args(logits, stride, rows, bias, bias_alt, input_ids, ids64, ntnp, ntnp64, weights, ids, image_token,
                     scale, eps, apply_scale, sum_mode, true, budget, max_count, mtree, rtree, 1, threads, pdl, 0, 0,
                     stream, 0, 0, ktree, record, tree_ids);
  a.rows_per_cta = static_cast<int>(rows_per_cta);
  a.ticket = reinterpret_cast<unsigned int*>(ticket);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>((rows + rows_per_cta - 1) / rows_per_cta), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  switch (threads) {
    case 32: check_runtime(cudaLaunchKernelEx(&cfg, route_nc_last_kernel<32>, a), "nc launch"); break;
    case 64: check_runtime(cudaLaunchKernelEx(&cfg, route_nc_last_kernel<64>, a), "nc launch"); break;
    case 128: check_runtime(cudaLaunchKernelEx(&cfg, route_nc_last_kernel<128>, a), "nc launch"); break;
    default: check_runtime(cudaLaunchKernelEx(&cfg, route_nc_last_kernel<256>, a), "nc launch"); break;
  }
}

// Lever RTP: top-k + E5 (+ record) on a plain grid of ceil(rows / (threads / 32)) CTAs, one row a warp, the rows
// exchanged through `table` (int64 [2 * kMaxRows * 2], zeroed once) tagged by `epoch` (uint32, 0 at allocation; the
// kernel advances it): one table per (device, MoE layer); launches on a table are stream-ordered. max_count <= 1.
// ids_in / w_in (0: off): the given rows instead of the top-k (test mode). clocks (0: off): 7 %globaltimer stamps.
void run_rtp(int64_t logits, int64_t stride, int64_t rows, int64_t bias, int64_t bias_alt, int64_t input_ids,
             bool ids64, int64_t ntnp, bool ntnp64, int64_t weights, int64_t ids, int64_t image_token, double scale,
             double eps, bool apply_scale, int64_t sum_mode, double budget, int64_t max_count,
             std::vector<int64_t> mtree, std::vector<int64_t> rtree, int64_t threads, bool pdl, int64_t stream,
             std::vector<int64_t> ktree, std::vector<int64_t> record, std::vector<int64_t> tree_ids, int64_t table,
             int64_t epoch, int64_t ids_in, int64_t w_in, int64_t clocks, int64_t trigger) {
  require(ids_in != 0 || logits != 0, "logits");
  require(table != 0 && table % 16 == 0 && epoch != 0 && epoch % 8 == 0, "table / epoch: device pointers");
  require(threads == 128 || threads == 256 || threads == 512, "threads 128 / 256 / 512");
  require(max_count >= 0 && max_count <= 1, "RTP: max_count 0 / 1 (the drop test is the candidate test)");
  require(trigger == 0 || trigger == 1, "trigger 0..1");
  require(weights % 4 == 0 && ids % 4 == 0, "weights / ids 4-byte aligned");
  Args a = make_args(logits != 0 ? logits : 16, stride, rows, bias, bias_alt, input_ids, ids64, ntnp, ntnp64,
                     weights - weights % 8, ids - ids % 8, image_token, scale, eps, apply_scale, sum_mode, true, budget,
                     max_count, mtree, rtree, 1, 256, pdl, 0, 0, stream, ids_in, w_in, ktree, record, tree_ids);
  a.logits = logits != 0 ? reinterpret_cast<const float*>(logits) : nullptr;
  a.weights = reinterpret_cast<float*>(weights);
  a.ids = reinterpret_cast<int32_t*>(ids);
  a.trigger = static_cast<int>(trigger);
  a.rtp_table = reinterpret_cast<unsigned long long*>(table);
  a.rtp_epoch = reinterpret_cast<unsigned int*>(epoch);
  a.rtp_clocks = clocks ? reinterpret_cast<unsigned long long*>(clocks) : nullptr;
  const int warps = static_cast<int>(threads / 32);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>((rows + warps - 1) / warps), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  switch (threads) {
    case 128: check_runtime(cudaLaunchKernelEx(&cfg, route_rtp_kernel<128>, a), "rtp launch"); break;
    case 256: check_runtime(cudaLaunchKernelEx(&cfg, route_rtp_kernel<256>, a), "rtp launch"); break;
    default: check_runtime(cudaLaunchKernelEx(&cfg, route_rtp_kernel<512>, a), "rtp launch"); break;
  }
}

// Clusters of `cluster` CTAs of `threads` threads that can be co-resident on this GPU (0: cannot run).
int64_t cluster_fits(int64_t cluster, int64_t threads) {
  if (cluster < 2 || cluster > kMaxCluster) return cluster == 1 ? 1 : 0;
  allow_big_clusters();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(cluster), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = static_cast<unsigned>(cluster);
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = 0;
  const void* fn = threads == 32    ? reinterpret_cast<const void*>(route_e5_kernel<32, true>)
                   : threads == 64  ? reinterpret_cast<const void*>(route_e5_kernel<64, true>)
                   : threads == 128 ? reinterpret_cast<const void*>(route_e5_kernel<128, true>)
                                    : reinterpret_cast<const void*>(route_e5_kernel<256, true>);
  const cudaError_t err = cudaOccupancyMaxActiveClusters(&n, fn, &cfg);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n;
}

}  // namespace route_e5

// ---- lever R5P, folded into this unit by gen_rtp.py --r5p-fold (kits/r5p_unit.cu after its verbatim
// copy of route_e5's routing block, which is this file's own) ----
namespace r5p {
using namespace route_e5;

// ---- lever R5P ------------------------------------------------------------------------------------------------------
constexpr int kR5Rows = 128;                // rows r5p_kernel serves at most (leader record: thread = row)

// e5_and_store's comparisons (route_e5.cu, after route_row), the same instructions
RP_DEVICE bool lt_f32_e(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.lt.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}
RP_DEVICE bool le_f32_e(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.le.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}
RP_DEVICE bool eq_f32_e(float a, float b) {
  uint32_t p;
  asm("{\n .reg .pred q;\n setp.eq.f32 q, %1, %2;\n selp.u32 %0, 1, 0, q;\n}" : "=r"(p) : "f"(a), "f"(b));
  return p != 0;
}

RP_DEVICE void cluster_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
RP_DEVICE void cluster_wait_acquire() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
RP_DEVICE void red_cluster_add_if(uint32_t addr, uint32_t v, uint32_t pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q red.shared::cluster.add.u32 [%0], %1;\n}"
               ::"r"(addr), "r"(v), "r"(pred) : "memory");
}
RP_DEVICE void red_cluster_or_if(uint32_t addr, uint32_t v, uint32_t pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q red.shared::cluster.or.b32 [%0], %1;\n}"
               ::"r"(addr), "r"(v), "r"(pred) : "memory");
}
RP_DEVICE uint32_t ld_cluster_if(uint32_t addr, uint32_t pred) {   // 0 where pred is off
  uint32_t v = 0;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.shared::cluster.u32 %0, [%1];\n}"
               : "+r"(v) : "r"(addr), "r"(pred) : "memory");
  return v;
}

struct R5Args {
  Args a;                       // route_e5's arguments (route_row reads them); a.rows_per_cta = rows a CTA (<= warps)
  int fast;                     // 1: max_count <= 1 -> drop = candidate (no candidate-count histogram)
  int tok_vec;                  // 1: rr_tok 16-byte aligned (int4 tile-padding fill)
};

template <int kThreads>
struct R5Shared {
  int32_t id[kR5Rows * kTopK];  // record: the final ids, from the routing warps
  int32_t cnt[kBins];           // picks per expert (bin id + 1), from the routing warps
  int32_t cnd[kBins];           // candidate picks per expert (max_count >= 2 only)
  int32_t gone[kExperts];       // record: experts E5 dropped
  int32_t offs[kExperts];       // record: first CTA of each local expert
  RowScratch rs[kThreads / 32];
  int32_t wsum[kThreads / 32];
  uint32_t rmask[kMaxLocal * (kR5Rows / 32)];   // record: per local expert, its rows' bits (pre-prune)
};

// The leader after the last barrier: lever RR's record of the final ids (record_build's values, counts from the
// histogram). Thread t = local expert t for the counts / scan / CTA arrays, = row t for the map.
template <int kThreads>
RP_DEVICE void record_final(const R5Args& r, R5Shared<kThreads>& s) {
  constexpr int kWarps = kThreads / 32;
  const Args& a = r.a;
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  const int sh = a.rr_tile_log2;
  const int nl = a.rr_num_local;
  const int words = (a.rows + 31) >> 5;
  int c = 0, n = 0;
  if (t < nl) {
    const int e = a.rr_offset + t;
    c = s.gone[e] ? 0 : s.cnt[e + 1];
    n = (c + (1 << sh) - 1) >> sh;
  }
  int incl = n;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) s.wsum[warp] = incl;
  __syncthreads();
  int before = 0, total = 0;
#pragma unroll
  for (int q = 0; q < kWarps; ++q) {
    const int x = s.wsum[q];
    before += q < warp ? x : 0;
    total += x;
  }
  const int off = incl - n + before;
  stamp(a, 6);
  if (t < nl) {
    s.offs[t] = off;
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
      a.rr_batch[off + q] = t;
      a.rr_limit[off + q] = min((off + q + 1) << sh, (off << sh) + c);
    }
  }
  if (t == 0) {
    a.rr_padded[0] = total << sh;
    a.rr_nonexit[0] = total;
  }
  const int ntok = total << sh;
  if (r.tok_vec) {
    const int nv = ntok >> 2;
    int4* tv = reinterpret_cast<int4*>(a.rr_tok);
#pragma unroll 1
    for (int i = t; i < nv; i += kThreads) tv[i] = make_int4(-1, -1, -1, -1);
#pragma unroll 1
    for (int i = (nv << 2) + t; i < ntok; i += kThreads) a.rr_tok[i] = -1;
  } else {
#pragma unroll 1
    for (int i = t; i < ntok; i += kThreads) a.rr_tok[i] = -1;
  }
  stamp(a, 7);
  __syncthreads();           // offsets visible; the padding fill before the map's token stores
  if (t < a.rows) {
    const uint32_t below = (1u << lane) - 1u;
    int out[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const int id = s.id[t * kTopK + k];
      const unsigned le = unsigned(id - a.rr_offset);
      out[k] = -1;
      if (id >= 0 && le < unsigned(nl)) {
        const uint32_t* mk = s.rmask + le * words;
        int rank = __popc(mk[warp] & below);
#pragma unroll 1
        for (int q = 0; q < warp; ++q) rank += __popc(mk[q]);
        out[k] = (s.offs[le] << sh) + rank;
        a.rr_tok[out[k]] = t;
      }
    }
    int2* mp = reinterpret_cast<int2*>(a.rr_map + t * kTopK);
#pragma unroll
    for (int k = 0; k < kTopK / 2; ++k) mp[k] = make_int2(out[2 * k], out[2 * k + 1]);
  }
}

// A cluster of ceil(rows / R) CTAs (R = a.rows_per_cta <= kThreads / 32); warp w < R of CTA c routes row c * R + w.
template <int kThreads>
__global__ void __launch_bounds__(kThreads, 1) r5p_kernel(const R5Args r) {
  __shared__ __align__(16) R5Shared<kThreads> s;
  const Args& a = r.a;
  stamp(a, 0);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const uint32_t crank = cluster_ctarank();
  const int words = (a.rows + 31) >> 5;
  if (crank == 0) {          // before the wait: input independent; released to the cluster by the first barrier
    for (int e = threadIdx.x; e < kBins; e += kThreads) {
      s.cnt[e] = 0;
      s.cnd[e] = 0;
    }
    if (a.rr_on) {
      for (int e = threadIdx.x; e < kExperts; e += kThreads) s.gone[e] = 0;
      const int n = a.rr_num_local * words;
      for (int i = threadIdx.x; i < n; i += kThreads) s.rmask[i] = 0u;
    }
  }
  float b[kPerLane], alt[kPerLane];
  load_bias(a, lane, b, alt);                                          // parameters: safe before the wait
  cluster_arrive_release();
  grid_wait();
  stamp(a, 1);
  const int64_t live = live_rows(a);
  const int row = int(crank) * a.rows_per_cta + warp;
  const bool has = warp < a.rows_per_cta && row < a.rows;              // warp-uniform
  int32_t oid = -1;
  float ow = 0.0f;
  if (has) {
    if (a.ids_in != nullptr) {   // prune-only mode (the self-tests' budget-boundary rows): the given rows, no top-k
      const int k = lane < kTopK ? lane : 0;
      oid = a.ids_in[row * kTopK + k];
      ow = a.w_in[row * kTopK + k];
    } else {
      route_row(a, row, live, lane, b, alt, s.rs[warp], oid, ow);
    }
  }
  stamp(a, 2);
  const uint32_t cnt_base = mapa_leader(&s.cnt[0]);
  const uint32_t cnd_base = mapa_leader(&s.cnd[0]);
  const uint32_t id_base = mapa_leader(&s.id[0]);
  const uint32_t gone_base = mapa_leader(&s.gone[0]);
  const uint32_t mask_base = mapa_leader(&s.rmask[0]);
  const uint32_t valid = (has && lane < kTopK && oid >= 0) ? 1u : 0u;
  const uint32_t bin = cnt_base + uint32_t(valid ? oid + 1 : 0) * 4u;
  cluster_wait_acquire();    // every CTA has started and the leader's zeroed histogram / masks are visible
  red_cluster_add_if(bin, 1u, valid);
  if (a.rr_on) {
    const unsigned le = unsigned(oid - a.rr_offset);
    const uint32_t local = (valid != 0u && le < unsigned(a.rr_num_local)) ? 1u : 0u;
    red_cluster_or_if(mask_base + uint32_t(local ? int(le) * words + (row >> 5) : 0) * 4u, 1u << (row & 31), local);
  }
  cluster_sync_all();        // (1) release / acquire: every row's picks are in the histogram
  stamp(a, 3);
  // E5 of this row, lane j = pick j: e5_and_store's operations on the complete histogram
  const int c = int(ld_cluster_if(bin, valid));
  const bool elig = valid != 0u && c <= a.max_count;
  bool cand = false;
  if (has) {
    float w[kTopK];
    bool el[kTopK];
    float mv[8];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      w[k] = __shfl_sync(0xffffffffu, ow, k);
      el[k] = __shfl_sync(0xffffffffu, int(elig), k) != 0;
      const bool v = __shfl_sync(0xffffffffu, int(valid), k) != 0;
      mv[k] = v ? w[k] : 0.0f;
    }
    mv[6] = 0.0f;
    mv[7] = 0.0f;
    const float mass = tree8(mv, a.mtree, a.mtree_id);
    float thr;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(thr) : "f"(a.budget), "f"(mass));
    float tv[8];
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
      const bool before = el[i] && (lt_f32_e(w[i], ow) || (eq_f32_e(w[i], ow) && i < lane));
      tv[i] = before ? w[i] : 0.0f;
    }
    tv[6] = 0.0f;
    tv[7] = 0.0f;
    const float run = add_rn_v(tree8(tv, a.rtree, a.rtree_id), ow);
    cand = elig && le_f32_e(run, thr);
  }
  if (!r.fast) {             // (uniform) e5_and_store's candidate-count histogram
    red_cluster_add_if(cnd_base + uint32_t(cand ? oid + 1 : 0) * 4u, 1u, cand ? 1u : 0u);
    cluster_sync_all();      // (1b) every candidate is counted
    cand = cand && int(ld_cluster_if(cnd_base + uint32_t(oid + 1) * 4u, 1u)) == c;
  }
  if (has && lane < kTopK) {
    const int32_t fid = cand ? -1 : oid;
    const float fw = cand ? 0.0f : ow;
    const int slot = row * kTopK + lane;
    a.ids[slot] = fid;
    a.weights[slot] = fw;
    if (a.rr_on) {
      st_cluster_if(id_base + uint32_t(slot) * 4u, uint32_t(fid), 1u);
      st_cluster_if(gone_base + uint32_t(cand ? oid : 0) * 4u, 1u, cand ? 1u : 0u);
    }
  }
  stamp(a, 4);
  cluster_sync_all();        // (2) the final ids / dropped experts are in the leader; its shared memory is free after
  if (crank != 0 || !a.rr_on) return;
  record_final<kThreads>(r, s);
  stamp(a, 5);
}

// ---- host ----------------------------------------------------------------------------------------------------------
static void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("r5p_unit: " + what);
}

static void check_runtime(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    cudaGetLastError();
    require(false, std::string(what) + ": " + cudaGetErrorString(err));
  }
}

static void put_tree(uint8_t* dst, const std::vector<int64_t>& steps, const char* what) {
  require(static_cast<int>(steps.size()) == 2 * kTreeSteps, std::string(what) + " needs 14 step indices");
  for (int s = 0; s < kTreeSteps; ++s) {
    for (int h = 0; h < 2; ++h) {
      const int64_t v = steps[2 * s + h];
      require(v >= 0 && v < 8 + s, std::string(what) + ": a step reads an unwritten value");
      dst[2 * s + h] = static_cast<uint8_t>(v);
    }
  }
}

template <int kThreads>
static void allow_cluster() {
  static bool done[64] = {};
  int dev = 0;
  check_runtime(cudaGetDevice(&dev), "device");
  if (dev >= 0 && dev < 64 && !done[dev]) {
    check_runtime(cudaFuncSetAttribute(r5p_kernel<kThreads>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
                  "cluster attribute");
    done[dev] = true;
  }
}

// One call = route_e5.run's fused top-k + E5 (+ record) of `logits` [rows, 384] fp32 into weights / ids [rows, 6]
// (same arguments, prune on, trigger 0) on a cluster of ceil(rows / rows_per_cta) CTAs of `threads` threads, one row
// a warp (rows_per_cta 0: threads / 32). ids_in / w_in (0: off): prune-only mode -- E5 (+ record) on the given
// [rows, 6] ids / weights (self-tests). Returns the cluster size.
int64_t run(int64_t logits, int64_t stride, int64_t rows, int64_t bias, int64_t bias_alt, int64_t input_ids,
            bool ids64, int64_t ntnp, bool ntnp64, int64_t weights, int64_t ids, int64_t image_token, double scale,
            double eps, bool apply_scale, int64_t sum_mode, double budget, int64_t max_count,
            std::vector<int64_t> mtree, std::vector<int64_t> rtree, int64_t threads, bool pdl, int64_t clocks,
            int64_t stream, std::vector<int64_t> ktree, std::vector<int64_t> record, std::vector<int64_t> tree_ids,
            int64_t ids_in, int64_t w_in, int64_t rows_per_cta) {
  require(threads == 384 || threads == 512, "threads 384 / 512");
  const int64_t per = rows_per_cta > 0 ? rows_per_cta : threads / 32;
  require(per >= 1 && per <= threads / 32, "rows_per_cta: 0 (one a warp) or 1..threads / 32");
  require(rows >= 1 && rows <= kR5Rows && rows <= threads, "1..128 rows");
  const int64_t cluster = (rows + per - 1) / per;
  require(cluster >= 1 && cluster <= 16, "cluster 1..16");
  require((ids_in == 0) == (w_in == 0), "prune-only mode: both inputs");
  require(ids_in != 0 || (logits != 0 && logits % 16 == 0 && stride % 4 == 0 && stride >= kExperts),
          "logits rows must be 16-byte aligned");
  require(ids_in == 0 || (ids_in % 4 == 0 && w_in % 4 == 0), "prune-only inputs: 4-byte aligned");
  require(bias != 0 && bias % 16 == 0 && (bias_alt == 0 || bias_alt % 16 == 0), "biases must be 16-byte aligned");
  require(weights != 0 && ids != 0 && weights % 4 == 0 && ids % 4 == 0, "weights / ids must be 4-byte aligned");
  require(input_ids == 0 || bias_alt != 0, "input_ids need bias_alt");
  require(sum_mode >= 0 && sum_mode < kSumModes, "sum_mode 0..5");
  require(max_count >= 0 && max_count <= kMaxRows, "max_count");
  R5Args r = {};
  Args& a = r.a;
  a.logits = reinterpret_cast<const float*>(logits);
  a.stride = stride;
  a.bias = reinterpret_cast<const float*>(bias);
  a.bias_alt = bias_alt ? reinterpret_cast<const float*>(bias_alt) : nullptr;
  a.input_ids = input_ids ? reinterpret_cast<const void*>(input_ids) : nullptr;
  a.ids64 = ids64 ? 1 : 0;
  a.ntnp = ntnp ? reinterpret_cast<const void*>(ntnp) : nullptr;
  a.ntnp64 = ntnp64 ? 1 : 0;
  a.weights = reinterpret_cast<float*>(weights);
  a.ids = reinterpret_cast<int32_t*>(ids);
  a.clocks = clocks ? reinterpret_cast<long long*>(clocks) : nullptr;
  a.ids_in = ids_in ? reinterpret_cast<const int32_t*>(ids_in) : nullptr;
  a.w_in = w_in ? reinterpret_cast<const float*>(w_in) : nullptr;
  a.rows = static_cast<int>(rows);
  a.rows_per_cta = static_cast<int>(per);
  a.image_token = static_cast<int>(image_token);
  a.sum_mode = static_cast<int>(sum_mode);
  a.apply_scale = apply_scale ? 1 : 0;
  a.prune = 1;
  a.trigger = 0;
  a.scale = static_cast<float>(scale);
  a.eps = static_cast<float>(eps);
  a.budget = static_cast<float>(budget);
  a.max_count = static_cast<int>(max_count);
  put_tree(a.mtree, mtree, "mtree");
  put_tree(a.rtree, rtree, "rtree");
  put_tree(a.ktree, ktree, "ktree");
  require(tree_ids.size() == 3, "tree_ids: [mtree, rtree, ktree] ids");
  for (int i = 0; i < 3; ++i) require(tree_ids[i] >= -1 && tree_ids[i] < kFixedTrees, "tree id -1..2");
  a.mtree_id = static_cast<int8_t>(tree_ids[0]);
  a.rtree_id = static_cast<int8_t>(tree_ids[1]);
  a.ktree_id = static_cast<int8_t>(tree_ids[2]);
  a.ticket = nullptr;
  r.fast = max_count <= 1 ? 1 : 0;
  require(record.empty() || record.size() == 9, "record: [] or 9 values");
  if (!record.empty()) {
    for (int i = 0; i < 6; ++i) require(record[i] != 0 && record[i] % 4 == 0, "record buffers: int32 device pointers");
    require(record[0] % 8 == 0, "record map: 8-byte aligned");
    int log2 = -1;
    for (int b = 0; b <= 8; ++b)
      if ((int64_t(1) << b) == record[6]) log2 = b;
    require(log2 >= 0, "record tile: a power of two <= 256");
    require(record[8] >= 1 && record[8] <= kExperts && record[7] >= 0 && record[7] + record[8] <= kExperts,
            "record local expert range");
    require(record[8] <= threads, "record: local experts <= threads (one a thread)");
    a.rr_map = reinterpret_cast<int32_t*>(record[0]);
    a.rr_tok = reinterpret_cast<int32_t*>(record[1]);
    a.rr_batch = reinterpret_cast<int32_t*>(record[2]);
    a.rr_limit = reinterpret_cast<int32_t*>(record[3]);
    a.rr_padded = reinterpret_cast<int32_t*>(record[4]);
    a.rr_nonexit = reinterpret_cast<int32_t*>(record[5]);
    a.rr_tile_log2 = log2;
    a.rr_offset = static_cast<int>(record[7]);
    a.rr_num_local = static_cast<int>(record[8]);
    a.rr_on = 1;
    r.tok_vec = record[1] % 16 == 0 ? 1 : 0;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(cluster), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = static_cast<unsigned>(cluster);
  attrs[1].val.clusterDim.y = 1;
  attrs[1].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  if (threads == 384) {
    allow_cluster<384>();
    check_runtime(cudaLaunchKernelEx(&cfg, r5p_kernel<384>, r), "r5p launch");
  } else {
    allow_cluster<512>();
    check_runtime(cudaLaunchKernelEx(&cfg, r5p_kernel<512>, r), "r5p launch");
  }
  return cluster;
}

// Clusters of `cluster` CTAs of `threads` threads that can be co-resident on this GPU (0: cannot run).
int64_t cluster_fits(int64_t cluster, int64_t threads) {
  require(threads == 384 || threads == 512, "threads 384 / 512");
  if (cluster < 1 || cluster > 16) return 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(cluster), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = static_cast<unsigned>(cluster);
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = 0;
  cudaError_t err;
  if (threads == 384) {
    allow_cluster<384>();
    err = cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(r5p_kernel<384>), &cfg);
  } else {
    allow_cluster<512>();
    err = cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(r5p_kernel<512>), &cfg);
  }
  if (err != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n;
}

}  // namespace r5p

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &route_e5::run, "stock vision_topk + the fork's E5 prune as one kernel");
  m.def("cluster_fits", &route_e5::cluster_fits, "co-resident clusters of this size (0: cannot run)");
  m.def("run_grid", &route_e5::run_grid, "lever RE5NC: stock vision_topk's top-k on a plain (no cluster) grid");
  m.def("run_nc", &route_e5::run_nc, "lever RE5NC mode 3: top-k + E5 on a plain grid, E5 in the last CTA");
  m.def("run_rtp", &route_e5::run_rtp, "lever RTP: top-k + E5 + record on a plain grid, one row a warp, epoch table");
  m.attr("RTP_TABLE_WORDS") = int64_t(4 * route_e5::kMaxRows);
  m.def("r5p_run", &r5p::run, "lever R5P (folded): RE5's fused top-k + E5 (+ record) on a cluster, one row a warp");
  m.def("r5p_cluster_fits", &r5p::cluster_fits, "lever R5P (folded): co-resident clusters of this size");
  m.attr("R5P_MAX_ROWS") = int64_t(r5p::kR5Rows);
  m.attr("MAX_ROWS") = int64_t(route_e5::kMaxRows);
  m.attr("TOP_K") = int64_t(route_e5::kTopK);
  m.attr("EXPERTS") = int64_t(route_e5::kExperts);
  m.attr("SUM_MODES") = int64_t(route_e5::kSumModes);
  m.attr("BINS") = int64_t(route_e5::kBins);
}
