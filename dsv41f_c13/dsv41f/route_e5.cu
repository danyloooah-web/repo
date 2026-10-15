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
void run(int64_t logits, int64_t stride, int64_t rows, int64_t bias, int64_t bias_alt, int64_t input_ids, bool ids64,
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &route_e5::run, "stock vision_topk + the fork's E5 prune as one kernel");
  m.def("cluster_fits", &route_e5::cluster_fits, "co-resident clusters of this size (0: cannot run)");
  m.attr("MAX_ROWS") = int64_t(route_e5::kMaxRows);
  m.attr("TOP_K") = int64_t(route_e5::kTopK);
  m.attr("EXPERTS") = int64_t(route_e5::kExperts);
  m.attr("SUM_MODES") = int64_t(route_e5::kSumModes);
  m.attr("BINS") = int64_t(route_e5::kBins);
}
