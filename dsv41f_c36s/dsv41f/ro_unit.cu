// lev_RO: the inverse RoPE DeepseekV4's MQALayer.forward applies to the attention output's rope tails -- stock
// fused_rope_inplace(o[..., -64:], None, freqs_cis, positions, inverse=True), in place, every layer, prefill (eager
// EXTEND) and decode / verify (CUDA graphs) -- with stock's per-pair arithmetic and far more memory-level parallelism.
//
// Stock (fork 5e72ce5, JIT sglang::deepseek_rope_kernel<kUsePDL, true, Pos>, deepseek_v4/rope.cuh): one warp per
// (token, head), one bf16x2 pair per lane, a dependent chain position -> freq -> data -> store: ~8 KB of loads in flight
// per SM, latency bound (1.26 TB/s at 12096 rows). Here a token is 8 * heads / kIters threads: thread (head group g,
// 16-byte chunk c of the 128-byte tail) loads its four freq pairs (32 bytes) once and rotates chunk c of kIters heads
// (kIters independent 16-byte loads; every 8 consecutive lanes cover one whole tail). Large row counts take kIters 8
// (a warp a token: bandwidth from many bytes in flight per thread), small ones fewer heads per thread (more threads,
// every load issued at once: stock's latency chain with 4x fewer, 16-byte accesses); the caller picks (ro.py).
//
// Arithmetic per pair = stock's sm_103a SASS (nvcc 13.0, -O3, no fast math; tests/ro_test.py compares both builds):
//   FMUL t1 = f_imag * x_real; FMUL t2 = f_imag * x_imag; FFMA im = f_real * x_imag - t1; FFMA re = f_real * x_real + t2;
//   F2FP.BF16.F32.PACK_AB (hi im, lo re)
// x_real / x_imag = the pair's low / high bf16 widened exactly (stock's __bfloat1622float2). Every float op is inline
// PTX with an explicit rounding mode (.rn: no contraction, no .ftz) because bundle units build with --use_fast_math.
// PDL as stock: the position and freq loads run before griddepcontrol.wait, the data loads after it;
// griddepcontrol.launch_dependents after the stores. The pybind11 launch sets programmatic stream serialization when
// the caller passes pdl = 1 (stock: is_arch_support_pdl()).
#include <cuda_runtime.h>
#include <stdint.h>

namespace ro {

constexpr int kRope = 64;            // rope dims of a head's tail (32 complex pairs, 128 bytes of bf16)
constexpr int kMaxWarps = 16;        // warps per CTA

__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float neg_f(float a) {
  float r;
  asm("neg.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ uint32_t pack_bf16x2(float hi, float lo) {   // cvt.rn.bf16x2.f32 d, hi, lo
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// one complex pair (lo bf16 = real, hi bf16 = imag) times conj(fr + i fi), stock's instruction order
__device__ __forceinline__ uint32_t inv_rot(uint32_t w, float fr, float fi) {
  const float xr = __uint_as_float(w << 16);
  const float xi = __uint_as_float(w & 0xffff0000u);
  const float t1 = mul_rn(fi, xr);
  const float t2 = mul_rn(fi, xi);
  const float im = fma_rn(fr, xi, neg_f(t1));
  const float re = fma_rn(fr, xr, t2);
  return pack_bf16x2(im, re);
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// x: the tails' base (bf16 element 0 of token 0 / head 0), token stride sb and head stride sh in elements (multiples
// of 8); freqs: [*, 64] fp32 (pair j = floats 2j, 2j + 1 = (real, imag)); pos: [rows] int32 / int64. A token is
// 8 * kGroups threads: thread (g, c) rotates 16-byte chunk c of heads g, g + kGroups, ..., g + kGroups (kIters - 1)
// (kIters independent loads; every 8 consecutive lanes one whole 128-byte tail). kSh > 0: the head stride as a
// compile-time constant (the model's 512): immediate load / store offsets from one address -- ptxas then sinks loads
// behind the first rotations at 4 / 8 heads a thread (measured slower there; levers.RO_TILES leaves it off).
// pos / freqs carry no __restrict__: plain ld.global, which ptxas keeps ahead of griddepcontrol.wait as in stock's
// build (read-only .nc loads are scheduled past it).
template <int kIters, int kGroups, typename Pos, int kSh>
__global__ void __launch_bounds__(32 * kMaxWarps) inv_rope_tail(uint16_t* __restrict__ x, long long sb, long long sh,
                                                                const float* freqs, const Pos* pos, int rows) {
  constexpr int kTok = 8 * kGroups;
  const int t = threadIdx.x;
  const int row = blockIdx.x * (blockDim.x / kTok) + t / kTok;
  if (row >= rows) return;
  const int r = t % kTok;
  const int c = r & 7;
  const int g = r >> 3;
  const long long p = static_cast<long long>(pos[row]);
  const float4* fp = reinterpret_cast<const float4*>(freqs + p * kRope + 8 * c);
  const float4 f0 = fp[0];
  const float4 f1 = fp[1];
  pdl_wait();
  const long long hs = kSh > 0 ? static_cast<long long>(kSh) : sh;
  uint16_t* base = x + static_cast<long long>(row) * sb + static_cast<long long>(g) * hs + 8 * c;
  uint4 v[kIters];
#pragma unroll
  for (int u = 0; u < kIters; ++u) v[u] = *reinterpret_cast<const uint4*>(base + u * kGroups * hs);
#pragma unroll
  for (int u = 0; u < kIters; ++u) {
    uint4 o;
    o.x = inv_rot(v[u].x, f0.x, f0.y);
    o.y = inv_rot(v[u].y, f0.z, f0.w);
    o.z = inv_rot(v[u].z, f1.x, f1.y);
    o.w = inv_rot(v[u].w, f1.z, f1.w);
    *reinterpret_cast<uint4*>(base + u * kGroups * hs) = o;
  }
  pdl_launch();
}

template <int kIters, int kGroups, typename Pos>
cudaError_t launch(const cudaLaunchConfig_t& cfg, int64_t x, int64_t sb, int64_t sh, int64_t freqs, int64_t pos,
                   bool fixed, int rows) {
  auto* xp = reinterpret_cast<uint16_t*>(x);
  const auto* fp = reinterpret_cast<const float*>(freqs);
  const auto* pp = reinterpret_cast<const Pos*>(pos);
  const long long sbl = static_cast<long long>(sb), shl = static_cast<long long>(sh);
  if (fixed && sh == 512) return cudaLaunchKernelEx(&cfg, inv_rope_tail<kIters, kGroups, Pos, 512>, xp, sbl, shl, fp, pp, rows);
  return cudaLaunchKernelEx(&cfg, inv_rope_tail<kIters, kGroups, Pos, 0>, xp, sbl, shl, fp, pp, rows);
}

template <int kHeads, typename Pos>
cudaError_t launch_iters(const cudaLaunchConfig_t& cfg, int64_t x, int64_t sb, int64_t sh, int64_t freqs, int64_t pos,
                         int iters, bool fixed, int rows) {
  switch (iters) {
    case 1: return launch<1, kHeads, Pos>(cfg, x, sb, sh, freqs, pos, fixed, rows);
    case 2: return launch<2, kHeads / 2, Pos>(cfg, x, sb, sh, freqs, pos, fixed, rows);
    case 4: return launch<4, kHeads / 4, Pos>(cfg, x, sb, sh, freqs, pos, fixed, rows);
    default: return launch<8, kHeads / 8, Pos>(cfg, x, sb, sh, freqs, pos, fixed, rows);
  }
}

template <typename Pos>
cudaError_t launch_heads(const cudaLaunchConfig_t& cfg, int64_t x, int64_t sb, int64_t sh, int64_t heads, int64_t freqs,
                         int64_t pos, int iters, bool fixed, int rows) {
  if (heads == 32) return launch_iters<32, Pos>(cfg, x, sb, sh, freqs, pos, iters, fixed, rows);
  if (heads == 16) return launch_iters<16, Pos>(cfg, x, sb, sh, freqs, pos, iters, fixed, rows);
  return launch_iters<64, Pos>(cfg, x, sb, sh, freqs, pos, iters, fixed, rows);
}

}  // namespace ro

// ===================================================================================================================
// lev_SPA (gen_spa.py splices this section into the bundle COPY of ro_unit.cu, before its pybind11 block, and adds
// m.def("spa_run", ...)): our own sparse MLA decode attention for the indexed layers (compress ratio 1 / 2 / 4) of
// decode / verify calls, NON-EXACT (flagged): it replaces trtllm-gen's sparse MLA kernel (+ lever RO's inverse RoPE
// launch) for the rows levers.SPA_ROWS admits.
//
// Problem (TP2 rank): q [R, 32 heads, 512] e4m3 (QR's FP8 bytes: RoPE already applied), one shared K = V row of 512
// e4m3 per slot (uniform-FP8 pools, unit scale), per row 128 SWA slots (swa_idx, -1 = empty) + the first len[r] of 512
// compressed slots (topk_idx), a per-head sink logit; out bf16 [R, 32, 512], the last 64 dims of every head rotated
// back (lever RO's per-pair arithmetic) when rope = 1.
//
// -1 semantics (zero_mode 2 = trtllm-gen's, probed): SWA columns at or past seq_len ignored; -1 below it and -1 among
// the first len top-k columns = zero K/V rows with logit 0; top-k columns past len ignored.
// split > 1 (levers.SPA_SPLIT, off by default: slower in the micro) deals a row's tiles over a 2 / 3-CTA cluster and
// merges the fp32 partials in rank 0 by DSMEM.
// One CTA per row (512 threads = 16 warps, one CTA an SM: <= 148 rows run as one wave; warp w = 4 g + q reads TMEM
// lane quarter q and works on column group g), the row's 640-slot table walked as up to five 128-slot tiles (tile 0 =
// SWA, tiles 1..4 = the compressed top-k; tiles past len are skipped):
//   gather   cp.async 16 B pieces of the tile's 128 KV rows into a 128B-swizzled stage [4 chunks][128 slots][128 B]
//            (three stages: tiles t + 1, t + 2 are in flight while tile t computes); Q once, [4][32 heads][128 B]
//   S^T      = K . Q^T on tcgen05 kind::f8f6f4 (M 128 slots, N 32 heads, K 512) into TMEM (lane = slot); tile t + 1's
//            S^T is issued right behind tile t's PV MMA (two S^T buffers), so only tile 0's sits on the critical path
//   softmax  per head over the tile in the exp2 domain (thread = slot x 8 heads): tile max m_t (warp butterfly, then
//            the 4 lane quarters), p = 2^(s - m_t), P^T = e4m3(256 p) [32 heads][128 slots] (swizzled, K-major B),
//            l_t = sum of the fp32 p
//   O^T      = V^T . P on tcgen05 (A = the same KV stage read MN-major, M 128 value dims a chunk, K 128 slots) into a
//            TMEM accumulator, drained by every thread (value dim x 32 heads of one chunk) into fp32 registers with the
//            online merge acc = acc * 2^(M_old - M_new) + O_t * 2^(m_t - M_new), M starting at the sink logit, den alike
//   out      acc / (256 den) -> bf16 (+ the inverse RoPE pairs by lane shuffle) -> out[row][head][dim]
// Differences from trtllm-gen (why it is NON-exact): P is quantized against each tile's own max (trtllm-gen: its
// running max over its KV split), the per-head sums / merges run in this kernel's fp32 order, ex2.approx. No bf16
// partials (the merge stays in fp32 registers) and no second launch.
// PDL: launched with programmatic stream serialization; every global read of a predecessor output (q, indices,
// lengths, KV) is issued after griddepcontrol.wait through cp.async.cg / plain ld.global (no .nc, no __restrict__
// const); the RoPE table / positions (graph inputs) are read after the wait too. launch_dependents after the loop.
// ===================================================================================================================
#include <cuda_fp8.h>
#include <math_constants.h>

namespace spa {

constexpr int kHeads = 32;
constexpr int kDim = 512;
constexpr int kTile = 128;                         // slots a tile == the S MMA's M
constexpr int kTiles = 5;                          // 1 SWA + 4 top-k
constexpr int kThreads = 512;               // 16 warps: lane quarter q = warp & 3, column group g = warp >> 2
constexpr int kChunk = 128;                        // bytes of one 128B-swizzle row
constexpr int kChunks = kDim / kChunk;             // 4
constexpr int kStages = 3;
constexpr int kKvChunk = kTile * kChunk;           // 16 KiB: one chunk of a stage
constexpr int kKvStage = kChunks * kKvChunk;       // 64 KiB
constexpr int kQChunk = kHeads * kChunk;           // 4 KiB
constexpr int kQBytes = kChunks * kQChunk;         // 16 KiB
constexpr int kPBytes = kHeads * kTile;            // 4 KiB
constexpr uint32_t kTmemCols = 256;                // O^T: cols [0, 128) (chunk c at 32 c), S^T: [128, 192) (2 bufs)
constexpr float kPScale = 256.0f;
constexpr float kLog2e = 1.4426950408889634f;

struct Smem {
  uint8_t kv[kStages][kKvStage];                   // 1024-B aligned stages
  uint8_t q[kQBytes];
  uint8_t p[kPBytes];
  float red_max[4][kHeads];
  float red_sum[4][kHeads];
  alignas(16) float ref[kHeads];                   // this tile's P reference (its max, 0 if none)
  alignas(16) float alpha[kHeads];                 // 2^(M_old - M_new)
  alignas(16) float beta[kHeads];                  // 2^(m_t - M_new)
  alignas(16) float inv[kHeads];                   // 1 / (256 den) at the end (split: this CTA's merge weight)
  float stat_m[kHeads];                            // split: this CTA's running max / den (read by the cluster)
  float stat_d[kHeads];
  uint64_t bar_s;
  uint64_t bar_o;
  uint32_t taddr;
};
constexpr size_t kSmemBytes = sizeof(Smem) + 1024;  // + the manual 1024-B alignment

struct Params {
  const uint8_t* q;        // [R, 32, 512] e4m3, row stride q_stride bytes (heads 512 B apart)
  const uint8_t* swa_kv;   // [slots, 512] e4m3
  const uint8_t* c_kv;     // [slots, 512] e4m3
  const int32_t* swa_idx;  // [R, 128] (row stride swa_stride)
  const int32_t* topk_idx; // [R, 512] (row stride topk_stride)
  const void* topk_len;    // [R] int32 / int64
  const float* sinks;      // [32]
  const float* freqs;      // [P, 64] fp32 (pair j = (cos, sin) at 2j, 2j + 1); rope only
  const void* pos;         // [R] int32 / int64; rope only
  uint16_t* out;           // [R, 32, 512] bf16 contiguous
  long long q_stride;
  long long swa_stride;
  long long topk_stride;
  float scale_log2;        // softmax scale * log2(e)
  int len64;
  int pos64;
  int rope;
  int split;               // CTAs a row (a cluster): 1, 2 or 3; CTA k walks tiles k, k + split, ...
  int zero_mode;           // -1 slots inside the scanned table: 0 ignored, 1 zero K/V rows with logit 0 (every SWA
                           // column and the first len top-k columns scanned), 2 as 1 but SWA columns >= seq_len ignored
  const int* seq_lens;     // zero_mode 2: [R] int32 (the causal lengths)
};

// ---- PTX ----------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "SPA_WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra SPA_WAIT_%=;\n\t}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_alloc(uint32_t dst, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst), "r"(cols));
}
__device__ __forceinline__ void tc_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tc_dealloc(uint32_t taddr, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols));
}
__device__ __forceinline__ void tc_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_ld32(uint32_t taddr, float (&v)[32]) {
  uint32_t* d = reinterpret_cast<uint32_t*>(v);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"
      "%30,%31}, [%32];"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]), "=r"(d[8]),
        "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]), "=r"(d[16]),
        "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]), "=r"(d[24]),
        "=r"(d[25]), "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31])
      : "r"(taddr));
}
// D (TMEM) [+]= A (smem desc) x B (smem desc), kind::f8f6f4
__device__ __forceinline__ void tc_mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
// shared-memory matrix descriptor: 128B swizzle, SBO 1024 B (8-row groups), LBO unused (one swizzle atom wide)
__device__ __forceinline__ uint64_t sdesc(uint32_t addr) {
  return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t(1024u >> 4) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}
// instruction descriptor kind::f8f6f4: D f32 (bit 4), A / B e4m3 (format 0), A MN-major (bit 15) if a_mn, N >> 3 at 17,
// M >> 4 at 24
__host__ __device__ constexpr uint32_t idesc_e4m3(uint32_t M, uint32_t N, bool a_mn) {
  return (1u << 4) | (a_mn ? (1u << 15) : 0u) | (((N >> 3) & 0x3Fu) << 17) | (((M >> 4) & 0x1Fu) << 24);
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait(int pending) {
  if (pending <= 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
  else if (pending == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else asm volatile("cp.async.wait_group 2;" ::: "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float ex2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {   // lo -> low byte
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
// lever RO's per-pair arithmetic (ro::inv_rot): explicit .rn ops (the unit builds with --use_fast_math)
__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float neg_f(float a) {
  float r;
  asm("neg.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ uint16_t bf16_rn(float x) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ uint32_t mapa_u32(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ float ld_cluster(uint32_t addr) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void st_cluster4(uint32_t addr, float a, float b, float c, float d) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "f"(a), "f"(b), "f"(c), "f"(d)
               : "memory");
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
constexpr uint32_t kRedStride = 36 * 4;            // split merge: [src][dim 512][36 floats] fp32 rows (conflict-free v4)
constexpr uint32_t kRedBytes = kDim * kRedStride;  // 72 KiB a source, in the (idle) KV stages

// byte offset of (row, byte col) in [rows][128 B] with the 128B swizzle (1024-B aligned base)
__device__ __forceinline__ uint32_t swz(uint32_t row, uint32_t col) {
  return row * 128u + ((((col >> 4) ^ (row & 7u)) << 4) | (col & 15u));
}

// 8 values a lane (heads 8 g .. 8 g + 7 of this lane's slot): lane l ends with op over the warp's 32 lanes of value
// (l & 7) -- 3 transposing butterfly rounds (lane bits 2..0 pick the value), then 2 plain rounds over lane bits 3, 4
template <bool kMax>
__device__ __forceinline__ float warp_reduce8(float (&v)[8], uint32_t lane) {
#pragma unroll
  for (int half = 4; half >= 1; half >>= 1) {
    const bool upper = (lane & half) != 0;
#pragma unroll
    for (int i = 0; i < half; ++i) {
      const float send = upper ? v[i] : v[i + half];
      const float keep = upper ? v[i + half] : v[i];
      const float got = __shfl_xor_sync(0xFFFFFFFFu, send, half);
      v[i] = kMax ? fmaxf(keep, got) : keep + got;
    }
  }
  float r = v[0];
#pragma unroll
  for (int m = 8; m <= 16; m <<= 1) {
    const float got = __shfl_xor_sync(0xFFFFFFFFu, r, m);
    r = kMax ? fmaxf(r, got) : r + got;
  }
  return r;
}

// warp (q, g) gathers slots 32 q + 8 g .. 32 q + 8 g + 7 of a tile (lane l of the warp owns slot 32 q + l's index):
// lane l = 16-B piece l of each 512-B row (coalesced); idx < 0 -> zero fill, nothing read
__device__ __forceinline__ void gather_tile(uint32_t stage, const uint8_t* pool, int32_t own, uint32_t q, uint32_t g,
                                            uint32_t lane) {
  const uint32_t chunk = lane >> 3;
  const uint32_t col = (lane & 7u) * 16u;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const int32_t idx = __shfl_sync(0xFFFFFFFFu, own, 8u * g + i);
    const uint32_t slot = q * 32u + 8u * g + i;
    const uint8_t* src = pool + (idx >= 0 ? static_cast<long long>(idx) * kDim : 0LL) + lane * 16u;
    cp16(stage + chunk * kKvChunk + swz(slot, col), src, idx >= 0 ? 16u : 0u);
  }
}

__device__ __forceinline__ void tc_ld8(uint32_t taddr, float (&v)[8]) {
  uint32_t* d = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7])
               : "r"(taddr));
}

__device__ __forceinline__ void issue_s(uint32_t t_s, uint32_t stage, uint32_t q_base) {
  constexpr uint32_t kIdS = idesc_e4m3(kTile, kHeads, false);
  const uint64_t a0 = sdesc(stage), b0 = sdesc(q_base);
#pragma unroll
  for (uint32_t c = 0; c < kChunks; ++c)
#pragma unroll
    for (uint32_t kb = 0; kb < 4; ++kb)
      tc_mma(t_s, a0 + ((c * kKvChunk + kb * 32u) >> 4), b0 + ((c * kQChunk + kb * 32u) >> 4), kIdS, (c | kb) != 0);
}

// thread (warp w = 4 g + q, lane): TMEM lane quarter q (slots / value dims 32 q + lane), column group g (heads
// 8 g .. 8 g + 7 in the softmax, chunk g of O^T in the merge and the output)
__global__ void __launch_bounds__(kThreads, 1) attn_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t base = (raw + 1023u) & ~1023u;
  Smem& s = *reinterpret_cast<Smem*>(smem_raw + (base - raw));
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5;
  const uint32_t lane = tid & 31u;
  const uint32_t q4 = warp & 3u;
  const uint32_t g4 = warp >> 2;
  const uint32_t split = static_cast<uint32_t>(p.split);
  const uint32_t row = blockIdx.x / split;
  const uint32_t rank = blockIdx.x % split;          // == %cluster_ctarank (cluster dims (split, 1, 1))
  const uint32_t slot = q4 * 32u + lane;

  if (warp == 0) {
    tc_alloc(smem_u32(&s.taddr), kTmemCols);
    tc_relinquish();
  } else if (tid == 32) {
    mbar_init(smem_u32(&s.bar_s), 1);
    mbar_init(smem_u32(&s.bar_o), 1);
    fence_mbar_init();
  }
  pdl_wait();   // q (QR), the indices (indexer), the lengths and the KV rows are this step's producers' output

  const uint32_t q_base = smem_u32(s.q);
  const uint8_t* q_row = p.q + static_cast<long long>(row) * p.q_stride;
#pragma unroll
  for (uint32_t i = 0; i < 1024 / kThreads; ++i) {
    const uint32_t piece = i * kThreads + tid;
    const uint32_t head = piece >> 5;
    const uint32_t l = piece & 31u;
    cp16(q_base + (l >> 3) * kQChunk + swz(head, (l & 7u) * 16u), q_row + head * kDim + l * 16u, 16u);
  }
  const long long len_raw = p.len64 ? static_cast<const long long*>(p.topk_len)[row]
                                    : static_cast<long long>(static_cast<const int*>(p.topk_len)[row]);
  const int len = static_cast<int>(len_raw < 0 ? 0 : (len_raw > 512 ? 512 : len_raw));
  const int ntiles = 1 + (len + kTile - 1) / kTile;
  // this CTA's tiles: t = rank + i * split, i < nt
  const int nt = ntiles > static_cast<int>(rank) ? (ntiles - static_cast<int>(rank) + static_cast<int>(split) - 1) /
                                                       static_cast<int>(split)
                                                 : 0;
  // slot `slot`'s index in every tile (the 4 column groups load the same words); -2 = outside the scanned table
  int32_t own[kTiles];
  own[0] = p.swa_idx[static_cast<long long>(row) * p.swa_stride + slot];
  if (p.zero_mode == 2 && static_cast<int>(slot) >= p.seq_lens[row]) own[0] = -2;
#pragma unroll
  for (int t = 1; t < kTiles; ++t) {
    const int j = (t - 1) * kTile + static_cast<int>(slot);
    own[t] = j < len ? p.topk_idx[static_cast<long long>(row) * p.topk_stride + j] : -2;
  }
  const uint32_t kv_base = smem_u32(s.kv[0]);
  int committed = 0;
#pragma unroll
  for (int i = 0; i < kStages; ++i) {
    if (i < nt) {
      const int t = static_cast<int>(rank) + i * static_cast<int>(split);
      int32_t idx = own[kTiles - 1];
#pragma unroll
      for (int u = 0; u < kTiles; ++u)
        if (u == t) idx = own[u];
      gather_tile(kv_base + i * kKvStage, t == 0 ? p.swa_kv : p.c_kv, idx, q4, g4, lane);
      cp_commit();
      ++committed;
    }
  }
  // the RoPE pair of output dims 448..511 (chunk 3 = column group 3, lane quarters 2 / 3): position + table entry now
  float fr = 0.f, fi = 0.f;
  if (p.rope && g4 == kChunks - 1 && q4 >= 2) {
    const long long pos = p.pos64 ? static_cast<const long long*>(p.pos)[row]
                                  : static_cast<long long>(static_cast<const int*>(p.pos)[row]);
    const uint32_t j = (q4 * 32u + lane - 64u) >> 1;
    fr = p.freqs[pos * 64 + 2 * j];
    fi = p.freqs[pos * 64 + 2 * j + 1];
  }
  float run_m = 0.f, den = 0.f;     // warp 0, lane h: head h's running max / denominator (sink included)
  if (warp == 0) {
    if (rank == 0) {   // the sink belongs to the CTA holding tile 0
      const float sk = p.sinks[lane] * kLog2e;
      run_m = sk;
      den = sk == -CUDART_INF_F ? 0.f : 1.f;
    } else {
      run_m = -CUDART_INF_F;
      den = 0.f;
    }
  }
  float acc[kHeads];
#pragma unroll
  for (int h = 0; h < kHeads; ++h) acc[h] = 0.f;

  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t taddr = s.taddr;
  const uint32_t t_o = taddr;            // O^T chunk c at columns 32 c
  const uint32_t t_s0 = taddr + 128u;    // S^T of even tiles; odd tiles at + 32
  const uint32_t lane_off = (q4 * 32u) << 16;
  const uint32_t p_base = smem_u32(s.p);
  constexpr uint32_t kIdO = idesc_e4m3(kChunk, kHeads, true);

  // the first tile's S^T
  if (nt > 0) {
    cp_wait(committed - 1);
    fence_async_smem();
    __syncthreads();
    if (tid == 0) {
      tc_fence_after();
      issue_s(t_s0, kv_base, q_base);
      tc_commit(smem_u32(&s.bar_s));
    }
  }

#pragma unroll 1
  for (int i = 0; i < nt; ++i) {
    const int t = static_cast<int>(rank) + i * static_cast<int>(split);
    const uint32_t stage = kv_base + (i % kStages) * kKvStage;
    const uint32_t t_s = t_s0 + 32u * static_cast<uint32_t>(i & 1);
    int32_t my_idx = own[0];
#pragma unroll
    for (int u = 1; u < kTiles; ++u)
      if (u == t) my_idx = own[u];
    const bool valid = my_idx >= 0 || (my_idx == -1 && p.zero_mode != 0);   // a zero K row scores exactly 0
    mbar_wait(smem_u32(&s.bar_s), static_cast<uint32_t>(i & 1));
    tc_fence_after();
    // ---- softmax: TMEM lane == slot, heads 8 g4 .. 8 g4 + 7 --------------------------------------------------------
    float x[8];
    tc_ld8(t_s + lane_off + 8u * g4, x);
    tc_wait_ld();
#pragma unroll
    for (int h = 0; h < 8; ++h) x[h] = valid ? x[h] * p.scale_log2 : -CUDART_INF_F;
    {
      const float r = warp_reduce8<true>(x, lane);   // destroys x: S^T is read again below
      if (lane < 8) s.red_max[q4][8u * g4 + lane] = r;
    }
    tc_ld8(t_s + lane_off + 8u * g4, x);
    tc_wait_ld();
#pragma unroll
    for (int h = 0; h < 8; ++h) x[h] = valid ? x[h] * p.scale_log2 : -CUDART_INF_F;
    __syncthreads();
    float alpha = 1.f, beta = 0.f;
    if (warp == 0) {
      const float m_t = fmaxf(fmaxf(s.red_max[0][lane], s.red_max[1][lane]),
                              fmaxf(s.red_max[2][lane], s.red_max[3][lane]));
      const float m_new = fmaxf(run_m, m_t);
      alpha = m_new == run_m ? 1.f : ex2(run_m - m_new);
      beta = m_t == -CUDART_INF_F ? 0.f : ex2(m_t - m_new);
      run_m = m_new;
      s.ref[lane] = m_t == -CUDART_INF_F ? 0.f : m_t;
      s.alpha[lane] = alpha;
      s.beta[lane] = beta;
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 8; h += 2) {
      const uint32_t hh = 8u * g4 + h;
      const float e0 = ex2(x[h] - s.ref[hh]);
      const float e1 = ex2(x[h + 1] - s.ref[hh + 1]);
      x[h] = e0;
      x[h + 1] = e1;
      const uint32_t pair = e4m3x2(e0 * kPScale, e1 * kPScale);
      s.p[swz(hh, slot)] = static_cast<uint8_t>(pair & 0xFFu);
      s.p[swz(hh + 1, slot)] = static_cast<uint8_t>(pair >> 8);
    }
    {
      const float r = warp_reduce8<false>(x, lane);
      if (lane < 8) s.red_sum[q4][8u * g4 + lane] = r;
    }
    const bool next = i + 1 < nt;
    if (next) cp_wait(committed - (i + 2));   // the next tile's stage landed: its S^T goes out with this PV
    fence_async_smem();
    tc_fence_before();
    __syncthreads();
    if (tid == 0) {
      tc_fence_after();
      const uint64_t a0 = sdesc(stage), b0 = sdesc(p_base);
#pragma unroll
      for (uint32_t c = 0; c < kChunks; ++c)
#pragma unroll
        for (uint32_t kb = 0; kb < 4; ++kb)
          tc_mma(t_o + 32u * c, a0 + ((c * kKvChunk + kb * 32u * kChunk) >> 4), b0 + ((kb * 32u) >> 4), kIdO, kb != 0);
      tc_commit(smem_u32(&s.bar_o));
      if (next) {
        issue_s(t_s0 + 32u * static_cast<uint32_t>((i + 1) & 1), kv_base + ((i + 1) % kStages) * kKvStage, q_base);
        tc_commit(smem_u32(&s.bar_s));
      }
    }
    if (warp == 0) {
      const float l_t = (s.red_sum[0][lane] + s.red_sum[1][lane]) + (s.red_sum[2][lane] + s.red_sum[3][lane]);
      den = den * alpha + l_t * beta;
    }
    // ---- merge O_t: TMEM lane == value dim 32 q4 + lane of chunk g4 ------------------------------------------------
    mbar_wait(smem_u32(&s.bar_o), static_cast<uint32_t>(i & 1));
    tc_fence_after();
    {
      float o[kHeads];
      tc_ld32(t_o + 32u * g4 + lane_off, o);
      tc_wait_ld();
      const float4* al = reinterpret_cast<const float4*>(s.alpha);
      const float4* be = reinterpret_cast<const float4*>(s.beta);
#pragma unroll
      for (int h4 = 0; h4 < kHeads / 4; ++h4) {
        const float4 a = al[h4], b = be[h4];
        acc[4 * h4 + 0] = fmaf(acc[4 * h4 + 0], a.x, o[4 * h4 + 0] * b.x);
        acc[4 * h4 + 1] = fmaf(acc[4 * h4 + 1], a.y, o[4 * h4 + 1] * b.y);
        acc[4 * h4 + 2] = fmaf(acc[4 * h4 + 2], a.z, o[4 * h4 + 2] * b.z);
        acc[4 * h4 + 3] = fmaf(acc[4 * h4 + 3], a.w, o[4 * h4 + 3] * b.w);
      }
    }
    tc_fence_before();
    // the stage is free (its PV MMA completed): gather this CTA's tile i + 3 into it
    if (i + kStages < nt) {
      const int tn = t + kStages * static_cast<int>(split);
      int32_t nidx = own[kTiles - 1];
#pragma unroll
      for (int u = 0; u < kTiles; ++u)
        if (u == tn) nidx = own[u];
      gather_tile(stage, p.c_kv, nidx, q4, g4, lane);
      cp_commit();
      ++committed;
    }
  }
  pdl_launch();
  if (split == 1) {
    if (warp == 0) s.inv[lane] = 1.0f / (den * kPScale);
    __syncthreads();
  } else {
    // cluster merge: every CTA's (max, den) -> its weight 2^(m_k - M) / (256 D); the scaled partials of ranks > 0 go
    // into rank 0's idle KV stages by DSMEM, rank 0 sums them (ascending rank) and writes the row
    if (warp == 0) {
      s.stat_m[lane] = run_m;
      s.stat_d[lane] = den;
    }
    cluster_sync_all();
    if (warp == 0) {
      float mk[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F}, dk[3] = {0.f, 0.f, 0.f};
      float M = -CUDART_INF_F;
      for (uint32_t k = 0; k < split; ++k) {
        mk[k] = ld_cluster(mapa_u32(smem_u32(&s.stat_m[lane]), k));
        dk[k] = ld_cluster(mapa_u32(smem_u32(&s.stat_d[lane]), k));
        M = fmaxf(M, mk[k]);
      }
      float D = 0.f;
      for (uint32_t k = 0; k < split; ++k) D += mk[k] == -CUDART_INF_F ? 0.f : dk[k] * ex2(mk[k] - M);
      s.inv[lane] = run_m == -CUDART_INF_F ? 0.f : ex2(run_m - M) / (D * kPScale);
    }
    __syncthreads();
    const uint32_t dd = g4 * kChunk + q4 * 32u + lane;
    if (rank != 0) {
      const uint32_t dst = mapa_u32(kv_base + (rank - 1) * kRedBytes + dd * kRedStride, 0);
      const float4* iv4 = reinterpret_cast<const float4*>(s.inv);
#pragma unroll
      for (int h4 = 0; h4 < kHeads / 4; ++h4) {
        const float4 w = iv4[h4];
        st_cluster4(dst + 16u * h4, acc[4 * h4] * w.x, acc[4 * h4 + 1] * w.y, acc[4 * h4 + 2] * w.z,
                    acc[4 * h4 + 3] * w.w);
      }
    }
    cluster_sync_all();
    if (rank != 0) {
      tc_fence_before();
      __syncthreads();
      if (warp == 0) {
        tc_fence_after();
        tc_dealloc(s.taddr, kTmemCols);
      }
      return;
    }
#pragma unroll
    for (int h = 0; h < kHeads; ++h) acc[h] *= s.inv[h];
    for (uint32_t k = 1; k < split; ++k) {
      const float4* src = reinterpret_cast<const float4*>(s.kv[0] + (k - 1) * kRedBytes + dd * kRedStride);
#pragma unroll
      for (int h4 = 0; h4 < kHeads / 4; ++h4) {
        const float4 v = src[h4];
        acc[4 * h4] += v.x;
        acc[4 * h4 + 1] += v.y;
        acc[4 * h4 + 2] += v.z;
        acc[4 * h4 + 3] += v.w;
      }
    }
    __syncthreads();   // every thread has read its weights
    if (warp == 0) s.inv[lane] = 1.0f;
    __syncthreads();
  }

  // ---- out[row][h][d], d = 128 g4 + 32 q4 + lane; the inverse RoPE on d >= 448 (g4 3, q4 2 / 3) -----------------
  uint16_t* orow = p.out + static_cast<long long>(row) * (kHeads * kDim);
  const uint32_t d = g4 * kChunk + q4 * 32u + lane;
  const float4* iv = reinterpret_cast<const float4*>(s.inv);
  if (p.rope && g4 == kChunks - 1 && q4 >= 2) {
    const bool real = (lane & 1u) == 0;
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      const uint16_t b = bf16_rn(acc[h] * s.inv[h]);
      const float own_v = __uint_as_float(static_cast<uint32_t>(b) << 16);
      const float other = __shfl_xor_sync(0xFFFFFFFFu, own_v, 1);
      // real lane: re = fr * xr + fi * xi; imag lane: im = fr * xi - fi * xr (lever RO's FMUL / FFMA order)
      const float t2 = mul_rn(fi, other);
      const float r = fma_rn(fr, own_v, real ? t2 : neg_f(t2));
      orow[h * kDim + d] = bf16_rn(r);
    }
  } else {
#pragma unroll
    for (int h4 = 0; h4 < kHeads / 4; ++h4) {
      const float4 v = iv[h4];
      orow[(4 * h4 + 0) * kDim + d] = bf16_rn(acc[4 * h4 + 0] * v.x);
      orow[(4 * h4 + 1) * kDim + d] = bf16_rn(acc[4 * h4 + 1] * v.y);
      orow[(4 * h4 + 2) * kDim + d] = bf16_rn(acc[4 * h4 + 2] * v.z);
      orow[(4 * h4 + 3) * kDim + d] = bf16_rn(acc[4 * h4 + 3] * v.w);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) {
    tc_fence_after();
    tc_dealloc(taddr, kTmemCols);
  }
}

}  // namespace spa

#include <stdexcept>
#include <string>

// lever SPA launch (raw device pointers as int64): q [rows, 32, 512] e4m3 (row stride q_stride bytes, 16-B aligned);
// swa_kv / c_kv flat [slots, 512] e4m3 pools; swa_idx [rows, 128] / topk_idx [rows, 512] int32 (row strides in
// elements); topk_len [rows] int32 (len64 0) / int64 (1); sinks [32] fp32; freqs [P, 64] fp32 + pos [rows] int32 / int64
// (pos64) when rope = 1; out [rows, 32, 512] bf16 contiguous; scale = softmax scale; pdl 1 = programmatic stream
// serialization; stream = the raw cudaStream_t
void spa_run(int64_t q, int64_t q_stride, int64_t swa_kv, int64_t c_kv, int64_t swa_idx, int64_t swa_stride,
             int64_t topk_idx, int64_t topk_stride, int64_t topk_len, int64_t len64, int64_t sinks, int64_t freqs,
             int64_t pos, int64_t pos64, int64_t out, int64_t rows, double scale, int64_t rope, int64_t pdl,
             int64_t split, int64_t zero_mode, int64_t seq_lens, int64_t stream) {
  if (split < 1 || split > 3) throw std::runtime_error("spa: split 1, 2 or 3");
  if (zero_mode < 0 || zero_mode > 2 || (zero_mode == 2 && (seq_lens == 0 || seq_lens % 4)))
    throw std::runtime_error("spa: zero_mode / seq_lens");
  if (rows < 1 || rows > 4096) throw std::runtime_error("spa: rows");
  if (q % 16 || q_stride % 16 || swa_kv % 16 || c_kv % 16 || out % 16) throw std::runtime_error("spa: alignment");
  if (swa_idx % 4 || topk_idx % 4 || sinks % 4) throw std::runtime_error("spa: index alignment");
  if (len64 != 0 && len64 != 1) throw std::runtime_error("spa: len64");
  if (rope != 0 && rope != 1) throw std::runtime_error("spa: rope");
  if (rope && (freqs % 16 || (pos64 != 0 && pos64 != 1))) throw std::runtime_error("spa: rope table");
  static bool attr_done = false;
  if (!attr_done) {
    cudaError_t e = cudaFuncSetAttribute(spa::attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(spa::kSmemBytes));
    if (e != cudaSuccess) throw std::runtime_error(std::string("spa: smem attribute: ") + cudaGetErrorString(e));
    attr_done = true;
  }
  spa::Params prm{};
  prm.q = reinterpret_cast<const uint8_t*>(q);
  prm.swa_kv = reinterpret_cast<const uint8_t*>(swa_kv);
  prm.c_kv = reinterpret_cast<const uint8_t*>(c_kv);
  prm.swa_idx = reinterpret_cast<const int32_t*>(swa_idx);
  prm.topk_idx = reinterpret_cast<const int32_t*>(topk_idx);
  prm.topk_len = reinterpret_cast<const void*>(topk_len);
  prm.sinks = reinterpret_cast<const float*>(sinks);
  prm.freqs = reinterpret_cast<const float*>(freqs);
  prm.pos = reinterpret_cast<const void*>(pos);
  prm.out = reinterpret_cast<uint16_t*>(out);
  prm.q_stride = q_stride;
  prm.swa_stride = swa_stride;
  prm.topk_stride = topk_stride;
  prm.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  prm.len64 = static_cast<int>(len64);
  prm.pos64 = static_cast<int>(pos64);
  prm.rope = static_cast<int>(rope);
  prm.split = static_cast<int>(split);
  prm.zero_mode = static_cast<int>(zero_mode);
  prm.seq_lens = reinterpret_cast<const int*>(seq_lens);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(rows * split));
  cfg.blockDim = dim3(spa::kThreads);
  cfg.dynamicSmemBytes = spa::kSmemBytes;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[2];
  int na = 0;
  if (pdl) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  if (split > 1) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = static_cast<unsigned>(split);
    attr[na].val.clusterDim.y = 1;
    attr[na].val.clusterDim.z = 1;
    ++na;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  const cudaError_t rc = cudaLaunchKernelEx(&cfg, spa::attn_kernel, prm);
  if (rc != cudaSuccess) throw std::runtime_error(std::string("spa: launch failed: ") + cudaGetErrorString(rc));
}

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

// x: rope tails [rows, heads, 64] bf16 (last dim contiguous) at token stride sb / head stride sh in elements, 16-byte
// aligned, rewritten in place; heads 16, 32 or 64; freqs: [*, 64] fp32 contiguous, 16-byte aligned (stock's
// view_as_real(freqs_cis).flatten(-2)); pos: [rows] int64 (pos64 = 1) or int32 contiguous; iters: heads per thread
// (1, 2, 4 or 8; a token = 8 * heads / iters threads); tokens: tokens per CTA (CTA <= 512 threads); fixed: 1 = the
// compile-time 512 head stride when sh == 512; pdl: 1 = launch with programmatic stream serialization (stock's
// kUsePDL), 0 = plain launch
void run(int64_t x, int64_t sb, int64_t sh, int64_t heads, int64_t freqs, int64_t pos, int64_t pos64, int64_t rows,
         int64_t iters, int64_t tokens, int64_t fixed, int64_t pdl, int64_t stream) {
  if (rows < 1 || rows > (int64_t(1) << 30)) throw std::runtime_error("ro: rows");
  if (heads != 16 && heads != 32 && heads != 64) throw std::runtime_error("ro: heads 16, 32 or 64");
  if (x % 16 || sb % 8 || sh % 8 || sh < ro::kRope) throw std::runtime_error("ro: 16-byte aligned tails");
  if (freqs % 16) throw std::runtime_error("ro: 16-byte aligned freqs");
  if (pos64 != 0 && pos64 != 1) throw std::runtime_error("ro: pos64 0 or 1");
  if (pos % (pos64 ? 8 : 4)) throw std::runtime_error("ro: aligned positions");
  if (iters != 1 && iters != 2 && iters != 4 && iters != 8) throw std::runtime_error("ro: iters 1, 2, 4 or 8");
  const int64_t per_token = 8 * heads / iters;
  if (tokens < 1 || tokens * per_token > 32 * ro::kMaxWarps) throw std::runtime_error("ro: tokens per CTA");
  if (fixed != 0 && fixed != 1) throw std::runtime_error("ro: fixed 0 or 1");
  if (pdl != 0 && pdl != 1) throw std::runtime_error("ro: pdl 0 or 1");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>((rows + tokens - 1) / tokens));
  cfg.blockDim = dim3(static_cast<unsigned>(tokens * per_token));
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const int r = static_cast<int>(rows), it = static_cast<int>(iters);
  const bool fx = fixed == 1;
  const cudaError_t rc = pos64 ? ro::launch_heads<int64_t>(cfg, x, sb, sh, heads, freqs, pos, it, fx, r)
                               : ro::launch_heads<int32_t>(cfg, x, sb, sh, heads, freqs, pos, it, fx, r);
  if (rc != cudaSuccess) throw std::runtime_error(std::string("ro: launch failed: ") + cudaGetErrorString(rc));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "DeepseekV4 attention-output inverse RoPE on the rope tails in place, stock's bytes");
  m.def("spa_run", &spa_run, "lever SPA: sparse MLA decode attention (+ inverse RoPE)");
}
