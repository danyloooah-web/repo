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
// lev_BSX v6 (kits/bsx6_unit.cu): as below with the split arrival / merge made cheaper (same bytes): the part's stores,
// a compute barrier, then ONE thread's __threadfence + atomicAdd (cooperative groups' grid-sync pattern, instead of
// 512 threads' fences); the merging CTA's thread 0 fences, then pulls every part's 32 KiB bf16 slice and its stats
// into the (free) KV stages with cp.async.bulk (one mbarrier, complete_tx), and the merge reads them from smem.
// lev_BSX (gen_bsx.py swaps this section in for lever BSA2's in the bundle copy of ro_unit.cu; entry bsx_run = bsa_run
// + the two pools' slot counts): lever BSA2's bitwise sparse MLA decode attention RE-SCHEDULED. Every float op, every
// MMA (shape, operands, K order, accumulate flags), the softmax / P quantization / denominator orders, the part split
// and the merge arithmetic are BSA2's (= stock trtllm-gen's), so the output is BIT-IDENTICAL to BSA2's and to stock's;
// only who issues what, and when, moves (lever SPX's techniques where stock's arithmetic allows them):
//   * a dedicated converged MMA-issuer warp (warp 16, elect.sync): S^T(i + 2) / PV(i) leave the softmax warps' barrier
//     chain (BSA2: tid 32 issued S^T(i + 1) between two CTA barriers, tid 0 issued PV(i), ~1k cycles each a tile)
//   * KV tiles on the TMA engine (cp.async.bulk.tensor tile::gather4, 2-D uint8 [slots, 512] maps, box 128 B x 1,
//     SW128 == swz(), row -1 = out-of-bounds zero fill == cp.async's src-size 0), 8 gather4 a warp a tile from lane 0,
//     one expect_tx of 64 KiB on the stage's barrier; the warps never block on the copies (BSA2: 16 warps x ~1k
//     cycles of cp.async issue a tile, 3 tiles of it in the prologue)
//   * one-round-trip prologue: the length, every slot index word, seq_len and the RoPE position loaded together (the
//     masks applied after: the same values), TMEM alloc + barrier init before griddepcontrol.wait
//   * S^T(i + 2) issued as soon as tile i's P is out (two S^T TMEM buffers, one barrier each)
//   * the split merge (rows <= 74, stock's MultiCtasKv parts in real CTAs): the merging CTA loads every part's (max,
//     den) and bf16 O slice in ONE round trip, then merges in part order with fmhaReduction's ops (BSA2: two dependent
//     round trips a part)
// Not taken from SPX (they would change bits): double O^T buffers / the deferred register merge (stock accumulates O in
// TMEM: O = O * alpha, then PV accumulates onto it), SPX's softmax. The one-CTA replay of the split (BSA2 ctas = 1) is
// not implemented: bsx_run takes ctas == vsplit only.
// Problem, -1 semantics, layouts: lever BSA2's (see its section); TMEM O^T [0, 128) (chunk c at 32 c), S^T at 128 /
// 160. PDL: every global read of a predecessor output (q, indices, lengths, KV) is issued after griddepcontrol.wait
// through cp.async.cg / cp.async.bulk.tensor / plain ld.global / ld.global.cg (no .nc, no __restrict__ const);
// launch_dependents after the tile loop.
// ===================================================================================================================
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <math_constants.h>

namespace bsx {

constexpr int kHeads = 32;
constexpr int kDim = 512;
constexpr int kTile = 128;
constexpr int kTiles = 5;
constexpr int kCompute = 512;                  // 16 compute warps: lane quarter q = warp & 3, column group g = warp >> 2
constexpr int kThreads = kCompute + 32;        // + the MMA issuer warp
constexpr int kIssuer = kCompute / 32;         // its warp index (16)
constexpr int kChunk = 128;
constexpr int kChunks = kDim / kChunk;
constexpr int kStages = 3;
constexpr int kKvChunk = kTile * kChunk;
constexpr int kKvStage = kChunks * kKvChunk;
constexpr int kQChunk = kHeads * kChunk;
constexpr int kQBytes = kChunks * kQChunk;
constexpr int kPBytes = kHeads * kTile;
constexpr int kMaxParts = 3;
constexpr uint32_t kTmemCols = 512;   // O^T [0, 128), S^T [128, 160) / [160, 192); 512 = one CTA an SM (as BSA2)
constexpr float kLog2_448 = 8.8073549270629882812f;
constexpr float kLog2e = 1.4426950216293334961f;
constexpr float kNegMax = -3.402823466e+38f;

struct Smem {
  uint8_t kv[kStages][kKvStage];
  uint8_t q[kQBytes];
  uint8_t p[2][kPBytes];
  float red_max[4][kHeads];
  float red_sum[4][kHeads];
  alignas(16) float ref[kHeads];
  alignas(16) float alpha[kHeads];
  alignas(16) float beta[kMaxParts][kHeads];
  alignas(16) float gam[kMaxParts][kHeads];
  alignas(16) float inv[kHeads];
  int moved;            // some head's alpha != 1 this tile (block-uniform)
  int last;             // ctas > 1: this CTA completed the row's arrivals (it merges)
  uint64_t bar_go;      // compute -> issuer: phase 0 = q landed; phase i + 1 = P(i) out + O corrected
  uint64_t bar_s[2];    // S^T(i) done: bar_s[i & 1], phase (i >> 1) & 1
  uint64_t bar_o;       // PV(i) done: phase i & 1
  uint64_t bar_kv[kStages];   // tile i's gather landed: bar_kv[i % 3], phase (i / 3) & 1
  uint64_t bar_m;       // ctas > 1, merging CTA: every part's slice + stats landed in smem (phase 0)
  uint32_t taddr;
};
constexpr size_t kSmemBytes = sizeof(Smem) + 1024;

struct Params {
  CUtensorMap tm_swa;      // 2-D [slots, 512] uint8 views of the pools, box 128 B x 1 row, 128B swizzle (gather4)
  CUtensorMap tm_c;
  const uint8_t* q;
  const int32_t* swa_idx;
  const int32_t* topk_idx;
  const void* topk_len;
  const float* sinks;
  const float* freqs;
  const void* pos;
  uint16_t* out;
  long long q_stride;
  long long swa_stride;
  long long topk_stride;
  float scale_log2;
  int len64;
  int pos64;
  int rope;
  int zero_mode;
  const int* seq_lens;
  int vsplit;
  int ctas;                // 1 (vsplit 1) or vsplit: one CTA a part
  uint16_t* part_o;        // ctas > 1: [rows * 3][512][32] bf16 partial O (bf16(O rcp 448))
  float* part_s;           // ctas > 1: [rows * 3][2][32] (raw running max, den) of each part
  int* counters;           // ctas > 1: [rows] arrivals (0 between launches: the merging CTA resets it)
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
      "BSX_WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra BSX_WAIT_%=;\n\t}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_arrive(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// the 512 compute threads' barrier (named barrier 1; the issuer warp never joins it)
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, 512;" ::: "memory"); }
__device__ __forceinline__ void tc_alloc(uint32_t dst, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst), "r"(cols));
}
__device__ __forceinline__ void tc_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tc_dealloc(uint32_t taddr, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols));
}
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
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
__device__ __forceinline__ void tc_st32(uint32_t taddr, const float (&v)[32]) {
  const uint32_t* d = reinterpret_cast<const uint32_t*>(v);
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"
      "%30,%31,%32};" ::"r"(taddr),
      "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]), "r"(d[4]), "r"(d[5]), "r"(d[6]), "r"(d[7]), "r"(d[8]), "r"(d[9]),
      "r"(d[10]), "r"(d[11]), "r"(d[12]), "r"(d[13]), "r"(d[14]), "r"(d[15]), "r"(d[16]), "r"(d[17]), "r"(d[18]),
      "r"(d[19]), "r"(d[20]), "r"(d[21]), "r"(d[22]), "r"(d[23]), "r"(d[24]), "r"(d[25]), "r"(d[26]), "r"(d[27]),
      "r"(d[28]), "r"(d[29]), "r"(d[30]), "r"(d[31])
      : "memory");
}
// 16 lanes x 256 bits (8 columns): thread T holds (lane T/4, cols 2(T%4), +1) then (lane T/4 + 8, same cols)
__device__ __forceinline__ void tc_ld16x256(uint32_t taddr, float (&v)[4]) {
  uint32_t* d = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(taddr));
}
// issuer warp (converged): one elected lane issues
__device__ __forceinline__ void tc_mma_e(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p, e;\n\tsetp.ne.b32 p, %4, 0;\n\telect.sync _|e, 0xffffffff;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void tc_commit_e(uint32_t bar) {
  asm volatile(
      "{\n\t.reg .pred e;\n\telect.sync _|e, 0xffffffff;\n\t"
      "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n\t}\n" ::"r"(bar)
      : "memory");
}
__device__ __forceinline__ uint64_t sdesc(uint32_t addr) {
  return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t(1024u >> 4) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}
__host__ __device__ constexpr uint32_t idesc_e4m3(uint32_t M, uint32_t N, bool a_mn) {
  return (1u << 4) | (a_mn ? (1u << 15) : 0u) | (((N >> 3) & 0x3Fu) << 17) | (((M >> 4) & 0x1Fu) << 24);
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// float ops with stock's rounding / FTZ (BSA2's, verbatim)
__device__ __forceinline__ float ex2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float rcp_approx(float a) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float r;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float add_ftz(float a, float b) {
  float r;
  asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float neg_f(float a) {
  float r;
  asm("neg.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {   // lo -> low byte
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint16_t bf16_rn(float x) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float bf16_round(float x) {
  return __uint_as_float(static_cast<uint32_t>(bf16_rn(x)) << 16);
}
__device__ __forceinline__ uint32_t swz(uint32_t row, uint32_t col) {
  return row * 128u + ((((col >> 4) ^ (row & 7u)) << 4) | (col & 15u));
}
__device__ __forceinline__ float ld_cg_f32(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void ld_cg_v4(const void* p, uint32_t (&w)[4]) {
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "l"(p));
}

// 1-D bulk copy global -> shared (16-B aligned, bytes % 16 == 0), completion counted on bar
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void fence_proxy_async_all() { asm volatile("fence.proxy.async;" ::: "memory"); }

// TMA gather4: rows r0..r3 (row < 0: out of bounds, zero fill) x 128 B at column col of a 2-D uint8 tensor map into
// 4 consecutive 128-B rows of smem, 128B-swizzled by address (== swz())
__device__ __forceinline__ void tma_gather4(uint32_t dst, const void* tm, int col, int r0, int r1, int r2, int r3,
                                            uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, "
      "%4, %5, %6}], [%7];" ::"r"(dst),
      "l"(tm), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(bar)
      : "memory");
}
// warp (q, g): slots 32 q + 8 g .. + 7 of a tile (BSA2's gather_tile's slots; lane l of the warp holds slot 32 q + l's
// index), as two gather4 quads x 4 chunks issued by lane 0; idx < 0 -> row -1 (zero fill, as BSA2's src-size 0)
__device__ __forceinline__ void tma_tile(uint32_t stage, const void* tm, int32_t own, uint32_t q, uint32_t g,
                                         uint32_t lane, uint32_t bar) {
  int32_t r[8];
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const int32_t v = __shfl_sync(0xFFFFFFFFu, own, 8u * g + i);
    r[i] = v < 0 ? -1 : v;
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
      const uint32_t slot0 = q * 32u + 8u * g + 4u * h;
#pragma unroll
      for (uint32_t c = 0; c < kChunks; ++c)
        tma_gather4(stage + c * kKvChunk + slot0 * kChunk, tm, static_cast<int>(c * kChunk), r[4 * h], r[4 * h + 1],
                    r[4 * h + 2], r[4 * h + 3], bar);
    }
  }
}

// S^T = K . Q^T of one tile (BSA2's issue_s: chunk outer, K block inner, one accumulator)
__device__ __forceinline__ void issue_s_e(uint32_t t_s, uint32_t stage, uint32_t q_base) {
  constexpr uint32_t kIdS = idesc_e4m3(kTile, kHeads, false);
  const uint64_t a0 = sdesc(stage), b0 = sdesc(q_base);
#pragma unroll
  for (uint32_t c = 0; c < kChunks; ++c)
#pragma unroll
    for (uint32_t kb = 0; kb < 4; ++kb)
      tc_mma_e(t_s, a0 + ((c * kKvChunk + kb * 32u) >> 4), b0 + ((c * kQChunk + kb * 32u) >> 4), kIdS, (c | kb) != 0);
}
// O^T (+)= V^T . P of one tile (BSA2's PV: chunk c's accumulator at 32 c, K blocks of 32 slots in order; the first
// tile of the CTA's part starts the accumulator, every later one accumulates onto the corrected O)
__device__ __forceinline__ void issue_pv_e(uint32_t t_o, uint32_t stage, uint32_t p_base, bool first) {
  constexpr uint32_t kIdO = idesc_e4m3(kChunk, kHeads, true);
  const uint64_t a0 = sdesc(stage), b0 = sdesc(p_base);
#pragma unroll
  for (uint32_t c = 0; c < kChunks; ++c)
#pragma unroll
    for (uint32_t kb = 0; kb < 4; ++kb)
      tc_mma_e(t_o + 32u * c, a0 + ((c * kKvChunk + kb * 32u * kChunk) >> 4), b0 + ((kb * 32u) >> 4), kIdO,
               (kb != 0 || !first) ? 1u : 0u);
}

__device__ __forceinline__ int32_t pick(const int32_t (&own)[kTiles], int t) {
  int32_t v = own[0];
#pragma unroll
  for (int u = 1; u < kTiles; ++u)
    if (u == t) v = own[u];
  return v;
}

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
  const uint32_t ctas = static_cast<uint32_t>(p.ctas);
  const uint32_t row = blockIdx.x / ctas;
  const uint32_t kpart = blockIdx.x % ctas;
  const uint32_t slot = q4 * 32u + lane;
  const uint32_t a = lane >> 2;
  const uint32_t h0 = 8u * g4 + 2u * (lane & 3u);

  // TMEM + barriers before the predecessor's data is needed (nothing here reads it)
  if (warp == kIssuer) {
    tc_alloc(smem_u32(&s.taddr), kTmemCols);
    tc_relinquish();
  } else if (tid == 0) {
    mbar_init(smem_u32(&s.bar_go), 1);
    mbar_init(smem_u32(&s.bar_s[0]), 1);
    mbar_init(smem_u32(&s.bar_s[1]), 1);
    mbar_init(smem_u32(&s.bar_o), 1);
    mbar_init(smem_u32(&s.bar_kv[0]), 1);
    mbar_init(smem_u32(&s.bar_kv[1]), 1);
    mbar_init(smem_u32(&s.bar_kv[2]), 1);
    mbar_init(smem_u32(&s.bar_m), 1);
    fence_mbar_init();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t taddr = s.taddr;
  const uint32_t t_o = taddr;
  const uint32_t t_s0 = taddr + 128u;
  const uint32_t lane_off = (q4 * 32u) << 16;
  const uint32_t q_base = smem_u32(s.q);
  const uint32_t kv_base = smem_u32(s.kv[0]);

  pdl_wait();
  // ---- one round trip: the length, every index word of this thread's slot, seq_len, the RoPE position; q on cp.async
  const long long len_raw = p.len64 ? static_cast<const long long*>(p.topk_len)[row]
                                    : static_cast<long long>(static_cast<const int*>(p.topk_len)[row]);
  int32_t own[kTiles];
  int seq = kTile;
  long long pos = 0;
  const bool rope_lane = p.rope && warp < kIssuer && g4 == kChunks - 1 && q4 >= 2;
  if (warp < kIssuer) {
    const uint8_t* q_row = p.q + static_cast<long long>(row) * p.q_stride;
#pragma unroll
    for (uint32_t i = 0; i < 1024 / kCompute; ++i) {
      const uint32_t piece = i * kCompute + tid;
      const uint32_t head = piece >> 5;
      const uint32_t l = piece & 31u;
      cp16(q_base + (l >> 3) * kQChunk + swz(head, (l & 7u) * 16u), q_row + head * kDim + l * 16u, 16u);
    }
    cp_commit();
    own[0] = p.swa_idx[static_cast<long long>(row) * p.swa_stride + slot];
#pragma unroll
    for (int t = 1; t < kTiles; ++t)
      own[t] = p.topk_idx[static_cast<long long>(row) * p.topk_stride + (t - 1) * kTile + static_cast<int>(slot)];
    if (p.zero_mode == 2) seq = p.seq_lens[row];
    if (rope_lane)
      pos = p.pos64 ? static_cast<const long long*>(p.pos)[row]
                    : static_cast<long long>(static_cast<const int*>(p.pos)[row]);
  }
  const int len = static_cast<int>(len_raw < 0 ? 0 : (len_raw > 512 ? 512 : len_raw));
  const int nt = 1 + (len + kTile - 1) / kTile;
  const int vs = p.vsplit > 1 ? p.vsplit : 1;
  const int nk = nt < vs ? nt : vs;
  const int tpc = (nt + nk - 1) / nk;
  const int n_parts = (nt + tpc - 1) / tpc;
  const int t0 = ctas > 1 ? static_cast<int>(kpart) * tpc : 0;
  if (ctas > 1 && static_cast<int>(kpart) >= n_parts) {
    // an empty part: nothing to do (the row's count waits for n_parts arrivals only)
    if (warp < kIssuer) cp_wait_all();
    if (warp == kIssuer) tc_dealloc(taddr, kTmemCols);
    return;
  }
  const int cnt = ctas > 1 ? (nt - t0 < tpc ? nt - t0 : tpc) : nt;

  if (warp == kIssuer) {
    // ---- MMA issuer: S^T(0), S^T(1); then per tile i: PV(i) once P(i) is out and O corrected, S^T(i + 2) ---------
    mbar_wait(smem_u32(&s.bar_go), 0u);
    mbar_wait(smem_u32(&s.bar_kv[0]), 0u);
    tc_fence_after();
    issue_s_e(t_s0, kv_base, q_base);
    tc_commit_e(smem_u32(&s.bar_s[0]));
    if (cnt > 1) {
      mbar_wait(smem_u32(&s.bar_kv[1]), 0u);
      tc_fence_after();
      issue_s_e(t_s0 + 32u, kv_base + kKvStage, q_base);
      tc_commit_e(smem_u32(&s.bar_s[1]));
    }
#pragma unroll 1
    for (int i = 0; i < cnt; ++i) {
      mbar_wait(smem_u32(&s.bar_go), static_cast<uint32_t>((i + 1) & 1));
      tc_fence_after();
      issue_pv_e(t_o, kv_base + (i % kStages) * kKvStage, smem_u32(s.p[i & 1]), i == 0);
      tc_commit_e(smem_u32(&s.bar_o));
      if (i + 2 < cnt) {
        mbar_wait(smem_u32(&s.bar_kv[(i + 2) % kStages]), static_cast<uint32_t>(((i + 2) / kStages) & 1));
        tc_fence_after();
        issue_s_e(t_s0 + 32u * static_cast<uint32_t>(i & 1), kv_base + ((i + 2) % kStages) * kKvStage, q_base);
        tc_commit_e(smem_u32(&s.bar_s[i & 1]));
      }
    }
    pdl_launch();
  } else {
    // ---- compute warps -----------------------------------------------------------------------------------------
    if (p.zero_mode == 2 && static_cast<int>(slot) >= seq) own[0] = -2;
#pragma unroll
    for (int t = 1; t < kTiles; ++t)
      if ((t - 1) * kTile + static_cast<int>(slot) >= len) own[t] = -2;
    // tiles 0..2 of this CTA's part on the TMA engine
#pragma unroll
    for (int i = 0; i < kStages; ++i) {
      if (i < cnt) {
        const uint32_t bar = smem_u32(&s.bar_kv[i]);
        if (tid == 0) mbar_expect_arrive(bar, static_cast<uint32_t>(kKvStage));
        tma_tile(kv_base + i * kKvStage,
                 t0 + i == 0 ? static_cast<const void*>(&p.tm_swa) : static_cast<const void*>(&p.tm_c),
                 pick(own, t0 + i), q4, g4, lane, bar);
      }
    }
    float fr = 0.f, fi = 0.f;
    if (rope_lane) {
      const uint32_t j = (q4 * 32u + lane - 64u) >> 1;
      fr = p.freqs[pos * 64 + 2 * j];
      fi = p.freqs[pos * 64 + 2 * j + 1];
    }
    cp_wait_all();   // q
    fence_async_smem();
    cbar();
    if (tid == 0) mbar_arrive(smem_u32(&s.bar_go));   // go phase 0
    // warp 0, lane h: head h's running raw max of this CTA's part
    float run_m = kNegMax;
    // per-thread denominator partials (heads h0, h0 + 1), stock's layout and order
    float d0 = 0.f, d1 = 0.f;

#pragma unroll 1
    for (int i = 0; i < cnt; ++i) {
      const uint32_t t_s = t_s0 + 32u * static_cast<uint32_t>(i & 1);
      const int32_t my_idx = pick(own, t0 + i);
      const bool valid_own = my_idx >= 0 || (my_idx == -1 && p.zero_mode != 0);
      const uint32_t vmask = __ballot_sync(0xFFFFFFFFu, valid_own);   // bit l = slot 32 q4 + l scores
      mbar_wait(smem_u32(&s.bar_s[i & 1]), static_cast<uint32_t>((i >> 1) & 1));
      tc_fence_after();
      // ---- S^T in stock's 16x256b layout: x[2k + j] = (slot 32 q4 + a + 8k, head h0 + j) ----------------------------
      float x[8];
      {
        float v0[4], v1[4];
        tc_ld16x256(t_s + lane_off + 8u * g4, v0);
        tc_ld16x256(t_s + lane_off + (16u << 16) + 8u * g4, v1);
        tc_wait_ld();
        x[0] = v0[0]; x[1] = v0[1]; x[2] = v0[2]; x[3] = v0[3];
        x[4] = v1[0]; x[5] = v1[1]; x[6] = v1[2]; x[7] = v1[3];
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const bool ok = (vmask >> (a + 8u * k)) & 1u;
        x[2 * k] = ok ? x[2 * k] : kNegMax;
        x[2 * k + 1] = ok ? x[2 * k + 1] : kNegMax;
      }
      {
        float m0 = fmaxf(fmaxf(x[0], x[2]), fmaxf(x[4], x[6]));
        float m1 = fmaxf(fmaxf(x[1], x[3]), fmaxf(x[5], x[7]));
#pragma unroll
        for (uint32_t off = 4; off <= 16; off <<= 1) {
          m0 = fmaxf(m0, __shfl_xor_sync(0xFFFFFFFFu, m0, off));
          m1 = fmaxf(m1, __shfl_xor_sync(0xFFFFFFFFu, m1, off));
        }
        if (lane < 4) {
          s.red_max[q4][h0] = m0;
          s.red_max[q4][h0 + 1] = m1;
        }
      }
      cbar();
      if (warp == 0) {
        const float m_t = fmaxf(fmaxf(s.red_max[0][lane], s.red_max[1][lane]),
                                fmaxf(s.red_max[2][lane], s.red_max[3][lane]));
        const float m_new = fmaxf(run_m, m_t);
        const float dm = add_rn(run_m, neg_f(m_new));
        const float alpha = dm != 0.f ? ex2(mul_rn(p.scale_log2, dm)) : 1.f;
        run_m = m_new;
        s.ref[lane] = fma_rn(m_new, neg_f(p.scale_log2), kLog2_448);
        s.alpha[lane] = alpha;
        const uint32_t mv = __ballot_sync(0xFFFFFFFFu, alpha != 1.f);
        if (lane == 0) s.moved = mv != 0u;
      }
      cbar();
      // ---- P (e4m3, P^T [head][slot] in buffer i & 1) and the per-thread denominator partials ------------------------
      {
        const float r0 = s.ref[h0], r1 = s.ref[h0 + 1];
        const float al0 = s.alpha[h0], al1 = s.alpha[h0 + 1];
        float e[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          e[2 * k] = ex2(fma_rn(x[2 * k], p.scale_log2, r0));
          e[2 * k + 1] = ex2(fma_rn(x[2 * k + 1], p.scale_log2, r1));
        }
        d0 = add_rn(add_rn(add_rn(fma_rn(al0, d0, e[0]), e[2]), e[4]), e[6]);
        d1 = add_rn(add_rn(add_rn(fma_rn(al1, d1, e[1]), e[3]), e[5]), e[7]);
        uint8_t* pb = s.p[i & 1];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const uint32_t sl = q4 * 32u + a + 8u * static_cast<uint32_t>(k);
          const uint32_t pair = e4m3x2(e[2 * k], e[2 * k + 1]);
          pb[swz(h0, sl)] = static_cast<uint8_t>(pair & 0xFFu);
          pb[swz(h0 + 1, sl)] = static_cast<uint8_t>(pair >> 8);
        }
      }
      // ---- O correction in TMEM (O = O * alpha) before this tile's accumulating PV; frees tile i - 1's stage --------
      if (i > 0) {
        mbar_wait(smem_u32(&s.bar_o), static_cast<uint32_t>((i - 1) & 1));   // PV(i - 1)
        tc_fence_after();
        if (s.moved) {
          float o[kHeads];
          const uint32_t ta = t_o + 32u * g4 + lane_off;
          tc_ld32(ta, o);
          tc_wait_ld();
          const float4* al = reinterpret_cast<const float4*>(s.alpha);
#pragma unroll
          for (int h4 = 0; h4 < kHeads / 4; ++h4) {
            const float4 w = al[h4];
            o[4 * h4 + 0] = mul_rn(o[4 * h4 + 0], w.x);
            o[4 * h4 + 1] = mul_rn(o[4 * h4 + 1], w.y);
            o[4 * h4 + 2] = mul_rn(o[4 * h4 + 2], w.z);
            o[4 * h4 + 3] = mul_rn(o[4 * h4 + 3], w.w);
          }
          tc_st32(ta, o);
          tc_wait_st();
        }
        const int j = i - 1;   // its stage is free (PV(j) complete): gather tile j + 3 into it
        if (j + kStages < cnt) {
          const uint32_t bar = smem_u32(&s.bar_kv[j % kStages]);
          if (tid == 0) mbar_expect_arrive(bar, static_cast<uint32_t>(kKvStage));
          tma_tile(kv_base + (j % kStages) * kKvStage, &p.tm_c, pick(own, t0 + j + kStages), q4, g4, lane, bar);
        }
      }
      fence_async_smem();
      tc_fence_before();
      cbar();
      if (tid == 0) mbar_arrive(smem_u32(&s.bar_go));   // go phase i + 1: PV(i)
    }
    pdl_launch();

    // ---- den = reduce(partials) (butterfly xor 16, 8, 4; then the four lane quarters in order) ---------------------
    {
      float r0 = d0, r1 = d1;
#pragma unroll
      for (uint32_t off = 16; off >= 4; off >>= 1) {
        r0 = add_rn(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, off));
        r1 = add_rn(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, off));
      }
      if (lane < 4) {
        s.red_sum[q4][h0] = r0;
        s.red_sum[q4][h0 + 1] = r1;
      }
    }
    mbar_wait(smem_u32(&s.bar_o), static_cast<uint32_t>((cnt - 1) & 1));   // the last PV
    tc_fence_after();
    cbar();
    float acc[kHeads];
    bool emit = true;
    const uint32_t dd = g4 * kChunk + q4 * 32u + lane;
    if (ctas > 1) {
      // ---- one CTA a part: bf16(O rcp 448) + (max, den) to the scratch; the row's last arriving CTA merges every
      // part in part order (fmhaReduction) from ONE round trip of loads ------------------------------------------------
      const long long sb = static_cast<long long>(row) * 3 + kpart;
      if (warp == 0) {
        const float dk = add_rn(add_rn(add_rn(s.red_sum[0][lane], s.red_sum[1][lane]), s.red_sum[2][lane]),
                                s.red_sum[3][lane]);
        p.part_s[sb * 64 + lane] = run_m;
        p.part_s[sb * 64 + 32 + lane] = dk;
      }
      {
        const float ps = rcp_approx(448.0f);
        float o[kHeads];
        tc_ld32(t_o + 32u * g4 + lane_off, o);
        tc_wait_ld();
        uint4* dst = reinterpret_cast<uint4*>(p.part_o + (sb * kDim + dd) * kHeads);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          uint32_t w[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int h = 8 * v + 2 * j;
            w[j] = static_cast<uint32_t>(bf16_rn(mul_rn(o[h], ps))) |
                   (static_cast<uint32_t>(bf16_rn(mul_rn(o[h + 1], ps))) << 16);
          }
          dst[v] = make_uint4(w[0], w[1], w[2], w[3]);
        }
      }
      cbar();   // every thread's part stores issued (bar.sync orders them before thread 0's fence: cumulativity)
      if (tid == 0) {
        __threadfence();
        const int last = atomicAdd(p.counters + row, 1) == n_parts - 1;
        if (last) {
          __threadfence();
          fence_proxy_async_all();   // the parts' generic stores -> this CTA's async-proxy (bulk) reads
          const uint32_t bar = smem_u32(&s.bar_m);
          mbar_expect_arrive(bar, static_cast<uint32_t>(n_parts) * (kDim * kHeads * 2u + 256u));
          for (int k = 0; k < n_parts; ++k) {
            const long long kb = static_cast<long long>(row) * 3 + k;
            bulk_g2s(smem_u32(s.kv[k]), p.part_o + kb * kDim * kHeads, kDim * kHeads * 2u, bar);
            bulk_g2s(smem_u32(s.p[0]) + 256u * k, p.part_s + kb * 64, 256u, bar);
          }
          p.counters[row] = 0;   // the next launch starts from 0
        }
        s.last = last;
      }
      cbar();
      emit = s.last != 0;
      if (emit) {
        mbar_wait(smem_u32(&s.bar_m), 0u);
        const float* st = reinterpret_cast<const float*>(s.p[0]);   // part k: (max [32], den [32]) at 64 k
        if (warp == 0) {
          float mrg_m = kNegMax, mrg_d = 0.f;
#pragma unroll
          for (int k = 0; k < kMaxParts; ++k) {
            if (k < n_parts) {
              const float mk = st[64 * k + lane], dk = st[64 * k + 32 + lane];
              const float M = fmaxf(mrg_m, mk);
              const float c0 = ex2(mul_ftz(add_ftz(mrg_m, neg_f(M)), p.scale_log2));
              const float c1 = ex2(mul_ftz(add_ftz(mk, neg_f(M)), p.scale_log2));
              mrg_d = fma_ftz(c0, mrg_d, mul_ftz(dk, c1));
              mrg_m = M;
              s.gam[k][lane] = c0;
              s.beta[k][lane] = c1;
            }
          }
          const float e = ex2(fma_ftz(p.sinks[lane], kLog2e, neg_f(mul_ftz(mrg_m, p.scale_log2))));
          const float dsum = fma_ftz(448.0f, e, mrg_d);
          s.inv[lane] = mul_ftz(448.0f, rcp_approx(dsum));
        }
        cbar();
#pragma unroll
        for (int h = 0; h < kHeads; ++h) acc[h] = 0.f;
#pragma unroll
        for (int k = 0; k < kMaxParts; ++k) {
          if (k < n_parts) {
            const uint4* src = reinterpret_cast<const uint4*>(s.kv[k] + dd * (kHeads * 2u));
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              const uint4 q4w = src[v];
              const uint32_t w[4] = {q4w.x, q4w.y, q4w.z, q4w.w};
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                const int h = 8 * v + 2 * j;
                const float lo = __uint_as_float(w[j] << 16), hi = __uint_as_float(w[j] & 0xFFFF0000u);
                acc[h] = fma_rn(lo, s.beta[k][h], mul_rn(acc[h], s.gam[k][h]));
                acc[h + 1] = fma_rn(hi, s.beta[k][h + 1], mul_rn(acc[h + 1], s.gam[k][h + 1]));
              }
            }
          }
        }
      }
    } else {
      // stock's single-CTA epilogue: den = reduce(partials) + 448 ex2(fma(sink, log2e, -(m sl2))); out = O rcp(den)
      if (warp == 0) {
        const float dk = add_rn(add_rn(add_rn(s.red_sum[0][lane], s.red_sum[1][lane]), s.red_sum[2][lane]),
                                s.red_sum[3][lane]);
        const float e = ex2(fma_ftz(p.sinks[lane], kLog2e, neg_f(mul_ftz(run_m, p.scale_log2))));
        s.inv[lane] = rcp_approx(add_ftz(dk, mul_ftz(e, 448.0f)));
      }
      cbar();
      tc_ld32(t_o + 32u * g4 + lane_off, acc);
      tc_wait_ld();
    }

    if (emit) {
      // ---- out[row][h][d], d = 128 g4 + 32 q4 + lane; the inverse RoPE on d >= 448 (g4 3, q4 2 / 3) -----------------
      uint16_t* orow = p.out + static_cast<long long>(row) * (kHeads * kDim);
      if (rope_lane) {
        const bool real = (lane & 1u) == 0;
#pragma unroll
        for (int h = 0; h < kHeads; ++h) {
          const uint16_t b = bf16_rn(mul_rn(acc[h], s.inv[h]));
          const float own_v = __uint_as_float(static_cast<uint32_t>(b) << 16);
          const float other = __shfl_xor_sync(0xFFFFFFFFu, own_v, 1);
          const float t2 = mul_rn(fi, other);
          const float r = fma_rn(fr, own_v, real ? t2 : neg_f(t2));
          orow[h * kDim + dd] = bf16_rn(r);
        }
      } else {
        const float4* iv = reinterpret_cast<const float4*>(s.inv);
#pragma unroll
        for (int h4 = 0; h4 < kHeads / 4; ++h4) {
          const float4 v = iv[h4];
          orow[(4 * h4 + 0) * kDim + dd] = bf16_rn(mul_rn(acc[4 * h4 + 0], v.x));
          orow[(4 * h4 + 1) * kDim + dd] = bf16_rn(mul_rn(acc[4 * h4 + 1], v.y));
          orow[(4 * h4 + 2) * kDim + dd] = bf16_rn(mul_rn(acc[4 * h4 + 2], v.z));
          orow[(4 * h4 + 3) * kDim + dd] = bf16_rn(mul_rn(acc[4 * h4 + 3], v.w));
        }
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == kIssuer) {
    tc_fence_after();
    tc_dealloc(taddr, kTmemCols);
  }
}

}  // namespace bsx

#include <stdexcept>
#include <string>
#include <vector>

namespace bsx {
PFN_cuTensorMapEncodeTiled_v12000 g_encode = nullptr;
struct MapEnt {
  int64_t ptr;
  int64_t rows;
  CUtensorMap tm;
};
std::vector<MapEnt> g_maps;   // per pool (ptr, slots): its gather4 map, encoded once (host only)

// 2-D uint8 view [rows][512] of a pool, box 128 B x 1 row (tile::gather4 loads 4 rows), 128B swizzle, zero OOB fill
CUtensorMap pool_map(int64_t ptr, int64_t rows) {
  for (const MapEnt& e : g_maps)
    if (e.ptr == ptr && e.rows == rows) return e.tm;
  if (!g_encode) {
    cudaDriverEntryPointQueryResult qr;
    void* fn = nullptr;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &qr) != cudaSuccess ||
        qr != cudaDriverEntryPointSuccess || fn == nullptr)
      throw std::runtime_error("bsx: cuTensorMapEncodeTiled not available");
    g_encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }
  MapEnt e;
  e.ptr = ptr;
  e.rows = rows;
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(kDim), static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(kDim)};
  cuuint32_t box[2] = {static_cast<cuuint32_t>(kChunk), 1};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = g_encode(&e.tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, reinterpret_cast<void*>(ptr), dims, strides, box,
                              es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("bsx: cuTensorMapEncodeTiled failed " + std::to_string(int(r)));
  if (g_maps.size() >= 512) g_maps.erase(g_maps.begin());
  g_maps.push_back(e);
  return e.tm;
}
}  // namespace bsx

// lever BSX launch = lever BSA2's bsa_run arguments (ctas must equal vsplit) + swa_rows / c_rows, the pools' slot
// counts (the TMA maps' row dims; every index must be below them, as for any read of the pool)
void bsx_run(int64_t q, int64_t q_stride, int64_t swa_kv, int64_t c_kv, int64_t swa_idx, int64_t swa_stride,
             int64_t topk_idx, int64_t topk_stride, int64_t topk_len, int64_t len64, int64_t sinks, int64_t freqs,
             int64_t pos, int64_t pos64, int64_t out, int64_t rows, double scale, int64_t rope, int64_t pdl,
             int64_t zero_mode, int64_t seq_lens, int64_t vsplit, int64_t ctas, int64_t part_o, int64_t part_s,
             int64_t counters, int64_t stream, int64_t swa_rows, int64_t c_rows) {
  if (vsplit < 1 || vsplit > 3) throw std::runtime_error("bsx: vsplit 1, 2 or 3");
  if (ctas != vsplit) throw std::runtime_error("bsx: ctas must equal vsplit (the one-CTA split replay is BSA2's)");
  if (ctas > 1 && (part_o == 0 || part_o % 16 || part_s == 0 || part_s % 16 || counters == 0 || counters % 4 ||
                   rows * ctas > 148 * 3 || rows > 148))
    throw std::runtime_error("bsx: split scratch");
  if (zero_mode < 0 || zero_mode > 2 || (zero_mode == 2 && (seq_lens == 0 || seq_lens % 4)))
    throw std::runtime_error("bsx: zero_mode / seq_lens");
  if (rows < 1 || rows > 4096) throw std::runtime_error("bsx: rows");
  if (q % 16 || q_stride % 16 || swa_kv % 16 || c_kv % 16 || out % 16) throw std::runtime_error("bsx: alignment");
  if (swa_idx % 4 || topk_idx % 4 || sinks % 4) throw std::runtime_error("bsx: index alignment");
  if (len64 != 0 && len64 != 1) throw std::runtime_error("bsx: len64");
  if (rope != 0 && rope != 1) throw std::runtime_error("bsx: rope");
  if (rope && (freqs % 16 || (pos64 != 0 && pos64 != 1))) throw std::runtime_error("bsx: rope table");
  if (swa_rows < 1 || c_rows < 1 || swa_rows > (1LL << 31) || c_rows > (1LL << 31))
    throw std::runtime_error("bsx: pool rows");
  static bool attr_done = false;
  if (!attr_done) {
    cudaError_t e = cudaFuncSetAttribute(bsx::attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(bsx::kSmemBytes));
    if (e != cudaSuccess) throw std::runtime_error(std::string("bsx: smem attribute: ") + cudaGetErrorString(e));
    attr_done = true;
  }
  bsx::Params prm{};
  prm.tm_swa = bsx::pool_map(swa_kv, swa_rows);
  prm.tm_c = bsx::pool_map(c_kv, c_rows);
  prm.q = reinterpret_cast<const uint8_t*>(q);
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
  prm.zero_mode = static_cast<int>(zero_mode);
  prm.seq_lens = reinterpret_cast<const int*>(seq_lens);
  prm.vsplit = static_cast<int>(vsplit);
  prm.ctas = static_cast<int>(ctas);
  prm.part_o = reinterpret_cast<uint16_t*>(part_o);
  prm.part_s = reinterpret_cast<float*>(part_s);
  prm.counters = reinterpret_cast<int*>(counters);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(rows * ctas));
  cfg.blockDim = dim3(bsx::kThreads);
  cfg.dynamicSmemBytes = bsx::kSmemBytes;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  int na = 0;
  if (pdl) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  const cudaError_t rc = cudaLaunchKernelEx(&cfg, bsx::attn_kernel, prm);
  if (rc != cudaSuccess) throw std::runtime_error(std::string("bsx: launch failed: ") + cudaGetErrorString(rc));
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
  m.def("bsx_run", &bsx_run, "lever BSX: bitwise sparse MLA decode attention (+ inverse RoPE), BSA2 re-scheduled");
}
