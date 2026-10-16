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
// lev_BSA (gen_bsa.py splices this section into the bundle COPY of ro_unit.cu, before its pybind11 block, and adds
// m.def("bsa_run", ...)): sparse MLA decode attention with stock trtllm-gen's arithmetic op for op (FlashInfer 0.7.1
// fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1{MultiCtasKvVarSeq..Static,VarSeq..Persistent}
// SwapsAbForGen, read from their SASS, + the open fmhaReduction.cu merge), so that its output is BIT-IDENTICAL to
// stock's; lever SPM's kernel with the two remaining differences removed:
//   * PV accumulation: O lives in TMEM across a part's tiles; before tile t's PV MMA the running O is rescaled in
//     TMEM (O = O * alpha_t, fp32 mul.rn, skipped when every head's alpha is 1 = exact) and the PV MMAs accumulate
//     onto it (stock's correction warpgroup + accumulate-on BMM2) [SPM: per-tile O drained and fma'd in registers]
//   * the denominator: per-thread partials in stock's TMEM-load layout (tcgen05.ld 16x256b: thread T of the warp
//     reading lane quarter q holds slots 32 q + T / 4 + {0, 8, 16, 24} x heads 8 g + 2 (T % 4) + {0, 1}):
//     d = fma(alpha, d, p(a)) + p(a + 8) + p(a + 16) + p(a + 24) per tile (fp32 p, before the e4m3 cast), reduced at
//     the part's end by the butterfly d += shfl_xor(d, 16), 8, 4 and then over the four lane quarters in order
//     ((W0 + W1) + W2) + W3 [SPM: per-tile warp tree + fma(den, alpha, l_t)]
// Schedule: thread 32 issues tile t + 1's S^T right after tile t's max exchange (it overlaps tile t's softmax), thread
// 0 issues tile t's PV after the O correction; PV(t) completes under softmax(t + 1).
// Kept from SPM (already stock's): P = e4m3(ex2(fma(s, sl2, fma(m_run, -sl2, log2 448)))) against the running raw
// max (masked -FLT_MAX), alpha = ex2(sl2 (m_old - m_new)), the S MMA chain, the sink / rcp epilogue, rows <= 74:
// stock's MultiCtasKv split (min(3, 148 / rows) parts of ceil(T / nk) contiguous tiles, part O -> bf16(O rcp 448),
// fmhaReduction's merge order and fma forms) replayed inside ONE CTA (merged F in TMEM cols 256..383).
// Every float op is explicit PTX with stock's rounding / FTZ choice (the unit builds with --use_fast_math).
//
// Problem (TP2 rank): q [R, 32 heads, 512] e4m3 (RoPE applied), one shared K = V row of 512 e4m3 per slot, per row 128
// SWA slots (swa_idx, -1 = empty) + the first len[r] of 512 compressed slots (topk_idx), a per-head sink logit; out
// bf16 [R, 32, 512], the last 64 dims of every head rotated back (lever RO's per-pair arithmetic) when rope = 1.
// -1 semantics (zero_mode 2 = trtllm-gen's): SWA columns at or past seq_len ignored; -1 below it and -1 among the first
// len top-k columns = zero K/V rows with logit 0; top-k columns past len ignored.
// One CTA per row, 512 threads = 16 warps (warp w = 4 g + q: TMEM lane quarter q, column group g), the row's table
// walked as up to five 128-slot tiles (tile 0 = SWA, tiles 1..4 = top-k; tiles past len skipped), 3 KV stages
// (cp.async 16-B pieces into a 128B-swizzled [4 chunks][128 slots][128 B] stage), S^T = K . Q^T (tcgen05 kind::f8f6f4
// M 128 slots N 32 heads K 512, two TMEM buffers), O^T = V^T . P (A = the KV stage MN-major, B = P^T [32][128] e4m3,
// two smem buffers) accumulated in TMEM cols [0, 128).
// PDL: every global read of a predecessor output (q, indices, lengths, KV) is issued after griddepcontrol.wait through
// cp.async.cg / plain ld.global (no .nc, no __restrict__ const); launch_dependents after the tile loop.
// ===================================================================================================================
#include <cuda_fp8.h>
#include <math_constants.h>

namespace bsa {

constexpr int kHeads = 32;
constexpr int kDim = 512;
constexpr int kTile = 128;
constexpr int kTiles = 5;
constexpr int kThreads = 512;
constexpr int kChunk = 128;
constexpr int kChunks = kDim / kChunk;
constexpr int kStages = 3;
constexpr int kKvChunk = kTile * kChunk;
constexpr int kKvStage = kChunks * kKvChunk;
constexpr int kQChunk = kHeads * kChunk;
constexpr int kQBytes = kChunks * kQChunk;
constexpr int kPBytes = kHeads * kTile;
constexpr uint32_t kTmemCols = 512;   // O^T [0, 128) (chunk c at 32 c), S^T [128, 192) (2 bufs), merged F [256, 384)
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
  alignas(16) float beta[kHeads];
  alignas(16) float gam[kHeads];
  alignas(16) float inv[kHeads];
  int moved;            // some head's alpha != 1 this tile (block-uniform)
  uint64_t bar_s;
  uint64_t bar_o;
  uint32_t taddr;
};
constexpr size_t kSmemBytes = sizeof(Smem) + 1024;

struct Params {
  const uint8_t* q;
  const uint8_t* swa_kv;
  const uint8_t* c_kv;
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
      "BSA_WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra BSA_WAIT_%=;\n\t}\n" ::"r"(bar),
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
__device__ __forceinline__ void tc_mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
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
__device__ __forceinline__ void cp_wait(int pending) {
  if (pending <= 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
  else if (pending == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else asm volatile("cp.async.wait_group 2;" ::: "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// float ops with stock's rounding / FTZ (MUFU.EX2 / MUFU.RCP; FFMA.FTZ etc. for the scalar statistics, the non-FTZ
// FFMA2 / FADD2 / FMUL2 forms for the vector work)
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

__device__ __forceinline__ void issue_s(uint32_t t_s, uint32_t stage, uint32_t q_base) {
  constexpr uint32_t kIdS = idesc_e4m3(kTile, kHeads, false);
  const uint64_t a0 = sdesc(stage), b0 = sdesc(q_base);
#pragma unroll
  for (uint32_t c = 0; c < kChunks; ++c)
#pragma unroll
    for (uint32_t kb = 0; kb < 4; ++kb)
      tc_mma(t_s, a0 + ((c * kKvChunk + kb * 32u) >> 4), b0 + ((c * kQChunk + kb * 32u) >> 4), kIdS, (c | kb) != 0);
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
  const uint32_t row = blockIdx.x;
  const uint32_t slot = q4 * 32u + lane;           // this thread's gather / index slot
  // softmax layout (stock's tcgen05.ld 16x256b): slots 32 q4 + a + 8 k (k = 0..3), heads h0 = 8 g4 + 2 (lane & 3), h0 + 1
  const uint32_t a = lane >> 2;
  const uint32_t h0 = 8u * g4 + 2u * (lane & 3u);

  if (warp == 0) {
    tc_alloc(smem_u32(&s.taddr), kTmemCols);
    tc_relinquish();
  } else if (tid == 32) {
    mbar_init(smem_u32(&s.bar_s), 1);
    mbar_init(smem_u32(&s.bar_o), 1);
    fence_mbar_init();
  }
  pdl_wait();

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
  const int nt = 1 + (len + kTile - 1) / kTile;
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
      gather_tile(kv_base + i * kKvStage, i == 0 ? p.swa_kv : p.c_kv, pick(own, i), q4, g4, lane);
      cp_commit();
      ++committed;
    }
  }
  float fr = 0.f, fi = 0.f;
  if (p.rope && g4 == kChunks - 1 && q4 >= 2) {
    const long long pos = p.pos64 ? static_cast<const long long*>(p.pos)[row]
                                  : static_cast<long long>(static_cast<const int*>(p.pos)[row]);
    const uint32_t j = (q4 * 32u + lane - 64u) >> 1;
    fr = p.freqs[pos * 64 + 2 * j];
    fi = p.freqs[pos * 64 + 2 * j + 1];
  }
  // warp 0, lane h: head h's running raw max of the current part and the merged (max, den) of the closed parts
  float run_m = kNegMax, mrg_m = kNegMax, mrg_d = 0.f, part_d = 0.f;
  // per-thread denominator partials (heads h0, h0 + 1) of the current part, stock's layout and order
  float d0 = 0.f, d1 = 0.f;
  const int vs = p.vsplit > 1 ? p.vsplit : 1;
  const int nk = nt < vs ? nt : vs;
  const int tpc = (nt + nk - 1) / nk;

  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t taddr = s.taddr;
  const uint32_t t_o = taddr;
  const uint32_t t_s0 = taddr + 128u;
  const uint32_t t_f = taddr + 256u;
  const uint32_t lane_off = (q4 * 32u) << 16;
  constexpr uint32_t kIdO = idesc_e4m3(kChunk, kHeads, true);
  int pv_waits = 0;

  if (nt > 0) {
    cp_wait(committed - 1);
    fence_async_smem();
    __syncthreads();
    if (tid == 32) {
      tc_fence_after();
      issue_s(t_s0, kv_base, q_base);
      tc_commit(smem_u32(&s.bar_s));
    }
  }

#pragma unroll 1
  for (int i = 0; i < nt; ++i) {
    const uint32_t stage = kv_base + (i % kStages) * kKvStage;
    const uint32_t t_s = t_s0 + 32u * static_cast<uint32_t>(i & 1);
    const uint32_t p_buf = smem_u32(s.p[i & 1]);
    const bool part_first = (i % tpc) == 0;
    const bool part_last = vs > 1 && ((i + 1) % tpc == 0 || i + 1 == nt);
    const int32_t my_idx = pick(own, i);
    const bool valid_own = my_idx >= 0 || (my_idx == -1 && p.zero_mode != 0);
    const uint32_t vmask = __ballot_sync(0xFFFFFFFFu, valid_own);   // bit l = slot 32 q4 + l scores
    mbar_wait(smem_u32(&s.bar_s), static_cast<uint32_t>(i & 1));
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
    const bool next = i + 1 < nt;
    if (next) cp_wait(committed - (i + 2));   // tile i + 1's stage landed: its S^T goes out now
    fence_async_smem();
    __syncthreads();
    if (next && tid == 32) {
      tc_fence_after();
      issue_s(t_s0 + 32u * static_cast<uint32_t>((i + 1) & 1), kv_base + ((i + 1) % kStages) * kKvStage, q_base);
      tc_commit(smem_u32(&s.bar_s));
    }
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
    __syncthreads();
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
    // ---- O correction in TMEM (O = O * alpha) before this tile's accumulating PV; frees the previous stage ---------
    if (!part_first) {
      mbar_wait(smem_u32(&s.bar_o), static_cast<uint32_t>(pv_waits & 1));   // PV(i - 1)
      ++pv_waits;
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
      const int j = i - 1;   // its stage is free: gather tile j + 3
      if (j + kStages < nt) {
        gather_tile(kv_base + (j % kStages) * kKvStage, p.c_kv, pick(own, j + kStages), q4, g4, lane);
        cp_commit();
        ++committed;
      }
    }
    fence_async_smem();
    tc_fence_before();
    __syncthreads();
    if (tid == 0) {
      tc_fence_after();
      const uint64_t a0 = sdesc(stage), b0 = sdesc(p_buf);
#pragma unroll
      for (uint32_t c = 0; c < kChunks; ++c)
#pragma unroll
        for (uint32_t kb = 0; kb < 4; ++kb)
          tc_mma(t_o + 32u * c, a0 + ((c * kKvChunk + kb * 32u * kChunk) >> 4), b0 + ((kb * 32u) >> 4), kIdO,
                 (kb != 0 || !part_first) ? 1u : 0u);
      tc_commit(smem_u32(&s.bar_o));
    }
    if (part_last) {
      // ---- stock's MultiCtasKv part end: den = reduce(partials); O -> bf16(O rcp 448) merged into F (part order) ----
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
      d0 = 0.f;
      d1 = 0.f;
      mbar_wait(smem_u32(&s.bar_o), static_cast<uint32_t>(pv_waits & 1));   // PV(i)
      ++pv_waits;
      tc_fence_after();
      __syncthreads();
      if (warp == 0) {
        const float dk = add_rn(add_rn(add_rn(s.red_sum[0][lane], s.red_sum[1][lane]), s.red_sum[2][lane]),
                                s.red_sum[3][lane]);
        const float M = fmaxf(mrg_m, run_m);
        const float c0 = ex2(mul_ftz(add_ftz(mrg_m, neg_f(M)), p.scale_log2));
        const float c1 = ex2(mul_ftz(add_ftz(run_m, neg_f(M)), p.scale_log2));
        mrg_d = fma_ftz(c0, mrg_d, mul_ftz(dk, c1));
        mrg_m = M;
        s.gam[lane] = c0;
        s.beta[lane] = c1;
        run_m = kNegMax;
      }
      __syncthreads();
      const float ps = rcp_approx(448.0f);
      float o[kHeads], f[kHeads];
      const uint32_t tfa = t_f + 32u * g4 + lane_off;
      tc_ld32(t_o + 32u * g4 + lane_off, o);
      if (i + 1 > tpc) tc_ld32(tfa, f);
      tc_wait_ld();
      if (i + 1 <= tpc) {
#pragma unroll
        for (int h = 0; h < kHeads; ++h) f[h] = 0.f;
      }
#pragma unroll
      for (int h = 0; h < kHeads; ++h) {
        const float part = bf16_round(mul_rn(o[h], ps));
        f[h] = fma_rn(part, s.beta[h], mul_rn(f[h], s.gam[h]));
      }
      tc_st32(tfa, f);
      tc_wait_st();
      if (i + kStages < nt) {
        gather_tile(stage, p.c_kv, pick(own, i + kStages), q4, g4, lane);
        cp_commit();
        ++committed;
      }
      tc_fence_before();
    }
  }
  pdl_launch();
  float acc[kHeads];
  if (vs > 1) {
    // fmhaReduction tail: den = fma(448, ex2(fma(sink, log2e, -(M sl2))), den); out = F * (448 rcp(den))
    if (warp == 0) {
      const float e = ex2(fma_ftz(p.sinks[lane], kLog2e, neg_f(mul_ftz(mrg_m, p.scale_log2))));
      const float d = fma_ftz(448.0f, e, mrg_d);
      s.inv[lane] = mul_ftz(448.0f, rcp_approx(d));
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    tc_ld32(t_f + 32u * g4 + lane_off, acc);
    tc_wait_ld();
  } else {
    // stock's single-CTA epilogue: den = reduce(partials) + 448 ex2(fma(sink, log2e, -(m sl2))); out = O rcp(den)
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
    mbar_wait(smem_u32(&s.bar_o), static_cast<uint32_t>(pv_waits & 1));   // the last PV
    ++pv_waits;
    tc_fence_after();
    __syncthreads();
    if (warp == 0) {
      const float dk = add_rn(add_rn(add_rn(s.red_sum[0][lane], s.red_sum[1][lane]), s.red_sum[2][lane]),
                              s.red_sum[3][lane]);
      const float e = ex2(fma_ftz(p.sinks[lane], kLog2e, neg_f(mul_ftz(run_m, p.scale_log2))));
      s.inv[lane] = rcp_approx(add_ftz(dk, mul_ftz(e, 448.0f)));
    }
    __syncthreads();
    tc_ld32(t_o + 32u * g4 + lane_off, acc);
    tc_wait_ld();
  }
  (void)part_d;

  // ---- out[row][h][d], d = 128 g4 + 32 q4 + lane; the inverse RoPE on d >= 448 (g4 3, q4 2 / 3) -------------------
  uint16_t* orow = p.out + static_cast<long long>(row) * (kHeads * kDim);
  const uint32_t d = g4 * kChunk + q4 * 32u + lane;
  if (p.rope && g4 == kChunks - 1 && q4 >= 2) {
    const bool real = (lane & 1u) == 0;
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      const uint16_t b = bf16_rn(mul_rn(acc[h], s.inv[h]));
      const float own_v = __uint_as_float(static_cast<uint32_t>(b) << 16);
      const float other = __shfl_xor_sync(0xFFFFFFFFu, own_v, 1);
      const float t2 = mul_rn(fi, other);
      const float r = fma_rn(fr, own_v, real ? t2 : neg_f(t2));
      orow[h * kDim + d] = bf16_rn(r);
    }
  } else {
    const float4* iv = reinterpret_cast<const float4*>(s.inv);
#pragma unroll
    for (int h4 = 0; h4 < kHeads / 4; ++h4) {
      const float4 v = iv[h4];
      orow[(4 * h4 + 0) * kDim + d] = bf16_rn(mul_rn(acc[4 * h4 + 0], v.x));
      orow[(4 * h4 + 1) * kDim + d] = bf16_rn(mul_rn(acc[4 * h4 + 1], v.y));
      orow[(4 * h4 + 2) * kDim + d] = bf16_rn(mul_rn(acc[4 * h4 + 2], v.z));
      orow[(4 * h4 + 3) * kDim + d] = bf16_rn(mul_rn(acc[4 * h4 + 3], v.w));
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) {
    tc_fence_after();
    tc_dealloc(taddr, kTmemCols);
  }
}

}  // namespace bsa

#include <stdexcept>
#include <string>

// lever BSA launch (raw device pointers as int64; the same arguments as lever SPM's spm_run minus split): q [rows, 32,
// 512] e4m3 (row stride q_stride bytes, 16-B aligned); swa_kv / c_kv flat [slots, 512] e4m3 pools; swa_idx [rows, 128]
// / topk_idx [rows, 512] int32 (row strides in elements); topk_len [rows] int32 (len64 0) / int64 (1); sinks [32] fp32;
// freqs [P, 64] fp32 + pos [rows] int32 / int64 (pos64) when rope = 1; out [rows, 32, 512] bf16 contiguous; scale =
// softmax scale; pdl 1 = programmatic stream serialization; vsplit = stock's CTAs a row (min(3, 148 / rows), 1 above
// 74 rows); stream = the raw cudaStream_t
void bsa_run(int64_t q, int64_t q_stride, int64_t swa_kv, int64_t c_kv, int64_t swa_idx, int64_t swa_stride,
             int64_t topk_idx, int64_t topk_stride, int64_t topk_len, int64_t len64, int64_t sinks, int64_t freqs,
             int64_t pos, int64_t pos64, int64_t out, int64_t rows, double scale, int64_t rope, int64_t pdl,
             int64_t zero_mode, int64_t seq_lens, int64_t vsplit, int64_t stream) {
  if (vsplit < 1 || vsplit > 3) throw std::runtime_error("bsa: vsplit 1, 2 or 3");
  if (zero_mode < 0 || zero_mode > 2 || (zero_mode == 2 && (seq_lens == 0 || seq_lens % 4)))
    throw std::runtime_error("bsa: zero_mode / seq_lens");
  if (rows < 1 || rows > 4096) throw std::runtime_error("bsa: rows");
  if (q % 16 || q_stride % 16 || swa_kv % 16 || c_kv % 16 || out % 16) throw std::runtime_error("bsa: alignment");
  if (swa_idx % 4 || topk_idx % 4 || sinks % 4) throw std::runtime_error("bsa: index alignment");
  if (len64 != 0 && len64 != 1) throw std::runtime_error("bsa: len64");
  if (rope != 0 && rope != 1) throw std::runtime_error("bsa: rope");
  if (rope && (freqs % 16 || (pos64 != 0 && pos64 != 1))) throw std::runtime_error("bsa: rope table");
  static bool attr_done = false;
  if (!attr_done) {
    cudaError_t e = cudaFuncSetAttribute(bsa::attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(bsa::kSmemBytes));
    if (e != cudaSuccess) throw std::runtime_error(std::string("bsa: smem attribute: ") + cudaGetErrorString(e));
    attr_done = true;
  }
  bsa::Params prm{};
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
  prm.zero_mode = static_cast<int>(zero_mode);
  prm.seq_lens = reinterpret_cast<const int*>(seq_lens);
  prm.vsplit = static_cast<int>(vsplit);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(rows));
  cfg.blockDim = dim3(bsa::kThreads);
  cfg.dynamicSmemBytes = bsa::kSmemBytes;
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
  const cudaError_t rc = cudaLaunchKernelEx(&cfg, bsa::attn_kernel, prm);
  if (rc != cudaSuccess) throw std::runtime_error(std::string("bsa: launch failed: ") + cudaGetErrorString(rc));
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
  m.def("bsa_run", &bsa_run, "lever BSA: bitwise sparse MLA decode attention (+ inverse RoPE)");
}
