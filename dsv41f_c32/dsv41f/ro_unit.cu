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
}
