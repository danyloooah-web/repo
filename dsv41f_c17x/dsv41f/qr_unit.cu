// lev_QR: the fork's FP8 query rope kernel of the DeepSeek-V4.1 attention (q_head_norm = False), bit for bit:
// stock = sglang::fused_q_norm_rope<bf16, 512, 64, PosT, kUsePDL, kApplyNorm = false, kOutFp8 = true>
// (python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh, launched by
// sglang.kernels.ops.attention.dsv4.elementwise.fused_q_norm_rope(q, q_out, None, freqs_cis, positions), called from
// MQALayer._compute_q_b with q_out float8_e4m3fn for the trtllm sparse-MLA backend).
//
// Same work split as stock: 256-thread CTAs, one warp per (token, head) work item (work = 8 * blockIdx + warp, token =
// work / heads, head = work % heads), head_dim 512 = 2 tiles of 16 bytes per lane (tile i, lane l holds elements
// 8 (32 i + l) .. + 7), the rope tail = elements 448..511 = tile 1 of lanes 24..31, staged through shared memory and
// re-read as one (real, imag) bf16 pair per lane (elements 448 + 2 l, 449 + 2 l). The position is read before
// griddepcontrol.wait (as stock: positions are not produced by the previous kernel), every other global access after
// it; launch_dependents after the noPE stores (+ __syncwarp), as stock. Grid div_ceil(rows * heads, 8).
//
// Arithmetic (stock's SASS on sm_103a, nvcc 13.0, no fast-math: FMUL / FFMA without FTZ):
//   noPE    fp8(x) of each input bf16 x (no norm: kApplyNorm = false)
//   rope    real = fma.rn(x_real, f_real, -mul.rn(x_imag, f_imag)), imag = fma.rn(x_real, f_imag, mul.rn(x_imag, f_real))
//           (the contraction nvcc chose for `x_real * f_real - x_imag * f_imag`, `x_real * f_imag + x_imag * f_real`),
//           rounded to bf16 (cvt.rn.bf16x2.f32), then fp8 of each bf16
//   fp8(v)  stock's torch_fp8_e4m3fn: RNE, subnormals below 2^-6, |v| that rounds past 448 (and +-Inf) -> 0x7E with the
//           sign, NaN -> 0x7F without sign. Here: the hardware cvt.rn.satfinite.e4m3x2.f32 (two values an instruction)
//           and a NaN fix-up per 4 bytes (0xFF, the only NaN code with a sign, -> 0x7F). The conversion input is always a
//           bf16 value widened exactly; tests/qr_test.py proves the pair equal to stock over all 65536 bf16 patterns.
// Every float op is inline PTX with an explicit .rn and no .ftz: bundle units build with --use_fast_math (ftz,
// contraction), the stock JIT build does not.
// Loads: plain coherent ld.global (stock's LDG.E), never ld.global.nc: the kernel pointers carry no __restrict__, so
// nvcc cannot prove the input read-only and does not pick the non-coherent path. Under PDL the input q is written by
// the primary (wq_b) while this grid is already resident, and the PTX memory model -- which is what makes the
// primary's writes visible after griddepcontrol.wait -- does not cover ld.global.nc (tests/qr_test.py asserts no
// .CONSTANT load in the SASS).
// Test-only macros: QR_NAN_FIX=0 drops the fix-up, QR_ROPE_ORDER=1 / 2 = the other contraction / none, QR_PDL_WAIT=0
// drops griddepcontrol.wait (controls).
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef QR_NAN_FIX
#define QR_NAN_FIX 1
#endif
#ifndef QR_ROPE_ORDER
#define QR_ROPE_ORDER 0
#endif
#ifndef QR_PDL_WAIT
#define QR_PDL_WAIT 1
#endif

namespace qr {

constexpr int kD = 512;        // head_dim
constexpr int kRope = 64;      // rope tail width (32 (real, imag) pairs: one per lane)
constexpr int kWarps = 8;      // warps (work items) per CTA, = stock's kFusedQNumWarps
constexpr int kThreads = 32 * kWarps;

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
// {hi, lo} -> bf16x2 (hi in the upper half), round to nearest even
__device__ __forceinline__ uint32_t bf16x2_rn(float hi, float lo) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
// {hi, lo} -> e4m3x2 (hi in the upper byte), round to nearest even, saturating to +-448
__device__ __forceinline__ uint32_t e4m3x2(float hi, float lo) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
// per byte: 0xFF (NaN with the sign bit) -> 0x7F (stock's NaN); bytes 0..0x7F of (w & 0x7F7F7F7F) + 1 never carry
__device__ __forceinline__ uint32_t nan_fix4(uint32_t w) {
#if QR_NAN_FIX
  return w ^ (((w & 0x7F7F7F7Fu) + 0x01010101u) & w & 0x80808080u);
#else
  return w;
#endif
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// 8 bf16 (16 bytes, element 0 in the low half of x) -> 8 e4m3 bytes in element order
__device__ __forceinline__ uint2 fp8x8(const uint4 v) {
  const uint32_t a = e4m3x2(bf_hi(v.x), bf_lo(v.x)) | (e4m3x2(bf_hi(v.y), bf_lo(v.y)) << 16);
  const uint32_t b = e4m3x2(bf_hi(v.z), bf_lo(v.z)) | (e4m3x2(bf_hi(v.w), bf_lo(v.w)) << 16);
  return make_uint2(nan_fix4(a), nan_fix4(b));
}

// no __restrict__ on purpose (coherent loads, see the header)
template <typename PosT, bool kPDL>
__global__ __launch_bounds__(kThreads, 8) void q_rope_fp8(const uint16_t* q, uint8_t* out, const float* freqs,
                                                          const PosT* positions, long long in_stride,
                                                          long long out_stride, unsigned total, unsigned heads,
                                                          int log2_heads) {
  const unsigned warp = threadIdx.x >> 5;
  const unsigned lane = threadIdx.x & 31;
  const unsigned work = blockIdx.x * kWarps + warp;
  if (work >= total) return;
  unsigned token, head;
  if (log2_heads >= 0) {
    token = work >> log2_heads;
    head = work & (heads - 1);
  } else {
    token = work / heads;
    head = work - token * heads;
  }
  const uint4* in = reinterpret_cast<const uint4*>(q + static_cast<long long>(token) * in_stride +
                                                   static_cast<long long>(head) * kD);
  uint8_t* o = out + (static_cast<long long>(token) * out_stride + static_cast<long long>(head) * kD);
  const int position = static_cast<int>(positions[token]);

  __shared__ uint4 s_rope[kWarps][kRope / 8];

#if QR_PDL_WAIT
  if (kPDL) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

  const uint4 v0 = in[lane];
  const uint4 v1 = in[32 + lane];
  const float2 f = reinterpret_cast<const float2*>(freqs + static_cast<long long>(position) * kRope)[lane];

  *reinterpret_cast<uint2*>(o + 8 * lane) = fp8x8(v0);
  if (lane >= 32 - kRope / 8) {
    s_rope[warp][lane - (32 - kRope / 8)] = v1;
  } else {
    *reinterpret_cast<uint2*>(o + 256 + 8 * lane) = fp8x8(v1);
  }
  __syncwarp();

  if (kPDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const uint32_t e = reinterpret_cast<const uint32_t*>(s_rope[warp])[lane];
  const float xr = bf_lo(e), xi = bf_hi(e);
#if QR_ROPE_ORDER == 0
  const float re = fma_rn(xr, f.x, neg_f(mul_rn(xi, f.y)));
  const float im = fma_rn(xr, f.y, mul_rn(xi, f.x));
#elif QR_ROPE_ORDER == 1
  const float re = fma_rn(neg_f(xi), f.y, mul_rn(xr, f.x));
  const float im = fma_rn(xi, f.x, mul_rn(xr, f.y));
#else
  const float re = add_rn(mul_rn(xr, f.x), neg_f(mul_rn(xi, f.y)));
  const float im = add_rn(mul_rn(xr, f.y), mul_rn(xi, f.x));
#endif
  const uint32_t b = bf16x2_rn(im, re);
  const uint32_t pair = nan_fix4(e4m3x2(bf_hi(b), bf_lo(b)));
  *reinterpret_cast<uint16_t*>(o + (kD - kRope) + 2 * lane) = static_cast<uint16_t>(pair);
}

}  // namespace qr

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

// pointers: q [rows, heads, 512] bf16 with row stride in_stride (elements; head stride 512), out [rows, heads, 512]
// e4m3 with row stride out_stride (bytes = elements), freqs [max_pos, 64] fp32 (re, im interleaved, contiguous),
// positions [rows] int32 (pos_bytes 4) or int64 (8); pdl 1 = programmatic stream serialization (stock's kUsePDL)
void run(int64_t q, int64_t in_stride, int64_t out, int64_t out_stride, int64_t freqs, int64_t positions,
         int64_t pos_bytes, int64_t rows, int64_t heads, int64_t pdl, int64_t stream) {
  if (rows < 0 || heads < 1 || rows * heads >= (int64_t(1) << 31)) throw std::runtime_error("qr: rows / heads");
  if (pos_bytes != 4 && pos_bytes != 8) throw std::runtime_error("qr: positions int32 or int64");
  if (q % 16 || in_stride % 8 || out % 8 || out_stride % 8 || freqs % 8 || positions % pos_bytes)
    throw std::runtime_error("qr: alignment");
  if (in_stride < heads * qr::kD || out_stride < heads * qr::kD) throw std::runtime_error("qr: row strides");
  if (pdl != 0 && pdl != 1) throw std::runtime_error("qr: pdl 0 or 1");
  if (rows == 0) return;
  const unsigned total = static_cast<unsigned>(rows * heads);
  const unsigned h = static_cast<unsigned>(heads);
  const int log2h = (h & (h - 1)) == 0 ? __builtin_ctz(h) : -1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((total + qr::kWarps - 1) / qr::kWarps);
  cfg.blockDim = dim3(qr::kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const auto* qp = reinterpret_cast<const uint16_t*>(q);
  auto* op = reinterpret_cast<uint8_t*>(out);
  const auto* fp = reinterpret_cast<const float*>(freqs);
  const long long si = static_cast<long long>(in_stride), so = static_cast<long long>(out_stride);
  cudaError_t rc;
  if (pos_bytes == 8) {
    const auto* pp = reinterpret_cast<const int64_t*>(positions);
    rc = pdl ? cudaLaunchKernelEx(&cfg, qr::q_rope_fp8<int64_t, true>, qp, op, fp, pp, si, so, total, h, log2h)
             : cudaLaunchKernelEx(&cfg, qr::q_rope_fp8<int64_t, false>, qp, op, fp, pp, si, so, total, h, log2h);
  } else {
    const auto* pp = reinterpret_cast<const int32_t*>(positions);
    rc = pdl ? cudaLaunchKernelEx(&cfg, qr::q_rope_fp8<int32_t, true>, qp, op, fp, pp, si, so, total, h, log2h)
             : cudaLaunchKernelEx(&cfg, qr::q_rope_fp8<int32_t, false>, qp, op, fp, pp, si, so, total, h, log2h);
  }
  if (rc != cudaSuccess) throw std::runtime_error(std::string("qr: launch failed: ") + cudaGetErrorString(rc));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "DeepSeek-V4.1 FP8 query rope (head_dim 512, rope 64): stock fused_q_norm_rope's bytes, no norm");
}
