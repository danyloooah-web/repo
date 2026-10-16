// lev_GQ: the shared expert's gate_up input quantize (bf16 rows of hidden 5120 -> MXFP8: e4m3 values + UE8M0 scales in
// FlashInfer's 128x4 swizzled layout, every scale byte of the padded rows zero) on a SMALL grid -- ceil(rows /
// rows_per_cta) CTAs of 640 threads instead of FlashInfer 0.7.1's 128-CTA MXFP8QuantizeSwizzledKernel launch -- bit for
// bit stock's bytes (flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)) at any row count.
//
// Why: in the decode / verify graphs the stock quantize (alt stream, forked before the router) is still spread over
// ~128 SMs when the router GEMM ends; RE5's cluster then waits for a whole free GPC behind the side chain. A 6..16-CTA
// grid leaves the GPCs free.
//
// Arithmetic (lev_HQ's / lev_QN mode 0's quantize, FlashInfer's MXFP8 on bf16 input), per 32-column group (4 lanes of
// 8 columns): amax by the bf16x2 and-mask / max.bf16x2 tree, max.f32 of the halves, max.f32 over lanes xor 1, 2;
// n = mul(amax, 1 / 448 as f32); FlashInfer's UE8M0 (positive rounding, subnormals) and inverse scale 2^(127 - sf) bit
// for bit; q = cvt.rn.satfinite.e4m3x2 of max(min(mul(x, inv), 448), -448). Every float op is inline PTX (bundle
// units build with --use_fast_math). Thread t of a CTA owns columns 8 t .. 8 t + 7 of each of its rows (one 16-byte
// load a row, up to kRows rows in flight). PDL: griddepcontrol.wait before any global access (x is the previous kernel's
// output: plain coherent loads, never ld.global.nc / __ldg / const __restrict__), launch_dependents after the stores
// (pdl 2, as FlashInfer's own kernel) or right after the wait (pdl 1); pdl 0 = plain launch.
#include <cuda_runtime.h>
#include <stdint.h>

namespace gq {

constexpr int kH = 5120;              // hidden width
constexpr int kSfCols = kH / 32;      // 160 scale columns (a multiple of 4: no column padding)
constexpr int kTileBytes = 128 * kSfCols;
constexpr int kThreads = kH / 8;      // 640: one 16-byte chunk of a row a thread
constexpr int kRows = 8;              // rows a thread holds in flight
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float mul_f(float a, float b) {
  float r;
  asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float max_f(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float min_f(float a, float b) {
  float r;
  asm("min.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint32_t max_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ uint32_t e4m3x2(float hi, float lo) {         // cvt.rn.satfinite.e4m3x2.f32 d, hi, lo
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bfly(float v, int mask) { return __shfl_xor_sync(kFull, v, mask, 32); }

// FlashInfer's float_to_ue8m0_fast and ue8m0_to_inv_scale_fast, verbatim (as lev_QN / lev_HQ)
__device__ __forceinline__ uint32_t ue8m0(float value) {
  uint32_t r;
  asm("{\n"
      ".reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
      ".reg .u32 bits, exp_biased, mantissa, bump, result;\n"
      "setp.le.f32 p_zero, %1, 0f00000000;\n"
      "mov.b32 bits, %1;\n"
      "shr.b32 exp_biased, bits, 23;\n"
      "and.b32 exp_biased, exp_biased, 255;\n"
      "and.b32 mantissa, bits, 0x7FFFFF;\n"
      "setp.ne.u32 p_has_mant, mantissa, 0;\n"
      "selp.u32 bump, 1, 0, p_has_mant;\n"
      "setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
      "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
      "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
      "@p_tiny_sub mov.u32 bump, 0;\n"
      "add.u32 result, exp_biased, bump;\n"
      "setp.gt.u32 p_ovf, result, 254;\n"
      "selp.u32 result, 254, result, p_ovf;\n"
      "selp.u32 %0, 0, result, p_zero;\n"
      "}\n"
      : "=r"(r)
      : "f"(value));
  return r;
}
__device__ __forceinline__ float inv_scale(uint32_t sf) {
  float r;
  asm("{\n"
      ".reg .s32 new_exp;\n"
      ".reg .b32 float_bits;\n"
      ".reg .pred p_zero;\n"
      "setp.eq.u32 p_zero, %1, 0;\n"
      "sub.s32 new_exp, 254, %1;\n"
      "max.s32 new_exp, new_exp, 0;\n"
      "shl.b32 float_bits, new_exp, 23;\n"
      "mov.b32 %0, float_bits;\n"
      "@p_zero mov.b32 %0, 0;\n"
      "}\n"
      : "=f"(r)
      : "r"(sf));
  return r;
}

// FlashInfer swizzled 128x4 scale offset (compute_sf_index_swizzled_128x4_gpu, padded_cols 160)
__device__ __forceinline__ long long sf_offset(int row, int col) {
  return (col & 3) + (col >> 2) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 +
         static_cast<long long>(row >> 7) * kTileBytes;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// one row's 8 columns of this thread (w: the 16-byte chunk, little-endian bf16 pairs) -> q bytes + (lane % 4 == 0) scale
__device__ __forceinline__ void quant_chunk(const uint4 v, int row, int col, int lane, uint8_t* q, uint8_t* sf) {
  const uint32_t yw[4] = {v.x, v.y, v.z, v.w};
  const uint32_t a0 = yw[0] & 0x7FFF7FFFu, a1 = yw[1] & 0x7FFF7FFFu;
  const uint32_t a2 = yw[2] & 0x7FFF7FFFu, a3 = yw[3] & 0x7FFF7FFFu;
  const uint32_t mx = max_bf16x2(max_bf16x2(a0, a1), max_bf16x2(a2, a3));
  float amax = max_f(bf_lo(mx), bf_hi(mx));
  amax = max_f(amax, bfly(amax, 1));
  amax = max_f(amax, bfly(amax, 2));
  const uint32_t sc = ue8m0(mul_f(amax, 0.002232142857142857f));
  const float inv = inv_scale(sc);
  uint32_t qw[2];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float lo = max_f(min_f(mul_f(bf_lo(yw[k]), inv), 448.0f), -448.0f);
    const float hi = max_f(min_f(mul_f(bf_hi(yw[k]), inv), 448.0f), -448.0f);
    const uint32_t pair = e4m3x2(hi, lo);
    if (k & 1) qw[k >> 1] |= pair << 16;
    else qw[k >> 1] = pair;
  }
  *reinterpret_cast<uint2*>(q + static_cast<long long>(row) * kH + col) = make_uint2(qw[0], qw[1]);
  if ((lane & 3) == 0) sf[sf_offset(row, col >> 5)] = static_cast<uint8_t>(sc);
}

// x: [m, 5120] bf16 rows with row stride sx (elements); CTA b quantizes rows [b * rpc, min(m, (b + 1) * rpc)), then
// every CTA zeroes its share (grid-stride) of the padded rows' scale bytes of the last 128-row tile
__global__ void __launch_bounds__(kThreads, 1) quant_kernel(const uint16_t* x, long long sx, uint8_t* __restrict__ q,
                                                            uint8_t* __restrict__ sf, int m, int rpc, int early) {
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int col = 8 * t;
  pdl_wait();
  if (early) pdl_launch();
  const int first = static_cast<int>(blockIdx.x) * rpc;
  const int end = min(first + rpc, m);
  for (int r0 = first; r0 < end; r0 += kRows) {
    uint4 v[kRows];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const int row = r0 + i;
      if (row < end) v[i] = *reinterpret_cast<const uint4*>(x + static_cast<long long>(row) * sx + col);
    }
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const int row = r0 + i;   // uniform over the CTA: the shuffles inside see every lane
      if (row < end) quant_chunk(v[i], row, col, lane, q, sf);
    }
  }
  const int padded = ((m + 127) >> 7) << 7;
  const int npad = (padded - m) * kSfCols;
  for (int i = static_cast<int>(blockIdx.x) * kThreads + t; i < npad; i += static_cast<int>(gridDim.x) * kThreads) {
    sf[sf_offset(m + i / kSfCols, i % kSfCols)] = 0;
  }
  if (!early) pdl_launch();
}

}  // namespace gq

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

// pointers: x [m, 5120] bf16 with row stride sx (elements, a multiple of 8, 16-byte aligned), q [m, 5120] e4m3
// (contiguous), sf [padded_m * 160] u8 (padded_m = m rounded up to 128); rpc = rows a CTA (grid = ceil(m / rpc));
// pdl 0 = plain launch, 1 = PDL with the dependents released right after griddepcontrol.wait, 2 = after the stores
void run(int64_t x, int64_t sx, int64_t q, int64_t sf, int64_t m, int64_t rpc, int64_t pdl, int64_t stream) {
  if (m < 1 || m > (int64_t(1) << 20)) throw std::runtime_error("gq: rows");
  if (rpc < 1 || rpc > 4096) throw std::runtime_error("gq: rows a CTA");
  if ((x | q) % 16 || sx % 8 || sx < gq::kH) throw std::runtime_error("gq: 16-byte aligned rows");
  if (pdl < 0 || pdl > 2) throw std::runtime_error("gq: pdl 0, 1 or 2");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>((m + rpc - 1) / rpc));
  cfg.blockDim = dim3(static_cast<unsigned>(gq::kThreads));
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t rc = cudaLaunchKernelEx(&cfg, gq::quant_kernel, reinterpret_cast<const uint16_t*>(x),
                                            static_cast<long long>(sx), reinterpret_cast<uint8_t*>(q),
                                            reinterpret_cast<uint8_t*>(sf), static_cast<int>(m), static_cast<int>(rpc),
                                            pdl == 1 ? 1 : 0);
  if (rc != cudaSuccess) throw std::runtime_error(std::string("gq: launch failed: ") + cudaGetErrorString(rc));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "MXFP8 quantize of bf16 rows of hidden 5120 (FlashInfer's 128x4 swizzled scales), stock's bytes");
}
