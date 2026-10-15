// lev_QN: MqaAttentionBase._normalize_q_lora (q_lora_rank K = 1280, bf16) as ONE kernel: the q-path RMSNorm and the wq_b
// linear's MXFP8 quantize (e4m3 values + UE8M0 scales in FlashInfer's 128x4 swizzled layout, every scale byte of the
// padded rows zero), bit for bit the stock path of the call's row count:
//   mode 0 (stock above 64 rows): sgl_kernel.rmsnorm -> flashinfer.norm.rmsnorm -> FlashInfer 0.7.1's CuTe-DSL
//          RMSNormKernel (H 1280: 32 threads a row, 4 rows a 128-thread CTA), then the linear's own
//          flashinfer_mxfp8_quantize (CuTe-DSL MXFP8QuantizeSwizzledKernel, 128x4);
//   mode 1 (stock at 1..64 rows): the fork's Triton _rmsnorm_mxfp8_kernel (num_warps 8, one program a row).
// One warp a row (4 rows a 128-thread CTA). Lane l holds columns 256 j + 8 l + v (j < 5, v < 8): the CuTe kernel's
// thread-value layout, and the Triton kernel's (its thread t = 32 j + l of 8 warps holds 8 t .. 8 t + 7), so the modes
// differ only in their reduction trees and epilogue op order, taken from the references' PTX:
//   mode 0  acc = fma.rn(x, x, acc) over j, v ascending from 0 (cutlass-dsl 4.8 on sm_103: fma.rn.f32.bf16, the same
//           exact-product single rounding); acc = add(bfly(acc), acc) for xor 1, 2, 4, 8, 16;
//           mean = div.rn(acc, 1280); rstd = rsqrt.approx.ftz(add(eps, mean));
//           y = cvt.rn.bf16x2(mul(add(w, 0), mul(rstd, x)))
//   mode 1  s_j = ((p0 + p2) + (p4 + p6)) + ((p1 + p3) + (p5 + p7)) with p = mul.rn(x, x), add.rn (Triton 3.8.0's
//           f32x2 pairwise reduction on sm_103; Triton 3.7 / sm_90 chains them instead); W_j = add.rn(s, bfly(s)) for
//           xor 16, 8, 4, 2, 1; the 8-warp tree with warps 5..7 all zero: T = ((W0 + W4) + (W2 + 0)) + ((W1 + 0) + (W3 + 0));
//           mean = div.full(T, 1280); rstd = rsqrt.approx.ftz(add.rn(mean, eps)); y = cvt.rn.bf16(mul.rn(mul.rn(rstd, x), w))
// MXFP8 epilogue on the bf16 y, 32-column groups (4 lanes): amax by NaN-ignoring max (mode 0: FlashInfer's bf16x2
// and-mask / max.bf16x2 tree, max.f32 of the halves, max.f32 over lanes xor 1, 2; mode 1: Triton's abs.f32 / max.f32
// chain, lanes xor 2, 1); n = mul(amax, 1 / 448 as f32); FlashInfer's UE8M0 (positive rounding, subnormals) and inverse
// scale 2^(127 - sf) bit for bit; q = cvt.rn.satfinite.e4m3x2 of mul(y, inv) clamped to +-448 (FlashInfer min then max,
// Triton max then min).
// Every float op is inline PTX: bundle units build with --use_fast_math (ftz, approximate division, contraction), the
// references do not. PDL: the weight loads run before griddepcontrol.wait (nothing in the stream writes the weight),
// every other global access after it (an output buffer may be memory the previous kernel still reads);
// launch_dependents after the stores, as FlashInfer's kernels do.
#include <cuda_runtime.h>
#include <stdint.h>

namespace qn {

constexpr int kK = 1280;
constexpr int kChunks = 5;            // 256-column chunks; a lane holds 8 columns of each
constexpr int kRowsPerCta = 4;
constexpr int kThreads = 32 * kRowsPerCta;
constexpr int kSfCols = kK / 32;      // 40 scale columns (a multiple of 4: no column padding)
constexpr int kTileBytes = 128 * kSfCols;
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float add_f(float a, float b) {
  float r;
  asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float mul_f(float a, float b) {
  float r;
  asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float div_rn(float a, float b) {
  float r;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rsqrt_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
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
__device__ __forceinline__ float abs_f(float a) {
  float r;
  asm("abs.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ uint32_t max_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float hi, float lo) {   // cvt.rn.bf16x2.f32 d, hi, lo
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t to_bf16(float a) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ uint32_t e4m3x2(float hi, float lo) {         // cvt.rn.satfinite.e4m3x2.f32 d, hi, lo
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bfly(float v, int mask) { return __shfl_xor_sync(kFull, v, mask, 32); }

// FlashInfer's float_to_ue8m0_fast (= the Triton ue8m0_scale integer form) and ue8m0_to_inv_scale_fast, verbatim
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

// FlashInfer swizzled 128x4 scale offset (compute_sf_index_swizzled_128x4_gpu, padded_cols 40)
__device__ __forceinline__ int sf_offset(int row, int col) {
  return (col & 3) + (col >> 2) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (row >> 7) * kTileBytes;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

template <int kMode>
__global__ void __launch_bounds__(kThreads) qnorm_mxfp8(const uint16_t* __restrict__ x, long long sx,
                                                         const uint16_t* __restrict__ w, uint16_t* __restrict__ y,
                                                         uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int m,
                                                         int padded_m, float eps) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerCta + (threadIdx.x >> 5);
  uint4 wv[kChunks];
#pragma unroll
  for (int j = 0; j < kChunks; ++j) wv[j] = __ldg(reinterpret_cast<const uint4*>(w + 256 * j + 8 * lane));
  pdl_wait();
  if (row >= padded_m) {
    pdl_launch();
    return;
  }
  if (row >= m) {                    // padding row of the last 128-row tile: its 40 scale bytes are zero
    sf[sf_offset(row, lane)] = 0;
    if (lane < kSfCols - 32) sf[sf_offset(row, lane + 32)] = 0;
    pdl_launch();
    return;
  }
  const uint16_t* xr = x + static_cast<long long>(row) * sx;
  uint4 xv[kChunks];
#pragma unroll
  for (int j = 0; j < kChunks; ++j) xv[j] = *reinterpret_cast<const uint4*>(xr + 256 * j + 8 * lane);
  float xf[kChunks][8];
  float wf[kChunks][8];
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
    const uint32_t xw[4] = {xv[j].x, xv[j].y, xv[j].z, xv[j].w};
    const uint32_t ww[4] = {wv[j].x, wv[j].y, wv[j].z, wv[j].w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      xf[j][2 * k] = bf_lo(xw[k]);
      xf[j][2 * k + 1] = bf_hi(xw[k]);
      wf[j][2 * k] = bf_lo(ww[k]);
      wf[j][2 * k + 1] = bf_hi(ww[k]);
    }
  }
  float rstd;
  if (kMode == 0) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kChunks; ++j) {
#pragma unroll
      for (int v = 0; v < 8; ++v) acc = fma_rn(xf[j][v], xf[j][v], acc);
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) acc = add_f(bfly(acc, o), acc);
    rstd = rsqrt_ftz(add_f(eps, div_rn(acc, static_cast<float>(kK))));
  } else {
    float wsum[kChunks];
#pragma unroll
    for (int j = 0; j < kChunks; ++j) {
      float p[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) p[v] = mul_rn(xf[j][v], xf[j][v]);
#ifndef QN_MODE1_CHAIN
      // Triton 3.8 on sm_103 reduces a thread's 8 squares in f32x2 pairs: {p0 + p2, p1 + p3}, {p4 + p6, p5 + p7},
      // their sum, then the two lanes of the pair
      const float lo = add_rn(add_rn(p[0], p[2]), add_rn(p[4], p[6]));
      const float hi = add_rn(add_rn(p[1], p[3]), add_rn(p[5], p[7]));
      float s = add_rn(lo, hi);
#else
      // tests only (qn_test.py's sensitivity control): Triton 3.7 / sm_90's chain, which sm_103's reference is not
      float s = add_rn(p[0], p[1]);
#pragma unroll
      for (int v = 2; v < 8; ++v) s = add_rn(p[v], s);
#endif
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s = add_rn(s, bfly(s, o));
      wsum[j] = s;
    }
    const float t = add_rn(add_rn(add_rn(wsum[0], wsum[4]), add_rn(wsum[2], 0.0f)),
                           add_rn(add_rn(wsum[1], 0.0f), add_rn(wsum[3], 0.0f)));
    rstd = rsqrt_ftz(add_rn(div_full(t, static_cast<float>(kK)), eps));
  }
  uint16_t* yr = y + static_cast<long long>(row) * kK;
  uint8_t* qr = q + static_cast<long long>(row) * kK;
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
    uint32_t yw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (kMode == 0) {
        const float lo = mul_f(add_f(wf[j][2 * k], 0.0f), mul_f(rstd, xf[j][2 * k]));
        const float hi = mul_f(add_f(wf[j][2 * k + 1], 0.0f), mul_f(rstd, xf[j][2 * k + 1]));
        yw[k] = pack_bf16x2(hi, lo);
      } else {
        const uint32_t lo = to_bf16(mul_rn(mul_rn(rstd, xf[j][2 * k]), wf[j][2 * k]));
        const uint32_t hi = to_bf16(mul_rn(mul_rn(rstd, xf[j][2 * k + 1]), wf[j][2 * k + 1]));
        yw[k] = lo | (hi << 16);
      }
    }
    *reinterpret_cast<uint4*>(yr + 256 * j + 8 * lane) = make_uint4(yw[0], yw[1], yw[2], yw[3]);
    float amax;
    if (kMode == 0) {
      const uint32_t a0 = yw[0] & 0x7FFF7FFFu, a1 = yw[1] & 0x7FFF7FFFu;
      const uint32_t a2 = yw[2] & 0x7FFF7FFFu, a3 = yw[3] & 0x7FFF7FFFu;
      const uint32_t mx = max_bf16x2(max_bf16x2(a0, a1), max_bf16x2(a2, a3));
      amax = max_f(bf_lo(mx), bf_hi(mx));
      amax = max_f(amax, bfly(amax, 1));
      amax = max_f(amax, bfly(amax, 2));
    } else {
      amax = max_f(abs_f(bf_lo(yw[0])), abs_f(bf_hi(yw[0])));
#pragma unroll
      for (int k = 1; k < 4; ++k) {
        amax = max_f(amax, abs_f(bf_lo(yw[k])));
        amax = max_f(amax, abs_f(bf_hi(yw[k])));
      }
      amax = max_f(amax, bfly(amax, 2));
      amax = max_f(amax, bfly(amax, 1));
    }
    const float norm = kMode == 0 ? mul_f(amax, 0.002232142857142857f) : mul_rn(amax, 0.002232142857142857f);
    const uint32_t s = ue8m0(norm);
    const float inv = inv_scale(s);
    uint32_t qw[2];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float lo, hi;
      if (kMode == 0) {
        lo = max_f(min_f(mul_f(bf_lo(yw[k]), inv), 448.0f), -448.0f);
        hi = max_f(min_f(mul_f(bf_hi(yw[k]), inv), 448.0f), -448.0f);
      } else {
        lo = min_f(max_f(mul_rn(bf_lo(yw[k]), inv), -448.0f), 448.0f);
        hi = min_f(max_f(mul_rn(bf_hi(yw[k]), inv), -448.0f), 448.0f);
      }
      const uint32_t pair = e4m3x2(hi, lo);
      if (k & 1) qw[k >> 1] |= pair << 16;
      else qw[k >> 1] = pair;
    }
    *reinterpret_cast<uint2*>(qr + 256 * j + 8 * lane) = make_uint2(qw[0], qw[1]);
    if ((lane & 3) == 0) sf[sf_offset(row, 8 * j + (lane >> 2))] = static_cast<uint8_t>(s);
  }
  pdl_launch();
}

}  // namespace qn

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

// pointers: x [m, 1280] bf16 with row stride sx (elements), w [1280] bf16, y [m, 1280] bf16, q [m, 1280] e4m3,
// sf [padded_m * 40] u8 (padded_m = m rounded up to 128); mode 0 = FlashInfer RMSNorm + quantize, 1 = the Triton kernel
void run(int64_t x, int64_t sx, int64_t w, int64_t y, int64_t q, int64_t sf, int64_t m, int64_t padded_m, double eps,
         int64_t mode, int64_t pdl, int64_t stream) {
  if (m < 1 || m > (int64_t(1) << 24) || padded_m != (m + 127) / 128 * 128) throw std::runtime_error("qn: rows");
  if ((x | y | w | q) % 16 || sx % 8 || sx < qn::kK) throw std::runtime_error("qn: 16-byte aligned rows");
  if (mode != 0 && mode != 1) throw std::runtime_error("qn: mode 0 or 1");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(padded_m / qn::kRowsPerCta));
  cfg.blockDim = dim3(qn::kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const auto* xp = reinterpret_cast<const uint16_t*>(x);
  const auto* wp = reinterpret_cast<const uint16_t*>(w);
  auto* yp = reinterpret_cast<uint16_t*>(y);
  auto* qp = reinterpret_cast<uint8_t*>(q);
  auto* sp = reinterpret_cast<uint8_t*>(sf);
  const long long sxl = static_cast<long long>(sx);
  const int mi = static_cast<int>(m), pmi = static_cast<int>(padded_m);
  const float e = static_cast<float>(eps);
  const cudaError_t rc = mode == 0 ? cudaLaunchKernelEx(&cfg, qn::qnorm_mxfp8<0>, xp, sxl, wp, yp, qp, sp, mi, pmi, e)
                                   : cudaLaunchKernelEx(&cfg, qn::qnorm_mxfp8<1>, xp, sxl, wp, yp, qp, sp, mi, pmi, e);
  if (rc != cudaSuccess) throw std::runtime_error(std::string("qn: launch failed: ") + cudaGetErrorString(rc));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "q-path RMSNorm + MXFP8 quantize (K 1280), bit-identical to stock's path for the row count");
}
