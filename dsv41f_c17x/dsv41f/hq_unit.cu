// lev_HQ: the attention hyper-connection combine of DeepseekV4DecoderLayer above 128 rows as ONE kernel: stock's
// hc_combine_norm (four-stream combine + RMSNorm, bf16 hidden 5120) and the MXFP8 quantize wqkv_a's linear runs on its
// output (FlashInfer 0.7.1's CuTe-DSL MXFP8QuantizeSwizzledKernel: e4m3 values + UE8M0 scales in the 128x4 swizzled
// layout, every scale byte of the padded rows zero), bit for bit, at any row count.
//
// y is the fork's Triton kernel of the call's row count as Triton 3.8.0 compiles it for sm_103 (tests/hq_test.py reads
// the reduction trees back from the references' PTX on the GPU):
//   kWarps 8  _hc_combine_norm (rows < 4096; 8 warps, BLOCK 8192): thread t holds columns 2048 r + 8 t + e (r < 4)
//   kWarps 4  _hc_combine_norm_prefill (rows >= 4096; 4 warps): columns 1024 r + 8 t + e (r < 8)
// per column  v = fma(p3, x3, fma(p2, x2, fma(p1, x1, fma(p0, x0, +0)))) (fma.rn.f32x2, p = pre row in fp32), then
//             rounded to bf16 (cvt.rn.bf16.f32) and widened back; masked columns are +0;
// per thread  the thread's columns as pairs P_k = (column 2k, 2k + 1) of its 16-byte words in rep-major order; per
//             pair lane: t_j = fma.rn(P_2j, P_2j, mul(P_2j+1, P_2j+1)), then a balanced pairwise tree of add over the
//             t_j (((t0 + t1) + (t2 + t3)) + ...), then lo + hi;
// per warp    add with shfl.bfly 16, 8, 4, 2, 1;
// per CTA     warp sums through shared memory, each lane reads warp (lane % kWarps) and adds with shfl.bfly
//             kWarps / 2 .. 1: ((W0 + W4) + (W2 + W6)) + ((W1 + W5) + (W3 + W7)) for 8 warps, (W0 + W2) + (W1 + W3)
//             for 4;
// then        rstd = rsqrt.approx.ftz(add(div.full(T, 5120), eps)); y = cvt.rn.bf16x2(mul(mul(rstd, v), w)).
// MXFP8 on the bf16 y, 32-column groups (4 lanes), FlashInfer's arithmetic (as lev_QN's mode 0): amax by its bf16x2
// and-mask / max.bf16x2 tree, max.f32 of the halves, max.f32 over lanes xor 1, 2; n = mul(amax, 1 / 448 as f32);
// FlashInfer's UE8M0 (positive rounding, subnormals) and inverse scale 2^(127 - sf) bit for bit; q =
// cvt.rn.satfinite.e4m3x2 of max(min(mul(y, inv), 448), -448).
// Every float op is inline PTX: bundle units build with --use_fast_math (ftz, approximate division, contraction), the
// references do not. One CTA a row; CTAs 0..39 also zero the padded rows' bytes of one 512-byte column tile of the last
// 128-row scale tile (the grid has at least 40 CTAs). PDL: the weight loads run before griddepcontrol.wait (nothing in
// the stream writes the weight), every other global access after it (an output buffer may be memory the previous
// kernel still reads). launch_dependents (run's pdl argument): 1 = right after the wait -- the consumer is wqkv_a's GEMM,
// which itself waits for this grid's completion before its loads (FlashInfer's dense_blockscaled_gemm_sm100), so only
// its setup overlaps (measured best on B300); 2 = after the stores, as FlashInfer's own kernels do; 0 = no PDL.
#include <cuda_runtime.h>
#include <stdint.h>

namespace hq {

constexpr int kH = 5120;              // hidden width (one stream)
constexpr int kStreams = 4;           // hyper-connection streams (hc_mult)
constexpr int kSfCols = kH / 32;      // 160 scale columns (a multiple of 4: no column padding)
constexpr int kTileBytes = 128 * kSfCols;
constexpr int kColTiles = kSfCols / 4;   // 40 column tiles of 512 bytes in a 128-row scale tile
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
__device__ __forceinline__ float mul_f(float a, float b) {
  float r;
  asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
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
__device__ __forceinline__ uint32_t max_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float round_bf16(float a) {   // cvt.rn.bf16.f32, widened back (exact)
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(a));
  return __uint_as_float(static_cast<uint32_t>(r) << 16);
}
__device__ __forceinline__ uint32_t pack_bf16x2(float hi, float lo) {   // cvt.rn.bf16x2.f32 d, hi, lo
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t e4m3x2(float hi, float lo) {         // cvt.rn.satfinite.e4m3x2.f32 d, hi, lo
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bfly(float v, int mask) { return __shfl_xor_sync(kFull, v, mask, 32); }

// FlashInfer's float_to_ue8m0_fast and ue8m0_to_inv_scale_fast, verbatim (as lev_QN)
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

// x and pre (written by the PDL primary) carry no __restrict__: plain coherent ld.global after griddepcontrol.wait, never
// ld.global.nc (the PTX memory model makes the primary's writes visible to coherent loads only); the weight, which no
// kernel writes, keeps its __ldg before the wait
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// a balanced pairwise add tree over n (a power of two) leaves, left to right at every level (Triton's reduce)
template <int kN>
__device__ __forceinline__ float tree(float* v) {
#pragma unroll
  for (int step = 1; step < kN; step <<= 1) {
#pragma unroll
    for (int i = 0; i < kN; i += 2 * step) v[i] = add_f(v[i], v[i + step]);
  }
  return v[0];
}

template <int kWarps>
__global__ void __launch_bounds__(32 * kWarps) combine_norm_mxfp8(const uint16_t* x, long long sx,
                                                                  const float* pre, long long sp,
                                                                  const uint16_t* __restrict__ w,
                                                                  uint16_t* __restrict__ y, uint8_t* __restrict__ q,
                                                                  uint8_t* __restrict__ sf, int m, float eps,
                                                                  int early) {
  constexpr int kThreads = 32 * kWarps;
  constexpr int kRep = 8 * kThreads;          // columns one rep of the Triton layout covers (2048 / 1024)
  constexpr int kReps = 8192 / kRep;          // reps over Triton's BLOCK 8192 (4 / 8)
  constexpr int kLive = (kH + kRep - 1) / kRep;   // reps that can hold columns < 5120 (3 / 5)
  constexpr int kPairs = 4 * kReps;           // f32x2 pair slots a thread holds (16 / 32)
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int row = blockIdx.x;
  __shared__ float red[kWarps];

  uint4 wv[kLive];
#pragma unroll
  for (int r = 0; r < kLive; ++r) {
    const int col = r * kRep + 8 * t;
    wv[r] = col < kH ? __ldg(reinterpret_cast<const uint4*>(w + col)) : make_uint4(0, 0, 0, 0);
  }
  pdl_wait();
  // the next kernel (wqkv_a's CuTe-DSL GEMM: griddepcontrol.wait before its first TMA load) may set up now; it still
  // waits for this grid's completion and memory flush before reading anything
  if (early) pdl_launch();
  if (row < m) {
    const float* pr = pre + static_cast<long long>(row) * sp;
    const float p0 = pr[0], p1 = pr[1], p2 = pr[2], p3 = pr[3];
    const uint16_t* xr = x + static_cast<long long>(row) * sx;
    float vlo[kPairs], vhi[kPairs];   // the thread's bf16-rounded combined values (pair lanes), +0 where masked
#pragma unroll
    for (int k = 0; k < kPairs; ++k) vlo[k] = vhi[k] = 0.0f;
#pragma unroll
    for (int r = 0; r < kLive; ++r) {
      const int col = r * kRep + 8 * t;
      if (col < kH) {
        uint4 xv[kStreams];
#pragma unroll
        for (int c = 0; c < kStreams; ++c) xv[c] = *reinterpret_cast<const uint4*>(xr + c * kH + col);
        const uint32_t s0[4] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w};
        const uint32_t s1[4] = {xv[1].x, xv[1].y, xv[1].z, xv[1].w};
        const uint32_t s2[4] = {xv[2].x, xv[2].y, xv[2].z, xv[2].w};
        const uint32_t s3[4] = {xv[3].x, xv[3].y, xv[3].z, xv[3].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          float lo = fma_rn(p0, bf_lo(s0[k]), 0.0f);
          float hi = fma_rn(p0, bf_hi(s0[k]), 0.0f);
          lo = fma_rn(p1, bf_lo(s1[k]), lo);
          hi = fma_rn(p1, bf_hi(s1[k]), hi);
          lo = fma_rn(p2, bf_lo(s2[k]), lo);
          hi = fma_rn(p2, bf_hi(s2[k]), hi);
          lo = fma_rn(p3, bf_lo(s3[k]), lo);
          hi = fma_rn(p3, bf_hi(s3[k]), hi);
          vlo[4 * r + k] = round_bf16(lo);
          vhi[4 * r + k] = round_bf16(hi);
        }
      }
    }
    // the thread's sum of squares: t_j = fma(P_2j, P_2j, P_2j+1 ^ 2) per pair lane, the pairwise tree, lo + hi
    float tl[kPairs / 2], th[kPairs / 2];
#pragma unroll
    for (int j = 0; j < kPairs / 2; ++j) {
      tl[j] = fma_rn(vlo[2 * j], vlo[2 * j], mul_f(vlo[2 * j + 1], vlo[2 * j + 1]));
      th[j] = fma_rn(vhi[2 * j], vhi[2 * j], mul_f(vhi[2 * j + 1], vhi[2 * j + 1]));
    }
#ifndef HQ_TREE_CHAIN
    float s = add_f(tree<kPairs / 2>(tl), tree<kPairs / 2>(th));
#else
    // tests only (hq_test.py's sensitivity control): the t_j chained left to right, which stock's tree is not
    float s = add_f(tl[0], th[0]);
#pragma unroll
    for (int j = 1; j < kPairs / 2; ++j) s = add_f(add_f(s, tl[j]), th[j]);
#endif
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = add_f(s, bfly(s, o));
    if (lane == 0) red[t >> 5] = s;
    __syncthreads();
    float tot = red[lane & (kWarps - 1)];
#pragma unroll
    for (int o = kWarps / 2; o > 0; o >>= 1) tot = add_f(tot, bfly(tot, o));
    const float rstd = rsqrt_ftz(add_f(div_full(tot, static_cast<float>(kH)), eps));

    uint16_t* yr = y + static_cast<long long>(row) * kH;
    uint8_t* qr = q + static_cast<long long>(row) * kH;
#pragma unroll
    for (int r = 0; r < kLive; ++r) {
      const int col = r * kRep + 8 * t;
      if (col < kH) {
        const uint32_t ww[4] = {wv[r].x, wv[r].y, wv[r].z, wv[r].w};
        uint32_t yw[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float lo = mul_f(mul_f(rstd, vlo[4 * r + k]), bf_lo(ww[k]));
          const float hi = mul_f(mul_f(rstd, vhi[4 * r + k]), bf_hi(ww[k]));
          yw[k] = pack_bf16x2(hi, lo);
        }
        *reinterpret_cast<uint4*>(yr + col) = make_uint4(yw[0], yw[1], yw[2], yw[3]);
        // MXFP8 group of 32 columns = 4 consecutive lanes (col / 32 = r * kRep / 32 + t / 4)
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
        *reinterpret_cast<uint2*>(qr + col) = make_uint2(qw[0], qw[1]);
        if ((lane & 3) == 0) sf[sf_offset(row, col >> 5)] = static_cast<uint8_t>(sc);
      }
    }
  }
  if (row < kColTiles) {
    // the padded rows of the last 128-row tile: their bytes of column tile `row` are zero (stock's quantize writes them)
    const int last = (m - 1) >> 7;
    uint8_t* tile = sf + static_cast<long long>(last) * kTileBytes + row * 512;
#pragma unroll
    for (int b = t; b < 512; b += kThreads) {
      const int prow = last * 128 + ((b >> 2) & 3) * 32 + (b >> 4);
      if (prow >= m) tile[b] = 0;
    }
  }
  if (!early) pdl_launch();
}

}  // namespace hq

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

// pointers: x [m, 4 * 5120] bf16 with row stride sx (elements, a multiple of 8), pre [m, 4] fp32 with row stride sp
// (elements), w [5120] bf16, y [m, 5120] bf16, q [m, 5120] e4m3, sf [padded_m * 160] u8 (padded_m = m rounded up to
// 128); warps 8 = stock's _hc_combine_norm (rows < 4096), 4 = _hc_combine_norm_prefill (rows >= 4096); pdl 0 = plain
// launch, 1 = PDL with the dependents released right after griddepcontrol.wait, 2 = PDL released after the stores
void run(int64_t x, int64_t sx, int64_t pre, int64_t sp, int64_t w, int64_t y, int64_t q, int64_t sf, int64_t m,
         double eps, int64_t warps, int64_t pdl, int64_t stream) {
  if (m < 1 || m > (int64_t(1) << 24)) throw std::runtime_error("hq: rows");
  if ((x | y | w | q) % 16 || sx % 8 || sx < hq::kStreams * hq::kH) throw std::runtime_error("hq: 16-byte aligned rows");
  if (pre % 4 || sp < hq::kStreams) throw std::runtime_error("hq: pre rows");
  if (warps != 8 && warps != 4) throw std::runtime_error("hq: warps 8 or 4");
  if (pdl < 0 || pdl > 2) throw std::runtime_error("hq: pdl 0, 1 or 2");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(m < hq::kColTiles ? hq::kColTiles : m));
  cfg.blockDim = dim3(static_cast<unsigned>(32 * warps));
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const auto* xp = reinterpret_cast<const uint16_t*>(x);
  const auto* pp = reinterpret_cast<const float*>(pre);
  const auto* wp = reinterpret_cast<const uint16_t*>(w);
  auto* yp = reinterpret_cast<uint16_t*>(y);
  auto* qp = reinterpret_cast<uint8_t*>(q);
  auto* sp8 = reinterpret_cast<uint8_t*>(sf);
  const long long sxl = static_cast<long long>(sx), spl = static_cast<long long>(sp);
  const int mi = static_cast<int>(m);
  const float e = static_cast<float>(eps);
  const int early = pdl == 1 ? 1 : 0;
  const cudaError_t rc =
      warps == 8 ? cudaLaunchKernelEx(&cfg, hq::combine_norm_mxfp8<8>, xp, sxl, pp, spl, wp, yp, qp, sp8, mi, e, early)
                 : cudaLaunchKernelEx(&cfg, hq::combine_norm_mxfp8<4>, xp, sxl, pp, spl, wp, yp, qp, sp8, mi, e, early);
  if (rc != cudaSuccess) throw std::runtime_error(std::string("hq: launch failed: ") + cudaGetErrorString(rc));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "hc combine + RMSNorm + MXFP8 quantize (hidden 5120, 4 streams), stock's bytes per row count");
}
