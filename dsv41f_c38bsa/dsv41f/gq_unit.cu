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
// + lev_PWN (gen_pwn.py): namespace pwn below (kits/pwn_unit.cu / kits/pwn_host.cu): lever PWS's entry-trigger
//   MXFP8 quantize of wo_b's / wq_b's inputs (K 4096 / 1280) as a native kernel; lev_GQ's code unchanged.
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

// ---------------------------------------------------------------------------------------------------- lev_PWN
// lever PWN (gen_pwn.py inserts this text into the bundle COPY of gq_unit.cu; it uses namespace gq's helpers above):
// FlashInfer 0.7.1's MXFP8 quantize of bf16 rows (mxfp8_quantize_cute_dsl(x, is_sf_swizzled_layout=True,
// alignment=32) -> MXFP8QuantizeSwizzledKernel at use_2t_per_sf False, 128x4 layout, the small-K multi-row path) as a
// native kernel whose PDL trigger is at its ENTRY: griddepcontrol.launch_dependents, then griddepcontrol.wait, then the
// first global access. Lever PWS's dense MXFP8 GEMM after it (wo_b, the indexer's wq_b, wq_b's quantize form) may then
// land its CTAs and pull its static weight while this grid still waits on its predecessor; the GEMM's own
// griddepcontrol.wait still waits for this whole grid's completion and memory flush, so the trigger point cannot
// change a byte.
//
// Launch = FlashInfer's: _compute_optimal_warps(K, 8) warps a CTA (K 4096: 32 -> 1024 threads, 2 rows a CTA
// iteration; K 1280: 30 -> 960 threads, 6 rows), grid min(ceil(padded_m / rows), 4 x SMs) (host side), a grid-stride
// loop over row batches of the padded rows. Thread t: row t / (4 K/32), scale column (t % (4 K/32)) / 4, part t % 4
// (8 columns: one 16-byte load). Arithmetic = FlashInfer's, op for op (gq::quant_chunk's ops, as lev_GQ / lev_HQ /
// lev_QN mode 0): amax by and 0x7FFF7FFF / max.bf16x2 tree, max.f32 of the halves, max.f32 over lanes xor 1, 2;
// ue8m0(mul.f32(amax, 1 / 448)); inverse scale; q = cvt.rn.satfinite.e4m3x2(max(min(mul(x, inv), 448), -448)); the
// scale byte by part 0; a padded row (m <= row < padded_m) writes 0 to each of its scale bytes. Every float op is
// inline PTX (bundle units build with --use_fast_math). Loads: plain coherent ld.global.v4.u32 after the wait (x is
// the predecessor's output: never ld.global.nc / __ldg / const __restrict__).
namespace pwn {

template <int K, int kWarps>
struct Cfg {
  static constexpr int kSf = K / 32;                 // scale blocks a row
  static constexpr int kPadSf = (kSf + 3) / 4 * 4;   // padded scale columns
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kTpr = kSf * 4;               // threads a row (4 threads a scale block)
  static constexpr int kRpb = kThreads / 4 / kSf;    // rows a CTA iteration
  static_assert(K % 32 == 0 && kSf <= kThreads / 4, "FlashInfer's multi-row (small K) path");
  static_assert(kRpb * kTpr == kThreads, "FlashInfer's optimal warps: every thread owns a column unit");
  static_assert(kTpr % 32 == 0, "a warp within one row (the lane-xor shuffles see one row)");
};

__device__ __forceinline__ uint4 ld_v4(const uint16_t* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

// compute_sf_index_swizzled_128x4_gpu(row, col, padded_cols)
template <int kPadSf>
__device__ __forceinline__ long long sf_off(int row, int col) {
  return (col & 3) + (col >> 2) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 +
         static_cast<long long>(row >> 7) * (128 * kPadSf);
}

template <int K, int kWarps>
__global__ void __launch_bounds__(kWarps * 32, 2) quant_kernel(const uint16_t* x, long long sx,
                                                               uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                                               int m, int padded_m) {
  using C = Cfg<K, kWarps>;
  gq::pdl_launch();   // lever PWN: the dependents may launch now (entry trigger)
  gq::pdl_wait();     // before any global access
  const int t = threadIdx.x;
  const int row_in_block = t / C::kTpr;
  const int local = t % C::kTpr;
  const int sf_col = local >> 2;
  const int part = local & 3;
  for (int rb = static_cast<int>(blockIdx.x); rb * C::kRpb < padded_m; rb += static_cast<int>(gridDim.x)) {
    const int row = rb * C::kRpb + row_in_block;   // warp-uniform (a warp lies within one row)
    if (row >= padded_m) continue;
    if (row >= m) {   // padded row: zero its scale bytes
      if (part == 0) {
        for (int pc = sf_col; pc < C::kPadSf; pc += C::kSf) sf[sf_off<C::kPadSf>(row, pc)] = 0;
      }
      continue;
    }
    const int e = sf_col * 32 + part * 8;
    const uint4 v = ld_v4(x + static_cast<long long>(row) * sx + e);
    const uint32_t yw[4] = {v.x, v.y, v.z, v.w};
    const uint32_t a0 = yw[0] & 0x7FFF7FFFu, a1 = yw[1] & 0x7FFF7FFFu;
    const uint32_t a2 = yw[2] & 0x7FFF7FFFu, a3 = yw[3] & 0x7FFF7FFFu;
    const uint32_t mx = gq::max_bf16x2(gq::max_bf16x2(a0, a1), gq::max_bf16x2(a2, a3));
    float amax = gq::max_f(gq::bf_lo(mx), gq::bf_hi(mx));
    amax = gq::max_f(amax, gq::bfly(amax, 1));
    amax = gq::max_f(amax, gq::bfly(amax, 2));
    const uint32_t sc = gq::ue8m0(gq::mul_f(amax, 0.002232142857142857f));
    const float inv = gq::inv_scale(sc);
    uint32_t qw[2];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float lo = gq::max_f(gq::min_f(gq::mul_f(gq::bf_lo(yw[k]), inv), 448.0f), -448.0f);
      const float hi = gq::max_f(gq::min_f(gq::mul_f(gq::bf_hi(yw[k]), inv), 448.0f), -448.0f);
      const uint32_t pair = gq::e4m3x2(hi, lo);
      if (k & 1) qw[k >> 1] |= pair << 16;
      else qw[k >> 1] = pair;
    }
    *reinterpret_cast<uint2*>(q + static_cast<long long>(row) * K + e) = make_uint2(qw[0], qw[1]);
    if (part == 0) sf[sf_off<C::kPadSf>(row, sf_col)] = static_cast<uint8_t>(sc);
  }
}

using Cfg4096 = Cfg<4096, 32>;   // wo_b's input
using Cfg1280 = Cfg<1280, 30>;   // wq_b's / the indexer wq_b's input

}  // namespace pwn
// ---------------------------------------------------------------------------------------------------- lev_PWN end

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

// lever PWN host side (gen_pwn.py inserts this before the bundle copy's PYBIND11_MODULE).
// x [m, k] bf16 rows with row stride sx (elements; 16-byte aligned rows), q [m, k] e4m3 (contiguous), sf
// [padded_m * k / 32] u8 (padded_m = m rounded up to 128); k 4096 or 1280; grid = FlashInfer's num_blocks (the caller
// computes min(ceil(padded_m / rows), 4 x SMs) with pwn_cfg's rows); always a PDL launch (programmatic stream
// serialization), as FlashInfer's own launch on SM >= 90.
template <int K, int kWarps>
static cudaError_t pwn_launch(int64_t x, int64_t sx, int64_t q, int64_t sf, int64_t m, int64_t grid, int64_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(static_cast<unsigned>(pwn::Cfg<K, kWarps>::kThreads));
  cfg.dynamicSmemBytes = 0;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const int padded = static_cast<int>(((m + 127) / 128) * 128);
  return cudaLaunchKernelEx(&cfg, pwn::quant_kernel<K, kWarps>, reinterpret_cast<const uint16_t*>(x),
                            static_cast<long long>(sx), reinterpret_cast<uint8_t*>(q), reinterpret_cast<uint8_t*>(sf),
                            static_cast<int>(m), padded);
}

void pwn_run(int64_t x, int64_t sx, int64_t q, int64_t sf, int64_t m, int64_t k, int64_t grid, int64_t stream) {
  if (k != 4096 && k != 1280) throw std::runtime_error("pwn: k 4096 or 1280");
  if (m < 1 || m * (k / 32) >= 65536) throw std::runtime_error("pwn: rows (FlashInfer's 4-threads-a-block range)");
  if (grid < 1 || grid > 65535) throw std::runtime_error("pwn: grid");
  if ((x | q) % 16 || sx % 8 || sx < k) throw std::runtime_error("pwn: 16-byte aligned rows");
  const cudaError_t rc = k == 4096 ? pwn_launch<4096, 32>(x, sx, q, sf, m, grid, stream)
                                   : pwn_launch<1280, 30>(x, sx, q, sf, m, grid, stream);
  if (rc != cudaSuccess) throw std::runtime_error(std::string("pwn: launch failed: ") + cudaGetErrorString(rc));
}

// (threads a CTA, rows a CTA iteration) of the k kernel: FlashInfer's warps_per_block * 32 and rows_per_block
pybind11::tuple pwn_cfg(int64_t k) {
  if (k == 4096) return pybind11::make_tuple(pwn::Cfg4096::kThreads, pwn::Cfg4096::kRpb);
  if (k == 1280) return pybind11::make_tuple(pwn::Cfg1280::kThreads, pwn::Cfg1280::kRpb);
  throw std::runtime_error("pwn: k 4096 or 1280");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "MXFP8 quantize of bf16 rows of hidden 5120 (FlashInfer's 128x4 swizzled scales), stock's bytes");
  m.def("pwn", &pwn_run, "lever PWN: FlashInfer's MXFP8 quantize (bf16, K 4096 / 1280, 128x4 swizzled scales) at FlashInfer's launch with the PDL trigger at entry, stock's bytes");
  m.def("pwn_cfg", &pwn_cfg, "lever PWN: (threads a CTA, rows a CTA iteration) of the K kernel");
}
