// lever MHC: the mHC stats PARTIAL of a decode / verify sublayer -- per K slice s (80 slices of 256 columns, stock's
// _num_slices_for(20480)) the 24 mixing dot products x[row] . W[:, slice] and the row sum of squares over the slice --
// written in stock's partial layout (part_mix [80, m, 24], part_sq [80, m] fp32), so lever SK's reduce + sinkhorn
// (sk_unit / skw_unit: slice-ordered sums, stock's epilogue) runs unchanged on them. Stock: the Triton
// _hc_mix_stats_partial_kernel (tf32x3 dot, one stage), 240..640 CTAs for 10..31 us in-engine at 48..127 rows.
// Two kernels:
//   xpartial_kernel (configs "x<MT>", SERVED): stock's arithmetic BIT FOR BIT (see its comment below), 8 warps a CTA,
//     one CTA a (slice, 16 MT rows), all loads of a warp in flight at once.
//   partial_kernel (configs "p<P>t<MT>", tolerance, kept for comparison): bf16 tensor cores (mma.sync m16n8k16,
//     fp32 accumulate) with x (bf16, exact) against the fork's bf16x3 split of the fp32 weight (split_bf16_hc_weight:
//     P = 3 parts ~24 bits, P = 2 ~16 bits), a fixed per-slice order (warp w of 8: columns [32 w, 32 w + 32) as two
//     k16 steps under one fixed column permutation for x and W; warps summed 0..7) -- NOT bit-identical to stock.
// Both: deterministic, the same in eager and CUDA-graph replay, a row's values independent of the row count, its
// position and the grid; no atomics, no PDL (inputs come from completed kernels), no local memory.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace mhc {

constexpr int kK = 20480;
constexpr int kMix = 24;
constexpr int kCols = kMix + 1;   // 24 mixes + the sum of squares
constexpr int kSlices = 80;
constexpr int kSliceCols = kK / kSlices;   // 256
constexpr int kWarps = kSliceCols / 32;    // 8: one 32-column group a warp
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float add_f(float a, float b) {
  float r;
  asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
// 16-byte global load (inputs written by kernels that completed before this launch: no PDL here)
__device__ __forceinline__ uint4 ld16(const void* p) {
  uint4 r;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(p));
  return r;
}
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float sq8(uint4 v, float s) {   // s + the squares of 8 bf16 in column order (fp32 FMA)
  const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float lo = __uint_as_float(u[i] << 16);
    const float hi = __uint_as_float(u[i] & 0xffff0000u);
    s = fma_rn(lo, lo, s);
    s = fma_rn(hi, hi, s);
  }
  return s;
}

template <int MT>
constexpr int smem_bytes() {
  return kWarps * MT * 16 * kCols * 4;
}

// P weight parts (2 or 3), MT m-tiles (16 rows each) a CTA
template <int P, int MT>
__global__ void __launch_bounds__(kWarps * 32, 1)
    partial_kernel(const __nv_bfloat16* __restrict__ x, long long xs, const __nv_bfloat16* __restrict__ w0,
                   const __nv_bfloat16* __restrict__ w1, const __nv_bfloat16* __restrict__ w2,
                   float* __restrict__ part_mix, float* __restrict__ part_sq, int m) {
  constexpr int kRows = MT * 16;
  extern __shared__ float s_part[];                 // [kWarps][kRows][kCols]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, q = lane & 3;
  const int slice = blockIdx.x;
  const int row_base = blockIdx.y * kRows;
  const int rows = min(kRows, m - row_base);
  const int ntiles = (rows + 15) >> 4;
  const int k = slice * kSliceCols + warp * 32 + 8 * q;
  const __nv_bfloat16* wp[3] = {w0, w1, w2};

  uint4 b[3][P];
#pragma unroll
  for (int t = 0; t < 3; ++t)
#pragma unroll
    for (int p = 0; p < P; ++p) b[t][p] = ld16(wp[p] + size_t(t * 8 + g) * kK + k);
  uint4 a_lo[MT], a_hi[MT];
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int r0 = mt * 16 + g, r1 = r0 + 8;
    a_lo[mt] = (r0 < rows) ? ld16(x + size_t(row_base + r0) * xs + k) : z;
    a_hi[mt] = (r1 < rows) ? ld16(x + size_t(row_base + r1) * xs + k) : z;
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    if (mt < ntiles) {                                 // warp-uniform
      float acc[3][4];
#pragma unroll
      for (int t = 0; t < 3; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
      // k16 step 0 = columns k + {0, 1, 2, 3} of lane q (MMA k slots 2q, 2q+1 | 2q+8, 2q+9), step 1 = k + {4..7}
#pragma unroll
      for (int t = 0; t < 3; ++t)
#pragma unroll
        for (int p = 0; p < P; ++p) {
          mma16816(acc[t], a_lo[mt].x, a_hi[mt].x, a_lo[mt].y, a_hi[mt].y, b[t][p].x, b[t][p].y);
          mma16816(acc[t], a_lo[mt].z, a_hi[mt].z, a_lo[mt].w, a_hi[mt].w, b[t][p].z, b[t][p].w);
        }
      float s0 = sq8(a_lo[mt], 0.f), s1 = sq8(a_hi[mt], 0.f);
      s0 = add_f(s0, __shfl_xor_sync(kFull, s0, 1));
      s0 = add_f(s0, __shfl_xor_sync(kFull, s0, 2));
      s1 = add_f(s1, __shfl_xor_sync(kFull, s1, 1));
      s1 = add_f(s1, __shfl_xor_sync(kFull, s1, 2));
      float* p0 = s_part + (warp * kRows + mt * 16 + g) * kCols;
      float* p1 = p0 + 8 * kCols;
#pragma unroll
      for (int t = 0; t < 3; ++t) {
        p0[t * 8 + 2 * q] = acc[t][0];
        p0[t * 8 + 2 * q + 1] = acc[t][1];
        p1[t * 8 + 2 * q] = acc[t][2];
        p1[t * 8 + 2 * q + 1] = acc[t][3];
      }
      if (q == 0) {
        p0[kMix] = s0;
        p1[kMix] = s1;
      }
    }
  }
  __syncthreads();
  // the slice's partial: warps summed in order 0..7, stored in stock's layout
  for (int i = threadIdx.x; i < rows * kCols; i += kWarps * 32) {
    float a = s_part[i];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) a = add_f(a, s_part[w * kRows * kCols + i]);
    const int r = i / kCols, c = i - r * kCols;
    const int row = row_base + r;
    if (c < kMix) part_mix[(size_t(slice) * m + row) * kMix + c] = a;
    else part_sq[size_t(slice) * m + row] = a;
  }
}


// ---------------------------------------------------------------------------------------------------------------------
// EXACT kernel (xpartial, the served one): stock's Triton _hc_mix_stats_partial_kernel arithmetic (Triton 3.8 PTX for
// sm_103a: BLOCK_K 64, DOT_PRECISION tf32x3, the kWidth = 2 operand layout), bit for bit, re-issued with all loads of a
// warp in flight at once. Per 64-column block kb of a slice:
//   d = 0; pass 1: d = mma.m16n8k8.tf32(x - rna(x), rna(w), d) over the block's 8 k8 steps; pass 2:
//   d = mma(rna(x), w - rna(w), d); d = isnan(d) ? 0 : d; pass 3: d = mma(rna(x), rna(w), d); acc = d + acc (kb
//   ascending from +0). k8 step 2t takes the EVEN columns of the block's 16-column chunk t, step 2t + 1 the odd ones
//   (slot q <-> column 16t + 2q + (step & 1), slot q + 4 <-> + 8): Triton's kWidth = 2 tf32 operand layout.
//   Row sum of squares of the block, per 32-column half h (column j = bits b4..b0): Triton's reduction tree:
//   s = fma(x_j, x_j, rn(x_{j^16}^2)) (b4), then + over b3, b2, b1, b0 in that order; block = half0 + half1;
//   sq = block + sq (kb ascending from +0). (fp32 adds are commutative, so only the tree shape is replicated.)
// Verified bitwise vs stock's kernel (2- and 4-warp launch shapes, rows 1..127, special values) by tests/mhc_micro.py X.
// One CTA a (slice, 16 MT rows), 8 warps: warp w takes column block kb = w & 3 and m-tiles (w >> 2) + 2 i, all three
// n-tiles (24 mixes) as three interleaved MMA chains. Plain IEEE fp32 ops in inline asm (the build uses fast math).
__device__ __forceinline__ float sub_f(float a, float b) {
  float r;
  asm("sub.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float mul_f(float a, float b) {
  float r;
  asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint32_t rna_tf32(float a) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float num_or_zero(float a) {
  float r;
  asm("{\n .reg .pred p;\n setp.num.f32 p, %1, %1;\n selp.f32 %0, %1, 0f00000000, p;\n}" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ld4(const void* p) {
  uint32_t r;
  asm("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ float2 ld8f(const void* p) {
  float2 r;
  asm("ld.global.nc.v2.f32 {%0, %1}, [%2];" : "=f"(r.x), "=f"(r.y) : "l"(p));
  return r;
}

constexpr int kXWarps = 8;
template <int MT>
constexpr int xsmem_bytes() {
  return (4 * MT * 16 * kMix + 4 * MT * 16) * 4;
}

template <int MT>
__global__ void __launch_bounds__(kXWarps * 32, 1)
    xpartial_kernel(const __nv_bfloat16* __restrict__ x, long long xs, const float* __restrict__ w,
                    float* __restrict__ part_mix, float* __restrict__ part_sq, int m) {
  constexpr int kRows = MT * 16;
  extern __shared__ float xsm[];
  float* s_d = xsm;                       // [4 kb][kRows][24]
  float* s_b = xsm + 4 * kRows * kMix;    // [4 kb][kRows]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q = lane & 3;
  const int slice = blockIdx.x;
  const int row_base = blockIdx.y * kRows;
  const int rows = min(kRows, m - row_base);
  const int ntiles = (rows + 15) >> 4;
  const int kb = warp & 3;
  const int col0 = slice * kSliceCols + kb * 64 + 2 * q;   // + 16 t (+ 8)
  // B: W[n][col0 + 16t + {0, 1, 8, 9}] for the three n-tiles (n = 8 nt + g): big / small tf32 parts
  uint32_t bb[3][4][4], bs[3][4][4];
#pragma unroll
  for (int nt = 0; nt < 3; ++nt) {
    const float* wr = w + size_t(nt * 8 + g) * kK + col0;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float2 lo = ld8f(wr + 16 * t), hi = ld8f(wr + 16 * t + 8);
      const float v[4] = {lo.x, lo.y, hi.x, hi.y};   // (step 2t, slot q), (2t + 1, q), (2t, q + 4), (2t + 1, q + 4)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        bb[nt][t][e] = rna_tf32(v[e]);
        bs[nt][t][e] = __float_as_uint(sub_f(v[e], __uint_as_float(bb[nt][t][e])));
      }
    }
  }
  for (int tile = warp >> 2; tile < ntiles; tile += kXWarps / 4) {
    const int r0 = tile * 16 + g, r1 = r0 + 8;
    // A: x[r][col0 + 16t + {0, 1}] and [+ 8, + 9] (bf16 pairs), rows r0 / r1; zero beyond the rows
    float xa[2][4][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = h ? r1 : r0;
      const bool in = r < rows;
      const __nv_bfloat16* xr = x + size_t(row_base + (in ? r : 0)) * xs + col0;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const uint32_t p0 = in ? ld4(xr + 16 * t) : 0u, p1 = in ? ld4(xr + 16 * t + 8) : 0u;
        xa[h][t][0] = __uint_as_float(p0 << 16);          // column 16t + 2q     (step 2t, slot q)
        xa[h][t][1] = __uint_as_float(p0 & 0xffff0000u);  // column 16t + 2q + 1 (step 2t + 1, slot q)
        xa[h][t][2] = __uint_as_float(p1 << 16);          // column 16t + 2q + 8 (step 2t, slot q + 4)
        xa[h][t][3] = __uint_as_float(p1 & 0xffff0000u);  // column 16t + 2q + 9 (step 2t + 1, slot q + 4)
      }
    }
    float d[3][4];
#pragma unroll
    for (int nt = 0; nt < 3; ++nt) d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.f;
    // pass 1: (x - rna(x)) x rna(w)
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      const int t = st >> 1, o = st & 1;
      const uint32_t a0 = __float_as_uint(sub_f(xa[0][t][o], __uint_as_float(rna_tf32(xa[0][t][o]))));
      const uint32_t a1 = __float_as_uint(sub_f(xa[1][t][o], __uint_as_float(rna_tf32(xa[1][t][o]))));
      const uint32_t a2 = __float_as_uint(sub_f(xa[0][t][2 + o], __uint_as_float(rna_tf32(xa[0][t][2 + o]))));
      const uint32_t a3 = __float_as_uint(sub_f(xa[1][t][2 + o], __uint_as_float(rna_tf32(xa[1][t][2 + o]))));
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) mma1688(d[nt], a0, a1, a2, a3, bb[nt][t][o], bb[nt][t][2 + o]);
    }
    // pass 2: rna(x) x (w - rna(w))
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      const int t = st >> 1, o = st & 1;
      const uint32_t a0 = rna_tf32(xa[0][t][o]), a1 = rna_tf32(xa[1][t][o]);
      const uint32_t a2 = rna_tf32(xa[0][t][2 + o]), a3 = rna_tf32(xa[1][t][2 + o]);
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) mma1688(d[nt], a0, a1, a2, a3, bs[nt][t][o], bs[nt][t][2 + o]);
    }
#pragma unroll
    for (int nt = 0; nt < 3; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) d[nt][e] = num_or_zero(d[nt][e]);
    // pass 3: rna(x) x rna(w)
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      const int t = st >> 1, o = st & 1;
      const uint32_t a0 = rna_tf32(xa[0][t][o]), a1 = rna_tf32(xa[1][t][o]);
      const uint32_t a2 = rna_tf32(xa[0][t][2 + o]), a3 = rna_tf32(xa[1][t][2 + o]);
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) mma1688(d[nt], a0, a1, a2, a3, bb[nt][t][o], bb[nt][t][2 + o]);
    }
#pragma unroll
    for (int nt = 0; nt < 3; ++nt) {
      float* o0 = s_d + (kb * kRows + tile * 16 + g) * kMix + nt * 8 + 2 * q;
      float* o1 = o0 + 8 * kMix;
      o0[0] = d[nt][0];
      o0[1] = d[nt][1];
      o1[0] = d[nt][2];
      o1[1] = d[nt][3];
    }
    // the block's row sums of squares (Triton's tree): this thread holds, per row and half h (t = 2h, 2h + 1), the
    // columns j = 16 (t - 2h) [b4] + 8 e [b3] + 2 q [b2 b1] + f [b0]
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      float half[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float s4[2];
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          float s2 = 0.f;
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float a = xa[hr][2 * h][2 * e + f], b = xa[hr][2 * h + 1][2 * e + f];   // b4 = 0 / 1
            const float s1 = fma_rn(a, a, mul_f(b, b));
            s2 = e ? add_f(s2, s1) : s1;                                                  // b3
          }
          float s3 = add_f(s2, __shfl_xor_sync(kFull, s2, 2));                            // b2 (q bit 1)
          s4[f] = add_f(s3, __shfl_xor_sync(kFull, s3, 1));                               // b1 (q bit 0)
        }
        half[h] = add_f(s4[0], s4[1]);                                                    // b0
      }
      if (q == 0) s_b[kb * kRows + tile * 16 + g + 8 * hr] = add_f(half[0], half[1]);
    }
  }
  __syncthreads();
  for (int i = tid; i < rows * kCols; i += kXWarps * 32) {
    const int r = i / kCols, c = i - r * kCols;
    float a = 0.f;
    if (c < kMix) {
#pragma unroll
      for (int k4 = 0; k4 < 4; ++k4) a = add_f(s_d[(k4 * kRows + r) * kMix + c], a);
      part_mix[(size_t(slice) * m + row_base + r) * kMix + c] = a;
    } else {
#pragma unroll
      for (int k4 = 0; k4 < 4; ++k4) a = add_f(s_b[k4 * kRows + r], a);
      part_sq[size_t(slice) * m + row_base + r] = a;
    }
  }
}

struct Config {
  const char* name;
  int parts, mt;
  const void* fn;
  int smem;
};
#define MHC_CFG(P, MT) {"p" #P "t" #MT, P, MT, reinterpret_cast<const void*>(&partial_kernel<P, MT>), smem_bytes<MT>()}
#define MHC_XCFG(MT) {"x" #MT, 0, MT, reinterpret_cast<const void*>(&xpartial_kernel<MT>), xsmem_bytes<MT>()}
static const Config kConfigs[] = {MHC_CFG(3, 8), MHC_CFG(3, 4), MHC_CFG(2, 8), MHC_CFG(2, 4),
                                  MHC_XCFG(4), MHC_XCFG(8), MHC_XCFG(2)};
#undef MHC_XCFG
#undef MHC_CFG
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);
static bool g_init = false;

}  // namespace mhc

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

namespace mhc {

void init() {
  if (g_init) return;
  for (int i = 0; i < kNumConfigs; ++i) {
    const Config& c = kConfigs[i];
    const cudaError_t rc = cudaFuncSetAttribute(c.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, c.smem);
    if (rc != cudaSuccess)
      throw std::runtime_error(std::string("mhc: cudaFuncSetAttribute failed for ") + c.name + ": " +
                               cudaGetErrorString(rc));
  }
  g_init = true;
}

int config_index(const std::string& name) {
  for (int i = 0; i < kNumConfigs; ++i)
    if (name == kConfigs[i].name) return i;
  return -1;
}

int64_t config_ctas(int64_t cfg_index, int64_t m) {
  if (cfg_index < 0 || cfg_index >= kNumConfigs || m < 1) return -1;
  const int rows = kConfigs[cfg_index].mt * 16;
  return int64_t(kSlices) * ((m + rows - 1) / rows);
}

// x [m, 20480] bf16 (row stride xs elements, unit column stride), w0 / w1 / w2 [24, 20480] bf16 contiguous (w2 unused
// at 2 parts); part_mix [80, m, 24], part_sq [80, m] fp32 contiguous (stock's partial layout)
void run(int64_t cfg_index, int64_t x, int64_t xs, int64_t w0, int64_t w1, int64_t w2, int64_t part_mix,
         int64_t part_sq, int64_t m, int64_t stream) {
  if (!g_init) throw std::runtime_error("mhc: init() first (outside capture)");
  if (cfg_index < 0 || cfg_index >= kNumConfigs) throw std::runtime_error("mhc: bad config index");
  if (m < 1 || m > 4096) throw std::runtime_error("mhc: rows 1-4096");
  if ((x | w0 | w1 | w2) % 16 || xs % 8 || xs < kK) throw std::runtime_error("mhc: 16-byte aligned rows");
  if ((part_mix | part_sq) % 4) throw std::runtime_error("mhc: partials 4-byte aligned");
  const Config& c = kConfigs[cfg_index];
  const int rows = c.mt * 16;
  const dim3 grid(kSlices, static_cast<unsigned>((m + rows - 1) / rows));
  if (c.parts == 0) {   // exact variant: w0 = the fp32 mixing weight [24, 20480]
    using KX = void (*)(const __nv_bfloat16*, long long, const float*, float*, float*, int);
    const KX fx = reinterpret_cast<KX>(const_cast<void*>(c.fn));
    fx<<<grid, kXWarps * 32, c.smem, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), static_cast<long long>(xs), reinterpret_cast<const float*>(w0),
        reinterpret_cast<float*>(part_mix), reinterpret_cast<float*>(part_sq), static_cast<int>(m));
    const cudaError_t rx = cudaGetLastError();
    if (rx != cudaSuccess) throw std::runtime_error(std::string("mhc: launch failed: ") + cudaGetErrorString(rx));
    return;
  }
  using K = void (*)(const __nv_bfloat16*, long long, const __nv_bfloat16*, const __nv_bfloat16*,
                     const __nv_bfloat16*, float*, float*, int);
  const K fn = reinterpret_cast<K>(const_cast<void*>(c.fn));
  fn<<<grid, kWarps * 32, c.smem, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(x), static_cast<long long>(xs), reinterpret_cast<const __nv_bfloat16*>(w0),
      reinterpret_cast<const __nv_bfloat16*>(w1), reinterpret_cast<const __nv_bfloat16*>(c.parts == 3 ? w2 : w1),
      reinterpret_cast<float*>(part_mix), reinterpret_cast<float*>(part_sq), static_cast<int>(m));
  const cudaError_t rc = cudaGetLastError();
  if (rc != cudaSuccess) throw std::runtime_error(std::string("mhc: launch failed: ") + cudaGetErrorString(rc));
}

}  // namespace mhc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &mhc::init, "set the configs' shared memory limits (outside capture)");
  m.def("config_index", &mhc::config_index, "config index of a name such as 'p3t8' (-1 = none)");
  m.def("config_ctas", &mhc::config_ctas, "CTAs a config launches at a row count");
  m.def("run", &mhc::run, "mHC stats partials (80 K slices, stock's layout) on bf16x3 tensor cores");
  m.attr("NUM_CONFIGS") = int64_t(mhc::kNumConfigs);
  m.attr("SLICES") = int64_t(mhc::kSlices);
}
