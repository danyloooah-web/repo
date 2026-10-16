// lev_SK: the mHC stats reduce + sinkhorn of a decode / verify sublayer (stock: Triton _hc_mix_reduce_sinkhorn_kernel,
// one 1-warp CTA a row), bit-identical, with the row's K-slice partials fetched in one memory round trip.
// Stock's single warp issues 80 slices x 4 scalar loads under a register cap, so the partials arrive over several L2
// round trips; beside the routed GEMMs that is 11-22 us a call on the k=1 critical path (the MoE finalize all-reduce
// waits for it), and at 144 rows its 144 CTAs hold every SM. Here a 128-thread CTA takes 4 rows: it loads all their
// partials into shared memory at once (float4), one thread per output sums its column over the slices in stock's order
// (acc = 0; acc += v_s for s ascending; deep_gemm path: v_s = hi_s + lo_s), and warp w runs row w's epilogue as stock
// does, instruction for instruction (Triton 3.8 PTX for sm_103a, num_warps 1):
//   rsqrt = div.full(1, sqrt.approx.ftz(fma.rn(inv_k, sq, rms_eps)))
//   pre   = div.full(1, ex2.approx(mul(fma.rn(mul(a_pre, rsqrt), s0, base), -log2e)) + 1) + eps
//   post  = 2 * the same sigmoid on a_post / s1 (as add(sig, sig))
//   comb  = fma.rn(mul(a_comb, rsqrt), s2, base); row max (bfly 2, 1); ex2.approx(mul(sub(c, max), log2e));
//           row sum (bfly 2, 1) -> div.full; + eps; col sum (bfly 8, 4) + eps -> div.full;
//           ITERS - 1 x { row sum + eps -> div.full; col sum + eps -> div.full }
// on stock's lane layout ([4] vectors: lane & 3; [4, 4]: col = lane & 3, row = (lane >> 2) & 3, lanes 16-31 mirror).
#include <cuda_runtime.h>
#include <stdint.h>

namespace sk {

constexpr int kThreads = 128;
constexpr int kMix = 24;          // (2 + HC) * HC at HC = 4
constexpr int kMaxSlices = 80;
constexpr int kVec = kMix / 4;    // float4 a slice row
constexpr int kMaxRows = 4;       // rows a CTA (one warp each for the epilogue)
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float ex2_approx(float a) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float sqrt_approx_ftz(float a) {
  float r;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
// stock's exact PTX ops (plain add / sub / mul: ptxas may contract them with div.full's multiply as it does for
// stock; non-ftz), whatever the build's fast-math / ftz flags
__device__ __forceinline__ float add_f(float a, float b) {
  float r;
  asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
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
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float max_f(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float bfly(float v, int mask) { return __shfl_xor_sync(kFull, v, mask, 32); }

// kRows rows a CTA (warp w runs row w's epilogue; 1 at small batches, where one row's loads per 128 threads is
// fastest, 4 for the deep_gemm path's 128+ rows: a quarter of stock's one-warp CTAs, fewer SMs held -- B300 runs one
// kernel's CTAs on an SM at a time). kResidual: the deep_gemm path's low part (stock: v = hi + lo a slice, then
// acc += v); kSlices: the slice cap (80 small path, 16 deep_gemm path).
template <int kRows, int kSlices, bool kResidual>
__global__ void __launch_bounds__(kThreads) reduce_sinkhorn(const float* __restrict__ part_mix,
                                                             const float* __restrict__ residual,
                                                             const float* __restrict__ part_sq,
                                                             const float* __restrict__ scale,
                                                             const float* __restrict__ base, float* __restrict__ pre,
                                                             float* __restrict__ post, float* __restrict__ comb, int m,
                                                             float inv_k, float rms_eps, int slices, int iters,
                                                             float eps) {
  constexpr int kLoads = (kRows * kSlices * kVec + kThreads - 1) / kThreads;
  constexpr int kSqLoads = (kRows * kSlices + kThreads - 1) / kThreads;
  __shared__ float4 s_mix[kRows][kSlices * kVec];
  __shared__ float s_sq[kRows][kSlices];
  __shared__ float s_sum[kRows][kMix + 1];
  const int row0 = blockIdx.x * kRows;
  const int rows = min(kRows, m - row0);
  const int t = threadIdx.x;
  const int n4 = slices * kVec;          // float4 a row
  const int total = rows * n4;
  // every partial in flight at once: registers first, then shared memory
  float4 v[kLoads];
  float4 w[kResidual ? kLoads : 1];
#pragma unroll
  for (int j = 0; j < kLoads; ++j) {
    const int i = t + j * kThreads;
    if (i < total) {
      const int rr = kRows == 1 ? 0 : i / n4;   // one row a CTA: no runtime division
      const int rest = i - rr * n4;
      const int s = rest / kVec;
      const int q = rest - s * kVec;
      const size_t at = (size_t(s) * m + row0 + rr) * kMix;
      v[j] = __ldcg(reinterpret_cast<const float4*>(part_mix + at) + q);
      if constexpr (kResidual) w[j] = __ldcg(reinterpret_cast<const float4*>(residual + at) + q);
    }
  }
  float sq[kSqLoads];
#pragma unroll
  for (int j = 0; j < kSqLoads; ++j) {
    const int i = t + j * kThreads;
    if (i < rows * slices) {
      const int rr = kRows == 1 ? 0 : i / slices;
      sq[j] = __ldcg(part_sq + size_t(i - rr * slices) * m + row0 + rr);
    }
  }
#pragma unroll
  for (int j = 0; j < kLoads; ++j) {
    const int i = t + j * kThreads;
    if (i < total) {
      const int rr = kRows == 1 ? 0 : i / n4;
      float4 x = v[j];
      if constexpr (kResidual) {       // stock: v = hi + lo (add.f32), before the running sum
        x.x = add_f(x.x, w[j].x);
        x.y = add_f(x.y, w[j].y);
        x.z = add_f(x.z, w[j].z);
        x.w = add_f(x.w, w[j].w);
      }
      s_mix[rr][i - rr * n4] = x;
    }
  }
#pragma unroll
  for (int j = 0; j < kSqLoads; ++j) {
    const int i = t + j * kThreads;
    if (i < rows * slices) {
      const int rr = kRows == 1 ? 0 : i / slices;
      s_sq[rr][i - rr * slices] = sq[j];
    }
  }
  __syncthreads();
  // stock's per-output order: acc = 0, acc += (combined) partial of slice 0, 1, ...
  if (t < rows * (kMix + 1)) {
    const int rr = kRows == 1 ? 0 : t / (kMix + 1);
    const int c = t - rr * (kMix + 1);
    float acc = 0.f;
    if (c < kMix) {
      const float* col = reinterpret_cast<const float*>(s_mix[rr]) + c;
#pragma unroll 8
      for (int s = 0; s < slices; ++s) acc = add_f(acc, col[s * kMix]);
    } else {
#pragma unroll 8
      for (int s = 0; s < slices; ++s) acc = add_f(acc, s_sq[rr][s]);
    }
    s_sum[rr][c] = acc;
  }
  __syncthreads();
  const int wr = t >> 5;
  if (wr >= rows) return;
  const int row = row0 + wr;
  const float* sum_r = s_sum[wr];
  const int lane = t & 31;
  const int j = lane & 3;
  const int r = (lane >> 2) & 3;
  const float log2e = __int_as_float(0x3FB8AA3B);
  const float neg_log2e = __int_as_float(0xBFB8AA3B);
  const float rsq = div_full(1.0f, sqrt_approx_ftz(fma_rn(inv_k, sum_r[kMix], rms_eps)));
  const float s0 = scale[0], s1 = scale[1], s2 = scale[2];

  float z = fma_rn(mul_f(sum_r[j], rsq), s0, base[j]);
  float sig = div_full(1.0f, add_f(ex2_approx(mul_f(z, neg_log2e)), 1.0f));
  const float pre_v = add_f(sig, eps);
  z = fma_rn(mul_f(sum_r[4 + j], rsq), s1, base[4 + j]);
  sig = div_full(1.0f, add_f(ex2_approx(mul_f(z, neg_log2e)), 1.0f));
  const float post_v = add_f(sig, sig);
  if (lane < 4) {
    pre[row * 4 + j] = pre_v;
    post[row * 4 + j] = post_v;
  }

  float c = fma_rn(mul_f(sum_r[8 + r * 4 + j], rsq), s2, base[8 + r * 4 + j]);
  float mx = max_f(c, bfly(c, 2));
  mx = max_f(mx, bfly(mx, 1));
  const float e = ex2_approx(mul_f(sub_f(c, mx), log2e));
  float sum = add_f(e, bfly(e, 2));
  sum = add_f(sum, bfly(sum, 1));
  c = add_f(div_full(e, sum), eps);
  sum = add_f(c, bfly(c, 8));
  sum = add_f(sum, bfly(sum, 4));
  c = div_full(c, add_f(sum, eps));
#pragma unroll 1
  for (int it = 1; it < iters; ++it) {
    sum = add_f(c, bfly(c, 2));
    sum = add_f(sum, bfly(sum, 1));
    c = div_full(c, add_f(sum, eps));
    sum = add_f(c, bfly(c, 8));
    sum = add_f(sum, bfly(sum, 4));
    c = div_full(c, add_f(sum, eps));
  }
  if (lane < 16) comb[row * 16 + r * 4 + j] = c;
}

}  // namespace sk

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

// pointers: part_mix [slices, m, 24] fp32 (+ residual of the same shape, or 0), part_sq [slices, m], scale [3],
// base [24], pre / post [m, 4], comb [m, 4, 4]
void run(int64_t part_mix, int64_t residual, int64_t part_sq, int64_t scale, int64_t base, int64_t pre, int64_t post,
         int64_t comb, int64_t m, double inv_k, double rms_eps, int64_t slices, int64_t iters, double eps,
         int64_t rows_per_cta, int64_t stream) {
  if (m < 1 || m > (int64_t(1) << 20)) throw std::runtime_error("sk: rows 1-2^20");
  if (slices < 1 || slices > sk::kMaxSlices || iters < 1) throw std::runtime_error("sk: 1-80 slices, iters >= 1");
  if (part_mix % 16 || residual % 16) throw std::runtime_error("sk: partials 16-byte aligned");
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  const auto launch = [&](auto kernel, int per) {
    const int grid = static_cast<int>((m + per - 1) / per);
    kernel<<<grid, sk::kThreads, 0, st>>>(
        reinterpret_cast<const float*>(part_mix), reinterpret_cast<const float*>(residual),
        reinterpret_cast<const float*>(part_sq), reinterpret_cast<const float*>(scale),
        reinterpret_cast<const float*>(base), reinterpret_cast<float*>(pre), reinterpret_cast<float*>(post),
        reinterpret_cast<float*>(comb), static_cast<int>(m), static_cast<float>(inv_k), static_cast<float>(rms_eps),
        static_cast<int>(slices), static_cast<int>(iters), static_cast<float>(eps));
  };
  if (rows_per_cta != 1 && rows_per_cta != sk::kMaxRows) throw std::runtime_error("sk: 1 or 4 rows a CTA");
  if (residual != 0) {
    if (slices > 16) throw std::runtime_error("sk: residual path takes <= 16 slices");
    if (rows_per_cta == 1) launch(sk::reduce_sinkhorn<1, 16, true>, 1);
    else launch(sk::reduce_sinkhorn<sk::kMaxRows, 16, true>, sk::kMaxRows);
  } else {
    if (rows_per_cta == 1) launch(sk::reduce_sinkhorn<1, sk::kMaxSlices, false>, 1);
    else launch(sk::reduce_sinkhorn<sk::kMaxRows, sk::kMaxSlices, false>, sk::kMaxRows);
  }
  const cudaError_t rc = cudaGetLastError();
  if (rc != cudaSuccess) throw std::runtime_error(std::string("sk: launch failed: ") + cudaGetErrorString(rc));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "mHC stats reduce + sinkhorn, bit-identical to the fork's Triton kernel (1 or 4 rows a CTA)");
}
