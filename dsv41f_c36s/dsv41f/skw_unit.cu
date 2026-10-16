// lev_SKW: lever SK's mHC stats reduce + sinkhorn (sk_unit.cu, itself bit-identical to the fork's Triton
// _hc_mix_reduce_sinkhorn_kernel) on a CAPPED grid: G <= grid_cap CTAs of 32 x R threads walk the row blocks of R rows
// (block b, b + G, b + 2G, ...), so the kernel never holds more than grid_cap SMs however many rows the batch has.
//
// Why: B300 runs ONE kernel's CTAs on an SM at a time. sk_unit takes one 128-thread CTA a row (120-144 CTAs at 120-144
// verify rows) on the low-priority mHC stats stream; where its launch races the main chain it lands on the SMs that the
// one-CTA-a-row persistent sparse-attention kernel (fmha, 120 / 144 CTAs) or the routed FC1 needs at launch: fmha then
// waits ~5-10 us for its last CTAs (c26rtr k20: fmha 17.8 -> 24.6 us in 34 % of the attention halves), FC1 starts
// ~3 us late and runs ~2 us longer (k24: +7.4 us a layer in 28 % of the FFN halves), and when the sinkhorn loses the
// race the other way it can end after FC2 (the MoE all-reduce waits). The race is decided per rank, so it is also the
// largest source of TP cross-rank skew at the all-reduces. Lever SKW serves <= 136 rows with <= 8 CTAs (kits/skw.py):
// fmha's <= 136 CTAs always find free SMs.
//
// Exactness: every row runs sk_unit's arithmetic unchanged -- the same partial loads (__ldcg, fp32), the deep_gemm
// path's v = hi + lo (add.f32) per slice before the sum, each output's running sum acc = 0; acc += v_s for s ascending
// (add.f32), and the epilogue below copied from sk_unit.cu instruction for instruction (div.full / ex2.approx /
// sqrt.approx.ftz / fma.rn / non-ftz add / sub / mul / max as inline PTX, warp butterflies 2 / 1 rows, 8 / 4 columns on
// stock's lane layout). Only WHICH CTA / warp computes a row changes; rows are independent, so the bytes are those of
// sk_unit (and stock). Checked bitwise vs sk_unit (and stock) by tests/skw_test.py, by a self-test at prepare (both
// ranks, agreed) and by the first eager call per (rows, path, slices) in the engine (kits/skw.py).
// No PDL: launched like sk_unit on the caller's stream after its partials are complete.
#include <cuda_runtime.h>
#include <stdint.h>

namespace skw {

constexpr int kMix = 24;          // (2 + HC) * HC at HC = 4
constexpr int kMaxSlices = 80;
constexpr int kVec = kMix / 4;    // float4 a slice row
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
// stock's exact PTX ops (plain add / sub / mul, non-ftz), whatever the build's fast-math / ftz flags (= sk_unit.cu)
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

template <int kR, int kSlices, bool kLowCopy = false>
constexpr int smem_bytes() {
  return (kLowCopy ? 2 : 1) * kR * kSlices * kVec * 16 + kR * kSlices * 4 + kR * (kMix + 1) * 4;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// kR rows a block (warp w < kR runs row w's epilogue), kThreads >= 32 x kR threads (all of them load), row blocks
// walked with stride gridDim.x. kResidual: the deep_gemm path's low part; kSlices: the slice cap (80 small path, 16
// deep_gemm path).
template <int kR, int kThreads, int kSlices, bool kResidual, bool kAsync>
__global__ void __launch_bounds__(kThreads) reduce_sinkhorn_g(const float* __restrict__ part_mix,
                                                              const float* __restrict__ residual,
                                                              const float* __restrict__ part_sq,
                                                              const float* __restrict__ scale,
                                                              const float* __restrict__ base, float* __restrict__ pre,
                                                              float* __restrict__ post, float* __restrict__ comb, int m,
                                                              float inv_k, float rms_eps, int slices, int iters,
                                                              float eps) {
  static_assert(kThreads >= 32 * kR && kThreads <= 1024 && kThreads % 32 == 0, "threads");
  static_assert(kR * (kMix + 1) <= kThreads, "one thread an output");
  constexpr int kLoads = (kR * kSlices * kVec + kThreads - 1) / kThreads;
  constexpr int kSqLoads = (kR * kSlices + kThreads - 1) / kThreads;
  constexpr int kRow4 = kSlices * kVec;            // float4 a row in shared memory
  constexpr bool kLow = kAsync && kResidual;      // async path: the low part lands in its own array
  extern __shared__ float4 smem4[];
  float4* s_mix = smem4;                                                     // [kR][kSlices * kVec]
  float4* s_low = smem4 + kR * kRow4;                                        // [kR][kSlices * kVec] (kLow only)
  float* s_sq = reinterpret_cast<float*>(smem4 + (kLow ? 2 : 1) * kR * kRow4);  // [kR][kSlices]
  float* s_sum = s_sq + kR * kSlices;                                        // [kR][kMix + 1]
  const int t = threadIdx.x;
  const int n4 = slices * kVec;          // float4 a row in global memory
  const int wr = t >> 5;
  const int lane = t & 31;
  for (int blk = blockIdx.x; blk * kR < m; blk += gridDim.x) {
    const int row0 = blk * kR;
    const int rows = min(kR, m - row0);
    const int total = rows * n4;
    if constexpr (kAsync) {
      // every partial of the block in flight at once, global -> shared memory with no register staging (cp.async,
      // L2-coherent .cg for the 16-byte words); the bytes that land are the bytes sk_unit's __ldcg reads
#pragma unroll
      for (int j = 0; j < kLoads; ++j) {
        const int i = t + j * kThreads;
        if (i < total) {
          const int rr = i / n4;
          const int rest = i - rr * n4;
          const int s = rest / kVec;
          const int q = rest - s * kVec;
          const size_t at = (size_t(s) * m + row0 + rr) * kMix;
          cp_async16(s_mix + rr * kRow4 + rest, reinterpret_cast<const float4*>(part_mix + at) + q);
          if constexpr (kResidual) cp_async16(s_low + rr * kRow4 + rest, reinterpret_cast<const float4*>(residual + at) + q);
        }
      }
#pragma unroll
      for (int j = 0; j < kSqLoads; ++j) {
        const int i = t + j * kThreads;
        if (i < rows * slices) {
          const int rr = i / slices;
          cp_async4(s_sq + rr * kSlices + (i - rr * slices), part_sq + size_t(i - rr * slices) * m + row0 + rr);
        }
      }
      cp_async_wait_all();
    } else {
    // every partial of the block in flight at once: registers first, then shared memory (sk_unit's loads)
    float4 v[kLoads];
    float4 w[kResidual ? kLoads : 1];
#pragma unroll
    for (int j = 0; j < kLoads; ++j) {
      const int i = t + j * kThreads;
      if (i < total) {
        const int rr = i / n4;
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
        const int rr = i / slices;
        sq[j] = __ldcg(part_sq + size_t(i - rr * slices) * m + row0 + rr);
      }
    }
#pragma unroll
    for (int j = 0; j < kLoads; ++j) {
      const int i = t + j * kThreads;
      if (i < total) {
        const int rr = i / n4;
        float4 x = v[j];
        if constexpr (kResidual) {       // stock: v = hi + lo (add.f32), before the running sum
          x.x = add_f(x.x, w[j].x);
          x.y = add_f(x.y, w[j].y);
          x.z = add_f(x.z, w[j].z);
          x.w = add_f(x.w, w[j].w);
        }
        s_mix[rr * kRow4 + (i - rr * n4)] = x;
      }
    }
#pragma unroll
    for (int j = 0; j < kSqLoads; ++j) {
      const int i = t + j * kThreads;
      if (i < rows * slices) {
        const int rr = i / slices;
        s_sq[rr * kSlices + (i - rr * slices)] = sq[j];
      }
    }
    }
    __syncthreads();
    // stock's per-output order: acc = 0, acc += (combined) partial of slice 0, 1, ...
    if (t < rows * (kMix + 1)) {
      const int rr = t / (kMix + 1);
      const int c = t - rr * (kMix + 1);
      float acc = 0.f;
      if (c < kMix) {
        const float* col = reinterpret_cast<const float*>(s_mix + rr * kRow4) + c;
        if constexpr (kLow) {          // stock: v = hi + lo (add.f32) a slice, then acc += v
          const float* lowc = reinterpret_cast<const float*>(s_low + rr * kRow4) + c;
#pragma unroll 8
          for (int s = 0; s < slices; ++s) acc = add_f(acc, add_f(col[s * kMix], lowc[s * kMix]));
        } else {
#pragma unroll 8
          for (int s = 0; s < slices; ++s) acc = add_f(acc, col[s * kMix]);
        }
      } else {
        const float* sqr = s_sq + rr * kSlices;
#pragma unroll 8
        for (int s = 0; s < slices; ++s) acc = add_f(acc, sqr[s]);
      }
      s_sum[rr * (kMix + 1) + c] = acc;
    }
    __syncthreads();
    if (wr < rows) {
      // sk_unit.cu's epilogue, instruction for instruction
      const int row = row0 + wr;
      const float* sum_r = s_sum + wr * (kMix + 1);
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
    __syncthreads();   // the next row block reuses the shared buffers
  }
}

}  // namespace skw

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

namespace {

bool g_init = false;

template <int kR, int kThreads, int kSlices, bool kResidual, bool kAsync>
void set_smem() {
  const cudaError_t rc = cudaFuncSetAttribute(skw::reduce_sinkhorn_g<kR, kThreads, kSlices, kResidual, kAsync>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              skw::smem_bytes<kR, kSlices, kResidual && kAsync>());
  if (rc != cudaSuccess) throw std::runtime_error(std::string("skw: smem attribute: ") + cudaGetErrorString(rc));
}

}  // namespace

// once per process, outside any graph capture: opt the instantiations into their dynamic shared memory
void init() {
  if (g_init) return;
  set_smem<8, 512, skw::kMaxSlices, false, false>();
  set_smem<16, 1024, skw::kMaxSlices, false, false>();
  set_smem<16, 512, 16, true, false>();
  set_smem<32, 1024, 16, true, false>();
  set_smem<8, 512, skw::kMaxSlices, false, true>();
  set_smem<16, 1024, skw::kMaxSlices, false, true>();
  set_smem<16, 512, 16, true, true>();
  set_smem<32, 1024, 16, true, true>();
  g_init = true;
}

// pointers as sk_unit.run: part_mix [slices, m, 24] fp32 (+ residual of the same shape, or 0), part_sq [slices, m],
// scale [3], base [24], pre / post [m, 4], comb [m, 4, 4]; rows_per_block 8 / 16 (small path) or 16 / 32 (residual
// path); grid_cap >= 1; async 1: cp.async loads into shared memory (no register staging), 0: sk_unit's register loads.
// Returns the grid launched.
int64_t run(int64_t part_mix, int64_t residual, int64_t part_sq, int64_t scale, int64_t base, int64_t pre, int64_t post,
            int64_t comb, int64_t m, double inv_k, double rms_eps, int64_t slices, int64_t iters, double eps,
            int64_t rows_per_block, int64_t grid_cap, int64_t async, int64_t stream) {
  if (!g_init) throw std::runtime_error("skw: init() first");
  if (m < 1 || m > (int64_t(1) << 20)) throw std::runtime_error("skw: rows 1-2^20");
  if (slices < 1 || slices > skw::kMaxSlices || iters < 1) throw std::runtime_error("skw: 1-80 slices, iters >= 1");
  if (part_mix % 16 || residual % 16) throw std::runtime_error("skw: partials 16-byte aligned");
  if (part_sq % 4) throw std::runtime_error("skw: part_sq 4-byte aligned");
  if (grid_cap < 1) throw std::runtime_error("skw: grid_cap >= 1");
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  int64_t launched = 0;
  const auto launch = [&](auto kernel, int per, int threads, int smem) {
    const int64_t blocks = (m + per - 1) / per;
    const int grid = static_cast<int>(blocks < grid_cap ? blocks : grid_cap);
    kernel<<<grid, threads, smem, st>>>(
        reinterpret_cast<const float*>(part_mix), reinterpret_cast<const float*>(residual),
        reinterpret_cast<const float*>(part_sq), reinterpret_cast<const float*>(scale),
        reinterpret_cast<const float*>(base), reinterpret_cast<float*>(pre), reinterpret_cast<float*>(post),
        reinterpret_cast<float*>(comb), static_cast<int>(m), static_cast<float>(inv_k), static_cast<float>(rms_eps),
        static_cast<int>(slices), static_cast<int>(iters), static_cast<float>(eps));
    launched = grid;
  };
  using skw::kMaxSlices;
  using skw::reduce_sinkhorn_g;
  using skw::smem_bytes;
  if (residual != 0) {
    if (slices > 16) throw std::runtime_error("skw: residual path takes <= 16 slices");
    if (rows_per_block == 16 && async) launch(reduce_sinkhorn_g<16, 512, 16, true, true>, 16, 512, smem_bytes<16, 16, true>());
    else if (rows_per_block == 16) launch(reduce_sinkhorn_g<16, 512, 16, true, false>, 16, 512, smem_bytes<16, 16>());
    else if (rows_per_block == 32 && async)
      launch(reduce_sinkhorn_g<32, 1024, 16, true, true>, 32, 1024, smem_bytes<32, 16, true>());
    else if (rows_per_block == 32) launch(reduce_sinkhorn_g<32, 1024, 16, true, false>, 32, 1024, smem_bytes<32, 16>());
    else throw std::runtime_error("skw: residual path takes 16 or 32 rows a block");
  } else {
    if (rows_per_block == 8 && async)
      launch(reduce_sinkhorn_g<8, 512, kMaxSlices, false, true>, 8, 512, smem_bytes<8, kMaxSlices>());
    else if (rows_per_block == 8)
      launch(reduce_sinkhorn_g<8, 512, kMaxSlices, false, false>, 8, 512, smem_bytes<8, kMaxSlices>());
    else if (rows_per_block == 16 && async)
      launch(reduce_sinkhorn_g<16, 1024, kMaxSlices, false, true>, 16, 1024, smem_bytes<16, kMaxSlices>());
    else if (rows_per_block == 16)
      launch(reduce_sinkhorn_g<16, 1024, kMaxSlices, false, false>, 16, 1024, smem_bytes<16, kMaxSlices>());
    else throw std::runtime_error("skw: small path takes 8 or 16 rows a block");
  }
  const cudaError_t rc = cudaGetLastError();
  if (rc != cudaSuccess) throw std::runtime_error(std::string("skw: launch failed: ") + cudaGetErrorString(rc));
  return launched;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "opt the kernels into their dynamic shared memory (once, outside capture)");
  m.def("run", &run, "mHC stats reduce + sinkhorn on a capped grid, sk_unit's bits (8 / 16 or 16 / 32 rows a block, "
        "register or cp.async loads)");
}
