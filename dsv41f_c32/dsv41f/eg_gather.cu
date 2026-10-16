// lev_EG: Engram row gather from the per-rank host table (fp8 e4m3 rows of DIM bytes, e8m0 scales a BLK-element block:
// 32 on the arena checkpoint, 128 also built), the bits of
// stock's Triton _engram_gather_kernel (sglang/kernels/ops/embeddings/engram_gather.py): out = bf16(fp32(fp8) *
// 2**(e - 127)) rounded once (RN), 2**-127 for e == 0, zero rows for ids outside [row_lo, row_hi).
// One warp per lookup: lane l moves bytes [8l, 8l + 8) of the 256-byte row (one coalesced 256-byte host read where
// stock's 128-thread program issues 64-byte warp reads plus per-thread scale bytes) and writes 16 bytes of output.
// UNROLL lookups per warp are in flight before any use; the grid is bounded (grid-stride), so the kernel can run beside
// other work with a fixed SM budget.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace eg {

constexpr int DIM = 256;

__device__ __forceinline__ float e8m0_scale(uint32_t e) {
  return e == 0 ? __uint_as_float(0x00400000u) : __uint_as_float(e << 23);
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ float fp8f(uint32_t byte) {
  __nv_fp8_e4m3 v;
  v.__x = static_cast<__nv_fp8_storage_t>(byte);
  return static_cast<float>(v);
}

template <typename IdT, int UNROLL, int BLK>
__global__ void __launch_bounds__(256) gather_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                                                     const IdT* __restrict__ ids, int64_t id_row_stride,
                                                     int64_t id_cols, __nv_bfloat16* __restrict__ out, int64_t n,
                                                     int64_t row_lo, int64_t row_hi) {
  const int lane = threadIdx.x & 31;
  const int64_t warp = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int64_t nwarps = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 5;
  for (int64_t base = warp * UNROLL; base < n; base += nwarps * UNROLL) {
    uint2 v[UNROLL];
    uint32_t e[UNROLL];
    bool own[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int64_t r = base + u;
      own[u] = false;
      v[u] = make_uint2(0u, 0u);
      e[u] = 0u;
      if (r < n) {
        const int64_t idx = static_cast<int64_t>(ids[(r / id_cols) * id_row_stride + (r % id_cols)]);
        own[u] = idx >= row_lo && idx < row_hi;
        if (own[u]) {
          const int64_t local = idx - row_lo;
          v[u] = __ldg(reinterpret_cast<const uint2*>(w + local * DIM) + lane);
          e[u] = __ldg(s + local * (DIM / BLK) + (lane * 8) / BLK);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int64_t r = base + u;
      if (r >= n) break;
      uint4 o = make_uint4(0u, 0u, 0u, 0u);
      if (own[u]) {
        const float sc = e8m0_scale(e[u]);
        const uint32_t a = v[u].x, b = v[u].y;
        o.x = pack2(__fmul_rn(fp8f(a & 0xff), sc), __fmul_rn(fp8f((a >> 8) & 0xff), sc));
        o.y = pack2(__fmul_rn(fp8f((a >> 16) & 0xff), sc), __fmul_rn(fp8f(a >> 24), sc));
        o.z = pack2(__fmul_rn(fp8f(b & 0xff), sc), __fmul_rn(fp8f((b >> 8) & 0xff), sc));
        o.w = pack2(__fmul_rn(fp8f((b >> 16) & 0xff), sc), __fmul_rn(fp8f(b >> 24), sc));
      }
      reinterpret_cast<uint4*>(out + r * DIM)[lane] = o;
    }
  }
}

}  // namespace eg

// ids: int32 / int64 lookups addressed as [rows, id_cols] with row stride id_row_stride (elements), so a column slice
// of the [T, layers, cols] hash ids needs no copy; out: [n, 256] bf16 contiguous, n = rows * id_cols.
extern "C" int eg_gather_launch(uint64_t w, uint64_t s, uint64_t ids, int ids_is64, int64_t id_row_stride,
                                int64_t id_cols, uint64_t out, int64_t n, int64_t row_lo, int64_t row_hi, int grid,
                                int unroll, int blk, uint64_t stream) {
  if (n <= 0) return 0;
  if (blk != 32 && blk != 128) return -1;
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  const uint8_t* wp = reinterpret_cast<const uint8_t*>(w);
  const uint8_t* sp = reinterpret_cast<const uint8_t*>(s);
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out);
#define EG_LAUNCH(T, U)                                                                                          \
  (blk == 32 ? (eg::gather_kernel<T, U, 32><<<grid, 256, 0, st>>>(wp, sp, reinterpret_cast<const T*>(ids),           \
                                                                  id_row_stride, id_cols, op, n, row_lo, row_hi),   \
                0)                                                                                                   \
             : (eg::gather_kernel<T, U, 128><<<grid, 256, 0, st>>>(wp, sp, reinterpret_cast<const T*>(ids),          \
                                                                   id_row_stride, id_cols, op, n, row_lo, row_hi),  \
                0))
  if (ids_is64) {
    if (unroll >= 4) EG_LAUNCH(int64_t, 4);
    else if (unroll == 2) EG_LAUNCH(int64_t, 2);
    else EG_LAUNCH(int64_t, 1);
  } else {
    if (unroll >= 4) EG_LAUNCH(int32_t, 4);
    else if (unroll == 2) EG_LAUNCH(int32_t, 2);
    else EG_LAUNCH(int32_t, 1);
  }
#undef EG_LAUNCH
  return static_cast<int>(cudaGetLastError());
}

// lever AUX (gen_aux.py, aux.py): the DSPARK draft's aux mean out[r, c] = mean over the HC = 4 mHC streams of
// x[r, k, c] (bf16 [rows, 4, cols], unit column stride) with torch's reduce_kernel<MeanOps<BFloat16, float>> arithmetic
// for a 4-element reduction over a strided dim (one thread an output, vt0 = 4 accumulators from ident 0, combined in
// order, project = acc * factor with factor = 0.25 exactly, c10::BFloat16(float) = cvt.rn.bf16.f32):
//   out = bf16_rn(((((0 + a0) + (0 + a1)) + (0 + a2)) + (0 + a3)) * 0.25f)
// as IEEE fp32 PTX without FTZ (the bundle builds with --use_fast_math: -ftz would flush the subnormals bf16 shares with
// fp32; f32x2 lanes: each lane the scalar op's bits) and cvt.rn.bf16x2.f32 per output pair. Grid-stride over
// 16-byte chunks (8 outputs a thread a step, UNROLL steps' loads in flight, 512 threads), a small fixed grid
// (aux.py AUX_LAUNCH) so it holds few SMs beside the main chain.
// Plain coherent loads (no __restrict__ / __ldg: the input is the previous kernel's output on another stream).
namespace auxm {

constexpr int THREADS = 512;

// f32x2 IEEE ops (two lanes, each the scalar op's bits: add.rn.f32 / mul.rn.f32, no .ftz)
__device__ __forceinline__ uint64_t add2(uint64_t a, uint64_t b) {
  uint64_t d;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}

__device__ __forceinline__ uint64_t mul2(uint64_t a, uint64_t b) {
  uint64_t d;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}

// the two bf16 of a 32-bit word as an f32x2 {low element, high element} (exact widening)
__device__ __forceinline__ uint64_t widen(uint32_t w) {
  return (static_cast<uint64_t>(w & 0xffff0000u) << 32) | static_cast<uint64_t>(w << 16);
}

// torch's per-output order on both elements: ((((0 + a0) + (0 + a1)) + (0 + a2)) + (0 + a3)) * 0.25f, bf16 RN
__device__ __forceinline__ uint32_t pair(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
  const uint64_t z = 0ull;                                  // {+0.0f, +0.0f}: torch's ident
  const uint64_t q = 0x3e8000003e800000ull;                 // {0.25f, 0.25f}: MeanOps' factor
  const uint64_t v0 = add2(z, widen(w0)), v1 = add2(z, widen(w1)), v2 = add2(z, widen(w2)), v3 = add2(z, widen(w3));
  const uint64_t r = mul2(add2(add2(add2(v0, v1), v2), v3), q);
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(__uint_as_float(static_cast<uint32_t>(r >> 32))),
      "f"(__uint_as_float(static_cast<uint32_t>(r))));
  return d;
}

// x: chunk pointer (16 bytes = 8 bf16); row_stride16 / hc_stride16 in chunks; CPR (0: cpr) chunks a row;
// n = rows * cpr < 2^31; out [rows, cols] contiguous = chunk i. Grid-stride, UNROLL chunks' 4 loads in flight a thread.
template <int UNROLL, int CPR>
__global__ void __launch_bounds__(THREADS, 1) mean_kernel(const uint4* x, int row_stride16, int hc_stride16,
                                                          uint4* out, int cpr_rt, int n) {
  const unsigned cpr = CPR ? static_cast<unsigned>(CPR) : static_cast<unsigned>(cpr_rt);
  const int nt = gridDim.x * THREADS;
  for (int base = blockIdx.x * THREADS + threadIdx.x; base < n; base += nt * UNROLL) {
    uint4 v[UNROLL][4];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int i = base + u * nt;
      if (i < n) {
        const unsigned r = static_cast<unsigned>(i) / cpr;
        const uint4* p = x + static_cast<int64_t>(r) * row_stride16 + (static_cast<unsigned>(i) - r * cpr);
#pragma unroll
        for (int k = 0; k < 4; ++k) v[u][k] = p[static_cast<int64_t>(k) * hc_stride16];
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int i = base + u * nt;
      if (i < n) {
        uint4 o;
        o.x = pair(v[u][0].x, v[u][1].x, v[u][2].x, v[u][3].x);
        o.y = pair(v[u][0].y, v[u][1].y, v[u][2].y, v[u][3].y);
        o.z = pair(v[u][0].z, v[u][1].z, v[u][2].z, v[u][3].z);
        o.w = pair(v[u][0].w, v[u][1].w, v[u][2].w, v[u][3].w);
        out[i] = o;
      }
    }
  }
}

}  // namespace auxm

// x [rows, 4, cols] bf16 (element strides row_stride / hc_stride, unit column stride), out [rows, cols] contiguous.
extern "C" int aux_mean_launch(uint64_t x, int64_t row_stride, int64_t hc_stride, uint64_t out, int64_t rows,
                               int64_t cols, int grid, uint64_t stream) {
  if (rows <= 0) return 0;
  const int64_t cpr = cols / 8, n = rows * cpr;
  if (n >= (int64_t(1) << 31) || row_stride / 8 >= (int64_t(1) << 31) || hc_stride / 8 >= (int64_t(1) << 28)) return -2;
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  const uint4* xp = reinterpret_cast<const uint4*>(x);
  uint4* op = reinterpret_cast<uint4*>(out);
  if (cpr == 640)
    auxm::mean_kernel<4, 640><<<grid, auxm::THREADS, 0, st>>>(xp, static_cast<int>(row_stride / 8),
                                                               static_cast<int>(hc_stride / 8), op, 640,
                                                               static_cast<int>(n));
  else
    auxm::mean_kernel<4, 0><<<grid, auxm::THREADS, 0, st>>>(xp, static_cast<int>(row_stride / 8),
                                                             static_cast<int>(hc_stride / 8), op,
                                                             static_cast<int>(cpr), static_cast<int>(n));
  return static_cast<int>(cudaGetLastError());
}

// ---- Python binding (no torch headers: torch 2.14's C++ API needs C++20, cacheon's build_cuda_ext compiles C++17;
// eg.py passes the device pointers, shapes and the current stream and checks dtypes / contiguity)
#include <pybind11/pybind11.h>

#include <stdexcept>

// out[n, 256] (bf16, contiguous) = rows ids of the host table at (w, s) owned by [lo, hi); ids [rows, id_cols] int32 /
// int64 with row stride id_row_stride (elements) and unit column stride, n = rows * id_cols.
void gather(int64_t w, int64_t s, int64_t ids, bool ids64, int64_t id_row_stride, int64_t id_cols, int64_t out,
            int64_t n, int64_t lo, int64_t hi, int64_t grid, int64_t unroll, int64_t blk, int64_t stream) {
  if (grid <= 0 || grid > 65535) throw std::runtime_error("eg_gather: grid");
  if (out % 16 != 0) throw std::runtime_error("eg_gather: out must be 16-byte aligned");
  const int rc = eg_gather_launch(static_cast<uint64_t>(w), static_cast<uint64_t>(s), static_cast<uint64_t>(ids),
                                  ids64 ? 1 : 0, id_row_stride, id_cols, static_cast<uint64_t>(out), n, lo, hi,
                                  static_cast<int>(grid), static_cast<int>(unroll), static_cast<int>(blk),
                                  static_cast<uint64_t>(stream));
  if (rc != 0) throw std::runtime_error("eg_gather: kernel launch failed (" + std::to_string(rc) + ")");
}

// lever AUX: out[rows, cols] (bf16 contiguous) = stock's x.mean(dim=1) bytes of x [rows, 4, cols] bf16 at x (element
// strides row_stride, hc_stride; unit column stride; 16-byte aligned, strides and cols multiples of 8).
void aux_mean(int64_t x, int64_t row_stride, int64_t hc_stride, int64_t out, int64_t rows, int64_t cols, int64_t grid,
              int64_t stream) {
  if (grid <= 0 || grid > 65535) throw std::runtime_error("aux_mean: grid");
  if (x % 16 != 0 || out % 16 != 0) throw std::runtime_error("aux_mean: x / out must be 16-byte aligned");
  if (cols <= 0 || cols % 8 != 0 || row_stride % 8 != 0 || hc_stride % 8 != 0 || rows < 0)
    throw std::runtime_error("aux_mean: cols / strides must be multiples of 8");
  const int rc = aux_mean_launch(static_cast<uint64_t>(x), row_stride, hc_stride, static_cast<uint64_t>(out), rows,
                                 cols, static_cast<int>(grid), static_cast<uint64_t>(stream));
  if (rc != 0) throw std::runtime_error("aux_mean: kernel launch failed (" + std::to_string(rc) + ")");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aux_mean", &aux_mean, "lever AUX: DSPARK aux mean over the 4 mHC streams, stock's bytes, small grid");
  m.def("gather", &gather, "Engram host-table row gather (stock _engram_gather_kernel's bits, warp per row)");
}
