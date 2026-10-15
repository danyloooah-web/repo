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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather", &gather, "Engram host-table row gather (stock _engram_gather_kernel's bits, warp per row)");
}
