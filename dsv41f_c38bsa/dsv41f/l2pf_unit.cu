// lev_L2PF: L2 prefetch hints for a decode / verify layer's WO-A + wo_b weights, issued during its attention.
// Window B (l2_fill) moved those bytes HBM -> shared memory on a few dozen resident blocks, which halved WO-A / wo_b
// time in the fork's k=1 profile but held SMs the TRT-LLM sparse decode attention wanted (B300: one kernel's CTAs per
// SM at a time). Here a short-lived grid issues prefetch instructions only -- no shared memory, no data returned to the
// SM, nothing written -- and exits, leaving the memory system to fill L2 while the attention runs.
//   mode 0: per-thread `prefetch.global.L2::evict_last` over 128-byte lines (grid-stride)
//   mode 1: `cp.async.bulk.prefetch.L2.global` of `chunk`-byte pieces (one thread a piece, grid-stride)
// Exactness: hints only; every consumer reads the same bytes. Scheduling only.
#include <cuda_runtime.h>
#include <stdint.h>

namespace l2pf {

constexpr int kMaxRegions = 16;

struct Regions {
  unsigned long long ptr[kMaxRegions];
  unsigned long long bytes[kMaxRegions];
  int n;
};

__global__ void __launch_bounds__(256) line_kernel(const Regions r) {
  const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = uint64_t(gridDim.x) * blockDim.x * 128u;
  for (int i = 0; i < r.n; ++i) {
    const char* base = reinterpret_cast<const char*>(r.ptr[i]);
    for (uint64_t off = tid * 128u; off < r.bytes[i]; off += step) {
      asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(base + off));
    }
  }
}

__global__ void __launch_bounds__(256) bulk_kernel(const Regions r, uint32_t chunk) {
  const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = uint64_t(gridDim.x) * blockDim.x * chunk;
  for (int i = 0; i < r.n; ++i) {
    const char* base = reinterpret_cast<const char*>(r.ptr[i]);
    for (uint64_t off = tid * chunk; off < r.bytes[i]; off += step) {
      const uint64_t left = r.bytes[i] - off;
      const uint32_t len = static_cast<uint32_t>(left < chunk ? left : chunk) & ~15u;
      if (len) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + off), "r"(len) : "memory");
    }
  }
}

}  // namespace l2pf

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

// regions (1-16) = device pointers (16-byte aligned) + byte sizes (multiples of 16); returns the bytes covered
int64_t prefetch(const std::vector<int64_t>& ptrs, const std::vector<int64_t>& sizes, int64_t grid, int64_t block,
                 int64_t mode, int64_t chunk, int64_t stream) {
  if (ptrs.size() != sizes.size() || ptrs.empty() || ptrs.size() > l2pf::kMaxRegions)
    throw std::runtime_error("l2pf: 1-16 regions");
  if (grid < 1 || grid > 1024 || block < 32 || block > 256 || block % 32)
    throw std::runtime_error("l2pf: grid 1-1024, block 32-256 (warp multiple)");
  l2pf::Regions r{};
  int64_t total = 0;
  for (size_t i = 0; i < ptrs.size(); ++i) {
    if (ptrs[i] % 16 || sizes[i] % 16 || sizes[i] <= 0) throw std::runtime_error("l2pf: 16-byte aligned regions");
    r.ptr[i] = static_cast<unsigned long long>(ptrs[i]);
    r.bytes[i] = static_cast<unsigned long long>(sizes[i]);
    total += sizes[i];
  }
  r.n = static_cast<int>(ptrs.size());
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  if (mode == 0) {
    l2pf::line_kernel<<<static_cast<int>(grid), static_cast<int>(block), 0, st>>>(r);
  } else {
    if (chunk < 16 || chunk % 16) throw std::runtime_error("l2pf: chunk multiple of 16");
    l2pf::bulk_kernel<<<static_cast<int>(grid), static_cast<int>(block), 0, st>>>(r, static_cast<uint32_t>(chunk));
  }
  const cudaError_t rc = cudaGetLastError();
  if (rc != cudaSuccess) throw std::runtime_error(std::string("l2pf: launch failed: ") + cudaGetErrorString(rc));
  return total;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefetch", &prefetch, "L2 prefetch hints over device regions (no data movement into SMs)");
}
