"""Lever EPA (levers.EPA; generated into the bundle by gen_epa.py, which inlines _HEADER): the attention all-reduce's
push moves into wo_b's GEMM epilogue.

Stock / c36: wo_b (lever PWN: DbwDenseGemmKernel) writes its bf16 rows; then the attention post (lever FIN's copy of
stock's finalize -> 1shot push all-reduce -> HC=4 mHC post + RMSNorm kernel) waits for the GEMM's completion, loads
the rows, applies the identity finalize (fmaf(x, 1, 0), bf16 cast, Lamport +0 marker remap), pushes them to its own
and the peer's plane slot, polls both slots, reduces in rank order and runs the mHC / norm epilogue.

Ours: wo_b runs EpaGemmKernel (epa_gemm.py = pws_gemm.py + a second TMA store of every C tile to the peer rank's copy
of the slot and a 0x0000 -> 0x8000 select on the bf16 tile before its smem stage; the MMA, accumulator and bf16
conversion are PWN's unchanged), so the NVLink flight starts inside the GEMM. The post (epa_attn_post_kernel, FIN's
header + kits/epa_ar.cuh) only polls this rank's two slots, rebuilds the exact bytes stock's push would have carried
(stock's finalize arithmetic on the received rows + clear_pos_zero: fmaf(+-0, 1, +0) = +0 makes the producer's zero
remap invisible), and runs FIN's reduce + mHC + norm epilogue verbatim, then re-arms its slots.

Slots: one symmetric bf16 buffer per rank (torch symmetric memory, rendezvous on the TP group), 2 regions (layer
parity) x 2 source ranks x EPA_MAX_ROWS rows. A region's next use comes after at least one later cross-rank round
(the layer's MoE all-reduce) completed on both ranks, so the peer has re-armed it (stock's two-phase plane argument).

Binding (no served module touched): lever L2PF's lifted attention forward (fork_attn.py) returns
epa.project_wo_b(self, o, defer) instead of self._project_wo_b(o, defer) (stock's method, pinned via lever AG's pin);
the replicas' wo_b linears (lever DT's private Fp8LinearMethod copies, already wrapped by PWN) get EPA's wrapper on top;
the replicas' lifted forwards (hq_layer / fg_layer) call epa.run_attn_post, which hands every call without a pending
EPA push to fin.run_attn_post unchanged.

Exactness, fail closed: the first EAGER call per row count runs the wrapped linear (PWN / DT / stock) + FIN's post into
the real outputs and EPA's GEMM + post into scratch on the same inputs; it compares the GEMM's own-slot bytes with the
wrapped linear's output (0x0000 as 0x8000) and mhc_out / normalized byte for byte, rank-agreed (MIN). Any difference or
exception turns that row count off for the process (a partial push is scrubbed: both ranks sync, zero their buffer,
sync). CUDA-graph capture takes EPA only on verified row counts.
"""
import inspect
import sys

import torch

from dsv41f import fin, levers

HIDDEN = 5120
K_IN = 4096
WORLD = 2
CLUSTER = 5
BF16 = torch.bfloat16
_HEADER = r"""// Lever EPA (dsv41f bundle): lever FIN's header (namespace dsv41f_epa) + the EPA attention post appended at the end.
// Lever FIN (dsv41f bundle): a scheduling-only copy of SGLang's fused MoE-finalize -> 1shot push all-reduce -> HC=4
// mHC post kernel. Derived from SGLang python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh
// (md5 36d9ef4de405a0d7e5070094913d2fe7, fork 5e72ce5), Copyright the SGLang authors, Apache License 2.0
// (http://www.apache.org/licenses/LICENSE-2.0); modified by the dsv41f bundle authors as listed below.
//
// Every floating-point operation and its order are the original's. The edits only move loads, drop one store and
// generate the identity routing in the kernel:
//   (a) everything the mHC / norm epilogue reads apart from the reduced vector (the 4 residual vectors, post, comb,
//       pre and the norm weight) is loaded right after the push, before the poll loop (all of it is complete there:
//       the kernel passed the PDL wait before the push, and post / comb / pre arrive through stream joins);
//       mhc_post_vec is split into mhc_post_load + mhc_post_compute with the same fmaf / __fmul_rn / cast chain;
//   (d) kMhc paths do not store the reduced row into `out` (every caller discards it; `out` is still verified);
//   (e) kIdentity: the attention post's identity routing (stock: idx = arange, fp32 weights = 1, top_k 1) is
//       generated in the kernel: rows[0] = token, w[0] = params.identity_weight (a runtime fp32 1.0f, so the
//       finalize stays fmaf(x, w, 0.0f) exactly as compiled for stock);
//   (f) kEarlyPdl (default off): the PDL trigger right after the push instead of after the epilogue stores;
//   kPrePoll selects which epilogue loads move before the poll (bit 0 residual, bit 1 post / comb / pre / norm
//   weight; 3 = all, 0 = the original's placement after the poll).
// The push-plane protocol (slots, +0 / -0 markers, bumper cluster, per-row epoch flip) is byte-for-byte the
// original's, so this kernel and the original interleave on one plane and either rank may run either one.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace sglang {
namespace dsv41f_epa {

using device::distributed::PushWorkSpace;
using host::distributed::CommunicatorRef;

using Lamport = device::distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;
using StageVec = device::AlignedVector<bf16x2_t, 4>;

SGL_DEVICE void barrier_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

SGL_DEVICE void barrier_cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

template <uint32_t kWorldSize, typename WeightT>
struct FinParams {
  bf16_t* out;                         // [num_tokens, kHiddenDim]; not written by the kMhc paths
  const bf16_t* gemm2;                 // [P, kHiddenDim]
  const int32_t* idx;                  // [num_tokens * kTopK] (nullptr with kIdentity)
  const WeightT* weights;              // [num_tokens, kTopK] (nullptr with kIdentity)
  const bf16_t* shared;
  const float* shared_gate = nullptr;
  const bf16_t* norm_weight;
  float norm_eps;
  bool prefetch_metadata;
  uint32_t rank;
  uint32_t num_tokens;
  uint32_t num_push_counters;
  PushWorkSpace<kWorldSize> ws;
  bf16_t* mhc_out = nullptr;
  const bf16_t* residual = nullptr;
  const float* post = nullptr;
  const float* comb = nullptr;
  const float* pre = nullptr;
  bf16_t* normalized = nullptr;
  fp8_e4m3_t* quantized = nullptr;
  uint8_t* scales = nullptr;
  float identity_weight = 1.0f;        // kIdentity: the unit routing weight, a runtime value (1.0f)
};

template <uint32_t kHiddenDim, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE void mhc_quant_vec(
    const FinParams<kWorldSize, WeightT>& params, const StageVec& value, uint32_t token, uint32_t hvec) {
  using namespace device;
  fp32x2_t v[4];
  float amax = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    v[j] = cast<fp32x2_t>(value[j]);
    amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));
  const float normalized = amax * (1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(normalized);
  const uint32_t exponent = (bits >> 23) & 255;
  const uint32_t mantissa = bits & 0x7fffff;
  const bool bump = mantissa != 0 && !(exponent == 0 && mantissa <= 0x400000);
  const uint32_t sf = normalized <= 0 ? 0 : min(exponent + uint32_t(bump), 254u);
  const float inv_scale = __uint_as_float(sf == 0 ? 0 : (254 - sf) << 23);
  AlignedVector<fp8x2_e4m3_t, 4> q;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    q[j] = cast<fp8x2_e4m3_t>(
        fp32x2_t{fminf(fmaxf(v[j].x * inv_scale, -448.0f), 448.0f), fminf(fmaxf(v[j].y * inv_scale, -448.0f), 448.0f)});
  }
  q.store(params.quantized + static_cast<int64_t>(token) * kHiddenDim, hvec);
  if (hvec % 4 == 0) {
    const uint32_t g = hvec / 4;
    const uint32_t off = (g / 4) * 512 + ((token % 32) * 4 + (token / 32) % 4) * 4 + g % 4;
    params.scales[off] = sf;
  }
}

/// The HC=4 post's inputs other than the reduced vector, held in registers across the poll.
struct MhcPostInputs {
  StageVec residual[4];
  float post[4];
  float comb[4][4];  // comb[c][r] = params.comb[token * 16 + r * 4 + c]
  float pre[4];
};

/// Edit (a): the loads of the original mhc_post_vec (residual vectors, then the post / comb / pre scalars).
template <uint32_t kHiddenDim, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE void mhc_post_load_residual(
    const FinParams<kWorldSize, WeightT>& params, uint32_t token, uint32_t hvec, MhcPostInputs& in) {
#pragma unroll
  for (uint32_t c = 0; c < 4; ++c) {
    in.residual[c].load(params.residual + (static_cast<int64_t>(token) * 4 + c) * kHiddenDim, hvec);
  }
}

template <bool kCollapse, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE void mhc_post_load_coeffs(const FinParams<kWorldSize, WeightT>& params, uint32_t token, MhcPostInputs& in) {
#pragma unroll
  for (uint32_t c = 0; c < 4; ++c) {
    in.post[c] = params.post[token * 4 + c];
#pragma unroll
    for (uint32_t r = 0; r < 4; ++r)
      in.comb[c][r] = params.comb[token * 16 + r * 4 + c];
    if constexpr (kCollapse) in.pre[c] = params.pre[token * 4 + c];
  }
}

/// The original mhc_post_vec's arithmetic and stores, on the preloaded inputs: round comb[0]*residual[0], FMA
/// post*x, then the remaining three residual streams; the collapse FMAs pre over the rounded outputs.
template <uint32_t kHiddenDim, bool kCollapse, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE StageVec mhc_post_compute(
    const FinParams<kWorldSize, WeightT>& params,
    const MhcPostInputs& in,
    const StageVec& red,
    uint32_t token,
    uint32_t hvec) {
  using namespace device;
  fp32x2_t collapsed[4] = {};
#pragma unroll
  for (uint32_t c = 0; c < 4; ++c) {
    const float post = in.post[c];
    float comb[4];
#pragma unroll
    for (uint32_t r = 0; r < 4; ++r)
      comb[r] = in.comb[c][r];
    StageVec out;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const auto x = cast<fp32x2_t>(red[j]);
      const auto r0 = cast<fp32x2_t>(in.residual[0][j]);
      fp32x2_t acc{fmaf(post, x.x, __fmul_rn(comb[0], r0.x)), fmaf(post, x.y, __fmul_rn(comb[0], r0.y))};
#pragma unroll
      for (uint32_t r = 1; r < 4; ++r) {
        const auto v = cast<fp32x2_t>(in.residual[r][j]);
        acc.x = fmaf(comb[r], v.x, acc.x);
        acc.y = fmaf(comb[r], v.y, acc.y);
      }
      out[j] = cast<bf16x2_t>(acc);
      if constexpr (kCollapse) {
        const auto rounded = cast<fp32x2_t>(out[j]);
        const float pre = in.pre[c];
        collapsed[j].x = fmaf(rounded.x, pre, collapsed[j].x);
        collapsed[j].y = fmaf(rounded.y, pre, collapsed[j].y);
      }
    }
    out.store(params.mhc_out + (static_cast<int64_t>(token) * 4 + c) * kHiddenDim, hvec);
  }
  StageVec result;
  if constexpr (kCollapse) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j)
      result[j] = cast<bf16x2_t>(collapsed[j]);
  }
  return result;
}

template <uint32_t kHiddenDim, uint32_t kClusterSize>
struct FinRowTrait {
  static constexpr uint32_t kRowVecs = kHiddenDim / 8;
  static constexpr uint32_t kBlockSize = kRowVecs / kClusterSize;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static_assert(kHiddenDim % 8 == 0, "hidden must be a whole number of 16B vectors");
  static_assert(1 <= kClusterSize && kClusterSize <= 8, "portable cluster sizes only");
  static_assert(kRowVecs % kClusterSize == 0, "cluster size must divide the row's vector count");
  static_assert(kBlockSize % device::kWarpThreads == 0, "block must be whole warps");
  static_assert(kBlockSize <= 1024, "block too large: raise the cluster size");
};

template <
    uint32_t kHiddenDim,
    uint32_t kTopK,
    bool kHasShared,
    bool kUsePDL,
    uint32_t kWorldSize,
    typename WeightT,
    bool kGatedShared,
    bool kIdentity>
SGL_DEVICE StageVec finalize_vec(const FinParams<kWorldSize, WeightT>& params, uint32_t token, uint32_t hvec) {
  using namespace device;
  int32_t rows[kTopK];
  WeightT w[kTopK];
  if constexpr (kIdentity) {
    // edit (e): the original's idx = arange(T), weights = ones(T, 1) fp32, top_k 1
    static_assert(kTopK == 1 && std::is_same_v<WeightT, fp32_t>);
    rows[0] = static_cast<int32_t>(token);
    w[0] = params.identity_weight;
  } else {
    const auto* idx = params.idx + static_cast<int64_t>(token) * kTopK;
    const auto* weights = params.weights + static_cast<int64_t>(token) * kTopK;
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      rows[k] = idx[k];
      w[k] = weights[k];
    }
  }

  PDLWaitPrimary<kUsePDL>();

  StageVec shared_in;
  if constexpr (kHasShared) {
    shared_in.load(params.shared + static_cast<int64_t>(token) * kHiddenDim, hvec);
  }

  StageVec in[kTopK];
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    if (rows[k] >= 0) in[k].load(params.gemm2 + static_cast<int64_t>(rows[k]) * kHiddenDim, hvec);
  }

  fp32x2_t acc[4];
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    acc[j] = fp32x2_t{0.0f, 0.0f};
  }

#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    if (rows[k] < 0) continue;
#if SGL_ARCH_BLACKWELL_OR_GREATER
    if constexpr (std::is_same_v<WeightT, bf16_t>) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        acc[j].x = math::fma_f32_bf16(in[k][j].x, w[k], acc[j].x);
        acc[j].y = math::fma_f32_bf16(in[k][j].y, w[k], acc[j].y);
      }
    } else
#endif
    {
      const auto w_fp32 = cast<fp32_t>(w[k]);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const auto [x, y] = cast<fp32x2_t>(in[k][j]);
        acc[j].x = fmaf(x, w_fp32, acc[j].x);
        acc[j].y = fmaf(y, w_fp32, acc[j].y);
      }
    }
  }
  StageVec out;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    if constexpr (kHasShared) {
      const auto routed = cast<fp32x2_t>(cast<bf16x2_t>(acc[j]));
      const auto sh = cast<fp32x2_t>(shared_in[j]);
      if constexpr (kGatedShared) {
        const float gate = params.shared_gate[token];
        out[j] = cast<bf16x2_t>(fp32x2_t{fmaf(gate, sh.x, routed.x), fmaf(gate, sh.y, routed.y)});
      } else {
        out[j] = cast<bf16x2_t>(fp32x2_t{routed.x + sh.x, routed.y + sh.y});
      }
    } else {
      out[j] = cast<bf16x2_t>(acc[j]);
    }
  }
  return out;
}

// Grid dim3(num_tokens [+ 1], kClusterSize) with the cluster along y, as the original.
template <
    uint32_t kWorldSize,
    uint32_t kHiddenDim,
    uint32_t kTopK,
    uint32_t kClusterSize,
    bool kUsePDL,
    bool kHasShared,
    bool kNorm,
    typename WeightT,
    bool kMhc,
    bool kQuant,
    bool kCollapse,
    bool kGatedShared,
    bool kIdentity,
    bool kEarlyPdl,
    uint32_t kPrePoll>
__global__ __launch_bounds__(FinRowTrait<kHiddenDim, kClusterSize>::kBlockSize)
    __cluster_dims__(1, kClusterSize, 1) void fin_moe_finalize_all_reduce_kernel(
        const __grid_constant__ FinParams<kWorldSize, WeightT> params) {
  namespace cg = cooperative_groups;
  using namespace device;
  using T = FinRowTrait<kHiddenDim, kClusterSize>;
  constexpr uint32_t kRowVecs = T::kRowVecs;
  constexpr uint32_t kBlockSize = T::kBlockSize;
  constexpr uint32_t kNumWarps = T::kNumWarps;

  const auto tx = threadIdx.x;
  const auto row_idx = blockIdx.x;
  const auto cluster_rank = blockIdx.y;
  const auto hvec = cluster_rank * kBlockSize + tx;

  if (!params.prefetch_metadata) PDLWaitPrimary<kUsePDL>();

  if (row_idx == params.num_tokens) {
    PDLWaitPrimary<kUsePDL>();
    if constexpr (kQuant) {
      for (uint32_t off = hvec; off < (kHiddenDim / 32) * 128; off += kRowVecs) {
        const uint32_t swizzled_row = (off % 512) / 4;
        const uint32_t row = swizzled_row / 4 + (swizzled_row % 4) * 32;
        if (row >= params.num_tokens) params.scales[off] = 0;
      }
    }
    if (cluster_rank == 0) {
      const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};
      __syncthreads();
      epoch.unsafe_flip_range(row_idx, params.num_push_counters);
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};
  const auto r = params.rank;
  void* push_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    push_ptrs[i] = epoch.slot_ptr(/*dst=*/i, /*src=*/r);
  }

  const auto vid = row_idx * kRowVecs + hvec;
  {
    auto vec = finalize_vec<kHiddenDim, kTopK, kHasShared, kUsePDL, kWorldSize, WeightT, kGatedShared, kIdentity>(
        params, row_idx, hvec);
    Lamport::clear_pos_zero(vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_relaxed_16B(vec, push_ptrs[i], vid);
    }
  }

  // edit (a): the epilogue's other inputs, in flight while the poll below waits for the peers. The compiler barrier
  // keeps them behind the push stores (no memory clobber there), so they never compete with the finalize gathers.
  asm volatile("" ::: "memory");
  // kPrePoll bit 0: the residual vectors before the poll; bit 1: the post / comb / pre scalars and the norm weight
  constexpr bool kPostCollapse = kNorm || kCollapse;
  constexpr bool kPreResidual = (kPrePoll & 1) != 0;
  constexpr bool kPreCoeffs = (kPrePoll & 2) != 0;
  [[maybe_unused]] MhcPostInputs mhc_in;
  [[maybe_unused]] StageVec w;
  if constexpr (kMhc && kPreResidual) mhc_post_load_residual<kHiddenDim>(params, row_idx, hvec, mhc_in);
  if constexpr (kMhc && kPreCoeffs) mhc_post_load_coeffs<kPostCollapse>(params, row_idx, mhc_in);
  if constexpr (kNorm && kPreCoeffs) w.load(params.norm_weight, hvec);
  // edit (f)
  if constexpr (kEarlyPdl) PDLTriggerSecondary<kUsePDL>();

  if constexpr (!kNorm) barrier_cluster_arrive_relaxed();

  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = epoch.slot_ptr(/*dst=*/r, /*src=*/i);
  }
  StageVec vec[kWorldSize];
  do {
    bool has_zero = false;
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], vid);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      has_zero |= Lamport::has_pos_zero(vec[i].data());
    }
    if (!has_zero) break;
  } while (true);
  if constexpr (kMhc && !kPreResidual) mhc_post_load_residual<kHiddenDim>(params, row_idx, hvec, mhc_in);
  if constexpr (kMhc && !kPreCoeffs) mhc_post_load_coeffs<kPostCollapse>(params, row_idx, mhc_in);
  if constexpr (kNorm && !kPreCoeffs) w.load(params.norm_weight, hvec);

  if constexpr (!kNorm) {
    const auto red = reduce_vec(vec);
    // edit (d): the kMhc callers discard the reduced row
    if constexpr (!kMhc) ptx::st_global_16B(red, params.out, vid);
    if constexpr (kMhc) {
      if constexpr (kCollapse) {
        const auto combined = mhc_post_compute<kHiddenDim, true>(params, mhc_in, red, row_idx, hvec);
        ptx::st_global_16B(combined, params.normalized, vid);
      } else {
        mhc_post_compute<kHiddenDim, false>(params, mhc_in, red, row_idx, hvec);
      }
    }
    barrier_cluster_wait();
  } else {
    __shared__ float smem_sq[kClusterSize][kNumWarps];
    auto red = reduce_vec(vec);
    if constexpr (kMhc) {
      red = mhc_post_compute<kHiddenDim, true>(params, mhc_in, red, row_idx, hvec);
    }
    const auto cluster = cg::this_cluster();
    fp32x2_t acc[4];
    float sq = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      acc[j] = cast<fp32x2_t>(red[j]);
      sq = fmaf(acc[j].x, acc[j].x, sq);
      sq = fmaf(acc[j].y, acc[j].y, sq);
    }
    sq = warp::reduce_sum(sq);
    const auto lane = tx % kWarpThreads;
    const auto warp = tx / kWarpThreads;
    if (lane < kClusterSize) {
      *cluster.map_shared_rank(&smem_sq[cluster_rank][warp], lane) = sq;
    }
    cluster.sync();
    float total = 0.0f;
#pragma unroll
    for (uint32_t c = 0; c < kClusterSize; ++c) {
#pragma unroll
      for (uint32_t wp = 0; wp < kNumWarps; ++wp) {
        total += smem_sq[c][wp];
      }
    }
    const auto factor = math::rsqrt(total / static_cast<float>(kHiddenDim) + params.norm_eps);
    StageVec out;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const auto [wa, wb] = cast<fp32x2_t>(w[j]);
      out[j] = cast<bf16x2_t>(fp32x2_t{acc[j].x * factor * wa, acc[j].y * factor * wb});
    }
    ptx::st_global_16B(out, kMhc ? params.normalized : params.out, vid);
    if constexpr (kQuant) mhc_quant_vec<kHiddenDim>(params, out, row_idx, hvec);
  }
  if constexpr (!kEarlyPdl) PDLTriggerSecondary<kUsePDL>();

  StageVec zero_vec;
  Lamport::fill_pos_zero(zero_vec.data());
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    ptx::st_global_16B(zero_vec, poll_ptrs[i], vid);
  }

  if (cluster_rank == 0) epoch.flip();
}

// --- host --------------------------------------------------------------------

template <
    uint32_t kWorldSize,
    uint32_t kHiddenDim,
    uint32_t kTopK,
    uint32_t kClusterSize,
    bool kUsePDL,
    typename WeightT,
    bool kMhc,
    bool kQuant,
    bool kIdentity,
    bool kEarlyPdl,
    uint32_t kPrePoll = 3>
struct FinMhcAllReduce {
 private:
  static_assert(std::is_same_v<WeightT, bf16_t> || std::is_same_v<WeightT, fp32_t>);
  static_assert(kMhc && kHiddenDim == 5120);
  static_assert(!kIdentity || (kTopK == 1 && std::is_same_v<WeightT, fp32_t> && !kQuant));
  using TensorView = tvm::ffi::TensorView;
  using Params = FinParams<kWorldSize, WeightT>;
  using Trait = FinRowTrait<kHiddenDim, kClusterSize>;

  template <bool kHasShared, bool kNorm>
  static constexpr auto kernel = fin_moe_finalize_all_reduce_kernel<
      kWorldSize,
      kHiddenDim,
      kTopK,
      kClusterSize,
      kUsePDL,
      kHasShared,
      kNorm,
      WeightT,
      kMhc,
      kQuant,
      /*kCollapse=*/false,
      /*kGatedShared=*/false,
      kIdentity,
      kEarlyPdl,
      kPrePoll>;

 public:
  static void run_mhc_post(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb) {
    static_assert(!kIdentity);
    run_impl(ref, out, gemm2_out, permuted_idx, expert_weights, shared_output, std::nullopt, 0.0, mhc_out, residual,
             post, comb, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 1.0f);
  }

  static void run_mhc_post_combine_norm(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      TensorView pre,
      TensorView norm_weight,
      double eps,
      TensorView normalized) {
    static_assert(!kIdentity);
    run_impl(ref, out, gemm2_out, permuted_idx, expert_weights, shared_output, norm_weight, eps, mhc_out, residual,
             post, comb, pre, normalized, std::nullopt, std::nullopt, 1.0f);
  }

  static void run_mhc_post_combine_norm_quant(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      TensorView pre,
      TensorView norm_weight,
      double eps,
      TensorView normalized,
      TensorView quantized,
      TensorView scales) {
    static_assert(kQuant && !kIdentity);
    run_impl(ref, out, gemm2_out, permuted_idx, expert_weights, shared_output, norm_weight, eps, mhc_out, residual,
             post, comb, pre, normalized, quantized, scales, 1.0f);
  }

  /// The attention post: plain all-reduce of x with the mHC + norm epilogue, identity routing in the kernel.
  static void run_identity_norm(
      CommunicatorRef ref,
      TensorView out,
      TensorView x,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      TensorView pre,
      TensorView norm_weight,
      double eps,
      TensorView normalized,
      double identity_weight) {
    static_assert(kIdentity);
    run_impl(ref, out, x, std::nullopt, std::nullopt, std::nullopt, norm_weight, eps, mhc_out, residual, post, comb,
             pre, normalized, std::nullopt, std::nullopt, static_cast<float>(identity_weight));
  }

 private:
  static void run_impl(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      std::optional<TensorView> permuted_idx,
      std::optional<TensorView> expert_weights,
      std::optional<TensorView> shared_output,
      std::optional<TensorView> norm_weight,
      double eps,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      std::optional<TensorView> pre,
      std::optional<TensorView> normalized,
      std::optional<TensorView> quantized,
      std::optional<TensorView> scales,
      float identity_weight) {
    using namespace host;
    const auto& comm = *ref.get();
    const auto& push = comm.get_push_obj();
    CHECK_HOST(push.world_size == kWorldSize)
        << "communicator holds " << push.world_size << " ranks, kernel built for " << kWorldSize;

    auto T = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"num_permuted_rows"};
    auto TK = SymbolicSize{"num_expanded"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({T, kHiddenDim})
        .with_strides({kHiddenDim, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(out);
    TensorMatcher({P, kHiddenDim})
        .with_strides({kHiddenDim, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(gemm2_out);
    if constexpr (kIdentity) {
      CHECK_HOST(!permuted_idx.has_value() && !expert_weights.has_value() && !shared_output.has_value());
      CHECK_HOST(P.unwrap() == T.unwrap()) << "identity routing reads row t of x for token t";
    } else {
      CHECK_HOST(permuted_idx.has_value() && expert_weights.has_value());
      TensorMatcher({T, kTopK})
          .with_strides({kTopK, 1})
          .with_dtype<WeightT>()
          .template with_device<kDLCUDA>(device)
          .verify(expert_weights.value());
      TK.set_value(T.unwrap() * kTopK);
      TensorMatcher({TK})
          .with_strides({1})
          .with_dtype<int32_t>()
          .with_device<kDLCUDA>(device)
          .verify(permuted_idx.value());
    }
    if (shared_output.has_value()) {
      TensorMatcher({T, kHiddenDim})
          .with_strides({kHiddenDim, 1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(shared_output.value());
    }
    if (norm_weight.has_value()) {
      TensorMatcher({kHiddenDim})
          .with_strides({1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(norm_weight.value());
    }
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    if constexpr (kQuant) {
      CHECK_HOST(num_tokens <= 128) << "the quant epilogue writes one 128-row scale tile";
      CHECK_HOST(norm_weight.has_value());
      TensorMatcher({T, kHiddenDim}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device).verify(quantized.value());
      TensorMatcher({(kHiddenDim / 32) * 128})
          .with_dtype<uint8_t>()
          .with_device<kDLCUDA>(device)
          .verify(scales.value());
    }
    if (norm_weight.has_value()) {
      TensorMatcher({T, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(pre.value());
      TensorMatcher({T, kHiddenDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(normalized.value());
    }
    TensorMatcher({T, 4, kHiddenDim})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(mhc_out)
        .verify(residual);
    TensorMatcher({T, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(post);
    TensorMatcher({T, 4, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(comb);
    CHECK_HOST(num_tokens > 0) << "num_tokens must be positive";
    CHECK_HOST(reinterpret_cast<uintptr_t>(gemm2_out.data_ptr()) % 16 == 0) << "gemm2_out must be 16B aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0) << "out must be 16B aligned";

    const int64_t nbytes = num_tokens * int64_t(kHiddenDim) * sizeof(bf16_t);
    CHECK_HOST(nbytes <= push.slot_bytes) << "num_tokens * hidden * 2 = " << nbytes << " bytes exceeds the "
                                          << push.slot_bytes << "-byte push slot";
    CHECK_HOST(num_tokens <= push.num_blocks)
        << "num_tokens = " << num_tokens << " exceeds the " << push.num_blocks << " push phase counters of the plane";
    const uint32_t num_clusters = num_tokens + (num_tokens < push.num_blocks ? 1 : 0);

    const auto params = Params{
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .gemm2 = static_cast<const bf16_t*>(gemm2_out.data_ptr()),
        .idx = permuted_idx.has_value() ? static_cast<const int32_t*>(permuted_idx.value().data_ptr()) : nullptr,
        .weights = expert_weights.has_value() ? static_cast<const WeightT*>(expert_weights.value().data_ptr())
                                              : nullptr,
        .shared = shared_output.has_value() ? static_cast<const bf16_t*>(shared_output.value().data_ptr()) : nullptr,
        .shared_gate = nullptr,
        .norm_weight = norm_weight.has_value() ? static_cast<const bf16_t*>(norm_weight.value().data_ptr()) : nullptr,
        .norm_eps = static_cast<float>(eps),
        .prefetch_metadata = false,
        .rank = push.rank,
        .num_tokens = num_tokens,
        .num_push_counters = push.num_blocks,
        .ws = push.get_workspace<kWorldSize>(nbytes),
        .mhc_out = static_cast<bf16_t*>(mhc_out.data_ptr()),
        .residual = static_cast<const bf16_t*>(residual.data_ptr()),
        .post = static_cast<const float*>(post.data_ptr()),
        .comb = static_cast<const float*>(comb.data_ptr()),
        .pre = pre.has_value() ? static_cast<const float*>(pre.value().data_ptr()) : nullptr,
        .normalized = normalized.has_value() ? static_cast<bf16_t*>(normalized.value().data_ptr()) : nullptr,
        .quantized = quantized.has_value() ? static_cast<fp8_e4m3_t*>(quantized.value().data_ptr()) : nullptr,
        .scales = scales.has_value() ? static_cast<uint8_t*>(scales.value().data_ptr()) : nullptr,
        .identity_weight = identity_weight,
    };

    const auto has_shared = shared_output.has_value();
    const auto has_norm = norm_weight.has_value();
    const auto kern = has_shared ? (has_norm ? kernel<true, true> : kernel<true, false>)
                                 : (has_norm ? kernel<false, true> : kernel<false, false>);
    LaunchKernel(dim3(num_clusters, kClusterSize), Trait::kBlockSize, out.device()).enable_pdl(kUsePDL)(kern, params);
  }
};

// --- lever EPA (dsv41f bundle) -------------------------------------------------------------------------------
// The attention post with wo_b's GEMM epilogue as the producer. Every rank's wo_b GEMM (epa_gemm.py) TMA-stores its
// bf16 rows, with each 0x0000 word written as 0x8000, into its own slot of its own EPA region AND into its slot of
// the peer's EPA region. This kernel polls both slots of this rank's region (Lamport: a 0x00000000 32-bit word
// is "not arrived"), rebuilds the exact bytes stock's push would have stored for each rank (finalize_vec's identity
// arithmetic acc = fmaf(x, w, 0.0f) from 0 + one bf16 cast, then Lamport::clear_pos_zero; fmaf(-0, 1, +0) = +0, so
// the producer's 0x0000 -> 0x8000 change cannot reach the reduction), then runs the FIN kernel's reduce + HC=4 mHC
// post + RMSNorm epilogue unchanged and re-arms both slots (+0). No plane round, no epoch: a region is reused only
// after a later cross-rank round completed on both ranks (callers alternate two regions by layer parity).
template <uint32_t kWorldSize>
struct EpaParams {
  FinParams<kWorldSize, fp32_t> fin;  // only norm_weight / norm_eps / mhc_out / residual / post / comb / pre /
                                      // normalized / identity_weight / num_tokens are read
  bf16_t* slots;                      // this rank's region: kWorldSize slots of slot_vecs 16-byte vectors
  uint32_t slot_vecs;
};

// kEarly (levers.EPA_EARLY): the GEMM triggers right after its own griddepcontrol.wait, so this kernel starts while
// the GEMM runs: everything it loads besides the slots was written by grids that completed before that wait, the
// slots are polled (Lamport), and it waits for the GEMM's completion only at its end (so its own completion still
// implies the GEMM's for every later grid). Its outputs are fresh buffers the GEMM never touches (the caller keeps
// the GEMM's activation alive until this launch).
template <uint32_t kWorldSize, uint32_t kHiddenDim, uint32_t kClusterSize, bool kUsePDL, uint32_t kPrePoll, bool kEarly>
__global__ __launch_bounds__(FinRowTrait<kHiddenDim, kClusterSize>::kBlockSize)
    __cluster_dims__(1, kClusterSize, 1) void epa_attn_post_kernel(const __grid_constant__ EpaParams<kWorldSize> ep) {
  namespace cg = cooperative_groups;
  using namespace device;
  using T = FinRowTrait<kHiddenDim, kClusterSize>;
  constexpr uint32_t kRowVecs = T::kRowVecs;
  constexpr uint32_t kBlockSize = T::kBlockSize;
  constexpr uint32_t kNumWarps = T::kNumWarps;
  const auto& params = ep.fin;

  const auto tx = threadIdx.x;
  const auto row_idx = blockIdx.x;
  const auto cluster_rank = blockIdx.y;
  const auto hvec = cluster_rank * kBlockSize + tx;
  const auto vid = row_idx * kRowVecs + hvec;

  if constexpr (!kEarly) PDLWaitPrimary<kUsePDL>();

  constexpr bool kPreResidual = (kPrePoll & 1) != 0;
  constexpr bool kPreCoeffs = (kPrePoll & 2) != 0;
  MhcPostInputs mhc_in;
  StageVec w;
  if constexpr (kPreResidual) mhc_post_load_residual<kHiddenDim>(params, row_idx, hvec, mhc_in);
  if constexpr (kPreCoeffs) mhc_post_load_coeffs<true>(params, row_idx, mhc_in);
  if constexpr (kPreCoeffs) w.load(params.norm_weight, hvec);

  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = ep.slots + static_cast<int64_t>(i) * ep.slot_vecs * 8;
  }
  StageVec vec[kWorldSize];
  do {
    bool has_zero = false;
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], vid);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      has_zero |= Lamport::has_pos_zero(vec[i].data());
    }
    if (!has_zero) break;
  } while (true);
  if constexpr (!kPreResidual) mhc_post_load_residual<kHiddenDim>(params, row_idx, hvec, mhc_in);
  if constexpr (!kPreCoeffs) mhc_post_load_coeffs<true>(params, row_idx, mhc_in);
  if constexpr (!kPreCoeffs) w.load(params.norm_weight, hvec);

  // stock's pushed bytes of rank i's row: finalize_vec<kIdentity, fp32 weight> + Lamport::clear_pos_zero
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    const auto w_fp32 = cast<fp32_t>(params.identity_weight);
    StageVec pushed;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      fp32x2_t acc = fp32x2_t{0.0f, 0.0f};
      const auto [x, y] = cast<fp32x2_t>(vec[i][j]);
      acc.x = fmaf(x, w_fp32, acc.x);
      acc.y = fmaf(y, w_fp32, acc.y);
      pushed[j] = cast<bf16x2_t>(acc);
    }
    Lamport::clear_pos_zero(pushed.data());
    vec[i] = pushed;
  }

  // the FIN kernel's kNorm + kMhc epilogue, verbatim
  __shared__ float smem_sq[kClusterSize][kNumWarps];
  auto red = reduce_vec(vec);
  red = mhc_post_compute<kHiddenDim, true>(params, mhc_in, red, row_idx, hvec);
  const auto cluster = cg::this_cluster();
  fp32x2_t acc[4];
  float sq = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    acc[j] = cast<fp32x2_t>(red[j]);
    sq = fmaf(acc[j].x, acc[j].x, sq);
    sq = fmaf(acc[j].y, acc[j].y, sq);
  }
  sq = warp::reduce_sum(sq);
  const auto lane = tx % kWarpThreads;
  const auto warp = tx / kWarpThreads;
  if (lane < kClusterSize) {
    *cluster.map_shared_rank(&smem_sq[cluster_rank][warp], lane) = sq;
  }
  cluster.sync();
  float total = 0.0f;
#pragma unroll
  for (uint32_t c = 0; c < kClusterSize; ++c) {
#pragma unroll
    for (uint32_t wp = 0; wp < kNumWarps; ++wp) {
      total += smem_sq[c][wp];
    }
  }
  const auto factor = math::rsqrt(total / static_cast<float>(kHiddenDim) + params.norm_eps);
  StageVec out;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    const auto [wa, wb] = cast<fp32x2_t>(w[j]);
    out[j] = cast<bf16x2_t>(fp32x2_t{acc[j].x * factor * wa, acc[j].y * factor * wb});
  }
  ptx::st_global_16B(out, params.normalized, vid);
  PDLTriggerSecondary<kUsePDL>();

  StageVec zero_vec;
  Lamport::fill_pos_zero(zero_vec.data());
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    ptx::st_global_16B(zero_vec, poll_ptrs[i], vid);
  }
  if constexpr (kEarly) PDLWaitPrimary<kUsePDL>();
}

template <uint32_t kWorldSize, uint32_t kHiddenDim, uint32_t kClusterSize, bool kUsePDL, uint32_t kPrePoll, bool kEarly>
struct EpaAttnPost {
  using TensorView = tvm::ffi::TensorView;
  using Trait = FinRowTrait<kHiddenDim, kClusterSize>;
  static constexpr auto kernel = epa_attn_post_kernel<kWorldSize, kHiddenDim, kClusterSize, kUsePDL, kPrePoll, kEarly>;

  /// slots: this rank's EPA region [kWorldSize, S] bf16 (S >= rows * hidden); outputs mhc_out [T, 4, H] and
  /// normalized [T, H]; residual [T, 4, H] bf16, post [T, 4] / comb [T, 4, 4] / pre [T, 4] fp32, norm_weight [H].
  static void run(
      TensorView slots,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      TensorView pre,
      TensorView norm_weight,
      double eps,
      TensorView normalized,
      double identity_weight) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto S = SymbolicSize{"slot_elems"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({kWorldSize, S}).with_strides({S, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(slots);
    TensorMatcher({T, 4, kHiddenDim})
        .with_strides({4 * kHiddenDim, kHiddenDim, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(mhc_out)
        .verify(residual);
    TensorMatcher({T, 4}).with_strides({4, 1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(post);
    TensorMatcher({T, 4, 4}).with_strides({16, 4, 1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(comb);
    TensorMatcher({T, 4}).with_strides({4, 1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(pre);
    TensorMatcher({kHiddenDim}).with_strides({1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(norm_weight);
    TensorMatcher({T, kHiddenDim})
        .with_strides({kHiddenDim, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(normalized);
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto slot_elems = static_cast<int64_t>(S.unwrap());
    CHECK_HOST(num_tokens > 0) << "num_tokens must be positive";
    CHECK_HOST(static_cast<int64_t>(num_tokens) * kHiddenDim <= slot_elems)
        << num_tokens << " rows exceed the " << slot_elems << "-element EPA slot";
    CHECK_HOST(slot_elems % 8 == 0 && reinterpret_cast<uintptr_t>(slots.data_ptr()) % 16 == 0)
        << "EPA slots must be 16-byte vectors";
    auto params = EpaParams<kWorldSize>{};
    params.fin.norm_weight = static_cast<const bf16_t*>(norm_weight.data_ptr());
    params.fin.norm_eps = static_cast<float>(eps);
    params.fin.num_tokens = num_tokens;
    params.fin.mhc_out = static_cast<bf16_t*>(mhc_out.data_ptr());
    params.fin.residual = static_cast<const bf16_t*>(residual.data_ptr());
    params.fin.post = static_cast<const float*>(post.data_ptr());
    params.fin.comb = static_cast<const float*>(comb.data_ptr());
    params.fin.pre = static_cast<const float*>(pre.data_ptr());
    params.fin.normalized = static_cast<bf16_t*>(normalized.data_ptr());
    params.fin.identity_weight = static_cast<float>(identity_weight);
    params.slots = static_cast<bf16_t*>(slots.data_ptr());
    params.slot_vecs = static_cast<uint32_t>(slot_elems / 8);
    LaunchKernel(dim3(num_tokens, kClusterSize), Trait::kBlockSize, slots.device()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace dsv41f_epa
}  // namespace sglang
"""
_STATE = {"detail": "not installed", "on": False, "fast": 0, "checks": 0, "bound": 0, "off_keys": 0}
_H: dict = {}
_OK: dict = {}            # rows -> True (rank-agreed bytes == FIN / stock) / False
_GEMM_CACHE: dict = {}
_ARM: list = []           # the armed call (project_wo_b -> the wo_b linear)
_PEND: list = []          # the pushed call (the wo_b linear -> run_attn_post)
_REG: dict = {}           # "st": State, "failed": reason


class Arm:
    def __init__(self, mode, rows, region, tactic):
        self.mode, self.rows, self.region, self.tactic = mode, rows, region, tactic


class Pending:
    def __init__(self, arm, ptr, gemm_ok=True, own=None, ref=None, why="", keep=None):
        self.arm, self.ptr, self.gemm_ok, self.own, self.ref, self.why = arm, ptr, gemm_ok, own, ref, why
        self.keep = keep


# ------------------------------------------------------------------------------------------------ symmetric slots
class State:
    """One symmetric bf16 buffer per rank: [2 regions][WORLD sources][max_rows * HIDDEN]."""

    def __init__(self, group, rank, world, max_rows, device):
        import torch.distributed._symmetric_memory as symm_mem

        if world != WORLD:
            raise RuntimeError(f"epa: TP{WORLD} only (world {world})")
        self.rank, self.max_rows = rank, max_rows
        self.slot = max_rows * HIDDEN
        with torch.inference_mode(False), torch.no_grad():
            buf = symm_mem.empty((2 * WORLD * self.slot,), dtype=BF16, device=device)
        hdl = symm_mem.rendezvous(buf, group=group)
        if hdl.rank != rank or hdl.world_size != world:
            raise RuntimeError(f"epa: symm handle rank {hdl.rank}/{hdl.world_size} != {rank}/{world}")
        ptrs = [int(p) for p in hdl.buffer_ptrs]
        if ptrs[rank] != buf.data_ptr():
            raise RuntimeError("epa: local buffer pointer mismatch")
        delta = ptrs[1 - rank] - ptrs[rank]
        if delta % 1024:
            raise RuntimeError("epa: peer buffer not 1 KiB aligned relative to ours")
        self.buf, self.hdl = buf, hdl
        self.peer_delta = delta // 2          # bf16 elements from our buffer to the peer's
        with torch.inference_mode(False), torch.no_grad():
            buf.zero_()
        torch.cuda.synchronize(device)

    def own_slot(self, region: int, rows: int):
        """[rows, HIDDEN] view of this rank's own-source slot of `region` (the GEMM's C; the peer copy is at
        + peer_delta)."""
        off = (region * WORLD + self.rank) * self.slot
        return self.buf[off: off + rows * HIDDEN].view(rows, HIDDEN)

    def region(self, region: int):
        """[WORLD, slot] view of `region` (the post's poll slots, source rank order)."""
        off = region * WORLD * self.slot
        return self.buf[off: off + WORLD * self.slot].view(WORLD, self.slot)

    def scrub(self, group):
        import torch.distributed as dist

        torch.cuda.synchronize()
        dist.barrier(group=group)
        with torch.inference_mode(False), torch.no_grad():
            self.buf.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)


def _tp():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().tp_group


def _state(device):
    """The slots, allocated at the first eager eligible call (collective: both ranks reach it on the same call)."""
    st = _REG.get("st")
    if st is not None or _REG.get("failed"):
        return st
    group = _tp()
    why = ""
    try:
        st = State(group.device_group, group.rank_in_group, group.world_size, int(levers.EPA_MAX_ROWS), device)
    except Exception as e:  # noqa: BLE001
        st, why = None, f"{type(e).__name__}: {str(e)[:200]}"
    if not fin._agree(st is not None):
        _REG["failed"] = why or "peer allocation failed"
        levers.announce("epa-alloc", f"lever EPA: slots not allocated ({_REG['failed']}): off")
        return None
    import torch.distributed as dist

    dist.barrier(group=group.device_group)
    _REG["st"] = st
    levers.announce("epa-alloc", f"lever EPA: slots {2 * WORLD * st.slot * 2 / 2 ** 20:.1f} MiB / rank, "
                                 f"peer delta {st.peer_delta} elements")
    return st


# ------------------------------------------------------------------------------------------------ install
def install() -> str:
    """JIT-build the post kernel (one load_jit call, FIN's flags) once in prepare; needs lever FIN on."""
    if _STATE["on"] or _STATE["detail"].startswith("off"):
        return _STATE["detail"]
    if not getattr(levers, "EPA", False):
        _STATE["detail"] = "off (levers.EPA)"
        return _STATE["detail"]
    try:
        if not fin._STATE.get("on"):
            raise RuntimeError(f"lever FIN is {fin._STATE.get('detail')}")
        from dsv41f import ag

        mod = sys.modules.get("sglang.srt.models.deepseek_v4")
        if mod is None:
            from sglang.srt.models import deepseek_v4 as mod
        want = levers.AG_PINS.get("sglang.srt.models.deepseek_v4:MQALayer._project_wo_b")
        got = ag.source_hash(inspect.getsource(mod), "MQALayer._project_wo_b")
        if want is None or got != want:
            raise RuntimeError(f"MQALayer._project_wo_b {got} != pinned {want}")
        from sglang.kernels.jit.utils import load_jit
        from sglang.kernels.jit.utils.compile import cache as jit_cache

        path = _write_header(str(jit_cache.cache_root()))
        import os

        sha = os.path.basename(os.path.dirname(path))
        pdl = bool(fin._STATE.get("pdl"))
        prepoll = int(fin._STATE.get("prepoll", 3))
        early = bool(getattr(levers, "EPA_EARLY", False))
        k = f"dsv41f_epa::EpaAttnPost<{WORLD}, {HIDDEN}, {CLUSTER}, {fin._cpp(pdl)}, {prepoll}, {fin._cpp(early)}>::run"
        jm = load_jit("dsv41f_epa", sha, f"pdl{int(pdl)}", f"p{prepoll}", f"e{int(early)}", cuda_files=[path],
                      cuda_wrappers=[("epa_attn", k)])
        from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce

        _H.update(mod=jm, dsv4=mod, skip=should_skip_mlp_all_reduce)
    except Exception as e:  # noqa: BLE001 - any changed internal or a failed build: FIN's post as before
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:400]})"
        return _STATE["detail"]
    _STATE["on"] = True
    _STATE["detail"] = (f"on: wo_b's GEMM epilogue pushes its rows to both ranks' EPA slots, the attention post only "
                        f"polls (rows {levers.EPA_ROWS[0]}..{min(levers.EPA_ROWS[1], levers.EPA_MAX_ROWS)}, header "
                        f"{sha}, PDL {pdl}, pre-poll {prepoll}, early start {early}); first eager call per row count checked vs FIN / "
                        f"stock (rank-agreed), captures on verified row counts only")
    return _STATE["detail"]


def _write_header(root: str) -> str:
    import hashlib
    import os

    data = _HEADER.encode()
    sha = hashlib.sha256(data).hexdigest()[:16]
    d = os.path.join(root, "dsv41f_src", sha)
    path = os.path.join(d, "epa_arf.cuh")
    if os.path.exists(path):
        try:
            if fin._read_bytes(path) == data:
                return path
        except OSError:
            pass
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
    finally:
        os.close(fd)
    os.replace(tmp, path)
    if fin._read_bytes(path) != data:
        raise RuntimeError(f"{path}: content differs after write")
    return path


# ------------------------------------------------------------------------------------------------ the GEMM
def gemm_into(a, b, a_scale, b_scale, tactic, out, peer_delta: int):
    """pwn.gemm's launch on EpaGemmKernel: C = out (a [m, n] row-major view of our slot), the peer copy at
    out + peer_delta elements. a: (m, k) fp8 activation, b: weight.t() (k, n)."""
    import cutlass
    from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _compile_block_scaled_gemm, _prepare_alpha_for_launch

    from dsv41f.epa_gemm import EpaGemmKernel

    m, real_k = a.shape
    n = b.shape[1]
    tiler, cluster, swap, pf, split = tactic
    if split != 1:
        raise ValueError(f"lever EPA: split-K tactic {tactic}")
    if swap:
        kernel_m, kernel_n = n, m
        ka, kb, ka_sf, kb_sf = b.T, a, b_scale, a_scale
        w_op = "a"
    else:
        kernel_m, kernel_n = m, n
        ka, kb, ka_sf, kb_sf = a, b.T, a_scale, b_scale
        w_op = "b"
    sf_m = (kernel_m + 127) // 128
    sf_n = (kernel_n + 127) // 128
    sf_k = (real_k // 32 + 3) // 4
    # lever EPA_EARLY: the GEMM releases the post right after its own wait (the post polls, waits at its end)
    early = bool(levers.PWS_EARLY_GEMM_TRIGGER) or bool(getattr(levers, "EPA_EARLY", False))
    pre_stages, l2_blocks = int(levers.PWS_PRE_STAGES), int(levers.PWS_L2_BLOCKS)
    key = ("epa", 32, tuple(tiler), tuple(cluster), bool(swap), bool(pf), True, BF16, 1, w_op, early, pre_stages,
           l2_blocks, int(peer_delta))
    compiled, _ = _compile_block_scaled_gemm(
        _GEMM_CACHE, key,
        lambda: EpaGemmKernel(32, tuple(tiler), tuple(cluster), bool(pf), True, w_operand=w_op, early_trigger=early,
                              w_pre_stages=pre_stages, w_l2_blocks=l2_blocks, peer_delta=int(peer_delta)),
        ab_cutlass_dtype=cutlass.Float8E4M3FN, sf_dtype=cutlass.Float8E8M0FNU, c_cutlass_dtype=cutlass.BFloat16,
        ab_assumed_align=16, cluster_shape_mn=tuple(cluster), swap_ab=bool(swap), sf_m=sf_m, sf_n=sf_n, sf_k=sf_k,
        batch_size=1, cluster_shape_k=1)
    if tuple(out.shape) != (m, n) or out.stride() != (n, 1) or out.dtype is not BF16:
        raise ValueError("lever EPA: C must be a contiguous [m, n] bf16 view")
    launch_out = out.as_strided(out.shape, (1, out.shape[0])) if swap else out
    compiled(ka, kb, launch_out, sf_m, sf_n, sf_k, ka_sf.data_ptr(), kb_sf.data_ptr(),
             _prepare_alpha_for_launch(None, a.device))
    return out


def _epa_gemm(input, weight, weight_scale, arm, st):
    from dsv41f import pwn

    x2 = input.view(-1, input.shape[-1])
    q, s = pwn.quant(x2, early=levers.PWS_EARLY_QUANT)
    w_scale = weight_scale.t() if weight_scale.ndim == 2 else weight_scale
    out = st.own_slot(arm.region, arm.rows)
    gemm_into(q, weight.t(), s, w_scale, arm.tactic, out, st.peer_delta)
    return out, (q, s)   # q / s stay referenced until the post is launched (lever EPA_EARLY)


# ------------------------------------------------------------------------------------------------ binding
def project_wo_b(attn, o, defer_all_reduce: bool = False):
    """MQALayer._project_wo_b (pinned) for lever L2PF's lifted attention forward: stock's method, with EPA armed
    for the wo_b call of a deferred (fused-post) attention all-reduce at an EPA row count."""
    arm = _arm_for(attn, o, defer_all_reduce)
    if arm is None:
        return attn._project_wo_b(o, defer_all_reduce)
    _ARM.append(arm)
    try:
        y = attn._project_wo_b(o, defer_all_reduce)
    finally:
        _ARM.clear()
    if _PEND and _PEND[-1].arm is arm:
        m = fin._mhc()
        if not (isinstance(y, m.AttnOutput) and y.partial.data_ptr() == _PEND[-1].ptr):
            raise RuntimeError("lever EPA: the pushed wo_b rows did not reach a deferred attention post")
    return y


def _arm_for(attn, o, defer):
    if not defer or not _STATE["on"] or not fin._STATE.get("on") or not isinstance(o, torch.Tensor) or _PEND:
        return None
    if torch.compiler.is_compiling():
        return None
    rows = int(o.shape[0])
    lo, hi = levers.EPA_ROWS
    if rows < max(lo, int(levers.FIN_MIN_ROWS)) or rows > min(hi, int(levers.EPA_MAX_ROWS), int(levers.FIN_MAX_ROWS)):
        return None
    wo_b = getattr(attn, "wo_b", None)
    if wo_b is None or not getattr(wo_b, "dsv4f_epa", False) or wo_b.bias is not None or not wo_b.input_is_parallel:
        return None
    w = wo_b._parameters.get("weight")
    if w is None or tuple(w.shape) != (HIDDEN, K_IN) or o.numel() != rows * K_IN or o.dtype is not BF16:
        return None
    # _project_wo_b's own defer conditions (stock source pinned at install)
    if not (attn.attn_tp_size > 1 and attn.attn_tp_size == _H["dsv4"].get_parallel().tp_size
            and wo_b.reduce_results and not wo_b.use_decode_attn_tp and not _H["skip"]()):
        return None
    if attn.attn_tp_size != WORLD:
        return None
    ok = _OK.get(rows)
    if ok is False:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if ok is None and capturing:
        return None
    st = _REG.get("st")
    if st is None:
        if capturing:
            return None
        st = _state(w.device)
        if st is None:
            return None
    from dsv41f import pwn

    tactic = pwn.served_tactic(rows, HIDDEN, K_IN, w.device)
    if tactic is None:
        return None
    region = int(getattr(attn, "layer_id", 0)) % 2
    return Arm("fast" if ok else "check", rows, region, tactic)


def _make_linear(inner):
    def linear(input, weight, weight_scale, input_scale=None, bias=None, **kwargs):
        arm = _ARM[0] if _ARM else None
        if (arm is None or kwargs or bias is not None or input_scale is not None
                or tuple(weight.shape) != (HIDDEN, K_IN) or input.dtype is not BF16 or not input.is_contiguous()
                or input.numel() != arm.rows * K_IN):
            return inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=input_scale, bias=bias,
                         **kwargs)
        _ARM.clear()   # one GEMM per armed call
        st = _REG["st"]
        if arm.mode == "fast":
            out, keep = _epa_gemm(input, weight, weight_scale, arm, st)
            _PEND.append(Pending(arm, out.data_ptr(), keep=keep))
            return out.view(*input.shape[:-1], HIDDEN)
        ref = inner(input=input, weight=weight, weight_scale=weight_scale, input_scale=None, bias=None)
        try:
            out, keep = _epa_gemm(input, weight, weight_scale, arm, st)
            p = Pending(arm, ref.data_ptr(), True, out.clone(), ref, keep=keep)
        except Exception as e:  # noqa: BLE001 - nothing (or a partial push) launched: scrubbed in run_attn_post
            p = Pending(arm, ref.data_ptr(), False, None, ref, f"gemm {type(e).__name__}: {str(e)[:200]}")
        _PEND.append(p)
        return ref

    linear.inner = inner   # lever FG's _served_partial walks .stock / .inner down to stock's partial
    return linear


def bind(replica) -> int:
    """EPA's wrapper on the replica's wo_b linear (lever DT's private copy, already wrapped by PWN); returns how many."""
    if not getattr(levers, "EPA", False):
        return 0
    count = 0
    attn = replica._modules.get("self_attn")
    if attn is None:
        return 0
    wo_b = attn._modules.get("wo_b")
    if wo_b is None or getattr(wo_b, "dsv4f_epa", False) or not getattr(wo_b, "dsv4f_dt", False):
        return 0
    method = getattr(wo_b, "quant_method", None)
    w = wo_b._parameters.get("weight")
    if method is None or w is None or tuple(w.shape) != (HIDDEN, K_IN) or not getattr(wo_b, "dsv4f_pwn", False):
        return 0
    method.w8a8_mxfp8_linear = _make_linear(method.w8a8_mxfp8_linear)
    wo_b.dsv4f_epa = True
    count += 1
    _STATE["bound"] += count
    return count


# ------------------------------------------------------------------------------------------------ the post
def _post(st, region, rows, residual, post, comb, pre, norm_weight, eps):
    mhc_out = torch.empty_like(residual)
    normalized = torch.empty((rows, HIDDEN), dtype=BF16, device=residual.device)
    _H["mod"].epa_attn(st.region(region), mhc_out, residual, post, comb, pre, norm_weight, eps, normalized, 1.0)
    return mhc_out, normalized


def run_attn_post(hc, out, residual, *, stats_stream, next, world_size, precomputed=None):
    """fin.run_attn_post for every call without a pending EPA push; with one: its AttnOutput branch on EPA's post."""
    if not _PEND:
        return fin.run_attn_post(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                                 precomputed=precomputed)
    p = _PEND.pop()
    m = fin._mhc()
    if not isinstance(out, m.AttnOutput) or out.partial.data_ptr() != p.ptr or next is None or world_size != WORLD:
        raise RuntimeError("lever EPA: a pending wo_b push without its fused attention post")
    st = _REG["st"]
    arm = p.arm
    pre, post_mix, comb = m._compute_triplet(hc, residual, stats_stream, precomputed)
    eps = next.norm.variance_epsilon
    if arm.mode == "fast":
        updated, normalized = _post(st, arm.region, arm.rows, residual, post_mix, comb, pre, next.norm.weight, eps)
        _STATE["fast"] += 1
        return m.HcState(updated, pre, m.HcNormed(normalized))
    # first eager call at this row count: FIN's post into the real outputs, EPA's into scratch, byte compare
    _, updated, normalized = fin.all_reduce_mhc_post_combine_norm(out.partial, residual, post_mix, comb, pre,
                                                                  next.norm.weight, eps, world_size=world_size)
    _STATE["checks"] += 1
    why = p.why
    launched = fin._agree(p.gemm_ok)
    ok = False
    if launched:
        expect = p.ref.reshape(arm.rows, HIDDEN).contiguous().view(torch.int16).clone()
        expect[expect == 0] = -32768
        g_ok = torch.equal(p.own.view(torch.int16), expect)
        if not g_ok:
            why = "GEMM bytes differ from the wrapped linear's"
        ran = True
        a_ok = False
        try:
            got_u, got_n = _post(st, arm.region, arm.rows, residual, post_mix, comb, pre, next.norm.weight, eps)
            a_ok = fin._bytes_equal(updated, got_u) and fin._bytes_equal(normalized, got_n)
            if not a_ok and not why:
                why = "post outputs differ"
        except Exception as e:  # noqa: BLE001
            ran = False
            why = f"post {type(e).__name__}: {str(e)[:200]}"
        all_ran = fin._agree(ran)
        ok = fin._agree(g_ok and a_ok and ran)
        if not all_ran:
            st.scrub(_tp().device_group)
    else:
        st.scrub(_tp().device_group)
    _OK[arm.rows] = ok
    if not ok:
        _STATE["off_keys"] += 1
    levers.announce(f"epa-{arm.rows}", f"lever EPA: {arm.rows} rows (tactic {arm.tactic}) first eager call: "
                    + ("EPA == FIN byte for byte: on" if ok else f"OFF ({why or 'peer rank differed'})"))
    return m.HcState(updated, pre, m.HcNormed(normalized))


def describe(layers) -> str:
    n = sum(1 for r in layers if getattr(getattr(r._modules.get("self_attn"), "_modules", {}).get("wo_b"),
                                         "dsv4f_epa", False))
    return f"{_STATE['detail']}; wo_b linears bound {n}"
