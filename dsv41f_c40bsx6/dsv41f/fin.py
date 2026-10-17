"""Lever FIN (no CUDA unit; one JIT module): the attention post and the MoE post of OUR layer replicas run a
scheduling-only copy of stock's fused MoE-finalize -> 1shot push all-reduce -> HC=4 mHC post kernel.

Stock (fork 5e72ce5): deepseek_v4_mhc.run_attn_post drives all_reduce_mhc_post_combine_norm (the finalize kernel with
identity routing: idx = arange, fp32 weights = 1, top_k 1, mHC + norm epilogue) and run_moe_post drives
moe_finalize_all_reduce_mhc_post_combine_norm_quant (<= 128 rows with an MXFP8 next input) or
moe_finalize_all_reduce_mhc_post, both JIT builds of csrc/distributed/all_reduce_fusion.cuh.

Ours (_HEADER below, Apache-2.0, derived from that header, md5 pinned): the same kernel with the same floating-point
operations in the same order and the same push-plane protocol, with (a) every epilogue input but the reduced vector
(residual, post, comb, pre, norm weight) loaded right after the push, before the poll; (d) no store of the reduced row
the callers discard; (e) the attention post's identity routing generated in the kernel (rows[0] = token, w[0] = a
runtime fp32 1.0f); (f) optionally (levers.FIN_EARLY_PDL) the PDL trigger right after the push.

Build: install() (model_node.prepare, after lever AR) writes the header content-addressed under the JIT cache root and
makes ONE load_jit call (wrappers attn_norm, moe_quant_*, moe_norm_*, moe_post_* for fp32 / bf16 expert weights); any
exception keeps stock. Binding: the replicas' lifted forward (hq_layer / fg_layer) calls fin.run_attn_post /
fin.run_moe_post: stock's fused branch copied verbatim with only the op replaced by ours; every other case is handed to
the module's own deepseek_v4_mhc.run_attn_post / run_moe_post (whatever is bound there, e.g. lever MO's wrapper). No
sglang function, class or served module is patched.

Exactness: per (variant, rows, top_k, weight dtype, shared) the first EAGER call runs stock's op into the real outputs
and ours into scratch on the same inputs (two plane rounds on both ranks; an exception in ours is answered with a
stock round so the plane stays in step) and compares mhc_out, normalized, quantized and the full scale tile byte for
byte; the verdict is agreed over the TP group (MIN). A difference or exception keeps stock for that key. Under CUDA
graph capture only verified keys run ours; unverified keys run stock.
"""
import ast
import hashlib
import inspect
import os
import sys

import torch

from dsv41f import levers

HIDDEN = 5120
WORLD = 2
CLUSTER = 5
STOCK_CUH_MD5 = "36d9ef4de405a0d7e5070094913d2fe7"
_STATE = {"detail": "not installed", "on": False, "fast": 0, "stock": 0, "checks": 0, "early": False}
_OK: dict = {}            # key -> True (bytes == stock, rank-agreed) / False
_H: dict = {}             # mod, stock ops, group, top_k, dtypes

# sha256[:16] of the source segment of every stock function lever FIN copies or relies on (fork 5e72ce5 image)
PINNED = {
    ("sglang.srt.models.deepseek_v4_mhc", "run_attn_post"): "4b8a2e74df252a1f",
    ("sglang.srt.models.deepseek_v4_mhc", "run_moe_post"): "5b60fa71e8a576a9",
    ("sglang.srt.models.deepseek_v4_mhc", "_compute_triplet"): "eeaad45953d06be9",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "all_reduce_mhc_post_combine_norm"): "83fa25aaf0b4058d",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "moe_finalize_all_reduce_mhc_post"): "ec7c8a69ca664e95",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "moe_finalize_all_reduce_mhc_post_combine_norm_quant"):
        "53b840214c6bfb15",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "moe_finalize_all_reduce_mhc_post_combine_norm"):
        "11fdbf39b4f0c69e",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "_identity_routing"): "504ae75b02df3868",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "_jit_mhc_module"): "586c85bee27b468f",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "_jit_mhc_quant_module"): "fb92f756c1cdf133",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "_moe_finalize_all_reduce_mhc_post_op"): "348a35981221ae70",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "_moe_finalize_all_reduce_mhc_post_combine_norm_op"):
        "4cc6352a91e1976c",
    ("sglang.kernels.ops.communication.all_reduce_mhc", "_moe_finalize_all_reduce_mhc_post_combine_norm_quant_op"):
        "c841e16af34e87c0",
    ("sglang.kernels.ops.communication.all_reduce_fusion", "default_cluster_size"): "483880e8502c01b5",
    ("sglang.kernels.ops.communication.all_reduce_fusion", "get_registered_comm"): "da85cd507e8e0b7a",
    ("sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe", "can_fuse_all_reduce"): "95addc69e95cc676",
}

_HEADER = r"""// Lever FIN (dsv41f bundle): a scheduling-only copy of SGLang's fused MoE-finalize -> 1shot push all-reduce -> HC=4
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
namespace dsv41f_fin {

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

}  // namespace dsv41f_fin
}  // namespace sglang
"""


def source_hash(module, qualname: str, cache=None):
    """sha256[:16] of the exact source text of `qualname` ("f" or "Class.f") in `module` (None: not found)."""
    hit = cache.get(module.__name__) if cache is not None else None
    if hit is None:
        src = inspect.getsource(module)
        hit = (src, ast.parse(src))
        if cache is not None:
            cache[module.__name__] = hit
    src, tree = hit
    node = tree
    for part in qualname.split("."):
        found = None
        for child in ast.walk(node):
            if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part
                    and child is not node):
                found = child
                break
        if found is None:
            return None
        node = found
    seg = ast.get_source_segment(src, node)
    return hashlib.sha256(seg.encode()).hexdigest()[:16] if seg is not None else None


def _modules():
    import sglang.kernels.ops.communication.all_reduce_fusion  # noqa: F401
    import sglang.kernels.ops.communication.all_reduce_mhc  # noqa: F401
    import sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe  # noqa: F401
    import sglang.srt.models.deepseek_v4_mhc  # noqa: F401
    return {name: sys.modules[name] for name, _ in PINNED}


def _read_bytes(path: str) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _write_header(root: str) -> str:
    """The header at <root>/dsv41f_src/<sha256[:16]>/fin_arf.cuh (written once, atomically; content-checked)."""
    data = _HEADER.encode()
    sha = hashlib.sha256(data).hexdigest()[:16]
    d = os.path.join(root, "dsv41f_src", sha)
    path = os.path.join(d, "fin_arf.cuh")
    if os.path.exists(path):
        try:
            if _read_bytes(path) == data:
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
    if _read_bytes(path) != data:
        raise RuntimeError(f"{path}: content differs after write")
    return path


def _cpp(v) -> str:
    return "true" if v is True else "false" if v is False else str(v)


def _wrappers(pdl: bool, early: bool, top_k: int, prepoll: int = 3, suffix: str = "", dtypes=None):
    def k(top, wt, quant, ident):
        return (f"dsv41f_fin::FinMhcAllReduce<{WORLD}, {HIDDEN}, {top}, {CLUSTER}, {_cpp(pdl)}, {wt}, true, "
                f"{_cpp(quant)}, {_cpp(ident)}, {_cpp(early)}, {int(prepoll)}>")
    out = [(f"attn_norm{suffix}", k(1, "fp32_t", False, True) + "::run_identity_norm")]
    for tag, wt in (dtypes or (("f32", "fp32_t"), ("bf16", "bf16_t"))):
        out += [(f"moe_quant_{tag}{suffix}", k(top_k, wt, True, False) + "::run_mhc_post_combine_norm_quant"),
                (f"moe_norm_{tag}{suffix}", k(top_k, wt, False, False) + "::run_mhc_post_combine_norm"),
                (f"moe_post_{tag}{suffix}", k(top_k, wt, False, False) + "::run_mhc_post")]
    return out


def install() -> str:
    """Pin stock, write + JIT-build our header (one load_jit call), record stock's ops; once, in prepare."""
    if _STATE["on"] or _STATE["detail"].startswith("off"):
        return _STATE["detail"]
    if not getattr(levers, "FIN", False):
        _STATE["detail"] = "off (levers.FIN)"
        return _STATE["detail"]
    try:
        mods = _modules()
        cache: dict = {}
        bad = [f"{mod}.{name} {got}" for (mod, name), want in PINNED.items()
               if (got := source_hash(mods[mod], name, cache)) != want]
        if bad:
            raise RuntimeError("source differs from the mirror: " + ", ".join(bad))
        from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit
        from sglang.kernels.jit.utils.compile import cache as jit_cache
        from sglang.kernels.jit.utils.compile import paths as jit_paths

        stock_cuh = os.path.join(str(jit_paths.KERNEL_PATH), "csrc", "distributed", "all_reduce_fusion.cuh")
        md5 = hashlib.md5(_read_bytes(stock_cuh)).hexdigest()
        if md5 != STOCK_CUH_MD5:
            raise RuntimeError(f"stock all_reduce_fusion.cuh md5 {md5} != {STOCK_CUH_MD5}")
        arf = mods["sglang.kernels.ops.communication.all_reduce_fusion"]
        arm = mods["sglang.kernels.ops.communication.all_reduce_mhc"]
        mhc = mods["sglang.srt.models.deepseek_v4_mhc"]
        arf.require_cluster_launch_arch()
        if int(arf.default_cluster_size(HIDDEN)) != CLUSTER or arm._MHC_HIDDEN_DIM != HIDDEN:
            raise RuntimeError(f"stock geometry {arf.default_cluster_size(HIDDEN)} / {arm._MHC_HIDDEN_DIM}")
        for n in ("AttnOutput", "HcState", "HcNormed", "HcQuantized", "MoEOutput", "Mxfp8SwizzledInput",
                  "is_deferred_finalize", "_compute_triplet", "run_attn_post", "run_moe_post"):
            if not hasattr(mhc, n):
                raise RuntimeError(f"deepseek_v4_mhc.{n} missing")
        pdl = bool(is_arch_support_pdl())
        early = bool(getattr(levers, "FIN_EARLY_PDL", False))
        if getattr(levers, "FIN_MAILBOX", False):
            raise RuntimeError("FIN_MAILBOX is not built")
        top_k = int(getattr(levers, "FIN_TOP_K", 6))
        prepoll = int(getattr(levers, "FIN_PREPOLL", 3))
        if prepoll not in (0, 1, 2, 3):
            raise RuntimeError(f"FIN_PREPOLL {prepoll}")
        path = _write_header(str(jit_cache.cache_root()))
        sha = os.path.basename(os.path.dirname(path))
        mod = load_jit("dsv41f_fin", sha, f"pdl{int(pdl)}", f"e{int(early)}", f"k{top_k}", f"p{prepoll}",
                       cuda_files=[path], cuda_wrappers=_wrappers(pdl, early, top_k, prepoll))
        _H.update(mod=mod, arf=arf, mhc=mhc, top_k=top_k,
                  s_attn=arm.all_reduce_mhc_post_combine_norm,
                  s_quant=arm.moe_finalize_all_reduce_mhc_post_combine_norm_quant,
                  s_post=arm.moe_finalize_all_reduce_mhc_post,
                  fns={("quant", torch.float32): mod.moe_quant_f32, ("quant", torch.bfloat16): mod.moe_quant_bf16,
                       ("post", torch.float32): mod.moe_post_f32, ("post", torch.bfloat16): mod.moe_post_bf16})
        _STATE.update(early=early, path=path, pdl=pdl, prepoll=prepoll)
    except Exception as e:  # noqa: BLE001 - any changed internal or a failed build keeps stock
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:400]})"
        return _STATE["detail"]
    _STATE["on"] = True
    _STATE["detail"] = (f"on: attention + MoE posts on our finalize/all-reduce/mHC kernel (pre-poll epilogue loads, no "
                        f"dead reduced-row store, in-kernel identity routing; pre-poll loads {prepoll}; early PDL {'on' if early else 'off'}), "
                        f"PDL {'on' if pdl else 'off'}, top_k {top_k}, rows {levers.FIN_MIN_ROWS}..{levers.FIN_MAX_ROWS}; "
                        f"header {sha}; first eager call per key checked vs stock (rank-agreed), captures on verified "
                        f"keys only")
    return _STATE["detail"]


def _agree(ok: bool) -> bool:
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group.device_group
    if dist.get_world_size(group) != WORLD:
        return False
    t = torch.tensor([1 if ok else 0], device=torch.cuda.current_device(), dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def _bytes_equal(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _gate(key, rows: int, world_size: int):
    """False: stock; None: first eager call (check); True: ours."""
    if not _STATE["on"] or world_size != WORLD or torch.compiler.is_compiling():
        return False
    if rows < int(levers.FIN_MIN_ROWS) or rows > int(levers.FIN_MAX_ROWS):
        return False
    ok = _OK.get(key)
    if ok is None and torch.cuda.is_current_stream_capturing():
        return False
    return ok


def _first(key, stock_call, ours_call, compare):
    """Stock into the real outputs, ours into scratch (2 plane rounds on every rank), byte compare, rank-agreed."""
    ref = stock_call()
    _STATE["checks"] += 1
    why = ""
    try:
        got = ours_call()
    except Exception as e:  # noqa: BLE001 - nothing launched: a stock round keeps the plane in step
        why = f"{type(e).__name__}: {str(e)[:200]}"
        got = None
        stock_call()
    if got is not None:
        bad = [i for i in compare if not _bytes_equal(ref[i], got[i])]
        if bad:
            why = f"outputs {bad} differ"
    ok = _agree(not why)
    _OK[key] = ok
    levers.announce(f"fin-{'-'.join(map(str, key))}",
                    f"lever FIN: {key} first eager call: " + ("ours == stock byte for byte: on" if ok else
                                                              f"OFF ({why or 'peer rank differed'}): stock"))
    return ref


def _fail(key, e):
    _OK[key] = False
    levers.announce(f"fin-err-{'-'.join(map(str, key))}",
                    f"lever FIN: {key} failed ({type(e).__name__}: {str(e)[:160]}): stock for this key")


def _ours_attn(comm, x, residual, post, comb, pre, norm_weight, eps):
    out = torch.empty((x.shape[0], HIDDEN), dtype=torch.bfloat16, device=x.device)
    mhc_out = torch.empty_like(residual)
    normalized = torch.empty_like(out)
    _H["mod"].attn_norm(comm, out, x, mhc_out, residual, post, comb, pre, norm_weight, eps, normalized, 1.0)
    return out, mhc_out, normalized


def all_reduce_mhc_post_combine_norm(x, residual, post, comb, pre, norm_weight, eps, *, world_size):
    """Stock's all_reduce_mhc_post_combine_norm contract (returns (reduced, mhc_out, normalized); reduced is not
    written by ours: every caller discards it)."""
    def stock():
        return _H["s_attn"](x, residual, post, comb, pre, norm_weight, eps, world_size=world_size)
    rows = int(x.shape[0])
    key = ("attn", rows)
    ok = _gate(key, rows, world_size)
    if ok is False:
        _STATE["stock"] += 1
        return stock()
    comm = _H["arf"].get_registered_comm(world_size)
    if comm is None or x.dtype is not torch.bfloat16:
        _STATE["stock"] += 1
        return stock()

    def ours():
        return _ours_attn(comm, x, residual, post, comb, pre, norm_weight, eps)
    if ok is None:
        return _first(key, stock, ours, (1, 2))
    try:
        got = ours()
    except Exception as e:  # noqa: BLE001 - nothing launched: stock's round instead
        _fail(key, e)
        return stock()
    _STATE["fast"] += 1
    return got


def _ours_moe(kind, comm, gemm2, idx, weights, shared, residual, post, comb, pre=None, norm_weight=None, eps=None):
    rows = weights.shape[0]
    out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=gemm2.device)
    mhc_out = torch.empty_like(residual)
    fn = _H["fns"][(kind, weights.dtype)]
    if kind == "post":
        fn(comm, out, gemm2, idx, weights, shared, mhc_out, residual, post, comb)
        return out, mhc_out
    normalized = torch.empty_like(out)
    quantized = torch.empty_like(out, dtype=torch.float8_e4m3fn)
    scales = torch.empty((HIDDEN // 32) * 128, device=gemm2.device, dtype=torch.uint8)
    fn(comm, out, gemm2, idx, weights, shared, mhc_out, residual, post, comb, pre, norm_weight, eps, normalized,
       quantized, scales)
    return out, mhc_out, normalized, quantized, scales


def _moe(kind, stock, compare, gemm2, idx, weights, top_k, shared, residual, post, comb, pre, norm_weight, eps,
         world_size):
    rows = int(weights.shape[0])
    key = (kind, rows, int(top_k), str(weights.dtype).replace("torch.", ""), shared is not None)
    ok = _gate(key, rows, world_size)
    if ok is not False and (int(top_k) != _H["top_k"] or (kind, weights.dtype) not in _H["fns"]
                            or gemm2.dtype is not torch.bfloat16):
        ok = False
    comm = _H["arf"].get_registered_comm(world_size) if ok is not False else None
    if ok is False or comm is None:
        _STATE["stock"] += 1
        return stock()

    def ours():
        return _ours_moe(kind, comm, gemm2, idx, weights, shared, residual, post, comb, pre, norm_weight, eps)
    if ok is None:
        return _first(key, stock, ours, compare)
    try:
        got = ours()
    except Exception as e:  # noqa: BLE001 - nothing launched: stock's round instead
        _fail(key, e)
        return stock()
    _STATE["fast"] += 1
    return got


def moe_finalize_all_reduce_mhc_post_combine_norm_quant(gemm2, idx, weights, top_k, shared, residual, post, comb,
                                                        pre, norm_weight, eps, *, world_size):
    """Stock's moe_finalize_all_reduce_mhc_post_combine_norm_quant contract (reduced not written by ours)."""
    def stock():
        return _H["s_quant"](gemm2, idx, weights, top_k, shared, residual, post, comb, pre, norm_weight, eps,
                             world_size=world_size)
    return _moe("quant", stock, (1, 2, 3, 4), gemm2, idx, weights, top_k, shared, residual, post, comb, pre,
                norm_weight, eps, world_size)


def moe_finalize_all_reduce_mhc_post(gemm2, idx, weights, top_k, shared, residual, post, comb, *, world_size):
    """Stock's moe_finalize_all_reduce_mhc_post contract (reduced not written by ours)."""
    def stock():
        return _H["s_post"](gemm2, idx, weights, top_k, shared, residual, post, comb, world_size=world_size)
    if not int(weights.shape[0]):
        return stock()
    return _moe("post", stock, (1,), gemm2, idx, weights, top_k, shared, residual, post, comb, None, None, None,
                world_size)


def _mhc():
    m = _H.get("mhc")
    if m is None:
        from sglang.srt.models import deepseek_v4_mhc as m
    return m


def run_attn_post(hc, out, residual, *, stats_stream, next, world_size, precomputed=None):
    """deepseek_v4_mhc.run_attn_post (pinned): its AttnOutput branch copied with only the op replaced by ours; every
    other call goes to the module's run_attn_post itself."""
    m = _mhc()
    if not _STATE["on"] or not isinstance(out, m.AttnOutput):
        return m.run_attn_post(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                               precomputed=precomputed)
    assert next is not None
    pre, post_mix, comb = m._compute_triplet(hc, residual, stats_stream, precomputed)
    _, updated, normalized = all_reduce_mhc_post_combine_norm(
        out.partial,
        residual,
        post_mix,
        comb,
        pre,
        next.norm.weight,
        next.norm.variance_epsilon,
        world_size=world_size,
    )
    return m.HcState(updated, pre, m.HcNormed(normalized))


def run_moe_post(hc, out, residual, *, stats_stream, next, world_size, precomputed=None):
    """deepseek_v4_mhc.run_moe_post (pinned): its fused deferred-finalize branch copied with only the ops replaced by
    ours; every other call goes to the module's run_moe_post itself."""
    m = _mhc()
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        can_fuse_all_reduce,
    )

    if not (
        _STATE["on"]
        and isinstance(out, m.MoEOutput)
        and m.is_deferred_finalize(out.routed)
        and can_fuse_all_reduce(out.routed.expert_weights.shape[0], hc.cfg.hidden)
    ):
        return m.run_moe_post(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                              precomputed=precomputed)
    pre, post_mix, comb = m._compute_triplet(hc, residual, stats_stream, precomputed)
    args = (
        out.routed.gemm2_out,
        out.routed.expanded_idx_to_permuted_idx,
        out.routed.expert_weights,
        out.routed.top_k,
        out.shared,
        residual,
        post_mix,
        comb,
    )
    if (
        next is not None
        and next.accepts_mxfp8
        and next.norm_fusable
        # The quant epilogue supports at most 128 batch size
        and out.routed.expert_weights.shape[0] <= 128
    ):
        _, updated, normalized, q, sf = (
            moe_finalize_all_reduce_mhc_post_combine_norm_quant(
                *args,
                pre,
                next.norm.weight,
                next.norm.variance_epsilon,
                world_size=world_size,
            )
        )
        return m.HcState(
            updated, pre, m.HcQuantized(normalized, m.Mxfp8SwizzledInput(q, sf))
        )
    _, updated = moe_finalize_all_reduce_mhc_post(*args, world_size=world_size)
    return m.HcState(updated, pre)


def stats() -> str:
    return (f"fast {_STATE['fast']}, stock {_STATE['stock']}, checks {_STATE['checks']}, keys on "
            f"{sorted(k for k, v in _OK.items() if v)}, off {sorted(k for k, v in _OK.items() if not v)}")
