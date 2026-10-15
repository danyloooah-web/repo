// lev_RTR (rtr.py): the MoE router gate GEMM at decode / verify rows as ONE kernel, bit for bit stock's
//   logits[M, 384] fp32 = x[M, 5120] bf16 . W[384, 5120]^T bf16
// which stock runs as torch.mm(x, w.t(), out_dtype=torch.float32) -> cuBLASLt 13.1 (B300): an nvjet tcgen05 kernel
// that splits K into S partitions (S per row count: 1, 2 or 3, rtr.py learns it from stock on every rank) followed, at
// S > 1, by cublasLt::splitKreduce_kernel. The bits (tests/rtr/rtr_bits.py, rtr_struct.py: 0 mismatches over every
// row count 17..384, real gate weights of all 40 layers, special values):
//   partition s of K: S = 1 [0, 5120); S = 2 [0, 2560) [2560, 5120); S = 3 [0, 1728) [1728, 3456) [3456, 5120);
//   chain s = ONE accumulator per output element, K = 16 MMA steps in ascending K from +0 (an mma.sync m16n8k16
//            bf16 -> f32 step == a tcgen05 kind::f16 step bit for bit: products and accumulator aligned to the largest
//            exponent with 2 guard bits, truncated, rounded toward zero);
//   splitKreduce (beta 0, its SASS): acc = +0; acc = add.rn(acc, p_s) for s = 0 .. S-1; out = mul.rn(acc, alpha 1);
//   S = 1: out = mul.rn(p0, 1). Round to nearest, no FTZ: every float op of the epilogue is inline PTX (bundle units
//   build with --use_fast_math). Tile, cluster and 2-CTA choices of cuBLAS do not change the bits.
// Layout: a cluster of S CTAs per tile of ET experts x RT rows (grid (S, 384 / ET, ceil(M / RT))), CTA rank s = chain
// s; warps 0 .. NW-1 run the chains (warp = 16 experts x 8 rows, mma.sync), warp NW = the TMA producer (3-D tensor
// maps (64-element K block, rows, K / 64) with SWIZZLE_128B, a STAGES-deep ring of NKB K blocks of W and x). Ranks >= 1
// push their fp32 partials into rank 0's shared memory (DSMEM, after a cluster barrier phase that every CTA arrived at
// on entry: all CTAs of the cluster have started), one release / acquire cluster barrier, rank 0 sums in partition
// order and stores. No workspace, no second launch.
// PDL: the producer issues the weight's TMA for the first STAGES stages before griddepcontrol.wait (nothing in the
// stream writes a gate weight), x's TMA after it; every global store (the output, which may be memory the previous
// kernel still uses) follows data that arrived after the wait; no thread reads global memory with ld.global.nc (x is
// read by TMA only). launch_dependents once the producer has issued its last load.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// No torch headers (cacheon's build_cuda_ext compiles C++17): Python passes raw device pointers, sizes and the current
// stream; rtr.py checks dtypes, shapes, contiguity and alignment.

namespace rtr {

constexpr int kE = 384, kK = 5120, kKB = kK / 64;   // experts, hidden, 64-element K blocks
constexpr int kMaxRows = 4096;
constexpr int kMaxWeights = 1024;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nRTR_WAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RTR_WAIT_%=;\n}" ::"r"(
          bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, int c2, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(
          dst),
      "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void hmma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float fadd_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fmul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void st_cluster_f32(uint32_t local_addr, uint32_t rank, float v) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(local_addr), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(ra), "f"(v) : "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_sync_rel_acq() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// K-block ranges of partition s (64-element blocks: 2560 = 40 blocks, 1728 = 27, 3456 = 54)
__host__ __device__ constexpr int kb_lo(int S, int s) { return S == 1 ? 0 : S == 2 ? 40 * s : (s == 0 ? 0 : s == 1 ? 27 : 54); }
__host__ __device__ constexpr int kb_hi(int S, int s) {
  return S == 1 ? 80 : S == 2 ? 40 * (s + 1) : (s == 0 ? 27 : s == 1 ? 54 : 80);
}

template <int ET, int RT, int S, int NKB, int STAGES>
struct Smem {
  static constexpr int W_STAGE = NKB * ET * 128, X_STAGE = NKB * RT * 128, STAGE = W_STAGE + X_STAGE;
  static constexpr int OFF_RED = STAGES * STAGE;                    // float [S - 1][ET][RT]: partials of ranks >= 1
  static constexpr int OFF_BAR = OFF_RED + (S - 1) * ET * RT * 4;   // full[STAGES], empty[STAGES]
  static constexpr int BYTES = OFF_BAR + 2 * STAGES * 8 + 1024;     // + 1024-byte alignment slack (SWIZZLE_128B)
};

template <int ET, int RT>
constexpr int kThreads = ((ET / 16) * (RT / 8) + 1) * 32;

template <int ET, int RT, int S, int NKB, int STAGES>
__global__ void __launch_bounds__(kThreads<ET, RT>, 1)
    rtr_kernel(const __grid_constant__ CUtensorMap tm_w, const __grid_constant__ CUtensorMap tm_x, float* out, int M,
               int pdl) {
  using L = Smem<ET, RT, S, NKB, STAGES>;
  constexpr int NMT = ET / 16, NNT = RT / 8, NW = NMT * NNT;
  static_assert(ET % 16 == 0 && RT % 8 == 0 && L::BYTES <= 227 * 1024, "tile");
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  const uint32_t sbase = smem_u32(smem);
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + L::OFF_BAR);
  float* red = reinterpret_cast<float*>(smem + L::OFF_RED);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int s = (S == 1) ? 0 : int(cluster_rank());
  const int e0 = blockIdx.y * ET, r0 = blockIdx.z * RT;
  const int kb0 = kb_lo(S, s), nkb = kb_hi(S, s) - kb0;
  const int nst = (nkb + NKB - 1) / NKB;
  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(smem_u32(&bars[i]), 1);
      mbar_init(smem_u32(&bars[STAGES + i]), NW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (S > 1) cluster_arrive_relaxed();   // phase 0: this CTA has started (waited on before any DSMEM store)

  if (warp == NW) {
    // ---- producer
    if (lane == 0) {
      const int pre = nst < STAGES ? nst : STAGES;
      for (int i = 0; i < pre; ++i) {
        const uint32_t bar = smem_u32(&bars[i]);
        mbar_expect_tx(bar, L::STAGE);
        tma_3d(sbase + i * L::STAGE, &tm_w, 0, e0, kb0 + i * NKB, bar);
      }
      if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int i = 0; i < pre; ++i)
        tma_3d(sbase + i * L::STAGE + L::W_STAGE, &tm_x, 0, r0, kb0 + i * NKB, smem_u32(&bars[i]));
      for (int i = pre; i < nst; ++i) {
        const int slot = i % STAGES;
        mbar_wait(smem_u32(&bars[STAGES + slot]), ((i / STAGES) - 1) & 1);
        const uint32_t bar = smem_u32(&bars[slot]);
        mbar_expect_tx(bar, L::STAGE);
        tma_3d(sbase + slot * L::STAGE, &tm_w, 0, e0, kb0 + i * NKB, bar);
        tma_3d(sbase + slot * L::STAGE + L::W_STAGE, &tm_x, 0, r0, kb0 + i * NKB, bar);
      }
      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    __syncwarp();
    if (S > 1) {
      cluster_wait();
      cluster_sync_rel_acq();
    }
    return;
  }
  // ---- chains: warp = m-tile mt (16 experts) x n-tile nt (8 rows)
  const int mt = warp / NNT, nt = warp % NNT;
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  // ldmatrix lane roles. A (W, 16 experts x 16 K) x4: matrix q = lane / 8: rows (q & 1) * 8 + (lane & 7), 16-byte unit
  // + (q >> 1). B (x, 8 rows x 16 K) x2 per n-tile; x4 loads two K steps' B at once: matrices (u, u + 1) of ks, ks + 1.
  const int a_row = mt * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), a_uadd = lane >> 4;
  const int b_row = nt * 8 + (lane & 7), b_uadd = lane >> 3;   // x4: units 0..3 of a 64-byte half
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(smem_u32(&bars[slot]), (i / STAGES) & 1);
    const uint32_t wst = sbase + slot * L::STAGE, xst = wst + L::W_STAGE;
    const int kbn = (nkb - i * NKB) < NKB ? (nkb - i * NKB) : NKB;
#pragma unroll
    for (int kb = 0; kb < NKB; ++kb) {
      if (kb < kbn) {
        const uint32_t wslab = wst + kb * (ET * 128), xslab = xst + kb * (RT * 128);
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
          // B for K steps 2 kh and 2 kh + 1 (units 4 kh .. 4 kh + 3)
          uint32_t b0, b1, b2, b3;
          {
            const int u = 4 * kh + b_uadd;
            ldsm4(xslab + b_row * 128 + ((u ^ (b_row & 7)) << 4), b0, b1, b2, b3);
          }
#pragma unroll
          for (int kq = 0; kq < 2; ++kq) {
            const int ks = 2 * kh + kq;
            uint32_t a0, a1, a2, a3;
            const int u = 2 * ks + a_uadd;
            ldsm4(wslab + a_row * 128 + ((u ^ (a_row & 7)) << 4), a0, a1, a2, a3);
            if (kq == 0)
              hmma(c, a0, a1, a2, a3, b0, b1);
            else
              hmma(c, a0, a1, a2, a3, b2, b3);
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(&bars[STAGES + slot]));
  }
  // ---- epilogue: c = (expert g, rows 2t, 2t + 1) and (expert g + 8, rows 2t, 2t + 1) of tile (mt, nt)
  const int g = lane >> 2, t = lane & 3;
  const int rr = nt * 8 + 2 * t, ee = mt * 16 + g;
  if (S > 1) {
    cluster_wait();   // phase 0 complete: every CTA of the cluster is running
    if (s > 0) {
      const uint32_t base = smem_u32(red) + uint32_t((s - 1) * ET * RT) * 4;
      st_cluster_f32(base + uint32_t(ee * RT + rr) * 4, 0, c[0]);
      st_cluster_f32(base + uint32_t(ee * RT + rr + 1) * 4, 0, c[1]);
      st_cluster_f32(base + uint32_t((ee + 8) * RT + rr) * 4, 0, c[2]);
      st_cluster_f32(base + uint32_t((ee + 8) * RT + rr + 1) * 4, 0, c[3]);
    }
    cluster_sync_rel_acq();
  }
  if (s == 0) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int e = ee + (q >> 1) * 8, r = rr + (q & 1);
      float acc = c[q];
      if (S > 1) {
        acc = fadd_rn(0.f, acc);
#pragma unroll
        for (int p = 1; p < S; ++p) acc = fadd_rn(acc, red[(p - 1) * ET * RT + e * RT + r]);
      }
      acc = fmul_rn(acc, 1.0f);
      if (r0 + r < M) out[size_t(r0 + r) * kE + e0 + e] = acc;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------------------------------------------------
PFN_cuTensorMapEncodeTiled_v12000 g_encode = nullptr;

// 3-D view (64 elements, rows, K / 64 blocks) of a row-major [rows][5120] bf16 matrix; box (64, box_rows, nkb), SW128
int encode(CUtensorMap* tm, const void* ptr, int rows, int box_rows, int nkb) {
  if (!g_encode) return -1;
  cuuint64_t dims[3] = {64, cuuint64_t(rows), cuuint64_t(kKB)};
  cuuint64_t strides[2] = {cuuint64_t(kK) * 2, 128};
  cuuint32_t box[3] = {64, cuuint32_t(box_rows), cuuint32_t(nkb)};
  cuuint32_t es[3] = {1, 1, 1};
  CUresult r = g_encode(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, es,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? 0 : 1000 + int(r);
}

// the weight maps a layer needs: one per (ET, NKB) a config uses (ET 16 / 32 / 48, NKB 4 / 8)
constexpr int kWMaps = 6;
constexpr int wmap_index(int et, int nkb) { return (et / 16 - 1) * 2 + (nkb == 8 ? 0 : 1); }
struct WMaps {
  CUtensorMap tm[kWMaps];
  uintptr_t ptr;
};
std::vector<WMaps> g_weights;
bool g_init = false;

using LaunchFn = int (*)(const CUtensorMap&, const void*, float*, int, int, cudaStream_t);
using AttrFn = int (*)();

template <int ET, int RT, int S, int NKB, int STAGES>
int set_attr() {
  using L = Smem<ET, RT, S, NKB, STAGES>;
  return int(cudaFuncSetAttribute(rtr_kernel<ET, RT, S, NKB, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  L::BYTES));
}

template <int ET, int RT, int S, int NKB, int STAGES>
int launch_t(const CUtensorMap& tw, const void* x, float* out, int M, int pdl, cudaStream_t st) {
  using L = Smem<ET, RT, S, NKB, STAGES>;
  alignas(64) CUtensorMap tx;
  int rc = encode(&tx, x, M, RT, NKB);
  if (rc) return rc;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(S, kE / ET, (M + RT - 1) / RT);
  cfg.blockDim = dim3(kThreads<ET, RT>);
  cfg.dynamicSmemBytes = L::BYTES;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (S > 1) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = S;
    at[na].val.clusterDim.y = 1;
    at[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  return int(cudaLaunchKernelEx(&cfg, rtr_kernel<ET, RT, S, NKB, STAGES>, tw, tx, out, M, pdl));
}

struct Config {
  const char* name;
  int et, rt, s, nkb, stages;
  LaunchFn fn;
  AttrFn attr;
};
#define RTR_C(ET, RT, S, NKB, ST)                                                                      \
  {"e" #ET "r" #RT "s" #S "k" #NKB "st" #ST, ET, RT, S, NKB, ST, &launch_t<ET, RT, S, NKB, ST>, \
   &set_attr<ET, RT, S, NKB, ST>}
// tile configs (all share the arithmetic: the bits do not depend on the tile; tests/rtr_test.py checks every config at
// every served row count of its S). Grid CTAs = S x (384 / ET) x ceil(M / RT); B300 runs one kernel's CTAs per SM, so
// a config also decides how many SMs stay free for the side streams (NQ, the shared expert, the mHC stats).
const Config kConfigs[] = {
    RTR_C(16, 24, 2, 8, 4),   // 0  default S 2, M <= 72
    RTR_C(32, 16, 2, 8, 4),   // 1  default S 2, 73 <= M <= 111
    RTR_C(16, 24, 2, 4, 4),   // 2  (2 CTAs an SM: at 144 rows 288 CTAs fill every SM; slower in-engine, c26rtr rtr1)
    RTR_C(32, 32, 2, 8, 3),   // 3  default S 2, M >= 145
    RTR_C(16, 24, 1, 8, 4),   // 4  default S 1, M < 144
    RTR_C(16, 32, 1, 8, 4),   // 5  default S 1, M >= 144
    RTR_C(16, 24, 3, 8, 4),   // 6  default S 3, M < 100
    RTR_C(32, 24, 3, 8, 3),   // 7  default S 3, M >= 100
    RTR_C(32, 24, 2, 8, 3),   // 8  default S 2, 112 <= M <= 144
    RTR_C(16, 40, 2, 8, 3),   // 9
    RTR_C(32, 40, 2, 8, 3),   // 10
    RTR_C(48, 24, 2, 8, 3),   // 11
    RTR_C(16, 40, 1, 8, 3),   // 12
    RTR_C(32, 24, 1, 8, 3),   // 13
};
// measured (tests/rtr_test.py Tc, 40-call cold chain, us p0 / p1): <= 72 rows S 2 e16r24k8st4 5.24-5.33 / 3.76-4.22 beats
// every other tile (e16r24k8st5 5.51-5.67: deeper rings are slower); 96 rows e32r16 5.90 / 4.77; S 1 108-126 rows
// e16r24k8st4 6.76-6.90 / 5.29-5.55. Fewer-CTA tiles (e16r40, e32r24, e48r24, e32r40) cost +0.8..+3 us.
constexpr int kNumConfigs = int(sizeof(kConfigs) / sizeof(kConfigs[0]));

int pick(int64_t M, int64_t S) {
  if (M < 1 || M > kMaxRows) return -1;
  if (S == 2) return M <= 72 ? 0 : M <= 111 ? 1 : M <= 144 ? 8 : 3;
  if (S == 1) return M < 144 ? 4 : 5;
  if (S == 3) return M < 100 ? 6 : 7;
  return -1;
}

// the config index for (rows, S): `name` ('' = the default dispatch above), -1 = none / S mismatch
int64_t config_index(int64_t M, int64_t S, const std::string& name) {
  if (name.empty()) return pick(M, S);
  if (M < 1 || M > kMaxRows) return -1;
  for (int i = 0; i < kNumConfigs; ++i)
    if (name == kConfigs[i].name) return kConfigs[i].s == S ? i : -1;
  return -1;
}

std::string config_name(int64_t i) {
  return (i < 0 || i >= kNumConfigs) ? std::string() : std::string(kConfigs[i].name);
}

int64_t config_ctas(int64_t i, int64_t M) {
  if (i < 0 || i >= kNumConfigs) return 0;
  const Config& c = kConfigs[i];
  return int64_t(c.s) * (kE / c.et) * ((M + c.rt - 1) / c.rt);
}

// once, outside any capture: the driver's tensor-map encoder + every config's dynamic shared memory limit
void init() {
  if (g_init) return;
  if (!g_encode) {
    cudaDriverEntryPointQueryResult q;
    void* fn = nullptr;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || fn == nullptr)
      throw std::runtime_error("rtr: cuTensorMapEncodeTiled not available");
    g_encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }
  for (int i = 0; i < kNumConfigs; ++i) {
    const int rc = kConfigs[i].attr();
    if (rc) throw std::runtime_error(std::string("rtr: cudaFuncSetAttribute failed for ") + kConfigs[i].name + ": " +
                                     cudaGetErrorString(cudaError_t(rc)));
  }
  g_init = true;
}

// a gate weight [384, 5120] bf16 (contiguous, 16-byte aligned, static for the process): its tensor maps -> handle
int64_t add_weight(int64_t w) {
  if (!g_init) throw std::runtime_error("rtr: init() first");
  if (w == 0 || w % 16) throw std::runtime_error("rtr: weight not 16-byte aligned");
  if (int(g_weights.size()) >= kMaxWeights) throw std::runtime_error("rtr: too many weights");
  WMaps m;
  m.ptr = uintptr_t(w);
  for (int et = 16; et <= 48; et += 16)
    for (int nkb = 4; nkb <= 8; nkb += 4) {
      const int rc = encode(&m.tm[wmap_index(et, nkb)], reinterpret_cast<const void*>(w), kE, et, nkb);
      if (rc) throw std::runtime_error("rtr: weight tensor map encode failed: " + std::to_string(rc));
    }
  g_weights.push_back(m);
  return int64_t(g_weights.size()) - 1;
}

// logits [M, 384] fp32 (out, contiguous) = x [M, 5120] bf16 (contiguous, 16-byte aligned) . W(handle)^T, S partitions,
// tile config `cfg` (config_index)
void run(int64_t handle, int64_t w, int64_t x, int64_t out, int64_t M, int64_t S, int64_t cfg, int64_t pdl,
         int64_t stream) {
  if (!g_init) throw std::runtime_error("rtr: init() first");
  if (handle < 0 || handle >= int64_t(g_weights.size())) throw std::runtime_error("rtr: bad weight handle");
  const WMaps& wm = g_weights[size_t(handle)];
  if (uintptr_t(w) != wm.ptr) throw std::runtime_error("rtr: weight pointer differs from the handle's");
  if (x == 0 || x % 16 || out == 0 || out % 16) throw std::runtime_error("rtr: x / out not 16-byte aligned");
  if (M < 1 || M > kMaxRows) throw std::runtime_error("rtr: rows");
  if (cfg < 0 || cfg >= kNumConfigs || kConfigs[cfg].s != S) throw std::runtime_error("rtr: config / split mismatch");
  const Config& c = kConfigs[cfg];
  const int rc = c.fn(wm.tm[wmap_index(c.et, c.nkb)], reinterpret_cast<const void*>(x), reinterpret_cast<float*>(out),
                      int(M), pdl ? 1 : 0, reinterpret_cast<cudaStream_t>(stream));
  if (rc) {
    std::string why = rc >= 1000 ? "tensor map encode " + std::to_string(rc) : cudaGetErrorString(cudaError_t(rc));
    throw std::runtime_error("rtr: launch failed (" + std::string(c.name) + "): " + why);
  }
}

}  // namespace rtr

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &rtr::init, "resolve the tensor-map encoder, set every config's shared memory limit (outside capture)");
  m.def("add_weight", &rtr::add_weight, "encode a [384, 5120] bf16 gate weight's tensor maps; returns its handle");
  m.def("run", &rtr::run, "router logits = x . W^T as stock's cuBLAS split-K GEMM + splitKreduce, bit for bit");
  m.def("config_index", &rtr::config_index, "config index for (rows, S, name; '' = default dispatch), -1 = none");
  m.def("config_name", &rtr::config_name, "a config index's name ('' = none)");
  m.def("config_ctas", &rtr::config_ctas, "CTAs a config launches at a row count");
  m.attr("NUM_CONFIGS") = int64_t(rtr::kNumConfigs);
  m.attr("EXPERTS") = int64_t(rtr::kE);
  m.attr("HIDDEN") = int64_t(rtr::kK);
  m.attr("MAX_ROWS") = int64_t(rtr::kMaxRows);
}
