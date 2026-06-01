/*******************************************************************************
 *
 * Standalone 64×64×32 i8 GEMM — rocBLAS-oracle tensile lego harness
 * with optional rocWMMA cross-checks.
 * Compile:
 *   hipcc -O3 -ffast-math --offload-arch=gfx1100 -std=c++20 -fopenmp -DNDEBUG \
 *     -I/mnt/f/git/rocm-libraries/projects/rocwmma/library/include \
 *     -I/mnt/f/git/Tnn-miner \
 *     -o perf_gemm64x64x32 /mnt/f/git/Tnn-miner/perf_gemm64x64x32.cpp -lrocblas
 *
 * Run:
 *   ./perf_gemm64x64x32 [M [N [K [runs warmups [alpha beta [wmmaCrossCheck]]]]]]]
 *
 ******************************************************************************/

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocblas/rocblas.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(expr)                                         \
  do                                                                      \
  {                                                                       \
    if (auto status = (expr); status != rocblas_status_success)           \
    {                                                                     \
      std::cerr << "rocBLAS error: '" << rocblas_status_to_string(status) \
                << "' (" << status << ") at " << __FILE__ << ":"          \
                << __LINE__ << std::endl;                                 \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (false)
#endif

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(expr)                                    \
  do                                                             \
  {                                                              \
    if (auto status = (expr); status != hipSuccess)              \
    {                                                            \
      std::cerr << "HIP error: '" << hipGetErrorString(status)   \
                << "' (" << status << ") at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                        \
      std::exit(EXIT_FAILURE);                                   \
    }                                                            \
  } while (false)
#endif

using InputT = int8_t;
using OutputT = int32_t;
using ComputeT = int32_t;

constexpr uint32_t kRocwmmaM = 16u;
constexpr uint32_t kRocwmmaN = 16u;
constexpr uint32_t kRocwmmaK = 16u;
constexpr uint32_t kWarpSize = rocwmma::Constants::AMDGCN_WAVE_SIZE_32;

struct Problem
{
  uint32_t m = 7168;
  uint32_t n = 7168;
  uint32_t k = 7168;
  ComputeT alpha = 2;
  ComputeT beta = 2;
  uint32_t warmups = 2;
  uint32_t runs = 5;
  bool wmmaCrossCheck = false;
};

struct Buffers
{
  std::vector<InputT> hA;
  std::vector<InputT> hB;
  std::vector<OutputT> hC;
  InputT *dA = nullptr;
  InputT *dB = nullptr;
  OutputT *dC = nullptr;
  OutputT *dD = nullptr;
  OutputT *dRef = nullptr;
  OutputT *dRoc = nullptr;

  uint32_t *dMismatch = nullptr;

  // new
  uint32_t *dMismatchCount = nullptr;
  uint32_t *dMismatchList = nullptr;
};

template <typename ProbeT>
inline void dump_probe(const ProbeT &p)
{
  auto print4 = [](const char *name, const int32_t *v)
  {
    std::cout << name << ": [" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << "]" << std::endl;
  };
  auto print8 = [](const char *name, const int32_t *v)
  {
    std::cout << name << ": [";
    for (int i = 0; i < 8; ++i)
    {
      if (i)
        std::cout << ", ";
      std::cout << v[i];
    }
    std::cout << "]" << std::endl;
  };

  print4("b0", p.b0);
  print4("b1", p.b1);
  print4("b2", p.b2);
  print4("b3", p.b3);
  print8("acc00", p.acc00);
  print8("acc01", p.acc01);
  print8("acc10", p.acc10);
  print8("acc11", p.acc11);
}

inline double gops(const Problem &p)
{
  return 2.0 * static_cast<double>(p.m) * static_cast<double>(p.n) * static_cast<double>(p.k) * 1.0e-9;
}

inline double macs(const Problem &p)
{
  return static_cast<double>(p.m) * static_cast<double>(p.n) * static_cast<double>(p.k) * 1.0e-12;
}

#include "perf_i8gemm_tensile_lego.hpp"

// ============================================================================
// GemmParams (wmma reference)
// ============================================================================

template <uint32_t BlocksM,
          uint32_t BlocksN,
          uint32_t TBlockX,
          uint32_t TBlockY,
          uint32_t KGroup,
          uint32_t LdsPad = 0,
          uint32_t RocwmmaK = kRocwmmaK,
          uint32_t LdsPadA = 0,
          uint32_t LdsPadB = LdsPad,
          typename SchedulerT = rocwmma::fragment_scheduler::coop_row_major_2d<TBlockX, TBlockY>>
struct GemmParams
{
  using DataLayoutA = rocwmma::col_major;
  using DataLayoutB = rocwmma::row_major;
  using DataLayoutC = rocwmma::row_major;
  using DataLayoutLds = rocwmma::col_major;

  static constexpr uint32_t ROCWMMA_M = kRocwmmaM;
  static constexpr uint32_t ROCWMMA_N = kRocwmmaN;
  static constexpr uint32_t ROCWMMA_K = RocwmmaK;
  static constexpr uint32_t BLOCKS_M = BlocksM;
  static constexpr uint32_t BLOCKS_N = BlocksN;
  static constexpr uint32_t TBLOCK_X = TBlockX;
  static constexpr uint32_t TBLOCK_Y = TBlockY;
  static constexpr uint32_t WARP_SIZE = kWarpSize;
  static constexpr uint32_t K_GROUP = KGroup;
  static constexpr uint32_t LDS_PAD_A = LdsPadA;
  static constexpr uint32_t LDS_PAD_B = LdsPadB;
  static constexpr uint32_t LDS_PAD = LDS_PAD_A + LDS_PAD_B;

  static constexpr uint32_t WARP_TILE_M = BLOCKS_M * ROCWMMA_M;
  static constexpr uint32_t WARP_TILE_N = BLOCKS_N * ROCWMMA_N;
  static constexpr uint32_t WARP_TILE_K = ROCWMMA_K;
  static constexpr uint32_t WARPS_M = TBLOCK_X / WARP_SIZE;
  static constexpr uint32_t WARPS_N = TBLOCK_Y;
  static constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M;
  static constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N;
  static constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;

  using MmaFragA = rocwmma::fragment<rocwmma::matrix_a, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutA>;
  using MmaFragB = rocwmma::fragment<rocwmma::matrix_b, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutB>;
  using MmaFragC = rocwmma::fragment<rocwmma::accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, OutputT, DataLayoutC>;
  using MmaFragD = MmaFragC;
  using MmaFragAcc = rocwmma::fragment<rocwmma::accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT>;

  using CoopScheduler = SchedulerT;
  using GRFragA = rocwmma::fragment<rocwmma::matrix_a, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                                    InputT, DataLayoutA, CoopScheduler>;
  using GRFragB = rocwmma::fragment<rocwmma::matrix_b, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                                    InputT, DataLayoutB, CoopScheduler>;

  using LWFragA = rocwmma::apply_data_layout_t<GRFragA, DataLayoutLds>;
  using LWFragB = rocwmma::apply_data_layout_t<rocwmma::apply_transpose_t<GRFragB>, DataLayoutLds>;
  using LRFragA = rocwmma::apply_data_layout_t<MmaFragA, DataLayoutLds>;
  using LRFragB = rocwmma::apply_data_layout_t<rocwmma::apply_transpose_t<MmaFragB>, DataLayoutLds>;
};

// 64×64×32 wmma reference: BlocksM=2, BlocksN=2, TBlockX=64, TBlockY=2, KGroup=2
using Wmma64x64x32 = GemmParams<2u, 2u, 64u, 2u, 2u>;

// ============================================================================
// WMMA grouped kernel (reference)
// ============================================================================

template <typename P>
__global__ __launch_bounds__(P::TBLOCK_X *P::TBLOCK_Y) void rocwmma_grouped_kernel(
    uint32_t m, uint32_t n, uint32_t k,
    InputT const *a, InputT const *b, OutputT const *c, OutputT *d,
    uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t ldd,
    ComputeT alpha, ComputeT beta)
{
  using namespace rocwmma;

  constexpr auto warpTileSize = make_coord2d(P::WARP_TILE_M, P::WARP_TILE_N);
  constexpr auto macroTileSize = make_coord2d(P::MACRO_TILE_M, P::MACRO_TILE_N);

  const auto localWarpCoord = make_coord2d(threadIdx.x / P::WARP_SIZE, threadIdx.y);
  const auto macroTileCoord = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize + localWarpCoord * warpTileSize;

  if (get<0>(macroTileCoord) >= m || get<1>(macroTileCoord) >= n)
    return;

  using MmaFragAMap1d = GetDataLayout_t<typename P::MmaFragA>;
  using MmaFragBMap1d = GetDataLayout_t<typename P::MmaFragB>;
  using MmaFragCMap1d = GetDataLayout_t<typename P::MmaFragC>;

  typename P::MmaFragAcc accFrag;
  fill_fragment(accFrag, 0);

  const uint32_t numKSteps = k / P::MACRO_TILE_K;
  for (uint32_t kStep = 0; kStep < numKSteps; ++kStep)
  {
    const uint32_t kBase = kStep * P::MACRO_TILE_K;

    typename P::MmaFragA aFrag;
    typename P::MmaFragB bFrag;

    const auto aCoord = MmaFragAMap1d::fromMatrixCoord(
        make_coord2d(get<0>(macroTileCoord), kBase), lda);
    const auto bCoord = MmaFragBMap1d::fromMatrixCoord(
        make_coord2d(kBase, get<1>(macroTileCoord)), ldb);

    load_matrix_sync(aFrag, a + aCoord, lda);
    load_matrix_sync(bFrag, b + bCoord, ldb);

    mma_sync(accFrag, aFrag, bFrag, accFrag);
  }

  // Apply D = alpha * (A * B) + beta * C.
  typename P::MmaFragC cFrag;
  typename P::MmaFragD dFrag;
  const auto dCoord = MmaFragCMap1d::fromMatrixCoord(
      macroTileCoord, ldd);
  const auto cCoord = MmaFragCMap1d::fromMatrixCoord(
      macroTileCoord, ldc);
  load_matrix_sync(cFrag, c + cCoord, ldc);

#pragma unroll
  for (uint32_t i = 0; i < dFrag.num_elements; ++i)
  {
    dFrag.x[i] = static_cast<OutputT>(alpha * accFrag.x[i] + beta * static_cast<ComputeT>(cFrag.x[i]));
  }

  store_matrix_sync(d + dCoord, dFrag, ldd);
}

// ============================================================================
// Include the combined header (pearl-style combined kernel)
// ============================================================================

// Before including, define CombinedParams alias for 64×64×32:
//   BlocksM=2, BlocksN=2, TBLOCK_X=64, TBLOCK_Y=2, KGroup=2,
//   MSubpasses=1, BStride=0, LdsPad=0, PcSplit=0

bool compare_output(const Problem &p, Buffers &buffers);

#include "perf_i8gemm_combined.hpp"

// 64×64×32 combined: Pearl64x64x32_BL
using Pearl64x64x32_BL = CombinedParams<2u, 2u, 64u, 2u, 2u, 1u>;

// ============================================================================
// compare, init, free helpers
// ============================================================================

__global__ void compare_output_kernel(OutputT const *actual,
                                      OutputT const *expected,
                                      uint32_t total,
                                      uint32_t *mismatchIndex)
{
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = tid; i < total; i += stride)
  {
    if (actual[i] != expected[i])
    {
      atomicMin(mismatchIndex, i);
    }
  }
}


__global__ void collect_mismatches_kernel(OutputT const* actual,
                                          OutputT const* expected,
                                          uint32_t total,
                                          uint32_t n,
                                          uint32_t* mismatchCount,
                                          uint32_t* mismatchList,
                                          uint32_t maxDump,
                                          uint32_t tileM,
                                          uint32_t tileN)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for(uint32_t i = tid; i < total; i += stride)
    {
        if(actual[i] != expected[i])
        {
            const uint32_t row = i / n;
            const uint32_t col = i - row * n;

            if(row >= tileM || col >= tileN)
                continue;

            const uint32_t slot = atomicAdd(mismatchCount, 1u);
            if(slot < maxDump)
            {
                mismatchList[slot] = i;
            }
        }
    }
}

constexpr uint32_t kMaxMismatchDump = 256u;

void dump_tile_diff(const Problem& p,
                    OutputT const* dActual,
                    OutputT const* dExpected,
                    uint32_t rows = 32,
                    uint32_t cols = 64)
{
    std::vector<OutputT> hActual(static_cast<size_t>(p.m) * p.n);
    std::vector<OutputT> hExpected(static_cast<size_t>(p.m) * p.n);

    CHECK_HIP_ERROR(hipMemcpy(hActual.data(), dActual,
                              hActual.size() * sizeof(OutputT),
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hExpected.data(), dExpected,
                              hExpected.size() * sizeof(OutputT),
                              hipMemcpyDeviceToHost));

    std::cout << "--- diff mask first " << rows << "x" << cols
              << " (. match, X mismatch) ---\n";

    for(uint32_t r = 0; r < rows; ++r)
    {
        std::cout << "r" << std::setw(2) << r << " ";
        for(uint32_t c = 0; c < cols; ++c)
        {
            uint32_t idx = r * p.n + c;
            std::cout << (hActual[idx] == hExpected[idx] ? '.' : 'X');
            if((c & 15u) == 15u) std::cout << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "--- actual first 16 rows x 32 cols ---\n";
    for(uint32_t r = 0; r < 16; ++r)
    {
        std::cout << "r" << std::setw(2) << r << " ";
        for(uint32_t c = 0; c < 32; ++c)
        {
            uint32_t idx = r * p.n + c;
            std::cout << std::setw(8) << hActual[idx] << " ";
        }
        std::cout << '\n';
    }

    std::cout << "--- expected first 16 rows x 32 cols ---\n";
    for(uint32_t r = 0; r < 16; ++r)
    {
        std::cout << "r" << std::setw(2) << r << " ";
        for(uint32_t c = 0; c < 32; ++c)
        {
            uint32_t idx = r * p.n + c;
            std::cout << std::setw(8) << hExpected[idx] << " ";
        }
        std::cout << '\n';
    }
}

bool compare_output(const Problem &p, OutputT const *dActual, OutputT const *dExpected, Buffers &buffers)
{
  const uint32_t total = p.m * p.n;
  const uint32_t noMismatch = 0xffffffffu;
  CHECK_HIP_ERROR(hipMemcpy(buffers.dMismatch, &noMismatch, sizeof(uint32_t), hipMemcpyHostToDevice));
  compare_output_kernel<<<1024, 256>>>(dActual, dExpected, total, buffers.dMismatch);
  CHECK_HIP_ERROR(hipGetLastError());
  uint32_t mismatch = noMismatch;
  CHECK_HIP_ERROR(hipMemcpy(&mismatch, buffers.dMismatch, sizeof(uint32_t), hipMemcpyDeviceToHost));
  CHECK_HIP_ERROR(hipDeviceSynchronize());
  if (mismatch != noMismatch)
  {
    OutputT actual = 0;
    OutputT expected = 0;
    CHECK_HIP_ERROR(hipMemcpy(&actual, dActual + mismatch, sizeof(OutputT), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(&expected, dExpected + mismatch, sizeof(OutputT), hipMemcpyDeviceToHost));
    std::cout << " first mismatch idx=" << mismatch
              << " row=" << (mismatch / p.n)
              << " col=" << (mismatch % p.n)
              << " actual=" << actual
              << " expected=" << expected
              << std::endl;

    constexpr uint32_t maxDump = kMaxMismatchDump;

    uint32_t zero = 0;
    CHECK_HIP_ERROR(hipMemcpy(buffers.dMismatchCount,
                              &zero,
                              sizeof(uint32_t),
                              hipMemcpyHostToDevice));

    collect_mismatches_kernel<<<1024, 256>>>(dActual,
                                             dExpected,
                                             total,
                                             p.n,
                                             buffers.dMismatchCount,
                                             buffers.dMismatchList,
                                             maxDump,
                                             64u,
                                             64u);
    CHECK_HIP_ERROR(hipGetLastError());
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    uint32_t mismatchCount = 0;
    CHECK_HIP_ERROR(hipMemcpy(&mismatchCount,
                              buffers.dMismatchCount,
                              sizeof(uint32_t),
                              hipMemcpyDeviceToHost));

    const uint32_t dumpCount = std::min(mismatchCount, maxDump);
    std::vector<uint32_t> hMismatchList(dumpCount);

    if (dumpCount)
    {
      CHECK_HIP_ERROR(hipMemcpy(hMismatchList.data(),
                                buffers.dMismatchList,
                                dumpCount * sizeof(uint32_t),
                                hipMemcpyDeviceToHost));
    }

    // std::cout << " mismatch count in first 64x64 tile = " << mismatchCount
    //           << ", dumping " << dumpCount << std::endl;

    // for (uint32_t j = 0; j < dumpCount; ++j)
    // {
    //   uint32_t idx = hMismatchList[j];

    //   OutputT aVal = 0;
    //   OutputT eVal = 0;
    //   CHECK_HIP_ERROR(hipMemcpy(&aVal, dActual + idx, sizeof(OutputT), hipMemcpyDeviceToHost));
    //   CHECK_HIP_ERROR(hipMemcpy(&eVal, dExpected + idx, sizeof(OutputT), hipMemcpyDeviceToHost));

    //   uint32_t row = idx / p.n;
    //   uint32_t col = idx - row * p.n;

    //   std::cout << "  mismatch[" << j << "]"
    //             << " idx=" << idx
    //             << " row=" << row
    //             << " col=" << col
    //             << " actual=" << aVal
    //             << " expected=" << eVal
    //             << " diff=" << (aVal - eVal)
    //             << std::endl;
    // }
  }
  return mismatch == noMismatch;
}

bool compare_output(const Problem &p, Buffers &buffers)
{
  return compare_output(p, buffers.dD, buffers.dRef, buffers);
}

void init_buffers(const Problem &p, Buffers &buffers)
{
  buffers.hA.resize(static_cast<size_t>(p.m) * p.k);
  buffers.hB.resize(static_cast<size_t>(p.k) * p.n);
  buffers.hC.resize(static_cast<size_t>(p.m) * p.n);

  auto make_i8 = [](uint32_t x) -> InputT {
      // integer mix, then map to signed int8 range
      x ^= x >> 16;
      x *= 0x7feb352dU;
      x ^= x >> 15;
      x *= 0x846ca68bU;
      x ^= x >> 16;
      return static_cast<InputT>(static_cast<int32_t>(x & 0xffU) - 128);
  };

  // A is addressed as: a[row + kk * lda], lda = p.m
  for(uint32_t kk = 0; kk < p.k; ++kk)
  {
      for(uint32_t row = 0; row < p.m; ++row)
      {
          size_t i = row + kk * static_cast<size_t>(p.m);
          buffers.hA[i] = make_i8(
              0xA501u
              ^ row * 0x1f123bb5u
              ^ kk  * 0x9e3779b9u);
      }
  }

  // B for rocBLAS/column-major B is addressed as: b[kk + col * ldb], ldb = p.k
  for(uint32_t col = 0; col < p.n; ++col)
  {
      for(uint32_t kk = 0; kk < p.k; ++kk)
      {
          size_t i = kk + col * static_cast<size_t>(p.k);
          buffers.hB[i] = make_i8(
              0xB701u
              ^ kk  * 0x85ebca6bu
              ^ col * 0xc2b2ae35u);
      }
  }
  
  for (size_t i = 0; i < buffers.hC.size(); ++i)
    buffers.hC[i] = static_cast<OutputT>((i * 7 + 5) & 0x7f);

  CHECK_HIP_ERROR(hipMalloc(&buffers.dA, buffers.hA.size() * sizeof(InputT)));
  CHECK_HIP_ERROR(hipMalloc(&buffers.dB, buffers.hB.size() * sizeof(InputT)));
  CHECK_HIP_ERROR(hipMalloc(&buffers.dC, buffers.hC.size() * sizeof(OutputT)));
  CHECK_HIP_ERROR(hipMalloc(&buffers.dD, buffers.hC.size() * sizeof(OutputT)));
  CHECK_HIP_ERROR(hipMalloc(&buffers.dRef, buffers.hC.size() * sizeof(OutputT)));
  CHECK_HIP_ERROR(hipMalloc(&buffers.dRoc, buffers.hC.size() * sizeof(OutputT)));
  CHECK_HIP_ERROR(hipMalloc(&buffers.dMismatch, sizeof(uint32_t)));

  CHECK_HIP_ERROR(hipMalloc(&buffers.dMismatchCount, sizeof(uint32_t)));
  CHECK_HIP_ERROR(hipMalloc(&buffers.dMismatchList,
                            kMaxMismatchDump * sizeof(uint32_t)));

  CHECK_HIP_ERROR(hipMemcpy(buffers.dA, buffers.hA.data(), buffers.hA.size() * sizeof(InputT), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(buffers.dB, buffers.hB.data(), buffers.hB.size() * sizeof(InputT), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(buffers.dC, buffers.hC.data(), buffers.hC.size() * sizeof(OutputT), hipMemcpyHostToDevice));
}

void free_buffers(Buffers &buffers)
{
  CHECK_HIP_ERROR(hipFree(buffers.dA));
  CHECK_HIP_ERROR(hipFree(buffers.dB));
  CHECK_HIP_ERROR(hipFree(buffers.dC));
  CHECK_HIP_ERROR(hipFree(buffers.dD));
  CHECK_HIP_ERROR(hipFree(buffers.dRef));
  CHECK_HIP_ERROR(hipFree(buffers.dRoc));
  CHECK_HIP_ERROR(hipFree(buffers.dMismatch));
  CHECK_HIP_ERROR(hipFree(buffers.dMismatchCount));
  CHECK_HIP_ERROR(hipFree(buffers.dMismatchList));
}

// ============================================================================
// Build reference (wmma grouped kernel)
// ============================================================================

template <typename P>
void build_reference(const Problem &p, Buffers &buffers)
{
  if ((p.m % P::ROCWMMA_M) || (p.n % P::ROCWMMA_N) || (p.k % (P::K_GROUP * P::ROCWMMA_K)))
  {
    std::cerr << "reference skipped: dimensions not divisible by fragment/group size" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const dim3 block(P::TBLOCK_X, P::TBLOCK_Y, 1);
  const dim3 grid(rocwmma::ceil_div(p.m, P::MACRO_TILE_M),
                  rocwmma::ceil_div(p.n, P::MACRO_TILE_N),
                  1);
  constexpr uint32_t sizeLdsOne = (P::MACRO_TILE_M + P::LDS_PAD_A + P::MACRO_TILE_N + P::LDS_PAD_B) * P::MACRO_TILE_K;
  const uint32_t sharedBytes = 2u * P::K_GROUP * sizeLdsOne * sizeof(InputT);

  CHECK_HIP_ERROR(hipMemset(buffers.dRef, 0, buffers.hC.size() * sizeof(OutputT)));
  hipLaunchKernelGGL((rocwmma_grouped_kernel<P>),
                     grid, block, sharedBytes, 0,
                     p.m, p.n, p.k,
                     buffers.dA, buffers.dB, buffers.dC, buffers.dRef,
                     p.m, p.n, p.n, p.n,
                     p.alpha, p.beta);
  CHECK_HIP_ERROR(hipGetLastError());
  CHECK_HIP_ERROR(hipDeviceSynchronize());
}

void dump_value_origin_map(const Problem& p,
                           OutputT const* dActual,
                           OutputT const* dExpected,
                           uint32_t rows = 32,
                           uint32_t cols = 64,
                           uint32_t searchRows = 64,
                           uint32_t searchCols = 64)
{
    std::vector<OutputT> hActual(static_cast<size_t>(p.m) * p.n);
    std::vector<OutputT> hExpected(static_cast<size_t>(p.m) * p.n);

    CHECK_HIP_ERROR(hipMemcpy(hActual.data(), dActual,
                              hActual.size() * sizeof(OutputT),
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hExpected.data(), dExpected,
                              hExpected.size() * sizeof(OutputT),
                              hipMemcpyDeviceToHost));

    auto find_unique = [&](OutputT v, uint32_t& hitR, uint32_t& hitC) -> int {
        int hits = 0;
        hitR = hitC = 0xffffffffu;

        for(uint32_t r = 0; r < searchRows; ++r)
        {
            for(uint32_t c = 0; c < searchCols; ++c)
            {
                if(hExpected[r * p.n + c] == v)
                {
                    ++hits;
                    hitR = r;
                    hitC = c;
                    if(hits > 1)
                        return hits;
                }
            }
        }
        return hits;
    };

    std::cout << "--- origin map actual(r,c) -> expected(r,c), first "
              << rows << "x" << cols << " searched in expected "
              << searchRows << "x" << searchCols << " ---\n";

    for(uint32_t r = 0; r < rows; ++r)
    {
        std::cout << "r" << std::setw(2) << r << " ";
        for(uint32_t c = 0; c < cols; ++c)
        {
            const OutputT v = hActual[r * p.n + c];

            uint32_t er = 0, ec = 0;
            int hits = find_unique(v, er, ec);

            if(hits == 1)
            {
                std::cout << "(" << std::setw(2) << er << "," << std::setw(2) << ec << ") ";
            }
            else if(hits == 0)
            {
                std::cout << "(--,--) ";
            }
            else
            {
                std::cout << "(multi) ";
            }

            if((c & 15u) == 15u)
                std::cout << " ";
        }
        std::cout << "\n";
    }
}

template <typename ProbeT>
void dump_acc_origin_table(const Problem& p,
                           const ProbeT& probe,
                           OutputT const* dExpected,
                           uint32_t searchRows = 64,
                           uint32_t searchCols = 64)
{
    std::vector<OutputT> hExpected(static_cast<size_t>(p.m) * p.n);
    CHECK_HIP_ERROR(hipMemcpy(hExpected.data(), dExpected,
                              hExpected.size() * sizeof(OutputT),
                              hipMemcpyDeviceToHost));

    auto find_unique = [&](int32_t v, uint32_t& hitR, uint32_t& hitC) -> int {
        int hits = 0;
        hitR = hitC = 0xffffffffu;

        for(uint32_t r = 0; r < searchRows; ++r)
        {
            for(uint32_t c = 0; c < searchCols; ++c)
            {
                if(hExpected[r * p.n + c] == v)
                {
                    ++hits;
                    hitR = r;
                    hitC = c;
                    if(hits > 1)
                        return hits;
                }
            }
        }
        return hits;
    };

    auto dump_group = [&](const char* name, const int32_t* v, uint32_t lane)
    {
        for(uint32_t e = 0; e < 8; ++e)
        {
            uint32_t r = 0, c = 0;
            int hits = find_unique(v[e], r, c);

            std::cout << "lane=" << std::setw(2) << lane
                      << " " << name << "[" << e << "]=" << std::setw(8) << v[e]
                      << " -> ";

            if(hits == 1)
                std::cout << "E(" << r << "," << c << ")";
            else if(hits == 0)
                std::cout << "NO_UNIQUE_HIT";
            else
                std::cout << "MULTI_HIT";

            std::cout << "\n";
        }
    };

    for(uint32_t warpM = 0; warpM < 2; ++warpM)
    {
        for(uint32_t warpN = 0; warpN < 2; ++warpN)
        {
            std::cout << "--- accumulator origin table warpM=" << warpM
                      << " warpN=" << warpN
                      << ", searched expected "
                      << searchRows << "x" << searchCols << " ---\n";

            for(uint32_t lane = 0; lane < 32; ++lane)
            {
                dump_group("acc00", probe.lane[warpM][warpN][lane].acc00, lane);
                dump_group("acc01", probe.lane[warpM][warpN][lane].acc01, lane);
                dump_group("acc10", probe.lane[warpM][warpN][lane].acc10, lane);
                dump_group("acc11", probe.lane[warpM][warpN][lane].acc11, lane);
            }
        }
    }
}

// ============================================================================
// Run combined (pearl) kernel
// ============================================================================

template <typename Policy>
void run_tensile_lego_kernel(const std::string &label,
                             const Problem &p,
                             Buffers &buffers,
                             OutputT const *dExpected,
                             bool captureProbe)
{
  if ((p.m % Policy::MACRO_TILE_M) || (p.n % Policy::MACRO_TILE_N) || (p.k % Policy::MACRO_TILE_K))
  {
    std::cout << label << " skipped: dimensions not divisible" << std::endl;
    return;
  }

  const dim3 block(Policy::TBLOCK_X, Policy::TBLOCK_Y, 1);
  const dim3 grid(rocwmma::ceil_div(p.m, Policy::MACRO_TILE_M),
                  rocwmma::ceil_div(p.n, Policy::MACRO_TILE_N),
                  1);

  constexpr uint32_t sharedBytes = tensile_lego::LdsPlanAilkBljk<Policy>::SharedBytes;
  int maxSharedBytes = 0;
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &maxSharedBytes, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
  if (sharedBytes > static_cast<uint32_t>(maxSharedBytes))
  {
    std::cout << label << " skipped: shared memory " << sharedBytes
              << " exceeds device limit " << maxSharedBytes << std::endl;
    return;
  }

  tensile_lego::DebugProbe *dProbe = nullptr;
  tensile_lego::DebugProbe hProbe{};
  if (captureProbe)
  {
    CHECK_HIP_ERROR(hipMalloc(&dProbe, sizeof(tensile_lego::DebugProbe)));
    CHECK_HIP_ERROR(hipMemset(dProbe, 0, sizeof(tensile_lego::DebugProbe)));
  }

  auto launch = [&]()
  {
    hipLaunchKernelGGL((tensile_lego::kernel<Policy>),
                       grid, block, sharedBytes, 0,
                       p.m, p.n, p.k,
                       buffers.dA, buffers.dB,
                       buffers.dC, buffers.dD,
                       p.m, p.k, p.n, p.n,
                       p.alpha, p.beta,
                       dProbe, captureProbe);
    CHECK_HIP_ERROR(hipGetLastError());
  };

  CHECK_HIP_ERROR(hipMemset(buffers.dD, 0, buffers.hC.size() * sizeof(OutputT)));
  for (uint32_t i = 0; i < p.warmups; ++i)
  {
    launch();
  }
  CHECK_HIP_ERROR(hipDeviceSynchronize());

  hipEvent_t startEvent{}, stopEvent{};
  CHECK_HIP_ERROR(hipEventCreate(&startEvent));
  CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
  CHECK_HIP_ERROR(hipEventRecord(startEvent));
  for (uint32_t i = 0; i < p.runs; ++i)
  {
    launch();
  }
  CHECK_HIP_ERROR(hipEventRecord(stopEvent));
  CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

  float elapsedMs = 0.0f;
  CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
  CHECK_HIP_ERROR(hipEventDestroy(startEvent));
  CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

  const double perEvalMs = static_cast<double>(elapsedMs) / p.runs;
  const double topPerSec = gops(p) / elapsedMs * static_cast<double>(p.runs);
  const double tmacPerSec = macs(p) / (perEvalMs * 1.0e-3);

  const bool ok = compare_output(p, buffers.dD, dExpected, buffers);

  if(captureProbe)
  {
      CHECK_HIP_ERROR(hipMemcpy(&hProbe,
                                dProbe,
                                sizeof(tensile_lego::DebugProbe),
                                hipMemcpyDeviceToHost));
  }

  if(!ok)
  {
      dump_tile_diff(p, buffers.dD, dExpected, 32, 64);
      dump_value_origin_map(p, buffers.dD, dExpected, 32, 64, 64, 64);
      dump_acc_origin_table(p, hProbe, dExpected, 64, 64);
  }
  std::cout << std::left
            << std::setw(34) << label
            << std::setw(6) << (ok ? "yes" : "no")
            << std::setw(13) << elapsedMs
            << std::setw(13) << perEvalMs
            << std::setw(13) << gops(p)
            << std::setw(13) << topPerSec
            << std::setw(13) << tmacPerSec
            << std::endl;

  if (captureProbe)
  {
    CHECK_HIP_ERROR(hipMemcpy(&hProbe, dProbe, sizeof(tensile_lego::DebugProbe), hipMemcpyDeviceToHost));
    std::cout << "--- Probe: " << label << " block(0,0) lane0 first K-step ---" << std::endl;
    dump_probe(hProbe);
    CHECK_HIP_ERROR(hipFree(dProbe));
  }
}

template <typename P>
void run_combined_kernel(const std::string &label,
                         const Problem &p,
                         Buffers &buffers,
                         OutputT const *dExpected)
{
  constexpr uint32_t effMacroM = P::EFF_MACRO_TILE_M;
  constexpr uint32_t effMacroN = P::EFF_MACRO_TILE_N;
  constexpr uint32_t kGroupStep = P::K_GROUP * P::MACRO_TILE_K;

  if ((p.m % P::ROCWMMA_M) || (p.n % P::ROCWMMA_N) || (p.k % kGroupStep) || (p.m % effMacroM) || (p.n % effMacroN))
  {
    std::cout << label << " skipped: dimensions not divisible" << std::endl;
    return;
  }

  const dim3 block(P::TBLOCK_X, P::TBLOCK_Y, 1);
  const dim3 grid(rocwmma::ceil_div(p.m, effMacroM),
                  rocwmma::ceil_div(p.n, effMacroN),
                  1);

  using LWFragA = typename P::LWFragA;
  using LWFragB = typename P::LWFragB;
  using LWFragAShape = rocwmma::GetIOShape_t<LWFragA>;
  using LWFragBShape = rocwmma::GetIOShape_t<LWFragB>;
  constexpr uint32_t ldsHeightA = LWFragAShape::BlockHeight;
  constexpr uint32_t ldsHeightB = P::N_SUBPASSES * LWFragBShape::BlockHeight;
  constexpr uint32_t ldsWidth = P::EFF_B_STRIDE;
  constexpr uint32_t ldsHeight = ldsHeightA + P::LDS_PAD_A + ldsHeightB + P::LDS_PAD_B;
  constexpr uint32_t sizeLdsOne = ldsHeight * ldsWidth;
  const uint32_t sharedBytes = 2u * P::K_GROUP * sizeLdsOne * sizeof(InputT);

  int maxSharedBytes = 0;
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &maxSharedBytes, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
  if (sharedBytes > static_cast<uint32_t>(maxSharedBytes))
  {
    std::cout << label << " skipped: shared memory " << sharedBytes
              << " exceeds device limit " << maxSharedBytes << std::endl;
    return;
  }

  auto launch = [&]()
  {
    hipLaunchKernelGGL((rocwmma_combined_kernel<P>),
                       grid, block, sharedBytes, 0,
                       p.m, p.n, p.k,
                       buffers.dA, buffers.dB,
                       buffers.dC, buffers.dD,
                       p.m, p.n, p.n, p.n,
                       p.alpha, p.beta);
    CHECK_HIP_ERROR(hipGetLastError());
  };

  CHECK_HIP_ERROR(hipMemset(buffers.dD, 0, buffers.hC.size() * sizeof(OutputT)));
  for (uint32_t i = 0; i < p.warmups; ++i)
  {
    launch();
  }
  CHECK_HIP_ERROR(hipDeviceSynchronize());

  hipEvent_t startEvent{}, stopEvent{};
  CHECK_HIP_ERROR(hipEventCreate(&startEvent));
  CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
  CHECK_HIP_ERROR(hipEventRecord(startEvent));
  for (uint32_t i = 0; i < p.runs; ++i)
  {
    launch();
  }
  CHECK_HIP_ERROR(hipEventRecord(stopEvent));
  CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

  float elapsedMs = 0.0f;
  CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
  CHECK_HIP_ERROR(hipEventDestroy(startEvent));
  CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

  const double perEvalMs = static_cast<double>(elapsedMs) / p.runs;
  const double topPerSec = gops(p) / elapsedMs * static_cast<double>(p.runs);
  const double tmacPerSec = macs(p) / (perEvalMs * 1.0e-3);

  const bool ok = compare_output(p, buffers.dD, dExpected, buffers);
  if(!ok)
  {
      dump_tile_diff(p, buffers.dD, dExpected, 32, 64);
  }
  std::cout << std::left
            << std::setw(34) << label
            << std::setw(6) << (ok ? "yes" : "no")
            << std::setw(13) << elapsedMs
            << std::setw(13) << perEvalMs
            << std::setw(13) << gops(p)
            << std::setw(13) << topPerSec
            << std::setw(13) << tmacPerSec
            << std::endl;
}

// ============================================================================
// Run rocBLAS
// ============================================================================

void run_rocblas(const Problem &p, Buffers &buffers)
{
  rocblas_handle handle{};
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
  CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

  auto launch = [&]()
  {
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                        rocblas_operation_none,
                                        rocblas_operation_none,
                                        p.m, p.n, p.k,
                                        &p.alpha,
                                        buffers.dA, rocblas_datatype_i8_r, p.m,
                                        buffers.dB, rocblas_datatype_i8_r, p.k,
                                        &p.beta,
                                        buffers.dC, rocblas_datatype_i32_r, p.m,
                                        buffers.dRoc, rocblas_datatype_i32_r, p.m,
                                        rocblas_datatype_i32_r,
                                        rocblas_gemm_algo_standard, 0, 0));
  };

  CHECK_HIP_ERROR(hipMemset(buffers.dRoc, 0, buffers.hC.size() * sizeof(OutputT)));
  for (uint32_t i = 0; i < p.warmups; ++i)
  {
    launch();
  }
  CHECK_HIP_ERROR(hipDeviceSynchronize());

  hipEvent_t startEvent{}, stopEvent{};
  CHECK_HIP_ERROR(hipEventCreate(&startEvent));
  CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
  CHECK_HIP_ERROR(hipEventRecord(startEvent));
  for (uint32_t i = 0; i < p.runs; ++i)
  {
    launch();
  }
  CHECK_HIP_ERROR(hipEventRecord(stopEvent));
  CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

  float elapsedMs = 0.0f;
  CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
  CHECK_HIP_ERROR(hipEventDestroy(startEvent));
  CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
  CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

  const double perEvalMs = static_cast<double>(elapsedMs) / p.runs;
  const double topPerSec = gops(p) / elapsedMs * static_cast<double>(p.runs);
  const double tmacPerSec = macs(p) / (perEvalMs * 1.0e-3);

  std::cout << std::left
            << std::setw(34) << "rocBLAS_i8"
            << std::setw(6) << "-"
            << std::setw(13) << elapsedMs
            << std::setw(13) << perEvalMs
            << std::setw(13) << gops(p)
            << std::setw(13) << topPerSec
            << std::setw(13) << tmacPerSec
            << std::endl;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char **argv)
{
  Problem p{};
  if (argc >= 4)
  {
    p.m = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
    p.n = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
    p.k = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
  }
  if (argc >= 6)
  {
    p.runs = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
    p.warmups = static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10));
  }
  if (argc >= 8)
  {
    p.alpha = static_cast<ComputeT>(std::strtol(argv[6], nullptr, 10));
    p.beta = static_cast<ComputeT>(std::strtol(argv[7], nullptr, 10));
  }
  if (argc >= 9)
  {
    p.wmmaCrossCheck = std::strtoul(argv[8], nullptr, 10) != 0;
  }
  if (const char *env = std::getenv("TNN_WMMA_CROSSCHECK"))
  {
    p.wmmaCrossCheck = std::strtoul(env, nullptr, 10) != 0;
  }
  const bool probeB = std::getenv("TNN_LEGO_DIAG_B") != nullptr;

  std::cout << "dims M=" << p.m << " N=" << p.n << " K=" << p.k
            << " runs=" << p.runs << " warmups=" << p.warmups
            << " alpha=" << p.alpha << " beta=" << p.beta
            << " wmmaCrossCheck=" << (p.wmmaCrossCheck ? 1 : 0) << std::endl;

  Buffers buffers{};
  init_buffers(p, buffers);

  std::cout << "\n--- Results (OK compares against rocBLAS_i8) ---" << std::endl;
  std::cout << std::left
            << std::setw(34) << "Kernel"
            << std::setw(6) << "OK"
            << std::setw(13) << "elapsedMs"
            << std::setw(13) << "perEvalMs"
            << std::setw(13) << "GOp"
            << std::setw(13) << "TOp/s"
            << std::setw(13) << "TMac/s"
            << std::endl;

  run_rocblas(p, buffers);
  run_tensile_lego_kernel<tensile_lego::RocblasLike64x64x32>(
      "tensile_lego_64x64x32", p, buffers, buffers.dRoc, probeB);

  if (p.wmmaCrossCheck)
  {
    std::cout << "\n--- Building wmma reference (64x64x32) ---" << std::endl;
    build_reference<Wmma64x64x32>(p, buffers);
    std::cout << "\n--- Optional Cross-check: rocBLAS / tensile_lego vs wmma reference ---" << std::endl;
    const bool rocMatchesRef = compare_output(p, buffers.dRoc, buffers.dRef, buffers);
    std::cout << "rocBLAS matches wmma reference: " << (rocMatchesRef ? "PASS" : "FAIL") << std::endl;
    const bool legoMatchesRef = compare_output(p, buffers.dD, buffers.dRef, buffers);
    std::cout << "tensile_lego matches wmma reference: " << (legoMatchesRef ? "PASS" : "FAIL") << std::endl;
  }

  free_buffers(buffers);
  return 0;
}
