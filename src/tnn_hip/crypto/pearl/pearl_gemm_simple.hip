// ---------------------------------------------------------------------------
// rocWMMA cooperative i8 GEMM kernel (TNN-adapted from AMD perf_i8gemm pattern)
//
// Matches TNN's host data layout:
//   A : (M, K) row-major  → DataLayoutA = row_major (lda = k)
//   B : (N, K) row-major  → rocWMMA matrix_b sees (K, N) → DataLayoutB = col_major (ldb = k)
//   D : (M, N) row-major  → DataLayoutD = row_major (ldd = n)
//
// D = A * B  (no alpha/beta/C — pure GEMM matching compute_host_gemm_ref)
//
// Grid: gridDim(ceil_div(m, macroTileM), ceil_div(n, macroTileN))
//       blockIdx.x = M-index, blockIdx.y = N-index
// Block: (TBLOCK_X, TBLOCK_Y)
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include "rocwmma.hpp"
#include "rocwmma_transforms.hpp"
#include <cstdint>

// ROCWMMA_KERNEL: extern "C" + amdgpu workgroup size attributes
#ifndef ROCWMMA_KERNEL
#define ROCWMMA_KERNEL extern "C" __global__ __launch_bounds__(256)
#endif

// ---- gfx11 params ----
namespace gfx11Params {
    using namespace rocwmma;

    using InputT   = int8_t;
    using ComputeT = int32_t;
    using OutputT  = int32_t;

    // Data layouts matching TNN host layout:
    //   A: (M,K) row-major  → row_major
    //   B: (N,K) row-major  → col_major  (rocWMMA matrix_b is (K,N))
    //   D: (M,N) row-major  → row_major
    using DataLayoutA   = row_major;
    using DataLayoutB   = col_major;
    using DataLayoutD   = row_major;
    using DataLayoutLds = col_major;

    constexpr uint32_t ROCWMMA_M = 16u;
    constexpr uint32_t ROCWMMA_N = 16u;
    constexpr uint32_t ROCWMMA_K = 16u;
    constexpr uint32_t BLOCKS_M  = 4u;
    constexpr uint32_t BLOCKS_N  = 2u;
    constexpr uint32_t TBLOCK_X  = 64u;
    constexpr uint32_t TBLOCK_Y  = 2u;
    constexpr uint32_t WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32;

    constexpr uint32_t WARP_TILE_M = BLOCKS_M * ROCWMMA_M;
    constexpr uint32_t WARP_TILE_N = BLOCKS_N * ROCWMMA_N;
    constexpr uint32_t WARP_TILE_K = ROCWMMA_K;

    constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE;
    constexpr uint32_t WARPS_N      = TBLOCK_Y;
    constexpr uint32_t WARP_COUNT   = WARPS_M * WARPS_N;
    constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M;
    constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N;
    constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;

    // --- fragment types ---
    using MmaFragA   = fragment<matrix_a,    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT,  DataLayoutA>;
    using MmaFragB   = fragment<matrix_b,    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT,  DataLayoutB>;
    using MmaFragAcc = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT>;
    using MmaFragD   = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, OutputT, DataLayoutD>;

    using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

    using GRFragA = fragment<matrix_a, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                             InputT, DataLayoutA, CoopScheduler>;
    using GRFragB = fragment<matrix_b, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                             InputT, DataLayoutB, CoopScheduler>;

    using LWFragA = apply_data_layout_t<GRFragA, DataLayoutLds>;
    using LWFragB = apply_data_layout_t<apply_transpose_t<GRFragB>, DataLayoutLds>;

    using LRFragA = apply_data_layout_t<MmaFragA, DataLayoutLds>;
    using LRFragB = apply_data_layout_t<apply_transpose_t<MmaFragB>, DataLayoutLds>;

    // --- transform helpers ---
    ROCWMMA_DEVICE auto transformGRFragAToLWFragA(GRFragA const& f) {
        return apply_data_layout<DataLayoutLds>(f);
    }
    ROCWMMA_DEVICE auto transformGRFragBToLWFragB(GRFragB const& f) {
        return apply_data_layout<DataLayoutLds>(apply_transpose(f));
    }
    ROCWMMA_DEVICE auto transformLRFragAToMmaFragA(LRFragA const& f) {
        return apply_data_layout<DataLayoutA>(f);
    }
    ROCWMMA_DEVICE auto transformLRFragBToMmaFragB(LRFragB const& f) {
        return apply_data_layout<DataLayoutB>(apply_transpose(f));
    }
} // namespace gfx11Params

// ---- gfx9 params ----
#if ROCWMMA_ARCH_GFX9
namespace gfx9Params {
    using namespace rocwmma;

    using InputT   = int8_t;
    using ComputeT = int32_t;
    using OutputT  = int32_t;

    // Data layouts match TNN host layout (same as gfx11)
    using DataLayoutA   = row_major;
    using DataLayoutB   = col_major;
    using DataLayoutD   = row_major;
    using DataLayoutLds = col_major;

    constexpr uint32_t ROCWMMA_M = 32u;
    constexpr uint32_t ROCWMMA_N = 32u;
    constexpr uint32_t ROCWMMA_K = 16u;
    constexpr uint32_t BLOCKS_M  = 2u;
    constexpr uint32_t BLOCKS_N  = 2u;
    constexpr uint32_t TBLOCK_X  = 128u;
    constexpr uint32_t TBLOCK_Y  = 2u;
    constexpr uint32_t WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64;

    constexpr uint32_t WARP_TILE_M = BLOCKS_M * ROCWMMA_M; // 64
    constexpr uint32_t WARP_TILE_N = BLOCKS_N * ROCWMMA_N; // 64
    constexpr uint32_t WARP_TILE_K = ROCWMMA_K;

    constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE; // 2
    constexpr uint32_t WARPS_N      = TBLOCK_Y;             // 2
    constexpr uint32_t WARP_COUNT   = WARPS_M * WARPS_N;
    constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M; // 128
    constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N; // 128
    constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;

    // --- fragment types ---
    using MmaFragA   = fragment<matrix_a,    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT,  DataLayoutA>;
    using MmaFragB   = fragment<matrix_b,    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT,  DataLayoutB>;
    using MmaFragAcc = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT>;
    using MmaFragD   = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, OutputT, DataLayoutD>;

    using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

    using GRFragA = fragment<matrix_a, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                             InputT, DataLayoutA, CoopScheduler>;
    using GRFragB = fragment<matrix_b, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                             InputT, DataLayoutB, CoopScheduler>;

    using LWFragA = apply_data_layout_t<GRFragA, DataLayoutLds>;
    using LWFragB = apply_data_layout_t<apply_transpose_t<GRFragB>, DataLayoutLds>;

    using LRFragA = apply_data_layout_t<MmaFragA, DataLayoutLds>;
    using LRFragB = apply_data_layout_t<apply_transpose_t<MmaFragB>, DataLayoutLds>;

    // --- transform helpers ---
    ROCWMMA_DEVICE auto transformGRFragAToLWFragA(GRFragA const& f) {
        return apply_data_layout<DataLayoutLds>(f);
    }
    ROCWMMA_DEVICE auto transformGRFragBToLWFragB(GRFragB const& f) {
        return apply_data_layout<DataLayoutLds>(apply_transpose(f));
    }
    ROCWMMA_DEVICE auto transformLRFragAToMmaFragA(LRFragA const& f) {
        return apply_data_layout<DataLayoutA>(f);
    }
    ROCWMMA_DEVICE auto transformLRFragBToMmaFragB(LRFragB const& f) {
        return apply_data_layout<DataLayoutB>(apply_transpose(f));
    }
}
#endif

namespace rocwmma_trial_params {
#if ROCWMMA_ARCH_GFX9
    using namespace gfx9Params;
#else
    using namespace gfx11Params;
#endif
} // namespace rocwmma_trial_params

// ---- Type aliases for kernel signature ----
using InputT   = int8_t;
using ComputeT = int32_t;
using OutputT  = int32_t;

// ---------------------------------------------------------------------------
// Kernel: D = A * B
//
// Host data layout:
//   A[ i * k + l ]  = row i, column l  of (M×K) row-major → lda = k
//   B[ j * k + l ]  = row j, column l  of (N×K) row-major → ldb = k
//   D[ i * n + j ]  = row i, column j  of (M×N) row-major → ldd = n
//
// Launched as:
//   gridDim(ceil_div(m, MACRO_TILE_M), ceil_div(n, MACRO_TILE_N))
//   blockDim(TBLOCK_X, TBLOCK_Y)
//   sharedMem = 2 * sizeof(InputT) * (LWFragAShape::BlockHeight + LWFragBShape::BlockHeight) * MACRO_TILE_K
// ---------------------------------------------------------------------------
extern "C" ROCWMMA_KERNEL void pearl_gemm_simple(
    uint32_t      m,
    uint32_t      n,
    uint32_t      k,
    InputT const* a,   // (M, K) row-major
    InputT const* b,   // (N, K) row-major  — rocWMMA sees as (K, N) col-major
    OutputT*      d,   // (M, N) row-major
    uint32_t      lda,
    uint32_t      ldb,
    uint32_t      ldd)
{
    using namespace rocwmma_trial_params;

    constexpr auto warpTileSize  = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto macroTileSize = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);

    auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto localWarpOffset = localWarpCoord * warpTileSize;

    auto macroTileCoord = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
    auto warpTileCoord  = macroTileCoord + localWarpOffset;

    if (get<0>(warpTileCoord + warpTileSize) > m ||
        get<1>(warpTileCoord + warpTileSize) > n)
        return;

    // ---- global read address setup ----
    using GRFragAMap1d = GetDataLayout_t<GRFragA>;
    using GRFragBMap1d = GetDataLayout_t<GRFragB>;

    auto globalReadOffsetA =
        GRFragAMap1d::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
    // B is (N,K) row-major → col_major from rocWMMA matrix_b (K,N) perspective
    // Start at column get<1>(macroTileCoord) in N, row 0 in K
    auto globalReadOffsetB =
        GRFragBMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);

    auto kStepOffsetA = GRFragAMap1d::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStepOffsetB = GRFragBMap1d::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

    // ---- LDS setup ----
    HIP_DYNAMIC_SHARED(void*, localMemPtr);

    using LWFragAShape = GetIOShape_t<LWFragA>;
    using LWFragBShape = GetIOShape_t<LWFragB>;
    using LWFragAMap1d = GetDataLayout_t<LWFragA>;
    using LWFragBMap1d = GetDataLayout_t<LWFragB>;

    constexpr uint32_t ldsWidth  = MACRO_TILE_K;
    constexpr uint32_t ldsHeight = LWFragAShape::BlockHeight +
                                   LWFragBShape::BlockHeight;
    constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
    constexpr uint32_t ldsld     = is_same_v<DataLayoutLds, row_major>
                                       ? ldsWidth : ldsHeight;

    auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
    auto* ldsPtrHi = ldsPtrLo + sizeLds;

    // LDS write offsets
    auto ldsWriteOffsetA = 0u;
    auto ldsWriteOffsetB = LWFragBMap1d::fromMatrixCoord(
        make_coord2d(LWFragAShape::BlockHeight, 0u), ldsld);
        // NOTE: B's offset uses LWFragBMap1d because we're computing the
        // 1D offset for the B fragment's position in LDS. The offset
        // is measured from the start of B's territory (after A).
        // The coordinate (LWFragAShape::BlockHeight, 0) is in the
        // unified LDS space, at the boundary between A and B data.

    // LDS read offsets for this warp's A / B
    // Each warp reads its own A and B slices from LDS
    auto ldsReadOffsetA = ldsWriteOffsetA + LWFragAMap1d::fromMatrixCoord(
        make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
    auto ldsReadOffsetB = ldsWriteOffsetB + LWFragBMap1d::fromMatrixCoord(
        make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

    // ---- pre-fetch K=0 ----
    GRFragA grFragA;
    GRFragB grFragB;
    load_matrix_sync(grFragA, a + globalReadOffsetA, lda);
    load_matrix_sync(grFragB, b + globalReadOffsetB, ldb);
    globalReadOffsetA += kStepOffsetA;
    globalReadOffsetB += kStepOffsetB;

    store_matrix_sync(ldsPtrLo + ldsWriteOffsetA,
                      transformGRFragAToLWFragA(grFragA), ldsld);
    store_matrix_sync(ldsPtrLo + ldsWriteOffsetB,
                      transformGRFragBToLWFragB(grFragB), ldsld);

    MmaFragAcc mmaFragAcc;
    fill_fragment(mmaFragAcc, static_cast<ComputeT>(0));

    synchronize_workgroup();

    // ---- main K loop with double-buffered prefetch ----
    for (uint32_t currentK = MACRO_TILE_K; currentK < k; currentK += MACRO_TILE_K)
    {
        // Read from LDS buffer 0
        LRFragA lrFragA;
        LRFragB lrFragB;
        load_matrix_sync(lrFragA, ldsPtrLo + ldsReadOffsetA, ldsld);
        load_matrix_sync(lrFragB, ldsPtrLo + ldsReadOffsetB, ldsld);

        // Prefetch next global tiles
        load_matrix_sync(grFragA, a + globalReadOffsetA, lda);
        load_matrix_sync(grFragB, b + globalReadOffsetB, ldb);
        globalReadOffsetA += kStepOffsetA;
        globalReadOffsetB += kStepOffsetB;

        // Accumulate A * B
        mma_sync(mmaFragAcc,
                 transformLRFragAToMmaFragA(lrFragA),
                 transformLRFragBToMmaFragB(lrFragB),
                 mmaFragAcc);

        // Write prefetch to LDS buffer 1
        store_matrix_sync(ldsPtrHi + ldsWriteOffsetA,
                          transformGRFragAToLWFragA(grFragA), ldsld);
        store_matrix_sync(ldsPtrHi + ldsWriteOffsetB,
                          transformGRFragBToLWFragB(grFragB), ldsld);

        synchronize_workgroup();

        auto* tmp  = ldsPtrLo;
        ldsPtrLo   = ldsPtrHi;
        ldsPtrHi   = tmp;
    }

    // ---- tail K step ----
    LRFragA lrFragA;
    LRFragB lrFragB;
    load_matrix_sync(lrFragA, ldsPtrLo + ldsReadOffsetA, ldsld);
    load_matrix_sync(lrFragB, ldsPtrLo + ldsReadOffsetB, ldsld);
    mma_sync(mmaFragAcc,
             transformLRFragAToMmaFragA(lrFragA),
             transformLRFragBToMmaFragB(lrFragB),
             mmaFragAcc);

    // ---- write D = accum (pure GEMM, no alpha/beta/C) ----
    using MmaFragDMap1d = GetDataLayout_t<MmaFragD>;
    MmaFragD mmaFragD;

    #pragma unroll
    for (int i = 0; i < mmaFragD.num_elements; ++i)
        mmaFragD.x[i] = static_cast<OutputT>(mmaFragAcc.x[i]);

    store_matrix_sync(d + MmaFragDMap1d::fromMatrixCoord(warpTileCoord, ldd),
                      mmaFragD, ldd);
}
