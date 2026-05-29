// ---------------------------------------------------------------------------
// rocWMMA cooperative i8 GEMM kernel (verbatim from AMD perf_i8gemm sample)
// gfx11: ROCWMMA_M=16 ROCWMMA_N=16 ROCWMMA_K=16 BLOCKS_M=4 BLOCKS_N=2
//        TBLOCK_X=64 TBLOCK_Y=2  -->  WARP_TILE=64x32  MACRO_TILE=128x64
//
// D = alpha * A * B + beta * C
//
// Grid dims: gridDim(ceil_div(m, macroTileM), ceil_div(n, macroTileN))
//            blockIdx.x = M-index, blockIdx.y = N-index
// Block dims: (TBLOCK_X, TBLOCK_Y)
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include "rocwmma.hpp"
#include "rocwmma_transforms.hpp"
#include <cstdint>

// ROCWMMA_KERNEL: adds extern "C" + amdgpu workgroup size attributes
// (Matches the perf_i8gemm sample macro pattern)
#ifndef ROCWMMA_KERNEL
#define ROCWMMA_KERNEL extern "C" __global__ __launch_bounds__(256)
#endif

// ---- gfx11 params ----
namespace gfx11Params {
    using namespace rocwmma;

    // Type aliases matching perf_i8gemm sample
    using InputT    = int8_t;
    using ComputeT  = int32_t;
    using OutputT   = int32_t;

    // Data layouts matching the perf_i8gemm sample
    using DataLayoutA   = col_major;
    using DataLayoutB   = row_major;
    using DataLayoutC   = row_major;
    using DataLayoutD   = row_major;
    using DataLayoutLds = col_major;

    // --- tile sizes ---
    constexpr uint32_t ROCWMMA_M = 16u;
    constexpr uint32_t ROCWMMA_N = 16u;
    constexpr uint32_t ROCWMMA_K = 16u;
    constexpr uint32_t BLOCKS_M  = 4u;
    constexpr uint32_t BLOCKS_N  = 2u;
    constexpr uint32_t TBLOCK_X  = 64u;
    constexpr uint32_t TBLOCK_Y  = 2u;
    constexpr uint32_t WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32;

    // Warp tile
    constexpr uint32_t WARP_TILE_M = BLOCKS_M * ROCWMMA_M;
    constexpr uint32_t WARP_TILE_N = BLOCKS_N * ROCWMMA_N;
    constexpr uint32_t WARP_TILE_K = ROCWMMA_K;

    // Macro tile
    constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE;
    constexpr uint32_t WARPS_N      = TBLOCK_Y;
    constexpr uint32_t WARP_COUNT   = WARPS_M * WARPS_N;
    constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M;
    constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N;
    constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;

    // --- fragment types ---
    // MMA fragments (per warp)
    using MmaFragA   = fragment<matrix_a,   WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutA>;
    using MmaFragB   = fragment<matrix_b,   WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutB>;
    using MmaFragC   = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, OutputT, DataLayoutC>;
    using MmaFragAcc = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT>;
    using MmaFragD   = MmaFragC;

    // Global-read fragments (cooperative, whole macro tile)
    using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;
    using GRFragA       = fragment<matrix_a, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                                   InputT, DataLayoutA, CoopScheduler>;
    using GRFragB       = fragment<matrix_b, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                                   InputT, DataLayoutB, CoopScheduler>;

    // Local-write fragments (after transform to LDS layout)
    using LWFragA = apply_data_layout_t<GRFragA, DataLayoutLds>;
    using LWFragB = apply_data_layout_t<apply_transpose_t<GRFragB>, DataLayoutLds>;

    // Local-read fragments (back to MMA layout)
    using LRFragA = apply_data_layout_t<MmaFragA, DataLayoutLds>;
    using LRFragB = apply_data_layout_t<apply_transpose_t<MmaFragB>, DataLayoutLds>;

    // --- transform helpers (matching perf_i8gemm naming) ---
    ROCWMMA_DEVICE auto transformGRFragAToLWFragA(GRFragA const& g) {
        return apply_data_layout<DataLayoutLds>(g);
    }
    ROCWMMA_DEVICE auto transformGRFragBToLWFragB(GRFragB const& g) {
        return apply_data_layout<DataLayoutLds>(apply_transpose(g));
    }
    ROCWMMA_DEVICE auto transformLRFragAToMmaFragA(LRFragA const& l) {
        return apply_data_layout<DataLayoutA>(l);
    }
    ROCWMMA_DEVICE auto transformLRFragBToMmaFragB(LRFragB const& l) {
        return apply_data_layout<DataLayoutB>(apply_transpose(l));
    }
} // namespace gfx11Params

// ---- gfx9 params (placeholder; same pattern with different tile sizes) ----
#if(ROCWMMA_ARCH_GFX9)
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

// ---- Runtime param selection ----
namespace rocwmma_trial_params {
#if(ROCWMMA_ARCH_GFX9)
    using namespace gfx9Params;
#else
    using namespace gfx11Params;
#endif
} // namespace rocwmma_trial_params

// Make type aliases available at file scope for the kernel signature
using InputT   = int8_t;
using ComputeT = int32_t;
using OutputT  = int32_t;

// ---------------------------------------------------------------------------
// Kernel: D = alpha * A * B + beta * C
//
// Grid dims: gridDim(ceil_div(m, macroTileM), ceil_div(n, macroTileN))
//            blockIdx.x = M-index (rows), blockIdx.y = N-index (cols)
// Block dims: (TBLOCK_X, TBLOCK_Y)
//
// Parameters match perf_i8gemm convention:
//   m, n, k   = matrix dimensions (uint32_t)
//   a, b      = input matrices (InputT = int8_t, no __restrict__)
//   c         = input C matrix (OutputT = int32_t, no __restrict__)
//   d         = output D matrix (OutputT = int32_t, no __restrict__)
//   lda, ldb, ldc, ldd = leading dimensions (uint32_t)
//   alpha, beta = scaling factors (ComputeT = int32_t)
// ---------------------------------------------------------------------------
extern "C" void pearl_rocwmma_gemm(
    uint32_t m, uint32_t n, uint32_t k,
    InputT const* a,
    InputT const* b,
    OutputT const* c,
    OutputT* d,
    uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t ldd,
    ComputeT alpha, ComputeT beta)
{
    using namespace rocwmma_trial_params;

    constexpr auto warpTileSize  = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto macroTileSize = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);
    constexpr auto warpDims      = make_coord2d(WARPS_M, WARPS_N);

    auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto localWarpOffset = localWarpCoord * warpTileSize;

    // blockIdx.x = M-index (rows), blockIdx.y = N-index (cols)
    auto macroTileCoord = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
    auto warpTileCoord  = macroTileCoord + localWarpOffset;

    // Bounds check at warp-tile granularity
    auto warpTileBound = warpTileCoord + warpTileSize;
    if (get<0>(warpTileBound) > m || get<1>(warpTileBound) > n)
        return;

    // 1D global read coordinate transforms
    using GRFragAMap1d = GetDataLayout_t<GRFragA>;
    using GRFragBMap1d = GetDataLayout_t<GRFragB>;

    // Initial global read addresses
    auto globalReadOffsetA = GRFragAMap1d::fromMatrixCoord(
        make_coord2d(get<0>(macroTileCoord), 0u), lda);
    auto globalReadOffsetB = GRFragBMap1d::fromMatrixCoord(
        make_coord2d(0u, get<1>(macroTileCoord)), ldb);

    // K-step increments
    auto kStepOffsetA = GRFragAMap1d::fromMatrixCoord(
        make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStepOffsetB = GRFragBMap1d::fromMatrixCoord(
        make_coord2d(MACRO_TILE_K, 0u), ldb);

    // Pre-fetch first global tiles
    GRFragA grFragA;
    GRFragB grFragB;
    load_matrix_sync(grFragA, a + globalReadOffsetA, lda);
    load_matrix_sync(grFragB, b + globalReadOffsetB, ldb);
    globalReadOffsetA += kStepOffsetA;
    globalReadOffsetB += kStepOffsetB;

    // LDS double buffer
    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    using LWFragAShape = GetIOShape_t<LWFragA>;
    using LWFragBShape = GetIOShape_t<LWFragB>;
    using LWFragAMap1d = GetDataLayout_t<LWFragA>;
    using LWFragBMap1d = GetDataLayout_t<LWFragB>;

    constexpr uint32_t ldsWidth  = MACRO_TILE_K;
    // ldsHeight computed from actual fragment shapes (not hardcoded)
    constexpr uint32_t ldsHeight = LWFragAShape::BlockHeight +
                                   LWFragBShape::BlockHeight;
    constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
    constexpr uint32_t ldsld     = is_same_v<DataLayoutLds, row_major>
                                       ? ldsWidth : ldsHeight;

    auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
    auto* ldsPtrHi = ldsPtrLo + sizeLds;

    // LDS write offsets for A / B
    // B data goes below A in the unified LDS block,
    // offset by LWFragAShape::BlockHeight rows.
    auto ldsWriteOffsetA = 0u;
    auto ldsWriteOffsetB = LWFragBMap1d::fromMatrixCoord(
        make_coord2d(LWFragAShape::BlockHeight, 0u), ldsld);

    // LDS read offsets for this warp's A / B
    // A data read uses LWFragAMap1d (matches A's data layout)
    // B data read uses LWFragBMap1d (matches B's data layout)
    auto ldsReadOffsetA = ldsWriteOffsetA + LWFragAMap1d::fromMatrixCoord(
        make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
    auto ldsReadOffsetB = ldsWriteOffsetB + LWFragBMap1d::fromMatrixCoord(
        make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

    // Store first prefetch to LDS buffer 0
    store_matrix_sync(ldsPtrLo + ldsWriteOffsetA,
                      transformGRFragAToLWFragA(grFragA), ldsld);
    store_matrix_sync(ldsPtrLo + ldsWriteOffsetB,
                      transformGRFragBToLWFragB(grFragB), ldsld);

    // Initialize accumulator
    MmaFragAcc mmaFragAcc;
    fill_fragment(mmaFragAcc, 0);

    synchronize_workgroup();

    // Main K loop with double-buffered prefetch
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

        // Swap buffers
        auto* tmp   = ldsPtrLo;
        ldsPtrLo    = ldsPtrHi;
        ldsPtrHi    = tmp;
    }

    // Tail: read last LDS buffer and accumulate
    LRFragA lrFragA;
    LRFragB lrFragB;
    load_matrix_sync(lrFragA, ldsPtrLo + ldsReadOffsetA, ldsld);
    load_matrix_sync(lrFragB, ldsPtrLo + ldsReadOffsetB, ldsld);
    mma_sync(mmaFragAcc,
             transformLRFragAToMmaFragA(lrFragA),
             transformLRFragBToMmaFragB(lrFragB),
             mmaFragAcc);

    // ---- D = alpha * A * B + beta * C ----
    // Load C matrix for this warp's tile
    using MmaFragCMap1d = GetDataLayout_t<MmaFragC>;
    auto cOffset = MmaFragCMap1d::fromMatrixCoord(warpTileCoord, ldc);
    MmaFragC cFrag;
    load_matrix_sync(cFrag, c + cOffset, ldc);

    // Compute D = alpha * A * B + beta * C with unrolled FMA
    MmaFragD mmaFragD;
    #pragma unroll
    for (int i = 0; i < mmaFragAcc.num_elements; ++i) {
        mmaFragD.x[i] = alpha * mmaFragAcc.x[i] + beta * cFrag.x[i];
    }

    // Store result to D
    using MmaFragDMap1d = GetDataLayout_t<MmaFragD>;
    store_matrix_sync(d + MmaFragDMap1d::fromMatrixCoord(warpTileCoord, ldd),
                      mmaFragD, ldd);
}
