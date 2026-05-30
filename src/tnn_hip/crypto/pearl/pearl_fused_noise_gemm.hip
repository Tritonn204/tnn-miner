// ---------------------------------------------------------------------------
// pearl_fused_noise_gemm.hip
// Phase 1: Fused BLAKE3 noise generation + rocWMMA cooperative i8 GEMM
//
// Generates EAL (128xR) and EBR (64xR) dense noise matrices on-the-fly
// via keyed BLAKE3 into LDS, then composes noised A/B tiles per K-iteration
// directly into the rocWMMA pipeline.
//
// Grid:  (ceil_div(m, 128), ceil_div(n, 64), 1)
// Block: (64, 2, 1) = 128 threads/wg
// LDS:   12288 bytes = EAL(4096) + EBR(2048) + GEMM double-buffer(6144)
//
// Parameters:
//   m, n, k, R     – matrix dimensions (R = rank = 32)
//   key_a, key_b   – 32B BLAKE3 keys for A/B-side noise
//   seed_a, seed_b – 32B seed labels for A/B (e.g. "A_tensor\0...")
//   s_a_seed       – 8 u32 for deterministic signal A
//   s_b_seed       – 8 u32 for deterministic signal B
//   ear_first/ear_second – k values: sparse column index pairs for A noise
//   ebl_first/ebl_second – k values: sparse column index pairs for B noise
//   d              – output C: MxN int32 row-major
//   ldd            – leading dimension of d (= n)
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include "rocwmma.hpp"
#include "rocwmma_transforms.hpp"
#include <cstdint>

#ifndef ROCWMMA_KERNEL
#define ROCWMMA_KERNEL extern "C" __global__ __launch_bounds__(256)
#endif

#ifndef KEYED_HASH
#define KEYED_HASH (1 << 4)
#endif

#include "blake3-inline.hip.inc"

// ---- gfx11 params (verbatim from pearl_gemm_simple.hip) ----
namespace gfx11Params {
    using namespace rocwmma;

    using InputT   = int8_t;
    using ComputeT = int32_t;
    using OutputT  = int32_t;

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

// ---- gfx9 params (verbatim from pearl_gemm_simple.hip) ----
#if ROCWMMA_ARCH_GFX9
namespace gfx9Params {
    using namespace rocwmma;

    using InputT   = int8_t;
    using ComputeT = int32_t;
    using OutputT  = int32_t;

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

// ---- Device helpers ----

__device__ __forceinline__ int8_t clampi8(int32_t v)
{
    return v;
    // return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
}

template <typename AccFragT>
__device__ __forceinline__ void jackpot_xor_checkpoint(
    const AccFragT& tileAcc,
    uint32_t*       jackpot_partial,
    uint32_t        tile_idx,
    uint32_t        linear_tid)
{
    using namespace rocwmma_trial_params;

    uint32_t thread_xor = 0;

    #pragma unroll
    for (int i = 0; i < tileAcc.num_elements; ++i)
        thread_xor ^= static_cast<uint32_t>(tileAcc.x[i]);

    uint32_t warp_xor = thread_xor;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        warp_xor ^= __shfl_xor(warp_xor, offset, WARP_SIZE);

    const uint32_t lane_id = linear_tid & (WARP_SIZE - 1);

    if (lane_id == 0)
        atomicXor(&jackpot_partial[tile_idx], warp_xor);
}

// ---------------------------------------------------------------------------
// Kernel: C = fused_noise_gemm(m, n, k, R, ...)
// ---------------------------------------------------------------------------
extern "C" ROCWMMA_KERNEL void pearl_fused_noise_gemm(
    uint32_t      m,
    uint32_t      n,
    uint32_t      k,
    uint32_t      R,
    uint8_t const* key_a,
    uint8_t const* key_b,
    uint8_t const* seed_a,
    uint8_t const* seed_b,
    uint32_t const* s_a_seed,
    uint32_t const* s_b_seed,
    int32_t const* ear_first,
    int32_t const* ear_second,
    int32_t const* ebl_first,
    int32_t const* ebl_second,
    int32_t*       d,
    uint32_t       ldd,
    uint32_t*      jackpot_partial)
{
    using namespace rocwmma_trial_params;

    const uint32_t tid = threadIdx.y * TBLOCK_X + threadIdx.x;

    const uint32_t m_base = blockIdx.x * MACRO_TILE_M;
    const uint32_t n_base = blockIdx.y * MACRO_TILE_N;

    // ---- LDS layout ----
    // eal_lds: [0        .. 4095]   128*32 int8
    // ebr_lds: [4096     .. 6143]    64*32 int8
    // gemm_lds:[6144     .. 12287]   2 * 192*16 int8
    HIP_DYNAMIC_SHARED(int8_t, lds);

    int8_t* eal_lds  = lds;
    int8_t* ebr_lds  = lds + MACRO_TILE_M * R;
    int8_t* gemm_lds = ebr_lds + MACRO_TILE_N * R;

    // =====================================================================
    // STEP A: Generate EAL dense noise into LDS
    // =====================================================================
    {
        const uint32_t thread_coord = m_base + tid + 1;

        uint32_t key_u32[8];
        uint32_t seed_u32[8];

        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            key_u32[i]  = blake3_load32(key_a  + i * 4);
            seed_u32[i] = blake3_load32(seed_a + i * 4);
        }

        uint8_t block[64];

        #pragma unroll
        for (int i = 0; i < 32; ++i)
            block[i] = 0;

        block[0] = (uint8_t)(thread_coord & 0xff);
        block[1] = (uint8_t)((thread_coord >> 8) & 0xff);
        block[2] = (uint8_t)((thread_coord >> 16) & 0xff);
        block[3] = (uint8_t)((thread_coord >> 24) & 0xff);

        #pragma unroll
        for (int i = 0; i < 32; ++i)
            block[32 + i] = (uint8_t)((seed_u32[i / 4] >> (8 * (i % 4))) & 0xff);

        uint32_t cv[8];

        #pragma unroll
        for (int i = 0; i < 8; ++i)
            cv[i] = key_u32[i];

        const uint8_t flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
        blake3_compress_in_place(cv, block, 64, 0, flags);

        #pragma unroll
        for (int j = 0; j < 32; ++j)
        {
            int8_t raw = ((int8_t*)cv)[j];
            eal_lds[tid * R + j] = (int8_t)(((raw + 128) % 64) - 32);
        }
    }

    synchronize_workgroup();

    // =====================================================================
    // STEP B: Generate EBR dense noise into LDS
    // =====================================================================
    {
        if (tid < MACRO_TILE_N)
        {
            const uint32_t thread_coord = n_base + tid + 1;

            uint32_t key_u32[8];
            uint32_t seed_u32[8];

            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                key_u32[i]  = blake3_load32(key_b  + i * 4);
                seed_u32[i] = blake3_load32(seed_b + i * 4);
            }

            uint8_t block[64];

            #pragma unroll
            for (int i = 0; i < 32; ++i)
                block[i] = 0;

            block[0] = (uint8_t)(thread_coord & 0xff);
            block[1] = (uint8_t)((thread_coord >> 8) & 0xff);
            block[2] = (uint8_t)((thread_coord >> 16) & 0xff);
            block[3] = (uint8_t)((thread_coord >> 24) & 0xff);

            #pragma unroll
            for (int i = 0; i < 32; ++i)
                block[32 + i] = (uint8_t)((seed_u32[i / 4] >> (8 * (i % 4))) & 0xff);

            uint32_t cv[8];

            #pragma unroll
            for (int i = 0; i < 8; ++i)
                cv[i] = key_u32[i];

            const uint8_t flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
            blake3_compress_in_place(cv, block, 64, 0, flags);

            #pragma unroll
            for (int j = 0; j < 32; ++j)
            {
                int8_t raw = ((int8_t*)cv)[j];
                ebr_lds[tid * R + j] = (int8_t)(((raw + 128) % 64) - 32);
            }
        }
    }

    synchronize_workgroup();

    // =====================================================================
    // STEP C: Load deterministic s_a_seed, s_b_seed into registers
    // =====================================================================
    uint32_t sa_seed[8];
    uint32_t sb_seed[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        sa_seed[i] = s_a_seed[i];
        sb_seed[i] = s_b_seed[i];
    }

    // =====================================================================
    // STEP D: rocWMMA GEMM with on-the-fly noise composition
    // =====================================================================

    constexpr auto warpTileSize  = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto macroTileSize = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);

    auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto localWarpOffset = localWarpCoord * warpTileSize;

    auto macroTileCoord = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
    auto warpTileCoord  = macroTileCoord + localWarpOffset;

    if (get<0>(warpTileCoord + warpTileSize) > m ||
        get<1>(warpTileCoord + warpTileSize) > n)
        return;

    using LWFragAShape = GetIOShape_t<LWFragA>;
    using LWFragBShape = GetIOShape_t<LWFragB>;
    using LWFragAMap1d = GetDataLayout_t<LWFragA>;
    using LWFragBMap1d = GetDataLayout_t<LWFragB>;

    constexpr uint32_t ldsWidthGEMM = MACRO_TILE_K;
    constexpr uint32_t ldsHeightA   = LWFragAShape::BlockHeight;
    constexpr uint32_t ldsHeightB   = LWFragBShape::BlockHeight;
    constexpr uint32_t ldsHtGEMM    = ldsHeightA + ldsHeightB;

    constexpr uint32_t ldsldGEMM =
        is_same_v<DataLayoutLds, row_major> ? ldsWidthGEMM : ldsHtGEMM;

    constexpr uint32_t sizeLdsGEMM = ldsHtGEMM * ldsWidthGEMM;

    constexpr auto ldsWriteOffsetA = 0u;

    constexpr auto ldsWriteOffsetB = LWFragBMap1d::fromMatrixCoord(
        make_coord2d(LWFragAShape::BlockHeight, 0u), ldsldGEMM);

    auto ldsReadOffsetA = ldsWriteOffsetA + LWFragAMap1d::fromMatrixCoord(
        make_coord2d(get<0>(localWarpOffset), 0u), ldsldGEMM);

    auto ldsReadOffsetB = ldsWriteOffsetB + LWFragBMap1d::fromMatrixCoord(
        make_coord2d(get<1>(localWarpOffset), 0u), ldsldGEMM);

    int8_t* buf0 = gemm_lds;
    int8_t* buf1 = gemm_lds + sizeLdsGEMM;

    // ---- Pre-fill first K tile into buf0 ----
    {
        int ear_f[16];
        int ear_s[16];
        int ebl_f[16];
        int ebl_s[16];

        #pragma unroll
        for (uint32_t l = 0; l < MACRO_TILE_K; ++l)
        {
            ear_f[l] = ear_first[l];
            ear_s[l] = ear_second[l];
            ebl_f[l] = ebl_first[l];
            ebl_s[l] = ebl_second[l];
        }

        {
            const uint32_t row = tid;

            #pragma unroll
            for (uint32_t l = 0; l < MACRO_TILE_K; ++l)
            {
                const uint32_t idx = (m_base + row) * k + l;

                const int32_t s_val =
                    (int32_t)(((idx * 0x9e3779b9u) % 129) - 64);

                const int32_t noise_val =
                    (int32_t)eal_lds[row * R + ear_f[l]] -
                    (int32_t)eal_lds[row * R + ear_s[l]];

                buf0[l * ldsldGEMM + row] = clampi8(s_val + noise_val);
            }
        }

        {
            const uint32_t col = tid;

            if (col < MACRO_TILE_N)
            {
                #pragma unroll
                for (uint32_t l = 0; l < MACRO_TILE_K; ++l)
                {
                    const uint32_t idx = (n_base + col) * k + l;

                    const int32_t s_val =
                        (int32_t)((((idx * 0x9e3779b9u + 0x55) % 129) - 64));

                    const int32_t noise_val =
                        (int32_t)ebr_lds[col * R + ebl_f[l]] -
                        (int32_t)ebr_lds[col * R + ebl_s[l]];

                    buf0[l * ldsldGEMM + MACRO_TILE_M + col] =
                        clampi8(s_val + noise_val);
                }
            }
        }
    }

    synchronize_workgroup();

    int8_t* buf_rd = buf0;
    int8_t* buf_wr = buf1;

    MmaFragAcc mmaFragAcc;
    fill_fragment(mmaFragAcc, static_cast<ComputeT>(0));

    MmaFragAcc tileAcc;
    fill_fragment(tileAcc, static_cast<ComputeT>(0));

    uint32_t tile_idx   = 0;
    uint32_t wmma_count = 0;

    const uint32_t checkpoint_period = R / MACRO_TILE_K;

    // ---- Main K-loop with double-buffered fill ----
    for (uint32_t k_base = MACRO_TILE_K; k_base < k; k_base += MACRO_TILE_K)
    {
        int ear_f[16];
        int ear_s[16];
        int ebl_f[16];
        int ebl_s[16];

        #pragma unroll
        for (uint32_t l = 0; l < MACRO_TILE_K; ++l)
        {
            const uint32_t ki = k_base + l;

            ear_f[l] = ear_first[ki];
            ear_s[l] = ear_second[ki];
            ebl_f[l] = ebl_first[ki];
            ebl_s[l] = ebl_second[ki];
        }

        LRFragA lrFragA;
        LRFragB lrFragB;

        load_matrix_sync(lrFragA, buf_rd + ldsReadOffsetA, ldsldGEMM);
        load_matrix_sync(lrFragB, buf_rd + ldsReadOffsetB, ldsldGEMM);

        {
            const uint32_t row = tid;

            #pragma unroll
            for (uint32_t l = 0; l < MACRO_TILE_K; ++l)
            {
                const uint32_t idx = (m_base + row) * k + (k_base + l);

                const int32_t s_val =
                    (int32_t)(((idx * 0x9e3779b9u) % 129) - 64);

                const int32_t noise_val =
                    (int32_t)eal_lds[row * R + ear_f[l]] -
                    (int32_t)eal_lds[row * R + ear_s[l]];

                buf_wr[l * ldsldGEMM + row] = clampi8(s_val + noise_val);
            }

            const uint32_t col = tid;

            if (col < MACRO_TILE_N)
            {
                #pragma unroll
                for (uint32_t l = 0; l < MACRO_TILE_K; ++l)
                {
                    const uint32_t idx = (n_base + col) * k + (k_base + l);

                    const int32_t s_val =
                        (int32_t)((((idx * 0x9e3779b9u + 0x55) % 129) - 64));

                    const int32_t noise_val =
                        (int32_t)ebr_lds[col * R + ebl_f[l]] -
                        (int32_t)ebr_lds[col * R + ebl_s[l]];

                    buf_wr[l * ldsldGEMM + MACRO_TILE_M + col] =
                        clampi8(s_val + noise_val);
                }
            }
        }

        MmaFragAcc stepAcc;
        fill_fragment(stepAcc, static_cast<ComputeT>(0));

        mma_sync(stepAcc,
                 transformLRFragAToMmaFragA(lrFragA),
                 transformLRFragBToMmaFragB(lrFragB),
                 stepAcc);

        #pragma unroll
        for (int i = 0; i < mmaFragAcc.num_elements; ++i)
        {
            mmaFragAcc.x[i] += stepAcc.x[i];
            tileAcc.x[i]    += stepAcc.x[i];
        }

        wmma_count++;

        if (jackpot_partial != nullptr &&
            checkpoint_period != 0 &&
            (wmma_count % checkpoint_period) == 0)
        {
            jackpot_xor_checkpoint(tileAcc, jackpot_partial, tile_idx, tid);
            tile_idx++;
        }

        synchronize_workgroup();

        int8_t* tmp = buf_rd;
        buf_rd = buf_wr;
        buf_wr = tmp;
    }

    // ---- Tail: final read + accumulate from remaining buffer ----
    {
        LRFragA lrFragA;
        LRFragB lrFragB;

        load_matrix_sync(lrFragA, buf_rd + ldsReadOffsetA, ldsldGEMM);
        load_matrix_sync(lrFragB, buf_rd + ldsReadOffsetB, ldsldGEMM);

        MmaFragAcc stepAcc;
        fill_fragment(stepAcc, static_cast<ComputeT>(0));

        mma_sync(stepAcc,
                 transformLRFragAToMmaFragA(lrFragA),
                 transformLRFragBToMmaFragB(lrFragB),
                 stepAcc);

        #pragma unroll
        for (int i = 0; i < mmaFragAcc.num_elements; ++i)
        {
            mmaFragAcc.x[i] += stepAcc.x[i];
            tileAcc.x[i]    += stepAcc.x[i];
        }
    }

    // ---- Jackpot checkpoint for final rank tile ----
    if (jackpot_partial != nullptr)
    {
        jackpot_xor_checkpoint(tileAcc, jackpot_partial, tile_idx, tid);
    }

    // ---- Store output C tile ----
    using MmaFragDMap1d = GetDataLayout_t<MmaFragD>;

    MmaFragD mmaFragD;

    #pragma unroll
    for (int i = 0; i < mmaFragD.num_elements; ++i)
        mmaFragD.x[i] = static_cast<OutputT>(mmaFragAcc.x[i]);

    store_matrix_sync(d + MmaFragDMap1d::fromMatrixCoord(warpTileCoord, ldd),
                      mmaFragD,
                      ldd);
}