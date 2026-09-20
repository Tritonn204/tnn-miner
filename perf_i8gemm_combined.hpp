/*******************************************************************************
 *
 * 2-Stage Orchestration i8 GEMM System — Tensile ASM mirror using C++20
 *
 * STAGE 1 (Compile-time): consteval TileConfig enumeration + templated kernel
 * STAGE 2 (Runtime):      lightweight dispatch selects best variant for {M,N,K}
 *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <array>
#include <type_traits>
#include <iomanip>
#include <iostream>
#include <string>

#include "tnn_matrix_decode.hpp"

#if defined(__HIPCC__) && !defined(__HIPRTC__)
#include <hip/hip_runtime.h>
#endif

// ============================================================================
// Architecture-parameterized constants
// ============================================================================

namespace gfx9_params {
    constexpr uint32_t kWmmaM = 32u;
    constexpr uint32_t kWmmaN = 32u;
    constexpr uint32_t kWmmaK = 16u;
    constexpr uint32_t kBlocksM = 2u;
    constexpr uint32_t kBlocksN = 2u;
    constexpr uint32_t kThreadsX = 64u;
    constexpr uint32_t kThreadsY = 4u;
    constexpr uint32_t kWarpSize = 64u;
}

namespace gfx11_params {
    constexpr uint32_t kWmmaM = 16u;
    constexpr uint32_t kWmmaN = 16u;
    constexpr uint32_t kWmmaK = 16u;
    constexpr uint32_t kBlocksM = 4u;
    constexpr uint32_t kBlocksN = 2u;
    constexpr uint32_t kThreadsX = 64u;
    constexpr uint32_t kThreadsY = 2u;
    constexpr uint32_t kWarpSize = 32u;
}

#if defined(__gfx900__) || defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
using namespace gfx9_params;
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1200__) || defined(__gfx1201__)
using namespace gfx11_params;
#elif defined(ROCWMMA_ARCH_GFX9)
using namespace gfx9_params;
#elif defined(ROCWMMA_ARCH_GFX11)
using namespace gfx11_params;
#else
using namespace gfx11_params;
#endif

constexpr uint32_t kBlockThreads = kThreadsX * kThreadsY;
constexpr uint32_t kMaxWarps = kBlockThreads / kWarpSize;

// ============================================================================
// VGPR abstraction types
// ============================================================================

using VRegI32   = int32_t;
using VRegI32x4 = int32_t __attribute__((__vector_size__(16)));
using VRegI32x8 = int32_t __attribute__((__vector_size__(32)));

struct CombinedOperandProbe
{
    uint32_t seen;
    uint32_t warp_m;
    uint32_t warp_n;
    uint32_t lane_id;
    uint32_t k_substep;
    uint32_t seen_lane8;
    uint32_t seen_lane16;
    uint32_t seen_lane24;
    int32_t a_lo[16];
    int32_t a_hi[16];
    int32_t a0[4];
    int32_t a1[4];
    int32_t a2[4];
    int32_t a3[4];
    int32_t b0[4];
    int32_t b1[4];
    int32_t a0_lane8[4];
    int32_t a1_lane8[4];
    int32_t a2_lane8[4];
    int32_t a3_lane8[4];
    int32_t b0_lane8[4];
    int32_t a0_lane16[4];
    int32_t a1_lane16[4];
    int32_t a2_lane16[4];
    int32_t a3_lane16[4];
    int32_t b0_lane16[4];
    int32_t a0_lane24[4];
    int32_t a1_lane24[4];
    int32_t a2_lane24[4];
    int32_t a3_lane24[4];
    int32_t b0_lane24[4];
    uint32_t seen_acc_lane0;
    uint32_t seen_acc_lane8;
    uint32_t seen_acc_lane16;
    uint32_t seen_acc_lane24;
    int32_t acc_lane0[8];
    int32_t acc_lane8[8];
    int32_t acc_lane16[8];
    int32_t acc_lane24[8];
    uint32_t seen_k1_lane0;
    uint32_t seen_k1_lane8;
    uint32_t seen_k1_lane16;
    uint32_t seen_k1_lane24;
    int32_t b0_k1_lane0[4];
    int32_t b0_k1_lane8[4];
    int32_t b0_k1_lane16[4];
    int32_t b0_k1_lane24[4];
    int32_t b0_k1_lane15[4];
    int32_t b0_k1_lane31[4];
    int32_t dbg_a0b0_lane0[8];
    int32_t dbg_a1b0_lane0[8];
    int32_t dbg_a2b0_lane0[8];
    int32_t dbg_a3b0_lane0[8];
    int32_t dbg_a0b0_lane16[8];
    int32_t dbg_a1b0_lane16[8];
    int32_t dbg_a2b0_lane16[8];
    int32_t dbg_a3b0_lane16[8];
    uint32_t seen_lane15;
    uint32_t seen_lane31;
    int32_t a0_lane15[4];
    int32_t a1_lane15[4];
    int32_t a2_lane15[4];
    int32_t a3_lane15[4];
    int32_t b0_lane15[4];
    int32_t acc_lane15[8];
    int32_t dbg_a0b0_lane15[8];
    int32_t dbg_a1b0_lane15[8];
    int32_t dbg_a2b0_lane15[8];
    int32_t dbg_a3b0_lane15[8];
    int32_t a0_lane31[4];
    int32_t a1_lane31[4];
    int32_t a2_lane31[4];
    int32_t a3_lane31[4];
    int32_t b0_lane31[4];
    int32_t acc_lane31[8];
    int32_t dbg_a0b0_lane31[8];
    int32_t dbg_a1b0_lane31[8];
    int32_t dbg_a2b0_lane31[8];
    int32_t dbg_a3b0_lane31[8];
};

// ============================================================================
// GPU intrinsics
// ============================================================================

__device__ inline void s_setprio_3() { asm volatile("s_setprio 3" ::: "memory"); }
__device__ inline void s_setprio_2() { asm volatile("s_setprio 2" ::: "memory"); }
__device__ inline void s_setprio_1() { asm volatile("s_setprio 1" ::: "memory"); }
__device__ inline void s_wait_lgkmcnt0()  { __builtin_amdgcn_s_waitcnt(0xF700); }
__device__ inline void s_wait_vmcnt0()    { __builtin_amdgcn_s_waitcnt(0xFF00); }
__device__ inline void s_wait_vmcnt0_lgkmcnt0() { __builtin_amdgcn_s_waitcnt(0x0000); }
__device__ inline void s_barrier()        { __builtin_amdgcn_s_barrier(); }

__device__ inline VRegI32 v_lshl_or_b32(VRegI32 hi, VRegI32 lo)
{
    VRegI32 r;
    asm volatile("v_lshl_or_b32 %0, %1, 8, %2" : "=v"(r) : "v"(hi), "v"(lo));
    return r;
}

__device__ inline VRegI32x4 ds_read_b128(VRegI32 addr)
{
    VRegI32x4 r;
    asm volatile("ds_read_b128 %0, %1" : "=v"(r) : "v"(addr) : "memory");
    return r;
}

__device__ inline void ds_write_b128(VRegI32 addr, VRegI32x4 data)
{
    asm volatile("ds_write_b128 %0, %1" :: "v"(addr), "v"(data) : "memory");
}

__device__ inline void ds_read_u8_pair(VRegI32& lo, VRegI32& hi, VRegI32 addrLo, VRegI32 addrHi)
{
    lo = 0; hi = 0;
    asm volatile("ds_read_u8 %0, %2" : "=v"(lo) : "v"(lo), "v"(addrLo) : "memory");
    asm volatile("ds_read_u8 %0, %2" : "=v"(hi) : "v"(hi), "v"(addrHi) : "memory");
}

__device__ inline void ds_read_u8_d16_hi_pair(VRegI32& lo, VRegI32& hi, VRegI32 addrLo, VRegI32 addrHi)
{
    asm volatile("ds_read_u8_d16_hi %0, %2" : "=v"(lo) : "v"(lo), "v"(addrLo) : "memory");
    asm volatile("ds_read_u8_d16_hi %0, %2" : "=v"(hi) : "v"(hi), "v"(addrHi) : "memory");
}

__device__ inline VRegI32x8 wmma_iu8(VRegI32x4 a, VRegI32x4 b, VRegI32x8 accIn)
{
    VRegI32x8 r = accIn;
    asm volatile("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %3 neg_lo:[1,1,1]"
        : "+v"(r) : "v"(b), "v"(a), "v"(accIn));
    return r;
}

// ============================================================================
// STAGE 1: Compile-time Kernel Variant Descriptor (consteval-compatible)
// ============================================================================

template <uint32_t TileM_, uint32_t TileN_, uint32_t TileK_, uint32_t UnrollK_>
struct TileConfig
{
    static constexpr uint32_t tile_m   = TileM_;
    static constexpr uint32_t tile_n   = TileN_;
    static constexpr uint32_t tile_k   = TileK_;
    static constexpr uint32_t unroll_k = UnrollK_;

    static constexpr uint32_t mma_m = kWmmaM;
    static constexpr uint32_t mma_n = kWmmaN;
    static constexpr uint32_t mma_k = kWmmaK;

    static constexpr uint32_t blocks_m = tile_m / mma_m;
    static constexpr uint32_t blocks_n = tile_n / mma_n;
    static constexpr uint32_t num_acc  = blocks_m * blocks_n;
    static constexpr uint32_t vgpr_per_acc = 8u;

    static constexpr uint32_t warps_m = (blocks_m <= 2u) ? blocks_m : 2u;
    static constexpr uint32_t warps_n = (blocks_n <= 2u) ? blocks_n : 2u;
    static constexpr uint32_t active_warps = warps_m * warps_n;
    static_assert(active_warps <= kMaxWarps, "Tile too large for workgroup");
    static_assert(blocks_m % warps_m == 0u, "blocks_m not divisible by warps_m");
    static_assert(blocks_n % warps_n == 0u, "blocks_n not divisible by warps_n");

    static constexpr uint32_t per_warp_blocks_m = blocks_m / warps_m;
    static constexpr uint32_t per_warp_blocks_n = blocks_n / warps_n;
    static constexpr uint32_t per_warp_acc = per_warp_blocks_m * per_warp_blocks_n;

    static constexpr uint32_t k_group_size = unroll_k * mma_k;
    static constexpr uint32_t k_groups_per_tile = tile_k / k_group_size;
    static_assert(tile_k % k_group_size == 0u, "tile_k must be divisible by k_group_size");

    static constexpr uint32_t lds_a_elems = tile_m * k_group_size;
    static constexpr uint32_t lds_b_elems = tile_n * k_group_size;
    static constexpr uint32_t lds_elems_per_buffer = lds_a_elems + lds_b_elems;

    static constexpr uint32_t lds_a_stride = tile_m;
    static constexpr uint32_t lds_b_stride = tile_n;

    static constexpr uint32_t tblock_x = kThreadsX;
    static constexpr uint32_t tblock_y = kThreadsY;
    static constexpr uint32_t tblock_size = tblock_x * tblock_y;

    static constexpr uint32_t lds_bytes  = lds_elems_per_buffer * sizeof(int8_t);
    static constexpr uint32_t lds_bytes_total = 2u * lds_bytes;
};

// ============================================================================
// LDS address offsets — computed from config geometry
// ============================================================================

template<typename C>
struct LdsGeometry
{
    static constexpr uint32_t a_base = 0u;
    static constexpr uint32_t b_base = C::lds_a_elems;
    static constexpr uint32_t a_stride = C::lds_a_stride;
    static constexpr uint32_t b_stride = C::lds_b_stride;
    static constexpr uint32_t buffer_stride = C::lds_elems_per_buffer;
};

// ============================================================================
// B LDS offsets — compute from warp/block index rather than magic numbers.
// B in LDS is K-major / N-contiguous: dims = [k_group_size][tile_n].
// Each lane reads one logical column ownership slice across 16 K positions.
// ============================================================================

template<typename C>
__device__ inline constexpr uint32_t b_lds_offset(uint32_t warp_n, uint32_t k_substep)
{
    return (k_substep) * (C::tile_n) + (warp_n) * (C::mma_n);
}

template<typename C>
__device__ inline constexpr uint32_t a_lds_offset(uint32_t warp_m, uint32_t k_col)
{
    return (warp_m) * (C::mma_m) + (k_col) * (C::tile_m);
}

// ============================================================================
// Accumulator manager — zero-init and store for any blocks_m × blocks_n
// ============================================================================

template<typename C>
struct KLoopState
{
    int8_t* lds_cur_a;
    int8_t* lds_cur_b;
    int8_t* lds_next_a;
    int8_t* lds_next_b;

    __device__ void init(int8_t* buf)
    {
        lds_cur_a = buf;
        lds_cur_b = buf + C::lds_a_elems;
        lds_next_a = buf + C::lds_elems_per_buffer;
        lds_next_b = buf + C::lds_elems_per_buffer + C::lds_a_elems;
    }

    __device__ void swap()
    {
        int8_t* tmpA = lds_cur_a; lds_cur_a = lds_next_a; lds_next_a = tmpA;
        int8_t* tmpB = lds_cur_b; lds_cur_b = lds_next_b; lds_next_b = tmpB;
    }
};

template<typename C>
struct AccGroup
{
    static constexpr uint32_t N = C::per_warp_acc;
    VRegI32x8 data[N];

    __device__ void zero()
    {
        #pragma unroll
        for (uint32_t i = 0; i < N; ++i)
        {
            #pragma unroll
            for (uint32_t j = 0; j < 8; ++j)
                data[i][j] = 0;
    }
}
};

// ============================================================================
// Global → LDS prefetch: cooperative thread load + LDS write
// ============================================================================

template<typename C>
__device__ void prefetch_a_lds(
    int8_t*        ldsA,
    const int8_t*  globalA,
    uint32_t       macro_m,
    uint32_t       k_global,
    uint32_t       lda,
    uint32_t       flat_tid)
{
    constexpr uint32_t M = C::tile_m;
    constexpr uint32_t K = C::k_group_size;
    constexpr uint32_t threads = C::active_warps * kWarpSize;

    for(uint32_t idx = flat_tid; idx < M * K; idx += threads)
    {
        uint32_t k_col = idx / M;
        uint32_t m_row = idx % M;
        uint32_t gRow  = macro_m + m_row;
        uint32_t gCol  = k_global + k_col;
        ldsA[m_row + k_col * C::lds_a_stride] = globalA[gRow + gCol * lda];
    }
}

template<typename C>
__device__ void prefetch_b_lds(
    int8_t*        ldsB,
    const int8_t*  globalB,
    uint32_t       macro_n,
    uint32_t       k_global,
    uint32_t       ldb,
    uint32_t       flat_tid)
{
    constexpr uint32_t N = C::tile_n;
    constexpr uint32_t K = C::k_group_size;
    constexpr uint32_t threads = C::active_warps * kWarpSize;

    for(uint32_t idx = flat_tid; idx < N * K; idx += threads)
    {
        uint32_t n_col = idx / K;
        uint32_t k_row = idx % K;
        uint32_t gRow  = k_global + k_row;
        uint32_t gCol  = macro_n + n_col;
        ldsB[n_col + k_row * C::lds_b_stride] = globalB[gRow + gCol * ldb];
    }
}

// ============================================================================
// A LDS reader — reads bytes for one K-substep into a_lo[16], a_hi[16].
// For i8 WMMA, each lane loads 16 bytes (8 cols × 2 halves) per K-substep.
// A in LDS is col_major [tile_m][k_group]. Each warp reads its warp_m block.
// ============================================================================

template<typename C>
__device__ void load_lds_a_one_kstep(
    VRegI32 (&a_lo)[16],
    VRegI32 (&a_hi)[16],
    int8_t const* ldsA_base,
    uint32_t      warp_m,
    uint32_t      k_substep,
    uint32_t      lane_id)
{
    constexpr uint32_t M_warp = C::mma_m;
    constexpr uint32_t stride = C::lds_a_stride;
    auto const* ldsA = reinterpret_cast<volatile int8_t const*>(ldsA_base);
    const uint32_t lane_row = lane_id & 0xFu;
    const uint32_t lane_half = (lane_id >> 4) & 0x1u;
    const uint32_t row_base = warp_m * M_warp + lane_row;
    const uint32_t idx_base = lane_half * 8u;
    const uint32_t col_base = k_substep * 16u;

    #pragma unroll
    for(uint32_t row_sel = 0; row_sel < 2u; ++row_sel)
    {
        const uint32_t row = row_base + row_sel;
        #pragma unroll
        for(uint32_t reg = 0; reg < 4u; ++reg)
        {
            const uint32_t idx = idx_base + row_sel * 4u + reg;
            const uint32_t k0 = col_base + reg * 4u + 0u;
            const uint32_t k1 = col_base + reg * 4u + 1u;
            const uint32_t k2 = col_base + reg * 4u + 2u;
            const uint32_t k3 = col_base + reg * 4u + 3u;

            const uint32_t off0 = row + k0 * stride;
            const uint32_t off1 = row + k1 * stride;
            const uint32_t off2 = row + k2 * stride;
            const uint32_t off3 = row + k3 * stride;

            const uint32_t b0 = static_cast<uint8_t>(ldsA[off0]);
            const uint32_t b1 = static_cast<uint8_t>(ldsA[off1]);
            const uint32_t b2 = static_cast<uint8_t>(ldsA[off2]);
            const uint32_t b3 = static_cast<uint8_t>(ldsA[off3]);

            a_lo[idx] = static_cast<VRegI32>(b0 | (b2 << 16));
            a_hi[idx] = static_cast<VRegI32>(b1 | (b3 << 16));
        }
    }
}

// ============================================================================
// B LDS reader — one WMMA B operand (4 VGPRs) for one (N-block, K-substep).
// B in LDS is N-contiguous at fixed K row. Lane ownership selects the logical
// output-column slice; k_substep selects the 16 K positions.
// ============================================================================

template<typename C>
__device__ inline VRegI32x4 load_lds_b_one_block(
    int8_t const* ldsB_base,
    uint32_t      warp_n,
    uint32_t      block_n_offset,
    uint32_t      k_substep,
    uint32_t      lane_id)
{
    constexpr uint32_t K_sub  = C::mma_k;
    constexpr uint32_t stride = C::lds_b_stride;
    auto const* ldsB = reinterpret_cast<volatile int8_t const*>(ldsB_base);

    const uint32_t col_base = warp_n * C::per_warp_blocks_n * C::mma_n + block_n_offset * C::mma_n;
    const uint32_t row_in_blk = lane_id & 0xFu;
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1200__) || defined(__gfx1201__) || defined(ROCWMMA_ARCH_GFX11)
    const uint32_t lane_col = ((lane_id >> 4) & 0x1u) * 8u + ((lane_id >> 2) & 0x3u);
#else
    const uint32_t lane_col = ((lane_id >> 4) & 0x1u) * 8u + ((lane_id >> 2) & 0x3u);
#endif
    const uint32_t col = col_base + lane_col;
    const uint32_t row_base = k_substep * K_sub;

    VRegI32x4 out{};
    #pragma unroll
    for(uint32_t reg = 0; reg < 4u; ++reg)
    {
        const uint32_t r0 = row_base + reg * 4u + 0u;
        const uint32_t r1 = row_base + reg * 4u + 1u;
        const uint32_t r2 = row_base + reg * 4u + 2u;
        const uint32_t r3 = row_base + reg * 4u + 3u;
        const uint32_t b0 = static_cast<uint8_t>(ldsB[col + r0 * stride]);
        const uint32_t b1 = static_cast<uint8_t>(ldsB[col + r1 * stride]);
        const uint32_t b2 = static_cast<uint8_t>(ldsB[col + r2 * stride]);
        const uint32_t b3 = static_cast<uint8_t>(ldsB[col + r3 * stride]);
        out[reg] = static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
    return out;
}

template<typename C>
__device__ inline VRegI32x4 load_lds_a_wmma_frag_single(
    int8_t const* ldsA_base,
    uint32_t      warp_m,
    uint32_t      k_substep,
    uint32_t      lane_id)
{
    constexpr uint32_t stride = C::lds_a_stride;
    auto const* ldsA = reinterpret_cast<volatile int8_t const*>(ldsA_base);
    const uint32_t row = warp_m * C::mma_m + (lane_id & 0xFu);
    const uint32_t col_base = k_substep * C::mma_k;

    VRegI32x4 out{};
    #pragma unroll
    for (uint32_t reg = 0; reg < 4u; ++reg)
    {
        const uint32_t k0 = col_base + reg * 4u + 0u;
        const uint32_t k1 = col_base + reg * 4u + 1u;
        const uint32_t k2 = col_base + reg * 4u + 2u;
        const uint32_t k3 = col_base + reg * 4u + 3u;
        const uint32_t b0 = static_cast<uint8_t>(ldsA[row + k0 * stride]);
        const uint32_t b1 = static_cast<uint8_t>(ldsA[row + k1 * stride]);
        const uint32_t b2 = static_cast<uint8_t>(ldsA[row + k2 * stride]);
        const uint32_t b3 = static_cast<uint8_t>(ldsA[row + k3 * stride]);
        out[reg] = static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
    return out;
}

template<typename C>
__device__ inline VRegI32x4 load_lds_b_wmma_frag_single(
    int8_t const* ldsB_base,
    uint32_t      warp_n,
    uint32_t      k_substep,
    uint32_t      lane_id)
{
    constexpr uint32_t stride = C::lds_b_stride;
    auto const* ldsB = reinterpret_cast<volatile int8_t const*>(ldsB_base);
    const uint32_t col = warp_n * C::mma_n + (lane_id & 0xFu);
    const uint32_t row_base = k_substep * C::mma_k;

    VRegI32x4 out{};
    #pragma unroll
    for (uint32_t reg = 0; reg < 4u; ++reg)
    {
        const uint32_t r0 = row_base + reg * 4u + 0u;
        const uint32_t r1 = row_base + reg * 4u + 1u;
        const uint32_t r2 = row_base + reg * 4u + 2u;
        const uint32_t r3 = row_base + reg * 4u + 3u;
        const uint32_t b0 = static_cast<uint8_t>(ldsB[col + r0 * stride]);
        const uint32_t b1 = static_cast<uint8_t>(ldsB[col + r1 * stride]);
        const uint32_t b2 = static_cast<uint8_t>(ldsB[col + r2 * stride]);
        const uint32_t b3 = static_cast<uint8_t>(ldsB[col + r3 * stride]);
        out[reg] = static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
    return out;
}

// ============================================================================
// A operand packing — combines lo/hi byte pairs via v_lshl_or_b32 into WMMA operands.
// ============================================================================

__device__ inline void pack_a_first_half(
    const VRegI32 (&a_lo)[16], const VRegI32 (&a_hi)[16],
    VRegI32x4& a0, VRegI32x4& a1)
{
    a0[0] = v_lshl_or_b32(a_hi[0], a_lo[0]);
    a0[1] = v_lshl_or_b32(a_hi[1], a_lo[1]);
    a0[2] = v_lshl_or_b32(a_hi[2], a_lo[2]);
    a0[3] = v_lshl_or_b32(a_hi[3], a_lo[3]);
    a1[0] = v_lshl_or_b32(a_hi[4], a_lo[4]);
    a1[1] = v_lshl_or_b32(a_hi[5], a_lo[5]);
    a1[2] = v_lshl_or_b32(a_hi[6], a_lo[6]);
    a1[3] = v_lshl_or_b32(a_hi[7], a_lo[7]);
}

__device__ inline void pack_a_second_half(
    const VRegI32 (&a_lo)[16], const VRegI32 (&a_hi)[16],
    VRegI32x4& a2, VRegI32x4& a3)
{
    a2[0] = v_lshl_or_b32(a_hi[8],  a_lo[8]);
    a2[1] = v_lshl_or_b32(a_hi[9],  a_lo[9]);
    a2[2] = v_lshl_or_b32(a_hi[10], a_lo[10]);
    a2[3] = v_lshl_or_b32(a_hi[11], a_lo[11]);
    a3[0] = v_lshl_or_b32(a_hi[12], a_lo[12]);
    a3[1] = v_lshl_or_b32(a_hi[13], a_lo[13]);
    a3[2] = v_lshl_or_b32(a_hi[14], a_lo[14]);
    a3[3] = v_lshl_or_b32(a_hi[15], a_lo[15]);
}

// ============================================================================
// WMMA iteration for one acc block — issues remaining WMMA instructions after
// the first acc += a0*b0 has already been issued (for dual-issue scheduling).
// K-contiguous B: each B operand covers all 16 K positions.
// 4 A operands (a0-a3: 2 M-row-halves × 2 K-halves) × 1 B operand = 4 WMMA total.
// ============================================================================

template<typename C>
__device__ inline void mfma_iter_one_block(
    VRegI32x8&               acc,
    const VRegI32x4&         a0,
    const VRegI32x4&         a1,
    const VRegI32x4&         a2,
    const VRegI32x4&         a3,
    const VRegI32x4&         b)
{
    acc = wmma_iu8(a1, b, acc);
    acc = wmma_iu8(a2, b, acc);
    acc = wmma_iu8(a3, b, acc);
}

// ============================================================================
// Full compute for one K-substep, all blocks in the warp.
// K-contiguous B layout: one ds_read_b128 per (N-block, k_substep).
// ============================================================================

template<typename C>
__device__ void compute_warp_blocks_one_kstep(
    AccGroup<C>& acc,
    int8_t const* ldsA,
    int8_t const* ldsB,
    uint32_t      warp_m,
    uint32_t      warp_n,
    uint32_t      k_substep,
    uint32_t      lane_id,
    CombinedOperandProbe* probe)
{
    constexpr uint32_t Pn = C::per_warp_blocks_n;

    if constexpr (C::per_warp_blocks_m == 1u && C::per_warp_blocks_n == 1u)
    {
        VRegI32x4 a_frag = load_lds_a_wmma_frag_single<C>(ldsA, warp_m, k_substep, lane_id);
        VRegI32x4 b_frag = load_lds_b_wmma_frag_single<C>(ldsB, warp_n, k_substep, lane_id);

        if(probe && blockIdx.x == 0 && blockIdx.y == 0 &&
           warp_m == 0 && warp_n == 0 && k_substep == 0)
        {
            if (lane_id == 0)
            {
                probe->seen = 1u;
                probe->warp_m = warp_m;
                probe->warp_n = warp_n;
                probe->lane_id = lane_id;
                probe->k_substep = k_substep;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i)
                {
                    probe->a0[i] = a_frag[i];
                    probe->a1[i] = 0;
                    probe->a2[i] = 0;
                    probe->a3[i] = 0;
                    probe->b0[i] = b_frag[i];
                    probe->b1[i] = 0;
                }
            }
            else if (lane_id == 8)
            {
                probe->seen_lane8 = 1u;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i)
                {
                    probe->a0_lane8[i] = a_frag[i];
                    probe->a1_lane8[i] = 0;
                    probe->a2_lane8[i] = 0;
                    probe->a3_lane8[i] = 0;
                    probe->b0_lane8[i] = b_frag[i];
                }
            }
            else if (lane_id == 15)
            {
                probe->seen_lane15 = 1u;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i)
                {
                    probe->a0_lane15[i] = a_frag[i];
                    probe->a1_lane15[i] = 0;
                    probe->a2_lane15[i] = 0;
                    probe->a3_lane15[i] = 0;
                    probe->b0_lane15[i] = b_frag[i];
                }
            }
            else if (lane_id == 16)
            {
                probe->seen_lane16 = 1u;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i)
                {
                    probe->a0_lane16[i] = a_frag[i];
                    probe->a1_lane16[i] = 0;
                    probe->a2_lane16[i] = 0;
                    probe->a3_lane16[i] = 0;
                    probe->b0_lane16[i] = b_frag[i];
                }
            }
            else if (lane_id == 24)
            {
                probe->seen_lane24 = 1u;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i)
                {
                    probe->a0_lane24[i] = a_frag[i];
                    probe->a1_lane24[i] = 0;
                    probe->a2_lane24[i] = 0;
                    probe->a3_lane24[i] = 0;
                    probe->b0_lane24[i] = b_frag[i];
                }
            }
            else if (lane_id == 31)
            {
                probe->seen_lane31 = 1u;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i)
                {
                    probe->a0_lane31[i] = a_frag[i];
                    probe->a1_lane31[i] = 0;
                    probe->a2_lane31[i] = 0;
                    probe->a3_lane31[i] = 0;
                    probe->b0_lane31[i] = b_frag[i];
                }
            }
        }

        if(probe && blockIdx.x == 0 && blockIdx.y == 0 && warp_m == 0 && warp_n == 0 && k_substep == 1)
        {
            if(lane_id == 0)
            {
                probe->seen_k1_lane0 = 1u;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i) probe->b0_k1_lane0[i] = b_frag[i];
            }
            else if(lane_id == 8)
            {
                probe->seen_k1_lane8 = 1u;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i) probe->b0_k1_lane8[i] = b_frag[i];
            }
            else if(lane_id == 16)
            {
                probe->seen_k1_lane16 = 1u;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i) probe->b0_k1_lane16[i] = b_frag[i];
            }
            else if(lane_id == 24)
            {
                probe->seen_k1_lane24 = 1u;
                #pragma unroll
                for(uint32_t i = 0; i < 4u; ++i) probe->b0_k1_lane24[i] = b_frag[i];
            }
        }

        s_setprio_3();
        acc.data[0] = wmma_iu8(a_frag, b_frag, acc.data[0]);
        s_setprio_1();
        return;
    }

    VRegI32   a_lo[16]{}, a_hi[16]{};
    VRegI32x4 a0{}, a1{}, a2{}, a3{};
    VRegI32x4 b[2]{};

    load_lds_a_one_kstep<C>(a_lo, a_hi, ldsA, warp_m, k_substep, lane_id);

    #pragma unroll
    for (uint32_t n = 0; n < Pn; ++n)
        b[n] = load_lds_b_one_block<C>(ldsB, warp_n, n, k_substep, lane_id);

    s_wait_lgkmcnt0();

    pack_a_first_half(a_lo, a_hi, a0, a1);
    pack_a_second_half(a_lo, a_hi, a2, a3);

    if(probe && blockIdx.x == 0 && blockIdx.y == 0 &&
       warp_m == 0 && warp_n == 0 && lane_id == 0 && k_substep == 0)
    {
        probe->seen = 1u;
        probe->warp_m = warp_m;
        probe->warp_n = warp_n;
        probe->lane_id = lane_id;
        probe->k_substep = k_substep;
        #pragma unroll
        for(uint32_t i = 0; i < 16u; ++i)
        {
            probe->a_lo[i] = a_lo[i];
            probe->a_hi[i] = a_hi[i];
        }
        #pragma unroll
        for(uint32_t i = 0; i < 4u; ++i)
        {
            probe->a0[i] = a0[i];
            probe->a1[i] = a1[i];
            probe->a2[i] = a2[i];
            probe->a3[i] = a3[i];
            probe->b0[i] = b[0][i];
            probe->b1[i] = (Pn > 1u) ? b[1][i] : 0;
        }
        VRegI32x8 z0{};
        VRegI32x8 z1{};
        VRegI32x8 z2{};
        VRegI32x8 z3{};
        #pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            z0[i] = 0;
            z1[i] = 0;
            z2[i] = 0;
            z3[i] = 0;
        }
        z0 = wmma_iu8(a0, b[0], z0);
        z1 = wmma_iu8(a1, b[0], z1);
        z2 = wmma_iu8(a2, b[0], z2);
        z3 = wmma_iu8(a3, b[0], z3);
        #pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            probe->dbg_a0b0_lane0[i] = z0[i];
            probe->dbg_a1b0_lane0[i] = z1[i];
            probe->dbg_a2b0_lane0[i] = z2[i];
            probe->dbg_a3b0_lane0[i] = z3[i];
        }
    }
    if(probe && blockIdx.x == 0 && blockIdx.y == 0 &&
       warp_m == 0 && warp_n == 0 && lane_id == 16 && k_substep == 0)
    {
        VRegI32x8 z0{};
        VRegI32x8 z1{};
        VRegI32x8 z2{};
        VRegI32x8 z3{};
        #pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            z0[i] = 0;
            z1[i] = 0;
            z2[i] = 0;
            z3[i] = 0;
        }
        z0 = wmma_iu8(a0, b[0], z0);
        z1 = wmma_iu8(a1, b[0], z1);
        z2 = wmma_iu8(a2, b[0], z2);
        z3 = wmma_iu8(a3, b[0], z3);
        #pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            probe->dbg_a0b0_lane16[i] = z0[i];
            probe->dbg_a1b0_lane16[i] = z1[i];
            probe->dbg_a2b0_lane16[i] = z2[i];
            probe->dbg_a3b0_lane16[i] = z3[i];
        }
    }
    if(probe && blockIdx.x == 0 && blockIdx.y == 0 &&
       warp_m == 0 && warp_n == 0 && lane_id == 8 && k_substep == 0)
    {
        probe->seen_lane8 = 1u;
        #pragma unroll
        for(uint32_t i = 0; i < 4u; ++i)
        {
            probe->a0_lane8[i] = a0[i];
            probe->a1_lane8[i] = a1[i];
            probe->a2_lane8[i] = a2[i];
            probe->a3_lane8[i] = a3[i];
            probe->b0_lane8[i] = b[0][i];
        }
    }
    if(probe && blockIdx.x == 0 && blockIdx.y == 0 &&
       warp_m == 0 && warp_n == 0 && lane_id == 16 && k_substep == 0)
    {
        probe->seen_lane16 = 1u;
        #pragma unroll
        for(uint32_t i = 0; i < 4u; ++i)
        {
            probe->a0_lane16[i] = a0[i];
            probe->a1_lane16[i] = a1[i];
            probe->a2_lane16[i] = a2[i];
            probe->a3_lane16[i] = a3[i];
            probe->b0_lane16[i] = b[0][i];
        }
    }
    if(probe && blockIdx.x == 0 && blockIdx.y == 0 &&
       warp_m == 0 && warp_n == 0 && lane_id == 24 && k_substep == 0)
    {
        probe->seen_lane24 = 1u;
        #pragma unroll
        for(uint32_t i = 0; i < 4u; ++i)
        {
            probe->a0_lane24[i] = a0[i];
            probe->a1_lane24[i] = a1[i];
            probe->a2_lane24[i] = a2[i];
            probe->a3_lane24[i] = a3[i];
            probe->b0_lane24[i] = b[0][i];
        }
    }
    if(probe && blockIdx.x == 0 && blockIdx.y == 0 &&
       warp_m == 0 && warp_n == 0 && lane_id == 15 && k_substep == 0)
    {
        probe->seen_lane15 = 1u;
        #pragma unroll
        for(uint32_t i = 0; i < 4u; ++i)
        {
            probe->a0_lane15[i] = a0[i];
            probe->a1_lane15[i] = a1[i];
            probe->a2_lane15[i] = a2[i];
            probe->a3_lane15[i] = a3[i];
            probe->b0_lane15[i] = b[0][i];
        }
        VRegI32x8 z0{};
        VRegI32x8 z1{};
        VRegI32x8 z2{};
        VRegI32x8 z3{};
        #pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            z0[i] = 0; z1[i] = 0; z2[i] = 0; z3[i] = 0;
        }
        z0 = wmma_iu8(a0, b[0], z0);
        z1 = wmma_iu8(a1, b[0], z1);
        z2 = wmma_iu8(a2, b[0], z2);
        z3 = wmma_iu8(a3, b[0], z3);
        #pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            probe->dbg_a0b0_lane15[i] = z0[i];
            probe->dbg_a1b0_lane15[i] = z1[i];
            probe->dbg_a2b0_lane15[i] = z2[i];
            probe->dbg_a3b0_lane15[i] = z3[i];
        }
    }
    if(probe && blockIdx.x == 0 && blockIdx.y == 0 &&
       warp_m == 0 && warp_n == 0 && lane_id == 31 && k_substep == 0)
    {
        probe->seen_lane31 = 1u;
        #pragma unroll
        for(uint32_t i = 0; i < 4u; ++i)
        {
            probe->a0_lane31[i] = a0[i];
            probe->a1_lane31[i] = a1[i];
            probe->a2_lane31[i] = a2[i];
            probe->a3_lane31[i] = a3[i];
            probe->b0_lane31[i] = b[0][i];
        }
        VRegI32x8 z0{};
        VRegI32x8 z1{};
        VRegI32x8 z2{};
        VRegI32x8 z3{};
        #pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            z0[i] = 0; z1[i] = 0; z2[i] = 0; z3[i] = 0;
        }
        z0 = wmma_iu8(a0, b[0], z0);
        z1 = wmma_iu8(a1, b[0], z1);
        z2 = wmma_iu8(a2, b[0], z2);
        z3 = wmma_iu8(a3, b[0], z3);
        #pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            probe->dbg_a0b0_lane31[i] = z0[i];
            probe->dbg_a1b0_lane31[i] = z1[i];
            probe->dbg_a2b0_lane31[i] = z2[i];
            probe->dbg_a3b0_lane31[i] = z3[i];
        }
    }
    if(probe && blockIdx.x == 0 && blockIdx.y == 0 && warp_m == 0 && warp_n == 0 && k_substep == 1)
    {
        if(lane_id == 0)
        {
            probe->seen_k1_lane0 = 1u;
            #pragma unroll
            for(uint32_t i = 0; i < 4u; ++i) probe->b0_k1_lane0[i] = b[0][i];
        }
        else if(lane_id == 8)
        {
            probe->seen_k1_lane8 = 1u;
            #pragma unroll
            for(uint32_t i = 0; i < 4u; ++i) probe->b0_k1_lane8[i] = b[0][i];
        }
        else if(lane_id == 16)
        {
            probe->seen_k1_lane16 = 1u;
            #pragma unroll
            for(uint32_t i = 0; i < 4u; ++i) probe->b0_k1_lane16[i] = b[0][i];
        }
        else if(lane_id == 24)
        {
            probe->seen_k1_lane24 = 1u;
            #pragma unroll
            for(uint32_t i = 0; i < 4u; ++i) probe->b0_k1_lane24[i] = b[0][i];
        }
    }

    s_setprio_3();

    if constexpr (C::per_warp_blocks_m == 1u && C::per_warp_blocks_n == 1u)
    {
        const uint32_t lane_quarter = lane_id >> 3;
        VRegI32x4 a_sel{};
        switch(lane_quarter)
        {
        case 0u: a_sel = a0; break;
        case 1u: a_sel = a1; break;
        case 2u: a_sel = a2; break;
        default: a_sel = a3; break;
        }
        acc.data[0] = wmma_iu8(a_sel, b[0], acc.data[0]);
    }
    else
    {
        #pragma unroll
        for (uint32_t n = 0; n < Pn; ++n)
        {
            acc.data[n] = wmma_iu8(a0, b[n], acc.data[n]);
            mfma_iter_one_block<C>(acc.data[n], a0, a1, a2, a3, b[n]);
        }
    }

    s_setprio_1();
}

// ============================================================================
// Full compute for one K-group (unroll_k K-substeps).
// ============================================================================

template<typename C>
__device__ void compute_k_group(
    AccGroup<C>& acc,
    int8_t const* ldsA,
    int8_t const* ldsB,
    uint32_t      warp_m,
    uint32_t      warp_n,
    uint32_t      lane_id,
    CombinedOperandProbe* probe)
{
    constexpr uint32_t U = C::unroll_k;
    for(uint32_t u = 0; u < U; ++u)
    {
        compute_warp_blocks_one_kstep<C>(acc, ldsA, ldsB, warp_m, warp_n, u, lane_id, probe);
    }
}


// ============================================================================
// Store output — write accumulator to D matrix with edge masking.
// ============================================================================

template<typename C>
__device__ void store_output_tile(
    const AccGroup<C>& acc,
    int32_t*           d,
    const int32_t*     c,
    uint32_t           macro_m,
    uint32_t           macro_n,
    uint32_t           M_global,
    uint32_t           N_global,
    uint32_t           ldc,
    uint32_t           ldd,
    uint32_t           lane_id,
    int32_t            alpha,
    int32_t            beta)
{
    constexpr uint32_t Pbm = C::per_warp_blocks_m;
    constexpr uint32_t Pbn = C::per_warp_blocks_n;
    constexpr uint32_t Mm  = C::mma_m;
    constexpr uint32_t Mn  = C::mma_n;

    #pragma unroll
    for(uint32_t bm = 0; bm < Pbm; ++bm)
    {
        #pragma unroll
        for(uint32_t bn = 0; bn < Pbn; ++bn)
        {
            const VRegI32x8& group = acc.data[bm * Pbn + bn];
            uint32_t base_row = macro_m + bm * Mm;
            uint32_t base_col = macro_n + bn * Mn;

            #pragma unroll
            for(uint32_t e = 0; e < 8u; ++e)
            {
                uint32_t out_row;
                uint32_t out_col;

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(ROCWMMA_ARCH_GFX11)
                const auto coord = tnn::decode::kGfx11WmmaI32_16x16x16_Iu8_Wave32[lane_id][e];
                out_row = base_row + coord.row;
                out_col = base_col + coord.col;
#elif defined(__gfx1200__) || defined(__gfx1201__)
                const auto coord = tnn::decode::kGfx12WmmaI32_16x16x16_Iu8_Wave32[lane_id][e];
                out_row = base_row + coord.row;
                out_col = base_col + coord.col;
#else
                uint32_t row_in_blk = lane_id & 0xF;
                uint32_t lane_half  = lane_id >> 4;
                uint32_t col_quad = lane_half;
                out_row = base_row + (e & 0x3) * 4u + (row_in_blk & 0x3);
                out_col = base_col + col_quad * 8u + ((e >> 2) & 0x1) * 4u + (row_in_blk >> 2);
#endif

                if((out_row < M_global) && (out_col < N_global)) [[likely]]
                {
                    int32_t c_val = c[out_row * ldc + out_col];
                    int32_t tmp   = alpha * group[e] + beta * c_val;
                    d[out_row * ldd + out_col] = tmp;
                }
            }
        }
    }
}

// ============================================================================
// Main kernel driver — one full tile with K-loop, double-buffering, and tail.
//
// Structurally mirrors Tensile kernelBody:
//   1. Prefetch tile_0 into LDS
//   2. K-loop: compute from current LDS buffer, prefetch next K into alt buffer
//   3. Swap buffers, wait + barrier, repeat
//   4. Final compute from last LDS buffer
//   5. Tail loop for K remainder
//   6. Store output with edge masking
// ============================================================================

template<typename C>
__global__ __launch_bounds__(C::tblock_size) void i8gemm_kernel(
    uint32_t           M,
    uint32_t           N,
    uint32_t           K,
    const int8_t*      A,
    const int8_t*      B,
    const int32_t*     Cmat,
    int32_t*           D,
    uint32_t           lda,
    uint32_t           ldb,
    uint32_t           ldc,
    uint32_t           ldd,
    int32_t            alpha,
    int32_t            beta,
    CombinedOperandProbe* probe)
{
    const uint32_t flat_tid = threadIdx.x + threadIdx.y * C::tblock_x;
    const uint32_t lane_id  = flat_tid & 0x1F;
    const uint32_t warp_id  = flat_tid >> 5;

    constexpr uint32_t Wm = C::warps_m;
    constexpr uint32_t Wn = C::warps_n;

    if(warp_id >= Wm * Wn) [[unlikely]] return;

    const uint32_t warp_m = warp_id / Wn;
    const uint32_t warp_n = warp_id % Wn;

    const uint32_t macro_m = blockIdx.x * C::tile_m + warp_m * C::per_warp_blocks_m * C::mma_m;
    const uint32_t macro_n = blockIdx.y * C::tile_n + warp_n * C::per_warp_blocks_n * C::mma_n;

    __shared__ int8_t lds_buf[C::lds_elems_per_buffer * 2u];

    KLoopState<C> lds;
    lds.init(lds_buf);

    AccGroup<C> acc;
    acc.zero();

    constexpr uint32_t k_step = C::k_group_size;
    const uint32_t    k_full = (K / k_step) * k_step;
    const uint32_t    k_rem  = K - k_full;
    (void)k_full;

    if(K >= k_step) [[likely]]
    {
        prefetch_a_lds<C>(lds.lds_next_a, A, macro_m, 0u, lda, flat_tid);
        prefetch_b_lds<C>(lds.lds_next_b, B, macro_n, 0u, ldb, flat_tid);
        s_barrier();

        lds.swap();

        for(uint32_t k_iter = k_step; k_iter < k_full; k_iter += k_step)
        {
            if(k_iter + k_step <= K) [[likely]]
            {
                prefetch_a_lds<C>(lds.lds_next_a, A, macro_m, k_iter, lda, flat_tid);
                prefetch_b_lds<C>(lds.lds_next_b, B, macro_n, k_iter, ldb, flat_tid);
            }

            compute_k_group<C>(acc, lds.lds_cur_a, lds.lds_cur_b, warp_m, warp_n, lane_id, probe);

            s_wait_vmcnt0_lgkmcnt0();
            s_barrier();

            lds.swap();
        }

        compute_k_group<C>(acc, lds.lds_cur_a, lds.lds_cur_b, warp_m, warp_n, lane_id, probe);
    }

    if(k_rem > 0u) [[unlikely]]
    {
        prefetch_a_lds<C>(lds.lds_next_a, A, macro_m, k_full, lda, flat_tid);
        prefetch_b_lds<C>(lds.lds_next_b, B, macro_n, k_full, ldb, flat_tid);

        s_wait_vmcnt0_lgkmcnt0();
        s_barrier();

        lds.swap();

        compute_k_group<C>(acc, lds.lds_cur_a, lds.lds_cur_b, warp_m, warp_n, lane_id, probe);
    }

    s_wait_lgkmcnt0();
    s_barrier();

    if(probe && blockIdx.x == 0 && blockIdx.y == 0 && warp_m == 0 && warp_n == 0)
    {
        if(lane_id == 0)
        {
            probe->seen_acc_lane0 = 1u;
            #pragma unroll
            for(uint32_t i = 0; i < 8u; ++i)
                probe->acc_lane0[i] = acc.data[0][i];
        }
        else if(lane_id == 8)
        {
            probe->seen_acc_lane8 = 1u;
            #pragma unroll
            for(uint32_t i = 0; i < 8u; ++i)
                probe->acc_lane8[i] = acc.data[0][i];
        }
        else if(lane_id == 16)
        {
            probe->seen_acc_lane16 = 1u;
            #pragma unroll
            for(uint32_t i = 0; i < 8u; ++i)
                probe->acc_lane16[i] = acc.data[0][i];
        }
        else if(lane_id == 24)
        {
            probe->seen_acc_lane24 = 1u;
            #pragma unroll
            for(uint32_t i = 0; i < 8u; ++i)
                probe->acc_lane24[i] = acc.data[0][i];
        }
        else if(lane_id == 15)
        {
            #pragma unroll
            for(uint32_t i = 0; i < 8u; ++i)
                probe->acc_lane15[i] = acc.data[0][i];
        }
        else if(lane_id == 31)
        {
            #pragma unroll
            for(uint32_t i = 0; i < 8u; ++i)
                probe->acc_lane31[i] = acc.data[0][i];
        }
    }

    store_output_tile<C>(acc, D, Cmat, macro_m, macro_n, M, N, ldc, ldd, lane_id, alpha, beta);
}

// ============================================================================
// GFX9 MFMA Fallback Path — for CDNA (gfx908/gfx90a/gfx942) using
//   __builtin_amdgcn_mfma_i32_32x32x16i8_i8()
//   32×32×16 tile, Wave64, 256 threads, AccVGPR accumulators.
// Single i8 element per VGPR for A/B inputs (4 elements packed per VGPR).
// Compiled only when __gfx90*__ or __gfx942__ is defined (NOT gfx11+).
// ============================================================================

#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)

namespace mfma_gfx9_params {
    constexpr uint32_t kTileM       = 32u;
    constexpr uint32_t kTileN       = 32u;
    constexpr uint32_t kTileK       = 32u;
    constexpr uint32_t kMmaM        = 32u;
    constexpr uint32_t kMmaN        = 32u;
    constexpr uint32_t kMmaK        = 16u;
    constexpr uint32_t kBlocksM     = 1u;
    constexpr uint32_t kBlocksN     = 1u;
    constexpr uint32_t kThreadsX    = 64u;
    constexpr uint32_t kThreadsY    = 4u;
    constexpr uint32_t kWarpSize    = 64u;
    constexpr uint32_t kUnrollK     = 2u;
    constexpr uint32_t kKGroupSize  = kMmaK * kUnrollK;
    constexpr uint32_t kBlockThreads= kThreadsX * kThreadsY;
    constexpr uint32_t kAccRegs     = 16u;
}

template<uint32_t TileM_, uint32_t TileN_, uint32_t TileK_, uint32_t UnrollK_>
struct MFMATileConfig
{
    static constexpr uint32_t tile_m   = TileM_;
    static constexpr uint32_t tile_n   = TileN_;
    static constexpr uint32_t tile_k   = TileK_;
    static constexpr uint32_t unroll_k = UnrollK_;
    static constexpr uint32_t mma_m    = mfma_gfx9_params::kMmaM;
    static constexpr uint32_t mma_n    = mfma_gfx9_params::kMmaN;
    static constexpr uint32_t mma_k    = mfma_gfx9_params::kMmaK;
    static constexpr uint32_t blocks_m = tile_m / mma_m;
    static constexpr uint32_t blocks_n = tile_n / mma_n;
    static constexpr uint32_t k_group_size = unroll_k * mma_k;
    static constexpr uint32_t lds_a_elems  = tile_m * k_group_size;
    static constexpr uint32_t lds_b_elems  = tile_n * k_group_size;
    static constexpr uint32_t lds_elems_per_buffer = lds_a_elems + lds_b_elems;
    static constexpr uint32_t lds_a_stride = tile_m;
    static constexpr uint32_t lds_b_stride = k_group_size;
    static constexpr uint32_t tblock_x = mfma_gfx9_params::kThreadsX;
    static constexpr uint32_t tblock_y = mfma_gfx9_params::kThreadsY;
    static constexpr uint32_t tblock_size = tblock_x * tblock_y;
    static constexpr uint32_t kWarpSize  = mfma_gfx9_params::kWarpSize;
    static constexpr uint32_t lds_bytes  = lds_elems_per_buffer * sizeof(int8_t);
    static constexpr uint32_t lds_bytes_total = 2u * lds_bytes;
    static constexpr uint32_t warp_m = 0u;
    static constexpr uint32_t warp_n = 0u;
    static constexpr uint32_t per_warp_blocks_m = blocks_m;
    static constexpr uint32_t per_warp_blocks_n = blocks_n;
    static constexpr uint32_t per_warp_acc = blocks_m * blocks_n;
    static_assert(blocks_m == 1u && blocks_n == 1u, "MFMA path: 1 block per warp");
    static_assert(MFMATileConfig::kWarpSize == 64u, "MFMA path: Wave64 required");
};

// MFMA i32_32x32x16i8 intrinsic — srcA/B are single VGPRs (4 i8 values packed per VGPR).
__device__ inline void mfma_i32_32x32x16i8(
    int32_t (&acc)[16],
    int32_t   a_val,
    int32_t   b_val)
{
    asm volatile("v_mfma_i32_32x32x16i8 %0, %1, %2, %3"
        : "+a"(acc[0])
        : "v"(b_val), "v"(a_val), "a"(acc[0]));
}

// Global→LDS prefetch for the MFMA path (A is col-major [tile_m][k_group]).
template<typename C>
__device__ void mfma_prefetch_a_lds(
    int8_t* ldsA, const int8_t* globalA, uint32_t macro_m,
    uint32_t k_global, uint32_t lda, uint32_t flat_tid)
{
    constexpr uint32_t M = C::tile_m, K = C::k_group_size, threads = C::tblock_size;
    for (uint32_t idx = flat_tid; idx < M * K; idx += threads)
    {
        uint32_t k_col = idx / M;
        uint32_t m_row = idx % M;
        ldsA[m_row + k_col * C::lds_a_stride] = globalA[macro_m + m_row + (k_global + k_col) * lda];
    }
}

// Global→LDS prefetch for B (K-contiguous: [k_group][tile_n]).
template<typename C>
__device__ void mfma_prefetch_b_lds(
    int8_t* ldsB, const int8_t* globalB, uint32_t macro_n,
    uint32_t k_global, uint32_t ldb, uint32_t flat_tid)
{
    constexpr uint32_t N = C::tile_n, K = C::k_group_size, threads = C::tblock_size;
    for (uint32_t idx = flat_tid; idx < N * K; idx += threads)
    {
        uint32_t n_col = idx / K;
        uint32_t k_row = idx % K;
        ldsB[k_row + n_col * C::lds_b_stride] = globalB[(k_global + k_row) * ldb + macro_n + n_col];
    }
}

// A LDS reader for MFMA — 4 consecutive K-positions packed into one VGPR.
// A in LDS is col-major [tile_m][k_group], row = lane_id & 0x1F (Wave64→32 rows).
template<typename C>
__device__ void mfma_read_a_vgprs(
    int32_t (&a_vgpr)[4],
    int8_t const* ldsA_base,
    uint32_t k_substep,
    uint32_t lane_id)
{
    constexpr uint32_t stride = C::lds_a_stride;
    uint32_t row_gbl = lane_id & 0x1F;

    #pragma unroll
    for (uint32_t i = 0; i < 4u; ++i)
    {
        VRegI32 bytes[4] = {0, 0, 0, 0};
        #pragma unroll
        for (uint32_t b = 0; b < 4u; ++b)
        {
            uint32_t col = k_substep * C::mma_k + i * 4u + b;
            uint32_t off = row_gbl + col * stride;
            VRegI32 addr = reinterpret_cast<VRegI32>(static_cast<uintptr_t>(
                reinterpret_cast<uintptr_t>(ldsA_base) + off));
            asm volatile("ds_read_u8 %0, %2" : "=v"(bytes[b]) : "v"(bytes[b]), "v"(addr) : "memory");
        }
        s_wait_lgkmcnt0();
        a_vgpr[i] = (VRegI32)(
            ((VRegI32)((uint32_t)(bytes[0]) & 0xFFu))       |
            ((VRegI32)(((uint32_t)(bytes[1]) & 0xFFu) << 8)) |
            ((VRegI32)(((uint32_t)(bytes[2]) & 0xFFu) << 16))|
            ((VRegI32)(((uint32_t)(bytes[3]) & 0xFFu) << 24)));
    }
}

// B LDS reader for MFMA — 4 consecutive K-positions packed into one VGPR.
// B in LDS is K-contiguous [k_group][tile_n], col = lane_id & 0x1F (Wave64→32 cols).
template<typename C>
__device__ void mfma_read_b_vgprs(
    int32_t (&b_vgpr)[4],
    int8_t const* ldsB_base,
    uint32_t k_substep,
    uint32_t lane_id)
{
    constexpr uint32_t stride = C::lds_b_stride;
    uint32_t col = lane_id & 0x1F;

    #pragma unroll
    for (uint32_t i = 0; i < 4u; ++i)
    {
        VRegI32 bytes[4] = {0, 0, 0, 0};
        uint32_t k_row_base = k_substep * C::mma_k + i * 4u;
        #pragma unroll
        for (uint32_t b = 0; b < 4u; ++b)
        {
            uint32_t off = (k_row_base + b) + col * stride;
            VRegI32 addr = reinterpret_cast<VRegI32>(static_cast<uintptr_t>(
                reinterpret_cast<uintptr_t>(ldsB_base) + off));
            asm volatile("ds_read_u8 %0, %2" : "=v"(bytes[b]) : "v"(bytes[b]), "v"(addr) : "memory");
        }
        s_wait_lgkmcnt0();
        b_vgpr[i] = (VRegI32)(
            ((VRegI32)((uint32_t)(bytes[0]) & 0xFFu))       |
            ((VRegI32)(((uint32_t)(bytes[1]) & 0xFFu) << 8)) |
            ((VRegI32)(((uint32_t)(bytes[2]) & 0xFFu) << 16))|
            ((VRegI32)(((uint32_t)(bytes[3]) & 0xFFu) << 24)));
    }
}

template<typename C>
__global__ __launch_bounds__(C::tblock_size) void mfma_i8gemm_kernel(
    uint32_t M, uint32_t N, uint32_t K,
    const int8_t* A, const int8_t* B,
    const int32_t* Cmat, int32_t* D,
    uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t ldd,
    int32_t alpha, int32_t beta)
{
    const uint32_t flat_tid = threadIdx.x + threadIdx.y * C::tblock_x;
    const uint32_t lane_id  = flat_tid & 0x3F;
    const uint32_t row_idx  = lane_id & 0x1F;
    const uint32_t macro_m = blockIdx.x * C::tile_m;
    const uint32_t macro_n = blockIdx.y * C::tile_n;

    __shared__ int8_t lds_buf[C::lds_elems_per_buffer * 2u];

    __attribute__((amdgcn_accvgpr)) int32_t acc[C::per_warp_acc * 16] = {};

    constexpr uint32_t k_step = C::k_group_size;
    uint32_t k_global = 0u;
    uint32_t buffer_sel = 0u;

    auto lds_buffer = [&](uint32_t buf) -> int8_t* {
        return lds_buf + buf * C::lds_elems_per_buffer;
    };

    if (K >= k_step) [[likely]]
    {
        mfma_prefetch_a_lds<C>(lds_buffer(1), A, macro_m, 0u, lda, flat_tid);
        mfma_prefetch_b_lds<C>(lds_buffer(1) + C::lds_a_elems, B, macro_n, 0u, ldb, flat_tid);
        s_barrier();
        buffer_sel = 1u;
        k_global = k_step;
    }

    for (; k_global < K; k_global += k_step)
    {
        if (k_global + k_step <= K) [[likely]]
        {
            uint32_t next_buf = 1u - buffer_sel;
            mfma_prefetch_a_lds<C>(lds_buffer(next_buf), A, macro_m, k_global, lda, flat_tid);
            mfma_prefetch_b_lds<C>(lds_buffer(next_buf) + C::lds_a_elems, B, macro_n, k_global, ldb, flat_tid);
        }

        int8_t* ldsA = lds_buffer(buffer_sel);
        int8_t* ldsB = ldsA + C::lds_a_elems;

        s_wait_lgkmcnt0();

        for (uint32_t u = 0; u < C::unroll_k; ++u)
        {
            int32_t a_vgpr[4], b_vgpr[4];

            mfma_read_a_vgprs<C>(a_vgpr, ldsA, u, lane_id);
            mfma_read_b_vgprs<C>(b_vgpr, ldsB, u, lane_id);

            s_wait_lgkmcnt0();

            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i)
                mfma_i32_32x32x16i8(acc, a_vgpr[i], b_vgpr[i]);
        }

        s_wait_vmcnt0_lgkmcnt0();
        s_barrier();
        buffer_sel = 1u - buffer_sel;
    }

    {
        uint32_t base_row = macro_m + row_idx;

        #pragma unroll
        for (uint32_t e = 0; e < 16u; ++e)
            {
                uint32_t out_row = base_row;
                uint32_t out_col = macro_n + e;
                if (out_row < M && out_col < N) [[likely]]
                {
                    int32_t c_val = Cmat[out_row * ldc + out_col];
                    D[out_row * ldd + out_col] = alpha * acc[e] + beta * c_val;
                }
            }
        }
    }

// MFMA tile config alias
using MFMACfg = MFMATileConfig<mfma_gfx9_params::kTileM, mfma_gfx9_params::kTileN,
                               mfma_gfx9_params::kTileK, mfma_gfx9_params::kUnrollK>;

#endif // __gfx908__ || __gfx90a__ || __gfx942__

// ============================================================================
// STAGE 2: Runtime Dispatch System
// ============================================================================

enum class KernelVariant : uint32_t
{
    V_16x16x32_U2                       = 0,
    V_16x16x64_U2                       = 1,
    V_16x16x64_U4                       = 2,
    V_16x16x128_U2                      = 3,
    V_16x16x128_U4                      = 4,
    V_16x16x128_U8                      = 5,
    V_16x32x32_U2                       = 6,
    V_16x32x64_U2                       = 7,
    V_16x32x64_U4                       = 8,
    V_16x32x128_U2                      = 9,
    V_16x32x128_U4                      = 10,
    V_16x32x128_U8                      = 11,
    V_16x64x32_U2                       = 12,
    V_16x64x64_U2                       = 13,
    V_16x64x64_U4                       = 14,
    V_16x64x128_U2                      = 15,
    V_16x64x128_U4                      = 16,
    V_16x64x128_U8                      = 17,
    V_16x128x32_U2                      = 18,
    V_16x128x64_U2                      = 19,
    V_16x128x64_U4                      = 20,
    V_16x128x128_U2                     = 21,
    V_16x128x128_U4                     = 22,
    V_16x128x128_U8                     = 23,
    V_16x256x32_U2                      = 24,
    V_16x256x64_U2                      = 25,
    V_16x256x64_U4                      = 26,
    V_16x256x128_U2                     = 27,
    V_16x256x128_U4                     = 28,
    V_32x32x32_U2                       = 29,
    V_32x32x64_U2                       = 30,
    V_32x32x64_U4                       = 31,
    V_32x32x128_U2                      = 32,
    V_32x32x128_U4                      = 33,
    V_32x32x128_U8                      = 34,
    V_32x64x32_U2                       = 35,
    V_32x64x64_U2                       = 36,
    V_32x64x64_U4                       = 37,
    V_32x64x128_U2                      = 38,
    V_32x64x128_U4                      = 39,
    V_32x64x128_U8                      = 40,
    V_32x128x32_U2                      = 41,
    V_32x128x64_U2                      = 42,
    V_32x128x64_U4                      = 43,
    V_32x128x128_U2                     = 44,
    V_32x128x128_U4                     = 45,
    V_32x128x128_U8                     = 46,
    V_32x256x32_U2                      = 47,
    V_32x256x64_U2                      = 48,
    V_32x256x64_U4                      = 49,
    V_32x256x128_U2                     = 50,
    V_32x256x128_U4                     = 51,
    V_64x64x32_U2                       = 52,
    V_64x64x64_U2                       = 53,
    V_64x64x64_U4                       = 54,
    V_64x64x128_U2                      = 55,
    V_64x64x128_U4                      = 56,
    V_64x64x128_U8                      = 57,
    V_64x128x32_U2                      = 58,
    V_64x128x64_U2                      = 59,
    V_64x128x64_U4                      = 60,
    V_64x128x128_U2                     = 61,
    V_64x128x128_U4                     = 62,
    V_64x128x128_U8                     = 63,
    V_64x256x32_U2                      = 64,
    V_64x256x64_U2                      = 65,
    V_64x256x64_U4                      = 66,
    V_64x256x128_U2                     = 67,
    V_64x256x128_U4                     = 68,
    V_128x128x32_U2                     = 69,
    V_128x128x64_U2                     = 70,
    V_128x128x64_U4                     = 71,
    V_128x128x128_U2                    = 72,
    V_128x128x128_U4                    = 73,
    V_128x128x128_U8                    = 74,
    V_128x256x32_U2                     = 75,
    V_128x256x64_U2                     = 76,
    V_128x256x64_U4                     = 77,
    V_128x256x128_U2                    = 78,
    V_128x256x128_U4                    = 79,
    V_256x256x32_U2                     = 80,
    V_256x256x64_U2                     = 81,
    V_256x256x64_U4                     = 82,
    V_256x256x128_U2                    = 83,
    V_256x256x128_U4                    = 84,
    Count
};

// Type aliases for each variant
using C16x16x32_U2                             = TileConfig< 16,  16,  32, 2>;
using C16x16x64_U2                             = TileConfig< 16,  16,  64, 2>;
using C16x16x64_U4                             = TileConfig< 16,  16,  64, 4>;
using C16x16x128_U2                            = TileConfig< 16,  16, 128, 2>;
using C16x16x128_U4                            = TileConfig< 16,  16, 128, 4>;
using C16x16x128_U8                            = TileConfig< 16,  16, 128, 8>;
using C16x32x32_U2                             = TileConfig< 16,  32,  32, 2>;
using C16x32x64_U2                             = TileConfig< 16,  32,  64, 2>;
using C16x32x64_U4                             = TileConfig< 16,  32,  64, 4>;
using C16x32x128_U2                            = TileConfig< 16,  32, 128, 2>;
using C16x32x128_U4                            = TileConfig< 16,  32, 128, 4>;
using C16x32x128_U8                            = TileConfig< 16,  32, 128, 8>;
using C16x64x32_U2                             = TileConfig< 16,  64,  32, 2>;
using C16x64x64_U2                             = TileConfig< 16,  64,  64, 2>;
using C16x64x64_U4                             = TileConfig< 16,  64,  64, 4>;
using C16x64x128_U2                            = TileConfig< 16,  64, 128, 2>;
using C16x64x128_U4                            = TileConfig< 16,  64, 128, 4>;
using C16x64x128_U8                            = TileConfig< 16,  64, 128, 8>;
using C16x128x32_U2                            = TileConfig< 16, 128,  32, 2>;
using C16x128x64_U2                            = TileConfig< 16, 128,  64, 2>;
using C16x128x64_U4                            = TileConfig< 16, 128,  64, 4>;
using C16x128x128_U2                           = TileConfig< 16, 128, 128, 2>;
using C16x128x128_U4                           = TileConfig< 16, 128, 128, 4>;
using C16x128x128_U8                           = TileConfig< 16, 128, 128, 8>;
using C16x256x32_U2                            = TileConfig< 16, 256,  32, 2>;
using C16x256x64_U2                            = TileConfig< 16, 256,  64, 2>;
using C16x256x64_U4                            = TileConfig< 16, 256,  64, 4>;
using C16x256x128_U2                           = TileConfig< 16, 256, 128, 2>;
using C16x256x128_U4                           = TileConfig< 16, 256, 128, 4>;
using C32x32x32_U2                             = TileConfig< 32,  32,  32, 2>;
using C32x32x64_U2                             = TileConfig< 32,  32,  64, 2>;
using C32x32x64_U4                             = TileConfig< 32,  32,  64, 4>;
using C32x32x128_U2                            = TileConfig< 32,  32, 128, 2>;
using C32x32x128_U4                            = TileConfig< 32,  32, 128, 4>;
using C32x32x128_U8                            = TileConfig< 32,  32, 128, 8>;
using C32x64x32_U2                             = TileConfig< 32,  64,  32, 2>;
using C32x64x64_U2                             = TileConfig< 32,  64,  64, 2>;
using C32x64x64_U4                             = TileConfig< 32,  64,  64, 4>;
using C32x64x128_U2                            = TileConfig< 32,  64, 128, 2>;
using C32x64x128_U4                            = TileConfig< 32,  64, 128, 4>;
using C32x64x128_U8                            = TileConfig< 32,  64, 128, 8>;
using C32x128x32_U2                            = TileConfig< 32, 128,  32, 2>;
using C32x128x64_U2                            = TileConfig< 32, 128,  64, 2>;
using C32x128x64_U4                            = TileConfig< 32, 128,  64, 4>;
using C32x128x128_U2                           = TileConfig< 32, 128, 128, 2>;
using C32x128x128_U4                           = TileConfig< 32, 128, 128, 4>;
using C32x128x128_U8                           = TileConfig< 32, 128, 128, 8>;
using C32x256x32_U2                            = TileConfig< 32, 256,  32, 2>;
using C32x256x64_U2                            = TileConfig< 32, 256,  64, 2>;
using C32x256x64_U4                            = TileConfig< 32, 256,  64, 4>;
using C32x256x128_U2                           = TileConfig< 32, 256, 128, 2>;
using C32x256x128_U4                           = TileConfig< 32, 256, 128, 4>;
using C64x64x32_U2                             = TileConfig< 64,  64,  32, 2>;
using C64x64x64_U2                             = TileConfig< 64,  64,  64, 2>;
using C64x64x64_U4                             = TileConfig< 64,  64,  64, 4>;
using C64x64x128_U2                            = TileConfig< 64,  64, 128, 2>;
using C64x64x128_U4                            = TileConfig< 64,  64, 128, 4>;
using C64x64x128_U8                            = TileConfig< 64,  64, 128, 8>;
using C64x128x32_U2                            = TileConfig< 64, 128,  32, 2>;
using C64x128x64_U2                            = TileConfig< 64, 128,  64, 2>;
using C64x128x64_U4                            = TileConfig< 64, 128,  64, 4>;
using C64x128x128_U2                           = TileConfig< 64, 128, 128, 2>;
using C64x128x128_U4                           = TileConfig< 64, 128, 128, 4>;
using C64x128x128_U8                           = TileConfig< 64, 128, 128, 8>;
using C64x256x32_U2                            = TileConfig< 64, 256,  32, 2>;
using C64x256x64_U2                            = TileConfig< 64, 256,  64, 2>;
using C64x256x64_U4                            = TileConfig< 64, 256,  64, 4>;
using C64x256x128_U2                           = TileConfig< 64, 256, 128, 2>;
using C64x256x128_U4                           = TileConfig< 64, 256, 128, 4>;
using C128x128x32_U2                           = TileConfig<128, 128,  32, 2>;
using C128x128x64_U2                           = TileConfig<128, 128,  64, 2>;
using C128x128x64_U4                           = TileConfig<128, 128,  64, 4>;
using C128x128x128_U2                          = TileConfig<128, 128, 128, 2>;
using C128x128x128_U4                          = TileConfig<128, 128, 128, 4>;
using C128x128x128_U8                          = TileConfig<128, 128, 128, 8>;
using C128x256x32_U2                           = TileConfig<128, 256,  32, 2>;
using C128x256x64_U2                           = TileConfig<128, 256,  64, 2>;
using C128x256x64_U4                           = TileConfig<128, 256,  64, 4>;
using C128x256x128_U2                          = TileConfig<128, 256, 128, 2>;
using C128x256x128_U4                          = TileConfig<128, 256, 128, 4>;
using C256x256x32_U2                           = TileConfig<256, 256,  32, 2>;
using C256x256x64_U2                           = TileConfig<256, 256,  64, 2>;
using C256x256x64_U4                           = TileConfig<256, 256,  64, 4>;
using C256x256x128_U2                          = TileConfig<256, 256, 128, 2>;
using C256x256x128_U4                          = TileConfig<256, 256, 128, 4>;

template<KernelVariant V> struct VariantToConfig;
template<> struct VariantToConfig<KernelVariant::V_16x16x32_U2> { using type = C16x16x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x16x64_U2> { using type = C16x16x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x16x64_U4> { using type = C16x16x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_16x16x128_U2> { using type = C16x16x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x16x128_U4> { using type = C16x16x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_16x16x128_U8> { using type = C16x16x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_16x32x32_U2> { using type = C16x32x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x32x64_U2> { using type = C16x32x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x32x64_U4> { using type = C16x32x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_16x32x128_U2> { using type = C16x32x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x32x128_U4> { using type = C16x32x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_16x32x128_U8> { using type = C16x32x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_16x64x32_U2> { using type = C16x64x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x64x64_U2> { using type = C16x64x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x64x64_U4> { using type = C16x64x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_16x64x128_U2> { using type = C16x64x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x64x128_U4> { using type = C16x64x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_16x64x128_U8> { using type = C16x64x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_16x128x32_U2> { using type = C16x128x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x128x64_U2> { using type = C16x128x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x128x64_U4> { using type = C16x128x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_16x128x128_U2> { using type = C16x128x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x128x128_U4> { using type = C16x128x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_16x128x128_U8> { using type = C16x128x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_16x256x32_U2> { using type = C16x256x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x256x64_U2> { using type = C16x256x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x256x64_U4> { using type = C16x256x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_16x256x128_U2> { using type = C16x256x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_16x256x128_U4> { using type = C16x256x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_32x32x32_U2> { using type = C32x32x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x32x64_U2> { using type = C32x32x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x32x64_U4> { using type = C32x32x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_32x32x128_U2> { using type = C32x32x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x32x128_U4> { using type = C32x32x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_32x32x128_U8> { using type = C32x32x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_32x64x32_U2> { using type = C32x64x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x64x64_U2> { using type = C32x64x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x64x64_U4> { using type = C32x64x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_32x64x128_U2> { using type = C32x64x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x64x128_U4> { using type = C32x64x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_32x64x128_U8> { using type = C32x64x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_32x128x32_U2> { using type = C32x128x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x128x64_U2> { using type = C32x128x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x128x64_U4> { using type = C32x128x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_32x128x128_U2> { using type = C32x128x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x128x128_U4> { using type = C32x128x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_32x128x128_U8> { using type = C32x128x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_32x256x32_U2> { using type = C32x256x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x256x64_U2> { using type = C32x256x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x256x64_U4> { using type = C32x256x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_32x256x128_U2> { using type = C32x256x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_32x256x128_U4> { using type = C32x256x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_64x64x32_U2> { using type = C64x64x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_64x64x64_U2> { using type = C64x64x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_64x64x64_U4> { using type = C64x64x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_64x64x128_U2> { using type = C64x64x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_64x64x128_U4> { using type = C64x64x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_64x64x128_U8> { using type = C64x64x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_64x128x32_U2> { using type = C64x128x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_64x128x64_U2> { using type = C64x128x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_64x128x64_U4> { using type = C64x128x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_64x128x128_U2> { using type = C64x128x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_64x128x128_U4> { using type = C64x128x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_64x128x128_U8> { using type = C64x128x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_64x256x32_U2> { using type = C64x256x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_64x256x64_U2> { using type = C64x256x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_64x256x64_U4> { using type = C64x256x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_64x256x128_U2> { using type = C64x256x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_64x256x128_U4> { using type = C64x256x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_128x128x32_U2> { using type = C128x128x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_128x128x64_U2> { using type = C128x128x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_128x128x64_U4> { using type = C128x128x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_128x128x128_U2> { using type = C128x128x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_128x128x128_U4> { using type = C128x128x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_128x128x128_U8> { using type = C128x128x128_U8; };
template<> struct VariantToConfig<KernelVariant::V_128x256x32_U2> { using type = C128x256x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_128x256x64_U2> { using type = C128x256x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_128x256x64_U4> { using type = C128x256x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_128x256x128_U2> { using type = C128x256x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_128x256x128_U4> { using type = C128x256x128_U4; };
template<> struct VariantToConfig<KernelVariant::V_256x256x32_U2> { using type = C256x256x32_U2; };
template<> struct VariantToConfig<KernelVariant::V_256x256x64_U2> { using type = C256x256x64_U2; };
template<> struct VariantToConfig<KernelVariant::V_256x256x64_U4> { using type = C256x256x64_U4; };
template<> struct VariantToConfig<KernelVariant::V_256x256x128_U2> { using type = C256x256x128_U2; };
template<> struct VariantToConfig<KernelVariant::V_256x256x128_U4> { using type = C256x256x128_U4; };

// ============================================================================
// Variant metadata (constexpr for compile-time introspection)
// ============================================================================

struct VariantInfo
{
    KernelVariant id;
    uint32_t tile_m;
    uint32_t tile_n;
    uint32_t tile_k;
    uint32_t unroll_k;
    uint32_t lds_bytes;
    const char* name;
};

consteval std::array<VariantInfo, 85> make_variant_table()
{
    std::array<VariantInfo, 85> t{};
    auto add = [&](uint32_t i, KernelVariant v, uint32_t m, uint32_t n, uint32_t k, uint32_t u,
                   const char* nm)
    {
        t[i] = {v, m, n, k, u, (m * k + n * k) * 2u, nm};
    };
    add(  0, KernelVariant::V_16x16x32_U2,   16,  16,  32, 2, "16x16x32_U2");
    add(  1, KernelVariant::V_16x16x64_U2,   16,  16,  64, 2, "16x16x64_U2");
    add(  2, KernelVariant::V_16x16x64_U4,   16,  16,  64, 4, "16x16x64_U4");
    add(  3, KernelVariant::V_16x16x128_U2,  16,  16, 128, 2, "16x16x128_U2");
    add(  4, KernelVariant::V_16x16x128_U4,  16,  16, 128, 4, "16x16x128_U4");
    add(  5, KernelVariant::V_16x16x128_U8,  16,  16, 128, 8, "16x16x128_U8");
    add(  6, KernelVariant::V_16x32x32_U2,   16,  32,  32, 2, "16x32x32_U2");
    add(  7, KernelVariant::V_16x32x64_U2,   16,  32,  64, 2, "16x32x64_U2");
    add(  8, KernelVariant::V_16x32x64_U4,   16,  32,  64, 4, "16x32x64_U4");
    add(  9, KernelVariant::V_16x32x128_U2,  16,  32, 128, 2, "16x32x128_U2");
    add( 10, KernelVariant::V_16x32x128_U4,  16,  32, 128, 4, "16x32x128_U4");
    add( 11, KernelVariant::V_16x32x128_U8,  16,  32, 128, 8, "16x32x128_U8");
    add( 12, KernelVariant::V_16x64x32_U2,   16,  64,  32, 2, "16x64x32_U2");
    add( 13, KernelVariant::V_16x64x64_U2,   16,  64,  64, 2, "16x64x64_U2");
    add( 14, KernelVariant::V_16x64x64_U4,   16,  64,  64, 4, "16x64x64_U4");
    add( 15, KernelVariant::V_16x64x128_U2,  16,  64, 128, 2, "16x64x128_U2");
    add( 16, KernelVariant::V_16x64x128_U4,  16,  64, 128, 4, "16x64x128_U4");
    add( 17, KernelVariant::V_16x64x128_U8,  16,  64, 128, 8, "16x64x128_U8");
    add( 18, KernelVariant::V_16x128x32_U2,  16, 128,  32, 2, "16x128x32_U2");
    add( 19, KernelVariant::V_16x128x64_U2,  16, 128,  64, 2, "16x128x64_U2");
    add( 20, KernelVariant::V_16x128x64_U4,  16, 128,  64, 4, "16x128x64_U4");
    add( 21, KernelVariant::V_16x128x128_U2, 16, 128, 128, 2, "16x128x128_U2");
    add( 22, KernelVariant::V_16x128x128_U4, 16, 128, 128, 4, "16x128x128_U4");
    add( 23, KernelVariant::V_16x128x128_U8, 16, 128, 128, 8, "16x128x128_U8");
    add( 24, KernelVariant::V_16x256x32_U2,  16, 256,  32, 2, "16x256x32_U2");
    add( 25, KernelVariant::V_16x256x64_U2,  16, 256,  64, 2, "16x256x64_U2");
    add( 26, KernelVariant::V_16x256x64_U4,  16, 256,  64, 4, "16x256x64_U4");
    add( 27, KernelVariant::V_16x256x128_U2, 16, 256, 128, 2, "16x256x128_U2");
    add( 28, KernelVariant::V_16x256x128_U4, 16, 256, 128, 4, "16x256x128_U4");
    add( 29, KernelVariant::V_32x32x32_U2,   32,  32,  32, 2, "32x32x32_U2");
    add( 30, KernelVariant::V_32x32x64_U2,   32,  32,  64, 2, "32x32x64_U2");
    add( 31, KernelVariant::V_32x32x64_U4,   32,  32,  64, 4, "32x32x64_U4");
    add( 32, KernelVariant::V_32x32x128_U2,  32,  32, 128, 2, "32x32x128_U2");
    add( 33, KernelVariant::V_32x32x128_U4,  32,  32, 128, 4, "32x32x128_U4");
    add( 34, KernelVariant::V_32x32x128_U8,  32,  32, 128, 8, "32x32x128_U8");
    add( 35, KernelVariant::V_32x64x32_U2,   32,  64,  32, 2, "32x64x32_U2");
    add( 36, KernelVariant::V_32x64x64_U2,   32,  64,  64, 2, "32x64x64_U2");
    add( 37, KernelVariant::V_32x64x64_U4,   32,  64,  64, 4, "32x64x64_U4");
    add( 38, KernelVariant::V_32x64x128_U2,  32,  64, 128, 2, "32x64x128_U2");
    add( 39, KernelVariant::V_32x64x128_U4,  32,  64, 128, 4, "32x64x128_U4");
    add( 40, KernelVariant::V_32x64x128_U8,  32,  64, 128, 8, "32x64x128_U8");
    add( 41, KernelVariant::V_32x128x32_U2,  32, 128,  32, 2, "32x128x32_U2");
    add( 42, KernelVariant::V_32x128x64_U2,  32, 128,  64, 2, "32x128x64_U2");
    add( 43, KernelVariant::V_32x128x64_U4,  32, 128,  64, 4, "32x128x64_U4");
    add( 44, KernelVariant::V_32x128x128_U2, 32, 128, 128, 2, "32x128x128_U2");
    add( 45, KernelVariant::V_32x128x128_U4, 32, 128, 128, 4, "32x128x128_U4");
    add( 46, KernelVariant::V_32x128x128_U8, 32, 128, 128, 8, "32x128x128_U8");
    add( 47, KernelVariant::V_32x256x32_U2,  32, 256,  32, 2, "32x256x32_U2");
    add( 48, KernelVariant::V_32x256x64_U2,  32, 256,  64, 2, "32x256x64_U2");
    add( 49, KernelVariant::V_32x256x64_U4,  32, 256,  64, 4, "32x256x64_U4");
    add( 50, KernelVariant::V_32x256x128_U2, 32, 256, 128, 2, "32x256x128_U2");
    add( 51, KernelVariant::V_32x256x128_U4, 32, 256, 128, 4, "32x256x128_U4");
    add( 52, KernelVariant::V_64x64x32_U2,   64,  64,  32, 2, "64x64x32_U2");
    add( 53, KernelVariant::V_64x64x64_U2,   64,  64,  64, 2, "64x64x64_U2");
    add( 54, KernelVariant::V_64x64x64_U4,   64,  64,  64, 4, "64x64x64_U4");
    add( 55, KernelVariant::V_64x64x128_U2,  64,  64, 128, 2, "64x64x128_U2");
    add( 56, KernelVariant::V_64x64x128_U4,  64,  64, 128, 4, "64x64x128_U4");
    add( 57, KernelVariant::V_64x64x128_U8,  64,  64, 128, 8, "64x64x128_U8");
    add( 58, KernelVariant::V_64x128x32_U2,  64, 128,  32, 2, "64x128x32_U2");
    add( 59, KernelVariant::V_64x128x64_U2,  64, 128,  64, 2, "64x128x64_U2");
    add( 60, KernelVariant::V_64x128x64_U4,  64, 128,  64, 4, "64x128x64_U4");
    add( 61, KernelVariant::V_64x128x128_U2, 64, 128, 128, 2, "64x128x128_U2");
    add( 62, KernelVariant::V_64x128x128_U4, 64, 128, 128, 4, "64x128x128_U4");
    add( 63, KernelVariant::V_64x128x128_U8, 64, 128, 128, 8, "64x128x128_U8");
    add( 64, KernelVariant::V_64x256x32_U2,  64, 256,  32, 2, "64x256x32_U2");
    add( 65, KernelVariant::V_64x256x64_U2,  64, 256,  64, 2, "64x256x64_U2");
    add( 66, KernelVariant::V_64x256x64_U4,  64, 256,  64, 4, "64x256x64_U4");
    add( 67, KernelVariant::V_64x256x128_U2, 64, 256, 128, 2, "64x256x128_U2");
    add( 68, KernelVariant::V_64x256x128_U4, 64, 256, 128, 4, "64x256x128_U4");
    add( 69, KernelVariant::V_128x128x32_U2, 128, 128,  32, 2, "128x128x32_U2");
    add( 70, KernelVariant::V_128x128x64_U2, 128, 128,  64, 2, "128x128x64_U2");
    add( 71, KernelVariant::V_128x128x64_U4, 128, 128,  64, 4, "128x128x64_U4");
    add( 72, KernelVariant::V_128x128x128_U2,128, 128, 128, 2, "128x128x128_U2");
    add( 73, KernelVariant::V_128x128x128_U4,128, 128, 128, 4, "128x128x128_U4");
    add( 74, KernelVariant::V_128x128x128_U8,128, 128, 128, 8, "128x128x128_U8");
    add( 75, KernelVariant::V_128x256x32_U2, 128, 256,  32, 2, "128x256x32_U2");
    add( 76, KernelVariant::V_128x256x64_U2, 128, 256,  64, 2, "128x256x64_U2");
    add( 77, KernelVariant::V_128x256x64_U4, 128, 256,  64, 4, "128x256x64_U4");
    add( 78, KernelVariant::V_128x256x128_U2,128, 256, 128, 2, "128x256x128_U2");
    add( 79, KernelVariant::V_128x256x128_U4,128, 256, 128, 4, "128x256x128_U4");
    add( 80, KernelVariant::V_256x256x32_U2, 256, 256,  32, 2, "256x256x32_U2");
    add( 81, KernelVariant::V_256x256x64_U2, 256, 256,  64, 2, "256x256x64_U2");
    add( 82, KernelVariant::V_256x256x64_U4, 256, 256,  64, 4, "256x256x64_U4");
    add( 83, KernelVariant::V_256x256x128_U2,256, 256, 128, 2, "256x256x128_U2");
    add( 84, KernelVariant::V_256x256x128_U4,256, 256, 128, 4, "256x256x128_U4");
    return t;
}

inline constexpr auto k_VariantTable = make_variant_table();

// ============================================================================
// Runtime dispatch: select best variant for given {M,N,K} dimensions
// ============================================================================

inline KernelVariant select_best_variant(uint32_t M, uint32_t N, uint32_t K)
{
    KernelVariant best = KernelVariant::V_32x64x64_U4;
    int32_t best_score = -1;

    for(const auto& info : k_VariantTable)
    {
        if((M % info.tile_m) == 0u && (N % info.tile_n) == 0u && (K % info.tile_k) == 0u)
        {
            int32_t score = static_cast<int32_t>(
                (M / info.tile_m) * (N / info.tile_n) * (K / info.tile_k)
                + info.tile_m * info.tile_n * info.unroll_k);
            if(score > best_score) { best_score = score; best = info.id; }
        }
    }
    return best;
}

inline bool variant_compatible(KernelVariant v, uint32_t M, uint32_t N, uint32_t K)
{
    for(const auto& info : k_VariantTable)
    {
        if(info.id == v)
            return (M % info.tile_m) == 0u && (N % info.tile_n) == 0u && (K % info.tile_k) == 0u;
    }
    return false;
}

template<typename Func>
inline void dispatch_variant(KernelVariant v, Func&& body)
{
    switch(v) {
    case KernelVariant::V_16x16x32_U2: body(C16x16x32_U2{}); break;
    case KernelVariant::V_16x16x64_U2: body(C16x16x64_U2{}); break;
    case KernelVariant::V_16x16x64_U4: body(C16x16x64_U4{}); break;
    case KernelVariant::V_16x16x128_U2: body(C16x16x128_U2{}); break;
    case KernelVariant::V_16x16x128_U4: body(C16x16x128_U4{}); break;
    case KernelVariant::V_16x16x128_U8: body(C16x16x128_U8{}); break;
    case KernelVariant::V_16x32x32_U2: body(C16x32x32_U2{}); break;
    case KernelVariant::V_16x32x64_U2: body(C16x32x64_U2{}); break;
    case KernelVariant::V_16x32x64_U4: body(C16x32x64_U4{}); break;
    case KernelVariant::V_16x32x128_U2: body(C16x32x128_U2{}); break;
    case KernelVariant::V_16x32x128_U4: body(C16x32x128_U4{}); break;
    case KernelVariant::V_16x32x128_U8: body(C16x32x128_U8{}); break;
    case KernelVariant::V_16x64x32_U2: body(C16x64x32_U2{}); break;
    case KernelVariant::V_16x64x64_U2: body(C16x64x64_U2{}); break;
    case KernelVariant::V_16x64x64_U4: body(C16x64x64_U4{}); break;
    case KernelVariant::V_16x64x128_U2: body(C16x64x128_U2{}); break;
    case KernelVariant::V_16x64x128_U4: body(C16x64x128_U4{}); break;
    case KernelVariant::V_16x64x128_U8: body(C16x64x128_U8{}); break;
    case KernelVariant::V_16x128x32_U2: body(C16x128x32_U2{}); break;
    case KernelVariant::V_16x128x64_U2: body(C16x128x64_U2{}); break;
    case KernelVariant::V_16x128x64_U4: body(C16x128x64_U4{}); break;
    case KernelVariant::V_16x128x128_U2: body(C16x128x128_U2{}); break;
    case KernelVariant::V_16x128x128_U4: body(C16x128x128_U4{}); break;
    case KernelVariant::V_16x128x128_U8: body(C16x128x128_U8{}); break;
    case KernelVariant::V_16x256x32_U2: body(C16x256x32_U2{}); break;
    case KernelVariant::V_16x256x64_U2: body(C16x256x64_U2{}); break;
    case KernelVariant::V_16x256x64_U4: body(C16x256x64_U4{}); break;
    case KernelVariant::V_16x256x128_U2: body(C16x256x128_U2{}); break;
    case KernelVariant::V_16x256x128_U4: body(C16x256x128_U4{}); break;
    case KernelVariant::V_32x32x32_U2: body(C32x32x32_U2{}); break;
    case KernelVariant::V_32x32x64_U2: body(C32x32x64_U2{}); break;
    case KernelVariant::V_32x32x64_U4: body(C32x32x64_U4{}); break;
    case KernelVariant::V_32x32x128_U2: body(C32x32x128_U2{}); break;
    case KernelVariant::V_32x32x128_U4: body(C32x32x128_U4{}); break;
    case KernelVariant::V_32x32x128_U8: body(C32x32x128_U8{}); break;
    case KernelVariant::V_32x64x32_U2: body(C32x64x32_U2{}); break;
    case KernelVariant::V_32x64x64_U2: body(C32x64x64_U2{}); break;
    case KernelVariant::V_32x64x64_U4: body(C32x64x64_U4{}); break;
    case KernelVariant::V_32x64x128_U2: body(C32x64x128_U2{}); break;
    case KernelVariant::V_32x64x128_U4: body(C32x64x128_U4{}); break;
    case KernelVariant::V_32x64x128_U8: body(C32x64x128_U8{}); break;
    case KernelVariant::V_32x128x32_U2: body(C32x128x32_U2{}); break;
    case KernelVariant::V_32x128x64_U2: body(C32x128x64_U2{}); break;
    case KernelVariant::V_32x128x64_U4: body(C32x128x64_U4{}); break;
    case KernelVariant::V_32x128x128_U2: body(C32x128x128_U2{}); break;
    case KernelVariant::V_32x128x128_U4: body(C32x128x128_U4{}); break;
    case KernelVariant::V_32x128x128_U8: body(C32x128x128_U8{}); break;
    case KernelVariant::V_32x256x32_U2: body(C32x256x32_U2{}); break;
    case KernelVariant::V_32x256x64_U2: body(C32x256x64_U2{}); break;
    case KernelVariant::V_32x256x64_U4: body(C32x256x64_U4{}); break;
    case KernelVariant::V_32x256x128_U2: body(C32x256x128_U2{}); break;
    case KernelVariant::V_32x256x128_U4: body(C32x256x128_U4{}); break;
    case KernelVariant::V_64x64x32_U2: body(C64x64x32_U2{}); break;
    case KernelVariant::V_64x64x64_U2: body(C64x64x64_U2{}); break;
    case KernelVariant::V_64x64x64_U4: body(C64x64x64_U4{}); break;
    case KernelVariant::V_64x64x128_U2: body(C64x64x128_U2{}); break;
    case KernelVariant::V_64x64x128_U4: body(C64x64x128_U4{}); break;
    case KernelVariant::V_64x64x128_U8: body(C64x64x128_U8{}); break;
    case KernelVariant::V_64x128x32_U2: body(C64x128x32_U2{}); break;
    case KernelVariant::V_64x128x64_U2: body(C64x128x64_U2{}); break;
    case KernelVariant::V_64x128x64_U4: body(C64x128x64_U4{}); break;
    case KernelVariant::V_64x128x128_U2: body(C64x128x128_U2{}); break;
    case KernelVariant::V_64x128x128_U4: body(C64x128x128_U4{}); break;
    case KernelVariant::V_64x128x128_U8: body(C64x128x128_U8{}); break;
    case KernelVariant::V_64x256x32_U2: body(C64x256x32_U2{}); break;
    case KernelVariant::V_64x256x64_U2: body(C64x256x64_U2{}); break;
    case KernelVariant::V_64x256x64_U4: body(C64x256x64_U4{}); break;
    case KernelVariant::V_64x256x128_U2: body(C64x256x128_U2{}); break;
    case KernelVariant::V_64x256x128_U4: body(C64x256x128_U4{}); break;
    case KernelVariant::V_128x128x32_U2: body(C128x128x32_U2{}); break;
    case KernelVariant::V_128x128x64_U2: body(C128x128x64_U2{}); break;
    case KernelVariant::V_128x128x64_U4: body(C128x128x64_U4{}); break;
    case KernelVariant::V_128x128x128_U2: body(C128x128x128_U2{}); break;
    case KernelVariant::V_128x128x128_U4: body(C128x128x128_U4{}); break;
    case KernelVariant::V_128x128x128_U8: body(C128x128x128_U8{}); break;
    case KernelVariant::V_128x256x32_U2: body(C128x256x32_U2{}); break;
    case KernelVariant::V_128x256x64_U2: body(C128x256x64_U2{}); break;
    case KernelVariant::V_128x256x64_U4: body(C128x256x64_U4{}); break;
    case KernelVariant::V_128x256x128_U2: body(C128x256x128_U2{}); break;
    case KernelVariant::V_128x256x128_U4: body(C128x256x128_U4{}); break;
    case KernelVariant::V_256x256x32_U2: body(C256x256x32_U2{}); break;
    case KernelVariant::V_256x256x64_U2: body(C256x256x64_U2{}); break;
    case KernelVariant::V_256x256x64_U4: body(C256x256x64_U4{}); break;
    case KernelVariant::V_256x256x128_U2: body(C256x256x128_U2{}); break;
    case KernelVariant::V_256x256x128_U4: body(C256x256x128_U4{}); break;
    default: break;
    }
}

inline const char* variant_name(KernelVariant v)
{
    for(const auto& info : k_VariantTable)
        if(info.id == v) return info.name;
    return "unknown";
}

inline uint32_t variant_tile_m(KernelVariant v)
{
    for(const auto& info : k_VariantTable)
        if(info.id == v) return info.tile_m;
    return 0;
}
inline uint32_t variant_tile_n(KernelVariant v)
{
    for(const auto& info : k_VariantTable)
        if(info.id == v) return info.tile_n;
    return 0;
}
inline uint32_t variant_tile_k(KernelVariant v)
{
    for(const auto& info : k_VariantTable)
        if(info.id == v) return info.tile_k;
    return 0;
}
inline uint32_t variant_unroll_k(KernelVariant v)
{
    for(const auto& info : k_VariantTable)
        if(info.id == v) return info.unroll_k;
    return 0;
}
inline uint32_t variant_lds_bytes(KernelVariant v)
{
    for(const auto& info : k_VariantTable)
        if(info.id == v) return info.lds_bytes;
    return 0;
}

// ============================================================================
// Runner: launch kernel variant, measure, validate.
// ============================================================================

template<typename C>
void launch_variant(
    KernelVariant  v,
    uint32_t       M, uint32_t       N, uint32_t    K,
    const int8_t*  A, const int8_t*  B,
    const int32_t* Cmat, int32_t*    D,
    uint32_t       lda, uint32_t     ldb,
    uint32_t       ldc, uint32_t     ldd,
    int32_t        alpha, int32_t    beta,
    CombinedOperandProbe* probe,
    hipStream_t    stream)
{
    const dim3 block(C::tblock_x, C::tblock_y, 1);
    const dim3 grid((M + C::tile_m - 1u) / C::tile_m,
                    (N + C::tile_n - 1u) / C::tile_n, 1);

    constexpr uint32_t shared_bytes = C::lds_bytes_total;

    hipFuncAttributes attr{};
    hipError_t attrStatus = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(i8gemm_kernel<C>));
    if(attrStatus != hipSuccess)
    {
        std::cerr << "HIP error: '"
                  << hipGetErrorString(attrStatus)
                  << "' while querying attributes for variant "
                  << variant_name(v) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int device = 0;
    CHECK_HIP_ERROR(hipGetDevice(&device));

    int maxBlockShared = 0;
    CHECK_HIP_ERROR(hipDeviceGetAttribute(
        &maxBlockShared,
        hipDeviceAttributeMaxSharedMemoryPerBlock,
        device));

    uint32_t maxDynamicShared = attr.maxDynamicSharedSizeBytes > 0
        ? static_cast<uint32_t>(attr.maxDynamicSharedSizeBytes)
        : static_cast<uint32_t>(maxBlockShared);
    if(shared_bytes > maxDynamicShared)
    {
        std::cerr << "Variant " << variant_name(v)
                  << " requires " << shared_bytes
                  << " bytes dynamic shared memory, exceeding kernel/device limit "
                  << maxDynamicShared << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if(attr.maxDynamicSharedSizeBytes > 0)
    {
        hipError_t setAttrStatus = hipFuncSetAttribute(
            reinterpret_cast<const void*>(i8gemm_kernel<C>),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes);
        if(setAttrStatus != hipSuccess)
        {
            std::cerr << "HIP error: '"
                      << hipGetErrorString(setAttrStatus)
                      << "' while opting in dynamic shared memory for variant "
                      << variant_name(v)
                      << " (" << shared_bytes << " bytes requested)"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    hipLaunchKernelGGL((i8gemm_kernel<C>), grid, block, shared_bytes, stream,
                       M, N, K, A, B, Cmat, D, lda, ldb, ldc, ldd, alpha, beta, probe);
    CHECK_HIP_ERROR(hipGetLastError());
}

inline void launch_dispatch(
    KernelVariant  v,
    uint32_t       M, uint32_t       N, uint32_t    K,
    const int8_t*  A, const int8_t*  B,
    const int32_t* Cmat, int32_t*    D,
    uint32_t       lda, uint32_t     ldb,
    uint32_t       ldc, uint32_t     ldd,
    int32_t        alpha, int32_t    beta,
    CombinedOperandProbe* probe,
    hipStream_t    stream)
{
    dispatch_variant(v, [&](auto tag) {
        using Cfg = typename std::decay_t<decltype(tag)>;
        launch_variant<Cfg>(v, M, N, K, A, B, Cmat, D, lda, ldb, ldc, ldd, alpha, beta, probe, stream);
    });
}

template<typename C>
inline void ensure_launch_size(KernelVariant v, uint32_t grid_m, uint32_t grid_n,
    uint32_t block_x, uint32_t block_y, uint32_t smem)
{
    (void)v; (void)grid_m; (void)grid_n; (void)block_x; (void)block_y; (void)smem;
}

// ============================================================================
// Printing helpers
// ============================================================================

inline void print_combined_header()
{
    std::cout << std::left
              << std::setw(22) << "Variant"
              << std::setw(6) << "OK"
              << std::setw(8) << "TileM"
              << std::setw(8) << "TileN"
              << std::setw(8) << "TileK"
              << std::setw(8) << "Unroll"
              << std::setw(8) << "LdsKB"
              << std::setw(13) << "elapsedMs"
              << std::setw(13) << "perEvalMs"
              << std::setw(13) << "GOp"
              << std::setw(13) << "TOp/s"
              << std::setw(13) << "TMac/s"
              << std::endl;
}

struct BenchmarkStats
{
    double elapsedMs;
    double perEvalMs;
    double gops;
    double topPerSec;
    double tmacPerSec;
};

inline BenchmarkStats compute_stats(uint32_t M, uint32_t N, uint32_t K,
    uint32_t runs, float elapsedMs)
{
    const double gopsVal = 2.0 * static_cast<double>(M) * N * K * 1.0e-9;
    const double perEval = static_cast<double>(elapsedMs) / static_cast<double>(runs);
    const double tops    = gopsVal / static_cast<double>(elapsedMs) * static_cast<double>(runs);
    const double tmacs   = static_cast<double>(M) * N * K / (perEval * 1.0e-3) * 1.0e-12;
    return {static_cast<double>(elapsedMs), perEval, gopsVal, tops, tmacs};
}

inline void print_result(const char* name, bool ok, uint32_t tile_m, uint32_t tile_n,
    uint32_t tile_k, uint32_t unroll, uint32_t lds_kb, const BenchmarkStats& s)
{
    std::cout << std::left
              << std::setw(22) << name
              << std::setw(6)  << (ok ? "yes" : "no")
              << std::setw(8)  << tile_m
              << std::setw(8)  << tile_n
              << std::setw(8)  << tile_k
              << std::setw(8)  << unroll
              << std::setw(8)  << lds_kb
              << std::setw(13) << s.elapsedMs
              << std::setw(13) << s.perEvalMs
              << std::setw(13) << s.gops
              << std::setw(13) << s.topPerSec
              << std::setw(13) << s.tmacPerSec
              << std::endl;
}
