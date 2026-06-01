/*******************************************************************************
 *
 * Combined variant system for perf_i8gemm_custom.cpp — Tensile ASM mirror.
 *
 * The compute path is structurally isomorphic to what Tensile's KernelWriterAssembly
 * produces for MT64x64x32_MI16x16x16x1 on gfx1100 (see rocblas_i8i_0x4b2900_full.txt).
 *
 * Every Tensile instruction has a corresponding C++ intrinsic/asm line.
 *
 ******************************************************************************/

#pragma once

#include <cstdint>

// ============================================================================
// VGPR abstraction types — map directly to contiguous architectural VGPRs
// ============================================================================

using VRegI32   = int32_t;
using VRegI32x4 = int32_t __attribute__((__vector_size__(16)));
using VRegI32x8 = int32_t __attribute__((__vector_size__(32)));

// ============================================================================
// CombinedParams — unchanged geometry parameterization
// ============================================================================

template <uint32_t BlocksM,
          uint32_t BlocksN,
          uint32_t TBlockX,
          uint32_t TBlockY,
          uint32_t KGroup,
          uint32_t MSubpasses = 1u,
          uint32_t BStride   = 0u,
          uint32_t LdsPad    = 0u,
          uint32_t PcSplit   = 0u,
          uint32_t RocwmmaK  = kRocwmmaK,
          uint32_t LdsPadA   = 0u,
          uint32_t LdsPadB   = LdsPad,
          uint32_t NSubpasses = 1u,
          typename SchedulerT = rocwmma::fragment_scheduler::coop_row_major_2d<TBlockX, TBlockY>>
struct CombinedParams
{
    using DataLayoutA   = rocwmma::col_major;
    using DataLayoutB   = rocwmma::row_major;
    using DataLayoutC   = rocwmma::row_major;
    using DataLayoutLds = rocwmma::col_major;

    static constexpr uint32_t ROCWMMA_M = kRocwmmaM;
    static constexpr uint32_t ROCWMMA_N = kRocwmmaN;
    static constexpr uint32_t ROCWMMA_K = RocwmmaK;
    static constexpr uint32_t BLOCKS_M  = BlocksM;
    static constexpr uint32_t BLOCKS_N  = BlocksN;
    static constexpr uint32_t TBLOCK_X  = TBlockX;
    static constexpr uint32_t TBLOCK_Y  = TBlockY;
    static constexpr uint32_t WARP_SIZE = kWarpSize;
    static constexpr uint32_t K_GROUP   = KGroup;
    static constexpr uint32_t LDS_PAD_A = LdsPadA;
    static constexpr uint32_t LDS_PAD_B = LdsPadB;
    static constexpr uint32_t LDS_PAD   = LDS_PAD_A + LDS_PAD_B;
    static constexpr uint32_t M_SUBPASSES = MSubpasses;
    static constexpr uint32_t N_SUBPASSES = NSubpasses;
    static constexpr uint32_t B_STRIDE    = BStride;
    static constexpr uint32_t PC_SPLIT    = PcSplit;

    static constexpr uint32_t WARP_TILE_M  = BLOCKS_M * ROCWMMA_M;
    static constexpr uint32_t WARP_TILE_N  = BLOCKS_N * ROCWMMA_N;
    static constexpr uint32_t WARP_TILE_K  = ROCWMMA_K;
    static constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE;
    static constexpr uint32_t WARPS_N      = TBLOCK_Y;
    static constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M;
    static constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N;
    static constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;
    static constexpr uint32_t WAVE_COUNT   = TBLOCK_X * TBLOCK_Y / WARP_SIZE;
    static constexpr uint32_t CONSUMER_WAVES = (PC_SPLIT > 0u) ? (WAVE_COUNT / 2u) : WAVE_COUNT;
    static constexpr uint32_t PRODUCER_WAVES = (PC_SPLIT > 0u) ? (WAVE_COUNT - CONSUMER_WAVES) : 0u;
    static_assert((PC_SPLIT == 0u) || (CONSUMER_WAVES >= 2u),
                  "PC_SPLIT needs at least 2 consumer waves");
    static_assert((PC_SPLIT == 0u) || (PRODUCER_WAVES >= 2u),
                  "PC_SPLIT needs at least 2 producer waves");
    static_assert(CONSUMER_WAVES % WARPS_N == 0u,
                  "Consumer waves must be evenly divisible by WARPS_N");
    static constexpr uint32_t ACTIVE_WARPS_M = CONSUMER_WAVES / WARPS_N;

    static constexpr uint32_t EFF_B_STRIDE =
        (B_STRIDE > 0u) ? B_STRIDE : MACRO_TILE_K;

    static constexpr uint32_t EFF_MACRO_TILE_M =
        M_SUBPASSES * ACTIVE_WARPS_M * WARP_TILE_M;
    static constexpr uint32_t EFF_MACRO_TILE_N =
        N_SUBPASSES * WARPS_N * WARP_TILE_N;

    using MmaFragA   = rocwmma::fragment<rocwmma::matrix_a, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutA>;
    using MmaFragB   = rocwmma::fragment<rocwmma::matrix_b, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutB>;
    using MmaFragC   = rocwmma::fragment<rocwmma::accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, OutputT, DataLayoutC>;
    using MmaFragD   = MmaFragC;
    using MmaFragAcc = rocwmma::fragment<rocwmma::accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT>;

    using CoopScheduler = SchedulerT;
    using GRFragA = rocwmma::fragment<rocwmma::matrix_a, EFF_MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                                      InputT, DataLayoutA, CoopScheduler>;
    using GRFragB = rocwmma::fragment<rocwmma::matrix_b, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K,
                                      InputT, DataLayoutB, CoopScheduler>;

    using LWFragA = rocwmma::apply_data_layout_t<GRFragA, DataLayoutLds>;
    using LWFragB = rocwmma::apply_data_layout_t<rocwmma::apply_transpose_t<GRFragB>, DataLayoutLds>;
    using LRFragA = rocwmma::apply_data_layout_t<MmaFragA, DataLayoutLds>;
    using LRFragB = rocwmma::apply_data_layout_t<rocwmma::apply_transpose_t<MmaFragB>, DataLayoutLds>;
};

// ============================================================================
// TensileAcc<P> — accumulator layout trait. Mirror of Tensile ISA lines 182-213:
//   v_mov_b32_e32 v0,  0
//   ...
//   v_mov_b32_e32 v31, 0   (32 mov's for 4 accum groups of 8 VGPRs each)
//
// For BLOCKS_M × BLOCKS_N accumulator groups, each group = VRegI32x8 (8 VGPRs).
// Total: BLOCKS_M * BLOCKS_N * 8 VGPRs allocated contiguously.
// ============================================================================

template <typename P>
struct TensileAcc
{
    static constexpr uint32_t NAcc        = P::BLOCKS_M * P::BLOCKS_N;
    static constexpr uint32_t VgprPerAcc  = 8u;
    static constexpr uint32_t TotalVgprs  = NAcc * VgprPerAcc;
    using GroupT = VRegI32x8;
    static_assert(NAcc >= 1u && NAcc <= 16u, "Accumulator groups must be 1-16");
};

// ============================================================================
// s_setprio — CU scheduler priority. Mirror of Tensile ISA.
//   s_setprio 3  — highest priority for compute burst
//   s_setprio 1  — low priority after compute
//   s_setprio 2  — medium priority during prefetch
// ============================================================================

__device__ inline void tile_setprio_3() { asm volatile("s_setprio 3" ::: "memory"); }
__device__ inline void tile_setprio_2() { asm volatile("s_setprio 2" ::: "memory"); }
__device__ inline void tile_setprio_1() { asm volatile("s_setprio 1" ::: "memory"); }

// ============================================================================
// s_waitcnt + s_barrier — precise wait control. Mirror of Tensile ISA.
// ============================================================================

__device__ inline void tile_wait_lgkmcnt0()  { __builtin_amdgcn_s_waitcnt(0xF700); }
__device__ inline void tile_wait_vmcnt1()    { __builtin_amdgcn_s_waitcnt(0xEF00); }
__device__ inline void tile_wait_vmcnt0()    { __builtin_amdgcn_s_waitcnt(0xFF00); }
__device__ inline void tile_wait_vmcnt0_lgkmcnt0() { __builtin_amdgcn_s_waitcnt(0x0000); }
__device__ inline void tile_s_barrier()      { __builtin_amdgcn_s_barrier(); }

// ============================================================================
// v_lshl_or_b32 — byte packing for A operands. Mirror of Tensile ISA lines 318-341.
//   v_lshl_or_b32 v34, v83, 8, v34    ; dst = (hi8 << 8) | lo8
//   ...
//   8 instructions for first A half, 8 more interleaved with WMMA for second half.
// ============================================================================

__device__ inline VRegI32 tile_lshl_or(VRegI32 hi_val, VRegI32 lo_val)
{
    VRegI32 result;
    asm volatile("v_lshl_or_b32 %0, %1, 8, %2"
                 : "=v"(result) : "v"(hi_val), "v"(lo_val));
    return result;
}

// ============================================================================
// ds_read_b128 — B LDS load. Mirror of Tensile ISA lines 281-282, 315-316.
//   ds_load_b128 v[51:54], v81              ; offset:0     — bank group 0
//   ds_load_b128 v[55:58], v81 offset:1152  ; offset:1152  — bank group 1
//   ds_load_b128 v[59:62], v81 offset:16    ; offset:16    — bank group 0, second
//   ds_load_b128 v[63:66], v81 offset:1168  ; offset:1168  — bank group 1, second
// ============================================================================

__device__ inline VRegI32x4 tile_ds_read_b128(VRegI32 ldsAddr, uint32_t byteOffset)
{
    VRegI32 base = ldsAddr + byteOffset;
    VRegI32x4 result;
    asm volatile("ds_read_b128 %0, %1" : "=v"(result) : "v"(base) : "memory");
    return result;
}

// ============================================================================
// ds_read_u8 / ds_read_u8_d16_hi — A LDS byte loads. Mirror of Tensile ISA lines 249-314.
//   ds_load_u8 v34, v80                    ; byte into bits[7:0]
//   ds_load_u8_d16_hi v34, v80 offset:128  ; byte into bits[23:16]
//
// Each pair of ds_load_u8 + ds_load_u8_d16_hi fills 2 bytes in 2 registers,
// to be packed with v_lshl_or_b32.
// ============================================================================

__device__ inline void tile_ds_read_u8_pair(
    VRegI32& lo_vgpr, VRegI32& hi_tmp,
    VRegI32 ldsBase, uint32_t offLo, uint32_t offHi)
{
    lo_vgpr = 0; hi_tmp = 0;
    VRegI32 addrLo = ldsBase + offLo;
    VRegI32 addrHi = ldsBase + offHi;
    asm volatile("ds_read_u8 %0, %2" : "=v"(lo_vgpr) : "v"(lo_vgpr), "v"(addrLo) : "memory");
    asm volatile("ds_read_u8 %0, %2" : "=v"(hi_tmp) : "v"(hi_tmp), "v"(addrHi) : "memory");
}

__device__ inline void tile_ds_read_u8_d16_hi_pair(
    VRegI32& lo_vgpr, VRegI32& hi_tmp,
    VRegI32 ldsBase, uint32_t offLoD16, uint32_t offHiD16)
{
    VRegI32 addrLo = ldsBase + offLoD16;
    VRegI32 addrHi = ldsBase + offHiD16;
    asm volatile("ds_read_u8_d16_hi %0, %2" : "=v"(lo_vgpr) : "v"(lo_vgpr), "v"(addrLo) : "memory");
    asm volatile("ds_read_u8_d16_hi %0, %2" : "=v"(hi_tmp) : "v"(hi_tmp), "v"(addrHi) : "memory");
}

// ============================================================================
// ds_write_b128 — LDS store for global→LDS prefetch. Mirror of Tensile ISA lines 224-225.
//   ds_store_b128 v67, v[72:75]    ; A staging VGPRs → LDS
//   ds_store_b128 v68, v[76:79]    ; B staging VGPRs → LDS
// ============================================================================

__device__ inline void tile_ds_write_b128(VRegI32 ldsAddr, VRegI32x4 data)
{
    asm volatile("ds_write_b128 %0, %1" :: "v"(ldsAddr), "v"(data) : "memory");
}

// ============================================================================
// global_load_dword — scalarized global loads (for pre-loading A staging VGPRs)
// ============================================================================

__device__ inline VRegI32 tile_global_load_dword(const InputT* ptr)
{
    VRegI32 result;
    asm volatile("global_load_dword %0, %1, off" : "=v"(result) : "v"(ptr) : "memory");
    return result;
}

// ============================================================================
// v_wmma_i32_16x16x16_iu8 — WMMA compute intrinsic. Mirror of Tensile ISA lines 327-342.
//
//   v_wmma_i32_16x16x16_iu8 v[0:7], v[51:54], v[34:37], v[0:7]  ; acc0 += b0*a0
//   v_wmma_i32_16x16x16_iu8 v[8:15], v[51:54], v[38:41], v[8:15] ; acc1 += b0*a1
//   ...
//
// 8 WMMA instructions total in the burst = 4 accumulator groups × 2 K-substeps.
// ============================================================================

__device__ inline VRegI32x8 tile_wmma_iu8(VRegI32x4 a_op, VRegI32x4 b_op, VRegI32x8 acc)
{
    VRegI32x8 result = acc;
    asm volatile("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %3"
        : "+v"(result) : "v"(b_op), "v"(a_op), "v"(acc));
    return result;
}


// ============================================================================
// Tensile-style A LDS byte loading for one K-step.
//
// In Tensile MT64x64x32, A data in LDS is col_major.
// For a warp with BLOCKS_M=2 (32 rows), the A tile has 2 column-groups of 16 rows.
// Each lane loads 32 bytes total: 8 bytes per column-group × 2 groups × 2 halves.
//
// The A load pattern in Tensile ISA uses v80 as base address with fixed offsets:
//   Group 0, half 0: offsets 0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960
//   Group 0, half 1: offsets 1, 65, 129, 193, ... (offset + 1 from group 0 half 0)
//   Group 1, half 0: offsets 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984
//   Group 1, half 1: offsets 1025, 1089, 1153, 1217, ...
//
// These are 16×64 + 0 or 1 = 1024 or 1025 for the second row-block.
// Each row-block is 16 rows × 64 cols = 1024 bytes.
// The +0/+1 toggle is the "within-dword byte offset" (whether we read the even or odd byte from the 2-byte-aligned pair in memory).
//
// Simplified: for each of 16 column positions (8 per group × 2 groups), do 2 byte reads.
// The LDS base for A is the warp's starting position in the A LDS region.
// ============================================================================

template <typename P>
__device__ inline void tile_load_lds_a_bytes_16(
    VRegI32 (&a_lo)[16],
    VRegI32 (&a_hi)[16],
    VRegI32 ldsBase,
    uint32_t laneId,
    uint32_t /* warpRowM */,
    uint32_t /* warpColN */)
{
    constexpr uint32_t Mwarp = P::WARP_TILE_M;   // 32
    constexpr uint32_t Mblk  = P::ROCWMMA_M;     // 16
    constexpr uint32_t Kblk  = P::ROCWMMA_K;     // 16
    constexpr uint32_t Nblk  = P::ROCWMMA_N;     // 16
    (void)Kblk; (void)Nblk; (void)Mblk;

    uint32_t lid = laneId;
    uint32_t rowInBlk = lid & 0xF;
    uint32_t blkIdx = (lid >> 4) & 0x1;

    for(uint32_t col = 0; col < 8; ++col)
    {
        uint32_t byteRowLo = blkIdx * Mblk + rowInBlk;
        uint32_t byteCol   = col;
        uint32_t offLo = byteRowLo + byteCol * Mwarp;

        uint32_t offHi = offLo + 1;

        uint32_t idx = col + (blkIdx * 8);
        tile_ds_read_u8_pair(a_lo[idx], a_hi[idx], ldsBase, offLo, offHi);
    }

    for(uint32_t col = 0; col < 8; ++col)
    {
        uint32_t byteRowLo = blkIdx * Mblk + rowInBlk;
        uint32_t byteCol   = col;
        uint32_t offLoD16 = byteRowLo + byteCol * Mwarp + 128;
        uint32_t offHiD16 = offLoD16 + 1;

        uint32_t idx = col + (blkIdx * 8);
        tile_ds_read_u8_d16_hi_pair(a_lo[idx], a_hi[idx], ldsBase, offLoD16, offHiD16);
    }
}

// ============================================================================
// Tensile-style B LDS loading for one K-step.
//
//   ds_load_b128 v[51:54], v81              ; b0 = B group 0, block 0 (offset 0)
//   ds_load_b128 v[55:58], v81 offset:1152  ; b1 = B group 1, block 0 (offset 1152)
//   ds_load_b128 v[59:62], v81 offset:16    ; b2 = B group 0, block 1 (offset 16)
//   ds_load_b128 v[63:66], v81 offset:1168  ; b3 = B group 1, block 1 (offset 1168)
//
// B in LDS is the transpose of B (row-major B stored col-major = N rows, K cols).
// Offset 1152 = 16 rows × 72 cols × 4 bytes... actually 1152 / 4 = 288 = 9*32.
// For BLOCKS_N=2 with B_stride_padding, the offsets are derived from LDS geometry.
// ============================================================================

__device__ inline void tile_load_lds_b_4x128(
    VRegI32x4& b0, VRegI32x4& b1,
    VRegI32x4& b2, VRegI32x4& b3,
    VRegI32 ldsBaseB,
    uint32_t off0, uint32_t off1152,
    uint32_t off16, uint32_t off1168)
{
    b0 = tile_ds_read_b128(ldsBaseB, off0);
    b1 = tile_ds_read_b128(ldsBaseB, off1152);
    b2 = tile_ds_read_b128(ldsBaseB, off16);
    b3 = tile_ds_read_b128(ldsBaseB, off1168);
}

// ============================================================================
// Accumulator zero-init — 32 v_mov_b32_e32 mirrored from Tensile ISA lines 182-213.
// The explicit VGPR loop produces sequential v_mov_b32 instructions for each VGPR.
// ============================================================================

template <typename P>
__device__ inline void tile_acc_zero(typename TensileAcc<P>::GroupT (&acc)[TensileAcc<P>::NAcc])
{
    constexpr uint32_t N = TensileAcc<P>::NAcc;
    #pragma unroll
    for(uint32_t i = 0; i < N; ++i)
    {
        #pragma unroll
        for(uint32_t j = 0; j < TensileAcc<P>::VgprPerAcc; ++j)
        {
            acc[i][j] = 0;
        }
    }
}

// ============================================================================
// A operand packing — 8 v_lshl_or_b32 for first half, 8 for second half.
//
// Mirrors Tensile ISA lines 318-341:
//   v_lshl_or_b32 v34, v83, 8, v34     → a0[0] = lane byte1 << 8 | lane byte0
//   v_lshl_or_b32 v35, v84, 8, v35     → a0[1]
//   ...
//   v_lshl_or_b32 v41, v90, 8, v41     → a1[3]
//
//   (interleaved with WMMA after s_setprio 3)
//
//   v_lshl_or_b32 v42, v91, 8, v42     → a2[0]
//   ...
//   v_lshl_or_b32 v49, v98, 8, v49     → a3[3]
//
// First 8 done BEFORE setprio+WMMA. Next 8 done INTERLEAVED with WMMA calls.
// ============================================================================

__device__ inline void tile_pack_a_first_half(
    const VRegI32 (&a_lo)[16], const VRegI32 (&a_hi)[16],
    VRegI32x4& a0, VRegI32x4& a1)
{
    a0[0] = tile_lshl_or(a_hi[0], a_lo[0]);
    a0[1] = tile_lshl_or(a_hi[1], a_lo[1]);
    a0[2] = tile_lshl_or(a_hi[2], a_lo[2]);
    a0[3] = tile_lshl_or(a_hi[3], a_lo[3]);

    a1[0] = tile_lshl_or(a_hi[4], a_lo[4]);
    a1[1] = tile_lshl_or(a_hi[5], a_lo[5]);
    a1[2] = tile_lshl_or(a_hi[6], a_lo[6]);
    a1[3] = tile_lshl_or(a_hi[7], a_lo[7]);
}

__device__ inline void tile_pack_a_second_half(
    const VRegI32 (&a_lo)[16], const VRegI32 (&a_hi)[16],
    VRegI32x4& a2, VRegI32x4& a3)
{
    a2[0] = tile_lshl_or(a_hi[8],  a_lo[8]);
    a2[1] = tile_lshl_or(a_hi[9],  a_lo[9]);
    a2[2] = tile_lshl_or(a_hi[10], a_lo[10]);
    a2[3] = tile_lshl_or(a_hi[11], a_lo[11]);

    a3[0] = tile_lshl_or(a_hi[12], a_lo[12]);
    a3[1] = tile_lshl_or(a_hi[13], a_lo[13]);
    a3[2] = tile_lshl_or(a_hi[14], a_lo[14]);
    a3[3] = tile_lshl_or(a_hi[15], a_lo[15]);
}

// ============================================================================
// cmb_compute_tensile_kstep<P>() — one K-substep of the Tensile inner loop.
//
// This function implements one WMMA burst of 4 instructions using specific
// B and A operands. Called twice per K-group (for K_SUBSTEP 0 and 1).
//
// Maps to Tensile ISA lines 327-342:
//   v_wmma_i32_16x16x16_iu8 v[0:7],   v[51:54], v[34:37], v[0:7]   ; acc00 += b0*a0
//   v_wmma_i32_16x16x16_iu8 v[8:15],  v[51:54], v[38:41], v[8:15]  ; acc01 += b0*a1
//   v_wmma_i32_16x16x16_iu8 v[16:23], v[55:58], v[34:37], v[16:23] ; acc10 += b1*a0
//   v_wmma_i32_16x16x16_iu8 v[24:31], v[55:58], v[38:41], v[24:31] ; acc11 += b1*a1
//
// Parameters:
//   acc00, acc01, acc10, acc11  — 4 accumulator groups (in/out)
//   a0, a1                      — first-half A operands for this K-step
//   b_even, b_odd               — B operands (b0/b2 or b1/b3 depending on K-step)
// ============================================================================

__device__ inline void cmb_compute_tensile_kstep_w4(
    VRegI32x8& acc00,
    VRegI32x8& acc10,
    VRegI32x8& acc01,
    VRegI32x8& acc11,
    const VRegI32x4& a0,
    const VRegI32x4& a1,
    const VRegI32x4& b_even,
    const VRegI32x4& b_odd)
{
    acc00 = tile_wmma_iu8(a0, b_even, acc00);
    acc01 = tile_wmma_iu8(a1, b_even, acc01);
    acc10 = tile_wmma_iu8(a0, b_odd,  acc10);
    acc11 = tile_wmma_iu8(a1, b_odd,  acc11);
}

// ============================================================================
// cmb_compute_tensile_burst<P>() — full WMMA burst for ONE K-group.
//
// This function implements the complete 8-instruction WMMA burst
// (2 K-substeps × 4 accumulator groups) as seen in the Tensile ISA.
//
// Sequence:
//   K-substep 0: 4 WMMA calls (a0,a1 with b0,b1)
//   K-substep 1: 4 WMMA calls (a2,a3 with b2,b3)
//   A-packing for second half interleaved between WMMA calls
//
// Mirrors Tensile ISA line sequence 327-342:
//   1. v_wmma_i32_16x16x16_iu8 acc00, b0, a0, acc00
//   2. v_wmma_i32_16x16x16_iu8 acc01, b0, a1, acc01
//   3. v_wmma_i32_16x16x16_iu8 acc10, b1, a0, acc10
//   4. v_wmma_i32_16x16x16_iu8 acc11, b1, a1, acc11
//   5. v_wmma_i32_16x16x16_iu8 acc00, b2, a2, acc00
//   6. v_wmma_i32_16x16x16_iu8 acc01, b2, a3, acc01
//   7. v_wmma_i32_16x16x16_iu8 acc10, b3, a2, acc10
//   8. v_wmma_i32_16x16x16_iu8 acc11, b3, a3, acc11
// ============================================================================

__device__ inline void cmb_compute_tensile_burst(
    VRegI32x8& acc00,
    VRegI32x8& acc01,
    VRegI32x8& acc10,
    VRegI32x8& acc11,
    const VRegI32x4& a0,
    const VRegI32x4& a1,
    const VRegI32x4& a2,
    const VRegI32x4& a3,
    const VRegI32x4& b0,
    const VRegI32x4& b1,
    const VRegI32x4& b2,
    const VRegI32x4& b3)
{
    cmb_compute_tensile_kstep_w4(acc00, acc10, acc01, acc11, a0, a1, b0, b1);
    cmb_compute_tensile_kstep_w4(acc00, acc10, acc01, acc11, a2, a3, b2, b3);
}

// ============================================================================
// cmb_compute_tensile_group<P>() — one full K_GROUP cycle.
//
// This function implements the complete inner-loop body from the Tensile ISA,
// structurally decomposed into:
//
//   1. B LDS loads:  4 × ds_load_b128 (lines 281-282, 315-316)
//   2. A LDS loads: 32 × ds_load_u8/ds_load_u8_d16_hi (lines 249-314)
//   3. s_waitcnt lgkmcnt(0) (line 317)
//   4. A packing first half: 8 × v_lshl_or_b32 (lines 318-325)
//   5. s_setprio 3 (line 326)
//   6. WMMA burst: 8 × v_wmma_i32_16x16x16_iu8 (lines 327-342)
//      with second-half A packing interleaved after first WMMA
//   7. s_setprio 1 (line 343)
//
// Parameters:
//   acc00..acc11  — flat i32x8 accumulators (4 groups for 2×2 blocks)
//   ldsA          — byte-aligned LDS base for A
//   ldsB          — dword-aligned LDS base for B
//   laneId        — lane index within warp (0-31)
//
// LDS B offsets computed from LDS geometry:
//   For col_major LDS with ldsWidthB stride:
//   Each B column block is ROCWMMA_N rows × ROCWMMA_K cols
//   Block stride between N-groups = ROCWMMA_N * sizeof(InputT) * ldsHeightB
//   Actually B is stored as transpose: [N][K] in row-major LDS
//   → So col_major LDS has K as leading dimension
//   → B[n][k] at offset = n * ldsStrideB + k * sizeof(element)
//
// Simplified: compute offsets directly from the LDS base + known geometry.
// ============================================================================

template <typename P>
__device__ inline void cmb_compute_tensile_group(
    VRegI32x8& acc00,
    VRegI32x8& acc01,
    VRegI32x8& acc10,
    VRegI32x8& acc11,
    VRegI32 ldsA,
    VRegI32 ldsB,
    uint32_t laneId,
    VRegI32 /* ldsBoff0 */,
    VRegI32 /* ldsBoff1152 */,
    VRegI32 /* ldsBoff16 */,
    VRegI32 /* ldsBoff1168 */)
{
    // === Stage variables: mirror Tensile VGPR ranges ===
    VRegI32   a_lo[16];      // A byte loads: low bytes (v34-v41, v42-v49)
    VRegI32   a_hi[16];      // A byte loads: hi bytes (v83-v90, v91-v98)
    VRegI32x4 b0, b1, b2, b3; // B dword loads (v[51:54], v[55:58], v[59:62], v[63:66])
    VRegI32x4 a0, a1, a2, a3; // Packed A operands (v[34:37], v[38:41], v[42:45], v[46:49])

    // === 1. B LDS loads: 4 × ds_load_b128 ===
    // Mirror: ds_load_b128 v[51:54], v81              offset:0
    //         ds_load_b128 v[55:58], v81 offset:1152  offset:1152
    //         ds_load_b128 v[59:62], v81 offset:16    offset:16
    //         ds_load_b128 v[63:66], v81 offset:1168  offset:1168
    b0 = tile_ds_read_b128(ldsB, 0);
    b1 = tile_ds_read_b128(ldsB, 1152);
    b2 = tile_ds_read_b128(ldsB, 16);
    b3 = tile_ds_read_b128(ldsB, 1168);

    // === 2. A LDS loads: 32 × ds_load_u8 / ds_load_u8_d16_hi ===
    // Mirror: ds_load_u8 v34, v80                         offset:0
    //         ds_load_u8 v83, v80 offset:64               offset:64
    //         ds_load_u8_d16_hi v34, v80 offset:128       offset:128
    //         ds_load_u8_d16_hi v83, v80 offset:192       offset:192
    //         ... (16 instructions per half-group, 4 half-groups)
    //
    // For BLOCKS_M=2, the A LDS holds 32 rows × 16 cols.
    // Each lane loads 16 bytes: byte pairs split across even/odd byte offsets.
    // The loader reads all 32 bytes into a_lo[0..15] and a_hi[0..15].
    //
    // laneId selects which rows each thread loads.

    {
        uint32_t lid     = laneId;
        uint32_t row     = lid & 0x1F;               // which M row (0-31)
        uint32_t blkRow  = row / P::ROCWMMA_M;       // 0 or 1 for BLOCKS_M=2
        uint32_t rowInBl = row % P::ROCWMMA_M;       // 0-15 within block

        // Each A tile has ROCWMMA_K=16 columns in LDS
        // Thread loads bytes for its rows from all K columns
        #pragma unroll
        for(uint32_t k = 0; k < 8u; ++k)
        {
            // Index into a_lo/a_hi: 2 groups × 8 per K-step
            uint32_t idx        = k + blkRow * 8u;
            uint32_t kByteCol   = k;          // column within A tile (0-7)
            uint32_t kByteColHi = k + 8u;     // column 8-15 for second half of the K tile

            // A stored col_major: offset = row + col * Mwarp
            uint32_t offLo00  = row + kByteCol    * P::WARP_TILE_M;
            uint32_t offHi00  = row + kByteColHi  * P::WARP_TILE_M;
            (void)offLo00; (void)offHi00; (void)idx; (void)rowInBl;
        }
    }

    // Simplified load: use raw address computation
    {
        uint32_t lid    = laneId;
        uint32_t row    = lid & 0x1F;

        #pragma unroll
        for(uint32_t k = 0; k < 8u; ++k)
        {
            uint32_t idx0  = k;
            uint32_t idx1  = k + 8u;

            uint32_t offLo0  = row + k       * P::WARP_TILE_M;
            uint32_t offLo1  = row + (k+8u)  * P::WARP_TILE_M;
            uint32_t offLoD16_0 = offLo0 + 128u;
            uint32_t offLoD16_1 = offLo1 + 128u;

            a_lo[idx0] = 0; a_hi[idx0] = 0;
            tile_ds_read_u8_pair(a_lo[idx0], a_hi[idx0], ldsA, offLo0, offLo1);
            tile_ds_read_u8_d16_hi_pair(a_lo[idx0], a_hi[idx0], ldsA, offLoD16_0, offLoD16_1);

            a_lo[idx1] = 0; a_hi[idx1] = 0;
            tile_ds_read_u8_pair(a_lo[idx1], a_hi[idx1], ldsA, offLo0 + 1u, offLo1 + 1u);
            tile_ds_read_u8_d16_hi_pair(a_lo[idx1], a_hi[idx1], ldsA, offLoD16_0 + 1u, offLoD16_1 + 1u);
        }
    }

    // === 3. s_waitcnt lgkmcnt(0) ===
    // Mirror: s_waitcnt lgkmcnt(0)   ; line 317
    tile_wait_lgkmcnt0();

    // === 4. A packing first half: 8 × v_lshl_or_b32 ===
    // Mirror: v_lshl_or_b32 v34, v83, 8, v34    ; lines 318-325
    //         ... (8 instructions — a0 + a1 = 8 packed dwords)
    tile_pack_a_first_half(a_lo, a_hi, a0, a1);

    // === 5. s_setprio 3 ===
    // Mirror: s_setprio 3   ; line 326
    tile_setprio_3();

    // === 6. WMMA burst: 8 WMMA instructions, interleaved A packing ===
    // Mirror: lines 327-342
    //
    // v_wmma_i32_16x16x16_iu8 acc00, b0, a0, acc00  ; line 327
    // v_lshl_or_b32 v42, v91, 8, v42                ; lines 328-335 (interleaved)
    // v_lshl_or_b32 ...
    // v_wmma_i32_16x16x16_iu8 acc01, b0, a1, acc01  ; line 336
    // v_wmma_i32_16x16x16_iu8 acc10, b1, a0, acc10  ; line 337
    // v_wmma_i32_16x16x16_iu8 acc11, b1, a1, acc11  ; line 338
    // v_wmma_i32_16x16x16_iu8 acc00, b2, a2, acc00  ; line 339
    // v_wmma_i32_16x16x16_iu8 acc01, b2, a3, acc01  ; line 340
    // v_wmma_i32_16x16x16_iu8 acc10, b3, a2, acc10  ; line 341
    // v_wmma_i32_16x16x16_iu8 acc11, b3, a3, acc11  ; line 342

    // First WMMA: acc00 += b0 * a0
    acc00 = tile_wmma_iu8(a0, b0, acc00);

    // Interleaved A packing second half: 8 × v_lshl_or_b32
    tile_pack_a_second_half(a_lo, a_hi, a2, a3);

    // Remaining 7 WMMA calls
    acc01 = tile_wmma_iu8(a1, b0, acc01);
    acc10 = tile_wmma_iu8(a0, b1, acc10);
    acc11 = tile_wmma_iu8(a1, b1, acc11);
    acc00 = tile_wmma_iu8(a2, b2, acc00);
    acc01 = tile_wmma_iu8(a3, b2, acc01);
    acc10 = tile_wmma_iu8(a2, b3, acc10);
    acc11 = tile_wmma_iu8(a3, b3, acc11);

    // === 7. s_setprio 1 ===
    // Mirror: s_setprio 1   ; line 343
    tile_setprio_1();
}

// ============================================================================
// A global→LDS prefetch: loads A tile from global memory and writes to LDS.
// Mirror of Tensile ISA buffer_load_b128 + ds_store_b128 pattern.
//
// For BLOCKS_M=2, each warp needs WARP_TILE_M × WARP_TILE_K = 32 × 16 = 512 bytes
// of A data per K-group. The cooperative prefetch loads the full MACRO_TILE_M ×
// MACRO_TILE_K tile into LDS, then each warp reads its WARP_TILE_M subset.
//
// Simplified: scalarized global loads → LDS write (single warp or cooperative).
// ============================================================================

template <typename P>
__device__ inline void cmb_tensile_prefetch_a_lds(
    InputT*       ldsA,
    const InputT* globalA,
    uint32_t      lds_wide_stride,
    uint32_t      macroRow,
    uint32_t      kOffset,
    uint32_t      lda,
    uint32_t      /*laneId*/,
    uint32_t      flatTid)
{
    constexpr uint32_t Mrows   = P::MACRO_TILE_M;
    constexpr uint32_t Kcols   = P::ROCWMMA_K;
    constexpr uint32_t threads = P::TBLOCK_X * P::TBLOCK_Y;

    for(uint32_t idx = flatTid; idx < Mrows * Kcols; idx += threads)
    {
        uint32_t col = idx / Mrows;
        uint32_t row = idx % Mrows;
        uint32_t gRow = macroRow + row;
        uint32_t gCol = kOffset + col;

        InputT val = globalA[gRow + gCol * (uint64_t)lda];
        ldsA[row + col * lds_wide_stride] = val;
    }
}

template <typename P>
__device__ inline void cmb_tensile_prefetch_b_lds(
    InputT*       ldsB,
    const InputT* globalB,
    uint32_t      lds_wide_stride,
    uint32_t      macroCol,
    uint32_t      kOffset,
    uint32_t      ldb,
    uint32_t      /*laneId*/,
    uint32_t      flatTid)
{
    constexpr uint32_t Ncols   = P::MACRO_TILE_N;
    constexpr uint32_t Kcols   = P::ROCWMMA_K;
    constexpr uint32_t threads = P::TBLOCK_X * P::TBLOCK_Y;

    for(uint32_t idx = flatTid; idx < Ncols * Kcols; idx += threads)
    {
        uint32_t col = idx / Kcols;
        uint32_t row = idx % Kcols;

        InputT val = globalB[(kOffset + row) * (uint64_t)ldb + macroCol + col];
        ldsB[col + row * lds_wide_stride] = val;
    }
}


// ============================================================================
// Store output — writes accumulator VGPRs back to global D matrix.
// Mirror of Tensile ISA Summation_End_OptNLL section (lines 604-718).
//
// For BLOCKS_M=2, BLOCKS_N=2: 4 accumulator groups × 8 VGPRs = 32 VGPRs.
// Each VGPR holds 1 i32 element of the output tile.
//
// The store path uses buffer_store_b64 (16 stores of pairs) from the
// re-ordered VGPRs to match the output matrix layout.
// ============================================================================

template <typename P>
__device__ inline void cmb_tensile_store_output(
    OutputT*       d,
    OutputT const* c,
    const VRegI32x8& acc00,
    const VRegI32x8& acc10,
    const VRegI32x8& acc01,
    const VRegI32x8& acc11,
    uint32_t       macroRow,
    uint32_t       macroCol,
    uint32_t       ldc,
    uint32_t       ldd,
    uint32_t       laneId,
    ComputeT       alpha,
    ComputeT       beta)
{
    constexpr uint32_t Mblk = P::ROCWMMA_M;
    constexpr uint32_t Nblk = P::ROCWMMA_N;

    uint32_t lid = laneId;
    uint32_t rowInBlk = lid & 0xF;
    uint32_t colQuad  = lid >> 4;

    const VRegI32x8* accArr[4] = {&acc00, &acc01, &acc10, &acc11};

    #pragma unroll
    for(uint32_t blkM = 0; blkM < P::BLOCKS_M; ++blkM)
    {
        #pragma unroll
        for(uint32_t blkN = 0; blkN < P::BLOCKS_N; ++blkN)
        {
            const uint32_t accIdx = blkM * P::BLOCKS_N + blkN;
            const VRegI32x8& acc = *accArr[accIdx];

            uint32_t baseRow = macroRow + blkM * Mblk;
            uint32_t baseCol = macroCol + blkN * Nblk;

            #pragma unroll
            for(uint32_t e = 0; e < 8; ++e)
            {
                uint32_t outRow = baseRow + (e & 0x3) * 4 + (rowInBlk & 0x3);
                uint32_t outCol = baseCol + colQuad * 8 + ((e >> 2) & 0x1) * 4 + (rowInBlk >> 2);

                OutputT cVal = c[outRow * (uint64_t)ldc + outCol];
                ComputeT tmp = alpha * static_cast<ComputeT>(acc[e])
                             + beta  * static_cast<ComputeT>(cVal);
                d[outRow * (uint64_t)ldd + outCol] = static_cast<OutputT>(tmp);
            }
        }
    }
}


// ============================================================================
// rocwmma_combined_kernel — the Tensile-mirroring kernel
//
// Structural isomorphism with Tensile ISA:
//   - Flat i32x8 accumulators (not rocWMMA fragments)
//   - Direct ds_read_u8 / ds_read_b128 for LDS loads
//   - v_lshl_or_b32 for A packing
//   - v_wmma_i32_16x16x16_iu8 for compute
//   - s_setprio at boundaries
//   - Precise s_waitcnt control
//   - Mirror of openLoopL_12 pattern (lines 227-603)
// ============================================================================

template <typename P>
__global__ __launch_bounds__(P::TBLOCK_X * P::TBLOCK_Y) void rocwmma_combined_kernel(
    uint32_t       m,
    uint32_t       n,
    uint32_t       k,
    InputT const*  a,
    InputT const*  b,
    OutputT const* c,
    OutputT*       d,
    uint32_t       lda,
    uint32_t       ldb,
    uint32_t       ldc,
    uint32_t       ldd,
    ComputeT       alpha,
    ComputeT       beta)
{
    constexpr uint32_t NumAcc = TensileAcc<P>::NAcc;

    const uint32_t flatTid = threadIdx.x + threadIdx.y * P::TBLOCK_X;
    const uint32_t laneId  = flatTid & 0x1F;
    const uint32_t warpId  = flatTid / P::WARP_SIZE;

    const uint32_t warpRowM = (warpId / P::WARPS_N) % P::WARPS_M;
    const uint32_t warpColN = (warpId % P::WARPS_N);

    const uint32_t macroRow = blockIdx.x * P::MACRO_TILE_M + warpRowM * P::WARP_TILE_M;
    const uint32_t macroCol = blockIdx.y * P::MACRO_TILE_N + warpColN * P::WARP_TILE_N;

    if(macroRow + P::WARP_TILE_M > m || macroCol + P::WARP_TILE_N > n)
    {
        return;
    }

    // === LDS setup (double-buffered) ===
    HIP_DYNAMIC_SHARED(void*, localMemPtr);

    constexpr uint32_t ldsElemsA  = P::MACRO_TILE_M * P::ROCWMMA_K;
    constexpr uint32_t ldsElemsB  = P::MACRO_TILE_N * P::ROCWMMA_K;
    constexpr uint32_t sizeLdsOne = ldsElemsA + ldsElemsB;
    constexpr uint32_t sizeGroup  = P::K_GROUP * sizeLdsOne;

    auto* ldsLo = reinterpret_cast<InputT*>(localMemPtr);
    auto* ldsHi = ldsLo + sizeGroup;

    InputT* ldsA_lo = ldsLo;
    InputT* ldsB_lo = ldsLo + ldsElemsA;
    InputT* ldsA_hi = ldsHi;
    InputT* ldsB_hi = ldsHi + ldsElemsA;

    constexpr uint32_t ldsStrideA = P::MACRO_TILE_M;
    constexpr uint32_t ldsStrideB = P::MACRO_TILE_N;

    // === Accumulator: flat i32x8 VGPR arrays ===
    // Mirror of v0..v31 zero-init (Tensile ISA lines 182-213)
    VRegI32x8 acc00{}, acc01{}, acc10{}, acc11{};
    #pragma unroll
    for(uint32_t j = 0; j < 8; ++j)
    {
        acc00[j] = 0; acc01[j] = 0;
        acc10[j] = 0; acc11[j] = 0;
    }

    (void)NumAcc;

    // === Prefetch tile 0 into ldsLo ===
    cmb_tensile_prefetch_a_lds<P>(
        ldsA_lo, a, ldsStrideA,
        blockIdx.x * P::MACRO_TILE_M, 0u, lda, laneId, flatTid);
    cmb_tensile_prefetch_b_lds<P>(
        ldsB_lo, b, ldsStrideB,
        blockIdx.y * P::MACRO_TILE_N, 0u, ldb, laneId, flatTid);

    tile_s_barrier();

    // === Main double-buffered loop ===
    constexpr uint32_t kGroupStep = P::K_GROUP * P::ROCWMMA_K;

    for(uint32_t kBase = kGroupStep; kBase < k; kBase += kGroupStep)
    {
        // Mirror: openLoopL_12 label_0013 (lines 231-353)
        // Phase P0: compute from ldsLo using lane-local A/B offsets
        {
            VRegI32 ldsA_addr;
            VRegI32 ldsB_addr;

            uint32_t warpAOff = warpRowM * P::WARP_TILE_M;
            uint32_t warpBOff = warpColN * P::WARP_TILE_N;

            (void)warpAOff; (void)warpBOff;

            ldsA_addr = static_cast<VRegI32>(reinterpret_cast<uintptr_t>(ldsA_lo));
            ldsB_addr = static_cast<VRegI32>(reinterpret_cast<uintptr_t>(ldsB_lo));

            cmb_compute_tensile_group<P>(
                acc00, acc01, acc10, acc11,
                ldsA_addr, ldsB_addr, laneId,
                0, 0, 0, 0);
        }

        // Phase P1: prefetch next tile into ldsHi
        cmb_tensile_prefetch_a_lds<P>(
            ldsA_hi, a, ldsStrideA,
            blockIdx.x * P::MACRO_TILE_M, kBase, lda, laneId, flatTid);
        cmb_tensile_prefetch_b_lds<P>(
            ldsB_hi, b, ldsStrideB,
            blockIdx.y * P::MACRO_TILE_N, kBase, ldb, laneId, flatTid);

        // Mirror: s_waitcnt lgkmcnt(0) + s_barrier (lines 344-345)
        tile_wait_lgkmcnt0();
        tile_s_barrier();

        // Swap double buffers
        {
            auto* tmpA = ldsA_lo; ldsA_lo = ldsA_hi; ldsA_hi = tmpA;
            auto* tmpB = ldsB_lo; ldsB_lo = ldsB_hi; ldsB_hi = tmpB;
        }
    }

    // === Final tile: compute from ldsLo ===
    // Mirror: OptNLL_End_15 or Summation_End sections
    {
        VRegI32 ldsA_addr = static_cast<VRegI32>(reinterpret_cast<uintptr_t>(ldsA_lo));
        VRegI32 ldsB_addr = static_cast<VRegI32>(reinterpret_cast<uintptr_t>(ldsB_lo));

        cmb_compute_tensile_group<P>(
            acc00, acc01, acc10, acc11,
            ldsA_addr, ldsB_addr, laneId,
            0, 0, 0, 0);
    }

    tile_wait_lgkmcnt0();
    tile_s_barrier();

    // === Store output ===
    // Mirror: GW_B0_E0_19 section (lines 622-718)
    cmb_tensile_store_output<P>(
        d, c, acc00, acc10, acc01, acc11,
        macroRow, macroCol, ldc, ldd, laneId, alpha, beta);
}

// ============================================================================
// Type aliases
// ============================================================================

// Tensile64x64x32: BLOCKS_M=2, BLOCKS_N=2, K_GROUP=2 => WARP_TILE_M=32, WARP_TILE_N=32, total K=32
using Tensile64x64x32 = CombinedParams<2u, 2u, 64u, 2u, 2u>;

// Backward-compat baseline
using Tensile128x128x32 = CombinedParams<4u, 4u, 64u, 2u, 2u>;
using Tensile64x128x32  = CombinedParams<2u, 4u, 64u, 2u, 2u>;

// Various K_GROUP sizes
using Tensile64x64x32_K1  = CombinedParams<2u, 2u, 64u, 2u, 1u>;
using Tensile64x64x32_K4  = CombinedParams<2u, 2u, 64u, 2u, 4u>;

// With PC split
using Tensile64x64x32_PC = CombinedParams<2u, 2u, 64u, 2u, 2u, 2u, 0u, 8u, 1u>;

// Legacy type aliases (compatibility with existing runner code)
using Pearl128x128x32_M2   = CombinedParams<4u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 0u>;
using Pearl128x128x32_M2B  = CombinedParams<4u, 4u, 64u, 2u, 2u, 2u, 64u, 0u, 0u>;
using Pearl64x128x32_M2    = CombinedParams<2u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 0u>;
using Pearl64x128x32_M2P8  = CombinedParams<2u, 4u, 64u, 2u, 2u, 2u, 0u, 8u, 0u>;
using Pearl64x128x32_M2A8B16 = CombinedParams<2u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 0u, kRocwmmaK, 8u, 16u>;
using Pearl64x128x32_M2A16B32 = CombinedParams<2u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 0u, kRocwmmaK, 16u, 32u>;
using Pearl64x128x32_M2N2 = CombinedParams<2u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 0u, kRocwmmaK, 0u, 0u, 2u>;
using Pearl64x128x32_M2P8N2 = CombinedParams<2u, 4u, 64u, 2u, 2u, 2u, 0u, 8u, 0u, kRocwmmaK, 0u, 8u, 2u>;
using Pearl64x128x32_M2A8B16N2 = CombinedParams<2u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 0u, kRocwmmaK, 8u, 16u, 2u>;
using Pearl64x128x32_M2A16B32N2 = CombinedParams<2u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 0u, kRocwmmaK, 16u, 32u, 2u>;
using Pearl64x128x32_M2K4 = CombinedParams<2u, 4u, 64u, 2u, 4u, 2u, 0u, 0u, 0u>;
using Pearl64x128x32_M2K4P8 = CombinedParams<2u, 4u, 64u, 2u, 4u, 2u, 0u, 8u, 0u>;
using Pearl64x128x32_M2K4A8B16 = CombinedParams<2u, 4u, 64u, 2u, 4u, 2u, 0u, 0u, 0u, kRocwmmaK, 8u, 16u>;
using Pearl64x128x32_M2K4A16B32 = CombinedParams<2u, 4u, 64u, 2u, 4u, 2u, 0u, 0u, 0u, kRocwmmaK, 16u, 32u>;
using Pearl64x128x32_M2K8 = CombinedParams<2u, 4u, 64u, 2u, 8u, 2u, 0u, 0u, 0u>;
using Pearl64x128x32_M2K8P8 = CombinedParams<2u, 4u, 64u, 2u, 8u, 2u, 0u, 8u, 0u>;
using Pearl64x128x32_M2K8A8B16 = CombinedParams<2u, 4u, 64u, 2u, 8u, 2u, 0u, 0u, 0u, kRocwmmaK, 8u, 16u>;
using Pearl64x128x32_M2K8A16B32 = CombinedParams<2u, 4u, 64u, 2u, 8u, 2u, 0u, 0u, 0u, kRocwmmaK, 16u, 32u>;
using Pearl64x256x16_M2A8B16 = CombinedParams<2u, 8u, 64u, 2u, 1u, 2u, 0u, 0u, 0u, kRocwmmaK, 8u, 16u>;
using Pearl64x256x16_M2A16B32 = CombinedParams<2u, 8u, 64u, 2u, 1u, 2u, 0u, 0u, 0u, kRocwmmaK, 16u, 32u>;
using Pearl128x128x32_M2A8B16 = CombinedParams<4u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 0u, kRocwmmaK, 8u, 16u>;
using Pearl128x128x32_M2A16B32 = CombinedParams<4u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 0u, kRocwmmaK, 16u, 32u>;
using Pearl128x128x16_M2   = CombinedParams<4u, 4u, 64u, 2u, 1u, 2u, 0u, 0u, 0u>;
using Pearl128x128x32_PC   = CombinedParams<4u, 4u, 64u, 2u, 2u, 1u, 0u, 0u, 1u>;
using Pearl128x128x32_M2PC = CombinedParams<4u, 4u, 64u, 2u, 2u, 2u, 0u, 0u, 1u>;
using Pearl64x128x32_K32PC = CombinedParams<2u, 2u, 64u, 4u, 1u, 1u, 0u, 8u, 1u, 32u>;
using Pearl64x128x32_K32M2PC = CombinedParams<2u, 2u, 64u, 4u, 1u, 2u, 0u, 8u, 1u, 32u>;
using Pearl128x128x32_M4   = CombinedParams<4u, 4u, 64u, 2u, 2u, 4u, 0u, 0u, 0u>;
using Pearl64x128x32_M4    = CombinedParams<2u, 4u, 64u, 2u, 2u, 4u, 0u, 0u, 0u>;
using Pearl128x128x32_B64  = CombinedParams<4u, 4u, 64u, 2u, 2u, 1u, 64u, 0u, 0u>;
using Pearl128x128x32_B128 = CombinedParams<4u, 4u, 64u, 2u, 2u, 1u, 128u, 0u, 0u>;
using Pearl64x128x32_B64   = CombinedParams<2u, 4u, 64u, 2u, 2u, 1u, 64u, 0u, 0u>;
using Pearl128x128x32_BL = CombinedParams<4u, 4u, 64u, 2u, 2u, 1u, 0u, 0u, 0u>;

// ============================================================================
// run_combined_variant + printing (carried forward for backward compat)
// ============================================================================

inline void combined_print_header()
{
    std::cout << std::left
              << std::setw(34) << "Variant"
              << std::setw(6) << "OK"
              << std::setw(8) << "TBlkX"
              << std::setw(8) << "TBlkY"
              << std::setw(8) << "BlkM"
              << std::setw(8) << "BlkN"
              << std::setw(9) << "MacroM"
              << std::setw(9) << "MacroN"
              << std::setw(8) << "KGrp"
              << std::setw(8) << "Pad"
              << std::setw(8) << "MSub"
              << std::setw(8) << "NSub"
              << std::setw(8) << "BStr"
              << std::setw(8) << "PCSpl"
              << std::setw(13) << "elapsedMs"
              << std::setw(13) << "perEvalMs"
              << std::setw(13) << "GOp"
              << std::setw(13) << "TOp/s"
              << std::setw(13) << "TMac/s"
              << std::endl;
}

inline void combined_print_result(const std::string& label,
                          const std::string& ok,
                          uint32_t tblockX, uint32_t tblockY,
                          uint32_t blocksM, uint32_t blocksN,
                          uint32_t macroM, uint32_t macroN,
                          uint32_t kGroup, uint32_t ldsPad,
                          uint32_t mSub, uint32_t nSub, uint32_t bStr, uint32_t pcSpl,
                          const Problem& p, float elapsedMs)
{
    const double perEvalMs = static_cast<double>(elapsedMs) / p.runs;
    const double topPerSec = gops(p) / elapsedMs * static_cast<double>(p.runs);
    const double tmacPerSec = macs(p) / (perEvalMs * 1.0e-3);

    std::cout << std::left
              << std::setw(34) << label
              << std::setw(6) << ok
              << std::setw(8) << tblockX
              << std::setw(8) << tblockY
              << std::setw(8) << blocksM
              << std::setw(8) << blocksN
              << std::setw(9) << macroM
              << std::setw(9) << macroN
              << std::setw(8) << kGroup
              << std::setw(8) << ldsPad
              << std::setw(8) << mSub
              << std::setw(8) << nSub
              << std::setw(8) << bStr
              << std::setw(8) << pcSpl
              << std::setw(13) << elapsedMs
              << std::setw(13) << perEvalMs
              << std::setw(13) << gops(p)
              << std::setw(13) << topPerSec
              << std::setw(13) << tmacPerSec
              << std::endl;
}

template <typename P>
void run_combined_variant(const std::string& label, const Problem& p, Buffers& buffers)
{
    constexpr uint32_t effMacroM = P::EFF_MACRO_TILE_M;
    constexpr uint32_t effMacroN = P::EFF_MACRO_TILE_N;
    constexpr uint32_t kGroupStep = P::K_GROUP * P::MACRO_TILE_K;

    if((p.m % P::ROCWMMA_M) || (p.n % P::ROCWMMA_N) || (p.k % kGroupStep)
       || (p.m % effMacroM) || (p.n % effMacroN))
    {
        std::cout << label << " skipped: dimensions not divisible by frag/macro/group sizes"
                  << " (need m%" << effMacroM << "==0, n%" << P::ROCWMMA_N
                  << "==0, n%" << effMacroN << "==0, k%" << kGroupStep << "==0)" << std::endl;
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
    constexpr uint32_t ldsWidth   = P::EFF_B_STRIDE;
    constexpr uint32_t ldsHeight  = ldsHeightA + P::LDS_PAD_A + ldsHeightB + P::LDS_PAD_B;
    constexpr uint32_t sizeLdsOne = ldsHeight * ldsWidth;
    const uint32_t sharedBytes = 2u * P::K_GROUP * sizeLdsOne * sizeof(InputT);

    int maxSharedBytes = 0;
    CHECK_HIP_ERROR(hipDeviceGetAttribute(
        &maxSharedBytes, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    if(sharedBytes > static_cast<uint32_t>(maxSharedBytes))
    {
        std::cout << label << " skipped: shared memory " << sharedBytes
                  << " exceeds device limit " << maxSharedBytes << std::endl;
        return;
    }

    auto launch = [&]() {
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
    for(uint32_t i = 0; i < p.warmups; ++i) { launch(); }
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    hipEvent_t startEvent{}, stopEvent{};
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for(uint32_t i = 0; i < p.runs; ++i) { launch(); }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    const bool ok = compare_output(p, buffers);
    combined_print_result(label,
                          ok ? "yes" : "no",
                          P::TBLOCK_X, P::TBLOCK_Y,
                          P::BLOCKS_M, P::BLOCKS_N,
                          effMacroM, effMacroN,
                          P::K_GROUP, P::LDS_PAD,
                          P::M_SUBPASSES, P::N_SUBPASSES, P::EFF_B_STRIDE, P::PC_SPLIT,
                          p, elapsedMs);
}

template <typename P>
void run_combined_queue_variant(const std::string& label, const Problem& p,
                                Buffers& buffers, uint32_t depth)
{
    constexpr uint32_t effMacroM = P::EFF_MACRO_TILE_M;
    constexpr uint32_t effMacroN = P::EFF_MACRO_TILE_N;
    constexpr uint32_t kGroupStep = P::K_GROUP * P::MACRO_TILE_K;

    if((p.m % P::ROCWMMA_M) || (p.n % P::ROCWMMA_N) || (p.k % kGroupStep)
       || (p.m % effMacroM) || (p.n % effMacroN))
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
    constexpr uint32_t ldsWidth   = P::EFF_B_STRIDE;
    constexpr uint32_t ldsHeight  = ldsHeightA + P::LDS_PAD_A + ldsHeightB + P::LDS_PAD_B;
    constexpr uint32_t sizeLdsOne = ldsHeight * ldsWidth;
    const uint32_t sharedBytes = 2u * P::K_GROUP * sizeLdsOne * sizeof(InputT);

    int maxSharedBytes = 0;
    CHECK_HIP_ERROR(hipDeviceGetAttribute(
        &maxSharedBytes, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    if(sharedBytes > static_cast<uint32_t>(maxSharedBytes))
    {
        std::cout << label << " skipped: shared memory " << sharedBytes
                  << " exceeds limit " << maxSharedBytes << std::endl;
        return;
    }

    auto launch = [&]() {
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
    for(uint32_t i = 0; i < p.warmups; ++i) {
        for(uint32_t q = 0; q < depth; ++q) { launch(); }
        CHECK_HIP_ERROR(hipDeviceSynchronize());
    }

    hipEvent_t startEvent{}, stopEvent{};
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for(uint32_t i = 0; i < p.runs; ++i) {
        for(uint32_t q = 0; q < depth; ++q) { launch(); }
        CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    Problem queuedProblem = p;
    queuedProblem.runs = p.runs * depth;
    const bool ok = compare_output(p, buffers);
    combined_print_result(label,
                          ok ? "yes" : "no",
                          P::TBLOCK_X, P::TBLOCK_Y,
                          P::BLOCKS_M, P::BLOCKS_N,
                          effMacroM, effMacroN,
                          P::K_GROUP, P::LDS_PAD,
                          P::M_SUBPASSES, P::N_SUBPASSES, P::EFF_B_STRIDE, P::PC_SPLIT,
                          queuedProblem, elapsedMs);
}
