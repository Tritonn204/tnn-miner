#pragma once

#include "recipe.hpp"
#if !defined(__HIPCC_RTC__)
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#endif

namespace tnn::hip::iris::gemm::qualified {
namespace {

using I4 = int __attribute__((ext_vector_type(4)));
using I8 = int __attribute__((ext_vector_type(8)));
using I2 = int __attribute__((ext_vector_type(2)));

// Paired 128x256 ownership derived from the studied SourceSwap=1 schedule.
// A LDS: Schedule::tile_m contiguous rows per K (128 in the qualified recipe).
// B LDS: 32 contiguous K per column,
// padded by 16 bytes after every 128 bytes (LBSPPB128/LPB16).
template <class Arch, bool Swizzle = false>
__device__ __forceinline__ unsigned b_offset(unsigned n, unsigned k) {
    unsigned x = n * 32 + k;
    if constexpr (Swizzle)
        return x ^ ((n & 28u) << 2);
    else
        return Arch::b_offset(n, k);
}

template <int NB> struct Operands {
    I4 a0;
    I4 a1;
    I4 b[NB];
};

template <class Schedule, int KS, int NB, bool Paired = false, bool Swizzle = false,
          bool Half = false>
__device__ __forceinline__ Operands<NB> operands(const unsigned char *a, const unsigned char *b,
                                                 unsigned am, unsigned bn) {
    Operands<NB> x{};
    if constexpr (Schedule::rematerialize && !Schedule::shared_address) {
        unsigned fresh;
        asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(fresh));
        unsigned wave = __builtin_amdgcn_readfirstlane(threadIdx.x) / 32;
        am = (wave % (Schedule::tile_m / 32)) * 32 + (fresh % 16) * 2;
        bn = (wave / (Schedule::tile_m / 32)) * 16 + fresh % 16;
    }
    if constexpr (Half) {
        unsigned p0, p1, p2, p3, p4, p5, p6, p7;
        unsigned addr = unsigned(reinterpret_cast<uintptr_t>(a)) + am + KS * Schedule::tile_m;
        // As in the saved Tensile schedule, D16_HI combines independent LDS
        // reads into a register without a separate shift/or instruction.
        asm volatile("ds_read_u16 %0, %8 offset:0\n\t"
                     "ds_read_u16_d16_hi %0, %8 offset:64\n\t"
                     "ds_read_u16 %1, %8 offset:128\n\t"
                     "ds_read_u16_d16_hi %1, %8 offset:192\n\t"
                     "ds_read_u16 %2, %8 offset:256\n\t"
                     "ds_read_u16_d16_hi %2, %8 offset:320\n\t"
                     "ds_read_u16 %3, %8 offset:384\n\t"
                     "ds_read_u16_d16_hi %3, %8 offset:448\n\t"
                     "ds_read_u16 %4, %8 offset:512\n\t"
                     "ds_read_u16_d16_hi %4, %8 offset:576\n\t"
                     "ds_read_u16 %5, %8 offset:640\n\t"
                     "ds_read_u16_d16_hi %5, %8 offset:704\n\t"
                     "ds_read_u16 %6, %8 offset:768\n\t"
                     "ds_read_u16_d16_hi %6, %8 offset:832\n\t"
                     "ds_read_u16 %7, %8 offset:896\n\t"
                     "ds_read_u16_d16_hi %7, %8 offset:960\n\t"
                     "s_waitcnt lgkmcnt(0)"
                     : "=&v"(p0), "=&v"(p1), "=&v"(p2), "=&v"(p3), "=&v"(p4), "=&v"(p5), "=&v"(p6),
                       "=&v"(p7)
                     : "v"(addr)
                     : "memory");
        unsigned p[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            x.a0[r] = __builtin_amdgcn_perm(p[2 * r + 1], p[2 * r], 0x06040200);
            x.a1[r] = __builtin_amdgcn_perm(p[2 * r + 1], p[2 * r], 0x07050301);
        }
    } else if constexpr (Paired) {
        I2 p0, p1, p2, p3, p4, p5, p6, p7;
        // Four M rows per dword; lanes select the even or odd row pair.
        // DS offsets are in dwords. All results are ready at the explicit wait.
        unsigned addr =
            unsigned(reinterpret_cast<uintptr_t>(a)) + (am & ~3u) + KS * Schedule::tile_m;
        asm volatile("ds_read2_b32 %0, %8 offset0:%c9 offset1:%c10\n\t"
                     "ds_read2_b32 %1, %8 offset0:%c11 offset1:%c12\n\t"
                     "ds_read2_b32 %2, %8 offset0:%c13 offset1:%c14\n\t"
                     "ds_read2_b32 %3, %8 offset0:%c15 offset1:%c16\n\t"
                     "ds_read2_b32 %4, %25 offset0:%c17 offset1:%c18\n\t"
                     "ds_read2_b32 %5, %25 offset0:%c19 offset1:%c20\n\t"
                     "ds_read2_b32 %6, %25 offset0:%c21 offset1:%c22\n\t"
                     "ds_read2_b32 %7, %25 offset0:%c23 offset1:%c24\n\t"
                     "s_waitcnt lgkmcnt(0)"
                     : "=&v"(p0), "=&v"(p1), "=&v"(p2), "=&v"(p3), "=&v"(p4), "=&v"(p5), "=&v"(p6),
                       "=&v"(p7)
                     : "v"(addr), "n"(0 * Schedule::tile_m / 4), "n"(1 * Schedule::tile_m / 4),
                       "n"(2 * Schedule::tile_m / 4), "n"(3 * Schedule::tile_m / 4),
                       "n"(4 * Schedule::tile_m / 4), "n"(5 * Schedule::tile_m / 4),
                       "n"(6 * Schedule::tile_m / 4), "n"(7 * Schedule::tile_m / 4),
                       "n"((8 - (Schedule::tile_m == 128 ? 8 : 0)) * Schedule::tile_m / 4),
                       "n"((9 - (Schedule::tile_m == 128 ? 8 : 0)) * Schedule::tile_m / 4),
                       "n"((10 - (Schedule::tile_m == 128 ? 8 : 0)) * Schedule::tile_m / 4),
                       "n"((11 - (Schedule::tile_m == 128 ? 8 : 0)) * Schedule::tile_m / 4),
                       "n"((12 - (Schedule::tile_m == 128 ? 8 : 0)) * Schedule::tile_m / 4),
                       "n"((13 - (Schedule::tile_m == 128 ? 8 : 0)) * Schedule::tile_m / 4),
                       "n"((14 - (Schedule::tile_m == 128 ? 8 : 0)) * Schedule::tile_m / 4),
                       "n"((15 - (Schedule::tile_m == 128 ? 8 : 0)) * Schedule::tile_m / 4),
                       "v"(addr + (Schedule::tile_m == 128 ? 8 * Schedule::tile_m : 0))
                     : "memory");
        I2 p[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
        unsigned select = 0x05010400u + (am & 2) * 0x01010101u;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            unsigned lo = __builtin_amdgcn_perm(p[2 * r][1], p[2 * r][0], select);
            unsigned hi = __builtin_amdgcn_perm(p[2 * r + 1][1], p[2 * r + 1][0], select);
            x.a0[r] = __builtin_amdgcn_perm(hi, lo, 0x05040100);
            x.a1[r] = __builtin_amdgcn_perm(hi, lo, 0x07060302);
        }
    } else {
// Pair adjacent M rows per lane. This preserves the vendor's interleaved
// output mapping while using 16-bit reads rather than two byte gathers.
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            unsigned p0 = *reinterpret_cast<const unsigned short *>(a + am + (KS + 4 * r + 0) * 64);
            unsigned p1 = *reinterpret_cast<const unsigned short *>(a + am + (KS + 4 * r + 1) * 64);
            unsigned p2 = *reinterpret_cast<const unsigned short *>(a + am + (KS + 4 * r + 2) * 64);
            unsigned p3 = *reinterpret_cast<const unsigned short *>(a + am + (KS + 4 * r + 3) * 64);
            unsigned lo = p0 | (p1 << 16), hi = p2 | (p3 << 16);
            x.a0[r] = __builtin_amdgcn_perm(hi, lo, 0x06040200);
            x.a1[r] = __builtin_amdgcn_perm(hi, lo, 0x07050301);
        }
    }
#pragma unroll
    for (int i = 0; i < NB; ++i)
        x.b[i] = *reinterpret_cast<const I4 *>(
            b + b_offset<typename Schedule::Arch, Swizzle>(bn + i * 32, KS));
    return x;
}
template <class Arch, int NB, bool Priority>
__device__ __forceinline__ void compute(I8 (&c)[2 * NB], const Operands<NB> &x) {
    // SourceSwap=1: B is WMMA src0, A is src1. Both lane halves carry
    // identical inputs. Output M=2*(lane%16)+parity, N=2*reg+lane/16.
    if constexpr (Priority)
        asm volatile("s_setprio 3");
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        if constexpr (Priority) {
            asm volatile("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]"
                         : "+v"(c[2 * i])
                         : "v"(x.b[i]), "v"(x.a0));
            asm volatile("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]"
                         : "+v"(c[2 * i + 1])
                         : "v"(x.b[i]), "v"(x.a1));
        } else {
            c[2 * i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                Arch::signed_inputs, x.b[i], Arch::signed_inputs, x.a0, c[2 * i], false);
            c[2 * i + 1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                Arch::signed_inputs, x.b[i], Arch::signed_inputs, x.a1, c[2 * i + 1], false);
        }
    }
    if constexpr (Priority)
        asm volatile("s_setprio 0");
}

template <bool Local> __device__ __forceinline__ __attribute__((convergent)) void shared_barrier() {
    // Every thread follows the same K loop; only LDS is communicated. The
    // wait drains prior LDS writes/reads, then all eight waves rendezvous.
    // The compiler still owns global-load dependencies before the LDS stores.
    if constexpr (Local)
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    else
        __syncthreads();
}

} // namespace
} // namespace tnn::hip::iris::gemm::qualified
