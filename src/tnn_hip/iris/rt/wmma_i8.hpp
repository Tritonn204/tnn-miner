#pragma once

#include "arch_traits.hpp"
#include "coordinate.hpp"
#include "lds_view.hpp"
#include "warp_primitives.hpp"

namespace iris::hip {

using i8x8_t = signed char __attribute__((ext_vector_type(8)));
using i8x16_t = signed char __attribute__((ext_vector_type(16)));
using i32x2_t = int __attribute__((ext_vector_type(2)));
using i32x4_t = int __attribute__((ext_vector_type(4)));
using i32x8_t = int __attribute__((ext_vector_type(8)));

template <typename To, typename From>
IRIS_DEVICE_INLINE To bit_cast_vec(From value) {
#if IRIS_DEVICE_FRONTEND
    return __builtin_bit_cast(To, value);
#else
    (void)value;
    return To{};
#endif
}

IRIS_DEVICE_INLINE int wmma_swap16_i32(int value) {
#if IRIS_AMDGCN_FRONTEND
    return __builtin_amdgcn_ds_swizzle(value, 0x401f);
#else
    return value;
#endif
}

IRIS_DEVICE_INLINE i32x4_t concat_i32x2(i32x2_t lo, i32x2_t hi) {
    i32x4_t out{};
    out[0] = lo[0];
    out[1] = lo[1];
    out[2] = hi[0];
    out[3] = hi[1];
    return out;
}

template <int KPerBlock, int GroupCols, typename T>
IRIS_DEVICE_INLINE auto make_grouped_wmma_b_tile(T* ptr, int group) {
    using XorK16 = LdsXorSwizzle<KPerBlock, 16>;
    return make_lds_tile<T, KPerBlock, GroupCols, 0, XorK16>(
        ptr + group * (KPerBlock * GroupCols));
}

template <typename Wmma, typename Tile>
IRIS_DEVICE_INLINE typename Wmma::storage_fragment wmma_load_storage_row(
    const Tile& tile,
    int row,
    int col_base) {
    auto frag = Wmma::zero_storage();
    #pragma unroll
    for (int elem = 0; elem < Wmma::storage_elems; ++elem) {
        frag[elem] = tile.load(row, col_base + elem);
    }
    return frag;
}

template <typename Wmma, typename Tile>
IRIS_DEVICE_INLINE typename Wmma::storage_fragment wmma_load_storage_col(
    const Tile& tile,
    int row_base,
    int col) {
    auto frag = Wmma::zero_storage();
    #pragma unroll
    for (int elem = 0; elem < Wmma::storage_elems; ++elem) {
        frag[elem] = tile.load(row_base + elem, col);
    }
    return frag;
}

template <typename Wmma, typename Tile>
IRIS_DEVICE_INLINE typename Wmma::ab_fragment wmma_load_ab_row(
    const Tile& tile,
    int row,
    int col_base) {
    return Wmma::prepare_ab(wmma_load_storage_row<Wmma>(tile, row, col_base));
}

template <typename Wmma, typename Tile>
IRIS_DEVICE_INLINE typename Wmma::ab_fragment wmma_load_ab_col(
    const Tile& tile,
    int row_base,
    int col) {
    return Wmma::prepare_ab(wmma_load_storage_col<Wmma>(tile, row_base, col));
}

template <bool SignedA = true, bool SignedB = true, bool Clamp = false>
struct WmmaI8I8I32_16x16x16 {
    static constexpr int m = 16;
    static constexpr int n = 16;
    static constexpr int k = 16;
    static constexpr int c_elems = 8;

#if IRIS_AMDGCN_FRONTEND && (defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__))
    static constexpr bool native = true;
    static constexpr int ab_elems = 16;
    static constexpr int storage_elems = 8;
    using storage_fragment = i8x8_t;
    using ab_fragment = i8x16_t;
#elif IRIS_AMDGCN_FRONTEND && (defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__))
    // gfx12 native enablement stays separate. Input preparation mirrors gfx11 so
    // kernel-side fragment assembly can remain architecture-neutral.
    static constexpr bool native = false;
    static constexpr int ab_elems = 16;
    static constexpr int storage_elems = 8;
    using storage_fragment = i8x8_t;
    using ab_fragment = i8x16_t;
#else
    static constexpr bool native = false;
    static constexpr int ab_elems = 16;
    static constexpr int storage_elems = 16;
    using storage_fragment = i8x16_t;
    using ab_fragment = i8x16_t;
#endif

    using c_fragment = i32x8_t;

    IRIS_DEVICE_INLINE static storage_fragment zero_storage() {
        storage_fragment out{};
        return out;
    }

    IRIS_DEVICE_INLINE static ab_fragment zero_ab() {
        ab_fragment out{};
        return out;
    }

    IRIS_DEVICE_INLINE static c_fragment zero_c() {
        c_fragment out{};
        return out;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int c_row(int lane, int elem) {
        // RDNA3 wave32 D layout:
        //   row = 2 * gpr + lane_half
        // where lane_half is 0 for lanes 0..15 and 1 for lanes 16..31.
        return elem * 2 + lane / 16;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int c_col(int lane) {
        // RDNA3 wave32 D layout:
        //   col = lane % 16
        return lane & 15;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int storage_k_base(int lane) {
#if IRIS_AMDGCN_FRONTEND && (defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__))
        // RDNA3/RDNA4 wave32 A/B inputs are split across the two lane halves:
        //   lanes 0..15  -> K slice 0..7
        //   lanes 16..31 -> K slice 8..15
        // prepare_ab() then duplicates each half-wave slice into the 16-byte
        // operand contract expected by v_wmma_i32_16x16x16_iu8.
        return (lane >> 4) * storage_elems;
#else
        (void)lane;
        return 0;
#endif
    }

    IRIS_DEVICE_INLINE static ab_fragment prepare_ab(storage_fragment src) {
#if IRIS_AMDGCN_FRONTEND && (defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__))
        static_assert(storage_elems == 8);
        static_assert(ab_elems == 16);
        // RDNA3/RDNA4 wave32 contract:
        //   A[i][k] and B[k][j] are replicated across lane pairs i/i+16 and
        //   j/j+16 respectively. The low 8 bytes hold the lane's native half,
        //   and ds_swizzle(swap16) synthesizes the matching high-half copy.
        i32x2_t packed = bit_cast_vec<i32x2_t>(src);
        i32x2_t swapped{};
        swapped[0] = wmma_swap16_i32(packed[0]);
        swapped[1] = wmma_swap16_i32(packed[1]);
        return bit_cast_vec<ab_fragment>(concat_i32x2(packed, swapped));
#else
        return bit_cast_vec<ab_fragment>(src);
#endif
    }

    IRIS_DEVICE_INLINE static c_fragment mma(ab_fragment a, ab_fragment b, c_fragment c) {
#if IRIS_AMDGCN_FRONTEND && (defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__))
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            SignedA,
            bit_cast_vec<i32x4_t>(a),
            SignedB,
            bit_cast_vec<i32x4_t>(b),
            c,
            Clamp);
#else
        (void)a;
        (void)b;
        return c;
#endif
    }
};

template <typename Wmma, int KSteps>
IRIS_DEVICE_INLINE void wmma_cluster_m2n2_ksteps(
    const typename Wmma::ab_fragment (&a_frag0)[KSteps],
    const typename Wmma::ab_fragment (&a_frag1)[KSteps],
    const typename Wmma::ab_fragment (&b_frag0)[KSteps],
    const typename Wmma::ab_fragment (&b_frag1)[KSteps],
    typename Wmma::c_fragment& c00,
    typename Wmma::c_fragment& c01,
    typename Wmma::c_fragment& c10,
    typename Wmma::c_fragment& c11)
{
    #pragma unroll
    for (int step = 0; step < KSteps; ++step) {
        c00 = Wmma::mma(a_frag0[step], b_frag0[step], c00);
        c01 = Wmma::mma(a_frag0[step], b_frag1[step], c01);
        c10 = Wmma::mma(a_frag1[step], b_frag0[step], c10);
        c11 = Wmma::mma(a_frag1[step], b_frag1[step], c11);
    }
}

} // namespace iris::hip
