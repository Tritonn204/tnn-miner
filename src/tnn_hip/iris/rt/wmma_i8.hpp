#pragma once

#include "arch_traits.hpp"
#include "coordinate.hpp"
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

template <bool SignedA = true, bool SignedB = true, bool Clamp = false>
struct WmmaI8I8I32_16x16x16 {
    static constexpr int m = 16;
    static constexpr int n = 16;
    static constexpr int k = 16;
    static constexpr int c_elems = 8;

#if IRIS_AMDGCN_FRONTEND && (defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__))
    static constexpr bool native = true;
    static constexpr int ab_elems = 16;
    using ab_fragment = i8x16_t;
#elif IRIS_AMDGCN_FRONTEND && (defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__))
    // Keep the type distinction here so kernel code can remain architecture-neutral.
    // The first Pearl WMMA path uses gfx11; gfx12 fragment loading is intentionally
    // staged behind this trait before enabling the native call there.
    static constexpr bool native = false;
    static constexpr int ab_elems = 8;
    using ab_fragment = i8x8_t;
#else
    static constexpr bool native = false;
    static constexpr int ab_elems = 16;
    using ab_fragment = i8x16_t;
#endif

    using c_fragment = i32x8_t;

    IRIS_DEVICE_INLINE static ab_fragment zero_ab() {
        ab_fragment out{};
        return out;
    }

    IRIS_DEVICE_INLINE static c_fragment zero_c() {
        c_fragment out{};
        return out;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int c_row(int lane, int elem) {
        return elem * 2 + lane / 16;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int c_col(int lane) {
        return lane & 15;
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

} // namespace iris::hip
