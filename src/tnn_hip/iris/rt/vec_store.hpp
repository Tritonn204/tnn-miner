#pragma once

#include <cstdint>

#include "coordinate.hpp"

namespace iris::hip {

// ============================================================================
// Vector type mapping
// ============================================================================
template <typename T, int N>
struct vec_type;

template <> struct vec_type<int8_t,   1> { using type = int8_t; };
template <> struct vec_type<int8_t,   4> { using type = int32_t; };
template <> struct vec_type<int8_t,   8> { using type = int64_t; };
template <> struct vec_type<int8_t,  16> { using type = int4; };
template <> struct vec_type<uint8_t,  1> { using type = uint8_t; };
template <> struct vec_type<uint8_t,  4> { using type = uint32_t; };
template <> struct vec_type<uint8_t,  8> { using type = uint64_t; };
template <> struct vec_type<uint8_t, 16> { using type = uint4; };
template <> struct vec_type<float,    1> { using type = float; };
template <> struct vec_type<float,    2> { using type = float2; };
template <> struct vec_type<float,    4> { using type = float4; };
template <> struct vec_type<uint32_t, 1> { using type = uint32_t; };
template <> struct vec_type<uint32_t, 2> { using type = uint64_t; };
template <> struct vec_type<uint32_t, 4> { using type = uint4; };

// ============================================================================
// Vectorized load/store wrappers
// ============================================================================
template <typename T, int VecBytes = 16>
struct VectorAccess {
    static constexpr int VecElems = VecBytes / sizeof(T);
    using Vec = typename vec_type<T, VecElems>::type;

    IRIS_DEVICE_INLINE static Vec load_aligned(const T* p) {
        return *reinterpret_cast<const Vec*>(p);
    }
    IRIS_DEVICE_INLINE static void store_aligned(T* p, Vec v) {
        *reinterpret_cast<Vec*>(p) = v;
    }

    IRIS_DEVICE_INLINE static bool is_aligned(const T* p) {
        return (reinterpret_cast<uintptr_t>(p) & (VecBytes - 1)) == 0;
    }
};

IRIS_DEVICE_INLINE uint32_t pack_i8x4(const int8_t* p) {
    return (static_cast<uint32_t>(static_cast<uint8_t>(p[0])) << 0) |
           (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}

template <int N>
IRIS_DEVICE_INLINE void store_int8_vec(int8_t* ptr, const int8_t* values) {
    static_assert(N == 32, "Only N=32 supported for now");
    using Vec16 = typename vec_type<int8_t, 16>::type;
    const Vec16 v0{
        static_cast<int>(pack_i8x4(values + 0)),
        static_cast<int>(pack_i8x4(values + 4)),
        static_cast<int>(pack_i8x4(values + 8)),
        static_cast<int>(pack_i8x4(values + 12)),
    };
    const Vec16 v1{
        static_cast<int>(pack_i8x4(values + 16)),
        static_cast<int>(pack_i8x4(values + 20)),
        static_cast<int>(pack_i8x4(values + 24)),
        static_cast<int>(pack_i8x4(values + 28)),
    };
    VectorAccess<int8_t, 16>::store_aligned(ptr, v0);
    VectorAccess<int8_t, 16>::store_aligned(ptr + 16, v1);
}

} // namespace iris::hip
