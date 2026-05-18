#pragma once

#include <cstdint>

#ifndef IRIS_HIP_FRONTEND
#if defined(__HIPCC__) || defined(__HIPRTC__) || defined(__HIP_DEVICE_COMPILE__)
#define IRIS_HIP_FRONTEND 1
#else
#define IRIS_HIP_FRONTEND 0
#endif
#endif

#ifndef IRIS_CUDA_FRONTEND
#if defined(__CUDACC__) || defined(__CUDACC_RTC__) || defined(__NVRTC__)
#define IRIS_CUDA_FRONTEND 1
#else
#define IRIS_CUDA_FRONTEND 0
#endif
#endif

#ifndef IRIS_DEVICE_FRONTEND
#if IRIS_HIP_FRONTEND || IRIS_CUDA_FRONTEND
#define IRIS_DEVICE_FRONTEND 1
#else
#define IRIS_DEVICE_FRONTEND 0
#endif
#endif

#ifndef IRIS_AMDGCN_FRONTEND
#if IRIS_HIP_FRONTEND && !IRIS_CUDA_FRONTEND
#define IRIS_AMDGCN_FRONTEND 1
#else
#define IRIS_AMDGCN_FRONTEND 0
#endif
#endif

#ifndef IRIS_HOST_DEVICE
#if IRIS_DEVICE_FRONTEND
#define IRIS_HOST_DEVICE __host__ __device__
#else
#define IRIS_HOST_DEVICE
#endif
#endif

#ifndef IRIS_DEVICE
#if IRIS_DEVICE_FRONTEND
#define IRIS_DEVICE __device__
#else
#define IRIS_DEVICE
#endif
#endif

#ifndef IRIS_FORCEINLINE
#if IRIS_DEVICE_FRONTEND
#define IRIS_FORCEINLINE __forceinline__
#else
#define IRIS_FORCEINLINE inline
#endif
#endif

#ifndef IRIS_DEVICE_INLINE
#define IRIS_DEVICE_INLINE IRIS_DEVICE IRIS_FORCEINLINE
#endif

#ifndef IRIS_HOST_DEVICE_INLINE
#define IRIS_HOST_DEVICE_INLINE IRIS_HOST_DEVICE IRIS_FORCEINLINE
#endif

namespace iris::hip {

struct Coord1D {
    int i;
};

struct Coord2D {
    int i;
    int j;
};

struct Access1D {
    int offset;
    int count;
};

struct Access2D {
    int row;
    int col;
    int count;
};

struct PartitionIndex {
    int block;
    int thread;
    int wave;
    int lane;
};

template <typename T>
IRIS_HOST_DEVICE_INLINE constexpr T ceil_div(T x, T y) {
    return (x + y - 1) / y;
}

template <typename T>
IRIS_HOST_DEVICE_INLINE constexpr T round_up(T x, T y) {
    return ceil_div(x, y) * y;
}

template <typename T>
IRIS_HOST_DEVICE_INLINE constexpr bool is_power_of_two(T x) {
    return x > 0 && (x & (x - 1)) == 0;
}

template <typename T>
IRIS_HOST_DEVICE_INLINE constexpr T min_value(T a, T b) {
    return a < b ? a : b;
}

} // namespace iris::hip
