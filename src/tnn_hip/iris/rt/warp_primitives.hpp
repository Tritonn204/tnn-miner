#pragma once

#include "arch_traits.hpp"
#include "coordinate.hpp"

namespace iris::hip {

IRIS_DEVICE_INLINE uint32_t lane_id() {
#if IRIS_HIP_FRONTEND
    return __lane_id();
#elif IRIS_CUDA_FRONTEND
    return static_cast<uint32_t>(threadIdx.x & 31);
#else
    return 0;
#endif
}

IRIS_DEVICE_INLINE uint32_t byte_perm_u32(uint32_t hi, uint32_t lo, uint32_t selector) {
#if IRIS_AMDGCN_FRONTEND
    return __builtin_amdgcn_perm(hi, lo, selector);
#elif IRIS_CUDA_FRONTEND
    return __byte_perm(lo, hi, selector);
#else
    return lo;
#endif
}

IRIS_DEVICE_INLINE uint32_t exchange_lane_x16_u32(uint32_t v) {
#if IRIS_AMDGCN_FRONTEND && (defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__))
    return __builtin_amdgcn_permlanex16(0, v, 0x76543210, 0xfedcba98, false, true);
#else
    return v;
#endif
}

IRIS_DEVICE_INLINE uint32_t shuffle_xor_u32(uint32_t v, int lane_mask) {
#if IRIS_DEVICE_FRONTEND
    return __shfl_xor(v, lane_mask, arch::wave_size);
#else
    (void)lane_mask;
    return v;
#endif
}

template <int N>
IRIS_DEVICE_INLINE void copy_u32_words(uint32_t (&out)[N], const uint32_t (&in)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        out[i] = in[i];
    }
}

} // namespace iris::hip
