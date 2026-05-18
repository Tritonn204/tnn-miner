#pragma once

#include "arch_traits.hpp"
#include "warp_primitives.hpp"

namespace iris::hip {

template <typename T>
struct WmmaCToA2ByteTraits {
    static constexpr bool enabled = sizeof(T) == 2 && arch::has_rdna3_wmma;
    static constexpr int out_words_per_in_word = 2;
};

template <int InWords>
IRIS_DEVICE_INLINE void permute_gfx11_wmma_c_to_a_u32(uint32_t (&out)[InWords * 2], const uint32_t (&in)[InWords]) {
    static_assert(InWords > 0, "InWords must be positive");

#if IRIS_AMDGCN_FRONTEND && (defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__))
    const uint32_t selector0 = lane_id() < 16 ? 0x05040100u : 0x01000504u;
    const uint32_t selector1 = lane_id() < 16 ? 0x07060302u : 0x03020706u;

    #pragma unroll
    for (int i = 0; i < InWords; ++i) {
        const uint32_t v = in[i];
        const uint32_t w = exchange_lane_x16_u32(v);
        out[i * 2 + 0] = byte_perm_u32(w, v, selector0);
        out[i * 2 + 1] = byte_perm_u32(w, v, selector1);
    }
#else
    #pragma unroll
    for (int i = 0; i < InWords; ++i) {
        out[i * 2 + 0] = in[i];
        out[i * 2 + 1] = in[i];
    }
#endif
}

template <int N>
IRIS_DEVICE_INLINE void permute_gfx11_dropout_randvals_u8(uint8_t (&values)[N]) {
    static_assert(N % 8 == 0, "Dropout randval permutation expects groups of 8 bytes");

#if IRIS_AMDGCN_FRONTEND && (defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__))
    #pragma unroll
    for (int offset = 0; offset < N; offset += 8) {
        const uint32_t r0 =
            (static_cast<uint32_t>(values[offset + 0]) << 0) |
            (static_cast<uint32_t>(values[offset + 1]) << 8) |
            (static_cast<uint32_t>(values[offset + 2]) << 16) |
            (static_cast<uint32_t>(values[offset + 3]) << 24);
        const uint32_t r1 =
            (static_cast<uint32_t>(values[offset + 4]) << 0) |
            (static_cast<uint32_t>(values[offset + 5]) << 8) |
            (static_cast<uint32_t>(values[offset + 6]) << 16) |
            (static_cast<uint32_t>(values[offset + 7]) << 24);

        const uint32_t v0 = byte_perm_u32(r1, r0, 0x06040200u);
        const uint32_t v1 = byte_perm_u32(r1, r0, 0x07050301u);
        const uint32_t w0 = exchange_lane_x16_u32(v0);
        const uint32_t w1 = exchange_lane_x16_u32(v1);
        const uint32_t lo = lane_id() < 16 ? v0 : w1;
        const uint32_t hi = lane_id() < 16 ? w0 : v1;

        values[offset + 0] = static_cast<uint8_t>((lo >> 0) & 0xff);
        values[offset + 1] = static_cast<uint8_t>((lo >> 8) & 0xff);
        values[offset + 2] = static_cast<uint8_t>((lo >> 16) & 0xff);
        values[offset + 3] = static_cast<uint8_t>((lo >> 24) & 0xff);
        values[offset + 4] = static_cast<uint8_t>((hi >> 0) & 0xff);
        values[offset + 5] = static_cast<uint8_t>((hi >> 8) & 0xff);
        values[offset + 6] = static_cast<uint8_t>((hi >> 16) & 0xff);
        values[offset + 7] = static_cast<uint8_t>((hi >> 24) & 0xff);
    }
#endif
}

} // namespace iris::hip
