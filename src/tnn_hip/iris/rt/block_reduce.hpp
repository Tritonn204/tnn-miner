#pragma once

#include "arch_traits.hpp"
#include "coordinate.hpp"
#include "lds_view.hpp"
#include "warp_primitives.hpp"

namespace iris::hip {

template <int Threads>
IRIS_DEVICE_INLINE uint32_t block_reduce_xor_u32(uint32_t value, uint32_t* scratch) {
    static_assert(Threads > 0, "Threads must be positive");
    static_assert((Threads & (Threads - 1)) == 0, "Threads must be a power of two");

    const int tid = static_cast<int>(threadIdx.x);
    scratch[tid] = value;
    block_sync();

    for (int offset = Threads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] ^= scratch[tid + offset];
        }
        block_sync();
    }

    return scratch[0];
}

template <int Threads>
IRIS_DEVICE_INLINE uint32_t block_reduce_xor_u32_wave_first(uint32_t value, uint32_t* scratch) {
    static_assert(Threads > 0, "Threads must be positive");
    static_assert((Threads & (Threads - 1)) == 0, "Threads must be a power of two");
    static_assert(Threads % arch::wave_size == 0, "Threads must be a whole number of waves");

    constexpr int WaveSize = arch::wave_size;
    constexpr int Waves = Threads / WaveSize;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (WaveSize - 1);
    const int wave = tid / WaveSize;

    uint32_t x = value;
    for (int offset = WaveSize / 2; offset > 0; offset >>= 1) {
        x ^= shuffle_xor_u32(x, offset);
    }

    if (lane == 0) {
        scratch[wave] = x;
    }
    block_sync();

    uint32_t y = 0;
    if (wave == 0) {
        y = lane < Waves ? scratch[lane] : 0u;
        for (int offset = WaveSize / 2; offset > 0; offset >>= 1) {
            y ^= shuffle_xor_u32(y, offset);
        }
        if (lane == 0) {
            scratch[0] = y;
        }
    }
    block_sync();

    return scratch[0];
}

} // namespace iris::hip
