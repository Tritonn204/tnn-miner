#pragma once

#include "coordinate.hpp"
#include "lds_view.hpp"

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

} // namespace iris::hip
