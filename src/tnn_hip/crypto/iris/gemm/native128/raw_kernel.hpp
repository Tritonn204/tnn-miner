#pragma once

namespace tnn::hip::iris::gemm::native128 {
using I8 = int __attribute__((ext_vector_type(8)));
#include "src/tnn_hip/crypto/iris/gemm/native128/raw_slots.hpp"
template <class Configuration, class LoadPolicy, bool Partial>
__device__ __forceinline__ void raw_body(const int8_t *a, const int8_t *b, int32_t *d, unsigned m,
                                         unsigned n, unsigned k, unsigned lda, unsigned ldb,
                                         unsigned ldd) {
    using Arch = typename Configuration::Arch;
    using Storage = typename Configuration::Storage;
    // Uniform contract guard: this recipe only handles complete tiles.
    if (!m || !n || m % 128 || n % 128 || !k || k % 32)
        return;

    __shared__ __align__(16) unsigned char storage[Storage::allocation_bytes];
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned gm = m / 128, gn = n / 128;
    const unsigned group = blockIdx.x / (gm * 4);
    const unsigned width = min(4u, gn - group * 4);
    const unsigned local = blockIdx.x - group * gm * 4;
    const unsigned tile_m = local / width;
    const unsigned bm = tile_m * 128, bn = (group * 4 + local % width) * 128;
    I8 accumulators[16]{};

    // Buffer descriptors require packed matrices with byte spans below 2^32.
    if (lda != m || ldb != k || ldd != m || size_t(lda) * k > 0xffffffffull ||
        size_t(ldb) * n > 0xffffffffull || (reinterpret_cast<uintptr_t>(a) >> 48) ||
        (reinterpret_cast<uintptr_t>(b) >> 48))
        return;
    LoadPolicy loads(a, b, bm, bn, m, n, k, lane, wave);
    const unsigned base = unsigned(reinterpret_cast<uintptr_t>(storage));
    const unsigned read_a = base + Arch::a_row(wave, lane, 0);
    const unsigned read_b = base + 4096 + Arch::b_offset(Arch::b_col(wave, lane, 0), 0);
    const unsigned store_a = base + (lane % 8) * 16 + (wave * 8 + lane / 8) * 128;
    const unsigned store_b = base + 4096 + Arch::b_offset(wave * 32 + lane / 2, (lane % 2) * 16);
    slot_raw(accumulators, loads, k / 32, m * 32, read_a, read_b, store_a, store_b);
}


} // namespace tnn::hip::iris::gemm::native128
