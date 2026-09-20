#pragma once

#include "src/tnn_hip/crypto/iris/gemm/cdna/contract.hpp"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx908__) && !defined(__gfx90a__) && !defined(__gfx942__)
#error "CDNA MFMA baseline supports only independently mapped gfx908/gfx90a/gfx942"
#endif

namespace tnn::hip::iris::gemm::cdna {

using I4 = int __attribute__((ext_vector_type(4)));
using Accumulator = I4[8][4];

#if defined(__gfx942__)
using Atom = I8Atom<8>;
using Packed = long long;

__device__ __forceinline__ I4 mma(Packed a, Packed b, I4 c) {
    return __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c, 0, 0, 0);
}
#else
using Atom = I8Atom<4>;
using Packed = int;

__device__ __forceinline__ I4 mma(Packed a, Packed b, I4 c) {
    return __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, c, 0, 0, 0);
}
#endif

template<unsigned Step, unsigned Banks, unsigned LoadBytes>
struct Recipe {
    static_assert(Step == 32 && Banks == 1 && LoadBytes == 8,
                  "Only the correctness-first CDNA recipe is implemented");
    static constexpr unsigned step = Step;
    static constexpr unsigned lds_bytes = 2 * 128 * Step;
};

template<class R, class Consumer>
__device__ __forceinline__ void run(const signed char* a, const signed char* b,
                                   unsigned m, unsigned n, unsigned k,
                                   unsigned char* shared, Consumer& consumer) {
    const unsigned tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const unsigned bm = blockIdx.x / (n / 128) * 128;
    const unsigned bn = blockIdx.x % (n / 128) * 128;
    auto* sa = shared;
    auto* sb = shared + 128 * R::step;
    Accumulator accum{};

    for (unsigned base = 0; base < k; base += R::step) {
        for (unsigned offset = tid; offset < 128 * R::step; offset += 128) {
            sa[offset] = a[bm + offset % 128 + size_t(base + offset / 128) * m];
            sb[offset] = b[size_t(bn + offset / R::step) * k + base + offset % R::step];
        }
        __syncthreads();

#pragma unroll
        for (unsigned inner = 0; inner < R::step; inner += Atom::depth) {
            Packed af[8], bf[4];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                unsigned long long bits = 0;
#pragma unroll
                for (unsigned byte = 0; byte < Atom::input_bytes; ++byte)
                    bits |= static_cast<unsigned long long>(sa[(inner + Atom::input_k(lane, byte)) * 128 +
                                                              i * 16 + Atom::input_outer(lane)]) << (byte * 8);
                af[i] = Packed(bits);
            }
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                bf[j] = *reinterpret_cast<const Packed*>(sb +
                    (wave * 64 + j * 16 + Atom::input_outer(lane)) * R::step +
                    inner + Atom::input_k(lane, 0));

#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    accum[i][j] = mma(af[i], bf[j], accum[i][j]);
        }

        consumer.checkpoint(accum, base + R::step, lane, wave);
        __syncthreads();
    }
    consumer.finish(accum, bm, bn, lane, wave);
}

} // namespace tnn::hip::iris::gemm::cdna
