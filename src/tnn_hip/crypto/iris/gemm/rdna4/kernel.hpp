#pragma once

#include "src/tnn_hip/crypto/iris/gemm/rdna4/contract.hpp"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1200__) && !defined(__gfx1201__)
#error "RDNA4 WMMA requires gfx1200/gfx1201; never substitute gfx11 operands"
#endif

namespace tnn::hip::iris::gemm::rdna4 {

using I2 = int __attribute__((ext_vector_type(2)));
using I8 = int __attribute__((ext_vector_type(8)));
using U4 = unsigned __attribute__((ext_vector_type(4)));

__device__ __forceinline__ I8 mma(I2 a, I2 b, I8 c) {
    // Signed A and B, nonsaturating accumulation. Pearl's supported K range
    // cannot overflow int32 even at the signed INT8 extremes.
    asm volatile("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]"
        : "+v"(c) : "v"(a), "v"(b));
    return c;
}

// Generic tiled GEMM. The consumer sees cumulative accumulators after each
// K step; it owns checkpoint frequency and any epilogue/proof semantics.
template<class R, class Consumer>
__device__ __forceinline__ void run(const signed char* a, const signed char* b,
                                   unsigned m, unsigned n, unsigned k,
                                   unsigned char* shared, Consumer& consumer) {
    const unsigned tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const unsigned bm = blockIdx.x / (n / 128) * 128;
    const unsigned bn = blockIdx.x % (n / 128) * 128;
    I8 accum[4][4]{};

    // B is K-contiguous; A is M-contiguous (the production packed operand ABI).
    // The load width is independent from the WMMA fragment width.
    U4 pending_a[R::step / R::load_bytes];
    U4 pending_b[R::step / R::load_bytes];
    auto load = [&](unsigned base) {
#pragma unroll
        for (unsigned chunk = 0; chunk < R::step / R::load_bytes; ++chunk) {
            const unsigned index = (chunk * 128 + tid) * R::load_bytes;
            const auto* pa = a + bm + index % 128 + size_t(base + index / 128) * m;
            const auto* pb = b + size_t(bn + index / R::step) * k + base + index % R::step;
            if constexpr (R::load_bytes == 16) {
                pending_a[chunk] = *reinterpret_cast<const U4*>(pa);
                pending_b[chunk] = *reinterpret_cast<const U4*>(pb);
            } else {
                const I2 av = *reinterpret_cast<const I2*>(pa);
                const I2 bv = *reinterpret_cast<const I2*>(pb);
                pending_a[chunk] = U4{unsigned(av[0]), unsigned(av[1]), 0, 0};
                pending_b[chunk] = U4{unsigned(bv[0]), unsigned(bv[1]), 0, 0};
            }
        }
    };
    auto publish = [&](unsigned bank) {
#pragma unroll
        for (unsigned chunk = 0; chunk < R::step / R::load_bytes; ++chunk) {
            const unsigned offset = (chunk * 128 + tid) * R::load_bytes;
            auto* sa = shared + bank * 2 * R::operand_bytes + offset;
            auto* sb = sa + R::operand_bytes;
            if constexpr (R::load_bytes == 16) {
                *reinterpret_cast<U4*>(sa) = pending_a[chunk];
                *reinterpret_cast<U4*>(sb) = pending_b[chunk];
            } else {
                *reinterpret_cast<I2*>(sa) = I2{int(pending_a[chunk][0]), int(pending_a[chunk][1])};
                *reinterpret_cast<I2*>(sb) = I2{int(pending_b[chunk][0]), int(pending_b[chunk][1])};
            }
        }
        // Compiler-visible loads/stores and the full workgroup barrier own
        // readiness. No gfx11 wait-counter encodings enter this path.
        __syncthreads();
    };

    load(0);
    publish(0);
    for (unsigned base = 0, iteration = 0; base < k; base += R::step, ++iteration) {
        const unsigned bank = iteration % R::banks;
        const auto* sa = shared + bank * 2 * R::operand_bytes;
        const auto* sb = sa + R::operand_bytes;
        if constexpr (R::banks == 2)
            if (base + R::step < k) load(base + R::step);

#pragma unroll
        for (unsigned inner = 0; inner < R::step; inner += 16) {
            I2 af[4], bf[4];
#pragma unroll
            for (unsigned atom = 0; atom < 4; ++atom) {
                unsigned ap[2]{};
                const unsigned ar = wave % 2 * 16 + atom * 32 + lane % 16;
#pragma unroll
                for (unsigned byte = 0; byte < 8; ++byte)
                    ap[byte / 4] |= unsigned(sa[(inner + I8Atom::input_k(lane, byte)) * 128 + ar])
                                     << ((byte % 4) * 8);
                af[atom] = I2{int(ap[0]), int(ap[1])};
                const unsigned bc = wave / 2 * 64 + atom * 16 + lane % 16;
                bf[atom] = *reinterpret_cast<const I2*>(sb + bc * R::step + inner + lane / 16 * 8);
            }
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    accum[i][j] = mma(af[i], bf[j], accum[i][j]);
        }
        consumer.checkpoint(accum, base + R::step, lane, wave);
        __syncthreads(); // All old-bank readers retire before reuse.
        if (base + R::step < k) {
            if constexpr (R::banks == 1) load(base + R::step);
            publish((iteration + 1) % R::banks);
        }
    }
    consumer.finish(accum, bm, bn, lane, wave);
}

} // namespace tnn::hip::iris::gemm::rdna4
