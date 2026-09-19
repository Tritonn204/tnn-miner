#pragma once

// Integer-only fallback. No matrix instructions, target-specific descriptors,
// or numeric wait-counter assumptions. A is M-contiguous, B K-contiguous.
namespace tnn::hip::iris::gemm::simt {

using I8 = int __attribute__((ext_vector_type(8)));
using Accumulator = I8[4][4];

struct Tile128 {
    __device__ static unsigned row(unsigned group, unsigned atom, unsigned lane, unsigned element) {
        return group % 2 * 16 + atom * 32 + lane / 16 * 8 + element;
    }
    __device__ static unsigned col(unsigned group, unsigned atom, unsigned lane) {
        return group / 2 * 64 + atom * 16 + lane % 16;
    }
};

template<unsigned Step, unsigned Banks, unsigned LoadBytes, bool PackedDot = false>
struct Recipe {
    static_assert(Step == 32 && Banks == 1 && LoadBytes == 8,
                  "Only the baseline SIMT recipe is implemented");
    static constexpr unsigned step = Step;
    static constexpr bool packed_dot = PackedDot;
    static constexpr unsigned lds_bytes = 2 * 128 * Step;
};

template<class R, class Consumer>
__device__ __forceinline__ void run(const signed char* a, const signed char* b,
                                   unsigned m, unsigned n, unsigned k,
                                   unsigned char* shared, Consumer& consumer) {
    const unsigned tid = threadIdx.x;
    // Logical groups of 32 work on both hardware wave32 and wave64.
    const unsigned lane = tid % 32, group = tid / 32;
    const unsigned bm = blockIdx.x / (n / 128) * 128;
    const unsigned bn = blockIdx.x % (n / 128) * 128;
    auto* sa = reinterpret_cast<signed char*>(shared);
    auto* sb = sa + 128 * R::step;
    I8 accum[4][4]{};

    for (unsigned base = 0; base < k; base += R::step) {
        // Global A reads are coalesced across threads. B staging follows its
        // K-contiguous layout. Explicit barriers publish and retire the bank.
        for (unsigned offset = tid; offset < 128 * R::step; offset += 128) {
            sa[offset] = a[bm + offset % 128 + size_t(base + offset / 128) * m];
            sb[offset] = b[size_t(bn + offset / R::step) * k + base + offset % R::step];
        }
        __syncthreads();

        for (unsigned inner = 0; inner < R::step; inner += R::packed_dot ? 4 : 1) {
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) {
                    const unsigned row = group % 2 * 16 + i * 32 + lane / 16 * 8 + e;
                    unsigned packed_a = 0;
                    if constexpr (R::packed_dot) {
#pragma unroll
                        for (unsigned byte = 0; byte < 4; ++byte)
                            packed_a |= unsigned(static_cast<unsigned char>(sa[(inner + byte) * 128 + row])) << (byte * 8);
                    }
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j) {
                        const unsigned col = group / 2 * 64 + j * 16 + lane % 16;
                        if constexpr (R::packed_dot) {
                            const unsigned packed_b = *reinterpret_cast<const unsigned*>(sb + col * R::step + inner);
                            // Four signed products, with saturation disabled. This
                            // intrinsic must compile for the exact target; gfx900
                            // and RDNA1 intentionally retain scalar arithmetic.
#if defined(__gfx1100__)
                            // RDNA3's local validation control has independently
                            // selectable input signs instead of the older opcode.
                            accum[i][j][e] = __builtin_amdgcn_sudot4(
                                true, int(packed_a), true, int(packed_b), accum[i][j][e], false);
#else
                            accum[i][j][e] = __builtin_amdgcn_sdot4(
                                int(packed_a), int(packed_b), accum[i][j][e], false);
#endif
                        } else {
                            accum[i][j][e] += int(sa[inner * 128 + row]) * int(sb[col * R::step + inner]);
                        }
                    }
                }
            }
        }
        consumer.checkpoint(accum, base + R::step, lane, group);
        __syncthreads();
    }
    consumer.finish(accum, bm, bn, lane, group);
}

} // namespace tnn::hip::iris::gemm::simt
