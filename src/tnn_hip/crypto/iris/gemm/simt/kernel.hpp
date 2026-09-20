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

template<unsigned Step, unsigned Banks, unsigned LoadBytes, bool PackedDot = false,
         unsigned TileN = 128>
struct Recipe {
    static_assert(Step == 32 || Step == 64);
    static_assert(Banks == 1 || Banks == 2);
    static_assert(LoadBytes == 1 || LoadBytes == 8 || LoadBytes == 16);
    static_assert(TileN == 64 || TileN == 128);
    static constexpr unsigned tile_n = TileN;
    static constexpr unsigned elements = TileN / 16;
    using Vector = int __attribute__((ext_vector_type(elements)));
    using Accumulator = Vector[4][4];
    static constexpr unsigned step = Step;
    static constexpr unsigned banks = Banks;
    static constexpr unsigned load_bytes = LoadBytes;
    static constexpr bool packed_dot = PackedDot;
    static constexpr unsigned operand_bytes = 128 * Step;
    static constexpr unsigned bank_bytes = (128 + TileN) * Step;
    static constexpr unsigned lds_bytes = Banks * bank_bytes;
};

template<class R> struct Tile {
    __device__ static unsigned row(unsigned group, unsigned atom, unsigned lane, unsigned e) {
        return group % 2 * 16 + atom * 32 + lane / 16 * 8 + e +
               (R::tile_n == 64 ? group / 2 * 4 : 0);
    }
    __device__ static unsigned col(unsigned group, unsigned atom, unsigned lane) {
        return (R::tile_n == 128 ? group / 2 * 64 : 0) + atom * 16 + lane % 16;
    }
};

// Both operand layouts have contiguous, aligned spans. Keep the copy type
// explicit: a nominal LoadBytes parameter must actually change the loads.
template<class R>
__device__ __forceinline__ void stage(const signed char* a, const signed char* b,
                                     unsigned m, unsigned k, unsigned bm, unsigned bn,
                                     unsigned base, unsigned bank, unsigned char* shared) {
    auto* sa = reinterpret_cast<signed char*>(shared) + bank * R::bank_bytes;
    auto* sb = sa + R::operand_bytes;

    if constexpr (R::load_bytes == 1) {
        for (unsigned offset = threadIdx.x; offset < R::operand_bytes; offset += 128) {
            sa[offset] = a[bm + offset % 128 + size_t(base + offset / 128) * m];
            if (offset < R::tile_n * R::step)
                sb[offset] = b[size_t(bn + offset / R::step) * k + base + offset % R::step];
        }
    } else {
        using Copy = unsigned int __attribute__((ext_vector_type(R::load_bytes / 4)));
        for (unsigned offset = threadIdx.x * R::load_bytes;
             offset < R::operand_bytes; offset += 128 * R::load_bytes) {
            const auto* ga = a + bm + offset % 128 + size_t(base + offset / 128) * m;
            *reinterpret_cast<Copy*>(sa + offset) = *reinterpret_cast<const Copy*>(ga);
            if (offset < R::tile_n * R::step) {
                const auto* gb = b + size_t(bn + offset / R::step) * k + base + offset % R::step;
                *reinterpret_cast<Copy*>(sb + offset) = *reinterpret_cast<const Copy*>(gb);
            }
        }
    }
}

template<class R, class Consumer>
__device__ __forceinline__ void run(const signed char* a, const signed char* b,
                                   unsigned m, unsigned n, unsigned k,
                                   unsigned char* shared, Consumer& consumer) {
    const unsigned tid = threadIdx.x;
    // Logical groups of 32 work on both hardware wave32 and wave64.
    const unsigned lane = tid % 32, group = tid / 32;
    const unsigned bm = blockIdx.x / (n / R::tile_n) * 128;
    const unsigned bn = blockIdx.x % (n / R::tile_n) * R::tile_n;
    typename R::Accumulator accum{};

    stage<R>(a, b, m, k, bm, bn, 0, 0, shared);
    __syncthreads();

    for (unsigned base = 0; base < k; base += R::step) {
        const unsigned bank = (base / R::step) % R::banks;
        auto* sa = reinterpret_cast<signed char*>(shared) + bank * R::bank_bytes;
        auto* sb = sa + R::operand_bytes;

        // The next bank has no readers. Its stores may overlap current-bank
        // arithmetic; the end barrier publishes them before the bank swap.
        if constexpr (R::banks == 2) {
            if (base + R::step < k)
                stage<R>(a, b, m, k, bm, bn, base + R::step, bank ^ 1, shared);
        }

        for (unsigned inner = 0; inner < R::step; inner += R::packed_dot ? 4 : 1) {
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
#pragma unroll
                for (unsigned e = 0; e < R::elements; ++e) {
                    const unsigned row = Tile<R>::row(group, i, lane, e);
                    unsigned packed_a = 0;
                    if constexpr (R::packed_dot) {
#pragma unroll
                        for (unsigned byte = 0; byte < 4; ++byte)
                            packed_a |= unsigned(static_cast<unsigned char>(sa[(inner + byte) * 128 + row])) << (byte * 8);
                    }
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j) {
                        const unsigned col = Tile<R>::col(group, j, lane);
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

        if constexpr (R::banks == 1) {
            if (base + R::step < k) {
                stage<R>(a, b, m, k, bm, bn, base + R::step, 0, shared);
                __syncthreads();
            }
        }
    }
    consumer.finish(accum, bm, bn, lane, group);
}

} // namespace tnn::hip::iris::gemm::simt
