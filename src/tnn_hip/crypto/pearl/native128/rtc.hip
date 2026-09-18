#include <hip/hip_runtime.h>
#include "src/tnn_hip/iris/rt/gemm_schedule.hpp"
#include "src/tnn_hip/crypto/iris/gemm/native128/recipe.hpp"

#include "src/tnn_hip/crypto/iris/gemm/native128/buffer_inputs.hpp"

#include "src/tnn_hip/crypto/blake3-inline.hip.inc"
#include "layout.hpp"
#include "src/tnn_hip/crypto/iris/gemm/native128/raw_kernel.hpp"
struct NativeWinner {
    unsigned row, col, digest[8];
};

namespace tnn::pearl::gpu::native128 {
using namespace tnn::hip::iris::gemm::native128;
using I8 = int __attribute__((ext_vector_type(8)));
#include "slots.hpp"

template <unsigned Stages, class Traversal> struct InfrastructureRecipe : Native128 {
    // The physical-register assembly is qualified for exactly this schedule.
    // Other policies must get their own specialization, not silently use it.
    static_assert(Stages == 2);
    static_assert(__is_same(Traversal, ::iris::hip::LinearK));

    using Storage = ::iris::hip::LdsStages<4096, 4608, Stages, 16, 16384>;
};

template <class Configuration, class LoadPolicy, bool Diagnostic>
__device__ __forceinline__ void
prefetched_body(const int8_t *a, const int8_t *b, int32_t *d, unsigned m, unsigned n, unsigned k,
                unsigned lda, unsigned ldb, unsigned ldd, const uint32_t *key,
                const uint32_t *target, NativeWinner *winners, uint32_t *state, unsigned capacity,
                uint32_t *diagnostic) {
    using Arch = typename Configuration::Arch;
    // Uniform contract guard: this recipe only handles complete tiles.
    if (!m || !n || m % 128 || n % 128 || k < 2048 || k % 128)
        return;

    __shared__ __align__(16) unsigned char storage[NativeTranscript::allocation_bytes];
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
    slot_partial(accumulators, loads, k / 32, m * 32, read_a, read_b, store_a, store_b,
                 __builtin_amdgcn_readfirstlane(base), __builtin_amdgcn_readfirstlane(wave));

    // All checkpoints are now in LDS. Reconstruct the lane identity after
    // the fixed-register GEMM block, then hash each lane's own transcript.
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    unsigned fresh_tid;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\tv_add_u32 %0, %1, %0"
                 : "=&v"(fresh_tid)
                 : "s"(__builtin_amdgcn_readfirstlane(threadIdx.x)));
    const unsigned fresh_lane = fresh_tid % 32, fresh_wave = fresh_tid / 32;
    uint32_t transcript[16], digest[8];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
        transcript[i] =
            *reinterpret_cast<const uint32_t *>(storage + NativeTranscript::offset(i, fresh_tid));
    const unsigned row = bm + fresh_wave % 2 * 16 + fresh_lane % 16;
    const unsigned col = bn + fresh_wave / 2 * 64 + fresh_lane / 16 * 4;
    const unsigned position = (row / 128 * 32 + row % 128) * (n / 32) + col / 64 * 2 + col % 64 / 4;
    const unsigned positions = m * n / 128;
    // One complete keyed BLAKE3 root block per candidate. No probabilistic
    // gate and no cross-lane jackpot coordination are needed.
#pragma unroll
    for (unsigned i = 0; i < 8; ++i)
        digest[i] = key[i];

    blake3_compress_in_place(digest, reinterpret_cast<const uint8_t *>(transcript),
                            64, 0, CHUNK_START | CHUNK_END | ROOT | (1 << 4));

    if constexpr (Diagnostic) {
#pragma unroll
        for (unsigned i = 0; i < 16; ++i)
            diagnostic[size_t(i) * positions + position] = transcript[i];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i)
            diagnostic[size_t(i + 16) * positions + position] = digest[i];
    }
    // Targets are little-endian U256 values: compare the high word first.
    // Record every passing candidate; overflow is a host-visible failure.
    bool equal = true, lower = false;
#pragma unroll
    for (int i = 7; i >= 0; --i) {
        lower |= equal && digest[i] < target[i];
        equal &= digest[i] == target[i];
    }
    if (lower || equal) {
        const unsigned slot = atomicAdd(state, 1u);
        if (slot < capacity) {
            winners[slot].row = row;
            winners[slot].col = col;
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
                winners[slot].digest[i] = digest[i];
        } else
            atomicExch(state + 1, 1u);
    }
}

extern "C" __global__ __launch_bounds__(128) void pearl_iris_fused(
    const int8_t *a, const int8_t *b, int32_t *d, unsigned m, unsigned n, unsigned k, unsigned lda,
    unsigned ldb, unsigned ldd, const uint32_t *key, const uint32_t *target, NativeWinner *winners,
    uint32_t *state, unsigned capacity, uint32_t *diagnostic) {
    prefetched_body<InfrastructureRecipe<2, ::iris::hip::LinearK>,
                    tnn::hip::iris::gemm::native128::Gfx1100BufferInputs128<>, false>(
        a, b, d, m, n, k, lda, ldb, ldd, key, target, winners, state, capacity, diagnostic);
}
extern "C" __global__ __launch_bounds__(128) void pearl_iris_diagnostic(
    const int8_t *a, const int8_t *b, int32_t *d, unsigned m, unsigned n, unsigned k, unsigned lda,
    unsigned ldb, unsigned ldd, const uint32_t *key, const uint32_t *target, NativeWinner *winners,
    uint32_t *state, unsigned capacity, uint32_t *diagnostic) {
    prefetched_body<InfrastructureRecipe<2, ::iris::hip::LinearK>,
                    tnn::hip::iris::gemm::native128::Gfx1100BufferInputs128<>, true>(
        a, b, d, m, n, k, lda, ldb, ldd, key, target, winners, state, capacity, diagnostic);
}
} // namespace tnn::pearl::gpu::native128

namespace tnn::pearl::gpu::native128 {
extern "C" __global__ __launch_bounds__(128) void pearl_iris_raw(const int8_t *a, const int8_t *b,
                                                                 int32_t *d, unsigned m, unsigned n,
                                                                 unsigned k, unsigned lda,
                                                                 unsigned ldb, unsigned ldd) {
    raw_body<InfrastructureRecipe<2, ::iris::hip::LinearK>,
             tnn::hip::iris::gemm::native128::Gfx1100BufferInputs128<>, true>(a, b, d, m, n, k,
                                                                                 lda, ldb, ldd);
}
} // namespace tnn::pearl::gpu::native128
