#include <hip/hip_runtime.h>
#if defined(PEARL_CDNA_MFMA)
#include "src/tnn_hip/crypto/iris/gemm/cdna/kernel.hpp"
#include "src/tnn_hip/crypto/pearl/cdna/ownership.hpp"
#elif defined(PEARL_PORTABLE_SIMT)
#include "src/tnn_hip/crypto/iris/gemm/simt/kernel.hpp"
#else
#include "src/tnn_hip/crypto/iris/gemm/rdna4/kernel.hpp"
#endif
#include "src/tnn_hip/crypto/pearl/rdna4/ownership.hpp"
#include "src/tnn_hip/crypto/blake3-inline.hip.inc"

#ifndef PEARL_GFX12_RECIPE
#define PEARL_GFX12_RECIPE 0
#endif
static_assert(PEARL_GFX12_RECIPE >= 0 && PEARL_GFX12_RECIPE < 8);

#if defined(__HIP_DEVICE_COMPILE__) && defined(PEARL_PORTABLE_SIMT) && (PEARL_GFX12_RECIPE & 1)
static_assert(__AMDGCN_WAVEFRONT_SIZE == 64, "Requested wave64 validation compiled with a different wave size");
#endif

namespace tnn::pearl::gpu::rdna4 {
#if defined(PEARL_CDNA_MFMA)
namespace gemm = tnn::hip::iris::gemm::cdna;
#elif defined(PEARL_PORTABLE_SIMT)
namespace gemm = tnn::hip::iris::gemm::simt;
#else
namespace gemm = tnn::hip::iris::gemm::rdna4;
#endif
#ifdef PEARL_PORTABLE_SIMT
using R = gemm::Recipe<32, 1, 8, (PEARL_GFX12_RECIPE & 2) != 0>;
#else
using R = gemm::Recipe<(PEARL_GFX12_RECIPE & 4) ? 64 : 32,
                       (PEARL_GFX12_RECIPE & 2) ? 2 : 1,
                       (PEARL_GFX12_RECIPE & 1) ? 16 : 8>;
#endif

struct Winner { unsigned row, col, digest[8]; };

template<bool Diagnostic> struct Pearl {
    unsigned* transcript;
    unsigned n;
    const unsigned *key, *target;
    Winner* winners;
    unsigned *state, capacity, *diagnostic;

    __device__ void checkpoint(gemm::Accumulator& acc, unsigned end, unsigned lane, unsigned) {
        if (end % 128) return;
        unsigned reduction = 0;
#ifdef PEARL_CDNA_MFMA
        // Even/odd M atoms correspond to Pearl's rows separated by 32.
        // Both lane halves execute both reductions before selecting a ticket.
        // No RDNA accumulator reinterpretation or cross-wave LDS transpose.
#pragma unroll
        for (unsigned parity = 0; parity < 2; ++parity)
#pragma unroll
            for (unsigned e = 0; e < 4; ++e) {
                unsigned value = 0;
#pragma unroll
                for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j)
                        value ^= unsigned(acc[tnn::pearl::gpu::cdna::Ticket::atom_row(parity, i)][j][e]);
                value ^= __shfl_xor(value, 1, 64);
                value ^= __shfl_xor(value, 2, 64);
                value ^= __shfl_xor(value, 8, 64);
                value = __shfl(value, tnn::pearl::gpu::cdna::Ticket::source_lane(lane), 64);
                if (parity == lane / 32 && e == tnn::pearl::gpu::cdna::Ticket::source_element(lane))
                    reduction = value;
            }
#else
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            unsigned value = 0;
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    value ^= unsigned(acc[i][j][e]);
            // Reduce the eight ticket columns represented by each N atom.
            // Every lane executes every shuffle before choosing its row.
            value ^= __shfl_xor(value, 1, 32);
            value ^= __shfl_xor(value, 2, 32);
            value ^= __shfl_xor(value, 8, 32);
            value = __shfl(value, Ticket::source_lane(lane), 32);
            if (e == Ticket::source_element(lane)) reduction = value;
        }
#endif
        const unsigned slot = ((end / 128 - 1) % 16) * 128 + threadIdx.x;
        const unsigned previous = transcript[slot];
        transcript[slot] = ((previous << 13) | (previous >> 19)) ^ reduction;
    }

    __device__ void finish(gemm::Accumulator&, unsigned bm, unsigned bn,
                           unsigned lane, unsigned wave) {
#ifdef PEARL_CDNA_MFMA
        // The wire/proof ticket ordering is independent of hardware wave size.
        lane = threadIdx.x % 32;
        wave = threadIdx.x / 32;
#endif
        unsigned words[16], digest[8];
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) words[i] = transcript[i * 128 + threadIdx.x];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) digest[i] = key[i];
        blake3_compress_in_place(digest, reinterpret_cast<const uint8_t*>(words), 64, 0,
                                 CHUNK_START | CHUNK_END | ROOT | (1 << 4));
        const unsigned row = bm + Ticket::row(wave, lane);
        const unsigned col = bn + Ticket::col(wave, lane);
        if constexpr (Diagnostic) {
            // Tile-major diagnostic ABI, explicitly separate from proof coordinates.
            const unsigned position = blockIdx.x * 128 + threadIdx.x;
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) diagnostic[position * 24 + i] = words[i];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) diagnostic[position * 24 + 16 + i] = digest[i];
        }
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
                for (unsigned i = 0; i < 8; ++i) winners[slot].digest[i] = digest[i];
            } else atomicExch(state + 1, 1u);
        }
    }
};

template<bool Diagnostic>
__device__ void fused(const signed char* a, const signed char* b, unsigned m, unsigned n,
                      unsigned k, const unsigned* key, const unsigned* target, Winner* winners,
                      unsigned* state, unsigned capacity, unsigned* diagnostic) {
    if (!m || !n || m % 128 || n % 128 || !k || k % 128) return;
    __shared__ __align__(16) unsigned char storage[R::lds_bytes + 8192];
    auto* transcript = reinterpret_cast<unsigned*>(storage + R::lds_bytes);
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) transcript[i * 128 + threadIdx.x] = 0;
    Pearl<Diagnostic> consumer{transcript, n, key, target, winners, state, capacity, diagnostic};
    gemm::run<R>(a, b, m, n, k, storage, consumer);
}

struct Raw {
    int* d;
    unsigned m;
    __device__ void checkpoint(gemm::Accumulator&, unsigned, unsigned, unsigned) {}
    __device__ void finish(gemm::Accumulator& acc, unsigned bm, unsigned bn,
                           unsigned lane, unsigned wave) {
        if (!d) return; // Timed D-free control; volatile WMMA keeps compute live.
#pragma unroll
        for (unsigned i = 0; i < sizeof(acc) / sizeof(acc[0]); ++i)
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
#pragma unroll
                for (unsigned e = 0; e < sizeof(acc[0][0]) / sizeof(int); ++e)
                    d[bm + gemm::Tile128::row(wave, i, lane, e) +
                      size_t(bn + gemm::Tile128::col(wave, j, lane)) * m] = acc[i][j][e];
    }
};
} // namespace tnn::pearl::gpu::rdna4

using namespace tnn::pearl::gpu::rdna4;

extern "C" __global__ __launch_bounds__(128) void pearl_iris_fused(
    const signed char* a, const signed char* b, int*, unsigned m, unsigned n, unsigned k,
    unsigned, unsigned, unsigned, const unsigned* key, const unsigned* target,
    Winner* winners, unsigned* state, unsigned capacity, unsigned* diagnostic) {
    fused<false>(a, b, m, n, k, key, target, winners, state, capacity, diagnostic);
}

extern "C" __global__ __launch_bounds__(128) void pearl_iris_diagnostic(
    const signed char* a, const signed char* b, int*, unsigned m, unsigned n, unsigned k,
    unsigned, unsigned, unsigned, const unsigned* key, const unsigned* target,
    Winner* winners, unsigned* state, unsigned capacity, unsigned* diagnostic) {
    fused<true>(a, b, m, n, k, key, target, winners, state, capacity, diagnostic);
}

extern "C" __global__ __launch_bounds__(128) void pearl_iris_raw(
    const signed char* a, const signed char* b, int* d, unsigned m, unsigned n, unsigned k,
    unsigned, unsigned, unsigned) {
    if (!m || !n || m % 128 || n % 128 || !k || k % R::step) return;
    __shared__ __align__(16) unsigned char storage[R::lds_bytes];
    Raw consumer{d, m};
    gemm::run<R>(a, b, m, n, k, storage, consumer);
}
