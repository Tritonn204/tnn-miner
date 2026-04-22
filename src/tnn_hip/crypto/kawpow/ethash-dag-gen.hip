#ifndef ETHASH_DAG_GEN_HIP
#define ETHASH_DAG_GEN_HIP

// ============================================================================
// ethash-dag-gen.hip — GPU-accelerated DAG generation for KawPow/ethash
//
// Each thread generates one 2048-bit DAG item (4 × 512-bit sub-items) from
// the light cache using the standard ethash algorithm:
//   1. Init: cache lookup + seed XOR + keccak512
//   2. 512 parent rounds: fnv1(mix, cache[parent_index])
//   3. Final: keccak512
//
// keccak-f1600 is implemented inline below (dag_keccakf) with full 24
// rounds. The existing .hip.inc files (keccak-tiny, keccak-amd, keccak-nvidia)
// ALL have last-round pruning (only compute state[0..4] in round 24) —
// correct for KawPow's main kernel but wrong for keccak512.
// ============================================================================

// ALL existing keccak .hip.inc files (keccak-tiny, keccak-amd, keccak-nvidia)
// have last-round pruning: they only compute state[0..4] in round 24.
// This is correct for KawPow's main kernel (reads state[0..3]) but WRONG
// for keccak512 which needs state[0..7] (64 bytes).
//
// This file provides its own full 24-round keccak-f1600 for DAG generation.

#include "bitselect.hip.h"
#ifdef __HIP_PLATFORM_AMD__
#include "keccak-amd.hip.inc"
#else
#include "keccak-nvidia.hip.inc"
#endif

#define ETHASH_FNV_PRIME         0x01000193u
#define ETHASH_NUM_PARENTS       512
#define ETHASH_HASH512_WORDS     16   // 512 bits / 32 bits

// ============================================================================
// Device-side keccak512 for 64-byte input
//
// Standard keccak-f[1600] with rate=72, capacity=128 (keccak-512).
// Input is exactly 64 bytes → single-block absorb with 0x01 padding.
// Output is first 64 bytes of state.
// ============================================================================

static __device__ __forceinline__ void ethash_keccak512_64(
    const uint32_t input[16], uint32_t output[16])
{
    uint64_t state[25];

    #pragma unroll
    for (int i = 0; i < 25; i++)
        state[i] = 0;

    // Absorb: 64 bytes = 8 uint64
    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = (uint64_t)input[i * 2] | ((uint64_t)input[i * 2 + 1] << 32);

    // Padding: delimiter 0x01 at byte 64, final 0x80 at byte 71
    state[8] ^= 0x8000000000000001ULL;

    keccakf(state);

    // Squeeze: first 64 bytes
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        output[i * 2]     = (uint32_t)(state[i] & 0xFFFFFFFF);
        output[i * 2 + 1] = (uint32_t)(state[i] >> 32);
    }
}

// ============================================================================
// FNV-1 on a 512-bit (16 × uint32) block: mix = (mix * FNV_PRIME) ^ parent
// ============================================================================

static __device__ __forceinline__ void fnv1_hash512(
    uint32_t mix[ETHASH_HASH512_WORDS],
    const uint32_t parent[ETHASH_HASH512_WORDS])
{
    #pragma unroll
    for (int i = 0; i < ETHASH_HASH512_WORDS; i++)
        mix[i] = (mix[i] * ETHASH_FNV_PRIME) ^ parent[i];
}

// ============================================================================
// Main DAG generation kernel
//
// Each thread generates one 2048-bit item (4 × 512-bit sub-items).
// Light cache is in global memory (read-only, fits in L2).
// Output DAG is written to global memory.
// ============================================================================

extern "C" __global__ void ethash_dag_gen_kernel(
    const uint32_t* __restrict__ light_cache,   // [num_cache_items × 16] uint32
    uint32_t*       __restrict__ dag,            // [dag_items × 64] uint32
    uint32_t                     num_cache_items,
    uint32_t                     dag_items)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dag_items) return;

    uint32_t* out = dag + (uint64_t)idx * 64;

    // Generate 4 sub-items
    for (int s = 0; s < 4; s++) {
        uint32_t item_index = idx * 4 + s;
        uint32_t seed = item_index;

        // Init: load cache entry
        uint32_t cache_idx = item_index % num_cache_items;
        const uint32_t* cache_entry = light_cache + (uint64_t)cache_idx * ETHASH_HASH512_WORDS;

        uint32_t mix[ETHASH_HASH512_WORDS];
        #pragma unroll
        for (int i = 0; i < ETHASH_HASH512_WORDS; i++)
            mix[i] = cache_entry[i];

        // XOR seed into first word
        mix[0] ^= seed;

        // Init keccak512
        uint32_t tmp[ETHASH_HASH512_WORDS];
        ethash_keccak512_64(mix, tmp);
        #pragma unroll
        for (int i = 0; i < ETHASH_HASH512_WORDS; i++)
            mix[i] = tmp[i];

        // 512 parent rounds
        for (uint32_t j = 0; j < ETHASH_NUM_PARENTS; j++) {
            uint32_t t = (seed ^ j) * ETHASH_FNV_PRIME ^ mix[j % ETHASH_HASH512_WORDS];
            uint32_t parent_idx = t % num_cache_items;
            const uint32_t* parent_entry = light_cache + (uint64_t)parent_idx * ETHASH_HASH512_WORDS;

            #pragma unroll
            for (int i = 0; i < ETHASH_HASH512_WORDS; i++)
                mix[i] = (mix[i] * ETHASH_FNV_PRIME) ^ parent_entry[i];
        }

        // Final keccak512 + write
        ethash_keccak512_64(mix, out + s * ETHASH_HASH512_WORDS);
    }
}

#endif // ETHASH_DAG_GEN_HIP
