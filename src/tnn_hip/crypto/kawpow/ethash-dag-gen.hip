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

#define ETHASH_FNV_PRIME         0x01000193u
#define ETHASH_NUM_PARENTS       512
#define ETHASH_HASH512_WORDS     16   // 512 bits / 32 bits

// ============================================================================
// Full 24-round keccak-f[1600] — no last-round pruning
// ============================================================================

__device__ static const uint64_t DAG_RC[24] = {
    1ULL, 0x8082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x808bULL, 0x80000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x8aULL, 0x88ULL, 0x80008009ULL, 0x8000000aULL,
    0x8000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x80000001ULL, 0x8000000080008008ULL};

__device__ static constexpr unsigned char DAG_RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};

__device__ static constexpr unsigned char DAG_PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

static __device__ __forceinline__ uint64_t dag_rol64(uint64_t v, uint32_t r)
{
#ifdef __HIP_PLATFORM_AMD__
    uint32_t lo = (uint32_t)(v & 0xFFFFFFFF);
    uint32_t hi = (uint32_t)(v >> 32);
    uint32_t rlo, rhi;
    if (r <= 32) {
        rlo = __builtin_amdgcn_alignbit(lo, hi, 32 - r);
        rhi = __builtin_amdgcn_alignbit(hi, lo, 32 - r);
    } else {
        rlo = __builtin_amdgcn_alignbit(hi, lo, 64 - r);
        rhi = __builtin_amdgcn_alignbit(lo, hi, 64 - r);
    }
    return ((uint64_t)rhi << 32) | rlo;
#else
    return (v << r) | (v >> (64 - r));
#endif
}

static __device__ __forceinline__ void dag_keccakf(uint64_t* a)
{
    uint64_t b[5], t, v;

    #pragma unroll
    for (int i = 0; i < 24; i++)
    {
        // Theta
        b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
        b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
        b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
        b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
        b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];

        v = b[4]; t = b[0];
        b[4] = b[4] ^ dag_rol64(b[1], 1);
        b[0] = b[0] ^ dag_rol64(b[2], 1);
        b[1] = b[1] ^ dag_rol64(b[3], 1);
        b[2] = b[2] ^ dag_rol64(v, 1);
        b[3] = b[3] ^ dag_rol64(t, 1);

        a[0]  ^= b[4]; a[5]  ^= b[4]; a[10] ^= b[4]; a[15] ^= b[4]; a[20] ^= b[4];
        a[1]  ^= b[0]; a[6]  ^= b[0]; a[11] ^= b[0]; a[16] ^= b[0]; a[21] ^= b[0];
        a[2]  ^= b[1]; a[7]  ^= b[1]; a[12] ^= b[1]; a[17] ^= b[1]; a[22] ^= b[1];
        a[3]  ^= b[2]; a[8]  ^= b[2]; a[13] ^= b[2]; a[18] ^= b[2]; a[23] ^= b[2];
        a[4]  ^= b[3]; a[9]  ^= b[3]; a[14] ^= b[3]; a[19] ^= b[3]; a[24] ^= b[3];

        // Rho and pi
        t = a[1];
        a[DAG_PI[23]] = dag_rol64(a[DAG_PI[22]], DAG_RHO[23]);
        a[DAG_PI[22]] = dag_rol64(a[DAG_PI[21]], DAG_RHO[22]);
        a[DAG_PI[21]] = dag_rol64(a[DAG_PI[20]], DAG_RHO[21]);
        a[DAG_PI[20]] = dag_rol64(a[DAG_PI[19]], DAG_RHO[20]);
        a[DAG_PI[19]] = dag_rol64(a[DAG_PI[18]], DAG_RHO[19]);
        a[DAG_PI[18]] = dag_rol64(a[DAG_PI[17]], DAG_RHO[18]);
        a[DAG_PI[17]] = dag_rol64(a[DAG_PI[16]], DAG_RHO[17]);
        a[DAG_PI[16]] = dag_rol64(a[DAG_PI[15]], DAG_RHO[16]);
        a[DAG_PI[15]] = dag_rol64(a[DAG_PI[14]], DAG_RHO[15]);
        a[DAG_PI[14]] = dag_rol64(a[DAG_PI[13]], DAG_RHO[14]);
        a[DAG_PI[13]] = dag_rol64(a[DAG_PI[12]], DAG_RHO[13]);
        a[DAG_PI[12]] = dag_rol64(a[DAG_PI[11]], DAG_RHO[12]);
        a[DAG_PI[11]] = dag_rol64(a[DAG_PI[10]], DAG_RHO[11]);
        a[DAG_PI[10]] = dag_rol64(a[DAG_PI[ 9]], DAG_RHO[10]);
        a[DAG_PI[ 9]] = dag_rol64(a[DAG_PI[ 8]], DAG_RHO[ 9]);
        a[DAG_PI[ 8]] = dag_rol64(a[DAG_PI[ 7]], DAG_RHO[ 8]);
        a[DAG_PI[ 7]] = dag_rol64(a[DAG_PI[ 6]], DAG_RHO[ 7]);
        a[DAG_PI[ 6]] = dag_rol64(a[DAG_PI[ 5]], DAG_RHO[ 6]);
        a[DAG_PI[ 5]] = dag_rol64(a[DAG_PI[ 4]], DAG_RHO[ 5]);
        a[DAG_PI[ 4]] = dag_rol64(a[DAG_PI[ 3]], DAG_RHO[ 4]);
        a[DAG_PI[ 3]] = dag_rol64(a[DAG_PI[ 2]], DAG_RHO[ 3]);
        a[DAG_PI[ 2]] = dag_rol64(a[DAG_PI[ 1]], DAG_RHO[ 2]);
        a[DAG_PI[ 1]] = dag_rol64(a[DAG_PI[ 0]], DAG_RHO[ 1]);
        a[DAG_PI[ 0]] = dag_rol64(t, DAG_RHO[0]);

        // Chi
        v = a[0]; t = a[1];
        a[0] = bitselect(a[0] ^ a[2], a[0], a[1]);
        a[1] = bitselect(a[1] ^ a[3], a[1], a[2]);
        a[2] = bitselect(a[2] ^ a[4], a[2], a[3]);
        a[3] = bitselect(a[3] ^ v, a[3], a[4]);
        a[4] = bitselect(a[4] ^ t, a[4], v);

        v = a[5]; t = a[6];
        a[5] = bitselect(a[5] ^ a[7], a[5], a[6]);
        a[6] = bitselect(a[6] ^ a[8], a[6], a[7]);
        a[7] = bitselect(a[7] ^ a[9], a[7], a[8]);
        a[8] = bitselect(a[8] ^ v, a[8], a[9]);
        a[9] = bitselect(a[9] ^ t, a[9], v);

        v = a[10]; t = a[11];
        a[10] = bitselect(a[10] ^ a[12], a[10], a[11]);
        a[11] = bitselect(a[11] ^ a[13], a[11], a[12]);
        a[12] = bitselect(a[12] ^ a[14], a[12], a[13]);
        a[13] = bitselect(a[13] ^ v, a[13], a[14]);
        a[14] = bitselect(a[14] ^ t, a[14], v);

        v = a[15]; t = a[16];
        a[15] = bitselect(a[15] ^ a[17], a[15], a[16]);
        a[16] = bitselect(a[16] ^ a[18], a[16], a[17]);
        a[17] = bitselect(a[17] ^ a[19], a[17], a[18]);
        a[18] = bitselect(a[18] ^ v, a[18], a[19]);
        a[19] = bitselect(a[19] ^ t, a[19], v);

        v = a[20]; t = a[21];
        a[20] = bitselect(a[20] ^ a[22], a[20], a[21]);
        a[21] = bitselect(a[21] ^ a[23], a[21], a[22]);
        a[22] = bitselect(a[22] ^ a[24], a[22], a[23]);
        a[23] = bitselect(a[23] ^ v, a[23], a[24]);
        a[24] = bitselect(a[24] ^ t, a[24], v);

        // Iota
        a[0] ^= DAG_RC[i];
    }
}

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

    dag_keccakf(state);

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
