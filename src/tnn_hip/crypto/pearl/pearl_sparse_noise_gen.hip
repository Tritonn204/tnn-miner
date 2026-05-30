// ---------------------------------------------------------------------------
// pearl_sparse_noise_gen.hip
// Phase 2: Sparse noise index-pair generation via keyed BLAKE3 on GPU.
//
// Grid:  (ceil_div(k, 256), 1, 1)  → one thread per K element
// Block: (256, 1, 1)
//
// Parameters:
//   k              – number of sparse rows (= common_dim = 1024)
//   R              – rank (= 32)
//   key            – 32B BLAKE3 key
//   seed           – 32B seed/label
//   first_idx      – output: k int32_t values (column index r0)
//   second_idx     – output: k int32_t values (column index r1)
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef KEYED_HASH
#define KEYED_HASH (1 << 4)
#endif

#ifndef CHUNK_START
#define CHUNK_START (1 << 0)
#endif

#ifndef CHUNK_END
#define CHUNK_END   (1 << 1)
#endif

#ifndef ROOT
#define ROOT        (1 << 3)
#endif

// ---- Embedded BLAKE3 primitives (self-contained, no #include of .hip.inc) ----

// IV constants
#define SPN_IV_0 0x6A09E667UL
#define SPN_IV_1 0xBB67AE85UL
#define SPN_IV_2 0x3C6EF372UL
#define SPN_IV_3 0xA54FF53AUL
#define SPN_IV_4 0x510E527FUL
#define SPN_IV_5 0x9B05688CUL
#define SPN_IV_6 0x1F83D9ABUL
#define SPN_IV_7 0x5BE0CD19UL

__device__ __forceinline__ uint32_t spn_rotr16(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __byte_perm(x, x, 0x1032);
#else
    return (x >> 16) | (x << 16);
#endif
}

__device__ __forceinline__ uint32_t spn_rotr8(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __byte_perm(x, x, 0x0321);
#else
    return (x >> 8) | (x << 24);
#endif
}

__device__ __forceinline__ uint32_t spn_rotr12(uint32_t x) {
    return (x >> 12) | (x << 20);
}

__device__ __forceinline__ uint32_t spn_rotr7(uint32_t x) {
    return (x >> 7) | (x << 25);
}

__device__ __forceinline__ uint32_t spn_load32(const uint8_t* src) {
    return (uint32_t)src[0] |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

// Message schedule constants
#define SPN_Z00 0
#define SPN_Z01 1
#define SPN_Z02 2
#define SPN_Z03 3
#define SPN_Z04 4
#define SPN_Z05 5
#define SPN_Z06 6
#define SPN_Z07 7
#define SPN_Z08 8
#define SPN_Z09 9
#define SPN_Z0A 10
#define SPN_Z0B 11
#define SPN_Z0C 12
#define SPN_Z0D 13
#define SPN_Z0E 14
#define SPN_Z0F 15

#define SPN_Z10 2
#define SPN_Z11 6
#define SPN_Z12 3
#define SPN_Z13 10
#define SPN_Z14 7
#define SPN_Z15 0
#define SPN_Z16 4
#define SPN_Z17 13
#define SPN_Z18 1
#define SPN_Z19 11
#define SPN_Z1A 12
#define SPN_Z1B 5
#define SPN_Z1C 9
#define SPN_Z1D 14
#define SPN_Z1E 15
#define SPN_Z1F 8

#define SPN_Z20 3
#define SPN_Z21 4
#define SPN_Z22 10
#define SPN_Z23 12
#define SPN_Z24 13
#define SPN_Z25 2
#define SPN_Z26 7
#define SPN_Z27 14
#define SPN_Z28 6
#define SPN_Z29 5
#define SPN_Z2A 9
#define SPN_Z2B 0
#define SPN_Z2C 11
#define SPN_Z2D 15
#define SPN_Z2E 8
#define SPN_Z2F 1

#define SPN_Z30 10
#define SPN_Z31 7
#define SPN_Z32 12
#define SPN_Z33 9
#define SPN_Z34 14
#define SPN_Z35 3
#define SPN_Z36 13
#define SPN_Z37 15
#define SPN_Z38 4
#define SPN_Z39 0
#define SPN_Z3A 11
#define SPN_Z3B 2
#define SPN_Z3C 5
#define SPN_Z3D 8
#define SPN_Z3E 1
#define SPN_Z3F 6

#define SPN_Z40 12
#define SPN_Z41 13
#define SPN_Z42 9
#define SPN_Z43 11
#define SPN_Z44 15
#define SPN_Z45 10
#define SPN_Z46 14
#define SPN_Z47 8
#define SPN_Z48 7
#define SPN_Z49 2
#define SPN_Z4A 5
#define SPN_Z4B 3
#define SPN_Z4C 0
#define SPN_Z4D 1
#define SPN_Z4E 6
#define SPN_Z4F 4

#define SPN_Z50 9
#define SPN_Z51 14
#define SPN_Z52 11
#define SPN_Z53 5
#define SPN_Z54 8
#define SPN_Z55 12
#define SPN_Z56 15
#define SPN_Z57 1
#define SPN_Z58 13
#define SPN_Z59 3
#define SPN_Z5A 0
#define SPN_Z5B 10
#define SPN_Z5C 2
#define SPN_Z5D 6
#define SPN_Z5E 4
#define SPN_Z5F 7

#define SPN_Z60 11
#define SPN_Z61 15
#define SPN_Z62 5
#define SPN_Z63 0
#define SPN_Z64 1
#define SPN_Z65 9
#define SPN_Z66 8
#define SPN_Z67 6
#define SPN_Z68 14
#define SPN_Z69 10
#define SPN_Z6A 2
#define SPN_Z6B 12
#define SPN_Z6C 3
#define SPN_Z6D 4
#define SPN_Z6E 7
#define SPN_Z6F 13

__device__ __forceinline__ void spn_blake3_compress_in_place(
    uint32_t cv[8],
    const uint8_t block[64],
    uint8_t block_len,
    uint64_t counter,
    uint8_t flags)
{
    uint32_t state[16];
    uint32_t block_words[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        block_words[i] = spn_load32(block + i * 4);
    }

    state[0]  = cv[0];
    state[1]  = cv[1];
    state[2]  = cv[2];
    state[3]  = cv[3];
    state[4]  = cv[4];
    state[5]  = cv[5];
    state[6]  = cv[6];
    state[7]  = cv[7];
    state[8]  = SPN_IV_0;
    state[9]  = SPN_IV_1;
    state[10] = SPN_IV_2;
    state[11] = SPN_IV_3;
    state[12] = (uint32_t)counter;
    state[13] = (uint32_t)(counter >> 32);
    state[14] = (uint32_t)block_len;
    state[15] = (uint32_t)flags;

    uint32_t x, y;
    #define SPN_MSG(r,i) block_words[SPN_Z##r##i]
    #define SPN_G(a,b,c,d,mx,my)                                \
        do {                                                     \
            state[a] = state[a] + state[b] + (mx);              \
            state[d] = spn_rotr16(state[d] ^ state[a]);         \
            state[c] = state[c] + state[d];                     \
            state[b] = spn_rotr12(state[b] ^ state[c]);         \
            state[a] = state[a] + state[b] + (my);              \
            state[d] = spn_rotr8(state[d] ^ state[a]);          \
            state[c] = state[c] + state[d];                     \
            state[b] = spn_rotr7(state[b] ^ state[c]);          \
        } while (0)

    #define SPN_ROUND(r)                                        \
        do {                                                    \
            SPN_G(0,4,8,12, SPN_MSG(r,0), SPN_MSG(r,1));        \
            SPN_G(1,5,9,13, SPN_MSG(r,2), SPN_MSG(r,3));        \
            SPN_G(2,6,10,14, SPN_MSG(r,4), SPN_MSG(r,5));       \
            SPN_G(3,7,11,15, SPN_MSG(r,6), SPN_MSG(r,7));       \
            SPN_G(0,5,10,15, SPN_MSG(r,8), SPN_MSG(r,9));       \
            SPN_G(1,6,11,12, SPN_MSG(r,A), SPN_MSG(r,B));       \
            SPN_G(2,7,8,13, SPN_MSG(r,C), SPN_MSG(r,D));        \
            SPN_G(3,4,9,14, SPN_MSG(r,E), SPN_MSG(r,F));        \
        } while (0)

    SPN_ROUND(0);
    SPN_ROUND(1);
    SPN_ROUND(2);
    SPN_ROUND(3);
    SPN_ROUND(4);
    SPN_ROUND(5);
    SPN_ROUND(6);

    #undef SPN_G
    #undef SPN_ROUND
    #undef SPN_MSG

    cv[0] = state[0] ^ state[8];
    cv[1] = state[1] ^ state[9];
    cv[2] = state[2] ^ state[10];
    cv[3] = state[3] ^ state[11];
    cv[4] = state[4] ^ state[12];
    cv[5] = state[5] ^ state[13];
    cv[6] = state[6] ^ state[14];
    cv[7] = state[7] ^ state[15];
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void pearl_sparse_noise_gen(
    uint32_t      k,
    uint32_t      R,
    uint8_t const* key,
    uint8_t const* seed,
    int32_t*       first_idx,
    int32_t*       second_idx)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= k) return;

    const uint32_t thread_coord = (tid / 8) + 1;

    uint32_t key_u32[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        key_u32[i] = spn_load32(key + i * 4);
    }

    uint32_t seed_u32[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        seed_u32[i] = spn_load32(seed + i * 4);
    }

    uint8_t block[64];
    #pragma unroll
    for (int i = 0; i < 32; ++i) block[i] = 0;

    block[4] = (uint8_t)(thread_coord & 0xff);
    block[5] = (uint8_t)((thread_coord >> 8) & 0xff);
    block[6] = (uint8_t)((thread_coord >> 16) & 0xff);
    block[7] = (uint8_t)((thread_coord >> 24) & 0xff);

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        block[32 + i] = (uint8_t)((seed_u32[i / 4] >> (8 * (i % 4))) & 0xff);
    }

    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key_u32[i];

    const uint8_t flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
    spn_blake3_compress_in_place(cv, block, 64, 0, flags);

    const uint32_t u = cv[tid % 8];
    const uint32_t r0 = u & (R - 1);
    const uint32_t hi = (uint32_t)(((uint64_t)(R - 1) * (uint64_t)u) >> 32);
    const uint32_t r1 = r0 ^ (1u + hi);

    first_idx[tid]  = (int32_t)r0;
    second_idx[tid] = (int32_t)r1;
}
