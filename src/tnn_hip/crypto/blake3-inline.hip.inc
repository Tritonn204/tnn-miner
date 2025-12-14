#pragma once

// blake3-inline.hip.inc - Optimized for AMD HIP and NVIDIA CUDA
// Platform-specific rotation optimizations

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    #include <hip/hip_runtime.h>
#endif

#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_MAX_DEPTH 54

// Split IV constants (enables direct constant embedding)
#define IV_0 0x6A09E667UL
#define IV_1 0xBB67AE85UL
#define IV_2 0x3C6EF372UL
#define IV_3 0xA54FF53AUL
#define IV_4 0x510E527FUL
#define IV_5 0x9B05688CUL
#define IV_6 0x1F83D9ABUL
#define IV_7 0x5BE0CD19UL

// Flags
#define CHUNK_START (1 << 0)
#define CHUNK_END   (1 << 1)
#define PARENT      (1 << 2)
#define ROOT        (1 << 3)

// ============================================================================
// Message schedule constants (compile-time lookup)
// ============================================================================
#define REF_Z00 0
#define REF_Z01 1
#define REF_Z02 2
#define REF_Z03 3
#define REF_Z04 4
#define REF_Z05 5
#define REF_Z06 6
#define REF_Z07 7
#define REF_Z08 8
#define REF_Z09 9
#define REF_Z0A 10
#define REF_Z0B 11
#define REF_Z0C 12
#define REF_Z0D 13
#define REF_Z0E 14
#define REF_Z0F 15

#define REF_Z10 2
#define REF_Z11 6
#define REF_Z12 3
#define REF_Z13 10
#define REF_Z14 7
#define REF_Z15 0
#define REF_Z16 4
#define REF_Z17 13
#define REF_Z18 1
#define REF_Z19 11
#define REF_Z1A 12
#define REF_Z1B 5
#define REF_Z1C 9
#define REF_Z1D 14
#define REF_Z1E 15
#define REF_Z1F 8

#define REF_Z20 3
#define REF_Z21 4
#define REF_Z22 10
#define REF_Z23 12
#define REF_Z24 13
#define REF_Z25 2
#define REF_Z26 7
#define REF_Z27 14
#define REF_Z28 6
#define REF_Z29 5
#define REF_Z2A 9
#define REF_Z2B 0
#define REF_Z2C 11
#define REF_Z2D 15
#define REF_Z2E 8
#define REF_Z2F 1

#define REF_Z30 10
#define REF_Z31 7
#define REF_Z32 12
#define REF_Z33 9
#define REF_Z34 14
#define REF_Z35 3
#define REF_Z36 13
#define REF_Z37 15
#define REF_Z38 4
#define REF_Z39 0
#define REF_Z3A 11
#define REF_Z3B 2
#define REF_Z3C 5
#define REF_Z3D 8
#define REF_Z3E 1
#define REF_Z3F 6

#define REF_Z40 12
#define REF_Z41 13
#define REF_Z42 9
#define REF_Z43 11
#define REF_Z44 15
#define REF_Z45 10
#define REF_Z46 14
#define REF_Z47 8
#define REF_Z48 7
#define REF_Z49 2
#define REF_Z4A 5
#define REF_Z4B 3
#define REF_Z4C 0
#define REF_Z4D 1
#define REF_Z4E 6
#define REF_Z4F 4

#define REF_Z50 9
#define REF_Z51 14
#define REF_Z52 11
#define REF_Z53 5
#define REF_Z54 8
#define REF_Z55 12
#define REF_Z56 15
#define REF_Z57 1
#define REF_Z58 13
#define REF_Z59 3
#define REF_Z5A 0
#define REF_Z5B 10
#define REF_Z5C 2
#define REF_Z5D 6
#define REF_Z5E 4
#define REF_Z5F 7

#define REF_Z60 11
#define REF_Z61 15
#define REF_Z62 5
#define REF_Z63 0
#define REF_Z64 1
#define REF_Z65 9
#define REF_Z66 8
#define REF_Z67 6
#define REF_Z68 14
#define REF_Z69 10
#define REF_Z6A 2
#define REF_Z6B 12
#define REF_Z6C 3
#define REF_Z6D 4
#define REF_Z6E 7
#define REF_Z6F 13

#ifndef __HIPRTC__INTTYPES_DEFINED
#define __HIPRTC__INTTYPES_DEFINED

#ifdef __HIP_PLATFORM_AMD__
using uint8_t  = __hip_internal::uint8_t;
using uint16_t = __hip_internal::uint16_t;
#elif defined(__CUDACC_RTC__)
#include "stdint-jit.hip.inc"
#elif defined(__CUDACC__)
#include <stdint.h> 
#endif

#endif

// ============================================================================
// Rotation primitives - Platform-specific optimizations
// ============================================================================

// 16-bit rotation using byte permute (both platforms)
__device__ __forceinline__ uint32_t blake3_rotr16(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __byte_perm(x, x, 0x1032);
#else
    return (x >> 16) | (x << 16);
#endif
}

// 8-bit rotation using byte permute (both platforms)
__device__ __forceinline__ uint32_t blake3_rotr8(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __byte_perm(x, x, 0x0321);
#else
    return (x >> 8) | (x << 24);
#endif
}

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)

// NVIDIA: Use funnel shifts for 12-bit and 7-bit rotations
__device__ __forceinline__ uint32_t blake3_rotr12(uint32_t x) {
    uint32_t result;
    asm("shf.r.wrap.b32 %0, %1, %2, 12;" : "=r"(result) : "r"(x), "r"(x));
    return result;
}

__device__ __forceinline__ uint32_t blake3_rotr7(uint32_t x) {
    uint32_t result;
    asm("shf.r.wrap.b32 %0, %1, %2, 7;" : "=r"(result) : "r"(x), "r"(x));
    return result;
}

// Generic rotation for other values (shouldn't be needed in BLAKE3)
__device__ __forceinline__ uint32_t blake3_rotr32(uint32_t x, uint32_t n) {
    uint32_t result;
    asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(x), "r"(n));
    return result;
}

#else

// AMD: Let compiler optimize (it handles alignbit automatically)
__device__ __forceinline__ uint32_t blake3_rotr12(uint32_t x) {
    return (x >> 12) | (x << 20);
}

__device__ __forceinline__ uint32_t blake3_rotr7(uint32_t x) {
    return (x >> 7) | (x << 25);
}

__device__ __forceinline__ uint32_t blake3_rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

#endif

// ============================================================================
// Byte loading (explicit, avoids alignment issues)
// ============================================================================

__device__ __forceinline__ uint32_t blake3_load32(const uint8_t* src) {
    return (uint32_t)src[0] |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

__device__ __forceinline__ void blake3_store32(uint8_t* dst, uint32_t val) {
    dst[0] = (uint8_t)(val);
    dst[1] = (uint8_t)(val >> 8);
    dst[2] = (uint8_t)(val >> 16);
    dst[3] = (uint8_t)(val >> 24);
}

// ============================================================================
// G function macro - uses specialized rotations
// ============================================================================

#define BLAKE3_G(a, b, c, d, x, y)                      \
    do {                                                \
        state[a] = state[a] + state[b] + (x);           \
        state[d] = blake3_rotr16(state[d] ^ state[a]);  \
        state[c] = state[c] + state[d];                 \
        state[b] = blake3_rotr12(state[b] ^ state[c]);  \
        state[a] = state[a] + state[b] + (y);           \
        state[d] = blake3_rotr8(state[d] ^ state[a]);   \
        state[c] = state[c] + state[d];                 \
        state[b] = blake3_rotr7(state[b] ^ state[c]);   \
    } while (0)

// Message word lookup macro
#define BLAKE3_MSG(r, i) (block_words[REF_Z##r##i])

// Full round macro
#define BLAKE3_ROUND(r)                                             \
    do {                                                            \
        BLAKE3_G(0, 4, 8,  12, BLAKE3_MSG(r, 0), BLAKE3_MSG(r, 1)); \
        BLAKE3_G(1, 5, 9,  13, BLAKE3_MSG(r, 2), BLAKE3_MSG(r, 3)); \
        BLAKE3_G(2, 6, 10, 14, BLAKE3_MSG(r, 4), BLAKE3_MSG(r, 5)); \
        BLAKE3_G(3, 7, 11, 15, BLAKE3_MSG(r, 6), BLAKE3_MSG(r, 7)); \
        BLAKE3_G(0, 5, 10, 15, BLAKE3_MSG(r, 8), BLAKE3_MSG(r, 9)); \
        BLAKE3_G(1, 6, 11, 12, BLAKE3_MSG(r, A), BLAKE3_MSG(r, B)); \
        BLAKE3_G(2, 7, 8,  13, BLAKE3_MSG(r, C), BLAKE3_MSG(r, D)); \
        BLAKE3_G(3, 4, 9,  14, BLAKE3_MSG(r, E), BLAKE3_MSG(r, F)); \
    } while (0)

// ============================================================================
// CV state initialization
// ============================================================================

__device__ __forceinline__ void blake3_cv_init(uint32_t cv[8]) {
    cv[0] = IV_0;
    cv[1] = IV_1;
    cv[2] = IV_2;
    cv[3] = IV_3;
    cv[4] = IV_4;
    cv[5] = IV_5;
    cv[6] = IV_6;
    cv[7] = IV_7;
}

// ============================================================================
// Core compression function
// ============================================================================

__device__ __forceinline__ void blake3_compress_in_place(
    uint32_t cv[8],
    const uint8_t block[BLAKE3_BLOCK_LEN],
    uint8_t block_len,
    uint64_t counter,
    uint8_t flags)
{
    uint32_t state[16];
    uint32_t block_words[16];
    
    // Load block words
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        block_words[i] = blake3_load32(block + i * 4);
    }
    
    // Initialize state
    state[0] = cv[0];
    state[1] = cv[1];
    state[2] = cv[2];
    state[3] = cv[3];
    state[4] = cv[4];
    state[5] = cv[5];
    state[6] = cv[6];
    state[7] = cv[7];
    state[8] = IV_0;
    state[9] = IV_1;
    state[10] = IV_2;
    state[11] = IV_3;
    state[12] = (uint32_t)counter;
    state[13] = (uint32_t)(counter >> 32);
    state[14] = (uint32_t)block_len;
    state[15] = (uint32_t)flags;
    
    // 7 rounds using compile-time message schedule
    BLAKE3_ROUND(0);
    BLAKE3_ROUND(1);
    BLAKE3_ROUND(2);
    BLAKE3_ROUND(3);
    BLAKE3_ROUND(4);
    BLAKE3_ROUND(5);
    BLAKE3_ROUND(6);
    
    // Finalize
    cv[0] = state[0] ^ state[8];
    cv[1] = state[1] ^ state[9];
    cv[2] = state[2] ^ state[10];
    cv[3] = state[3] ^ state[11];
    cv[4] = state[4] ^ state[12];
    cv[5] = state[5] ^ state[13];
    cv[6] = state[6] ^ state[14];
    cv[7] = state[7] ^ state[15];
}

// ============================================================================
// Single-chunk hash (input_len <= 1024) - No streaming stores needed
// ============================================================================

__device__ __forceinline__ void blake3_hash_single_chunk(
    const uint8_t* input,
    size_t input_len,
    uint8_t output[BLAKE3_OUT_LEN])
{
    uint32_t cv[8];
    blake3_cv_init(cv);
    
    uint8_t block[BLAKE3_BLOCK_LEN];
    size_t offset = 0;
    
    while (offset < input_len || offset == 0) {
        size_t take = (input_len - offset >= BLAKE3_BLOCK_LEN) ? 
                      BLAKE3_BLOCK_LEN : (input_len - offset);
        
        // Copy and zero-pad
        #pragma unroll
        for (size_t i = 0; i < BLAKE3_BLOCK_LEN; i++) {
            block[i] = (i < take) ? input[offset + i] : 0;
        }
        
        uint8_t flags = 0;
        if (offset == 0) flags |= CHUNK_START;
        if (offset + BLAKE3_BLOCK_LEN >= input_len) flags |= CHUNK_END | ROOT;
        
        blake3_compress_in_place(cv, block, take, 0, flags);
        
        offset += BLAKE3_BLOCK_LEN;
        if (input_len == 0) break;
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        blake3_store32(output + i * 4, cv[i]);
    }
}

// ============================================================================
// Chunk CV computation (for multi-chunk)
// ============================================================================

__device__ __forceinline__ void blake3_chunk_cv(
    const uint8_t* input,
    size_t chunk_len,
    uint64_t chunk_counter,
    uint8_t cv_out[BLAKE3_OUT_LEN],
    bool is_root)
{
    uint32_t cv[8];
    blake3_cv_init(cv);
    
    uint8_t block[BLAKE3_BLOCK_LEN];
    size_t offset = 0;
    
    while (offset < chunk_len || offset == 0) {
        size_t take = (chunk_len - offset >= BLAKE3_BLOCK_LEN) ?
                      BLAKE3_BLOCK_LEN : (chunk_len - offset);
        
        #pragma unroll
        for (size_t i = 0; i < BLAKE3_BLOCK_LEN; i++) {
            block[i] = (i < take) ? input[offset + i] : 0;
        }
        
        uint8_t flags = 0;
        if (offset == 0) flags |= CHUNK_START;
        
        bool is_last_block = (offset + BLAKE3_BLOCK_LEN >= chunk_len) || (chunk_len == 0);
        if (is_last_block) {
            flags |= CHUNK_END;
            if (is_root) flags |= ROOT;
        }
        
        blake3_compress_in_place(cv, block, take, chunk_counter, flags);
        
        offset += BLAKE3_BLOCK_LEN;
        if (chunk_len == 0) break;
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        blake3_store32(cv_out + i * 4, cv[i]);
    }
}

// ============================================================================
// Parent node hash
// ============================================================================

__device__ __forceinline__ void blake3_hash_parent(
    const uint8_t left_cv[BLAKE3_OUT_LEN],
    const uint8_t right_cv[BLAKE3_OUT_LEN],
    uint8_t output[BLAKE3_OUT_LEN],
    bool is_root)
{
    uint32_t cv[8];
    blake3_cv_init(cv);
    
    uint8_t block[BLAKE3_BLOCK_LEN];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        block[i] = left_cv[i];
        block[i + 32] = right_cv[i];
    }
    
    uint8_t flags = PARENT;
    if (is_root) flags |= ROOT;
    
    blake3_compress_in_place(cv, block, BLAKE3_BLOCK_LEN, 0, flags);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        blake3_store32(output + i * 4, cv[i]);
    }
}

// ============================================================================
// Streaming store helpers for final output (platform-specific)
// ============================================================================

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)

__device__ __forceinline__ void blake3_store_output_streaming(uint8_t* output, uint32_t cv[8]) {
    // NVIDIA: Use cache-streaming stores for final output
    uint64_t* out64 = (uint64_t*)output;
    uint64_t v0 = ((uint64_t)cv[1] << 32) | cv[0];
    uint64_t v1 = ((uint64_t)cv[3] << 32) | cv[2];
    uint64_t v2 = ((uint64_t)cv[5] << 32) | cv[4];
    uint64_t v3 = ((uint64_t)cv[7] << 32) | cv[6];
    
    asm volatile("st.global.cs.v2.u64 [%0], {%1, %2};" 
                 :: "l"(out64), "l"(v0), "l"(v1) : "memory");
    asm volatile("st.global.cs.v2.u64 [%0], {%1, %2};" 
                 :: "l"(out64 + 2), "l"(v2), "l"(v3) : "memory");
}

#else

__device__ __forceinline__ void blake3_store_output_streaming(uint8_t* output, uint32_t cv[8]) {
    // AMD: Use non-temporal stores for final output
    uint64_t* out64 = (uint64_t*)output;
    uint64_t v0 = ((uint64_t)cv[1] << 32) | cv[0];
    uint64_t v1 = ((uint64_t)cv[3] << 32) | cv[2];
    uint64_t v2 = ((uint64_t)cv[5] << 32) | cv[4];
    uint64_t v3 = ((uint64_t)cv[7] << 32) | cv[6];
    
    __builtin_nontemporal_store(v0, out64);
    __builtin_nontemporal_store(v1, out64 + 1);
    __builtin_nontemporal_store(v2, out64 + 2);
    __builtin_nontemporal_store(v3, out64 + 3);
}

#endif

// Standard store (non-streaming) for small outputs
__device__ __forceinline__ void blake3_store_output(uint8_t* output, uint32_t cv[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        blake3_store32(output + i * 4, cv[i]);
    }
}

// ============================================================================
// Full multi-chunk hash with tree construction
// Uses streaming stores for final output only
// ============================================================================

// ============================================================================
// Prefetching support for NVIDIA GPUs
// ============================================================================

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)

// Prefetch macros for different cache levels
#define BLAKE3_PREFETCH_L1(addr) \
    asm volatile("prefetch.global.L1 [%0];" :: "l"((const char*)(addr)))

#define BLAKE3_PREFETCH_L2(addr) \
    asm volatile("prefetch.global.L2 [%0];" :: "l"((const char*)(addr)))

// Prefetch distance tuning (in blocks)
#define BLAKE3_PREFETCH_DISTANCE_L1 2
#define BLAKE3_PREFETCH_DISTANCE_L2 4

#else

// AMD: No explicit prefetch (hardware prefetcher is effective)
#define BLAKE3_PREFETCH_L1(addr) ((void)0)
#define BLAKE3_PREFETCH_L2(addr) ((void)0)

#endif

// ============================================================================
// Enhanced chunk CV computation with prefetching
// ============================================================================

__device__ __forceinline__ void blake3_chunk_cv_prefetch(
    const uint8_t* input,
    size_t chunk_len,
    uint64_t chunk_counter,
    uint8_t cv_out[BLAKE3_OUT_LEN],
    bool is_root)
{
    uint32_t cv[8];
    blake3_cv_init(cv);
    
    uint8_t block[BLAKE3_BLOCK_LEN];
    size_t offset = 0;
    
    // Initial prefetch for the first few blocks
    #if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
    if (chunk_len > 0) {
        BLAKE3_PREFETCH_L1(input);
        if (chunk_len > BLAKE3_BLOCK_LEN) {
            BLAKE3_PREFETCH_L1(input + BLAKE3_BLOCK_LEN);
        }
        if (chunk_len > 2 * BLAKE3_BLOCK_LEN) {
            BLAKE3_PREFETCH_L2(input + 2 * BLAKE3_BLOCK_LEN);
        }
    }
    #endif
    
    while (offset < chunk_len || offset == 0) {
        size_t take = (chunk_len - offset >= BLAKE3_BLOCK_LEN) ?
                      BLAKE3_BLOCK_LEN : (chunk_len - offset);
        
        // Prefetch future blocks while processing current one
        #if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
        size_t prefetch_l1_offset = offset + BLAKE3_PREFETCH_DISTANCE_L1 * BLAKE3_BLOCK_LEN;
        size_t prefetch_l2_offset = offset + BLAKE3_PREFETCH_DISTANCE_L2 * BLAKE3_BLOCK_LEN;
        
        if (prefetch_l1_offset < chunk_len) {
            BLAKE3_PREFETCH_L1(input + prefetch_l1_offset);
        }
        if (prefetch_l2_offset < chunk_len) {
            BLAKE3_PREFETCH_L2(input + prefetch_l2_offset);
        }
        #endif
        
        // Load current block with potential vectorization
        #pragma unroll
        for (size_t i = 0; i < BLAKE3_BLOCK_LEN; i++) {
            block[i] = (i < take) ? input[offset + i] : 0;
        }
        
        uint8_t flags = 0;
        if (offset == 0) flags |= CHUNK_START;
        
        bool is_last_block = (offset + BLAKE3_BLOCK_LEN >= chunk_len) || (chunk_len == 0);
        if (is_last_block) {
            flags |= CHUNK_END;
            if (is_root) flags |= ROOT;
        }
        
        blake3_compress_in_place(cv, block, take, chunk_counter, flags);
        
        offset += BLAKE3_BLOCK_LEN;
        if (chunk_len == 0) break;
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        blake3_store32(cv_out + i * 4, cv[i]);
    }
}

// ============================================================================
// Vectorized block loading with prefetch for NVIDIA
// ============================================================================

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)

__device__ __forceinline__ void blake3_load_block_vectorized(
    uint8_t block[BLAKE3_BLOCK_LEN],
    const uint8_t* input,
    size_t take)
{
    // Try to use 128-bit loads when aligned
    if (((uintptr_t)input & 15) == 0 && take == BLAKE3_BLOCK_LEN) {
        // Aligned 128-bit loads
        uint4* block_vec = (uint4*)block;
        const uint4* input_vec = (const uint4*)input;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            block_vec[i] = input_vec[i];
        }
    } else if (((uintptr_t)input & 3) == 0 && take >= 4) {
        // Aligned 32-bit loads
        uint32_t* block_u32 = (uint32_t*)block;
        const uint32_t* input_u32 = (const uint32_t*)input;
        size_t words = take / 4;
        
        #pragma unroll
        for (size_t i = 0; i < 16; i++) {
            if (i < words) {
                block_u32[i] = input_u32[i];
            } else {
                block_u32[i] = 0;
            }
        }
        
        // Handle remaining bytes
        size_t remaining_start = words * 4;
        #pragma unroll
        for (size_t i = remaining_start; i < BLAKE3_BLOCK_LEN; i++) {
            block[i] = (i < take) ? input[i] : 0;
        }
    } else {
        // Fallback to byte loads
        #pragma unroll
        for (size_t i = 0; i < BLAKE3_BLOCK_LEN; i++) {
            block[i] = (i < take) ? input[i] : 0;
        }
    }
}

#endif

// ============================================================================
// Full multi-chunk hash with tree construction and prefetching
// ============================================================================

__device__ __forceinline__ void blake3_hasher_hash(
    const uint8_t* input,
    size_t input_len,
    uint8_t output[BLAKE3_OUT_LEN])
{
    // Single chunk fast path
    if (input_len <= BLAKE3_CHUNK_LEN) {
        blake3_hash_single_chunk(input, input_len, output);
        return;
    }
    
    // Multi-chunk: build Merkle tree with prefetching
    uint8_t cv_stack[BLAKE3_MAX_DEPTH][BLAKE3_OUT_LEN];
    int stack_len = 0;
    
    size_t num_chunks = (input_len + BLAKE3_CHUNK_LEN - 1) / BLAKE3_CHUNK_LEN;
    
    // Prefetch the first chunk
    #if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
    BLAKE3_PREFETCH_L2(input);
    if (input_len > BLAKE3_CHUNK_LEN) {
        BLAKE3_PREFETCH_L2(input + BLAKE3_CHUNK_LEN);
    }
    #endif
    
    for (uint64_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        size_t chunk_start = chunk_idx * BLAKE3_CHUNK_LEN;
        size_t chunk_len = (input_len - chunk_start >= BLAKE3_CHUNK_LEN) ?
                          BLAKE3_CHUNK_LEN : (input_len - chunk_start);
        
        // Prefetch next chunk while processing current
        #if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
        if (chunk_idx + 1 < num_chunks) {
            size_t next_chunk_start = (chunk_idx + 1) * BLAKE3_CHUNK_LEN;
            BLAKE3_PREFETCH_L2(input + next_chunk_start);
            
            // Aggressive L1 prefetch for next chunk's first blocks
            if (next_chunk_start + BLAKE3_BLOCK_LEN < input_len) {
                BLAKE3_PREFETCH_L1(input + next_chunk_start);
            }
        }
        #endif
        
        bool is_only_chunk = (num_chunks == 1);
        
        uint8_t chunk_cv[BLAKE3_OUT_LEN];
        blake3_chunk_cv_prefetch(input + chunk_start, chunk_len, chunk_idx, chunk_cv, is_only_chunk);
        
        // Push to stack
        #pragma unroll
        for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
            cv_stack[stack_len][i] = chunk_cv[i];
        }
        stack_len++;
        
        // Merge pairs at the same tree level
        uint64_t merge_count = chunk_idx + 1;
        while ((merge_count & 1) == 0 && stack_len >= 2) {
            stack_len--;
            uint8_t right_cv[BLAKE3_OUT_LEN];
            #pragma unroll
            for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
                right_cv[i] = cv_stack[stack_len][i];
            }
            
            stack_len--;
            uint8_t left_cv[BLAKE3_OUT_LEN];
            #pragma unroll
            for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
                left_cv[i] = cv_stack[stack_len][i];
            }
            
            bool is_root_node = (stack_len == 0) && (chunk_idx == num_chunks - 1);
            
            uint8_t parent_cv[BLAKE3_OUT_LEN];
            blake3_hash_parent(left_cv, right_cv, parent_cv, is_root_node);
            
            #pragma unroll
            for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
                cv_stack[stack_len][i] = parent_cv[i];
            }
            stack_len++;
            
            merge_count >>= 1;
        }
    }
    
    // Final merges (same as before)
    while (stack_len > 1) {
        stack_len--;
        uint8_t right_cv[BLAKE3_OUT_LEN];
        #pragma unroll
        for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
            right_cv[i] = cv_stack[stack_len][i];
        }
        
        stack_len--;
        uint8_t left_cv[BLAKE3_OUT_LEN];
        #pragma unroll
        for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
            left_cv[i] = cv_stack[stack_len][i];
        }
        
        bool is_root_node = (stack_len == 0);
        
        uint8_t parent_cv[BLAKE3_OUT_LEN];
        blake3_hash_parent(left_cv, right_cv, parent_cv, is_root_node);
        
        #pragma unroll
        for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
            cv_stack[stack_len][i] = parent_cv[i];
        }
        stack_len++;
    }
    
    // Final output - use streaming stores
    uint32_t final_cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        final_cv[i] = blake3_load32(cv_stack[0] + i * 4);
    }
    blake3_store_output_streaming(output, final_cv);
}

// ============================================================================
// Convenience aliases (API compatibility)
// ============================================================================

__device__ __forceinline__ void blake3_inline_hash(
    uint8_t* input, 
    size_t input_len, 
    uint8_t* output)
{
    blake3_hasher_hash(input, input_len, output);
}

__device__ __forceinline__ void blake3_inline_hash_single(
    uint8_t* input,
    size_t input_len,
    uint8_t* output)
{
    blake3_hash_single_chunk(input, input_len, output);
}