#pragma once

// ============================================================================
// noise_generation_kernel.hpp
// Pearl noise generation kernel for TNN-IRIS-HIP.
//
// Generates dense (int8, range [-32, 32)) and sparse (1/-1 at random columns)
// noise matrices from BLAKE3 keyed hashing.
//
// Uses iris/hip/ framework and blake3-inline.hip.inc.
// hiprtc-friendly: all includes are inline headers.
// ============================================================================

#include "../../iris/rt/iris.hpp"

// BLAKE3 inline implementation (already has blake3_compress_in_place,
// rotation intrinsics, G-function, etc.)
#include "../../crypto/blake3-inline.hip.inc"

// ============================================================================
// BLAKE3 keyed hash helpers
// ============================================================================

// KEYED_HASH flag needed by noise generation
#ifndef KEYED_HASH
#define KEYED_HASH (1 << 4)
#endif

// Keyed single-block BLAKE3 compress.
// Inputs:
//   key[8]    - 32-byte key as uint32_t[8], used as chaining value
//   seed[8]   - 32-byte seed as uint32_t[8], placed in last 32 bytes of message
//   output[8] - 32-byte output as uint32_t[8]
// Message block layout:
//   [0..31] = 0 (zeros)
//   [32..63] = seed
// Counter = 0, block_len = 64, flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT
__device__ __forceinline__ void
pearl_blake3_keyed_single_block(const uint32_t key[8],
                                 const uint32_t seed[8],
                                 uint32_t output[8]) {
    // Setup chaining value = key
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key[i];

    // Build message block: zeros[0..31] + seed[32..63]
    uint8_t block[64];
    #pragma unroll
    for (int i = 0; i < 32; ++i) block[i] = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        block[32 + i] = static_cast<uint8_t>((seed[i / 4] >> (8 * (i % 4))) & 0xff);
    }

    uint8_t flags = static_cast<uint8_t>(KEYED_HASH | CHUNK_START | CHUNK_END | ROOT);

    blake3_compress_in_place(cv, block, 64, 0, flags);

    #pragma unroll
    for (int i = 0; i < 8; ++i) output[i] = cv[i];
}

// Alternative: keyed single-block compress taking byte pointers for key and seed.
// Same internal logic but avoids uint32 pointer casts.
__device__ __forceinline__ void
pearl_blake3_keyed_single_block_byte(const uint8_t* key_bytes,
                                      const uint8_t* seed_bytes,
                                      uint32_t output[8]) {
    uint32_t key[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        key[i] = blake3_load32(key_bytes + i * 4);
    }
    uint32_t seed[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        seed[i] = blake3_load32(seed_bytes + i * 4);
    }
    pearl_blake3_keyed_single_block(key, seed, output);
}

// ============================================================================
// Noise generation constants (matching pearl-gemm CUDA)
// ============================================================================
namespace pearl_noise {

static constexpr int NOISE_ABS_MAX = 128;
static constexpr int PERM_IDXS_PER_COL = 2;
// range = 128 / 2 = 64, so noise in [-32, 32)
static constexpr int NOISE_RANGE = NOISE_ABS_MAX / PERM_IDXS_PER_COL;

// Each thread produces 32 bytes from BLAKE3 = 32 int8 values
static constexpr int VALS_PER_THREAD = 32;

// Sparse: each uint32_t of the 8-word hash produces one pair (r0, r1)
static constexpr int KVALS_PER_THREAD = 8;

} // namespace pearl_noise

// ============================================================================
// Dense noise generation: BLAKE3 hash → clamp → vectorized store
// ============================================================================
//
// Each thread:
//   1. Generates BLAKE3 keyed hash → raw_hash[8] (uint32_t)
//   2. Interprets raw_hash as 32 int8 values
//   3. Clamps each to [-32, 32): ((val + 128) % 64) - 32
//   4. Stores 32 int8 values at its tile position in global memory
//
// Thread tile assignment (R=128, ValsPerThread=32, ThreadsPerRow=4):
//   Thread tid in a block of NumThreads:
//     row_in_tile = tid / ThreadsPerRow
//     col_start   = (tid % ThreadsPerRow) * ValsPerThread
//   Writes to gmem[(block_base_row + row_in_tile) * R + col_start + 0..31]
//
// No shared memory needed: each thread's 32 values are contiguous and adjacent
// threads' writes fall in the same cache line (coalesced).
// ============================================================================
template <int R, int NumThreads>
__global__ void noise_generation_dense_kernel(
    int8_t* __restrict__ ptr_out,
    int num_rows,
    const uint8_t* __restrict__ ptr_key,
    const uint8_t* __restrict__ ptr_seed_bytes,
    uint32_t thread_coord_base)
{
    constexpr int ThreadsPerRow = R / pearl_noise::VALS_PER_THREAD;
    constexpr int BlockSize = NumThreads / ThreadsPerRow;
    using DenseMap = iris::hip::RowMajorCoalesced<BlockSize, R, NumThreads, pearl_noise::VALS_PER_THREAD>;
    static_assert(R % pearl_noise::VALS_PER_THREAD == 0, "R must be multiple of ValsPerThread");
    static_assert(NumThreads % ThreadsPerRow == 0, "NumThreads must be multiple of ThreadsPerRow");

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // 1. BLAKE3 keyed hash
    uint32_t raw_hash[8];
    uint32_t thread_coord = thread_coord_base + bid * NumThreads + tid + 1;

    // Dense tensors use data[0] = thread_coord in the first message dword.
    uint32_t key[8];
    uint32_t seed[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        key[i] = blake3_load32(ptr_key + i * 4);
        seed[i] = blake3_load32(ptr_seed_bytes + i * 4);
    }

    // Build message: zeros[0..31] + seed[32..63], with data[0] = thread_coord
    uint8_t block[64];
    #pragma unroll
    for (int i = 0; i < 32; ++i) block[i] = 0;
    // data[0] = thread_coord (for dense tensors)
    block[0] = static_cast<uint8_t>(thread_coord & 0xff);
    block[1] = static_cast<uint8_t>((thread_coord >> 8) & 0xff);
    block[2] = static_cast<uint8_t>((thread_coord >> 16) & 0xff);
    block[3] = static_cast<uint8_t>((thread_coord >> 24) & 0xff);
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        block[32 + i] = static_cast<uint8_t>((seed[i / 4] >> (8 * (i % 4))) & 0xff);
    }

    // We can't use the simple wrapper because we need to modify data[0].
    // Use blake3_compress_in_place directly.
    {
        uint32_t cv[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = key[i];

        uint8_t flags = static_cast<uint8_t>(KEYED_HASH | CHUNK_START | CHUNK_END | ROOT);
        blake3_compress_in_place(cv, block, 64, 0, flags);

        #pragma unroll
        for (int i = 0; i < 8; ++i) raw_hash[i] = cv[i];
    }

    // 2. Interpret as int8[32], clamp to [-32, 32)
    int8_t vals[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int32_t raw = static_cast<int32_t>(reinterpret_cast<int8_t*>(raw_hash)[i]);
        vals[i] = static_cast<int8_t>(
            ((raw + pearl_noise::NOISE_ABS_MAX) % pearl_noise::NOISE_RANGE) -
            (pearl_noise::NOISE_RANGE / 2));
    }

    // 3. Vectorized store to global memory
    auto out_view = iris::hip::gmem_view(ptr_out, num_rows, R, R);
    iris::hip::store_row_major_vector<DenseMap>(out_view, bid * DenseMap::rows_per_block, 0, tid, vals);
}

// ============================================================================
// Sparse noise generation: BLAKE3 hash → column index pairs → scatter 1/-1
// ============================================================================
//
// For each of 8 uint32_t hash values, produce a pair (r0, r1):
//   r0 = u & (R - 1)                          // lowest log2(R) bits
//   x  = 1 + __umulhi((R - 1), u)             // nonzero
//   r1 = r0 ^ x                               // always distinct from r0
//
// Write 1 at column r0, -1 at column r1 for each K value.
// Supports both R-major and K-major sparse matrix layouts.
// ============================================================================
template <int R, int NumThreads>
__global__ void noise_generation_sparse_kernel(
    int8_t* __restrict__ ptr_out,
    int length,
    const uint8_t* __restrict__ ptr_key,
    const uint8_t* __restrict__ ptr_seed_bytes,
    bool r_major,
    uint32_t thread_coord_base)
{
    constexpr int KBlockSize = pearl_noise::KVALS_PER_THREAD * NumThreads;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int k_global_start = bid * KBlockSize;
    int k_local = tid;  // each thread handles KVALS_PER_THREAD consecutive K values

    // 1. BLAKE3 keyed hash (sparse: data[1] = thread_coord, not data[0])
    uint32_t raw_hash[8];
    uint32_t thread_coord = thread_coord_base + bid * NumThreads + tid + 1;

    uint32_t key[8];
    uint32_t seed[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        key[i] = blake3_load32(ptr_key + i * 4);
        seed[i] = blake3_load32(ptr_seed_bytes + i * 4);
    }

    uint8_t block[64];
    #pragma unroll
    for (int i = 0; i < 32; ++i) block[i] = 0;
    // data[1] = thread_coord (for sparse tensors)
    block[4] = static_cast<uint8_t>(thread_coord & 0xff);
    block[5] = static_cast<uint8_t>((thread_coord >> 8) & 0xff);
    block[6] = static_cast<uint8_t>((thread_coord >> 16) & 0xff);
    block[7] = static_cast<uint8_t>((thread_coord >> 24) & 0xff);
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        block[32 + i] = static_cast<uint8_t>((seed[i / 4] >> (8 * (i % 4))) & 0xff);
    }

    {
        uint32_t cv[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = key[i];
        uint8_t flags = static_cast<uint8_t>(KEYED_HASH | CHUNK_START | CHUNK_END | ROOT);
        blake3_compress_in_place(cv, block, 64, 0, flags);
        #pragma unroll
        for (int i = 0; i < 8; ++i) raw_hash[i] = cv[i];
    }

    // 2. Compute column index pairs and scatter write
    if (r_major) {
        // R-major layout: ptr[K][R], stride = R
        for (int k = 0; k < pearl_noise::KVALS_PER_THREAD; ++k) {
            int k_idx = k_local + k * NumThreads;
            if (k_idx >= length) break;

            uint32_t u = raw_hash[k];
            int r0 = static_cast<int>(u & (R - 1));
            int r1 = static_cast<int>(r0 ^ (1 + __umulhi(static_cast<uint32_t>(R - 1), u)));

            int8_t* row_ptr = ptr_out + k_idx * R;
            if (r0 < R) row_ptr[r0] = 1;
            if (r1 < R) row_ptr[r1] = -1;
        }
    } else {
        // K-major layout: ptr[R][K], stride = K
        for (int k = 0; k < pearl_noise::KVALS_PER_THREAD; ++k) {
            int k_idx = k_local + k * NumThreads;
            if (k_idx >= length) break;

            uint32_t u = raw_hash[k];
            int r0 = static_cast<int>(u & (R - 1));
            int r1 = static_cast<int>(r0 ^ (1 + __umulhi(static_cast<uint32_t>(R - 1), u)));

            if (r0 < R) ptr_out[r0 * length + k_idx] = 1;
            if (r1 < R) ptr_out[r1 * length + k_idx] = -1;
        }
    }
}

// ============================================================================
// Auxiliary buffer clear: zero int32/uint32 buffer
// ============================================================================
template <int NumThreads>
__global__ void noise_generation_clear_aux_kernel(
    uint32_t* __restrict__ ptr_aux,
    int aux_size)
{
    // Each thread clears 8 uint32_t values (32 bytes)
    constexpr int ValsPerThread = 8;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int base = (bid * NumThreads + tid) * ValsPerThread;

    #pragma unroll
    for (int i = 0; i < ValsPerThread; ++i) {
        int idx = base + i;
        if (idx < aux_size) {
            ptr_aux[idx] = 0;
        }
    }
}

// ============================================================================
// Combined noise generation kernel dispatch
// ============================================================================
//
// Matches the division of labor from pearl-gemm CUDA:
//   CTAs assigned in order: EAL, EAR_R_major, EAR_K_major,
//                            EBR, EBL_R_major, EBL_K_major,
//                            clear_aux
// ============================================================================
struct NoiseGenKernelArgs {
    int8_t*  ptr_EAL;
    int8_t*  ptr_EAR_R_major;
    int8_t*  ptr_EAR_K_major;
    int8_t*  ptr_EBR;
    int8_t*  ptr_EBL_R_major;
    int8_t*  ptr_EBL_K_major;
    int      num_rows_EAL;
    int      length_EAR;
    int      length_EBL;
    int      num_rows_EBR;
    const uint8_t* ptr_key_A;
    const uint8_t* ptr_key_B;
    uint32_t*      ptr_aux_buffer;
    int      aux_buffer_size;
};

__global__ void noise_generation_kernel(NoiseGenKernelArgs args) {
    // Each block handles its tile based on blockIdx.x.
    // We launch a flat 1D grid and partition blocks among the matrices.
    // For now, this is a placeholder — the actual dispatching logic
    // mirrors the CUDA version's bid_offset computation.
    //
    // Full implementation will be in noise_generation_launch.hpp
    // which selects the right kernel based on block index ranges.
}
