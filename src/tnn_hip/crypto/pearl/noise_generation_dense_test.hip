#include "hiprtc_types.hip.h"
#include "iris.hpp"
#include "blake3-inline.hip.inc"

#ifndef KEYED_HASH
#define KEYED_HASH (1 << 4)
#endif

namespace pearl_noise_test {

static constexpr int R = 32;
static constexpr int NumThreads = 128;
static constexpr int NoiseAbsMax = 128;
static constexpr int PermIdxsPerCol = 2;
static constexpr int NoiseRange = NoiseAbsMax / PermIdxsPerCol;
static constexpr int ValsPerThread = 32;
static constexpr int ThreadsPerRow = R / ValsPerThread;
static constexpr int BlockRows = NumThreads / ThreadsPerRow;

using DenseMap = iris::hip::RowMajorCoalesced<BlockRows, R, NumThreads, ValsPerThread>;
using DenseDistribution = iris::hip::TileDistribution2D<DenseMap>;

__device__ __forceinline__ void blake3_noise_hash(
    const uint8_t* key_bytes,
    const uint8_t* seed_bytes,
    uint32_t thread_coord,
    int coord_word,
    uint32_t raw_hash[8])
{
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        cv[i] = blake3_load32(key_bytes + i * 4);
    }

    uint8_t block[64];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        block[i] = 0;
    }

    const int coord_offset = coord_word * 4;
    block[coord_offset + 0] = static_cast<uint8_t>(thread_coord & 0xff);
    block[coord_offset + 1] = static_cast<uint8_t>((thread_coord >> 8) & 0xff);
    block[coord_offset + 2] = static_cast<uint8_t>((thread_coord >> 16) & 0xff);
    block[coord_offset + 3] = static_cast<uint8_t>((thread_coord >> 24) & 0xff);

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        block[32 + i] = seed_bytes[i];
    }

    const uint8_t flags = static_cast<uint8_t>(KEYED_HASH | CHUNK_START | CHUNK_END | ROOT);
    blake3_compress_in_place(cv, block, 64, 0, flags);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        raw_hash[i] = cv[i];
    }
}

__device__ __forceinline__ int8_t dense_noise_byte(const uint32_t raw_hash[8], int i) {
    const uint8_t raw_u8 = static_cast<uint8_t>((raw_hash[i / 4] >> (8 * (i % 4))) & 0xff);
    const int32_t raw = static_cast<int32_t>(static_cast<int8_t>(raw_u8));
    return static_cast<int8_t>(((raw + NoiseAbsMax) % NoiseRange) - (NoiseRange / 2));
}

__device__ __forceinline__ void sparse_noise_indices(uint32_t u, uint8_t& r0, uint8_t& r1) {
    r0 = static_cast<uint8_t>(u & (R - 1));
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(R - 1) * static_cast<uint64_t>(u)) >> 32);
    r1 = static_cast<uint8_t>(r0 ^ static_cast<uint8_t>(1 + hi));
}

} // namespace pearl_noise_test

extern "C" __global__ void pearl_noise_dense_32_128_kernel(
    int8_t* __restrict__ ptr_out,
    int num_rows,
    const uint8_t* __restrict__ ptr_key,
    const uint8_t* __restrict__ ptr_seed_bytes,
    uint32_t thread_coord_base)
{
    using namespace pearl_noise_test;

    const int bid = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const uint32_t thread_coord = thread_coord_base + static_cast<uint32_t>(bid * NumThreads + tid + 1);

    uint32_t raw_hash[8];
    blake3_noise_hash(ptr_key, ptr_seed_bytes, thread_coord, 0, raw_hash);

    auto out_view = iris::hip::make_global_row_major_tensor_view(ptr_out, num_rows, R, R);
    auto window = iris::hip::make_distributed_tile_window<DenseMap>(out_view, bid * DenseMap::rows_per_block, 0);
    iris::hip::StaticDistributedTensor<int8_t, DenseDistribution> reg{};

    #pragma unroll
    for (int i = 0; i < DenseDistribution::values_per_thread; ++i) {
        reg[i] = dense_noise_byte(raw_hash, i);
    }

    window.store_from(reg, tid);
}

extern "C" __global__ void pearl_noise_sparse_32_128_kernel(
    int8_t* __restrict__ ptr_out,
    int num_rows,
    const uint8_t* __restrict__ ptr_key,
    const uint8_t* __restrict__ ptr_seed_bytes,
    uint32_t thread_coord_base)
{
    using namespace pearl_noise_test;

    const int bid = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const uint32_t thread_coord = thread_coord_base + static_cast<uint32_t>(bid * NumThreads + tid + 1);

    uint32_t raw_hash[8];
    blake3_noise_hash(ptr_key, ptr_seed_bytes, thread_coord, 1, raw_hash);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (bid * NumThreads + tid) * 8 + i;
        if (row < num_rows) {
            uint8_t r0 = 0;
            uint8_t r1 = 0;
            sparse_noise_indices(raw_hash[i], r0, r1);
            ptr_out[row * R + static_cast<int>(r0)] = 1;
            ptr_out[row * R + static_cast<int>(r1)] = -1;
        }
    }
}
