// ============================================================================
// pearl_expand_jackpot.hip
// GPU expansion + jackpot computation kernel for Pearl PoUW.
//
// Current validation/benchmark path:
//   Phase 1 (pearl_expand_jackpot_tiled_partial):
//     Grid = ceil(w / 32) x ceil(h / 16), Block = 256 threads.
//     Each block handles a 16x32 tile of (u, v) pairs. A/B rank tiles are
//     staged through Iris LDS views, each thread keeps two cumulative pair sums,
//     and the block writes one XOR partial per rank tile.
//
//   Phase 2 (pearl_jackpot_tiled_reduce):
//     16 blocks x 256 threads, one block per jackpot word.
//     Each block walks that word's tile sequence, XORs all tiled partial
//     results, and applies ROT13 into jackpot_msg[word].
//
// Kernels accept dynamic sizes (h, w, k) so they work for both test
// vectors (h=4, w=8) and production (h=1024, w=1024).
// ============================================================================

#include "hiprtc_types.hip.h"
#include "iris.hpp"
#include "blake3-inline.hip.inc"
#include "../iris/gemm/gemm_backend.hpp"

#ifndef KEYED_HASH
#define KEYED_HASH (1 << 4)
#endif

// ============================================================================
// Constants
// ============================================================================
static constexpr int kNumThreads = 256;
static constexpr int kRank = 32;
static constexpr int kJackpotSize = 16;
static constexpr unsigned kLrotPerTile = 13;
static constexpr int kJackpotTileRowsA = 16;
static constexpr int kJackpotTileRowsB = 32;
static constexpr int kJackpotWmmaTileRowsA = 64;
static constexpr int kJackpotWmmaTileRowsB = 32;
static constexpr int kJackpotWmma64TileRowsB = 64;
static constexpr int kJackpotWmma96TileRowsB = 96;
static constexpr int kJackpotWmma128TileRowsB = 128;
static constexpr int kJackpotWmma160TileRowsB = 160;
static constexpr int kJackpotWmma192TileRowsB = 192;
static constexpr int kJackpotWmma256TileRowsB = 256;
static constexpr int kJackpotWmma320TileRowsB = 320;
static constexpr int kJackpotWmma384TileRowsB = 384;
static constexpr int kJackpotWmma448TileRowsB = 448;
static constexpr int kJackpotWmma512TileRowsB = 512;
static constexpr int kNumThreads512 = 512;
static constexpr int kJackpotWmma128TileRowsA = 128;
static constexpr int kPearlBenchmarkK = 4096;

IRIS_DEVICE_INLINE int8_t pearl_i8_lane(uint32_t packed, int lane) {
    return static_cast<int8_t>((packed >> (lane * 8)) & 0xffu);
}

IRIS_DEVICE_INLINE int8_t pearl_i8_lane(uint64_t packed, int lane) {
    return static_cast<int8_t>((packed >> (lane * 8)) & 0xffull);
}

struct alignas(16) PearlVec128 {
    uint64_t lo;
    uint64_t hi;
};

IRIS_DEVICE_INLINE auto pearl_vec128_as_i8x16(const PearlVec128& packed) {
    union PackedVec128 {
        PearlVec128 packed;
        iris::hip::VectorAccess<int8_t, 16>::Vec vec;
    };
    PackedVec128 converted{};
    converted.packed = packed;
    return converted.vec;
}

IRIS_DEVICE_INLINE PearlVec128 pearl_load_vec128_or_zero(const signed char* ptr, bool in_bounds) {
    PearlVec128 packed{0, 0};
    if (in_bounds) {
        packed = *reinterpret_cast<const PearlVec128*>(ptr);
    }
    return packed;
}

IRIS_DEVICE_INLINE void pearl_store_vec128_row_major_or_scalar(
    signed char* base,
    int row_stride,
    int row,
    int col,
    const PearlVec128& packed)
{
    using Access = iris::hip::VectorAccess<int8_t, 16>;
    int8_t* dst = reinterpret_cast<int8_t*>(base + row * row_stride + col);
    if (Access::is_aligned(dst)) {
        Access::store_aligned(dst, pearl_vec128_as_i8x16(packed));
        return;
    }

    #pragma unroll
    for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
        dst[lane_byte] = pearl_i8_lane(packed.lo, lane_byte);
        dst[8 + lane_byte] = pearl_i8_lane(packed.hi, lane_byte);
    }
}

IRIS_DEVICE_INLINE void pearl_store_vec128_col_major_or_scalar(
    signed char* base,
    int col_stride,
    int col,
    int row,
    const PearlVec128& packed)
{
    using Access = iris::hip::VectorAccess<int8_t, 16>;
    int8_t* dst = reinterpret_cast<int8_t*>(base + col * col_stride + row);
    if (Access::is_aligned(dst)) {
        Access::store_aligned(dst, pearl_vec128_as_i8x16(packed));
        return;
    }

    #pragma unroll
    for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
        dst[lane_byte] = pearl_i8_lane(packed.lo, lane_byte);
        dst[8 + lane_byte] = pearl_i8_lane(packed.hi, lane_byte);
    }
}

IRIS_DEVICE_INLINE void pearl_store_vec128_col_major_halfpad16_or_scalar(
    signed char* base,
    int col_stride,
    int col,
    int row,
    const PearlVec128& packed)
{
    using Access = iris::hip::VectorAccess<int8_t, 16>;
    const int physical_row = row + ((row >= 16) ? 16 : 0);
    int8_t* dst = reinterpret_cast<int8_t*>(base + col * col_stride + physical_row);
    if (Access::is_aligned(dst)) {
        Access::store_aligned(dst, pearl_vec128_as_i8x16(packed));
        return;
    }

    #pragma unroll
    for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
        dst[lane_byte] = pearl_i8_lane(packed.lo, lane_byte);
        dst[8 + lane_byte] = pearl_i8_lane(packed.hi, lane_byte);
    }
}

IRIS_DEVICE_INLINE void pearl_store_vec128_col_major_stepmajor16_or_scalar(
    signed char* base,
    int tile_rows,
    int col,
    int row,
    const PearlVec128& packed)
{
    using Access = iris::hip::VectorAccess<int8_t, 16>;
    const int step = row / 16;
    const int local_row = row - step * 16;
    int8_t* dst = reinterpret_cast<int8_t*>(
        base + step * (tile_rows * 16) + col * 16 + local_row);
    if (Access::is_aligned(dst)) {
        Access::store_aligned(dst, pearl_vec128_as_i8x16(packed));
        return;
    }

    #pragma unroll
    for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
        dst[lane_byte] = pearl_i8_lane(packed.lo, lane_byte);
        dst[8 + lane_byte] = pearl_i8_lane(packed.hi, lane_byte);
    }
}

IRIS_DEVICE_INLINE void pearl_noise_hash(
    const uint8_t* __restrict__ key_bytes,
    const uint8_t* __restrict__ seed_bytes,
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
    block[coord_offset + 0] = static_cast<uint8_t>(thread_coord & 0xffu);
    block[coord_offset + 1] = static_cast<uint8_t>((thread_coord >> 8) & 0xffu);
    block[coord_offset + 2] = static_cast<uint8_t>((thread_coord >> 16) & 0xffu);
    block[coord_offset + 3] = static_cast<uint8_t>((thread_coord >> 24) & 0xffu);

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

IRIS_DEVICE_INLINE signed char pearl_dense_noise_byte(uint32_t raw_hash[8], int i) {
    const uint8_t raw_u8 = static_cast<uint8_t>((raw_hash[i / 4] >> (8 * (i % 4))) & 0xffu);
    const int raw = static_cast<int>(static_cast<int8_t>(raw_u8));
    return static_cast<signed char>(((raw + 128) % 64) - 32);
}

IRIS_DEVICE_INLINE uint16_t pearl_sparse_noise_pair(uint32_t u) {
    const uint8_t r0 = static_cast<uint8_t>(u & (kRank - 1));
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(kRank - 1) * static_cast<uint64_t>(u)) >> 32);
    const uint8_t r1 = static_cast<uint8_t>(r0 ^ static_cast<uint8_t>(1 + hi));
    return static_cast<uint16_t>(r0) | (static_cast<uint16_t>(r1) << 8);
}

template <int Threads, bool ExtraSync = false>
IRIS_DEVICE_INLINE uint32_t pearl_wave_publish_block_reduce_xor_u32(
    uint32_t thread_xored,
    unsigned int* __restrict__ sh_xor)
{
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int wave = tid >> 5;
    uint32_t wave_xor = thread_xored;
    for (int offset = 16; offset > 0; offset >>= 1) {
        wave_xor ^= iris::hip::shuffle_xor_u32(wave_xor, offset);
    }
    if constexpr (ExtraSync) {
        iris::hip::block_sync();
    }
    if (lane == 0) {
        sh_xor[wave] = wave_xor;
    }
    iris::hip::block_sync();
    if constexpr (ExtraSync) {
        iris::hip::block_sync();
    }
    uint32_t block_xor = 0;
    if (wave == 0) {
        block_xor = (lane < (Threads / 32)) ? sh_xor[lane] : 0u;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_xor ^= iris::hip::shuffle_xor_u32(block_xor, offset);
        }
    }
    if constexpr (ExtraSync) {
        iris::hip::block_sync();
    }
    iris::hip::block_sync();
    return block_xor;
}


IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_wmma256_core_v1_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k);

IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_wmma256_core_v2_pipeline_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k);

IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_wmma256_core_v1_a_swizzle_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k);

// Dedicated 64x64x32 reset body.
// This intentionally starts as a clean owned implementation of the currently
// validated 64x64 / 128tpb / ungrouped path instead of another generic family
// wrapper. The first goal is to give the 64x64 family its own control-flow
// home so future DU32 scheduling, stagger, and vector-width tuning can happen
// locally without mutating the generic Pearl shell.
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_du32_v1_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k);

IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq128x128_d16_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k);

IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq256x256_d16_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k);

template <bool BMajorSwizzle>
IRIS_DEVICE_INLINE void pearl_compute_virtual_block_coords(
    int& virtual_block_x,
    int& virtual_block_y)
{
    if constexpr (!BMajorSwizzle) {
        virtual_block_x = static_cast<int>(blockIdx.x);
        virtual_block_y = static_cast<int>(blockIdx.y);
        return;
    }

    const int grid_x = static_cast<int>(gridDim.x);
    const int grid_y = static_cast<int>(gridDim.y);
    const int linear = static_cast<int>(blockIdx.y) * grid_x + static_cast<int>(blockIdx.x);
    virtual_block_x = linear / grid_y;
    virtual_block_y = linear - virtual_block_x * grid_y;
}

template <int StripeY>
IRIS_DEVICE_INLINE void pearl_compute_virtual_block_coords_bmajor_striped(
    int& virtual_block_x,
    int& virtual_block_y)
{
    static_assert(StripeY > 0, "StripeY must be positive");
    const int grid_x = static_cast<int>(gridDim.x);
    const int linear = static_cast<int>(blockIdx.y) * grid_x + static_cast<int>(blockIdx.x);
    const int span = grid_x * StripeY;
    const int stripe_group = linear / span;
    const int rem = linear - stripe_group * span;
    virtual_block_x = rem / StripeY;
    virtual_block_y = stripe_group * StripeY + (rem - virtual_block_x * StripeY);
}

template <int ATileRows, int BTileRows, int NGroups, int Threads, int ChunkGroups, bool PreStoreSync, bool Ungrouped>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_wmma_wide_precombined_vec8_share2a_core_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k);

template <int ATileRows, int BTileRows, int NGroups, int Threads, int ChunkGroups = NGroups, bool PreStoreSync = false, int ReduceMode = 0>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_wmma_wide_precombined_vec8_reduce_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    if constexpr (ATileRows == kJackpotWmmaTileRowsA && Threads == kNumThreads) {
        pearl_expand_jackpot_wmma_wide_precombined_vec8_share2a_core_impl<
            ATileRows, BTileRows, NGroups, Threads, ChunkGroups, PreStoreSync, true>(
                a_combined, b_combined, partial, owner_first, owner_last, owner_collision, h, w, k);
        return;
    }

    static_assert(BTileRows == NGroups * kJackpotWmmaTileRowsB, "BTileRows must be NGroups * 32");
    static_assert(ChunkGroups > 0, "ChunkGroups must be positive");
    static_assert(NGroups % ChunkGroups == 0, "NGroups must be divisible by ChunkGroups");
    using Wmma = iris::hip::WmmaI8I8I32_16x16x16<true, true, false>;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[ATileRows * kRank];
    __shared__ signed char sh_bt_storage[kRank * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    const int tile_u_base = static_cast<int>(blockIdx.y) * ATileRows;
    const int tile_v_base = static_cast<int>(blockIdx.x) * BTileRows;
    const int partial_block = static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);
    const int wave_m = wave >> 1;
    const int wave_n = wave & 1;
    const int local_m_base = wave_m * Wmma::m;
    const int local_n_base0 = wave_n * Wmma::n;

    auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage);
    auto sh_bt_group = [&](int group) {
        return iris::hip::make_grouped_wmma_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, group);
    };
    int32_t running_sums[NGroups][Wmma::c_elems];
    #pragma unroll
    for (int group = 0; group < NGroups; ++group) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            running_sums[group][elem] = 0;
        }
    }

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx8 = tid; idx8 < (ATileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (u < h && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(a_combined + static_cast<size_t>(u) * k + k_col);
            }
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_a.store(row, col + lane_byte, pearl_i8_lane(packed, lane_byte));
            }
        }

        for (int idx8 = tid; idx8 < (BTileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (v < w && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(b_combined + static_cast<size_t>(v) * k + k_col);
            }
            const int group = row / 32;
            const int local_row = row % 32;
            auto sh_bt = sh_bt_group(group);
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_bt.store(col + lane_byte, local_row, pearl_i8_lane(packed, lane_byte));
            }
        }

        iris::hip::block_sync();

        uint32_t thread_xored = 0;
        #pragma unroll
        for (int group_base = 0; group_base < NGroups; group_base += ChunkGroups) {
            typename Wmma::c_fragment c[ChunkGroups];
            #pragma unroll
            for (int group = 0; group < ChunkGroups; ++group) {
                c[group] = Wmma::zero_c();
            }

            #pragma unroll
            for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
                const auto a_frag = iris::hip::wmma_load_ab_row<Wmma>(
                    sh_a,
                    local_m_base + lane16,
                    k_half + k_pack_base);
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    auto sh_bt = sh_bt_group(group_base + group);
                    const auto b_frag = iris::hip::wmma_load_ab_col<Wmma>(
                        sh_bt,
                        k_half + k_pack_base,
                        local_n_base0 + lane16);
                    c[group] = Wmma::mma(a_frag, b_frag, c[group]);
                }
            }

            #pragma unroll
            for (int group = 0; group < ChunkGroups; ++group) {
                const int global_group = group_base + group;
                #pragma unroll
                for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                    const int local_u = local_m_base + Wmma::c_row(lane, elem);
                    const int local_v = local_n_base0 + global_group * kJackpotWmmaTileRowsB + Wmma::c_col(lane);
                    const int global_u = tile_u_base + local_u;
                    const int global_v = tile_v_base + local_v;
                    if (global_u < h && global_v < w) {
                        running_sums[global_group][elem] += c[group][elem];
                        thread_xored ^= static_cast<uint32_t>(running_sums[global_group][elem]);
                    }
                }
            }
        }

        sh_xor[tid] = thread_xored;
        uint32_t xored = 0;
        if constexpr (ReduceMode == 1) {
            xored = pearl_manual_block_reduce_xor_u32<Threads>(thread_xored, sh_xor);
        } else if constexpr (ReduceMode == 2) {
            xored = pearl_wave_publish_block_reduce_xor_u32<Threads, false>(thread_xored, sh_xor);
        } else if constexpr (ReduceMode == 3) {
            xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);
        } else {
            xored = iris::hip::block_reduce_xor_u32_wave_first<Threads>(thread_xored, sh_xor);
        }

        if constexpr (PreStoreSync) {
            iris::hip::block_sync();
        }

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }

        iris::hip::block_sync();
    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_core_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma64TileRowsB;
    constexpr int NGroups = 2;
    constexpr int Threads = 128;
    constexpr int ChunkGroups = 2;
    using Wmma = iris::hip::WmmaI8I8I32_16x16x16<true, true, false>;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[2][ATileRows * kRank];
    __shared__ signed char sh_bt_storage[2][kRank * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_sq64x64_du32_v1 expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_sq64x64_du32_v1 expects two M subpasses");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;

    int32_t running_sums[MSubpasses][NGroups][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < NGroups; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto load_stage = [&](int stage, int tile) {
        auto sh_a_stage = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage[stage]);
        const int base_col = tile * kRank;

        for (int idx8 = tid; idx8 < (ATileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (u < h && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(a_combined + static_cast<size_t>(u) * k + k_col);
            }
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_a_stage.store(row, col + lane_byte, pearl_i8_lane(packed, lane_byte));
            }
        }

        for (int idx8 = tid; idx8 < (BTileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (v < w && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(b_combined + static_cast<size_t>(v) * k + k_col);
            }
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            auto sh_bt_stage = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage[stage], group);
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_bt_stage.store(col + lane_byte, local_row, pearl_i8_lane(packed, lane_byte));
            }
        }
    };

    load_stage(0, 0);
    iris::hip::block_sync();

    for (int tile = 0; tile < tiles; ++tile) {
        const int curr_stage = tile & 1;
        const int next_stage = curr_stage ^ 1;
        auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage[curr_stage]);

        uint32_t thread_xored = 0;
        #pragma unroll
        for (int group_base = 0; group_base < NGroups; group_base += ChunkGroups) {
            typename Wmma::c_fragment c[MSubpasses][ChunkGroups];
            #pragma unroll
            for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    c[m_pass][group] = Wmma::zero_c();
                }
            }

            // DU32 means each outer tile is accumulated over two 16-wide WMMA K steps.
            #pragma unroll
            for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
                typename Wmma::ab_fragment a_frag[MSubpasses];
                #pragma unroll
                for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                    const int local_m_base = (wave_m + m_pass * WaveRowsPerSubpass) * Wmma::m;
                    a_frag[m_pass] = iris::hip::wmma_load_ab_row<Wmma>(
                        sh_a,
                        local_m_base + lane16,
                        k_half + k_pack_base);
                }
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    const int global_group = group_base + group;
                    const auto b_frag = iris::hip::wmma_load_ab_col<Wmma>(
                        pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage[curr_stage], global_group),
                        k_half + k_pack_base,
                        local_n_base0 + lane16);
                    #pragma unroll
                    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                        c[m_pass][group] = Wmma::mma(a_frag[m_pass], b_frag, c[m_pass][group]);
                    }
                }
            }

            #pragma unroll
            for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                const int local_m_base = (wave_m + m_pass * WaveRowsPerSubpass) * Wmma::m;
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    const int global_group = group_base + group;
                    #pragma unroll
                    for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                        const int local_u = local_m_base + Wmma::c_row(lane, elem);
                        const int local_v = local_n_base0 + global_group * kJackpotWmmaTileRowsB + Wmma::c_col(lane);
                        const int global_u = tile_u_base + local_u;
                        const int global_v = tile_v_base + local_v;
                        if (global_u < h && global_v < w) {
                            running_sums[m_pass][global_group][elem] += c[m_pass][group][elem];
                            thread_xored ^= static_cast<uint32_t>(running_sums[m_pass][global_group][elem]);
                        }
                    }
                }
            }
        }

        if (tile + 1 < tiles) {
            load_stage(next_stage, tile + 1);
        }

        iris::hip::block_sync();

        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }

    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_serial_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma64TileRowsB;
    constexpr int NGroups = 2;
    constexpr int Threads = 128;
    constexpr int ChunkGroups = 2;
    using Wmma = iris::hip::WmmaI8I8I32_16x16x16<true, true, false>;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[ATileRows * kRank];
    __shared__ signed char sh_bt_storage[kRank * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_sq64x64_serial expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_sq64x64_serial expects two M subpasses");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;

    auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage);

    int32_t running_sums[MSubpasses][NGroups][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < NGroups; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx8 = tid; idx8 < (ATileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (u < h && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(a_combined + static_cast<size_t>(u) * k + k_col);
            }
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_a.store(row, col + lane_byte, pearl_i8_lane(packed, lane_byte));
            }
        }

        for (int idx8 = tid; idx8 < (BTileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (v < w && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(b_combined + static_cast<size_t>(v) * k + k_col);
            }
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            auto sh_bt = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, group);
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_bt.store(col + lane_byte, local_row, pearl_i8_lane(packed, lane_byte));
            }
        }

        iris::hip::block_sync();

        uint32_t thread_xored = 0;
        #pragma unroll
        for (int group_base = 0; group_base < NGroups; group_base += ChunkGroups) {
            typename Wmma::c_fragment c[MSubpasses][ChunkGroups];
            #pragma unroll
            for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    c[m_pass][group] = Wmma::zero_c();
                }
            }

            #pragma unroll
            for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
                typename Wmma::ab_fragment a_frag[MSubpasses];
                #pragma unroll
                for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                    const int local_m_base = (wave_m + m_pass * WaveRowsPerSubpass) * Wmma::m;
                    a_frag[m_pass] = iris::hip::wmma_load_ab_row<Wmma>(
                        sh_a,
                        local_m_base + lane16,
                        k_half + k_pack_base);
                }
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    const int global_group = group_base + group;
                    const auto b_frag = iris::hip::wmma_load_ab_col<Wmma>(
                        pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, global_group),
                        k_half + k_pack_base,
                        local_n_base0 + lane16);
                    #pragma unroll
                    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                        c[m_pass][group] = Wmma::mma(a_frag[m_pass], b_frag, c[m_pass][group]);
                    }
                }
            }

            #pragma unroll
            for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                const int local_m_base = (wave_m + m_pass * WaveRowsPerSubpass) * Wmma::m;
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    const int global_group = group_base + group;
                    #pragma unroll
                    for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                        const int local_u = local_m_base + Wmma::c_row(lane, elem);
                        const int local_v = local_n_base0 + global_group * kJackpotWmmaTileRowsB + Wmma::c_col(lane);
                        const int global_u = tile_u_base + local_u;
                        const int global_v = tile_v_base + local_v;
                        if (global_u < h && global_v < w) {
                            running_sums[m_pass][global_group][elem] += c[m_pass][group][elem];
                            thread_xored ^= static_cast<uint32_t>(running_sums[m_pass][global_group][elem]);
                        }
                    }
                }
            }
        }

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_dense_serial_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma64TileRowsB;
    constexpr int Threads = 128;
    using Wmma = iris::hip::WmmaI8I8I32_16x16x16<true, true, false>;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[ATileRows * kRank];
    __shared__ signed char sh_bt_storage[kRank * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_sq64x64_dense_serial expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_sq64x64_dense_serial expects two M subpasses");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;
    const int local_m_base0 = wave_m * Wmma::m;
    const int local_m_base1 = (wave_m + WaveRowsPerSubpass) * Wmma::m;

    auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage);
    auto sh_bt0 = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, 0);
    auto sh_bt1 = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, 1);

    int32_t running_sums[MSubpasses][2][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto accumulate_fragment = [&](typename Wmma::c_fragment& frag, int32_t* accum, int local_m_base, int local_n_base, uint32_t& thread_xored) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                accum[elem] += frag[elem];
                thread_xored ^= static_cast<uint32_t>(accum[elem]);
            }
        }
    };

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx8 = tid; idx8 < (ATileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (u < h && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(a_combined + static_cast<size_t>(u) * k + k_col);
            }
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_a.store(row, col + lane_byte, pearl_i8_lane(packed, lane_byte));
            }
        }

        for (int idx8 = tid; idx8 < (BTileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (v < w && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(b_combined + static_cast<size_t>(v) * k + k_col);
            }
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            auto sh_bt = group == 0 ? sh_bt0 : sh_bt1;
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_bt.store(col + lane_byte, local_row, pearl_i8_lane(packed, lane_byte));
            }
        }

        iris::hip::block_sync();

        typename Wmma::c_fragment c00 = Wmma::zero_c();
        typename Wmma::c_fragment c01 = Wmma::zero_c();
        typename Wmma::c_fragment c10 = Wmma::zero_c();
        typename Wmma::c_fragment c11 = Wmma::zero_c();

        #pragma unroll
        for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
            const auto a_frag0 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base0 + lane16, k_half + k_pack_base);
            const auto a_frag1 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base1 + lane16, k_half + k_pack_base);
            const auto b_frag0 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt0, k_half + k_pack_base, local_n_base0 + lane16);
            const auto b_frag1 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt1, k_half + k_pack_base, local_n_base0 + lane16);

            c00 = Wmma::mma(a_frag0, b_frag0, c00);
            c01 = Wmma::mma(a_frag0, b_frag1, c01);
            c10 = Wmma::mma(a_frag1, b_frag0, c10);
            c11 = Wmma::mma(a_frag1, b_frag1, c11);
        }

        uint32_t thread_xored = 0;
        accumulate_fragment(c00, running_sums[0][0], local_m_base0, local_n_base0, thread_xored);
        accumulate_fragment(c01, running_sums[0][1], local_m_base0, local_n_base1, thread_xored);
        accumulate_fragment(c10, running_sums[1][0], local_m_base1, local_n_base0, thread_xored);
        accumulate_fragment(c11, running_sums[1][1], local_m_base1, local_n_base1, thread_xored);

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_dense_pipe2_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma64TileRowsB;
    constexpr int Threads = 128;
    using Wmma = iris::hip::WmmaI8I8I32_16x16x16<true, true, false>;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[2][ATileRows * kRank];
    __shared__ signed char sh_bt_storage[2][kRank * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_sq64x64_dense_pipe2 expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_sq64x64_dense_pipe2 expects two M subpasses");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;
    const int local_m_base0 = wave_m * Wmma::m;
    const int local_m_base1 = (wave_m + WaveRowsPerSubpass) * Wmma::m;

    int32_t running_sums[MSubpasses][2][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto load_stage = [&](int stage, int tile_idx) {
        const int base_col = tile_idx * kRank;
        auto sh_a_stage = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage[stage]);
        auto sh_bt0_stage = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage[stage], 0);
        auto sh_bt1_stage = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage[stage], 1);

        for (int idx8 = tid; idx8 < (ATileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (u < h && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(a_combined + static_cast<size_t>(u) * k + k_col);
            }
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_a_stage.store(row, col + lane_byte, pearl_i8_lane(packed, lane_byte));
            }
        }

        for (int idx8 = tid; idx8 < (BTileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (v < w && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(b_combined + static_cast<size_t>(v) * k + k_col);
            }
            const int local_row = row % kJackpotWmmaTileRowsB;
            auto sh_bt = row < kJackpotWmmaTileRowsB ? sh_bt0_stage : sh_bt1_stage;
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_bt.store(col + lane_byte, local_row, pearl_i8_lane(packed, lane_byte));
            }
        }
    };

    auto accumulate_fragment = [&](typename Wmma::c_fragment& frag, int32_t* accum, int local_m_base, int local_n_base, uint32_t& thread_xored) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                accum[elem] += frag[elem];
                thread_xored ^= static_cast<uint32_t>(accum[elem]);
            }
        }
    };

    auto compute_tile = [&](int curr_stage, uint32_t& thread_xored) {
        auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage[curr_stage]);
        auto sh_bt0 = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage[curr_stage], 0);
        auto sh_bt1 = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage[curr_stage], 1);

        typename Wmma::c_fragment c00 = Wmma::zero_c();
        typename Wmma::c_fragment c01 = Wmma::zero_c();
        typename Wmma::c_fragment c10 = Wmma::zero_c();
        typename Wmma::c_fragment c11 = Wmma::zero_c();

        #pragma unroll
        for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
            const auto a_frag0 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base0 + lane16, k_half + k_pack_base);
            const auto a_frag1 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base1 + lane16, k_half + k_pack_base);
            const auto b_frag0 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt0, k_half + k_pack_base, local_n_base0 + lane16);
            const auto b_frag1 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt1, k_half + k_pack_base, local_n_base0 + lane16);

            c00 = Wmma::mma(a_frag0, b_frag0, c00);
            c01 = Wmma::mma(a_frag0, b_frag1, c01);
            c10 = Wmma::mma(a_frag1, b_frag0, c10);
            c11 = Wmma::mma(a_frag1, b_frag1, c11);
        }

        accumulate_fragment(c00, running_sums[0][0], local_m_base0, local_n_base0, thread_xored);
        accumulate_fragment(c01, running_sums[0][1], local_m_base0, local_n_base1, thread_xored);
        accumulate_fragment(c10, running_sums[1][0], local_m_base1, local_n_base0, thread_xored);
        accumulate_fragment(c11, running_sums[1][1], local_m_base1, local_n_base1, thread_xored);
    };

    load_stage(0, 0);
    iris::hip::block_sync();

    for (int tile = 0; tile < tiles; ++tile) {
        const int curr_stage = tile & 1;
        const int next_stage = curr_stage ^ 1;
        uint32_t thread_xored = 0;

        if (tile + 1 < tiles) {
            load_stage(next_stage, tile + 1);
        }
        compute_tile(curr_stage, thread_xored);

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_dense_grouped_serial_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma64TileRowsB;
    constexpr int Threads = 128;
    using Wmma = iris::hip::WmmaI8I8I32_16x16x16<true, true, false>;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[ATileRows * kRank];
    __shared__ signed char sh_bt_storage[kRank * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_sq64x64_dense_grouped_serial expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_sq64x64_dense_grouped_serial expects two M subpasses");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;
    const int local_m_base0 = wave_m * Wmma::m;
    const int local_m_base1 = (wave_m + WaveRowsPerSubpass) * Wmma::m;

    auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage);
    auto sh_bt0 = iris::hip::make_grouped_wmma_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 0);
    auto sh_bt1 = iris::hip::make_grouped_wmma_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 1);

    int32_t running_sums[MSubpasses][2][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto accumulate_fragment = [&](typename Wmma::c_fragment& frag, int32_t* accum, int local_m_base, int local_n_base, uint32_t& thread_xored) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                accum[elem] += frag[elem];
                thread_xored ^= static_cast<uint32_t>(accum[elem]);
            }
        }
    };

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx8 = tid; idx8 < (ATileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (u < h && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(a_combined + static_cast<size_t>(u) * k + k_col);
            }
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_a.store(row, col + lane_byte, pearl_i8_lane(packed, lane_byte));
            }
        }

        for (int idx8 = tid; idx8 < (BTileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (v < w && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(b_combined + static_cast<size_t>(v) * k + k_col);
            }
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            auto sh_bt = group == 0 ? sh_bt0 : sh_bt1;
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_bt.store(col + lane_byte, local_row, pearl_i8_lane(packed, lane_byte));
            }
        }

        iris::hip::block_sync();

        typename Wmma::c_fragment c00 = Wmma::zero_c();
        typename Wmma::c_fragment c01 = Wmma::zero_c();
        typename Wmma::c_fragment c10 = Wmma::zero_c();
        typename Wmma::c_fragment c11 = Wmma::zero_c();

        #pragma unroll
        for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
            const auto a_frag0 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base0 + lane16, k_half + k_pack_base);
            const auto a_frag1 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base1 + lane16, k_half + k_pack_base);
            const auto b_frag0 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt0, k_half + k_pack_base, local_n_base0 + lane16);
            const auto b_frag1 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt1, k_half + k_pack_base, local_n_base0 + lane16);

            c00 = Wmma::mma(a_frag0, b_frag0, c00);
            c01 = Wmma::mma(a_frag0, b_frag1, c01);
            c10 = Wmma::mma(a_frag1, b_frag0, c10);
            c11 = Wmma::mma(a_frag1, b_frag1, c11);
        }

        uint32_t thread_xored = 0;
        accumulate_fragment(c00, running_sums[0][0], local_m_base0, local_n_base0, thread_xored);
        accumulate_fragment(c01, running_sums[0][1], local_m_base0, local_n_base1, thread_xored);
        accumulate_fragment(c10, running_sums[1][0], local_m_base1, local_n_base0, thread_xored);
        accumulate_fragment(c11, running_sums[1][1], local_m_base1, local_n_base1, thread_xored);

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_rdna3b_serial_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    using Backend = tnn::hip::iris::gemm::DefaultBackend;
    using Wmma = Backend::WmmaI8;
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma64TileRowsB;
    constexpr int BStride = Backend::kBStride;
    constexpr int Threads = 128;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[ATileRows * kRank];
    __shared__ signed char sh_bt_storage[BStride * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_sq64x64_rdna3b_serial expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_sq64x64_rdna3b_serial expects two M subpasses");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;
    const int local_m_base0 = wave_m * Wmma::m;
    const int local_m_base1 = (wave_m + WaveRowsPerSubpass) * Wmma::m;

    auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage);
    auto sh_bt0 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 0);
    auto sh_bt1 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 1);

    int32_t running_sums[MSubpasses][2][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto accumulate_fragment = [&](typename Wmma::c_fragment& frag, int32_t* accum, int local_m_base, int local_n_base, uint32_t& thread_xored) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                accum[elem] += frag[elem];
                thread_xored ^= static_cast<uint32_t>(accum[elem]);
            }
        }
    };

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx16 = tid; idx16 < (ATileRows * kRank) / 16; idx16 += Threads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            const PearlVec128 packed = pearl_load_vec128_or_zero(
                a_combined + static_cast<size_t>(u) * k + k_col,
                u < h && k_col + 15 < k);
            pearl_store_vec128_row_major_or_scalar(sh_a_storage, kRank, row, col, packed);
        }

        for (int idx16 = tid; idx16 < (BTileRows * kRank) / 16; idx16 += Threads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            const PearlVec128 packed = pearl_load_vec128_or_zero(
                b_combined + static_cast<size_t>(v) * k + k_col,
                v < w && k_col + 15 < k);
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            pearl_store_vec128_col_major_or_scalar(
                sh_bt_storage + group * (BStride * kJackpotWmmaTileRowsB),
                BStride,
                local_row,
                col,
                packed);
        }

        iris::hip::block_sync();

        typename Wmma::c_fragment c00 = Wmma::zero_c();
        typename Wmma::c_fragment c01 = Wmma::zero_c();
        typename Wmma::c_fragment c10 = Wmma::zero_c();
        typename Wmma::c_fragment c11 = Wmma::zero_c();

        #pragma unroll
        for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
            const auto a_frag0 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base0 + lane16, k_half + k_pack_base);
            const auto a_frag1 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base1 + lane16, k_half + k_pack_base);
            const auto b_frag0 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt0, k_half + k_pack_base, local_n_base0 + lane16);
            const auto b_frag1 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt1, k_half + k_pack_base, local_n_base0 + lane16);

            c00 = Wmma::mma(a_frag0, b_frag0, c00);
            c01 = Wmma::mma(a_frag0, b_frag1, c01);
            c10 = Wmma::mma(a_frag1, b_frag0, c10);
            c11 = Wmma::mma(a_frag1, b_frag1, c11);
        }

        uint32_t thread_xored = 0;
        accumulate_fragment(c00, running_sums[0][0], local_m_base0, local_n_base0, thread_xored);
        accumulate_fragment(c01, running_sums[0][1], local_m_base0, local_n_base1, thread_xored);
        accumulate_fragment(c10, running_sums[1][0], local_m_base1, local_n_base0, thread_xored);
        accumulate_fragment(c11, running_sums[1][1], local_m_base1, local_n_base1, thread_xored);

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}

template <typename Wmma>
IRIS_DEVICE_INLINE int pearl_rdna3b_ds_swizzle_swap16_asm(int value)
{
#if defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    int out;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)"
        : "=v"(out)
        : "v"(value));
    return out;
#else
    return iris::hip::wmma_swap16_i32(value);
#endif
}

template <typename Wmma>
IRIS_DEVICE_INLINE typename Wmma::ab_fragment pearl_rdna3b_prepare_ab_asm(
    typename Wmma::storage_fragment src)
{
    const auto packed = iris::hip::bit_cast_vec<iris::hip::i32x2_t>(src);
    iris::hip::i32x2_t swapped{};
    swapped[0] = pearl_rdna3b_ds_swizzle_swap16_asm<Wmma>(packed[0]);
    swapped[1] = pearl_rdna3b_ds_swizzle_swap16_asm<Wmma>(packed[1]);
    return iris::hip::bit_cast_vec<typename Wmma::ab_fragment>(
        iris::hip::concat_i32x2(packed, swapped));
}

template <typename Wmma>
IRIS_DEVICE_INLINE typename Wmma::c_fragment pearl_rdna3b_wmma_asm_step(
    typename Wmma::ab_fragment a,
    typename Wmma::ab_fragment b,
    typename Wmma::c_fragment c)
{
#if defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    auto out = c;
    const auto a_regs = iris::hip::bit_cast_vec<iris::hip::i32x4_t>(a);
    const auto b_regs = iris::hip::bit_cast_vec<iris::hip::i32x4_t>(b);
    asm volatile(
        "v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]"
        : "+v"(out)
        : "v"(a_regs), "v"(b_regs));
    return out;
#else
    return Wmma::mma(a, b, c);
#endif
}

IRIS_DEVICE_INLINE iris::hip::i8x16_t pearl_rdna3b_ds_load_b128_asm(const signed char* ptr)
{
#if defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    iris::hip::i32x4_t out{};
    const unsigned int addr = static_cast<unsigned int>(reinterpret_cast<uintptr_t>(ptr));
    asm volatile(
        "ds_read_b128 %0, %1"
        : "=v"(out)
        : "v"(addr));
    return iris::hip::bit_cast_vec<iris::hip::i8x16_t>(out);
#else
    return iris::hip::bit_cast_vec<iris::hip::i8x16_t>(
        iris::hip::VectorAccess<int8_t, 16>::load_aligned(ptr));
#endif
}

template <typename Wmma>
IRIS_DEVICE_INLINE typename Wmma::storage_fragment pearl_rdna3b_take_low_storage(
    iris::hip::i8x16_t wide)
{
    auto out = Wmma::zero_storage();
    #pragma unroll
    for (int i = 0; i < Wmma::storage_elems; ++i) {
        out[i] = wide[i];
    }
    return out;
}

IRIS_DEVICE_INLINE void pearl_store_dup8_record_from_global(
    signed char* dst,
    const signed char* src,
    bool row_in_bounds,
    int valid_cols)
{
    using Access = iris::hip::VectorAccess<int8_t, 16>;
    iris::hip::i8x16_t record{};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const signed char value = (row_in_bounds && i < valid_cols) ? src[i] : static_cast<signed char>(0);
        record[i] = value;
        record[8 + i] = value;
    }
    Access::store_aligned(dst, iris::hip::bit_cast_vec<typename Access::Vec>(record));
}

IRIS_DEVICE_INLINE int pearl_valid_record_cols(int col_base, int k)
{
    const int remaining = k - col_base;
    if (remaining <= 0) {
        return 0;
    }
    return remaining < 8 ? remaining : 8;
}

IRIS_DEVICE_INLINE constexpr int pearl_rdna3b_b_record_offset(
    int group,
    int step,
    int half,
    int local_col)
{
    return (((group * 2 + step) * 2 + half) * 32 + local_col) * 16;
}

template <int Rank, int BStride, bool BMajorSwizzle = false, int BMajorStripeY = 0, bool FullAsmCluster = false, bool BRecordLayout = false, bool BHalfPad16 = false, bool BStepMajor16 = false, bool ProducerOverlap = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_rdna3b_reg2_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    using Wmma = tnn::hip::iris::gemm::DefaultBackend::WmmaI8;
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma64TileRowsB;
    constexpr int Threads = 128;
    constexpr int KSteps = Rank / Wmma::k;
    constexpr int StageCount = ProducerOverlap ? 2 : 1;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    static_assert(!(ProducerOverlap && BRecordLayout),
        "producer overlap path currently keeps the original LDS tile contract");

    __shared__ signed char sh_a_storage[StageCount * ATileRows * Rank];
    __shared__ signed char sh_bt_storage[StageCount * BStride * BTileRows];
    __shared__ __attribute__((aligned(16))) signed char sh_b_records[2 * 2 * 2 * 32 * 16];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    if constexpr (BMajorStripeY > 0) {
        pearl_compute_virtual_block_coords_bmajor_striped<BMajorStripeY>(virtual_block_x, virtual_block_y);
    } else {
        pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    }
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / Rank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_sq64x64_rdna3b_reg2 expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_sq64x64_rdna3b_reg2 expects two M subpasses");
    static_assert((Rank % Wmma::k) == 0, "rank must be a multiple of WMMA K");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;
    const int local_m_base0 = wave_m * Wmma::m;
    const int local_m_base1 = (wave_m + WaveRowsPerSubpass) * Wmma::m;

    int32_t running_sums[MSubpasses][2][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto accumulate_fragment = [&](typename Wmma::c_fragment& frag, int32_t* accum, int local_m_base, int local_n_base, uint32_t& thread_xored) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                accum[elem] += frag[elem];
                thread_xored ^= static_cast<uint32_t>(accum[elem]);
            }
        }
    };

    auto stage_a_ptr = [&](int stage) {
        return sh_a_storage + stage * (ATileRows * Rank);
    };
    auto stage_b_ptr = [&](int stage) {
        return sh_bt_storage + stage * (BStride * BTileRows);
    };
    auto make_sh_a = [&](int stage) {
        return iris::hip::make_lds_tile<signed char, ATileRows, Rank>(stage_a_ptr(stage));
    };
    auto make_sh_bt0 = [&](int stage) {
        signed char* const stage_ptr = stage_b_ptr(stage);
        if constexpr (BStepMajor16) {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_stepmajor_tile<Rank, kJackpotWmmaTileRowsB, 16>(
                stage_ptr, 0);
        } else if constexpr (BHalfPad16) {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_halfpad_tile<Rank, kJackpotWmmaTileRowsB, BStride, 16, 16>(
                stage_ptr, 0);
        } else {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_crosswise_tile<Rank, kJackpotWmmaTileRowsB, BStride>(
                stage_ptr, 0);
        }
    };
    auto make_sh_bt1 = [&](int stage) {
        signed char* const stage_ptr = stage_b_ptr(stage);
        if constexpr (BStepMajor16) {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_stepmajor_tile<Rank, kJackpotWmmaTileRowsB, 16>(
                stage_ptr, 1);
        } else if constexpr (BHalfPad16) {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_halfpad_tile<Rank, kJackpotWmmaTileRowsB, BStride, 16, 16>(
                stage_ptr, 1);
        } else {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_crosswise_tile<Rank, kJackpotWmmaTileRowsB, BStride>(
                stage_ptr, 1);
        }
    };
    auto stage_tile = [&](int stage, int tile_idx) {
        const int base_col = tile_idx * Rank;
        signed char* const sh_a_stage = stage_a_ptr(stage);
        signed char* const sh_b_stage = stage_b_ptr(stage);

        for (int idx16 = tid; idx16 < (ATileRows * Rank) / 16; idx16 += Threads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / Rank;
            const int col = scalar_idx - row * Rank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            const PearlVec128 packed = pearl_load_vec128_or_zero(
                a_combined + static_cast<size_t>(u) * k + k_col,
                u < h && k_col + 15 < k);
            pearl_store_vec128_row_major_or_scalar(sh_a_stage, Rank, row, col, packed);
        }

        if constexpr (BRecordLayout) {
            constexpr int BGroups = 2;
            constexpr int BHalves = 2;
            constexpr int RecordCols = 32;
            for (int rec = tid; rec < BGroups * KSteps * BHalves * RecordCols; rec += Threads) {
                const int group = rec / (KSteps * BHalves * RecordCols);
                const int rem0 = rec - group * (KSteps * BHalves * RecordCols);
                const int step = rem0 / (BHalves * RecordCols);
                const int rem1 = rem0 - step * (BHalves * RecordCols);
                const int half = rem1 / RecordCols;
                const int local_col = rem1 - half * RecordCols;
                const int v = tile_v_base + group * kJackpotWmmaTileRowsB + local_col;
                const int k_col = base_col + step * Wmma::k + half * Wmma::storage_elems;
                pearl_store_dup8_record_from_global(
                    sh_b_records + pearl_rdna3b_b_record_offset(group, step, half, local_col),
                    b_combined + static_cast<size_t>(v) * k + k_col,
                    v < w,
                    pearl_valid_record_cols(k_col, k));
            }
        } else {
            for (int idx16 = tid; idx16 < (BTileRows * Rank) / 16; idx16 += Threads) {
                const int scalar_idx = idx16 * 16;
                const int row = scalar_idx / Rank;
                const int col = scalar_idx - row * Rank;
                const int v = tile_v_base + row;
                const int k_col = base_col + col;
                const PearlVec128 packed = pearl_load_vec128_or_zero(
                    b_combined + static_cast<size_t>(v) * k + k_col,
                    v < w && k_col + 15 < k);
                const int group = row / kJackpotWmmaTileRowsB;
                const int local_row = row % kJackpotWmmaTileRowsB;
                if constexpr (BStepMajor16) {
                    pearl_store_vec128_col_major_stepmajor16_or_scalar(
                        sh_b_stage + group * (Rank * kJackpotWmmaTileRowsB),
                        kJackpotWmmaTileRowsB,
                        local_row,
                        col,
                        packed);
                } else if constexpr (BHalfPad16) {
                    pearl_store_vec128_col_major_halfpad16_or_scalar(
                        sh_b_stage + group * (BStride * kJackpotWmmaTileRowsB),
                        BStride,
                        local_row,
                        col,
                        packed);
                } else {
                    pearl_store_vec128_col_major_or_scalar(
                        sh_b_stage + group * (BStride * kJackpotWmmaTileRowsB),
                        BStride,
                        local_row,
                        col,
                        packed);
                }
            }
        }
    };

    if constexpr (ProducerOverlap) {
        stage_tile(0, 0);
        iris::hip::block_sync();
    }

    for (int tile = 0; tile < tiles; ++tile) {
        const int stage = ProducerOverlap ? (tile & 1) : 0;
        if constexpr (!ProducerOverlap) {
            stage_tile(stage, tile);
            iris::hip::block_sync();
        }

        auto sh_a = make_sh_a(stage);
        auto sh_bt0 = make_sh_bt0(stage);
        auto sh_bt1 = make_sh_bt1(stage);

        typename Wmma::c_fragment c00 = Wmma::zero_c();
        typename Wmma::c_fragment c01 = Wmma::zero_c();
        typename Wmma::c_fragment c10 = Wmma::zero_c();
        typename Wmma::c_fragment c11 = Wmma::zero_c();

        if constexpr (FullAsmCluster) {
            static_assert(KSteps == 2, "full asm cluster currently expects rank 32 / ksteps 2");

            const auto a0_storage_s0 = iris::hip::wmma_load_storage_row<Wmma>(
                sh_a, local_m_base0 + lane16, k_pack_base);
            const auto a1_storage_s0 = iris::hip::wmma_load_storage_row<Wmma>(
                sh_a, local_m_base1 + lane16, k_pack_base);
            const auto b0_storage_s0 = [&]() {
                if constexpr (BRecordLayout) {
                    return pearl_rdna3b_take_low_storage<Wmma>(
                        pearl_rdna3b_ds_load_b128_asm(
                            sh_b_records + pearl_rdna3b_b_record_offset(0, 0, k_pack_base / Wmma::storage_elems, local_n_base0 + lane16)));
                } else {
                    return iris::hip::wmma_load_storage_col<Wmma>(
                        sh_bt0, k_pack_base, local_n_base0 + lane16);
                }
            }();
            const auto b1_storage_s0 = [&]() {
                if constexpr (BRecordLayout) {
                    return pearl_rdna3b_take_low_storage<Wmma>(
                        pearl_rdna3b_ds_load_b128_asm(
                            sh_b_records + pearl_rdna3b_b_record_offset(1, 0, k_pack_base / Wmma::storage_elems, local_n_base0 + lane16)));
                } else {
                    return iris::hip::wmma_load_storage_col<Wmma>(
                        sh_bt1, k_pack_base, local_n_base0 + lane16);
                }
            }();
            const auto a0_storage_s1 = iris::hip::wmma_load_storage_row<Wmma>(
                sh_a, local_m_base0 + lane16, Wmma::k + k_pack_base);
            const auto a1_storage_s1 = iris::hip::wmma_load_storage_row<Wmma>(
                sh_a, local_m_base1 + lane16, Wmma::k + k_pack_base);
            const auto b0_storage_s1 = [&]() {
                if constexpr (BRecordLayout) {
                    return pearl_rdna3b_take_low_storage<Wmma>(
                        pearl_rdna3b_ds_load_b128_asm(
                            sh_b_records + pearl_rdna3b_b_record_offset(0, 1, k_pack_base / Wmma::storage_elems, local_n_base0 + lane16)));
                } else {
                    return iris::hip::wmma_load_storage_col<Wmma>(
                        sh_bt0, Wmma::k + k_pack_base, local_n_base0 + lane16);
                }
            }();
            const auto b1_storage_s1 = [&]() {
                if constexpr (BRecordLayout) {
                    return pearl_rdna3b_take_low_storage<Wmma>(
                        pearl_rdna3b_ds_load_b128_asm(
                            sh_b_records + pearl_rdna3b_b_record_offset(1, 1, k_pack_base / Wmma::storage_elems, local_n_base0 + lane16)));
                } else {
                    return iris::hip::wmma_load_storage_col<Wmma>(
                        sh_bt1, Wmma::k + k_pack_base, local_n_base0 + lane16);
                }
            }();

            if constexpr (ProducerOverlap) {
                // Current-tile storage fragments are resident in VGPRs now, so
                // the alternate LDS stage can be safely populated before the
                // asm prepare+WMMA burst consumes them.
                if (tile + 1 < tiles) {
                    stage_tile(stage ^ 1, tile + 1);
                }
            }

            iris::hip::wait_lgkmcnt0();

            const auto a0s0 = pearl_rdna3b_prepare_ab_asm<Wmma>(a0_storage_s0);
            const auto a1s0 = pearl_rdna3b_prepare_ab_asm<Wmma>(a1_storage_s0);
            const auto b0s0 = pearl_rdna3b_prepare_ab_asm<Wmma>(b0_storage_s0);
            const auto b1s0 = pearl_rdna3b_prepare_ab_asm<Wmma>(b1_storage_s0);
            const auto a0s1 = pearl_rdna3b_prepare_ab_asm<Wmma>(a0_storage_s1);
            const auto a1s1 = pearl_rdna3b_prepare_ab_asm<Wmma>(a1_storage_s1);
            const auto b0s1 = pearl_rdna3b_prepare_ab_asm<Wmma>(b0_storage_s1);
            const auto b1s1 = pearl_rdna3b_prepare_ab_asm<Wmma>(b1_storage_s1);

            c00 = pearl_rdna3b_wmma_asm_step<Wmma>(a0s0, b0s0, c00);
            c01 = pearl_rdna3b_wmma_asm_step<Wmma>(a0s0, b1s0, c01);
            c10 = pearl_rdna3b_wmma_asm_step<Wmma>(a1s0, b0s0, c10);
            c11 = pearl_rdna3b_wmma_asm_step<Wmma>(a1s0, b1s0, c11);
            c00 = pearl_rdna3b_wmma_asm_step<Wmma>(a0s1, b0s1, c00);
            c01 = pearl_rdna3b_wmma_asm_step<Wmma>(a0s1, b1s1, c01);
            c10 = pearl_rdna3b_wmma_asm_step<Wmma>(a1s1, b0s1, c10);
            c11 = pearl_rdna3b_wmma_asm_step<Wmma>(a1s1, b1s1, c11);
        } else {
            typename Wmma::ab_fragment a_curr0 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base0 + lane16, k_pack_base);
            typename Wmma::ab_fragment a_curr1 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base1 + lane16, k_pack_base);
            typename Wmma::ab_fragment b_curr0 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt0, k_pack_base, local_n_base0 + lane16);
            typename Wmma::ab_fragment b_curr1 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt1, k_pack_base, local_n_base0 + lane16);

            #pragma unroll
            for (int step = 0; step < KSteps; ++step) {
                typename Wmma::ab_fragment a_next0 = Wmma::zero_ab();
                typename Wmma::ab_fragment a_next1 = Wmma::zero_ab();
                typename Wmma::ab_fragment b_next0 = Wmma::zero_ab();
                typename Wmma::ab_fragment b_next1 = Wmma::zero_ab();

                if constexpr (KSteps > 1) {
                    if (step + 1 < KSteps) {
                        const int k_half_next = (step + 1) * Wmma::k;
                        a_next0 = iris::hip::wmma_load_ab_row<Wmma>(
                            sh_a, local_m_base0 + lane16, k_half_next + k_pack_base);
                        a_next1 = iris::hip::wmma_load_ab_row<Wmma>(
                            sh_a, local_m_base1 + lane16, k_half_next + k_pack_base);
                        b_next0 = iris::hip::wmma_load_ab_col<Wmma>(
                            sh_bt0, k_half_next + k_pack_base, local_n_base0 + lane16);
                        b_next1 = iris::hip::wmma_load_ab_col<Wmma>(
                            sh_bt1, k_half_next + k_pack_base, local_n_base0 + lane16);
                    }
                }

                iris::hip::wait_lgkmcnt0();
                c00 = Wmma::mma(a_curr0, b_curr0, c00);
                c01 = Wmma::mma(a_curr0, b_curr1, c01);
                c10 = Wmma::mma(a_curr1, b_curr0, c10);
                c11 = Wmma::mma(a_curr1, b_curr1, c11);

                if constexpr (KSteps > 1) {
                    if (step + 1 < KSteps) {
                        a_curr0 = a_next0;
                        a_curr1 = a_next1;
                        b_curr0 = b_next0;
                        b_curr1 = b_next1;
                    }
                }
            }
        }

        if constexpr (ProducerOverlap && !FullAsmCluster) {
            if (tile + 1 < tiles) {
                stage_tile(stage ^ 1, tile + 1);
            }
        }

        uint32_t thread_xored = 0;
        accumulate_fragment(c00, running_sums[0][0], local_m_base0, local_n_base0, thread_xored);
        accumulate_fragment(c01, running_sums[0][1], local_m_base0, local_n_base1, thread_xored);
        accumulate_fragment(c10, running_sums[1][0], local_m_base1, local_n_base0, thread_xored);
        accumulate_fragment(c11, running_sums[1][1], local_m_base1, local_n_base1, thread_xored);

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}

template <
    int Rank,
    int BStride,
    bool BMajorSwizzle = false,
    int BMajorStripeY = 0,
    bool BHalfPad16 = false,
    bool BStepMajor16 = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq256x128x64_rdna3b_reg2_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    using Wmma = tnn::hip::iris::gemm::DefaultBackend::WmmaI8;

    constexpr int MicroATileRows = kJackpotWmmaTileRowsA;       // expected 64
    constexpr int LogicalATileRows = 256;
    constexpr int MPanelCount = LogicalATileRows / MicroATileRows;

    constexpr int BTileRows = kJackpotWmma64TileRowsB;          // expected 128
    constexpr int Threads = 128;
    constexpr int KSteps = Rank / Wmma::k;
    constexpr int StageCount = 1;

    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    static_assert(Rank == 64, "this variant is explicitly 256x128x64");
    static_assert(MicroATileRows == 64, "this variant expects the original 64-row micro M tile");
    static_assert(LogicalATileRows == 4 * MicroATileRows, "logical M tile must be four 64-row panels");
    static_assert((Rank % Wmma::k) == 0, "rank must be a multiple of WMMA K");

    // This variant keeps the existing WMMA scaffold and simply runs four
    // 64x128x64 panels per logical CTA.
    //
    // FullAsmCluster is intentionally not present here because the original
    // path asserts KSteps == 2, while Rank == 64 normally means KSteps == 4.
    //
    // BRecordLayout is intentionally not present here because the original
    // record LDS allocation is hard-sized for the old K-step contract.
    //
    // ProducerOverlap is intentionally not present here to avoid changing the
    // double-buffering contract while proving the logical 256x128x64 shape.

    __shared__ signed char sh_a_storage[StageCount * MicroATileRows * Rank];
    __shared__ signed char sh_bt_storage[StageCount * BStride * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);

    int virtual_block_x = 0;
    int virtual_block_y = 0;
    if constexpr (BMajorStripeY > 0) {
        pearl_compute_virtual_block_coords_bmajor_striped<BMajorStripeY>(
            virtual_block_x,
            virtual_block_y);
    } else {
        pearl_compute_virtual_block_coords<BMajorSwizzle>(
            virtual_block_x,
            virtual_block_y);
    }

    // Logical CTA coordinates: 256 rows by 128 cols.
    const int logical_tile_u_base = virtual_block_y * LogicalATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;

    const int partial_block =
        virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job =
        static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset =
        static_cast<int>(blockIdx.z) * partial_blocks_per_job;

    const int tiles = k / Rank;

    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "sq256x128x64_rdna3b_reg2 expects 4 waves");

    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses =
        MicroATileRows / (WaveRowsPerSubpass * Wmma::m);

    static_assert(MSubpasses == 2, "micro tile still expects two M subpasses");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;

    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;

    const int local_m_base0 = wave_m * Wmma::m;
    const int local_m_base1 = (wave_m + WaveRowsPerSubpass) * Wmma::m;

    // Separate running sums per 64-row M panel.
    int32_t running_sums[MPanelCount][MSubpasses][2][Wmma::c_elems];

    #pragma unroll
    for (int panel = 0; panel < MPanelCount; ++panel) {
        #pragma unroll
        for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
            #pragma unroll
            for (int group = 0; group < 2; ++group) {
                #pragma unroll
                for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                    running_sums[panel][m_pass][group][elem] = 0;
                }
            }
        }
    }

    auto accumulate_fragment =
        [&](typename Wmma::c_fragment& frag,
            int32_t* accum,
            int panel_u_base,
            int local_m_base,
            int local_n_base,
            uint32_t& thread_xored) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                const int local_u =
                    local_m_base + Wmma::c_row(lane, elem);
                const int local_v =
                    local_n_base + Wmma::c_col(lane);

                const int global_u = panel_u_base + local_u;
                const int global_v = tile_v_base + local_v;

                if (global_u < h && global_v < w) {
                    accum[elem] += frag[elem];
                    thread_xored ^= static_cast<uint32_t>(accum[elem]);
                }
            }
        };

    auto stage_a_ptr = [&](int stage) {
        return sh_a_storage + stage * (MicroATileRows * Rank);
    };

    auto stage_b_ptr = [&](int stage) {
        return sh_bt_storage + stage * (BStride * BTileRows);
    };

    auto make_sh_a = [&](int stage) {
        return iris::hip::make_lds_tile<signed char, MicroATileRows, Rank>(
            stage_a_ptr(stage));
    };

    auto make_sh_bt0 = [&](int stage) {
        signed char* const stage_ptr = stage_b_ptr(stage);

        if constexpr (BStepMajor16) {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_stepmajor_tile<
                Rank,
                kJackpotWmmaTileRowsB,
                16>(stage_ptr, 0);
        } else if constexpr (BHalfPad16) {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_halfpad_tile<
                Rank,
                kJackpotWmmaTileRowsB,
                BStride,
                16,
                16>(stage_ptr, 0);
        } else {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_crosswise_tile<
                Rank,
                kJackpotWmmaTileRowsB,
                BStride>(stage_ptr, 0);
        }
    };

    auto make_sh_bt1 = [&](int stage) {
        signed char* const stage_ptr = stage_b_ptr(stage);

        if constexpr (BStepMajor16) {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_stepmajor_tile<
                Rank,
                kJackpotWmmaTileRowsB,
                16>(stage_ptr, 1);
        } else if constexpr (BHalfPad16) {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_halfpad_tile<
                Rank,
                kJackpotWmmaTileRowsB,
                BStride,
                16,
                16>(stage_ptr, 1);
        } else {
            return tnn::hip::iris::gemm::layout::make_rdna3_b_crosswise_tile<
                Rank,
                kJackpotWmmaTileRowsB,
                BStride>(stage_ptr, 1);
        }
    };

    auto stage_tile = [&](int stage, int tile_idx, int panel_u_base) {
        const int base_col = tile_idx * Rank;

        signed char* const sh_a_stage = stage_a_ptr(stage);
        signed char* const sh_b_stage = stage_b_ptr(stage);

        // Load the 64x64 A micro-panel for this panel.
        for (int idx16 = tid;
             idx16 < (MicroATileRows * Rank) / 16;
             idx16 += Threads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / Rank;
            const int col = scalar_idx - row * Rank;

            const int u = panel_u_base + row;
            const int k_col = base_col + col;

            const PearlVec128 packed = pearl_load_vec128_or_zero(
                a_combined + static_cast<size_t>(u) * k + k_col,
                u < h && k_col + 15 < k);

            pearl_store_vec128_row_major_or_scalar(
                sh_a_stage,
                Rank,
                row,
                col,
                packed);
        }

        // Load the full 128x64 B tile.
        //
        // This reloads B for each of the four M panels. That is intentional for
        // the minimal/no-scaffold-rewrite version. Once this validates, B can be
        // hoisted outside the M-panel loop as an optimization.
        for (int idx16 = tid;
             idx16 < (BTileRows * Rank) / 16;
             idx16 += Threads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / Rank;
            const int col = scalar_idx - row * Rank;

            const int v = tile_v_base + row;
            const int k_col = base_col + col;

            const PearlVec128 packed = pearl_load_vec128_or_zero(
                b_combined + static_cast<size_t>(v) * k + k_col,
                v < w && k_col + 15 < k);

            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;

            if constexpr (BStepMajor16) {
                pearl_store_vec128_col_major_stepmajor16_or_scalar(
                    sh_b_stage + group * (Rank * kJackpotWmmaTileRowsB),
                    kJackpotWmmaTileRowsB,
                    local_row,
                    col,
                    packed);
            } else if constexpr (BHalfPad16) {
                pearl_store_vec128_col_major_halfpad16_or_scalar(
                    sh_b_stage + group * (BStride * kJackpotWmmaTileRowsB),
                    BStride,
                    local_row,
                    col,
                    packed);
            } else {
                pearl_store_vec128_col_major_or_scalar(
                    sh_b_stage + group * (BStride * kJackpotWmmaTileRowsB),
                    BStride,
                    local_row,
                    col,
                    packed);
            }
        }
    };

    for (int tile = 0; tile < tiles; ++tile) {
        uint32_t thread_xored_logical = 0;

        #pragma unroll
        for (int panel = 0; panel < MPanelCount; ++panel) {
            const int panel_u_base =
                logical_tile_u_base + panel * MicroATileRows;

            constexpr int stage = 0;

            stage_tile(stage, tile, panel_u_base);
            iris::hip::block_sync();

            auto sh_a = make_sh_a(stage);
            auto sh_bt0 = make_sh_bt0(stage);
            auto sh_bt1 = make_sh_bt1(stage);

            typename Wmma::c_fragment c00 = Wmma::zero_c();
            typename Wmma::c_fragment c01 = Wmma::zero_c();
            typename Wmma::c_fragment c10 = Wmma::zero_c();
            typename Wmma::c_fragment c11 = Wmma::zero_c();

            typename Wmma::ab_fragment a_curr0 =
                iris::hip::wmma_load_ab_row<Wmma>(
                    sh_a,
                    local_m_base0 + lane16,
                    k_pack_base);

            typename Wmma::ab_fragment a_curr1 =
                iris::hip::wmma_load_ab_row<Wmma>(
                    sh_a,
                    local_m_base1 + lane16,
                    k_pack_base);

            typename Wmma::ab_fragment b_curr0 =
                iris::hip::wmma_load_ab_col<Wmma>(
                    sh_bt0,
                    k_pack_base,
                    local_n_base0 + lane16);

            typename Wmma::ab_fragment b_curr1 =
                iris::hip::wmma_load_ab_col<Wmma>(
                    sh_bt1,
                    k_pack_base,
                    local_n_base0 + lane16);

            #pragma unroll
            for (int step = 0; step < KSteps; ++step) {
                typename Wmma::ab_fragment a_next0 = Wmma::zero_ab();
                typename Wmma::ab_fragment a_next1 = Wmma::zero_ab();
                typename Wmma::ab_fragment b_next0 = Wmma::zero_ab();
                typename Wmma::ab_fragment b_next1 = Wmma::zero_ab();

                if constexpr (KSteps > 1) {
                    if (step + 1 < KSteps) {
                        const int k_half_next = (step + 1) * Wmma::k;

                        a_next0 = iris::hip::wmma_load_ab_row<Wmma>(
                            sh_a,
                            local_m_base0 + lane16,
                            k_half_next + k_pack_base);

                        a_next1 = iris::hip::wmma_load_ab_row<Wmma>(
                            sh_a,
                            local_m_base1 + lane16,
                            k_half_next + k_pack_base);

                        b_next0 = iris::hip::wmma_load_ab_col<Wmma>(
                            sh_bt0,
                            k_half_next + k_pack_base,
                            local_n_base0 + lane16);

                        b_next1 = iris::hip::wmma_load_ab_col<Wmma>(
                            sh_bt1,
                            k_half_next + k_pack_base,
                            local_n_base0 + lane16);
                    }
                }

                iris::hip::wait_lgkmcnt0();

                c00 = Wmma::mma(a_curr0, b_curr0, c00);
                c01 = Wmma::mma(a_curr0, b_curr1, c01);
                c10 = Wmma::mma(a_curr1, b_curr0, c10);
                c11 = Wmma::mma(a_curr1, b_curr1, c11);

                if constexpr (KSteps > 1) {
                    if (step + 1 < KSteps) {
                        a_curr0 = a_next0;
                        a_curr1 = a_next1;
                        b_curr0 = b_next0;
                        b_curr1 = b_next1;
                    }
                }
            }

            uint32_t thread_xored_panel = 0;

            accumulate_fragment(
                c00,
                running_sums[panel][0][0],
                panel_u_base,
                local_m_base0,
                local_n_base0,
                thread_xored_panel);

            accumulate_fragment(
                c01,
                running_sums[panel][0][1],
                panel_u_base,
                local_m_base0,
                local_n_base1,
                thread_xored_panel);

            accumulate_fragment(
                c10,
                running_sums[panel][1][0],
                panel_u_base,
                local_m_base1,
                local_n_base0,
                thread_xored_panel);

            accumulate_fragment(
                c11,
                running_sums[panel][1][1],
                panel_u_base,
                local_m_base1,
                local_n_base1,
                thread_xored_panel);

            thread_xored_logical ^= thread_xored_panel;

            iris::hip::block_sync();
        }

        sh_xor[tid] = thread_xored_logical;

        const uint32_t xored =
            pearl_wave_publish_block_reduce_xor_u32<Threads, true>(
                thread_xored_logical,
                sh_xor);

        if (tid == 0) {
            const unsigned int slot =
                static_cast<unsigned int>(
                    (partial_job_offset + partial_block) * tiles + tile);

            partial[slot] = xored;

            if (owner_first != nullptr) {
                const unsigned int writer =
                    static_cast<unsigned int>(partial_job_offset + partial_block);

                const unsigned int prev =
                    atomicCAS(owner_first + slot, 0xffffffffu, writer);

                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }

                if (owner_collision != nullptr &&
                    prev != 0xffffffffu &&
                    prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }

        iris::hip::block_sync();
    }
}

template <int Rank>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_rdna3b_reg2_exact_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    using Wmma = tnn::hip::iris::gemm::DefaultBackend::WmmaI8;
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma64TileRowsB;
    constexpr int Threads = 128;
    constexpr int KSteps = Rank / Wmma::k;
    constexpr int ExactRecordBytes = 16;
    constexpr int ExactRecordsPerThread = 2 * KSteps;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    static_assert(KSteps == 2, "exact reg2 path currently expects rank 32 / ksteps 2");

    __shared__ __attribute__((aligned(16))) signed char sh_a_exact[Threads * ExactRecordsPerThread * ExactRecordBytes];
    __shared__ __attribute__((aligned(16))) signed char sh_b_exact[Threads * ExactRecordsPerThread * ExactRecordBytes];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<true>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / Rank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(WaveCount == 4);
    static_assert(MSubpasses == 2);

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;
    const int local_m_base0 = wave_m * Wmma::m;
    const int local_m_base1 = (wave_m + WaveRowsPerSubpass) * Wmma::m;

    auto exact_offset = [&](int slot) {
        return (tid * ExactRecordsPerThread + slot) * ExactRecordBytes;
    };

    int32_t running_sums[MSubpasses][2][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto accumulate_fragment = [&](typename Wmma::c_fragment& frag, int32_t* accum, int local_m_base, int local_n_base, uint32_t& thread_xored) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                accum[elem] += frag[elem];
                thread_xored ^= static_cast<uint32_t>(accum[elem]);
            }
        }
    };

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * Rank;

        #pragma unroll
        for (int step = 0; step < KSteps; ++step) {
            const int col_base = base_col + step * Wmma::k + k_pack_base;
            const int u0 = tile_u_base + local_m_base0 + lane16;
            const int u1 = tile_u_base + local_m_base1 + lane16;
            const int v0 = tile_v_base + local_n_base0 + lane16;
            const int v1 = tile_v_base + local_n_base1 + lane16;
            const int valid_cols = pearl_valid_record_cols(col_base, k);

            pearl_store_dup8_record_from_global(
                sh_a_exact + exact_offset(step * 2 + 0),
                a_combined + static_cast<size_t>(u0) * k + col_base,
                u0 < h,
                valid_cols);
            pearl_store_dup8_record_from_global(
                sh_a_exact + exact_offset(step * 2 + 1),
                a_combined + static_cast<size_t>(u1) * k + col_base,
                u1 < h,
                valid_cols);
            pearl_store_dup8_record_from_global(
                sh_b_exact + exact_offset(step * 2 + 0),
                b_combined + static_cast<size_t>(v0) * k + col_base,
                v0 < w,
                valid_cols);
            pearl_store_dup8_record_from_global(
                sh_b_exact + exact_offset(step * 2 + 1),
                b_combined + static_cast<size_t>(v1) * k + col_base,
                v1 < w,
                valid_cols);
        }

        iris::hip::block_sync();

        typename Wmma::c_fragment c00 = Wmma::zero_c();
        typename Wmma::c_fragment c01 = Wmma::zero_c();
        typename Wmma::c_fragment c10 = Wmma::zero_c();
        typename Wmma::c_fragment c11 = Wmma::zero_c();

        const auto a0_storage_s0 = pearl_rdna3b_take_low_storage<Wmma>(
            pearl_rdna3b_ds_load_b128_asm(sh_a_exact + exact_offset(0)));
        const auto a1_storage_s0 = pearl_rdna3b_take_low_storage<Wmma>(
            pearl_rdna3b_ds_load_b128_asm(sh_a_exact + exact_offset(1)));
        const auto b0_storage_s0 = pearl_rdna3b_take_low_storage<Wmma>(
            pearl_rdna3b_ds_load_b128_asm(sh_b_exact + exact_offset(0)));
        const auto b1_storage_s0 = pearl_rdna3b_take_low_storage<Wmma>(
            pearl_rdna3b_ds_load_b128_asm(sh_b_exact + exact_offset(1)));
        const auto a0_storage_s1 = pearl_rdna3b_take_low_storage<Wmma>(
            pearl_rdna3b_ds_load_b128_asm(sh_a_exact + exact_offset(2)));
        const auto a1_storage_s1 = pearl_rdna3b_take_low_storage<Wmma>(
            pearl_rdna3b_ds_load_b128_asm(sh_a_exact + exact_offset(3)));
        const auto b0_storage_s1 = pearl_rdna3b_take_low_storage<Wmma>(
            pearl_rdna3b_ds_load_b128_asm(sh_b_exact + exact_offset(2)));
        const auto b1_storage_s1 = pearl_rdna3b_take_low_storage<Wmma>(
            pearl_rdna3b_ds_load_b128_asm(sh_b_exact + exact_offset(3)));

        iris::hip::wait_lgkmcnt0();

        const auto a0s0 = pearl_rdna3b_prepare_ab_asm<Wmma>(a0_storage_s0);
        const auto a1s0 = pearl_rdna3b_prepare_ab_asm<Wmma>(a1_storage_s0);
        const auto b0s0 = pearl_rdna3b_prepare_ab_asm<Wmma>(b0_storage_s0);
        const auto b1s0 = pearl_rdna3b_prepare_ab_asm<Wmma>(b1_storage_s0);
        const auto a0s1 = pearl_rdna3b_prepare_ab_asm<Wmma>(a0_storage_s1);
        const auto a1s1 = pearl_rdna3b_prepare_ab_asm<Wmma>(a1_storage_s1);
        const auto b0s1 = pearl_rdna3b_prepare_ab_asm<Wmma>(b0_storage_s1);
        const auto b1s1 = pearl_rdna3b_prepare_ab_asm<Wmma>(b1_storage_s1);

        c00 = pearl_rdna3b_wmma_asm_step<Wmma>(a0s0, b0s0, c00);
        c01 = pearl_rdna3b_wmma_asm_step<Wmma>(a0s0, b1s0, c01);
        c10 = pearl_rdna3b_wmma_asm_step<Wmma>(a1s0, b0s0, c10);
        c11 = pearl_rdna3b_wmma_asm_step<Wmma>(a1s0, b1s0, c11);
        c00 = pearl_rdna3b_wmma_asm_step<Wmma>(a0s1, b0s1, c00);
        c01 = pearl_rdna3b_wmma_asm_step<Wmma>(a0s1, b1s1, c01);
        c10 = pearl_rdna3b_wmma_asm_step<Wmma>(a1s1, b0s1, c10);
        c11 = pearl_rdna3b_wmma_asm_step<Wmma>(a1s1, b1s1, c11);

        uint32_t thread_xored = 0;
        accumulate_fragment(c00, running_sums[0][0], local_m_base0, local_n_base0, thread_xored);
        accumulate_fragment(c01, running_sums[0][1], local_m_base0, local_n_base1, thread_xored);
        accumulate_fragment(c10, running_sums[1][0], local_m_base1, local_n_base0, thread_xored);
        accumulate_fragment(c11, running_sums[1][1], local_m_base1, local_n_base1, thread_xored);

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_sq64x64_rdna3b_pc11_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    using Backend = tnn::hip::iris::gemm::DefaultBackend;
    using Wmma = Backend::WmmaI8;
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma64TileRowsB;
    constexpr int BStride = Backend::kBStride;
    constexpr int Threads = 128;
    constexpr int Stages = 2;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[Stages][ATileRows * kRank];
    __shared__ signed char sh_bt_storage[Stages][BStride * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    constexpr int ConsumerWaves = 2;
    constexpr int ProducerWaves = WaveCount - ConsumerWaves;
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = ConsumerWaves / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(WaveCount == 4, "iris_sq64x64_rdna3b_pc11 expects 4 waves");
    static_assert(ConsumerWaves == 2 && ProducerWaves == 2, "iris_sq64x64_rdna3b_pc11 expects 2 consumer and 2 producer waves");
    static_assert(MSubpasses == 4, "iris_sq64x64_rdna3b_pc11 expects four M subpasses");

    const bool consumer_wave = wave < ConsumerWaves;
    const bool producer_wave = !consumer_wave;
    const int consumer_wave_id = wave;
    const int wave_m = consumer_wave_id / WaveCols;
    const int wave_n = consumer_wave_id % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;

    int32_t running_sums[MSubpasses][2][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto load_stage_by_producers = [&](int stage, int tile_idx) {
        if (!producer_wave) {
            return;
        }
        const int base_col = tile_idx * kRank;
        const int producer_tid = tid - ConsumerWaves * 32;
        constexpr int ProducerThreads = ProducerWaves * 32;

        for (int idx16 = producer_tid; idx16 < (ATileRows * kRank) / 16; idx16 += ProducerThreads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            const PearlVec128 packed = pearl_load_vec128_or_zero(
                a_combined + static_cast<size_t>(u) * k + k_col,
                u < h && k_col + 15 < k);
            pearl_store_vec128_row_major_or_scalar(sh_a_storage[stage], kRank, row, col, packed);
        }

        for (int idx16 = producer_tid; idx16 < (BTileRows * kRank) / 16; idx16 += ProducerThreads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            const PearlVec128 packed = pearl_load_vec128_or_zero(
                b_combined + static_cast<size_t>(v) * k + k_col,
                v < w && k_col + 15 < k);
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            pearl_store_vec128_col_major_or_scalar(
                sh_bt_storage[stage] + group * (BStride * kJackpotWmmaTileRowsB),
                BStride,
                local_row,
                col,
                packed);
        }
    };

    auto compute_tile = [&](int stage, uint32_t& thread_xored) {
        if (!consumer_wave) {
            return;
        }
        auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage[stage]);
        auto sh_bt0 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage[stage], 0);
        auto sh_bt1 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage[stage], 1);

        typename Wmma::c_fragment c[MSubpasses][2];
        #pragma unroll
        for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
            #pragma unroll
            for (int group = 0; group < 2; ++group) {
                c[m_pass][group] = Wmma::zero_c();
            }
        }

        #pragma unroll
        for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
            typename Wmma::ab_fragment a_frag[MSubpasses];
            #pragma unroll
            for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                const int local_m_base = (wave_m + m_pass * WaveRowsPerSubpass) * Wmma::m;
                a_frag[m_pass] = iris::hip::wmma_load_ab_row<Wmma>(
                    sh_a,
                    local_m_base + lane16,
                    k_half + k_pack_base);
            }

            const auto b_frag0 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt0, k_half + k_pack_base, local_n_base0 + lane16);
            const auto b_frag1 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt1, k_half + k_pack_base, local_n_base0 + lane16);

            #pragma unroll
            for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                c[m_pass][0] = Wmma::mma(a_frag[m_pass], b_frag0, c[m_pass][0]);
                c[m_pass][1] = Wmma::mma(a_frag[m_pass], b_frag1, c[m_pass][1]);
            }
        }

        #pragma unroll
        for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
            const int local_m_base = (wave_m + m_pass * WaveRowsPerSubpass) * Wmma::m;
            #pragma unroll
            for (int group = 0; group < 2; ++group) {
                const int local_n_base = local_n_base0 + group * kJackpotWmmaTileRowsB;
                #pragma unroll
                for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                    const int local_u = local_m_base + Wmma::c_row(lane, elem);
                    const int local_v = local_n_base + Wmma::c_col(lane);
                    const int global_u = tile_u_base + local_u;
                    const int global_v = tile_v_base + local_v;
                    if (global_u < h && global_v < w) {
                        running_sums[m_pass][group][elem] += c[m_pass][group][elem];
                        thread_xored ^= static_cast<uint32_t>(running_sums[m_pass][group][elem]);
                    }
                }
            }
        }
    };

    load_stage_by_producers(0, 0);
    iris::hip::block_sync();

    for (int tile = 0; tile < tiles; ++tile) {
        const int curr_stage = tile & 1;
        const int next_stage = curr_stage ^ 1;
        uint32_t thread_xored = 0;

        if (tile + 1 < tiles) {
            load_stage_by_producers(next_stage, tile + 1);
        }
        compute_tile(curr_stage, thread_xored);
        iris::hip::block_sync();

        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }

    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_64x128_serial_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma128TileRowsB;
    constexpr int NGroups = 4;
    constexpr int Threads = 128;
    constexpr int ChunkGroups = 4;
    using Wmma = iris::hip::WmmaI8I8I32_16x16x16<true, true, false>;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[ATileRows * kRank];
    __shared__ signed char sh_bt_storage[kRank * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_64x128_serial expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_64x128_serial expects two M subpasses");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;

    auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage);

    int32_t running_sums[MSubpasses][NGroups][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < NGroups; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx8 = tid; idx8 < (ATileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (u < h && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(a_combined + static_cast<size_t>(u) * k + k_col);
            }
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_a.store(row, col + lane_byte, pearl_i8_lane(packed, lane_byte));
            }
        }

        for (int idx8 = tid; idx8 < (BTileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (v < w && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(b_combined + static_cast<size_t>(v) * k + k_col);
            }
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            auto sh_bt = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, group);
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_bt.store(col + lane_byte, local_row, pearl_i8_lane(packed, lane_byte));
            }
        }

        iris::hip::block_sync();

        uint32_t thread_xored = 0;
        #pragma unroll
        for (int group_base = 0; group_base < NGroups; group_base += ChunkGroups) {
            typename Wmma::c_fragment c[MSubpasses][ChunkGroups];
            #pragma unroll
            for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    c[m_pass][group] = Wmma::zero_c();
                }
            }

            #pragma unroll
            for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
                typename Wmma::ab_fragment a_frag[MSubpasses];
                #pragma unroll
                for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                    const int local_m_base = (wave_m + m_pass * WaveRowsPerSubpass) * Wmma::m;
                    a_frag[m_pass] = iris::hip::wmma_load_ab_row<Wmma>(
                        sh_a,
                        local_m_base + lane16,
                        k_half + k_pack_base);
                }
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    const int global_group = group_base + group;
                    const auto b_frag = iris::hip::wmma_load_ab_col<Wmma>(
                        pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, global_group),
                        k_half + k_pack_base,
                        local_n_base0 + lane16);
                    #pragma unroll
                    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                        c[m_pass][group] = Wmma::mma(a_frag[m_pass], b_frag, c[m_pass][group]);
                    }
                }
            }

            #pragma unroll
            for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
                const int local_m_base = (wave_m + m_pass * WaveRowsPerSubpass) * Wmma::m;
                #pragma unroll
                for (int group = 0; group < ChunkGroups; ++group) {
                    const int global_group = group_base + group;
                    #pragma unroll
                    for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                        const int local_u = local_m_base + Wmma::c_row(lane, elem);
                        const int local_v = local_n_base0 + global_group * kJackpotWmmaTileRowsB + Wmma::c_col(lane);
                        const int global_u = tile_u_base + local_u;
                        const int global_v = tile_v_base + local_v;
                        if (global_u < h && global_v < w) {
                            running_sums[m_pass][global_group][elem] += c[m_pass][group][elem];
                            thread_xored ^= static_cast<uint32_t>(running_sums[m_pass][global_group][elem]);
                        }
                    }
                }
            }
        }

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_64x128_dense_serial_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma128TileRowsB;
    constexpr int Threads = 128;
    using Wmma = iris::hip::WmmaI8I8I32_16x16x16<true, true, false>;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[ATileRows * kRank];
    __shared__ signed char sh_bt_storage[kRank * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_64x128_dense_serial expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_64x128_dense_serial expects two M subpasses");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;
    const int local_n_base2 = local_n_base1 + kJackpotWmmaTileRowsB;
    const int local_n_base3 = local_n_base2 + kJackpotWmmaTileRowsB;
    const int local_m_base0 = wave_m * Wmma::m;
    const int local_m_base1 = (wave_m + WaveRowsPerSubpass) * Wmma::m;

    auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage);
    auto sh_bt0 = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, 0);
    auto sh_bt1 = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, 1);
    auto sh_bt2 = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, 2);
    auto sh_bt3 = pearl_make_ungrouped_wmma_b_subtile<BTileRows>(sh_bt_storage, 3);

    int32_t running_sums[MSubpasses][4][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto accumulate_fragment = [&](typename Wmma::c_fragment& frag, int32_t* accum, int local_m_base, int local_n_base, uint32_t& thread_xored) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                accum[elem] += frag[elem];
                thread_xored ^= static_cast<uint32_t>(accum[elem]);
            }
        }
    };

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx8 = tid; idx8 < (ATileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (u < h && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(a_combined + static_cast<size_t>(u) * k + k_col);
            }
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_a.store(row, col + lane_byte, pearl_i8_lane(packed, lane_byte));
            }
        }

        for (int idx8 = tid; idx8 < (BTileRows * kRank) / 8; idx8 += Threads) {
            const int scalar_idx = idx8 * 8;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            uint64_t packed = 0;
            if (v < w && k_col + 7 < k) {
                packed = *reinterpret_cast<const uint64_t*>(b_combined + static_cast<size_t>(v) * k + k_col);
            }
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            auto sh_bt = group == 0 ? sh_bt0 : (group == 1 ? sh_bt1 : (group == 2 ? sh_bt2 : sh_bt3));
            #pragma unroll
            for (int lane_byte = 0; lane_byte < 8; ++lane_byte) {
                sh_bt.store(col + lane_byte, local_row, pearl_i8_lane(packed, lane_byte));
            }
        }

        iris::hip::block_sync();

        typename Wmma::c_fragment c00 = Wmma::zero_c();
        typename Wmma::c_fragment c01 = Wmma::zero_c();
        typename Wmma::c_fragment c02 = Wmma::zero_c();
        typename Wmma::c_fragment c03 = Wmma::zero_c();
        typename Wmma::c_fragment c10 = Wmma::zero_c();
        typename Wmma::c_fragment c11 = Wmma::zero_c();
        typename Wmma::c_fragment c12 = Wmma::zero_c();
        typename Wmma::c_fragment c13 = Wmma::zero_c();

        #pragma unroll
        for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
            const auto a_frag0 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base0 + lane16, k_half + k_pack_base);
            const auto a_frag1 = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base1 + lane16, k_half + k_pack_base);
            const auto b_frag0 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt0, k_half + k_pack_base, local_n_base0 + lane16);
            const auto b_frag1 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt1, k_half + k_pack_base, local_n_base0 + lane16);
            const auto b_frag2 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt2, k_half + k_pack_base, local_n_base0 + lane16);
            const auto b_frag3 = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt3, k_half + k_pack_base, local_n_base0 + lane16);

            c00 = Wmma::mma(a_frag0, b_frag0, c00);
            c01 = Wmma::mma(a_frag0, b_frag1, c01);
            c02 = Wmma::mma(a_frag0, b_frag2, c02);
            c03 = Wmma::mma(a_frag0, b_frag3, c03);
            c10 = Wmma::mma(a_frag1, b_frag0, c10);
            c11 = Wmma::mma(a_frag1, b_frag1, c11);
            c12 = Wmma::mma(a_frag1, b_frag2, c12);
            c13 = Wmma::mma(a_frag1, b_frag3, c13);
        }

        uint32_t thread_xored = 0;
        accumulate_fragment(c00, running_sums[0][0], local_m_base0, local_n_base0, thread_xored);
        accumulate_fragment(c01, running_sums[0][1], local_m_base0, local_n_base1, thread_xored);
        accumulate_fragment(c02, running_sums[0][2], local_m_base0, local_n_base2, thread_xored);
        accumulate_fragment(c03, running_sums[0][3], local_m_base0, local_n_base3, thread_xored);
        accumulate_fragment(c10, running_sums[1][0], local_m_base1, local_n_base0, thread_xored);
        accumulate_fragment(c11, running_sums[1][1], local_m_base1, local_n_base1, thread_xored);
        accumulate_fragment(c12, running_sums[1][2], local_m_base1, local_n_base2, thread_xored);
        accumulate_fragment(c13, running_sums[1][3], local_m_base1, local_n_base3, thread_xored);

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_64x128_rdna3b_reg2_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    using Backend = tnn::hip::iris::gemm::DefaultBackend;
    using Wmma = Backend::WmmaI8;
    constexpr int ATileRows = kJackpotWmmaTileRowsA;
    constexpr int BTileRows = kJackpotWmma128TileRowsB;
    constexpr int BStride = Backend::kBStride;
    constexpr int Threads = 128;
    constexpr int KSteps = kRank / Wmma::k;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[ATileRows * kRank];
    __shared__ signed char sh_bt_storage[BStride * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 4, "iris_64x128_rdna3b_reg2 expects 4 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(MSubpasses == 2, "iris_64x128_rdna3b_reg2 expects two M subpasses");
    static_assert(KSteps == 2, "iris_64x128_rdna3b_reg2 assumes two K steps");

    const int wave_m = wave / WaveCols;
    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;
    const int local_n_base2 = local_n_base1 + kJackpotWmmaTileRowsB;
    const int local_n_base3 = local_n_base2 + kJackpotWmmaTileRowsB;
    const int local_m_base0 = wave_m * Wmma::m;
    const int local_m_base1 = (wave_m + WaveRowsPerSubpass) * Wmma::m;

    auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage);
    auto sh_bt0 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 0);
    auto sh_bt1 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 1);
    auto sh_bt2 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 2);
    auto sh_bt3 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 3);

    int32_t running_sums[MSubpasses][4][Wmma::c_elems];
    #pragma unroll
    for (int m_pass = 0; m_pass < MSubpasses; ++m_pass) {
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            #pragma unroll
            for (int elem = 0; elem < Wmma::c_elems; ++elem) {
                running_sums[m_pass][group][elem] = 0;
            }
        }
    }

    auto accumulate_fragment = [&](typename Wmma::c_fragment& frag, int32_t* accum, int local_m_base, int local_n_base, uint32_t& thread_xored) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                accum[elem] += frag[elem];
                thread_xored ^= static_cast<uint32_t>(accum[elem]);
            }
        }
    };

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx16 = tid; idx16 < (ATileRows * kRank) / 16; idx16 += Threads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            const PearlVec128 packed = pearl_load_vec128_or_zero(
                a_combined + static_cast<size_t>(u) * k + k_col,
                u < h && k_col + 15 < k);
            pearl_store_vec128_row_major_or_scalar(sh_a_storage, kRank, row, col, packed);
        }

        for (int idx16 = tid; idx16 < (BTileRows * kRank) / 16; idx16 += Threads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            const PearlVec128 packed = pearl_load_vec128_or_zero(
                b_combined + static_cast<size_t>(v) * k + k_col,
                v < w && k_col + 15 < k);
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            pearl_store_vec128_col_major_or_scalar(
                sh_bt_storage + group * (BStride * kJackpotWmmaTileRowsB),
                BStride,
                local_row,
                col,
                packed);
        }

        iris::hip::block_sync();

        typename Wmma::c_fragment c00 = Wmma::zero_c();
        typename Wmma::c_fragment c01 = Wmma::zero_c();
        typename Wmma::c_fragment c02 = Wmma::zero_c();
        typename Wmma::c_fragment c03 = Wmma::zero_c();
        typename Wmma::c_fragment c10 = Wmma::zero_c();
        typename Wmma::c_fragment c11 = Wmma::zero_c();
        typename Wmma::c_fragment c12 = Wmma::zero_c();
        typename Wmma::c_fragment c13 = Wmma::zero_c();

        typename Wmma::ab_fragment a_frag0[KSteps];
        typename Wmma::ab_fragment a_frag1[KSteps];
        typename Wmma::ab_fragment b_frag0[KSteps];
        typename Wmma::ab_fragment b_frag1[KSteps];
        typename Wmma::ab_fragment b_frag2[KSteps];
        typename Wmma::ab_fragment b_frag3[KSteps];

        #pragma unroll
        for (int step = 0; step < KSteps; ++step) {
            const int k_half = step * Wmma::k;
            a_frag0[step] = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base0 + lane16, k_half + k_pack_base);
            a_frag1[step] = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base1 + lane16, k_half + k_pack_base);
            b_frag0[step] = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt0, k_half + k_pack_base, local_n_base0 + lane16);
            b_frag1[step] = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt1, k_half + k_pack_base, local_n_base0 + lane16);
            b_frag2[step] = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt2, k_half + k_pack_base, local_n_base0 + lane16);
            b_frag3[step] = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt3, k_half + k_pack_base, local_n_base0 + lane16);
        }

        iris::hip::wait_lgkmcnt0();

        #pragma unroll
        for (int step = 0; step < KSteps; ++step) {
            c00 = Wmma::mma(a_frag0[step], b_frag0[step], c00);
            c01 = Wmma::mma(a_frag0[step], b_frag1[step], c01);
            c02 = Wmma::mma(a_frag0[step], b_frag2[step], c02);
            c03 = Wmma::mma(a_frag0[step], b_frag3[step], c03);
            c10 = Wmma::mma(a_frag1[step], b_frag0[step], c10);
            c11 = Wmma::mma(a_frag1[step], b_frag1[step], c11);
            c12 = Wmma::mma(a_frag1[step], b_frag2[step], c12);
            c13 = Wmma::mma(a_frag1[step], b_frag3[step], c13);
        }

        uint32_t thread_xored = 0;
        accumulate_fragment(c00, running_sums[0][0], local_m_base0, local_n_base0, thread_xored);
        accumulate_fragment(c01, running_sums[0][1], local_m_base0, local_n_base1, thread_xored);
        accumulate_fragment(c02, running_sums[0][2], local_m_base0, local_n_base2, thread_xored);
        accumulate_fragment(c03, running_sums[0][3], local_m_base0, local_n_base3, thread_xored);
        accumulate_fragment(c10, running_sums[1][0], local_m_base1, local_n_base0, thread_xored);
        accumulate_fragment(c11, running_sums[1][1], local_m_base1, local_n_base1, thread_xored);
        accumulate_fragment(c12, running_sums[1][2], local_m_base1, local_n_base2, thread_xored);
        accumulate_fragment(c13, running_sums[1][3], local_m_base1, local_n_base3, thread_xored);

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}

template <bool BMajorSwizzle = false>
IRIS_DEVICE_INLINE void pearl_expand_jackpot_iris_16x64_rdna3b_reg2_impl(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__ b_combined,
    unsigned int* __restrict__ partial,
    unsigned int* __restrict__ owner_first,
    unsigned int* __restrict__ owner_last,
    unsigned int* __restrict__ owner_collision,
    int h,
    int w,
    int k)
{
    using Backend = tnn::hip::iris::gemm::DefaultBackend;
    using Wmma = Backend::WmmaI8;
    constexpr int ATileRows = 16;
    constexpr int BTileRows = 64;
    constexpr int BStride = Backend::kBStride;
    constexpr int Threads = 64;
    constexpr int KSteps = kRank / Wmma::k;
    using BlockLayout = iris::hip::BlockThreadLayout<Threads>;

    __shared__ signed char sh_a_storage[ATileRows * kRank];
    __shared__ signed char sh_bt_storage[BStride * BTileRows];
    __shared__ unsigned int sh_xor[Threads];

    const int tid = static_cast<int>(threadIdx.x);
    int virtual_block_x = 0;
    int virtual_block_y = 0;
    pearl_compute_virtual_block_coords<BMajorSwizzle>(virtual_block_x, virtual_block_y);
    const int tile_u_base = virtual_block_y * ATileRows;
    const int tile_v_base = virtual_block_x * BTileRows;
    const int partial_block = virtual_block_y * static_cast<int>(gridDim.x) + virtual_block_x;
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int k_pack_base = Wmma::storage_k_base(lane);

    constexpr int WaveCount = Threads / 32;
    static_assert(WaveCount == 2, "iris_16x64_rdna3b_reg2 expects 2 waves");
    constexpr int WaveCols = 2;
    constexpr int WaveRowsPerSubpass = WaveCount / WaveCols;
    constexpr int MSubpasses = ATileRows / (WaveRowsPerSubpass * Wmma::m);
    static_assert(WaveRowsPerSubpass == 1, "iris_16x64_rdna3b_reg2 expects one wave row");
    static_assert(MSubpasses == 1, "iris_16x64_rdna3b_reg2 expects one M subpass");
    static_assert(KSteps == 2, "iris_16x64_rdna3b_reg2 assumes two K steps");

    const int wave_n = wave % WaveCols;
    const int local_n_base0 = wave_n * Wmma::n;
    const int local_n_base1 = local_n_base0 + kJackpotWmmaTileRowsB;
    const int local_m_base0 = 0;

    auto sh_a = iris::hip::make_lds_tile<signed char, ATileRows, kRank>(sh_a_storage);
    auto sh_bt0 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 0);
    auto sh_bt1 = tnn::hip::iris::gemm::default_arch::make_b_tile<kRank, kJackpotWmmaTileRowsB>(sh_bt_storage, 1);

    int32_t running_sums[2][Wmma::c_elems];
    #pragma unroll
    for (int group = 0; group < 2; ++group) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            running_sums[group][elem] = 0;
        }
    }

    auto accumulate_fragment = [&](typename Wmma::c_fragment& frag, int32_t* accum, int local_n_base, uint32_t& thread_xored) {
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base0 + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                accum[elem] += frag[elem];
                thread_xored ^= static_cast<uint32_t>(accum[elem]);
            }
        }
    };

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx16 = tid; idx16 < (ATileRows * kRank) / 16; idx16 += Threads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            const PearlVec128 packed = pearl_load_vec128_or_zero(
                a_combined + static_cast<size_t>(u) * k + k_col,
                u < h && k_col + 15 < k);
            pearl_store_vec128_row_major_or_scalar(sh_a_storage, kRank, row, col, packed);
        }

        for (int idx16 = tid; idx16 < (BTileRows * kRank) / 16; idx16 += Threads) {
            const int scalar_idx = idx16 * 16;
            const int row = scalar_idx / kRank;
            const int col = scalar_idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            const PearlVec128 packed = pearl_load_vec128_or_zero(
                b_combined + static_cast<size_t>(v) * k + k_col,
                v < w && k_col + 15 < k);
            const int group = row / kJackpotWmmaTileRowsB;
            const int local_row = row % kJackpotWmmaTileRowsB;
            pearl_store_vec128_col_major_or_scalar(
                sh_bt_storage + group * (BStride * kJackpotWmmaTileRowsB),
                BStride,
                local_row,
                col,
                packed);
        }

        iris::hip::block_sync();

        typename Wmma::c_fragment c00 = Wmma::zero_c();
        typename Wmma::c_fragment c01 = Wmma::zero_c();

        typename Wmma::ab_fragment a_frag0[KSteps];
        typename Wmma::ab_fragment b_frag0[KSteps];
        typename Wmma::ab_fragment b_frag1[KSteps];

        #pragma unroll
        for (int step = 0; step < KSteps; ++step) {
            const int k_half = step * Wmma::k;
            a_frag0[step] = iris::hip::wmma_load_ab_row<Wmma>(
                sh_a, local_m_base0 + lane16, k_half + k_pack_base);
            b_frag0[step] = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt0, k_half + k_pack_base, local_n_base0 + lane16);
            b_frag1[step] = iris::hip::wmma_load_ab_col<Wmma>(
                sh_bt1, k_half + k_pack_base, local_n_base0 + lane16);
        }

        iris::hip::wait_lgkmcnt0();

        #pragma unroll
        for (int step = 0; step < KSteps; ++step) {
            c00 = Wmma::mma(a_frag0[step], b_frag0[step], c00);
            c01 = Wmma::mma(a_frag0[step], b_frag1[step], c01);
        }

        uint32_t thread_xored = 0;
        accumulate_fragment(c00, running_sums[0], local_n_base0, thread_xored);
        accumulate_fragment(c01, running_sums[1], local_n_base1, thread_xored);

        iris::hip::block_sync();
        sh_xor[tid] = thread_xored;
        const uint32_t xored = pearl_wave_publish_block_reduce_xor_u32<Threads, true>(thread_xored, sh_xor);

        if (tid == 0) {
            const unsigned int slot = static_cast<unsigned int>((partial_job_offset + partial_block) * tiles + tile);
            partial[slot] = xored;
            if (owner_first != nullptr) {
                const unsigned int writer = static_cast<unsigned int>(partial_job_offset + partial_block);
                const unsigned int prev = atomicCAS(owner_first + slot, 0xffffffffu, writer);
                if (owner_last != nullptr) {
                    owner_last[slot] = writer;
                }
                if (owner_collision != nullptr && prev != 0xffffffffu && prev != writer) {
                    owner_collision[slot] = 1u;
                }
            }
        }
        iris::hip::block_sync();
    }
}


extern "C" __global__ void pearl_expand_jackpot_iris_sq64x64_rdna3b_reg2_swizzle_partial(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__,
    const signed char* __restrict__ b_combined,
    const signed char* __restrict__,
    unsigned int* __restrict__ partial,
    int h,
    int w,
    int k)
{
    pearl_expand_jackpot_iris_sq64x64_rdna3b_reg2_impl<kRank, tnn::hip::iris::gemm::DefaultBackend::kBStride, true>(
        a_combined, b_combined, partial, nullptr, nullptr, nullptr, h, w, k);
}

extern "C" __global__ void pearl_expand_jackpot_iris_sq64x64_rdna3b_reg2_bstride64_swizzle_partial(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__,
    const signed char* __restrict__ b_combined,
    const signed char* __restrict__,
    unsigned int* __restrict__ partial,
    int h,
    int w,
    int k)
{
    pearl_expand_jackpot_iris_sq64x64_rdna3b_reg2_impl<
        kRank,
        64,
        true>(
        a_combined, b_combined, partial, nullptr, nullptr, nullptr, h, w, k);
}

extern "C" __global__ void pearl_expand_jackpot_iris_16x64_rdna3b_reg2_swizzle_partial(
    const signed char* __restrict__ a_combined,
    const signed char* __restrict__,
    const signed char* __restrict__ b_combined,
    const signed char* __restrict__,
    unsigned int* __restrict__ partial,
    int h,
    int w,
    int k)
{
    pearl_expand_jackpot_iris_16x64_rdna3b_reg2_impl<true>(
        a_combined, b_combined, partial, nullptr, nullptr, nullptr, h, w, k);
}
