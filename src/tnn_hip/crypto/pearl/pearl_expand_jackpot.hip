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

// ============================================================================
// Iris-tiled Phase 1
//
// Grid: ceil(w / 32) x ceil(h / 16), Block: 256 threads.
// Each block owns a 16x32 tile of (u, v) pairs. The current rank tile from A
// and B is staged into LDS once, then every thread computes two pair dots and
// participates in a block XOR reduction for the cumulative tile result.
// ============================================================================
extern "C" __global__ void pearl_expand_jackpot_tiled_partial(
    const signed char* __restrict__ s_a,
    const signed char* __restrict__ noise_a,
    const signed char* __restrict__ s_b,
    const signed char* __restrict__ noise_b,
    unsigned int* __restrict__ partial,
    int h,
    int w,
    int k)
{
    using BlockLayout = iris::hip::BlockThreadLayout<kNumThreads>;
    __shared__ short sh_a_storage[kJackpotTileRowsA * kRank];
    __shared__ short sh_bt_storage[kRank * kJackpotTileRowsB];
    __shared__ unsigned int sh_xor[kNumThreads];

    const int tid = static_cast<int>(threadIdx.x);
    const int tile_u_base = static_cast<int>(blockIdx.y) * kJackpotTileRowsA;
    const int tile_v_base = static_cast<int>(blockIdx.x) * kJackpotTileRowsB;
    const int partial_block = static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;

    auto a_view = iris::hip::make_global_row_major_tensor_view(s_a, h, k, k);
    auto an_view = iris::hip::make_global_row_major_tensor_view(noise_a, h, k, k);
    auto b_view = iris::hip::make_global_row_major_tensor_view(s_b, w, k, k);
    auto bn_view = iris::hip::make_global_row_major_tensor_view(noise_b, w, k, k);
    auto sh_a = iris::hip::make_lds_tile<short, kJackpotTileRowsA, kRank>(sh_a_storage);
    auto sh_bt = iris::hip::make_lds_tile<short, kRank, kJackpotTileRowsB>(sh_bt_storage);

    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int local_u0 = wave * 2;
    const int local_u1 = local_u0 + 1;
    const int local_v = lane;
    const int global_u0 = tile_u_base + local_u0;
    const int global_u1 = tile_u_base + local_u1;
    const int global_v = tile_v_base + local_v;
    const bool pair0_in_bounds = global_u0 < h && global_v < w;
    const bool pair1_in_bounds = global_u1 < h && global_v < w;

    int running_sum0 = 0;
    int running_sum1 = 0;
    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx = tid; idx < kJackpotTileRowsA * kRank; idx += kNumThreads) {
            const int row = idx / kRank;
            const int col = idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            short value = 0;
            if (u < h && k_col < k) {
                value = static_cast<short>(
                    static_cast<int>(a_view.load(u, k_col)) +
                    static_cast<int>(an_view.load(u, k_col)));
            }
            sh_a.store(row, col, value);
        }

        for (int idx = tid; idx < kJackpotTileRowsB * kRank; idx += kNumThreads) {
            const int row = idx / kRank;
            const int col = idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            short value = 0;
            if (v < w && k_col < k) {
                value = static_cast<short>(
                    static_cast<int>(b_view.load(v, k_col)) +
                    static_cast<int>(bn_view.load(v, k_col)));
            }
            sh_bt.store(col, row, value);
        }

        iris::hip::block_sync();

        int dot0 = 0;
        int dot1 = 0;
        if (pair0_in_bounds || pair1_in_bounds) {
            #pragma unroll
            for (int l = 0; l < kRank; ++l) {
                const int b = static_cast<int>(sh_bt.load(l, local_v));
                dot0 += static_cast<int>(sh_a.load(local_u0, l)) * b;
                dot1 += static_cast<int>(sh_a.load(local_u1, l)) * b;
            }
            if (pair0_in_bounds) {
                running_sum0 += dot0;
            }
            if (pair1_in_bounds) {
                running_sum1 += dot1;
            }
        }

        uint32_t thread_xored = 0;
        if (pair0_in_bounds) {
            thread_xored ^= static_cast<uint32_t>(running_sum0);
        }
        if (pair1_in_bounds) {
            thread_xored ^= static_cast<uint32_t>(running_sum1);
        }

        const uint32_t xored = iris::hip::block_reduce_xor_u32<kNumThreads>(
            thread_xored, sh_xor);

        if (tid == 0) {
            partial[(partial_job_offset + partial_block) * tiles + tile] = xored;
        }

        iris::hip::block_sync();
    }
}

extern "C" __global__ void pearl_expand_jackpot_wmma_partial(
    const signed char* __restrict__ s_a,
    const signed char* __restrict__ noise_a,
    const signed char* __restrict__ s_b,
    const signed char* __restrict__ noise_b,
    unsigned int* __restrict__ partial,
    int h,
    int w,
    int k)
{
    using Wmma = iris::hip::WmmaI8I8I32_16x16x16<true, true, false>;
    using BlockLayout = iris::hip::BlockThreadLayout<kNumThreads>;

    __shared__ signed char sh_a_storage[kJackpotWmmaTileRowsA * kRank];
    __shared__ signed char sh_bt_storage[kRank * kJackpotWmmaTileRowsB];
    __shared__ unsigned int sh_xor[kNumThreads];

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = BlockLayout::wave(tid);
    const int lane = BlockLayout::lane(tid);
    const int lane16 = lane & 15;
    const int wave_m = wave >> 1;
    const int wave_n = wave & 1;
    const int local_m_base = wave_m * Wmma::m;
    const int local_n_base = wave_n * Wmma::n;
    const int tile_u_base = static_cast<int>(blockIdx.y) * kJackpotWmmaTileRowsA;
    const int tile_v_base = static_cast<int>(blockIdx.x) * kJackpotWmmaTileRowsB;
    const int partial_block = static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    const int partial_blocks_per_job = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    const int partial_job_offset = static_cast<int>(blockIdx.z) * partial_blocks_per_job;
    const int tiles = k / kRank;

    auto a_view = iris::hip::make_global_row_major_tensor_view(s_a, h, k, k);
    auto an_view = iris::hip::make_global_row_major_tensor_view(noise_a, h, k, k);
    auto b_view = iris::hip::make_global_row_major_tensor_view(s_b, w, k, k);
    auto bn_view = iris::hip::make_global_row_major_tensor_view(noise_b, w, k, k);
    auto sh_a = iris::hip::make_lds_tile<signed char, kJackpotWmmaTileRowsA, kRank>(sh_a_storage);
    auto sh_bt = iris::hip::make_lds_tile<signed char, kRank, kJackpotWmmaTileRowsB>(sh_bt_storage);

    typename Wmma::c_fragment c = Wmma::zero_c();

    for (int tile = 0; tile < tiles; ++tile) {
        const int base_col = tile * kRank;

        for (int idx = tid; idx < kJackpotWmmaTileRowsA * kRank; idx += kNumThreads) {
            const int row = idx / kRank;
            const int col = idx - row * kRank;
            const int u = tile_u_base + row;
            const int k_col = base_col + col;
            signed char value = 0;
            if (u < h && k_col < k) {
                value = static_cast<signed char>(
                    static_cast<int>(a_view.load(u, k_col)) +
                    static_cast<int>(an_view.load(u, k_col)));
            }
            sh_a.store(row, col, value);
        }

        for (int idx = tid; idx < kJackpotWmmaTileRowsB * kRank; idx += kNumThreads) {
            const int row = idx / kRank;
            const int col = idx - row * kRank;
            const int v = tile_v_base + row;
            const int k_col = base_col + col;
            signed char value = 0;
            if (v < w && k_col < k) {
                value = static_cast<signed char>(
                    static_cast<int>(b_view.load(v, k_col)) +
                    static_cast<int>(bn_view.load(v, k_col)));
            }
            sh_bt.store(col, row, value);
        }

        iris::hip::block_sync();

        #pragma unroll
        for (int k_half = 0; k_half < kRank; k_half += Wmma::k) {
            typename Wmma::ab_fragment a_frag = Wmma::zero_ab();
            typename Wmma::ab_fragment b_frag = Wmma::zero_ab();

            #pragma unroll
            for (int elem = 0; elem < Wmma::ab_elems; ++elem) {
                a_frag[elem] = sh_a.load(local_m_base + elem, k_half + lane16);
                b_frag[elem] = sh_bt.load(k_half + elem, local_n_base + lane16);
            }

            c = Wmma::mma(a_frag, b_frag, c);
        }

        uint32_t thread_xored = 0;
        #pragma unroll
        for (int elem = 0; elem < Wmma::c_elems; ++elem) {
            const int local_u = local_m_base + Wmma::c_row(lane, elem);
            const int local_v = local_n_base + Wmma::c_col(lane);
            const int global_u = tile_u_base + local_u;
            const int global_v = tile_v_base + local_v;
            if (global_u < h && global_v < w) {
                thread_xored ^= static_cast<uint32_t>(c[elem]);
            }
        }

        const uint32_t xored = iris::hip::block_reduce_xor_u32<kNumThreads>(
            thread_xored, sh_xor);

        if (tid == 0) {
            partial[(partial_job_offset + partial_block) * tiles + tile] = xored;
        }

        iris::hip::block_sync();
    }
}

extern "C" __global__ void pearl_jackpot_tiled_reduce(
    const unsigned int* __restrict__ partial,
    unsigned int* __restrict__ jackpot_msg,
    int partial_blocks,
    int k)
{
    __shared__ unsigned int sh_xor[kNumThreads];
    __shared__ unsigned int sh_jackpot_word;

    const int tid = static_cast<int>(threadIdx.x);
    const int word_idx = static_cast<int>(blockIdx.x);
    const int tiles = k / kRank;
    if (word_idx >= kJackpotSize) {
        return;
    }

    if (tid == 0) {
        sh_jackpot_word = 0;
    }
    iris::hip::block_sync();

    for (int tile = word_idx; tile < tiles; tile += kJackpotSize) {
        uint32_t xored = 0;
        for (int block = tid; block < partial_blocks; block += kNumThreads) {
            xored ^= partial[block * tiles + tile];
        }

        const uint32_t xored_tile = iris::hip::block_reduce_xor_u32<kNumThreads>(xored, sh_xor);

        if (tid == 0) {
            sh_jackpot_word =
                (sh_jackpot_word << kLrotPerTile) |
                (sh_jackpot_word >> (32 - kLrotPerTile));
            sh_jackpot_word ^= xored_tile;
        }
        iris::hip::block_sync();
    }

    if (tid == 0) {
        jackpot_msg[word_idx] = sh_jackpot_word;
    }
}

// ============================================================================
// Phase 1: Per-u partial jackpot computation
//
// Each block processes all v for a single u value.
// Grid: h blocks (one per u), Block: 256 threads
//
// Inputs:
//   s_a, noise_a: [h][k] int8
//   s_b, noise_b: [w][k] int8
//   h, w, k: matrix dimensions
//
// Output:
//   partial: [h * tiles] uint32 — partial XOR per (u, tile)
// ============================================================================
extern "C" __global__ void pearl_expand_jackpot_partial(
    const signed char* __restrict__ s_a,
    const signed char* __restrict__ noise_a,
    const signed char* __restrict__ s_b,
    const signed char* __restrict__ noise_b,
    unsigned int* __restrict__ partial,
    int h,
    int w,
    int k)
{
    __shared__ unsigned int sh_xor[kNumThreads];

    int tid = threadIdx.x;
    int u = blockIdx.x;
    if (u >= h) return;

    int tiles = k / kRank;
    int vals_per_thread = (w + kNumThreads - 1) / kNumThreads;

    // Note: vals_per_thread may be 0 for threads beyond w.
    // We compute max_pairs the thread actually handles.
    int local_v_start = tid;
    int local_v_count = 0;
    for (int v = local_v_start; v < w; v += kNumThreads) {
        local_v_count++;
    }

    // Running sums for this thread's v values (max 4 for typical w=1024, but
    // use dynamic sizing with a capped compile-time max of 4).
    // For very large w, we'd need a different mapping; but this covers
    // the expected usage (w <= 1024 with 256 threads = max 4 v's per thread).
    int running_sums[4];

    // Initialize running sums
    for (int i = 0; i < local_v_count; ++i) {
        running_sums[i] = 0;
    }

    // Process each tile sequentially
    for (int tile = 0; tile < tiles; ++tile) {
        int base_col = tile * kRank;
        unsigned int thread_xored = 0;
        int vi = 0;

        for (int v = local_v_start; v < w; v += kNumThreads) {
            // Compute dot product over this tile's columns:
            //   Σ_{l in tile} (s_a[u][l] + noise_a[u][l]) * (s_b[v][l] + noise_b[v][l])
            int dot = 0;
            #pragma unroll
            for (int l = 0; l < kRank; ++l) {
                int col = base_col + l;
                int a_val = (int)s_a[u * k + col] + (int)noise_a[u * k + col];
                int b_val = (int)s_b[v * k + col] + (int)noise_b[v * k + col];
                dot += a_val * b_val;
            }

            // Accumulate cumulative running sum across tiles
            running_sums[vi] += dot;

            // XOR cumulative sum into per-thread tile accumulator
            thread_xored ^= (unsigned int)running_sums[vi];
            vi++;
        }

        // Store per-thread XOR in shared memory
        sh_xor[tid] = thread_xored;
        __syncthreads();

        // Tree-reduce XOR across kNumThreads (256) threads
        if (tid < 128) { sh_xor[tid] ^= sh_xor[tid + 128]; } __syncthreads();
        if (tid < 64)  { sh_xor[tid] ^= sh_xor[tid + 64];  } __syncthreads();
        if (tid < 32)  { sh_xor[tid] ^= sh_xor[tid + 32];  } __syncthreads();
        if (tid < 16)  { sh_xor[tid] ^= sh_xor[tid + 16];  } __syncthreads();
        if (tid < 8)   { sh_xor[tid] ^= sh_xor[tid + 8];   } __syncthreads();
        if (tid < 4)   { sh_xor[tid] ^= sh_xor[tid + 4];   } __syncthreads();
        if (tid < 2)   { sh_xor[tid] ^= sh_xor[tid + 2];   } __syncthreads();
        if (tid == 0)  { sh_xor[0] ^= sh_xor[1]; }
        __syncthreads();

        // Thread 0 writes the block's partial XOR for this tile
        if (tid == 0) {
            partial[u * tiles + tile] = sh_xor[0];
        }
        __syncthreads();
    }
}

// ============================================================================
// Phase 2: Global reduction across all u blocks
//
// Reads partial[h * tiles], computes final jackpot_msg[kJackpotSize].
// Grid: 1 block, Block: 256 threads
// ============================================================================
extern "C" __global__ void pearl_jackpot_reduce(
    const unsigned int* __restrict__ partial,
    unsigned int* __restrict__ jackpot_msg,
    int h,
    int k)
{
    __shared__ unsigned int sh_xor[kNumThreads];
    __shared__ unsigned int sh_jackpot[kJackpotSize];

    int tid = threadIdx.x;
    int tiles = k / kRank;

    // Initialize jackpot in shared memory
    if (tid < kJackpotSize) {
        sh_jackpot[tid] = 0;
    }
    __syncthreads();

    // Process each tile
    for (int tile = 0; tile < tiles; ++tile) {
        unsigned int xored = 0;

        // Each thread XORs its assigned subset of partial results
        for (int u = tid; u < h; u += kNumThreads) {
            xored ^= partial[u * tiles + tile];
        }

        // Store in shared memory for tree reduction
        sh_xor[tid] = xored;
        __syncthreads();

        // Tree-reduce XOR across all threads
        if (tid < 128) { sh_xor[tid] ^= sh_xor[tid + 128]; } __syncthreads();
        if (tid < 64)  { sh_xor[tid] ^= sh_xor[tid + 64];  } __syncthreads();
        if (tid < 32)  { sh_xor[tid] ^= sh_xor[tid + 32];  } __syncthreads();
        if (tid < 16)  { sh_xor[tid] ^= sh_xor[tid + 16];  } __syncthreads();
        if (tid < 8)   { sh_xor[tid] ^= sh_xor[tid + 8];   } __syncthreads();
        if (tid < 4)   { sh_xor[tid] ^= sh_xor[tid + 4];   } __syncthreads();
        if (tid < 2)   { sh_xor[tid] ^= sh_xor[tid + 2];   } __syncthreads();
        if (tid == 0)  { sh_xor[0] ^= sh_xor[1]; }
        __syncthreads();

        // Thread 0 does the rotation + XOR into jackpot
        if (tid == 0) {
            unsigned int xored_tile = sh_xor[0];
            int word_idx = tile % kJackpotSize;

            // Rotate left by kLrotPerTile
            sh_jackpot[word_idx] =
                (sh_jackpot[word_idx] << kLrotPerTile) |
                (sh_jackpot[word_idx] >> (32 - kLrotPerTile));

            // XOR with this tile's final cumulative XOR
            sh_jackpot[word_idx] ^= xored_tile;
        }
        __syncthreads();
    }

    // Write final jackpot words to global memory
    if (tid < kJackpotSize) {
        jackpot_msg[tid] = sh_jackpot[tid];
    }
}
