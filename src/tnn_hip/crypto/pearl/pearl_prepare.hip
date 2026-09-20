// Reference-exact preparation, deliberately separate from qualified GEMM ISA.
// Algorithms follow BLAKE3 and pearl_native.cpp; see PEARL_REFERENCE_LICENSE.txt.
#ifndef PEARL_PREP_CPU
#include "hiprtc_types.hip.h"
#endif

namespace pearl_prepare {

struct Output {
    uint32_t cv[8];
    uint8_t block[64];
    uint32_t length, flags;
};
static_assert(sizeof(Output) == 104, "Base output ABI");

__device__ inline uint32_t load(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
__device__ inline void store(uint8_t* p, uint32_t v) {
    for (unsigned i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}
__device__ inline uint32_t rotate(uint32_t v, unsigned n) { return (v >> n) | (v << (32 - n)); }
__device__ inline void mix(uint32_t* s, int a, int b, int c, int d, uint32_t x, uint32_t y) {
    s[a] += s[b] + x; s[d] = rotate(s[d] ^ s[a], 16);
    s[c] += s[d]; s[b] = rotate(s[b] ^ s[c], 12);
    s[a] += s[b] + y; s[d] = rotate(s[d] ^ s[a], 8);
    s[c] += s[d]; s[b] = rotate(s[b] ^ s[c], 7);
}

__device__ inline void compress(const uint32_t* cv, const uint8_t* block,
                                uint64_t counter, uint32_t length, uint32_t flags, uint32_t* out) {
    uint32_t s[16], m[16];
    for (int i = 0; i < 8; ++i) s[i] = cv[i];
    s[8] = 0x6a09e667; s[9] = 0xbb67ae85; s[10] = 0x3c6ef372; s[11] = 0xa54ff53a;
    s[12] = uint32_t(counter); s[13] = uint32_t(counter >> 32); s[14] = length; s[15] = flags;
    for (int i = 0; i < 16; ++i) m[i] = load(block + 4 * i);
    constexpr unsigned permutation[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    #pragma unroll
    for (int round = 0; round < 7; ++round) {
        mix(s, 0, 4, 8, 12, m[0], m[1]); mix(s, 1, 5, 9, 13, m[2], m[3]);
        mix(s, 2, 6, 10, 14, m[4], m[5]); mix(s, 3, 7, 11, 15, m[6], m[7]);
        mix(s, 0, 5, 10, 15, m[8], m[9]); mix(s, 1, 6, 11, 12, m[10], m[11]);
        mix(s, 2, 7, 8, 13, m[12], m[13]); mix(s, 3, 4, 9, 14, m[14], m[15]);
        uint32_t next[16];
        for (int i = 0; i < 16; ++i) next[i] = m[permutation[i]];
        for (int i = 0; i < 16; ++i) m[i] = next[i];
    }
    for (int i = 0; i < 8; ++i) { out[i] = s[i] ^ s[i + 8]; out[i + 8] = s[i + 8] ^ cv[i]; }
}

__device__ inline void random(uint32_t index, const uint8_t* seed, bool a, bool sparse, uint32_t* out) {
    uint8_t message[64] = {};
    store(message + (sparse ? 4 : 0), index + 1);
    message[32] = a ? 'A' : 'B'; message[33] = '_'; message[34] = 't'; message[35] = 'e';
    message[36] = 'n'; message[37] = 's'; message[38] = 'o'; message[39] = 'r';
    uint32_t key[8];
    for (int i = 0; i < 8; ++i) key[i] = load(seed + 4 * i);
    compress(key, message, 0, 64, 16 | 1 | 2 | 8, out);
}
} // namespace pearl_prepare

extern "C" __global__ void pearl_prepare_base(const pearl_prepare::Output* descriptor, uint8_t* base, uint32_t bytes) {
    using namespace pearl_prepare;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= bytes / 64) return;
    uint32_t output[16];
    compress(descriptor->cv, descriptor->block, index, descriptor->length, descriptor->flags, output);
    for (int i = 0; i < 64; ++i) base[index * 64 + i] = uint8_t(int((output[i / 4] >> (8 * (i % 4))) & 127) - 64);
}

extern "C" __global__ void pearl_prepare_leaves(const uint8_t* base, const uint8_t* key, uint8_t* leaves, uint32_t count) {
    using namespace pearl_prepare;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t cv[8], output[16];
    for (int i = 0; i < 8; ++i) cv[i] = load(key + 4 * i);
    for (int block = 0; block < 16; ++block) {
        compress(cv, base + index * 1024 + block * 64, index, 64,
                 16 | (block == 0 ? 1 : 0) | (block == 15 ? 2 : 0), output);
        for (int i = 0; i < 8; ++i) cv[i] = output[i];
    }
    for (int i = 0; i < 8; ++i) store(leaves + index * 32 + i * 4, cv[i]);
}

extern "C" __global__ void pearl_prepare_parents(const uint8_t* children, const uint8_t* key, uint8_t* parents, uint32_t count) {
    using namespace pearl_prepare;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (count + 1) / 2) return;
    if (index * 2 + 1 == count) {
        for (int i = 0; i < 32; ++i) parents[index * 32 + i] = children[index * 64 + i];
        return;
    }
    uint32_t cv[8], output[16];
    for (int i = 0; i < 8; ++i) cv[i] = load(key + 4 * i);
    compress(cv, children + index * 64, 0, 64, 16 | 4 | (count == 2 ? 8 : 0), output);
    for (int i = 0; i < 8; ++i) store(parents + index * 32 + i * 4, output[i]);
}

__device__ __attribute__((always_inline)) inline void pearl_seed(const uint8_t* left, const uint8_t* right, uint8_t* seed,
                                                               uint32_t version, uint32_t dimension, uint32_t is_a) {
    using namespace pearl_prepare;
    if (blockIdx.x || threadIdx.x) return;
    const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t message[64] = {};
    uint32_t output[16];
    uint8_t bound[32];
    if (version == 3) {
        // Upstream cert-v3 domain salts, little-endian BLAKE3 key words.
        const uint32_t salt_a[8] = {0x6c404982,0x1615eda0,0x92f61696,0xf876f0fc,0x2adbdb92,0x52b82370,0x1977d4f0,0x7b0190c3};
        const uint32_t salt_b[8] = {0x32063011,0xca0163ec,0x71afe22b,0x4f4d3f8b,0x39c6e91a,0x04cce888,0x1d304448,0xa99ab871};
        for (int i = 0; i < 32; ++i) message[i] = right[i];
        store(message + 32, dimension);
        compress(is_a ? salt_a : salt_b, message, 0, 64, 1 | 2 | 8 | 16, output);
        for (int i = 0; i < 8; ++i) store(bound + i * 4, output[i]);
        right = bound;
    }
    for (int i = 0; i < 32; ++i) { message[i] = left[i]; message[i + 32] = right[i]; }
    compress(iv, message, 0, 64, 1 | 2 | 8, output);
    for (int i = 0; i < 8; ++i) store(seed + i * 4, output[i]);
}

extern "C" __global__ void pearl_prepare_seed(const uint8_t* left, const uint8_t* right, uint8_t* seed,
                                            uint32_t version, uint32_t dimension, uint32_t is_a) {
    pearl_seed(left, right, seed, version, dimension, is_a);
}

extern "C" __global__ void pearl_prepare_dense(const uint8_t* seed, uint8_t* dense, uint32_t rows, uint32_t is_a) {
    using namespace pearl_prepare;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * 4) return;
    // Mode 2 emits A factors rank-major. Adjacent lanes write adjacent rows;
    // packed column-major operand generation subsequently reads packed words.
    const uint32_t hash_index = is_a == 2 ? (index % rows) * 4 + index / rows : index;
    uint32_t output[16];
    random(hash_index, seed, is_a != 0, false, output);
    for (int i = 0; i < 32; ++i) {
        const uint32_t destination = is_a == 2 ? (index / rows * 32 + i) * rows + index % rows : index * 32 + i;
        dense[destination] = uint8_t(int((output[i / 4] >> (8 * (i % 4))) & 63) - 32);
    }
}

extern "C" __global__ void pearl_prepare_sparse(const uint8_t* seed, uint32_t* pairs, uint32_t k, uint32_t is_a) {
    using namespace pearl_prepare;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= k / 8) return;
    uint32_t output[16];
    random(index, seed, is_a, true, output);
    for (int i = 0; i < 8; ++i) {
        uint32_t first = output[i] & 127;
        pairs[(index * 8 + i) * 2] = first;
        pairs[(index * 8 + i) * 2 + 1] = first ^ (1 + uint32_t((uint64_t(127) * output[i]) >> 32));
    }
}

extern "C" __global__ void pearl_prepare_materialize(const int8_t* base, const int8_t* dense, const uint32_t* pairs,
                                                    int8_t* operand, uint32_t rows, uint32_t k, uint32_t is_a) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * k) return;
    uint32_t row = is_a ? index % rows : index / k;
    uint32_t col = is_a ? index / rows : index % k;
    operand[index] = int8_t(int(base[row * k + col]) + int(dense[row * 128 + pairs[col * 2]]) -
                           int(dense[row * 128 + pairs[col * 2 + 1]]));
}

extern "C" __global__ void pearl_prepare_gather(const uint8_t* base, const uint32_t* rows, uint8_t* sampled,
                                               uint32_t count, uint32_t k) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count * k) sampled[index] = base[rows[index / k] * k + index % k];
}

// One changed 1024-byte leaf, followed by its complete keyed Merkle path.
// The base is persistent. Only the first 19 int7 bytes encode the attempt nonce.
__device__ __attribute__((always_inline)) inline void pearl_incremental_path(uint8_t* base, const uint8_t* key,
                                                     uint8_t* tree, uint32_t count,
                                                     uint32_t leaf, uint64_t low, uint64_t high,
                                                     const uint8_t* seed_left, uint8_t* seed,
                                                     uint32_t version = 2, uint32_t dimension = 0) {
    using namespace pearl_prepare;
    if (blockIdx.x || threadIdx.x) return;
    for (unsigned i = 0; i < 19; ++i) {
        unsigned value = 0;
        for (unsigned j = 0; j < 7; ++j) {
            const unsigned bit = i * 7 + j;
            if (bit < 128) value |= unsigned(((bit < 64 ? low : high) >> (bit % 64)) & 1) << j;
        }
        base[leaf * 1024 + i] = uint8_t(int(value) - 64);
    }
    uint32_t cv[8], output[16];
    for (unsigned i = 0; i < 8; ++i) cv[i] = load(key + i * 4);
    for (unsigned block = 0; block < 16; ++block) {
        compress(cv, base + leaf * 1024 + block * 64, leaf, 64,
                 16 | (block == 0 ? 1 : 0) | (block == 15 ? 2 : 0), output);
        for (unsigned i = 0; i < 8; ++i) cv[i] = output[i];
    }
    for (unsigned i = 0; i < 8; ++i) store(tree + leaf * 32 + i * 4, cv[i]);
    uint32_t index = leaf;
    while (count > 1) {
        uint8_t* parents = tree + count * 32;
        const uint32_t pair = index & ~1u;
        if (pair + 1 == count) {
            for (unsigned i = 0; i < 32; ++i) parents[(index / 2) * 32 + i] = tree[index * 32 + i];
        } else {
            for (unsigned i = 0; i < 8; ++i) cv[i] = load(key + i * 4);
            compress(cv, tree + pair * 32, 0, 64, 16 | 4 | (count == 2 ? 8 : 0), output);
            for (unsigned i = 0; i < 8; ++i) store(parents + (index / 2) * 32 + i * 4, output[i]);
        }
        tree = parents;
        count = (count + 1) / 2;
        index /= 2;
    }
    if (seed) pearl_seed(seed_left, tree, seed, version, dimension, 1);
}

extern "C" __global__ void pearl_prepare_incremental(uint8_t* base, const uint8_t* key,
    uint8_t* tree, uint32_t count, uint32_t leaf, uint64_t low, uint64_t high) {
    pearl_incremental_path(base, key, tree, count, leaf, low, high, nullptr, nullptr);
}

extern "C" __global__ void pearl_prepare_incremental_seed(uint8_t* base, const uint8_t* key,
    uint8_t* tree, uint32_t count, uint32_t leaf, uint64_t low, uint64_t high,
    const uint8_t* seed_left, uint8_t* seed, uint32_t version, uint32_t dimension) {
    pearl_incremental_path(base, key, tree, count, leaf, low, high, seed_left, seed, version, dimension);
}

// Output-contiguous, four int8 values per thread. The zero base differs only
// at the nonce patch, so no full-sized strided base reads are necessary.
extern "C" __global__ void pearl_prepare_zero_operand(const int8_t* base, const int8_t* dense,
                                                       const uint32_t* pairs, uint32_t* operand,
                                                       uint32_t rows, uint32_t k) {
    const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= rows * k / 4) return;
    const uint32_t row = word * 4 % rows;
    const uint32_t col = word * 4 / rows;
    const uint32_t first = pairs[col * 2], second = pairs[col * 2 + 1];
    const uint32_t left = reinterpret_cast<const uint32_t*>(dense + first * rows)[row / 4];
    const uint32_t right = reinterpret_cast<const uint32_t*>(dense + second * rows)[row / 4];
    uint32_t packed = 0;
    for (unsigned lane = 0; lane < 4; ++lane) {
        const int value = (row + lane == 0 && col < 19 ? int(base[col]) : 0)
            + int(int8_t(left >> (lane * 8))) - int(int8_t(right >> (lane * 8)));
        packed |= uint32_t(uint8_t(value)) << (lane * 8);
    }
    operand[word] = packed;
}

#ifndef PEARL_PREP_CPU
// Recovered fusion strategy: generate each row's factors once per workgroup,
// retain them in LDS, and emit the entire K extent directly in GEMM layout.
// 32 rows x 128 factors = 4096 LDS bytes. No global dense-factor round trip.
// B stays materialized/cached; this specialization is only for nonce-patched A.
extern "C" __global__ void pearl_prepare_zero_fused(const int8_t* base, const uint8_t* seed,
                                                    const uint32_t* pairs, uint32_t* operand,
                                                    uint32_t rows, uint32_t k) {
    __shared__ int8_t factors[128][32];
    const uint32_t tid = threadIdx.x;
    const uint32_t row_base = blockIdx.x * 32;
    const uint32_t row = tid % 32;
    const uint32_t chunk = tid / 32;
    uint32_t hash[16];
    pearl_prepare::random((row_base + row) * 4 + chunk, seed, true, false, hash);
    for (unsigned i = 0; i < 32; ++i)
        factors[chunk * 32 + i][row] = int8_t(int((hash[i / 4] >> (8 * (i % 4))) & 63) - 32);
    __syncthreads();

    const uint32_t local_row = (tid % 8) * 4;
    for (uint32_t col = tid / 8; col < k; col += 16) {
        const uint32_t first = pairs[col * 2];
        const uint32_t second = pairs[col * 2 + 1];
        uint32_t packed = 0;
        for (unsigned lane = 0; lane < 4; ++lane) {
            const int signal = row_base + local_row + lane == 0 && col < 19 ? int(base[col]) : 0;
            const int value = signal + int(factors[first][local_row + lane])
                - int(factors[second][local_row + lane]);
            packed |= uint32_t(uint8_t(value)) << (lane * 8);
        }
        operand[(col * rows + row_base + local_row) / 4] = packed;
    }
}

// 32x32 input tile: coalesced base reads and coalesced column-major writes.
extern "C" __global__ void pearl_prepare_tiled(const int8_t* base, const int8_t* dense,
                                               const uint32_t* pairs, int8_t* operand,
                                               uint32_t rows, uint32_t k) {
    __shared__ int8_t tile[32][33];
    const uint32_t row_base = (blockIdx.x % (rows / 32)) * 32;
    const uint32_t col_base = (blockIdx.x / (rows / 32)) * 32;
    const uint32_t x = threadIdx.x % 32, y = threadIdx.x / 32;
    for (unsigned step = 0; step < 32; step += 4) {
        const uint32_t row = row_base + y + step, col = col_base + x;
        tile[y + step][x] = int8_t(int(base[row * k + col])
            + int(dense[row * 128 + pairs[col * 2]]) - int(dense[row * 128 + pairs[col * 2 + 1]]));
    }
    __syncthreads();
    for (unsigned step = 0; step < 32; step += 4)
        operand[(col_base + y + step) * rows + row_base + x] = tile[x][y + step];
}
#endif
