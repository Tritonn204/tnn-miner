// Included after pearl_prepare.hip; shares its reference-compatible primitives.
// Supplied by the host recipe when compiling production preparation.
#ifndef PEARL_BASE_VALUE
#define PEARL_BASE_VALUE 32
#endif
static_assert(PEARL_BASE_VALUE >= -64 && PEARL_BASE_VALUE <= 63, "Pearl base must be int7");

__device__ __attribute__((always_inline)) inline void
fresh_seed(const uint8_t *left, const uint8_t *right, uint8_t *seed, uint32_t version,
           uint32_t dimension, uint32_t is_a) {
    using namespace pearl_prepare;
    const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t message[64] = {};
    uint32_t output[16];
    uint8_t bound[32];

    if (version == 3) {
        // Upstream cert-v3 domain salts, little-endian BLAKE3 key words.
        const uint32_t salt_a[8] = {0x6c404982, 0x1615eda0, 0x92f61696, 0xf876f0fc,
                                    0x2adbdb92, 0x52b82370, 0x1977d4f0, 0x7b0190c3};
        const uint32_t salt_b[8] = {0x32063011, 0xca0163ec, 0x71afe22b, 0x4f4d3f8b,
                                    0x39c6e91a, 0x04cce888, 0x1d304448, 0xa99ab871};
        for (int i = 0; i < 32; ++i)
            message[i] = right[i];
        store(message + 32, dimension);
        compress(salt_a, message, 0, 64, 1 | 2 | 8 | 16, output);
        for (int i = 0; i < 8; ++i)
            store(bound + i * 4, output[i]);
        right = bound;
    }

    for (int i = 0; i < 32; ++i) {
        message[i] = left[i];
        message[i + 32] = right[i];
    }
    compress(iv, message, 0, 64, 1 | 2 | 8, output);
    for (int i = 0; i < 8; ++i)
        store(seed + i * 4, output[i]);
}

// Batched fresh A: independent patches, Merkle paths, seeds and operands.
extern "C" __global__ void pearl_batch_paths(const uint8_t *key, const uint8_t *template_tree,
                                             const uint8_t *b_seed, uint8_t *patches,
                                             uint8_t *paths, uint8_t *seeds, uint32_t leaves,
                                             uint32_t levels, uint32_t attempts,
                                             uint64_t first_nonce, uint64_t high, uint32_t version,
                                             uint32_t dimension) {
    using namespace pearl_prepare;
    const uint32_t attempt = blockIdx.x * blockDim.x + threadIdx.x;
    if (attempt >= attempts)
        return;

    // Encode the independent nonce into the first 19 base bytes. All other
    // bytes retain the constant base; the noise seed changes with the Merkle root.
    uint8_t block[64] = {};
    for (unsigned i = 0; i < 64; ++i)
        block[i] = PEARL_BASE_VALUE;

    const uint64_t low = first_nonce + attempt;
    for (unsigned i = 0; i < 19; ++i) {
        unsigned value = 0;
        for (unsigned j = 0; j < 7; ++j) {
            const unsigned bit = i * 7 + j;
            if (bit < 128)
                value |= unsigned(((bit < 64 ? low : high) >> (bit % 64)) & 1) << j;
        }
        block[i] = uint8_t(int(value) - 64);
        patches[attempt * 32 + i] = block[i];
    }

    // Rehash the changed 1024-byte leaf, then walk its leftmost Merkle path.
    // The sibling subtree at every level is unchanged and comes from the cache.
    uint32_t cv[8], out[16];
    for (unsigned i = 0; i < 8; ++i)
        cv[i] = load(key + i * 4);

    for (unsigned chunk = 0; chunk < 16; ++chunk) {
        compress(cv, block, 0, 64, 16 | (chunk == 0 ? 1 : 0) | (chunk == 15 ? 2 : 0), out);
        for (unsigned i = 0; i < 8; ++i)
            cv[i] = out[i];
        if (chunk == 0)
            for (unsigned i = 0; i < 19; ++i)
                block[i] = PEARL_BASE_VALUE;
    }
    auto path = paths + size_t(attempt) * levels * 32;
    for (unsigned i = 0; i < 8; ++i)
        store(path + i * 4, cv[i]);

    uint32_t count = leaves;
    uint32_t level = 0;
    while (count > 1) {
        for (unsigned i = 0; i < 8; ++i)
            store(block + i * 4, cv[i]);
        for (unsigned i = 0; i < 32; ++i)
            block[i + 32] = template_tree[32 + i];
        for (unsigned i = 0; i < 8; ++i)
            cv[i] = load(key + i * 4);
        compress(cv, block, 0, 64, 16 | 4 | (count == 2 ? 8 : 0), out);
        for (unsigned i = 0; i < 8; ++i)
            cv[i] = out[i];
        ++level;
        for (unsigned i = 0; i < 8; ++i)
            store(path + level * 32 + i * 4, cv[i]);
        template_tree += count * 32;
        count = (count + 1) / 2;
    }

    fresh_seed(b_seed, path + (levels - 1) * 32, seeds + attempt * 32, version, dimension, 1);
}

extern "C" __global__ void pearl_batch_sparse(const uint8_t *seeds, uint32_t *pairs, uint32_t k,
                                              uint32_t attempts) {
    using namespace pearl_prepare;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= attempts * (k / 8))
        return;
    const uint32_t attempt = index / (k / 8);
    index %= k / 8;
    uint32_t output[16];
    random(index, seeds + attempt * 32, true, true, output);
    pairs += size_t(attempt) * k * 2;
    for (int i = 0; i < 8; ++i) {
        const uint32_t first = output[i] & 127;
        pairs[(index * 8 + i) * 2] = first;
        pairs[(index * 8 + i) * 2 + 1] = first ^ (1 + uint32_t((uint64_t(127) * output[i]) >> 32));
    }
}

extern "C" __global__ void pearl_batch_materialize(const int8_t *base, const uint8_t *seed,
                                                   const uint32_t *pairs, uint32_t *operand,
                                                   uint32_t rows, uint32_t k, uint32_t first) {
    const uint32_t attempt = first + blockIdx.y;
    base += size_t(attempt) * 32;
    seed += size_t(attempt) * 32;
    pairs += size_t(attempt) * k * 2;
    operand += size_t(blockIdx.y) * rows * k / 4;
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
            const int signal = row_base + local_row + lane == 0 && col < 19
                                   ? int(base[col])
                                   : PEARL_BASE_VALUE;
            const int value = signal + int(factors[first][local_row + lane]) -
                              int(factors[second][local_row + lane]);
            packed |= uint32_t(uint8_t(value)) << (lane * 8);
        }
        operand[(col * rows + row_base + local_row) / 4] = packed;
    }
}
