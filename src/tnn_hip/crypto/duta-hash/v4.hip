#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROTL64(x,n) (((uint64_t)(x) << (n)) | ((uint64_t)(x) >> (64-(n))))
#define EPOCH_LEN   2048ULL
#define SCRATCHPAD_BYTES (256u * 1024u)
#define POW_STEPS 4096ull
#define RANDOM_DATASET_READS_PER_STEP 4u

__host__ __forceinline__ static uint64_t rotl64_u(uint64_t x, uint32_t r) {
    return __builtin_rotateleft64(x, r);
}

__host__ __device__ __forceinline__ uint64_t load_u64_le(const uint8_t* p) {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    return *reinterpret_cast<const uint64_t*>(p);
#else
    return ((uint64_t)p[0])       |
           ((uint64_t)p[1] << 8)  |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
#endif
}

__host__ __device__ __forceinline__ void store_u64_le(uint8_t* p, uint64_t v) {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    *reinterpret_cast<uint64_t*>(p) = v;
#else
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32);
    p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48);
    p[7] = (uint8_t)(v >> 56);
#endif
}

__device__ __forceinline__ uint32_t lz8(uint32_t x) {
    return __clz(x) - 24u;
}

#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ uint64_t load_uncached_u64(const uint64_t* ptr) {
    uint64_t val;
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__)
    asm volatile(
        "global_load_dwordx2 %0, %1, off glc slc dlc\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(val)
        : "v"(ptr)
        : "memory");
#else
    asm volatile(
        "global_load_dwordx2 %0, %1, off glc slc\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(val)
        : "v"(ptr)
        : "memory");
#endif
    return val;
}
__device__ __forceinline__ void store_uncached_u64(uint64_t* ptr, uint64_t val) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__)
    asm volatile("global_store_dwordx2 %0, %1, off glc slc dlc" : : "v"(ptr), "v"(val) : "memory");
#else
    asm volatile("global_store_dwordx2 %0, %1, off glc slc" : : "v"(ptr), "v"(val) : "memory");
#endif
}
#else
__device__ __forceinline__ uint64_t load_uncached_u64(const uint64_t* ptr) {
    uint64_t val;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(val) : "l"(ptr));
    return val;
}
__device__ __forceinline__ void store_uncached_u64(uint64_t* ptr, uint64_t val) {
    asm volatile("st.global.wt.u64 [%0], %1;" : : "l"(ptr), "l"(val));
}
#endif

__device__ __forceinline__ void sha3_256_device_small(const uint8_t* msg, size_t len, uint8_t out[32]) {
    const int RATE = 136;
    uint64_t st[25];
#pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0;
    size_t pos = 0;
    while (pos + RATE <= len) {
#pragma unroll
        for (int i = 0; i < 17; i++) st[i] ^= load_u64_le(msg + pos + i * 8);
        keccakf(st);
        pos += RATE;
    }
    size_t rem = len - pos;
    for (size_t i = 0; i < rem; i++) st[i >> 3] ^= (uint64_t)msg[pos + i] << ((i & 7u) * 8u);
    st[rem >> 3] ^= (uint64_t)0x06u << ((rem & 7u) * 8u);
    st[16] ^= (uint64_t)0x80u << 56;
    keccakf(st);
#pragma unroll
    for (int i = 0; i < 4; i++) store_u64_le(out + i * 8, st[i]);
}

/* ========================= Shared helpers ========================= */

__host__ __device__ __forceinline__ void append_u64le(uint8_t* dst, size_t* off, uint64_t v) {
    store_u64_le(dst + *off, v);
    *off += 8;
}

__host__ __device__ __forceinline__ void append_bytes(uint8_t* dst, size_t* off, const uint8_t* src, size_t n) {
    for (size_t i = 0; i < n; i++) dst[*off + i] = src[i];
    *off += n;
}

__host__ __device__ __forceinline__ uint64_t mix_u64_v4(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t z = (a + rotl64_u(b ^ c, 19)) * 0x9E3779B97F4A7C15ULL;
    uint64_t y = z ^ rotl64_u(a + c, 27);
    return y + rotl64_u(b, 41);
}

/* ========================= Host dataset builder ========================= */
static void build_dataset_v4_host(uint64_t epoch, const uint8_t anchor32[32], size_t mem_mb, uint8_t* ds_out) {
    const size_t m = mem_mb * 1024ull * 1024ull;
    const size_t nblocks = m / 64u;

    uint8_t seed_in[19 + 8 + 32];
    size_t off = 0;
    append_bytes(seed_in, &off, (const uint8_t*)"CPUCOIN_DATASET_V4|", 19);
    append_u64le(seed_in, &off, epoch);
    append_bytes(seed_in, &off, anchor32, 32);
    uint8_t seed[32];
    sha3_256_host_small(seed_in, off, seed);

    uint64_t s0 = load_u64_le(seed + 0);
    uint64_t s1 = load_u64_le(seed + 8);
    uint64_t s2 = load_u64_le(seed + 16);
    uint64_t s3 = load_u64_le(seed + 24);

    uint8_t inp[8 * 5 + 32];
    uint8_t h1[32], h2[32];
    for (size_t i = 0; i < nblocks; i++) {
        uint64_t iu = (uint64_t)i;
        s0 = mix_u64_v4(s0 ^ iu, s1, s3);
        s1 = mix_u64_v4(s1, s2 ^ rotl64_u(iu, 7), s0);
        s2 = mix_u64_v4(s2, s3, s1 ^ rotl64_u(iu, 13));
        s3 = mix_u64_v4(s3 ^ iu, s0, s2);

        off = 0;
        append_u64le(inp, &off, s0);
        append_u64le(inp, &off, s1);
        append_u64le(inp, &off, s2);
        append_u64le(inp, &off, s3);
        append_u64le(inp, &off, iu);
        append_bytes(inp, &off, anchor32, 32);

        sha3_256_host_small(inp, off, h1);
        sha3_256_host_small(h1, 32, h2);

        size_t out_off = i * 64u;
        for (int j = 0; j < 32; j++) ds_out[out_off + j] = h1[j];
        for (int j = 0; j < 32; j++) ds_out[out_off + 32 + j] = h2[j];
    }
    printf("HOST DATA BUILT\n");
}

/* ========================= Device PoW ========================= */

__global__ void init_scratch_kernel_v4(
    const uint8_t* header80,
    const uint8_t* anchor32,
    uint64_t height,
    uint64_t nonce_start,
    uint64_t* scratch_pool
) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t nonce = nonce_start + gid;
    const uint64_t ep = height / EPOCH_LEN;
    const uint32_t scratch_words = SCRATCHPAD_BYTES / 8u;
    uint64_t* my_scratch = scratch_pool + gid * (uint64_t)scratch_words;

    uint8_t fill[19 + 80 + 8 + 8 + 32];
    size_t fill_off = 0;
    append_bytes(fill, &fill_off, (const uint8_t*)"CPUCOIN_POW_V4_SCR|", 19);
    append_bytes(fill, &fill_off, header80, 80);
    append_u64le(fill, &fill_off, nonce);
    append_u64le(fill, &fill_off, height);
    append_bytes(fill, &fill_off, anchor32, 32);

    uint8_t seed_block[32];
    sha3_256_device_small(fill, fill_off, seed_block);

    for (uint32_t i = 0; i < scratch_words; i++) {
        if ((i & 3u) == 0u) {
            uint8_t next[32 + 8 + 8];
            size_t next_off = 0;
            append_bytes(next, &next_off, seed_block, 32);
            append_u64le(next, &next_off, (uint64_t)i);
            append_u64le(next, &next_off, ep);
            sha3_256_device_small(next, next_off, seed_block);

        }

        uint32_t byte_off = (i & 3u) * 8u;
        uint64_t val = load_u64_le(seed_block + byte_off) ^ ((uint64_t)i * 0x9E3779B9ULL);
        store_uncached_u64(my_scratch + i, val);
    }
}

__device__ __forceinline__ void pow_digest_v4_gpu(
    uint64_t out[4],
    const uint8_t* header80,
    uint64_t nonce,
    uint64_t height,
    const uint8_t* anchor32,
    const uint8_t* __restrict__ dataset,
    uint32_t nblocks,
    uint64_t* my_scratch
) {
    const uint64_t ep = height / EPOCH_LEN;
    const uint32_t scratch_words = SCRATCHPAD_BYTES / 8u;

    uint8_t seed_in[15 + 80 + 8 + 8 + 8 + 32];
    size_t off = 0;
    append_bytes(seed_in, &off, (const uint8_t*)"CPUCOIN_POW_V4|", 15);
    append_bytes(seed_in, &off, header80, 80);
    append_u64le(seed_in, &off, nonce);
    append_u64le(seed_in, &off, height);
    append_u64le(seed_in, &off, ep);
    append_bytes(seed_in, &off, anchor32, 32);
    uint8_t seed[32];
    sha3_256_device_small(seed_in, off, seed);

    uint64_t s0 = load_u64_le(seed + 0);
    uint64_t s1 = load_u64_le(seed + 8);
    uint64_t s2 = load_u64_le(seed + 16);
    uint64_t s3 = load_u64_le(seed + 24);
    uint64_t lane = s0 ^ s2;

    for (uint64_t t = 0; t < POW_STEPS; t++) {
        uint64_t idx0 = lane ^ rotl64_u(s1, 11) ^ (t * 0x9E3779B97F4A7C15ULL);
        uint32_t j0 = (uint32_t)(idx0 % (uint64_t)nblocks);
        const uint8_t* p0 = dataset + (size_t)j0 * 64u;
        uint64_t x0 = load_u64_le(p0 + 0);
        uint64_t x1 = load_u64_le(p0 + 16);

        uint64_t idx1 = s2 ^ rotl64_u(s3, 29) ^ x0 ^ x1 ^ (lane * 0xD1B54A32D192ED03ULL);
        uint32_t j1 = (uint32_t)(idx1 % (uint64_t)nblocks);
        const uint8_t* p1 = dataset + (size_t)j1 * 64u;
        uint64_t y0 = load_u64_le(p1 + 8);
        uint64_t y1 = load_u64_le(p1 + 24);

        uint32_t sp_idx0 = (uint32_t)((lane ^ s0 ^ x0) % scratch_words);
        uint32_t sp_idx1 = (uint32_t)((rotl64_u(s1, 9) ^ y0 ^ t) % scratch_words);
        uint32_t sp_idx2 = (uint32_t)((rotl64_u(s2, 17) ^ x1 ^ lane) % scratch_words);
        uint32_t sp_idx3 = (uint32_t)((rotl64_u(s3, 27) ^ y1 ^ s0) % scratch_words);

        uint64_t sp0 = load_uncached_u64(my_scratch + sp_idx0);
        uint64_t sp1 = load_uncached_u64(my_scratch + sp_idx1);
        uint64_t sp2 = load_uncached_u64(my_scratch + sp_idx2);
        uint64_t sp3 = load_uncached_u64(my_scratch + sp_idx3);

        uint64_t extra = 0;

        #pragma unroll
        for (uint32_t r = 0; r < RANDOM_DATASET_READS_PER_STEP; r++) {
            uint64_t mix;
            if (r == 0) mix = lane ^ sp0 ^ s2;
            else if (r == 1) mix = s0 ^ sp1 ^ y1;
            else if (r == 2) mix = s1 ^ sp2 ^ x0;
            else mix = s3 ^ sp3 ^ x1;

            uint64_t idx = mix ^ ((t + (uint64_t)r) * 0x94D049BB133111EBULL);
            uint32_t j = (uint32_t)(idx % (uint64_t)nblocks);
            const uint8_t* pp = dataset + (size_t)j * 64u;
            uint64_t lo = load_u64_le(pp + 0);
            uint64_t hi = load_u64_le(pp + 32);
            extra ^= mix_u64_v4(lo, hi, mix ^ (uint64_t)r);
        }

        lane = mix_u64_v4(lane ^ x0 ^ sp0, y1 ^ sp1, t ^ s3 ^ extra);
        s0 = mix_u64_v4(s0 ^ x0 ^ extra, y0 ^ sp2, lane);
        s1 = mix_u64_v4(s1 ^ sp1, x1 ^ lane ^ extra, s0);
        s2 = mix_u64_v4(s2 ^ y0 ^ sp3, s0 ^ extra, x1);
        s3 = mix_u64_v4(s3 ^ y1 ^ extra, s1 ^ sp0, lane);

        store_uncached_u64(my_scratch + sp_idx0, mix_u64_v4(sp0 ^ x0, lane, extra));
        store_uncached_u64(my_scratch + sp_idx1, mix_u64_v4(sp1 ^ y0, s0, lane));
        store_uncached_u64(my_scratch + sp_idx2, mix_u64_v4(sp2 ^ x1, s1, extra));
        store_uncached_u64(my_scratch + sp_idx3, mix_u64_v4(sp3 ^ y1, s2, s3));
    }

    uint8_t fin_in[19 + 80 + 8 + 8 + 8 + 32 + 8 * 8];
    size_t fin_off = 0;
    append_bytes(fin_in, &fin_off, (const uint8_t*)"CPUCOIN_POW_V4_FIN|", 19);
    append_bytes(fin_in, &fin_off, header80, 80);
    append_u64le(fin_in, &fin_off, nonce);
    append_u64le(fin_in, &fin_off, height);
    append_u64le(fin_in, &fin_off, ep);
    append_bytes(fin_in, &fin_off, anchor32, 32);
    append_u64le(fin_in, &fin_off, lane);
    append_u64le(fin_in, &fin_off, s0);
    append_u64le(fin_in, &fin_off, s1);
    append_u64le(fin_in, &fin_off, s2);
    append_u64le(fin_in, &fin_off, s3);
    append_u64le(fin_in, &fin_off, load_uncached_u64(my_scratch + 0));
    append_u64le(fin_in, &fin_off, load_uncached_u64(my_scratch + (scratch_words / 2u)));
    append_u64le(fin_in, &fin_off, load_uncached_u64(my_scratch + (scratch_words - 1u)));

    uint8_t digest[32];
    sha3_256_device_small(fin_in, fin_off, digest);
    out[0] = load_u64_le(digest + 0);
    out[1] = load_u64_le(digest + 8);
    out[2] = load_u64_le(digest + 16);
    out[3] = load_u64_le(digest + 24);
}

__device__ __forceinline__ uint32_t lzb256(const uint64_t h[4]) {
    uint32_t n = 0;
    for (int w = 0; w < 4; w++) {
        uint64_t word = h[w];
        for (int b = 0; b < 8; b++) {
            uint32_t byte = (uint32_t)((word >> (b * 8)) & 0xFFu);
            if (!byte) n += 8u;
            else return n + lz8(byte);
        }
    }
    return 256u;
}

__global__ void digest_kernel_v4(
    const uint8_t* __restrict__ dataset,
    uint32_t nblocks,
    const uint8_t* header80,
    const uint8_t* anchor32,
    uint64_t height,
    uint64_t nonce,
    uint64_t* scratch_pool,
    uint64_t* digest_out
) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t scratch_words = SCRATCHPAD_BYTES / 8u;
    uint64_t* my_scratch = scratch_pool + gid * (uint64_t)scratch_words;
    uint64_t digest[4];
    pow_digest_v4_gpu(digest, header80, nonce, height, anchor32, dataset, nblocks, my_scratch);
    digest_out[0] = digest[0];
    digest_out[1] = digest[1];
    digest_out[2] = digest[2];
    digest_out[3] = digest[3];
}

__global__ void mine_kernel_v4(
    const uint8_t* __restrict__ dataset,
    uint32_t nblocks,
    const uint8_t* header80,
    const uint8_t* anchor32,
    uint64_t height,
    uint32_t target_bits,
    uint64_t nonce_start,
    uint64_t* scratch_pool,
    uint64_t* found_nonce,
    int* found_flag
) {
    if (*found_flag) return;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t scratch_words = SCRATCHPAD_BYTES / 8u;
    uint64_t* my_scratch = scratch_pool + gid * (uint64_t)scratch_words;
    uint64_t digest[4];
    uint64_t nonce = nonce_start + gid;
    pow_digest_v4_gpu(digest, header80, nonce, height, anchor32, dataset, nblocks, my_scratch);
    if (lzb256(digest) >= target_bits) {
        if (atomicCAS(found_flag, 0, 1) == 0) *found_nonce = nonce;
    }
}