// xelis-hash-v3-internal.hpp -- shared helpers for FMV tier TUs
// Included by generated simd_sources/*.gen.cpp via the spec's //@ common block.
#pragma once

#include "xelis-hash.hpp"
#include <stdlib.h>
#include <iostream>
#include "aes.hpp"
#include "chacha20.h"
#include "chacha20_xelis_inline.h"
#include <crc32.h>
#include <memory>

#include <BLAKE3/c/blake3.h>

#include <byteswap.h>
#include <chacha20.h>
#include <cmath>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

#if defined(__x86_64__)
#include <emmintrin.h>
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#include <arm_acle.h>
#endif
#include <numeric>
#include <cstring>
#include <array>
#include <cassert>

#include "compile.h"

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

#define rl64(x, a) (((x << (a & 63)) | (x >> (64 - (a & 63)))))
#define rr64(x, a) (((x >> (a & 63)) | (x << (64 - (a & 63)))))

#define htonll(x) ((1 == htonl(1)) ? (x) : ((uint64_t)htonl((x) & 0xFFFFFFFF) << 32) | htonl((x) >> 32))
#define ntohll(x) ((1 == ntohl(1)) ? (x) : ((uint64_t)ntohl((x) & 0xFFFFFFFF) << 32) | ntohl((x) >> 32))

#define COMBINE_UINT64(high, low) (((__uint128_t)(high) << 64) | (low))

static inline uint64_t swap_bytes(uint64_t value)
{
    return ((value & 0xFF00000000000000ULL) >> 56) |
           ((value & 0x00FF000000000000ULL) >> 40) |
           ((value & 0x0000FF0000000000ULL) >> 24) |
           ((value & 0x000000FF00000000ULL) >> 8)  |
           ((value & 0x00000000FF000000ULL) << 8)  |
           ((value & 0x0000000000FF0000ULL) << 24) |
           ((value & 0x000000000000FF00ULL) << 40) |
           ((value & 0x00000000000000FFULL) << 56);
}

static inline void blake3(const uint8_t *input, int len, uint8_t *output)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, input, len);
    blake3_hasher_finalize(&hasher, output, BLAKE3_OUT_LEN);
}

#if !defined(__x86_64__)
static inline void aes_round(uint8_t *block, const uint8_t *key)
{
#if defined(__aarch64__) && defined(__AES__)
    uint8x16_t blck = vld1q_u8(block);
    uint8x16_t ky   = vld1q_u8(key);
    uint8x16_t rslt = vaesmcq_u8(vaeseq_u8(blck, (uint8x16_t){})) ^ ky;
    vst1q_u8(block, rslt);
#else
    aes_single_round_no_intrinsics(block, key);
#endif
}
#endif

#if defined(__x86_64__)
__attribute__((target("aes")))inline void aes_round(uint8_t *block, const uint8_t *key)
{
    __m128i block_vec = _mm_loadu_si128((const __m128i *)block);
    __m128i key_vec   = _mm_loadu_si128((const __m128i *)key);
    __m128i result    = _mm_aesenc_si128(block_vec, key_vec);
    _mm_storeu_si128((__m128i *)block, result);
}

__attribute__((target("default")))inline void aes_round(uint8_t *block, const uint8_t *key)
{
    aes_single_round_no_intrinsics(block, key);
}
#endif

constexpr uint8_t chaIn[XELIS_MEMORY_SIZE_V3 * 2] = {0};

static void chacha_encrypt(uint8_t *key, uint8_t *nonce, uint8_t *in, uint8_t *out,
                           size_t bytes, uint32_t rounds)
{
    uint8_t state[48] = {0};
    ChaCha20SetKey(state, key);
    ChaCha20SetNonce(state, nonce);
    ChaCha20EncryptBytes(state, in, out, bytes, rounds);
}

static inline void chacha_get_bytes_at_offset(const uint8_t key[32], const uint8_t nonce[12],
                                               size_t offset, uint8_t output[12], int rounds)
{
    uint8_t state[48] = {0};
    uint8_t block[64];

    ChaCha20SetKey(state, key);
    ChaCha20SetNonce(state, nonce);

    uint32_t  block_num = offset / 64;
    uint32_t *counter   = (uint32_t *)(state + 32);
    *counter            = block_num;

    ChaCha20EncryptBytes(state, NULL, block, 64, rounds);

    size_t block_offset = offset % 64;
    memcpy(output, block + block_offset, 12);
}

// ============================================================================
// STAGE 1 - shared key/nonce derivation body
// ============================================================================

static inline void stage_1_derive(const uint8_t *input, size_t input_len,
                                   uint8_t K2_values[4][32], uint8_t nonces[4][12])
{
    constexpr size_t chunk_size      = XELIS_CHUNK_SIZE;
    constexpr size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3;

    uint8_t buffer[64] = {0};
    uint8_t key[128]   = {0};

    memcpy(key, input, input_len);

    blake3(input, input_len, buffer);
    memcpy(nonces[0], buffer, 12);

    for (int i = 0; i < 4; i++)
    {
        if (i == 0) {
            memcpy(buffer + chunk_size, key, chunk_size);
        } else {
            memcpy(buffer, K2_values[i - 1], chunk_size);
            memcpy(buffer + chunk_size, key + i * chunk_size, chunk_size);
        }
        blake3(buffer, chunk_size * 2, K2_values[i]);
    }

    for (int i = 0; i < 3; i++) {
        chacha_get_bytes_at_offset(K2_values[i], nonces[i],
                                   bytes_per_chunk - 12, nonces[i + 1], 8);
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static inline uint64_t murmurhash3(uint64_t seed)
{
    seed ^= seed >> 55;
    seed *= XELIS_MURMUR_CONST1;
    seed ^= seed >> 32;
    seed *= XELIS_MURMUR_CONST2;
    seed ^= seed >> 15;
    return seed;
}

static inline uint64_t map_index(uint64_t x)
{
    x ^= x >> 33;
    x *= XELIS_MURMUR_CONST1;
    return (uint64_t)(((__uint128_t)x * XELIS_BUFFER_SIZE_V3) >> 64);
}

static inline int pick_half(uint64_t seed)
{
    return (murmurhash3(seed) & XELIS_PICK_HALF_BIT) != 0;
}

static inline uint64_t modular_power(uint64_t base, uint64_t exp, uint64_t mod)
{
    mod += (mod == 0);
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1)
            result = (uint64_t)(((__uint128_t)result * base) % mod);
        base = (uint64_t)(((__uint128_t)base * base) % mod);
        exp >>= 1;
    }
    return result;
}

alignas(64) static constexpr uint8_t isqrt_lut[256] = {
    0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 10,10,10,10,10,10,10,10,10,10,10,10,
    10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,
    11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,
    12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,
    13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
    13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,14,
    14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
    14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15};

#ifdef __x86_64__
static inline uint64_t isqrt(uint64_t n)
{
    if (n < 2) return n;
    
    // Initial estimate from bit position
    int lz = __builtin_clzll(n);
    uint64_t x = 1ULL << ((63 - lz + 1) >> 1);
    
    // Newton-Raphson iterations (4 is sufficient for 64-bit)
    x = (x + n/x) >> 1;
    x = (x + n/x) >> 1;
    x = (x + n/x) >> 1;
    x = (x + n/x) >> 1;
    
    // Final correction
    while (x * x > n) --x;
    while ((x + 1) <= n / (x + 1)) ++x;
    return x;
}
#endif

#ifdef __aarch64__
static inline uint64_t isqrt(uint64_t n)
{
    if (n < 2) return n;
    float64x1_t v = vdup_n_f64((double)n);
    float64x1_t s = vsqrt_f64(v);
    double droot;
    vst1_f64(&droot, s);
    uint64_t root = (uint64_t)droot;
    if ((root + 1) * (root + 1) <= n) ++root;
    else if (root * root > n)         --root;
    return root;
}
#endif

static inline __uint128_t combine_uint64(uint64_t high, uint64_t low)
{
    return ((__uint128_t)high << 64) | low;
}

#if defined(__AVX2__)
__attribute__((target("avx2")))void static inline uint64_to_le_bytes(uint64_t value, uint8_t *bytes)
{
    _mm_storel_epi64((__m128i *)bytes,
        _mm_shuffle_epi8(_mm_set1_epi64x(value),
            _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1, 7,6,5,4,3,2,1,0)));
}

__attribute__((target("avx2")))uint64_t static inline le_bytes_to_uint64(const uint8_t *bytes)
{
    return _mm_cvtsi128_si64(
        _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)bytes),
            _mm_set_epi8(15,14,13,12,11,10,9,8, 7,6,5,4,3,2,1,0)));
}
#endif

#if defined(__x86_64__)
__attribute__((target("default")))void static inline uint64_to_le_bytes(uint64_t value, uint8_t *bytes)
{
    for (int i = 0; i < 8; i++) { bytes[i] = value & 0xFF; value >>= 8; }
}

__attribute__((target("default")))uint64_t static inline le_bytes_to_uint64(const uint8_t *bytes)
{
    uint64_t value = 0;
    for (int i = 7; i >= 0; i--) value = (value << 8) | bytes[i];
    return value;
}
#elif defined(__aarch64__)
#if defined(__ARM_NEON) || defined(__aarch64__)
static inline void uint64_to_le_bytes(uint64_t value, uint8_t *bytes)
{
    uint8x8_t val = vreinterpret_u8_u64(vdup_n_u64(value));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    val = vrev64_u8(val);
#endif
    vst1_u8(bytes, val);
}

static inline uint64_t le_bytes_to_uint64(const uint8_t *bytes)
{
    uint8x8_t val = vld1_u8(bytes);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    val = vrev64_u8(val);
#endif
    return vget_lane_u64(vreinterpret_u64_u8(val), 0);
}
#else
void static inline uint64_to_le_bytes(uint64_t value, uint8_t *bytes)
{
    for (int i = 0; i < 8; i++) { bytes[i] = value & 0xFF; value >>= 8; }
}
uint64_t static inline le_bytes_to_uint64(const uint8_t *bytes)
{
    uint64_t value = 0;
    for (int i = 7; i >= 0; i--) value = (value << 8) | bytes[i];
    return value;
}
#endif
#endif

#if defined(__x86_64__)
void static inline aes_single_round(uint8_t *block, const uint8_t *key)
{
    __m128i block_vec = _mm_aesenc_si128(
        _mm_loadu_si128((const __m128i *)block),
        _mm_loadu_si128((const __m128i *)key));
    _mm_storeu_si128((__m128i *)block, block_vec);
}

static inline uint64_t ROTR(uint64_t x, uint32_t r)
{
    asm("rorq %%cl, %0" : "+r"(x) : "c"(r)); return x;
}
static inline uint64_t ROTL(uint64_t x, uint32_t r)
{
    asm("rolq %%cl, %0" : "+r"(x) : "c"(r)); return x;
}
#else
static inline uint64_t ROTR(uint64_t x, uint32_t r) { return __builtin_rotateright64(x, r); }
static inline uint64_t ROTL(uint64_t x, uint32_t r) { return __builtin_rotateleft64(x, r);  }
#endif

static inline uint64_t d128_64(__uint128_t a, uint64_t b)
{
    uint64_t dividend_hi = a >> 64;
    uint64_t dividend_lo = (uint64_t)a;
#if defined(__x86_64__)
    uint64_t q_hi = 0, q_lo = 0, remainder = 0;
    if (dividend_hi < b) {
        __asm__("divq %[divisor]"
                : "=a"(q_lo), "=d"(remainder)
                : [divisor] "r"(b), "a"(dividend_lo), "d"(dividend_hi) : "cc");
    } else {
        __asm__("divq %[divisor]"
                : "=a"(q_hi), "=d"(dividend_hi)
                : [divisor] "r"(b), "a"(dividend_hi), "d"(0ULL) : "cc");
        __asm__("divq %[divisor]"
                : "=a"(q_lo), "=d"(remainder)
                : [divisor] "r"(b), "a"(dividend_lo), "d"(dividend_hi) : "cc");
    }
    return q_lo;
#else
    return (uint64_t)(a / b);
#endif
}

static inline uint64_t mod128_64(__uint128_t dividend, uint64_t divisor)
{
    uint64_t dividend_hi = dividend >> 64;
    uint64_t dividend_lo = (uint64_t)dividend;
#if defined(__x86_64__)
    uint64_t remainder = 0, dummy_quotient;
    if (dividend_hi < divisor) {
        __asm__("divq %[divisor]"
                : "=a"(dummy_quotient), "=d"(remainder)
                : [divisor] "r"(divisor), "a"(dividend_lo), "d"(dividend_hi) : "cc");
    } else {
        __asm__("divq %[divisor]"
                : "=a"(dummy_quotient), "=d"(dividend_hi)
                : [divisor] "r"(divisor), "a"(dividend_hi), "d"(0ULL) : "cc");
        __asm__("divq %[divisor]"
                : "=a"(dummy_quotient), "=d"(remainder)
                : [divisor] "r"(divisor), "a"(dividend_lo), "d"(dividend_hi) : "cc");
    }
    return remainder;
#else
    return (uint64_t)(dividend % divisor);
#endif
}

static inline uint64_t umul64_hi(uint64_t a, uint64_t b)
{
#if defined(_MSC_VER) && defined(_M_X64)
    uint64_t h; (void)_umul128(a, b, &h); return h;
#elif defined(__SIZEOF_INT128__)
    return (uint64_t)(((__uint128_t)a * (__uint128_t)b) >> 64);
#elif defined(__x86_64__)
    uint64_t h, l; __asm__("mulq %3" : "=a"(l), "=d"(h) : "a"(a), "r"(b) : "cc"); return h;
#elif defined(__aarch64__)
    uint64_t h; __asm__("umulh %0, %1, %2" : "=r"(h) : "r"(a), "r"(b)); return h;
#else
#   error "No known way to compute high 64 bits of 64x64 multiply"
#endif
}

static inline uint64_t mod128_64_fast(__uint128_t t1, uint64_t denom)
{
#if defined(__x86_64__)
    uint64_t hi = t1 >> 64;
    uint64_t lo = (uint64_t)t1;
    if (hi < denom) {
        uint64_t remainder;
        __asm__("divq %[d]"
                : "=d"(remainder)
                : [d] "r"(denom), "a"(lo), "d"(hi) : "cc");
        return remainder;
    }
#endif
    return mod128_64(t1, denom);
}

static inline uint64_t hi(__uint128_t x) { return (uint64_t)(x >> 64); }
static inline uint64_t lo(__uint128_t x) { return (uint64_t)x; }

static inline uint64_t udiv(uint64_t high, uint64_t low, uint64_t divisor)
{
    __uint128_t dividend = ((__uint128_t)high << 64) | low;
    return d128_64(dividend, divisor);
}

// Montgomery multiplication avoids division entirely after setup
struct Montgomery64 {
    uint64_t n, n_inv, r2;
    
    static Montgomery64 init(uint64_t mod) {
        Montgomery64 m;
        m.n = mod | 1;
        // Compute -n^(-1) mod 2^64 via Newton iteration
        uint64_t x = m.n;
        for (int i = 0; i < 5; i++) x *= 2 - m.n * x;
        m.n_inv = -x;
        m.r2 = ((__uint128_t)1 << 64) % m.n;
        m.r2 = ((__uint128_t)m.r2 * m.r2) % m.n;
        return m;
    }
    
    uint64_t reduce(__uint128_t t) const {
        uint64_t m = (uint64_t)t * n_inv;
        uint64_t r = (t + (__uint128_t)m * n) >> 64;
        return r >= n ? r - n : r;
    }
    
    uint64_t mul(uint64_t a, uint64_t b) const {
        return reduce((__uint128_t)a * b);
    }
};

// The hot reduction, mu_lo/mu_hi expected to be in registers
static inline uint64_t barrett_core(
    __uint128_t x, uint64_t mod, uint64_t mu_lo, uint64_t mu_hi)
{
    uint64_t x_lo = (uint64_t)x;
    uint64_t x_hi = (uint64_t)(x >> 64);

    uint64_t p0_hi = umul64_hi(x_lo, mu_lo);
    __uint128_t p1 = (__uint128_t)x_lo * mu_hi;
    __uint128_t p2 = (__uint128_t)x_hi * mu_lo;
    __uint128_t p3 = (__uint128_t)x_hi * mu_hi;

    __uint128_t mid = (__uint128_t)p0_hi + (uint64_t)p1 + (uint64_t)p2;

    __uint128_t q = p3 + (p1 >> 64) + (p2 >> 64) + (uint64_t)(mid >> 64);

    // Use 128-bit subtraction to avoid truncation bug
    __uint128_t r = x - q * (__uint128_t)mod;

    // r < 2*mod guaranteed, and mod < 2^64, so r < 2^65
    // After one subtraction, r < mod < 2^64, fits in uint64_t
    if (r >= mod) r -= mod;

    return (uint64_t)r;
}

static inline uint64_t modular_power_fast(uint64_t base, uint64_t exp, uint64_t mod)
{
    mod += (mod == 0);
    base %= mod;
    if (base < 2) return base;

    // Precompute Barrett constant — one division at setup
    __uint128_t mu = (~(__uint128_t)0) / mod;
    uint64_t mu_lo = (uint64_t)mu;
    uint64_t mu_hi = (uint64_t)(mu >> 64);

    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            __uint128_t prod = (__uint128_t)result * base;
            result = barrett_core(prod, mod, mu_lo, mu_hi);
        }
        exp >>= 1;
        if (exp > 0) {
            __uint128_t prod = (__uint128_t)base * base;
            base = barrett_core(prod, mod, mu_lo, mu_hi);
            if (base == 0) return 0;
        }
    }
    return result;
}

// Op dispatch strategy selector
enum class OpDispatch { Switch, Goto };

// ---------- Strategy 1: Switch ----------

__attribute__((noinline))
static uint64_t execute_op_heavy_switch(
    uint32_t op_idx, uint64_t a, uint64_t b, uint64_t c,
    int r_next, uint64_t result, int i, int j_off)
{
    switch (op_idx) {
        case 7:  return mod128_64_fast(COMBINE_UINT64(a,b), c|1);
        case 8:  { uint64_t t2_hi = ROTL(result,r_next), t2_lo = a|2;
                   if (t2_hi>b||(t2_hi==b&&t2_lo>c)) return c;
                   __uint128_t dd=COMBINE_UINT64(b,c), ds=COMBINE_UINT64(t2_hi,t2_lo);
                   uint64_t q=(t2_hi!=0)?(b/t2_hi):0;
                   return (uint64_t)(dd-ds*q); }
        case 9:  return udiv(c,a,b|4);
        case 10: { uint64_t rr=ROTL(result,r_next);
                   return (rr>a||(rr==a&&b>(c|8)))?((a!=0)?(rr/a):0):(a^b); }
        case 13: { __uint128_t t1 = combine_uint64(a+i, isqrt(b+j_off));
                   uint64_t denom = murmurhash3(c^result^i^j_off)|1;
                   return (uint64_t)(t1 % denom); }
        case 14: { uint64_t sb = isqrt(b|2), sa = isqrt(a+j_off);
                   return ROTL((c+i)%sb, i+j_off)*sa; }
        case 15: return (isqrt(a+i)*isqrt(c+j_off))^(b+i+j_off);
        default: __builtin_unreachable();
    }
}

__attribute__((always_inline, hot))
static inline uint64_t execute_operation_switch(
    uint32_t op_idx, uint64_t a, uint64_t b, uint64_t c,
    int r_next, uint64_t result, int i, int j_off)
{
    constexpr uint16_t cheap_mask = 0x187F;  // bits 0-6, 11, 12

    if (__builtin_expect((cheap_mask >> op_idx) & 1, 1)) {
        switch (op_idx) {
            case 0:  return (a + b) * c;
            case 1:  return (b - c) * a;
            case 2:  return c - a + b;
            case 3:  return a - b + c;
            case 4:  return b * c + a;
            case 5:  return c * a + b;
            case 6:  return a * b * c;
            case 11: return umul64_hi(a, c) + b * c;
            case 12: { uint64_t rr = ROTR(result, r_next);
                       return a * b + c * rr + umul64_hi(c, b); }
            default: __builtin_unreachable();
        }
    }
    return execute_op_heavy_switch(op_idx, a, b, c, r_next, result, i, j_off);
}

// ---------- Strategy 2: Goto dispatch ----------

__attribute__((always_inline, hot, aligned(64)))
static inline uint64_t execute_operation_goto(
    uint32_t idx, uint64_t a, uint64_t b, uint64_t c,
    int r_next, uint64_t result, int i, int j_off)
{
    __asm__ volatile ("" ::: "memory");
    static void *const dispatch_table[] = {
        &&op0,&&op1,&&op2,&&op3,&&op4,&&op5,&&op6,&&op7,
        &&op8,&&op9,&&op10,&&op11,&&op12,&&op13,&&op14,&&op15};

    uint64_t v;
    goto *dispatch_table[idx];

op0: { __uint128_t t1 = combine_uint64(a+i, isqrt(b+j_off));
       uint64_t denom = murmurhash3(c^result^i^j_off)|1;
       v = (uint64_t)(t1 % denom); goto done; }
op1: { uint64_t sb = isqrt(b|2), sa = isqrt(a+j_off);
       v = ROTL((c+i)%sb, i+j_off)*sa; goto done; }
op2:   v = (isqrt(a+i)*isqrt(c+j_off))^(b+i+j_off); goto done;
op3:   v = (a+b)*c; goto done;
op4:   v = (b-c)*a; goto done;
op5:   v = c-a+b;   goto done;
op6:   v = a-b+c;   goto done;
op7:   v = b*c+a;   goto done;
op8:   v = c*a+b;   goto done;
op9:   v = a*b*c;   goto done;
op10:  v = mod128_64_fast(COMBINE_UINT64(a,b), c|1); goto done;
op11: { uint64_t t2_hi = ROTL(result,r_next), t2_lo = a|2;
        if (t2_hi>b||(t2_hi==b&&t2_lo>c)) { v=c; }
        else { __uint128_t dd=COMBINE_UINT64(b,c), ds=COMBINE_UINT64(t2_hi,t2_lo);
               uint64_t q=(t2_hi!=0)?(b/t2_hi):0;
               v=(uint64_t)(dd-ds*q); }
        goto done; }
op12:  v = udiv(c,a,b|4); goto done;
op13: { uint64_t rr=ROTL(result,r_next);
        v=(rr>a||(rr==a&&b>(c|8)))?((a!=0)?(rr/a):0):(a^b); goto done; }
op14:  v = umul64_hi(a,c)+b*c; goto done;
op15: { uint64_t rr=ROTR(result,r_next);
        v=a*b+c*rr+umul64_hi(c,b); goto done; }
done:
    return v;
}

// ---------- Unified dispatch ----------

template<OpDispatch D = OpDispatch::Goto>
__attribute__((always_inline, hot))
static inline uint64_t execute_operation(
    uint32_t raw_rotl, uint64_t a, uint64_t b, uint64_t c,
    int r_next, uint64_t result, int i, int j_off)
{
    if constexpr (D == OpDispatch::Switch) {
        uint32_t op_idx = (raw_rotl + 13) & 0xF;
        return execute_operation_switch(op_idx, a, b, c, r_next, result, i, j_off);
    } else {
        uint32_t op_idx = raw_rotl & 0xF;
        return execute_operation_goto(op_idx, a, b, c, r_next, result, i, j_off);
    }
}

// ============================================================================
// Prefetch
// ============================================================================

#ifdef __x86_64__
static inline void prefetch_L1(const void *p) {
    _mm_prefetch((const char *)p, _MM_HINT_T0);
}
static inline void prefetch_L2(const void *p) {
    _mm_prefetch((const char *)p, _MM_HINT_T1);
}
#else
static inline void prefetch_L1(const void *p) {
    __builtin_prefetch(p, 0, 3);
}
static inline void prefetch_L2(const void *p) {
    __builtin_prefetch(p, 0, 2);
}
#endif
static inline void prefetch_none(const void *) {}

#ifdef __x86_64__
static inline void prefetch_W(const void *p) {
    __builtin_prefetch(p, 1, 3);  // PREFETCHW — write-intent, L1
}
#else
static inline void prefetch_W(const void *p) {
    __builtin_prefetch(p, 1, 3);
}
#endif

// ============================================================================
// Shared stage_1 pipelined body (always_inline 4-way ChaCha into tier TU)
// ============================================================================

#define XELIS_STAGE1_INLINE(chacha_fn, POLICY) \
{ \
    constexpr size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3; \
    uint8_t *t = reinterpret_cast<uint8_t *>(sp); \
    uint8_t K2_values[4][32]; \
    uint8_t nonces[4][12]; \
\
    stage_1_derive(input, input_len, K2_values, nonces); \
\
    byte *outputs[4]; \
    for (int i = 0; i < 4; i++) \
        outputs[i] = t + i * bytes_per_chunk; \
    chacha_fn<POLICY>(K2_values, nonces, outputs, bytes_per_chunk, 8); \
}

// ============================================================================
// Shared stage_1 serial body (non-pipelined, sequential per-stream ChaCha)
// ============================================================================

#define XELIS_STAGE1_SERIAL_BODY \
{ \
    constexpr size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3; \
    uint8_t *t = reinterpret_cast<uint8_t *>(sp); \
    uint8_t K2_values[4][32]; \
    uint8_t nonces[4][12]; \
\
    stage_1_derive(input, input_len, K2_values, nonces); \
\
    for (int i = 0; i < 4; i++) \
    { \
        uint8_t state[48] = {0}; \
        ChaCha20SetKey(state, K2_values[i]); \
        ChaCha20SetNonce(state, nonces[i]); \
        ChaCha20EncryptBytes(state, NULL, t + i * bytes_per_chunk, bytes_per_chunk, 8); \
    } \
}

// ============================================================================
// Shared stage_3 body macros
// SOFT: software AES via aes_round() FMV (used by tiers without AES target)
// HW:   inline _mm_aesenc_si128, requires AES-NI in the function target
//
// _PF(PF)          — single prefetch function for lookahead
// _EX(PF,PF_W,PF_S)— lookahead + write-intent stores + scratch_pad read
// Plain versions    — L1 lookahead + scratch_pad[r] read-ahead (default)
// ============================================================================

#define XELIS_STAGE3_SOFT_AES_BODY_PF(PF) \
    XELIS_STAGE3_SOFT_AES_BODY_FULL(PF, prefetch_none, prefetch_none, OpDispatch::Goto)

// Soft AES full: PF=lookahead, PF_W=write-intent, PF_S=scratch read, OPD=dispatch mode
// Also L2-prefetches scratch_pad[r+32] unconditionally.
#define XELIS_STAGE3_SOFT_AES_BODY_FULL(PF, PF_W, PF_S, OPD) \
{ \
    constexpr uint8_t key[17] = "xelishash-pow-v3"; \
    uint8_t block[16]; \
\
    uint64_t *__restrict mem_buffer_a = scratch_pad; \
    uint64_t *__restrict mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V3; \
\
    uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1]; \
    uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32; \
    size_t r = 0; \
\
    for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; ++i) \
    { \
        uint64_t mem_a = mem_buffer_a[map_index(addr_a)]; \
        uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)]; \
\
        alignas(16) uint64_t block_u64[2] = { mem_b, mem_a }; \
        aes_round(reinterpret_cast<uint8_t*>(block_u64), key);  \
        uint64_t hash1 = block_u64[0];  \
        uint64_t hash2 = block_u64[1];  \
        uint64_t result = ~(hash1 ^ hash2); \
\
        uint64_t next_a_idx = map_index(result); \
        PF(&mem_buffer_a[next_a_idx]); \
\
        for (size_t j = 0; j < XELIS_BUFFER_SIZE_V3; ++j) \
        { \
            PF_S(&scratch_pad[r]); \
            prefetch_L2(&scratch_pad[r + 32]); \
            uint64_t rot_res = ~ROTR(result, r); \
            uint64_t a       = mem_buffer_a[next_a_idx]; \
            uint64_t b       = mem_buffer_b[map_index(a ^ rot_res)]; \
            uint64_t c       = scratch_pad[r]; \
            r++; \
\
            uint32_t op_raw = ROTL(result, (uint32_t)c); \
            uint64_t v      = execute_operation<OPD>(op_raw, a, b, c, r, result, i, j); \
\
            uint64_t idx_seed = v ^ result; \
            result            = ROTL(idx_seed, r); \
\
            next_a_idx = map_index(result); \
            PF(&mem_buffer_a[next_a_idx]); \
\
            int      use_b = pick_half(v); \
            uint64_t idx_t = map_index(idx_seed); \
            uint64_t t     = (use_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result; \
\
            uint64_t idx_a = map_index(t ^ result ^ XELIS_GOLDEN_RATIO); \
            uint64_t idx_b = map_index(idx_a ^ ~result ^ XELIS_SCATTER_CONST); \
\
            PF_W(&mem_buffer_a[idx_a]); \
            PF_W(&mem_buffer_b[idx_b]); \
            uint64_t mem_a_tmp = mem_buffer_a[idx_a]; \
            mem_buffer_a[idx_a] = t; \
            mem_buffer_b[idx_b] ^= mem_a_tmp ^ ROTR(t, i + j); \
        } \
\
        uint64_t addr_a_next = modular_power_fast(addr_a, addr_b, result); \
        uint64_t addr_b_next = isqrt(result) * (r + 1) * isqrt(addr_a_next); \
        addr_a = addr_a_next; \
        addr_b = addr_b_next; \
    } \
}

#define XELIS_STAGE3_SOFT_AES_BODY_EX(PF, PF_W, PF_S) \
    XELIS_STAGE3_SOFT_AES_BODY_FULL(PF, PF_W, PF_S, OpDispatch::Goto)

// Default: L1 lookahead + scratch_pad[r] read-ahead (goto dispatch)
#define XELIS_STAGE3_SOFT_AES_BODY XELIS_STAGE3_SOFT_AES_BODY_FULL(prefetch_L1, prefetch_none, prefetch_L1, OpDispatch::Goto)

// Switch dispatch variant
#define XELIS_STAGE3_SOFT_AES_BODY_SWITCH XELIS_STAGE3_SOFT_AES_BODY_FULL(prefetch_L1, prefetch_none, prefetch_L1, OpDispatch::Switch)

#if defined(__x86_64__)
// HW AES full: PF=lookahead, PF_W=write-intent, PF_S=scratch read, OPD=dispatch mode
// Also L2-prefetches scratch_pad[r+32] unconditionally.
#define XELIS_STAGE3_HW_AES_BODY_FULL(PF, PF_W, PF_S, OPD) \
{ \
    constexpr uint8_t key[17] = "xelishash-pow-v3"; \
    __m128i key_vec = _mm_loadu_si128((const __m128i *)key); \
\
    uint64_t *__restrict mem_buffer_a = scratch_pad; \
    uint64_t *__restrict mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V3; \
\
    uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1]; \
    uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32; \
    size_t r = 0; \
\
    for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; ++i) \
    { \
        uint64_t mem_a = mem_buffer_a[map_index(addr_a)]; \
        uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)]; \
\
        __m128i block_vec = _mm_set_epi64x(mem_a, mem_b); \
        block_vec = _mm_aesenc_si128(block_vec, key_vec); \
        uint64_t hash1 = _mm_extract_epi64(block_vec, 0); \
        uint64_t hash2 = _mm_extract_epi64(block_vec, 1); \
        uint64_t result = ~(hash1 ^ hash2); \
\
        uint64_t next_a_idx = map_index(result); \
        PF(&mem_buffer_a[next_a_idx]); \
\
        for (size_t j = 0; j < XELIS_BUFFER_SIZE_V3; ++j) \
        { \
            PF_S(&scratch_pad[r]); \
            prefetch_L2(&scratch_pad[r + 32]); \
            uint64_t rot_res = ~ROTR(result, r); \
            uint64_t a       = mem_buffer_a[next_a_idx]; \
            uint64_t b       = mem_buffer_b[map_index(a ^ rot_res)]; \
            uint64_t c       = scratch_pad[r]; \
            r++; \
\
            uint32_t op_raw = ROTL(result, (uint32_t)c); \
            uint64_t v      = execute_operation<OPD>(op_raw, a, b, c, r, result, i, j); \
\
            uint64_t idx_seed = v ^ result; \
            result            = ROTL(idx_seed, r); \
\
            next_a_idx = map_index(result); \
            PF(&mem_buffer_a[next_a_idx]); \
\
            int      use_b = pick_half(v); \
            uint64_t idx_t = map_index(idx_seed); \
            uint64_t t     = (use_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result; \
\
            uint64_t idx_a = map_index(t ^ result ^ XELIS_GOLDEN_RATIO); \
            uint64_t idx_b = map_index(idx_a ^ ~result ^ XELIS_SCATTER_CONST); \
\
            PF_W(&mem_buffer_a[idx_a]); \
            PF_W(&mem_buffer_b[idx_b]); \
            uint64_t mem_a_tmp = mem_buffer_a[idx_a]; \
            mem_buffer_a[idx_a] = t; \
            mem_buffer_b[idx_b] ^= mem_a_tmp ^ ROTR(t, i + j); \
        } \
\
        uint64_t addr_a_next = modular_power_fast(addr_a, addr_b, result); \
        uint64_t addr_b_next = isqrt(result) * (r + 1) * isqrt(addr_a_next); \
        addr_a = addr_a_next; \
        addr_b = addr_b_next; \
    } \
}

#define XELIS_STAGE3_HW_AES_BODY_PF(PF) \
    XELIS_STAGE3_HW_AES_BODY_FULL(PF, prefetch_none, prefetch_none, OpDispatch::Goto)

#define XELIS_STAGE3_HW_AES_BODY_EX(PF, PF_W, PF_S) \
    XELIS_STAGE3_HW_AES_BODY_FULL(PF, PF_W, PF_S, OpDispatch::Goto)

// Default: L1 lookahead + scratch_pad[r] read-ahead (goto dispatch)
#define XELIS_STAGE3_HW_AES_BODY XELIS_STAGE3_HW_AES_BODY_FULL(prefetch_L1, prefetch_none, prefetch_L1, OpDispatch::Goto)

// Switch dispatch variant
#define XELIS_STAGE3_HW_AES_BODY_SWITCH XELIS_STAGE3_HW_AES_BODY_FULL(prefetch_L1, prefetch_none, prefetch_L1, OpDispatch::Switch)
#endif
