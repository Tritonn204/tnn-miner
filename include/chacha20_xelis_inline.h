// chacha20_xelis_inline.h -- always_inline 4-way ChaCha for xelis tier TUs
// These get hoisted into the per-SIMD-tier text sections at compile time.
#pragma once

#include <immintrin.h>
#include "simd/simd_utils.h"

#ifdef __x86_64__

// ============================================================================
// SSSE3 4-way (128-bit lanes, 4 streams interleaved)
// ============================================================================

__attribute__((target("ssse3")))
static inline __m128i chacha_rotl7_128(__m128i v) {
    return _mm_or_si128(_mm_slli_epi32(v, 7), _mm_srli_epi32(v, 25));
}
__attribute__((target("ssse3")))
static inline __m128i chacha_rotl8_128(__m128i v) {
    return _mm_or_si128(_mm_slli_epi32(v, 8), _mm_srli_epi32(v, 24));
}
__attribute__((target("ssse3")))
static inline __m128i chacha_rotl12_128(__m128i v) {
    return _mm_or_si128(_mm_slli_epi32(v, 12), _mm_srli_epi32(v, 20));
}
__attribute__((target("ssse3")))
static inline __m128i chacha_rotl16_128(__m128i v) {
    return _mm_or_si128(_mm_slli_epi32(v, 16), _mm_srli_epi32(v, 16));
}

template<StorePolicy POLICY>
__attribute__((always_inline, target("ssse3")))
static inline void ChaCha20EncryptXelis_ssse3_inline(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    size_t bytes_per_stream,
    int rounds)
{
    const __m128i const0 = _mm_set1_epi32(0x61707865);
    const __m128i const1 = _mm_set1_epi32(0x3320646e);
    const __m128i const2 = _mm_set1_epi32(0x79622d32);
    const __m128i const3 = _mm_set1_epi32(0x6b206574);

    __m128i k0, k1, k2, k3, k4, k5, k6, k7;
    {
        __m128i key0 = _mm_loadu_si128((const __m128i*)keys[0]);
        __m128i key1 = _mm_loadu_si128((const __m128i*)keys[1]);
        __m128i key2 = _mm_loadu_si128((const __m128i*)keys[2]);
        __m128i key3 = _mm_loadu_si128((const __m128i*)keys[3]);

        __m128i key0b = _mm_loadu_si128((const __m128i*)(keys[0] + 16));
        __m128i key1b = _mm_loadu_si128((const __m128i*)(keys[1] + 16));
        __m128i key2b = _mm_loadu_si128((const __m128i*)(keys[2] + 16));
        __m128i key3b = _mm_loadu_si128((const __m128i*)(keys[3] + 16));

        __m128i t0 = _mm_unpacklo_epi32(key0, key1);
        __m128i t1 = _mm_unpacklo_epi32(key2, key3);
        __m128i t2 = _mm_unpackhi_epi32(key0, key1);
        __m128i t3 = _mm_unpackhi_epi32(key2, key3);

        k0 = _mm_unpacklo_epi64(t0, t1);
        k1 = _mm_unpackhi_epi64(t0, t1);
        k2 = _mm_unpacklo_epi64(t2, t3);
        k3 = _mm_unpackhi_epi64(t2, t3);

        t0 = _mm_unpacklo_epi32(key0b, key1b);
        t1 = _mm_unpacklo_epi32(key2b, key3b);
        t2 = _mm_unpackhi_epi32(key0b, key1b);
        t3 = _mm_unpackhi_epi32(key2b, key3b);

        k4 = _mm_unpacklo_epi64(t0, t1);
        k5 = _mm_unpackhi_epi64(t0, t1);
        k6 = _mm_unpacklo_epi64(t2, t3);
        k7 = _mm_unpackhi_epi64(t2, t3);
    }

    __m128i n0, n1, n2;
    {
        __m128i nonce_bytes[4];
        for (int i = 0; i < 4; i++) {
            nonce_bytes[i] = _mm_loadu_si128((const __m128i*)nonces[i]);
        }

        const __m128i mask = _mm_set_epi32(0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        for (int i = 0; i < 4; i++) {
            nonce_bytes[i] = _mm_and_si128(nonce_bytes[i], mask);
        }

        __m128i t0 = _mm_unpacklo_epi32(nonce_bytes[0], nonce_bytes[1]);
        __m128i t1 = _mm_unpacklo_epi32(nonce_bytes[2], nonce_bytes[3]);
        __m128i t2 = _mm_unpackhi_epi32(nonce_bytes[0], nonce_bytes[1]);
        __m128i t3 = _mm_unpackhi_epi32(nonce_bytes[2], nonce_bytes[3]);

        n0 = _mm_unpacklo_epi64(t0, t1);
        n1 = _mm_unpackhi_epi64(t0, t1);
        n2 = _mm_unpacklo_epi64(t2, t3);
    }

    __m128i counter = _mm_setzero_si128();
    size_t pos = 0;

    while (pos < bytes_per_stream)
    {
        __m128i x0 = const0, x1 = const1, x2 = const2, x3 = const3;
        __m128i x4 = k0, x5 = k1, x6 = k2, x7 = k3;
        __m128i x8 = k4, x9 = k5, x10 = k6, x11 = k7;
        __m128i x12 = counter, x13 = n0, x14 = n1, x15 = n2;

        __m128i s0 = x0, s1 = x1, s2 = x2, s3 = x3;
        __m128i s4 = x4, s5 = x5, s6 = x6, s7 = x7;
        __m128i s8 = x8, s9 = x9, s10 = x10, s11 = x11;
        __m128i s12 = x12, s13 = x13, s14 = x14, s15 = x15;

        for (int i = rounds; i > 0; i -= 2)
        {
            #define QR_SSSE3_INL(a, b, c, d) \
                a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a); \
                d = chacha_rotl16_128(d); \
                c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c); \
                b = chacha_rotl12_128(b); \
                a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a); \
                d = chacha_rotl8_128(d); \
                c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c); \
                b = chacha_rotl7_128(b);

            QR_SSSE3_INL(x0, x4, x8,  x12);
            QR_SSSE3_INL(x1, x5, x9,  x13);
            QR_SSSE3_INL(x2, x6, x10, x14);
            QR_SSSE3_INL(x3, x7, x11, x15);

            QR_SSSE3_INL(x0, x5, x10, x15);
            QR_SSSE3_INL(x1, x6, x11, x12);
            QR_SSSE3_INL(x2, x7, x8,  x13);
            QR_SSSE3_INL(x3, x4, x9,  x14);

            #undef QR_SSSE3_INL
        }

        x0 = _mm_add_epi32(x0, s0);   x1 = _mm_add_epi32(x1, s1);
        x2 = _mm_add_epi32(x2, s2);   x3 = _mm_add_epi32(x3, s3);
        x4 = _mm_add_epi32(x4, s4);   x5 = _mm_add_epi32(x5, s5);
        x6 = _mm_add_epi32(x6, s6);   x7 = _mm_add_epi32(x7, s7);
        x8 = _mm_add_epi32(x8, s8);   x9 = _mm_add_epi32(x9, s9);
        x10 = _mm_add_epi32(x10, s10); x11 = _mm_add_epi32(x11, s11);
        x12 = _mm_add_epi32(x12, s12); x13 = _mm_add_epi32(x13, s13);
        x14 = _mm_add_epi32(x14, s14); x15 = _mm_add_epi32(x15, s15);

        #define TRANSPOSE_STORE_SSSE3_INL(xa, xb, xc, xd, off) \
        { \
            __m128i t0 = _mm_unpacklo_epi32(xa, xb); \
            __m128i t1 = _mm_unpacklo_epi32(xc, xd); \
            __m128i t2 = _mm_unpackhi_epi32(xa, xb); \
            __m128i t3 = _mm_unpackhi_epi32(xc, xd); \
            simd_store_128<POLICY>((__m128i*)(outputs[0] + pos + off), _mm_unpacklo_epi64(t0, t1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[1] + pos + off), _mm_unpackhi_epi64(t0, t1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[2] + pos + off), _mm_unpacklo_epi64(t2, t3)); \
            simd_store_128<POLICY>((__m128i*)(outputs[3] + pos + off), _mm_unpackhi_epi64(t2, t3)); \
        }

        TRANSPOSE_STORE_SSSE3_INL(x0,  x1,  x2,  x3,  0);
        TRANSPOSE_STORE_SSSE3_INL(x4,  x5,  x6,  x7,  16);
        TRANSPOSE_STORE_SSSE3_INL(x8,  x9,  x10, x11, 32);
        TRANSPOSE_STORE_SSSE3_INL(x12, x13, x14, x15, 48);

        #undef TRANSPOSE_STORE_SSSE3_INL

        pos += 64;
        counter = _mm_add_epi32(counter, _mm_set1_epi32(1));
    }

    simd_fence<POLICY>();
}

// ============================================================================
// AVX2 4-way (256-bit lanes, 4 streams x 2 blocks interleaved)
// ============================================================================

__attribute__((target("avx2")))
static inline __m256i chacha_rotl7_256(__m256i v) {
    return _mm256_or_si256(_mm256_slli_epi32(v, 7), _mm256_srli_epi32(v, 25));
}
__attribute__((target("avx2")))
static inline __m256i chacha_rotl8_256(__m256i v) {
    return _mm256_shuffle_epi8(v,
        _mm256_set_epi8(14,13,12,15, 10,9,8,11, 6,5,4,7, 2,1,0,3,
                        14,13,12,15, 10,9,8,11, 6,5,4,7, 2,1,0,3));
}
__attribute__((target("avx2")))
static inline __m256i chacha_rotl12_256(__m256i v) {
    return _mm256_or_si256(_mm256_slli_epi32(v, 12), _mm256_srli_epi32(v, 20));
}
__attribute__((target("avx2")))
static inline __m256i chacha_rotl16_256(__m256i v) {
    return _mm256_shuffle_epi8(v,
        _mm256_set_epi8(13,12,15,14, 9,8,11,10, 5,4,7,6, 1,0,3,2,
                        13,12,15,14, 9,8,11,10, 5,4,7,6, 1,0,3,2));
}

template<StorePolicy POLICY>
__attribute__((always_inline, target("avx2")))
static inline void ChaCha20EncryptXelis_avx2_inline(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    const size_t bytes_per_stream,
    int rounds)
{
    const __m256i const0 = _mm256_set1_epi32(0x61707865);
    const __m256i const1 = _mm256_set1_epi32(0x3320646e);
    const __m256i const2 = _mm256_set1_epi32(0x79622d32);
    const __m256i const3 = _mm256_set1_epi32(0x6b206574);

    __m256i k0, k1, k2, k3, k4, k5, k6, k7;
    {
        __m128i key0_lo = _mm_loadu_si128((const __m128i*)keys[0]);
        __m128i key1_lo = _mm_loadu_si128((const __m128i*)keys[1]);
        __m128i key2_lo = _mm_loadu_si128((const __m128i*)keys[2]);
        __m128i key3_lo = _mm_loadu_si128((const __m128i*)keys[3]);

        __m128i key0_hi = _mm_loadu_si128((const __m128i*)(keys[0] + 16));
        __m128i key1_hi = _mm_loadu_si128((const __m128i*)(keys[1] + 16));
        __m128i key2_hi = _mm_loadu_si128((const __m128i*)(keys[2] + 16));
        __m128i key3_hi = _mm_loadu_si128((const __m128i*)(keys[3] + 16));

        __m128i t0 = _mm_unpacklo_epi32(key0_lo, key1_lo);
        __m128i t1 = _mm_unpacklo_epi32(key2_lo, key3_lo);
        __m128i t2 = _mm_unpackhi_epi32(key0_lo, key1_lo);
        __m128i t3 = _mm_unpackhi_epi32(key2_lo, key3_lo);

        __m128i s0 = _mm_unpacklo_epi64(t0, t1);
        __m128i s1 = _mm_unpackhi_epi64(t0, t1);
        __m128i s2 = _mm_unpacklo_epi64(t2, t3);
        __m128i s3 = _mm_unpackhi_epi64(t2, t3);

        t0 = _mm_unpacklo_epi32(key0_hi, key1_hi);
        t1 = _mm_unpacklo_epi32(key2_hi, key3_hi);
        t2 = _mm_unpackhi_epi32(key0_hi, key1_hi);
        t3 = _mm_unpackhi_epi32(key2_hi, key3_hi);

        __m128i s4 = _mm_unpacklo_epi64(t0, t1);
        __m128i s5 = _mm_unpackhi_epi64(t0, t1);
        __m128i s6 = _mm_unpacklo_epi64(t2, t3);
        __m128i s7 = _mm_unpackhi_epi64(t2, t3);

        k0 = _mm256_insertf128_si256(_mm256_castsi128_si256(s0), s0, 1);
        k1 = _mm256_insertf128_si256(_mm256_castsi128_si256(s1), s1, 1);
        k2 = _mm256_insertf128_si256(_mm256_castsi128_si256(s2), s2, 1);
        k3 = _mm256_insertf128_si256(_mm256_castsi128_si256(s3), s3, 1);
        k4 = _mm256_insertf128_si256(_mm256_castsi128_si256(s4), s4, 1);
        k5 = _mm256_insertf128_si256(_mm256_castsi128_si256(s5), s5, 1);
        k6 = _mm256_insertf128_si256(_mm256_castsi128_si256(s6), s6, 1);
        k7 = _mm256_insertf128_si256(_mm256_castsi128_si256(s7), s7, 1);
    }

    __m256i n0, n1, n2;
    {
        __m256i nonces01 = _mm256_loadu2_m128i(
            (const __m128i*)nonces[1], (const __m128i*)nonces[0]);
        __m256i nonces23 = _mm256_loadu2_m128i(
            (const __m128i*)nonces[3], (const __m128i*)nonces[2]);

        const __m256i mask = _mm256_set_epi32(0, -1, -1, -1, 0, -1, -1, -1);
        nonces01 = _mm256_and_si256(nonces01, mask);
        nonces23 = _mm256_and_si256(nonces23, mask);

        __m256i nonces02 = _mm256_permute2x128_si256(nonces01, nonces23, 0x20);
        __m256i nonces13 = _mm256_permute2x128_si256(nonces01, nonces23, 0x31);

        __m256i t0 = _mm256_unpacklo_epi32(nonces02, nonces13);
        __m256i t1 = _mm256_unpackhi_epi32(nonces02, nonces13);

        const __m256i idx_lo = _mm256_setr_epi32(0, 1, 4, 5, 0, 1, 4, 5);
        const __m256i idx_hi = _mm256_setr_epi32(2, 3, 6, 7, 2, 3, 6, 7);

        n0 = _mm256_permutevar8x32_epi32(t0, idx_lo);
        n1 = _mm256_permutevar8x32_epi32(t0, idx_hi);
        n2 = _mm256_permutevar8x32_epi32(t1, idx_lo);
    }

    size_t iterations = bytes_per_stream / 128;
    uint32_t counter_base = 0;

    for (size_t iter = 0; iter < iterations; iter++)
    {
        __m256i counter = _mm256_add_epi32(
            _mm256_set1_epi32(counter_base),
            _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0));

        __m256i x0 = const0, x1 = const1, x2 = const2, x3 = const3;
        __m256i x4 = k0, x5 = k1, x6 = k2, x7 = k3;
        __m256i x8 = k4, x9 = k5, x10 = k6, x11 = k7;
        __m256i x12 = counter, x13 = n0, x14 = n1, x15 = n2;

        const __m256i s0 = x0, s1 = x1, s2 = x2, s3 = x3;
        const __m256i s4 = x4, s5 = x5, s6 = x6, s7 = x7;
        const __m256i s8 = x8, s9 = x9, s10 = x10, s11 = x11;
        const __m256i s12 = x12, s13 = x13, s14 = x14, s15 = x15;

        for (int i = rounds; i > 0; i -= 2)
        {
            #define QR_AVX2_INL(a, b, c, d) \
                a = _mm256_add_epi32(a, b); d = _mm256_xor_si256(d, a); \
                d = chacha_rotl16_256(d); \
                c = _mm256_add_epi32(c, d); b = _mm256_xor_si256(b, c); \
                b = chacha_rotl12_256(b); \
                a = _mm256_add_epi32(a, b); d = _mm256_xor_si256(d, a); \
                d = chacha_rotl8_256(d); \
                c = _mm256_add_epi32(c, d); b = _mm256_xor_si256(b, c); \
                b = chacha_rotl7_256(b);

            QR_AVX2_INL(x0, x4, x8,  x12); QR_AVX2_INL(x1, x5, x9,  x13);
            QR_AVX2_INL(x2, x6, x10, x14); QR_AVX2_INL(x3, x7, x11, x15);
            QR_AVX2_INL(x0, x5, x10, x15); QR_AVX2_INL(x1, x6, x11, x12);
            QR_AVX2_INL(x2, x7, x8,  x13); QR_AVX2_INL(x3, x4, x9,  x14);

            #undef QR_AVX2_INL
        }

        x0 = _mm256_add_epi32(x0, s0);   x1 = _mm256_add_epi32(x1, s1);
        x2 = _mm256_add_epi32(x2, s2);   x3 = _mm256_add_epi32(x3, s3);
        x4 = _mm256_add_epi32(x4, s4);   x5 = _mm256_add_epi32(x5, s5);
        x6 = _mm256_add_epi32(x6, s6);   x7 = _mm256_add_epi32(x7, s7);
        x8 = _mm256_add_epi32(x8, s8);   x9 = _mm256_add_epi32(x9, s9);
        x10 = _mm256_add_epi32(x10, s10); x11 = _mm256_add_epi32(x11, s11);
        x12 = _mm256_add_epi32(x12, s12); x13 = _mm256_add_epi32(x13, s13);
        x14 = _mm256_add_epi32(x14, s14); x15 = _mm256_add_epi32(x15, s15);

        #define TRANSPOSE_STORE_AVX2_INL(xa, xb, xc, xd, off0, off1) \
        { \
            __m256i t0 = _mm256_unpacklo_epi32(xa, xb); \
            __m256i t1 = _mm256_unpackhi_epi32(xa, xb); \
            __m256i t2 = _mm256_unpacklo_epi32(xc, xd); \
            __m256i t3 = _mm256_unpackhi_epi32(xc, xd); \
            __m256i u0 = _mm256_unpacklo_epi64(t0, t2); \
            __m256i u1 = _mm256_unpackhi_epi64(t0, t2); \
            __m256i u2 = _mm256_unpacklo_epi64(t1, t3); \
            __m256i u3 = _mm256_unpackhi_epi64(t1, t3); \
            simd_store_128<POLICY>((__m128i*)(outputs[0] + off0), _mm256_extracti128_si256(u0, 0)); \
            simd_store_128<POLICY>((__m128i*)(outputs[0] + off1), _mm256_extracti128_si256(u0, 1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[1] + off0), _mm256_extracti128_si256(u1, 0)); \
            simd_store_128<POLICY>((__m128i*)(outputs[1] + off1), _mm256_extracti128_si256(u1, 1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[2] + off0), _mm256_extracti128_si256(u2, 0)); \
            simd_store_128<POLICY>((__m128i*)(outputs[2] + off1), _mm256_extracti128_si256(u2, 1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[3] + off0), _mm256_extracti128_si256(u3, 0)); \
            simd_store_128<POLICY>((__m128i*)(outputs[3] + off1), _mm256_extracti128_si256(u3, 1)); \
        }

        TRANSPOSE_STORE_AVX2_INL(x0,  x1,  x2,  x3,  0,  64);
        TRANSPOSE_STORE_AVX2_INL(x4,  x5,  x6,  x7,  16, 80);
        TRANSPOSE_STORE_AVX2_INL(x8,  x9,  x10, x11, 32, 96);
        TRANSPOSE_STORE_AVX2_INL(x12, x13, x14, x15, 48, 112);

        #undef TRANSPOSE_STORE_AVX2_INL

        outputs[0] += 128;
        outputs[1] += 128;
        outputs[2] += 128;
        outputs[3] += 128;
        counter_base += 2;
    }

    simd_fence<POLICY>();
}

// ============================================================================
// AVX512 4-way (512-bit lanes, 4 streams x 4 blocks interleaved)
// Native rotates via _mm512_rol_epi32.
// ============================================================================

template<StorePolicy POLICY>
__attribute__((always_inline, target("avx512f,avx512dq,avx512bw")))
static inline void ChaCha20EncryptXelis_avx512_inline(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    const size_t bytes_per_stream,
    int rounds)
{
    const __m512i const0 = _mm512_set1_epi32(0x61707865);
    const __m512i const1 = _mm512_set1_epi32(0x3320646e);
    const __m512i const2 = _mm512_set1_epi32(0x79622d32);
    const __m512i const3 = _mm512_set1_epi32(0x6b206574);

    // Load and transpose keys: k# = same key word for 4 streams, broadcast to 4 lanes
    __m512i k0, k1, k2, k3, k4, k5, k6, k7;
    {
        __m128i key0 = _mm_loadu_si128((const __m128i*)keys[0]);
        __m128i key1 = _mm_loadu_si128((const __m128i*)keys[1]);
        __m128i key2 = _mm_loadu_si128((const __m128i*)keys[2]);
        __m128i key3 = _mm_loadu_si128((const __m128i*)keys[3]);

        __m128i key0_hi = _mm_loadu_si128((const __m128i*)(keys[0] + 16));
        __m128i key1_hi = _mm_loadu_si128((const __m128i*)(keys[1] + 16));
        __m128i key2_hi = _mm_loadu_si128((const __m128i*)(keys[2] + 16));
        __m128i key3_hi = _mm_loadu_si128((const __m128i*)(keys[3] + 16));

        __m128i t0 = _mm_unpacklo_epi32(key0, key1);
        __m128i t1 = _mm_unpacklo_epi32(key2, key3);
        __m128i t2 = _mm_unpackhi_epi32(key0, key1);
        __m128i t3 = _mm_unpackhi_epi32(key2, key3);

        k0 = _mm512_broadcast_i32x4(_mm_unpacklo_epi64(t0, t1));
        k1 = _mm512_broadcast_i32x4(_mm_unpackhi_epi64(t0, t1));
        k2 = _mm512_broadcast_i32x4(_mm_unpacklo_epi64(t2, t3));
        k3 = _mm512_broadcast_i32x4(_mm_unpackhi_epi64(t2, t3));

        t0 = _mm_unpacklo_epi32(key0_hi, key1_hi);
        t1 = _mm_unpacklo_epi32(key2_hi, key3_hi);
        t2 = _mm_unpackhi_epi32(key0_hi, key1_hi);
        t3 = _mm_unpackhi_epi32(key2_hi, key3_hi);

        k4 = _mm512_broadcast_i32x4(_mm_unpacklo_epi64(t0, t1));
        k5 = _mm512_broadcast_i32x4(_mm_unpackhi_epi64(t0, t1));
        k6 = _mm512_broadcast_i32x4(_mm_unpacklo_epi64(t2, t3));
        k7 = _mm512_broadcast_i32x4(_mm_unpackhi_epi64(t2, t3));
    }

    // Load and transpose nonces
    __m512i n0, n1, n2;
    {
        uint32_t nonce_words[4][3];
        for (int i = 0; i < 4; i++)
            memcpy(nonce_words[i], nonces[i], 12);

        n0 = _mm512_broadcast_i32x4(_mm_setr_epi32(
            nonce_words[0][0], nonce_words[1][0], nonce_words[2][0], nonce_words[3][0]));
        n1 = _mm512_broadcast_i32x4(_mm_setr_epi32(
            nonce_words[0][1], nonce_words[1][1], nonce_words[2][1], nonce_words[3][1]));
        n2 = _mm512_broadcast_i32x4(_mm_setr_epi32(
            nonce_words[0][2], nonce_words[1][2], nonce_words[2][2], nonce_words[3][2]));
    }

    // Counter offsets: 4 blocks, same counter within each 128-bit lane
    // Lane 0 = block +0, lane 1 = block +1, lane 2 = block +2, lane 3 = block +3
    const __m512i ctr_inc = _mm512_setr_epi32(
        0, 0, 0, 0,   // lane 0: block 0
        1, 1, 1, 1,   // lane 1: block 1
        2, 2, 2, 2,   // lane 2: block 2
        3, 3, 3, 3    // lane 3: block 3
    );

    size_t iterations = bytes_per_stream / 256;  // 4 blocks x 64 bytes per stream
    uint32_t counter_base = 0;

    for (size_t iter = 0; iter < iterations; iter++)
    {
        __m512i counter = _mm512_add_epi32(_mm512_set1_epi32(counter_base), ctr_inc);

        __m512i x0 = const0, x1 = const1, x2 = const2, x3 = const3;
        __m512i x4 = k0, x5 = k1, x6 = k2, x7 = k3;
        __m512i x8 = k4, x9 = k5, x10 = k6, x11 = k7;
        __m512i x12 = counter, x13 = n0, x14 = n1, x15 = n2;

        const __m512i s0  = x0,  s1  = x1,  s2  = x2,  s3  = x3;
        const __m512i s4  = x4,  s5  = x5,  s6  = x6,  s7  = x7;
        const __m512i s8  = x8,  s9  = x9,  s10 = x10, s11 = x11;
        const __m512i s12 = x12, s13 = x13, s14 = x14, s15 = x15;

        for (int i = rounds; i > 0; i -= 2)
        {
            #define QR_AVX512_INL(a, b, c, d) \
                a = _mm512_add_epi32(a, b); d = _mm512_xor_si512(d, a); d = _mm512_rol_epi32(d, 16); \
                c = _mm512_add_epi32(c, d); b = _mm512_xor_si512(b, c); b = _mm512_rol_epi32(b, 12); \
                a = _mm512_add_epi32(a, b); d = _mm512_xor_si512(d, a); d = _mm512_rol_epi32(d, 8);  \
                c = _mm512_add_epi32(c, d); b = _mm512_xor_si512(b, c); b = _mm512_rol_epi32(b, 7);

            // Column round
            QR_AVX512_INL(x0, x4, x8,  x12); QR_AVX512_INL(x1, x5, x9,  x13);
            QR_AVX512_INL(x2, x6, x10, x14); QR_AVX512_INL(x3, x7, x11, x15);
            // Diagonal round
            QR_AVX512_INL(x0, x5, x10, x15); QR_AVX512_INL(x1, x6, x11, x12);
            QR_AVX512_INL(x2, x7, x8,  x13); QR_AVX512_INL(x3, x4, x9,  x14);

            #undef QR_AVX512_INL
        }

        x0  = _mm512_add_epi32(x0,  s0);  x1  = _mm512_add_epi32(x1,  s1);
        x2  = _mm512_add_epi32(x2,  s2);  x3  = _mm512_add_epi32(x3,  s3);
        x4  = _mm512_add_epi32(x4,  s4);  x5  = _mm512_add_epi32(x5,  s5);
        x6  = _mm512_add_epi32(x6,  s6);  x7  = _mm512_add_epi32(x7,  s7);
        x8  = _mm512_add_epi32(x8,  s8);  x9  = _mm512_add_epi32(x9,  s9);
        x10 = _mm512_add_epi32(x10, s10); x11 = _mm512_add_epi32(x11, s11);
        x12 = _mm512_add_epi32(x12, s12); x13 = _mm512_add_epi32(x13, s13);
        x14 = _mm512_add_epi32(x14, s14); x15 = _mm512_add_epi32(x15, s15);

        // Transpose via unpack (operates per 128-bit lane), then extract 4 lanes.
        // Each group of 4 state words (xa,xb,xc,xd) produces 16 bytes per stream per block.
        // off0..off3 = byte offsets within the 256-byte output for blocks 0..3.
        #define TRANSPOSE_STORE_AVX512_INL(xa, xb, xc, xd, off0, off1, off2, off3) \
        { \
            __m512i t0 = _mm512_unpacklo_epi32(xa, xb); \
            __m512i t1 = _mm512_unpackhi_epi32(xa, xb); \
            __m512i t2 = _mm512_unpacklo_epi32(xc, xd); \
            __m512i t3 = _mm512_unpackhi_epi32(xc, xd); \
            __m512i u0 = _mm512_unpacklo_epi64(t0, t2); \
            __m512i u1 = _mm512_unpackhi_epi64(t0, t2); \
            __m512i u2 = _mm512_unpacklo_epi64(t1, t3); \
            __m512i u3 = _mm512_unpackhi_epi64(t1, t3); \
            simd_store_128<POLICY>((__m128i*)(outputs[0] + off0), _mm512_castsi512_si128(u0)); \
            simd_store_128<POLICY>((__m128i*)(outputs[0] + off1), _mm512_extracti32x4_epi32(u0, 1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[0] + off2), _mm512_extracti32x4_epi32(u0, 2)); \
            simd_store_128<POLICY>((__m128i*)(outputs[0] + off3), _mm512_extracti32x4_epi32(u0, 3)); \
            simd_store_128<POLICY>((__m128i*)(outputs[1] + off0), _mm512_castsi512_si128(u1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[1] + off1), _mm512_extracti32x4_epi32(u1, 1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[1] + off2), _mm512_extracti32x4_epi32(u1, 2)); \
            simd_store_128<POLICY>((__m128i*)(outputs[1] + off3), _mm512_extracti32x4_epi32(u1, 3)); \
            simd_store_128<POLICY>((__m128i*)(outputs[2] + off0), _mm512_castsi512_si128(u2)); \
            simd_store_128<POLICY>((__m128i*)(outputs[2] + off1), _mm512_extracti32x4_epi32(u2, 1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[2] + off2), _mm512_extracti32x4_epi32(u2, 2)); \
            simd_store_128<POLICY>((__m128i*)(outputs[2] + off3), _mm512_extracti32x4_epi32(u2, 3)); \
            simd_store_128<POLICY>((__m128i*)(outputs[3] + off0), _mm512_castsi512_si128(u3)); \
            simd_store_128<POLICY>((__m128i*)(outputs[3] + off1), _mm512_extracti32x4_epi32(u3, 1)); \
            simd_store_128<POLICY>((__m128i*)(outputs[3] + off2), _mm512_extracti32x4_epi32(u3, 2)); \
            simd_store_128<POLICY>((__m128i*)(outputs[3] + off3), _mm512_extracti32x4_epi32(u3, 3)); \
        }

        TRANSPOSE_STORE_AVX512_INL(x0,  x1,  x2,  x3,  0,   64,  128, 192);
        TRANSPOSE_STORE_AVX512_INL(x4,  x5,  x6,  x7,  16,  80,  144, 208);
        TRANSPOSE_STORE_AVX512_INL(x8,  x9,  x10, x11, 32,  96,  160, 224);
        TRANSPOSE_STORE_AVX512_INL(x12, x13, x14, x15, 48,  112, 176, 240);

        #undef TRANSPOSE_STORE_AVX512_INL

        outputs[0] += 256;
        outputs[1] += 256;
        outputs[2] += 256;
        outputs[3] += 256;
        counter_base += 4;
    }

    simd_fence<POLICY>();
}

#endif // __x86_64__
