#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cstring>
#include <immintrin.h>

// --- Scalar BlaMka (fBlaMka) macros from Lyra PHC team ---
static inline uint64_t fBlaMka(uint64_t x, uint64_t y) {
    const uint64_t m = UINT64_C(0xFFFFFFFF);
    uint64_t xy = (x & m) * (y & m);
    return x + y + 2 * xy;
}

#define G(a, b, c, d)                                          \
    do {                                                      \
        a = fBlaMka(a, b);                                    \
        d = (d ^ a) >> 32 | (d ^ a) << (64 - 32);             \
        c = fBlaMka(c, d);                                    \
        b = (b ^ c) >> 24 | (b ^ c) << (64 - 24);             \
        a = fBlaMka(a, b);                                    \
        d = (d ^ a) >> 16 | (d ^ a) << (64 - 16);             \
        c = fBlaMka(c, d);                                    \
        b = (b ^ c) >> 63 | (b ^ c) << (64 - 63);             \
    } while (0)

#define G1(a, b, c, d)                                          \
    do {                                                      \
        a = fBlaMka(a, b);                                    \
        d = (d ^ a) >> 32 | (d ^ a) << (64 - 32);             \
        c = fBlaMka(c, d);                                    \
        b = (b ^ c) >> 24 | (b ^ c) << (64 - 24);             \
    } while (0)

#define G2(a, b, c, d)                                          \
    do {                                                      \
        a = fBlaMka(a, b);                                    \
        d = (d ^ a) >> 16 | (d ^ a) << (64 - 16);             \
        c = fBlaMka(c, d);                                    \
        b = (b ^ c) >> 63 | (b ^ c) << (64 - 63);             \
    } while (0)

#define BLAKE2_ROUND_NOMSG(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15) \
    do {                                                                         \
        G(v0, v4, v8,  v12);                                                     \
        G(v1, v5, v9,  v13);                                                     \
        G(v2, v6, v10, v14);                                                     \
        G(v3, v7, v11, v15);                                                     \
        G(v0, v5, v10, v15);                                                     \
        G(v1, v6, v11, v12);                                                     \
        G(v2, v7, v8,  v13);                                                     \
        G(v3, v4, v9,  v14);                                                     \
    } while (0)

// Print scalar 16-word state
void print_state16(const char *label, uint64_t v[16]) {
    printf("--- %s ---\n", label);
    for (int i = 0; i < 16; i++) {
        printf("v%2d=0x%016llx%s", i,
               (unsigned long long)v[i],
               (i % 4 == 3) ? "\n" : ", ");
    }
    printf("----------------\n");
}

#define rotr32(x)   _mm256_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24(x)   _mm256_shuffle_epi8(x, _mm256_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, 3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10))
#define rotr16(x)   _mm256_shuffle_epi8(x, _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9))
#define rotr63(x)   _mm256_xor_si256(_mm256_srli_epi64((x), 63), _mm256_add_epi64((x), (x)))

#define rotr32_512(x)   _mm512_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24_512(x)   _mm512_shuffle_epi8(x, _mm512_set_epi8( \
    10, 9, 8, 15, 14, 13, 12, 11, 2, 1, 0, 7, 6, 5, 4, 3, \
    10, 9, 8, 15, 14, 13, 12, 11, 2, 1, 0, 7, 6, 5, 4, 3, \
    10, 9, 8, 15, 14, 13, 12, 11, 2, 1, 0, 7, 6, 5, 4, 3, \
    10, 9, 8, 15, 14, 13, 12, 11, 2, 1, 0, 7, 6, 5, 4, 3))
#define rotr16_512(x)   _mm512_shuffle_epi8(x, _mm512_set_epi8( \
    9, 8, 15, 14, 13, 12, 11, 10, 1, 0, 7, 6, 5, 4, 3, 2, \
    9, 8, 15, 14, 13, 12, 11, 10, 1, 0, 7, 6, 5, 4, 3, 2, \
    9, 8, 15, 14, 13, 12, 11, 10, 1, 0, 7, 6, 5, 4, 3, 2, \
    9, 8, 15, 14, 13, 12, 11, 10, 1, 0, 7, 6, 5, 4, 3, 2))
#define rotr63_512(x)   _mm512_xor_si512(_mm512_srli_epi64((x), 63), _mm512_add_epi64((x), (x)))

// Quarter-round with multiply-add (BlaMka) on two parallel lanes
#define G1_AVX2(A0, A1, B0, B1, C0, C1, D0, D1)       \
    do {                                             \
        __m256i ml = _mm256_mul_epu32(A0, B0);       \
        ml = _mm256_add_epi64(ml, ml);              \
        A0 = _mm256_add_epi64(A0, _mm256_add_epi64(B0, ml)); \
        D0 = _mm256_xor_si256(D0, A0);               \
        D0 = rotr32(D0);                             \
                                                    \
        ml = _mm256_mul_epu32(C0, D0);               \
        ml = _mm256_add_epi64(ml, ml);               \
        C0 = _mm256_add_epi64(C0, _mm256_add_epi64(D0, ml)); \
                                                    \
        B0 = _mm256_xor_si256(B0, C0);               \
        B0 = rotr24(B0);                             \
                                                    \
        ml = _mm256_mul_epu32(A1, B1);               \
        ml = _mm256_add_epi64(ml, ml);               \
        A1 = _mm256_add_epi64(A1, _mm256_add_epi64(B1, ml)); \
        D1 = _mm256_xor_si256(D1, A1);               \
        D1 = rotr32(D1);                             \
                                                    \
        ml = _mm256_mul_epu32(C1, D1);               \
        ml = _mm256_add_epi64(ml, ml);               \
        C1 = _mm256_add_epi64(C1, _mm256_add_epi64(D1, ml)); \
                                                    \
        B1 = _mm256_xor_si256(B1, C1);               \
        B1 = rotr24(B1);                             \
    } while((void)0, 0);

#define G2_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
    do { \
        __m256i ml = _mm256_mul_epu32(A0, B0); \
        ml = _mm256_add_epi64(ml, ml); \
        A0 = _mm256_add_epi64(A0, _mm256_add_epi64(B0, ml)); \
        D0 = _mm256_xor_si256(D0, A0); \
        D0 = rotr16(D0); \
        \
        ml = _mm256_mul_epu32(C0, D0); \
        ml = _mm256_add_epi64(ml, ml); \
        C0 = _mm256_add_epi64(C0, _mm256_add_epi64(D0, ml)); \
        B0 = _mm256_xor_si256(B0, C0); \
        B0 = rotr63(B0); \
        \
        ml = _mm256_mul_epu32(A1, B1); \
        ml = _mm256_add_epi64(ml, ml); \
        A1 = _mm256_add_epi64(A1, _mm256_add_epi64(B1, ml)); \
        D1 = _mm256_xor_si256(D1, A1); \
        D1 = rotr16(D1); \
        \
        ml = _mm256_mul_epu32(C1, D1); \
        ml = _mm256_add_epi64(ml, ml); \
        C1 = _mm256_add_epi64(C1, _mm256_add_epi64(D1, ml)); \
        B1 = _mm256_xor_si256(B1, C1); \
        B1 = rotr63(B1); \
    } while((void)0, 0);

#define DIAGONALIZE_1(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        B0 = _mm256_permute4x64_epi64(B0, _MM_SHUFFLE(0, 3, 2, 1)); \
        C0 = _mm256_permute4x64_epi64(C0, _MM_SHUFFLE(1, 0, 3, 2)); \
        D0 = _mm256_permute4x64_epi64(D0, _MM_SHUFFLE(2, 1, 0, 3)); \
        \
        B1 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(0, 3, 2, 1)); \
        C1 = _mm256_permute4x64_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
        D1 = _mm256_permute4x64_epi64(D1, _MM_SHUFFLE(2, 1, 0, 3)); \
    } while((void)0, 0);

#define DIAGONALIZE_2(A0, A1, B0, B1, C0, C1, D0, D1) \
    do { \
        __m256i tmp1 = _mm256_blend_epi32(B0, B1, 0xCC); \
        __m256i tmp2 = _mm256_blend_epi32(B0, B1, 0x33); \
        B1 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        B0 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
        \
        tmp1 = C0; \
        C0 = C1; \
        C1 = tmp1; \
        \
        tmp1 = _mm256_blend_epi32(D0, D1, 0xCC); \
        tmp2 = _mm256_blend_epi32(D0, D1, 0x33); \
        D0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        D1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
    } while((void)0, 0);

#define UNDIAGONALIZE_1(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        B0 = _mm256_permute4x64_epi64(B0, _MM_SHUFFLE(2, 1, 0, 3)); \
        C0 = _mm256_permute4x64_epi64(C0, _MM_SHUFFLE(1, 0, 3, 2)); \
        D0 = _mm256_permute4x64_epi64(D0, _MM_SHUFFLE(0, 3, 2, 1)); \
        \
        B1 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(2, 1, 0, 3)); \
        C1 = _mm256_permute4x64_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
        D1 = _mm256_permute4x64_epi64(D1, _MM_SHUFFLE(0, 3, 2, 1)); \
    } while((void)0, 0);

#define UNDIAGONALIZE_2(A0, A1, B0, B1, C0, C1, D0, D1) \
    do { \
        __m256i tmp1 = _mm256_blend_epi32(B0, B1, 0xCC); \
        __m256i tmp2 = _mm256_blend_epi32(B0, B1, 0x33); \
        B0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        B1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
        \
        tmp1 = C0; \
        C0 = C1; \
        C1 = tmp1; \
        \
        tmp1 = _mm256_blend_epi32(D0, D1, 0x33); \
        tmp2 = _mm256_blend_epi32(D0, D1, 0xCC); \
        D0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        D1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
    } while((void)0, 0);

#define BLAKE2_ROUND_1(A0, A1, B0, B1, C0, C1, D0, D1) \
    do{ \
        G1_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
        G2_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        DIAGONALIZE_1(A0, B0, C0, D0, A1, B1, C1, D1) \
        \
        G1_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
        G2_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        UNDIAGONALIZE_1(A0, B0, C0, D0, A1, B1, C1, D1) \
    } while((void)0, 0);

#define BLAKE2_ROUND_2(A0, A1, B0, B1, C0, C1, D0, D1) \
    do{ \
        G1_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
        G2_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        DIAGONALIZE_2(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        G1_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
        G2_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        UNDIAGONALIZE_2(A0, A1, B0, B1, C0, C1, D0, D1) \
    } while((void)0, 0);

#define BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, block, offset1, offset2) \
    do { \
        __m256i a_low = _mm256_loadu_si256((__m256i*)&(block)[(offset1)]); \
        __m256i a_high = _mm256_loadu_si256((__m256i*)&(block)[(offset2)]); \
        (A) = _mm512_inserti64x4(_mm512_castsi256_si512(a_low), a_high, 1); \
        \
        __m256i b_low = _mm256_loadu_si256((__m256i*)&(block)[(offset1) + 4]); \
        __m256i b_high = _mm256_loadu_si256((__m256i*)&(block)[(offset2) + 4]); \
        (B) = _mm512_inserti64x4(_mm512_castsi256_si512(b_low), b_high, 1); \
        \
        __m256i c_low = _mm256_loadu_si256((__m256i*)&(block)[(offset1) + 8]); \
        __m256i c_high = _mm256_loadu_si256((__m256i*)&(block)[(offset2) + 8]); \
        (C) = _mm512_inserti64x4(_mm512_castsi256_si512(c_low), c_high, 1); \
        \
        __m256i d_low = _mm256_loadu_si256((__m256i*)&(block)[(offset1) + 12]); \
        __m256i d_high = _mm256_loadu_si256((__m256i*)&(block)[(offset2) + 12]); \
        (D) = _mm512_inserti64x4(_mm512_castsi256_si512(d_low), d_high, 1); \
    } while(0)

#define BLAKE2_STORE_COLUMNS_AVX512(block, A, B, C, D, offset1, offset2) \
    do { \
        _mm256_storeu_si256((__m256i*)&(block)[(offset1)], _mm512_extracti64x4_epi64((A), 0)); \
        _mm256_storeu_si256((__m256i*)&(block)[(offset2)], _mm512_extracti64x4_epi64((A), 1)); \
        \
        _mm256_storeu_si256((__m256i*)&(block)[(offset1) + 4], _mm512_extracti64x4_epi64((B), 0)); \
        _mm256_storeu_si256((__m256i*)&(block)[(offset2) + 4], _mm512_extracti64x4_epi64((B), 1)); \
        \
        _mm256_storeu_si256((__m256i*)&(block)[(offset1) + 8], _mm512_extracti64x4_epi64((C), 0)); \
        _mm256_storeu_si256((__m256i*)&(block)[(offset2) + 8], _mm512_extracti64x4_epi64((C), 1)); \
        \
        _mm256_storeu_si256((__m256i*)&(block)[(offset1) + 12], _mm512_extracti64x4_epi64((D), 0)); \
        _mm256_storeu_si256((__m256i*)&(block)[(offset2) + 12], _mm512_extracti64x4_epi64((D), 1)); \
    } while(0)

// Row-wise loading (loads two rows with strided access)
#define BLAKE2_LOAD_ROWS_AVX512(A, B, C, D, block, idx) \
    do { \
        /* Create index vectors for gathering values */ \
        const __m512i idx_a = _mm512_set_epi64( \
            (4+(idx))*4+3, (4+(idx))*4+2, (4+(idx))*4+1, (4+(idx))*4+0, \
            (0+(idx))*4+3, (0+(idx))*4+2, (0+(idx))*4+1, (0+(idx))*4+0); \
        const __m512i idx_b = _mm512_set_epi64( \
            (12+(idx))*4+3, (12+(idx))*4+2, (12+(idx))*4+1, (12+(idx))*4+0, \
            (8+(idx))*4+3, (8+(idx))*4+2, (8+(idx))*4+1, (8+(idx))*4+0); \
        const __m512i idx_c = _mm512_set_epi64( \
            (20+(idx))*4+3, (20+(idx))*4+2, (20+(idx))*4+1, (20+(idx))*4+0, \
            (16+(idx))*4+3, (16+(idx))*4+2, (16+(idx))*4+1, (16+(idx))*4+0); \
        const __m512i idx_d = _mm512_set_epi64( \
            (28+(idx))*4+3, (28+(idx))*4+2, (28+(idx))*4+1, (28+(idx))*4+0, \
            (24+(idx))*4+3, (24+(idx))*4+2, (24+(idx))*4+1, (24+(idx))*4+0); \
        \
        /* Gather values directly into AVX512 registers */ \
        (A) = _mm512_i64gather_epi64(idx_a, (const long long int*)&(block)[0], 8); \
        (B) = _mm512_i64gather_epi64(idx_b, (const long long int*)&(block)[0], 8); \
        (C) = _mm512_i64gather_epi64(idx_c, (const long long int*)&(block)[0], 8); \
        (D) = _mm512_i64gather_epi64(idx_d, (const long long int*)&(block)[0], 8); \
    } while(0)

#define BLAKE2_STORE_ROWS_AVX512(block, A, B, C, D, idx) \
    do { \
        /* Create index vectors for scattering values */ \
        const __m512i idx_a = _mm512_set_epi64( \
            (4+(idx))*4+3, (4+(idx))*4+2, (4+(idx))*4+1, (4+(idx))*4+0, \
            (0+(idx))*4+3, (0+(idx))*4+2, (0+(idx))*4+1, (0+(idx))*4+0); \
        const __m512i idx_b = _mm512_set_epi64( \
            (12+(idx))*4+3, (12+(idx))*4+2, (12+(idx))*4+1, (12+(idx))*4+0, \
            (8+(idx))*4+3, (8+(idx))*4+2, (8+(idx))*4+1, (8+(idx))*4+0); \
        const __m512i idx_c = _mm512_set_epi64( \
            (20+(idx))*4+3, (20+(idx))*4+2, (20+(idx))*4+1, (20+(idx))*4+0, \
            (16+(idx))*4+3, (16+(idx))*4+2, (16+(idx))*4+1, (16+(idx))*4+0); \
        const __m512i idx_d = _mm512_set_epi64( \
            (28+(idx))*4+3, (28+(idx))*4+2, (28+(idx))*4+1, (28+(idx))*4+0, \
            (24+(idx))*4+3, (24+(idx))*4+2, (24+(idx))*4+1, (24+(idx))*4+0); \
        \
        /* Scatter values directly from AVX512 registers */ \
        _mm512_i64scatter_epi64((long long int*)&(block)[0], idx_a, (A), 8); \
        _mm512_i64scatter_epi64((long long int*)&(block)[0], idx_b, (B), 8); \
        _mm512_i64scatter_epi64((long long int*)&(block)[0], idx_c, (C), 8); \
        _mm512_i64scatter_epi64((long long int*)&(block)[0], idx_d, (D), 8); \
    } while(0)
    
// 2. Define G1 operation for AVX512 (processing 8 columns at once)
#define G1_AVX512_DUAL(A, B, C, D) \
    do { \
        __m512i ml = _mm512_mul_epu32(A, B); \
        ml = _mm512_add_epi64(ml, ml); \
        A = _mm512_add_epi64(A, _mm512_add_epi64(B, ml)); \
        D = _mm512_xor_si512(D, A); \
        D = rotr32_512(D); \
        \
        ml = _mm512_mul_epu32(C, D); \
        ml = _mm512_add_epi64(ml, ml); \
        C = _mm512_add_epi64(C, _mm512_add_epi64(D, ml)); \
        \
        B = _mm512_xor_si512(B, C); \
        B = rotr24_512(B); \
    } while(0)

// 3. Define G2 operation for AVX512 (processing 8 columns at once)
#define G2_AVX512_DUAL(A, B, C, D) \
    do { \
        __m512i ml = _mm512_mul_epu32(A, B); \
        ml = _mm512_add_epi64(ml, ml); \
        A = _mm512_add_epi64(A, _mm512_add_epi64(B, ml)); \
        D = _mm512_xor_si512(D, A); \
        D = rotr16_512(D); \
        \
        ml = _mm512_mul_epu32(C, D); \
        ml = _mm512_add_epi64(ml, ml); \
        C = _mm512_add_epi64(C, _mm512_add_epi64(D, ml)); \
        \
        B = _mm512_xor_si512(B, C); \
        B = rotr63_512(B); \
    } while(0)

// 4. Combine G1+G2 for convenience
#define G_AVX512_DUAL(A, B, C, D) \
    do { \
        G1_AVX512_DUAL(A, B, C, D); \
        G2_AVX512_DUAL(A, B, C, D); \
    } while(0)

#define DIAGONALIZE_1_AVX512_DUAL(A, B, C, D) \
    do { \
        /* Extract 256-bit halves */ \
        __m256i b_low = _mm512_extracti64x4_epi64(B, 0); \
        __m256i b_high = _mm512_extracti64x4_epi64(B, 1); \
        __m256i c_low = _mm512_extracti64x4_epi64(C, 0); \
        __m256i c_high = _mm512_extracti64x4_epi64(C, 1); \
        __m256i d_low = _mm512_extracti64x4_epi64(D, 0); \
        __m256i d_high = _mm512_extracti64x4_epi64(D, 1); \
        \
        /* Apply same permutation as AVX2 DIAGONALIZE_1 to each half */ \
        b_low = _mm256_permute4x64_epi64(b_low, _MM_SHUFFLE(0, 3, 2, 1)); \
        b_high = _mm256_permute4x64_epi64(b_high, _MM_SHUFFLE(0, 3, 2, 1)); \
        c_low = _mm256_permute4x64_epi64(c_low, _MM_SHUFFLE(1, 0, 3, 2)); \
        c_high = _mm256_permute4x64_epi64(c_high, _MM_SHUFFLE(1, 0, 3, 2)); \
        d_low = _mm256_permute4x64_epi64(d_low, _MM_SHUFFLE(2, 1, 0, 3)); \
        d_high = _mm256_permute4x64_epi64(d_high, _MM_SHUFFLE(2, 1, 0, 3)); \
        \
        /* Recombine halves */ \
        B = _mm512_inserti64x4(_mm512_castsi256_si512(b_low), b_high, 1); \
        C = _mm512_inserti64x4(_mm512_castsi256_si512(c_low), c_high, 1); \
        D = _mm512_inserti64x4(_mm512_castsi256_si512(d_low), d_high, 1); \
    } while(0)

// Undiagonalize for BLAKE2_ROUND_1 pattern (8 columns at once)
#define UNDIAGONALIZE_1_AVX512_DUAL(A, B, C, D) \
    do { \
        /* Extract 256-bit halves */ \
        __m256i b_low = _mm512_extracti64x4_epi64(B, 0); \
        __m256i b_high = _mm512_extracti64x4_epi64(B, 1); \
        __m256i c_low = _mm512_extracti64x4_epi64(C, 0); \
        __m256i c_high = _mm512_extracti64x4_epi64(C, 1); \
        __m256i d_low = _mm512_extracti64x4_epi64(D, 0); \
        __m256i d_high = _mm512_extracti64x4_epi64(D, 1); \
        \
        /* Apply same permutation as AVX2 UNDIAGONALIZE_1 to each half */ \
        b_low = _mm256_permute4x64_epi64(b_low, _MM_SHUFFLE(2, 1, 0, 3)); \
        b_high = _mm256_permute4x64_epi64(b_high, _MM_SHUFFLE(2, 1, 0, 3)); \
        c_low = _mm256_permute4x64_epi64(c_low, _MM_SHUFFLE(1, 0, 3, 2)); \
        c_high = _mm256_permute4x64_epi64(c_high, _MM_SHUFFLE(1, 0, 3, 2)); \
        d_low = _mm256_permute4x64_epi64(d_low, _MM_SHUFFLE(0, 3, 2, 1)); \
        d_high = _mm256_permute4x64_epi64(d_high, _MM_SHUFFLE(0, 3, 2, 1)); \
        \
        /* Recombine halves */ \
        B = _mm512_inserti64x4(_mm512_castsi256_si512(b_low), b_high, 1); \
        C = _mm512_inserti64x4(_mm512_castsi256_si512(c_low), c_high, 1); \
        D = _mm512_inserti64x4(_mm512_castsi256_si512(d_low), d_high, 1); \
    } while(0)

// For BLAKE2_ROUND_2 pattern, we need the more complex diagonalization
#define DIAGONALIZE_2_AVX512_DUAL(A, B, C, D) \
    do { \
        /* This is more complex - we need to handle the blend operations */ \
        /* For now, let's extract to 256-bit and use AVX2 logic */ \
        __m256i a0 = _mm512_extracti64x4_epi64(A, 0); \
        __m256i a1 = _mm512_extracti64x4_epi64(A, 1); \
        __m256i b0 = _mm512_extracti64x4_epi64(B, 0); \
        __m256i b1 = _mm512_extracti64x4_epi64(B, 1); \
        __m256i c0 = _mm512_extracti64x4_epi64(C, 0); \
        __m256i c1 = _mm512_extracti64x4_epi64(C, 1); \
        __m256i d0 = _mm512_extracti64x4_epi64(D, 0); \
        __m256i d1 = _mm512_extracti64x4_epi64(D, 1); \
        \
        /* Apply DIAGONALIZE_2 logic */ \
        __m256i tmp1 = _mm256_blend_epi32(b0, b1, 0xCC); \
        __m256i tmp2 = _mm256_blend_epi32(b0, b1, 0x33); \
        b1 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        b0 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
        \
        tmp1 = c0; \
        c0 = c1; \
        c1 = tmp1; \
        \
        tmp1 = _mm256_blend_epi32(d0, d1, 0xCC); \
        tmp2 = _mm256_blend_epi32(d0, d1, 0x33); \
        d0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        d1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
        \
        /* Recombine */ \
        A = _mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1); \
        B = _mm512_inserti64x4(_mm512_castsi256_si512(b0), b1, 1); \
        C = _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1); \
        D = _mm512_inserti64x4(_mm512_castsi256_si512(d0), d1, 1); \
    } while(0)

#define UNDIAGONALIZE_2_AVX512_DUAL(A, B, C, D) \
    do { \
        /* Extract to 256-bit */ \
        __m256i a0 = _mm512_extracti64x4_epi64(A, 0); \
        __m256i a1 = _mm512_extracti64x4_epi64(A, 1); \
        __m256i b0 = _mm512_extracti64x4_epi64(B, 0); \
        __m256i b1 = _mm512_extracti64x4_epi64(B, 1); \
        __m256i c0 = _mm512_extracti64x4_epi64(C, 0); \
        __m256i c1 = _mm512_extracti64x4_epi64(C, 1); \
        __m256i d0 = _mm512_extracti64x4_epi64(D, 0); \
        __m256i d1 = _mm512_extracti64x4_epi64(D, 1); \
        \
        /* Apply UNDIAGONALIZE_2 logic */ \
        __m256i tmp1 = _mm256_blend_epi32(b0, b1, 0xCC); \
        __m256i tmp2 = _mm256_blend_epi32(b0, b1, 0x33); \
        b0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        b1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
        \
        tmp1 = c0; \
        c0 = c1; \
        c1 = tmp1; \
        \
        tmp1 = _mm256_blend_epi32(d0, d1, 0x33); \
        tmp2 = _mm256_blend_epi32(d0, d1, 0xCC); \
        d0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        d1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
        \
        /* Recombine */ \
        A = _mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1); \
        B = _mm512_inserti64x4(_mm512_castsi256_si512(b0), b1, 1); \
        C = _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1); \
        D = _mm512_inserti64x4(_mm512_castsi256_si512(d0), d1, 1); \
    } while(0)

// Update the BLAKE2 round macros to use the correct diagonalization
#define BLAKE2_ROUND_1_AVX512_DUAL(A, B, C, D) \
    do { \
        G1_AVX512_DUAL(A, B, C, D); \
        G2_AVX512_DUAL(A, B, C, D); \
        \
        DIAGONALIZE_1_AVX512_DUAL(A, B, C, D); \
        \
        G1_AVX512_DUAL(A, B, C, D); \
        G2_AVX512_DUAL(A, B, C, D); \
        \
        UNDIAGONALIZE_1_AVX512_DUAL(A, B, C, D); \
    } while(0)

#define BLAKE2_ROUND_2_AVX512_DUAL(A, B, C, D) \
    do { \
        G1_AVX512_DUAL(A, B, C, D); \
        G2_AVX512_DUAL(A, B, C, D); \
        \
        DIAGONALIZE_2_AVX512_DUAL(A, B, C, D); \
        \
        G1_AVX512_DUAL(A, B, C, D); \
        G2_AVX512_DUAL(A, B, C, D); \
        \
        UNDIAGONALIZE_2_AVX512_DUAL(A, B, C, D); \
    } while(0)

// Helper: print 4 lanes of a __m256i as 64-bit hex
void print256(const char *name, __m256i reg) {
    uint64_t vals[4];
    _mm256_storeu_si256((__m256i*)vals, reg);
    printf("%s = [0x%016llx, 0x%016llx, 0x%016llx, 0x%016llx]\n",
           name,
           (unsigned long long)vals[0],
           (unsigned long long)vals[1],
           (unsigned long long)vals[2],
           (unsigned long long)vals[3]);
}

void print512(const char *name, __m512i reg) {
    uint64_t vals[8];
    _mm512_storeu_si512((__m512i*)vals, reg);
    printf("%s = [", name);
    for (int i = 0; i < 8; i++) {
        printf("0x%016llx%s", (unsigned long long)vals[i], (i < 7 ? ", " : ""));
    }
    printf("]\n");
}

void compare_all_implementations(const uint64_t* v_scalar, const uint64_t* v_avx2, 
                                const uint64_t* v_avx512, 
                                const __m256i* avx2_regs, const __m512i* avx512_regs) {
    printf("\n--- COMPARISON RESULTS ---\n");
    int mismatch = 0;
    
    // Compare primary data (first 16 values)
    printf("Checking primary values...\n");
    for (int i = 0; i < 16; i++) {
        if (v_scalar[i] != v_avx2[i]) {
            printf("Primary mismatch at v[%d]: Scalar=0x%016llx, AVX2=0x%016llx\n", 
                   i, (unsigned long long)v_scalar[i], (unsigned long long)v_avx2[i]);
            mismatch = 1;
        }
        
#ifdef __AVX512F__
        if (v_scalar[i] != v_avx512[i]) {
            printf("Primary mismatch at v[%d]: Scalar=0x%016llx, AVX512=0x%016llx\n", 
                   i, (unsigned long long)v_scalar[i], (unsigned long long)v_avx512[i]);
            mismatch = 1;
        }
#endif
    }
    
    // For AVX2, check that A1/B1/C1/D1 match A0/B0/C0/D0 (they should be duplicates)
    if (avx2_regs) {
        printf("\nChecking AVX2 duplicate lanes...\n");
        uint64_t avx2_vals[32];
        _mm256_storeu_si256((__m256i*)&avx2_vals[0], avx2_regs[0]);  // A0
        _mm256_storeu_si256((__m256i*)&avx2_vals[4], avx2_regs[1]);  // B0
        _mm256_storeu_si256((__m256i*)&avx2_vals[8], avx2_regs[2]);  // C0
        _mm256_storeu_si256((__m256i*)&avx2_vals[12], avx2_regs[3]); // D0
        _mm256_storeu_si256((__m256i*)&avx2_vals[16], avx2_regs[4]); // A1
        _mm256_storeu_si256((__m256i*)&avx2_vals[20], avx2_regs[5]); // B1
        _mm256_storeu_si256((__m256i*)&avx2_vals[24], avx2_regs[6]); // C1
        _mm256_storeu_si256((__m256i*)&avx2_vals[28], avx2_regs[7]); // D1
        
        // Check A0 == A1, B0 == B1, etc.
        for (int i = 0; i < 16; i++) {
            if (avx2_vals[i] != avx2_vals[i + 16]) {
                printf("AVX2 duplicate mismatch at position %d: Lane0=0x%016llx, Lane1=0x%016llx\n",
                       i, (unsigned long long)avx2_vals[i], (unsigned long long)avx2_vals[i + 16]);
                mismatch = 1;
            }
        }
    }
    
#ifdef __AVX512F__
    // For AVX512, check that lanes 4-7 match lanes 0-3 (they should be duplicates)
    if (avx512_regs) {
        printf("\nChecking AVX512 duplicate lanes...\n");
        uint64_t avx512_vals[32];
        _mm512_storeu_si512((__m512i*)&avx512_vals[0], avx512_regs[0]);  // A512
        _mm512_storeu_si512((__m512i*)&avx512_vals[8], avx512_regs[1]);  // B512
        _mm512_storeu_si512((__m512i*)&avx512_vals[16], avx512_regs[2]); // C512
        _mm512_storeu_si512((__m512i*)&avx512_vals[24], avx512_regs[3]); // D512
        
        // Check that each register's upper half matches lower half
        const char* reg_names[] = {"A512", "B512", "C512", "D512"};
        for (int reg = 0; reg < 4; reg++) {
            for (int i = 0; i < 4; i++) {
                int idx = reg * 8 + i;
                if (avx512_vals[idx] != avx512_vals[idx + 4]) {
                    printf("AVX512 duplicate mismatch in %s[%d]: Lane0=0x%016llx, Lane1=0x%016llx\n",
                           reg_names[reg], i, 
                           (unsigned long long)avx512_vals[idx], 
                           (unsigned long long)avx512_vals[idx + 4]);
                    mismatch = 1;
                }
                
                // Also verify against scalar
                int scalar_idx = reg * 4 + i;
                if (avx512_vals[idx] != v_scalar[scalar_idx]) {
                    printf("AVX512 vs Scalar mismatch in %s[%d]: AVX512=0x%016llx, Scalar=0x%016llx\n",
                           reg_names[reg], i,
                           (unsigned long long)avx512_vals[idx],
                           (unsigned long long)v_scalar[scalar_idx]);
                    mismatch = 1;
                }
            }
        }
    }
#endif
    
    if (!mismatch) {
        printf("PASS: All implementations produced identical results!\n");
        printf("PASS: All duplicate lanes match correctly!\n");
    }
}

#define ARGON2_BLOCK_SIZE 128  // 128 uint64_t values = 1024 bytes

// Helper function to compute checksum of 8 values
static uint64_t checksum8(const uint64_t* v) {
    uint64_t sum = 0;
    for (int i = 0; i < 8; i++) {
        sum ^= v[i];
    }
    return sum;
}

// Helper to print 8 values if they differ
static void print_diff8(const char* label, int start_idx, 
                       const uint64_t* scalar, const uint64_t* avx2, const uint64_t* avx512) {
    printf("\n%s (indices %d-%d):\n", label, start_idx, start_idx + 7);
    for (int i = 0; i < 8; i++) {
        printf("  v[%3d]: Scalar=0x%016llx, AVX2=0x%016llx", 
               start_idx + i, 
               (unsigned long long)scalar[i], 
               (unsigned long long)avx2[i]);
        if (avx512) {
            printf(", AVX512=0x%016llx", (unsigned long long)avx512[i]);
        }
        if (scalar[i] != avx2[i] || (avx512 && scalar[i] != avx512[i])) {
            printf(" <-- MISMATCH");
        }
        printf("\n");
    }
}

void test_g_operations_only() {
    printf("\n=== Testing G1 Operations Only (No Diagonalization) ===\n");
    
    // Create test data - 32 values to test multiple scenarios
    uint64_t test_data[32];
    for (int i = 0; i < 32; i++) {
        test_data[i] = 0x0101010100000000ULL + i;
    }
    
    // Make copies for each implementation
    uint64_t v_scalar[32];
    uint64_t v_avx2[32];
    uint64_t v_avx512[32];
    
    memcpy(v_scalar, test_data, sizeof(v_scalar));
    memcpy(v_avx2, test_data, sizeof(v_avx2));
    memcpy(v_avx512, test_data, sizeof(v_avx512));
    
    // ... (checksums code remains the same) ...
    
    // --- Scalar G1 operations ---
    printf("\nApplying scalar G1 operations...\n");
    // Process first 16 values as one block
    G(v_scalar[0], v_scalar[4], v_scalar[8], v_scalar[12]);
    G(v_scalar[1], v_scalar[5], v_scalar[9], v_scalar[13]);
    G(v_scalar[2], v_scalar[6], v_scalar[10], v_scalar[14]);
    G(v_scalar[3], v_scalar[7], v_scalar[11], v_scalar[15]);
    
    // Process second 16 values
    G(v_scalar[16], v_scalar[20], v_scalar[24], v_scalar[28]);
    G(v_scalar[17], v_scalar[21], v_scalar[25], v_scalar[29]);
    G(v_scalar[18], v_scalar[22], v_scalar[26], v_scalar[30]);
    G(v_scalar[19], v_scalar[23], v_scalar[27], v_scalar[31]);
    
    // --- AVX2 G1 operations ---
    printf("Applying AVX2 G1 operations...\n");
    __m256i A0 = _mm256_loadu_si256((__m256i*)&v_avx2[0]);
    __m256i B0 = _mm256_loadu_si256((__m256i*)&v_avx2[4]);
    __m256i C0 = _mm256_loadu_si256((__m256i*)&v_avx2[8]);
    __m256i D0 = _mm256_loadu_si256((__m256i*)&v_avx2[12]);
    __m256i A1 = _mm256_loadu_si256((__m256i*)&v_avx2[16]);
    __m256i B1 = _mm256_loadu_si256((__m256i*)&v_avx2[20]);
    __m256i C1 = _mm256_loadu_si256((__m256i*)&v_avx2[24]);
    __m256i D1 = _mm256_loadu_si256((__m256i*)&v_avx2[28]);
    
    G1_AVX2(A0, A1, B0, B1, C0, C1, D0, D1);
    G2_AVX2(A0, A1, B0, B1, C0, C1, D0, D1);
    
    _mm256_storeu_si256((__m256i*)&v_avx2[0], A0);
    _mm256_storeu_si256((__m256i*)&v_avx2[4], B0);
    _mm256_storeu_si256((__m256i*)&v_avx2[8], C0);
    _mm256_storeu_si256((__m256i*)&v_avx2[12], D0);
    _mm256_storeu_si256((__m256i*)&v_avx2[16], A1);
    _mm256_storeu_si256((__m256i*)&v_avx2[20], B1);
    _mm256_storeu_si256((__m256i*)&v_avx2[24], C1);
    _mm256_storeu_si256((__m256i*)&v_avx2[28], D1);
    
#ifdef __AVX512F__
    // --- AVX512 G1 operations (dual block processing) ---
    printf("Applying AVX512 G1 operations (8 unique values per register)...\n");
    __m512i A, B, C, D;
    
    // Load two blocks at once
    BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512, 0, 16);
    
    // Process 8 columns at once
    G_AVX512_DUAL(A, B, C, D);
    
    // Store results back
    BLAKE2_STORE_COLUMNS_AVX512(v_avx512, A, B, C, D, 0, 16);
#endif
    
    // Compare results
    printf("\nG1 Results Comparison (checksums):\n");
    bool all_match = true;
    
    for (int i = 0; i < 4; i++) {
        uint64_t cs_scalar = checksum8(&v_scalar[i * 8]);
        uint64_t cs_avx2 = checksum8(&v_avx2[i * 8]);
#ifdef __AVX512F__
        uint64_t cs_avx512 = checksum8(&v_avx512[i * 8]);
#endif
        
        printf("  Group[%d]: Scalar=0x%016llx, AVX2=0x%016llx", 
               i, (unsigned long long)cs_scalar, (unsigned long long)cs_avx2);
#ifdef __AVX512F__
        printf(", AVX512=0x%016llx", (unsigned long long)cs_avx512);
#endif
        
        bool match = (cs_scalar == cs_avx2);
#ifdef __AVX512F__
        match = match && (cs_scalar == cs_avx512);
#endif
        
        if (!match) {
            printf(" <-- MISMATCH");
            all_match = false;
            
            // Print detailed comparison
            printf("\n  Detailed values for group %d:\n", i);
            for (int j = 0; j < 8; j++) {
                int idx = i * 8 + j;
                printf("    v[%2d]: Scalar=0x%016llx, AVX2=0x%016llx", 
                       idx, (unsigned long long)v_scalar[idx], 
                       (unsigned long long)v_avx2[idx]);
#ifdef __AVX512F__
                printf(", AVX512=0x%016llx", (unsigned long long)v_avx512[idx]);
#endif
                printf("\n");
            }
        }
        printf("\n");
    }
    
    if (all_match) {
        printf("\nPASS: All G1 operations match across implementations!\n");
        printf("This confirms the G operations work correctly on all 8 lanes.\n");
        printf("Only diagonalization patterns need adjustment for AVX512.\n");
    }
}

void test_blake2_round_with_diagonalization() {
    printf("\n=== Testing BLAKE2 Round with Diagonalization ===\n");
    
    // Create test data - 32 values to test multiple scenarios
    uint64_t test_data[32];
    for (int i = 0; i < 32; i++) {
        test_data[i] = 0x0101010100000000ULL + i;
    }
    
    // Make copies for each implementation
    uint64_t v_scalar[32];
    uint64_t v_avx2[32];
    uint64_t v_avx512[32];
    uint64_t v_avx512_step[32];  // For step-by-step verification
    
    memcpy(v_scalar, test_data, sizeof(v_scalar));
    memcpy(v_avx2, test_data, sizeof(v_avx2));
    memcpy(v_avx512, test_data, sizeof(v_avx512));
    memcpy(v_avx512_step, test_data, sizeof(v_avx512_step));
    
    printf("Initial checksums:\n");
    for (int i = 0; i < 4; i++) {
        uint64_t cs = checksum8(&test_data[i * 8]);
        printf("  Group[%d]: 0x%016llx\n", i, (unsigned long long)cs);
    }
    
    // --- Scalar BLAKE2 Round ---
    printf("\n--- Scalar BLAKE2 Round ---\n");
    
    // First G operations
    G1(v_scalar[0], v_scalar[4], v_scalar[8], v_scalar[12]);
    G1(v_scalar[1], v_scalar[5], v_scalar[9], v_scalar[13]);
    G1(v_scalar[2], v_scalar[6], v_scalar[10], v_scalar[14]);
    G1(v_scalar[3], v_scalar[7], v_scalar[11], v_scalar[15]);
    G2(v_scalar[0], v_scalar[4], v_scalar[8], v_scalar[12]);
    G2(v_scalar[1], v_scalar[5], v_scalar[9], v_scalar[13]);
    G2(v_scalar[2], v_scalar[6], v_scalar[10], v_scalar[14]);
    G2(v_scalar[3], v_scalar[7], v_scalar[11], v_scalar[15]);
    
    printf("After first G: ");
    uint64_t cs_after_g1 = checksum8(&v_scalar[0]);
    printf("0x%016llx\n", (unsigned long long)cs_after_g1);
    
    // Diagonalize (manual for scalar)
    // For diagonals: (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)
    G1(v_scalar[0], v_scalar[5], v_scalar[10], v_scalar[15]);
    G1(v_scalar[1], v_scalar[6], v_scalar[11], v_scalar[12]);
    G1(v_scalar[2], v_scalar[7], v_scalar[8], v_scalar[13]);
    G1(v_scalar[3], v_scalar[4], v_scalar[9], v_scalar[14]);
    G2(v_scalar[0], v_scalar[5], v_scalar[10], v_scalar[15]);
    G2(v_scalar[1], v_scalar[6], v_scalar[11], v_scalar[12]);
    G2(v_scalar[2], v_scalar[7], v_scalar[8], v_scalar[13]);
    G2(v_scalar[3], v_scalar[4], v_scalar[9], v_scalar[14]);
    
    printf("After diagonal G: ");
    uint64_t cs_after_diag = checksum8(&v_scalar[0]);
    printf("0x%016llx\n", (unsigned long long)cs_after_diag);
    
    // --- AVX2 BLAKE2 Round ---
    printf("\n--- AVX2 BLAKE2 Round ---\n");
    __m256i A0 = _mm256_loadu_si256((__m256i*)&v_avx2[0]);
    __m256i B0 = _mm256_loadu_si256((__m256i*)&v_avx2[4]);
    __m256i C0 = _mm256_loadu_si256((__m256i*)&v_avx2[8]);
    __m256i D0 = _mm256_loadu_si256((__m256i*)&v_avx2[12]);
    __m256i A1 = A0;  // Duplicate for AVX2 pattern
    __m256i B1 = B0;
    __m256i C1 = C0;
    __m256i D1 = D0;
    
    // First G operations
    G1_AVX2(A0, A1, B0, B1, C0, C1, D0, D1);
    G2_AVX2(A0, A1, B0, B1, C0, C1, D0, D1);
    
    // Store to check intermediate state
    _mm256_storeu_si256((__m256i*)&v_avx2[0], A0);
    _mm256_storeu_si256((__m256i*)&v_avx2[4], B0);
    _mm256_storeu_si256((__m256i*)&v_avx2[8], C0);
    _mm256_storeu_si256((__m256i*)&v_avx2[12], D0);
    
    printf("After first G: ");
    cs_after_g1 = checksum8(&v_avx2[0]);
    printf("0x%016llx\n", (unsigned long long)cs_after_g1);
    
    // Diagonalize
    DIAGONALIZE_1(A0, B0, C0, D0, A1, B1, C1, D1);
    
    // Second G operations
    G1_AVX2(A0, A1, B0, B1, C0, C1, D0, D1);
    G2_AVX2(A0, A1, B0, B1, C0, C1, D0, D1);
    
    // Undiagonalize
    UNDIAGONALIZE_1(A0, B0, C0, D0, A1, B1, C1, D1);
    
    // Store final result
    _mm256_storeu_si256((__m256i*)&v_avx2[0], A0);
    _mm256_storeu_si256((__m256i*)&v_avx2[4], B0);
    _mm256_storeu_si256((__m256i*)&v_avx2[8], C0);
    _mm256_storeu_si256((__m256i*)&v_avx2[12], D0);
    
    printf("After full round: ");
    uint64_t cs_final = checksum8(&v_avx2[0]);
    printf("0x%016llx\n", (unsigned long long)cs_final);
    
#ifdef __AVX512F__
    // --- AVX512 BLAKE2 Round (Step by Step) ---
    printf("\n--- AVX512 BLAKE2 Round (Step by Step) ---\n");
    __m512i A, B, C, D;
    
    // Load two blocks at once
    BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512_step, 0, 16);
    
    // First G operations
    G1_AVX512_DUAL(A, B, C, D);
    G2_AVX512_DUAL(A, B, C, D);
    
    // Store to check intermediate state
    BLAKE2_STORE_COLUMNS_AVX512(v_avx512_step, A, B, C, D, 0, 16);
    printf("After first G: ");
    cs_after_g1 = checksum8(&v_avx512_step[0]);
    printf("0x%016llx\n", (unsigned long long)cs_after_g1);
    
    // Reload for diagonalization
    BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512_step, 0, 16);
    
    // Diagonalize
    DIAGONALIZE_1_AVX512_DUAL(A, B, C, D);
    
    // Second G operations  
    G1_AVX512_DUAL(A, B, C, D);
    G2_AVX512_DUAL(A, B, C, D);
    
    // Undiagonalize
    UNDIAGONALIZE_1_AVX512_DUAL(A, B, C, D);
    
    // Store final result
    BLAKE2_STORE_COLUMNS_AVX512(v_avx512_step, A, B, C, D, 0, 16);
    printf("After full round: ");
    cs_final = checksum8(&v_avx512_step[0]);
    printf("0x%016llx\n", (unsigned long long)cs_final);
    
    // --- AVX512 BLAKE2 Round (Using Macro) ---
    printf("\n--- AVX512 BLAKE2 Round (Using Macro) ---\n");
    BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512, 0, 16);
    BLAKE2_ROUND_1_AVX512_DUAL(A, B, C, D);
    BLAKE2_STORE_COLUMNS_AVX512(v_avx512, A, B, C, D, 0, 16);
    
    printf("After full round: ");
    cs_final = checksum8(&v_avx512[0]);
    printf("0x%016llx\n", (unsigned long long)cs_final);
#endif
    
    // Final comparison
    printf("\n--- Final Comparison ---\n");
    bool all_match = true;
    
    // Compare first 16 values only (one block)
    for (int i = 0; i < 16; i++) {
        printf("v[%2d]: Scalar=0x%016llx, AVX2=0x%016llx", 
               i, (unsigned long long)v_scalar[i], (unsigned long long)v_avx2[i]);
#ifdef __AVX512F__
        printf(", AVX512=0x%016llx", (unsigned long long)v_avx512[i]);
        if (v_scalar[i] != v_avx512[i]) all_match = false;
#endif
        if (v_scalar[i] != v_avx2[i]) all_match = false;
        printf("\n");
    }
    
    if (all_match) {
        printf("\nPASS: All BLAKE2 round implementations match!\n");
    } else {
        printf("\nFAIL: BLAKE2 round implementations DO NOT match!\n");
    }
}

void test_blake2_both_rounds() {
    printf("\n=== Testing Both BLAKE2 Rounds (Column-wise and Row-wise) ===\n");
    
    // Create a full Argon2 block
    uint64_t v_block[ARGON2_BLOCK_SIZE];
    
    // Initialize with deterministic pattern
    for (int i = 0; i < ARGON2_BLOCK_SIZE; i++) {
        v_block[i] = 0x0101010100000000ULL + i;
    }
    
    // Make copies for each implementation
    uint64_t v_scalar[ARGON2_BLOCK_SIZE];
    uint64_t v_avx2[ARGON2_BLOCK_SIZE];
    uint64_t v_avx512[ARGON2_BLOCK_SIZE];
    
    memcpy(v_scalar, v_block, sizeof(v_scalar));
    memcpy(v_avx2, v_block, sizeof(v_avx2));
    memcpy(v_avx512, v_block, sizeof(v_avx512));
    
    printf("Initial block checksums:\n");
    for (int i = 0; i < 16; i++) {
        uint64_t cs = checksum8(&v_block[i * 8]);
        printf("  Block[%2d]: 0x%016llx\n", i, (unsigned long long)cs);
    }
    
    // ========== COLUMN-WISE ROUNDS (BLAKE2_ROUND_1) ==========
    printf("\n=== Column-wise BLAKE2 Rounds ===\n");
    
    // --- Scalar implementation ---
    printf("Running scalar column-wise rounds...\n");
    for (int i = 0; i < 8; ++i) {
        BLAKE2_ROUND_NOMSG(
            v_scalar[16 * i], v_scalar[16 * i + 1], v_scalar[16 * i + 2],
            v_scalar[16 * i + 3], v_scalar[16 * i + 4], v_scalar[16 * i + 5],
            v_scalar[16 * i + 6], v_scalar[16 * i + 7], v_scalar[16 * i + 8],
            v_scalar[16 * i + 9], v_scalar[16 * i + 10], v_scalar[16 * i + 11],
            v_scalar[16 * i + 12], v_scalar[16 * i + 13], v_scalar[16 * i + 14],
            v_scalar[16 * i + 15]);
    }
    
    // --- AVX2 implementation ---
    printf("Running AVX2 column-wise rounds...\n");
    __m256i state[ARGON2_BLOCK_SIZE / 4];  // 32 registers for 128 values
    
    // Load the data into state array
    for (int i = 0; i < ARGON2_BLOCK_SIZE / 4; i++) {
        state[i] = _mm256_loadu_si256((__m256i*)&v_avx2[i * 4]);
    }
    
    // Perform column-wise rounds
    for (int i = 0; i < 4; ++i) {
        BLAKE2_ROUND_1(state[8 * i + 0], state[8 * i + 4], state[8 * i + 1], state[8 * i + 5],
                       state[8 * i + 2], state[8 * i + 6], state[8 * i + 3], state[8 * i + 7]);
    }
    
    // Store results back
    for (int i = 0; i < ARGON2_BLOCK_SIZE / 4; i++) {
        _mm256_storeu_si256((__m256i*)&v_avx2[i * 4], state[i]);
    }
    
#ifdef __AVX512F__
    // --- AVX512 implementation ---
    printf("Running AVX512 column-wise rounds...\n");
    for (int i = 0; i < 4; ++i) {
        __m512i A, B, C, D;
        
        // Load two blocks at once
        BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512, i * 32, i * 32 + 16);
        
        // Process using ROUND_1 pattern
        BLAKE2_ROUND_1_AVX512_DUAL(A, B, C, D);
        
        // Store back
        BLAKE2_STORE_COLUMNS_AVX512(v_avx512, A, B, C, D, i * 32, i * 32 + 16);
    }
#endif
    
    // Compare column-wise results
    printf("\nColumn-wise checksums comparison:\n");
    bool columns_match = true;
    for (int i = 0; i < 16; i++) {
        uint64_t cs_scalar = checksum8(&v_scalar[i * 8]);
        uint64_t cs_avx2 = checksum8(&v_avx2[i * 8]);
#ifdef __AVX512F__
        uint64_t cs_avx512 = checksum8(&v_avx512[i * 8]);
#endif
        
        printf("  Group[%2d]: Scalar=0x%016llx, AVX2=0x%016llx", 
               i, (unsigned long long)cs_scalar, (unsigned long long)cs_avx2);
#ifdef __AVX512F__
        printf(", AVX512=0x%016llx", (unsigned long long)cs_avx512);
#endif
        
        bool match = (cs_scalar == cs_avx2);
#ifdef __AVX512F__
        match = match && (cs_scalar == cs_avx512);
#endif
        
        if (!match) {
            printf(" <-- MISMATCH");
            columns_match = false;
        }
        printf("\n");
    }
    
    if (columns_match) {
        printf("PASS: All column-wise implementations match!\n");
    } else {
        printf("FAIL: Column-wise implementations DO NOT match!\n");
    }
    
    // ========== ROW-WISE ROUNDS (BLAKE2_ROUND_2) ==========
    printf("\n=== Row-wise BLAKE2 Rounds ===\n");
    
    // --- Scalar implementation ---
    printf("Running scalar row-wise rounds...\n");
    for (int i = 0; i < 8; i++) {
        BLAKE2_ROUND_NOMSG(
            v_scalar[2 * i], v_scalar[2 * i + 1], v_scalar[2 * i + 16],
            v_scalar[2 * i + 17], v_scalar[2 * i + 32], v_scalar[2 * i + 33],
            v_scalar[2 * i + 48], v_scalar[2 * i + 49], v_scalar[2 * i + 64],
            v_scalar[2 * i + 65], v_scalar[2 * i + 80], v_scalar[2 * i + 81],
            v_scalar[2 * i + 96], v_scalar[2 * i + 97], v_scalar[2 * i + 112],
            v_scalar[2 * i + 113]);
    }
    
    // --- AVX2 implementation ---
    printf("Running AVX2 row-wise rounds...\n");
    
    // Reload the data after column rounds
    for (int i = 0; i < ARGON2_BLOCK_SIZE / 4; i++) {
        state[i] = _mm256_loadu_si256((__m256i*)&v_avx2[i * 4]);
    }
    
    // Perform row-wise rounds
    for (int i = 0; i < 4; ++i) {
        BLAKE2_ROUND_2(state[0 + i], state[4 + i], state[8 + i], state[12 + i],
                      state[16 + i], state[20 + i], state[24 + i], state[28 + i]);
    }
    
    // Store results back
    for (int i = 0; i < ARGON2_BLOCK_SIZE / 4; i++) {
        _mm256_storeu_si256((__m256i*)&v_avx2[i * 4], state[i]);
    }
    
#ifdef __AVX512F__
    // --- AVX512 implementation ---
    printf("Running AVX512 row-wise rounds...\n");
    
    // For row-wise, we need to use the row shuffle pattern
    for (int i = 0; i < 4; ++i) {
        __m512i A, B, C, D;
        BLAKE2_LOAD_ROWS_AVX512(A, B, C, D, v_avx512, i);
        BLAKE2_ROUND_2_AVX512_DUAL(A, B, C, D);
        BLAKE2_STORE_ROWS_AVX512(v_avx512, A, B, C, D, i);
    }
#endif
    
    // Compare final results after both rounds
    printf("\nFinal checksums after both rounds:\n");
    bool all_match = true;
    for (int i = 0; i < 16; i++) {
        uint64_t cs_scalar = checksum8(&v_scalar[i * 8]);
        uint64_t cs_avx2 = checksum8(&v_avx2[i * 8]);
#ifdef __AVX512F__
        uint64_t cs_avx512 = checksum8(&v_avx512[i * 8]);
#endif
        
        printf("  Group[%2d]: Scalar=0x%016llx, AVX2=0x%016llx", 
               i, (unsigned long long)cs_scalar, (unsigned long long)cs_avx2);
#ifdef __AVX512F__
        printf(", AVX512=0x%016llx", (unsigned long long)cs_avx512);
#endif
        
        bool match = (cs_scalar == cs_avx2);
#ifdef __AVX512F__
        match = match && (cs_scalar == cs_avx512);
#endif
        
        if (!match) {
            printf(" <-- MISMATCH");
            all_match = false;
            
            // Print detailed comparison for mismatched group
            print_diff8("Mismatch details", i * 8, 
                       &v_scalar[i * 8], &v_avx2[i * 8], 
#ifdef __AVX512F__
                       &v_avx512[i * 8]
#else
                       NULL
#endif
                       );
        }
        printf("\n");
    }
    
    if (all_match) {
        printf("\nPASS: All implementations match after both rounds!\n");
        printf("PASS: Column-wise (BLAKE2_ROUND_1) and Row-wise (BLAKE2_ROUND_2) processing verified!\n");
    } else {
        printf("\nFAIL: Implementations DO NOT match after both rounds!\n");
    }
}

void test_diagonalize_operations() {
    printf("\n=== Testing Diagonalization Operations ===\n");
    
    // Create test data with recognizable pattern
    uint64_t test_data[32];
    for (int i = 0; i < 32; i++) {
        test_data[i] = 0x0101010100000000ULL + i;
    }
    
    // Make copies for AVX2 and AVX512
    uint64_t v_avx2[32];
    uint64_t v_avx512[32];
    
    memcpy(v_avx2, test_data, sizeof(v_avx2));
    memcpy(v_avx512, test_data, sizeof(v_avx512));
    
    printf("Initial data pattern:\n");
    for (int i = 0; i < 32; i += 4) {
        printf("  [%2d-%2d]: 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n", 
               i, i+3,
               (unsigned long long)test_data[i], 
               (unsigned long long)test_data[i+1],
               (unsigned long long)test_data[i+2],
               (unsigned long long)test_data[i+3]);
    }
    
    // === Test DIAGONALIZE_1 ===
    printf("\n--- Testing DIAGONALIZE_1 ---\n");
    
    // AVX2 implementation
    __m256i A0 = _mm256_loadu_si256((__m256i*)&v_avx2[0]);
    __m256i B0 = _mm256_loadu_si256((__m256i*)&v_avx2[4]);
    __m256i C0 = _mm256_loadu_si256((__m256i*)&v_avx2[8]);
    __m256i D0 = _mm256_loadu_si256((__m256i*)&v_avx2[12]);
    __m256i A1 = _mm256_loadu_si256((__m256i*)&v_avx2[16]);
    __m256i B1 = _mm256_loadu_si256((__m256i*)&v_avx2[20]);
    __m256i C1 = _mm256_loadu_si256((__m256i*)&v_avx2[24]);
    __m256i D1 = _mm256_loadu_si256((__m256i*)&v_avx2[28]);
    
    // Apply diagonalization
    DIAGONALIZE_1(A0, B0, C0, D0, A1, B1, C1, D1);
    
    // Store back
    _mm256_storeu_si256((__m256i*)&v_avx2[0], A0);
    _mm256_storeu_si256((__m256i*)&v_avx2[4], B0);
    _mm256_storeu_si256((__m256i*)&v_avx2[8], C0);
    _mm256_storeu_si256((__m256i*)&v_avx2[12], D0);
    _mm256_storeu_si256((__m256i*)&v_avx2[16], A1);
    _mm256_storeu_si256((__m256i*)&v_avx2[20], B1);
    _mm256_storeu_si256((__m256i*)&v_avx2[24], C1);
    _mm256_storeu_si256((__m256i*)&v_avx2[28], D1);
    
#ifdef __AVX512F__
    // AVX512 implementation
    __m512i A, B, C, D;
    
    // Load with proper interleaving
    BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512, 0, 16);
    
    // Apply diagonalization
    DIAGONALIZE_1_AVX512_DUAL(A, B, C, D);
    
    // Store back
    BLAKE2_STORE_COLUMNS_AVX512(v_avx512, A, B, C, D, 0, 16);
#endif
    
    printf("After DIAGONALIZE_1:\n");
    printf("AVX2 result:\n");
    for (int i = 0; i < 32; i += 4) {
        printf("  [%2d-%2d]: 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n", 
               i, i+3,
               (unsigned long long)v_avx2[i], 
               (unsigned long long)v_avx2[i+1],
               (unsigned long long)v_avx2[i+2],
               (unsigned long long)v_avx2[i+3]);
    }
    
#ifdef __AVX512F__
    printf("AVX512 result:\n");
    for (int i = 0; i < 32; i += 4) {
        printf("  [%2d-%2d]: 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n", 
               i, i+3,
               (unsigned long long)v_avx512[i], 
               (unsigned long long)v_avx512[i+1],
               (unsigned long long)v_avx512[i+2],
               (unsigned long long)v_avx512[i+3]);
    }
#endif
    
    // Reset data for DIAGONALIZE_2 test
    memcpy(v_avx2, test_data, sizeof(v_avx2));
    memcpy(v_avx512, test_data, sizeof(v_avx512));
    
    // === Test DIAGONALIZE_2 ===
    printf("\n--- Testing DIAGONALIZE_2 ---\n");
    
    // AVX2 implementation
    A0 = _mm256_loadu_si256((__m256i*)&v_avx2[0]);
    B0 = _mm256_loadu_si256((__m256i*)&v_avx2[4]);
    C0 = _mm256_loadu_si256((__m256i*)&v_avx2[8]);
    D0 = _mm256_loadu_si256((__m256i*)&v_avx2[12]);
    A1 = _mm256_loadu_si256((__m256i*)&v_avx2[16]);
    B1 = _mm256_loadu_si256((__m256i*)&v_avx2[20]);
    C1 = _mm256_loadu_si256((__m256i*)&v_avx2[24]);
    D1 = _mm256_loadu_si256((__m256i*)&v_avx2[28]);
    
    // Apply diagonalization
    DIAGONALIZE_2(A0, A1, B0, B1, C0, C1, D0, D1);
    
    // Store back
    _mm256_storeu_si256((__m256i*)&v_avx2[0], A0);
    _mm256_storeu_si256((__m256i*)&v_avx2[4], B0);
    _mm256_storeu_si256((__m256i*)&v_avx2[8], C0);
    _mm256_storeu_si256((__m256i*)&v_avx2[12], D0);
    _mm256_storeu_si256((__m256i*)&v_avx2[16], A1);
    _mm256_storeu_si256((__m256i*)&v_avx2[20], B1);
    _mm256_storeu_si256((__m256i*)&v_avx2[24], C1);
    _mm256_storeu_si256((__m256i*)&v_avx2[28], D1);
    
#ifdef __AVX512F__
    // AVX512 implementation - the problem is likely here
    BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512, 0, 16);
    
    // Apply diagonalization
    DIAGONALIZE_2_AVX512_DUAL(A, B, C, D);
    
    // Store back
    BLAKE2_STORE_COLUMNS_AVX512(v_avx512, A, B, C, D, 0, 16);
#endif
    
    printf("After DIAGONALIZE_2:\n");
    printf("AVX2 result:\n");
    for (int i = 0; i < 32; i += 4) {
        printf("  [%2d-%2d]: 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n", 
               i, i+3,
               (unsigned long long)v_avx2[i], 
               (unsigned long long)v_avx2[i+1],
               (unsigned long long)v_avx2[i+2],
               (unsigned long long)v_avx2[i+3]);
    }
    
#ifdef __AVX512F__
    printf("AVX512 result:\n");
    for (int i = 0; i < 32; i += 4) {
        printf("  [%2d-%2d]: 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n", 
               i, i+3,
               (unsigned long long)v_avx512[i], 
               (unsigned long long)v_avx512[i+1],
               (unsigned long long)v_avx512[i+2],
               (unsigned long long)v_avx512[i+3]);
    }
#endif
    
    // Compare if AVX2 and AVX512 diagonalization results match
    bool diag1_match = true;
    bool diag2_match = true;
    
    // Reset data for comparison
    memcpy(v_avx2, test_data, sizeof(v_avx2));
    memcpy(v_avx512, test_data, sizeof(v_avx512));
    
    // Test DIAGONALIZE_1 again for comparison
    A0 = _mm256_loadu_si256((__m256i*)&v_avx2[0]);
    B0 = _mm256_loadu_si256((__m256i*)&v_avx2[4]);
    C0 = _mm256_loadu_si256((__m256i*)&v_avx2[8]);
    D0 = _mm256_loadu_si256((__m256i*)&v_avx2[12]);
    A1 = _mm256_loadu_si256((__m256i*)&v_avx2[16]);
    B1 = _mm256_loadu_si256((__m256i*)&v_avx2[20]);
    C1 = _mm256_loadu_si256((__m256i*)&v_avx2[24]);
    D1 = _mm256_loadu_si256((__m256i*)&v_avx2[28]);
    
    DIAGONALIZE_1(A0, B0, C0, D0, A1, B1, C1, D1);
    
    _mm256_storeu_si256((__m256i*)&v_avx2[0], A0);
    _mm256_storeu_si256((__m256i*)&v_avx2[4], B0);
    _mm256_storeu_si256((__m256i*)&v_avx2[8], C0);
    _mm256_storeu_si256((__m256i*)&v_avx2[12], D0);
    _mm256_storeu_si256((__m256i*)&v_avx2[16], A1);
    _mm256_storeu_si256((__m256i*)&v_avx2[20], B1);
    _mm256_storeu_si256((__m256i*)&v_avx2[24], C1);
    _mm256_storeu_si256((__m256i*)&v_avx2[28], D1);
    
#ifdef __AVX512F__
    BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512, 0, 16);
    DIAGONALIZE_1_AVX512_DUAL(A, B, C, D);
    BLAKE2_STORE_COLUMNS_AVX512(v_avx512, A, B, C, D, 0, 16);
    
    for (int i = 0; i < 32; i++) {
        if (v_avx2[i] != v_avx512[i]) {
            diag1_match = false;
            break;
        }
    }
#endif
    
    // Reset data for DIAGONALIZE_2 comparison
    memcpy(v_avx2, test_data, sizeof(v_avx2));
    memcpy(v_avx512, test_data, sizeof(v_avx512));
    
    // Test DIAGONALIZE_2 again for comparison
    A0 = _mm256_loadu_si256((__m256i*)&v_avx2[0]);
    B0 = _mm256_loadu_si256((__m256i*)&v_avx2[4]);
    C0 = _mm256_loadu_si256((__m256i*)&v_avx2[8]);
    D0 = _mm256_loadu_si256((__m256i*)&v_avx2[12]);
    A1 = _mm256_loadu_si256((__m256i*)&v_avx2[16]);
    B1 = _mm256_loadu_si256((__m256i*)&v_avx2[20]);
    C1 = _mm256_loadu_si256((__m256i*)&v_avx2[24]);
    D1 = _mm256_loadu_si256((__m256i*)&v_avx2[28]);
    
    DIAGONALIZE_2(A0, A1, B0, B1, C0, C1, D0, D1);
    
    _mm256_storeu_si256((__m256i*)&v_avx2[0], A0);
    _mm256_storeu_si256((__m256i*)&v_avx2[4], B0);
    _mm256_storeu_si256((__m256i*)&v_avx2[8], C0);
    _mm256_storeu_si256((__m256i*)&v_avx2[12], D0);
    _mm256_storeu_si256((__m256i*)&v_avx2[16], A1);
    _mm256_storeu_si256((__m256i*)&v_avx2[20], B1);
    _mm256_storeu_si256((__m256i*)&v_avx2[24], C1);
    _mm256_storeu_si256((__m256i*)&v_avx2[28], D1);
    
#ifdef __AVX512F__
    BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512, 0, 16);
    DIAGONALIZE_2_AVX512_DUAL(A, B, C, D);
    BLAKE2_STORE_COLUMNS_AVX512(v_avx512, A, B, C, D, 0, 16);
    
    for (int i = 0; i < 32; i++) {
        if (v_avx2[i] != v_avx512[i]) {
            diag2_match = false;
            break;
        }
    }
#endif
    
    printf("\n--- Comparison Results ---\n");
    printf("DIAGONALIZE_1: %s\n", diag1_match ? "MATCH" : "MISMATCH");
    printf("DIAGONALIZE_2: %s\n", diag2_match ? "MATCH" : "MISMATCH");
    
    if (!diag1_match || !diag2_match) {
        printf("\nDetailed comparison:\n");
        memcpy(v_avx2, test_data, sizeof(v_avx2));
        memcpy(v_avx512, test_data, sizeof(v_avx512));
        
        // Test again and print specific differences
        if (!diag2_match) {
            printf("\nDIAGONALIZE_2 differences:\n");
            
            A0 = _mm256_loadu_si256((__m256i*)&v_avx2[0]);
            B0 = _mm256_loadu_si256((__m256i*)&v_avx2[4]);
            C0 = _mm256_loadu_si256((__m256i*)&v_avx2[8]);
            D0 = _mm256_loadu_si256((__m256i*)&v_avx2[12]);
            A1 = _mm256_loadu_si256((__m256i*)&v_avx2[16]);
            B1 = _mm256_loadu_si256((__m256i*)&v_avx2[20]);
            C1 = _mm256_loadu_si256((__m256i*)&v_avx2[24]);
            D1 = _mm256_loadu_si256((__m256i*)&v_avx2[28]);
            
            DIAGONALIZE_2(A0, A1, B0, B1, C0, C1, D0, D1);
            
            _mm256_storeu_si256((__m256i*)&v_avx2[0], A0);
            _mm256_storeu_si256((__m256i*)&v_avx2[4], B0);
            _mm256_storeu_si256((__m256i*)&v_avx2[8], C0);
            _mm256_storeu_si256((__m256i*)&v_avx2[12], D0);
            _mm256_storeu_si256((__m256i*)&v_avx2[16], A1);
            _mm256_storeu_si256((__m256i*)&v_avx2[20], B1);
            _mm256_storeu_si256((__m256i*)&v_avx2[24], C1);
            _mm256_storeu_si256((__m256i*)&v_avx2[28], D1);
            
#ifdef __AVX512F__
            BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, v_avx512, 0, 16);
            DIAGONALIZE_2_AVX512_DUAL(A, B, C, D);
            BLAKE2_STORE_COLUMNS_AVX512(v_avx512, A, B, C, D, 0, 16);
#endif
            
            for (int i = 0; i < 32; i++) {
                if (v_avx2[i] != v_avx512[i]) {
                    printf("  [%2d]: AVX2=0x%016llx, AVX512=0x%016llx\n", 
                           i, (unsigned long long)v_avx2[i], 
                           (unsigned long long)v_avx512[i]);
                }
            }
        }
    }
}

#define ARGON2_BLOCK_SIZE 128
#define ARGON2_QWORDS_IN_BLOCK ARGON2_BLOCK_SIZE
#define ARGON2_HWORDS_IN_BLOCK 32  // Half-words (256-bit words) in a block

typedef struct block {
    uint64_t v[ARGON2_BLOCK_SIZE];
} block;

static void copy_block(block* dst, const block* src) {
    memcpy(dst->v, src->v, sizeof(uint64_t) * ARGON2_QWORDS_IN_BLOCK);
}

static void xor_block(block* dst, const block* src) {
    int i;
    for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
        dst->v[i] ^= src->v[i];
    }
}

// Scalar implementation - DIRECT FROM MINER
static void fill_block(const block *prev_block, const block *ref_block,
                       block *next_block, int with_xor) {
    block blockR, block_tmp;
    unsigned i;

    copy_block(&blockR, ref_block);
    xor_block(&blockR, prev_block);
    copy_block(&block_tmp, &blockR);
    /* Now blockR = ref_block + prev_block and block_tmp = ref_block + prev_block */
    if (with_xor) {
        /* Saving the next block contents for XOR over: */
        xor_block(&block_tmp, next_block);
        /* Now blockR = ref_block + prev_block and
           block_tmp = ref_block + prev_block + next_block */
    }

    /* Apply Blake2 on columns of 64-bit words: (0,1,...,15) , then
       (16,17,..31)... finally (112,113,...127) */
    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND_NOMSG(
            blockR.v[16 * i], blockR.v[16 * i + 1], blockR.v[16 * i + 2],
            blockR.v[16 * i + 3], blockR.v[16 * i + 4], blockR.v[16 * i + 5],
            blockR.v[16 * i + 6], blockR.v[16 * i + 7], blockR.v[16 * i + 8],
            blockR.v[16 * i + 9], blockR.v[16 * i + 10], blockR.v[16 * i + 11],
            blockR.v[16 * i + 12], blockR.v[16 * i + 13], blockR.v[16 * i + 14],
            blockR.v[16 * i + 15]);
    }

    /* Apply Blake2 on rows of 64-bit words: (0,1,16,17,...112,113), then
       (2,3,18,19,...,114,115).. finally (14,15,30,31,...,126,127) */
    for (i = 0; i < 8; i++) {
        BLAKE2_ROUND_NOMSG(
            blockR.v[2 * i], blockR.v[2 * i + 1], blockR.v[2 * i + 16],
            blockR.v[2 * i + 17], blockR.v[2 * i + 32], blockR.v[2 * i + 33],
            blockR.v[2 * i + 48], blockR.v[2 * i + 49], blockR.v[2 * i + 64],
            blockR.v[2 * i + 65], blockR.v[2 * i + 80], blockR.v[2 * i + 81],
            blockR.v[2 * i + 96], blockR.v[2 * i + 97], blockR.v[2 * i + 112],
            blockR.v[2 * i + 113]);
    }

    copy_block(next_block, &block_tmp);
    xor_block(next_block, &blockR);
}

// AVX2 implementation - DIRECT FROM MINER
static void fill_block_avx2(__m256i* state, const block* ref_block,
                           block* next_block, int with_xor) {
    __m256i block_XY[ARGON2_HWORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            state[i] = _mm256_xor_si256(
                state[i], _mm256_loadu_si256((const __m256i*)ref_block->v + i));
            block_XY[i] = _mm256_xor_si256(
                state[i], _mm256_loadu_si256((const __m256i*)next_block->v + i));
        }
    }
    else {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = _mm256_xor_si256(
                state[i], _mm256_loadu_si256((const __m256i*)ref_block->v + i));
        }
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND_1(state[8 * i + 0], state[8 * i + 4], state[8 * i + 1], state[8 * i + 5],
            state[8 * i + 2], state[8 * i + 6], state[8 * i + 3], state[8 * i + 7]);
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND_2(state[0 + i], state[4 + i], state[8 + i], state[12 + i],
            state[16 + i], state[20 + i], state[24 + i], state[28 + i]);
    }

    for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
        state[i] = _mm256_xor_si256(state[i], block_XY[i]);
        _mm256_storeu_si256((__m256i*)next_block->v + i, state[i]);
    }
}

// AVX512 implementation
static void fill_block_avx512(__m512i* state, const block* ref_block,
                             block* next_block, int with_xor) {
    uint64_t temp_block[ARGON2_BLOCK_SIZE];  // Temporary storage for intermediate results
    __m512i block_XY[ARGON2_HWORDS_IN_BLOCK / 2];
    unsigned int i;

    // State is already loaded by the caller (via memcpy in the miner)

    // XOR ref_block into state and prepare block_XY
    if (with_xor) {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK / 2; i++) {
            state[i] = _mm512_xor_si512(
                state[i], _mm512_loadu_si512((const __m512i*)ref_block->v + i));
            block_XY[i] = _mm512_xor_si512(
                state[i], _mm512_loadu_si512((const __m512i*)next_block->v + i));
        }
    } else {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK / 2; i++) {
            block_XY[i] = state[i] = _mm512_xor_si512(
                state[i], _mm512_loadu_si512((const __m512i*)ref_block->v + i));
        }
    }

    // Copy state to temp_block for column processing
    for (i = 0; i < ARGON2_HWORDS_IN_BLOCK / 2; i++) {
        _mm512_storeu_si512((__m512i*)temp_block + i, state[i]);
    }

    // Column-wise rounds
    for (i = 0; i < 4; ++i) {
        __m512i A, B, C, D;

        BLAKE2_LOAD_COLUMNS_AVX512(A, B, C, D, temp_block, i * 32, i * 32 + 16);
        BLAKE2_ROUND_1_AVX512_DUAL(A, B, C, D);
        BLAKE2_STORE_COLUMNS_AVX512(temp_block, A, B, C, D, i * 32, i * 32 + 16);
    }

    // Row-wise rounds
    for (i = 0; i < 4; ++i) {
        __m512i A, B, C, D;
        BLAKE2_LOAD_ROWS_AVX512(A, B, C, D, temp_block, i);
        BLAKE2_ROUND_2_AVX512_DUAL(A, B, C, D);
        BLAKE2_STORE_ROWS_AVX512(temp_block, A, B, C, D, i);
    }

    // Load processed data back to state
    for (i = 0; i < ARGON2_HWORDS_IN_BLOCK / 2; i++) {
        state[i] = _mm512_loadu_si512((__m512i*)temp_block + i);
    }

    // XOR state with block_XY and store to next_block
    for (i = 0; i < ARGON2_HWORDS_IN_BLOCK / 2; i++) {
        state[i] = _mm512_xor_si512(state[i], block_XY[i]);
        _mm512_storeu_si512((__m512i*)next_block->v + i, state[i]);
    }
}

// Test function to validate all implementations
bool test_fill_block_miner_ready() {
    printf("\n=== Testing fill_block with Miner-Ready State Management ===\n");
    printf("=========================================================\n");
    
    // --- set up blocks and states for all three implementations ---
    block prev_block, ref_block;
    block next_block_scalar, next_block_avx2, next_block_avx512;
    
    __m256i state_avx2[ARGON2_HWORDS_IN_BLOCK];
    __m512i state_avx512[ARGON2_HWORDS_IN_BLOCK / 2];
    
    // Initialize with deterministic patterns
    for (int i = 0; i < ARGON2_BLOCK_SIZE; i++) {
        prev_block.v[i] = 0x3000000000000000ULL + i;  // Same as state init below
        ref_block.v[i] = 0x1000000000000000ULL + i;
        next_block_scalar.v[i] = 0x2000000000000000ULL + i;
        next_block_avx2.v[i] = 0x2000000000000000ULL + i;
        next_block_avx512.v[i] = 0x2000000000000000ULL + i;
    }
    
    // Initialize state arrays exactly as in your test
    for(int i = 0; i < ARGON2_HWORDS_IN_BLOCK; ++i) {
        uint64_t* ptr = (uint64_t*)&state_avx2[i];
        for(int j = 0; j < 4; ++j) {
            ptr[j] = 0x3000000000000000ULL + (i*4 + j);
        }
    }
    
    for(int i = 0; i < ARGON2_HWORDS_IN_BLOCK / 2; ++i) {
        uint64_t* ptr = (uint64_t*)&state_avx512[i];
        for(int j = 0; j < 8; ++j) {
            ptr[j] = 0x3000000000000000ULL + (i*8 + j);
        }
    }
    
    // --- run all three fill_block variants ---
    printf("\n=== Running fill_block functions (with XOR) ===\n");
    
    printf("Running scalar fill_block...\n");
    fill_block(&prev_block, &ref_block, &next_block_scalar, 1);
    
    printf("Running AVX2 fill_block...\n");
    fill_block_avx2(state_avx2, &ref_block, &next_block_avx2, 1);
    
    printf("Running AVX512 fill_block...\n");
    fill_block_avx512(state_avx512, &ref_block, &next_block_avx512, 1);
    
    // --- compare outputs side by side ---
    printf("\n=== Results Comparison ===\n");
    printf("\nScalar Result Block (first 32 words):\n");
    for(int i = 0; i < 32; ++i) {
        printf(" [%2d]: %016llx\n", i, next_block_scalar.v[i]);
    }
    
    printf("\nAVX2 Result Block (first 32 words):\n");
    for(int i = 0; i < 32; ++i) {
        printf(" [%2d]: %016llx\n", i, next_block_avx2.v[i]);
    }
    
    printf("\nAVX512 Result Block (first 32 words):\n");
    for(int i = 0; i < 32; ++i) {
        printf(" [%2d]: %016llx\n", i, next_block_avx512.v[i]);
    }
    
    // --- detailed diff, showing all values ---
    printf("\n=== Detailed Comparison ===\n");
    bool blocks_match = true;
    int first_diff = -1;
    for(int i = 0; i < 128; ++i) {
        uint64_t sc = next_block_scalar.v[i];
        uint64_t a2 = next_block_avx2.v[i];
        uint64_t a5 = next_block_avx512.v[i];
        
        if(sc != a2 || sc != a5) {
            if(first_diff < 0) first_diff = i;
            blocks_match = false;
            printf(
              "DIFF[%3d]: Scalar=%016llx  AVX2=%016llx  AVX512=%016llx\n",
               i, sc, a2, a5
            );
        }
    }
    
    if(blocks_match) {
        printf("\nSUCCESS: All three implementations agree!\n");
    } else {
        printf("\nMISMATCH: first difference at index %d\n", first_diff);
    }
    
    // Also test without XOR
    printf("\n=== Running fill_block functions (without XOR) ===\n");
    
    // Reset next blocks
    for (int i = 0; i < ARGON2_BLOCK_SIZE; i++) {
        next_block_scalar.v[i] = 0x2000000000000000ULL + i;
        next_block_avx2.v[i] = 0x2000000000000000ULL + i;
        next_block_avx512.v[i] = 0x2000000000000000ULL + i;
    }
    
    // Reset state arrays
    for(int i = 0; i < ARGON2_HWORDS_IN_BLOCK; ++i) {
        uint64_t* ptr = (uint64_t*)&state_avx2[i];
        for(int j = 0; j < 4; ++j) {
            ptr[j] = 0x3000000000000000ULL + (i*4 + j);
        }
    }
    
    for(int i = 0; i < ARGON2_HWORDS_IN_BLOCK / 2; ++i) {
        uint64_t* ptr = (uint64_t*)&state_avx512[i];
        for(int j = 0; j < 8; ++j) {
            ptr[j] = 0x3000000000000000ULL + (i*8 + j);
        }
    }
    
    printf("Running scalar fill_block...\n");
    fill_block(&prev_block, &ref_block, &next_block_scalar, 0);
    
    printf("Running AVX2 fill_block...\n");
    fill_block_avx2(state_avx2, &ref_block, &next_block_avx2, 0);
    
    printf("Running AVX512 fill_block...\n");
    fill_block_avx512(state_avx512, &ref_block, &next_block_avx512, 0);
    
    // Check results without XOR
    printf("\n=== Results Without XOR ===\n");
    bool blocks_match_no_xor = true;
    int first_diff_no_xor = -1;
    for(int i = 0; i < 128; ++i) {
        uint64_t sc = next_block_scalar.v[i];
        uint64_t a2 = next_block_avx2.v[i];
        uint64_t a5 = next_block_avx512.v[i];
        
        if(sc != a2 || sc != a5) {
            if(first_diff_no_xor < 0) first_diff_no_xor = i;
            blocks_match_no_xor = false;
            if (i < 32) {  // Show only first 32 mismatches
                printf(
                  "DIFF[%3d]: Scalar=%016llx  AVX2=%016llx  AVX512=%016llx\n",
                   i, sc, a2, a5
                );
            }
        }
    }
    
    if(blocks_match_no_xor) {
        printf("\nSUCCESS: All three implementations agree without XOR!\n");
    } else {
        printf("\nMISMATCH: first difference at index %d\n", first_diff_no_xor);
    }
    
    // Final summary
    printf("\n=== Test Summary ===\n");
    printf("With XOR: %s\n", blocks_match ? "PASS" : "FAIL");
    printf("Without XOR: %s\n", blocks_match_no_xor ? "PASS" : "FAIL");
    
    if (blocks_match && blocks_match_no_xor) {
        printf("\n✓✓✓ All tests PASSED! The AVX512 implementation is ready for your miner.\n");
    }
    
    return blocks_match && blocks_match_no_xor;
}

int main() {
    test_g_operations_only();
    test_diagonalize_operations();
    // test_blake2_round_with_diagonalization();
    test_blake2_both_rounds();
    test_fill_block_miner_ready();
}