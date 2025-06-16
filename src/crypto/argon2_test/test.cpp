/*
 * BLAKE2 AVX2/AVX512 Implementation Validation
 * Tests equivalence between AVX2 and AVX512 BLAKE2 round implementations
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <stdbool.h>

#include "../argon2/argon2_core.h"

// Mock blake2-impl.h (minimal required definitions)
#ifndef BLAKE2_IMPL_H
#define BLAKE2_IMPL_H
// Add any required definitions here if needed
#endif

// Include both AVX2 and AVX512 macro definitions
// AVX2 Macros (from paste-2.txt)
#define rotr32(x)   _mm256_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24(x)   _mm256_shuffle_epi8(x, _mm256_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, 3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10))
#define rotr16(x)   _mm256_shuffle_epi8(x, _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9))
#define rotr63(x)   _mm256_xor_si256(_mm256_srli_epi64((x), 63), _mm256_add_epi64((x), (x)))

// AVX512 Macros (from paste.txt)
#define rotr32_512(x)   _mm512_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24_512(x)   _mm512_shuffle_epi8(x, _mm512_broadcast_i32x4(_mm_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10)))
#define rotr16_512(x)   _mm512_shuffle_epi8(x, _mm512_broadcast_i32x4(_mm_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9)))
#define rotr63_512(x)   _mm512_xor_si512(_mm512_srli_epi64((x), 63), _mm512_add_epi64((x), (x)))

// AVX2 G functions
#define G1_AVX2(A0, A1, B0, B1, C0, C1, D0, D1) \
  do { \
    __m256i ml = _mm256_mul_epu32(A0, B0); \
    ml = _mm256_add_epi64(ml, ml); \
    A0 = _mm256_add_epi64(A0, _mm256_add_epi64(B0, ml)); \
    D0 = _mm256_xor_si256(D0, A0); \
    D0 = rotr32(D0); \
    \
    ml = _mm256_mul_epu32(C0, D0); \
    ml = _mm256_add_epi64(ml, ml); \
    C0 = _mm256_add_epi64(C0, _mm256_add_epi64(D0, ml)); \
    \
    B0 = _mm256_xor_si256(B0, C0); \
    B0 = rotr24(B0); \
    \
    ml = _mm256_mul_epu32(A1, B1); \
    ml = _mm256_add_epi64(ml, ml); \
    A1 = _mm256_add_epi64(A1, _mm256_add_epi64(B1, ml)); \
    D1 = _mm256_xor_si256(D1, A1); \
    D1 = rotr32(D1); \
    \
    ml = _mm256_mul_epu32(C1, D1); \
    ml = _mm256_add_epi64(ml, ml); \
    C1 = _mm256_add_epi64(C1, _mm256_add_epi64(D1, ml)); \
    \
    B1 = _mm256_xor_si256(B1, C1); \
    B1 = rotr24(B1); \
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

#define G1_AVX512(A0, A1, B0, B1, C0, C1, D0, D1)            \
  do {                                        \
    __m512i ml = _mm512_mul_epu32(A0, B0); \
    ml = _mm512_add_epi64(ml, ml); \
    A0 = _mm512_add_epi64(A0, _mm512_add_epi64(B0, ml)); \
    D0 = _mm512_xor_si512(D0, A0); \
    D0 = rotr32_512(D0); \
    \
    ml = _mm512_mul_epu32(C0, D0); \
    ml = _mm512_add_epi64(ml, ml); \
    C0 = _mm512_add_epi64(C0, _mm512_add_epi64(D0, ml)); \
    \
    B0 = _mm512_xor_si512(B0, C0); \
    B0 = rotr24_512(B0); \
    \
    ml = _mm512_mul_epu32(A1, B1); \
    ml = _mm512_add_epi64(ml, ml); \
    A1 = _mm512_add_epi64(A1, _mm512_add_epi64(B1, ml)); \
    D1 = _mm512_xor_si512(D1, A1); \
    D1 = rotr32_512(D1); \
    \
    ml = _mm512_mul_epu32(C1, D1); \
    ml = _mm512_add_epi64(ml, ml); \
    C1 = _mm512_add_epi64(C1, _mm512_add_epi64(D1, ml)); \
    \
    B1 = _mm512_xor_si512(B1, C1); \
    B1 = rotr24_512(B1); \
  } while(0);

#define G2_AVX512(A0, A1, B0, B1, C0, C1, D0, D1)            \
  do {                                        \
    __m512i ml = _mm512_mul_epu32(A0, B0); \
    ml = _mm512_add_epi64(ml, ml); \
    A0 = _mm512_add_epi64(A0, _mm512_add_epi64(B0, ml)); \
    D0 = _mm512_xor_si512(D0, A0); \
    D0 = rotr16_512(D0); \
    \
    ml = _mm512_mul_epu32(C0, D0); \
    ml = _mm512_add_epi64(ml, ml); \
    C0 = _mm512_add_epi64(C0, _mm512_add_epi64(D0, ml)); \
    B0 = _mm512_xor_si512(B0, C0); \
    B0 = rotr63_512(B0); \
    \
    ml = _mm512_mul_epu32(A1, B1); \
    ml = _mm512_add_epi64(ml, ml); \
    A1 = _mm512_add_epi64(A1, _mm512_add_epi64(B1, ml)); \
    D1 = _mm512_xor_si512(D1, A1); \
    D1 = rotr16_512(D1); \
    \
    ml = _mm512_mul_epu32(C1, D1); \
    ml = _mm512_add_epi64(ml, ml); \
    C1 = _mm512_add_epi64(C1, _mm512_add_epi64(D1, ml)); \
    B1 = _mm512_xor_si512(B1, C1); \
    B1 = rotr63_512(B1); \
  } while(0);

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
    } while(0);

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

// AVX512 Diagonalization macros
#define DIAGONALIZE_AVX512(A, B, C, D, A1, B1, C1, D1) \
    do { \
        B = _mm512_permutex_epi64(B, _MM_SHUFFLE(0, 3, 2, 1)); \
        C = _mm512_permutex_epi64(C, _MM_SHUFFLE(1, 0, 3, 2)); \
        D = _mm512_permutex_epi64(D, _MM_SHUFFLE(2, 1, 0, 3)); \
        \
        B1 = _mm512_permutex_epi64(B1, _MM_SHUFFLE(0, 3, 2, 1)); \
        C1 = _mm512_permutex_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
        D1 = _mm512_permutex_epi64(D1, _MM_SHUFFLE(2, 1, 0, 3)); \
    } while((void)0, 0);

#define UNDIAGONALIZE_AVX512(A, B, C, D, A1, B1, C1, D1) \
    do { \
        B = _mm512_permutex_epi64(B, _MM_SHUFFLE(2, 1, 0, 3)); \
        C = _mm512_permutex_epi64(C, _MM_SHUFFLE(1, 0, 3, 2)); \
        D = _mm512_permutex_epi64(D, _MM_SHUFFLE(0, 3, 2, 1)); \
        \
        B1 = _mm512_permutex_epi64(B1, _MM_SHUFFLE(2, 1, 0, 3)); \
        C1 = _mm512_permutex_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
        D1 = _mm512_permutex_epi64(D1, _MM_SHUFFLE(0, 3, 2, 1)); \
    } while((void)0, 0);

static const int _diagR2_mask64_hi = 0xCC33;
static const int _diagR2_mask64_lo = 0x33CC;

__m512i mskA = _mm512_set_epi64(2,7,0,5,6,3,4,1);
__m512i mskB = _mm512_set_epi64(6,3,4,1,2,7,0,5);

#define DIAGONALIZE_AVX512_R2(A0, A1, B0, B1, C0, C1, D0, D1)         \
do {                                                                  \
  __m512i tmp1 = _mm512_mask_blend_epi64(_diagR2_mask64_hi, B0, B1);  \
  __m512i tmp2 = _mm512_mask_blend_epi64(_diagR2_mask64_lo, B0, B1);  \
  B0 = _mm512_permutexvar_epi64(mskA, tmp1);                          \
  B1 = _mm512_permutexvar_epi64(mskA, tmp2);                          \
  \
  tmp1 = C0;                                                          \
  C0 = _mm512_shuffle_i64x2(C1, C1, 0x4E);                            \
  C1 = _mm512_shuffle_i64x2(tmp1, tmp1, 0x4E);                        \
  \
  tmp1 = _mm512_mask_blend_epi64(_diagR2_mask64_hi, D0, D1);          \
  tmp2 = _mm512_mask_blend_epi64(_diagR2_mask64_lo, D0, D1);          \
  D0 = _mm512_permutexvar_epi64(mskB, tmp1);                          \
  D1 = _mm512_permutexvar_epi64(mskB, tmp2);                          \
} while (0);

#define UNDIAGONALIZE_AVX512_R2(A0, A1, B0, B1, C0, C1, D0, D1)       \
do {                                                                  \
  __m512i tmp1 = _mm512_permutexvar_epi64(mskB, B0);                  \
  __m512i tmp2 = _mm512_permutexvar_epi64(mskB, B1);                  \
  B0 = _mm512_mask_blend_epi64(_diagR2_mask64_hi, tmp1, tmp2);        \
  B1 = _mm512_mask_blend_epi64(_diagR2_mask64_lo, tmp1, tmp2);        \
  \
  tmp1 = _mm512_shuffle_i64x2(C0, C0, 0x4E);                          \
  C0 = _mm512_shuffle_i64x2(C1, C1, 0x4E);                            \
  C1 = tmp1;                                                          \
  \
  tmp1 = _mm512_permutexvar_epi64(mskA, D0);                          \
  tmp2 = _mm512_permutexvar_epi64(mskA, D1);                          \
  D0 = _mm512_mask_blend_epi64(_diagR2_mask64_hi, tmp1, tmp2);        \
  D1 = _mm512_mask_blend_epi64(_diagR2_mask64_lo, tmp1, tmp2);        \
} while (0);

// Complete round macros
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

#define BLAKE2_ROUND_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
    do{ \
        G1_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
        G2_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        DIAGONALIZE_AVX512(A0, B0, C0, D0, A1, B1, C1, D1) \
        \
        G1_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
        G2_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        UNDIAGONALIZE_AVX512(A0, B0, C0, D0, A1, B1, C1, D1) \
    } while((void)0, 0);

#define BLAKE2_ROUND_AVX512_R2(A0, A1, B0, B1, C0, C1, D0, D1)   \
    do{ \
        G1_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
        G2_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        DIAGONALIZE_AVX512_R2(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        G1_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
        G2_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        UNDIAGONALIZE_AVX512_R2(A0, A1, B0, B1, C0, C1, D0, D1) \
    } while((void)0, 0);

// Utility functions for testing
void print_m256i(const char* name, __m256i vec) {
    uint64_t data[4];
    _mm256_storeu_si256((__m256i*)data, vec);
    printf("%s: %016lx %016lx %016lx %016lx\n", name, data[3], data[2], data[1], data[0]);
}

void print_m512i(const char* name, __m512i vec) {
    uint64_t data[8];
    _mm512_storeu_si512((__m512i*)data, vec);
    printf("%s: %016lx %016lx %016lx %016lx %016lx %016lx %016lx %016lx\n", 
           name, data[7], data[6], data[5], data[4], data[3], data[2], data[1], data[0]);
}

bool compare_vectors(__m256i avx2_0, __m256i avx2_1, __m512i avx512) {
    uint64_t avx2_data[8];
    uint64_t avx512_data[8];
    
    _mm256_storeu_si256((__m256i*)&avx2_data[0], avx2_0);
    _mm256_storeu_si256((__m256i*)&avx2_data[4], avx2_1);
    _mm512_storeu_si512((__m512i*)avx512_data, avx512);
    
    return memcmp(avx2_data, avx512_data, 64) == 0;
}

// Test rotation functions
bool test_rotations() {
    printf("Testing rotation functions...\n");
    
    // Test data
    uint64_t test_data[8] = {
        0x123456789abcdef0ULL, 0xfedcba9876543210ULL,
        0x0f1e2d3c4b5a6978ULL, 0x8796a5b4c3d2e1f0ULL,
        0x1111222233334444ULL, 0x5555666677778888ULL,
        0x9999aaaabbbbccccULL, 0xddddeeeeffffULL
    };
    
    __m256i avx2_0 = _mm256_loadu_si256((__m256i*)&test_data[0]);
    __m256i avx2_1 = _mm256_loadu_si256((__m256i*)&test_data[4]);
    __m512i avx512 = _mm512_loadu_si512((__m512i*)test_data);
    
    bool all_passed = true;
    
    // Test rotr32
    __m256i avx2_0_rot32 = rotr32(avx2_0);
    __m256i avx2_1_rot32 = rotr32(avx2_1);
    __m512i avx512_rot32 = rotr32_512(avx512);
    
    if (!compare_vectors(avx2_0_rot32, avx2_1_rot32, avx512_rot32)) {
        printf("FAIL: rotr32 mismatch\n");
        print_m256i("AVX2_0 rotr32", avx2_0_rot32);
        print_m256i("AVX2_1 rotr32", avx2_1_rot32);
        print_m512i("AVX512 rotr32", avx512_rot32);
        all_passed = false;
    } else {
        printf("PASS: rotr32\n");
    }
    
    // Test rotr24
    __m256i avx2_0_rot24 = rotr24(avx2_0);
    __m256i avx2_1_rot24 = rotr24(avx2_1);
    __m512i avx512_rot24 = rotr24_512(avx512);
    
    if (!compare_vectors(avx2_0_rot24, avx2_1_rot24, avx512_rot24)) {
        printf("FAIL: rotr24 mismatch\n");
        print_m256i("AVX2_0 rotr24", avx2_0_rot24);
        print_m256i("AVX2_1 rotr24", avx2_1_rot24);
        print_m512i("AVX512 rotr24", avx512_rot24);
        all_passed = false;
    } else {
        printf("PASS: rotr24\n");
    }
    
    // Test rotr16
    __m256i avx2_0_rot16 = rotr16(avx2_0);
    __m256i avx2_1_rot16 = rotr16(avx2_1);
    __m512i avx512_rot16 = rotr16_512(avx512);
    
    if (!compare_vectors(avx2_0_rot16, avx2_1_rot16, avx512_rot16)) {
        printf("FAIL: rotr16 mismatch\n");
        print_m256i("AVX2_0 rotr16", avx2_0_rot16);
        print_m256i("AVX2_1 rotr16", avx2_1_rot16);
        print_m512i("AVX512 rotr16", avx512_rot16);
        all_passed = false;
    } else {
        printf("PASS: rotr16\n");
    }
    
    // Test rotr63
    __m256i avx2_0_rot63 = rotr63(avx2_0);
    __m256i avx2_1_rot63 = rotr63(avx2_1);
    __m512i avx512_rot63 = rotr63_512(avx512);
    
    if (!compare_vectors(avx2_0_rot63, avx2_1_rot63, avx512_rot63)) {
        printf("FAIL: rotr63 mismatch\n");
        print_m256i("AVX2_0 rotr63", avx2_0_rot63);
        print_m256i("AVX2_1 rotr63", avx2_1_rot63);
        print_m512i("AVX512 rotr63", avx512_rot63);
        all_passed = false;
    } else {
        printf("PASS: rotr63\n");
    }
    
    return all_passed;
}

// Helpers to print one 256-vector lane array
static void dump_lane4(const char *name, __m256i v) {
    uint64_t x[4];
    _mm256_storeu_si256((__m256i*)x, v);
    printf("  %s = [%016llx  %016llx  %016llx  %016llx]\n",
           name, x[3], x[2], x[1], x[0]);
}

void debug_round2_xors(__m256i A0, __m256i B0, __m256i C0, __m256i D0) {
    // --- G1 first XOR: D0 ^= A0 ---
    // but A0 must already have been updated by A0 += B0 + ml
    dump_lane4(" pre-G1 D0", D0);
    dump_lane4(" post-A0 (A0')", A0);
    {
        uint64_t d[4], a[4], r[4];
        _mm256_storeu_si256((__m256i*)d, D0);
        _mm256_storeu_si256((__m256i*)a, A0);
        for (int i = 0; i < 4; i++) {
            r[i] = d[i] ^ a[i];
        }
        printf("  D0 ^ A0 = [%016llx  %016llx  %016llx  %016llx]\n",
               r[3], r[2], r[1], r[0]);
    }

    // assume you then do D0 = rotr16(D0 ^ A0) and update C0…

    // --- G1 second XOR: B0 ^= C0 ---
    dump_lane4(" pre-G1 B0", B0);
    dump_lane4(" post-C0 (C0')", C0);
    {
        uint64_t b[4], c[4], r2[4];
        _mm256_storeu_si256((__m256i*)b, B0);
        _mm256_storeu_si256((__m256i*)c, C0);
        for (int i = 0; i < 4; i++) {
            r2[i] = b[i] ^ c[i];
        }
        printf("  B0 ^ C0 = [%016llx  %016llx  %016llx  %016llx]\n",
               r2[3], r2[2], r2[1], r2[0]);
    }
}

bool test_blake2_round() {
    printf("\nTesting full-width BLAKE2 round (16 lanes)...\n");

    // 64 test words = two 512-bit lanes of 8×64-bit each
    uint64_t test_data[64] = {
        // first 32 words (as before)
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
        0x0123456789abcdefULL, 0xfedcba9876543210ULL,
        0x1111222233334444ULL, 0x5555666677778888ULL,
        0x9999aaaabbbbccccULL, 0x0000ddddeeeeffffULL,
        0x0f1e2d3c4b5a6978ULL, 0x8796a5b4c3d2e1f0ULL,
        0x0123456789abcdefULL, 0xfedcba9876543210ULL,
        0x1111222233334444ULL, 0x5555666677778888ULL,
        0x9999aaaabbbbccccULL, 0x0000ddddeeeeffffULL,
        0x0f1e2d3c4b5a6978ULL, 0x8796a5b4c3d2e1f0ULL,

        0x6a09e667f3bccA08ULL, 0xbb67ae8584caa83bULL,
        0x3c6ef372fe94fa2bULL, 0xa54ff53a5f1d37f1ULL,
        0x510e527fade683d1ULL, 0x9b05688c2b3e6d1fULL,
        0x1f83d9abfb41be6bULL, 0x5be0cd19137e2279ULL,
        0x6a09e667f3bccA08ULL, 0xbb67ae8584caa83bULL,
        0x3c6ef372fe94fa2bULL, 0xa54ff53a5f1d37f1ULL,
        0x510e527fade683d1ULL, 0x9b05688c2b3e6d1fULL,
        0x1f83d9abfb41be6bULL, 0x5be0cd19137e2279ULL,
        0x1123456789abcdefULL, 0xeedcba9876543210ULL,
        0x2111222233334444ULL, 0x6555666677778888ULL,
        0xa999aaaabbbbccccULL, 0x1000ddddeeeeffffULL,
        0x1f1e2d3c4b5a6978ULL, 0x9796a5b4c3d2e1f0ULL,
        0x1123456789abcdefULL, 0xeedcba9876543210ULL,
        0x2111222233334444ULL, 0x6555666677778888ULL,
        0xa999aaaabbbbccccULL, 0x1000ddddeeeeffffULL,
        0x1f1e2d3c4b5a6978ULL, 0x9796a5b4c3d2e1f0ULL
    };

    // Load four 256-bit chunks (4×64-bit = 16 words)
    __m256i A0_avx2 = _mm256_loadu_si256((__m256i*)&test_data[ 0]);
    __m256i A1_avx2 = _mm256_loadu_si256((__m256i*)&test_data[ 4]);
    __m256i A2_avx2 = _mm256_loadu_si256((__m256i*)&test_data[ 8]);
    __m256i A3_avx2 = _mm256_loadu_si256((__m256i*)&test_data[12]);
    __m256i B0_avx2 = _mm256_loadu_si256((__m256i*)&test_data[16]);
    __m256i B1_avx2 = _mm256_loadu_si256((__m256i*)&test_data[20]);
    __m256i B2_avx2 = _mm256_loadu_si256((__m256i*)&test_data[24]);
    __m256i B3_avx2 = _mm256_loadu_si256((__m256i*)&test_data[28]);
    __m256i C0_avx2 = _mm256_loadu_si256((__m256i*)&test_data[32]);
    __m256i C1_avx2 = _mm256_loadu_si256((__m256i*)&test_data[36]);
    __m256i C2_avx2 = _mm256_loadu_si256((__m256i*)&test_data[40]);
    __m256i C3_avx2 = _mm256_loadu_si256((__m256i*)&test_data[44]);
    __m256i D0_avx2 = _mm256_loadu_si256((__m256i*)&test_data[48]);
    __m256i D1_avx2 = _mm256_loadu_si256((__m256i*)&test_data[52]);
    __m256i D2_avx2 = _mm256_loadu_si256((__m256i*)&test_data[56]);
    __m256i D3_avx2 = _mm256_loadu_si256((__m256i*)&test_data[60]);

    // Pack into two 512-bit registers each (8 lanes)
    __m512i A0_avx512 = _mm512_inserti64x4(_mm512_castsi256_si512(A0_avx2), A1_avx2, 1);
    __m512i A1_avx512 = _mm512_inserti64x4(_mm512_castsi256_si512(A2_avx2), A3_avx2, 1);
    __m512i B0_avx512 = _mm512_inserti64x4(_mm512_castsi256_si512(B0_avx2), B1_avx2, 1);
    __m512i B1_avx512 = _mm512_inserti64x4(_mm512_castsi256_si512(B2_avx2), B3_avx2, 1);
    __m512i C0_avx512 = _mm512_inserti64x4(_mm512_castsi256_si512(C0_avx2), C1_avx2, 1);
    __m512i C1_avx512 = _mm512_inserti64x4(_mm512_castsi256_si512(C2_avx2), C3_avx2, 1);
    __m512i D0_avx512 = _mm512_inserti64x4(_mm512_castsi256_si512(D0_avx2), D1_avx2, 1);
    __m512i D1_avx512 = _mm512_inserti64x4(_mm512_castsi256_si512(D2_avx2), D3_avx2, 1);

    printf("Initial state:\n");
    print_m256i("  A0_avx2", A0_avx2);
    print_m256i("  A1_avx2", A1_avx2);
    print_m512i("  A_avx512", A0_avx512);
    print_m256i("  B0_avx2", B0_avx2);
    print_m256i("  B1_avx2", B1_avx2);
    print_m512i("  B_avx512", B0_avx512);
    print_m256i("  C0_avx2", C0_avx2);
    print_m256i("  C1_avx2", C1_avx2);
    print_m512i("  C_avx512", C0_avx512);
    print_m256i("  D0_avx2", D0_avx2);
    print_m256i("  D1_avx2", D1_avx2);
    print_m512i("  D_avx512", D0_avx512);

    // Row-wise (Round 1): two AVX2 calls vs. one AVX-512 call
    BLAKE2_ROUND_1(             A0_avx2, A1_avx2, B0_avx2, B1_avx2, C0_avx2, C1_avx2, D0_avx2, D1_avx2);
    BLAKE2_ROUND_1(             A2_avx2, A3_avx2, B2_avx2, B3_avx2, C2_avx2, C3_avx2, D2_avx2, D3_avx2);
    BLAKE2_ROUND_AVX512(   A0_avx512, A1_avx512, B0_avx512, B1_avx512,
                               C0_avx512, C1_avx512, D0_avx512, D1_avx512);

    printf("\nAfter Round 1:\n");
    print_m256i("  A0_avx2", A0_avx2);
    print_m256i("  A1_avx2", A1_avx2);
    print_m512i("  A_avx512", A0_avx512);
    print_m256i("  B0_avx2", B0_avx2);
    print_m256i("  B1_avx2", B1_avx2);
    print_m512i("  B_avx512", B0_avx512);
    print_m256i("  C0_avx2", C0_avx2);
    print_m256i("  C1_avx2", C1_avx2);
    print_m512i("  C_avx512", C0_avx512);
    print_m256i("  D0_avx2", D0_avx2);
    print_m256i("  D1_avx2", D1_avx2);
    print_m512i("  D_avx512", D0_avx512);

    // Column-wise (Round 2): two AVX2 vs. one AVX-512
    BLAKE2_ROUND_2(             A0_avx2, A1_avx2, B0_avx2, B1_avx2, C0_avx2, C1_avx2, D0_avx2, D1_avx2);
    BLAKE2_ROUND_2(             A2_avx2, A3_avx2, B2_avx2, B3_avx2, C2_avx2, C3_avx2, D2_avx2, D3_avx2);
    BLAKE2_ROUND_AVX512_R2(A0_avx512, A1_avx512, B0_avx512, B1_avx512,
                               C0_avx512, C1_avx512, D0_avx512, D1_avx512);

    printf("\nAfter Round 2:\n");
    print_m256i("  A0_avx2", A0_avx2);
    print_m256i("  A1_avx2", A1_avx2);
    print_m512i("  A_avx512", A0_avx512);
    print_m256i("  B0_avx2", B0_avx2);
    print_m256i("  B1_avx2", B1_avx2);
    print_m512i("  B_avx512", B0_avx512);
    print_m256i("  C0_avx2", C0_avx2);
    print_m256i("  C1_avx2", C1_avx2);
    print_m512i("  C_avx512", C0_avx512);
    print_m256i("  D0_avx2", D0_avx2);
    print_m256i("  D1_avx2", D1_avx2);
    print_m512i("  D_avx512", D0_avx512);

    // Compare each half
    bool a_match =  compare_vectors(A0_avx2, A1_avx2, A0_avx512)
                  && compare_vectors(A2_avx2, A3_avx2, A1_avx512);
    bool b_match =  compare_vectors(B0_avx2, B1_avx2, B0_avx512)
                  && compare_vectors(B2_avx2, B3_avx2, B1_avx512);
    bool c_match =  compare_vectors(C0_avx2, C1_avx2, C0_avx512)
                  && compare_vectors(C2_avx2, C3_avx2, C1_avx512);
    bool d_match =  compare_vectors(D0_avx2, D1_avx2, D0_avx512)
                  && compare_vectors(D2_avx2, D3_avx2, D1_avx512);

    printf("\nComparison results:\n");
    printf("  A vectors match: %s\n", a_match ? "YES" : "NO");
    printf("  B vectors match: %s\n", b_match ? "YES" : "NO");
    printf("  C vectors match: %s\n", c_match ? "YES" : "NO");
    printf("  D vectors match: %s\n", d_match ? "YES" : "NO");

    return a_match && b_match && c_match && d_match;
}

void print_avx2_avx512_mapping(const block* input_block) {
    // Prepare AVX2 and AVX-512 register state from same block input
    __m256i state2[32];
    __m512i state512[16];

    // Load as done in typical fill_block logic
    for (int i = 0; i < 32; ++i) {
        state2[i] = _mm256_loadu_si256((const __m256i*)(&input_block->v[4*i]));
    }
    for (int i = 0; i < 16; ++i) {
        state512[i] = _mm512_loadu_si512((const __m512i*)(&input_block->v[8*i]));
    }

    printf("Word |   AVX2 reg:lane   |         Value         |   AVX512 reg:lane  |         Value        | Match\n");
    printf("-----+-------------------+-----------------------+--------------------+----------------------+-------\n");
    for (int word = 0; word < 128; ++word) {
        int avx2_reg = word / 4;
        int avx2_lane = word % 4;
        int avx512_reg = word / 8;
        int avx512_lane = word % 8;

        uint64_t avx2_val = ((uint64_t*)&state2[avx2_reg])[avx2_lane];
        uint64_t avx512_val = ((uint64_t*)&state512[avx512_reg])[avx512_lane];

        printf("%4d |  [%2d]:%d           | %016llx |   [%2d]:%d         | %016llx |  %s\n",
            word, avx2_reg, avx2_lane, (unsigned long long)avx2_val,
            avx512_reg, avx512_lane, (unsigned long long)avx512_val,
            avx2_val == avx512_val ? "OK" : "MISMATCH"
        );
    }
}

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

  for (i = 0; i < 2; ++i) {
      BLAKE2_ROUND_1(
          state[8*i+0], state[8*i+4], state[8*i+1], state[8*i+5],
          state[8*i+2], state[8*i+6], state[8*i+3], state[8*i+7]
      );
  }

	// for (i = 0; i < 4; ++i) {
	// 	BLAKE2_ROUND_2(state[0 + i], state[4 + i], state[8 + i], state[12 + i],
	// 		state[16 + i], state[20 + i], state[24 + i], state[28 + i]);
	// }

	for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
		state[i] = _mm256_xor_si256(state[i], block_XY[i]);
		_mm256_storeu_si256((__m256i*)next_block->v + i, state[i]);
	}
}

static void fill_block_avx512(__m512i* state, const block* ref_block,
  block* next_block, int with_xor) {
  __m512i block_XY[ARGON2_512BIT_WORDS_IN_BLOCK];
  unsigned int i;


  // Load and XOR operations
  if (with_xor) {
    for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
      state[i] = _mm512_xor_si512(
        state[i], _mm512_loadu_si512((const __m512i*)ref_block->v + i));
      block_XY[i] = _mm512_xor_si512(
        state[i], _mm512_loadu_si512((const __m512i*)next_block->v + i));
    }
  }
  else {
    for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
      block_XY[i] = state[i] = _mm512_xor_si512(
        state[i], _mm512_loadu_si512((const __m512i*)ref_block->v + i));
    }
  }

  // Row-wise rounds using dual processing (2 rounds per iteration)
  BLAKE2_ROUND_AVX512(
      state[0],state[10], state[7],  state[13],  // A0,A1,B0,B1 
      state[4], state[1], state[6], state[3]   // C0,C1,D0,D1
  );
  
  // Second row-wise round: elements 8-15 (maps to AVX2's second call)  
  BLAKE2_ROUND_AVX512(
      state[2], state[8], state[5], state[15],   // A0,A1,B0,B1
      state[12], state[9], state[13], state[11]  // C0,C1,D0,D1
  );
  // BLAKE2_ROUND_AVX512(
  //     state[8*i+1], state[8*i+5], state[8*i+0], state[8*i+4],  // A0..D0 (block 0)
  //     state[8*i+2], state[8*i+6], state[8*i+3], state[8*i+7]   // A1..D1 (block 1)
  // );

  // // Column-wise rounds using dual processing (2 rounds per iteration)  
  // for (i = 0; i < 2; ++i) {
  //   BLAKE2_ROUND_AVX512_R2(
  //     state[0+i], state[8+i], state[16+i], state[24+i],
  //     state[32+i], state[40+i], state[48+i], state[56+i]
  //   );
  // }

  // Final XOR and store
  for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
      state[i] = _mm512_xor_si512(state[i], block_XY[i]);
      _mm512_storeu_si512((__m512i*)next_block->v + i, state[i]);
  }
}

bool test_fill_block() {
    printf("\nTesting fill_block functions...\n");
    printf("================================\n");
    
    // Create test blocks with recognizable patterns
    block ref_block, next_block_avx2, next_block_avx512;
    __m256i state_avx2[ARGON2_HWORDS_IN_BLOCK];
    __m512i state_avx512[ARGON2_512BIT_WORDS_IN_BLOCK];
    
    // Initialize reference block with pattern
    for (int i = 0; i < 128; i++) {
        ref_block.v[i] = 0x1000000000000000ULL + i;
    }
    
    // Initialize next block with different pattern  
    for (int i = 0; i < 128; i++) {
        next_block_avx2.v[i] = 0x2000000000000000ULL + i;
        next_block_avx512.v[i] = 0x2000000000000000ULL + i;
    }
    
    // Initialize state with another pattern
    for (int i = 0; i < ARGON2_HWORDS_IN_BLOCK/2; i++) {
        uint64_t* state_ptr = (uint64_t*)&state_avx2[i];
        for (int j = 0; j < 4; j++) {
            state_ptr[j] = 0x3000000000000000ULL + (i * 4 + j);
        }
    }
    
    // for (int i = 0; i < 8; ++i) {
    //   __m512i v = _mm512_castsi256_si512(state_avx2[i]); // lower 256
    //   v = _mm512_inserti64x4(v, state_avx2[i+1], 1);     // upper 256
    //   state_avx512[i] = v;
    // }

    for (int i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i+= 2) {
        uint64_t base = 0x3000000000000000ULL + (i * 8);
        state_avx512[i] = _mm512_set_epi64(
            base + 7, base + 6, base + 5, base + 4,
            base + 3, base + 2, base + 1, base + 0
        );

        state_avx512[i+1] = _mm512_set_epi64(
            base + 15, base + 14, base + 13, base + 12,  // upper 4 lanes
            base + 11, base + 10, base + 9, base + 8   // lower 4 lanes
        );
    }

    // Print initial states
    printf("\nInitial States:\n");
    uint64_t* avx2_state_u64 = (uint64_t*)state_avx2;
    uint64_t* avx512_state_u64 = (uint64_t*)state_avx512;
    
    for (int i = 0; i < 16; i++) {
        char label[32];

        snprintf(label, sizeof(label), "  S%d_avx2", i);
        print_m256i(label, state_avx2[i]);
    }
    
    for (int i = 0; i < 8; i++) {
        char label[32];
        snprintf(label, sizeof(label), "  S%d_avx512", i);
        print_m512i(label, state_avx512[i]);
    }

    printf("\nReference Block (first 16 values):\n");
    for (int i = 0; i < 16; i++) {
        printf("  [%d]: %016llx\n", i, ref_block.v[i]);
    }
    
    printf("\nNext Block (first 16 values):\n");
    for (int i = 0; i < 16; i++) {
        printf("  [%d]: %016llx\n", i, next_block_avx2.v[i]);
    }
    
    print_avx2_avx512_mapping(&ref_block);

    // Run fill_block functions
    printf("\n=== Running fill_block functions ===\n");
    
    // AVX2 version
    printf("Running AVX2 fill_block...\n");
    fill_block_avx2(state_avx2, &ref_block, &next_block_avx2, 1);
    
    // AVX512 version  
    printf("Running AVX512 fill_block...\n");
    fill_block_avx512(state_avx512, &ref_block, &next_block_avx512, 1);
    
    // Compare outputs
    printf("\n=== Results Comparison ===\n");
    
    printf("\nAVX2 Result Block (first 32 values):\n");
    for (int i = 0; i < 32; i++) {
        printf("  [%2d]: %016llx\n", i, next_block_avx2.v[i]);
    }
    
    printf("\nAVX512 Result Block (first 32 values):\n");
    for (int i = 0; i < 32; i++) {
        printf("  [%2d]: %016llx\n", i, next_block_avx512.v[i]);
    }
    
    // Compare byte by byte
    printf("\n=== Detailed Comparison ===\n");
    bool blocks_match = true;
    int first_diff = -1;
    
    for (int i = 0; i < 128; i++) {
        if (next_block_avx2.v[i] != next_block_avx512.v[i]) {
            if (first_diff == -1) first_diff = i;
            blocks_match = false;
            if (i < 64) { // Show first 32 differences
                printf("DIFF[%3d]: AVX2=%016llx  AVX512=%016llx\n", 
                       i, next_block_avx2.v[i], next_block_avx512.v[i]);
            }
        }
    }
    
    if (blocks_match) {
        printf("SUCCESS: All values match!\n");
    } else {
        printf("MISMATCH: First difference at index %d\n", first_diff);
        
        // Show state after processing too
        printf("\nFinal States:\n");
        printf("AVX2 State (first 8 values):\n");
        for (int i = 0; i < 8; i++) {
            printf("  [%d]: %016llx\n", i, avx2_state_u64[i]);
        }
        
        printf("\nAVX512 State (first 8 values):\n"); 
        for (int i = 0; i < 8; i++) {
            printf("  [%d]: %016llx\n", i, avx512_state_u64[i]);
        }
    }
    
    return blocks_match;
}

bool test_G1_equivalence() {
    printf("\nTesting G1 Parity\n");
    uint64_t data[16] = {
        0x1111111122222222ULL, 0x3333333344444444ULL,
        0x5555555566666666ULL, 0x7777777788888888ULL,
        0x99999999aaaaaaaaULL, 0xbbbbbbbbccccccccULL,
        0xddddddddffffffffULL, 0x0123456789abcdefULL,
        0xdeadbeefdeadbeefULL, 0xcafebabecafebabeULL,
        0x0badf00d0badf00dULL, 0xfeedfacefeedfaceULL,
        0xfaceb00cfaceb00cULL, 0xabad1deaabad1deaULL,
        0xdefec8eddeadcafeULL, 0x123456789abcdef0ULL
    };

    // load two independent AVX2 halves (4 lanes each) for the first 8 words
    __m256i A0_2 = _mm256_loadu_si256((__m256i*)&data[ 0]);
    __m256i B0_2 = _mm256_loadu_si256((__m256i*)&data[ 2]);
    __m256i C0_2 = _mm256_loadu_si256((__m256i*)&data[ 4]);
    __m256i D0_2 = _mm256_loadu_si256((__m256i*)&data[ 6]);
    // and for the *second* 8 words
    __m256i A1_2 = _mm256_loadu_si256((__m256i*)&data[ 8]);
    __m256i B1_2 = _mm256_loadu_si256((__m256i*)&data[10]);
    __m256i C1_2 = _mm256_loadu_si256((__m256i*)&data[12]);
    __m256i D1_2 = _mm256_loadu_si256((__m256i*)&data[14]);

    // pack each pair into two *distinct* 512-bit registers
    __m512i A0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(A0_2), A1_2, 1);
    __m512i B0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(B0_2), B1_2, 1);
    __m512i C0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(C0_2), C1_2, 1);
    __m512i D0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(D0_2), D1_2, 1);

    __m512i A1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(A0_2), A1_2, 1);
    __m512i B1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(B0_2), B1_2, 1);
    __m512i C1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(C0_2), C1_2, 1);
    __m512i D1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(D0_2), D1_2, 1);

    // ——— G1 on AVX2 (two passes covering 16 lanes) ———
    G1_AVX2(A0_2, A1_2, B0_2, B1_2, C0_2, C1_2, D0_2, D1_2);

    // ——— G1 on AVX-512 (one pass over full 16 lanes) ———
    G1_AVX512( A0_512, A1_512,
               B0_512, B1_512,
               C0_512, C1_512,
               D0_512, D1_512 );
    // (or if you have dual-wide G1_AVX512_DUAL, call that instead)

    // Print results
    printf("=== AVX2 results (two 256-bit halves) ===\n");
    print_m256i("A0_avx2", A0_2);
    print_m256i("A1_avx2", A1_2);
    print_m256i("B0_avx2", B0_2);
    print_m256i("B1_avx2", B1_2);
    print_m256i("C0_avx2", C0_2);
    print_m256i("C1_avx2", C1_2);
    print_m256i("D0_avx2", D0_2);
    print_m256i("D1_avx2", D1_2);

    printf("\n=== AVX-512 results (one 512-bit lane) ===\n");
    print_m512i("A_avx512", A0_512);
    print_m512i("B_avx512", B0_512);
    print_m512i("C_avx512", C0_512);
    print_m512i("D_avx512", D0_512);

    // Compare lane-by-lane
    // extract back to two __m256i for comparison
    __m256i A0_x = _mm512_castsi512_si256(A0_512);
    __m256i A1_x = _mm512_extracti64x4_epi64(A0_512, 1);

    bool okA = compare_vectors(A0_2, A1_2, A0_512);
    bool okB = compare_vectors(B0_2, B1_2, B0_512);
    bool okC = compare_vectors(C0_2, C1_2, C0_512);
    bool okD = compare_vectors(D0_2, D1_2, D0_512);

    return okA && okB && okC && okD;
}

bool test_G2_equivalence() {
    printf("\nTesting G2 Parity\n");
    uint64_t data[16] = {
        0x1111111122222222ULL, 0x3333333344444444ULL,
        0x5555555566666666ULL, 0x7777777788888888ULL,
        0x99999999aaaaaaaaULL, 0xbbbbbbbbccccccccULL,
        0xddddddddffffffffULL, 0x0123456789abcdefULL,
        0xdeadbeefdeadbeefULL, 0xcafebabecafebabeULL,
        0x0badf00d0badf00dULL, 0xfeedfacefeedfaceULL,
        0xfaceb00cfaceb00cULL, 0xabad1deaabad1deaULL,
        0xdefec8eddeadcafeULL, 0x123456789abcdef0ULL
    };

    // load two independent AVX2 halves (4 lanes each) for the first 8 words
    __m256i A0_2 = _mm256_loadu_si256((__m256i*)&data[ 0]);
    __m256i B0_2 = _mm256_loadu_si256((__m256i*)&data[ 2]);
    __m256i C0_2 = _mm256_loadu_si256((__m256i*)&data[ 4]);
    __m256i D0_2 = _mm256_loadu_si256((__m256i*)&data[ 6]);
    // and for the *second* 8 words
    __m256i A1_2 = _mm256_loadu_si256((__m256i*)&data[ 8]);
    __m256i B1_2 = _mm256_loadu_si256((__m256i*)&data[10]);
    __m256i C1_2 = _mm256_loadu_si256((__m256i*)&data[12]);
    __m256i D1_2 = _mm256_loadu_si256((__m256i*)&data[14]);

    // pack each pair into two *distinct* 512-bit registers
    __m512i A0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(A0_2), A1_2, 1);
    __m512i B0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(B0_2), B1_2, 1);
    __m512i C0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(C0_2), C1_2, 1);
    __m512i D0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(D0_2), D1_2, 1);

    __m512i A1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(A0_2), A1_2, 1);
    __m512i B1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(B0_2), B1_2, 1);
    __m512i C1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(C0_2), C1_2, 1);
    __m512i D1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(D0_2), D1_2, 1);

    // ——— G1 on AVX2 (two passes covering 16 lanes) ———
    G2_AVX2(A0_2, A1_2, B0_2, B1_2, C0_2, C1_2, D0_2, D1_2);

    // ——— G1 on AVX-512 (one pass over full 16 lanes) ———
    G2_AVX512( A0_512, A1_512,
               B0_512, B1_512,
               C0_512, C1_512,
               D0_512, D1_512 );
    // (or if you have dual-wide G1_AVX512_DUAL, call that instead)

    // Print results
    printf("=== AVX2 results (two 256-bit halves) ===\n");
    print_m256i("A0_avx2", A0_2);
    print_m256i("A1_avx2", A1_2);
    print_m256i("B0_avx2", B0_2);
    print_m256i("B1_avx2", B1_2);
    print_m256i("C0_avx2", C0_2);
    print_m256i("C1_avx2", C1_2);
    print_m256i("D0_avx2", D0_2);
    print_m256i("D1_avx2", D1_2);

    printf("\n=== AVX-512 results (one 512-bit lane) ===\n");
    print_m512i("A_avx512", A0_512);
    print_m512i("B_avx512", B0_512);
    print_m512i("C_avx512", C0_512);
    print_m512i("D_avx512", D0_512);

    // Compare lane-by-lane
    // extract back to two __m256i for comparison
    __m256i A0_x = _mm512_castsi512_si256(A0_512);
    __m256i A1_x = _mm512_extracti64x4_epi64(A0_512, 1);

    bool okA = compare_vectors(A0_2, A1_2, A0_512);
    bool okB = compare_vectors(B0_2, B1_2, B0_512);
    bool okC = compare_vectors(C0_2, C1_2, C0_512);
    bool okD = compare_vectors(D0_2, D1_2, D0_512);

    return okA && okB && okC && okD;
}

bool test_diagonalize1_equivalence() {
    printf("\nTesting DIAGONALIZE_1 parity\n");

    // 16 distinct 64-bit words → two independent 256-bit lanes
    uint64_t data[16] = {
        /* lane 0 (words 0–7) */ 
        0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008,
        /* lane 1 (words 8–15), choose a different pattern */
        0x1111, 0x2222, 0x3333, 0x4444,
        0x5555, 0x6666, 0x7777, 0x8888
    };

    // ——— load two AVX2 halves ———
    __m256i A0_2 = _mm256_loadu_si256((__m256i*)&data[ 0]);
    __m256i B0_2 = _mm256_loadu_si256((__m256i*)&data[ 2]);
    __m256i C0_2 = _mm256_loadu_si256((__m256i*)&data[ 4]);
    __m256i D0_2 = _mm256_loadu_si256((__m256i*)&data[ 6]);
    __m256i A1_2 = _mm256_loadu_si256((__m256i*)&data[ 8]);
    __m256i B1_2 = _mm256_loadu_si256((__m256i*)&data[10]);
    __m256i C1_2 = _mm256_loadu_si256((__m256i*)&data[12]);
    __m256i D1_2 = _mm256_loadu_si256((__m256i*)&data[14]);

    __m256i A0_o = A0_2, B0_o = B0_2, C0_o = C0_2, D0_o = D0_2;
    __m256i A1_o = A1_2, B1_o = B1_2, C1_o = C1_2, D1_o = D1_2;

    // ——— pack into two 512-bit registers ———
    __m512i A0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(A0_2), A1_2, 1);
    __m512i B0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(B0_2), B1_2, 1);
    __m512i C0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(C0_2), C1_2, 1);
    __m512i D0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(D0_2), D1_2, 1);

    __m512i A1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(A0_2), A1_2, 1);
    __m512i B1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(B0_2), B1_2, 1);
    __m512i C1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(C0_2), C1_2, 1);
    __m512i D1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(D0_2), D1_2, 1);

    // ——— apply the AVX2 DIAGONALIZE_1 macro ———
    DIAGONALIZE_1(A0_2, B0_2, C0_2, D0_2,  A1_2, B1_2, C1_2, D1_2);

    // ——— apply the AVX-512 version (dual-wide) ———
    DIAGONALIZE_AVX512(
        A0_512, B0_512, C0_512, D0_512,
        A1_512, B1_512, C1_512, D1_512
    );

    // ——— print for eyeballing ———
    printf("AVX2 half0 after diag1:\n"); print_m256i(" A0", A0_2); print_m256i(" B0", B0_2);
    printf("AVX2 half1 after diag1:\n"); print_m256i(" A1", A1_2); print_m256i(" B1", B1_2);
    printf("AVX512 block0 after diag1:\n"); print_m512i(" AB", A0_512); /* shows A0..A1 packed */
    printf("AVX512 block1 after diag1:\n"); print_m512i(" CD", C0_512);

    // ——— compare lane‐for‐lane ———
    bool okA = compare_vectors(A0_2, A1_2, A0_512);
    bool okB = compare_vectors(B0_2, B1_2, B0_512);
    bool okC = compare_vectors(C0_2, C1_2, C0_512);
    bool okD = compare_vectors(D0_2, D1_2, D0_512);

    okA &= compare_vectors(A0_2, A1_2, A1_512);
    okB &= compare_vectors(B0_2, B1_2, B1_512);
    okC &= compare_vectors(C0_2, C1_2, C1_512);
    okD &= compare_vectors(D0_2, D1_2, D1_512);

    printf("DIAGONALIZE_1 parity: %s\n", (okA && okB && okC && okD) ? "PASS" : "FAIL");

    // === Stage 2: undiagonalize (round-trip) ===
    UNDIAGONALIZE_1(A0_2, B0_2, C0_2, D0_2,  A1_2, B1_2, C1_2, D1_2);
    UNDIAGONALIZE_AVX512(
        A0_512, B0_512, C0_512, D0_512,
        A1_512, B1_512, C1_512, D1_512
    );

    // compare back to originals
    bool rtA = compare_vectors(A0_2, A1_2, A0_512);
    bool rtB = compare_vectors(B0_2, B1_2, B0_512);
    bool rtC = compare_vectors(C0_2, C1_2, C0_512);
    bool rtD = compare_vectors(D0_2, D1_2, D0_512);

    rtA &= compare_vectors(A0_2, A1_2, A1_512);
    rtB &= compare_vectors(B0_2, B1_2, B1_512);
    rtC &= compare_vectors(C0_2, C1_2, C1_512);
    rtD &= compare_vectors(D0_2, D1_2, D1_512);

    printf("  round-trip  : %s\n", (rtA&&rtB&&rtC&&rtD) ? "PASS":"FAIL");

    return okA&&okB&&okC&&okD && rtA&&rtB&&rtC&&rtD;
}

static void print_compare_m512i(const char* label, __m256i lo, __m256i hi, __m512i ref) {
    uint64_t lo_words[4], hi_words[4], ref_words[8];
    _mm256_storeu_si256((__m256i*)lo_words, lo);
    _mm256_storeu_si256((__m256i*)hi_words, hi);
    _mm512_storeu_si512((__m512i*)ref_words, ref);

    printf("%s:\n", label);
    for (int i = 0; i < 8; ++i) {
        uint64_t avx2_word = (i < 4) ? lo_words[i] : hi_words[i - 4];
        uint64_t avx512_word = ref_words[i];
        printf("  [%d] avx2: %016llx  avx512: %016llx%s\n", i, 
               (unsigned long long)avx2_word, (unsigned long long)avx512_word,
               avx2_word == avx512_word ? "" : "  <== MISMATCH");
    }
}

bool test_diagonalize2_equivalence() {
    printf("\nTesting DIAGONALIZE_2 parity\n");

    // 16 distinct words = two 256-bit lanes
    uint64_t data[16] = {
        /* lane 0 */ 0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL,
                     0x1112131415161718ULL, 0x191a1b1c1d1e1f20ULL,
        /* lane 1 */ 0x2122232425262728ULL, 0x292a2b2c2d2e2f30ULL,
                     0x3132333435363738ULL, 0x393a3b3c3d3e3f40ULL,
        /* lane 2 (distinct) */ 0xa1a2a3a4a5a6a7a8ULL, 0xa9aaabacadaeafb0ULL,
                           0xb1b2b3b4b5b6b7b8ULL, 0xb9babbbcbdbebfc0ULL,
        /* lane 3 */         0xc1c2c3c4c5c6c7c8ULL, 0xc9cacbcccdcecfd0ULL,
                           0xd1d2d3d4d5d6d7d8ULL, 0xd9dadbdcdddedfe0ULL
    };

    // load AVX2 halves
    __m256i A0_2 = _mm256_loadu_si256((__m256i*)&data[ 0]);
    __m256i B0_2 = _mm256_loadu_si256((__m256i*)&data[ 2]);
    __m256i C0_2 = _mm256_loadu_si256((__m256i*)&data[ 4]);
    __m256i D0_2 = _mm256_loadu_si256((__m256i*)&data[ 6]);
    __m256i A1_2 = _mm256_loadu_si256((__m256i*)&data[ 8]);
    __m256i B1_2 = _mm256_loadu_si256((__m256i*)&data[10]);
    __m256i C1_2 = _mm256_loadu_si256((__m256i*)&data[12]);
    __m256i D1_2 = _mm256_loadu_si256((__m256i*)&data[14]);

    // keep copies for round‐trip check
    __m256i A0_o = A0_2, B0_o = B0_2, C0_o = C0_2, D0_o = D0_2;
    __m256i A1_o = A1_2, B1_o = B1_2, C1_o = C1_2, D1_o = D1_2;

    // pack into AVX-512 lanes
    __m512i A0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(A0_2), A1_2, 1);
    __m512i B0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(B0_2), B1_2, 1);
    __m512i C0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(C0_2), C1_2, 1);
    __m512i D0_512 = _mm512_inserti64x4(_mm512_castsi256_si512(D0_2), D1_2, 1);

    __m512i A1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(A0_2), A1_2, 1);
    __m512i B1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(B0_2), B1_2, 1);
    __m512i C1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(C0_2), C1_2, 1);
    __m512i D1_512 = _mm512_inserti64x4(_mm512_castsi256_si512(D0_2), D1_2, 1);

    // ——— do DIAGONALIZE_2 ———
    DIAGONALIZE_2(A0_2, A1_2, B0_2, B1_2, C0_2, C1_2, D0_2, D1_2);
    DIAGONALIZE_AVX512_R2(A0_512, A1_512,  B0_512, B1_512,
                               C0_512, C1_512,  D0_512, D1_512);

    // print
    printf("AVX2 half0 after diag2:\n"); print_m256i(" A0", A0_2); print_m256i(" B0", B0_2);
    printf("AVX2 half1 after diag2:\n"); print_m256i(" A1", A1_2); print_m256i(" B1", B1_2);
    printf("AVX2 half2 after diag2:\n"); print_m256i(" C0", C0_2); print_m256i(" D0", D0_2);
    printf("AVX2 half3 after diag2:\n"); print_m256i(" C1", C1_2); print_m256i(" D1", D1_2);

    printf("AVX512 after diag2:\n");     
    print_m512i(" A0A1", A0_512); print_m512i(" B0B1", B0_512);
    print_m512i(" C0C1", C0_512); print_m512i(" D0D1", D0_512); 
    print_m512i(" A0A1_2", A1_512); print_m512i(" B0B1_2", B1_512);
    print_m512i(" C0C1_2", C1_512); print_m512i(" D0D1_2", D1_512);
    

    // compare
    bool okA = compare_vectors(A0_2, A1_2, A0_512);
    bool okB = compare_vectors(B0_2, B1_2, B0_512);
    bool okC = compare_vectors(C0_2, C1_2, C0_512);
    bool okD = compare_vectors(D0_2, D1_2, D0_512);

    okA &= compare_vectors(A0_2, A1_2, A1_512);
    okB &= compare_vectors(B0_2, B1_2, B1_512);
    okC &= compare_vectors(C0_2, C1_2, C1_512);
    okD &= compare_vectors(D0_2, D1_2, D1_512);
    
    printf("  DIAGONALIZE_2 parity: %s\n", (okA&&okB&&okC&&okD)?"PASS":"FAIL");

    // ——— now undo it and verify round-trip ———
    UNDIAGONALIZE_2(A0_2, A1_2, B0_2, B1_2, C0_2, C1_2, D0_2, D1_2);
    UNDIAGONALIZE_AVX512_R2(A0_512, A1_512, B0_512, B1_512,
                                 C0_512, C1_512, D0_512, D1_512);

    printf("AVX2 half0 after undiag2:\n"); print_m256i(" A0", A0_2); print_m256i(" B0", B0_2);
    printf("AVX2 half1 after undiag2:\n"); print_m256i(" A1", A1_2); print_m256i(" B1", B1_2);
    printf("AVX2 half2 after undiag2:\n"); print_m256i(" C0", C0_2); print_m256i(" D0", D0_2);
    printf("AVX2 half3 after undiag2:\n"); print_m256i(" C1", C1_2); print_m256i(" D1", D1_2);

    printf("AVX512 after undiag2:\n");     
    print_m512i(" A0A1", A0_512); print_m512i(" B0B1", B0_512);
    print_m512i(" C0C1", C0_512); print_m512i(" D0D1", D0_512);

    // compare back to original
    bool rtA = compare_vectors(A0_2, A1_2, A0_512);
    bool rtB = compare_vectors(B0_2, B1_2, B0_512);
    bool rtC = compare_vectors(C0_2, C1_2, C0_512);
    bool rtD = compare_vectors(D0_2, D1_2, D0_512);

    rtA &= compare_vectors(A0_2, A1_2, A1_512);
    rtB &= compare_vectors(B0_2, B1_2, B1_512);
    rtC &= compare_vectors(C0_2, C1_2, C1_512);
    rtD &= compare_vectors(D0_2, D1_2, D1_512);

    printf("  round-trip      : %s\n", (rtA&&rtB&&rtC&&rtD)?"PASS":"FAIL");

    return okA&&okB&&okC&&okD && rtA&&rtB&&rtC&&rtD;
}

bool test_avx2_avx512_packing() {
    printf("Testing AVX2/AVX512 canonical packing...\n");

    // 128 canonical words
    uint64_t canonical[128];
    for (int i = 0; i < 128; i++)
        canonical[i] = 0x3000000000000000ULL + i;

    // AVX2 pack: 32 registers, 4 words each
    __m256i state_avx2[32];
    for (int i = 0; i < 32; i++)
        state_avx2[i] = _mm256_loadu_si256((__m256i*)&canonical[4*i]);

    // AVX512 pack: 16 registers, 8 words each (vertical interleaving)
    __m512i state_avx512[16];
    for (int i = 0; i < 16; ++i)
        state_avx512[i] = _mm512_inserti64x4(
            _mm512_castsi256_si512(state_avx2[i]),    // lo
            state_avx2[i+4],                         // hi
            1);

    // Compare
    bool all_match = true;
    for (int i = 0; i < 16; ++i) {
        bool ok = compare_vectors(state_avx2[i], state_avx2[i+4], state_avx512[i]);
        printf("Register %2d: %s\n", i, ok ? "OK" : "MISMATCH");
        if (!ok) all_match = false;
    }
    printf("Packing %s\n", all_match ? "SUCCESSFUL" : "FAILED");
    return all_match;
}

int main() {
    printf("BLAKE2 AVX2/AVX512 Validation Test\n");
    printf("===================================\n\n");
    
    // Check CPU capabilities
    printf("Checking CPU capabilities...\n");
    printf("Assuming AVX2 and AVX512 are available\n\n");
    
    bool rotation_tests_passed = test_rotations();
    bool g1_tests = test_G1_equivalence();
    bool g2_tests = test_G2_equivalence();
    bool diag1_tests = test_diagonalize1_equivalence();
    bool diag2_tests = test_diagonalize2_equivalence();
    bool round_tests_passed = test_blake2_round();
    bool fill_block_tests_passed = test_fill_block();
    
    printf("\n===================================\n");
    printf("Test Summary:\n");
    printf("Rotation tests: %s\n", rotation_tests_passed ? "PASSED" : "FAILED");
    printf("G1 tests: %s\n", g1_tests ? "PASSED" : "FAILED");
    printf("G2 tests: %s\n", g1_tests ? "PASSED" : "FAILED");
    printf("Diagonalize 1 tests: %s\n", diag1_tests ? "PASSED" : "FAILED");
    printf("Diagonalize 2 tests: %s\n", diag2_tests ? "PASSED" : "FAILED");
    printf("BLAKE2 tests: %s\n", round_tests_passed ? "PASSED" : "FAILED");
    printf("Fill block tests: %s\n", fill_block_tests_passed ? "PASSED" : "FAILED");
    printf("Overall: %s\n", (rotation_tests_passed && round_tests_passed && fill_block_tests_passed) ? "PASSED" : "FAILED");
    
    test_avx2_avx512_packing();

    return (rotation_tests_passed && round_tests_passed && fill_block_tests_passed) ? 0 : 1;
}