/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Original code from Argon2 reference source code package used under CC0 Licence
 * https://github.com/P-H-C/phc-winner-argon2
 * Copyright 2015
 * Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, and Samuel Neves
*/

/* AVX512 translation - converts AVX2 256-bit operations to AVX512 512-bit operations */

#ifndef BLAKE_ROUND_MKA_OPT_AVX512_H
#define BLAKE_ROUND_MKA_OPT_AVX512_H

#include "blake2-impl.h"

#ifdef __GNUC__
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

// AVX512 rotation macros - operating on 512-bit registers
#define rotr32_512(x)   _mm512_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24_512(x)   _mm512_shuffle_epi8(x, _mm512_broadcast_i32x4(_mm_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10)))
#define rotr16_512(x)   _mm512_shuffle_epi8(x, _mm512_broadcast_i32x4(_mm_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9)))
#define rotr63_512(x)   _mm512_xor_si512(_mm512_srli_epi64((x), 63), _mm512_add_epi64((x), (x)))

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
    } while(0);

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
    } while(0);

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
    } while(0);

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
    } while(0);
    
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
    } while(0);

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
    } while(0);

// 4. Combine G1+G2 for convenience
#define G_AVX512_DUAL(A, B, C, D) \
    do { \
        G1_AVX512_DUAL(A, B, C, D); \
        G2_AVX512_DUAL(A, B, C, D); \
    } while(0);

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
    } while(0);

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
    } while(0);

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
    } while(0);

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
    } while(0);

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
    } while(0);

#endif