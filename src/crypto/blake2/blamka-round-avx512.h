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

// G1 function for AVX512 - processes 8 64-bit lanes instead of 4
#define G1_AVX512(A, B, C, D) \
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
    } while((void)0, 0);

// G2 function for AVX512
#define G2_AVX512(A, B, C, D) \
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
        B = _mm512_xor_si512(B, C); \
        B = rotr63_512(B); \
    } while((void)0, 0);

// Diagonalization for AVX512 - rearranges 8 64-bit lanes
#define DIAGONALIZE_AVX512(A, B, C, D) \
    do { \
        B = _mm512_permutex_epi64(B, _MM_SHUFFLE(0, 3, 2, 1)); \
        C = _mm512_permutex_epi64(C, _MM_SHUFFLE(1, 0, 3, 2)); \
        D = _mm512_permutex_epi64(D, _MM_SHUFFLE(2, 1, 0, 3)); \
    } while((void)0, 0);

// Undiagonalization for AVX512
#define UNDIAGONALIZE_AVX512(A, B, C, D) \
    do { \
        B = _mm512_permutex_epi64(B, _MM_SHUFFLE(2, 1, 0, 3)); \
        C = _mm512_permutex_epi64(C, _MM_SHUFFLE(1, 0, 3, 2)); \
        D = _mm512_permutex_epi64(D, _MM_SHUFFLE(0, 3, 2, 1)); \
    } while((void)0, 0);

// Alternative diagonalization using AVX512 cross-lane operations
#define DIAGONALIZE_CROSS_AVX512(A, B, C, D) \
    do { \
        B = _mm512_shuffle_i64x2(B, B, _MM_SHUFFLE(0, 3, 2, 1)); \
        C = _mm512_shuffle_i64x2(C, C, _MM_SHUFFLE(1, 0, 3, 2)); \
        D = _mm512_shuffle_i64x2(D, D, _MM_SHUFFLE(2, 1, 0, 3)); \
    } while((void)0, 0);

// Alternative undiagonalization using AVX512 cross-lane operations
#define UNDIAGONALIZE_CROSS_AVX512(A, B, C, D) \
    do { \
        B = _mm512_shuffle_i64x2(B, B, _MM_SHUFFLE(2, 1, 0, 3)); \
        C = _mm512_shuffle_i64x2(C, C, _MM_SHUFFLE(1, 0, 3, 2)); \
        D = _mm512_shuffle_i64x2(D, D, _MM_SHUFFLE(0, 3, 2, 1)); \
    } while((void)0, 0);

// Complete BLAKE2 round using AVX512 - operates on single 512-bit vectors
#define BLAKE2_ROUND_AVX512(A, B, C, D) \
    do{ \
        G1_AVX512(A, B, C, D) \
        G2_AVX512(A, B, C, D) \
        \
        DIAGONALIZE_AVX512(A, B, C, D) \
        \
        G1_AVX512(A, B, C, D) \
        G2_AVX512(A, B, C, D) \
        \
        UNDIAGONALIZE_AVX512(A, B, C, D) \
    } while((void)0, 0);

// Alternative BLAKE2 round using cross-lane operations
#define BLAKE2_ROUND_CROSS_AVX512(A, B, C, D) \
    do{ \
        G1_AVX512(A, B, C, D) \
        G2_AVX512(A, B, C, D) \
        \
        DIAGONALIZE_CROSS_AVX512(A, B, C, D) \
        \
        G1_AVX512(A, B, C, D) \
        G2_AVX512(A, B, C, D) \
        \
        UNDIAGONALIZE_CROSS_AVX512(A, B, C, D) \
    } while((void)0, 0);

// Dual-vector version for processing two 512-bit blocks simultaneously
#define BLAKE2_ROUND_DUAL_AVX512(A0, B0, C0, D0, A1, B1, C1, D1) \
    do{ \
        G1_AVX512(A0, B0, C0, D0) \
        G1_AVX512(A1, B1, C1, D1) \
        G2_AVX512(A0, B0, C0, D0) \
        G2_AVX512(A1, B1, C1, D1) \
        \
        DIAGONALIZE_AVX512(A0, B0, C0, D0) \
        DIAGONALIZE_AVX512(A1, B1, C1, D1) \
        \
        G1_AVX512(A0, B0, C0, D0) \
        G1_AVX512(A1, B1, C1, D1) \
        G2_AVX512(A0, B0, C0, D0) \
        G2_AVX512(A1, B1, C1, D1) \
        \
        UNDIAGONALIZE_AVX512(A0, B0, C0, D0) \
        UNDIAGONALIZE_AVX512(A1, B1, C1, D1) \
    } while((void)0, 0);

#endif /* BLAKE_ROUND_MKA_OPT_AVX512_H */