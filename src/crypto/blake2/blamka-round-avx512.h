/*
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

#ifndef BLAKE_ROUND_MKA_AVX512_H
#define BLAKE_ROUND_MKA_AVX512_H

#include "blake2-impl.h"

#ifdef __GNUC__
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

#define rotr32_512(x)   _mm512_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24_512(x)   _mm512_shuffle_epi8(x, _mm512_set_epi8( \
    59, 60, 61, 62, 63, 56, 57, 58, 51, 52, 53, 54, 55, 48, 49, 50, \
    43, 44, 45, 46, 47, 40, 41, 42, 35, 36, 37, 38, 39, 32, 33, 34, \
    27, 28, 29, 30, 31, 24, 25, 26, 19, 20, 21, 22, 23, 16, 17, 18, \
    11, 12, 13, 14, 15,  8,  9, 10,  3,  4,  5,  6,  7,  0,  1,  2))
#define rotr16_512(x)   _mm512_shuffle_epi8(x, _mm512_set_epi8( \
    58, 59, 60, 61, 62, 63, 56, 57, 50, 51, 52, 53, 54, 55, 48, 49, \
    42, 43, 44, 45, 46, 47, 40, 41, 34, 35, 36, 37, 38, 39, 32, 33, \
    26, 27, 28, 29, 30, 31, 24, 25, 18, 19, 20, 21, 22, 23, 16, 17, \
    10, 11, 12, 13, 14, 15,  8,  9,  2,  3,  4,  5,  6,  7,  0,  1))
#define rotr63_512(x)   _mm512_xor_si512(_mm512_srli_epi64((x), 63), _mm512_add_epi64((x), (x)))

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

#define DIAGONALIZE_512_1(A, B, C, D) \
    do { \
        B = _mm512_permutex_epi64(B, _MM_SHUFFLE(0, 3, 2, 1)); \
        C = _mm512_permutex_epi64(C, _MM_SHUFFLE(1, 0, 3, 2)); \
        D = _mm512_permutex_epi64(D, _MM_SHUFFLE(2, 1, 0, 3)); \
    } while((void)0, 0);

#define UNDIAGONALIZE_512_1(A, B, C, D) \
    do { \
        B = _mm512_permutex_epi64(B, _MM_SHUFFLE(2, 1, 0, 3)); \
        C = _mm512_permutex_epi64(C, _MM_SHUFFLE(1, 0, 3, 2)); \
        D = _mm512_permutex_epi64(D, _MM_SHUFFLE(0, 3, 2, 1)); \
    } while((void)0, 0);

#define BLAKE2_ROUND_1(A0, A1, A2, A3, A4, A5, A6, A7) \
    do{ \
        G1_AVX512(A0, A2, A4, A6) \
        G1_AVX512(A1, A3, A5, A7) \
        G2_AVX512(A0, A2, A4, A6) \
        G2_AVX512(A1, A3, A5, A7) \
        \
        DIAGONALIZE_512_1(A0, A2, A4, A6) \
        DIAGONALIZE_512_1(A1, A3, A5, A7) \
        \
        G1_AVX512(A0, A2, A4, A6) \
        G1_AVX512(A1, A3, A5, A7) \
        G2_AVX512(A0, A2, A4, A6) \
        G2_AVX512(A1, A3, A5, A7) \
        \
        UNDIAGONALIZE_512_1(A0, A2, A4, A6) \
        UNDIAGONALIZE_512_1(A1, A3, A5, A7) \
    } while((void)0, 0);

#define DIAGONALIZE_512_2(A0, A1, B0, B1, C0, C1, D0, D1) \
    do { \
        __m512i tmp1 = _mm512_shuffle_i64x2(B0, B1, _MM_SHUFFLE(3, 2, 1, 0)); \
        __m512i tmp2 = _mm512_shuffle_i64x2(B0, B1, _MM_SHUFFLE(1, 0, 3, 2)); \
        B0 = tmp2; \
        B1 = tmp1; \
        \
        tmp1 = _mm512_shuffle_i64x2(C0, C1, _MM_SHUFFLE(2, 3, 0, 1)); \
        tmp2 = _mm512_shuffle_i64x2(C0, C1, _MM_SHUFFLE(0, 1, 2, 3)); \
        C0 = tmp1; \
        C1 = tmp2; \
        \
        tmp1 = _mm512_shuffle_i64x2(D0, D1, _MM_SHUFFLE(1, 0, 3, 2)); \
        tmp2 = _mm512_shuffle_i64x2(D0, D1, _MM_SHUFFLE(3, 2, 1, 0)); \
        D0 = tmp1; \
        D1 = tmp2; \
    } while(0);

#define UNDIAGONALIZE_512_2(A0, A1, B0, B1, C0, C1, D0, D1) \
    do { \
        __m512i tmp1 = _mm512_shuffle_i64x2(B0, B1, _MM_SHUFFLE(1, 0, 3, 2)); \
        __m512i tmp2 = _mm512_shuffle_i64x2(B0, B1, _MM_SHUFFLE(3, 2, 1, 0)); \
        B0 = tmp1; \
        B1 = tmp2; \
        \
        tmp1 = _mm512_shuffle_i64x2(C0, C1, _MM_SHUFFLE(2, 3, 0, 1)); \
        tmp2 = _mm512_shuffle_i64x2(C0, C1, _MM_SHUFFLE(0, 1, 2, 3)); \
        C0 = tmp1; \
        C1 = tmp2; \
        \
        tmp1 = _mm512_shuffle_i64x2(D0, D1, _MM_SHUFFLE(1, 0, 3, 2)); \
        tmp2 = _mm512_shuffle_i64x2(D0, D1, _MM_SHUFFLE(3, 2, 1, 0)); \
        D0 = tmp1; \
        D1 = tmp2; \
    } while((void)0, 0);

#define BLAKE2_ROUND_2(A0, A1, B0, B1, C0, C1, D0, D1) \
    do{ \
        G1_AVX512(A0, B0, C0, D0) \
        G1_AVX512(A1, B1, C1, D1) \
        G2_AVX512(A0, B0, C0, D0) \
        G2_AVX512(A1, B1, C1, D1) \
        \
        DIAGONALIZE_512_2(A0, A1, B0, B1, C0, C1, D0, D1) \
        \
        G1_AVX512(A0, B0, C0, D0) \
        G1_AVX512(A1, B1, C1, D1) \
        G2_AVX512(A0, B0, C0, D0) \
        G2_AVX512(A1, B1, C1, D1) \
        \
        UNDIAGONALIZE_512_2(A0, A1, B0, B1, C0, C1, D0, D1) \
    } while((void)0, 0);

#endif /* BLAKE_ROUND_MKA_AVX512_H */