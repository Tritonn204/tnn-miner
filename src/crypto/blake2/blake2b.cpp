/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>

All rights reserved.

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

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "blake2.h"
#include "blake2-impl.h"

#include "compile.h"

static const uint64_t blake2b_IV[8] = {
	UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
	UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
	UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
	UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179) };

static const unsigned int blake2b_sigma[12][16] = {
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
	{7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
	{2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
	{6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

static FORCE_INLINE void blake2b_set_lastnode(blake2b_state *S) {
	S->f[1] = (uint64_t)-1;
}

static FORCE_INLINE void blake2b_set_lastblock(blake2b_state *S) {
	if (S->last_node) {
		blake2b_set_lastnode(S);
	}
	S->f[0] = (uint64_t)-1;
}

static FORCE_INLINE void blake2b_increment_counter(blake2b_state *S,
	uint64_t inc) {
	S->t[0] += inc;
	S->t[1] += (S->t[0] < inc);
}

static FORCE_INLINE void blake2b_invalidate_state(blake2b_state *S) {
	//clear_internal_memory(S, sizeof(*S));      /* wipe */
	blake2b_set_lastblock(S); /* invalidate for further use */
}

static FORCE_INLINE void blake2b_init0(blake2b_state *S) {
	memset(S, 0, sizeof(*S));
	memcpy(S->h, blake2b_IV, sizeof(S->h));
}

int blake2b_init_param(blake2b_state *S, const blake2b_param *P) {
	const unsigned char *p = (const unsigned char *)P;
	unsigned int i;

	if (NULL == P || NULL == S) {
		return -1;
	}

	blake2b_init0(S);
	/* IV XOR Parameter Block */
	for (i = 0; i < 8; ++i) {
		S->h[i] ^= load64(&p[i * sizeof(S->h[i])]);
	}
	S->outlen = P->digest_length;
	return 0;
}

/* Sequential blake2b initialization */
int blake2b_init(blake2b_state *S, size_t outlen) {
	blake2b_param P;

	if (S == NULL) {
		return -1;
	}

	if ((outlen == 0) || (outlen > BLAKE2B_OUTBYTES)) {
		blake2b_invalidate_state(S);
		return -1;
	}

	/* Setup Parameter Block for unkeyed BLAKE2 */
	P.digest_length = (uint8_t)outlen;
	P.key_length = 0;
	P.fanout = 1;
	P.depth = 1;
	P.leaf_length = 0;
	P.node_offset = 0;
	P.node_depth = 0;
	P.inner_length = 0;
	memset(P.reserved, 0, sizeof(P.reserved));
	memset(P.salt, 0, sizeof(P.salt));
	memset(P.personal, 0, sizeof(P.personal));

	return blake2b_init_param(S, &P);
}

int blake2b_init_key(blake2b_state *S, size_t outlen, const void *key, size_t keylen) {
	blake2b_param P;

	if (S == NULL) {
		return -1;
	}

	if ((outlen == 0) || (outlen > BLAKE2B_OUTBYTES)) {
		blake2b_invalidate_state(S);
		return -1;
	}

	if ((key == 0) || (keylen == 0) || (keylen > BLAKE2B_KEYBYTES)) {
		blake2b_invalidate_state(S);
		return -1;
	}

	/* Setup Parameter Block for keyed BLAKE2 */
	P.digest_length = (uint8_t)outlen;
	P.key_length = (uint8_t)keylen;
	P.fanout = 1;
	P.depth = 1;
	P.leaf_length = 0;
	P.node_offset = 0;
	P.node_depth = 0;
	P.inner_length = 0;
	memset(P.reserved, 0, sizeof(P.reserved));
	memset(P.salt, 0, sizeof(P.salt));
	memset(P.personal, 0, sizeof(P.personal));

	if (blake2b_init_param(S, &P) < 0) {
		blake2b_invalidate_state(S);
		return -1;
	}

	{
		uint8_t block[BLAKE2B_BLOCKBYTES];
		memset(block, 0, BLAKE2B_BLOCKBYTES);
		memcpy(block, key, keylen);
		blake2b_update(S, block, BLAKE2B_BLOCKBYTES);
		/* Burn the key from stack */
		//clear_internal_memory(block, BLAKE2B_BLOCKBYTES);
	}
	return 0;
}

#ifdef __x86_64__
#include <x86intrin.h>

__attribute__((target("ssse3")))
static void blake2b_compress(blake2b_state *S, const uint8_t *block) {
    __m128i row1l, row1h, row2l, row2h, row3l, row3h, row4l, row4h;
    __m128i t0, t1;
    __m128i b0, b1;
    uint64_t m[16];
    unsigned int i, r;

    // Rotation constants for SSSE3
    const __m128i r16 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9);
    const __m128i r24 = _mm_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10);

    // Define rotation macros for SSSE3
    #define rotr32_sse(x) _mm_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))
    #define rotr24_sse(x) _mm_shuffle_epi8(x, r24)
    #define rotr16_sse(x) _mm_shuffle_epi8(x, r16)
    #define rotr63_sse(x) _mm_xor_si128(_mm_srli_epi64(x, 63), _mm_add_epi64(x, x))

    // Load message block
    for (i = 0; i < 16; ++i) {
        m[i] = load64(block + i * 8);
    }

    // Initialize working variables (split into low and high parts)
    row1l = _mm_set_epi64x(S->h[1], S->h[0]);
    row1h = _mm_set_epi64x(S->h[3], S->h[2]);
    row2l = _mm_set_epi64x(S->h[5], S->h[4]);
    row2h = _mm_set_epi64x(S->h[7], S->h[6]);
    row3l = _mm_set_epi64x(blake2b_IV[1], blake2b_IV[0]);
    row3h = _mm_set_epi64x(blake2b_IV[3], blake2b_IV[2]);
    row4l = _mm_set_epi64x(blake2b_IV[5] ^ S->t[1], blake2b_IV[4] ^ S->t[0]);
    row4h = _mm_set_epi64x(blake2b_IV[7] ^ S->f[1], blake2b_IV[6] ^ S->f[0]);

    for (r = 0; r < 12; r++) {
        // Column step
        // G(0,4,8,12) and G(1,5,9,13)
        b0 = _mm_set_epi64x(m[blake2b_sigma[r][2]], m[blake2b_sigma[r][0]]);
        b1 = _mm_set_epi64x(m[blake2b_sigma[r][3]], m[blake2b_sigma[r][1]]);

        row1l = _mm_add_epi64(_mm_add_epi64(row1l, row2l), b0);
        row4l = rotr32_sse(_mm_xor_si128(row4l, row1l));
        row3l = _mm_add_epi64(row3l, row4l);
        row2l = rotr24_sse(_mm_xor_si128(row2l, row3l));
        row1l = _mm_add_epi64(_mm_add_epi64(row1l, row2l), b1);
        row4l = rotr16_sse(_mm_xor_si128(row4l, row1l));
        row3l = _mm_add_epi64(row3l, row4l);
        row2l = rotr63_sse(_mm_xor_si128(row2l, row3l));

        // G(2,6,10,14) and G(3,7,11,15)
        b0 = _mm_set_epi64x(m[blake2b_sigma[r][6]], m[blake2b_sigma[r][4]]);
        b1 = _mm_set_epi64x(m[blake2b_sigma[r][7]], m[blake2b_sigma[r][5]]);

        row1h = _mm_add_epi64(_mm_add_epi64(row1h, row2h), b0);
        row4h = rotr32_sse(_mm_xor_si128(row4h, row1h));
        row3h = _mm_add_epi64(row3h, row4h);
        row2h = rotr24_sse(_mm_xor_si128(row2h, row3h));
        row1h = _mm_add_epi64(_mm_add_epi64(row1h, row2h), b1);
        row4h = rotr16_sse(_mm_xor_si128(row4h, row1h));
        row3h = _mm_add_epi64(row3h, row4h);
        row2h = rotr63_sse(_mm_xor_si128(row2h, row3h));

        // Diagonalize
        t0 = _mm_alignr_epi8(row2h, row2l, 8);
        t1 = _mm_alignr_epi8(row2l, row2h, 8);
        row2l = t0;
        row2h = t1;

        t0 = row3l;
        row3l = row3h;
        row3h = t0;

        t0 = _mm_alignr_epi8(row4h, row4l, 8);
        t1 = _mm_alignr_epi8(row4l, row4h, 8);
        row4l = t1;
        row4h = t0;

        // Diagonal step
        // G(0,5,10,15) and G(1,6,11,12)
        b0 = _mm_set_epi64x(m[blake2b_sigma[r][10]], m[blake2b_sigma[r][8]]);
        b1 = _mm_set_epi64x(m[blake2b_sigma[r][11]], m[blake2b_sigma[r][9]]);

        row1l = _mm_add_epi64(_mm_add_epi64(row1l, row2l), b0);
        row4l = rotr32_sse(_mm_xor_si128(row4l, row1l));
        row3l = _mm_add_epi64(row3l, row4l);
        row2l = rotr24_sse(_mm_xor_si128(row2l, row3l));
        row1l = _mm_add_epi64(_mm_add_epi64(row1l, row2l), b1);
        row4l = rotr16_sse(_mm_xor_si128(row4l, row1l));
        row3l = _mm_add_epi64(row3l, row4l);
        row2l = rotr63_sse(_mm_xor_si128(row2l, row3l));

        // G(2,7,8,13) and G(3,4,9,14)
        b0 = _mm_set_epi64x(m[blake2b_sigma[r][14]], m[blake2b_sigma[r][12]]);
        b1 = _mm_set_epi64x(m[blake2b_sigma[r][15]], m[blake2b_sigma[r][13]]);

        row1h = _mm_add_epi64(_mm_add_epi64(row1h, row2h), b0);
        row4h = rotr32_sse(_mm_xor_si128(row4h, row1h));
        row3h = _mm_add_epi64(row3h, row4h);
        row2h = rotr24_sse(_mm_xor_si128(row2h, row3h));
        row1h = _mm_add_epi64(_mm_add_epi64(row1h, row2h), b1);
        row4h = rotr16_sse(_mm_xor_si128(row4h, row1h));
        row3h = _mm_add_epi64(row3h, row4h);
        row2h = rotr63_sse(_mm_xor_si128(row2h, row3h));

        // Undiagonalize
        t0 = _mm_alignr_epi8(row2l, row2h, 8);
        t1 = _mm_alignr_epi8(row2h, row2l, 8);
        row2l = t0;
        row2h = t1;

        t0 = row3l;
        row3l = row3h;
        row3h = t0;

        t0 = _mm_alignr_epi8(row4l, row4h, 8);
        t1 = _mm_alignr_epi8(row4h, row4l, 8);
        row4l = t1;
        row4h = t0;
    }

    // Finalize
    row1l = _mm_xor_si128(_mm_xor_si128(row1l, row3l), _mm_set_epi64x(S->h[1], S->h[0]));
    row1h = _mm_xor_si128(_mm_xor_si128(row1h, row3h), _mm_set_epi64x(S->h[3], S->h[2]));
    row2l = _mm_xor_si128(_mm_xor_si128(row2l, row4l), _mm_set_epi64x(S->h[5], S->h[4]));
    row2h = _mm_xor_si128(_mm_xor_si128(row2h, row4h), _mm_set_epi64x(S->h[7], S->h[6]));

    // Store results
    _mm_storeu_si128((__m128i*)&S->h[0], row1l);
    _mm_storeu_si128((__m128i*)&S->h[2], row1h);
    _mm_storeu_si128((__m128i*)&S->h[4], row2l);
    _mm_storeu_si128((__m128i*)&S->h[6], row2h);

    #undef rotr32_sse
    #undef rotr24_sse
    #undef rotr16_sse
    #undef rotr63_sse
}

#ifndef TNN_LEGACY_AMD64

#define rotr32(x) _mm256_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24(x) _mm256_or_si256(_mm256_srli_epi64(x, 24), _mm256_slli_epi64(x, 40))
#define rotr16(x) _mm256_or_si256(_mm256_srli_epi64(x, 16), _mm256_slli_epi64(x, 48))
#define rotr63(x) _mm256_or_si256(_mm256_srli_epi64(x, 63), _mm256_slli_epi64(x, 1))

TNN_TARGET_CLONE(
  blake2b_compress,
  static void,
  (blake2b_state *S, const uint8_t *block),
  {
    __m256i m[4];
    __m256i row1;
    __m256i row2;
    __m256i row3;
    __m256i row4;
    __m256i t0;
    __m256i t1;
    __m256i b0;
    __m256i b1;

    uint64_t m_scalar[16];
    unsigned int r;

    // Load message block using AVX2
    m[0] = _mm256_loadu_si256((const __m256i*)(block));
    m[1] = _mm256_loadu_si256((const __m256i*)(block + 32));
    m[2] = _mm256_loadu_si256((const __m256i*)(block + 64));
    m[3] = _mm256_loadu_si256((const __m256i*)(block + 96));
    
    // Store to scalar array for sigma indexing
    _mm256_storeu_si256((__m256i*)&m_scalar[0], m[0]);
    _mm256_storeu_si256((__m256i*)&m_scalar[4], m[1]);
    _mm256_storeu_si256((__m256i*)&m_scalar[8], m[2]);
    _mm256_storeu_si256((__m256i*)&m_scalar[12], m[3]);

    // Initialize working variables
    row1 = _mm256_setr_epi64x(S->h[0], S->h[1], S->h[2], S->h[3]);
    row2 = _mm256_setr_epi64x(S->h[4], S->h[5], S->h[6], S->h[7]);
    row3 = _mm256_setr_epi64x(blake2b_IV[0], blake2b_IV[1], blake2b_IV[2], blake2b_IV[3]);
    row4 = _mm256_setr_epi64x(
        blake2b_IV[4] ^ S->t[0],
        blake2b_IV[5] ^ S->t[1],
        blake2b_IV[6] ^ S->f[0],
        blake2b_IV[7] ^ S->f[1]
    );

    for (r = 0; r < 12; r++) {
        // Column step
        b0 = _mm256_setr_epi64x(
            m_scalar[blake2b_sigma[r][0]],
            m_scalar[blake2b_sigma[r][2]],
            m_scalar[blake2b_sigma[r][4]],
            m_scalar[blake2b_sigma[r][6]]
        );
        b1 = _mm256_setr_epi64x(
            m_scalar[blake2b_sigma[r][1]],
            m_scalar[blake2b_sigma[r][3]],
            m_scalar[blake2b_sigma[r][5]],
            m_scalar[blake2b_sigma[r][7]]
        );

        row1 = _mm256_add_epi64(_mm256_add_epi64(row1, row2), b0);
        row4 = rotr32(_mm256_xor_si256(row4, row1));
        row3 = _mm256_add_epi64(row3, row4);
        row2 = rotr24(_mm256_xor_si256(row2, row3));
        row1 = _mm256_add_epi64(_mm256_add_epi64(row1, row2), b1);
        row4 = rotr16(_mm256_xor_si256(row4, row1));
        row3 = _mm256_add_epi64(row3, row4);
        row2 = rotr63(_mm256_xor_si256(row2, row3));

        // Diagonal step
        row2 = _mm256_permute4x64_epi64(row2, _MM_SHUFFLE(0, 3, 2, 1));
        row3 = _mm256_permute4x64_epi64(row3, _MM_SHUFFLE(1, 0, 3, 2));
        row4 = _mm256_permute4x64_epi64(row4, _MM_SHUFFLE(2, 1, 0, 3));

        b0 = _mm256_setr_epi64x(
            m_scalar[blake2b_sigma[r][8]],
            m_scalar[blake2b_sigma[r][10]],
            m_scalar[blake2b_sigma[r][12]],
            m_scalar[blake2b_sigma[r][14]]
        );
        b1 = _mm256_setr_epi64x(
            m_scalar[blake2b_sigma[r][9]],
            m_scalar[blake2b_sigma[r][11]],
            m_scalar[blake2b_sigma[r][13]],
            m_scalar[blake2b_sigma[r][15]]
        );

        row1 = _mm256_add_epi64(_mm256_add_epi64(row1, row2), b0);
        row4 = rotr32(_mm256_xor_si256(row4, row1));
        row3 = _mm256_add_epi64(row3, row4);
        row2 = rotr24(_mm256_xor_si256(row2, row3));
        row1 = _mm256_add_epi64(_mm256_add_epi64(row1, row2), b1);
        row4 = rotr16(_mm256_xor_si256(row4, row1));
        row3 = _mm256_add_epi64(row3, row4);
        row2 = rotr63(_mm256_xor_si256(row2, row3));

        // Undiagonalize
        row2 = _mm256_permute4x64_epi64(row2, _MM_SHUFFLE(2, 1, 0, 3));
        row3 = _mm256_permute4x64_epi64(row3, _MM_SHUFFLE(1, 0, 3, 2));
        row4 = _mm256_permute4x64_epi64(row4, _MM_SHUFFLE(0, 3, 2, 1));
    }

    // Finalize
    t0 = _mm256_xor_si256(_mm256_xor_si256(row1, row3), _mm256_loadu_si256((const __m256i*)&S->h[0]));
    t1 = _mm256_xor_si256(_mm256_xor_si256(row2, row4), _mm256_loadu_si256((const __m256i*)&S->h[4]));

    _mm256_storeu_si256((__m256i*)&S->h[0], t0);
    _mm256_storeu_si256((__m256i*)&S->h[4], t1);
  },
  TNN_TARGETS_X86_AVX2
)

#undef rotr32
#undef rotr24
#undef rotr16
#undef rotr63


#define rotr32_512(x)   _mm512_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24_512(x)   _mm512_shuffle_epi8(x, _mm512_broadcast_i32x4(_mm_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10)))
#define rotr16_512(x)   _mm512_shuffle_epi8(x, _mm512_broadcast_i32x4(_mm_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9)))
#define rotr63_512(x)   _mm512_xor_si512(_mm512_srli_epi64((x), 63), _mm512_add_epi64((x), (x)))

TNN_TARGET_CLONE(
  blake2b_compress,
  static void,
  (blake2b_state *S, const uint8_t *block),
  {
    __m512i row1;
    __m512i row2;
    __m512i row3;
    __m512i row4;
    __m512i m0;
    __m512i m1;

    uint64_t m_scalar[16];
    unsigned int r;

    // Load message using AVX512
    __m512i msg_block[2];
    msg_block[0] = _mm512_loadu_si512((const __m512i*)block);
    msg_block[1] = _mm512_loadu_si512((const __m512i*)(block + 64));
    
    // Store to scalar for sigma indexing
    _mm512_storeu_si512((__m512i*)&m_scalar[0], msg_block[0]);
    _mm512_storeu_si512((__m512i*)&m_scalar[8], msg_block[1]);

    // Initialize state - broadcast 256-bit values to fill 512-bit registers
    __m256i h_low = _mm256_setr_epi64x(S->h[0], S->h[1], S->h[2], S->h[3]);
    __m256i h_high = _mm256_setr_epi64x(S->h[4], S->h[5], S->h[6], S->h[7]);
    __m256i iv_low = _mm256_setr_epi64x(blake2b_IV[0], blake2b_IV[1], blake2b_IV[2], blake2b_IV[3]);
    __m256i iv_high = _mm256_setr_epi64x(
        blake2b_IV[4] ^ S->t[0],
        blake2b_IV[5] ^ S->t[1],
        blake2b_IV[6] ^ S->f[0],
        blake2b_IV[7] ^ S->f[1]
    );
    
    // Duplicate the 256-bit values to fill 512-bit registers
    row1 = _mm512_broadcast_i64x4(h_low);
    row2 = _mm512_broadcast_i64x4(h_high);
    row3 = _mm512_broadcast_i64x4(iv_low);
    row4 = _mm512_broadcast_i64x4(iv_high);

    // Create permutation indices for diagonal/undiagonal
    const __m512i diag_perm = _mm512_setr_epi64(1, 2, 3, 0, 5, 6, 7, 4);
    const __m512i undiag_perm = _mm512_setr_epi64(3, 0, 1, 2, 7, 4, 5, 6);

    for (r = 0; r < 12; r++) {
        // Column step
        m0 = _mm512_setr_epi64(
            m_scalar[blake2b_sigma[r][0]], m_scalar[blake2b_sigma[r][2]], 
            m_scalar[blake2b_sigma[r][4]], m_scalar[blake2b_sigma[r][6]],
            m_scalar[blake2b_sigma[r][0]], m_scalar[blake2b_sigma[r][2]], 
            m_scalar[blake2b_sigma[r][4]], m_scalar[blake2b_sigma[r][6]]
        );
        m1 = _mm512_setr_epi64(
            m_scalar[blake2b_sigma[r][1]], m_scalar[blake2b_sigma[r][3]], 
            m_scalar[blake2b_sigma[r][5]], m_scalar[blake2b_sigma[r][7]],
            m_scalar[blake2b_sigma[r][1]], m_scalar[blake2b_sigma[r][3]], 
            m_scalar[blake2b_sigma[r][5]], m_scalar[blake2b_sigma[r][7]]
        );

        // G function
        row1 = _mm512_add_epi64(_mm512_add_epi64(row1, row2), m0);
        row4 = rotr32_512(_mm512_xor_si512(row4, row1));
        row3 = _mm512_add_epi64(row3, row4);
        row2 = rotr24_512(_mm512_xor_si512(row2, row3));
        row1 = _mm512_add_epi64(_mm512_add_epi64(row1, row2), m1);
        row4 = rotr16_512(_mm512_xor_si512(row4, row1));
        row3 = _mm512_add_epi64(row3, row4);
        row2 = rotr63_512(_mm512_xor_si512(row2, row3));

        // Diagonal
        row2 = _mm512_permutexvar_epi64(diag_perm, row2);
        row3 = _mm512_permutexvar_epi64(_mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5), row3);
        row4 = _mm512_permutexvar_epi64(_mm512_setr_epi64(3, 0, 1, 2, 7, 4, 5, 6), row4);

        // Diagonal step
        m0 = _mm512_setr_epi64(
            m_scalar[blake2b_sigma[r][8]], m_scalar[blake2b_sigma[r][10]], 
            m_scalar[blake2b_sigma[r][12]], m_scalar[blake2b_sigma[r][14]],
            m_scalar[blake2b_sigma[r][8]], m_scalar[blake2b_sigma[r][10]], 
            m_scalar[blake2b_sigma[r][12]], m_scalar[blake2b_sigma[r][14]]
        );
        m1 = _mm512_setr_epi64(
            m_scalar[blake2b_sigma[r][9]], m_scalar[blake2b_sigma[r][11]], 
            m_scalar[blake2b_sigma[r][13]], m_scalar[blake2b_sigma[r][15]],
            m_scalar[blake2b_sigma[r][9]], m_scalar[blake2b_sigma[r][11]], 
            m_scalar[blake2b_sigma[r][13]], m_scalar[blake2b_sigma[r][15]]
        );

        // G function
        row1 = _mm512_add_epi64(_mm512_add_epi64(row1, row2), m0);
        row4 = rotr32_512(_mm512_xor_si512(row4, row1));
        row3 = _mm512_add_epi64(row3, row4);
        row2 = rotr24_512(_mm512_xor_si512(row2, row3));
        row1 = _mm512_add_epi64(_mm512_add_epi64(row1, row2), m1);
        row4 = rotr16_512(_mm512_xor_si512(row4, row1));
        row3 = _mm512_add_epi64(row3, row4);
        row2 = rotr63_512(_mm512_xor_si512(row2, row3));

        // Undiagonal
        row2 = _mm512_permutexvar_epi64(undiag_perm, row2);
        row3 = _mm512_permutexvar_epi64(_mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5), row3);
        row4 = _mm512_permutexvar_epi64(diag_perm, row4);
    }

    // Finalize - extract lower 256 bits
    __m256i t0 = _mm256_xor_si256(
        _mm256_xor_si256(
            _mm512_extracti64x4_epi64(row1, 0), 
            _mm512_extracti64x4_epi64(row3, 0)
        ),
        _mm256_loadu_si256((const __m256i*)&S->h[0])
    );
    __m256i t1 = _mm256_xor_si256(
        _mm256_xor_si256(
            _mm512_extracti64x4_epi64(row2, 0), 
            _mm512_extracti64x4_epi64(row4, 0)
        ),
        _mm256_loadu_si256((const __m256i*)&S->h[4])
    );

    _mm256_storeu_si256((__m256i*)&S->h[0], t0);
    _mm256_storeu_si256((__m256i*)&S->h[4], t1);
  },
  TNN_TARGETS_X86_AVX512BW
)

#undef rotr32_512
#undef rotr24_512
#undef rotr16_512
#undef rotr63_512

#endif

__attribute__((target("default")))
#endif
static void blake2b_compress(blake2b_state *S, const uint8_t *block) {
	uint64_t m[16];
	uint64_t v[16];
	unsigned int i, r;

	for (i = 0; i < 16; ++i) {
		m[i] = load64(block + i * sizeof(m[i]));
	}

	for (i = 0; i < 8; ++i) {
		v[i] = S->h[i];
	}

	v[8] = blake2b_IV[0];
	v[9] = blake2b_IV[1];
	v[10] = blake2b_IV[2];
	v[11] = blake2b_IV[3];
	v[12] = blake2b_IV[4] ^ S->t[0];
	v[13] = blake2b_IV[5] ^ S->t[1];
	v[14] = blake2b_IV[6] ^ S->f[0];
	v[15] = blake2b_IV[7] ^ S->f[1];

#define G(r, i, a, b, c, d)                                                    \
    do {                                                                       \
        a = a + b + m[blake2b_sigma[r][2 * i + 0]];                            \
        d = rotr64(d ^ a, 32);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 24);                                                 \
        a = a + b + m[blake2b_sigma[r][2 * i + 1]];                            \
        d = rotr64(d ^ a, 16);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 63);                                                 \
    } while ((void)0, 0)

#define ROUND(r)                                                               \
    do {                                                                       \
        G(r, 0, v[0], v[4], v[8], v[12]);                                      \
        G(r, 1, v[1], v[5], v[9], v[13]);                                      \
        G(r, 2, v[2], v[6], v[10], v[14]);                                     \
        G(r, 3, v[3], v[7], v[11], v[15]);                                     \
        G(r, 4, v[0], v[5], v[10], v[15]);                                     \
        G(r, 5, v[1], v[6], v[11], v[12]);                                     \
        G(r, 6, v[2], v[7], v[8], v[13]);                                      \
        G(r, 7, v[3], v[4], v[9], v[14]);                                      \
    } while ((void)0, 0)

	for (r = 0; r < 12; ++r) {
		ROUND(r);
	}

	for (i = 0; i < 8; ++i) {
		S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
	}

#undef G
#undef ROUND
}

int blake2b_update(blake2b_state *S, const void *in, size_t inlen) {
	const uint8_t *pin = (const uint8_t *)in;

	if (inlen == 0) {
		return 0;
	}

	/* Sanity check */
	if (S == NULL || in == NULL) {
		return -1;
	}

	/* Is this a reused state? */
	if (S->f[0] != 0) {
		return -1;
	}

	if (S->buflen + inlen > BLAKE2B_BLOCKBYTES) {
		/* Complete current block */
		size_t left = S->buflen;
		size_t fill = BLAKE2B_BLOCKBYTES - left;
		memcpy(&S->buf[left], pin, fill);
		blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
		blake2b_compress(S, S->buf);
		S->buflen = 0;
		inlen -= fill;
		pin += fill;
		/* Avoid buffer copies when possible */
		while (inlen > BLAKE2B_BLOCKBYTES) {
			blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
			blake2b_compress(S, pin);
			inlen -= BLAKE2B_BLOCKBYTES;
			pin += BLAKE2B_BLOCKBYTES;
		}
	}
	memcpy(&S->buf[S->buflen], pin, inlen);
	S->buflen += (unsigned int)inlen;
	return 0;
}

int blake2b_final(blake2b_state *S, void *out, size_t outlen) {
	uint8_t buffer[BLAKE2B_OUTBYTES] = { 0 };
	unsigned int i;

	/* Sanity checks */
	if (S == NULL || out == NULL || outlen < S->outlen) {
		return -1;
	}

	/* Is this a reused state? */
	if (S->f[0] != 0) {
		return -1;
	}

	blake2b_increment_counter(S, S->buflen);
	blake2b_set_lastblock(S);
	memset(&S->buf[S->buflen], 0, BLAKE2B_BLOCKBYTES - S->buflen); /* Padding */
	blake2b_compress(S, S->buf);

	for (i = 0; i < 8; ++i) { /* Output full hash to temp buffer */
		store64(buffer + sizeof(S->h[i]) * i, S->h[i]);
	}

	memcpy(out, buffer, S->outlen);
	//clear_internal_memory(buffer, sizeof(buffer));
	//clear_internal_memory(S->buf, sizeof(S->buf));
	//clear_internal_memory(S->h, sizeof(S->h));
	return 0;
}

#ifdef __x86_64__ 

#ifndef TNN_LEGACY_AMD64
TNN_TARGET_CLONE(
  blake2b,
  int,
  (void *out, size_t outlen, const void *in, size_t inlen,
	const void *key, size_t keylen),
  {
    blake2b_state S;
    int ret = -1;

    /* Verify parameters */
    if (NULL == in && inlen > 0) {
      goto fail;
    }

    if (NULL == out || outlen == 0 || outlen > BLAKE2B_OUTBYTES) {
      goto fail;
    }

    if ((NULL == key && keylen > 0) || keylen > BLAKE2B_KEYBYTES) {
      goto fail;
    }

    if (keylen > 0) {
      if (blake2b_init_key(&S, outlen, key, keylen) < 0) {
        goto fail;
      }
    }
    else {
      if (blake2b_init(&S, outlen) < 0) {
        goto fail;
      }
    }

    if (blake2b_update(&S, in, inlen) < 0) {
      goto fail;
    }
    ret = blake2b_final(&S, out, outlen);

  fail:
    //clear_internal_memory(&S, sizeof(S));
    return ret;
  },
  TNN_TARGETS_X86_AVX2, TNN_TARGETS_X86_AVX512
)
#endif

__attribute__((target("default")))
#endif
int blake2b(void *out, size_t outlen, const void *in, size_t inlen,
	const void *key, size_t keylen) {
  blake2b_state S;
  int ret = -1;

  /* Verify parameters */
  if (NULL == in && inlen > 0) {
    goto fail;
  }

  if (NULL == out || outlen == 0 || outlen > BLAKE2B_OUTBYTES) {
    goto fail;
  }

  if ((NULL == key && keylen > 0) || keylen > BLAKE2B_KEYBYTES) {
    goto fail;
  }

  if (keylen > 0) {
    if (blake2b_init_key(&S, outlen, key, keylen) < 0) {
      goto fail;
    }
  }
  else {
    if (blake2b_init(&S, outlen) < 0) {
      goto fail;
    }
  }

  if (blake2b_update(&S, in, inlen) < 0) {
    goto fail;
  }
  ret = blake2b_final(&S, out, outlen);

fail:
  //clear_internal_memory(&S, sizeof(S));
  return ret;
}

/* Argon2 Team - Begin Code */
int blake2b_long(void *pout, size_t outlen, const void *in, size_t inlen) {
	uint8_t *out = (uint8_t *)pout;
	blake2b_state blake_state;
	uint8_t outlen_bytes[sizeof(uint32_t)] = { 0 };
	int ret = -1;

	if (outlen > UINT32_MAX) {
		goto fail;
	}

	/* Ensure little-endian byte order! */
	store32(outlen_bytes, (uint32_t)outlen);

#define TRY(statement)                                                         \
    do {                                                                       \
        ret = statement;                                                       \
        if (ret < 0) {                                                         \
            goto fail;                                                         \
        }                                                                      \
    } while ((void)0, 0)

	if (outlen <= BLAKE2B_OUTBYTES) {
		TRY(blake2b_init(&blake_state, outlen));
		TRY(blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
		TRY(blake2b_update(&blake_state, in, inlen));
		TRY(blake2b_final(&blake_state, out, outlen));
	}
	else {
		uint32_t toproduce;
		uint8_t out_buffer[BLAKE2B_OUTBYTES];
		uint8_t in_buffer[BLAKE2B_OUTBYTES];
		TRY(blake2b_init(&blake_state, BLAKE2B_OUTBYTES));
		TRY(blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
		TRY(blake2b_update(&blake_state, in, inlen));
		TRY(blake2b_final(&blake_state, out_buffer, BLAKE2B_OUTBYTES));
		memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
		out += BLAKE2B_OUTBYTES / 2;
		toproduce = (uint32_t)outlen - BLAKE2B_OUTBYTES / 2;

		while (toproduce > BLAKE2B_OUTBYTES) {
			memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
			TRY(blake2b(out_buffer, BLAKE2B_OUTBYTES, in_buffer,
				BLAKE2B_OUTBYTES, NULL, 0));
			memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
			out += BLAKE2B_OUTBYTES / 2;
			toproduce -= BLAKE2B_OUTBYTES / 2;
		}

		memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
		TRY(blake2b(out_buffer, toproduce, in_buffer, BLAKE2B_OUTBYTES, NULL,
			0));
		memcpy(out, out_buffer, toproduce);
	}
fail:
	//clear_internal_memory(&blake_state, sizeof(blake_state));
	return ret;
#undef TRY
}
/* Argon2 Team - End Code */

