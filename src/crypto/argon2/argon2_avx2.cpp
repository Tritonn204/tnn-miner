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
#include <stdlib.h>

#include "argon2.h"

void randomx_argon2_fill_segment_avx2(const argon2_instance_t* instance,
	argon2_position_t position);

randomx_argon2_impl* randomx_argon2_impl_avx2() {
#if defined(__AVX2__)
	return &randomx_argon2_fill_segment_avx2;
#endif
	return NULL;
}

#if defined(__AVX2__)

#include "argon2_core.h"

#include "blake2/blamka-round-avx2.h"
#include "blake2/blake2-impl.h"
#include "blake2/blake2.h"

static void fill_block(__m256i* state, const block* ref_block,
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

void randomx_argon2_fill_segment_avx2(const argon2_instance_t* instance,
	argon2_position_t position) {
	block* ref_block = NULL, * curr_block = NULL;
	block address_block, input_block;
	uint64_t pseudo_rand, ref_index, ref_lane;
	uint32_t prev_offset, curr_offset;
	uint32_t starting_index, i;
	__m256i state[ARGON2_HWORDS_IN_BLOCK];

	if (instance == NULL) {
		return;
	}

	starting_index = 0;

	if ((0 == position.pass) && (0 == position.slice)) {
		starting_index = 2; /* we have already generated the first two blocks */
	}

	/* Offset of the current block */
	curr_offset = position.lane * instance->lane_length +
		position.slice * instance->segment_length + starting_index;

	if (0 == curr_offset % instance->lane_length) {
		/* Last block in this lane */
		prev_offset = curr_offset + instance->lane_length - 1;
	}
	else {
		/* Previous block */
		prev_offset = curr_offset - 1;
	}

	memcpy(state, ((instance->memory + prev_offset)->v), ARGON2_BLOCK_SIZE);

	for (i = starting_index; i < instance->segment_length;
		++i, ++curr_offset, ++prev_offset) {
		/*1.1 Rotating prev_offset if needed */
		if (curr_offset % instance->lane_length == 1) {
			prev_offset = curr_offset - 1;
		}

		/* 1.2 Computing the index of the reference block */
		/* 1.2.1 Taking pseudo-random value from the previous block */
		pseudo_rand = instance->memory[prev_offset].v[0];

		/* 1.2.2 Computing the lane of the reference block */
		ref_lane = ((pseudo_rand >> 32)) % instance->lanes;

		if ((position.pass == 0) && (position.slice == 0)) {
			/* Can not reference other lanes yet */
			ref_lane = position.lane;
		}

		/* 1.2.3 Computing the number of possible reference block within the
		 * lane.
		 */
		position.index = i;
		ref_index = randomx_argon2_index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF,
			ref_lane == position.lane);

		/* 2 Creating a new block */
		ref_block =
			instance->memory + instance->lane_length * ref_lane + ref_index;
		curr_block = instance->memory + curr_offset;
		if (ARGON2_VERSION_10 == instance->version) {
			/* version 1.2.1 and earlier: overwrite, not XOR */
			fill_block(state, ref_block, curr_block, 0);
		}
		else {
			if (0 == position.pass) {
				fill_block(state, ref_block, curr_block, 0);
			}
			else {
				fill_block(state, ref_block, curr_block, 1);
			}
		}
	}
}

void argon2_fill_segment_avx2(const argon2_instance_t* instance,
                              argon2_position_t position) {
    block* ref_block = NULL;
    block* curr_block = NULL;
    uint64_t pseudo_rand, ref_index, ref_lane;
    uint32_t prev_offset, curr_offset;
    uint32_t starting_index, i;
    __m256i state[ARGON2_HWORDS_IN_BLOCK];

    if (instance == NULL || instance->memory == NULL)
        return;

    starting_index = 0;
    if (position.pass == 0 && position.slice == 0)
        starting_index = 2; // First two blocks already generated

    curr_offset = position.lane * instance->lane_length +
                  position.slice * instance->segment_length + starting_index;

    prev_offset = (curr_offset % instance->lane_length == 0)
                      ? curr_offset + instance->lane_length - 1
                      : curr_offset - 1;

    // Load previous block into SIMD state
    memcpy(state, instance->memory[prev_offset].v, ARGON2_BLOCK_SIZE);

    for (i = starting_index; i < instance->segment_length; ++i, ++curr_offset) {
        if (curr_offset % instance->lane_length == 0)
            prev_offset = curr_offset + instance->lane_length - 1;
        else
            prev_offset = curr_offset - 1;

        pseudo_rand = instance->memory[prev_offset].v[0];
        ref_lane = (position.pass == 0 && position.slice == 0)
                       ? position.lane
                       : ((pseudo_rand >> 32) % instance->lanes);

        position.index = i;
        ref_index = randomx_argon2_index_alpha(instance, &position,
                                               (uint32_t)pseudo_rand,
                                               ref_lane == position.lane);

        ref_block = instance->memory + ref_lane * instance->lane_length + ref_index;
        curr_block = instance->memory + curr_offset;

        int with_xor = (position.pass != 0);
        fill_block(state, ref_block, curr_block, with_xor);
    }
}


#include <stdio.h>
#include <nmmintrin.h>

uint64_t block_checksum(const void* data, size_t size) {
    const uint64_t* ptr = (const uint64_t*)data;
    size_t count = size / sizeof(uint64_t);
    uint64_t crc = 0;
    for (size_t i = 0; i < count; ++i) {
        crc = _mm_crc32_u64(crc, ptr[i]);
    }
    return crc;
}

void argon2_finalize_avx2(const argon2_instance_t* instance, uint8_t* out, size_t outlen) {
    if (instance == NULL || out == NULL || outlen == 0 || outlen > ARGON2_BLOCK_SIZE) {
        return;
    }

    if (instance->lanes == 1) {
        const uint32_t last_block_offset = instance->lane_length - 1;
        const block* last_block = &(instance->memory[last_block_offset]);

        blake2b_long(out, 32, (uint8_t*)last_block->v, ARGON2_BLOCK_SIZE);
        return;
    }

    __m256i blockhash[ARGON2_HWORDS_IN_BLOCK];
    const uint32_t last_block_offset = instance->lane_length - 1;
    memcpy(blockhash, instance->memory[last_block_offset].v, ARGON2_BLOCK_SIZE);

    // 2. XOR in all other lane tail blocks
    for (uint32_t l = 1; l < instance->lanes; ++l) {
        const block* lane_block = instance->memory + l * instance->lane_length + last_block_offset;
        for (uint32_t i = 0; i < ARGON2_HWORDS_IN_BLOCK; ++i) {
            blockhash[i] = _mm256_xor_si256(blockhash[i], _mm256_load_si256((__m256i*)(lane_block->v + i * 4)));
        }
    }

    // TODO Blake2
}

#endif
