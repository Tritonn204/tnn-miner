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

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "argon2.h"

#include <immintrin.h>

#include "argon2_core.h"

#include "blake2/blamka-round-avx512.h"
#include "blake2/blake2-impl.h"
#include "blake2/blake2.h"

__attribute__((target("avx512f, avx512bw")))
static void fill_block(__m512i* state, const block* ref_block,
	block* next_block, int with_xor) {
	__m512i block_XY[ARGON2_512BIT_WORDS_IN_BLOCK];
	unsigned int i;

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

	BLAKE2_ROUND_1(state[0], state[1], state[2], state[3],
		state[4], state[5], state[6], state[7]);
	BLAKE2_ROUND_1(state[8], state[9], state[10], state[11],
		state[12], state[13], state[14], state[15]);

	BLAKE2_ROUND_2(state[0], state[2], state[4], state[6],
		state[8], state[10], state[12], state[14]);
	BLAKE2_ROUND_2(state[1], state[3], state[5], state[7],
		state[9], state[11], state[13], state[15]);

	for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
		state[i] = _mm512_xor_si512(state[i], block_XY[i]);
		_mm512_storeu_si512((__m512i*)next_block->v + i, state[i]);
	}
}

__attribute__((target("avx512f, avx512bw")))
void randomx_argon2_fill_segment_avx512(const argon2_instance_t* instance,
	argon2_position_t position) {
	block* ref_block = NULL, * curr_block = NULL;
	block address_block, input_block;
	uint64_t pseudo_rand, ref_index, ref_lane;
	uint32_t prev_offset, curr_offset;
	uint32_t starting_index, i;
	__m512i state[ARGON2_512BIT_WORDS_IN_BLOCK];

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

__attribute__((target("default")))
void randomx_argon2_fill_segment_avx512(const argon2_instance_t* instance,
	argon2_position_t position) {}

void avx512_dispatch(const argon2_instance_t* instance,
	argon2_position_t position) {
    randomx_argon2_fill_segment_avx512(instance, position);
}

randomx_argon2_impl* randomx_argon2_impl_avx512() {
	return &avx512_dispatch;
}