#include "chacha20.h"

#if defined(__x86_64__)

#include <immintrin.h>
#include <inttypes.h>
#include <memory.h>
#include <simd/simd_utils.h>
 
__attribute__((target("sse2")))
static inline void PartialXor(const __m128i val, uint8_t* Src, uint8_t* Dest, uint64_t Size)
{
	alignas(16) uint8_t BuffForPartialOp[16];
	memcpy(BuffForPartialOp, Src, Size);
	_mm_store_si128((__m128i*)(BuffForPartialOp), _mm_xor_si128(val, _mm_loadu_si128((const __m128i*)BuffForPartialOp)));
	memcpy(Dest, BuffForPartialOp, Size);
}

__attribute__((target("sse2")))
static inline void PartialStore(const __m128i val, uint8_t* Dest, uint64_t Size)
{
	alignas(16) uint8_t BuffForPartialOp[16];
	_mm_store_si128((__m128i*)(BuffForPartialOp), val);
	memcpy(Dest, BuffForPartialOp, Size);
}

__attribute__((target("sse2")))
static inline __m128i RotateLeft7(const __m128i val)
{
	return _mm_or_si128(_mm_slli_epi32(val, 7), _mm_srli_epi32(val, 32 - 7));
}

__attribute__((target("sse2")))
static inline __m128i RotateLeft8(const __m128i val)
{
	return _mm_or_si128(_mm_slli_epi32(val, 8), _mm_srli_epi32(val, 32 - 8));
}

__attribute__((target("sse2")))
static inline __m128i RotateLeft12(const __m128i val)
{
	return _mm_or_si128(_mm_slli_epi32(val, 12), _mm_srli_epi32(val, 32 - 12));
}

__attribute__((target("sse2")))
static inline __m128i RotateLeft16(const __m128i val)
{
	return _mm_or_si128(_mm_slli_epi32(val, 16), _mm_srli_epi32(val, 32 - 16));
}

__attribute__((target("sse2")))
void ChaCha20EncryptBytes(uint8_t* state, uint8_t* In, uint8_t* Out, uint64_t Size, int rounds)
{
	uint8_t* CurrentIn = In;
	uint8_t* CurrentOut = Out;

	uint64_t FullBlocksCount = Size / 256;
	uint64_t RemainingBytes = Size % 256;

	const __m128i state0 = _mm_set_epi32(1797285236, 2036477234, 857760878, 1634760805); //"expand 32-byte k"
	const __m128i state1 = _mm_loadu_si128((const __m128i*)(state));
	const __m128i state2 = _mm_loadu_si128((const __m128i*)((state)+16));

	for (int64_t n = 0; n < FullBlocksCount; n++)
	{

		const __m128i state3 = _mm_loadu_si128((const __m128i*)((state)+32));

		__m128i r0_0 = state0;
		__m128i r0_1 = state1;
		__m128i r0_2 = state2;
		__m128i r0_3 = state3;

		__m128i r1_0 = state0;
		__m128i r1_1 = state1;
		__m128i r1_2 = state2;
		__m128i r1_3 = _mm_add_epi64(r0_3, _mm_set_epi32(0, 0, 0, 1));

		__m128i r2_0 = state0;
		__m128i r2_1 = state1;
		__m128i r2_2 = state2;
		__m128i r2_3 = _mm_add_epi64(r0_3, _mm_set_epi32(0, 0, 0, 2));

		__m128i r3_0 = state0;
		__m128i r3_1 = state1;
		__m128i r3_2 = state2;
		__m128i r3_3 = _mm_add_epi64(r0_3, _mm_set_epi32(0, 0, 0, 3));

		for (int i = rounds; i > 0; i -= 2)
		{
			r0_0 = _mm_add_epi32(r0_0, r0_1);
			r1_0 = _mm_add_epi32(r1_0, r1_1);
			r2_0 = _mm_add_epi32(r2_0, r2_1);
			r3_0 = _mm_add_epi32(r3_0, r3_1);

			r0_3 = _mm_xor_si128(r0_3, r0_0);
			r1_3 = _mm_xor_si128(r1_3, r1_0);
			r2_3 = _mm_xor_si128(r2_3, r2_0);
			r3_3 = _mm_xor_si128(r3_3, r3_0);

			r0_3 = RotateLeft16(r0_3);
			r1_3 = RotateLeft16(r1_3);
			r2_3 = RotateLeft16(r2_3);
			r3_3 = RotateLeft16(r3_3);

			r0_2 = _mm_add_epi32(r0_2, r0_3);
			r1_2 = _mm_add_epi32(r1_2, r1_3);
			r2_2 = _mm_add_epi32(r2_2, r2_3);
			r3_2 = _mm_add_epi32(r3_2, r3_3);

			r0_1 = _mm_xor_si128(r0_1, r0_2);
			r1_1 = _mm_xor_si128(r1_1, r1_2);
			r2_1 = _mm_xor_si128(r2_1, r2_2);
			r3_1 = _mm_xor_si128(r3_1, r3_2);

			r0_1 = RotateLeft12(r0_1);
			r1_1 = RotateLeft12(r1_1);
			r2_1 = RotateLeft12(r2_1);
			r3_1 = RotateLeft12(r3_1);

			r0_0 = _mm_add_epi32(r0_0, r0_1);
			r1_0 = _mm_add_epi32(r1_0, r1_1);
			r2_0 = _mm_add_epi32(r2_0, r2_1);
			r3_0 = _mm_add_epi32(r3_0, r3_1);

			r0_3 = _mm_xor_si128(r0_3, r0_0);
			r1_3 = _mm_xor_si128(r1_3, r1_0);
			r2_3 = _mm_xor_si128(r2_3, r2_0);
			r3_3 = _mm_xor_si128(r3_3, r3_0);

			r0_3 = RotateLeft8(r0_3);
			r1_3 = RotateLeft8(r1_3);
			r2_3 = RotateLeft8(r2_3);
			r3_3 = RotateLeft8(r3_3);

			r0_2 = _mm_add_epi32(r0_2, r0_3);
			r1_2 = _mm_add_epi32(r1_2, r1_3);
			r2_2 = _mm_add_epi32(r2_2, r2_3);
			r3_2 = _mm_add_epi32(r3_2, r3_3);

			r0_1 = _mm_xor_si128(r0_1, r0_2);
			r1_1 = _mm_xor_si128(r1_1, r1_2);
			r2_1 = _mm_xor_si128(r2_1, r2_2);
			r3_1 = _mm_xor_si128(r3_1, r3_2);

			r0_1 = RotateLeft7(r0_1);
			r1_1 = RotateLeft7(r1_1);
			r2_1 = RotateLeft7(r2_1);
			r3_1 = RotateLeft7(r3_1);

			r0_1 = _mm_shuffle_epi32(r0_1, _MM_SHUFFLE(0, 3, 2, 1));
			r0_2 = _mm_shuffle_epi32(r0_2, _MM_SHUFFLE(1, 0, 3, 2));
			r0_3 = _mm_shuffle_epi32(r0_3, _MM_SHUFFLE(2, 1, 0, 3));

			r1_1 = _mm_shuffle_epi32(r1_1, _MM_SHUFFLE(0, 3, 2, 1));
			r1_2 = _mm_shuffle_epi32(r1_2, _MM_SHUFFLE(1, 0, 3, 2));
			r1_3 = _mm_shuffle_epi32(r1_3, _MM_SHUFFLE(2, 1, 0, 3));

			r2_1 = _mm_shuffle_epi32(r2_1, _MM_SHUFFLE(0, 3, 2, 1));
			r2_2 = _mm_shuffle_epi32(r2_2, _MM_SHUFFLE(1, 0, 3, 2));
			r2_3 = _mm_shuffle_epi32(r2_3, _MM_SHUFFLE(2, 1, 0, 3));

			r3_1 = _mm_shuffle_epi32(r3_1, _MM_SHUFFLE(0, 3, 2, 1));
			r3_2 = _mm_shuffle_epi32(r3_2, _MM_SHUFFLE(1, 0, 3, 2));
			r3_3 = _mm_shuffle_epi32(r3_3, _MM_SHUFFLE(2, 1, 0, 3));

			r0_0 = _mm_add_epi32(r0_0, r0_1);
			r1_0 = _mm_add_epi32(r1_0, r1_1);
			r2_0 = _mm_add_epi32(r2_0, r2_1);
			r3_0 = _mm_add_epi32(r3_0, r3_1);

			r0_3 = _mm_xor_si128(r0_3, r0_0);
			r1_3 = _mm_xor_si128(r1_3, r1_0);
			r2_3 = _mm_xor_si128(r2_3, r2_0);
			r3_3 = _mm_xor_si128(r3_3, r3_0);

			r0_3 = RotateLeft16(r0_3);
			r1_3 = RotateLeft16(r1_3);
			r2_3 = RotateLeft16(r2_3);
			r3_3 = RotateLeft16(r3_3);

			r0_2 = _mm_add_epi32(r0_2, r0_3);
			r1_2 = _mm_add_epi32(r1_2, r1_3);
			r2_2 = _mm_add_epi32(r2_2, r2_3);
			r3_2 = _mm_add_epi32(r3_2, r3_3);

			r0_1 = _mm_xor_si128(r0_1, r0_2);
			r1_1 = _mm_xor_si128(r1_1, r1_2);
			r2_1 = _mm_xor_si128(r2_1, r2_2);
			r3_1 = _mm_xor_si128(r3_1, r3_2);

			r0_1 = RotateLeft12(r0_1);
			r1_1 = RotateLeft12(r1_1);
			r2_1 = RotateLeft12(r2_1);
			r3_1 = RotateLeft12(r3_1);

			r0_0 = _mm_add_epi32(r0_0, r0_1);
			r1_0 = _mm_add_epi32(r1_0, r1_1);
			r2_0 = _mm_add_epi32(r2_0, r2_1);
			r3_0 = _mm_add_epi32(r3_0, r3_1);

			r0_3 = _mm_xor_si128(r0_3, r0_0);
			r1_3 = _mm_xor_si128(r1_3, r1_0);
			r2_3 = _mm_xor_si128(r2_3, r2_0);
			r3_3 = _mm_xor_si128(r3_3, r3_0);

			r0_3 = RotateLeft8(r0_3);
			r1_3 = RotateLeft8(r1_3);
			r2_3 = RotateLeft8(r2_3);
			r3_3 = RotateLeft8(r3_3);

			r0_2 = _mm_add_epi32(r0_2, r0_3);
			r1_2 = _mm_add_epi32(r1_2, r1_3);
			r2_2 = _mm_add_epi32(r2_2, r2_3);
			r3_2 = _mm_add_epi32(r3_2, r3_3);

			r0_1 = _mm_xor_si128(r0_1, r0_2);
			r1_1 = _mm_xor_si128(r1_1, r1_2);
			r2_1 = _mm_xor_si128(r2_1, r2_2);
			r3_1 = _mm_xor_si128(r3_1, r3_2);

			r0_1 = RotateLeft7(r0_1);
			r1_1 = RotateLeft7(r1_1);
			r2_1 = RotateLeft7(r2_1);
			r3_1 = RotateLeft7(r3_1);

			r0_1 = _mm_shuffle_epi32(r0_1, _MM_SHUFFLE(2, 1, 0, 3));
			r0_2 = _mm_shuffle_epi32(r0_2, _MM_SHUFFLE(1, 0, 3, 2));
			r0_3 = _mm_shuffle_epi32(r0_3, _MM_SHUFFLE(0, 3, 2, 1));

			r1_1 = _mm_shuffle_epi32(r1_1, _MM_SHUFFLE(2, 1, 0, 3));
			r1_2 = _mm_shuffle_epi32(r1_2, _MM_SHUFFLE(1, 0, 3, 2));
			r1_3 = _mm_shuffle_epi32(r1_3, _MM_SHUFFLE(0, 3, 2, 1));

			r2_1 = _mm_shuffle_epi32(r2_1, _MM_SHUFFLE(2, 1, 0, 3));
			r2_2 = _mm_shuffle_epi32(r2_2, _MM_SHUFFLE(1, 0, 3, 2));
			r2_3 = _mm_shuffle_epi32(r2_3, _MM_SHUFFLE(0, 3, 2, 1));

			r3_1 = _mm_shuffle_epi32(r3_1, _MM_SHUFFLE(2, 1, 0, 3));
			r3_2 = _mm_shuffle_epi32(r3_2, _MM_SHUFFLE(1, 0, 3, 2));
			r3_3 = _mm_shuffle_epi32(r3_3, _MM_SHUFFLE(0, 3, 2, 1));
		}

		r0_0 = _mm_add_epi32(r0_0, state0);
		r0_1 = _mm_add_epi32(r0_1, state1);
		r0_2 = _mm_add_epi32(r0_2, state2);
		r0_3 = _mm_add_epi32(r0_3, state3);

		r1_0 = _mm_add_epi32(r1_0, state0);
		r1_1 = _mm_add_epi32(r1_1, state1);
		r1_2 = _mm_add_epi32(r1_2, state2);
		r1_3 = _mm_add_epi32(r1_3, state3);
		r1_3 = _mm_add_epi64(r1_3, _mm_set_epi32(0, 0, 0, 1));

		r2_0 = _mm_add_epi32(r2_0, state0);
		r2_1 = _mm_add_epi32(r2_1, state1);
		r2_2 = _mm_add_epi32(r2_2, state2);
		r2_3 = _mm_add_epi32(r2_3, state3);
		r2_3 = _mm_add_epi64(r2_3, _mm_set_epi32(0, 0, 0, 2));

		r3_0 = _mm_add_epi32(r3_0, state0);
		r3_1 = _mm_add_epi32(r3_1, state1);
		r3_2 = _mm_add_epi32(r3_2, state2);
		r3_3 = _mm_add_epi32(r3_3, state3);
		r3_3 = _mm_add_epi64(r3_3, _mm_set_epi32(0, 0, 0, 3));


		if (In)
		{
			_mm_storeu_si128((__m128i*)(CurrentOut + 0 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 0 * 16)), r0_0));
			_mm_storeu_si128((__m128i*)(CurrentOut + 1 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 1 * 16)), r0_1));
			_mm_storeu_si128((__m128i*)(CurrentOut + 2 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 2 * 16)), r0_2));
			_mm_storeu_si128((__m128i*)(CurrentOut + 3 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 3 * 16)), r0_3));

			_mm_storeu_si128((__m128i*)(CurrentOut + 4 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 4 * 16)), r1_0));
			_mm_storeu_si128((__m128i*)(CurrentOut + 5 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 5 * 16)), r1_1));
			_mm_storeu_si128((__m128i*)(CurrentOut + 6 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 6 * 16)), r1_2));
			_mm_storeu_si128((__m128i*)(CurrentOut + 7 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 7 * 16)), r1_3));

			_mm_storeu_si128((__m128i*)(CurrentOut + 8 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 8 * 16)), r2_0));
			_mm_storeu_si128((__m128i*)(CurrentOut + 9 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 9 * 16)), r2_1));
			_mm_storeu_si128((__m128i*)(CurrentOut + 10 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 10 * 16)), r2_2));
			_mm_storeu_si128((__m128i*)(CurrentOut + 11 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 11 * 16)), r2_3));

			_mm_storeu_si128((__m128i*)(CurrentOut + 12 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 12 * 16)), r3_0));
			_mm_storeu_si128((__m128i*)(CurrentOut + 13 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 13 * 16)), r3_1));
			_mm_storeu_si128((__m128i*)(CurrentOut + 14 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 14 * 16)), r3_2));
			_mm_storeu_si128((__m128i*)(CurrentOut + 15 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 15 * 16)), r3_3));
			CurrentIn += 256;
		}
		else
		{
			_mm_storeu_si128((__m128i*)(CurrentOut + 0 * 16), r0_0);
			_mm_storeu_si128((__m128i*)(CurrentOut + 1 * 16), r0_1);
			_mm_storeu_si128((__m128i*)(CurrentOut + 2 * 16), r0_2);
			_mm_storeu_si128((__m128i*)(CurrentOut + 3 * 16), r0_3);

			_mm_storeu_si128((__m128i*)(CurrentOut + 4 * 16), r1_0);
			_mm_storeu_si128((__m128i*)(CurrentOut + 5 * 16), r1_1);
			_mm_storeu_si128((__m128i*)(CurrentOut + 6 * 16), r1_2);
			_mm_storeu_si128((__m128i*)(CurrentOut + 7 * 16), r1_3);

			_mm_storeu_si128((__m128i*)(CurrentOut + 8 * 16), r2_0);
			_mm_storeu_si128((__m128i*)(CurrentOut + 9 * 16), r2_1);
			_mm_storeu_si128((__m128i*)(CurrentOut + 10 * 16), r2_2);
			_mm_storeu_si128((__m128i*)(CurrentOut + 11 * 16), r2_3);

			_mm_storeu_si128((__m128i*)(CurrentOut + 12 * 16), r3_0);
			_mm_storeu_si128((__m128i*)(CurrentOut + 13 * 16), r3_1);
			_mm_storeu_si128((__m128i*)(CurrentOut + 14 * 16), r3_2);
			_mm_storeu_si128((__m128i*)(CurrentOut + 15 * 16), r3_3);
		}

		CurrentOut += 256;
		
		ChaCha20AddCounter(state, 4);

	}

	if (RemainingBytes == 0) return;


	while(1)
	{
		const __m128i state3 = _mm_loadu_si128((const __m128i*)((state)+32));

		__m128i r0_0 = state0;
		__m128i r0_1 = state1;
		__m128i r0_2 = state2;
		__m128i r0_3 = state3;

		

		for (int i = rounds; i > 0; i -= 2)
		{
			r0_0 = _mm_add_epi32(r0_0, r0_1);

			r0_3 = _mm_xor_si128(r0_3, r0_0);

			r0_3 = RotateLeft16(r0_3);

			r0_2 = _mm_add_epi32(r0_2, r0_3);

			r0_1 = _mm_xor_si128(r0_1, r0_2);

			r0_1 = RotateLeft12(r0_1);

			r0_0 = _mm_add_epi32(r0_0, r0_1);

			r0_3 = _mm_xor_si128(r0_3, r0_0);

			r0_3 = RotateLeft8(r0_3);

			r0_2 = _mm_add_epi32(r0_2, r0_3);

			r0_1 = _mm_xor_si128(r0_1, r0_2);

			r0_1 = RotateLeft7(r0_1);

			r0_1 = _mm_shuffle_epi32(r0_1, _MM_SHUFFLE(0, 3, 2, 1));
			r0_2 = _mm_shuffle_epi32(r0_2, _MM_SHUFFLE(1, 0, 3, 2));
			r0_3 = _mm_shuffle_epi32(r0_3, _MM_SHUFFLE(2, 1, 0, 3));


			r0_0 = _mm_add_epi32(r0_0, r0_1);

			r0_3 = _mm_xor_si128(r0_3, r0_0);

			r0_3 = RotateLeft16(r0_3);

			r0_2 = _mm_add_epi32(r0_2, r0_3);

			r0_1 = _mm_xor_si128(r0_1, r0_2);

			r0_1 = RotateLeft12(r0_1);

			r0_0 = _mm_add_epi32(r0_0, r0_1);

			r0_3 = _mm_xor_si128(r0_3, r0_0);

			r0_3 = RotateLeft8(r0_3);

			r0_2 = _mm_add_epi32(r0_2, r0_3);

			r0_1 = _mm_xor_si128(r0_1, r0_2);

			r0_1 = RotateLeft7(r0_1);

			r0_1 = _mm_shuffle_epi32(r0_1, _MM_SHUFFLE(2, 1, 0, 3));
			r0_2 = _mm_shuffle_epi32(r0_2, _MM_SHUFFLE(1, 0, 3, 2));
			r0_3 = _mm_shuffle_epi32(r0_3, _MM_SHUFFLE(0, 3, 2, 1));

		}

		r0_0 = _mm_add_epi32(r0_0, state0);
		r0_1 = _mm_add_epi32(r0_1, state1);
		r0_2 = _mm_add_epi32(r0_2, state2);
		r0_3 = _mm_add_epi32(r0_3, state3);

		if (RemainingBytes>=64)
		{

			if (In)
			{
				_mm_storeu_si128((__m128i*)(CurrentOut + 0 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 0 * 16)), r0_0));
				_mm_storeu_si128((__m128i*)(CurrentOut + 1 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 1 * 16)), r0_1));
				_mm_storeu_si128((__m128i*)(CurrentOut + 2 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 2 * 16)), r0_2));
				_mm_storeu_si128((__m128i*)(CurrentOut + 3 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 3 * 16)), r0_3));
				CurrentIn += 64;
			}
			else
			{
				_mm_storeu_si128((__m128i*)(CurrentOut + 0 * 16), r0_0);
				_mm_storeu_si128((__m128i*)(CurrentOut + 1 * 16), r0_1);
				_mm_storeu_si128((__m128i*)(CurrentOut + 2 * 16), r0_2);
				_mm_storeu_si128((__m128i*)(CurrentOut + 3 * 16), r0_3);

			}
			CurrentOut += 64;
			ChaCha20AddCounter(state, 1);
			RemainingBytes -= 64;
			if (RemainingBytes == 0) return;
			continue;
		}
		else
		{
			alignas(16) uint8_t TmpBuf[64];
			if (In)
			{
				memcpy(TmpBuf, CurrentIn, RemainingBytes);
				_mm_store_si128((__m128i*)(TmpBuf + 0 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(TmpBuf + 0 * 16)), r0_0));
				_mm_store_si128((__m128i*)(TmpBuf + 1 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(TmpBuf + 1 * 16)), r0_1));
				_mm_store_si128((__m128i*)(TmpBuf + 2 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(TmpBuf + 2 * 16)), r0_2));
				_mm_store_si128((__m128i*)(TmpBuf + 3 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(TmpBuf + 3 * 16)), r0_3));
			}
			else
			{
				_mm_store_si128((__m128i*)(TmpBuf + 0 * 16), r0_0);
				_mm_store_si128((__m128i*)(TmpBuf + 1 * 16), r0_1);
				_mm_store_si128((__m128i*)(TmpBuf + 2 * 16), r0_2);
				_mm_store_si128((__m128i*)(TmpBuf + 3 * 16), r0_3);
			}
			memcpy(CurrentOut, TmpBuf, RemainingBytes);
			ChaCha20AddCounter(state, 1);
			return;
		}
	}
}

// 4-way, no input to XOR with
template<StorePolicy POLICY>
__attribute__((target("ssse3")))
static void ChaCha20EncryptXelis_ssse3_impl(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    size_t bytes_per_stream,
    int rounds)
{
    // Constants
    const __m128i const0 = _mm_set1_epi32(0x61707865);
    const __m128i const1 = _mm_set1_epi32(0x3320646e);
    const __m128i const2 = _mm_set1_epi32(0x79622d32);
    const __m128i const3 = _mm_set1_epi32(0x6b206574);
    
    // Load and transpose keys
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
    
    // Load and transpose nonces
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
        
        // ChaCha rounds
        for (int i = rounds; i > 0; i -= 2)
        {
            #define QR_SSSE3(a, b, c, d) \
                a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a); \
                d = RotateLeft16(d); \
                c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c); \
                b = RotateLeft12(b); \
                a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a); \
                d = RotateLeft8(d); \
                c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c); \
                b = RotateLeft7(b);
            
            // Column round
            QR_SSSE3(x0, x4, x8,  x12);
            QR_SSSE3(x1, x5, x9,  x13);
            QR_SSSE3(x2, x6, x10, x14);
            QR_SSSE3(x3, x7, x11, x15);
            
            // Diagonal round
            QR_SSSE3(x0, x5, x10, x15);
            QR_SSSE3(x1, x6, x11, x12);
            QR_SSSE3(x2, x7, x8,  x13);
            QR_SSSE3(x3, x4, x9,  x14);
            
            #undef QR_SSSE3
        }
        
        // Add initial state
        x0 = _mm_add_epi32(x0, s0);   x1 = _mm_add_epi32(x1, s1);
        x2 = _mm_add_epi32(x2, s2);   x3 = _mm_add_epi32(x3, s3);
        x4 = _mm_add_epi32(x4, s4);   x5 = _mm_add_epi32(x5, s5);
        x6 = _mm_add_epi32(x6, s6);   x7 = _mm_add_epi32(x7, s7);
        x8 = _mm_add_epi32(x8, s8);   x9 = _mm_add_epi32(x9, s9);
        x10 = _mm_add_epi32(x10, s10); x11 = _mm_add_epi32(x11, s11);
        x12 = _mm_add_epi32(x12, s12); x13 = _mm_add_epi32(x13, s13);
        x14 = _mm_add_epi32(x14, s14); x15 = _mm_add_epi32(x15, s15);
        
        // Transpose and store (using policy-controlled stores)
        #define TRANSPOSE_STORE_SSSE3(xa, xb, xc, xd, off) \
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
        
        TRANSPOSE_STORE_SSSE3(x0,  x1,  x2,  x3,  0);
        TRANSPOSE_STORE_SSSE3(x4,  x5,  x6,  x7,  16);
        TRANSPOSE_STORE_SSSE3(x8,  x9,  x10, x11, 32);
        TRANSPOSE_STORE_SSSE3(x12, x13, x14, x15, 48);
        
        #undef TRANSPOSE_STORE_SSSE3
        
        pos += 64;
        counter = _mm_add_epi32(counter, _mm_set1_epi32(1));
    }
    
    simd_fence<POLICY>();
}

__attribute__((target("ssse3")))
void ChaCha20EncryptXelis(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    size_t bytes_per_stream,
    int rounds)
{
    ChaCha20EncryptXelis_ssse3_impl<StorePolicy::TEMPORAL>(
        keys, nonces, outputs, bytes_per_stream, rounds);
}

__attribute__((target("ssse3")))
void ChaCha20EncryptXelisNT(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    size_t bytes_per_stream,
    int rounds)
{
    ChaCha20EncryptXelis_ssse3_impl<StorePolicy::NON_TEMPORAL>(
        keys, nonces, outputs, bytes_per_stream, rounds);
}

#endif
