#include "chacha20.h"
#include <memory.h>



#if defined(USE_SSE) || defined (USE_AVX2) || defined(USE_AVX512)
#include <immintrin.h>

#endif

static const int32_t KeyDataSize = 48;
static const int32_t rounds = 20;



static const uint32_t ConstState[4] = { 1634760805, 857760878, 2036477234, 1797285236 };  //"expand 32-byte k";;













void ChaCha20SetKey(uint8_t* state, const uint8_t* Key)
{
	memcpy(state, Key, 32);
}

void ChaCha20SetNonce(uint8_t* state, const uint8_t* Nonce)
{
	memcpy(state + 36, Nonce, 12);
}

void ChaCha20SetCtr(uint8_t* state, const uint8_t* Ctr)
{
	memcpy(state + 32, Ctr, 4);
}


void ChaCha20IncrementNonce(uint8_t* state)
{
	uint32_t* State32bits = (uint32_t*)state;
	State32bits[8] = 0; //reset counter
	++State32bits[9];
	if (State32bits[9] == 0)
	{
		++State32bits[10];
		if (State32bits[10] == 0) ++State32bits[11];
	}
}




		//todo

	





/*	

#if defined(USE_SSE)
	//sse2 version
	while (1)
	{
		const __m128i state0 = _mm_loadu_si128((const __m128i*)(ConstState));
		const __m128i state1 = _mm_loadu_si128((const __m128i*)(state));
		const __m128i state2 = _mm_loadu_si128((const __m128i*)((state)+16));
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

		for (int i = 20; i > 0; i -= 2)
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

		if (RemainingSize >= CHACHA20_FULLBLOCKBYTES) //full block 4x64bytes
		{
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
			RemainingSize -= CHACHA20_FULLBLOCKBYTES;
			CurrentOut += CHACHA20_FULLBLOCKBYTES;
			if (In) CurrentIn += CHACHA20_FULLBLOCKBYTES;
			AddCounter(state, 4);
			if (RemainingSize == 0) return;
			continue;
		}
		else
		{ //partial block
			if (In) //encrypt
			{
				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r0_0, CurrentIn, CurrentOut, RemainingSize);
					AddCounter(state, 1);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 0 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 0 * 16)), r0_0));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 1);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r0_1, CurrentIn + 16, CurrentOut + 16, RemainingSize);
					AddCounter(state, 1);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 1 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 1 * 16)), r0_1));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 1);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r0_2, CurrentIn + 32, CurrentOut + 32, RemainingSize);
					AddCounter(state, 1);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 2 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 2 * 16)), r0_2));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 1);
					return;
				}


				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r0_3, CurrentIn + 48, CurrentOut + 48, RemainingSize);
					AddCounter(state, 1);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 3 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 3 * 16)), r0_3));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 1);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r1_0, CurrentIn + 64, CurrentOut + 64, RemainingSize);
					AddCounter(state, 2);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 4 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 4 * 16)), r1_0));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 2);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r1_1, CurrentIn + 80, CurrentOut + 80, RemainingSize);
					AddCounter(state, 2);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 5 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 5 * 16)), r1_1));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 2);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r1_2, CurrentIn + 96, CurrentOut + 96, RemainingSize);
					AddCounter(state, 2);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 6 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 6 * 16)), r1_2));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 2);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r1_3, CurrentIn + 112, CurrentOut + 112, RemainingSize);
					AddCounter(state, 2);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 7 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 7 * 16)), r1_3));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 2);
					return;
				}


				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r2_0, CurrentIn + 128, CurrentOut + 128, RemainingSize);
					AddCounter(state, 3);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 8 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 8 * 16)), r2_0));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 3);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r2_1, CurrentIn + 144, CurrentOut + 144, RemainingSize);
					AddCounter(state, 3);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 9 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 9 * 16)), r2_1));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 3);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r2_2, CurrentIn + 160, CurrentOut + 160, RemainingSize);
					AddCounter(state, 3);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 10 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 10 * 16)), r2_2));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 3);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r2_3, CurrentIn + 176, CurrentOut + 176, RemainingSize);
					AddCounter(state, 3);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 11 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 11 * 16)), r2_3));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 3);
					return;
				}


				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r3_0, CurrentIn + 192, CurrentOut + 192, RemainingSize);
					AddCounter(state, 4);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 12 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 12 * 16)), r3_0));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 4);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r3_1, CurrentIn + 208, CurrentOut + 208, RemainingSize);
					AddCounter(state, 4);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 13 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 13 * 16)), r3_1));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 4);
					return;
				}
				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialXor(r3_2, CurrentIn + 224, CurrentOut + 224, RemainingSize);
					AddCounter(state, 4);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 14 * 16), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(CurrentIn + 14 * 16)), r3_2));
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 4);
					return;
				}
				PartialXor(r3_3, CurrentIn + 240, CurrentOut + 240, RemainingSize);
				AddCounter(state, 4);
				return;

			}
			else //store
			{
				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r0_0, CurrentOut, RemainingSize);
					AddCounter(state, 1);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 0 * 16), r0_0);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 1);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r0_1, CurrentOut + 16, RemainingSize);
					AddCounter(state, 1);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 1 * 16), r0_1);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 1);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r0_2, CurrentOut + 32, RemainingSize);
					AddCounter(state, 1);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 2 * 16), r0_2);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 1);
					return;
				}


				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r0_3, CurrentOut + 48, RemainingSize);
					AddCounter(state, 1);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 3 * 16), r0_3);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 1);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r1_0, CurrentOut + 64, RemainingSize);
					AddCounter(state, 2);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 4 * 16), r1_0);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 2);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r1_1, CurrentOut + 80, RemainingSize);
					AddCounter(state, 2);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 5 * 16), r1_1);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 2);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r1_2, CurrentOut + 96, RemainingSize);
					AddCounter(state, 2);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 6 * 16), r1_2);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 2);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r1_3, CurrentOut + 112, RemainingSize);
					AddCounter(state, 2);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 7 * 16), r1_3);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 2);
					return;
				}


				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r2_0, CurrentOut + 128, RemainingSize);
					AddCounter(state, 3);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 8 * 16), r2_0);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 3);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r2_1, CurrentOut + 144, RemainingSize);
					AddCounter(state, 3);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 9 * 16), r2_1);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 3);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r2_2, CurrentOut + 160, RemainingSize);
					AddCounter(state, 3);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 10 * 16), r2_2);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 3);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r2_3, CurrentOut + 176, RemainingSize);
					AddCounter(state, 3);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 11 * 16), r2_3);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 3);
					return;
				}


				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r3_0, CurrentOut + 192, RemainingSize);
					AddCounter(state, 4);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 12 * 16), r3_0);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 4);
					return;
				}

				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r3_1, CurrentOut + 208, RemainingSize);
					AddCounter(state, 4);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 13 * 16), r3_1);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 4);
					return;
				}
				if (RemainingSize < CHACHA20_MINBLOCKBYTES)
				{
					PartialStore(r3_2, CurrentOut + 224, RemainingSize);
					AddCounter(state, 4);
					return;
				}
				_mm_storeu_si128((__m128i*)(CurrentOut + 14 * 16), r3_2);
				RemainingSize -= CHACHA20_MINBLOCKBYTES;
				if (RemainingSize == 0)
				{
					AddCounter(state, 4);
					return;
				}
				PartialStore(r3_3, CurrentOut + 240, RemainingSize);
				AddCounter(state, 4);
				return;
			}
		}
	}



*/

__attribute__((target("default")))
void ChaCha20EncryptBytes(uint8_t* state, uint8_t* In, uint8_t* Out, uint64_t Size, int rounds)
{
	//portable chacha, no simd
	uint8_t* CurrentIn = In;
	uint8_t* CurrentOut = Out;
	uint64_t RemainingBytes = Size;
	uint32_t* state_dwords = (uint32_t*)state;
	uint32_t b[16];
	while (1)
	{
		b[0] = ConstState[0];
		b[1] = ConstState[1];
		b[2] = ConstState[2];
		b[3] = ConstState[3];
		memcpy(((uint8_t*)b) + 16, state, 48);


		for (int i = 0; i < rounds; i+= 2)
		{
			b[0] = b[0] + b[4];
			b[12] = (b[12] ^ b[0]) << 16 | (b[12] ^ b[0]) >> 16;
			b[8] = b[8] + b[12]; b[4] = (b[4] ^ b[8]) << 12 | (b[4] ^ b[8]) >> 20;
			b[0] = b[0] + b[4];
			b[12] = (b[12] ^ b[0]) << 8 | (b[12] ^ b[0]) >> 24;
			b[8] = b[8] + b[12];
			b[4] = (b[4] ^ b[8]) << 7 | (b[4] ^ b[8]) >> 25;
			b[1] = b[1] + b[5];
			b[13] = (b[13] ^ b[1]) << 16 | (b[13] ^ b[1]) >> 16;
			b[9] = b[9] + b[13];
			b[5] = (b[5] ^ b[9]) << 12 | (b[5] ^ b[9]) >> 20;
			b[1] = b[1] + b[5];
			b[13] = (b[13] ^ b[1]) << 8 | (b[13] ^ b[1]) >> 24;
			b[9] = b[9] + b[13];
			b[5] = (b[5] ^ b[9]) << 7 | (b[5] ^ b[9]) >> 25;
			b[2] = b[2] + b[6];
			b[14] = (b[14] ^ b[2]) << 16 | (b[14] ^ b[2]) >> 16;
			b[10] = b[10] + b[14];
			b[6] = (b[6] ^ b[10]) << 12 | (b[6] ^ b[10]) >> 20;
			b[2] = b[2] + b[6];
			b[14] = (b[14] ^ b[2]) << 8 | (b[14] ^ b[2]) >> 24;
			b[10] = b[10] + b[14];
			b[6] = (b[6] ^ b[10]) << 7 | (b[6] ^ b[10]) >> 25;
			b[3] = b[3] + b[7];
			b[15] = (b[15] ^ b[3]) << 16 | (b[15] ^ b[3]) >> 16;
			b[11] = b[11] + b[15];
			b[7] = (b[7] ^ b[11]) << 12 | (b[7] ^ b[11]) >> 20;
			b[3] = b[3] + b[7];
			b[15] = (b[15] ^ b[3]) << 8 | (b[15] ^ b[3]) >> 24;
			b[11] = b[11] + b[15];
			b[7] = (b[7] ^ b[11]) << 7 | (b[7] ^ b[11]) >> 25;
			b[0] = b[0] + b[5];
			b[15] = (b[15] ^ b[0]) << 16 | (b[15] ^ b[0]) >> 16;
			b[10] = b[10] + b[15];
			b[5] = (b[5] ^ b[10]) << 12 | (b[5] ^ b[10]) >> 20;
			b[0] = b[0] + b[5];
			b[15] = (b[15] ^ b[0]) << 8 | (b[15] ^ b[0]) >> 24;
			b[10] = b[10] + b[15];
			b[5] = (b[5] ^ b[10]) << 7 | (b[5] ^ b[10]) >> 25;
			b[1] = b[1] + b[6];
			b[12] = (b[12] ^ b[1]) << 16 | (b[12] ^ b[1]) >> 16;
			b[11] = b[11] + b[12];
			b[6] = (b[6] ^ b[11]) << 12 | (b[6] ^ b[11]) >> 20;
			b[1] = b[1] + b[6];
			b[12] = (b[12] ^ b[1]) << 8 | (b[12] ^ b[1]) >> 24;
			b[11] = b[11] + b[12];
			b[6] = (b[6] ^ b[11]) << 7 | (b[6] ^ b[11]) >> 25;
			b[2] = b[2] + b[7];
			b[13] = (b[13] ^ b[2]) << 16 | (b[13] ^ b[2]) >> 16;
			b[8] = b[8] + b[13];
			b[7] = (b[7] ^ b[8]) << 12 | (b[7] ^ b[8]) >> 20;
			b[2] = b[2] + b[7];
			b[13] = (b[13] ^ b[2]) << 8 | (b[13] ^ b[2]) >> 24;
			b[8] = b[8] + b[13];
			b[7] = (b[7] ^ b[8]) << 7 | (b[7] ^ b[8]) >> 25;
			b[3] = b[3] + b[4];
			b[14] = (b[14] ^ b[3]) << 16 | (b[14] ^ b[3]) >> 16;
			b[9] = b[9] + b[14];
			b[4] = (b[4] ^ b[9]) << 12 | (b[4] ^ b[9]) >> 20;
			b[3] = b[3] + b[4];
			b[14] = (b[14] ^ b[3]) << 8 | (b[14] ^ b[3]) >> 24;
			b[9] = b[9] + b[14];
			b[4] = (b[4] ^ b[9]) << 7 | (b[4] ^ b[9]) >> 25;
		}

		for (uint32_t i = 0; i < 4; ++i)
		{
			b[i] += ConstState[i];
		}
		for (uint32_t i = 0; i < 12; ++i)
		{
			b[i + 4] += state_dwords[i];
		}

		++state_dwords[8]; //counter

		if (RemainingBytes >= 64)
		{
			if (In)
			{
				uint32_t* In32bits = (uint32_t*)CurrentIn;
				uint32_t* Out32bits = (uint32_t*)CurrentOut;
				for (uint32_t i = 0; i < 16; i++)
				{
					Out32bits[i] = In32bits[i] ^ b[i];
				}
			}
			else
				memcpy(CurrentOut, b, 64);

			if (In) CurrentIn += 64;
			CurrentOut += 64;
			RemainingBytes -= 64;
			if (RemainingBytes == 0) return;
			continue;
		}
		else
		{
			if (In)
			{
				for (int32_t i = 0; i < RemainingBytes; i++)
					CurrentOut[i] = CurrentIn[i] ^ ((uint8_t*)b)[i];
			}
			else memcpy(CurrentOut, b, RemainingBytes);
			return;
		}
	}
}
//}