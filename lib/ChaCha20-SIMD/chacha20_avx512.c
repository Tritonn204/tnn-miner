#include "chacha20.h"

#if defined(__x86_64__)

#include <immintrin.h>
#include <memory.h>

__attribute__((target("avx512f")))
static inline void PartialXor(const __m512i val, uint8_t* Src, uint8_t* Dest, uint64_t Size)
{
	_Alignas(64) uint8_t BuffForPartialOp[64];
	memcpy(BuffForPartialOp, Src, Size);
	_mm512_storeu_si512((__m512i*)(BuffForPartialOp), _mm512_xor_si512(val, _mm512_loadu_si512((const __m512i*)BuffForPartialOp)));
	memcpy(Dest, BuffForPartialOp, Size);
}

__attribute__((target("avx512f")))
static inline void PartialStore(const __m512i val, uint8_t* Dest, uint64_t Size)
{
	_Alignas(64) uint8_t BuffForPartialOp[64];
	_mm512_storeu_si512((__m512i*)(BuffForPartialOp), val);
	memcpy(Dest, BuffForPartialOp, Size);
}

__attribute__((target("avx512f")))
void ChaCha20EncryptBytes(uint8_t* state, uint8_t* In, uint8_t* Out, uint64_t Size, int rounds)
{

	uint8_t* CurrentIn = In;
	uint8_t* CurrentOut = Out;


	uint64_t FullBlocksCount = Size / 1024;
	uint64_t RemainingBytes = Size % 1024;





	const __m512i state0 = _mm512_broadcast_i32x4(_mm_set_epi32(1797285236, 2036477234, 857760878, 1634760805)); //"expand 32-byte k"
	const __m512i state1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)(state)));
	const __m512i state2 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)(state + 16)));


	//permutation indexes for results
	const __m512i P1 = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
	const __m512i P2 = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
	const __m512i P3 = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
	const __m512i P4 = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);

	__m512i T1, T2, T3, T4; //temporary registers
	__m512i ctr_increment;

	T1 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*)(state + 32)));
	T2 = _mm512_set_epi64(0, 3, 0, 2, 0, 1, 0, 0);

	__m512i state3_0 = _mm512_add_epi32(T1, T2);

	if (FullBlocksCount > 0)
	{
		T3 = _mm512_set_epi64(0, 7, 0, 6, 0, 5, 0, 4);





		__m512i state3_1 = _mm512_add_epi32(T1, T3);

		T2 = _mm512_set_epi64(0, 11, 0, 10, 0, 9, 0, 8);
		T3 = _mm512_set_epi64(0, 15, 0, 14, 0, 13, 0, 12);

		__m512i state3_2 = _mm512_add_epi32(T1, T2);
		__m512i state3_3 = _mm512_add_epi32(T1, T3);





		ctr_increment = _mm512_set_epi64(0, 16, 0, 16, 0, 16, 0, 16);



		for (uint64_t n = 0; n < FullBlocksCount; n++)
		{

			__m512i X0_0 = state0;
			__m512i X0_1 = state1;
			__m512i X0_2 = state2;
			__m512i X0_3 = state3_0;

			__m512i X1_0 = state0;
			__m512i X1_1 = state1;
			__m512i X1_2 = state2;
			__m512i X1_3 = state3_1;

			__m512i X2_0 = state0;
			__m512i X2_1 = state1;
			__m512i X2_2 = state2;
			__m512i X2_3 = state3_2;

			__m512i X3_0 = state0;
			__m512i X3_1 = state1;
			__m512i X3_2 = state2;
			__m512i X3_3 = state3_3;



			for (int i = rounds; i > 0; i -= 2)
			{
				X0_0 = _mm512_add_epi32(X0_0, X0_1);
				X1_0 = _mm512_add_epi32(X1_0, X1_1);
				X2_0 = _mm512_add_epi32(X2_0, X2_1);
				X3_0 = _mm512_add_epi32(X3_0, X3_1);


				X0_3 = _mm512_xor_si512(X0_3, X0_0);
				X1_3 = _mm512_xor_si512(X1_3, X1_0);
				X2_3 = _mm512_xor_si512(X2_3, X2_0);
				X2_3 = _mm512_xor_si512(X3_3, X3_0);

				X0_3 = _mm512_rol_epi32(X0_3, 16);
				X1_3 = _mm512_rol_epi32(X1_3, 16);
				X2_3 = _mm512_rol_epi32(X2_3, 16);
				X3_3 = _mm512_rol_epi32(X3_3, 16);

				X0_2 = _mm512_add_epi32(X0_2, X0_3);
				X1_2 = _mm512_add_epi32(X1_2, X1_3);
				X2_2 = _mm512_add_epi32(X2_2, X2_3);
				X3_2 = _mm512_add_epi32(X3_2, X3_3);


				X0_1 = _mm512_xor_si512(X0_1, X0_2);
				X1_1 = _mm512_xor_si512(X1_1, X1_2);
				X2_1 = _mm512_xor_si512(X2_1, X2_2);
				X3_1 = _mm512_xor_si512(X3_1, X3_2);


				X0_1 = _mm512_rol_epi32(X0_1, 12);
				X1_1 = _mm512_rol_epi32(X1_1, 12);
				X2_1 = _mm512_rol_epi32(X2_1, 12);
				X3_1 = _mm512_rol_epi32(X3_1, 12);

				X0_0 = _mm512_add_epi32(X0_0, X0_1);
				X1_0 = _mm512_add_epi32(X1_0, X1_1);
				X2_0 = _mm512_add_epi32(X2_0, X2_1);
				X3_0 = _mm512_add_epi32(X3_0, X3_1);


				X0_3 = _mm512_xor_si512(X0_3, X0_0);
				X1_3 = _mm512_xor_si512(X1_3, X1_0);
				X2_3 = _mm512_xor_si512(X2_3, X2_0);
				X3_3 = _mm512_xor_si512(X3_3, X3_0);


				X0_3 = _mm512_rol_epi32(X0_3, 8);
				X1_3 = _mm512_rol_epi32(X1_3, 8);
				X2_3 = _mm512_rol_epi32(X2_3, 8);
				X3_3 = _mm512_rol_epi32(X3_3, 8);

				X0_2 = _mm512_add_epi32(X0_2, X0_3);
				X1_2 = _mm512_add_epi32(X1_2, X1_3);
				X2_2 = _mm512_add_epi32(X2_2, X2_3);
				X3_2 = _mm512_add_epi32(X3_2, X3_3);

				X0_1 = _mm512_xor_si512(X0_1, X0_2);
				X1_1 = _mm512_xor_si512(X1_1, X1_2);
				X2_1 = _mm512_xor_si512(X2_1, X2_2);
				X3_1 = _mm512_xor_si512(X3_1, X3_2);


				X0_1 = _mm512_rol_epi32(X0_1, 7);
				X1_1 = _mm512_rol_epi32(X1_1, 7);
				X2_1 = _mm512_rol_epi32(X2_1, 7);
				X3_1 = _mm512_rol_epi32(X2_1, 7);


				X0_1 = _mm512_shuffle_epi32(X0_1, _MM_SHUFFLE(0, 3, 2, 1));
				X0_2 = _mm512_shuffle_epi32(X0_2, _MM_SHUFFLE(1, 0, 3, 2));
				X0_3 = _mm512_shuffle_epi32(X0_3, _MM_SHUFFLE(2, 1, 0, 3));

				X1_1 = _mm512_shuffle_epi32(X1_1, _MM_SHUFFLE(0, 3, 2, 1));
				X1_2 = _mm512_shuffle_epi32(X1_2, _MM_SHUFFLE(1, 0, 3, 2));
				X1_3 = _mm512_shuffle_epi32(X1_3, _MM_SHUFFLE(2, 1, 0, 3));

				X2_1 = _mm512_shuffle_epi32(X2_1, _MM_SHUFFLE(0, 3, 2, 1));
				X2_2 = _mm512_shuffle_epi32(X2_2, _MM_SHUFFLE(1, 0, 3, 2));
				X2_3 = _mm512_shuffle_epi32(X2_3, _MM_SHUFFLE(2, 1, 0, 3));

				X3_1 = _mm512_shuffle_epi32(X3_1, _MM_SHUFFLE(0, 3, 2, 1));
				X3_2 = _mm512_shuffle_epi32(X3_2, _MM_SHUFFLE(1, 0, 3, 2));
				X3_3 = _mm512_shuffle_epi32(X3_3, _MM_SHUFFLE(2, 1, 0, 3));


				X0_0 = _mm512_add_epi32(X0_0, X0_1);
				X1_0 = _mm512_add_epi32(X1_0, X1_1);
				X2_0 = _mm512_add_epi32(X2_0, X2_1);
				X3_0 = _mm512_add_epi32(X3_0, X3_1);


				X0_3 = _mm512_xor_si512(X0_3, X0_0);
				X1_3 = _mm512_xor_si512(X1_3, X1_0);
				X2_3 = _mm512_xor_si512(X2_3, X2_0);
				X3_3 = _mm512_xor_si512(X3_3, X3_0);


				X0_3 = _mm512_rol_epi32(X0_3, 16);
				X1_3 = _mm512_rol_epi32(X1_3, 16);
				X2_3 = _mm512_rol_epi32(X2_3, 16);
				X3_3 = _mm512_rol_epi32(X3_3, 16);


				X0_2 = _mm512_add_epi32(X0_2, X0_3);
				X1_2 = _mm512_add_epi32(X1_2, X1_3);
				X2_2 = _mm512_add_epi32(X2_2, X2_3);
				X3_2 = _mm512_add_epi32(X3_2, X3_3);

				X0_1 = _mm512_xor_si512(X0_1, X0_2);
				X1_1 = _mm512_xor_si512(X1_1, X1_2);
				X2_1 = _mm512_xor_si512(X2_1, X2_2);
				X3_1 = _mm512_xor_si512(X3_1, X3_2);


				X0_1 = _mm512_rol_epi32(X0_1, 12);
				X1_1 = _mm512_rol_epi32(X1_1, 12);
				X2_1 = _mm512_rol_epi32(X2_1, 12);
				X3_1 = _mm512_rol_epi32(X3_1, 12);


				X0_0 = _mm512_add_epi32(X0_0, X0_1);
				X1_0 = _mm512_add_epi32(X1_0, X1_1);
				X2_0 = _mm512_add_epi32(X2_0, X2_1);
				X3_0 = _mm512_add_epi32(X3_0, X3_1);


				X0_3 = _mm512_xor_si512(X0_3, X0_0);
				X1_3 = _mm512_xor_si512(X1_3, X1_0);
				X2_3 = _mm512_xor_si512(X2_3, X3_0);
				X3_3 = _mm512_xor_si512(X3_3, X3_0);


				X0_3 = _mm512_rol_epi32(X0_3, 8);
				X1_3 = _mm512_rol_epi32(X1_3, 8);
				X2_3 = _mm512_rol_epi32(X2_3, 8);
				X3_3 = _mm512_rol_epi32(X3_3, 8);


				X0_2 = _mm512_add_epi32(X0_2, X0_3);
				X1_2 = _mm512_add_epi32(X1_2, X1_3);
				X2_2 = _mm512_add_epi32(X2_2, X3_3);
				X3_2 = _mm512_add_epi32(X2_2, X3_3);


				X0_1 = _mm512_xor_si512(X0_1, X0_2);
				X1_1 = _mm512_xor_si512(X1_1, X1_2);
				X2_1 = _mm512_xor_si512(X2_1, X2_2);
				X3_1 = _mm512_xor_si512(X3_1, X3_2);


				X0_1 = _mm512_rol_epi32(X0_1, 7);
				X1_1 = _mm512_rol_epi32(X1_1, 7);
				X2_1 = _mm512_rol_epi32(X2_1, 7);
				X3_1 = _mm512_rol_epi32(X3_1, 7);



				X0_1 = _mm512_shuffle_epi32(X0_1, _MM_SHUFFLE(2, 1, 0, 3));
				X0_2 = _mm512_shuffle_epi32(X0_2, _MM_SHUFFLE(1, 0, 3, 2));
				X0_3 = _mm512_shuffle_epi32(X0_3, _MM_SHUFFLE(0, 3, 2, 1));

				X1_1 = _mm512_shuffle_epi32(X1_1, _MM_SHUFFLE(2, 1, 0, 3));
				X1_2 = _mm512_shuffle_epi32(X1_2, _MM_SHUFFLE(1, 0, 3, 2));
				X1_3 = _mm512_shuffle_epi32(X1_3, _MM_SHUFFLE(0, 3, 2, 1));

				X2_1 = _mm512_shuffle_epi32(X2_1, _MM_SHUFFLE(2, 1, 0, 3));
				X2_2 = _mm512_shuffle_epi32(X2_2, _MM_SHUFFLE(1, 0, 3, 2));
				X2_3 = _mm512_shuffle_epi32(X2_3, _MM_SHUFFLE(0, 3, 2, 1));

				X3_1 = _mm512_shuffle_epi32(X3_1, _MM_SHUFFLE(2, 1, 0, 3));
				X3_2 = _mm512_shuffle_epi32(X3_2, _MM_SHUFFLE(1, 0, 3, 2));
				X3_3 = _mm512_shuffle_epi32(X3_3, _MM_SHUFFLE(0, 3, 2, 1));
			}




			X0_0 = _mm512_add_epi32(X0_0, state0);
			X0_1 = _mm512_add_epi32(X0_1, state1);
			X0_2 = _mm512_add_epi32(X0_2, state2);
			X0_3 = _mm512_add_epi32(X0_3, state3_0);

			X1_0 = _mm512_add_epi32(X1_0, state0);
			X1_1 = _mm512_add_epi32(X1_1, state1);
			X1_2 = _mm512_add_epi32(X1_2, state2);
			X1_3 = _mm512_add_epi32(X1_3, state3_1);

			X2_0 = _mm512_add_epi32(X2_0, state0);
			X2_1 = _mm512_add_epi32(X2_1, state1);
			X2_2 = _mm512_add_epi32(X2_2, state2);
			X2_3 = _mm512_add_epi32(X2_3, state3_2);

			X3_0 = _mm512_add_epi32(X3_0, state0);
			X3_1 = _mm512_add_epi32(X3_1, state1);
			X3_2 = _mm512_add_epi32(X3_2, state2);
			X3_3 = _mm512_add_epi32(X3_3, state3_3);




			//now making results contiguous, one 64-bytes block per register
			T1 = _mm512_permutex2var_epi64(X0_0, P1, X0_1);
			T2 = _mm512_permutex2var_epi64(X0_0, P2, X0_1);
			T3 = _mm512_permutex2var_epi64(X0_2, P1, X0_3);
			T4 = _mm512_permutex2var_epi64(X0_2, P2, X0_3);

			X0_0 = _mm512_permutex2var_epi64(T1, P3, T3);
			X0_2 = _mm512_permutex2var_epi64(T1, P4, T3);
			X0_1 = _mm512_permutex2var_epi64(T2, P3, T4);
			X0_3 = _mm512_permutex2var_epi64(T2, P4, T4);


			T1 = _mm512_permutex2var_epi64(X1_0, P1, X1_1);
			T2 = _mm512_permutex2var_epi64(X1_0, P2, X1_1);
			T3 = _mm512_permutex2var_epi64(X1_2, P1, X1_3);
			T4 = _mm512_permutex2var_epi64(X1_2, P2, X1_3);

			X1_0 = _mm512_permutex2var_epi64(T1, P3, T3);
			X1_2 = _mm512_permutex2var_epi64(T1, P4, T3);
			X1_1 = _mm512_permutex2var_epi64(T2, P3, T4);
			X1_3 = _mm512_permutex2var_epi64(T2, P4, T4);


			T1 = _mm512_permutex2var_epi64(X2_0, P1, X2_1);
			T2 = _mm512_permutex2var_epi64(X2_0, P2, X2_1);
			T3 = _mm512_permutex2var_epi64(X2_2, P1, X2_3);
			T4 = _mm512_permutex2var_epi64(X2_2, P2, X2_3);

			X2_0 = _mm512_permutex2var_epi64(T1, P3, T3);
			X2_2 = _mm512_permutex2var_epi64(T1, P4, T3);
			X2_1 = _mm512_permutex2var_epi64(T2, P3, T4);
			X2_3 = _mm512_permutex2var_epi64(T2, P4, T4);

			T1 = _mm512_permutex2var_epi64(X3_0, P1, X3_1);
			T2 = _mm512_permutex2var_epi64(X3_0, P2, X3_1);
			T3 = _mm512_permutex2var_epi64(X3_2, P1, X3_3);
			T4 = _mm512_permutex2var_epi64(X3_2, P2, X3_3);

			X3_0 = _mm512_permutex2var_epi64(T1, P3, T3);
			X3_2 = _mm512_permutex2var_epi64(T1, P4, T3);
			X3_1 = _mm512_permutex2var_epi64(T2, P3, T4);
			X3_3 = _mm512_permutex2var_epi64(T2, P4, T4);




			if (In)
			{
				T1 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 0 * 64));
				T2 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 1 * 64));
				T3 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 2 * 64));
				T4 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 3 * 64));

				T1 = _mm512_xor_si512(T1, X0_0);
				T2 = _mm512_xor_si512(T2, X0_1);
				T3 = _mm512_xor_si512(T3, X0_2);
				T4 = _mm512_xor_si512(T4, X0_3);

				_mm512_storeu_si512(CurrentOut + 0 * 64, T1);
				_mm512_storeu_si512(CurrentOut + 1 * 64, T2);
				_mm512_storeu_si512(CurrentOut + 2 * 64, T3);
				_mm512_storeu_si512(CurrentOut + 3 * 64, T4);

				T1 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 4 * 64));
				T2 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 5 * 64));
				T3 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 6 * 64));
				T4 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 7 * 64));

				T1 = _mm512_xor_si512(T1, X1_0);
				T2 = _mm512_xor_si512(T2, X1_1);
				T3 = _mm512_xor_si512(T3, X1_2);
				T4 = _mm512_xor_si512(T4, X1_3);

				_mm512_storeu_si512(CurrentOut + 4 * 64, T1);
				_mm512_storeu_si512(CurrentOut + 5 * 64, T2);
				_mm512_storeu_si512(CurrentOut + 6 * 64, T3);
				_mm512_storeu_si512(CurrentOut + 7 * 64, T4);


				T1 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 8 * 64));
				T2 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 9 * 64));
				T3 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 10 * 64));
				T4 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 11 * 64));

				T1 = _mm512_xor_si512(T1, X2_0);
				T2 = _mm512_xor_si512(T2, X2_1);
				T3 = _mm512_xor_si512(T3, X2_2);
				T4 = _mm512_xor_si512(T4, X2_3);

				_mm512_storeu_si512(CurrentOut + 8 * 64, T1);
				_mm512_storeu_si512(CurrentOut + 9 * 64, T2);
				_mm512_storeu_si512(CurrentOut + 10 * 64, T3);
				_mm512_storeu_si512(CurrentOut + 11 * 64, T4);

				T1 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 12 * 64));
				T2 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 13 * 64));
				T3 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 14 * 64));
				T4 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 15 * 64));

				T1 = _mm512_xor_si512(T1, X3_0);
				T2 = _mm512_xor_si512(T2, X3_1);
				T3 = _mm512_xor_si512(T3, X3_2);
				T4 = _mm512_xor_si512(T4, X3_3);

				_mm512_storeu_si512(CurrentOut + 12 * 64, T1);
				_mm512_storeu_si512(CurrentOut + 13 * 64, T2);
				_mm512_storeu_si512(CurrentOut + 14 * 64, T3);
				_mm512_storeu_si512(CurrentOut + 15 * 64, T4);

			}
			else
			{

				_mm512_storeu_si512(CurrentOut + 0 * 64, X0_0);
				_mm512_storeu_si512(CurrentOut + 1 * 64, X0_1);
				_mm512_storeu_si512(CurrentOut + 2 * 64, X0_2);
				_mm512_storeu_si512(CurrentOut + 3 * 64, X0_3);

				_mm512_storeu_si512(CurrentOut + 4 * 64, X1_0);
				_mm512_storeu_si512(CurrentOut + 5 * 64, X1_1);
				_mm512_storeu_si512(CurrentOut + 6 * 64, X1_2);
				_mm512_storeu_si512(CurrentOut + 7 * 64, X1_3);

				_mm512_storeu_si512(CurrentOut + 8 * 64, X2_0);
				_mm512_storeu_si512(CurrentOut + 9 * 64, X2_1);
				_mm512_storeu_si512(CurrentOut + 10 * 64, X2_2);
				_mm512_storeu_si512(CurrentOut + 11 * 64, X2_3);

				_mm512_storeu_si512(CurrentOut + 12 * 64, X3_0);
				_mm512_storeu_si512(CurrentOut + 13 * 64, X3_1);
				_mm512_storeu_si512(CurrentOut + 14 * 64, X3_2);
				_mm512_storeu_si512(CurrentOut + 15 * 64, X3_3);


			}


			if (CurrentIn) CurrentIn += 1024;
			CurrentOut += 1024;

			state3_0 = _mm512_add_epi32(state3_0, ctr_increment);
			state3_1 = _mm512_add_epi32(state3_1, ctr_increment);
			state3_2 = _mm512_add_epi32(state3_2, ctr_increment);
			state3_3 = _mm512_add_epi32(state3_3, ctr_increment);

		}

		ChaCha20AddCounter(state, FullBlocksCount * 16);
	}
	if (RemainingBytes == 0) return;
	//now computing rest in 4-blocks cycle
	ctr_increment = _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4);

	while (1)
	{
		__m512i X0_0 = state0;
		__m512i X0_1 = state1;
		__m512i X0_2 = state2;
		__m512i X0_3 = state3_0;




		for (int i = rounds; i > 0; i -= 2)
		{
			X0_0 = _mm512_add_epi32(X0_0, X0_1);

			X0_3 = _mm512_xor_si512(X0_3, X0_0);

			X0_3 = _mm512_rol_epi32(X0_3, 16);

			X0_2 = _mm512_add_epi32(X0_2, X0_3);

			X0_1 = _mm512_xor_si512(X0_1, X0_2);

			X0_1 = _mm512_rol_epi32(X0_1, 12);

			X0_0 = _mm512_add_epi32(X0_0, X0_1);

			X0_3 = _mm512_xor_si512(X0_3, X0_0);

			X0_3 = _mm512_rol_epi32(X0_3, 8);

			X0_2 = _mm512_add_epi32(X0_2, X0_3);

			X0_1 = _mm512_xor_si512(X0_1, X0_2);

			X0_1 = _mm512_rol_epi32(X0_1, 7);

			X0_1 = _mm512_shuffle_epi32(X0_1, _MM_SHUFFLE(0, 3, 2, 1));
			X0_2 = _mm512_shuffle_epi32(X0_2, _MM_SHUFFLE(1, 0, 3, 2));
			X0_3 = _mm512_shuffle_epi32(X0_3, _MM_SHUFFLE(2, 1, 0, 3));

			X0_0 = _mm512_add_epi32(X0_0, X0_1);

			X0_3 = _mm512_xor_si512(X0_3, X0_0);

			X0_3 = _mm512_rol_epi32(X0_3, 16);

			X0_2 = _mm512_add_epi32(X0_2, X0_3);
			X0_1 = _mm512_xor_si512(X0_1, X0_2);

			X0_1 = _mm512_rol_epi32(X0_1, 12);

			X0_0 = _mm512_add_epi32(X0_0, X0_1);

			X0_3 = _mm512_xor_si512(X0_3, X0_0);

			X0_3 = _mm512_rol_epi32(X0_3, 8);

			X0_2 = _mm512_add_epi32(X0_2, X0_3);

			X0_1 = _mm512_xor_si512(X0_1, X0_2);

			X0_1 = _mm512_rol_epi32(X0_1, 7);

			X0_1 = _mm512_shuffle_epi32(X0_1, _MM_SHUFFLE(2, 1, 0, 3));
			X0_2 = _mm512_shuffle_epi32(X0_2, _MM_SHUFFLE(1, 0, 3, 2));
			X0_3 = _mm512_shuffle_epi32(X0_3, _MM_SHUFFLE(0, 3, 2, 1));
		}


		X0_0 = _mm512_add_epi32(X0_0, state0);
		X0_1 = _mm512_add_epi32(X0_1, state1);
		X0_2 = _mm512_add_epi32(X0_2, state2);
		X0_3 = _mm512_add_epi32(X0_3, state3_0);





		//now making results contiguous, one 64-bytes block per register
		T1 = _mm512_permutex2var_epi64(X0_0, P1, X0_1);
		T2 = _mm512_permutex2var_epi64(X0_0, P2, X0_1);
		T3 = _mm512_permutex2var_epi64(X0_2, P1, X0_3);
		T4 = _mm512_permutex2var_epi64(X0_2, P2, X0_3);

		X0_0 = _mm512_permutex2var_epi64(T1, P3, T3);
		X0_2 = _mm512_permutex2var_epi64(T1, P4, T3);
		X0_1 = _mm512_permutex2var_epi64(T2, P3, T4);
		X0_3 = _mm512_permutex2var_epi64(T2, P4, T4);


		if (RemainingBytes >= 256)
		{
			if (In)
			{
				T1 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 0 * 64));
				T2 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 1 * 64));
				T3 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 2 * 64));
				T4 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 3 * 64));

				T1 = _mm512_xor_si512(T1, X0_0);
				T2 = _mm512_xor_si512(T2, X0_1);
				T3 = _mm512_xor_si512(T3, X0_2);
				T4 = _mm512_xor_si512(T4, X0_3);

				_mm512_storeu_si512(CurrentOut + 0 * 64, T1);
				_mm512_storeu_si512(CurrentOut + 1 * 64, T2);
				_mm512_storeu_si512(CurrentOut + 2 * 64, T3);
				_mm512_storeu_si512(CurrentOut + 3 * 64, T4);



			}
			else
			{

				_mm512_storeu_si512(CurrentOut + 0 * 64, X0_0);
				_mm512_storeu_si512(CurrentOut + 1 * 64, X0_1);
				_mm512_storeu_si512(CurrentOut + 2 * 64, X0_2);
				_mm512_storeu_si512(CurrentOut + 3 * 64, X0_3);


			}

			ChaCha20AddCounter(state, 4);
			RemainingBytes -= 256;
			if (RemainingBytes == 0) return;

			if (CurrentIn) CurrentIn += 256;
			CurrentOut += 256;
			state3_0 = _mm512_add_epi32(state3_0, ctr_increment);
			continue;
		}
		else
		{

			if (In)
			{

				if (RemainingBytes < 64)
				{
					PartialXor(X0_0, CurrentIn, CurrentOut, RemainingBytes);
					ChaCha20AddCounter(state, 1);
					return;
				}
				T1 = _mm512_loadu_si512((const __m512i*)(CurrentIn + 0 * 64));
				T1 = _mm512_xor_si512(T1, X0_0);
				_mm512_storeu_si512(CurrentOut + 0 * 64, T1);

				RemainingBytes -= 64;
				if (RemainingBytes == 0)
				{
					ChaCha20AddCounter(state, 1);
					return;
				}

				CurrentIn += 64;
				CurrentOut += 64;

				if (RemainingBytes < 64)
				{
					PartialXor(X0_1, CurrentIn, CurrentOut, RemainingBytes);
					ChaCha20AddCounter(state, 2);
					return;
				}
				T1 = _mm512_loadu_si512((const __m512i*)(CurrentIn));
				T1 = _mm512_xor_si512(T1, X0_1);
				_mm512_storeu_si512(CurrentOut, T1);

				RemainingBytes -= 64;
				if (RemainingBytes == 0)
				{
					ChaCha20AddCounter(state, 2);
					return;
				}


				CurrentIn += 64;
				CurrentOut += 64;

				if (RemainingBytes < 64)
				{
					PartialXor(X0_2, CurrentIn, CurrentOut, RemainingBytes);
					ChaCha20AddCounter(state, 3);
					return;
				}
				T1 = _mm512_loadu_si512((const __m512i*)(CurrentIn));
				T1 = _mm512_xor_si512(T1, X0_2);
				_mm512_storeu_si512(CurrentOut, T1);

				RemainingBytes -= 64;
				if (RemainingBytes == 0)
				{
					ChaCha20AddCounter(state, 3);
					return;
				}

				PartialXor(X0_3, CurrentIn, CurrentOut, RemainingBytes);
				ChaCha20AddCounter(state, 4);
				return;


			}
			else
			{
				if (RemainingBytes < 64)
				{
					PartialStore(X0_0, CurrentOut, RemainingBytes);
					ChaCha20AddCounter(state, 1);
					return;
				}
				_mm512_storeu_si512((__m512i*)(CurrentOut), X0_0);
				RemainingBytes -= 64;
				if (RemainingBytes == 0)
				{
					ChaCha20AddCounter(state, 1);
					return;
				}
				CurrentOut += 64;

				if (RemainingBytes < 64)
				{
					PartialStore(X0_1, CurrentOut, RemainingBytes);
					ChaCha20AddCounter(state, 2);
					return;
				}
				_mm512_storeu_si512((__m512i*)(CurrentOut), X0_1);
				RemainingBytes -= 64;
				if (RemainingBytes == 0)
				{
					ChaCha20AddCounter(state, 2);
					return;
				}
				CurrentOut += 64;

				if (RemainingBytes < 64)
				{
					PartialStore(X0_2, CurrentOut, RemainingBytes);
					ChaCha20AddCounter(state, 3);
					return;
				}
				_mm512_storeu_si512((__m512i*)(CurrentOut), X0_2);
				RemainingBytes -= 64;
				if (RemainingBytes == 0)
				{
					ChaCha20AddCounter(state, 3);
					return;
				}
				CurrentOut += 64;

				PartialStore(X0_3, CurrentOut, RemainingBytes);
				ChaCha20AddCounter(state, 4);
				return;

			}
		}
	}
}

#endif