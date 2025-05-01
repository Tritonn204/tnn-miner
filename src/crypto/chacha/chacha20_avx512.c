#include "chacha20.h"

#if defined(__x86_64__)

#include <immintrin.h>
#include <memory.h>

__attribute__((target("avx512f,avx512dq,avx512bw")))
static inline void PartialXor(const __m512i val, uint8_t* Src, uint8_t* Dest, uint64_t Size)
{
	_Alignas(64) uint8_t BuffForPartialOp[64];
	memcpy(BuffForPartialOp, Src, Size);
	_mm512_storeu_si512((__m512i*)(BuffForPartialOp), _mm512_xor_si512(val, _mm512_loadu_si512((const __m512i*)BuffForPartialOp)));
	memcpy(Dest, BuffForPartialOp, Size);
}

__attribute__((target("avx512f,avx512dq,avx512bw")))
static inline void PartialStore(const __m512i val, uint8_t* Dest, uint64_t Size)
{
	_Alignas(64) uint8_t BuffForPartialOp[64];
	_mm512_storeu_si512((__m512i*)(BuffForPartialOp), val);
	memcpy(Dest, BuffForPartialOp, Size);
}

__attribute__((target("avx512f,avx512dq,avx512bw"))) void ChaCha20EncryptBytes(
    uint8_t *state, uint8_t *In, uint8_t *Out, uint64_t Size, int rounds)
{
  uint8_t *CurrentIn = In;
  uint8_t *CurrentOut = Out;

  uint64_t FullBlocksCount = Size / 1024;
  uint64_t RemainingBytes = Size % 1024;

  const __m512i state0 = _mm512_broadcast_i32x4(_mm_set_epi32(
      1797285236, 2036477234, 857760878, 1634760805)); //"expand 32-byte k"
  const __m512i state1 =
      _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(state)));
  const __m512i state2 =
      _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(state + 16)));

  // AVX2 for partial blocks
  const __m256i state0_r = _mm256_broadcastsi128_si256(_mm_set_epi32(
      1797285236, 2036477234, 857760878, 1634760805)); //"expand 32-byte k"
  const __m256i state1_r =
      _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(state)));
  const __m256i state2_r = _mm256_broadcastsi128_si256(
      _mm_load_si128((const __m128i *)(state + 16)));

  // end of AVX2 definitions

  // __m512i state3_r = _mm512_broadcast_i32x4(
  //     _mm_load_si128((const __m128i*)(state + 32)));

  __m512i CTR0 = _mm512_set_epi64(0, 0, 0, 4, 0, 8, 0, 12);
  const __m512i CTR1 = _mm512_set_epi64(0, 1, 0, 5, 0, 9, 0, 13);
  const __m512i CTR2 = _mm512_set_epi64(0, 2, 0, 6, 0, 10, 0, 14);
  const __m512i CTR3 = _mm512_set_epi64(0, 3, 0, 7, 0, 11, 0, 15);

  // permutation indexes for results
  const __m512i P1 = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  const __m512i P2 = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  const __m512i P3 = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
  const __m512i P4 = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);

  __m512i T1;
  __m512i T2;
  __m512i T3;
  __m512i T4;

  if (FullBlocksCount > 0)
  {
    for (uint64_t n = 0; n < FullBlocksCount; n++)
    {
      const __m512i state3 = _mm512_broadcast_i32x4(
          _mm_loadu_si128((const __m128i *)(state + 32)));

      __m512i X0_0 = state0;
      __m512i X0_1 = state1;
      __m512i X0_2 = state2;
      __m512i X0_3 = _mm512_add_epi32(state3, CTR0);

      __m512i X1_0 = state0;
      __m512i X1_1 = state1;
      __m512i X1_2 = state2;
      __m512i X1_3 = _mm512_add_epi32(state3, CTR1);

      __m512i X2_0 = state0;
      __m512i X2_1 = state1;
      __m512i X2_2 = state2;
      __m512i X2_3 = _mm512_add_epi32(state3, CTR2);

      __m512i X3_0 = state0;
      __m512i X3_1 = state1;
      __m512i X3_2 = state2;
      __m512i X3_3 = _mm512_add_epi32(state3, CTR3);

      for (int i = rounds; i > 0; i -= 2)
      {
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

        //

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

        //

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

        //

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
        X3_1 = _mm512_rol_epi32(X3_1, 7);

        //

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

        //

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

        //

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

        //

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

        //

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
        X3_1 = _mm512_rol_epi32(X3_1, 7);

        //

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
      X0_3 = _mm512_add_epi32(X0_3, state3);
      X0_3 = _mm512_add_epi32(X0_3, CTR0);

      X1_0 = _mm512_add_epi32(X1_0, state0);
      X1_1 = _mm512_add_epi32(X1_1, state1);
      X1_2 = _mm512_add_epi32(X1_2, state2);
      X1_3 = _mm512_add_epi32(X1_3, state3);
      X1_3 = _mm512_add_epi32(X1_3, CTR1);

      X2_0 = _mm512_add_epi32(X2_0, state0);
      X2_1 = _mm512_add_epi32(X2_1, state1);
      X2_2 = _mm512_add_epi32(X2_2, state2);
      X2_3 = _mm512_add_epi32(X2_3, state3);
      X2_3 = _mm512_add_epi32(X2_3, CTR2);

      X3_0 = _mm512_add_epi32(X3_0, state0);
      X3_1 = _mm512_add_epi32(X3_1, state1);
      X3_2 = _mm512_add_epi32(X3_2, state2);
      X3_3 = _mm512_add_epi32(X3_3, state3);
      X3_3 = _mm512_add_epi32(X3_3, CTR3);

      // permutation indexes
      __m512i idx1 = _mm512_set_epi64(15, 14, 7, 6, 15, 14, 7, 6);
      __m512i idx2 = _mm512_set_epi64(13, 12, 5, 4, 13, 12, 5, 4);
      __m512i idx3 = _mm512_set_epi64(11, 10, 3, 2, 11, 10, 3, 2);
      __m512i idx4 = _mm512_set_epi64(9, 8, 1, 0, 9, 8, 1, 0);

      // Blend the results
      __m512i X0_0F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X0_0, idx1, X0_1),
          _mm512_permutex2var_epi64(X0_2, idx1, X0_3));
      __m512i X0_1F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X1_0, idx1, X1_1),
          _mm512_permutex2var_epi64(X1_2, idx1, X1_3));
      __m512i X0_2F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X2_0, idx1, X2_1),
          _mm512_permutex2var_epi64(X2_2, idx1, X2_3));
      __m512i X0_3F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X3_0, idx1, X3_1),
          _mm512_permutex2var_epi64(X3_2, idx1, X3_3));

      //

      __m512i X1_0F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X0_0, idx2, X0_1),
          _mm512_permutex2var_epi64(X0_2, idx2, X0_3));
      __m512i X1_1F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X1_0, idx2, X1_1),
          _mm512_permutex2var_epi64(X1_2, idx2, X1_3));
      __m512i X1_2F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X2_0, idx2, X2_1),
          _mm512_permutex2var_epi64(X2_2, idx2, X2_3));
      __m512i X1_3F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X3_0, idx2, X3_1),
          _mm512_permutex2var_epi64(X3_2, idx2, X3_3));

      //

      __m512i X2_0F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X0_0, idx3, X0_1),
          _mm512_permutex2var_epi64(X0_2, idx3, X0_3));
      __m512i X2_1F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X1_0, idx3, X1_1),
          _mm512_permutex2var_epi64(X1_2, idx3, X1_3));
      __m512i X2_2F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X2_0, idx3, X2_1),
          _mm512_permutex2var_epi64(X2_2, idx3, X2_3));
      __m512i X2_3F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X3_0, idx3, X3_1),
          _mm512_permutex2var_epi64(X3_2, idx3, X3_3));

      //

      __m512i X3_0F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X0_0, idx4, X0_1),
          _mm512_permutex2var_epi64(X0_2, idx4, X0_3));
      __m512i X3_1F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X1_0, idx4, X1_1),
          _mm512_permutex2var_epi64(X1_2, idx4, X1_3));
      __m512i X3_2F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X2_0, idx4, X2_1),
          _mm512_permutex2var_epi64(X2_2, idx4, X2_3));
      __m512i X3_3F = _mm512_mask_blend_epi64(
          0xF0,
          _mm512_permutex2var_epi64(X3_0, idx4, X3_1),
          _mm512_permutex2var_epi64(X3_2, idx4, X3_3));

      if (In)
      {
        T1 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 0 * 64));
        T2 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 1 * 64));
        T3 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 2 * 64));
        T4 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 3 * 64));

        T1 = _mm512_xor_si512(T1, X0_0F);
        T2 = _mm512_xor_si512(T2, X0_1F);
        T3 = _mm512_xor_si512(T3, X0_2F);
        T4 = _mm512_xor_si512(T4, X0_3F);

        _mm512_storeu_si512(CurrentOut + 0 * 64, T1);
        _mm512_storeu_si512(CurrentOut + 1 * 64, T2);
        _mm512_storeu_si512(CurrentOut + 2 * 64, T3);
        _mm512_storeu_si512(CurrentOut + 3 * 64, T4);

        T1 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 4 * 64));
        T2 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 5 * 64));
        T3 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 6 * 64));
        T4 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 7 * 64));

        T1 = _mm512_xor_si512(T1, X1_0F);
        T2 = _mm512_xor_si512(T2, X1_1F);
        T3 = _mm512_xor_si512(T3, X1_2F);
        T4 = _mm512_xor_si512(T4, X1_3F);

        _mm512_storeu_si512(CurrentOut + 4 * 64, T1);
        _mm512_storeu_si512(CurrentOut + 5 * 64, T2);
        _mm512_storeu_si512(CurrentOut + 6 * 64, T3);
        _mm512_storeu_si512(CurrentOut + 7 * 64, T4);

        T1 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 8 * 64));
        T2 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 9 * 64));
        T3 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 10 * 64));
        T4 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 11 * 64));

        T1 = _mm512_xor_si512(T1, X2_0F);
        T2 = _mm512_xor_si512(T2, X2_1F);
        T3 = _mm512_xor_si512(T3, X2_2F);
        T4 = _mm512_xor_si512(T4, X2_3F);

        _mm512_storeu_si512(CurrentOut + 8 * 64, T1);
        _mm512_storeu_si512(CurrentOut + 9 * 64, T2);
        _mm512_storeu_si512(CurrentOut + 10 * 64, T3);
        _mm512_storeu_si512(CurrentOut + 11 * 64, T4);

        T1 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 12 * 64));
        T2 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 13 * 64));
        T3 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 14 * 64));
        T4 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 15 * 64));

        T1 = _mm512_xor_si512(T1, X3_0F);
        T2 = _mm512_xor_si512(T2, X3_1F);
        T3 = _mm512_xor_si512(T3, X3_2F);
        T4 = _mm512_xor_si512(T4, X3_3F);

        _mm512_storeu_si512(CurrentOut + 12 * 64, T1);
        _mm512_storeu_si512(CurrentOut + 13 * 64, T2);
        _mm512_storeu_si512(CurrentOut + 14 * 64, T3);
        _mm512_storeu_si512(CurrentOut + 15 * 64, T4);
      }
      else
      {
        _mm512_storeu_si512(CurrentOut + 0 * 64, X0_0F);
        _mm512_storeu_si512(CurrentOut + 1 * 64, X0_1F);
        _mm512_storeu_si512(CurrentOut + 2 * 64, X0_2F);
        _mm512_storeu_si512(CurrentOut + 3 * 64, X0_3F);

        _mm512_storeu_si512(CurrentOut + 4 * 64, X1_0F);
        _mm512_storeu_si512(CurrentOut + 5 * 64, X1_1F);
        _mm512_storeu_si512(CurrentOut + 6 * 64, X1_2F);
        _mm512_storeu_si512(CurrentOut + 7 * 64, X1_3F);

        _mm512_storeu_si512(CurrentOut + 8 * 64, X2_0F);
        _mm512_storeu_si512(CurrentOut + 9 * 64, X2_1F);
        _mm512_storeu_si512(CurrentOut + 10 * 64, X2_2F);
        _mm512_storeu_si512(CurrentOut + 11 * 64, X2_3F);

        _mm512_storeu_si512(CurrentOut + 12 * 64, X3_0F);
        _mm512_storeu_si512(CurrentOut + 13 * 64, X3_1F);
        _mm512_storeu_si512(CurrentOut + 14 * 64, X3_2F);
        _mm512_storeu_si512(CurrentOut + 15 * 64, X3_3F);
      }

      ChaCha20AddCounter(state, 16);
      if (CurrentIn)
        CurrentIn += 1024;
      CurrentOut += 1024;
    }
  }

  if (RemainingBytes == 0)
    return;
  // now computing rest in 4-blocks cycle

  CTR0 = _mm512_set_epi64(0, 0, 0, 1, 0, 2, 0, 3);

  while (1)
  {
    const __m512i state3 = _mm512_broadcast_i32x4(
        _mm_load_si128((const __m128i *)(state + 32)));

    __m512i X0_0 = state0;
    __m512i X0_1 = state1;
    __m512i X0_2 = state2;
    __m512i X0_3 = _mm512_add_epi32(state3, CTR0);

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
    X0_3 = _mm512_add_epi32(X0_3, state3);
    X0_3 = _mm512_add_epi32(X0_3, CTR0);

    __m512i idx1 = _mm512_set_epi64(15, 14, 7, 6, 15, 14, 7, 6);
    __m512i idx2 = _mm512_set_epi64(13, 12, 5, 4, 13, 12, 5, 4);
    __m512i idx3 = _mm512_set_epi64(11, 10, 3, 2, 11, 10, 3, 2);
    __m512i idx4 = _mm512_set_epi64(9, 8, 1, 0, 9, 8, 1, 0);

    // Blend the results
    __m512i X0_0F = _mm512_mask_blend_epi64(
        0xF0,
        _mm512_permutex2var_epi64(X0_0, idx1, X0_1),
        _mm512_permutex2var_epi64(X0_2, idx1, X0_3));
    __m512i X0_1F = _mm512_mask_blend_epi64(
        0xF0,
        _mm512_permutex2var_epi64(X0_0, idx2, X0_1),
        _mm512_permutex2var_epi64(X0_2, idx2, X0_3));
    __m512i X0_2F = _mm512_mask_blend_epi64(
        0xF0,
        _mm512_permutex2var_epi64(X0_0, idx3, X0_1),
        _mm512_permutex2var_epi64(X0_2, idx3, X0_3));
    __m512i X0_3F = _mm512_mask_blend_epi64(
        0xF0,
        _mm512_permutex2var_epi64(X0_0, idx4, X0_1),
        _mm512_permutex2var_epi64(X0_2, idx4, X0_3));

    if (RemainingBytes >= 256)
    {
      if (In)
      {
        T1 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 0 * 64));
        T2 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 1 * 64));
        T3 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 2 * 64));
        T4 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 3 * 64));

        T1 = _mm512_xor_si512(T1, X0_0F);
        T2 = _mm512_xor_si512(T2, X0_1F);
        T3 = _mm512_xor_si512(T3, X0_2F);
        T4 = _mm512_xor_si512(T4, X0_3F);

        _mm512_storeu_si512(CurrentOut + 0 * 64, T1);
        _mm512_storeu_si512(CurrentOut + 1 * 64, T2);
        _mm512_storeu_si512(CurrentOut + 2 * 64, T3);
        _mm512_storeu_si512(CurrentOut + 3 * 64, T4);
      }
      else
      {
        _mm512_storeu_si512(CurrentOut + 0 * 64, X0_0F);
        _mm512_storeu_si512(CurrentOut + 1 * 64, X0_1F);
        _mm512_storeu_si512(CurrentOut + 2 * 64, X0_2F);
        _mm512_storeu_si512(CurrentOut + 3 * 64, X0_3F);
      }
      ChaCha20AddCounter(state, 4);
      RemainingBytes -= 256;
      if (RemainingBytes == 0)
        return;
      if (CurrentIn)
        CurrentIn += 256;
      CurrentOut += 256;
      continue;
    }
    else
    {
      if (In)
      {
        if (RemainingBytes < 64)
        {
          PartialXor(X0_0F, CurrentIn, CurrentOut, RemainingBytes);
          ChaCha20AddCounter(state, 1);
          return;
        }
        T1 = _mm512_loadu_si512((const __m512i *)(CurrentIn + 0 * 64));
        T1 = _mm512_xor_si512(T1, X0_0F);
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
          PartialXor(X0_1F, CurrentIn, CurrentOut, RemainingBytes);
          ChaCha20AddCounter(state, 2);
          return;
        }
        T1 = _mm512_loadu_si512((const __m512i *)(CurrentIn));
        T1 = _mm512_xor_si512(T1, X0_1F);
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
          PartialXor(X0_2F, CurrentIn, CurrentOut, RemainingBytes);
          ChaCha20AddCounter(state, 3);
          return;
        }
        T1 = _mm512_loadu_si512((const __m512i *)(CurrentIn));
        T1 = _mm512_xor_si512(T1, X0_2F);
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
          PartialStore(X0_0F, CurrentOut, RemainingBytes);
          ChaCha20AddCounter(state, 1);
          return;
        }
        _mm512_storeu_si512((__m512i *)(CurrentOut), X0_0F);
        RemainingBytes -= 64;
        if (RemainingBytes == 0)
        {
          ChaCha20AddCounter(state, 1);
          return;
        }
        CurrentOut += 64;

        if (RemainingBytes < 64)
        {
          PartialStore(X0_1F, CurrentOut, RemainingBytes);
          ChaCha20AddCounter(state, 2);
          return;
        }
        _mm512_storeu_si512((__m512i *)(CurrentOut), X0_1F);
        RemainingBytes -= 64;
        if (RemainingBytes == 0)
        {
          ChaCha20AddCounter(state, 2);
          return;
        }
        CurrentOut += 64;

        if (RemainingBytes < 64)
        {
          PartialStore(X0_2F, CurrentOut, RemainingBytes);
          ChaCha20AddCounter(state, 3);
          return;
        }
        _mm512_storeu_si512((__m512i *)(CurrentOut), X0_2F);
        RemainingBytes -= 64;
        if (RemainingBytes == 0)
        {
          ChaCha20AddCounter(state, 3);
          return;
        }
        CurrentOut += 64;

        PartialStore(X0_3F, CurrentOut, RemainingBytes);
        ChaCha20AddCounter(state, 4);
        return;
      }
    }
  }
}

#endif