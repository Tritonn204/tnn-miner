#include "chacha20.h"

#if defined(__x86_64__)

#include <immintrin.h>
#include <inttypes.h>
#include <memory.h>

#define PartialXor(val, Src, Dest, Size)                                 \
do {                                                                             \
    {                                                                            \
        alignas(64) uint8_t BuffForPartialOp[64];                                \
        memcpy(BuffForPartialOp, Src, Size);                                     \
        _mm512_store_si512((__m512i*)(BuffForPartialOp),                         \
            _mm512_xor_si512(val, _mm512_loadu_si512((const __m512i*)BuffForPartialOp))); \
        memcpy(Dest, BuffForPartialOp, Size);                                    \
    }                                                                            \
} while (0)


#define PartialStore(val, Dest, Size)                                    \
do {                                                                             \
    {                                                                            \
        alignas(64) uint8_t BuffForPartialOp[64];                                \
        _mm512_store_si512((__m512i*)(BuffForPartialOp), val);                  \
        memcpy(Dest, BuffForPartialOp, Size);                                    \
    }                                                                            \
} while (0)

TNN_TARGET_CLONE(
  ChaCha20EncryptBytes,
  void,
  (uint8_t *state, uint8_t *In, uint8_t *Out, uint64_t Size, int rounds),
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
  },
  TNN_TARGETS_X86_CHACHA512
)

#define STORE_STREAM(stream_idx, perm_idx) \
{ \
    __m512i px0 = _mm512_permutexvar_epi32(perm_idx, x0); \
    __m512i px4 = _mm512_permutexvar_epi32(perm_idx, x4); \
    __m512i px8 = _mm512_permutexvar_epi32(perm_idx, x8); \
    __m512i px12 = _mm512_permutexvar_epi32(perm_idx, x12); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 0), _mm512_castsi512_si128(px0)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 16), _mm512_castsi512_si128(px4)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 32), _mm512_castsi512_si128(px8)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 48), _mm512_castsi512_si128(px12)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 64), _mm512_extracti32x4_epi32(px0, 1)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 80), _mm512_extracti32x4_epi32(px4, 1)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 96), _mm512_extracti32x4_epi32(px8, 1)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 112), _mm512_extracti32x4_epi32(px12, 1)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 128), _mm512_extracti32x4_epi32(px0, 2)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 144), _mm512_extracti32x4_epi32(px4, 2)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 160), _mm512_extracti32x4_epi32(px8, 2)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 176), _mm512_extracti32x4_epi32(px12, 2)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 192), _mm512_extracti32x4_epi32(px0, 3)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 208), _mm512_extracti32x4_epi32(px4, 3)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 224), _mm512_extracti32x4_epi32(px8, 3)); \
    _mm_storeu_si128((__m128i*)(outputs[stream_idx] + 240), _mm512_extracti32x4_epi32(px12, 3)); \
}

// TNN_TARGET_CLONE(
//   ChaCha20EncryptXelis,
//   void,
//   (const uint8_t (*keys)[32], const uint8_t (*nonces)[12], uint8_t **outputs, size_t bytes_per_stream, int rounds),
//   {
//     // ChaCha constants
//     const __m512i const0 = _mm512_set1_epi32(0x61707865);
//     const __m512i const1 = _mm512_set1_epi32(0x3320646e);
//     const __m512i const2 = _mm512_set1_epi32(0x79622d32);
//     const __m512i const3 = _mm512_set1_epi32(0x6b206574);
    
//     // Load and broadcast keys for each stream
//     // Each k# contains the same key word broadcast across all lanes,
//     // but different for each of the 4 streams (interleaved)
//     __m512i k0;
//     __m512i k1;
//     __m512i k2;
//     __m512i k3;
//     __m512i k4;
//     __m512i k5;
//     __m512i k6;
//     __m512i k7;
//     {
//         // Load 4 keys
//         __m128i key0 = _mm_loadu_si128((const __m128i*)keys[0]);
//         __m128i key1 = _mm_loadu_si128((const __m128i*)keys[1]);
//         __m128i key2 = _mm_loadu_si128((const __m128i*)keys[2]);
//         __m128i key3 = _mm_loadu_si128((const __m128i*)keys[3]);
        
//         __m128i key0_hi = _mm_loadu_si128((const __m128i*)(keys[0] + 16));
//         __m128i key1_hi = _mm_loadu_si128((const __m128i*)(keys[1] + 16));
//         __m128i key2_hi = _mm_loadu_si128((const __m128i*)(keys[2] + 16));
//         __m128i key3_hi = _mm_loadu_si128((const __m128i*)(keys[3] + 16));
        
//         // Transpose: want k0 = [key0[0], key1[0], key2[0], key3[0], ...repeated 4x]
//         __m128i t0 = _mm_unpacklo_epi32(key0, key1);     // k0[0],k1[0],k0[1],k1[1]
//         __m128i t1 = _mm_unpacklo_epi32(key2, key3);     // k2[0],k3[0],k2[1],k3[1]
//         __m128i t2 = _mm_unpackhi_epi32(key0, key1);     // k0[2],k1[2],k0[3],k1[3]
//         __m128i t3 = _mm_unpackhi_epi32(key2, key3);     // k2[2],k3[2],k2[3],k3[3]
        
//         __m128i u0 = _mm_unpacklo_epi64(t0, t1);         // k0[0],k1[0],k2[0],k3[0]
//         __m128i u1 = _mm_unpackhi_epi64(t0, t1);         // k0[1],k1[1],k2[1],k3[1]
//         __m128i u2 = _mm_unpacklo_epi64(t2, t3);         // k0[2],k1[2],k2[2],k3[2]
//         __m128i u3 = _mm_unpackhi_epi64(t2, t3);         // k0[3],k1[3],k2[3],k3[3]
        
//         // Broadcast to 512-bit (4 copies of the 128-bit pattern)
//         k0 = _mm512_broadcast_i32x4(u0);
//         k1 = _mm512_broadcast_i32x4(u1);
//         k2 = _mm512_broadcast_i32x4(u2);
//         k3 = _mm512_broadcast_i32x4(u3);
        
//         // Same for high half of keys
//         t0 = _mm_unpacklo_epi32(key0_hi, key1_hi);
//         t1 = _mm_unpacklo_epi32(key2_hi, key3_hi);
//         t2 = _mm_unpackhi_epi32(key0_hi, key1_hi);
//         t3 = _mm_unpackhi_epi32(key2_hi, key3_hi);
        
//         u0 = _mm_unpacklo_epi64(t0, t1);
//         u1 = _mm_unpackhi_epi64(t0, t1);
//         u2 = _mm_unpacklo_epi64(t2, t3);
//         u3 = _mm_unpackhi_epi64(t2, t3);
        
//         k4 = _mm512_broadcast_i32x4(u0);
//         k5 = _mm512_broadcast_i32x4(u1);
//         k6 = _mm512_broadcast_i32x4(u2);
//         k7 = _mm512_broadcast_i32x4(u3);
//     }
    
//     // Load and transpose nonces
//     __m512i n0;
//     __m512i n1;
//     __m512i n2;
//     {
//         // Load 4 nonces (12 bytes each, but load 16 and mask)
//         uint32_t nonce_words[4][3];
//         for (int i = 0; i < 4; i++) {
//             memcpy(nonce_words[i], nonces[i], 12);
//         }
        
//         // Transpose nonce words
//         __m128i nn0 = _mm_set_epi32(nonce_words[3][0], nonce_words[2][0], 
//                                     nonce_words[1][0], nonce_words[0][0]);
//         __m128i nn1 = _mm_set_epi32(nonce_words[3][1], nonce_words[2][1], 
//                                     nonce_words[1][1], nonce_words[0][1]);
//         __m128i nn2 = _mm_set_epi32(nonce_words[3][2], nonce_words[2][2], 
//                                     nonce_words[1][2], nonce_words[0][2]);
        
//         n0 = _mm512_broadcast_i32x4(nn0);
//         n1 = _mm512_broadcast_i32x4(nn1);
//         n2 = _mm512_broadcast_i32x4(nn2);
//     }
    
//     // Counter offsets for 4 blocks per stream
//     const __m512i ctr_inc = _mm512_set_epi32(
//         3, 3, 3, 3,  // 4th block offset
//         2, 2, 2, 2,  // 3rd block offset  
//         1, 1, 1, 1,  // 2nd block offset
//         0, 0, 0, 0   // 1st block offset
//     );
//     const __m512i ctr_add4 = _mm512_set1_epi32(4);
    
//     size_t iterations = bytes_per_stream / 256;  // 256 bytes = 4 blocks per stream
//     __m512i counter_base = _mm512_setzero_si512();
    
//     for (size_t iter = 0; iter < iterations; iter++) {
//         // Set up counter for this iteration
//         __m512i counter = _mm512_add_epi32(counter_base, ctr_inc);
        
//         // Initialize state
//         __m512i x0 = const0;
//         __m512i x1 = const1;
//         __m512i x2 = const2;
//         __m512i x3 = const3;
//         __m512i x4 = k0;
//         __m512i x5 = k1;
//         __m512i x6 = k2;
//         __m512i x7 = k3;
//         __m512i x8 = k4;
//         __m512i x9 = k5;
//         __m512i x10 = k6;
//         __m512i x11 = k7;
//         __m512i x12 = counter;
//         __m512i x13 = n0;
//         __m512i x14 = n1;
//         __m512i x15 = n2;
        
//         // Save initial state
//         __m512i s0  = x0;
//         __m512i s1  = x1;
//         __m512i s2  = x2;
//         __m512i s3  = x3;

//         __m512i s4  = x4;
//         __m512i s5  = x5;
//         __m512i s6  = x6;
//         __m512i s7  = x7;

//         __m512i s8  = x8;
//         __m512i s9  = x9;
//         __m512i s10 = x10;
//         __m512i s11 = x11;

//         __m512i s12 = x12;
//         __m512i s13 = x13;
//         __m512i s14 = x14;
//         __m512i s15 = x15;
        
//         // ChaCha rounds with native AVX-512 rotates
//         for (int i = rounds; i > 0; i -= 2) {
//             // Column round
//             x0 = _mm512_add_epi32(x0, x4);   x12 = _mm512_xor_si512(x12, x0);  x12 = _mm512_rol_epi32(x12, 16);
//             x8 = _mm512_add_epi32(x8, x12);  x4 = _mm512_xor_si512(x4, x8);    x4 = _mm512_rol_epi32(x4, 12);
//             x0 = _mm512_add_epi32(x0, x4);   x12 = _mm512_xor_si512(x12, x0);  x12 = _mm512_rol_epi32(x12, 8);
//             x8 = _mm512_add_epi32(x8, x12);  x4 = _mm512_xor_si512(x4, x8);    x4 = _mm512_rol_epi32(x4, 7);
            
//             x1 = _mm512_add_epi32(x1, x5);   x13 = _mm512_xor_si512(x13, x1);  x13 = _mm512_rol_epi32(x13, 16);
//             x9 = _mm512_add_epi32(x9, x13);  x5 = _mm512_xor_si512(x5, x9);    x5 = _mm512_rol_epi32(x5, 12);
//             x1 = _mm512_add_epi32(x1, x5);   x13 = _mm512_xor_si512(x13, x1);  x13 = _mm512_rol_epi32(x13, 8);
//             x9 = _mm512_add_epi32(x9, x13);  x5 = _mm512_xor_si512(x5, x9);    x5 = _mm512_rol_epi32(x5, 7);
            
//             x2 = _mm512_add_epi32(x2, x6);   x14 = _mm512_xor_si512(x14, x2);  x14 = _mm512_rol_epi32(x14, 16);
//             x10 = _mm512_add_epi32(x10, x14); x6 = _mm512_xor_si512(x6, x10);  x6 = _mm512_rol_epi32(x6, 12);
//             x2 = _mm512_add_epi32(x2, x6);   x14 = _mm512_xor_si512(x14, x2);  x14 = _mm512_rol_epi32(x14, 8);
//             x10 = _mm512_add_epi32(x10, x14); x6 = _mm512_xor_si512(x6, x10);  x6 = _mm512_rol_epi32(x6, 7);
            
//             x3 = _mm512_add_epi32(x3, x7);   x15 = _mm512_xor_si512(x15, x3);  x15 = _mm512_rol_epi32(x15, 16);
//             x11 = _mm512_add_epi32(x11, x15); x7 = _mm512_xor_si512(x7, x11);  x7 = _mm512_rol_epi32(x7, 12);
//             x3 = _mm512_add_epi32(x3, x7);   x15 = _mm512_xor_si512(x15, x3);  x15 = _mm512_rol_epi32(x15, 8);
//             x11 = _mm512_add_epi32(x11, x15); x7 = _mm512_xor_si512(x7, x11);  x7 = _mm512_rol_epi32(x7, 7);
            
//             // Diagonal round  
//             x0 = _mm512_add_epi32(x0, x5);   x15 = _mm512_xor_si512(x15, x0);  x15 = _mm512_rol_epi32(x15, 16);
//             x10 = _mm512_add_epi32(x10, x15); x5 = _mm512_xor_si512(x5, x10);  x5 = _mm512_rol_epi32(x5, 12);
//             x0 = _mm512_add_epi32(x0, x5);   x15 = _mm512_xor_si512(x15, x0);  x15 = _mm512_rol_epi32(x15, 8);
//             x10 = _mm512_add_epi32(x10, x15); x5 = _mm512_xor_si512(x5, x10);  x5 = _mm512_rol_epi32(x5, 7);
            
//             x1 = _mm512_add_epi32(x1, x6);   x12 = _mm512_xor_si512(x12, x1);  x12 = _mm512_rol_epi32(x12, 16);
//             x11 = _mm512_add_epi32(x11, x12); x6 = _mm512_xor_si512(x6, x11);  x6 = _mm512_rol_epi32(x6, 12);
//             x1 = _mm512_add_epi32(x1, x6);   x12 = _mm512_xor_si512(x12, x1);  x12 = _mm512_rol_epi32(x12, 8);
//             x11 = _mm512_add_epi32(x11, x12); x6 = _mm512_xor_si512(x6, x11);  x6 = _mm512_rol_epi32(x6, 7);
            
//             x2 = _mm512_add_epi32(x2, x7);   x13 = _mm512_xor_si512(x13, x2);  x13 = _mm512_rol_epi32(x13, 16);
//             x8 = _mm512_add_epi32(x8, x13);  x7 = _mm512_xor_si512(x7, x8);    x7 = _mm512_rol_epi32(x7, 12);
//             x2 = _mm512_add_epi32(x2, x7);   x13 = _mm512_xor_si512(x13, x2);  x13 = _mm512_rol_epi32(x13, 8);
//             x8 = _mm512_add_epi32(x8, x13);  x7 = _mm512_xor_si512(x7, x8);    x7 = _mm512_rol_epi32(x7, 7);
            
//             x3 = _mm512_add_epi32(x3, x4);   x14 = _mm512_xor_si512(x14, x3);  x14 = _mm512_rol_epi32(x14, 16);
//             x9 = _mm512_add_epi32(x9, x14);  x4 = _mm512_xor_si512(x4, x9);    x4 = _mm512_rol_epi32(x4, 12);
//             x3 = _mm512_add_epi32(x3, x4);   x14 = _mm512_xor_si512(x14, x3);  x14 = _mm512_rol_epi32(x14, 8);
//             x9 = _mm512_add_epi32(x9, x14);  x4 = _mm512_xor_si512(x4, x9);    x4 = _mm512_rol_epi32(x4, 7);
//         }
        
//         // Add initial state
//         x0 = _mm512_add_epi32(x0, s0);   x1 = _mm512_add_epi32(x1, s1);
//         x2 = _mm512_add_epi32(x2, s2);   x3 = _mm512_add_epi32(x3, s3);
//         x4 = _mm512_add_epi32(x4, s4);   x5 = _mm512_add_epi32(x5, s5);
//         x6 = _mm512_add_epi32(x6, s6);   x7 = _mm512_add_epi32(x7, s7);
//         x8 = _mm512_add_epi32(x8, s8);   x9 = _mm512_add_epi32(x9, s9);
//         x10 = _mm512_add_epi32(x10, s10); x11 = _mm512_add_epi32(x11, s11);
//         x12 = _mm512_add_epi32(x12, s12); x13 = _mm512_add_epi32(x13, s13);
//         x14 = _mm512_add_epi32(x14, s14); x15 = _mm512_add_epi32(x15, s15);
        
//         // Transpose and store
//         const __m512i idx_s0 = _mm512_setr_epi32(0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12);
//         const __m512i idx_s1 = _mm512_setr_epi32(1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13);
//         const __m512i idx_s2 = _mm512_setr_epi32(2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14);
//         const __m512i idx_s3 = _mm512_setr_epi32(3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15);
        
//         // Stream 0, Block 0
//         __m128i s0_b0_w0_3 = _mm512_castsi512_si128(_mm512_permutexvar_epi32(idx_s0, x0));
//         __m128i s0_b0_w4_7 = _mm512_castsi512_si128(_mm512_permutexvar_epi32(idx_s0, x4));
//         __m128i s0_b0_w8_11 = _mm512_castsi512_si128(_mm512_permutexvar_epi32(idx_s0, x8));
//         __m128i s0_b0_w12_15 = _mm512_castsi512_si128(_mm512_permutexvar_epi32(idx_s0, x12));
        
//         _mm_storeu_si128((__m128i*)(outputs[0]), s0_b0_w0_3);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 16), s0_b0_w4_7);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 32), s0_b0_w8_11);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 48), s0_b0_w12_15);
        
//         // Stream 0, Block 1
//         __m128i s0_b1_w0_3 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x0), 1);
//         __m128i s0_b1_w4_7 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x4), 1);
//         __m128i s0_b1_w8_11 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x8), 1);
//         __m128i s0_b1_w12_15 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x12), 1);
        
//         _mm_storeu_si128((__m128i*)(outputs[0] + 64), s0_b1_w0_3);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 80), s0_b1_w4_7);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 96), s0_b1_w8_11);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 112), s0_b1_w12_15);
        
//         // Stream 0, Blocks 2–3
//         __m128i s0_b2_w0_3 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x0), 2);
//         __m128i s0_b2_w4_7 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x4), 2);
//         __m128i s0_b2_w8_11 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x8), 2);
//         __m128i s0_b2_w12_15 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x12), 2);
        
//         _mm_storeu_si128((__m128i*)(outputs[0] + 128), s0_b2_w0_3);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 144), s0_b2_w4_7);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 160), s0_b2_w8_11);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 176), s0_b2_w12_15);
        
//         __m128i s0_b3_w0_3 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x0), 3);
//         __m128i s0_b3_w4_7 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x4), 3);
//         __m128i s0_b3_w8_11 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x8), 3);
//         __m128i s0_b3_w12_15 = _mm512_extracti32x4_epi32(_mm512_permutexvar_epi32(idx_s0, x12), 3);
        
//         _mm_storeu_si128((__m128i*)(outputs[0] + 192), s0_b3_w0_3);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 208), s0_b3_w4_7);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 224), s0_b3_w8_11);
//         _mm_storeu_si128((__m128i*)(outputs[0] + 240), s0_b3_w12_15);
        
//         // Streams 1–3
//         STORE_STREAM(1, idx_s1);
//         STORE_STREAM(2, idx_s2);
//         STORE_STREAM(3, idx_s3);        

//         // Advance output pointers (256 bytes = 4 blocks per stream)
//         outputs[0] += 256;
//         outputs[1] += 256;
//         outputs[2] += 256;
//         outputs[3] += 256;
        
//         counter_base = _mm512_add_epi32(counter_base, ctr_add4);
//     }
//   },
//   TNN_TARGETS_X86_CHACHA512
// )
#undef STORE_STREAM

#endif