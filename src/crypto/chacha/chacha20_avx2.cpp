#include "chacha20.h"

#if defined(__x86_64__)

#include <immintrin.h>
#include <memory.h>

#define PartialXor(val, Src, Dest, Size)                                      \
    {                                                                         \
        alignas(32) uint8_t BuffForPartialOp[32];                             \
        memcpy(BuffForPartialOp, (Src), (Size));                              \
        _mm256_store_si256((__m256i*)(BuffForPartialOp),                      \
            _mm256_xor_si256((val),                                           \
            _mm256_loadu_si256((const __m256i*)BuffForPartialOp)));          \
        memcpy((Dest), BuffForPartialOp, (Size));                             \
    }

#define PartialStore(val, Dest, Size)                                         \
    {                                                                         \
        alignas(32) uint8_t BuffForPartialOp[32];                             \
        _mm256_store_si256((__m256i*)(BuffForPartialOp), (val));             \
        memcpy((Dest), BuffForPartialOp, (Size));                             \
    }

#define RotateLeft7(val)                                                      \
    _mm256_or_si256(_mm256_slli_epi32((val), 7), _mm256_srli_epi32((val), 25))

#define RotateLeft12(val)                                                     \
    _mm256_or_si256(_mm256_slli_epi32((val), 12), _mm256_srli_epi32((val), 20))

#define RotateLeft8(val)                                                      \
    _mm256_shuffle_epi8((val),                                                \
        _mm256_set_epi8(14, 13, 12, 15, 10, 9, 8, 11,                          \
                        6, 5, 4, 7, 2, 1, 0, 3,                                \
                        14, 13, 12, 15, 10, 9, 8, 11,                          \
                        6, 5, 4, 7, 2, 1, 0, 3))

#define RotateLeft16(val)                                                    \
    _mm256_shuffle_epi8((val),                                               \
        _mm256_set_epi8(13, 12, 15, 14, 9, 8, 11, 10,                         \
                        5, 4, 7, 6, 1, 0, 3, 2,                               \
                        13, 12, 15, 14, 9, 8, 11, 10,                         \
                        5, 4, 7, 6, 1, 0, 3, 2))

TNN_TARGET_CLONE(
  ChaCha20EncryptBytes,
  void,
  (uint8_t* state, uint8_t* In, uint8_t* Out, uint64_t Size, int rounds),
  {
    uint8_t* CurrentIn = In;
    uint8_t* CurrentOut = Out;
    
    uint64_t FullBlocksCount = Size / 512;
    uint64_t RemainingBytes = Size % 512;

    const __m256i state0 = _mm256_broadcastsi128_si256(_mm_set_epi32(1797285236, 2036477234, 857760878, 1634760805)); //"expand 32-byte k"
    const __m256i state1 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)(state)));
    const __m256i state2 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)(state + 16)));

    __m256i CTR0 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 4);
    const __m256i CTR1 = _mm256_set_epi32(0, 0, 0, 1, 0, 0, 0, 5);
    const __m256i CTR2 = _mm256_set_epi32(0, 0, 0, 2, 0, 0, 0, 6);
    const __m256i CTR3 = _mm256_set_epi32(0, 0, 0, 3, 0, 0, 0, 7);

    for (int64_t n = 0; n < FullBlocksCount; n++)
    {

      const __m256i state3 = _mm256_broadcastsi128_si256(
        _mm_load_si128((const __m128i*)(state + 32)));

      __m256i X0_0 = state0;
      __m256i X0_1 = state1;
      __m256i X0_2 = state2;
      __m256i X0_3 = _mm256_add_epi32(state3, CTR0);

      __m256i X1_0 = state0;
      __m256i X1_1 = state1;
      __m256i X1_2 = state2;
      __m256i X1_3 = _mm256_add_epi32(state3, CTR1);

      __m256i X2_0 = state0;
      __m256i X2_1 = state1;
      __m256i X2_2 = state2;
      __m256i X2_3 = _mm256_add_epi32(state3, CTR2);

      __m256i X3_0 = state0;
      __m256i X3_1 = state1;
      __m256i X3_2 = state2;
      __m256i X3_3 = _mm256_add_epi32(state3, CTR3);

      for (int i = rounds; i > 0; i -= 2)
      {
        X0_0 = _mm256_add_epi32(X0_0, X0_1);
        X1_0 = _mm256_add_epi32(X1_0, X1_1);
        X2_0 = _mm256_add_epi32(X2_0, X2_1);
        X3_0 = _mm256_add_epi32(X3_0, X3_1);

        X0_3 = _mm256_xor_si256(X0_3, X0_0);
        X1_3 = _mm256_xor_si256(X1_3, X1_0);
        X2_3 = _mm256_xor_si256(X2_3, X2_0);
        X3_3 = _mm256_xor_si256(X3_3, X3_0);

        X0_3 = RotateLeft16(X0_3);
        X1_3 = RotateLeft16(X1_3);
        X2_3 = RotateLeft16(X2_3);
        X3_3 = RotateLeft16(X3_3);

        X0_2 = _mm256_add_epi32(X0_2, X0_3);
        X1_2 = _mm256_add_epi32(X1_2, X1_3);
        X2_2 = _mm256_add_epi32(X2_2, X2_3);
        X3_2 = _mm256_add_epi32(X3_2, X3_3);

        X0_1 = _mm256_xor_si256(X0_1, X0_2);
        X1_1 = _mm256_xor_si256(X1_1, X1_2);
        X2_1 = _mm256_xor_si256(X2_1, X2_2);
        X3_1 = _mm256_xor_si256(X3_1, X3_2);

        X0_1 = RotateLeft12(X0_1);
        X1_1 = RotateLeft12(X1_1);
        X2_1 = RotateLeft12(X2_1);
        X3_1 = RotateLeft12(X3_1);

        X0_0 = _mm256_add_epi32(X0_0, X0_1);
        X1_0 = _mm256_add_epi32(X1_0, X1_1);
        X2_0 = _mm256_add_epi32(X2_0, X2_1);
        X3_0 = _mm256_add_epi32(X3_0, X3_1);

        X0_3 = _mm256_xor_si256(X0_3, X0_0);
        X1_3 = _mm256_xor_si256(X1_3, X1_0);
        X2_3 = _mm256_xor_si256(X2_3, X2_0);
        X3_3 = _mm256_xor_si256(X3_3, X3_0);

        X0_3 = RotateLeft8(X0_3);
        X1_3 = RotateLeft8(X1_3);
        X2_3 = RotateLeft8(X2_3);
        X3_3 = RotateLeft8(X3_3);

        X0_2 = _mm256_add_epi32(X0_2, X0_3);
        X1_2 = _mm256_add_epi32(X1_2, X1_3);
        X2_2 = _mm256_add_epi32(X2_2, X2_3);
        X3_2 = _mm256_add_epi32(X3_2, X3_3);

        X0_1 = _mm256_xor_si256(X0_1, X0_2);
        X1_1 = _mm256_xor_si256(X1_1, X1_2);
        X2_1 = _mm256_xor_si256(X2_1, X2_2);
        X3_1 = _mm256_xor_si256(X3_1, X3_2);

        X0_1 = RotateLeft7(X0_1);
        X1_1 = RotateLeft7(X1_1);
        X2_1 = RotateLeft7(X2_1);
        X3_1 = RotateLeft7(X3_1);

        X0_1 = _mm256_shuffle_epi32(X0_1, _MM_SHUFFLE(0, 3, 2, 1));
        X0_2 = _mm256_shuffle_epi32(X0_2, _MM_SHUFFLE(1, 0, 3, 2));
        X0_3 = _mm256_shuffle_epi32(X0_3, _MM_SHUFFLE(2, 1, 0, 3));

        X1_1 = _mm256_shuffle_epi32(X1_1, _MM_SHUFFLE(0, 3, 2, 1));
        X1_2 = _mm256_shuffle_epi32(X1_2, _MM_SHUFFLE(1, 0, 3, 2));
        X1_3 = _mm256_shuffle_epi32(X1_3, _MM_SHUFFLE(2, 1, 0, 3));

        X2_1 = _mm256_shuffle_epi32(X2_1, _MM_SHUFFLE(0, 3, 2, 1));
        X2_2 = _mm256_shuffle_epi32(X2_2, _MM_SHUFFLE(1, 0, 3, 2));
        X2_3 = _mm256_shuffle_epi32(X2_3, _MM_SHUFFLE(2, 1, 0, 3));

        X3_1 = _mm256_shuffle_epi32(X3_1, _MM_SHUFFLE(0, 3, 2, 1));
        X3_2 = _mm256_shuffle_epi32(X3_2, _MM_SHUFFLE(1, 0, 3, 2));
        X3_3 = _mm256_shuffle_epi32(X3_3, _MM_SHUFFLE(2, 1, 0, 3));

        X0_0 = _mm256_add_epi32(X0_0, X0_1);
        X1_0 = _mm256_add_epi32(X1_0, X1_1);
        X2_0 = _mm256_add_epi32(X2_0, X2_1);
        X3_0 = _mm256_add_epi32(X3_0, X3_1);

        X0_3 = _mm256_xor_si256(X0_3, X0_0);
        X1_3 = _mm256_xor_si256(X1_3, X1_0);
        X2_3 = _mm256_xor_si256(X2_3, X2_0);
        X3_3 = _mm256_xor_si256(X3_3, X3_0);

        X0_3 = RotateLeft16(X0_3);
        X1_3 = RotateLeft16(X1_3);
        X2_3 = RotateLeft16(X2_3);
        X3_3 = RotateLeft16(X3_3);

        X0_2 = _mm256_add_epi32(X0_2, X0_3);
        X1_2 = _mm256_add_epi32(X1_2, X1_3);
        X2_2 = _mm256_add_epi32(X2_2, X2_3);
        X3_2 = _mm256_add_epi32(X3_2, X3_3);

        X0_1 = _mm256_xor_si256(X0_1, X0_2);
        X1_1 = _mm256_xor_si256(X1_1, X1_2);
        X2_1 = _mm256_xor_si256(X2_1, X2_2);
        X3_1 = _mm256_xor_si256(X3_1, X3_2);

        X0_1 = RotateLeft12(X0_1);
        X1_1 = RotateLeft12(X1_1);
        X2_1 = RotateLeft12(X2_1);
        X3_1 = RotateLeft12(X3_1);

        X0_0 = _mm256_add_epi32(X0_0, X0_1);
        X1_0 = _mm256_add_epi32(X1_0, X1_1);
        X2_0 = _mm256_add_epi32(X2_0, X2_1);
        X3_0 = _mm256_add_epi32(X3_0, X3_1);

        X0_3 = _mm256_xor_si256(X0_3, X0_0);
        X1_3 = _mm256_xor_si256(X1_3, X1_0);
        X2_3 = _mm256_xor_si256(X2_3, X2_0);
        X3_3 = _mm256_xor_si256(X3_3, X3_0);

        X0_3 = RotateLeft8(X0_3);
        X1_3 = RotateLeft8(X1_3);
        X2_3 = RotateLeft8(X2_3);
        X3_3 = RotateLeft8(X3_3);

        X0_2 = _mm256_add_epi32(X0_2, X0_3);
        X1_2 = _mm256_add_epi32(X1_2, X1_3);
        X2_2 = _mm256_add_epi32(X2_2, X2_3);
        X3_2 = _mm256_add_epi32(X3_2, X3_3);

        X0_1 = _mm256_xor_si256(X0_1, X0_2);
        X1_1 = _mm256_xor_si256(X1_1, X1_2);
        X2_1 = _mm256_xor_si256(X2_1, X2_2);
        X3_1 = _mm256_xor_si256(X3_1, X3_2);

        X0_1 = RotateLeft7(X0_1);
        X1_1 = RotateLeft7(X1_1);
        X2_1 = RotateLeft7(X2_1);
        X3_1 = RotateLeft7(X3_1);

        X0_1 = _mm256_shuffle_epi32(X0_1, _MM_SHUFFLE(2, 1, 0, 3));
        X0_2 = _mm256_shuffle_epi32(X0_2, _MM_SHUFFLE(1, 0, 3, 2));
        X0_3 = _mm256_shuffle_epi32(X0_3, _MM_SHUFFLE(0, 3, 2, 1));

        X1_1 = _mm256_shuffle_epi32(X1_1, _MM_SHUFFLE(2, 1, 0, 3));
        X1_2 = _mm256_shuffle_epi32(X1_2, _MM_SHUFFLE(1, 0, 3, 2));
        X1_3 = _mm256_shuffle_epi32(X1_3, _MM_SHUFFLE(0, 3, 2, 1));

        X2_1 = _mm256_shuffle_epi32(X2_1, _MM_SHUFFLE(2, 1, 0, 3));
        X2_2 = _mm256_shuffle_epi32(X2_2, _MM_SHUFFLE(1, 0, 3, 2));
        X2_3 = _mm256_shuffle_epi32(X2_3, _MM_SHUFFLE(0, 3, 2, 1));

        X3_1 = _mm256_shuffle_epi32(X3_1, _MM_SHUFFLE(2, 1, 0, 3));
        X3_2 = _mm256_shuffle_epi32(X3_2, _MM_SHUFFLE(1, 0, 3, 2));
        X3_3 = _mm256_shuffle_epi32(X3_3, _MM_SHUFFLE(0, 3, 2, 1));
      }

      X0_0 = _mm256_add_epi32(X0_0, state0);
      X0_1 = _mm256_add_epi32(X0_1, state1);
      X0_2 = _mm256_add_epi32(X0_2, state2);
      X0_3 = _mm256_add_epi32(X0_3, state3);
      X0_3 = _mm256_add_epi32(X0_3, CTR0);

      X1_0 = _mm256_add_epi32(X1_0, state0);
      X1_1 = _mm256_add_epi32(X1_1, state1);
      X1_2 = _mm256_add_epi32(X1_2, state2);
      X1_3 = _mm256_add_epi32(X1_3, state3);
      X1_3 = _mm256_add_epi32(X1_3, CTR1);

      X2_0 = _mm256_add_epi32(X2_0, state0);
      X2_1 = _mm256_add_epi32(X2_1, state1);
      X2_2 = _mm256_add_epi32(X2_2, state2);
      X2_3 = _mm256_add_epi32(X2_3, state3);
      X2_3 = _mm256_add_epi32(X2_3, CTR2);

      X3_0 = _mm256_add_epi32(X3_0, state0);
      X3_1 = _mm256_add_epi32(X3_1, state1);
      X3_2 = _mm256_add_epi32(X3_2, state2);
      X3_3 = _mm256_add_epi32(X3_3, state3);
      X3_3 = _mm256_add_epi32(X3_3, CTR3);

      //


      if (In)
      {
        _mm256_storeu_si256((__m256i*)(CurrentOut + 0 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X0_0, X0_1, 1 + (3 << 4)),
            _mm256_loadu_si256((__m256i*)(CurrentIn + 0 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 1 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X0_2, X0_3, 1 + (3 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 1 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 2 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X1_0, X1_1, 1 + (3 << 4)),
            _mm256_loadu_si256(((const __m256i*)(CurrentIn + 2 * 32)))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 3 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X1_2, X1_3, 1 + (3 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 3 * 32))));

        _mm256_storeu_si256((__m256i*)(CurrentOut + 4 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X2_0, X2_1, 1 + (3 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 4 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 5 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X2_2, X2_3, 1 + (3 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 5 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 6 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X3_0, X3_1, 1 + (3 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 6 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 7 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X3_2, X3_3, 1 + (3 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 7 * 32))));

        _mm256_storeu_si256(
          (__m256i*)(CurrentOut + 8 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X0_0, X0_1, 0 + (2 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 8 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 9 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X0_2, X0_3, 0 + (2 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 9 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 10 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X1_0, X1_1, 0 + (2 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 10 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 11 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X1_2, X1_3, 0 + (2 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 11 * 32))));

        _mm256_storeu_si256((__m256i*)(CurrentOut + 12 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X2_0, X2_1, 0 + (2 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 12 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 13 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X2_2, X2_3, 0 + (2 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 13 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 14 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X3_0, X3_1, 0 + (2 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 14 * 32))));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 15 * 32),
          _mm256_xor_si256(_mm256_permute2x128_si256(X3_2, X3_3, 0 + (2 << 4)),
            _mm256_loadu_si256((const __m256i*)(CurrentIn + 15 * 32))));
      }
      else
      {
        _mm256_storeu_si256((__m256i*)(CurrentOut + 0 * 32),
          _mm256_permute2x128_si256(X0_0, X0_1, 1 + (3 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 1 * 32),
          _mm256_permute2x128_si256(X0_2, X0_3, 1 + (3 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 2 * 32),
          _mm256_permute2x128_si256(X1_0, X1_1, 1 + (3 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 3 * 32),
          _mm256_permute2x128_si256(X1_2, X1_3, 1 + (3 << 4)));

        _mm256_storeu_si256((__m256i*)(CurrentOut + 4 * 32),
          _mm256_permute2x128_si256(X2_0, X2_1, 1 + (3 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 5 * 32),
          _mm256_permute2x128_si256(X2_2, X2_3, 1 + (3 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 6 * 32),
          _mm256_permute2x128_si256(X3_0, X3_1, 1 + (3 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 7 * 32),
          _mm256_permute2x128_si256(X3_2, X3_3, 1 + (3 << 4)));

        _mm256_storeu_si256((__m256i*)(CurrentOut + 8 * 32),
          _mm256_permute2x128_si256(X0_0, X0_1, 0 + (2 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 9 * 32),
          _mm256_permute2x128_si256(X0_2, X0_3, 0 + (2 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 10 * 32),
          _mm256_permute2x128_si256(X1_0, X1_1, 0 + (2 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 11 * 32),
          _mm256_permute2x128_si256(X1_2, X1_3, 0 + (2 << 4)));

        _mm256_storeu_si256((__m256i*)(CurrentOut + 12 * 32),
          _mm256_permute2x128_si256(X2_0, X2_1, 0 + (2 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 13 * 32),
          _mm256_permute2x128_si256(X2_2, X2_3, 0 + (2 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 14 * 32),
          _mm256_permute2x128_si256(X3_0, X3_1, 0 + (2 << 4)));
        _mm256_storeu_si256((__m256i*)(CurrentOut + 15 * 32),
          _mm256_permute2x128_si256(X3_2, X3_3, 0 + (2 << 4)));
      }

      ChaCha20AddCounter(state, 8);
      if (CurrentIn)
        CurrentIn += 512;
      CurrentOut += 512;

    }

    if (RemainingBytes == 0) return;

    CTR0 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 1);

    while (1)
    {

      const __m256i state3 = _mm256_broadcastsi128_si256(
        _mm_load_si128((const __m128i*)(state + 32)));

      __m256i X0_0 = state0;
      __m256i X0_1 = state1;
      __m256i X0_2 = state2;
      __m256i X0_3 = _mm256_add_epi32(state3, CTR0);


      for (int i = rounds; i > 0; i -= 2)
      {
        X0_0 = _mm256_add_epi32(X0_0, X0_1);


        X0_3 = _mm256_xor_si256(X0_3, X0_0);

        X0_3 = RotateLeft16(X0_3);

        X0_2 = _mm256_add_epi32(X0_2, X0_3);


        X0_1 = _mm256_xor_si256(X0_1, X0_2);

        X0_1 = RotateLeft12(X0_1);

        X0_0 = _mm256_add_epi32(X0_0, X0_1);

        X0_3 = _mm256_xor_si256(X0_3, X0_0);

        X0_3 = RotateLeft8(X0_3);

        X0_2 = _mm256_add_epi32(X0_2, X0_3);


        X0_1 = _mm256_xor_si256(X0_1, X0_2);

        X0_1 = RotateLeft7(X0_1);

        X0_1 = _mm256_shuffle_epi32(X0_1, _MM_SHUFFLE(0, 3, 2, 1));
        X0_2 = _mm256_shuffle_epi32(X0_2, _MM_SHUFFLE(1, 0, 3, 2));
        X0_3 = _mm256_shuffle_epi32(X0_3, _MM_SHUFFLE(2, 1, 0, 3));


        X0_0 = _mm256_add_epi32(X0_0, X0_1);

        X0_3 = _mm256_xor_si256(X0_3, X0_0);

        X0_3 = RotateLeft16(X0_3);

        X0_2 = _mm256_add_epi32(X0_2, X0_3);

        X0_1 = _mm256_xor_si256(X0_1, X0_2);

        X0_1 = RotateLeft12(X0_1);

        X0_0 = _mm256_add_epi32(X0_0, X0_1);

        X0_3 = _mm256_xor_si256(X0_3, X0_0);

        X0_3 = RotateLeft8(X0_3);

        X0_2 = _mm256_add_epi32(X0_2, X0_3);

        X0_1 = _mm256_xor_si256(X0_1, X0_2);

        X0_1 = RotateLeft7(X0_1);

        X0_1 = _mm256_shuffle_epi32(X0_1, _MM_SHUFFLE(2, 1, 0, 3));
        X0_2 = _mm256_shuffle_epi32(X0_2, _MM_SHUFFLE(1, 0, 3, 2));
        X0_3 = _mm256_shuffle_epi32(X0_3, _MM_SHUFFLE(0, 3, 2, 1));


      }

      X0_0 = _mm256_add_epi32(X0_0, state0);
      X0_1 = _mm256_add_epi32(X0_1, state1);
      X0_2 = _mm256_add_epi32(X0_2, state2);
      X0_3 = _mm256_add_epi32(X0_3, state3);
      X0_3 = _mm256_add_epi32(X0_3, CTR0);

      //todo

      if (RemainingBytes >= 128)
      {
        if (In)
        {
          _mm256_storeu_si256((__m256i*)(CurrentOut + 0 * 32),
            _mm256_xor_si256(_mm256_permute2x128_si256(X0_0, X0_1, 1 + (3 << 4)),
              _mm256_loadu_si256((__m256i*)(CurrentIn + 0 * 32))));
          _mm256_storeu_si256((__m256i*)(CurrentOut + 1 * 32),
            _mm256_xor_si256(_mm256_permute2x128_si256(X0_2, X0_3, 1 + (3 << 4)),
              _mm256_loadu_si256((const __m256i*)(CurrentIn + 1 * 32))));
          _mm256_storeu_si256((__m256i*)(CurrentOut + 2 * 32),
            _mm256_xor_si256(_mm256_permute2x128_si256(X0_0, X0_1, 0 + (2 << 4)),
              _mm256_loadu_si256((const __m256i*)(CurrentIn + 2 * 32))));
          _mm256_storeu_si256((__m256i*)(CurrentOut + 3 * 32),
            _mm256_xor_si256(_mm256_permute2x128_si256(X0_2, X0_3, 0 + (2 << 4)),
              _mm256_loadu_si256((const __m256i*)(CurrentIn + 3 * 32))));

        }
        else
        {
          _mm256_storeu_si256((__m256i*)(CurrentOut + 0 * 32),
            _mm256_permute2x128_si256(X0_0, X0_1, 1 + (3 << 4)));
          _mm256_storeu_si256((__m256i*)(CurrentOut + 1 * 32),
            _mm256_permute2x128_si256(X0_2, X0_3, 1 + (3 << 4)));
          _mm256_storeu_si256((__m256i*)(CurrentOut + 2 * 32),
            _mm256_permute2x128_si256(X0_0, X0_1, 0 + (2 << 4)));
          _mm256_storeu_si256((__m256i*)(CurrentOut + 3 * 32),
            _mm256_permute2x128_si256(X0_2, X0_3, 0 + (2 << 4)));

        }
        ChaCha20AddCounter(state, 2);
        RemainingBytes -= 128;
        if (RemainingBytes == 0) return;
        if (CurrentIn)
          CurrentIn += 128;
        CurrentOut += 128;
        continue;
      }
      else //last, partial block
      {
        __m256i tmp;
        if (In) // encrypt
        {
          tmp = _mm256_permute2x128_si256(X0_0, X0_1, 1 + (3 << 4));
          if (RemainingBytes < 32)
          {
            PartialXor(tmp, CurrentIn, CurrentOut, RemainingBytes);
            ChaCha20AddCounter(state, 1);
            return;
          }
          _mm256_storeu_si256((__m256i*)(CurrentOut), _mm256_xor_si256(tmp, _mm256_loadu_si256((const __m256i*)(CurrentIn))));
          RemainingBytes -= 32;
          if (RemainingBytes == 0)
          {
            ChaCha20AddCounter(state, 1);
            return;
          }

          CurrentIn += 32;
          CurrentOut += 32;



          tmp = _mm256_permute2x128_si256(X0_2, X0_3, 1 + (3 << 4));
          if (RemainingBytes < 32)
          {
            PartialXor(tmp, CurrentIn, CurrentOut, RemainingBytes);
            ChaCha20AddCounter(state, 1);
            return;
          }
          _mm256_storeu_si256((__m256i*)(CurrentOut), _mm256_xor_si256(tmp, _mm256_loadu_si256((const __m256i*)(CurrentIn))));
          RemainingBytes -= 32;
          if (RemainingBytes == 0)
          {
            ChaCha20AddCounter(state, 1);
            return;
          }
          CurrentIn += 32;
          CurrentOut += 32;


          tmp = _mm256_permute2x128_si256(X0_0, X0_1, 0 + (2 << 4));
          if (RemainingBytes < 32)
          {
            PartialXor(tmp, CurrentIn, CurrentOut, RemainingBytes);
            ChaCha20AddCounter(state, 2);
            return;
          }
          _mm256_storeu_si256((__m256i*)(CurrentOut), _mm256_xor_si256(tmp, _mm256_loadu_si256((const __m256i*)(CurrentIn))));
          RemainingBytes -= 32;
          if (RemainingBytes == 0)
          {
            ChaCha20AddCounter(state, 2);
            return;
          }
          CurrentIn += 32;
          CurrentOut += 32;


          tmp = _mm256_permute2x128_si256(X0_2, X0_3, 0 + (2 << 4));
          PartialXor(tmp, CurrentIn, CurrentOut, RemainingBytes);
          ChaCha20AddCounter(state, 2);
          return;
        }
        else
        {

          tmp = _mm256_permute2x128_si256(X0_0, X0_1, 1 + (3 << 4));
          if (RemainingBytes < 32)
          {
            PartialStore(tmp, CurrentOut, RemainingBytes);
            ChaCha20AddCounter(state, 1);
            return;
          }
          _mm256_storeu_si256((__m256i*)(CurrentOut), tmp);
          RemainingBytes -= 32;
          if (RemainingBytes == 0)
          {
            ChaCha20AddCounter(state, 1);
            return;
          }
          CurrentOut += 32;


          tmp = _mm256_permute2x128_si256(X0_2, X0_3, 1 + (3 << 4));

          if (RemainingBytes < 32)
          {
            PartialStore(tmp, CurrentOut, RemainingBytes);
            ChaCha20AddCounter(state, 1);
            return;
          }
          _mm256_storeu_si256((__m256i*)(CurrentOut), tmp);
          RemainingBytes -= 32;
          if (RemainingBytes == 0)
          {
            ChaCha20AddCounter(state, 1);
            return;
          }
          CurrentOut += 32;


          tmp = _mm256_permute2x128_si256(X0_0, X0_1, 0 + (2 << 4));
          if (RemainingBytes < 32)
          {
            PartialStore(tmp, CurrentOut, RemainingBytes);
            ChaCha20AddCounter(state, 2);
            return;
          }
          _mm256_storeu_si256((__m256i*)(CurrentOut), tmp);
          RemainingBytes -= 32;
          if (RemainingBytes == 0)
          {
            ChaCha20AddCounter(state, 2);
            return;
          }
          CurrentOut += 32;


          tmp = _mm256_permute2x128_si256(X0_2, X0_3, 0 + (2 << 4));
          PartialStore(tmp, CurrentOut, RemainingBytes);
          ChaCha20AddCounter(state, 2);
          return;
        }
      }
    }
  },
  TNN_TARGETS_X86_AVX2
)

__attribute__((target("avx2")))
void ChaCha20EncryptXelis(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    size_t bytes_per_stream,
    int rounds)
{
    // Constants
    const __m256i const0 = _mm256_set1_epi32(0x61707865);
    const __m256i const1 = _mm256_set1_epi32(0x3320646e);
    const __m256i const2 = _mm256_set1_epi32(0x79622d32);
    const __m256i const3 = _mm256_set1_epi32(0x6b206574);
    
    // Load and transpose keys using SIMD
    __m256i k0, k1, k2, k3, k4, k5, k6, k7;
    {
        __m128i key0_lo = _mm_loadu_si128((const __m128i*)keys[0]);
        __m128i key1_lo = _mm_loadu_si128((const __m128i*)keys[1]);
        __m128i key2_lo = _mm_loadu_si128((const __m128i*)keys[2]);
        __m128i key3_lo = _mm_loadu_si128((const __m128i*)keys[3]);
        
        __m128i key0_hi = _mm_loadu_si128((const __m128i*)(keys[0] + 16));
        __m128i key1_hi = _mm_loadu_si128((const __m128i*)(keys[1] + 16));
        __m128i key2_hi = _mm_loadu_si128((const __m128i*)(keys[2] + 16));
        __m128i key3_hi = _mm_loadu_si128((const __m128i*)(keys[3] + 16));
        
        // Transpose first 4 words
        __m128i t0 = _mm_unpacklo_epi32(key0_lo, key1_lo);
        __m128i t1 = _mm_unpacklo_epi32(key2_lo, key3_lo);
        __m128i t2 = _mm_unpackhi_epi32(key0_lo, key1_lo);
        __m128i t3 = _mm_unpackhi_epi32(key2_lo, key3_lo);
        
        __m128i s0 = _mm_unpacklo_epi64(t0, t1);
        __m128i s1 = _mm_unpackhi_epi64(t0, t1);
        __m128i s2 = _mm_unpacklo_epi64(t2, t3);
        __m128i s3 = _mm_unpackhi_epi64(t2, t3);
        
        // Transpose second 4 words
        t0 = _mm_unpacklo_epi32(key0_hi, key1_hi);
        t1 = _mm_unpacklo_epi32(key2_hi, key3_hi);
        t2 = _mm_unpackhi_epi32(key0_hi, key1_hi);
        t3 = _mm_unpackhi_epi32(key2_hi, key3_hi);
        
        __m128i s4 = _mm_unpacklo_epi64(t0, t1);
        __m128i s5 = _mm_unpackhi_epi64(t0, t1);
        __m128i s6 = _mm_unpacklo_epi64(t2, t3);
        __m128i s7 = _mm_unpackhi_epi64(t2, t3);
        
        // Duplicate to both lanes
        k0 = _mm256_insertf128_si256(_mm256_castsi128_si256(s0), s0, 1);
        k1 = _mm256_insertf128_si256(_mm256_castsi128_si256(s1), s1, 1);
        k2 = _mm256_insertf128_si256(_mm256_castsi128_si256(s2), s2, 1);
        k3 = _mm256_insertf128_si256(_mm256_castsi128_si256(s3), s3, 1);
        k4 = _mm256_insertf128_si256(_mm256_castsi128_si256(s4), s4, 1);
        k5 = _mm256_insertf128_si256(_mm256_castsi128_si256(s5), s5, 1);
        k6 = _mm256_insertf128_si256(_mm256_castsi128_si256(s6), s6, 1);
        k7 = _mm256_insertf128_si256(_mm256_castsi128_si256(s7), s7, 1);
    }
    
    // Load and transpose nonces
    __m256i n0, n1, n2;
    {
        __m256i nonces01 = _mm256_loadu2_m128i((const __m128i*)nonces[1], (const __m128i*)nonces[0]);
        __m256i nonces23 = _mm256_loadu2_m128i((const __m128i*)nonces[3], (const __m128i*)nonces[2]);
        
        // Mask the 4th dword in each nonce using 256-bit operation
        const __m256i mask = _mm256_set_epi32(0, -1, -1, -1, 0, -1, -1, -1);
        nonces01 = _mm256_and_si256(nonces01, mask);
        nonces23 = _mm256_and_si256(nonces23, mask);
        
        // Rearrange using cross-lane permutes
        __m256i nonces02 = _mm256_permute2x128_si256(nonces01, nonces23, 0x20);
        __m256i nonces13 = _mm256_permute2x128_si256(nonces01, nonces23, 0x31);
        
        // Transpose using 256-bit unpacks
        __m256i t0 = _mm256_unpacklo_epi32(nonces02, nonces13);
        __m256i t1 = _mm256_unpackhi_epi32(nonces02, nonces13);
        
        // Use vpermd to get final arrangement and broadcast in one operation
        const __m256i idx_lo = _mm256_setr_epi32(0, 1, 4, 5, 0, 1, 4, 5);
        const __m256i idx_hi = _mm256_setr_epi32(2, 3, 6, 7, 2, 3, 6, 7);
        
        n0 = _mm256_permutevar8x32_epi32(t0, idx_lo);
        n1 = _mm256_permutevar8x32_epi32(t0, idx_hi);
        n2 = _mm256_permutevar8x32_epi32(t1, idx_lo);
    }
    
    // Process exactly 429 iterations (858 blocks total, 2 per iteration)
    uint32_t counter_base = 0;
    
    // Xelis: exactly 429 iterations, no partial blocks
    for (int iter = 0; iter < 858; iter++) {
        // Set up counter - block 0 and block 1 for each stream
        __m256i counter = _mm256_add_epi32(
            _mm256_set1_epi32(counter_base),
            _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0)
        );
        
        // Initialize state
        __m256i x0 = const0;
        __m256i x1 = const1;
        __m256i x2 = const2;
        __m256i x3 = const3;
        __m256i x4 = k0;
        __m256i x5 = k1;
        __m256i x6 = k2;
        __m256i x7 = k3;
        __m256i x8 = k4;
        __m256i x9 = k5;
        __m256i x10 = k6;
        __m256i x11 = k7;
        __m256i x12 = counter;
        __m256i x13 = n0;
        __m256i x14 = n1;
        __m256i x15 = n2;
        
        // Save initial state
        __m256i s0 = x0, s1 = x1, s2 = x2, s3 = x3;
        __m256i s4 = x4, s5 = x5, s6 = x6, s7 = x7;
        __m256i s8 = x8, s9 = x9, s10 = x10, s11 = x11;
        __m256i s12 = x12, s13 = x13, s14 = x14, s15 = x15;
        
        // ChaCha rounds
        for (int i = rounds; i > 0; i -= 2) {
            // Column round
            x0 = _mm256_add_epi32(x0, x4); x12 = _mm256_xor_si256(x12, x0); x12 = RotateLeft16(x12);
            x8 = _mm256_add_epi32(x8, x12); x4 = _mm256_xor_si256(x4, x8); x4 = RotateLeft12(x4);
            x0 = _mm256_add_epi32(x0, x4); x12 = _mm256_xor_si256(x12, x0); x12 = RotateLeft8(x12);
            x8 = _mm256_add_epi32(x8, x12); x4 = _mm256_xor_si256(x4, x8); x4 = RotateLeft7(x4);
            
            x1 = _mm256_add_epi32(x1, x5); x13 = _mm256_xor_si256(x13, x1); x13 = RotateLeft16(x13);
            x9 = _mm256_add_epi32(x9, x13); x5 = _mm256_xor_si256(x5, x9); x5 = RotateLeft12(x5);
            x1 = _mm256_add_epi32(x1, x5); x13 = _mm256_xor_si256(x13, x1); x13 = RotateLeft8(x13);
            x9 = _mm256_add_epi32(x9, x13); x5 = _mm256_xor_si256(x5, x9); x5 = RotateLeft7(x5);
            
            x2 = _mm256_add_epi32(x2, x6); x14 = _mm256_xor_si256(x14, x2); x14 = RotateLeft16(x14);
            x10 = _mm256_add_epi32(x10, x14); x6 = _mm256_xor_si256(x6, x10); x6 = RotateLeft12(x6);
            x2 = _mm256_add_epi32(x2, x6); x14 = _mm256_xor_si256(x14, x2); x14 = RotateLeft8(x14);
            x10 = _mm256_add_epi32(x10, x14); x6 = _mm256_xor_si256(x6, x10); x6 = RotateLeft7(x6);
            
            x3 = _mm256_add_epi32(x3, x7); x15 = _mm256_xor_si256(x15, x3); x15 = RotateLeft16(x15);
            x11 = _mm256_add_epi32(x11, x15); x7 = _mm256_xor_si256(x7, x11); x7 = RotateLeft12(x7);
            x3 = _mm256_add_epi32(x3, x7); x15 = _mm256_xor_si256(x15, x3); x15 = RotateLeft8(x15);
            x11 = _mm256_add_epi32(x11, x15); x7 = _mm256_xor_si256(x7, x11); x7 = RotateLeft7(x7);
            
            // Diagonal round
            x0 = _mm256_add_epi32(x0, x5); x15 = _mm256_xor_si256(x15, x0); x15 = RotateLeft16(x15);
            x10 = _mm256_add_epi32(x10, x15); x5 = _mm256_xor_si256(x5, x10); x5 = RotateLeft12(x5);
            x0 = _mm256_add_epi32(x0, x5); x15 = _mm256_xor_si256(x15, x0); x15 = RotateLeft8(x15);
            x10 = _mm256_add_epi32(x10, x15); x5 = _mm256_xor_si256(x5, x10); x5 = RotateLeft7(x5);
            
            x1 = _mm256_add_epi32(x1, x6); x12 = _mm256_xor_si256(x12, x1); x12 = RotateLeft16(x12);
            x11 = _mm256_add_epi32(x11, x12); x6 = _mm256_xor_si256(x6, x11); x6 = RotateLeft12(x6);
            x1 = _mm256_add_epi32(x1, x6); x12 = _mm256_xor_si256(x12, x1); x12 = RotateLeft8(x12);
            x11 = _mm256_add_epi32(x11, x12); x6 = _mm256_xor_si256(x6, x11); x6 = RotateLeft7(x6);
            
            x2 = _mm256_add_epi32(x2, x7); x13 = _mm256_xor_si256(x13, x2); x13 = RotateLeft16(x13);
            x8 = _mm256_add_epi32(x8, x13); x7 = _mm256_xor_si256(x7, x8); x7 = RotateLeft12(x7);
            x2 = _mm256_add_epi32(x2, x7); x13 = _mm256_xor_si256(x13, x2); x13 = RotateLeft8(x13);
            x8 = _mm256_add_epi32(x8, x13); x7 = _mm256_xor_si256(x7, x8); x7 = RotateLeft7(x7);
            
            x3 = _mm256_add_epi32(x3, x4); x14 = _mm256_xor_si256(x14, x3); x14 = RotateLeft16(x14);
            x9 = _mm256_add_epi32(x9, x14); x4 = _mm256_xor_si256(x4, x9); x4 = RotateLeft12(x4);
            x3 = _mm256_add_epi32(x3, x4); x14 = _mm256_xor_si256(x14, x3); x14 = RotateLeft8(x14);
            x9 = _mm256_add_epi32(x9, x14); x4 = _mm256_xor_si256(x4, x9); x4 = RotateLeft7(x4);
        }
        
        // Add initial state
        x0 = _mm256_add_epi32(x0, s0);
        x1 = _mm256_add_epi32(x1, s1);
        x2 = _mm256_add_epi32(x2, s2);
        x3 = _mm256_add_epi32(x3, s3);
        x4 = _mm256_add_epi32(x4, s4);
        x5 = _mm256_add_epi32(x5, s5);
        x6 = _mm256_add_epi32(x6, s6);
        x7 = _mm256_add_epi32(x7, s7);
        x8 = _mm256_add_epi32(x8, s8);
        x9 = _mm256_add_epi32(x9, s9);
        x10 = _mm256_add_epi32(x10, s10);
        x11 = _mm256_add_epi32(x11, s11);
        x12 = _mm256_add_epi32(x12, s12);
        x13 = _mm256_add_epi32(x13, s13);
        x14 = _mm256_add_epi32(x14, s14);
        x15 = _mm256_add_epi32(x15, s15);
        
        // Xelis-optimized extraction: always 128 bytes, no conditionals
        // Process words 0-3
        __m256i t0 = _mm256_unpacklo_epi32(x0, x1);
        __m256i t1 = _mm256_unpackhi_epi32(x0, x1);
        __m256i t2 = _mm256_unpacklo_epi32(x2, x3);
        __m256i t3 = _mm256_unpackhi_epi32(x2, x3);
        
        __m256i u0 = _mm256_unpacklo_epi64(t0, t2);  // Stream 0, words 0-3
        __m256i u1 = _mm256_unpackhi_epi64(t0, t2);  // Stream 1, words 0-3
        __m256i u2 = _mm256_unpacklo_epi64(t1, t3);  // Stream 2, words 0-3
        __m256i u3 = _mm256_unpackhi_epi64(t1, t3);  // Stream 3, words 0-3
        
        // Store words 0-3 for each stream
        _mm_storeu_si128((__m128i*)outputs[0], _mm256_extracti128_si256(u0, 0));
        _mm_storeu_si128((__m128i*)(outputs[0] + 64), _mm256_extracti128_si256(u0, 1));
        
        _mm_storeu_si128((__m128i*)outputs[1], _mm256_extracti128_si256(u1, 0));
        _mm_storeu_si128((__m128i*)(outputs[1] + 64), _mm256_extracti128_si256(u1, 1));
        
        _mm_storeu_si128((__m128i*)outputs[2], _mm256_extracti128_si256(u2, 0));
        _mm_storeu_si128((__m128i*)(outputs[2] + 64), _mm256_extracti128_si256(u2, 1));
        
        _mm_storeu_si128((__m128i*)outputs[3], _mm256_extracti128_si256(u3, 0));
        _mm_storeu_si128((__m128i*)(outputs[3] + 64), _mm256_extracti128_si256(u3, 1));
        
        // Process words 4-7
        t0 = _mm256_unpacklo_epi32(x4, x5);
        t1 = _mm256_unpackhi_epi32(x4, x5);
        t2 = _mm256_unpacklo_epi32(x6, x7);
        t3 = _mm256_unpackhi_epi32(x6, x7);
        
        u0 = _mm256_unpacklo_epi64(t0, t2);  // Stream 0, words 4-7
        u1 = _mm256_unpackhi_epi64(t0, t2);  // Stream 1, words 4-7
        u2 = _mm256_unpacklo_epi64(t1, t3);  // Stream 2, words 4-7
        u3 = _mm256_unpackhi_epi64(t1, t3);  // Stream 3, words 4-7
        
        // Store words 4-7
        _mm_storeu_si128((__m128i*)(outputs[0] + 16), _mm256_extracti128_si256(u0, 0));
        _mm_storeu_si128((__m128i*)(outputs[0] + 80), _mm256_extracti128_si256(u0, 1));
        
        _mm_storeu_si128((__m128i*)(outputs[1] + 16), _mm256_extracti128_si256(u1, 0));
        _mm_storeu_si128((__m128i*)(outputs[1] + 80), _mm256_extracti128_si256(u1, 1));
        
        _mm_storeu_si128((__m128i*)(outputs[2] + 16), _mm256_extracti128_si256(u2, 0));
        _mm_storeu_si128((__m128i*)(outputs[2] + 80), _mm256_extracti128_si256(u2, 1));
        
        _mm_storeu_si128((__m128i*)(outputs[3] + 16), _mm256_extracti128_si256(u3, 0));
        _mm_storeu_si128((__m128i*)(outputs[3] + 80), _mm256_extracti128_si256(u3, 1));
        
        // Process words 8-11
        t0 = _mm256_unpacklo_epi32(x8, x9);
        t1 = _mm256_unpackhi_epi32(x8, x9);
        t2 = _mm256_unpacklo_epi32(x10, x11);
        t3 = _mm256_unpackhi_epi32(x10, x11);
        
        u0 = _mm256_unpacklo_epi64(t0, t2);
        u1 = _mm256_unpackhi_epi64(t0, t2);
        u2 = _mm256_unpacklo_epi64(t1, t3);
        u3 = _mm256_unpackhi_epi64(t1, t3);
        
        // Store words 8-11
        _mm_storeu_si128((__m128i*)(outputs[0] + 32), _mm256_extracti128_si256(u0, 0));
        _mm_storeu_si128((__m128i*)(outputs[0] + 96), _mm256_extracti128_si256(u0, 1));
        
        _mm_storeu_si128((__m128i*)(outputs[1] + 32), _mm256_extracti128_si256(u1, 0));
        _mm_storeu_si128((__m128i*)(outputs[1] + 96), _mm256_extracti128_si256(u1, 1));
        
        _mm_storeu_si128((__m128i*)(outputs[2] + 32), _mm256_extracti128_si256(u2, 0));
        _mm_storeu_si128((__m128i*)(outputs[2] + 96), _mm256_extracti128_si256(u2, 1));
        
        _mm_storeu_si128((__m128i*)(outputs[3] + 32), _mm256_extracti128_si256(u3, 0));
        _mm_storeu_si128((__m128i*)(outputs[3] + 96), _mm256_extracti128_si256(u3, 1));
        
        // Process words 12-15
        t0 = _mm256_unpacklo_epi32(x12, x13);
        t1 = _mm256_unpackhi_epi32(x12, x13);
        t2 = _mm256_unpacklo_epi32(x14, x15);
        t3 = _mm256_unpackhi_epi32(x14, x15);
        
        u0 = _mm256_unpacklo_epi64(t0, t2);
        u1 = _mm256_unpackhi_epi64(t0, t2);
        u2 = _mm256_unpacklo_epi64(t1, t3);
        u3 = _mm256_unpackhi_epi64(t1, t3);
        
        // Store words 12-15
        _mm_storeu_si128((__m128i*)(outputs[0] + 48), _mm256_extracti128_si256(u0, 0));
        _mm_storeu_si128((__m128i*)(outputs[0] + 112), _mm256_extracti128_si256(u0, 1));
        
        _mm_storeu_si128((__m128i*)(outputs[1] + 48), _mm256_extracti128_si256(u1, 0));
        _mm_storeu_si128((__m128i*)(outputs[1] + 112), _mm256_extracti128_si256(u1, 1));
        
        _mm_storeu_si128((__m128i*)(outputs[2] + 48), _mm256_extracti128_si256(u2, 0));
        _mm_storeu_si128((__m128i*)(outputs[2] + 112), _mm256_extracti128_si256(u2, 1));
        
        _mm_storeu_si128((__m128i*)(outputs[3] + 48), _mm256_extracti128_si256(u3, 0));
        _mm_storeu_si128((__m128i*)(outputs[3] + 112), _mm256_extracti128_si256(u3, 1));
        
        outputs[0] += 128;
        outputs[1] += 128;
        outputs[2] += 128;
        outputs[3] += 128;
        
        counter_base += 2;
    }
}

#endif