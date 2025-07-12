#pragma once
#include <immintrin.h>

#ifndef TNN_LEGACY_AMD64

// AVX512+VAES implementation
__attribute__((target("avx512f,avx512vaes"))) inline void aesenc_8way(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                                                                      __m128i *out4, __m128i *out5, __m128i *out6, __m128i *out7,
                                                                      const __m128i in0, const __m128i in1, const __m128i in2, const __m128i in3,
                                                                      const __m128i in4, const __m128i in5, const __m128i in6, const __m128i in7,
                                                                      const __m128i key0, const __m128i key1, const __m128i key2, const __m128i key3,
                                                                      const __m128i key4, const __m128i key5, const __m128i key6, const __m128i key7)
{
  // Pack inputs into two 512-bit vectors
  __m512i data_lo = _mm512_inserti64x2(
      _mm512_inserti64x2(
          _mm512_inserti64x2(_mm512_castsi128_si512(in0), in1, 1),
          in2, 2),
      in3, 3);

  __m512i data_hi = _mm512_inserti64x2(
      _mm512_inserti64x2(
          _mm512_inserti64x2(_mm512_castsi128_si512(in4), in5, 1),
          in6, 2),
      in7, 3);

  __m512i keys_lo = _mm512_inserti64x2(
      _mm512_inserti64x2(
          _mm512_inserti64x2(_mm512_castsi128_si512(key0), key1, 1),
          key2, 2),
      key3, 3);

  __m512i keys_hi = _mm512_inserti64x2(
      _mm512_inserti64x2(
          _mm512_inserti64x2(_mm512_castsi128_si512(key4), key5, 1),
          key6, 2),
      key7, 3);

  // Two VAES instructions for 8 blocks
  __m512i result_lo = _mm512_aesenc_epi128(data_lo, keys_lo);
  __m512i result_hi = _mm512_aesenc_epi128(data_hi, keys_hi);

  // Extract results
  *out0 = _mm512_extracti64x2_epi64(result_lo, 0);
  *out1 = _mm512_extracti64x2_epi64(result_lo, 1);
  *out2 = _mm512_extracti64x2_epi64(result_lo, 2);
  *out3 = _mm512_extracti64x2_epi64(result_lo, 3);
  *out4 = _mm512_extracti64x2_epi64(result_hi, 0);
  *out5 = _mm512_extracti64x2_epi64(result_hi, 1);
  *out6 = _mm512_extracti64x2_epi64(result_hi, 2);
  *out7 = _mm512_extracti64x2_epi64(result_hi, 3);
}

// AVX2+VAES implementation
__attribute__((target("avx2,vaes"))) inline void aesenc_8way(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                                                             __m128i *out4, __m128i *out5, __m128i *out6, __m128i *out7,
                                                             const __m128i in0, const __m128i in1, const __m128i in2, const __m128i in3,
                                                             const __m128i in4, const __m128i in5, const __m128i in6, const __m128i in7,
                                                             const __m128i key0, const __m128i key1, const __m128i key2, const __m128i key3,
                                                             const __m128i key4, const __m128i key5, const __m128i key6, const __m128i key7)
{
  // AVX2+VAES path - 4 instructions for 8
  __m256i data01 = _mm256_inserti128_si256(_mm256_castsi128_si256(in0), in1, 1);
  __m256i data23 = _mm256_inserti128_si256(_mm256_castsi128_si256(in2), in3, 1);
  __m256i data45 = _mm256_inserti128_si256(_mm256_castsi128_si256(in4), in5, 1);
  __m256i data67 = _mm256_inserti128_si256(_mm256_castsi128_si256(in6), in7, 1);

  __m256i keys01 = _mm256_inserti128_si256(_mm256_castsi128_si256(key0), key1, 1);
  __m256i keys23 = _mm256_inserti128_si256(_mm256_castsi128_si256(key2), key3, 1);
  __m256i keys45 = _mm256_inserti128_si256(_mm256_castsi128_si256(key4), key5, 1);
  __m256i keys67 = _mm256_inserti128_si256(_mm256_castsi128_si256(key6), key7, 1);

  __m256i result01 = _mm256_aesenc_epi128(data01, keys01);
  __m256i result23 = _mm256_aesenc_epi128(data23, keys23);
  __m256i result45 = _mm256_aesenc_epi128(data45, keys45);
  __m256i result67 = _mm256_aesenc_epi128(data67, keys67);

  *out0 = _mm256_castsi256_si128(result01);
  *out1 = _mm256_extracti128_si256(result01, 1);
  *out2 = _mm256_castsi256_si128(result23);
  *out3 = _mm256_extracti128_si256(result23, 1);
  *out4 = _mm256_castsi256_si128(result45);
  *out5 = _mm256_extracti128_si256(result45, 1);
  *out6 = _mm256_castsi256_si128(result67);
  *out7 = _mm256_extracti128_si256(result67, 1);
}

#endif

// Basic AES-NI implementation
__attribute__((target("aes"))) inline void aesenc_8way(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                                                       __m128i *out4, __m128i *out5, __m128i *out6, __m128i *out7,
                                                       const __m128i in0, const __m128i in1, const __m128i in2, const __m128i in3,
                                                       const __m128i in4, const __m128i in5, const __m128i in6, const __m128i in7,
                                                       const __m128i key0, const __m128i key1, const __m128i key2, const __m128i key3,
                                                       const __m128i key4, const __m128i key5, const __m128i key6, const __m128i key7)
{
  // Fallback - 8 separate instructions
  *out0 = _mm_aesenc_si128(in0, key0);
  *out1 = _mm_aesenc_si128(in1, key1);
  *out2 = _mm_aesenc_si128(in2, key2);
  *out3 = _mm_aesenc_si128(in3, key3);
  *out4 = _mm_aesenc_si128(in4, key4);
  *out5 = _mm_aesenc_si128(in5, key5);
  *out6 = _mm_aesenc_si128(in6, key6);
  *out7 = _mm_aesenc_si128(in7, key7);
}

// Default fallback
inline void aesenc_8way(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                        __m128i *out4, __m128i *out5, __m128i *out6, __m128i *out7,
                        const __m128i in0, const __m128i in1, const __m128i in2, const __m128i in3,
                        const __m128i in4, const __m128i in5, const __m128i in6, const __m128i in7,
                        const __m128i key0, const __m128i key1, const __m128i key2, const __m128i key3,
                        const __m128i key4, const __m128i key5, const __m128i key6, const __m128i key7)
{
  // Default implementation - same as basic AES-NI
  *out0 = _mm_aesenc_si128(in0, key0);
  *out1 = _mm_aesenc_si128(in1, key1);
  *out2 = _mm_aesenc_si128(in2, key2);
  *out3 = _mm_aesenc_si128(in3, key3);
  *out4 = _mm_aesenc_si128(in4, key4);
  *out5 = _mm_aesenc_si128(in5, key5);
  *out6 = _mm_aesenc_si128(in6, key6);
  *out7 = _mm_aesenc_si128(in7, key7);
}

#ifndef TNN_LEGACY_AMD64

// Similarly for 4-way
__attribute__((target("avx512f,avx512vaes"))) inline void aesenc_4way(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                                                                      const __m128i in0, const __m128i in1, const __m128i in2, const __m128i in3,
                                                                      const __m128i key0, const __m128i key1, const __m128i key2, const __m128i key3)
{
  __m512i data = _mm512_inserti64x2(
      _mm512_inserti64x2(
          _mm512_inserti64x2(_mm512_castsi128_si512(in0), in1, 1),
          in2, 2),
      in3, 3);

  __m512i keys = _mm512_inserti64x2(
      _mm512_inserti64x2(
          _mm512_inserti64x2(_mm512_castsi128_si512(key0), key1, 1),
          key2, 2),
      key3, 3);

  __m512i result = _mm512_aesenc_epi128(data, keys);

  *out0 = _mm512_extracti64x2_epi64(result, 0);
  *out1 = _mm512_extracti64x2_epi64(result, 1);
  *out2 = _mm512_extracti64x2_epi64(result, 2);
  *out3 = _mm512_extracti64x2_epi64(result, 3);
}

__attribute__((target("avx2,vaes"))) inline void aesenc_4way(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                                                             const __m128i in0, const __m128i in1, const __m128i in2, const __m128i in3,
                                                             const __m128i key0, const __m128i key1, const __m128i key2, const __m128i key3)
{
  __m256i data01 = _mm256_inserti128_si256(_mm256_castsi128_si256(in0), in1, 1);
  __m256i data23 = _mm256_inserti128_si256(_mm256_castsi128_si256(in2), in3, 1);

  __m256i keys01 = _mm256_inserti128_si256(_mm256_castsi128_si256(key0), key1, 1);
  __m256i keys23 = _mm256_inserti128_si256(_mm256_castsi128_si256(key2), key3, 1);

  __m256i result01 = _mm256_aesenc_epi128(data01, keys01);
  __m256i result23 = _mm256_aesenc_epi128(data23, keys23);

  *out0 = _mm256_castsi256_si128(result01);
  *out1 = _mm256_extracti128_si256(result01, 1);
  *out2 = _mm256_castsi256_si128(result23);
  *out3 = _mm256_extracti128_si256(result23, 1);
}

#endif

__attribute__((target("aes"))) inline void aesenc_4way(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                                                       const __m128i in0, const __m128i in1, const __m128i in2, const __m128i in3,
                                                       const __m128i key0, const __m128i key1, const __m128i key2, const __m128i key3)
{
  *out0 = _mm_aesenc_si128(in0, key0);
  *out1 = _mm_aesenc_si128(in1, key1);
  *out2 = _mm_aesenc_si128(in2, key2);
  *out3 = _mm_aesenc_si128(in3, key3);
}

#ifndef TNN_LEGACY_AMD64

// 2-way
__attribute__((target("avx2,vaes"))) inline __m256i aesenc_2way(__m256i data, __m256i key)
{
  return _mm256_aesenc_epi128(data, key);
}

#endif

__attribute__((target("aes"))) inline __m256i aesenc_2way(__m256i data, __m256i key)
{
  __m128i lo = _mm256_castsi256_si128(data);
  __m128i hi = _mm256_extracti128_si256(data, 1);
  __m128i key_lo = _mm256_castsi256_si128(key);
  __m128i key_hi = _mm256_extracti128_si256(key, 1);

  lo = _mm_aesenc_si128(lo, key_lo);
  hi = _mm_aesenc_si128(hi, key_hi);

  return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

// Convenience functions for when keys are the same
inline void aesenc_4way_same_key(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                                 const __m128i in0, const __m128i in1,
                                 const __m128i in2, const __m128i in3,
                                 const __m128i key)
{
  aesenc_4way(out0, out1, out2, out3, in0, in1, in2, in3, key, key, key, key);
}

inline void aesenc_8way_same_key(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                                 __m128i *out4, __m128i *out5, __m128i *out6, __m128i *out7,
                                 const __m128i in0, const __m128i in1,
                                 const __m128i in2, const __m128i in3,
                                 const __m128i in4, const __m128i in5,
                                 const __m128i in6, const __m128i in7,
                                 const __m128i key)
{
  aesenc_8way(out0, out1, out2, out3, out4, out5, out6, out7,
              in0, in1, in2, in3, in4, in5, in6, in7,
              key, key, key, key, key, key, key, key);
}

// For CryptoNight where we need multiple rounds
template <int ROUNDS>
inline void aesenc_8way_rounds(__m128i *out0, __m128i *out1, __m128i *out2, __m128i *out3,
                               __m128i *out4, __m128i *out5, __m128i *out6, __m128i *out7,
                               const __m128i *keys)
{
  for (int i = 0; i < ROUNDS; i++)
  {
    aesenc_8way(out0, out1, out2, out3, out4, out5, out6, out7,
                *out0, *out1, *out2, *out3, *out4, *out5, *out6, *out7,
                keys[i], keys[i], keys[i], keys[i],
                keys[i], keys[i], keys[i], keys[i]);
  }
}