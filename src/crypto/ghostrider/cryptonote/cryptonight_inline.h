/*
 * Copyright 2021 Delgon
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "algo/blake/sph_blake.h"
#include "algo/jh/sph_jh.h"
#include <crypto/skein/skein.h>
#include "cryptonight.h"
#include "simd-utils.h"
#include "keccak_wrappers.h"
#include "aes/aes_multiway.h"
#include <immintrin.h>
#include <stdio.h>

#include <array>
#include <type_traits>
#include <utility>

#ifndef __AES__
#include "algo/groestl/sph_groestl.h"
#include "soft_aes.h"
#else
#include "algo/groestl/aes_ni/hash-groestl256.h"
#endif

#ifdef __VAES__
static constexpr bool kVAES = true;
static constexpr bool kAES = true;
#elif defined(__AES__)
static constexpr bool kVAES = false;
static constexpr bool kAES = true;
#else
static constexpr bool kVAES = false;
static constexpr bool kAES = false;
#endif

extern __thread uint8_t *__restrict__ hp_state;

static inline void do_blake_hash(const void *input, void *output) {
  sph_blake256_context ctx __attribute__((aligned(64)));
  sph_blake256_init(&ctx);
  sph_blake256(&ctx, input, 200);
  sph_blake256_close(&ctx, output);
}

static inline void do_groestl_hash(const void *input, void *output) {
#ifdef __AES__
  hashState_groestl256 ctx __attribute__((aligned(64)));
  groestl256_full(&ctx, output, input, 1600);
#else
  sph_groestl256_context ctx __attribute__((aligned(64)));
  sph_groestl256_init(&ctx);
  sph_groestl256(&ctx, input, 200);
  sph_groestl256_close(&ctx, output);
#endif
}

static inline void do_jh_hash(const void *input, void *output) {
  sph_jh256_context ctx __attribute__((aligned(64)));
  sph_jh256_init(&ctx);
  sph_jh256(&ctx, input, 200);
  sph_jh256_close(&ctx, output);
}

static inline void do_skein_hash(const void *input, void *output) {
  skein_hash(256,
    (const BitSequence *)input, 
    200 * 8,
    (BitSequence *)output);
}

static inline void (*const extra_hashes[4])(const void *, void *) = {
    do_blake_hash, do_groestl_hash, do_jh_hash, do_skein_hash};

// This will shift and xor tmp1 into
// itself as 4 32-bit vals such as
// sl_xor(a1 a2 a3 a4) = a1 (a2^a1) (a3^a2^a1) (a4^a3^a2^a1)
static __attribute__((always_inline)) inline __m128i sl_xor(__m128i tmp1) {
  __m128i tmp4;
  tmp4 = _mm_slli_si128(tmp1, 0x04);
  tmp1 = _mm_xor_si128(tmp1, tmp4);
  tmp4 = _mm_slli_si128(tmp4, 0x04);
  tmp1 = _mm_xor_si128(tmp1, tmp4);
  tmp4 = _mm_slli_si128(tmp4, 0x04);
  tmp1 = _mm_xor_si128(tmp1, tmp4);
  return tmp1;
}

template <uint8_t kRcon>
static __attribute__((always_inline)) inline void
aes_genkey_sub(__m128i *xout0, __m128i *xout2) {
#ifdef __AES__
  __m128i xout1 = _mm_aeskeygenassist_si128(*xout2, kRcon);
#else
  __m128i xout1 = soft_aeskeygenassist(*xout2, kRcon);
#endif
  // see PSHUFD, set all elems to 4th elem
  xout1 = _mm_shuffle_epi32(xout1, 0xFF);
  *xout0 = sl_xor(*xout0);
  *xout0 = _mm_xor_si128(*xout0, xout1);
#ifdef __AES__
  xout1 = _mm_aeskeygenassist_si128(*xout0, 0x00);
#else
  xout1 = soft_aeskeygenassist(*xout0, 0x00);
#endif
  // see PSHUFD, set all elems to 3th elem
  xout1 = _mm_shuffle_epi32(xout1, 0xAA);
  *xout2 = sl_xor(*xout2);
  *xout2 = _mm_xor_si128(*xout2, xout1);
}

template <typename T>
static inline std::array<T, 10> aes_genkey(const __m128i *memory) {
  alignas(64) std::array<T, 10> key;
  __m128i xout0 = _mm_load_si128(memory);
  __m128i xout2 = _mm_load_si128(memory + 1);
  key[0] = xout0;
  key[1] = xout2;

  aes_genkey_sub<0x01>(&xout0, &xout2);
  key[2] = xout0;
  key[3] = xout2;

  aes_genkey_sub<0x02>(&xout0, &xout2);
  key[4] = xout0;
  key[5] = xout2;

  aes_genkey_sub<0x04>(&xout0, &xout2);
  key[6] = xout0;
  key[7] = xout2;

  aes_genkey_sub<0x08>(&xout0, &xout2);
  key[8] = std::move(xout0);
  key[9] = std::move(xout2);

  return key;
}

template <>
inline std::array<__m256i, 10> aes_genkey<__m256i>(const __m128i *memory) {
  alignas(64) std::array<__m256i, 10> key;
  __m128i xout0 = _mm_load_si128(memory);
  __m128i xout2 = _mm_load_si128(memory + 1);
  key[0] = _mm256_set_m128i(xout0, xout0);
  key[1] = _mm256_set_m128i(xout2, xout2);

  aes_genkey_sub<0x01>(&xout0, &xout2);
  key[2] = _mm256_set_m128i(xout0, xout0);
  key[3] = _mm256_set_m128i(xout2, xout2);

  aes_genkey_sub<0x02>(&xout0, &xout2);
  key[4] = _mm256_set_m128i(xout0, xout0);
  key[5] = _mm256_set_m128i(xout2, xout2);

  aes_genkey_sub<0x04>(&xout0, &xout2);
  key[6] = _mm256_set_m128i(xout0, xout0);
  key[7] = _mm256_set_m128i(xout2, xout2);

  aes_genkey_sub<0x08>(&xout0, &xout2);
  key[8] = _mm256_set_m128i(xout0, xout0);
  key[9] = _mm256_set_m128i(xout2, xout2);

  return key;
}

template <typename T, size_t kSize>
static inline std::array<std::array<T, 10>, kSize>
aes_multikey(const uint8_t *memory, size_t shift) {
  if constexpr (kSize == 4) {
    return {aes_genkey<T>((__m128i *)(memory + shift)),
            aes_genkey<T>((__m128i *)(memory + shift + 208)),
            aes_genkey<T>((__m128i *)(memory + shift + 208 * 2)),
            aes_genkey<T>((__m128i *)(memory + shift + 208 * 3))};
  } else if constexpr (kSize == 2) {
    return {aes_genkey<T>((__m128i *)(memory + shift)),
            aes_genkey<T>((__m128i *)(memory + shift + 208))};
  } else { // kSize 1?
    return {aes_genkey<T>((__m128i *)(memory + shift))};
  }
}

#ifndef __AES__

static __attribute__((noinline)) __m128i soft_aesenc(const __m128i &in,
                                                     const __m128i &key) {
  // saes_table is implemented in "soft_aes.h"
  const uint32_t x0 = ((uint32_t *)(&in))[0];
  const uint32_t x1 = ((uint32_t *)(&in))[1];
  const uint32_t x2 = ((uint32_t *)(&in))[2];
  const uint32_t x3 = ((uint32_t *)(&in))[3];

  return _mm_xor_si128(
      _mm_set_epi32(
          (saes_table[0][(uint8_t)x3] ^ saes_table[1][(uint8_t)(x0 >> 8)] ^
           saes_table[2][(uint8_t)(x1 >> 16)] ^
           saes_table[3][(uint8_t)(x2 >> 24)]),
          (saes_table[0][(uint8_t)x2] ^ saes_table[1][(uint8_t)(x3 >> 8)] ^
           saes_table[2][(uint8_t)(x0 >> 16)] ^
           saes_table[3][(uint8_t)(x1 >> 24)]),
          (saes_table[0][(uint8_t)x1] ^ saes_table[1][(uint8_t)(x2 >> 8)] ^
           saes_table[2][(uint8_t)(x3 >> 16)] ^
           saes_table[3][(uint8_t)(x0 >> 24)]),
          (saes_table[0][(uint8_t)x0] ^ saes_table[1][(uint8_t)(x1 >> 8)] ^
           saes_table[2][(uint8_t)(x2 >> 16)] ^
           saes_table[3][(uint8_t)(x3 >> 24)])),
      key);
}

#endif

static __attribute__((always_inline)) inline auto aesenc(const auto &x,
                                                         const auto &key) {
  // We need __m256i check to make sure we are using standard _mm_aesenc_si128
  // in the main loop of cryptonight.
  if constexpr (kVAES && sizeof(x) == sizeof(__m256i)) {
    return _mm256_aesenc_epi128(x, key);
  } else if constexpr (kAES) {
    return _mm_aesenc_si128(x, key);
  } else {
    return soft_aesenc(x, key);
  }
}

static __attribute__((always_inline)) inline auto aesenc(auto &&x,
                                                         const auto &key) {
  // We need __m256i check to make sure we are using standard _mm_aesenc_si128
  // in the main loop of cryptonight.
  if constexpr (kVAES && sizeof(x) == sizeof(__m256i)) {
    return _mm256_aesenc_epi128(x, key);
  } else if constexpr (kAES) {
    return _mm_aesenc_si128(x, key);
  } else {
    return soft_aesenc(x, key);
  }
}

static __attribute__((always_inline)) inline void aes_batch(const auto &key,
                                                            auto &x) {

  if constexpr (kVAES) {
#pragma GCC unroll 4
    for (size_t i = 0; i < 4; ++i) {
      x[i] = aesenc(std::move(x[i]), key[0]);
      x[i] = aesenc(std::move(x[i]), key[1]);
      x[i] = aesenc(std::move(x[i]), key[2]);
      x[i] = aesenc(std::move(x[i]), key[3]);
      x[i] = aesenc(std::move(x[i]), key[4]);
      x[i] = aesenc(std::move(x[i]), key[5]);
      x[i] = aesenc(std::move(x[i]), key[6]);
      x[i] = aesenc(std::move(x[i]), key[7]);
      x[i] = aesenc(std::move(x[i]), key[8]);
      x[i] = aesenc(std::move(x[i]), key[9]);
    }
  } else {
#pragma GCC unroll 8
    for (size_t i = 0; i < 8; ++i) {
      x[i] = aesenc(std::move(x[i]), key[0]);
      x[i] = aesenc(std::move(x[i]), key[1]);
      x[i] = aesenc(std::move(x[i]), key[2]);
      x[i] = aesenc(std::move(x[i]), key[3]);
      x[i] = aesenc(std::move(x[i]), key[4]);
      x[i] = aesenc(std::move(x[i]), key[5]);
      x[i] = aesenc(std::move(x[i]), key[6]);
      x[i] = aesenc(std::move(x[i]), key[7]);
      x[i] = aesenc(std::move(x[i]), key[8]);
      x[i] = aesenc(std::move(x[i]), key[9]);
    }
  }
}

static __attribute__((always_inline)) inline void xor_batch(auto &dst,
                                                            const auto &src) {
  // The reason why VAES is checked, not AVX2 is coz using it on only AVX2
  // capable CPUs has negative impact on the performance instead of using
  // SSE2 xor.
  if constexpr (kVAES) {
#pragma GCC unroll 4
    for (size_t i = 0; i < 4; ++i) {
      dst[i] = _mm256_xor_si256(std::move(dst[i]), src[i]);
    }
  } else {
#pragma GCC unroll 8
    for (size_t i = 0; i < 8; ++i) {
      dst[i] = _mm_xor_si128(std::move(dst[i]), src[i]);
    }
  }
}

static __attribute__((always_inline)) inline void
xor_batch_ptr(auto &dst, const auto *src) {
  // The reason why VAES is checked, not AVX2 is coz using it on only AVX2
  // capable CPUs has negative impact on the performance instead of using
  // SSE2 xor.
  if constexpr (kVAES) {
#pragma GCC unroll 4
    for (size_t i = 0; i < 4; ++i) {
      dst[i] = _mm256_xor_si256(std::move(dst[i]), src[i]);
    }
  } else {
#pragma GCC unroll 8
    for (size_t i = 0; i < 8; ++i) {
      dst[i] = _mm_xor_si128(std::move(dst[i]), src[i]);
    }
  }
}

#define PREFETCH_W(ptr) __builtin_prefetch((ptr), 1, 3)
// Equivalent of:
// #define PREFETCH_W(ptr) _mm_prefetch((ptr), _MM_HINT_ET0)

// Not recommended to use!
#define PREFETCH_W_NTA(ptr) __builtin_prefetch((ptr), 1, 0)
// Equivalent of:
// #define PREFETCH_W_NTA(ptr) _mm_prefetch((ptr), _MM_HINT_ENTA)
// ^ does not exist on most architectures by itself but __builtin does "work".

#define PREFETCH_R(ptr) __builtin_prefetch((ptr), 0, 1);
// Equivalent of:
// #define PREFETCH_R(ptr) _mm_prefetch((ptr), _MM_HINT_T0)

#define PREFETCH_R_NTA(ptr) __builtin_prefetch((ptr), 0, 0);
// Equivalent of:
// #define PREFETCH_R_NTA(ptr) _mm_prefetch((ptr), _MM_HINT_NTA)

#ifdef __VAES__
typedef __m256i _mXXXi;
#else
typedef __m128i _mXXXi;
#endif
constexpr size_t kLineSize = 64;
constexpr size_t kStateSize = 128;
constexpr size_t kWPL = kLineSize / sizeof(__m128i);
constexpr size_t kWPS = kStateSize / sizeof(__m128i);
constexpr size_t kArrSize = kStateSize / sizeof(_mXXXi);

template <size_t kMemory, size_t kBatches, size_t kPrefetchSize>
static inline void explode_scratchpad(const uint8_t *state, __m128i *ls) {
  constexpr size_t kPrefetchLines = kPrefetchSize / kLineSize;
  constexpr size_t kPrefetchShift0 = kPrefetchLines * kWPL;
  constexpr size_t kPrefetchShift1 = kPrefetchLines * kWPL + kWPL;
  constexpr size_t kMemoryEnd = kMemory / kStateSize;

  alignas(32) std::array<std::array<_mXXXi, kArrSize>, kBatches> x;
  const auto k = aes_multikey<_mXXXi, kBatches>(state, 0);

  for (size_t i = 0; i < kPrefetchShift0; i += kWPL) {
    PREFETCH_W(ls + i);
  }

  for (size_t batch = 0; batch < kBatches; ++batch) {
    memcpy(x[batch].data(), &state[208 * batch + 64], kStateSize);

    for (size_t i = 0; i < kMemoryEnd; ++i) {
      PREFETCH_W(ls + kPrefetchShift0);
      PREFETCH_W(ls + kPrefetchShift1);
      aes_batch(k[batch], x[batch]);

      memcpy(ls, x[batch].data(), kStateSize);
      ls += kWPS;
    }
  }
}



template <size_t kMemory, bool kHalf, size_t kBatches, size_t kPrefetchSize,
          bool kL1Prefetch>
static inline void implode_scratchpad(const __m128i *ls, uint8_t *state) {
  constexpr size_t kPrefetchLines = kPrefetchSize / kLineSize;
  constexpr size_t kPrefetchShift0 = kPrefetchLines * kWPL;
  constexpr size_t kPrefetchShift1 = kPrefetchLines * kWPL + kWPL;
  constexpr size_t kMemoryEnd =
      kHalf ? kMemory / kStateSize / 2 : kMemory / kStateSize;

  alignas(32) std::array<std::array<_mXXXi, kArrSize>, kBatches> x;
  const auto k = aes_multikey<_mXXXi, kBatches>(state, 32);

  for (size_t i = 0; i < kPrefetchShift0; i += kWPL) {
    if constexpr (kL1Prefetch) {
      PREFETCH_R(ls + i);
    } else {
      PREFETCH_R_NTA(ls + i);
    }
  }

  for (size_t batch = 0; batch < kBatches; ++batch) {
    memcpy(x[batch].data(), &state[208 * batch + 64], kStateSize);

    for (size_t i = 0; i < kMemoryEnd; ++i) {
      if constexpr (kL1Prefetch) {
        PREFETCH_R(ls + kPrefetchShift0);
        PREFETCH_R(ls + kPrefetchShift1);
      } else {
        PREFETCH_R_NTA(ls + kPrefetchShift0);
        PREFETCH_R_NTA(ls + kPrefetchShift1);
      }
      if constexpr (kVAES) {
        xor_batch_ptr(x[batch], (__m256i *)ls);
      } else {
        xor_batch_ptr(x[batch], ls);
      }
      aes_batch(k[batch], x[batch]);
      ls += kWPS;
    }
    memcpy(&state[208 * batch + 64], x[batch].data(), kStateSize);
    if constexpr (kHalf) {
      memcpy(x[batch].data(), ls, kStateSize);
      if constexpr (kL1Prefetch) {
        PREFETCH_R(ls + kPrefetchShift0);
        PREFETCH_R(ls + kPrefetchShift1);
      } else {
        PREFETCH_R_NTA(ls + kPrefetchShift0);
        PREFETCH_R_NTA(ls + kPrefetchShift1);
      }
      ls += kWPS;
    }
  }

  if constexpr (kHalf) {
    alignas(32) std::array<_mXXXi, kArrSize> x1;
    const auto k1 = aes_multikey<_mXXXi, kBatches>(state, 0);
    for (size_t batch = 0; batch < kBatches; ++batch) {
      memcpy(x1.data(), &state[208 * batch + 64], kStateSize);
      for (size_t i = 0; i < kMemoryEnd; ++i) {
        xor_batch(x1, x[batch]);
        aes_batch(k1[batch], x[batch]);
        aes_batch(k[batch], x1);
      }
      memcpy(&state[208 * batch + 64], x1.data(), kStateSize);
    }
  }
}

#define AES(suffix)                                                            \
  const __m128i cx##suffix = aesenc(*sp_loc0##suffix, ax##suffix);             \
  if constexpr (kPrefetch) {                                                   \
    sp_loc1##suffix = (__m128i *)(&l##suffix[cx##suffix[0] & kMask]);          \
    PREFETCH_W(sp_loc1##suffix);                                               \
  }

static constexpr std::array<uint64_t, 8> table = {
    ((0x7531 >> 0) & 0x3) << 28,  ((0x7531 >> 2) & 0x3) << 28,
    ((0x7531 >> 4) & 0x3) << 28,  ((0x7531 >> 6) & 0x3) << 28,
    ((0x7531 >> 8) & 0x3) << 28,  ((0x7531 >> 10) & 0x3) << 28,
    ((0x7531 >> 12) & 0x3) << 28, ((0x7531 >> 14) & 0x3) << 28};

#define TWEAK(suffix)                                                          \
  {                                                                            \
    *sp_loc0##suffix = _mm_xor_si128(std::move(bx##suffix), cx##suffix);       \
    bx##suffix = cx##suffix;                                                   \
                                                                               \
    const uint32_t x = static_cast<uint32_t>((*sp_loc0##suffix)[1] >> 24);     \
    (*sp_loc0##suffix)[1] ^= table[((x >> 3) & 6) | (x & 1)];                  \
  }

#define POST_AES(suffix)                                                       \
  if constexpr (!kPrefetch) {                                                  \
    sp_loc1##suffix = (__m128i *)(&l##suffix[cx##suffix[0] & kMask]);          \
  }                                                                            \
  {                                                                            \
    const uint64_t cl = (*sp_loc1##suffix)[0];                                 \
    const uint64_t ch = (*sp_loc1##suffix)[1];                                 \
    {                                                                          \
      uint64_t hi, lo;                                                         \
      asm("mulq %[y]\n\t"                                                      \
          : "=d"(hi), "=a"(lo)                                                 \
          : "1"(cx##suffix[0]), [y] "ri"(cl)                                   \
          : "cc");                                                             \
      ax##suffix[0] += hi;                                                     \
      ax##suffix[1] += lo;                                                     \
    }                                                                          \
    (*sp_loc1##suffix)[0] = ax##suffix[0];                                     \
    (*sp_loc1##suffix)[1] = ax##suffix[1] ^ tweak##suffix;                     \
                                                                               \
    ax##suffix[0] ^= cl;                                                       \
    ax##suffix[1] ^= ch;                                                       \
  }                                                                            \
  sp_loc0##suffix = (__m128i *)(&l##suffix[ax##suffix[0] & kMask]);            \
  if constexpr (kPrefetch) {                                                   \
    PREFETCH_W(sp_loc0##suffix);                                               \
  }

#define CRYPTONIGHT_INIT(suffix)                                               \
  uint8_t *__restrict__ l##suffix =                                            \
      (uint8_t *)(__builtin_assume_aligned(&hp_state[suffix * kShift], 16));   \
  uint64_t *h##suffix = (uint64_t *)(&state[208 * suffix]);                    \
                                                                               \
  keccak1600(((const uint8_t *)(input##suffix)), 64, (uint8_t *)h##suffix);

#define CRYPTONIGHT_VARIABLES(suffix)                                          \
  const uint64_t tweak##suffix =                                               \
      *((const uint64_t *)(&(((const uint8_t *)input##suffix)[35]))) ^         \
      h##suffix[24];                                                           \
                                                                               \
  __m128i ax##suffix =                                                         \
      _mm_set_epi64x(static_cast<int64_t>(h##suffix[1] ^ h##suffix[5]),        \
                     static_cast<int64_t>(h##suffix[0] ^ h##suffix[4]));       \
                                                                               \
  __m128i bx##suffix =                                                         \
      _mm_set_epi64x(static_cast<int64_t>(h##suffix[3] ^ h##suffix[7]),        \
                     static_cast<int64_t>(h##suffix[2] ^ h##suffix[6]));       \
  __m128i *__restrict__ sp_loc0##suffix =                                      \
      (__m128i *)(&l##suffix[ax##suffix[0] & kMask]);                          \
  __m128i *__restrict__ sp_loc1##suffix = nullptr;

#define CRYPTONIGHT_FINISH(suffix)                                             \
  keccakf(h##suffix, 24);                                                      \
  extra_hashes[state[208 * suffix] & 3](&state[208 * suffix], output##suffix); \
  memset(&((uint8_t *)output##suffix)[32], 0, 32);

template <size_t kMemory, size_t kIterations, uint32_t kMask, size_t kShift,
          size_t kPrefetchW, size_t kPrefetchR, bool kL1Prefetch>
void cryptonight_hash(const void *input0, void *output0) {
  uint8_t state[208] __attribute__((aligned(16)));
  CRYPTONIGHT_INIT(0);
  explode_scratchpad<kShift, 1, kPrefetchW>(state, ((__m128i *)l0));
  CRYPTONIGHT_VARIABLES(0);
  constexpr bool kPrefetch = false;
  for (size_t i = 0; i < kIterations; ++i) {
    // AES
    AES(0);
    TWEAK(0);

    // Post AES
    POST_AES(0);
  }
  implode_scratchpad<kMemory, kMemory != kShift, 1, kPrefetchR, kL1Prefetch>(
      ((const __m128i *)l0), state);
  CRYPTONIGHT_FINISH(0);
}

template void cryptonight_hash<262144, 32768, 0x3FFF, 262144, 512, 512, true>(const void*, void*);
template void cryptonight_hash<262144, 32768, 0x3FFF, 262144, 512, 512, true>(const void*, void*);

#define AES_2WAY(suffix0, suffix1) \
  __m128i cx##suffix0, cx##suffix1; \
  { \
    __m256i data = _mm256_set_m128i(*sp_loc0##suffix1, *sp_loc0##suffix0); \
    __m256i keys = _mm256_set_m128i(ax##suffix1, ax##suffix0); \
    __m256i result = aesenc_2way(data, keys); \
    cx##suffix0 = _mm256_castsi256_si128(result); \
    cx##suffix1 = _mm256_extracti128_si256(result, 1); \
  } \
  if constexpr (kPrefetch) { \
    sp_loc1##suffix0 = (__m128i *)(&l##suffix0[cx##suffix0[0] & kMask]); \
    sp_loc1##suffix1 = (__m128i *)(&l##suffix1[cx##suffix1[0] & kMask]); \
    PREFETCH_W(sp_loc1##suffix0); \
    PREFETCH_W(sp_loc1##suffix1); \
  }

#define AES_4WAY() \
  __m128i cx0, cx1, cx2, cx3; \
  aesenc_4way(&cx0, &cx1, &cx2, &cx3, \
              *sp_loc00, *sp_loc01, *sp_loc02, *sp_loc03, \
              ax0, ax1, ax2, ax3); \
  if constexpr (kPrefetch) { \
    sp_loc10 = (__m128i *)(&l0[cx0[0] & kMask]); \
    sp_loc11 = (__m128i *)(&l1[cx1[0] & kMask]); \
    sp_loc12 = (__m128i *)(&l2[cx2[0] & kMask]); \
    sp_loc13 = (__m128i *)(&l3[cx3[0] & kMask]); \
    PREFETCH_W(sp_loc10); \
    PREFETCH_W(sp_loc11); \
    PREFETCH_W(sp_loc12); \
    PREFETCH_W(sp_loc13); \
  }

// Requires 2x memory allocated in hp_state.
template <size_t kMemory, size_t kIterations, uint32_t kMask, size_t kShift,
          size_t kPrefetchW, size_t kPrefetchR, bool kL1Prefetch>
void cryptonight_2way_hash(const void *input0, const void *input1,
                           void *output0, void *output1) {
  uint8_t state[208 * 2] __attribute__((aligned(16)));
  CRYPTONIGHT_INIT(0);
  CRYPTONIGHT_INIT(1);

  explode_scratchpad<kShift, 2, kPrefetchW>(state, ((__m128i *)l0));
  CRYPTONIGHT_VARIABLES(0);
  CRYPTONIGHT_VARIABLES(1);
  constexpr bool kPrefetch = true;
  for (size_t i = 0; i < kIterations; ++i) {
    // AES
    AES_2WAY(0, 1);

    TWEAK(0);
    TWEAK(1);

    POST_AES(0);
    POST_AES(1);
  }

  implode_scratchpad<kMemory, kMemory != kShift, 2, kPrefetchR, kL1Prefetch>(
      ((const __m128i *)l0), state);
  CRYPTONIGHT_FINISH(0);
  CRYPTONIGHT_FINISH(1);
}

#ifdef __AVX2__

// Requires 4x memory allocated in hp_state.
template <size_t kMemory, size_t kIterations, uint32_t kMask, size_t kShift,
          size_t kPrefetchW, size_t kPrefetchR, bool kL1Prefetch>
void cryptonight_4way_hash(const void *input0, const void *input1,
                           const void *input2, const void *input3,
                           void *output0, void *output1, void *output2,
                           void *output3) {
  uint8_t state[208 * 4] __attribute__((aligned(16)));
  
  // Define the scratchpad pointers BEFORE using CRYPTONIGHT_VARIABLES
  uint8_t *__restrict__ l0 = (uint8_t *)(__builtin_assume_aligned(&hp_state[0 * kShift], 16));
  uint8_t *__restrict__ l1 = (uint8_t *)(__builtin_assume_aligned(&hp_state[1 * kShift], 16));
  uint8_t *__restrict__ l2 = (uint8_t *)(__builtin_assume_aligned(&hp_state[2 * kShift], 16));
  uint8_t *__restrict__ l3 = (uint8_t *)(__builtin_assume_aligned(&hp_state[3 * kShift], 16));
  
  // Process all 4 Keccak hashes in parallel
  keccak1600_4way((const uint8_t*)input0, (const uint8_t*)input1,
                  (const uint8_t*)input2, (const uint8_t*)input3,
                  &state[0], &state[208], &state[416], &state[624]);

  explode_scratchpad<kShift, 4, kPrefetchW>(state, ((__m128i *)l0));
  
  CRYPTONIGHT_VARIABLES(0);
  CRYPTONIGHT_VARIABLES(1);
  CRYPTONIGHT_VARIABLES(2);
  CRYPTONIGHT_VARIABLES(3);
  
  constexpr bool kPrefetch = true;
  for (size_t i = 0; i < kIterations; ++i) {
    // AES
    AES_4WAY();

    TWEAK(0);
    TWEAK(1);
    TWEAK(2);
    TWEAK(3);

    POST_AES(0);
    POST_AES(1);
    POST_AES(2);
    POST_AES(3);
  }

  implode_scratchpad<kMemory, kMemory != kShift, 4, kPrefetchR, kL1Prefetch>(
      ((const __m128i *)l0), state);

  keccakf_4way((uint64_t*)&state[0], (uint64_t*)&state[208],
               (uint64_t*)&state[416], (uint64_t*)&state[624]);

  extra_hashes[state[208 * 0] & 3](&state[208 * 0], output0);
  memset(&((uint8_t *)output0)[32], 0, 32);
  
  extra_hashes[state[208 * 1] & 3](&state[208 * 1], output1);
  memset(&((uint8_t *)output1)[32], 0, 32);
  
  extra_hashes[state[208 * 2] & 3](&state[208 * 2], output2);
  memset(&((uint8_t *)output2)[32], 0, 32);
  
  extra_hashes[state[208 * 3] & 3](&state[208 * 3], output3);
  memset(&((uint8_t *)output3)[32], 0, 32);
}

#endif // AVX2