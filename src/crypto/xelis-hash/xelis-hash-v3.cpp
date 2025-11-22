#include "xelis-hash.hpp"
#include <stdlib.h>
#include <iostream>
#include "aes.hpp"
#include "chacha20.h"
#include <crc32.h>

#include <BLAKE3/c/blake3.h>

#include <byteswap.h>
#include <chacha20.h>
#include <cmath> // NEW: needed for sqrt() in isqrt

#if defined(__x86_64__)
#include <emmintrin.h>
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#include <arm_acle.h>
#endif
#include <numeric>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <array>
#include <cassert>
#include <chrono>

#include "compile.h"

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

#define rl64(x, a) (((x << (a & 63)) | (x >> (64 - (a & 63)))))
#define rr64(x, a) (((x >> (a & 63)) | (x << (64 - (a & 63)))))

#define htonll(x) ((1 == htonl(1)) ? (x) : ((uint64_t)htonl((x) & 0xFFFFFFFF) << 32) | htonl((x) >> 32))
#define ntohll(x) ((1 == ntohl(1)) ? (x) : ((uint64_t)ntohl((x) & 0xFFFFFFFF) << 32) | ntohl((x) >> 32))

#define COMBINE_UINT64(high, low) (((__uint128_t)(high) << 64) | (low))

const int sign_bit_values_avx512[16][16] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

alignas(32) const int sign_bit_values_avx2[8][8] = {
    {0, 0, 0, 0, 0, 0, 0, -1},
    {0, 0, 0, 0, 0, 0, -1, 0},
    {0, 0, 0, 0, 0, -1, 0, 0},
    {0, 0, 0, 0, -1, 0, 0, 0},
    {0, 0, 0, -1, 0, 0, 0, 0},
    {0, 0, -1, 0, 0, 0, 0, 0},
    {0, -1, 0, 0, 0, 0, 0, 0},
    {-1, 0, 0, 0, 0, 0, 0, 0}};

alignas(16) const int sign_bit_values_sse[4][4] = {
    {0, 0, 0, -1},
    {0, 0, -1, 0},
    {0, -1, 0, 0},
    {-1, 0, 0, 0}};

static inline uint64_t swap_bytes(uint64_t value)
{
  return ((value & 0xFF00000000000000ULL) >> 56) |
         ((value & 0x00FF000000000000ULL) >> 40) |
         ((value & 0x0000FF0000000000ULL) >> 24) |
         ((value & 0x000000FF00000000ULL) >> 8) |
         ((value & 0x00000000FF000000ULL) << 8) |
         ((value & 0x0000000000FF0000ULL) << 24) |
         ((value & 0x000000000000FF00ULL) << 40) |
         ((value & 0x00000000000000FFULL) << 56);
}

static inline void blake3(const uint8_t *input, int len, uint8_t *output)
{
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, input, len);
  blake3_hasher_finalize(&hasher, output, BLAKE3_OUT_LEN);
}

#if !defined(__x86_64__)
static inline void aes_round(uint8_t *block, const uint8_t *key)
{
#if defined(__aarch64__) && defined(__AES__)
  uint8x16_t blck = vld1q_u8(block);
  uint8x16_t ky = vld1q_u8(key);
  // This magic sauce is from here: https://blog.michaelbrase.com/2018/06/04/optimizing-x86-aes-intrinsics-on-armv8-a/
  uint8x16_t rslt = vaesmcq_u8(vaeseq_u8(blck, (uint8x16_t){})) ^ ky;
  vst1q_u8(block, rslt);
#else
  aes_single_round_no_intrinsics(block, key);
#endif
}
#endif

#if defined(__x86_64__)

__attribute__((target("aes"))) inline void aes_round(uint8_t *block, const uint8_t *key)
{
  __m128i block_vec = _mm_loadu_si128((const __m128i *)block);
  __m128i key_vec = _mm_loadu_si128((const __m128i *)key);
  __m128i result = _mm_aesenc_si128(block_vec, key_vec);
  _mm_storeu_si128((__m128i *)block, result);
}

__attribute__((target("default"))) inline void aes_round(uint8_t *block, const uint8_t *key)
{
  aes_single_round_no_intrinsics(block, key);
}

#endif

const uint8_t chaIn[XELIS_MEMORY_SIZE_V3 * 2] = {0};

static void chacha_encrypt(uint8_t *key, uint8_t *nonce, uint8_t *in, uint8_t *out, size_t bytes, uint32_t rounds)
{
  uint8_t state[48] = {0};
  ChaCha20SetKey(state, key);
  ChaCha20SetNonce(state, nonce);
  ChaCha20EncryptBytes(state, in, out, bytes, rounds);
}

static inline void chacha_get_bytes_at_offset(const uint8_t key[32], const uint8_t nonce[12],
                                              size_t offset, uint8_t output[12], int rounds)
{
  uint8_t state[48] = {0};
  uint8_t block[64];

  ChaCha20SetKey(state, key);
  ChaCha20SetNonce(state, nonce);

  uint32_t block_num = offset / 64;
  uint32_t *counter = (uint32_t *)(state + 32);
  *counter = block_num;

  ChaCha20EncryptBytes(state, NULL, block, 64, rounds);

  size_t block_offset = offset % 64;
  memcpy(output, block + block_offset, 12);
}

// #ifdef __x86_64__
// TNN_TARGET_CLONE(
//   stage_1,
//   void,
//   (const uint8_t *input, uint64_t *sp, size_t input_len),
//   {
//     const size_t chunk_size = 32;
//     const size_t output_size = XELIS_MEMORY_SIZE_V2 * 8;
//     const size_t bytes_per_chunk = output_size / 4;

//     uint8_t *t = reinterpret_cast<uint8_t *>(sp);
//     uint8_t K2_values[4][32];
//     uint8_t nonces[4][12];
//     uint8_t buffer[64] = {0};
//     uint8_t key[128] = {0};

//     memcpy(key, input, input_len);

//     blake3(input, input_len, buffer);
//     memcpy(nonces[0], buffer, 12);

//     for (int i = 0; i < 4; i++) {
//       if (i == 0) {
//         memcpy(buffer + chunk_size, key, chunk_size);
//       } else {
//         memcpy(buffer, K2_values[i-1], chunk_size);
//         memcpy(buffer + chunk_size, key + i*chunk_size, chunk_size);
//       }
//       blake3(buffer, chunk_size*2, K2_values[i]);
//     }

//     for (int i = 0; i < 3; i++) {
//       chacha_get_bytes_at_offset(K2_values[i], nonces[i],
//                                   bytes_per_chunk - 12, nonces[i+1], 8);
//     }

//     byte* outputs[4];
//     for (int i = 0; i < 4; i++) {
//       outputs[i] = t + i*bytes_per_chunk;
//     }

//     ChaCha20EncryptXelis(K2_values, nonces, outputs, bytes_per_chunk, 8);
//   },
//   "ssse3", "avx2"
// )
// __attribute__((target("default")))
// #endif

#ifdef __x86_64__
TNN_TARGET_CLONE(
  stage_1,
  static void,
  (const uint8_t *input, uint64_t *sp, size_t input_len),
  {
    const size_t chunk_size = XELIS_CHUNK_SIZE;
    const size_t output_size = XELIS_OUTPUT_SIZE_V3;
    const size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3;

    uint8_t *t = reinterpret_cast<uint8_t *>(sp);
    uint8_t K2_values[4][32];
    uint8_t nonces[4][12];
    uint8_t buffer[64] = {0};
    uint8_t key[128] = {0};

    memcpy(key, input, input_len);

    blake3(input, input_len, buffer);
    memcpy(nonces[0], buffer, 12);

    for (int i = 0; i < 4; i++)
    {
      if (i == 0)
      {
        memcpy(buffer + chunk_size, key, chunk_size);
      }
      else
      {
        memcpy(buffer, K2_values[i - 1], chunk_size);
        memcpy(buffer + chunk_size, key + i * chunk_size, chunk_size);
      }
      blake3(buffer, chunk_size * 2, K2_values[i]);
    }

    for (int i = 0; i < 3; i++)
    {
      chacha_get_bytes_at_offset(K2_values[i], nonces[i],
                                  bytes_per_chunk - 12, nonces[i + 1], 8);
    }

    for (int i = 0; i < 4; i++)
    {
      uint8_t state[48] = {0};
      ChaCha20SetKey(state, K2_values[i]);
      ChaCha20SetNonce(state, nonces[i]);
      ChaCha20EncryptBytes(state, NULL, t + i * bytes_per_chunk, bytes_per_chunk, 8);
    }
  },
  TNN_TARGETS_X86_CHACHA512
)

__attribute__((target("default"))) static void stage_1(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  const size_t chunk_size = XELIS_CHUNK_SIZE;                  // 32
  const size_t output_size = XELIS_OUTPUT_SIZE_V3;             // XELIS_MEMORY_SIZE_V3 * 8
  constexpr size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3; // output_size / 4

  uint8_t *t = reinterpret_cast<uint8_t *>(sp);
  uint8_t K2_values[4][32];
  uint8_t nonces[4][12];
  uint8_t buffer[64] = {0};
  uint8_t key[128] = {0};

  memcpy(key, input, input_len);

  blake3(input, input_len, buffer);
  memcpy(nonces[0], buffer, 12);

  for (int i = 0; i < 4; i++)
  {
    if (i == 0)
    {
      memcpy(buffer + chunk_size, key, chunk_size);
    }
    else
    {
      memcpy(buffer, K2_values[i - 1], chunk_size);
      memcpy(buffer + chunk_size, key + i * chunk_size, chunk_size);
    }
    blake3(buffer, chunk_size * 2, K2_values[i]);
  }

  for (int i = 0; i < 3; i++)
  {
    chacha_get_bytes_at_offset(K2_values[i], nonces[i],
                               bytes_per_chunk - 12, nonces[i + 1], 8);
  }

  byte *outputs[4];
  for (int i = 0; i < 4; i++)
  {
    outputs[i] = t + i * bytes_per_chunk;
  }

  ChaCha20EncryptXelis(K2_values, nonces, outputs, bytes_per_chunk, 8);
}
#else
static void stage_1(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  const size_t chunk_size = XELIS_CHUNK_SIZE;
  const size_t output_size = XELIS_OUTPUT_SIZE_V3;
  const size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3;

  uint8_t *t = reinterpret_cast<uint8_t *>(sp);
  uint8_t K2_values[4][32];
  uint8_t nonces[4][12];
  uint8_t buffer[64] = {0};
  uint8_t key[128] = {0};

  memcpy(key, input, input_len);

  blake3(input, input_len, buffer);
  memcpy(nonces[0], buffer, 12);

  for (int i = 0; i < 4; i++)
  {
    if (i == 0)
    {
      memcpy(buffer + chunk_size, key, chunk_size);
    }
    else
    {
      memcpy(buffer, K2_values[i - 1], chunk_size);
      memcpy(buffer + chunk_size, key + i * chunk_size, chunk_size);
    }
    blake3(buffer, chunk_size * 2, K2_values[i]);
  }

  for (int i = 0; i < 3; i++)
  {
    chacha_get_bytes_at_offset(K2_values[i], nonces[i],
                               bytes_per_chunk - 12, nonces[i + 1], 8);
  }

  for (int i = 0; i < 4; i++)
  {
    uint8_t state[48] = {0};
    ChaCha20SetKey(state, K2_values[i]);
    ChaCha20SetNonce(state, nonces[i]);
    ChaCha20EncryptBytes(state, NULL, t + i * bytes_per_chunk, bytes_per_chunk, 8);
  }
}
#endif

// MurmurHash3 finalizer - used for randomization
static inline uint64_t murmurhash3(uint64_t seed)
{
  seed ^= seed >> 55;
  seed *= XELIS_MURMUR_CONST1;
  seed ^= seed >> 32;
  seed *= XELIS_MURMUR_CONST2;
  seed ^= seed >> 15;
  return seed;
}

// Map index - converts hash to buffer index (0 to BUFSIZE-1)
static inline uint64_t map_index(uint64_t x)
{
  x ^= x >> 33;
  x *= XELIS_MURMUR_CONST1;
  return (uint64_t)(((__uint128_t)x * XELIS_BUFFER_SIZE_V3) >> 64);
}

// Pick which half buffer to use for writes
static inline int pick_half(uint64_t seed)
{
  return (murmurhash3(seed) & XELIS_PICK_HALF_BIT) != 0;
}

// Modular exponentiation - used for address update in v3
static inline uint64_t modular_power(uint64_t base, uint64_t exp, uint64_t mod)
{
  mod += (mod == 0);

  uint64_t result = 1;
  base %= mod;

  while (exp > 0)
  {
    if (exp & 1)
    {
      result = (uint64_t)(((__uint128_t)result * base) % mod);
    }
    base = (uint64_t)(((__uint128_t)base * base) % mod);
    exp >>= 1;
  }

  return result;
}

alignas(64) static const uint8_t isqrt_lut[256] = {
    0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};

#ifdef __x86_64__
__attribute__((target("sse2"))) static inline uint64_t isqrt(uint64_t n)
{
  if (n < 2)
    return n;

  __m128d v = _mm_set_sd((double)n);
  __m128d s = _mm_sqrt_sd(v, v);
  uint64_t root = (uint64_t)_mm_cvtsd_f64(s);
  if ((root + 1) * (root + 1) <= n)
    ++root;
  else if (root * root > n)
    --root;
  return root;
}

__attribute__((target("default"))) static inline uint64_t isqrt(uint64_t n)
{
  if (n < 2)
  {
    return n;
  }

  uint64_t x = n;
  uint64_t y = (x + 1) >> 1;

  while (y < x)
  {
    x = y;
    y = (x + n / x) >> 1;
  }

  return x;
}
#endif
#ifdef __aarch64__
static inline uint64_t isqrt(uint64_t n)
{
  if (n < 2)
    return n;

  float64x1_t v = vdup_n_f64((double)n);
  float64x1_t s = vsqrt_f64(v);
  double droot;
  vst1_f64(&droot, s);
  uint64_t root = (uint64_t)droot;
  if ((root + 1) * (root + 1) <= n)
    ++root;
  else if (root * root > n)
    --root;
  return root;
}
#endif

static inline __uint128_t combine_uint64(uint64_t high, uint64_t low)
{
  return ((__uint128_t)high << 64) | low;
}

#if defined(__AVX2__)
__attribute__((target("avx2"))) void static inline uint64_to_le_bytes(uint64_t value, uint8_t *bytes)
{
  _mm_storel_epi64((__m128i *)bytes, _mm_shuffle_epi8(_mm_set1_epi64x(value), _mm_set_epi8(
                                                                                  -1, -1, -1, -1, -1, -1, -1, -1,
                                                                                  7, 6, 5, 4, 3, 2, 1, 0)));
}

__attribute__((target("avx2")))
uint64_t static inline le_bytes_to_uint64(const uint8_t *bytes)
{
  return _mm_cvtsi128_si64(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)bytes), _mm_set_epi8(
                                                                                         15, 14, 13, 12, 11, 10, 9, 8,
                                                                                         7, 6, 5, 4, 3, 2, 1, 0)));
}
#endif

#if defined(__x86_64__)
__attribute__((target("default"))) void static inline uint64_to_le_bytes(uint64_t value, uint8_t *bytes)
{
  for (int i = 0; i < 8; i++)
  {
    bytes[i] = value & 0xFF;
    value >>= 8;
  }
}

__attribute__((target("default")))
uint64_t static inline le_bytes_to_uint64(const uint8_t *bytes)
{
  uint64_t value = 0;
  for (int i = 7; i >= 0; i--)
    value = (value << 8) | bytes[i];
  return value;
}
#elif defined(__aarch64__)
#if defined(__ARM_NEON) || defined(__aarch64__)
static inline void uint64_to_le_bytes(uint64_t value, uint8_t *bytes)
{
  uint8x8_t val = vreinterpret_u8_u64(vdup_n_u64(value));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  val = vrev64_u8(val);
#endif

  vst1_u8(bytes, val);
}

static inline uint64_t le_bytes_to_uint64(const uint8_t *bytes)
{
  uint8x8_t val = vld1_u8(bytes);

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  val = vrev64_u8(val);
#endif

  return vget_lane_u64(vreinterpret_u64_u8(val), 0);
}
#else
void static inline uint64_to_le_bytes(uint64_t value, uint8_t *bytes)
{
  for (int i = 0; i < 8; i++)
  {
    bytes[i] = value & 0xFF;
    value >>= 8;
  }
}

uint64_t static inline le_bytes_to_uint64(const uint8_t *bytes)
{
  uint64_t value = 0;
  for (int i = 7; i >= 0; i--)
    value = (value << 8) | bytes[i];
  return value;
}
#endif
#endif
#if defined(__x86_64__)
void static inline aes_single_round(uint8_t *block, const uint8_t *key)
{
  // Perform single AES encryption round
  __m128i block_vec = _mm_aesenc_si128(_mm_loadu_si128((const __m128i *)block), _mm_loadu_si128((const __m128i *)key));
  _mm_storeu_si128((__m128i *)block, block_vec);
}

static inline uint64_t ROTR(uint64_t x, uint32_t r)
{
  asm("rorq %%cl, %0" : "+r"(x) : "c"(r));
  return x;
}

static inline uint64_t ROTL(uint64_t x, uint32_t r)
{
  asm("rolq %%cl, %0" : "+r"(x) : "c"(r));
  return x;
}

#else // aarch64

static inline uint64_t ROTR(uint64_t x, uint32_t r)
{
  return __builtin_rotateright64(x, r);
}

static inline uint64_t ROTL(uint64_t x, uint32_t r)
{
  return __builtin_rotateleft64(x, r);
}

#endif

static inline uint64_t d128_64(__uint128_t a, uint64_t b)
{
  uint64_t dividend_hi = a >> 64;
  uint64_t dividend_lo = (uint64_t)a;

#if defined(__x86_64__)
  uint64_t q_hi = 0, q_lo = 0, remainder = 0;

  if (dividend_hi < b)
  {
    __asm__("divq %[divisor]"
            : "=a"(q_lo), "=d"(remainder)
            : [divisor] "r"(b), "a"(dividend_lo), "d"(dividend_hi)
            : "cc");
  }
  else
  {
    __asm__("divq %[divisor]"
            : "=a"(q_hi), "=d"(dividend_hi)
            : [divisor] "r"(b), "a"(dividend_hi), "d"(0ULL)
            : "cc");

    __asm__("divq %[divisor]"
            : "=a"(q_lo), "=d"(remainder)
            : [divisor] "r"(b), "a"(dividend_lo), "d"(dividend_hi)
            : "cc");
  }

  return q_lo;
#else
  return (uint64_t)(a / b);
#endif
}

static inline __uint128_t d128(__uint128_t a, uint64_t b)
{
  uint64_t dividend_hi = a >> 64;
  uint64_t dividend_lo = (uint64_t)a;

#if defined(__x86_64__)
  uint64_t q_hi = 0, q_lo = 0, remainder = 0;

  if (dividend_hi < b)
  {
    __asm__("divq %[divisor]"
            : "=a"(q_lo), "=d"(remainder)
            : [divisor] "r"(b), "a"(dividend_lo), "d"(dividend_hi)
            : "cc");
  }
  else
  {
    __asm__("divq %[divisor]"
            : "=a"(q_hi), "=d"(dividend_hi)
            : [divisor] "r"(b), "a"(dividend_hi), "d"(0ULL)
            : "cc");

    __asm__("divq %[divisor]"
            : "=a"(q_lo), "=d"(remainder)
            : [divisor] "r"(b), "a"(dividend_lo), "d"(dividend_hi)
            : "cc");
  }

  return ((__uint128_t)q_hi << 64) | q_lo;
#else
  return a / b;
#endif
}

static inline uint64_t mod128_64(__uint128_t dividend, uint64_t divisor)
{
  uint64_t dividend_hi = dividend >> 64;
  uint64_t dividend_lo = (uint64_t)dividend;

#if defined(__x86_64__)
  uint64_t remainder = 0;
  uint64_t dummy_quotient;

  if (dividend_hi < divisor)
  {
    __asm__("divq %[divisor]"
            : "=a"(dummy_quotient), "=d"(remainder)
            : [divisor] "r"(divisor), "a"(dividend_lo), "d"(dividend_hi)
            : "cc");
  }
  else
  {
    __asm__("divq %[divisor]"
            : "=a"(dummy_quotient), "=d"(dividend_hi)
            : [divisor] "r"(divisor), "a"(dividend_hi), "d"(0ULL)
            : "cc");

    __asm__("divq %[divisor]"
            : "=a"(dummy_quotient), "=d"(remainder)
            : [divisor] "r"(divisor), "a"(dividend_lo), "d"(dividend_hi)
            : "cc");
  }

  return remainder;
#else
  return (uint64_t)(dividend % divisor);
#endif
}

static inline uint64_t mod128_64_fast(__uint128_t t1, uint64_t denom)
{
#if defined(__x86_64__)
  uint64_t hi = t1 >> 64;
  uint64_t lo = (uint64_t)t1;

  if (hi < denom)
  {
    uint64_t remainder;
    __asm__("divq %[d]"
            : "=d"(remainder)
            : [d] "r"(denom), "a"(lo), "d"(hi)
            : "cc");
    return remainder;
  }
#endif
  return mod128_64(t1, denom);
}

// Very cool, very performant, but unnecessary for Xelis given inter-operand relations from the ChaCha dataset
// #ifdef __x86_64__

// #ifndef TNN_LEGACY_AMD64
// #define div_targets_extra , TNN_TARGETS_X86_AVX2, TNN_TARGETS_X86_AVX512
// #else
// #define div_targets_extra
// #endif
// TNN_TARGET_CLONE(
//   div128_128_clz_improved,
//   static inline __uint128_t,
//   (__uint128_t dividend, __uint128_t divisor),
//   {
//     if (divisor > dividend) return 0;
//     if (divisor == dividend) return 1;
//     if ((divisor >> 64) == 0) return d128(dividend, (uint64_t)divisor);

//     int dividend_clz = (dividend >> 64) ? __builtin_clzll(dividend >> 64) :
//                       64 + __builtin_clzll((uint64_t)dividend);
//     int divisor_clz = (divisor >> 64) ? __builtin_clzll(divisor >> 64) :
//                       64 + __builtin_clzll((uint64_t)divisor);

//     int shift = divisor_clz - dividend_clz;
//     if (shift < 0) return 0;

//     __uint128_t shifted_divisor = divisor << shift;
//     uint64_t quotient = 0;

//     if (shift >= 64) {
//       uint64_t div_hi = shifted_divisor >> 64;
//       uint64_t div_lo = (uint64_t)shifted_divisor;
//       uint64_t rem_hi = dividend >> 64;
//       uint64_t rem_lo = (uint64_t)dividend;

//       for (int i = 0; i <= shift; i++) {
//         quotient <<= 1;

//         if (rem_hi > div_hi || (rem_hi == div_hi && rem_lo >= div_lo)) {
//           __m128i rem = _mm_set_epi64x(rem_hi, rem_lo);
//           __m128i div = _mm_set_epi64x(div_hi, div_lo);

//           uint64_t borrow = (rem_lo < div_lo) ? 1 : 0;
//           rem_lo -= div_lo;
//           rem_hi -= div_hi + borrow;

//           quotient |= 1;
//         }

//         div_lo = (div_lo >> 1) | (div_hi << 63);
//         div_hi >>= 1;
//       }

//       dividend = ((__uint128_t)rem_hi << 64) | rem_lo;
//     } else {
//       for (int i = 0; i <= shift; i++) {
//         quotient <<= 1;
//         if (dividend >= shifted_divisor) {
//           dividend -= shifted_divisor;
//           quotient |= 1;
//         }
//         shifted_divisor >>= 1;
//       }
//     }

//     return quotient;
//   },
//   "sse2", "ssse3" div_targets_extra
// )
// __attribute__((target("default")))
// #endif
// static inline __uint128_t div128_128_clz_improved(__uint128_t dividend, __uint128_t divisor) {
//   if (divisor > dividend) return 0;
//   if (divisor == dividend) return 1;
//   if ((divisor >> 64) == 0) return d128(dividend, (uint64_t)divisor);

//   int dividend_clz = (dividend >> 64) ? __builtin_clzll(dividend >> 64) :
//                     64 + __builtin_clzll((uint64_t)dividend);
//   int divisor_clz = (divisor >> 64) ? __builtin_clzll(divisor >> 64) :
//                     64 + __builtin_clzll((uint64_t)divisor);

//   int shift = divisor_clz - dividend_clz;
//   if (shift < 0) return 0;

//   __uint128_t shifted_divisor = divisor << shift;
//   uint64_t quotient = 0;

//   int chunks = (shift + 1) / 8;
//   int remaining = (shift + 1) % 8;

//   for (int chunk = 0; chunk < chunks; chunk++) {
//     uint64_t chunk_quotient = 0;
//     uint64_t chunk_divisor = shifted_divisor >> (8 * chunk);

//     for (int bit = 0; bit < 8; bit++) {
//       chunk_quotient <<= 1;
//       if (dividend >= chunk_divisor) {
//         dividend -= chunk_divisor;
//         chunk_quotient |= 1;
//       }
//       chunk_divisor >>= 1;
//     }

//     quotient = (quotient << 8) | chunk_quotient;
//   }

//   if (remaining > 0) {
//     uint64_t remaining_quotient = 0;
//     uint64_t remaining_divisor = shifted_divisor >> (8 * chunks);

//     for (int bit = 0; bit < remaining; bit++) {
//       remaining_quotient <<= 1;
//       if (dividend >= remaining_divisor) {
//         dividend -= remaining_divisor;
//         remaining_quotient |= 1;
//       }
//       remaining_divisor >>= 1;
//     }

//     quotient = (quotient << remaining) | remaining_quotient;
//   }

//   return quotient;
// }

static inline uint64_t div128_128_large(__uint128_t *dividend, __uint128_t divisor)
{
  uint64_t div_hi = divisor >> 64;

  // If high part of divisor is 0, use 128/64 division instead
  if (div_hi == 0)
  {
    uint64_t div_lo = (uint64_t)divisor;
    if (div_lo == 0)
      return 0; // Prevent div by zero
    __uint128_t q = d128(*dividend, div_lo);
    *dividend = *dividend - q * divisor;
    return (uint64_t)q;
  }

  uint64_t quotient = 0;
  uint64_t dividend_hi = *dividend >> 64;

  if (dividend_hi >= div_hi)
  {
    uint64_t q_hi = dividend_hi / div_hi;
    if (q_hi > 0)
    {
      __uint128_t sub = divisor * q_hi;
      if (sub <= *dividend)
      {
        *dividend -= sub;
        quotient += q_hi;
      }
      else
      {
        q_hi--;
        if (q_hi > 0)
        {
          sub = divisor * q_hi;
          *dividend -= sub;
          quotient += q_hi;
        }
      }
    }
  }

  return quotient;
}

static inline uint64_t udiv(uint64_t high, uint64_t low, uint64_t divisor)
{
  __uint128_t dividend = ((__uint128_t)high << 64) | low;
  return d128_64(dividend, divisor);
}

// __attribute__((noinline))
static inline uint64_t case_0(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  __uint128_t t1 = combine_uint64(a + i, isqrt(b + j));
  uint64_t denom = murmurhash3(c ^ result ^ i ^ j) | 1;
  return (uint64_t)(t1 % denom);
}
// __attribute__((noinline))
static inline uint64_t case_1(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return ROTL((c + i) % isqrt(b | 2), i + j) * isqrt(a + j);
}
// __attribute__((noinline))
static inline uint64_t case_2(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return (isqrt(a + i) * isqrt(c + j)) ^ (b + i + j);
}
// __attribute__((noinline))
static inline uint64_t case_3(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return (a + b) * c;
}
// __attribute__((noinline))
static inline uint64_t case_4(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return (b - c) * a;
}
// __attribute__((noinline))
static inline uint64_t case_5(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return c - a + b;
}
// __attribute__((noinline))
static inline uint64_t case_6(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return a - b + c;
}
// __attribute__((noinline))
static inline uint64_t case_7(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return b * c + a;
}
// __attribute__((noinline))
static inline uint64_t case_8(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return c * a + b;
}
// __attribute__((noinline))
static inline uint64_t case_9(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return a * b * c;
}
// __attribute__((noinline))
static inline uint64_t case_10(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return mod128_64_fast(COMBINE_UINT64(a, b), c | 1);
}
// __attribute__((noinline))
static inline uint64_t case_11(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  uint64_t t2_hi = ROTL(result, r);
  uint64_t t2_lo = a | 2;

  if (t2_hi > b || (t2_hi == b && t2_lo > c))
  {
    return c;
  }

  if (t2_hi == 0)
  {
    return mod128_64_fast(COMBINE_UINT64(b, c), t2_lo);
  }

  __uint128_t t2 = COMBINE_UINT64(t2_hi, t2_lo);
  __uint128_t dividend = COMBINE_UINT64(b, c);

  uint64_t quotient = div128_128_large(&dividend, t2);
  return (uint64_t)(dividend - quotient * t2);
}
// __attribute__((noinline))
static inline uint64_t case_12(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  return udiv(c, a, b | 4);
}
// __attribute__((noinline))
static inline uint64_t case_13(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  uint64_t t1_hi = ROTL(result, r);
  uint64_t t1_lo = b;
  uint64_t t2_hi = a;
  uint64_t t2_lo = c | 8;

  if (t1_hi > t2_hi || (t1_hi == t2_hi && t1_lo > t2_lo))
  {
    __uint128_t dividend = ((__uint128_t)t1_hi << 64) | t1_lo;
    __uint128_t divisor = ((__uint128_t)t2_hi << 64) | t2_lo;
    return div128_128_large(&dividend, divisor);
  }
  else
  {
    return a ^ b;
  }
}
// __attribute__((noinline))
static inline uint64_t case_14(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  __uint128_t ac = (__uint128_t)a * c;
  uint64_t ac_hi = ac >> 64;
  uint64_t bc = b * c;
  return ac_hi + bc;
}
// __attribute__((noinline))
static inline uint64_t case_15(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j)
{
  uint64_t b1 = ROTR(result, r);
  uint64_t b0 = b;

  __uint128_t z0 = (__uint128_t)c * b0;
  __uint128_t z1 = (__uint128_t)a * b0 + (__uint128_t)c * b1;

  uint64_t z0_hi = z0 >> 64;
  uint64_t z1_lo = (uint64_t)z1;

  return z0_hi + z1_lo;
}

typedef uint64_t (*operation_func)(uint64_t, uint64_t, uint64_t, int, uint64_t, int, int);

static operation_func operations[] = {
    case_0,
    case_1,
    case_2,
    case_3,
    case_4,
    case_5,
    case_6,
    case_7,
    case_8,
    case_9,
    case_10,
    case_11,
    case_12,
    case_13,
    case_14,
    case_15,
};

static inline uint64_t execute_operation_goto(uint32_t idx, uint64_t a, uint64_t b,
                                              uint64_t c, int r_next, uint64_t result,
                                              int i, int j_off)
{
  uint64_t v;

  static void *const dispatch_table[] = {
      &&op0, &&op1, &&op2, &&op3, &&op4, &&op5, &&op6, &&op7,
      &&op8, &&op9, &&op10, &&op11, &&op12, &&op13, &&op14, &&op15};

  void *target = (void *)dispatch_table[idx];
  goto *target;

op0:
{
  __uint128_t t1 = combine_uint64(a + i, isqrt(b + j_off));
  uint64_t denom = murmurhash3(c ^ result ^ i ^ j_off) | 1;
  v = (uint64_t)(t1 % denom);
  goto done;
}
op1:
{
  uint64_t input_b = b | 2;
  uint64_t input_a = a + j_off;

  uint64_t sqrt_b, sqrt_a;
  sqrt_b = isqrt(input_b);
  sqrt_a = isqrt(input_a);

  v = ROTL((c + i) % sqrt_b, i + j_off) * sqrt_a;
  goto done;
}
op2:
{
  v = (isqrt(a + i) * isqrt(c + j_off)) ^ (b + i + j_off);
  goto done;
}
op3:
  v = (a + b) * c;
  goto done;
op4:
  v = (b - c) * a;
  goto done;
op5:
  v = c - a + b;
  goto done;
op6:
  v = a - b + c;
  goto done;
op7:
  v = b * c + a;
  goto done;
op8:
  v = c * a + b;
  goto done;
op9:
  v = a * b * c;
  goto done;
op10:
  v = mod128_64_fast(COMBINE_UINT64(a, b), c | 1);
  goto done;
op11:
{
  uint64_t t2_hi = ROTL(result, r_next);
  uint64_t t2_lo = a | 2;
  if (t2_hi > b || (t2_hi == b && t2_lo > c))
  {
    v = c;
  }
  else
  {
    __uint128_t t2 = COMBINE_UINT64(t2_hi, t2_lo);
    __uint128_t dividend = COMBINE_UINT64(b, c);
    div128_128_large(&dividend, t2);
    v = dividend;
  }
  goto done;
}
op12:
  v = udiv(c, a, b | 4);
  goto done;
op13:
{
  uint64_t t1_hi = ROTL(result, r_next);
  uint64_t t1_lo = b;
  uint64_t t2_hi = a;
  uint64_t t2_lo = c | 8;
  if (t1_hi > t2_hi || (t1_hi == t2_hi && t1_lo > t2_lo))
  {
    __uint128_t dividend = ((__uint128_t)t1_hi << 64) | t1_lo;
    __uint128_t divisor = ((__uint128_t)t2_hi << 64) | t2_lo;
    v = div128_128_large(&dividend, divisor);
  }
  else
  {
    v = a ^ b;
  }
  goto done;
}
op14:
{
  __uint128_t ac = (__uint128_t)a * c;
  uint64_t ac_hi = ac >> 64;
  uint64_t bc = b * c;
  v = ac_hi + bc;
  goto done;
}
op15:
{
  uint64_t b1 = ROTR(result, r_next);
  uint64_t b0 = b;
  __uint128_t z0 = (__uint128_t)c * b0;
  __uint128_t z1 = (__uint128_t)a * b0 + (__uint128_t)c * b1;
  uint64_t z0_hi = z0 >> 64;
  uint64_t z1_lo = (uint64_t)z1;
  v = z0_hi + z1_lo;
  goto done;
}

done:
  return v;
}

static inline int pick_half_fast(uint64_t v)
{
  v ^= v >> 55;
  v *= XELIS_MURMUR_CONST1;
  v ^= v >> 32;
  return (v & XELIS_PICK_HALF_BIT) != 0;
}

#ifdef __x86_64__

__attribute__((target("aes"))) static void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  const uint8_t key[17] = "xelishash-pow-v3";
  __m128i key_vec = _mm_loadu_si128((const __m128i *)key);

  uint64_t *__restrict mem_buffer_a = scratch_pad;
  uint64_t *__restrict mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V3;

  uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
  uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;

  size_t r = 0;

  for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; ++i)
  {
    uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
    uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];

    __m128i block_vec = _mm_set_epi64x(mem_a, mem_b);
    block_vec = _mm_aesenc_si128(block_vec, key_vec);
    uint64_t hash1 = _mm_extract_epi64(block_vec, 0);
    uint64_t hash2 = _mm_extract_epi64(block_vec, 1);

    uint64_t result = ~(hash1 ^ hash2);

    for (size_t j = 0; j < XELIS_BUFFER_SIZE_V3; ++j)
    {
      uint64_t a = mem_buffer_a[map_index(result)];
      uint64_t b = mem_buffer_b[map_index(a ^ ~ROTR(result, r))];
      uint64_t c = scratch_pad[r];

      r = (r + 1) % XELIS_MEMORY_SIZE_V3;

      uint32_t op_idx = ROTL(result, (uint32_t)c) & 0xF;
      uint64_t v = execute_operation_goto(op_idx, a, b, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      int use_buffer_b = pick_half(v);
      uint64_t idx_t = map_index(idx_seed);
      uint64_t t = (use_buffer_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

      uint64_t idx_a = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
      uint64_t idx_b = map_index(idx_a ^ ~result ^ XELIS_SCATTER_CONST);

      // __builtin_prefetch(&mem_buffer_a[idx_a], 1, 0);
      // __builtin_prefetch(&mem_buffer_b[idx_b], 1, 0);

      uint64_t mem_a_tmp = mem_buffer_a[idx_a];
      mem_buffer_a[idx_a] = t;
      mem_buffer_b[idx_b] ^= mem_a_tmp ^ ROTR(t, i + j);
    }

    addr_a = modular_power(addr_a, addr_b, result);
    addr_b = isqrt(result) * (r + 1) * isqrt(addr_a);
  }
}

__attribute__((target("default"))) static void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  const uint8_t key[17] = "xelishash-pow-v3";
  uint8_t block[16];

  uint64_t *__restrict mem_buffer_a = scratch_pad;
  uint64_t *__restrict mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V3;

  uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
  uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;

  size_t r = 0;

  for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; ++i)
  {
    uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
    uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];

    uint64_to_le_bytes(mem_b, block);
    uint64_to_le_bytes(mem_a, block + 8);
    aes_round(block, key);
    uint64_t hash1 = le_bytes_to_uint64(block);
    uint64_t hash2 = le_bytes_to_uint64(block + 8);

    uint64_t result = ~(hash1 ^ hash2);

    for (size_t j = 0; j < XELIS_BUFFER_SIZE_V3; ++j)
    {
      uint64_t a = mem_buffer_a[map_index(result)];
      uint64_t b = mem_buffer_b[map_index(a ^ ~ROTR(result, r))];
      uint64_t c = scratch_pad[r];

      r = (r + 1) % XELIS_MEMORY_SIZE_V3;

      uint32_t op_idx = ROTL(result, (uint32_t)c) & 0xF;
      uint64_t v = execute_operation_goto(op_idx, a, b, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      int use_buffer_b = pick_half(v);
      uint64_t idx_t = map_index(idx_seed);
      uint64_t t = (use_buffer_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

      uint64_t idx_a = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
      uint64_t idx_b = map_index(idx_a ^ ~result ^ XELIS_SCATTER_CONST);

      // __builtin_prefetch(&mem_buffer_a[idx_a], 1, 0);
      // __builtin_prefetch(&mem_buffer_b[idx_b], 1, 0);

      uint64_t mem_a_tmp = mem_buffer_a[idx_a];
      mem_buffer_a[idx_a] = t;
      mem_buffer_b[idx_b] ^= mem_a_tmp ^ ROTR(t, i + j);
    }

    addr_a = modular_power(addr_a, addr_b, result);
    addr_b = isqrt(result) * (r + 1) * isqrt(addr_a);
  }
}

#else

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_CRYPTO)

static inline uint64x2_t aes_round_v3_register(uint64_t mem_a, uint64_t mem_b,
                                               const uint8x16_t key_vec)
{
  uint64x2_t input_u64 = {mem_b, mem_a};
  uint8x16_t block = vreinterpretq_u8_u64(input_u64);

  block = vaeseq_u8(block, vdupq_n_u8(0));
  block = vaesmcq_u8(block);
  block = veorq_u8(block, key_vec);

  return vreinterpretq_u64_u8(block);
}

static void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  const uint8_t key[17] = "xelishash-pow-v3";

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_CRYPTO)
  uint8x16_t key_vec = vld1q_u8(key);
#endif

  uint64_t *__restrict mem_buffer_a = scratch_pad;
  uint64_t *__restrict mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V3;

  uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
  uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;

  size_t r = 0;

  for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; ++i)
  {
    uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
    uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];

    uint64_t hash1, hash2;

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_CRYPTO)
    {
      uint64x2_t result_u64 = aes_round_v3_register(mem_a, mem_b, key_vec);
      hash1 = vgetq_lane_u64(result_u64, 0);
      hash2 = vgetq_lane_u64(result_u64, 1);
    }
#else
    uint8_t block_bytes[16];
    uint64_to_le_bytes(mem_b, block_bytes);
    uint64_to_le_bytes(mem_a, block_bytes + 8);
    aes_round(block_bytes, key);
    hash1 = le_bytes_to_uint64(block_bytes);
    hash2 = le_bytes_to_uint64(block_bytes + 8);
#endif

    uint64_t result = ~(hash1 ^ hash2);

    for (size_t j = 0; j < XELIS_BUFFER_SIZE_V3; ++j)
    {
      uint64_t a = mem_buffer_a[map_index(result)];
      uint64_t b = mem_buffer_b[map_index(a ^ ~ROTR(result, r))];
      uint64_t c = scratch_pad[r];

      r = (r + 1) % XELIS_MEMORY_SIZE_V3;

      uint32_t op_idx = ROTL(result, (uint32_t)c) & 0xF;
      uint64_t v = execute_operation_goto(op_idx, a, b, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      int use_buffer_b = pick_half(v);
      uint64_t idx_t = map_index(idx_seed);
      uint64_t t = (use_buffer_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

      uint64_t idx_a = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
      uint64_t idx_b = map_index(idx_a ^ ~result ^ XELIS_SCATTER_CONST);

      // __builtin_prefetch(&mem_buffer_a[idx_a], 1, 0);
      // __builtin_prefetch(&mem_buffer_b[idx_b], 1, 0);

      uint64_t mem_a_tmp = mem_buffer_a[idx_a];
      mem_buffer_a[idx_a] = t;
      mem_buffer_b[idx_b] ^= mem_a_tmp ^ ROTR(t, i + j);
    }

    addr_a = modular_power(addr_a, addr_b, result);
    addr_b = isqrt(result) * (r + 1) * isqrt(addr_a);
  }
}
#endif
#endif

void xelis_hash_v3(byte *input, workerData_xelis_v3 &worker, byte *hashResult)
{
  stage_1(input, worker.scratchPad, 112);
  stage_3(worker.scratchPad, worker);
  blake3((uint8_t *)worker.scratchPad, XELIS_OUTPUT_SIZE_V3, hashResult);
}

void xelis_benchmark_cpu_hash_v3()
{
  const uint32_t ITERATIONS = 10000;
  byte input[112] = {0};
  workerData_xelis_v3 worker; // V3 worker type
  byte hash_result[XELIS_HASH_SIZE] = {0};

  printf("v3 bench\n");
  fflush(stdout);

  auto start = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < ITERATIONS; ++i)
  {
    memset(worker.scratchPad, 0, XELIS_MEMORY_SIZE_V3 * 8); // V3 memory size
    xelis_hash_v3(input, worker, hash_result);
  }
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time took: " << elapsed.count() << " ms" << std::endl;
  std::cout << "H/s: " << ((ITERATIONS * 1000.0) / elapsed.count()) << std::endl;
  std::cout << "ms per hash: " << (elapsed.count() / ITERATIONS) << std::endl;
  fflush(stdout);

  for (int i = 0; i < 32; i++)
  {
    printf("%02x", hash_result[i]);
  }
  printf("\n");
  fflush(stdout);
}

namespace xelis_tests_v3
{
  using Hash = std::array<byte, XELIS_HASH_SIZE>;

  bool test_input(const char *test_name, byte *input, size_t input_size, const Hash &expected_hash)
  {
    auto worker = std::make_unique<workerData_xelis_v3>();
    byte hash_result[XELIS_HASH_SIZE] = {0};

    xelis_hash_v3(input, *worker, hash_result);

    if (std::memcmp(hash_result, expected_hash.data(), XELIS_HASH_SIZE) != 0)
    {
      std::cout << "Test '" << test_name << "' failed!" << std::endl;
      std::cout << "Expected hash: ";
      for (size_t i = 0; i < XELIS_HASH_SIZE; ++i)
      {
        printf("%02x", expected_hash[i]);
      }
      std::cout << std::endl;
      std::cout << "Actual hash:   ";
      for (size_t i = 0; i < XELIS_HASH_SIZE; ++i)
      {
        printf("%02x", hash_result[i]);
      }
      std::cout << std::endl;
      fflush(stdout);
      return false;
    }
    return true;
  }

  bool test_reused_scratchpad()
  {
    workerData_xelis_v3 worker;
    byte input[112];

    // Fill with random data
    srand(time(NULL));
    for (int i = 0; i < 112; i++)
    {
      input[i] = rand() & 0xFF;
    }

    byte expected_hash[XELIS_HASH_SIZE];

    // First hash
    xelis_hash_v3(input, worker, expected_hash);

    // Second hash with dirty scratch pad but same input
    byte hash_result[XELIS_HASH_SIZE];
    xelis_hash_v3(input, worker, hash_result);

    if (std::memcmp(hash_result, expected_hash, XELIS_HASH_SIZE) != 0)
    {
      std::cout << "test_reused_scratchpad failed!" << std::endl;
      return false;
    }

    std::cout << "test_reused_scratchpad passed!" << std::endl;
    return true;
  }

  bool test_zero_hash()
  {
    alignas(32) byte input[112] = {0};
    Hash expected_hash = {
        105, 172, 103, 40, 94, 253, 92, 162,
        42, 252, 5, 196, 236, 238, 91, 218,
        22, 157, 228, 233, 239, 8, 250, 57,
        212, 166, 121, 132, 148, 205, 103, 163};

    return test_input("test_zero_hash", input, sizeof(input), expected_hash);
  }

  bool test_verify_output()
  {
    byte input[] = {
        172, 236, 108, 212, 181, 31, 109, 45, 44, 242, 54, 225, 143, 133,
        89, 44, 179, 108, 39, 191, 32, 116, 229, 33, 63, 130, 33, 120, 185, 89,
        146, 141, 10, 79, 183, 107, 238, 122, 92, 222, 25, 134, 90, 107, 116,
        110, 236, 53, 255, 5, 214, 126, 24, 216, 97, 199, 148, 239, 253, 102,
        199, 184, 232, 253, 158, 145, 86, 187, 112, 81, 78, 70, 80, 110, 33,
        37, 159, 233, 198, 1, 178, 108, 210, 100, 109, 155, 106, 124, 124, 83,
        89, 50, 197, 115, 231, 32, 74, 2, 92, 47, 25, 220, 135, 249, 122,
        172, 220, 137, 143, 234, 68, 188};

    Hash expected_hash = {
        242, 8, 176, 222, 203, 27, 104,
        187, 22, 40, 68, 73, 79, 79, 65,
        83, 138, 101, 10, 116, 194, 41, 153,
        21, 92, 163, 12, 206, 231, 156, 70, 83};

    return test_input("test_verify_output", input, sizeof(input), expected_hash);
  }

  bool test_pick_half()
  {
    int ones = 0;
    int zeros = 0;
    const int iterations = 1000000;

    // Test with sequential values
    for (uint64_t i = 0; i < iterations; i++)
    {
      if (pick_half(i))
      {
        ones++;
      }
      else
      {
        zeros++;
      }
    }

    double ratio = (double)ones / (double)(ones + zeros);
    bool balanced = (std::abs(ratio - 0.5) < 0.01);

    if (!balanced)
    {
      std::cout << "test_pick_half failed: ratio=" << ratio << std::endl;
      return false;
    }

    std::cout << "test_pick_half passed: ratio=" << ratio << std::endl;
    return true;
  }

  bool test_stage3_single_iteration()
  {
    byte input[112] = {0};
    workerData_xelis_v3 worker;
    memset(worker.scratchPad, 0, XELIS_MEMORY_SIZE_V3 * 8);

    stage_1(input, worker.scratchPad, 112);

    uint64_t *mem_buffer_a = worker.scratchPad;
    uint64_t *mem_buffer_b = worker.scratchPad + XELIS_BUFFER_SIZE_V3;

    uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
    uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;

    printf("\nInitial values:\n");
    printf("  addr_a = 0x%016llx\n", (unsigned long long)addr_a);
    printf("  addr_b = 0x%016llx\n", (unsigned long long)addr_b);

    // First AES round
    uint64_t mem_a = mem_buffer_a[addr_a % XELIS_BUFFER_SIZE_V3];
    uint64_t mem_b = mem_buffer_b[addr_b % XELIS_BUFFER_SIZE_V3];

    printf("  mem_a = 0x%016llx (from idx %llu)\n",
           (unsigned long long)mem_a, (unsigned long long)(addr_a % XELIS_BUFFER_SIZE_V3));
    printf("  mem_b = 0x%016llx (from idx %llu)\n",
           (unsigned long long)mem_b, (unsigned long long)(addr_b % XELIS_BUFFER_SIZE_V3));

    // AES round
    uint8_t key[17] = "xelishash-pow-v3";
    uint8_t block[16];
    uint64_to_le_bytes(mem_b, block);
    uint64_to_le_bytes(mem_a, block + 8);
    aes_round(block, key);
    uint64_t hash1 = le_bytes_to_uint64(block);
    uint64_t hash2 = le_bytes_to_uint64(block + 8);
    uint64_t result = ~(hash1 ^ hash2);

    printf("  hash1 = 0x%016llx\n", (unsigned long long)hash1);
    printf("  hash2 = 0x%016llx\n", (unsigned long long)hash2);
    printf("  result = 0x%016llx\n", (unsigned long long)result);

    // First few iterations of inner loop
    size_t r = 0;
    printf("\nFirst 5 inner loop iterations (i=0):\n");
    for (size_t j = 0; j < 5; ++j)
    {
      uint64_t a = mem_buffer_a[map_index(result)];
      uint64_t b = mem_buffer_b[map_index(~ROTR(result, r))];
      uint64_t c = (r < XELIS_BUFFER_SIZE_V3) ? mem_buffer_a[r] : mem_buffer_b[r - XELIS_BUFFER_SIZE_V3];

      size_t r_before = r;
      r = (r + 1) % XELIS_MEMORY_SIZE_V3;

      uint32_t op_idx = ROTL(result, (uint32_t)c) & 0xF;

      printf("  j=%zu: r_before=%zu, r_after=%zu\n", j, r_before, r);
      printf("        a=0x%016llx (idx=%llu)\n", (unsigned long long)a, (unsigned long long)map_index(result));
      printf("        b=0x%016llx (idx=%llu)\n", (unsigned long long)b, (unsigned long long)map_index(~ROTR(result, r_before)));
      printf("        c=0x%016llx (idx=%zu)\n", (unsigned long long)c, r_before);
      printf("        op_idx=%u\n", op_idx);

      uint64_t v = execute_operation_goto(op_idx, a, b, c, r, result, 0, j);
      printf("        v=0x%016llx\n", (unsigned long long)v);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);
      printf("        new result=0x%016llx\n", (unsigned long long)result);
    }

    return true;
  }

  bool test_map_index()
  {
    // Test that map_index always returns valid indices
    const int iterations = 1000000;

    for (int i = 0; i < iterations; i++)
    {
      uint64_t x = ((uint64_t)rand() << 32) | rand();
      uint64_t index = map_index(x);

      if (index >= XELIS_BUFFER_SIZE_V3)
      {
        printf("test_map_index failed: index %llu >= BUFFER_SIZE %zu\n",
               (unsigned long long)index, XELIS_BUFFER_SIZE_V3);
        return false;
      }
    }

    // Test specific edge cases
    if (map_index(0) >= XELIS_BUFFER_SIZE_V3)
    {
      std::cout << "test_map_index failed for input 0" << std::endl;
      return false;
    }

    if (map_index(UINT64_MAX) >= XELIS_BUFFER_SIZE_V3)
    {
      std::cout << "test_map_index failed for input UINT64_MAX" << std::endl;
      return false;
    }

    std::cout << "test_map_index passed!" << std::endl;
    return true;
  }

  bool test_isqrt_correctness()
  {
    // Test edge cases
    if (isqrt(0) != 0)
      return false;
    if (isqrt(1) != 1)
      return false;
    if (isqrt(2) != 1)
      return false;
    if (isqrt(3) != 1)
      return false;
    if (isqrt(4) != 2)
      return false;

    // Test perfect squares
    for (uint64_t i = 0; i < 100000; i++)
    {
      uint64_t square = i * i;
      if (isqrt(square) != i)
      {
        printf("test_isqrt failed: isqrt(%llu) = %llu, expected %llu\n",
               (unsigned long long)square, (unsigned long long)isqrt(square),
               (unsigned long long)i);
        return false;
      }
    }

    // Test non-perfect squares
    for (uint64_t i = 1; i < 10000; i++)
    {
      uint64_t n = i * i + i / 2; // Not a perfect square
      uint64_t root = isqrt(n);

      if (root * root > n || (root + 1) * (root + 1) <= n)
      {
        printf("test_isqrt failed for %llu: got %llu\n",
               (unsigned long long)n, (unsigned long long)root);
        return false;
      }
    }

    std::cout << "test_isqrt_correctness passed!" << std::endl;
    return true;
  }

  bool test_modular_power()
  {
    // Test edge cases
    if (modular_power(2, 3, 5) != 3)
      return false; // 2^3 % 5 = 8 % 5 = 3
    if (modular_power(3, 4, 7) != 4)
      return false; // 3^4 % 7 = 81 % 7 = 4
    if (modular_power(5, 0, 13) != 1)
      return false; // Any number^0 = 1
    if (modular_power(0, 5, 13) != 0)
      return false; // 0^5 = 0
    if (modular_power(7, 1, 13) != 7)
      return false; // 7^1 % 13 = 7

    // Test with mod = 0 (should handle gracefully)
    modular_power(2, 3, 0); // Should not crash

    std::cout << "test_modular_power passed!" << std::endl;
    return true;
  }
}

int xelis_runTests_v3()
{
  std::cout << "\n=== Running Xelis Hash v3 Tests ===" << std::endl;
  fflush(stdout);

  bool all_tests_passed = true;

  // Core hash tests
  all_tests_passed &= xelis_tests_v3::test_zero_hash();
  all_tests_passed &= xelis_tests_v3::test_verify_output();
  all_tests_passed &= xelis_tests_v3::test_reused_scratchpad();

  // Component tests
  all_tests_passed &= xelis_tests_v3::test_map_index();
  all_tests_passed &= xelis_tests_v3::test_pick_half();
  all_tests_passed &= xelis_tests_v3::test_isqrt_correctness();
  all_tests_passed &= xelis_tests_v3::test_modular_power();
  all_tests_passed &= xelis_tests_v3::test_stage3_single_iteration();

  if (all_tests_passed)
  {
    std::cout << "\nXELIS-HASH-V3: All tests passed! ✓" << std::endl;
    fflush(stdout);
    return 0;
  }
  else
  {
    std::cout << "\nXELIS-HASH-V3: Some tests failed! ✗" << std::endl;
    fflush(stdout);
    return 1;
  }
}