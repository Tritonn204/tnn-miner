#include "xelis-hash.hpp"
#include <stdlib.h>
#include <iostream>

#if defined(__x86_64__)
  #include <emmintrin.h>
  #include <immintrin.h>
#endif
#include <numeric>
#include <chrono>
#include <cstring>
#include <array>
#include <cassert>
#if !defined(__AES__)
  #include <openssl/aes.h>
#endif

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

#if defined(__x86_64__)

#define rl64(x, a) (((x << a % 64) | (x >> (64 - a % 64))))

#define htonll(x) ((1 == htonl(1)) ? (x) : ((uint64_t)htonl((x) & 0xFFFFFFFF) << 32) | htonl((x) >> 32))
#define ntohll(x) ((1 == ntohl(1)) ? (x) : ((uint64_t)ntohl((x) & 0xFFFFFFFF) << 32) | ntohl((x) >> 32))

alignas(64) const int sign_bit_values_avx512[16][16] = {
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

uint64_t swap_bytes(uint64_t value)
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

void aes_round(uint8_t *block, const uint8_t *key)
{
  #if defined(__AES__)
  __m128i block_m128i = _mm_load_si128((__m128i *)block);
  __m128i key_m128i = _mm_load_si128((__m128i *)key);
  __m128i result = _mm_aesenc_si128(block_m128i, key_m128i);
  _mm_store_si128((__m128i *)block, result);
  #else
    AES_KEY aes_key;
    AES_set_encrypt_key(key, 128, &aes_key);
    AES_encrypt(block, block, &aes_key);
  #endif
}

alignas(64) static const uint64_t RC[12] = {
    0x000000008000808bULL,
    0x800000000000008bULL,
    0x8000000000008089ULL,
    0x8000000000008003ULL,
    0x8000000000008002ULL,
    0x8000000000000080ULL,
    0x000000000000800aULL,
    0x800000008000000aULL,
    0x8000000080008081ULL,
    0x8000000000008080ULL,
    0x0000000080000001ULL,
    0x8000000080008008ULL};

alignas(64) static const uint32_t RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};

alignas(64) static const uint32_t PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

void print_reversed_words(const uint64_t *state, int num_words)
{
  for (int i = 0; i < num_words; ++i)
  {
    printf("%016lx ", state[i]);
  }
  printf("\n");
}

void keccakp_1600_first(uint64_t state[25])
{
  {
    alignas(64) uint64_t C[5] = {0};
    alignas(64) uint64_t D[5] = {0};

    // Theta step
    for (int i = 0; i < 5; ++i)
    {
      C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
    }
    for (int i = 0; i < 5; ++i)
    {
      D[i] = C[(i + 4) % 5] ^ rl64(C[(i + 1) % 5], 1);
    }
    for (int i = 0; i < 25; ++i)
    {
      state[i] ^= D[i % 5];
    }

    // Rho and Pi steps
    uint64_t last = state[1];
    for (int i = 0; i < 24; ++i)
    {
      uint32_t j = PI[i];
      uint64_t temp = state[j];
      state[j] = rl64(last, RHO[i]);
      last = temp;
    }

    // Chi step
    for (int j = 0; j < 25; j += 5)
    {
      for (int i = 0; i < 5; ++i)
      {
        C[i] = state[j + i];
      }
      for (int i = 0; i < 5; ++i)
      {
        state[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
      }
    }

    // Iota step
    state[0] ^= RC[0];
  }

  for (int round = 1; round < 12; ++round)
  {
    alignas(64) uint64_t C[5] = {0};
    alignas(64) uint64_t D[5] = {0};

    // Theta step
    for (int i = 0; i < 5; ++i)
    {
      C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
    }
    for (int i = 0; i < 5; ++i)
    {
      D[i] = C[(i + 4) % 5] ^ rl64(C[(i + 1) % 5], 1);
    }
    for (int i = 0; i < 25; ++i)
    {
      state[i] ^= D[i % 5];
    }

    // Rho and Pi steps
    uint64_t last = state[1];
    for (int i = 0; i < 24; ++i)
    {
      uint32_t j = PI[i];
      uint64_t temp = state[j];
      state[j] = rl64(last, RHO[i]);
      last = temp;
    }

    // Chi step
    for (int j = 0; j < 25; j += 5)
    {
      for (int i = 0; i < 5; ++i)
      {
        C[i] = state[j + i];
      }
      for (int i = 0; i < 5; ++i)
      {
        state[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
      }
    }

    // Iota step
    state[0] ^= RC[round];
  }
}

void keccakp_1600_12(uint64_t state[25])
{
  for (int round = 0; round < 12; ++round)
  {
    alignas(64) uint64_t C[5] = {0};
    alignas(64) uint64_t D[5] = {0};

    // Theta step
    for (int i = 0; i < 5; ++i)
    {
      C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
    }
    for (int i = 0; i < 5; ++i)
    {
      D[i] = C[(i + 4) % 5] ^ rl64(C[(i + 1) % 5], 1);
    }
    for (int i = 0; i < 25; ++i)
    {
      state[i] ^= D[i % 5];
    }

    // Rho and Pi steps
    uint64_t last = state[1];
    for (int i = 0; i < 24; ++i)
    {
      uint32_t j = PI[i];
      uint64_t temp = state[j];
      state[j] = rl64(last, RHO[i]);
      last = temp;
    }

    // Chi step
    for (int j = 0; j < 25; j += 5)
    {
      for (int i = 0; i < 5; ++i)
      {
        C[i] = state[j + i];
      }
      for (int i = 0; i < 5; ++i)
      {
        state[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
      }
    }

    // Iota step
    state[0] ^= RC[round];
  }
}

void keccakp_1600_12_unrolled(uint64_t state[25])
{
  for (int round = 0; round < 12; ++round)
  {
    uint64_t C[5] = {0};
    uint64_t D[5] = {0};

    // Theta step
    C[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
    C[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
    C[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
    C[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
    C[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

    D[0] = C[4] ^ rl64(C[1], 1);
    D[1] = C[0] ^ rl64(C[2], 1);
    D[2] = C[1] ^ rl64(C[3], 1);
    D[3] = C[2] ^ rl64(C[4], 1);
    D[4] = C[3] ^ rl64(C[0], 1);

    state[0] ^= D[0];
    state[5] ^= D[0];
    state[10] ^= D[0];
    state[15] ^= D[0];
    state[20] ^= D[0];

    state[1] ^= D[1];
    state[6] ^= D[1];
    state[11] ^= D[1];
    state[16] ^= D[1];
    state[21] ^= D[1];

    state[2] ^= D[2];
    state[7] ^= D[2];
    state[12] ^= D[2];
    state[17] ^= D[2];
    state[22] ^= D[2];

    state[3] ^= D[3];
    state[8] ^= D[3];
    state[13] ^= D[3];
    state[18] ^= D[3];
    state[23] ^= D[3];

    state[4] ^= D[4];
    state[9] ^= D[4];
    state[14] ^= D[4];
    state[19] ^= D[4];
    state[24] ^= D[4];

    // Rho and Pi steps
    uint64_t last = state[1];
    state[1] = rl64(state[6], 44);
    state[6] = rl64(state[9], 20);
    state[9] = rl64(state[22], 61);
    state[22] = rl64(state[14], 39);
    state[14] = rl64(state[20], 18);
    state[20] = rl64(state[2], 62);
    state[2] = rl64(state[12], 43);
    state[12] = rl64(state[13], 25);
    state[13] = rl64(state[19], 8);
    state[19] = rl64(state[23], 56);
    state[23] = rl64(state[15], 41);
    state[15] = rl64(state[4], 27);
    state[4] = rl64(state[24], 14);
    state[24] = rl64(state[21], 2);
    state[21] = rl64(state[8], 55);
    state[8] = rl64(state[16], 45);
    state[16] = rl64(state[5], 36);
    state[5] = rl64(state[3], 28);
    state[3] = rl64(state[18], 21);
    state[18] = rl64(state[17], 15);
    state[17] = rl64(state[11], 10);
    state[11] = rl64(state[7], 6);
    state[7] = rl64(state[10], 3);
    state[10] = rl64(last, 1);

    // Chi step
    for (int j = 0; j < 25; j += 5)
    {
      C[0] = state[j];
      C[1] = state[j + 1];
      C[2] = state[j + 2];
      C[3] = state[j + 3];
      C[4] = state[j + 4];

      state[j] ^= (~C[1]) & C[2];
      state[j + 1] ^= (~C[2]) & C[3];
      state[j + 2] ^= (~C[3]) & C[4];
      state[j + 3] ^= (~C[4]) & C[0];
      state[j + 4] ^= (~C[0]) & C[1];
    }

    // Iota step
    state[0] ^= RC[round];
  }
}

void stage_1(uint64_t *int_input, uint64_t *scratchPad,
             uint16_t A1, uint16_t A2, byte B1, byte B2)
{
  for (size_t i = A1; i <= A2; ++i)
  {
    __builtin_prefetch(&int_input[0], 0, 3);
    __builtin_prefetch(&scratchPad[(i + 1) * XELIS_KECCAK_WORDS], 1, 3);

    if (i == 0)
      keccakp_1600_first(int_input);
    else
      keccakp_1600_12(int_input);

    uint64_t rand_int = 0;
    for (size_t j = B1; j <= B2; ++j)
    {
      byte pair_idx = (j + 1) % XELIS_KECCAK_WORDS;
      byte pair_idx2 = (j + 2) % XELIS_KECCAK_WORDS;

      size_t target_idx = i * XELIS_KECCAK_WORDS + j;
      uint64_t a = int_input[j] ^ rand_int;

      uint64_t left = int_input[pair_idx];
      uint64_t right = int_input[pair_idx2];

      uint64_t xor_result = left ^ right;
      uint64_t v;
      switch (xor_result & 0x3)
      {
      case 0:
        v = left & right;
        break;
      case 1:
        v = ~(left & right);
        break;
      case 2:
        v = ~xor_result;
        break;
      case 3:
        v = xor_result;
        break;
      }
      uint64_t b = a ^ v;
      rand_int = b;
      scratchPad[target_idx] = b;
    }
  }
}

/*
__attribute__((target("avx512f"))) void stage_2(uint64_t *input, uint32_t *smallPad, byte *indices, uint32_t *slots)
{
  for (byte iter = 0; iter < XELIS_ITERS; ++iter)
  {
    for (uint16_t j = 0; j < (XELIS_MEMORY_SIZE * 2) / XELIS_SLOT_LENGTH; ++j)
    {
      __builtin_prefetch(&smallPad[(j + 1) * XELIS_SLOT_LENGTH], 0, 3);
      std::iota(indices, indices + XELIS_SLOT_LENGTH, 0);
      for (int slot_idx = XELIS_SLOT_LENGTH - 1; slot_idx >= 0; --slot_idx)
      {
        byte index_in_indices = smallPad[j * XELIS_SLOT_LENGTH + slot_idx] % (slot_idx + 1);
        byte index = indices[index_in_indices];
        indices[index_in_indices] = indices[slot_idx];

        // AVX-512 implementation
        __m512i sum_buffer = _mm512_setzero_si512();
        byte sign = slots[index] >> 31;

        for (uint16_t k = 0; k < XELIS_SLOT_LENGTH; k += 16)
        {
          __m512i slot_vector = _mm512_load_si512(&slots[k]);
          __m512i values = _mm512_load_si512(&smallPad[j * XELIS_SLOT_LENGTH + k]);

          __mmask16 sign_mask = _mm512_cmpeq_epi32_mask(_mm512_setzero_si512(), _mm512_srli_epi32(slot_vector, 31));
          sum_buffer = _mm512_mask_add_epi32(sum_buffer, sign_mask, sum_buffer, values);
          sum_buffer = _mm512_mask_sub_epi32(sum_buffer, ~sign_mask, sum_buffer, values);
        }

        __m512i adjustment = _mm512_set1_epi32(smallPad[j * XELIS_SLOT_LENGTH + index]);
        __m512i sign_bit = _mm512_load_si512((__m512i *)sign_bit_values_avx512[index % 16]);

        if (sign == 0)
        {
            sum_buffer = _mm512_sub_epi32(sum_buffer, _mm512_and_si512(adjustment, sign_bit));
        }
        else
        {
            sum_buffer = _mm512_add_epi32(sum_buffer, _mm512_and_si512(adjustment, sign_bit));
        }

        __m256i sum_low_256 = _mm512_extracti64x4_epi64(sum_buffer, 0);
        __m256i sum_high_256 = _mm512_extracti64x4_epi64(sum_buffer, 1);
        __m256i sum_256 = _mm256_add_epi32(sum_low_256, sum_high_256);

        __m128i sum_low = _mm256_extracti128_si256(sum_256, 0);
        __m128i sum_high = _mm256_extracti128_si256(sum_256, 1);
        __m128i sum_128 = _mm_add_epi32(sum_low, sum_high);

        sum_128 = _mm_hadd_epi32(sum_128, sum_128);
        sum_128 = _mm_hadd_epi32(sum_128, sum_128);

        uint32_t reduced_sum = _mm_extract_epi32(sum_128, 0);
        slots[index] += reduced_sum;
      }
    }
  }

  // Copy slots back to the last SLOT_LENGTH elements of smallPad
  std::copy(&slots[0], &slots[XELIS_SLOT_LENGTH], &smallPad[XELIS_MEMORY_SIZE * 2 - XELIS_SLOT_LENGTH]);
}

__attribute__((target("avx2"))) void stage_2(uint64_t *input, uint32_t *smallPad, byte *indices, uint32_t *slots)
{
  for (byte iter = 0; iter < XELIS_ITERS; ++iter)
  {
    for (uint16_t j = 0; j < (XELIS_MEMORY_SIZE * 2) / XELIS_SLOT_LENGTH; ++j)
    {
      __builtin_prefetch(&smallPad[j * XELIS_SLOT_LENGTH], 0, 3);
      std::iota(indices, indices + XELIS_SLOT_LENGTH, 0);

      __m256i sum_buffer = _mm256_setzero_si256();

      for (uint16_t k = 0; k < XELIS_SLOT_LENGTH; k += 8)
      {
        __m256i slot_vector = _mm256_load_si256((__m256i *)&slots[k]);
        __m256i values = _mm256_load_si256((__m256i *)&smallPad[j * XELIS_SLOT_LENGTH + k]);

        __m256i sign_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_srli_epi32(slot_vector, 31));
        sum_buffer = _mm256_blendv_epi8(_mm256_sub_epi32(sum_buffer, values), _mm256_add_epi32(sum_buffer, values), sign_mask);
      }

      __m128i sum_low = _mm256_extracti128_si256(sum_buffer, 0);
      __m128i sum_high = _mm256_extracti128_si256(sum_buffer, 1);
      __m128i sum_128 = _mm_add_epi32(sum_low, sum_high);

      sum_128 = _mm_hadd_epi32(sum_128, sum_128);
      sum_128 = _mm_hadd_epi32(sum_128, sum_128);

      alignas(32) uint32_t sum = _mm_extract_epi32(sum_128, 0);
      alignas(32) uint16_t offset = j * XELIS_SLOT_LENGTH;

      for (int slot_idx = XELIS_SLOT_LENGTH - 1; slot_idx >= 0; --slot_idx)
      {
        alignas(32) byte index_in_indices = smallPad[j * XELIS_SLOT_LENGTH + slot_idx] % (slot_idx + 1);
        alignas(32) byte index = indices[index_in_indices];
        indices[index_in_indices] = indices[slot_idx];     

        alignas(32) byte sign = slots[index] >> 31;
        switch(sign) {
          case 0:
            sum -= smallPad[offset + index];
            break;
          case 1:
            sum += smallPad[offset + index];
            break;
        }

        slots[index] += sum;

        sign = slots[index] >> 31;
        switch(sign) {
          case 0:
            sum += smallPad[offset + index];
            break;
          case 1:
            sum -= smallPad[offset + index];
            break;
        }
      }
    }
  }

  // Copy slots back to the last SLOT_LENGTH elements of smallPad
  std::copy(&slots[0], &slots[XELIS_SLOT_LENGTH], &smallPad[XELIS_MEMORY_SIZE * 2 - XELIS_SLOT_LENGTH]);
}
*/

__attribute__((target("sse4.1"))) void stage_2(uint64_t *input, uint32_t *smallPad, byte *indices, uint32_t *slots)
{
  for (byte iter = 0; iter < XELIS_ITERS; ++iter)
  {
    for (uint16_t j = 0; j < (XELIS_MEMORY_SIZE * 2) / XELIS_SLOT_LENGTH; ++j)
    {
      __builtin_prefetch(&smallPad[(j + 1) * XELIS_SLOT_LENGTH], 0, 3);
      std::iota(indices, indices + XELIS_SLOT_LENGTH, 0);

      // SSE implementation
      __m128i sum_buffer = _mm_setzero_si128();

      for (size_t k = 0; k < XELIS_SLOT_LENGTH; k += 4)
      {
        __m128i slot_vector = _mm_load_si128((__m128i *)&slots[k]);
        __m128i values = _mm_load_si128((__m128i *)&smallPad[j * XELIS_SLOT_LENGTH + k]);

        __m128i sign_mask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_srli_epi32(slot_vector, 31));
        sum_buffer = _mm_blendv_epi8(_mm_sub_epi32(sum_buffer, values), _mm_add_epi32(sum_buffer, values), sign_mask);
      }

      // Perform horizontal addition to reduce the sum
      sum_buffer = _mm_hadd_epi32(sum_buffer, sum_buffer);
      sum_buffer = _mm_hadd_epi32(sum_buffer, sum_buffer);

      // Extract the reduced sum
      uint32_t sum = _mm_extract_epi32(sum_buffer, 0);
      uint16_t offset = j * XELIS_SLOT_LENGTH;

      for (int slot_idx = XELIS_SLOT_LENGTH - 1; slot_idx >= 0; --slot_idx)
      {
        byte index_in_indices = smallPad[j * XELIS_SLOT_LENGTH + slot_idx] % (slot_idx + 1);
        byte index = indices[index_in_indices];
        indices[index_in_indices] = indices[slot_idx];     

        byte sign = slots[index] >> 31;
        switch(sign) {
          case 0:
            sum -= smallPad[offset + index];
            break;
          case 1:
            sum += smallPad[offset + index];
            break;
        }

        slots[index] += sum;

        sign = slots[index] >> 31;
        switch(sign) {
          case 0:
            sum += smallPad[offset + index];
            break;
          case 1:
            sum -= smallPad[offset + index];
            break;
        }
      }
    }
  }

  // Copy slots back to the last SLOT_LENGTH elements of smallPad
  std::copy(&slots[0], &slots[XELIS_SLOT_LENGTH], &smallPad[XELIS_MEMORY_SIZE * 2 - XELIS_SLOT_LENGTH]);
}

__attribute__((target("default"))) void stage_2(uint64_t *input, uint32_t *smallPad, byte *indices, uint32_t *slots)
{
  for (byte iter = 0; iter < XELIS_ITERS; ++iter)
  {
    for (uint16_t j = 0; j < (XELIS_MEMORY_SIZE * 2) / XELIS_SLOT_LENGTH; ++j)
    {
      __builtin_prefetch(&smallPad[(j + 1) * XELIS_SLOT_LENGTH], 0, 3);
      std::iota(indices, indices + XELIS_SLOT_LENGTH, 0);

      uint32_t sum = 0;
      uint16_t offset = j * XELIS_SLOT_LENGTH;
      for (uint16_t k = 0; k < 256; ++k)
      {
        uint32_t pad = smallPad[offset + k];
        sum = (slots[k] >> 31 == 0) ? sum + pad : sum - pad;
      }

      for (int slot_idx = XELIS_SLOT_LENGTH - 1; slot_idx >= 0; --slot_idx)
      {
        byte index_in_indices = smallPad[j * XELIS_SLOT_LENGTH + slot_idx] % (slot_idx + 1);
        byte index = indices[index_in_indices];
        indices[index_in_indices] = indices[slot_idx];     

        byte sign = slots[index] >> 31;
        switch(sign) {
          case 0:
            sum -= smallPad[offset + index];
            break;
          case 1:
            sum += smallPad[offset + index];
            break;
        }

        slots[index] += sum;

        sign = slots[index] >> 31;
        switch(sign) {
          case 0:
            sum += smallPad[offset + index];
            break;
          case 1:
            sum -= smallPad[offset + index];
            break;
        }
      }
    }
  }
  // Copy slots back to the last SLOT_LENGTH elements of smallPad
  std::copy(&slots[0], &slots[XELIS_SLOT_LENGTH], &smallPad[XELIS_MEMORY_SIZE * 2 - XELIS_SLOT_LENGTH]);
}


void stage_3(uint64_t *scratchPad, byte *hashResult)
{
  const byte key[16] = {0};
  alignas(64) byte block[16] = {0};

  alignas(64) uint64_t addr_a = (scratchPad[XELIS_MEMORY_SIZE - 1] >> 15) & 0x7FFF;
  alignas(64) uint64_t addr_b = scratchPad[XELIS_MEMORY_SIZE - 1] & 0x7FFF;

  alignas(64) uint64_t mem_buffer_a[XELIS_BUFFER_SIZE];
  alignas(64) uint64_t mem_buffer_b[XELIS_BUFFER_SIZE];

  for (byte i = 0; i < XELIS_BUFFER_SIZE; ++i)
  {
    mem_buffer_a[i] = scratchPad[(addr_a + i) % XELIS_MEMORY_SIZE];
    mem_buffer_b[i] = scratchPad[(addr_b + i) % XELIS_MEMORY_SIZE];
  }

  for (uint16_t i = 0; i < XELIS_SCRATCHPAD_ITERS; ++i)
  {
    __builtin_prefetch(&mem_buffer_a[(i + 1) % XELIS_BUFFER_SIZE], 0, 3);
    __builtin_prefetch(&mem_buffer_b[(i + 1) % XELIS_BUFFER_SIZE], 0, 3);
    byte *mem_a = reinterpret_cast<byte *>(&mem_buffer_a[i % XELIS_BUFFER_SIZE]);
    byte *mem_b = reinterpret_cast<byte *>(&mem_buffer_b[i % XELIS_BUFFER_SIZE]);

    std::copy(mem_b, mem_b + 8, block);
    std::copy(mem_a, mem_a + 8, block + 8);

    aes_round(block, key);

    uint64_t hash1 = *(reinterpret_cast<uint64_t *>(&block[0]));
    uint64_t hash2 = *(reinterpret_cast<uint64_t *>(mem_a)) ^ *(reinterpret_cast<uint64_t *>(mem_b));

    uint64_t result = ~(hash1 ^ hash2);

    for (size_t j = 0; j < XELIS_HASH_SIZE; ++j)
    {
      uint64_t a = mem_buffer_a[(j + i) % XELIS_BUFFER_SIZE];
      uint64_t b = mem_buffer_b[(j + i) % XELIS_BUFFER_SIZE];

      uint64_t v;
      switch ((result >> (j * 2)) & 0xf)
      {
      case 0:
        v = rl64(result, j) ^ b;
        break;
      case 1:
        v = ~(rl64(result, j) ^ a);
        break;
      case 2:
        v = ~(result ^ a);
        break;
      case 3:
        v = result ^ b;
        break;
      case 4:
        v = result ^ (a + b);
        break;
      case 5:
        v = result ^ (a - b);
        break;
      case 6:
        v = result ^ (b - a);
        break;
      case 7:
        v = result ^ (a * b);
        break;
      case 8:
        v = result ^ (a & b);
        break;
      case 9:
        v = result ^ (a | b);
        break;
      case 10:
        v = result ^ (a ^ b);
        break;
      case 11:
        v = result ^ (a - result);
        break;
      case 12:
        v = result ^ (b - result);
        break;
      case 13:
        v = result ^ (a + result);
        break;
      case 14:
        v = result ^ (result - a);
        break;
      case 15:
        v = result ^ (result - b);
        break;
      }

      result = v;
    }

    addr_b = result & 0x7FFF;
    mem_buffer_a[i % XELIS_BUFFER_SIZE] = result;
    mem_buffer_b[i % XELIS_BUFFER_SIZE] = scratchPad[addr_b];

    addr_a = (result >> 15) & 0x7FFF;
    scratchPad[addr_a] = result;

    size_t index = XELIS_SCRATCHPAD_ITERS - i - 1;
    // printf("post result: %llu\n", result);
    if (index < 4)
    {
      uint64_t be_result = htonll(result);
      std::copy(reinterpret_cast<byte *>(&be_result), reinterpret_cast<byte *>(&be_result) + 8, &hashResult[index * 8]);
    }
  }
}

void xelis_hash(byte *input, workerData_xelis &worker, byte *hashResult)
{
  // printf("input: ");
  // for (int i = 0; i < 200; i++) {
  //   printf("%02x", input[i]);
  // }
  // printf("\n\n");
  worker.int_input = reinterpret_cast<uint64_t *>(input);

  // Stage 1
  stage_1(worker.int_input, worker.scratchPad,
          0, XELIS_STAGE_1_MAX - 1,
          0, XELIS_KECCAK_WORDS - 1);
  stage_1(worker.int_input, worker.scratchPad,
          XELIS_STAGE_1_MAX, XELIS_STAGE_1_MAX,
          0, 17);

  // Stage 2
  __builtin_prefetch(worker.slots, 1, 3);
  __builtin_prefetch(worker.smallPad, 1, 3);
  worker.smallPad = reinterpret_cast<uint32_t *>(worker.scratchPad);

  std::copy(&worker.smallPad[XELIS_MEMORY_SIZE * 2 - XELIS_SLOT_LENGTH], &worker.smallPad[XELIS_MEMORY_SIZE * 2], worker.slots);

  // if (__builtin_cpu_supports("avx512f")) stage_2_avx512(worker.int_input, worker.smallPad, worker.indices, worker.slots);
  stage_2(worker.int_input, worker.smallPad, worker.indices, worker.slots);
  // else if (__builtin_cpu_supports("sse2")) stage_2_sse2(worker.int_input, worker.smallPad, worker.indices, worker.slots);
  // else stage_2_scalar(worker.int_input, worker.smallPad, worker.indices, worker.slots);

  // Stage 3
  stage_3(worker.scratchPad, hashResult);
}

void xelis_benchmark_cpu_hash()
{
  const uint32_t ITERATIONS = 1000;
  byte input[200] = {0};
  alignas(64) workerData_xelis worker;
  alignas(64) byte hash_result[XELIS_HASH_SIZE] = {0};

  auto start = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < ITERATIONS; ++i)
  {
    input[0] = i & 0xFF;
    input[1] = (i >> 8) & 0xFF;
    xelis_hash(input, worker, hash_result);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time took: " << elapsed.count() << " ms" << std::endl;
  std::cout << "H/s: " << (ITERATIONS * 1000.0 / elapsed.count()) << std::endl;
  std::cout << "ms per hash: " << (elapsed.count() / ITERATIONS) << std::endl;
}

int char2int(char input)
{
  if (input >= '0' && input <= '9')
    return input - '0';
  if (input >= 'A' && input <= 'F')
    return input - 'A' + 10;
  if (input >= 'a' && input <= 'f')
    return input - 'a' + 10;
  throw std::invalid_argument("Invalid input string");
}

// This function assumes src to be a zero terminated sanitized string with
// an even number of [0-9a-f] characters, and target to be sufficiently large
void hex2bin(const char *src, char *target)
{
  while (*src && src[1])
  {
    *(target++) = char2int(*src) * 16 + char2int(src[1]);
    src += 2;
  }
}

const char *testTemplate = "97dff4761917c2692df3be38e72ca7a59c3f55252e2245cc21564ef65fa8ea6f0000018f22fe78f80000000000064202f2a40463ccfcea839c4950a56ee38fa69c7ce2d4ba45d4b060cc63c297fb73b8a09c69661b1690b0a238d096a7ccb3cb204ce5dd604da9bb6c79c4ab00000000";

namespace tests
{
  using Hash = std::array<byte, XELIS_HASH_SIZE>;

  bool test_real()
  {
    alignas(64) workerData_xelis worker;
    byte hash_result[XELIS_HASH_SIZE] = {0};
    alignas(64) byte input[XELIS_BYTES_ARRAY_INPUT] = {0};

    hex2bin(testTemplate,
            (char *)input);

    printf("sanity check: ");
    for (int i = 0; i < 112; i++)
    {
      printf("%02x", input[i]);
    }
    printf("\n");

    xelis_hash(input, worker, hash_result);

    printf("hoping for: 446e381b592967518c2b184c7115f9446b65921358eeb751363663e3474e0300\ngot: ");
    for (int i = 0; i < 32; i++)
    {
      printf("%02x", hash_result[i]);
    }
    printf("\n");
    return true;
  }

  bool test_input(const char *test_name, byte *input, size_t input_size, const Hash &expected_hash)
  {
    alignas(64) workerData_xelis worker;
    byte hash_result[XELIS_HASH_SIZE] = {0};

    xelis_hash(input, worker, hash_result);

    if (std::memcmp(hash_result, expected_hash.data(), XELIS_HASH_SIZE) != 0)
    {
      std::cout << "Test '" << test_name << "' failed!" << std::endl;
      std::cout << "Expected hash: ";
      for (size_t i = 0; i < XELIS_HASH_SIZE; ++i)
      {
        printf("%02x ", expected_hash[i]);
      }
      std::cout << std::endl;
      std::cout << "Actual hash:   ";
      for (size_t i = 0; i < XELIS_HASH_SIZE; ++i)
      {
        printf("%02x ", hash_result[i]);
      }
      std::cout << std::endl;
      return false;
    }
    return true;
  }

  bool test_zero_input()
  {
    alignas(64) byte input[200] = {0};
    Hash expected_hash = {
        0x0e, 0xbb, 0xbd, 0x8a, 0x31, 0xed, 0xad, 0xfe, 0x09, 0x8f, 0x2d, 0x77, 0x0d, 0x84,
        0xb7, 0x19, 0x58, 0x86, 0x75, 0xab, 0x88, 0xa0, 0xa1, 0x70, 0x67, 0xd0, 0x0a, 0x8f,
        0x36, 0x18, 0x22, 0x65};

    return test_input("test_zero_input", input, sizeof(input), expected_hash);
  }

  bool test_xelis_input()
  {
    alignas(64) byte input[XELIS_BYTES_ARRAY_INPUT] = {0};

    const char *custom = "xelis-hashing-algorithm";
    std::memcpy(input, custom, std::strlen(custom));

    Hash expected_hash = {
        106, 106, 173, 8, 207, 59, 118, 108, 176, 196, 9, 124, 250, 195, 3,
        61, 30, 146, 238, 182, 88, 83, 115, 81, 139, 56, 3, 28, 176, 86, 68, 21};
    return test_input("test_xelis_input", input, sizeof(input), expected_hash);
  }
}

void xelis_runTests()
{
  bool all_tests_passed = true;
  all_tests_passed &= tests::test_zero_input();
  all_tests_passed &= tests::test_xelis_input();

  tests::test_real();

  if (all_tests_passed)
  {
    std::cout << "XELIS-HASH: All tests passed!" << std::endl;
  }
  else
  {
    std::cout << "XELIS-HASH: Some tests failed!" << std::endl;
  }
}

#else
// These are just to satisfy compilation on AARCH64
void xelis_hash(byte* input, workerData_xelis &worker, byte *hashResult) {
}

void xelis_benchmark_cpu_hash() {
}

void xelis_runTests() {
}

#endif // __x86_64__
