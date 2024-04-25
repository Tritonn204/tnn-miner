#include "xelis-hash.hpp"
#include <stdlib.h>
#include <inttypes.h>
#include <iostream>

#include <fastiota.h>
#include <emmintrin.h>
#include <immintrin.h>

const uint16_t MEMORY_SIZE = 32768;
const uint16_t SCRATCHPAD_ITERS = 5000;
const byte ITERS = 1;
const uint16_t BUFFER_SIZE = 42;
const uint16_t SLOT_LENGTH = 256;

const byte KECCAK_WORDS = 25;
const byte BYTES_ARRAY_INPUT = KECCAK_WORDS * 8;
const byte HASH_SIZE = 32;
const uint16_t STAGE_1_MAX = MEMORY_SIZE / KECCAK_WORDS;

uint64_t *int_input;
uint32_t *smallPad;
uint32_t slots[SLOT_LENGTH];
byte indices[SLOT_LENGTH];

void stage_1(uint64_t* int_input, uint64_t* scratchPad, 
  byte A1, uint16_t A2, byte B1, byte B2
) {
    for (size_t i = A1; i <= A2; ++i) {
        int_input = reinterpret_cast<uint64_t*>(keccak256(reinterpret_cast<byte*>(int_input), BYTES_ARRAY_INPUT));

        uint64_t rand_int = 0;
        for (size_t j = B1; j <= B2; ++j) {
            byte pair_idx = (j + 1) % KECCAK_WORDS;
            byte pair_idx2 = (j + 2) % KECCAK_WORDS;

            size_t target_idx = i * KECCAK_WORDS + j;
            uint64_t a = int_input[j] ^ rand_int;
            uint64_t left = int_input[pair_idx];
            uint64_t right = int_input[pair_idx2];
            uint64_t xor_result = left ^ right;
            uint64_t v;
            switch (xor_result & 0x3) {
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

void stage_2(uint64_t *input, uint32_t *smallPad, byte *indices) {
  for (byte iter = 0; iter < ITERS; ++iter) {
    for (uint16_t j = 0; j < (MEMORY_SIZE * 2) / SLOT_LENGTH; ++j) {
      // Initialize indices
      // for (byte k = 0; k < SLOT_LENGTH; ++k) {
      //   indices[k] = k;
      // }

      fastiota::iota(indices, indices+SLOT_LENGTH, 256, 0);

      for (int slot_idx = SLOT_LENGTH - 1; slot_idx >= 0; --slot_idx) {
        uint16_t index_in_indices = smallPad[j * SLOT_LENGTH + slot_idx] % (slot_idx + 1);
        uint16_t index = indices[index_in_indices];
        indices[index_in_indices] = indices[slot_idx];

        uint32_t sum = slots[index];

      #ifdef __AVX512F__
        // AVX-512 implementation
        __m512i sum_buffer = _mm512_setzero_si512();

        for (uint16_t k = 0; k < SLOT_LENGTH; k += 16) {
          __m512i slot_vector = _mm512_loadu_si512(&slots[k]);
          __m512i values = _mm512_loadu_si512(&smallPad[j * SLOT_LENGTH + k]);

          __mmask16 sign_mask = _mm512_cmpgt_epu32_mask(_mm512_setzero_si512(), _mm512_srli_epi32(slot_vector, 31));
          sum_buffer = _mm512_add_epi32(sum_buffer, _mm512_mask_blend_epi32(sign_mask, values, _mm512_sub_epi32(_mm512_setzero_si512(), values)));
        }

        sum += _mm512_reduce_add_epi32(sum_buffer);
      #elif defined(__AVX2__)
        // AVX2 implementation
        __m256i sum_buffer = _mm256_setzero_si256();

        for (uint16_t k = 0; k < SLOT_LENGTH; k += 8) {
          __m256i slot_vector = _mm256_loadu_si256((__m256i*)&slots[k]);
          __m256i values = _mm256_loadu_si256((__m256i*)&smallPad[j * SLOT_LENGTH + k]);

          __m256i sign_mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), _mm256_srli_epi32(slot_vector, 31));
          sum_buffer = _mm256_add_epi32(sum_buffer, _mm256_blendv_epi8(values, _mm256_sub_epi32(_mm256_setzero_si256(), values), sign_mask));
        }

        sum += _mm256_extract_epi32(sum_buffer, 0) + _mm256_extract_epi32(sum_buffer, 1) +
              _mm256_extract_epi32(sum_buffer, 2) + _mm256_extract_epi32(sum_buffer, 3) +
              _mm256_extract_epi32(sum_buffer, 4) + _mm256_extract_epi32(sum_buffer, 5) +
              _mm256_extract_epi32(sum_buffer, 6) + _mm256_extract_epi32(sum_buffer, 7);
      #elif defined(__SSE2__)
        // SSE implementation
        __m128i sum_buffer = _mm_setzero_si128();

        for (size_t k = 0; k < SLOT_LENGTH; k += 4) {
          __m128i slot_vector = _mm_loadu_si128((__m128i*)&slots[k]);
          __m128i values = _mm_loadu_si128((__m128i*)&smallPad[j * SLOT_LENGTH + k]);

          __m128i sign_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), _mm_srli_epi32(slot_vector, 31));
          sum_buffer = _mm_add_epi32(sum_buffer, _mm_blendv_epi8(values, _mm_sub_epi32(_mm_setzero_si128(), values), sign_mask));
        }

        sum += _mm_extract_epi32(sum_buffer, 0) + _mm_extract_epi32(sum_buffer, 1) +
              _mm_extract_epi32(sum_buffer, 2) + _mm_extract_epi32(sum_buffer, 3);
      #else
        // SCALAR implementation
        uint16_t offset = j * SLOT_LENGTH;

        for (uint16_t k = 0; k < index; ++k) {
          uint32_t pad = smallPad[offset + k];
          sum = (slots[k] >> 31 == 0) ? sum + pad : sum - pad;
        }

        for (uint16_t k = index + 1; k < SLOT_LENGTH; ++k) {
          uint32_t pad = smallPad[offset + k];
          sum = (slots[k] >> 31 == 0) ? sum + pad : sum - pad;
        }
      #endif
        slots[index] = sum;
      }
    }
  }

  // Copy slots back to the last SLOT_LENGTH elements of smallPad
  std::copy(slots, slots + SLOT_LENGTH, smallPad + (MEMORY_SIZE * 2 - SLOT_LENGTH));
}

void stage_3(byte* scratchPad, byte* hashResult) {
    const byte key[16] = {0};
    byte block[16] = {0};

    uint64_t addr_a = (scratchPad[MEMORY_SIZE - 1] >> 15) & 0x7FFF;
    uint64_t addr_b = scratchPad[MEMORY_SIZE - 1] & 0x7FFF;

    uint64_t mem_buffer_a[BUFFER_SIZE];
    uint64_t mem_buffer_b[BUFFER_SIZE];

    for (byte i = 0; i < BUFFER_SIZE; ++i) {
        mem_buffer_a[i] = scratchPad[(addr_a + i) % MEMORY_SIZE];
        mem_buffer_b[i] = scratchPad[(addr_b + i) % MEMORY_SIZE];
    }

    for (uint16_t i = 0; i < SCRATCHPAD_ITERS; ++i) {
        uint64_t mem_a = mem_buffer_a[i % BUFFER_SIZE];
        uint64_t mem_b = mem_buffer_b[i % BUFFER_SIZE];

        std::copy(mem_b, mem_b+8, block);
        std::copy(mem_a, mem_a+8, block + 8);

        aes_round(block, key);

        uint64_t hash1 = (static_cast<uint64_t>(block[0])) | (static_cast<uint64_t>(block[1]) << 32);
        uint64_t hash2 = mem_a ^ mem_b;

        uint64_t result = -((hash1 ^ hash2) + 1);

        for (size_t j = 0; j < HASH_SIZE; ++j) {
          uint64_t a = mem_buffer_a[(j + i) % BUFFER_SIZE];
          uint64_t b = mem_buffer_b[(j + i) % BUFFER_SIZE];

          uint64_t v;
          switch ((result >> (j * 2)) & 0xf) {
            case 0:
              v = result ^ ((result << j) | (result >> (64 - j))) ^ b;
              break;
            case 1:
              v = ~(result ^ ((result << j) | (result >> (64 - j))) ^ a);
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
        mem_buffer_a[i % BUFFER_SIZE] = result;
        mem_buffer_b[i % BUFFER_SIZE] = scratchPad[addr_b];

        addr_a = (result >> 15) & 0x7FFF;
        scratchPad[addr_a] = result;

        size_t index = SCRATCHPAD_ITERS - i - 1;
        if (index < 4) {
            memcpy(hashResult + index * 8, &result, (SCRATCHPAD_ITERS - i) * 8);
            std::copy(result, result + (SCRATCHPAD_ITERS - i) * 8 - (index * 8), hashResult + index * 8);
        }
    }
}

void xelis_hash(byte* input, byte* scratchPad, byte *hashResult) {
  int_input = reinterpret_cast<uint64_t*>(input);
  
  // Stage 1
  stage_1(int_input, reinterpret_cast<uint64_t*>(scratchPad), 
    0, STAGE_1_MAX - 1,
    0, KECCAK_WORDS - 1
  );
  stage_1(int_input, reinterpret_cast<uint64_t*>(scratchPad), 
    STAGE_1_MAX, STAGE_1_MAX,
    0, 17
  );

  // Stage 2
  std::fill_n(slots, SLOT_LENGTH, 0);
  smallPad = reinterpret_cast<uint32_t*>(scratchPad);

  // Copy the last SLOT_LENGTH elements from smallPad to slots
  std::copy(smallPad + (MEMORY_SIZE * 2 - SLOT_LENGTH), smallPad + (MEMORY_SIZE * 2), slots);
  std::fill_n(indices, SLOT_LENGTH, 0);

  stage_2(int_input, smallPad, indices);
  
  // Stage 3
  stage_3(scratchPad, hashResult);
}

/*
function xelis_hash(input, scratchPad):
    if input length is less than BYTES_ARRAY_INPUT or scratchPad length is less than MEMORY_SIZE:
        return Error

    int_input = convert input to an array of 64-bit integers

    // Stage 1
    stage_1(int_input, scratchPad, (0, STAGE_1_MAX - 1), (0, KECCAK_WORDS - 1))
    stage_1(int_input, scratchPad, (STAGE_1_MAX, STAGE_1_MAX), (0, 17))

    // Stage 2
    slots = copy last SLOT_LENGTH elements from scratchPad to a new array
    smallPad = convert scratchPad to an array of 32-bit integers

    for ITERS iterations:
        for each SLOT_LENGTH-sized chunk in smallPad:
            indices = initialize indices array
            for each slot in slots (in reverse order):
                index = select an index from indices based on smallPad value
                update slots[index] using a sum of smallPad values
                update indices

    copy slots back to the end of smallPad

    // Stage 3
    key = initialize a 16-byte array with zeros
    block = initialize a 16-byte array with zeros
    addr_a = calculate memory address based on last element of scratchPad
    addr_b = calculate memory address based on last element of scratchPad
    mem_buffer_a = copy BUFFER_SIZE elements from scratchPad starting at addr_a
    mem_buffer_b = copy BUFFER_SIZE elements from scratchPad starting at addr_b
    final_result = initialize a HASH_SIZE-byte array with zeros

    for SCRATCHPAD_ITERS iterations:
        mem_a = select an element from mem_buffer_a based on iteration
        mem_b = select an element from mem_buffer_b based on iteration
        block = combine mem_b and mem_a
        perform AES cipher round on block using key
        hash1 = convert first 8 bytes of block to a 64-bit integer
        hash2 = XOR mem_a and mem_b
        result = XOR hash1 and hash2
        for HASH_SIZE iterations:
            a = select an element from mem_buffer_a based on iteration and index
            b = select an element from mem_buffer_b based on iteration and index
            v = perform a series of operations on result, a, and b based on the current index
            result = v
        addr_b = update memory address based on result
        mem_buffer_a[i % BUFFER_SIZE] = result
        mem_buffer_b[i % BUFFER_SIZE] = select an element from scratchPad based on addr_b
        addr_a = update memory address based on result
        scratchPad[addr_a] = result
        if iteration is one of the last 4:
            copy result to final_result

    return final_result
  */