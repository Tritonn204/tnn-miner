#include "xelis-hash.hpp"
#include <stdlib.h>
#include <iostream>
#include "chacha20.hpp"
#include <crc32.h>

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

#include <openssl/evp.h>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

#if defined(__x86_64__)

#define rl64(x, a) (((x << (a & 63)) | (x >> (64 - (a & 63)))))
#define rr64(x, a) (((x >> (a & 63)) | (x << (64 - (a & 63)))))

#define htonll(x) ((1 == htonl(1)) ? (x) : ((uint64_t)htonl((x) & 0xFFFFFFFF) << 32) | htonl((x) >> 32))
#define ntohll(x) ((1 == ntohl(1)) ? (x) : ((uint64_t)ntohl((x) & 0xFFFFFFFF) << 32) | ntohl((x) >> 32))

alignas(64) static const byte KEY[] = "xelishash-pow-v2";

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

static uint64_t swap_bytes(uint64_t value)
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

static void aes_round(uint8_t *block, const uint8_t *key)
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

// void stage_1_old(uint64_t *int_input, uint64_t *scratchPad,
//              uint16_t A1, uint16_t A2, byte B1, byte B2)
// {
//   for (size_t i = A1; i <= A2; ++i)
//   {
//     __builtin_prefetch(&int_input[0], 0, 3);
//     __builtin_prefetch(&scratchPad[(i + 1) * XELIS_KECCAK_WORDS], 1, 3);

//     if (i == 0)
//       keccakp_1600_first(int_input);
//     else
//       keccakp_1600_12(int_input);

//     uint64_t rand_int = 0;
//     for (size_t j = B1; j <= B2; ++j)
//     {
//       byte pair_idx = (j + 1) % XELIS_KECCAK_WORDS;
//       byte pair_idx2 = (j + 2) % XELIS_KECCAK_WORDS;

//       size_t target_idx = i * XELIS_KECCAK_WORDS + j;
//       uint64_t a = int_input[j] ^ rand_int;

//       uint64_t left = int_input[pair_idx];
//       uint64_t right = int_input[pair_idx2];

//       uint64_t xor_result = left ^ right;
//       uint64_t v;
//       switch (xor_result & 0x3)
//       {
//       case 0:
//         v = left & right;
//         break;
//       case 1:
//         v = ~(left & right);
//         break;
//       case 2:
//         v = ~xor_result;
//         break;
//       case 3:
//         v = xor_result;
//         break;
//       }
//       uint64_t b = a ^ v;
//       rand_int = b;
//       scratchPad[target_idx] = b;
//     }
//   }
// }

void stage_1(uint8_t *input, uint64_t *scratch_pad, size_t input_len)
{
  const size_t chunk_size = 32;
  const size_t nonce_size = 12;
  const size_t output_size = XELIS_MEMORY_SIZE_V2 * 8; // MEMORY_SIZE is in u64, so multiply by 8 for bytes

  alignas(32) uint8_t nonce[nonce_size] = {0};
  alignas(32) uint8_t *output = reinterpret_cast<uint8_t *>(scratch_pad);
  size_t output_offset = 0;
  size_t num_chunks = (input_len + chunk_size - 1) / chunk_size;

  for (size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index)
  {
    // Calculate the start and end of the current chunk
    size_t chunk_start = chunk_index * chunk_size;
    size_t chunk_end = std::min(chunk_start + chunk_size, input_len);

    // Pad the chunk to 32 bytes if it is shorter
    alignas(32) uint8_t key[chunk_size] = {0};
    memcpy(key, &input[chunk_start], chunk_end - chunk_start);

    // Create a new ChaCha20 instance with the current chunk as the key
    chacha20::ChaCha20 chacha(key, nonce);

    // Create and initialize the ChaCha20 cipher context
    // EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    // EVP_EncryptInit_ex(ctx, EVP_chacha20(), NULL, NULL, NULL);

    // // Set the key and nonce
    // EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, 12, NULL);
    // EVP_EncryptInit_ex(ctx, NULL, NULL, key, nonce);

    // Calculate the remaining size and how much to generate this iteration
    size_t remaining_output_size = output_size - output_offset;
    size_t chunks_left = num_chunks - chunk_index;
    size_t chunk_output_size = remaining_output_size / chunks_left;
    int current_output_size = std::min(remaining_output_size, chunk_output_size);

    chacha.apply_keystream(output + output_offset, current_output_size);

    __builtin_prefetch(output + current_output_size + output_offset, 0, 3);

    // Generate the output using the keystream
    // EVP_EncryptUpdate(ctx, output + output_offset, &current_output_size, output + output_offset, current_output_size);
    // EVP_EncryptFinal_ex(ctx, output + output_offset + current_output_size, &current_output_size);
    // chacha.crypt(output + output_offset, current_output_size);
    // chacha(&k, &iv, output + output_offset, output + output_offset, chunk_output_size, 20);
    output_offset += current_output_size;

    // EVP_CIPHER_CTX_free(ctx);

    // Update the nonce with the last NONCE_SIZE bytes of the output
    size_t nonce_start = current_output_size - nonce_size;
    // std::copy(output + output_offset - nonce_size, output + output_offset, nonce);
    memcpy(nonce, output + output_offset - nonce_size, nonce_size);

    // printf("nonce: ");
    // for (int i = 0; i < 12; i++)
    // {
    //   printf("%02X ", nonce[i]);
    // }
    // printf("\n");
  }

  // Crc32 crc32;
  // crc32.input(scratch_pad, XELIS_MEMORY_SIZE_V2 * 8);
  // std::cout << "Stage 1 scratch pad CRC32: 0x" << std::hex << std::setw(8) << std::setfill('0') << crc32.result() << std::endl;
}

// void stage_3(uint64_t *scratchPad, byte *hashResult)
// {
//   const byte key[16] = {0};
//   alignas(64) byte block[16] = {0};

//   alignas(64) uint64_t addr_a = (scratchPad[XELIS_MEMORY_SIZE_V2 - 1] >> 15) & 0x7FFF;
//   alignas(64) uint64_t addr_b = scratchPad[XELIS_MEMORY_SIZE_V2 - 1] & 0x7FFF;

//   alignas(64) uint64_t mem_buffer_a[XELIS_BUFFER_SIZE_V2];
//   alignas(64) uint64_t mem_buffer_b[XELIS_BUFFER_SIZE_V2];

//   for (byte i = 0; i < XELIS_BUFFER_SIZE_V2; ++i)
//   {
//     mem_buffer_a[i] = scratchPad[(addr_a + i) % XELIS_MEMORY_SIZE];
//     mem_buffer_b[i] = scratchPad[(addr_b + i) % XELIS_MEMORY_SIZE];
//   }

//   for (uint16_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V2; ++i)
//   {
//     __builtin_prefetch(&mem_buffer_a[(i + 1) % XELIS_BUFFER_SIZE_V2], 0, 3);
//     __builtin_prefetch(&mem_buffer_b[(i + 1) % XELIS_BUFFER_SIZE_V2], 0, 3);
//     byte *mem_a = reinterpret_cast<byte *>(&mem_buffer_a[i % XELIS_BUFFER_SIZE_V2]);
//     byte *mem_b = reinterpret_cast<byte *>(&mem_buffer_b[i % XELIS_BUFFER_SIZE_V2]);

//     std::copy(mem_b, mem_b + 8, block);
//     std::copy(mem_a, mem_a + 8, block + 8);

//     aes_round(block, key);

//     uint64_t hash1 = *(reinterpret_cast<uint64_t *>(&block[0]));
//     uint64_t hash2 = *(reinterpret_cast<uint64_t *>(mem_a)) ^ *(reinterpret_cast<uint64_t *>(mem_b));

//     uint64_t result = ~(hash1 ^ hash2);

//     for (size_t j = 0; j < XELIS_HASH_SIZE; ++j)
//     {
//       uint64_t a = mem_buffer_a[(j + i) % XELIS_BUFFER_SIZE_V2];
//       uint64_t b = mem_buffer_b[(j + i) % XELIS_BUFFER_SIZE_V2];

//       uint64_t v;
//       switch ((result >> (j * 2)) & 0xf)
//       {
//       case 0:
//         v = rl64(result, j) ^ b;
//         break;
//       case 1:
//         v = ~(rl64(result, j) ^ a);
//         break;
//       case 2:
//         v = ~(result ^ a);
//         break;
//       case 3:
//         v = result ^ b;
//         break;
//       case 4:
//         v = result ^ (a + b);
//         break;
//       case 5:
//         v = result ^ (a - b);
//         break;
//       case 6:
//         v = result ^ (b - a);
//         break;
//       case 7:
//         v = result ^ (a * b);
//         break;
//       case 8:
//         v = result ^ (a & b);
//         break;
//       case 9:
//         v = result ^ (a | b);
//         break;
//       case 10:
//         v = result ^ (a ^ b);
//         break;
//       case 11:
//         v = result ^ (a - result);
//         break;
//       case 12:
//         v = result ^ (b - result);
//         break;
//       case 13:
//         v = result ^ (a + result);
//         break;
//       case 14:
//         v = result ^ (result - a);
//         break;
//       case 15:
//         v = result ^ (result - b);
//         break;
//       }

//       result = v;
//     }

//     addr_b = result & 0x7FFF;
//     mem_buffer_a[i % XELIS_BUFFER_SIZE_V2] = result;
//     mem_buffer_b[i % XELIS_BUFFER_SIZE_V2] = scratchPad[addr_b];

//     addr_a = (result >> 15) & 0x7FFF;
//     scratchPad[addr_a] = result;

//     size_t index = XELIS_SCRATCHPAD_ITERS_V2 - i - 1;
//     // printf("post result: %llu\n", result);
//     if (index < 4)
//     {
//       uint64_t be_result = htonll(result);
//       std::copy(reinterpret_cast<byte *>(&be_result), reinterpret_cast<byte *>(&be_result) + 8, &hashResult[index * 8]);
//     }
//   }
// }

static inline uint64_t isqrt(uint64_t n)
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

inline uint64_t wrapping_add(uint64_t a, uint64_t b) {
    return a + b;
}

inline uint64_t wrapping_sub(uint64_t a, uint64_t b) {
    return a - b;
}

inline uint64_t wrapping_mul(uint64_t a, uint64_t b) {
    return a * b;
}

void stage_3(uint64_t *scratch_pad)
{
    alignas(64) const uint8_t *key = KEY;
    alignas(64) uint8_t block[16] = {0};
    const uint64_t buffer_size = XELIS_BUFFER_SIZE_V2;

    alignas(64) uint64_t *mem_buffer_a = scratch_pad;
    alignas(64) uint64_t *mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V2;

    alignas(64) uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V2 - 1];
    alignas(64) uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V2 - 1] >> 32;
    size_t r = 0;

    for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V2; ++i) {
        alignas(64) uint64_t mem_a = mem_buffer_a[addr_a % buffer_size];
        alignas(64) uint64_t mem_b = mem_buffer_b[addr_b % buffer_size];

        std::copy(&mem_b, &mem_b + 8, block);
        std::copy(&mem_a, &mem_a + 8, block + 8);

        aes_round(block, key);

        uint64_t hash1 = 0, hash2 = 0;
        std::copy(block, block + 8, &hash1);
        // hash1 = _byteswap_uint64(hash1);
        hash2 = mem_a ^ mem_b;

        alignas(64) uint64_t result = ~(hash1 ^ hash2);

        // printf("pre result: %llu\n", result);

        for (size_t j = 0; j < XELIS_BUFFER_SIZE_V2; ++j) {
            uint64_t a = mem_buffer_a[(result % buffer_size)];
            uint64_t b = mem_buffer_b[~rr64(result, r) % buffer_size];
            uint64_t c = (r < XELIS_BUFFER_SIZE_V2) ? mem_buffer_a[r] : mem_buffer_b[r - XELIS_BUFFER_SIZE_V2];
            r = (r + 1) % XELIS_MEMORY_SIZE_V2;

            // printf("%d, ", rl64(result, (uint32_t)c) & 0xF);
            uint64_t v;
            switch (rl64(result, (uint32_t)c) & 0xF) {
                case 0: v = result ^ rl64(c, (uint32_t)(i * j)) ^ b; break;
                case 1: v = result ^ rr64(c, (uint32_t)(i * j)) ^ a; break;
                case 2: v = result ^ a ^ b ^ c; break;
                case 3: v = result ^ wrapping_mul(wrapping_add(a, b), c); break;
                case 4: v = result ^ wrapping_mul(wrapping_sub(b, c), a); break;
                case 5: v = result ^ wrapping_add(wrapping_sub(c, a), b); break;
                case 6: v = result ^ wrapping_add(wrapping_sub(a, b), c); break;
                case 7: v = result ^ wrapping_add(wrapping_mul(b, c), a); break;
                case 8: v = result ^ wrapping_add(wrapping_mul(c, a), b); break;
                case 9: v = result ^ wrapping_mul(wrapping_mul(a, b), c); break;
                case 10: v = result ^ ((__uint128_t)a << 64 | b) % (c + 1); break;
                case 11: v = result ^ ((__uint128_t)b << 64 | c) % ((__uint128_t)rl64(result, (uint32_t)r) << 64 | a + 2); break;
                case 12: v = result ^ ((__uint128_t)c << 64 | a) / ((__uint128_t)(b + 3)); break;
                case 13: v = result ^ ((__uint128_t)rl64(result, (uint32_t)r) << 64 | b) / (((__uint128_t)a << 64) | (c + 4)); break;
                case 14: v = result ^ (((__uint128_t)b << 64 | a) * c >> 64); break;
                case 15: v = result ^ (((__uint128_t)a << 64 | c) * ((__uint128_t)rr64(result, (uint32_t)r) << 64 | b) >> 64); break;
            }

            result = rl64(v, 1);

            // printf("post result: %llu\n", result);

            uint64_t t = mem_buffer_a[XELIS_BUFFER_SIZE_V2 - j - 1] ^ result;
            mem_buffer_a[XELIS_BUFFER_SIZE_V2 - j - 1] = t;
            mem_buffer_b[j] ^= rr64(t, (uint32_t)result);
        }
        // printf("post result: %llu\n", result);
        addr_a = result;
        addr_b = isqrt(result);
    }

  // Crc32 crc32;
  // crc32.input(scratch_pad, XELIS_MEMORY_SIZE_V2 * 8);
  // std::cout << "Stage 3 scratch pad CRC32: 0x" << std::hex << std::setw(8) << std::setfill('0') << crc32.result() << std::endl;
}

// void xelis_hash_old(byte *input, workerData_xelis_v2 &worker, byte *hashResult)
// {
//   // printf("input: ");
//   // for (int i = 0; i < 200; i++) {
//   //   printf("%02x", input[i]);
//   // }
//   // printf("\n\n");
//   worker.int_input = reinterpret_cast<uint64_t *>(input);

//   // Stage 1
//   stage_1(worker.int_input, worker.scratchPad,
//           0, XELIS_STAGE_1_MAX - 1,
//           0, XELIS_KECCAK_WORDS - 1);
//   stage_1(worker.int_input, worker.scratchPad,
//           XELIS_STAGE_1_MAX, XELIS_STAGE_1_MAX,
//           0, 17);

//   // Stage 2
//   __builtin_prefetch(worker.slots, 1, 3);
//   __builtin_prefetch(worker.smallPad, 1, 3);
//   worker.smallPad = reinterpret_cast<uint32_t *>(worker.scratchPad);

//   std::copy(&worker.smallPad[XELIS_MEMORY_SIZE_V2 * 2 - XELIS_SLOT_LENGTH], &worker.smallPad[XELIS_MEMORY_SIZE_V2 * 2], worker.slots);

//   // if (__builtin_cpu_supports("avx512f")) stage_2_avx512(worker.int_input, worker.smallPad, worker.indices, worker.slots);
//   stage_2(worker.int_input, worker.smallPad, worker.indices, worker.slots);
//   // else if (__builtin_cpu_supports("sse2")) stage_2_sse2(worker.int_input, worker.smallPad, worker.indices, worker.slots);
//   // else stage_2_scalar(worker.int_input, worker.smallPad, worker.indices, worker.slots);

//   // Stage 3
//   stage_3(worker.scratchPad, hashResult);
// }

void xelis_hash_v2(byte *input, workerData_xelis_v2 &worker, byte *hashResult)
{
  stage_1(input, worker.scratchPad, 112);
  // stage_3(worker.scratchPad);
}

void xelis_benchmark_cpu_hash_v2()
{
  const uint32_t ITERATIONS = 1000;
  byte input[112] = {0};
  alignas(64) workerData_xelis_v2 worker;
  alignas(64) byte hash_result[XELIS_HASH_SIZE] = {0};

  auto start = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < ITERATIONS; ++i)
  {
    input[0] = i & 0xFF;
    input[1] = (i >> 8) & 0xFF;
    xelis_hash_v2(input, worker, hash_result);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time took: " << elapsed.count() << " ms" << std::endl;
  std::cout << "H/s: " << (ITERATIONS * 1000.0 / elapsed.count()) << std::endl;
  std::cout << "ms per hash: " << (elapsed.count() / ITERATIONS) << std::endl;
}

static int char2int(char input)
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
static void hex2bin(const char *src, char *target)
{
  while (*src && src[1])
  {
    *(target++) = char2int(*src) * 16 + char2int(src[1]);
    src += 2;
  }
}

static const char *testTemplate = "97dff4761917c2692df3be38e72ca7a59c3f55252e2245cc21564ef65fa8ea6f0000018f22fe78f80000000000064202f2a40463ccfcea839c4950a56ee38fa69c7ce2d4ba45d4b060cc63c297fb73b8a09c69661b1690b0a238d096a7ccb3cb204ce5dd604da9bb6c79c4ab00000000";

namespace xelis_tests_v2
{
  using Hash = std::array<byte, XELIS_HASH_SIZE>;

  bool test_real()
  {
    alignas(64) workerData_xelis_v2 worker;
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

    xelis_hash_v2(input, worker, hash_result);

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
    alignas(64) workerData_xelis_v2 worker;
    byte hash_result[XELIS_HASH_SIZE] = {0};

    xelis_hash_v2(input, worker, hash_result);

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
    alignas(32) byte input[112] = {0};
    Hash expected_hash = {
        21, 151, 162, 132, 117, 237, 123, 209, 162, 83, 125, 103,
        120, 231, 142, 171, 252, 240, 191, 215, 40, 185,
        76, 205, 230, 124, 118, 192, 230, 103, 128, 118};

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

void xelis_runTests_v2()
{
  bool all_tests_passed = true;
  all_tests_passed &= xelis_tests_v2::test_zero_input();
  all_tests_passed &= xelis_tests_v2::test_xelis_input();

  xelis_tests_v2::test_real();

  if (all_tests_passed)
  {
    std::cout << "XELIS-HASH-V2: All tests passed!" << std::endl;
  }
  else
  {
    std::cout << "XELIS-HASH-V2: Some tests failed!" << std::endl;
  }
}

#else
// These are just to satisfy compilation on AARCH64
void xelis_hash_v2(byte *input, workerData_xelis_v2 &worker, byte *hashResult)
{
}

void xelis_benchmark_cpu_hash()
{
}

void xelis_runTests()
{
}

#endif // __x86_64__
