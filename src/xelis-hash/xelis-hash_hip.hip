#include "xelis-hash_hip.hpp"
#include <stdlib.h>
#include <iostream>

#include <emmintrin.h>
#include <immintrin.h>
#include <numeric>
#include <chrono>
#include <cstring>
#include <array>
#include <cassert>

#include "aes.tab"

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif
#define rl64(x, a) (((x << a%64) | (x >> (64 - a%64))))

#define SWAP_UINT64(x) (((uint64_t)__brev((uint32_t)(x)) << 32) | __brev((uint32_t)((x) >> 32)))

__device__ const int sign_bit_values[8][8] = {
    {0, 0, 0, 0, 0, 0, 0, -1},
    {0, 0, 0, 0, 0, 0, -1, 0},
    {0, 0, 0, 0, 0, -1, 0, 0},
    {0, 0, 0, 0, -1, 0, 0, 0},
    {0, 0, 0, -1, 0, 0, 0, 0},
    {0, 0, -1, 0, 0, 0, 0, 0},
    {0, -1, 0, 0, 0, 0, 0, 0},
    {-1, 0, 0, 0, 0, 0, 0, 0}
};

__device__ uint64_t swap_bytes(uint64_t value) {
    return ((value & 0xFF00000000000000ULL) >> 56) |
           ((value & 0x00FF000000000000ULL) >> 40) |
           ((value & 0x0000FF0000000000ULL) >> 24) |
           ((value & 0x000000FF00000000ULL) >> 8) |
           ((value & 0x00000000FF000000ULL) << 8) |
           ((value & 0x0000000000FF0000ULL) << 24) |
           ((value & 0x000000000000FF00ULL) << 40) |
           ((value & 0x00000000000000FFULL) << 56);
}


__device__ void aes_round(uint *block, const uint *key) {
    uint s0 = block[0];
    uint s1 = block[1];
    uint s2 = block[2];
    uint s3 = block[3];

    uint t0 = cTe0[s0 >> 24] ^ cTe1[(s1 >> 16) & 0xff] ^ cTe2[(s2 >>  8) & 0xff] ^ cTe3[s3 & 0xff] ^ key[0];
    uint t1 = cTe0[s1 >> 24] ^ cTe1[(s2 >> 16) & 0xff] ^ cTe2[(s3 >>  8) & 0xff] ^ cTe3[s0 & 0xff] ^ key[1];
    uint t2 = cTe0[s2 >> 24] ^ cTe1[(s3 >> 16) & 0xff] ^ cTe2[(s0 >>  8) & 0xff] ^ cTe3[s1 & 0xff] ^ key[2];
    uint t3 = cTe0[s3 >> 24] ^ cTe1[(s0 >> 16) & 0xff] ^ cTe2[(s1 >>  8) & 0xff] ^ cTe3[s2 & 0xff] ^ key[3];

    block[0] = t0;
    block[1] = t1;
    block[2] = t2;
    block[3] = t3;
}

__device__ static const  uint64_t RC[12] = {
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
    0x8000000080008008ULL
};

__device__ static const uint32_t RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ static const uint32_t PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

__device__ void keccakp_1600_12(uint64_t state[25]) {
    for (int round = 0; round < 12; ++round) {
        uint64_t C[5] = {0};
        uint64_t D[5] = {0};

        // Theta step
        for (int i = 0; i < 5; ++i) {
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        }
        for (int i = 0; i < 5; ++i) {
            D[i] = C[(i + 4) % 5] ^ rl64(C[(i + 1) % 5], 1);
        }
        for (int i = 0; i < 25; ++i) {
            state[i] ^= D[i % 5];
        }

        // Rho and Pi steps
        uint64_t last = state[1];
        for (int i = 0; i < 24; ++i) {
            uint32_t j = PI[i];
            uint64_t temp = state[j];
            state[j] = rl64(last, RHO[i]);
            last = temp;
        }

        // Chi step
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; ++i) {
                C[i] = state[j + i];
            }
            for (int i = 0; i < 5; ++i) {
                state[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
            }
        }

        // Iota step
        state[0] ^= RC[round];
    }
}

__device__ void keccakp_1600_12_unrolled(uint64_t state[25]) {
    for (int round = 0; round < 12; ++round) {
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
        for (int j = 0; j < 25; j += 5) {
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

__device__ void stage_1(uint64_t* int_input, uint64_t* scratchPad, 
  uint16_t A1, uint16_t A2, byte B1, byte B2
) {
  for (size_t i = A1; i <= A2; ++i) {
    keccakp_1600_12(int_input);

    uint64_t rand_int = 0;
    for (size_t j = B1; j <= B2; ++j) {
      byte pair_idx = (j + 1) % XELIS_HIP_KECCAK_WORDS;
      byte pair_idx2 = (j + 2) % XELIS_HIP_KECCAK_WORDS;

      size_t target_idx = i * XELIS_HIP_KECCAK_WORDS + j;
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
      if (target_idx == 7) printf("b = %llu\n", b);
      // if (i == A1 && j < 5) 
      //   printf("i = %llu, j = %llu, b = %llu\n", i, j, b);
    }
  }
}

__device__ void stage_2(workerData_xelis_hip &worker, uint64_t *input, uint32_t *smallPad, byte *indices, uint32_t *slots) {
  for (byte iter = 0; iter < XELIS_HIP_ITERS; ++iter) {
    for (uint16_t j = 0; j < (XELIS_HIP_MEMORY_SIZE * 2) / XELIS_HIP_SLOT_LENGTH; ++j) {
      if (j > 16) break;
      for (int i = 0; i < XELIS_HIP_SLOT_LENGTH; i++) {
        indices[i] = i;
      }
      for (int slot_idx = XELIS_HIP_SLOT_LENGTH - 1; slot_idx >= 0; --slot_idx) {
        uint16_t index_in_indices = smallPad[j * XELIS_HIP_SLOT_LENGTH + slot_idx] % (slot_idx + 1);
        if (slot_idx < 16 && j == 0) {
          printf("sIndex: %d\n", (j * XELIS_HIP_SLOT_LENGTH + slot_idx)/2);
          printf("scratchPad in question: %llu\n", *(uint64_t*)&smallPad[j * XELIS_HIP_SLOT_LENGTH + slot_idx]);
          printf("smallPad in question: %lu\n", smallPad[j * XELIS_HIP_SLOT_LENGTH + slot_idx]);
        }
        // uint16_t index = indices[index_in_indices];
        // indices[index_in_indices] = indices[slot_idx];

        // // SCALAR implementation
        // uint32_t sum = slots[index];
        // uint16_t offset = j * XELIS_HIP_SLOT_LENGTH;

        // for (uint16_t k = 0; k < index; ++k) {
        //   uint32_t pad = smallPad[offset + k];
        //   sum = (slots[k] >> 31 == 0) ? sum + pad : sum - pad;
        // }

        // for (uint16_t k = index + 1; k < XELIS_HIP_SLOT_LENGTH; ++k) {
        //   uint32_t pad = smallPad[offset + k];
        //   sum = (slots[k] >> 31 == 0) ? sum + pad : sum - pad;
        // }
        // slots[index] = sum;
        // // if (slot_idx < 16 && j == 0) printf("sum: %llu, index: %u\n", sum, index);
      }
    }
  }

  // Copy slots back to the last SLOT_LENGTH elements of smallPad
  memcpy(&smallPad[XELIS_HIP_MEMORY_SIZE*2-XELIS_HIP_SLOT_LENGTH], &slots[0], XELIS_HIP_SLOT_LENGTH*4);
}

__device__ void stage_3(uint64_t* scratchPad, byte* hashResult) {
    const byte key[16] = {0};
    byte block[16] = {0};

    uint64_t addr_a = (scratchPad[XELIS_HIP_MEMORY_SIZE - 1] >> 15) & 0x7FFF;
    uint64_t addr_b = scratchPad[XELIS_HIP_MEMORY_SIZE - 1] & 0x7FFF;

    uint64_t mem_buffer_a[XELIS_HIP_BUFFER_SIZE];
    uint64_t mem_buffer_b[XELIS_HIP_BUFFER_SIZE];

    for (byte i = 0; i < XELIS_HIP_BUFFER_SIZE; ++i) {
        mem_buffer_a[i] = scratchPad[(addr_a + i) % XELIS_HIP_MEMORY_SIZE];
        mem_buffer_b[i] = scratchPad[(addr_b + i) % XELIS_HIP_MEMORY_SIZE];
    }

    for (uint16_t i = 0; i < XELIS_HIP_SCRATCHPAD_ITERS; ++i) {
        byte *mem_a = reinterpret_cast<byte*>(&mem_buffer_a[i % XELIS_HIP_BUFFER_SIZE]);
        byte *mem_b = reinterpret_cast<byte*>(&mem_buffer_b[i % XELIS_HIP_BUFFER_SIZE]);

        memcpy(block, mem_b, 8);
        memcpy(block + 8, mem_a, 8);

        aes_round(
          reinterpret_cast<uint*>(const_cast<byte*>(block)), 
          reinterpret_cast<const uint*>(const_cast<const byte*>(key)
        ));

        uint64_t hash1 = *(reinterpret_cast<uint64_t*>(&block[0]));
        uint64_t hash2 = *(reinterpret_cast<uint64_t*>(mem_a)) ^ *(reinterpret_cast<uint64_t*>(mem_b));

        uint64_t result = ~(hash1 ^ hash2);

        for (size_t j = 0; j < XELIS_HIP_HASH_SIZE; ++j) {
          uint64_t a = mem_buffer_a[(j + i) % XELIS_HIP_BUFFER_SIZE];
          uint64_t b = mem_buffer_b[(j + i) % XELIS_HIP_BUFFER_SIZE];

          uint64_t v;
          switch ((result >> (j * 2)) & 0xf) {
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
        mem_buffer_a[i % XELIS_HIP_BUFFER_SIZE] = result;
        mem_buffer_b[i % XELIS_HIP_BUFFER_SIZE] = scratchPad[addr_b];

        addr_a = (result >> 15) & 0x7FFF;
        scratchPad[addr_a] = result;

        size_t index = XELIS_HIP_SCRATCHPAD_ITERS - i - 1;
        // printf("post result: %llu\n", result);
        if (index < 4) {
            uint64_t be_result = SWAP_UINT64(result);
            memcpy(&hashResult[index * 8], reinterpret_cast<byte*>(&be_result), 8);
        }
    }
}

void xelis_benchmark_gpu_hash_hip() {
    // const uint32_t ITERATIONS = 1000;
    // byte input[200] = {0};
    // workerData_xelis_hip worker;
    // byte hash_result[XELIS_HIP_HASH_SIZE] = {0};

    // auto start = std::chrono::high_resolution_clock::now();
    // for (uint32_t i = 0; i < ITERATIONS; ++i) {
    //     input[0] = i & 0xFF;
    //     input[1] = (i >> 8) & 0xFF;
    //     xelis_hash_hip(input, worker, hash_result);
    // }
    // auto end = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<double, std::milli> elapsed = end - start;
    // std::cout << "Time took: " << elapsed.count() << " ms" << std::endl;
    // std::cout << "H/s: " << (ITERATIONS * 1000.0 / elapsed.count()) << std::endl;
    // std::cout << "ms per hash: " << (elapsed.count() / ITERATIONS) << std::endl;
}

__global__ void s1_kernel(uint64_t* int_input, uint64_t* scratchPad, 
  uint16_t A1, uint16_t A2, byte B1, byte B2) {
  stage_1(int_input, scratchPad, A1, A2, B1, B2);
}

__global__ void s2_kernel(uint64_t* input, workerData_xelis_hip &worker) {
  // for (int i = 0; i < XELIS_HIP_BYTES_ARRAY_INPUT/8; i++) {
  //   printf("%llu, ", input[i]);
  // }
  // printf("\n");

  memcpy(worker.slots, &worker.smallPad[XELIS_HIP_MEMORY_SIZE*2-XELIS_HIP_SLOT_LENGTH], XELIS_HIP_SLOT_LENGTH*4);

  stage_2(worker, input, worker.smallPad, worker.indices, worker.slots);
}

__global__ void s3_kernel(uint64_t* scratchPad, byte* hashResult) {
  stage_3(scratchPad, hashResult);
}

void xelis_hash_hip(byte* input, workerData_xelis_hip &worker, byte *hashResult) {
  worker.int_input = reinterpret_cast<uint64_t*>(input);
  // Stage 1
  s1_kernel<<<1,1>>>(worker.int_input, reinterpret_cast<uint64_t*>(worker.scratchPad), 
    0, XELIS_HIP_STAGE_1_MAX - 1,
    0, XELIS_HIP_KECCAK_WORDS - 1
  );
  hipDeviceSynchronize();

  s1_kernel<<<1,1>>>(worker.int_input, reinterpret_cast<uint64_t*>(worker.scratchPad), 
    XELIS_HIP_STAGE_1_MAX, XELIS_HIP_STAGE_1_MAX,
    0, 17
  );
  hipDeviceSynchronize();

  worker.smallPad = reinterpret_cast<uint32_t*>(worker.scratchPad);
  for (int i = 0; i < 200; i++) printf("a");

  s2_kernel<<<1,1>>>(reinterpret_cast<uint64_t*>(input), worker);
  hipDeviceSynchronize();

  // // Stage 3
  // s3_kernel<<<1,1>>>(reinterpret_cast<uint64_t*>(worker.scratchPad), hashResult);
  // hipDeviceSynchronize();
}

int char2int_hip(char input)
{
  if(input >= '0' && input <= '9')
    return input - '0';
  if(input >= 'A' && input <= 'F')
    return input - 'A' + 10;
  if(input >= 'a' && input <= 'f')
    return input - 'a' + 10;
  throw std::invalid_argument("Invalid input string");
}

// This function assumes src to be a zero terminated sanitized string with
// an even number of [0-9a-f] characters, and target to be sufficiently large
void hex2bin_hip(const char* src, char* target)
{
  while(*src && src[1])
  {
    *(target++) = char2int_hip(*src)*16 + char2int_hip(src[1]);
    src += 2;
  }
}


namespace tests_hip {
    char* testTemplate = "97dff4761917c2692df3be38e72ca7a59c3f55252e2245cc21564ef65fa8ea6f0000018f22fe78f80000000000064202f2a40463ccfcea839c4950a56ee38fa69c7ce2d4ba45d4b060cc63c297fb73b8a09c69661b1690b0a238d096a7ccb3cb204ce5dd604da9bb6c79c4ab00000000";
    using Hash = std::array<byte, XELIS_HIP_HASH_SIZE>;

    bool test_input(const char* test_name, byte* input, size_t input_size, const Hash& expected_hash) {
        workerData_xelis_hip worker_d;
        byte dummyBytes[sizeof(workerData_xelis_hip)] = {0};
        hipMalloc((void**)&worker_d, sizeof(workerData_xelis_hip));
        hipMemcpy(&worker_d, dummyBytes, sizeof(workerData_xelis_hip), hipMemcpyHostToDevice);

        byte hash_result[XELIS_HIP_HASH_SIZE] = {0};
        byte *hash_result_d;

        byte *input_d;
        
        hipMalloc((void**)&input_d, input_size);
        hipMemcpy(input_d, input, input_size, hipMemcpyHostToDevice);

        hipMalloc((void**)&hash_result_d, XELIS_HIP_HASH_SIZE);
        xelis_hash_hip(input_d, worker_d, hash_result_d);

        hipMemcpy(hash_result, hash_result_d, XELIS_HIP_HASH_SIZE, hipMemcpyDeviceToHost);

        if (std::memcmp(hash_result, expected_hash.data(), XELIS_HIP_HASH_SIZE) != 0) {
            std::cout << "Test '" << test_name << "' failed!" << std::endl;
            std::cout << "HIP || Expected hash: ";
            for (size_t i = 0; i < XELIS_HIP_HASH_SIZE; ++i) {
                printf("%02x ", expected_hash[i]);
            }
            std::cout << std::endl;
            std::cout << "HIP || Actual hash:   ";
            for (size_t i = 0; i < XELIS_HIP_HASH_SIZE; ++i) {
                printf("%02x ", hash_result[i]);
            }
            std::cout << std::endl;
            return false;
        }
        return true;
    }

    bool test_zero_input() {
        byte input[200] = {0};
        Hash expected_hash = {
            0x0e, 0xbb, 0xbd, 0x8a, 0x31, 0xed, 0xad, 0xfe, 0x09, 0x8f, 0x2d, 0x77, 0x0d, 0x84,
            0xb7, 0x19, 0x58, 0x86, 0x75, 0xab, 0x88, 0xa0, 0xa1, 0x70, 0x67, 0xd0, 0x0a, 0x8f,
            0x36, 0x18, 0x22, 0x65
        };

        return test_input("test_zero_input", input, 200, expected_hash);
    }

    bool test_xelis_input() {
        byte input[XELIS_HIP_BYTES_ARRAY_INPUT] = {0};

        const char* custom = "xelis-hashing-algorithm";
        std::memcpy(input, custom, std::strlen(custom));

        Hash expected_hash = {
            106, 106, 173, 8, 207, 59, 118, 108, 176, 196, 9, 124, 250, 195, 3,
            61, 30, 146, 238, 182, 88, 83, 115, 81, 139, 56, 3, 28, 176, 86, 68, 21
        };
        return test_input("test_xelis_input", input, sizeof(input), expected_hash);
    }
}

void xelis_runTests_hip() {
  bool all_tests_passed = true;
  all_tests_passed &= tests_hip::test_zero_input();
  all_tests_passed &= tests_hip::test_xelis_input();

  if (all_tests_passed) {
      std::cout << "HIP || XELIS-HASH: All tests passed!" << std::endl;
  } else {
      std::cout << "HIP || XELIS-HASH: Some tests failed!" << std::endl;
  }
}