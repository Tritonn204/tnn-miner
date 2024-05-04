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

#include <stdio.h>
#include <stdint.h>

#include "aes.tab"

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif
#define rl64(x, a) (((x << a%64) | (x >> (64 - a%64))))

#define S1_TPB 128
#define S3_TPB 64

#define REVERSE_BYTES_64(value) \
    ((((value) & 0xFF00000000000000ULL) >> 56) | \
     (((value) & 0x00FF000000000000ULL) >> 40) | \
     (((value) & 0x0000FF0000000000ULL) >> 24) | \
     (((value) & 0x000000FF00000000ULL) >>  8) | \
     (((value) & 0x00000000FF000000ULL) <<  8) | \
     (((value) & 0x0000000000FF0000ULL) << 24) | \
     (((value) & 0x000000000000FF00ULL) << 40) | \
     (((value) & 0x00000000000000FFULL) << 56))

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

	__device__ int sbox[256] =   {
	//0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, //0
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, //1
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, //2
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, //3
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, //4
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, //5
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, //6
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, //7
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, //8
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, //9
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, //A
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, //B
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, //C
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, //D
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, //E
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 }; //F

// Function to perform the AddRoundKey step
__device__ void AddRoundKey(uint8_t state[], const uint8_t roundKey[]) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

// Function to perform the SubBytes step
__device__ void SubBytes(uint8_t state[]) {
    for (int i = 0; i < 16; i++) {
        state[i] = sbox[state[i]];
    }
}

// Function to perform the ShiftRows step
__device__ void ShiftRows(uint8_t state[]) {
    uint8_t temp;

    // Rotate first row 1 column to the left
    temp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = temp;

    // Rotate second row 2 columns to the left
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;

    temp = state[6];
    state[6] = state[14];
    state[14] = temp;

    // Rotate third row 3 columns to the left
    temp = state[3];
    state[3] = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = temp;
}

// Function to perform the MixColumns step
__device__ void MixColumns(uint8_t state[]) {
    uint8_t i;
    uint8_t Tmp, Tm;

    for (i = 0; i < 16; i += 4) {
        uint8_t t = state[i];
        Tmp = state[i] ^ state[i + 1] ^ state[i + 2] ^ state[i + 3];
        Tm = state[i] ^ state[i + 1];
        Tm = (Tm << 1) ^ (((Tm >> 7) & 1) * 0x1B);
        state[i] ^= Tm ^ Tmp;
        Tm = state[i + 1] ^ state[i + 2];
        Tm = (Tm << 1) ^ (((Tm >> 7) & 1) * 0x1B);
        state[i + 1] ^= Tm ^ Tmp;
        Tm = state[i + 2] ^ state[i + 3];
        Tm = (Tm << 1) ^ (((Tm >> 7) & 1) * 0x1B);
        state[i + 2] ^= Tm ^ Tmp;
        Tm = state[i + 3] ^ t;
        Tm = (Tm << 1) ^ (((Tm >> 7) & 1) * 0x1B);
        state[i + 3] ^= Tm ^ Tmp;
    }
}

// Function to perform a single round of AES encryption
__device__ void aes_single_round(uint8_t block[], const uint8_t roundKey[]) {
    AddRoundKey(block, roundKey);
    SubBytes(block);
    ShiftRows(block);
    MixColumns(block);
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

__global__ void s1_kernel(workerData_xelis_hip *worker, 
  uint16_t A1, uint16_t A2, byte B1, byte B2) {

  for (size_t i = A1; i <= A2; ++i) {
    keccakp_1600_12(worker->int_input);

    uint64_t rand_int = 0;
    for (size_t j = B1; j <= B2; ++j) {
      byte pair_idx = (j + 1) % XELIS_HIP_KECCAK_WORDS;
      byte pair_idx2 = (j + 2) % XELIS_HIP_KECCAK_WORDS;

      size_t target_idx = i * XELIS_HIP_KECCAK_WORDS + j;
      uint64_t a = worker->int_input[j] ^ rand_int;

      uint64_t left = worker->int_input[pair_idx];
      uint64_t right = worker->int_input[pair_idx2];
      
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
      worker->scratchPad[target_idx] = b;
      // if (target_idx == 7) printf("b = %llu\n", b);
      // if (target_idx == 7) printf("assigned b = %llu\n", scratchPad[target_idx]);
      // if (i == A1 && j < 5) 
      //   printf("i = %llu, j = %llu, b = %llu\n", i, j, b);
    }
  }
}

__global__ void s1_kernel_optimized(workerData_xelis_hip_optimized *worker, 
  uint16_t A1, uint16_t A2, byte B1, byte B2, int batchSize) {

  uint32_t flatid = blockIdx.x * blockDim.x + threadIdx.x;

  if (flatid < batchSize) {
    // printf("flatid = %d\n", flatid);

    // if (flatid == 1) printf("first check: %llu\n", worker->int_input[0]);
    for (size_t i = A1; i <= A2; ++i) {
      keccakp_1600_12(worker->int_input[flatid]);

      uint64_t rand_int = 0;
      for (size_t j = B1; j <= B2; ++j) {
        byte pair_idx = (j + 1) % XELIS_HIP_KECCAK_WORDS;
        byte pair_idx2 = (j + 2) % XELIS_HIP_KECCAK_WORDS;

        size_t target_idx = i * XELIS_HIP_KECCAK_WORDS + j;
        uint64_t a = worker->int_input[flatid][j] ^ rand_int;

        uint64_t left = worker->int_input[flatid][pair_idx];
        uint64_t right = worker->int_input[flatid][pair_idx2];
        
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
        worker->scratchPad[flatid*XELIS_HIP_MEMORY_SIZE + target_idx] = b;
        // if (target_idx == 7) printf("b = %llu\n", b);
        // if (target_idx == 7) printf("assigned b = %llu\n", scratchPad[target_idx]);
        // if (flatid == 1 && i == A1 && j < 5)
        //   printf("i = %llu, j = %llu, b = %llu\n", i, j, b);
      }
    }
  }
  __syncthreads();
}

__global__ void s2_kernel(workerData_xelis_hip *worker) {
  // for (int i = 0; i < XELIS_HIP_BYTES_ARRAY_INPUT/8; i++) {
  //   printf("%llu, ", input[i]);
  // }
  // printf("\n");

  worker->smallPad = reinterpret_cast<uint32_t*>(worker->scratchPad);
  memcpy(worker->slots, &worker->smallPad[XELIS_HIP_MEMORY_SIZE*2-XELIS_HIP_SLOT_LENGTH], XELIS_HIP_SLOT_LENGTH*4);

  for (uint16_t j = 0; j < (XELIS_HIP_MEMORY_SIZE * 2) / XELIS_HIP_SLOT_LENGTH; ++j) {
    for (int i = 0; i < XELIS_HIP_SLOT_LENGTH; i++) {
      worker->indices[i] = i;
    }
    for (int slot_idx = XELIS_HIP_SLOT_LENGTH - 1; slot_idx >= 0; --slot_idx) {
      uint16_t index_in_indices = worker->smallPad[j * XELIS_HIP_SLOT_LENGTH + slot_idx] % (slot_idx + 1);
      uint16_t index = worker->indices[index_in_indices];
      worker->indices[index_in_indices] = worker->indices[slot_idx];

      // SCALAR implementation
      uint32_t sum = worker->slots[index];
      uint16_t offset = j * XELIS_HIP_SLOT_LENGTH;

      for (uint16_t k = 0; k < index; ++k) {
        uint32_t pad = worker->smallPad[offset + k];
        sum = (worker->slots[k] >> 31 == 0) ? sum + pad : sum - pad;
      }

      for (uint16_t k = index + 1; k < XELIS_HIP_SLOT_LENGTH; ++k) {
        uint32_t pad = worker->smallPad[offset + k];
        sum = (worker->slots[k] >> 31 == 0) ? sum + pad : sum - pad;
      }
      worker->slots[index] = sum;
      // if (slot_idx < 16 && j == 0) printf("sum: %llu, index: %u\n", sum, index);
    }
  }

  // Copy slots back to the last SLOT_LENGTH elements of smallPad
  memcpy(&worker->smallPad[XELIS_HIP_MEMORY_SIZE*2-XELIS_HIP_SLOT_LENGTH], &worker->slots[0], XELIS_HIP_SLOT_LENGTH*4);
}

__global__ void s2_kernel_optimized(workerData_xelis_hip_optimized *worker) {
  __shared__ uint32_t shared_slots[XELIS_HIP_SLOT_LENGTH];
  __shared__ uint32_t shared_indices[XELIS_HIP_SLOT_LENGTH];

  __shared__ uint32_t sum;
  __shared__  byte index_in_indices;
  __shared__  byte index;

  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;

  shared_slots[tid] = worker->smallPad[bid][XELIS_HIP_MEMORY_SIZE*2-XELIS_HIP_SLOT_LENGTH + tid];
  __syncthreads();

// Iterate over the memory
  for (uint16_t j = 0; j < (XELIS_HIP_MEMORY_SIZE * 2) / XELIS_HIP_SLOT_LENGTH; ++j)
  {
    shared_indices[tid] = tid;
    __syncthreads();

    for (int slot_idx = XELIS_HIP_SLOT_LENGTH - 1; slot_idx >= 0; --slot_idx)
    {
      if (tid == 0) {
        // Fisher-Yates shuffle
        index_in_indices = worker->smallPad[bid][j * XELIS_HIP_SLOT_LENGTH + slot_idx] % (slot_idx + 1);
        index = shared_indices[index_in_indices];
        shared_indices[index_in_indices] = shared_indices[slot_idx];
        sum = shared_slots[index];
      }
      __syncthreads();

      if (tid != index) {
        uint32_t pad = worker->smallPad[bid][XELIS_HIP_SLOT_LENGTH * j + tid];
        if (shared_slots[tid] >> 31 == 0) atomicAdd(&sum, pad);
        else atomicSub(&sum, pad);
      }
      __syncthreads();
      if (tid == 0) {
        shared_slots[index] = sum;
        // if (bid == 1 && slot_idx == 0 && j % 100 == 0) printf("sum: %llu, index: %u\n", sum, index);
      }
      __syncthreads();      
    }
  }

  worker->slots[bid*XELIS_HIP_SLOT_LENGTH + tid] = shared_slots[tid];
  __syncthreads();
  if (tid == 0) memcpy(
    &worker->smallPad[bid][XELIS_HIP_MEMORY_SIZE*2-XELIS_HIP_SLOT_LENGTH], 
    &worker->slots[bid*XELIS_HIP_SLOT_LENGTH], XELIS_HIP_SLOT_LENGTH * sizeof(uint32_t)
  );
}

__global__ void s3_kernel(workerData_xelis_hip *worker, byte* hashResult) {
  const byte key[16] = {0};
  byte block[16] = {0};

  uint64_t addr_a = (worker->scratchPad[XELIS_HIP_MEMORY_SIZE - 1] >> 15) & 0x7FFF;
  uint64_t addr_b = worker->scratchPad[XELIS_HIP_MEMORY_SIZE - 1] & 0x7FFF;

  uint64_t mem_buffer_a[XELIS_HIP_BUFFER_SIZE];
  uint64_t mem_buffer_b[XELIS_HIP_BUFFER_SIZE];

  for (byte i = 0; i < XELIS_HIP_BUFFER_SIZE; ++i)
  {
    mem_buffer_a[i] = worker->scratchPad[(addr_a + i) % XELIS_HIP_MEMORY_SIZE];
    mem_buffer_b[i] = worker->scratchPad[(addr_b + i) % XELIS_HIP_MEMORY_SIZE];
  }

  for (uint16_t i = 0; i < XELIS_HIP_SCRATCHPAD_ITERS; ++i)
  {
    byte *mem_a = reinterpret_cast<byte *>(&mem_buffer_a[i % XELIS_HIP_BUFFER_SIZE]);
    byte *mem_b = reinterpret_cast<byte *>(&mem_buffer_b[i % XELIS_HIP_BUFFER_SIZE]);

    memcpy(block, mem_b, 8);
    memcpy(block + 8, mem_a, 8);

    aes_single_round(
        block,
        key);

    uint64_t hash1 = *(reinterpret_cast<uint64_t *>(&block[0]));
    uint64_t hash2 = *(reinterpret_cast<uint64_t *>(mem_a)) ^ *(reinterpret_cast<uint64_t *>(mem_b));

    uint64_t result = ~(hash1 ^ hash2);

    for (size_t j = 0; j < XELIS_HIP_HASH_SIZE; ++j)
    {
      uint64_t a = mem_buffer_a[(j + i) % XELIS_HIP_BUFFER_SIZE];
      uint64_t b = mem_buffer_b[(j + i) % XELIS_HIP_BUFFER_SIZE];

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
    mem_buffer_a[i % XELIS_HIP_BUFFER_SIZE] = result;
    mem_buffer_b[i % XELIS_HIP_BUFFER_SIZE] = worker->scratchPad[addr_b];

    addr_a = (result >> 15) & 0x7FFF;
    worker->scratchPad[addr_a] = result;

    size_t index = XELIS_HIP_SCRATCHPAD_ITERS - i - 1;
    // printf("post result: %llu\n", result);
    if (index < 4)
    {
      uint64_t be_result = REVERSE_BYTES_64(result);
      memcpy(&hashResult[index * 8], reinterpret_cast<byte *>(&be_result), 8);
    }
  }
}


__global__ void 
__launch_bounds__(S3_TPB, 2)
s3_kernel_optimized(workerData_xelis_hip_optimized *worker, byte* hashResult, int batchSize) {

  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t flatid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ byte key[16*S3_TPB];
  __shared__ byte block[16*S3_TPB];

  // __shared__ uint64_t addr_a[S3_TBP];
  // __shared__ uint64_t addr_b[S3_TBP];

  // if (flatid < batchSize) {
  //   addr_a[tid] = (worker->scratchPad[flatid*XELIS_HIP_MEMORY_SIZE + XELIS_HIP_MEMORY_SIZE - 1] >> 15) & 0x7FFF;
  //   addr_b[tid] = worker->scratchPad[flatid*XELIS_HIP_MEMORY_SIZE + XELIS_HIP_MEMORY_SIZE - 1] & 0x7FFF;
  // }
  // __syncthreads();

  __shared__ uint64_t mem_buffer_a[XELIS_HIP_BUFFER_SIZE*S3_TPB];
  __shared__ uint64_t mem_buffer_b[XELIS_HIP_BUFFER_SIZE*S3_TPB];

  byte *mem_a;
  byte *mem_b;
  
  for (int i = tid; i < 16*S3_TPB; i += gridDim.x) {
    key[i] = 0;
    block[i] = 0;
  }
  __syncthreads();


  uint64_t addr_a = (worker->scratchPad[flatid*XELIS_HIP_MEMORY_SIZE + XELIS_HIP_MEMORY_SIZE - 1] >> 15) & 0x7FFF;
  uint64_t addr_b = worker->scratchPad[flatid*XELIS_HIP_MEMORY_SIZE + XELIS_HIP_MEMORY_SIZE - 1] & 0x7FFF;

  // uint64_t mem_buffer_a[XELIS_HIP_BUFFER_SIZE];
  // uint64_t mem_buffer_b[XELIS_HIP_BUFFER_SIZE];

  if (flatid < batchSize) {
    for (byte i = 0; i < XELIS_HIP_BUFFER_SIZE; ++i)
    {
      mem_buffer_a[tid*XELIS_HIP_BUFFER_SIZE + i] = worker->scratchPad[flatid * XELIS_HIP_MEMORY_SIZE + ((addr_a + i) % XELIS_HIP_MEMORY_SIZE)];
      mem_buffer_b[tid*XELIS_HIP_BUFFER_SIZE + i] = worker->scratchPad[flatid * XELIS_HIP_MEMORY_SIZE + ((addr_b + i) % XELIS_HIP_MEMORY_SIZE)];
    }
  }

  __syncthreads();

  if (flatid < batchSize) {
    for (uint16_t i = 0; i < XELIS_HIP_SCRATCHPAD_ITERS; ++i)
    {
      // byte *mem_a = reinterpret_cast<byte *>(&mem_buffer_a[i % XELIS_HIP_BUFFER_SIZE]);
      // byte *mem_b = reinterpret_cast<byte *>(&mem_buffer_b[i % XELIS_HIP_BUFFER_SIZE]);
      mem_a = reinterpret_cast<byte *>(&mem_buffer_a[tid*XELIS_HIP_BUFFER_SIZE + (i % XELIS_HIP_BUFFER_SIZE)]);
      mem_b = reinterpret_cast<byte *>(&mem_buffer_b[tid*XELIS_HIP_BUFFER_SIZE + (i % XELIS_HIP_BUFFER_SIZE)]);
      
      memcpy(&block[tid*16], mem_b, 8);
      memcpy(&block[tid*16] + 8, mem_a, 8);

      aes_single_round(
          &block[tid*16],
          &key[tid*16]);

      uint64_t hash1 = *(reinterpret_cast<uint64_t *>(&block[tid*16]));
      uint64_t hash2 = *(reinterpret_cast<uint64_t *>(mem_a)) ^ *(reinterpret_cast<uint64_t *>(mem_b));

      uint64_t result = ~(hash1 ^ hash2);

      for (size_t j = 0; j < XELIS_HIP_HASH_SIZE; ++j)
      {
        uint64_t a = mem_buffer_a[tid*XELIS_HIP_BUFFER_SIZE + (j + i) % XELIS_HIP_BUFFER_SIZE];
        uint64_t b = mem_buffer_b[tid*XELIS_HIP_BUFFER_SIZE + (j + i) % XELIS_HIP_BUFFER_SIZE];

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
      mem_buffer_a[tid*XELIS_HIP_BUFFER_SIZE + (i % XELIS_HIP_BUFFER_SIZE)] = result;
      mem_buffer_b[tid*XELIS_HIP_BUFFER_SIZE + (i % XELIS_HIP_BUFFER_SIZE)] = worker->scratchPad[flatid * XELIS_HIP_MEMORY_SIZE + addr_b];

      addr_a = (result >> 15) & 0x7FFF;
      worker->scratchPad[flatid * XELIS_HIP_MEMORY_SIZE + addr_a] = result;

      size_t index = XELIS_HIP_SCRATCHPAD_ITERS - i - 1;
      // if (flatid == 1 && i % 500 == 0) printf("post result: %llu\n", result);
      if (index < 4)
      {
        uint64_t be_result = REVERSE_BYTES_64(result);
        memcpy(&hashResult[flatid*XELIS_HIP_HASH_SIZE + index * 8], reinterpret_cast<byte *>(&be_result), 8);
      }
    }
  }
}

void printArrayToFile(const uint64_t* arr, size_t size, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        // Handle file opening error
        return;
    }

    for (size_t i = 0; i < size; ++i) {
        fprintf(file, "%llu", (unsigned long long)arr[i]);
        if (i < size - 1) {
            fprintf(file, ",");
        }
    }

    fclose(file);
}

__global__ void iCopy_kernel(workerData_xelis_hip *worker, byte *input) {
  worker->int_input = reinterpret_cast<uint64_t*>(input);
}

void xelis_hash_hip(byte* input, workerData_xelis_hip *worker, byte *hashResult) {
  // worker.int_input = reinterpret_cast<uint64_t*>(input);

  iCopy_kernel<<<1,1>>>(worker, input);

  // Stage 1
  s1_kernel<<<1,1>>>(worker, 
    0, XELIS_HIP_STAGE_1_MAX - 1,
    0, XELIS_HIP_KECCAK_WORDS - 1
  );
  hipDeviceSynchronize();

  s1_kernel<<<1,1>>>(worker, 
    XELIS_HIP_STAGE_1_MAX, XELIS_HIP_STAGE_1_MAX,
    0, 17
  );
  hipDeviceSynchronize();

  // workerData_xelis_hip *tempWorker = (workerData_xelis_hip*)malloc(sizeof(workerData_xelis_hip));
  // hipMemcpy(tempWorker, worker, sizeof(workerData_xelis_hip), hipMemcpyDeviceToHost);

  // printArrayToFile(tempWorker->scratchPad, XELIS_HIP_MEMORY_SIZE, "scratchPad.txt");

  // memcpy(worker.scratchPad, worker.smallPad, XELIS_HIP_MEMORY_SIZE*8);

  // printf("\n");

  
  s2_kernel<<<1,1>>>(worker);
  hipDeviceSynchronize();

  // Stage 3
  s3_kernel<<<1,1>>>(worker, hashResult);
  hipDeviceSynchronize();
}

void xelis_hash_hip_optimized(workerData_xelis_hip_optimized *worker, byte *hashResult, int batchSize) {
  // Stage 1

  int s1Blocks = (batchSize + S1_TPB-1) / S1_TPB;

  s1_kernel_optimized<<<s1Blocks,S1_TPB>>>(worker, 
    0, XELIS_HIP_STAGE_1_MAX - 1,
    0, XELIS_HIP_KECCAK_WORDS - 1,
    batchSize
  );
  hipDeviceSynchronize();

  s1_kernel_optimized<<<s1Blocks,S1_TPB>>>(worker, 
    XELIS_HIP_STAGE_1_MAX, XELIS_HIP_STAGE_1_MAX,
    0, 17,
    batchSize
  );
  hipDeviceSynchronize();

  dim3 block(XELIS_HIP_SLOT_LENGTH);
  dim3 grid(batchSize);

  s2_kernel_optimized<<<grid, block>>>(worker);
  hipDeviceSynchronize();

  int s3Blocks = (batchSize + S3_TPB-1) / S3_TPB;
  // Stage 3
  s3_kernel_optimized<<<s3Blocks,S3_TPB>>>(worker, hashResult, batchSize);
  hipDeviceSynchronize();
}

void xelis_hash_hip_batch(byte* input, workerData_xelis_hip *worker, byte *hashResult) {

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

__global__ void xelis_initOptimizedWorker_hip(workerData_xelis_hip_optimized *worker, int batchSize) {
  worker->scratchPad = (uint64_t*)malloc(XELIS_HIP_MEMORY_SIZE*8*batchSize);
  memset(worker->scratchPad, 0, XELIS_HIP_MEMORY_SIZE*8*batchSize);
}

__global__ void xelis_initOptimizedWorker_hip2(workerData_xelis_hip_optimized *worker, int batchSize) {
  worker->input = (byte*)malloc(XELIS_HIP_BYTES_ARRAY_INPUT*batchSize);
  memset(worker->input, 0, XELIS_HIP_BYTES_ARRAY_INPUT*batchSize);
}

__global__ void xelis_initOptimizedWorker_hip3(workerData_xelis_hip_optimized *worker, int batchSize) {
  worker->int_input = (uint64_t**)malloc(sizeof(uint64_t*)*batchSize);
  worker->smallPad = (uint32_t**)malloc(sizeof(uint32_t*)*batchSize);

  worker->slots = (uint32_t*)malloc(XELIS_HIP_SLOT_LENGTH*4*batchSize);
  memset(worker->slots, 0, XELIS_HIP_SLOT_LENGTH*4*batchSize);
}

__global__ void xelis_initOptimizedWorker_hip4(workerData_xelis_hip_optimized *worker, int batchSize) {
  worker->indices = (byte*)malloc(XELIS_HIP_SLOT_LENGTH*batchSize);
  memset(worker->indices, 0, XELIS_HIP_SLOT_LENGTH*batchSize);
}

__global__ void xelis_interpretOptimizedWorker_hip(workerData_xelis_hip_optimized *worker, byte *input, int batchSize) {
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;

  worker->input[bid * XELIS_HIP_BYTES_ARRAY_INPUT + tid] = input[tid];
  __syncthreads();

  if (tid == 0) {
    worker->int_input[bid] = reinterpret_cast<uint64_t*>(&worker->input[bid*XELIS_HIP_BYTES_ARRAY_INPUT]);
    worker->smallPad[bid] = reinterpret_cast<uint32_t*>(&worker->scratchPad[bid*XELIS_HIP_MEMORY_SIZE]);
  }
  __syncthreads();
}

namespace tests_hip {
    char* testTemplate = "97dff4761917c2692df3be38e72ca7a59c3f55252e2245cc21564ef65fa8ea6f0000018f22fe78f80000000000064202f2a40463ccfcea839c4950a56ee38fa69c7ce2d4ba45d4b060cc63c297fb73b8a09c69661b1690b0a238d096a7ccb3cb204ce5dd604da9bb6c79c4ab00000000";
    using Hash = std::array<byte, XELIS_HIP_HASH_SIZE>;

    int obSize = 4096;

    bool test_input(const char* test_name, byte* input, size_t input_size, const Hash& expected_hash) {
        workerData_xelis_hip *worker_d;
        byte dummyBytes[sizeof(workerData_xelis_hip)] = {0};
        hipMalloc((void**)&worker_d, sizeof(workerData_xelis_hip));

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

    bool test_input_optimized(const char* test_name, byte* input, size_t input_size, const Hash& expected_hash) {
      workerData_xelis_hip_optimized *worker_d;
      byte dummyBytes[sizeof(workerData_xelis_hip_optimized)] = {0};
      hipMalloc((void**)&worker_d, sizeof(workerData_xelis_hip_optimized));

      byte hash_result[XELIS_HIP_HASH_SIZE*obSize];
      byte *hash_result_d;

      byte *input_d;
      
      hipMalloc((void**)&input_d, input_size);
      hipMemcpy(input_d, input, input_size, hipMemcpyHostToDevice);

      xelis_initOptimizedWorker_hip<<<1, 1>>>(worker_d, obSize);
      hipDeviceSynchronize();
      xelis_initOptimizedWorker_hip2<<<1, 1>>>(worker_d, obSize);
      hipDeviceSynchronize();
      xelis_initOptimizedWorker_hip3<<<1, 1>>>(worker_d, obSize);
      hipDeviceSynchronize();
      xelis_initOptimizedWorker_hip4<<<1, 1>>>(worker_d, obSize);
      hipDeviceSynchronize();
      
      xelis_interpretOptimizedWorker_hip<<<obSize, XELIS_HIP_BYTES_ARRAY_INPUT>>>(worker_d, input_d, obSize);
      hipDeviceSynchronize();

      printf("initialized\n");
      hipMalloc((void**)&hash_result_d, XELIS_HIP_HASH_SIZE*obSize);

      auto start = std::chrono::high_resolution_clock::now();
      xelis_hash_hip_optimized(worker_d, hash_result_d, obSize);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;

      std::cout << "Time took: " << elapsed.count() << " ms for " << obSize << " hashes" << std::endl;

      hipMemcpy(hash_result, hash_result_d, XELIS_HIP_HASH_SIZE*obSize, hipMemcpyDeviceToHost);

      if (std::memcmp(&hash_result[32*6], expected_hash.data(), XELIS_HIP_HASH_SIZE) != 0) {
          std::cout << "Test '" << test_name << "' failed!" << std::endl;
          std::cout << "HIP (optimized) || Expected hash: ";
          for (size_t i = 0; i < XELIS_HIP_HASH_SIZE; ++i) {
              printf("%02x ", expected_hash[i]);
          }
          std::cout << std::endl;
          std::cout << "HIP (optimized) || Actual hash:   ";
          for (size_t i = 0; i < XELIS_HIP_HASH_SIZE; ++i) {
              printf("%02x ", hash_result[i+32*6]);
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

      bool test_zero_input_optimized() {
        byte input[200] = {0};
        Hash expected_hash = {
            0x0e, 0xbb, 0xbd, 0x8a, 0x31, 0xed, 0xad, 0xfe, 0x09, 0x8f, 0x2d, 0x77, 0x0d, 0x84,
            0xb7, 0x19, 0x58, 0x86, 0x75, 0xab, 0x88, 0xa0, 0xa1, 0x70, 0x67, 0xd0, 0x0a, 0x8f,
            0x36, 0x18, 0x22, 0x65
        };

        return test_input_optimized("test_zero_input", input, 200, expected_hash);
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

    bool test_xelis_input_optimized() {
        byte input[XELIS_HIP_BYTES_ARRAY_INPUT] = {0};

        const char* custom = "xelis-hashing-algorithm";
        std::memcpy(input, custom, std::strlen(custom));

        Hash expected_hash = {
            106, 106, 173, 8, 207, 59, 118, 108, 176, 196, 9, 124, 250, 195, 3,
            61, 30, 146, 238, 182, 88, 83, 115, 81, 139, 56, 3, 28, 176, 86, 68, 21
        };
        return test_input_optimized("test_xelis_input", input, sizeof(input), expected_hash);
    }
}

void xelis_runTests_hip() {
  bool all_tests_passed = true;
  bool all_tests_passed_optimized = true;

  // all_tests_passed &= tests_hip::test_zero_input();
  // all_tests_passed &= tests_hip::test_xelis_input();

  // all_tests_passed_optimized &= tests_hip::test_zero_input_optimized();
  all_tests_passed_optimized &= tests_hip::test_xelis_input_optimized();

  // if (all_tests_passed) {
  //     std::cout << "HIP || XELIS-HASH: All tests passed!" << std::endl;
  // } else {
  //     std::cout << "HIP || XELIS-HASH: Some tests failed!" << std::endl;
  // }

  if (all_tests_passed_optimized) {
      std::cout << "HIP (optimized) || XELIS-HASH: All tests passed!" << std::endl;
  } else {
      std::cout << "HIP (optimized) || XELIS-HASH: Some tests failed!" << std::endl;
  }
}