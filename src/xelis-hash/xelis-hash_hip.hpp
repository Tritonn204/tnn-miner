#pragma once

#include <inttypes.h>
#include <algorithm>
#include <numeric>
#include <hip/hip_runtime.h>

typedef unsigned char byte;

const uint16_t XELIS_HIP_MEMORY_SIZE = 32768;
const uint16_t XELIS_HIP_SCRATCHPAD_ITERS = 5000;
const byte XELIS_HIP_ITERS = 1;
const uint16_t XELIS_HIP_BUFFER_SIZE = 42;
const uint16_t XELIS_HIP_SLOT_LENGTH = 256;
const int XELIS_HIP_TEMPLATE_SIZE = 112;

const byte XELIS_HIP_KECCAK_WORDS = 25;
const byte XELIS_HIP_BYTES_ARRAY_INPUT = XELIS_HIP_KECCAK_WORDS * 8;
const byte XELIS_HIP_HASH_SIZE = 32;
const uint16_t XELIS_HIP_STAGE_1_MAX = XELIS_HIP_MEMORY_SIZE / XELIS_HIP_KECCAK_WORDS;


typedef struct workerData_xelis_hip {
  uint64_t scratchPad[XELIS_HIP_MEMORY_SIZE] = {0};
  uint64_t *int_input;
  uint32_t *smallPad;
  uint32_t slots[XELIS_HIP_SLOT_LENGTH] = {0};
  byte indices[XELIS_HIP_SLOT_LENGTH] = {0};
} workerData_xelis_hip;

typedef struct workerData_xelis_hip_optimized {
  byte *input;
  uint64_t *scratchPad;
  uint64_t **int_input;
  uint32_t **smallPad;
  uint32_t *slots;
  byte *indices;
} workerData_xelis_hip_optimized;

typedef struct xelis_BlockMiner_hip {
    uint8_t header_work_hash[32];
    uint64_t timestamp;
    uint64_t nonce;
    uint8_t miner[32];
    uint8_t extra_nonce[32];
    // Other fields and methods...
} xelis_BlockMiner_hip;
void xelis_hash_hip(byte* input, workerData_xelis_hip &worker, byte *hashResult);
void xelis_benchmark_gpu_hash_hip();
void xelis_runTests_hip();