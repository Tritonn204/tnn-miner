#pragma once

#include <inttypes.h>
#include <algorithm>
#include <numeric>

typedef unsigned char byte;

#define XELIS_USE_AVX512 4
#define XELIS_USE_AVX2 3
#define XELIS_USE_SSE2 2
#define XELIS_USE_SCALAR 1

const uint16_t XELIS_MEMORY_SIZE = 32768;
const uint16_t XELIS_SCRATCHPAD_ITERS = 5000;
const byte XELIS_ITERS = 1;
const uint16_t XELIS_BUFFER_SIZE = 42;
const uint16_t XELIS_SLOT_LENGTH = 256;
const int XELIS_TEMPLATE_SIZE = 112;

const byte XELIS_KECCAK_WORDS = 25;
const byte XELIS_BYTES_ARRAY_INPUT = XELIS_KECCAK_WORDS * 8;
const byte XELIS_HASH_SIZE = 32;
const uint16_t XELIS_STAGE_1_MAX = XELIS_MEMORY_SIZE / XELIS_KECCAK_WORDS;

typedef struct workerData_xelis {
  alignas(64) uint64_t scratchPad[XELIS_MEMORY_SIZE] = {0};
  uint64_t *int_input;
  uint32_t *smallPad;
  alignas(64) uint32_t slots[XELIS_SLOT_LENGTH] = {0};
  alignas(64) byte indices[XELIS_SLOT_LENGTH] = {0};
} workerData_xelis;

typedef struct xelis_BlockMiner {
    uint8_t header_work_hash[32];
    uint64_t timestamp;
    uint64_t nonce;
    uint8_t miner[32];
    uint8_t extra_nonce[32];
    // Other fields and methods...
} xelis_BlockMiner;

void xelis_hash(byte* input, workerData_xelis &worker, byte *hashResult);
void xelis_benchmark_cpu_hash();
void xelis_runTests();

void mineXelis(int tid);
