#pragma once

#include <inttypes.h>
#include <algorithm>
#include <numeric>

typedef unsigned char byte;

// ============================================================================
// Compilation Target Flags
// ============================================================================
#define XELIS_USE_AVX512 4
#define XELIS_USE_AVX2 3
#define XELIS_USE_SSE2 2
#define XELIS_USE_SCALAR 1

// ============================================================================
// Common Constants (shared across versions)
// ============================================================================
constexpr int XELIS_TEMPLATE_SIZE = 112;
constexpr byte XELIS_HASH_SIZE = 32;
constexpr uint16_t XELIS_SLOT_LENGTH = 256;

// ============================================================================
// V1 Constants
// ============================================================================
constexpr uint16_t XELIS_MEMORY_SIZE = 32768;
constexpr uint16_t XELIS_SCRATCHPAD_ITERS = 5000;
constexpr byte XELIS_ITERS = 1;
constexpr uint16_t XELIS_BUFFER_SIZE = 42;
constexpr byte XELIS_KECCAK_WORDS = 25;
constexpr byte XELIS_BYTES_ARRAY_INPUT = XELIS_KECCAK_WORDS * 8;
constexpr uint16_t XELIS_STAGE_1_MAX = XELIS_MEMORY_SIZE / XELIS_KECCAK_WORDS;

// ============================================================================
// V2 Constants
// ============================================================================
#define XELIS_BATCHSIZE_V2 1

constexpr size_t XELIS_MEMORY_SIZE_V2 = 429 * 128;              // 54,912 uint64s = 439,296 bytes
constexpr uint16_t XELIS_SCRATCHPAD_ITERS_V2 = 3;
constexpr size_t XELIS_BUFFER_SIZE_V2 = XELIS_MEMORY_SIZE_V2 / 2;  // 27,456

// ============================================================================
// V3 Constants
// ============================================================================
#define XELIS_BATCHSIZE_V3 1

constexpr size_t XELIS_MEMORY_SIZE_V3 = 531 * 128;              // 67,968 uint64s = 543,744 bytes
constexpr uint16_t XELIS_SCRATCHPAD_ITERS_V3 = 2;
constexpr size_t XELIS_BUFFER_SIZE_V3 = XELIS_MEMORY_SIZE_V3 / 2;  // 33,984

constexpr size_t XELIS_CHUNK_SIZE = 32;
constexpr size_t XELIS_NONCE_SIZE = 12;
constexpr size_t XELIS_CHUNKS = 4;

constexpr size_t XELIS_OUTPUT_SIZE_V3 = XELIS_MEMORY_SIZE_V3 * 8;
constexpr size_t XELIS_BYTES_PER_CHUNK_V3 = XELIS_OUTPUT_SIZE_V3 / XELIS_CHUNKS;

constexpr size_t XELIS_OUTPUT_SIZE_V2 = XELIS_MEMORY_SIZE_V2 * 8;
constexpr size_t XELIS_BYTES_PER_CHUNK_V2 = XELIS_OUTPUT_SIZE_V2 / XELIS_CHUNKS;

// ============================================================================
// V3 Magic Constants (used in murmurhash3, map_index, write pattern)
// ============================================================================
constexpr uint64_t XELIS_MURMUR_CONST1 = 0xff51afd7ed558ccdULL;
constexpr uint64_t XELIS_MURMUR_CONST2 = 0xc4ceb9fe1a85ec53ULL;
constexpr uint64_t XELIS_GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t XELIS_SCATTER_CONST = 0xd2b74407b1ce6e93ULL;

// pick_half() bit position check
constexpr uint64_t XELIS_PICK_HALF_BIT = (1ULL << 58);

// ============================================================================
// Worker Data Structures
// ============================================================================

typedef struct workerData_xelis {
    alignas(64) uint64_t scratchPad[XELIS_MEMORY_SIZE] = {0};
    uint64_t *int_input;
    uint32_t *smallPad;
    alignas(64) uint32_t slots[XELIS_SLOT_LENGTH] = {0};
    alignas(64) byte indices[XELIS_SLOT_LENGTH] = {0};
} workerData_xelis;

typedef struct workerData_xelis_v2 {
    alignas(64) uint64_t scratchPad[XELIS_MEMORY_SIZE_V2] = {0};
} workerData_xelis_v2;

typedef struct workerData_xelis_v3 {
    alignas(64) uint64_t scratchPad[XELIS_MEMORY_SIZE_V3] = {0};
} workerData_xelis_v3;

// ============================================================================
// Block Miner Structure
// ============================================================================

typedef struct xelis_BlockMiner {
    uint8_t header_work_hash[32];
    uint64_t timestamp;
    uint64_t nonce;
    uint8_t miner[32];
    uint8_t extra_nonce[32];
} xelis_BlockMiner;

// ============================================================================
// V1 Function Declarations
// ============================================================================
void xelis_hash(byte* input, workerData_xelis &worker, byte *hashResult);
void xelis_benchmark_cpu_hash();
int xelis_runTests();

// ============================================================================
// V2 Function Declarations
// ============================================================================
void xelis_hash_v2(byte *input, workerData_xelis_v2 &worker, byte *hashResult);
void xelis_benchmark_cpu_hash_v2();
int xelis_runTests_v2();

// ============================================================================
// V3 Function Declarations (NEW)
// ============================================================================
void xelis_hash_v3(byte *input, workerData_xelis_v3 &worker, byte *hashResult);
void xelis_hash_v3_nt(byte *input, workerData_xelis_v3 &worker, byte *hashResult);
void xelis_hash_v3_switch(byte *input, workerData_xelis_v3 &worker, byte *hashResult);
void xelis_hash_v3_switch_nt(byte *input, workerData_xelis_v3 &worker, byte *hashResult);
void xelis_hash_v3_merged(byte *input, workerData_xelis_v3 &worker, byte *hashResult);
void xelis_hash_v3_merged_nt(byte *input, workerData_xelis_v3 &worker, byte *hashResult);
void xelis_benchmark_cpu_hash_v3();
int xelis_runTests_v3();

// Startup tune: compares baseline vs hybrid (NT stores for HT siblings).
// Sets xelis_v3_use_hybrid and xelis_v3_ht_threshold globals.
void xelis_tune_v3(int num_threads);
extern bool xelis_v3_use_hybrid;
extern bool xelis_v3_use_switch;
extern bool xelis_v3_use_merged;
extern unsigned xelis_v3_ht_threshold;

// Override stage 1 SIMD dispatch level.
// Valid levels: "avx512", "avx2", "aes"; anything else maps to "none" (scalar fallback).
bool xelis_set_simd_override(const char *level);

// Exposed stages for debugging/validation
void xelis_stage1_v3(const uint8_t *input, uint64_t *scratch_pad, size_t input_len);
void xelis_stage3_v3(uint64_t *scratch_pad);
void xelis_blake3_v3(const uint8_t *scratch_pad, byte *hashResult);

// ============================================================================
// Mining Entry Point
// ============================================================================
void mineXelis(int tid);

enum class Stage3Variant {
    ORIGINAL,       // Current implementation
    OPTIMIZED_V1,   // Prefetch improvements + fast modpow
    OPTIMIZED_V2,   // V1 + non-temporal stores
};

// // Benchmark both implementations
// void xelis_benchmark_stage3_comparison();

// // Selectable stage 3 for testing
// void xelis_hash_v3_variant(byte *input, workerData_xelis_v3 &worker, 
//                            byte *hashResult, Stage3Variant variant);