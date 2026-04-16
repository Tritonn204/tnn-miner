#pragma once
// ============================================================================
// ethash_dag_gen_simd.hpp — SIMD-accelerated DAG item generation
//
// Reimplements ethash::calculate_dataset_item_2048 with SIMD FNV-1 inner loop.
// The hot path is fnv1(hash512, hash512): 16 uint32 multiply+xor, called
// 512 times per sub-item × 4 sub-items = 2048 SIMD fnv1 calls per DAG entry.
//
// Tiers: AVX512 (1 op), AVX2 (2 ops), SSE4.1 (4 ops), scalar fallback.
// Runtime dispatch via function pointer set once at startup.
// ============================================================================

#include <cstdint>
#include <cstring>
#include <ethash/ethash.hpp>
#include <ethash/ethash-internal.hpp>
#include <ethash/keccak.h>

#ifdef __x86_64__
#include <immintrin.h>
#include <cpuid.h>
#endif

namespace ethash_dag_simd {

static constexpr uint32_t FNV_PRIME = 0x01000193u;
static constexpr int NUM_PARENTS = 512;
static constexpr int WORDS_PER_512 = 16;

// ---- Tier: scalar fallback ----

__attribute__((target("default")))
inline void fnv1_hash512(uint32_t* __restrict__ mix, const uint32_t* __restrict__ parent) {
    for (int i = 0; i < WORDS_PER_512; ++i)
        mix[i] = (mix[i] * FNV_PRIME) ^ parent[i];
}

#ifdef __x86_64__

// ---- Tier: SSE4.1 (mullo_epi32 requires SSE4.1) ----

__attribute__((target("sse4.1")))
inline void fnv1_hash512(uint32_t* __restrict__ mix, const uint32_t* __restrict__ parent) {
    __m128i prime = _mm_set1_epi32(FNV_PRIME);
    for (int i = 0; i < WORDS_PER_512; i += 4) {
        __m128i m = _mm_loadu_si128((const __m128i*)(mix + i));
        __m128i p = _mm_loadu_si128((const __m128i*)(parent + i));
        _mm_storeu_si128((__m128i*)(mix + i),
            _mm_xor_si128(_mm_mullo_epi32(m, prime), p));
    }
}

// ---- Tier: AVX2 ----

__attribute__((target("avx2")))
inline void fnv1_hash512(uint32_t* __restrict__ mix, const uint32_t* __restrict__ parent) {
    __m256i prime = _mm256_set1_epi32(FNV_PRIME);
    for (int i = 0; i < WORDS_PER_512; i += 8) {
        __m256i m = _mm256_loadu_si256((const __m256i*)(mix + i));
        __m256i p = _mm256_loadu_si256((const __m256i*)(parent + i));
        _mm256_storeu_si256((__m256i*)(mix + i),
            _mm256_xor_si256(_mm256_mullo_epi32(m, prime), p));
    }
}

// ---- Tier: AVX512 ----

__attribute__((target("avx512f")))
inline void fnv1_hash512(uint32_t* __restrict__ mix, const uint32_t* __restrict__ parent) {
    __m512i prime = _mm512_set1_epi32(FNV_PRIME);
    __m512i m = _mm512_loadu_si512(mix);
    __m512i p = _mm512_loadu_si512(parent);
    _mm512_storeu_si512(mix, _mm512_xor_si512(_mm512_mullo_epi32(m, prime), p));
}

#endif // __x86_64__

// ============================================================================
// Core DAG item generation — same algorithm as ethash::calculate_dataset_item_2048
// but with SIMD fnv1 inner loop.
// ============================================================================

// Single sub-item (512-bit) generation
inline void generate_sub_item(
    const ethash_hash512* cache, int64_t num_cache_items,
    int64_t index, ethash_hash512& out)
{
    uint32_t seed = (uint32_t)index;
    ethash_hash512 mix = cache[index % num_cache_items];
    mix.word32s[0] ^= seed; // x86 is LE, no conversion needed

    mix = ethash_keccak512_64(mix.bytes);

    for (uint32_t j = 0; j < NUM_PARENTS; ++j) {
        uint32_t t = (seed ^ j) * FNV_PRIME ^ mix.word32s[j % WORDS_PER_512];
        int64_t parent_index = t % num_cache_items;
        fnv1_hash512(mix.word32s, cache[parent_index].word32s);
    }

    out = ethash_keccak512_64(mix.bytes);
}

// Generate a single 2048-bit DAG item (4 interleaved sub-items)
inline void generate_item_2048(
    const ethash::epoch_context& ctx, uint32_t index, uint32_t* out)
{
    const auto* cache = ctx.light_cache;
    int64_t num_cache = ctx.light_cache_num_items;

    // 4 sub-items interleaved (same as reference calculate_dataset_item_2048)
    ethash_hash512 items[4];
    int64_t base = int64_t(index) * 4;

    // Init all 4
    uint32_t seeds[4];
    ethash_hash512 mixes[4];
    for (int s = 0; s < 4; ++s) {
        seeds[s] = (uint32_t)(base + s);
        mixes[s] = cache[(base + s) % num_cache];
        mixes[s].word32s[0] ^= seeds[s];
        mixes[s] = ethash_keccak512_64(mixes[s].bytes);
    }

    // Interleaved 512-round update (same order as reference for ILP)
    for (uint32_t j = 0; j < NUM_PARENTS; ++j) {
        for (int s = 0; s < 4; ++s) {
            uint32_t t = (seeds[s] ^ j) * FNV_PRIME ^ mixes[s].word32s[j % WORDS_PER_512];
            int64_t parent_index = t % num_cache;
            fnv1_hash512(mixes[s].word32s, cache[parent_index].word32s);
        }
    }

    // Final keccak + write to output
    for (int s = 0; s < 4; ++s) {
        ethash_hash512 final = ethash_keccak512_64(mixes[s].bytes);
        std::memcpy(out + s * 16, final.word32s, 64);
    }
}

// ============================================================================
// Batch DAG generation — drop-in replacement for the multithreaded loop
// ============================================================================
inline void generate_dag_range(
    const ethash::epoch_context& ctx, uint32_t start, uint32_t end, uint32_t* out)
{
    for (uint32_t i = start; i < end; ++i) {
        generate_item_2048(ctx, i, out + (size_t)i * 64);
    }
}

} // namespace ethash_dag_simd
