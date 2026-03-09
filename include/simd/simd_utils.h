#pragma once

enum class StorePolicy { TEMPORAL, NON_TEMPORAL };

#ifdef __x86_64__
#include <immintrin.h>
#include <emmintrin.h>

template<StorePolicy POLICY>
__attribute__((target("ssse3")))
static inline void simd_store_128(__m128i* dst, __m128i val) {
    if constexpr (POLICY == StorePolicy::NON_TEMPORAL) {
        _mm_stream_si128(dst, val);
    } else {
        _mm_storeu_si128(dst, val);
    }
}

template<StorePolicy POLICY>
__attribute__((target("avx2")))
static inline void simd_store_256(__m256i* dst, __m256i val) {
    if constexpr (POLICY == StorePolicy::NON_TEMPORAL) {
        _mm256_stream_si256(dst, val);
    } else {
        _mm256_storeu_si256(dst, val);
    }
}

template<StorePolicy POLICY>
static inline void simd_store_64(void* dst, int64_t val) {
    if constexpr (POLICY == StorePolicy::NON_TEMPORAL) {
        _mm_stream_si64((long long*)dst, val);
    } else {
        *(int64_t*)dst = val;
    }
}

template<StorePolicy POLICY>
static inline void simd_fence() {
    if constexpr (POLICY == StorePolicy::NON_TEMPORAL) {
        _mm_sfence();
    }
}

#endif // __x86_64__