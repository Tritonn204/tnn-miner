// pwxform.c
#include "pwxform-simd.h"
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <stddef.h>
#include <stdint.h>

// Scalar implementation
#if defined(__x86_64__)
__attribute__((target("default")))
#endif
void pwxform_simd(uint8_t* block, uint8_t* S0, uint8_t* S1, uint64_t Smask2) {
    uint64_t* x = (uint64_t*)block;
    for (int i = 0; i < 4; i++) {
        uint64_t x0 = x[i*2];
        uint64_t x1 = x[i*2+1];
        
        uint64_t mask = x0 & Smask2;
        uint64_t* p0 = (uint64_t*)(S0 + (uint32_t)mask);
        uint64_t* p1 = (uint64_t*)(S1 + (mask >> 32));
        
        x[i*2] = ((x0 >> 32) * (uint32_t)x0 + p0[0]) ^ p1[0];
        x[i*2+1] = ((x1 >> 32) * (uint32_t)x1 + p0[1]) ^ p1[1];
    }
}

#if defined(__x86_64__)
__attribute__((target("sse2")))
void pwxform_simd(uint8_t* block, uint8_t* S0, uint8_t* S1, uint64_t Smask2) {
    __m128i* X = (__m128i*)block;
    for (int i = 0; i < 4; i++) {
        __m128i x = X[i];
        uint64_t idx = _mm_cvtsi128_si64(x) & Smask2;
        
        __m128i s0 = _mm_loadu_si128((__m128i*)(S0 + (uint32_t)idx));
        __m128i s1 = _mm_loadu_si128((__m128i*)(S1 + (idx >> 32)));
        
        x = _mm_mul_epu32(_mm_srli_si128(x, 4), x);
        x = _mm_add_epi64(x, s0);
        x = _mm_xor_si128(x, s1);
        
        X[i] = x;
    }
}

__attribute__((target("avx2")))
void pwxform_simd(uint8_t* block, uint8_t* S0, uint8_t* S1, uint64_t Smask2) {
    __m256i* X = (__m256i*)block;
    for (int i = 0; i < 2; i++) {
        __m256i x = X[i];
        uint64_t idx_lo = _mm256_extract_epi64(x, 0) & Smask2;
        uint64_t idx_hi = _mm256_extract_epi64(x, 2) & Smask2;
        
        __m256i s0 = _mm256_set_m128i(
            _mm_loadu_si128((__m128i*)(S0 + (uint32_t)idx_hi)),
            _mm_loadu_si128((__m128i*)(S0 + (uint32_t)idx_lo))
        );
        
        __m256i s1 = _mm256_set_m128i(
            _mm_loadu_si128((__m128i*)(S1 + (idx_hi >> 32))),
            _mm_loadu_si128((__m128i*)(S1 + (idx_lo >> 32)))
        );
        
        x = _mm256_mul_epu32(_mm256_srli_si256(x, 4), x);
        x = _mm256_add_epi64(x, s0);
        x = _mm256_xor_si256(x, s1);
        
        X[i] = x;
    }
}

// __attribute__((target("avx512f,avx512vl")))
// void pwxform_simd(uint8_t* block, uint8_t* S0, uint8_t* S1, uint64_t Smask2) {
//     // Similar to AVX2 but with AVX512 instructions
//     // Could process all 64 bytes at once with AVX512
// }

// Pass 2 versions would look similar but include the S2 rotation and w handling:
__attribute__((target("avx2")))
void pwxform_simd_pass2(uint8_t* block, uint8_t* S0, uint8_t* S1, uint8_t* S2,
                       size_t* w, uint64_t Smask2) {
    pwxform_simd(block, S0, S1, Smask2);
    
    // Handle S-box rotation
    uint8_t* Stmp = S2;
    S2 = S1;
    S1 = S0;
    S0 = Stmp;
    
    // Update w
    *w &= Smask2;
}
#endif