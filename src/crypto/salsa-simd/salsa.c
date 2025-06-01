#include "salsa-simd.h"
#include <memory.h>
#include <stdbool.h>

#if defined(__x86_64__)
#include <immintrin.h>
#include <cpuid.h>
#endif

static bool cpu_has_avx512 = false;
static bool cpu_has_avx2 = false;
static bool cpu_has_sse2 = false;
static bool cpu_checked = false;

static void check_cpu_features(void) {
    if (cpu_checked) return;
    
#if defined(__x86_64__)
    unsigned int eax, ebx, ecx, edx;
    
    // Check for SSE2
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        cpu_has_sse2 = (edx & (1 << 26)) != 0;
    }
    
    // Check for AVX2
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        cpu_has_avx2 = (ebx & (1 << 5)) != 0;
        cpu_has_avx512 = (ebx & (1 << 16)) != 0; // AVX512F
    }
#endif
    
    cpu_checked = true;
}

void Salsa20SetState(uint8_t* state, const uint8_t* input) {
    memcpy(state, input, 64);
}

void Salsa20GetState(const uint8_t* state, uint8_t* output) {
    memcpy(output, state, 64);
}

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// Scalar implementation
void Salsa20Transform_scalar(uint8_t* state, int rounds) {
    uint32_t* x = (uint32_t*)state;
    uint32_t y[16];
    
    // Copy state
    for (int i = 0; i < 16; i++) {
        y[i] = x[i];
    }
    
    // Perform rounds
    for (int i = 0; i < rounds; i += 2) {
        // Column rounds
        y[4] ^= ROTL32(y[0] + y[12], 7);
        y[8] ^= ROTL32(y[4] + y[0], 9);
        y[12] ^= ROTL32(y[8] + y[4], 13);
        y[0] ^= ROTL32(y[12] + y[8], 18);
        
        y[9] ^= ROTL32(y[5] + y[1], 7);
        y[13] ^= ROTL32(y[9] + y[5], 9);
        y[1] ^= ROTL32(y[13] + y[9], 13);
        y[5] ^= ROTL32(y[1] + y[13], 18);
        
        y[14] ^= ROTL32(y[10] + y[6], 7);
        y[2] ^= ROTL32(y[14] + y[10], 9);
        y[6] ^= ROTL32(y[2] + y[14], 13);
        y[10] ^= ROTL32(y[6] + y[2], 18);
        
        y[3] ^= ROTL32(y[15] + y[11], 7);
        y[7] ^= ROTL32(y[3] + y[15], 9);
        y[11] ^= ROTL32(y[7] + y[3], 13);
        y[15] ^= ROTL32(y[11] + y[7], 18);
        
        // Row rounds
        y[1] ^= ROTL32(y[0] + y[3], 7);
        y[2] ^= ROTL32(y[1] + y[0], 9);
        y[3] ^= ROTL32(y[2] + y[1], 13);
        y[0] ^= ROTL32(y[3] + y[2], 18);
        
        y[6] ^= ROTL32(y[5] + y[4], 7);
        y[7] ^= ROTL32(y[6] + y[5], 9);
        y[4] ^= ROTL32(y[7] + y[6], 13);
        y[5] ^= ROTL32(y[4] + y[7], 18);
        
        y[11] ^= ROTL32(y[10] + y[9], 7);
        y[8] ^= ROTL32(y[11] + y[10], 9);
        y[9] ^= ROTL32(y[8] + y[11], 13);
        y[10] ^= ROTL32(y[9] + y[8], 18);
        
        y[12] ^= ROTL32(y[15] + y[14], 7);
        y[13] ^= ROTL32(y[12] + y[15], 9);
        y[14] ^= ROTL32(y[13] + y[12], 13);
        y[15] ^= ROTL32(y[14] + y[13], 18);
    }
    
    // Add back to state
    for (int i = 0; i < 16; i++) {
        x[i] += y[i];
    }
}

#if defined(__x86_64__)
__attribute__((target("default")))
#endif
void Salsa20Transform(uint8_t* state, int rounds) {
    Salsa20Transform_scalar(state, rounds);
}

#if defined(__x86_64__)
__attribute__((target("sse2")))
void Salsa20Transform(uint8_t* state, int rounds) {
    __m128i x0 = _mm_loadu_si128((__m128i*)(state + 0));
    __m128i x1 = _mm_loadu_si128((__m128i*)(state + 16));
    __m128i x2 = _mm_loadu_si128((__m128i*)(state + 32));
    __m128i x3 = _mm_loadu_si128((__m128i*)(state + 48));
    
    __m128i z0 = x0, z1 = x1, z2 = x2, z3 = x3;
    
    for (int i = 0; i < rounds; i += 2) {
        // Column rounds
        __m128i tmp = _mm_add_epi32(x0, x3);
        x1 = _mm_xor_si128(x1, _mm_or_si128(_mm_slli_epi32(tmp, 7), _mm_srli_epi32(tmp, 25)));
        
        tmp = _mm_add_epi32(x1, x0);
        x2 = _mm_xor_si128(x2, _mm_or_si128(_mm_slli_epi32(tmp, 9), _mm_srli_epi32(tmp, 23)));
        
        tmp = _mm_add_epi32(x2, x1);
        x3 = _mm_xor_si128(x3, _mm_or_si128(_mm_slli_epi32(tmp, 13), _mm_srli_epi32(tmp, 19)));
        
        tmp = _mm_add_epi32(x3, x2);
        x0 = _mm_xor_si128(x0, _mm_or_si128(_mm_slli_epi32(tmp, 18), _mm_srli_epi32(tmp, 14)));
        
        // Rotate for row rounds
        x1 = _mm_shuffle_epi32(x1, 0x93);
        x2 = _mm_shuffle_epi32(x2, 0x4e);
        x3 = _mm_shuffle_epi32(x3, 0x39);
        
        // Row rounds
        tmp = _mm_add_epi32(x0, x1);
        x3 = _mm_xor_si128(x3, _mm_or_si128(_mm_slli_epi32(tmp, 7), _mm_srli_epi32(tmp, 25)));
        
        tmp = _mm_add_epi32(x3, x0);
        x2 = _mm_xor_si128(x2, _mm_or_si128(_mm_slli_epi32(tmp, 9), _mm_srli_epi32(tmp, 23)));
        
        tmp = _mm_add_epi32(x2, x3);
        x1 = _mm_xor_si128(x1, _mm_or_si128(_mm_slli_epi32(tmp, 13), _mm_srli_epi32(tmp, 19)));
        
        tmp = _mm_add_epi32(x1, x2);
        x0 = _mm_xor_si128(x0, _mm_or_si128(_mm_slli_epi32(tmp, 18), _mm_srli_epi32(tmp, 14)));
        
        // Rotate back
        x1 = _mm_shuffle_epi32(x1, 0x39);
        x2 = _mm_shuffle_epi32(x2, 0x4e);
        x3 = _mm_shuffle_epi32(x3, 0x93);
    }
    
    // Add back
    x0 = _mm_add_epi32(x0, z0);
    x1 = _mm_add_epi32(x1, z1);
    x2 = _mm_add_epi32(x2, z2);
    x3 = _mm_add_epi32(x3, z3);
    
    _mm_storeu_si128((__m128i*)(state + 0), x0);
    _mm_storeu_si128((__m128i*)(state + 16), x1);
    _mm_storeu_si128((__m128i*)(state + 32), x2);
    _mm_storeu_si128((__m128i*)(state + 48), x3);
}

#if defined(__x86_64__)
__attribute__((target("default")))
#endif
void Salsa20Transform4(uint8_t* state0, uint8_t* state1, uint8_t* state2, uint8_t* state3, int rounds) {
    Salsa20Transform_scalar(state0, rounds);
    Salsa20Transform_scalar(state1, rounds);
    Salsa20Transform_scalar(state2, rounds);
    Salsa20Transform_scalar(state3, rounds);
}

#if defined(__x86_64__)
__attribute__((target("sse2")))
void Salsa20Transform4(uint8_t* state0, uint8_t* state1, uint8_t* state2, uint8_t* state3, int rounds) {
    Salsa20Transform(state0, rounds);
    Salsa20Transform(state1, rounds);
    Salsa20Transform(state2, rounds);
    Salsa20Transform(state3, rounds);
}

__attribute__((target("avx2")))
void Salsa20Transform(uint8_t* state, int rounds) {
    // Process 2 blocks in parallel using AVX2
    __m256i x0 = _mm256_loadu_si256((__m256i*)(state + 0));
    __m256i x1 = _mm256_loadu_si256((__m256i*)(state + 32));
    
    __m256i z0 = x0, z1 = x1;
    
    for (int i = 0; i < rounds; i += 2) {
        // Column rounds
        __m256i tmp = _mm256_add_epi32(x0, x1);
        x1 = _mm256_xor_si256(x1, _mm256_or_si256(
            _mm256_slli_epi32(tmp, 7), 
            _mm256_srli_epi32(tmp, 25)
        ));
        
        tmp = _mm256_add_epi32(x1, x0);
        x0 = _mm256_xor_si256(x0, _mm256_or_si256(
            _mm256_slli_epi32(tmp, 9),
            _mm256_srli_epi32(tmp, 23)
        ));
        
        tmp = _mm256_add_epi32(x0, x1);
        x1 = _mm256_xor_si256(x1, _mm256_or_si256(
            _mm256_slli_epi32(tmp, 13),
            _mm256_srli_epi32(tmp, 19)
        ));
        
        tmp = _mm256_add_epi32(x1, x0);
        x0 = _mm256_xor_si256(x0, _mm256_or_si256(
            _mm256_slli_epi32(tmp, 18),
            _mm256_srli_epi32(tmp, 14)
        ));
        
        // Shuffle words for diagonal rounds
        x1 = _mm256_shuffle_epi32(x1, _MM_SHUFFLE(2, 1, 0, 3));
        
        // Diagonal rounds
        tmp = _mm256_add_epi32(x0, x1);
        x1 = _mm256_xor_si256(x1, _mm256_or_si256(
            _mm256_slli_epi32(tmp, 7),
            _mm256_srli_epi32(tmp, 25)
        ));
        
        tmp = _mm256_add_epi32(x1, x0);
        x0 = _mm256_xor_si256(x0, _mm256_or_si256(
            _mm256_slli_epi32(tmp, 9),
            _mm256_srli_epi32(tmp, 23)
        ));
        
        tmp = _mm256_add_epi32(x0, x1);
        x1 = _mm256_xor_si256(x1, _mm256_or_si256(
            _mm256_slli_epi32(tmp, 13),
            _mm256_srli_epi32(tmp, 19)
        ));
        
        tmp = _mm256_add_epi32(x1, x0);
        x0 = _mm256_xor_si256(x0, _mm256_or_si256(
            _mm256_slli_epi32(tmp, 18),
            _mm256_srli_epi32(tmp, 14)
        ));
        
        // Shuffle back
        x1 = _mm256_shuffle_epi32(x1, _MM_SHUFFLE(0, 3, 2, 1));
    }
    
    // Add back
    x0 = _mm256_add_epi32(x0, z0);
    x1 = _mm256_add_epi32(x1, z1);
    
    _mm256_storeu_si256((__m256i*)(state + 0), x0);
    _mm256_storeu_si256((__m256i*)(state + 32), x1);
}

__attribute__((target("avx2")))
void Salsa20Transform4(uint8_t* state0, uint8_t* state1, uint8_t* state2, uint8_t* state3, int rounds) {
    // Process first pair
    __m256i x0_a = _mm256_loadu_si256((__m256i*)(state0 + 0));
    __m256i x1_a = _mm256_loadu_si256((__m256i*)(state0 + 32));
    __m256i x0_b = _mm256_loadu_si256((__m256i*)(state1 + 0));
    __m256i x1_b = _mm256_loadu_si256((__m256i*)(state1 + 32));
    
    __m256i z0_a = x0_a, z1_a = x1_a;
    __m256i z0_b = x0_b, z1_b = x1_b;
    
    // Process second pair
    __m256i x0_c = _mm256_loadu_si256((__m256i*)(state2 + 0));
    __m256i x1_c = _mm256_loadu_si256((__m256i*)(state2 + 32));
    __m256i x0_d = _mm256_loadu_si256((__m256i*)(state3 + 0));
    __m256i x1_d = _mm256_loadu_si256((__m256i*)(state3 + 32));
    
    __m256i z0_c = x0_c, z1_c = x1_c;
    __m256i z0_d = x0_d, z1_d = x1_d;
    
    for (int i = 0; i < rounds; i += 2) {
        // Process both pairs in parallel using same operations
        #define SALSA_QUARTER_ROUND(x0, x1) { \
            __m256i tmp = _mm256_add_epi32(x0, x1); \
            x1 = _mm256_xor_si256(x1, _mm256_or_si256( \
                _mm256_slli_epi32(tmp, 7), \
                _mm256_srli_epi32(tmp, 25) \
            )); \
            \
            tmp = _mm256_add_epi32(x1, x0); \
            x0 = _mm256_xor_si256(x0, _mm256_or_si256( \
                _mm256_slli_epi32(tmp, 9), \
                _mm256_srli_epi32(tmp, 23) \
            )); \
            \
            tmp = _mm256_add_epi32(x0, x1); \
            x1 = _mm256_xor_si256(x1, _mm256_or_si256( \
                _mm256_slli_epi32(tmp, 13), \
                _mm256_srli_epi32(tmp, 19) \
            )); \
            \
            tmp = _mm256_add_epi32(x1, x0); \
            x0 = _mm256_xor_si256(x0, _mm256_or_si256( \
                _mm256_slli_epi32(tmp, 18), \
                _mm256_srli_epi32(tmp, 14) \
            )); \
        }
        
        // Column rounds for all pairs
        SALSA_QUARTER_ROUND(x0_a, x1_a);
        SALSA_QUARTER_ROUND(x0_b, x1_b);
        SALSA_QUARTER_ROUND(x0_c, x1_c);
        SALSA_QUARTER_ROUND(x0_d, x1_d);
        
        // Shuffle words for diagonal rounds
        x1_a = _mm256_shuffle_epi32(x1_a, _MM_SHUFFLE(2, 1, 0, 3));
        x1_b = _mm256_shuffle_epi32(x1_b, _MM_SHUFFLE(2, 1, 0, 3));
        x1_c = _mm256_shuffle_epi32(x1_c, _MM_SHUFFLE(2, 1, 0, 3));
        x1_d = _mm256_shuffle_epi32(x1_d, _MM_SHUFFLE(2, 1, 0, 3));
        
        // Diagonal rounds for all pairs
        SALSA_QUARTER_ROUND(x0_a, x1_a);
        SALSA_QUARTER_ROUND(x0_b, x1_b);
        SALSA_QUARTER_ROUND(x0_c, x1_c);
        SALSA_QUARTER_ROUND(x0_d, x1_d);
        
        // Shuffle back
        x1_a = _mm256_shuffle_epi32(x1_a, _MM_SHUFFLE(0, 3, 2, 1));
        x1_b = _mm256_shuffle_epi32(x1_b, _MM_SHUFFLE(0, 3, 2, 1));
        x1_c = _mm256_shuffle_epi32(x1_c, _MM_SHUFFLE(0, 3, 2, 1));
        x1_d = _mm256_shuffle_epi32(x1_d, _MM_SHUFFLE(0, 3, 2, 1));
        
        #undef SALSA_QUARTER_ROUND
    }
    
    // Add back and store results for first pair
    x0_a = _mm256_add_epi32(x0_a, z0_a);
    x1_a = _mm256_add_epi32(x1_a, z1_a);
    x0_b = _mm256_add_epi32(x0_b, z0_b);
    x1_b = _mm256_add_epi32(x1_b, z1_b);
    
    _mm256_storeu_si256((__m256i*)(state0 + 0), x0_a);
    _mm256_storeu_si256((__m256i*)(state0 + 32), x1_a);
    _mm256_storeu_si256((__m256i*)(state1 + 0), x0_b);
    _mm256_storeu_si256((__m256i*)(state1 + 32), x1_b);
    
    // Add back and store results for second pair
    x0_c = _mm256_add_epi32(x0_c, z0_c);
    x1_c = _mm256_add_epi32(x1_c, z1_c);
    x0_d = _mm256_add_epi32(x0_d, z0_d);
    x1_d = _mm256_add_epi32(x1_d, z1_d);
    
    _mm256_storeu_si256((__m256i*)(state2 + 0), x0_c);
    _mm256_storeu_si256((__m256i*)(state2 + 32), x1_c);
    _mm256_storeu_si256((__m256i*)(state3 + 0), x0_d);
    _mm256_storeu_si256((__m256i*)(state3 + 32), x1_d);
}
#endif

__attribute__((target("avx512f,avx512vl")))
void Salsa20Transform(uint8_t* state, int rounds) {
    __m128i x0 = _mm_loadu_si128((__m128i*)(state + 0));
    __m128i x1 = _mm_loadu_si128((__m128i*)(state + 16));
    __m128i x2 = _mm_loadu_si128((__m128i*)(state + 32));
    __m128i x3 = _mm_loadu_si128((__m128i*)(state + 48));
    
    __m128i z0 = x0, z1 = x1, z2 = x2, z3 = x3;
    
    for (int i = 0; i < rounds; i += 2) {
        // Column rounds - use AVX512VL rotate instructions
        __m128i tmp = _mm_add_epi32(x0, x3);
        x1 = _mm_xor_si128(x1, _mm_rol_epi32(tmp, 7));
        
        tmp = _mm_add_epi32(x1, x0);
        x2 = _mm_xor_si128(x2, _mm_rol_epi32(tmp, 9));
        
        tmp = _mm_add_epi32(x2, x1);
        x3 = _mm_xor_si128(x3, _mm_rol_epi32(tmp, 13));
        
        tmp = _mm_add_epi32(x3, x2);
        x0 = _mm_xor_si128(x0, _mm_rol_epi32(tmp, 18));
        
        // Rotate for row rounds
        x1 = _mm_shuffle_epi32(x1, 0x93);
        x2 = _mm_shuffle_epi32(x2, 0x4e);
        x3 = _mm_shuffle_epi32(x3, 0x39);
        
        // Row rounds
        tmp = _mm_add_epi32(x0, x1);
        x3 = _mm_xor_si128(x3, _mm_rol_epi32(tmp, 7));
        
        tmp = _mm_add_epi32(x3, x0);
        x2 = _mm_xor_si128(x2, _mm_rol_epi32(tmp, 9));
        
        tmp = _mm_add_epi32(x2, x3);
        x1 = _mm_xor_si128(x1, _mm_rol_epi32(tmp, 13));
        
        tmp = _mm_add_epi32(x1, x2);
        x0 = _mm_xor_si128(x0, _mm_rol_epi32(tmp, 18));
        
        // Rotate back
        x1 = _mm_shuffle_epi32(x1, 0x39);
        x2 = _mm_shuffle_epi32(x2, 0x4e);
        x3 = _mm_shuffle_epi32(x3, 0x93);
    }
    
    // Add back
    x0 = _mm_add_epi32(x0, z0);
    x1 = _mm_add_epi32(x1, z1);
    x2 = _mm_add_epi32(x2, z2);
    x3 = _mm_add_epi32(x3, z3);
    
    _mm_storeu_si128((__m128i*)(state + 0), x0);
    _mm_storeu_si128((__m128i*)(state + 16), x1);
    _mm_storeu_si128((__m128i*)(state + 32), x2);
    _mm_storeu_si128((__m128i*)(state + 48), x3);
}

// 4-way batch implementations for yespower
__attribute__((target("avx512f,avx512vl")))
void Salsa20Transform4(uint8_t* state0, uint8_t* state1, uint8_t* state2, uint8_t* state3, int rounds) {
    // Load 4 states into AVX512 registers
    __m512i x0 = _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(
        _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)(state0 + 0))),
        _mm_loadu_si128((__m128i*)(state1 + 0)), 1),
        _mm_loadu_si128((__m128i*)(state2 + 0)), 2),
        _mm_loadu_si128((__m128i*)(state3 + 0)), 3);
        
    __m512i x1 = _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(
        _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)(state0 + 16))),
        _mm_loadu_si128((__m128i*)(state1 + 16)), 1),
        _mm_loadu_si128((__m128i*)(state2 + 16)), 2),
        _mm_loadu_si128((__m128i*)(state3 + 16)), 3);
        
    __m512i x2 = _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(
        _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)(state0 + 32))),
        _mm_loadu_si128((__m128i*)(state1 + 32)), 1),
        _mm_loadu_si128((__m128i*)(state2 + 32)), 2),
        _mm_loadu_si128((__m128i*)(state3 + 32)), 3);
        
    __m512i x3 = _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(
        _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)(state0 + 48))),
        _mm_loadu_si128((__m128i*)(state1 + 48)), 1),
        _mm_loadu_si128((__m128i*)(state2 + 48)), 2),
        _mm_loadu_si128((__m128i*)(state3 + 48)), 3);
    
    __m512i z0 = x0, z1 = x1, z2 = x2, z3 = x3;
    
    for (int i = 0; i < rounds; i += 2) {
        // Process 4 Salsa20 blocks in parallel
        __m512i tmp = _mm512_add_epi32(x0, x3);
        x1 = _mm512_xor_si512(x1, _mm512_rol_epi32(tmp, 7));
        
        tmp = _mm512_add_epi32(x1, x0);
        x2 = _mm512_xor_si512(x2, _mm512_rol_epi32(tmp, 9));
        
        tmp = _mm512_add_epi32(x2, x1);
        x3 = _mm512_xor_si512(x3, _mm512_rol_epi32(tmp, 13));
        
        tmp = _mm512_add_epi32(x3, x2);
        x0 = _mm512_xor_si512(x0, _mm512_rol_epi32(tmp, 18));
        
        // Shuffle for row rounds
        x1 = _mm512_shuffle_epi32(x1, 0x93);
        x2 = _mm512_shuffle_epi32(x2, 0x4e);
        x3 = _mm512_shuffle_epi32(x3, 0x39);
        
        // Row rounds
        tmp = _mm512_add_epi32(x0, x1);
        x3 = _mm512_xor_si512(x3, _mm512_rol_epi32(tmp, 7));
        
        tmp = _mm512_add_epi32(x3, x0);
        x2 = _mm512_xor_si512(x2, _mm512_rol_epi32(tmp, 9));
        
        tmp = _mm512_add_epi32(x2, x3);
        x1 = _mm512_xor_si512(x1, _mm512_rol_epi32(tmp, 13));
        
        tmp = _mm512_add_epi32(x1, x2);
        x0 = _mm512_xor_si512(x0, _mm512_rol_epi32(tmp, 18));
        
        // Shuffle back
        x1 = _mm512_shuffle_epi32(x1, 0x39);
        x2 = _mm512_shuffle_epi32(x2, 0x4e);
        x3 = _mm512_shuffle_epi32(x3, 0x93);
    }
    
    // Add back
    x0 = _mm512_add_epi32(x0, z0);
    x1 = _mm512_add_epi32(x1, z1);
    x2 = _mm512_add_epi32(x2, z2);
    x3 = _mm512_add_epi32(x3, z3);
    
    // Store back to individual states
    _mm_storeu_si128((__m128i*)(state0 + 0), _mm512_extracti32x4_epi32(x0, 0));
    _mm_storeu_si128((__m128i*)(state1 + 0), _mm512_extracti32x4_epi32(x0, 1));
    _mm_storeu_si128((__m128i*)(state2 + 0), _mm512_extracti32x4_epi32(x0, 2));
    _mm_storeu_si128((__m128i*)(state3 + 0), _mm512_extracti32x4_epi32(x0, 3));
    
    _mm_storeu_si128((__m128i*)(state0 + 16), _mm512_extracti32x4_epi32(x1, 0));
    _mm_storeu_si128((__m128i*)(state1 + 16), _mm512_extracti32x4_epi32(x1, 1));
    _mm_storeu_si128((__m128i*)(state2 + 16), _mm512_extracti32x4_epi32(x1, 2));
    _mm_storeu_si128((__m128i*)(state3 + 16), _mm512_extracti32x4_epi32(x1, 3));
    
    _mm_storeu_si128((__m128i*)(state0 + 32), _mm512_extracti32x4_epi32(x2, 0));
    _mm_storeu_si128((__m128i*)(state1 + 32), _mm512_extracti32x4_epi32(x2, 1));
    _mm_storeu_si128((__m128i*)(state2 + 32), _mm512_extracti32x4_epi32(x2, 2));
    _mm_storeu_si128((__m128i*)(state3 + 32), _mm512_extracti32x4_epi32(x2, 3));
    
    _mm_storeu_si128((__m128i*)(state0 + 48), _mm512_extracti32x4_epi32(x3, 0));
    _mm_storeu_si128((__m128i*)(state1 + 48), _mm512_extracti32x4_epi32(x3, 1));
    _mm_storeu_si128((__m128i*)(state2 + 48), _mm512_extracti32x4_epi32(x3, 2));
    _mm_storeu_si128((__m128i*)(state3 + 48), _mm512_extracti32x4_epi32(x3, 3));
}
#endif

#if defined(__x86_64__)

__attribute__((target("avx512f,avx512vl")))
void salsa20_core(uint8_t* state, int rounds) {
    // For larger workloads, use 4-way processing
    if (rounds >= 8) {
        uint8_t states[4][64];
        memcpy(states[0], state, 64);
        memcpy(states[1], state, 64);
        memcpy(states[2], state, 64);
        memcpy(states[3], state, 64);
        
        Salsa20Transform4(states[0], states[1], states[2], states[3], rounds);
        memcpy(state, states[0], 64);
    } else {
        Salsa20Transform(state, rounds);
    }
}

__attribute__((target("avx2")))
void salsa20_core(uint8_t* state, int rounds) {
    // Similar logic for AVX2
    if (rounds >= 8) {
        uint8_t states[4][64];
        memcpy(states[0], state, 64);
        memcpy(states[1], state, 64);
        memcpy(states[2], state, 64);
        memcpy(states[3], state, 64);
        
        Salsa20Transform4(states[0], states[1], states[2], states[3], rounds);
        memcpy(state, states[0], 64);
    } else {
        Salsa20Transform(state, rounds);
    }
}

__attribute__((target("sse2")))
void salsa20_core(uint8_t* state, int rounds) {
    Salsa20Transform(state, rounds);
}

__attribute__((target("default")))
void salsa20_core(uint8_t* state, int rounds) {
    Salsa20Transform(state, rounds);
}

// 4-way versions with similar dispatch
__attribute__((target("avx512f,avx512vl")))
void salsa20_core_4way(uint8_t* state0, uint8_t* state1, uint8_t* state2, uint8_t* state3, int rounds) {
    Salsa20Transform4(state0, state1, state2, state3, rounds);
}

__attribute__((target("avx2")))
void salsa20_core_4way(uint8_t* state0, uint8_t* state1, uint8_t* state2, uint8_t* state3, int rounds) {
    Salsa20Transform4(state0, state1, state2, state3, rounds);
}

__attribute__((target("default")))
void salsa20_core_4way(uint8_t* state0, uint8_t* state1, uint8_t* state2, uint8_t* state3, int rounds) {
    Salsa20Transform(state0, rounds);
    Salsa20Transform(state1, rounds);
    Salsa20Transform(state2, rounds);
    Salsa20Transform(state3, rounds);
}

#endif