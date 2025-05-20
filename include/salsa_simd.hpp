// salsa20_simd.hpp
#ifndef SALSA20_SIMD_HPP
#define SALSA20_SIMD_HPP

#include <cstdint>
#include <cstring>

#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace crypto {

class Salsa20 {
public:
    template<int ROUNDS>
    static void salsa20(uint32_t B[16]);
    
    template<int ROUNDS>
    static void salsa20_blocks(uint32_t* B, size_t blocks);
};

// Generic implementation (always available)
template<int ROUNDS>
#ifdef __x86_64__
__attribute__((target("default")))
#endif
void Salsa20::salsa20(uint32_t B[16]) {
    uint32_t x[16];
    memcpy(x, B, 64);
    
    for (int i = 0; i < ROUNDS; i += 2) {
        #define R(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
        // Column round
        x[ 4] ^= R(x[ 0]+x[12], 7);  x[ 8] ^= R(x[ 4]+x[ 0], 9);
        x[12] ^= R(x[ 8]+x[ 4],13);  x[ 0] ^= R(x[12]+x[ 8],18);
        x[ 9] ^= R(x[ 5]+x[ 1], 7);  x[13] ^= R(x[ 9]+x[ 5], 9);
        x[ 1] ^= R(x[13]+x[ 9],13);  x[ 5] ^= R(x[ 1]+x[13],18);
        x[14] ^= R(x[10]+x[ 6], 7);  x[ 2] ^= R(x[14]+x[10], 9);
        x[ 6] ^= R(x[ 2]+x[14],13);  x[10] ^= R(x[ 6]+x[ 2],18);
        x[ 3] ^= R(x[15]+x[11], 7);  x[ 7] ^= R(x[ 3]+x[15], 9);
        x[11] ^= R(x[ 7]+x[ 3],13);  x[15] ^= R(x[11]+x[ 7],18);
        
        // Row round
        x[ 1] ^= R(x[ 0]+x[ 3], 7);  x[ 2] ^= R(x[ 1]+x[ 0], 9);
        x[ 3] ^= R(x[ 2]+x[ 1],13);  x[ 0] ^= R(x[ 3]+x[ 2],18);
        x[ 6] ^= R(x[ 5]+x[ 4], 7);  x[ 7] ^= R(x[ 6]+x[ 5], 9);
        x[ 4] ^= R(x[ 7]+x[ 6],13);  x[ 5] ^= R(x[ 4]+x[ 7],18);
        x[11] ^= R(x[10]+x[ 9], 7);  x[ 8] ^= R(x[11]+x[10], 9);
        x[ 9] ^= R(x[ 8]+x[11],13);  x[10] ^= R(x[ 9]+x[ 8],18);
        x[12] ^= R(x[15]+x[14], 7);  x[13] ^= R(x[12]+x[15], 9);
        x[14] ^= R(x[13]+x[12],13);  x[15] ^= R(x[14]+x[13],18);
        #undef R
    }
    
    for (int i = 0; i < 16; i++)
        B[i] += x[i];
}

#ifdef __x86_64__
// Complete SSE2 implementation
template<int ROUNDS>
__attribute__((target("sse2")))
void Salsa20::salsa20(uint32_t B[16]) {
    __m128i X0, X1, X2, X3;
    __m128i T0, T1, T2, T3;
    
    // Load data with SIMD shuffle pattern (matching yespower's layout)
    // This shuffle pattern groups the data for efficient SIMD operations
    {
        __m128i B0 = _mm_loadu_si128((__m128i*)&B[0]);
        __m128i B1 = _mm_loadu_si128((__m128i*)&B[4]);
        __m128i B2 = _mm_loadu_si128((__m128i*)&B[8]);
        __m128i B3 = _mm_loadu_si128((__m128i*)&B[12]);
        
        // Extract individual elements and shuffle
        // Pattern: X0=[B0,B5,B10,B15], X1=[B4,B9,B14,B3], etc.
        X0 = _mm_set_epi32(B[15], B[10], B[5], B[0]);
        X1 = _mm_set_epi32(B[3], B[14], B[9], B[4]);
        X2 = _mm_set_epi32(B[7], B[2], B[13], B[8]);
        X3 = _mm_set_epi32(B[11], B[6], B[1], B[12]);
    }
    
    // Save initial values for final addition
    T0 = X0;
    T1 = X1;
    T2 = X2;
    T3 = X3;
    
    // ARX macro: Add-Rotate-XOR
    #define ARX(out, in1, in2, s) { \
        __m128i tmp = _mm_add_epi32(in1, in2); \
        out = _mm_xor_si128(out, _mm_slli_epi32(tmp, s)); \
        out = _mm_xor_si128(out, _mm_srli_epi32(tmp, 32 - s)); \
    }
    
    // Main loop - process ROUNDS/2 double-rounds
    for (int i = 0; i < ROUNDS; i += 2) {
        // Column round
        ARX(X1, X0, X3, 7)
        ARX(X2, X1, X0, 9)
        ARX(X3, X2, X1, 13)
        ARX(X0, X3, X2, 18)
        
        // Rearrange data for row round
        X1 = _mm_shuffle_epi32(X1, 0x93);  // 2,0,3,1
        X2 = _mm_shuffle_epi32(X2, 0x4E);  // 1,0,3,2
        X3 = _mm_shuffle_epi32(X3, 0x39);  // 0,3,2,1
        
        // Row round
        ARX(X3, X0, X1, 7)
        ARX(X2, X3, X0, 9)
        ARX(X1, X2, X3, 13)
        ARX(X0, X1, X2, 18)
        
        // Rearrange data back for next column round
        X1 = _mm_shuffle_epi32(X1, 0x39);  // 0,3,2,1
        X2 = _mm_shuffle_epi32(X2, 0x4E);  // 1,0,3,2
        X3 = _mm_shuffle_epi32(X3, 0x93);  // 2,0,3,1
    }
    #undef ARX
    
    // Add initial values
    X0 = _mm_add_epi32(X0, T0);
    X1 = _mm_add_epi32(X1, T1);
    X2 = _mm_add_epi32(X2, T2);
    X3 = _mm_add_epi32(X3, T3);
    
    // Unshuffle and store back
    // We need to reverse the initial shuffle pattern
    union {
        __m128i q;
        uint32_t w[4];
    } t0, t1, t2, t3;
    
    t0.q = X0;
    t1.q = X1;
    t2.q = X2;
    t3.q = X3;
    
    B[0]  = t0.w[0];
    B[5]  = t0.w[1];
    B[10] = t0.w[2];
    B[15] = t0.w[3];
    
    B[4]  = t1.w[0];
    B[9]  = t1.w[1];
    B[14] = t1.w[2];
    B[3]  = t1.w[3];
    
    B[8]  = t2.w[0];
    B[13] = t2.w[1];
    B[2]  = t2.w[2];
    B[7]  = t2.w[3];
    
    B[12] = t3.w[0];
    B[1]  = t3.w[1];
    B[6]  = t3.w[2];
    B[11] = t3.w[3];
}
#endif // __x86_64__

#ifdef __ARM_NEON
// ARM NEON implementation placeholder
template<int ROUNDS>
__attribute__((target("neon")))
void Salsa20::salsa20(uint32_t B[16]) {
    // TODO: NEON implementation
}
#endif

#ifdef __ARM_FEATURE_SVE
// ARM SVE implementation placeholder
template<int ROUNDS>
void Salsa20::salsa20(uint32_t B[16]) {
    // TODO: SVE implementation
}
#endif

// Template specializations
template void Salsa20::salsa20<2>(uint32_t B[16]);
template void Salsa20::salsa20<8>(uint32_t B[16]);
template void Salsa20::salsa20<20>(uint32_t B[16]);

} // namespace crypto

#endif // SALSA20_SIMD_HPP