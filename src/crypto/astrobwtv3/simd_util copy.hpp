#pragma once

#include <immintrin.h>
#include <emmintrin.h>

///////////////////////////////////////////////////////////////////////////////
// SIMD vector typedefs for uniform handling
///////////////////////////////////////////////////////////////////////////////
typedef __m128i simd128_t;
typedef __m256i simd256_t;
typedef __m512i simd512_t;

///////////////////////////////////////////////////////////////////////////////
// Generic mask generation - creates a mask with the specified number of bytes
///////////////////////////////////////////////////////////////////////////////
#if defined(__AVX512F__)
__attribute__((target("avx512f")))
inline simd512_t genMask(int bytes) {
    return _mm512_maskz_set1_epi8((1ULL << (bytes & 0x3F)) - 1, 0xFF);
}
#endif

__attribute__((target("avx2")))
inline simd256_t genMask(int bytes) {
    static const uint64_t masks[33] = {
        0x0000000000000000ULL, 0x00000000000000FFULL, 0x000000000000FFFFULL, 0x0000000000FFFFFFULL,
        0x00000000FFFFFFFFULL, 0x000000FFFFFFFFFFULL, 0x0000FFFFFFFFFFFFULL, 0x00FFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0x00000000000000FFULL, 0x000000000000FFFFULL, 0x0000000000FFFFFFULL,
        0x00000000FFFFFFFFULL, 0x000000FFFFFFFFFFULL, 0x0000FFFFFFFFFFFFULL, 0x00FFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0x00000000000000FFULL, 0x000000000000FFFFULL, 0x0000000000FFFFFFULL,
        0x00000000FFFFFFFFULL, 0x000000FFFFFFFFFFULL, 0x0000FFFFFFFFFFFFULL, 0x00FFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    
    bytes = (bytes < 0) ? 0 : (bytes > 32) ? 32 : bytes;
    return _mm256_setr_epi64x(masks[bytes], masks[bytes > 16 ? bytes - 16 : 0], 0, 0);
}

///////////////////////////////////////////////////////////////////////////////
// 8-bit integer multiplication
///////////////////////////////////////////////////////////////////////////////

// 256-bit versions
__attribute__((target("avx512bw")))
inline simd512_t simd_mul_epi8(simd512_t a, simd512_t b) {
    simd512_t a_lo = _mm512_unpacklo_epi8(a, _mm512_setzero_si512());
    simd512_t b_lo = _mm512_unpacklo_epi8(b, _mm512_setzero_si512());
    simd512_t result_lo = _mm512_mullo_epi16(a_lo, b_lo);
    
    simd512_t a_hi = _mm512_unpackhi_epi8(a, _mm512_setzero_si512());
    simd512_t b_hi = _mm512_unpackhi_epi8(b, _mm512_setzero_si512());
    simd512_t result_hi = _mm512_mullo_epi16(a_hi, b_hi);
    
    return _mm512_packus_epi16(result_lo, result_hi);
}

__attribute__((target("avx2")))
inline simd256_t simd_mul_epi8(simd256_t a, simd256_t b) {
    simd256_t a_lo = _mm256_unpacklo_epi8(a, _mm256_setzero_si256());
    simd256_t b_lo = _mm256_unpacklo_epi8(b, _mm256_setzero_si256());
    simd256_t result_lo = _mm256_mullo_epi16(a_lo, b_lo);
    
    simd256_t a_hi = _mm256_unpackhi_epi8(a, _mm256_setzero_si256());
    simd256_t b_hi = _mm256_unpackhi_epi8(b, _mm256_setzero_si256());
    simd256_t result_hi = _mm256_mullo_epi16(a_hi, b_hi);
    
    return _mm256_packus_epi16(result_lo, result_hi);
}

// 128-bit versions
__attribute__((target("sse2"))) 
inline simd128_t simd_mul_epi8(simd128_t a, simd128_t b) {
    simd128_t dst_even = _mm_mullo_epi16(a, b);
    simd128_t dst_odd = _mm_mullo_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8));
    return _mm_or_si128(_mm_slli_epi16(dst_odd, 8), _mm_srli_epi16(_mm_slli_epi16(dst_even, 8), 8));
}

///////////////////////////////////////////////////////////////////////////////
// Variable 8-bit shift left
///////////////////////////////////////////////////////////////////////////////
__attribute__((target("avx2")))
inline simd256_t simd_sllv_epi8(simd256_t a, simd256_t count) {
    simd256_t mask_hi = _mm256_set1_epi32(0xFF00FF00);
    simd256_t multiplier_lut = _mm256_set_epi8(
        0,0,0,0, 0,0,0,0, 0x80,0x40,0x20,0x10, 0x08,0x04,0x02,0x01,
        0,0,0,0, 0,0,0,0, 0x80,0x40,0x20,0x10, 0x08,0x04,0x02,0x01
    );

    simd256_t count_sat = _mm256_min_epu8(count, _mm256_set1_epi8(8));
    simd256_t multiplier = _mm256_shuffle_epi8(multiplier_lut, count_sat);
    
    simd256_t x_lo = _mm256_mullo_epi16(a, multiplier);
    simd256_t multiplier_hi = _mm256_srli_epi16(multiplier, 8);
    simd256_t a_hi = _mm256_and_si256(a, mask_hi);
    simd256_t x_hi = _mm256_mullo_epi16(a_hi, multiplier_hi);
    
    return _mm256_blendv_epi8(x_lo, x_hi, mask_hi);
}

///////////////////////////////////////////////////////////////////////////////
// Variable 8-bit shift right
///////////////////////////////////////////////////////////////////////////////
__attribute__((target("avx2")))
inline simd256_t simd_srlv_epi8(simd256_t a, simd256_t count) {
    simd256_t mask_hi = _mm256_set1_epi32(0xFF00FF00);
    simd256_t multiplier_lut = _mm256_set_epi8(
        0,0,0,0, 0,0,0,0, 0x01,0x02,0x04,0x08, 0x10,0x20,0x40,0x80,
        0,0,0,0, 0,0,0,0, 0x01,0x02,0x04,0x08, 0x10,0x20,0x40,0x80
    );

    simd256_t count_sat = _mm256_min_epu8(count, _mm256_set1_epi8(8));
    simd256_t multiplier = _mm256_shuffle_epi8(multiplier_lut, count_sat);
    simd256_t a_lo = _mm256_andnot_si256(mask_hi, a);
    simd256_t multiplier_lo = _mm256_andnot_si256(mask_hi, multiplier);
    simd256_t x_lo = _mm256_mullo_epi16(a_lo, multiplier_lo);
    x_lo = _mm256_srli_epi16(x_lo, 7);

    simd256_t multiplier_hi = _mm256_and_si256(mask_hi, multiplier);
    simd256_t x_hi = _mm256_mulhi_epu16(a, multiplier_hi);
    x_hi = _mm256_slli_epi16(x_hi, 1);
    
    return _mm256_blendv_epi8(x_lo, x_hi, mask_hi);
}

///////////////////////////////////////////////////////////////////////////////
// Variable 8-bit rotation
///////////////////////////////////////////////////////////////////////////////
__attribute__((target("avx512vbmi2")))
inline simd512_t simd_rolv_epi8(simd512_t x, simd512_t y) {
    simd512_t y_mod = _mm512_and_si512(y, _mm512_set1_epi8(7));
    simd512_t left_shift = _mm512_multishift_epi64_epi8(x, y_mod);
    simd512_t right_shift_counts = _mm512_sub_epi8(_mm512_set1_epi8(8), y_mod);
    simd512_t right_shift = _mm512_multishift_epi64_epi8(x, right_shift_counts);
    return _mm512_or_si512(left_shift, right_shift);
}

__attribute__((target("avx2")))
inline simd256_t simd_rolv_epi8(simd256_t x, simd256_t y) {
    simd256_t y_mod = _mm256_and_si256(y, _mm256_set1_epi8(7));
    simd256_t left_shift = simd_sllv_epi8(x, y_mod);
    simd256_t right_shift_counts = _mm256_sub_epi8(_mm256_set1_epi8(8), y_mod);
    simd256_t right_shift = simd_srlv_epi8(x, right_shift_counts);
    return _mm256_or_si256(left_shift, right_shift);
}

///////////////////////////////////////////////////////////////////////////////
// Fixed 8-bit rotation
///////////////////////////////////////////////////////////////////////////////
__attribute__((target("avx2")))
inline simd256_t simd_rol_epi8(simd256_t x, int r) {
    r &= 7; // Ensure r is in range [0,7]
    simd256_t mask1 = _mm256_set1_epi16(0x00FF);
    simd256_t mask2 = _mm256_set1_epi16(0xFF00);
    
    simd256_t a = _mm256_and_si256(x, mask1);
    simd256_t b = _mm256_and_si256(x, mask2);

    simd256_t shiftedA = _mm256_slli_epi16(a, r);
    simd256_t wrappedA = _mm256_srli_epi16(a, 8 - r);
    simd256_t rotatedA = _mm256_and_si256(_mm256_or_si256(shiftedA, wrappedA), mask1);

    simd256_t shiftedB = _mm256_slli_epi16(b, r);
    simd256_t wrappedB = _mm256_srli_epi16(b, 8 - r);
    simd256_t rotatedB = _mm256_and_si256(_mm256_or_si256(shiftedB, wrappedB), mask2);

    return _mm256_or_si256(rotatedA, rotatedB);
}

///////////////////////////////////////////////////////////////////////////////
// Population count
///////////////////////////////////////////////////////////////////////////////

// 128-bit parallel population count
__attribute__((target("ssse3")))
inline simd128_t simd_popcnt_epi8(simd128_t x) {
    const simd128_t mask4 = _mm_set1_epi8(0x0F);
    const simd128_t lookup = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    
    simd128_t low = _mm_and_si128(mask4, x);
    simd128_t high = _mm_and_si128(mask4, _mm_srli_epi16(x, 4));
    return _mm_add_epi8(_mm_shuffle_epi8(lookup, low), _mm_shuffle_epi8(lookup, high));
}

// 256-bit parallel population count
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512VL__)
__attribute__((target("avx512vpopcntdq,avx512vl")))
inline simd256_t simd_popcnt_epi8(simd256_t x) {
    return _mm256_popcnt_epi8(x); // Directly use AVX-512VPOPCNTDQ+VL if available
}
#endif

__attribute__((target("avx2")))
inline simd256_t simd_popcnt_epi8(simd256_t x) {
    simd128_t hi = _mm256_extractf128_si256(x, 1);
    simd128_t lo = _mm256_castsi256_si128(x);

    hi = simd_popcnt_epi8(hi);
    lo = simd_popcnt_epi8(lo);

    return _mm256_set_m128i(hi, lo);
}

///////////////////////////////////////////////////////////////////////////////
// Bit reversal
///////////////////////////////////////////////////////////////////////////////
__attribute__((target("avx2")))
inline simd256_t simd_reverse_epi8(simd256_t x) {
    const simd256_t mask_0f = _mm256_set1_epi8(0x0F);
    const simd256_t mask_33 = _mm256_set1_epi8(0x33);
    const simd256_t mask_55 = _mm256_set1_epi8(0x55);

    // Step 1: Swap nibbles
    simd256_t temp = _mm256_and_si256(x, mask_0f);
    temp = _mm256_slli_epi16(temp, 4);
    x = _mm256_and_si256(x, _mm256_set1_epi8(0xF0));
    x = _mm256_srli_epi16(x, 4);
    x = _mm256_or_si256(x, temp);

    // Step 2: Swap pairs of bits
    temp = _mm256_and_si256(x, mask_33);
    temp = _mm256_slli_epi16(temp, 2);
    x = _mm256_and_si256(x, _mm256_set1_epi8(0xCC));
    x = _mm256_srli_epi16(x, 2);
    x = _mm256_or_si256(x, temp);

    // Step 3: Swap individual bits
    temp = _mm256_and_si256(x, mask_55);
    temp = _mm256_slli_epi16(temp, 1);
    x = _mm256_and_si256(x, _mm256_set1_epi8(0xAA));
    x = _mm256_srli_epi16(x, 1);
    x = _mm256_or_si256(x, temp);

    return x;
}