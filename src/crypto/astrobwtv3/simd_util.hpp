#pragma once

#if defined(__x86_64__)
__attribute__((target("avx512f")))
inline const __m512i genMask_avx512(int bytes) {
    return _mm512_maskz_set1_epi8((1ULL << (bytes & 0x3F)) - 1, 0xFF);
}

__attribute__((target("avx2")))
inline const __m256i genMask_avx2(int bytes) {
  const __m256i sequence = _mm256_setr_epi8(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
  
  bytes = (bytes < 0) ? 0 : (bytes > 32) ? 32 : bytes;
  
  const __m256i count = _mm256_set1_epi8(bytes);
  return _mm256_cmpgt_epi8(count, sequence);
}

inline __m128i mullo_epi8(__m128i a, __m128i b)
{
    // unpack and multiply
    __m128i dst_even = _mm_mullo_epi16(a, b);
    __m128i dst_odd = _mm_mullo_epi16(_mm_srli_epi16(a, 8),_mm_srli_epi16(b, 8));
    // repack
#if defined(__AVX2__)
    // only faster if have access to VPBROADCASTW
    return _mm_or_si128(_mm_slli_epi16(dst_odd, 8), _mm_and_si128(dst_even, _mm_set1_epi16(0xFF)));
#else
    return _mm_or_si128(_mm_slli_epi16(dst_odd, 8), _mm_srli_epi16(_mm_slli_epi16(dst_even,8), 8));
#endif
}

inline __m256i _mm256_mul_epi8(__m256i x, __m256i y) {
  // Unpack and isolate 2 8 bit numbers from a 16 bit block in each vector using masks
  __m256i mask1 = _mm256_set1_epi16(0xFF00);
  __m256i mask2 = _mm256_set1_epi16(0x00FF);

  // Store the first and second members of the 8bit multiplication equation in their own 16 bit numbers, shifting down as necessary before calculating
  __m256i aa = _mm256_srli_epi16(_mm256_and_si256(x, mask1), 8);
  __m256i ab = _mm256_srli_epi16(_mm256_and_si256(y, mask1), 8);
  __m256i ba = _mm256_and_si256(x, mask2);
  __m256i bb = _mm256_and_si256(y, mask2);

  // Perform the multiplication, and undo any downshifting
  __m256i pa = _mm256_slli_epi16(_mm256_mullo_epi16(aa, ab), 8);
  __m256i pb = _mm256_mullo_epi16(ba, bb);

  // Mask out unwanted data to maintain isolation
  pa = _mm256_and_si256(pa, mask1);
  pb = _mm256_and_si256(pb, mask2);

  __m256i result = _mm256_or_si256(pa,pb);

  return result;
}

inline __m256i _mm256_sllv_epi8(__m256i a, __m256i count) {
    __m256i mask_hi        = _mm256_set1_epi32(0xFF00FF00);
    __m256i multiplier_lut = _mm256_set_epi8(0,0,0,0, 0,0,0,0, 0x80,0x40,0x20,0x10, 0x08,0x04,0x02,0x01, 0,0,0,0, 0,0,0,0, 0x80,0x40,0x20,0x10, 0x08,0x04,0x02,0x01);

    __m256i count_sat      = _mm256_min_epu8(count, _mm256_set1_epi8(8));     /* AVX shift counts are not masked. So a_i << n_i = 0 for n_i >= 8. count_sat is always less than 9.*/ 
    __m256i multiplier     = _mm256_shuffle_epi8(multiplier_lut, count_sat);  /* Select the right multiplication factor in the lookup table.                                      */
    
    __m256i x_lo           = _mm256_mullo_epi16(a, multiplier);               /* Unfortunately _mm256_mullo_epi8 doesn't exist. Split the 16 bit elements in a high and low part. */

    __m256i multiplier_hi  = _mm256_srli_epi16(multiplier, 8);                /* The multiplier of the high bits.                                                                 */
    __m256i a_hi           = _mm256_and_si256(a, mask_hi);                    /* Mask off the low bits.                                                                           */
    __m256i x_hi           = _mm256_mullo_epi16(a_hi, multiplier_hi);
    __m256i x              = _mm256_blendv_epi8(x_lo, x_hi, mask_hi);         /* Merge the high and low part. */

    return x;
}


inline __m256i _mm256_srlv_epi8(__m256i a, __m256i count) {
    __m256i mask_hi        = _mm256_set1_epi32(0xFF00FF00);
    __m256i multiplier_lut = _mm256_set_epi8(0,0,0,0, 0,0,0,0, 0x01,0x02,0x04,0x08, 0x10,0x20,0x40,0x80, 0,0,0,0, 0,0,0,0, 0x01,0x02,0x04,0x08, 0x10,0x20,0x40,0x80);

    __m256i count_sat      = _mm256_min_epu8(count, _mm256_set1_epi8(8));     /* AVX shift counts are not masked. So a_i >> n_i = 0 for n_i >= 8. count_sat is always less than 9.*/ 
    __m256i multiplier     = _mm256_shuffle_epi8(multiplier_lut, count_sat);  /* Select the right multiplication factor in the lookup table.                                      */
    __m256i a_lo           = _mm256_andnot_si256(mask_hi, a);                 /* Mask off the high bits.                                                                          */
    __m256i multiplier_lo  = _mm256_andnot_si256(mask_hi, multiplier);        /* The multiplier of the low bits.                                                                  */
    __m256i x_lo           = _mm256_mullo_epi16(a_lo, multiplier_lo);         /* Shift left a_lo by multiplying.                                                                  */
            x_lo           = _mm256_srli_epi16(x_lo, 7);                      /* Shift right by 7 to get the low bits at the right position.                                      */

    __m256i multiplier_hi  = _mm256_and_si256(mask_hi, multiplier);           /* The multiplier of the high bits.                                                                 */
    __m256i x_hi           = _mm256_mulhi_epu16(a, multiplier_hi);            /* Variable shift left a_hi by multiplying. Use a instead of a_hi because the a_lo bits don't interfere */
            x_hi           = _mm256_slli_epi16(x_hi, 1);                      /* Shift left by 1 to get the high bits at the right position.                                      */
    __m256i x              = _mm256_blendv_epi8(x_lo, x_hi, mask_hi);         /* Merge the high and low part.                                                                     */

    return x;
}

inline __m256i _mm256_rolv_epi8(__m256i x, __m256i y) {
    // Ensure the shift counts are within the range of 0 to 7
  __m256i y_mod = _mm256_and_si256(y, _mm256_set1_epi8(7));

  // Left shift x by y_mod
  __m256i left_shift = _mm256_sllv_epi8(x, y_mod);

  // Compute the right shift counts
  __m256i right_shift_counts = _mm256_sub_epi8(_mm256_set1_epi8(8), y_mod);

  // Right shift x by (8 - y_mod)
  __m256i right_shift = _mm256_srlv_epi8(x, right_shift_counts);

  // Combine the left-shifted and right-shifted results using bitwise OR
  __m256i rotated = _mm256_or_si256(left_shift, right_shift);

  return rotated;
}

// Rotates x left by r bits
inline __m256i _mm256_rol_epi8(__m256i x, int r) {
  // Unpack 2 8 bit numbers into their own vectors, and isolate them using masks
  __m256i mask1 = _mm256_set1_epi16(0x00FF);
  __m256i mask2 = _mm256_set1_epi16(0xFF00);
  __m256i a = _mm256_and_si256(x, mask1);
  __m256i b = _mm256_and_si256(x, mask2);

  // Apply the 8bit rotation to the lower 8 bits, then mask out any extra/overflow
  __m256i shiftedA = _mm256_slli_epi16(a, r);
  __m256i wrappedA = _mm256_srli_epi16(a, 8 - r);
  __m256i rotatedA = _mm256_or_si256(shiftedA, wrappedA);
  rotatedA = _mm256_and_si256(rotatedA, mask1);

  // Apply the 8bit rotation to the upper 8 bits, then mask out any extra/overflow
  __m256i shiftedB = _mm256_slli_epi16(b, r);
  __m256i wrappedB = _mm256_srli_epi16(b, 8 - r);
  __m256i rotatedB = _mm256_or_si256(shiftedB, wrappedB);
  rotatedB = _mm256_and_si256(rotatedB, mask2);

  // Re-pack the isolated results into a 16-bit block
  __m256i rotated = _mm256_or_si256(rotatedA, rotatedB);

  return rotated;
}

// parallelPopcnt16bytes - find population count for 8-bit groups in xmm (16 groups)
//                         each byte of xmm result contains a value ranging from 0 to 8
//
inline __m128i parallelPopcnt16bytes (__m128i xmm)
{
  const __m128i mask4 = _mm_set1_epi8 (0x0F);
  const __m128i lookup = _mm_setr_epi8 (0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  __m128i low, high, count;

  low = _mm_and_si128 (mask4, xmm);
  high = _mm_and_si128 (mask4, _mm_srli_epi16 (xmm, 4));
  count = _mm_add_epi8 (_mm_shuffle_epi8 (lookup, low), _mm_shuffle_epi8 (lookup, high));
  return count;
}

inline __m256i popcnt256_epi8(__m256i data) {
  __m128i hi = _mm256_extractf128_si256(data, 1);
  __m128i lo = _mm256_castsi256_si128(data);

  hi = parallelPopcnt16bytes(hi);
  lo = parallelPopcnt16bytes(lo);

  __m256i pop = _mm256_set_m128i(hi,lo);
  return pop;
}

/*
void testPopcnt256_epi8() {
    __m256i data = _mm256_setzero_si256();
    __m256i expected = _mm256_setzero_si256();

    // Test all possible byte values
    for (uint32_t i = 0; i < 256; ++i) {
        // Set the byte value in the data vector
        data = _mm256_set1_epi8(static_cast<int8_t>(i));

        // Calculate the expected population count for the byte value
        uint8_t popcnt = __builtin_popcount(i);
        expected = _mm256_set1_epi8(static_cast<int8_t>(popcnt));

        // Perform the population count using the function
        __m256i result = popcnt256_epi8(data);

        // Compare the result with the expected value
        if (!_mm256_testc_si256(result, expected)) {
            std::cout << "Test failed for byte value: " << i << std::endl;
            return;
        }
    }

    std::cout << "All tests passed!" << std::endl;
}
*/

inline __m256i _mm256_reverse_epi8(__m256i input) {
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_33 = _mm256_set1_epi8(0x33);
    const __m256i mask_55 = _mm256_set1_epi8(0x55);

    // b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    __m256i temp = _mm256_and_si256(input, mask_0f);
    temp = _mm256_slli_epi16(temp, 4);
    input = _mm256_and_si256(input, _mm256_andnot_si256(mask_0f, _mm256_set1_epi8(0xFF)));
    input = _mm256_srli_epi16(input, 4);
    input = _mm256_or_si256(input, temp);

    // b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    temp = _mm256_and_si256(input, mask_33);
    temp = _mm256_slli_epi16(temp, 2);
    input = _mm256_and_si256(input, _mm256_andnot_si256(mask_33, _mm256_set1_epi8(0xFF)));
    input = _mm256_srli_epi16(input, 2);
    input = _mm256_or_si256(input, temp);

    // b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    temp = _mm256_and_si256(input, mask_55);
    temp = _mm256_slli_epi16(temp, 1);
    input = _mm256_and_si256(input, _mm256_andnot_si256(mask_55, _mm256_set1_epi8(0xFF)));
    input = _mm256_srli_epi16(input, 1);
    input = _mm256_or_si256(input, temp);

    return input;
}

#endif