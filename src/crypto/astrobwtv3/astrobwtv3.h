#ifndef astrobwtv3
#define astrobwtv3

#include "astroworker.h"
#include "tnn-hugepages.h"

#ifdef _MSC_VER
#define NOMINMAX // Disable Windows macros for min and max
#endif

#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <compile.h>

#include <unordered_map>

#include <random>
//#include <chrono>
#include <math.h>
#include <Salsa20.h>

#include <openssl/sha.h>
#include <openssl/rc4.h>

#include <bitset>

#ifdef _WIN32
#include <winsock2.h>
#include <intrin.h>
#else
#include <arpa/inet.h>
#endif

#include <atomic>
#include <libcubwt.cuh>
// #include <cuda.h>
// #include <cuda_runtime.h>

#if defined(__x86_64__)
  #include "immintrin.h"
#endif
#if defined(__aarch64__)
  #include <arm_neon.h>
#endif

#ifndef POW_CONST
#define POW_CONST

#define TNN_TYPICAL_MAX 71000
#define INSERT	10
#define MAXST	64
#define DEEP	150
#define DEELCP	10000
#define OVER	1000
#define TERM	((1+OVER)*sizeof(unsigned long))
#define LCPART	8
#define ABIT	7
const int AMASK	= (1<<ABIT)-1;

#if defined(__AVX2__)
#ifdef __GNUC__ 
#if __GNUC__ < 8
#define _mm256_set_m128i(xmm1, xmm2) _mm256_permute2f128_si256(_mm256_castsi128_si256(xmm1), _mm256_castsi128_si256(xmm2), 2)
#define _mm256_set_m128f(xmm1, xmm2) _mm256_permute2f128_ps(_mm256_castps128_ps256(xmm1), _mm256_castps128_ps256(xmm2), 2)
#endif
#endif

#define ALIGNMENT 32
#else

#define ALIGNMENT 16

#endif

#if defined(__AVX2__)
alignas(32) inline __m256i g_maskTable[32];
#endif

#define rl8(x, y) ((x << (y%8) | x >> (8-(y%8))))

const int MINIBLOCK_SIZE = 48;

typedef unsigned int suffix;
typedef unsigned int t_index;
typedef unsigned char byte;
typedef unsigned short dbyte;
typedef unsigned long word;

const int sus_op = 1;

const std::vector<unsigned char> branchedOps_global = {
1,3,5,9,11,13,15,17,20,21,23,27,29,30,35,39,40,43,45,47,51,54,58,60,62,64,68,70,72,74,75,80,82,85,91,92,93,94,103,108,109,115,116,117,119,120,123,124,127,132,133,134,136,138,140,142,143,146,148,149,150,154,155,159,161,165,168,169,176,177,178,180,182,184,187,189,190,193,194,195,199,202,203,204,212,214,215,216,219,221,222,223,226,227,230,231,234,236,239,240,241,242,250,253
};

// const uint64_t maskTips[8] = {
//   0x0000000000000000,
//   0xFF00000000000000,
//   0xFFFF000000000000,
//   0xFFFFFF0000000000,
//   0xFFFFFFFF00000000,
//   0xFFFFFFFFFF000000,
//   0xFFFFFFFFFFFF0000,
//   0xFFFFFFFFFFFFFF00,
// };

const uint64_t maskTips[8] = {
  0x00,         // n%8 = 0 
  0xFF,         // n%8 = 1
  0xFFFF,       // n%8 = 2 
  0xFFFFFF,     // n%8 = 3
  0xFFFFFFFF,   // n%8 = 4
  0xFFFFFFFFFF, // n%8 = 5
  0xFFFFFFFFFFFF, // n%8 = 6
  0xFFFFFFFFFFFFFFF // n%8 = 7
};

const uint32_t sha_standard[8] = {
    0x6a09e667, 
    0xbb67ae85, 
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f, 
    0x9b05688c, 
    0x1f83d9ab,
    0x5be0cd19
};

const int deviceAllocMB = 5;

#endif

static const bool sInduction = true;
static const bool sTracking = true;

#if defined(__AVX2__)
template <unsigned int N>
__m256i shiftRight256(__m256i a);

template <unsigned int N> 
__m256i shiftLeft256(__m256i a);

const __m256i vec_3 = _mm256_set1_epi8(3);
#endif



extern void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex);

template <std::size_t N>
inline void generateInitVector(std::uint8_t (&iv_buff)[N]);

#if defined(__AVX2__)
inline __m256i genMask(int i) {
  __m256i temp = _mm256_setzero_si256(); // Initialize mask with all zeros

  __m128i lower_part = _mm_set1_epi64x(0);
  __m128i upper_part = _mm_set1_epi64x(0);

  if (i > 24) {
    lower_part = _mm_set1_epi64x(-1ULL);
    upper_part = _mm_set_epi64x(-1ULL >> (32-i)*8,-1ULL);
  } else if (i > 16) {
    lower_part = _mm_set_epi64x(-1ULL,-1ULL);
    upper_part = _mm_set_epi64x(0,-1ULL >> (24-i)*8);
  } else if (i > 8) {
    lower_part = _mm_set_epi64x(-1ULL >> (16-i)*8,-1ULL);
  } else if (i > 0) {
    lower_part = _mm_set_epi64x(0,-1ULL >> (8-i)*8);
  }

  temp = _mm256_insertf128_si256(temp, lower_part, 0); // Set lower 128 bits
  temp = _mm256_insertf128_si256(temp, upper_part, 1); // Set upper 128 bits
  return temp;
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

inline int check_results(__m256i avx_result, unsigned char* scalar_result, int num_elements) {
  union {
     __m256i avx;
     unsigned char scalar[32];
  } converter;

  converter.avx = avx_result;

  for(int i = 0; i < num_elements; ++i) {
    if (converter.scalar[i] != scalar_result[i]) {
      std::cout << "Mismatch at index: " << i << std::endl;
      std::cout << "AVX: " << static_cast<int>(converter.scalar[i]) 
                << " Scalar: " << static_cast<int>(scalar_result[i]) << std::endl;
      return 0;
    }
  }

  return 1; 
}

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

inline uint32_t revInt(uint32_t dword)
{
  return ((dword >> 24) & 0x000000FF) | ((dword >> 8) & 0x0000FF00) | ((dword << 8) & 0x00FF0000) | ((dword << 24) & 0xFF000000);
}

inline void initWorker(workerData &worker) {
  #if defined(__AVX2__)

  for(int i = 0; i < 33; i++) {
    int size = 32-i;

    uint32_t a = ~(size > 28 ? 0xFFFFFFFF >> (std::max(4-(size - 28), 0)*8) : 0);  
    uint32_t b = ~(size > 24 ? 0xFFFFFFFF >> (std::max(4-(size - 24), 0)*8) : 0);  
    uint32_t c = ~(size > 20 ? 0xFFFFFFFF >> (std::max(4-(size - 20), 0)*8) : 0);  
    uint32_t d = ~(size > 16 ? 0xFFFFFFFF >> (std::max(4-(size - 16), 0)*8) : 0);  
    uint32_t e = ~(size > 12 ? 0xFFFFFFFF >> (std::max(4-(size - 12), 0)*8) : 0);  
    uint32_t f = ~(size > 8 ? 0xFFFFFFFF >> (std::max(4-(size - 8), 0)*8) : 0);  
    uint32_t g = ~(size > 4 ? 0xFFFFFFFF >> (std::max(4-(size - 4), 0)*8) : 0);  
    uint32_t h = ~(size > 0 ? 0xFFFFFFFF >> (std::max(4-size,0)*8) : 0);
    uint32_t vec[8] = {revInt(a),revInt(b),revInt(c),revInt(d),revInt(e),revInt(f),revInt(g),revInt(h)};

    // printf("genMask bytes for length %d: ", 32-i);
    // printf("%08X, %08X, %08X, %08X, %08X, %08X, %08X, %08X", a, b, c, d, e, f, g, h);
    // printf("\n");

    byte *cpyPoint = &worker.maskTable_bytes[32*i];
    memcpy(cpyPoint, vec, 32);
  }
  // printf("worker.maskTable\n");
  // uint32_t v[8];
  // for(int i = 0; i < 32; i++) {
  //   _mm256_storeu_si256((__m256i*)v, _mm256_loadu_si256(&worker.maskTable[i]));
  //   printf("%02d v8_u32: %x %x %x %x %x %x %x %x\n", i, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
  // }

  #endif

  memcpy(worker.iota8, iota8_g, 256);

  std::copy(branchedOps_global.begin(), branchedOps_global.end(), worker.branchedOps);
  std::vector<byte> full(256);
  std::vector<byte> diff(256);
  std::iota(full.begin(), full.end(), 0);
  std::set_difference(full.begin(), full.end(), branchedOps_global.begin(), branchedOps_global.end(), std::inserter(diff, diff.begin()));
  std::copy(diff.begin(), diff.end(), worker.regularOps);

  for (int i = 0; i < 256; i++) {
    if (std::find(branchedOps_global.begin(), branchedOps_global.end(), i) != branchedOps_global.end())
      worker.isBranched.set(i, 1);
  }

  // printf("Branched Ops:\n");
  // for (int i = 0; i < branchedOps_size; i++) {
  //   std::printf("%02X, ", worker.branchedOps[i]);
  // }
  // printf("\n");
  // printf("Regular Ops:\n");
  // for (int i = 0; i < regOps_size; i++) {
  //   std::printf("%02X, ", worker.regularOps[i]);
  // }
  // printf("\n");
}

// inline std::ostream& operator<<(std::ostream& os, const workerData& wd) {
//     // Print values for dynamically allocated byte arrays (assuming 32 bytes for demonstration)
//     auto printByteArray = [&os](const byte* arr, size_t size) {
//         for (size_t i = 0; i < size; ++i) {
//             os << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(arr[i]) << " ";
//         }
//         os << std::dec << '\n'; // Switch back to decimal for other prints
//     };

//     os << "sHash: ";
//     printByteArray(wd.sHash, 32);
    
//     os << "sha_key: ";
//     printByteArray(wd.sha_key, 32);
    
//     os << "sha_key2: ";
//     printByteArray(wd.sha_key2, 32);
    
//     // Assuming data_len is the length of sData you're interested in printing
//     os << "sData: ";
//     printByteArray(wd.sData, MAX_LENGTH + 64);

//     // For int arrays like bA, bB, C, and B, assuming lengths based on your constructor (example sizes)
//     auto printIntArray = [&os](const int* arr, size_t size) {
//         for (size_t i = 0; i < size; ++i) {
//             os << arr[i] << " ";
//         }
//         os << '\n';
//     };

//     // Example: Assuming sizes from your description
//     // os << "bA: ";
//     // printIntArray(wd.bA, 256); // Based on allocation in init

//     // os << "bB: ";
//     // printIntArray(wd.bB, 256*256); // Based on allocation in init

//     os << '\n';

//     // If you have other arrays or variables to print, follow the same pattern:
//     // 1. Use printByteArray for byte* with known sizes
//     // 2. Use printIntArray for int* with known sizes
//     // 3. Directly iterate over and print contents of fixed-size arrays or std::vector

//     return os;
// }

inline byte
leftRotate8(byte n, unsigned d)
{ // rotate n by d bits
#if defined(_WIN32)
  return _rotl8(n, d);
#else
  d = d % 8;
  return (n << d) | (n >> (8 - d));
#endif
}

void bitCountLookup();
inline byte reverse8(byte b)
{
  return (b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
}

inline byte countSetBits(byte n)
{
  byte count = 0;
  while (n)
  {
    count += n & 1;
    n >>= 1;
  }
  return count;
}

inline byte signByte(byte A)
{
  A = (A + (A % 256)) % 256;
  return A;
}

template <std::size_t N>
inline void generateInitVector(std::uint8_t (&iv_buff)[N])
{
  using random_bytes_engine = std::independent_bits_engine<std::default_random_engine,
                                                           CHAR_BIT, unsigned short>;
  random_bytes_engine rbe(rand());

  std::generate(std::begin(iv_buff), std::end(iv_buff), rbe);
}

template <typename T>
inline void prefetch(T *data, int size, int hint) {
  const size_t prefetch_distance = 256; // Prefetch 8 cache lines ahead
  const size_t cache_line_size = 64; // Assuming a 64-byte cache line

  //for (size_t i = 0; i < size; i += prefetch_distance * cache_line_size) {
  //    __builtin_prefetch(&data[i], 0, hint);
  //}
  switch(hint) {
    case 0:
      for (size_t i = 0; i < size; i += prefetch_distance * cache_line_size) {
          __builtin_prefetch(&data[i], 0, 0);
      }
    break;
    case 1:
      for (size_t i = 0; i < size; i += prefetch_distance * cache_line_size) {
          __builtin_prefetch(&data[i], 0, 1);
      }
    break;
    case 2:
      for (size_t i = 0; i < size; i += prefetch_distance * cache_line_size) {
          __builtin_prefetch(&data[i], 0, 2);
      }
    break;
    case 3:
      for (size_t i = 0; i < size; i += prefetch_distance * cache_line_size) {
          __builtin_prefetch(&data[i], 0, 3);
      }
      break;
    default:
    break;
  }
}

template <typename T>
inline void insertElement(T* arr, int& size, int capacity, int index, const T& element) {
    if (size < capacity) {
        // Shift elements to the right
        for (int i = size - 1; i >= index; i--) {
            arr[i + 1] = arr[i];
        }

        // Insert the new element
        arr[index] = element;

        // Increase the size
        size++;
    } else {
        std::cout << "Array is full. Cannot insert element." << std::endl;
    }
}

void initWolfLUT();

void mineDero(int tid);

void processAfterMarker(workerData& worker);
void lookupCompute(workerData &worker, bool isTest, int wIndex);
TNN_TARGETS
void branchComputeCPU(workerData &worker, bool isTest, int wIndex);

uint8_t wolfBranch(uint8_t val, uint8_t pos2val, uint32_t opcode);
void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker);
void wolfPermute_avx2(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker);
void wolfSame(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker);

void wolfCompute(workerData &worker, bool isTest, int wIndex);

typedef void (*wolfPerm)(uint8_t *, uint8_t *, uint16_t, uint8_t, uint8_t, workerData&);

#if defined(__AVX2__)
inline wolfPerm wolfPerms[2] = {wolfPermute_avx2, wolfSame};
#else
inline wolfPerm wolfPerms[1] = {wolfPermute};
#endif

#if defined(__AVX2__)

void branchComputeCPU_avx2(workerData &worker, bool isTest, int wIndex);

void branchComputeCPU_avx2_zOptimized(workerData &worker, bool isTest, int wIndex);
#elif defined(__aarch64__)
// This is gross.  But we need to do this until 'workerData' gets pushed into it's own include file.
void branchComputeCPU_aarch64(workerData &worker, bool isTest, int wIndex);
#endif

struct AstroFunc {
  std::string funcName;
  void (*funcPtr)(workerData& worker, bool isTest, int wIndex);
};

extern AstroFunc allAstroFuncs[];
extern size_t numAstroFuncs;

bool setAstroAlgo(std::string desiredAlgo);
void astroTune(int num_threads, int tuneWarmupSec, int tuneDurationSec);

TNN_TARGETS
void AstroBWTv3(byte *input, int inputLen, byte *outputhash, workerData &scratch, bool unused, int tid = 0);
TNN_TARGETS
void AstroBWTv3_batch(byte *input, int inputLen, byte *outputhash, workerData &worker, bool unused);
void finishBatch(workerData &worker);

static void construct_SA_pre(const byte *T, int *SA,
             int *bucket_A, int *bucket_B,
             std::vector<std::vector<int>> &buckets_A,
             int n, int m);

#undef INSERT
#undef MAXST
#undef DEEP
#undef DEELCP
#undef OVER
#undef TERM
#undef LCPART
#undef ABIT

#endif
