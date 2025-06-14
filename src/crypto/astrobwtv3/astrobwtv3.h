#ifndef astrobwtv3
#define astrobwtv3

#include "astroworker.h"

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
#include "tnn-hugepages.h"

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

#ifndef TNN_LEGACY_AMD64
template <unsigned int N>
__m256i shiftRight256(__m256i a);

template <unsigned int N> 
__m256i shiftLeft256(__m256i a);
#endif



extern void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex);

template <std::size_t N>
inline void generateInitVector(std::uint8_t (&iv_buff)[N]);

#include "simd_util.hpp"

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

  // std::copy(branchedOps_global.begin(), branchedOps_global.end(), worker.branchedOps);
  // std::vector<byte> full(256);
  // std::vector<byte> diff(256);
  // std::iota(full.begin(), full.end(), 0);
  // std::set_difference(full.begin(), full.end(), branchedOps_global.begin(), branchedOps_global.end(), std::inserter(diff, diff.begin()));
  // std::copy(diff.begin(), diff.end(), worker.regularOps);

  // for (int i = 0; i < 256; i++) {
  //   if (std::find(branchedOps_global.begin(), branchedOps_global.end(), i) != branchedOps_global.end())
  //     worker.isBranched.set(i, 1);
  // }

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

void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker);
void wolfSame(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker);
uint8_t wolfSingle(uint8_t *in, uint16_t op, uint8_t idx, uint8_t pos2);

void wolfCompute(workerData &worker, bool isTest, int wIndex);

typedef void (*wolfPerm)(uint8_t *, uint8_t *, uint16_t, uint8_t, uint8_t, workerData&);

#ifdef __x86_64__
#ifndef TNN_LEGACY_AMD64
void branchComputeCPU_avx2(workerData &worker, bool isTest, int wIndex);
void branchComputeCPU_avx2_zOptimized(workerData &worker, bool isTest, int wIndex);
#endif
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
