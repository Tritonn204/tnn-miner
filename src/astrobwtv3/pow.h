#ifndef astrobwtv3
#define astrobwtv3

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <unordered_map>

#include <random>
#include <chrono>
#include <math.h>
#include <Salsa20.h>

#include <openssl/sha.h>
#include <openssl/rc4.h>

#ifdef _WIN32
#include <winsock2.h>
#include <intrin.h>
#else
#include <arpa/inet.h>
#endif

#include <libcubwt.cuh>
#include <hugepages.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

#include "immintrin.h"

#ifndef POW_CONST
#define POW_CONST

#ifdef __GNUC__ 
#if __GNUC__ < 8
#define _mm256_set_m128i(xmm1, xmm2) _mm256_permute2f128_si256(_mm256_castsi128_si256(xmm1), _mm256_castsi128_si256(xmm2), 2)
#define _mm256_set_m128f(xmm1, xmm2) _mm256_permute2f128_ps(_mm256_castps128_ps256(xmm1), _mm256_castps128_ps256(xmm2), 2)
#endif
#endif

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

const uint32_t MAX_LENGTH = (256 * 384) - 1; // this is the maximum
const int deviceAllocMB = 5;

#endif

static const bool sInduction = true;
static const bool sTracking = true;

template <unsigned int N>
__m256i shiftRight256(__m256i a);

template <unsigned int N> 
__m256i shiftLeft256(__m256i a);

typedef unsigned int suffix;
typedef unsigned int t_index;
typedef unsigned char byte;
typedef unsigned short dbyte;
typedef unsigned long word;

const __m256i vec_3 = _mm256_set1_epi8(3);


//--------------------------------------------------------//

class workerData
{
public:
  unsigned char *sHash;
  unsigned char *sha_key;
  unsigned char *sha_key2;
  unsigned char *sData;

  unsigned char *counter;

  int *bA;
  int *bB;

  int *C;  // Count array for characters
  int *B;
  int D[512];  // Temporary array used in LMS sort

  SHA256_CTX sha256;
  ucstk::Salsa20 salsa20;
  RC4_KEY key;

  int32_t *sa;

  unsigned char *step_3;
  uint64_t random_switcher;

  uint64_t lhash;
  uint64_t prev_lhash;
  uint64_t tries;

  unsigned char op;
  unsigned char pos1;
  unsigned char pos2;
  unsigned char t1;
  unsigned char t2;

  unsigned char A;
  uint32_t data_len;

  void *GPUData[16];
  // void *cudaStore;

  std::vector<std::vector<unsigned char>> workBlobs;
  std::vector<std::vector<unsigned char>> saInputs;
  std::vector<uint32_t> inputSizes;
  std::vector<std::vector<uint32_t>> saResults2;
  std::vector<std::vector<uint32_t>> saResults;
  std::vector<std::vector<unsigned char>> outputHashes;
  std::vector<std::vector<unsigned char>> refHashes;

  std::vector<byte> opsA;
  std::vector<byte> opsB;

  void init()
  {
    sHash = (byte*)malloc_huge_pages(32);
    sha_key = (byte*)malloc_huge_pages(32);
    sha_key2 = (byte*)malloc_huge_pages(32);
    sData = (byte*)malloc_huge_pages(MAX_LENGTH+64);

    counter = (byte*)malloc_huge_pages(64);

    bA = (int*)malloc_huge_pages(256*sizeof(int));
    bB = (int*)malloc_huge_pages(256*256*sizeof(int));

    sa = (int32_t *)malloc_huge_pages(MAX_LENGTH*sizeof(int32_t));
    step_3 = (unsigned char *)malloc_huge_pages(256+32);

    C = (int*)malloc_huge_pages(MAX_LENGTH*4);
    B = (int*)malloc_huge_pages(MAX_LENGTH*4);
  }

  workerData()
  {
    init();
  }

  friend std::ostream& operator<<(std::ostream& os, const workerData& wd);
};

inline std::ostream& operator<<(std::ostream& os, const workerData& wd) {
    // Print values for dynamically allocated unsigned char arrays (assuming 32 bytes for demonstration)
    auto printByteArray = [&os](const unsigned char* arr, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            os << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(arr[i]) << " ";
        }
        os << std::dec << '\n'; // Switch back to decimal for other prints
    };

    os << "sHash: ";
    printByteArray(wd.sHash, 32);
    
    os << "sha_key: ";
    printByteArray(wd.sha_key, 32);
    
    os << "sha_key2: ";
    printByteArray(wd.sha_key2, 32);
    
    // Assuming data_len is the length of sData you're interested in printing
    os << "sData: ";
    printByteArray(wd.sData, MAX_LENGTH + 64);

    // For int arrays like bA, bB, C, and B, assuming lengths based on your constructor (example sizes)
    auto printIntArray = [&os](const int* arr, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            os << arr[i] << " ";
        }
        os << '\n';
    };

    // Example: Assuming sizes from your description
    os << "bA: ";
    printIntArray(wd.bA, 256); // Based on allocation in init

    os << "bB: ";
    printIntArray(wd.bB, 256*256); // Based on allocation in init

    // Assuming similar allocation sizes for C and B as for bA for demonstration
    os << "C: ";
    printIntArray(wd.C, 256); // Adjust size as necessary

    os << "B: ";
    printIntArray(wd.B, 256); // Adjust size as necessary

    // Directly contained int array D
    os << "D: ";
    for (int i = 0; i < 512; ++i) {
        os << wd.D[i] << " ";
    }
    os << '\n';

    // If you have other arrays or variables to print, follow the same pattern:
    // 1. Use printByteArray for unsigned char* with known sizes
    // 2. Use printIntArray for int* with known sizes
    // 3. Directly iterate over and print contents of fixed-size arrays or std::vector

    return os;
}

inline unsigned char
leftRotate8(unsigned char n, unsigned d)
{ // rotate n by d bits
#if defined(_WIN32)
  return _rotl8(n, d);
#else
  d = d % 8;
  return (n << d) | (n >> (8 - d));
#endif
}

void bitCountLookup();
inline unsigned char reverse8(unsigned char b)
{
  return (b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
}

inline unsigned char countSetBits(unsigned char n)
{
  unsigned char count = 0;
  while (n)
  {
    count += n & 1;
    n >>= 1;
  }
  return count;
}

inline unsigned char signByte(unsigned char A)
{
  A = (A + (A % 256)) % 256;
  return A;
}

template <std::size_t N>
inline void generateInitVector(std::uint8_t (&iv_buff)[N])
{
  using random_bytes_engine = std::independent_bits_engine<std::default_random_engine,
                                                           CHAR_BIT, unsigned short>;
  random_bytes_engine rbe;

  std::generate(std::begin(iv_buff), std::end(iv_buff), rbe);
}

inline void hashSHA256(SHA256_CTX &sha256, const unsigned char *input, unsigned char *digest, unsigned long inputSize)
{
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, input, inputSize);
  SHA256_Final(digest, &sha256);
}

inline std::vector<uint8_t> padSHA256Input(const uint8_t* input, size_t length) {
    // Calculate the length of the padded message
    size_t paddedLength = length + 1; // Original length plus the 0x80 byte
    size_t mod = paddedLength % 64;
    if (mod > 56) {
        paddedLength += 64 + 56 - mod; // Pad so there's room for the length
    } else {
        paddedLength += 56 - mod; // Pad so there's room for the length
    }
    paddedLength += 8; // Add 8 bytes for the original length

    // Create the padded message
    std::vector<uint8_t> padded(paddedLength, 0);
    memcpy(padded.data(), input, length);

    // Append the '1' bit
    padded[length] = 0x80;

    // Append the original length in bits as a 64-bit big-endian integer
    uint64_t bitLength = static_cast<uint64_t>(length) * 8; // Convert length to bits
    for (size_t i = 0; i < 8; ++i) {
        padded[paddedLength - 1 - i] = static_cast<uint8_t>((bitLength >> (8 * i)) & 0xff);
    }

    return padded;
}


void branchComputeCPU(workerData &worker);
void branchComputeCPU_optimized(workerData &worker);
void AstroBWTv3(unsigned char *input, int inputLen, unsigned char *outputhash, workerData &scratch, bool gpuMine, bool simd=false);
void finishBatch(workerData &worker);

#endif