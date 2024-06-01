#include <endian.hpp>
#include <inttypes.h>

#include <unistd.h>

#include <bitset>
#include <iostream>
#include <fstream>

#include <fnv1a.h>
#include <xxhash64.h>
#include "astrobwtv3.h"
#include "astrotest.hpp"
#include "branched_AVX2.h"

#include <unordered_map>
#include <array>
#include <algorithm>

#if defined(__x86_64__)
  #include <xmmintrin.h>
  #include <emmintrin.h>
#endif
#if defined(__aarch64__)
  #include "astro_aarch64.hpp"
#endif

#include <random>
#include <chrono>

#include <Salsa20.h>

#include <highwayhash/sip_hash.h>
#include <filesystem>
#include <functional>
#include "lookupcompute.h"
#include "archon3r3.h"
// #include "archon4r0.h"
#include "archon.h"
#include "sais_lcp.hpp"
#include "dc3.hpp"

extern "C"
{
  #include "divsufsort_private.h"
  #include "divsufsort.h"
}

#include <utility>

#include <hex.h>
#include <openssl/rc4.h>
// #include "sais2.h"

#include <fstream>

#include <bit>
// #include <libcubwt.cuh>
// #include <device_sa.cuh>
#include <lookup.h>
// #include <sacak-lcp.h>
#if defined(__x86_64__)
  #include "immintrin.h"
#endif
#include "dc3.hpp"
// #include "fgsaca.hpp"
#include <hugepages.h>

using byte = unsigned char;

int ops[256];

uint16_t lookup2D[regOps_size*256*256];
byte lookup3D[branchedOps_size*256*256];

std::vector<byte> opsA;
std::vector<byte> opsB;

bool debugOpOrder = false;

void saveBufferToFile(const std::string& filename, const byte* buffer, size_t size) {
    // Generate unique filename using timestamp
    std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now().time_since_epoch()).count());
    std::string unique_filename = "tests/worker_sData_snapshot_" + timestamp;

    std::ofstream file(unique_filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(buffer), size);
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// TODO: Implement dynamic SIMD checks for branchCompute
/*
void checkSIMDSupport() {
    // Setup a function pointer to detect AVX2 
    void (*func_ptr)() = nullptr;
#ifdef __AVX2__
    func_ptr = __builtin_cpu_supports("avx2");
#endif
    if (func_ptr && func_ptr()) {
        // AVX2 is supported - use AVX2 intrinsics
    } else {
        // Setup a function pointer to detect SSE2
        func_ptr = nullptr; 
#ifdef __SSE2__ 
        func_ptr = __builtin_cpu_supports("sse2"); 
#endif
        if (func_ptr && func_ptr()) {
            // SSE2 is supported - use SSE2 intrinsics
        } else {
            // Use scalar code
        }
    }
}
*/

void hashSHA256(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize)
{
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, input, inputSize);
  SHA256_Final(digest, &sha256);
}

std::vector<uint8_t> padSHA256Input(const uint8_t* input, size_t length) {
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

void checkFalseRep(workerData &worker) {
  byte f = worker.prev_chunk[worker.pos1];
  bool falsePos = false;
  for (int i = worker.pos1; i < worker.pos2; i++) {
    if (worker.prev_chunk[i] != f) {
      falsePos = true;
      break;
    }
  }

  if (falsePos) {
    printf("false positive\n");
    for (int i = worker.pos1; i < worker.pos2; i++) {
      printf("%02x ", worker.prev_chunk[i]);
    }
    printf("\n");
  }
}

// Function to compare two suffixes based on lexicographical order
bool cmp(const uint8_t* data, const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) {
    const uint8_t* p1 = data + a.second;
    const uint8_t* p2 = data + b.second;

    while (*p1 != 0 && *p1 == *p2) {
        ++p1;
        ++p2;
    }

    return *p1 < *p2;
}

void buildSuffixArray(const uint8_t* data, int n, int* suffixArray, int* buckets, int* sorted) {
    if (n <= 0 || !data || !suffixArray || !buckets || !sorted) {
        // Handle invalid input
        return;
    }

    // Step 1: Counting sort on first character
    for (int i = 0; i < 256; ++i) {
        buckets[i] = 0;  // Initialize buckets
    }

    for (int i = 0; i < n; ++i) {
        buckets[data[i]]++;
    }

    for (int i = 1; i < 256; ++i) {
        buckets[i] += buckets[i - 1];
    }

    for (int i = n - 1; i >= 0; --i) {
        sorted[--buckets[data[i]]] = i;
    }

    // Step 2: Sort suffixes recursively
    int* indices = new int[n];
    indices[sorted[0]] = 0;
    int rank = 0;
    for (int i = 1; i < n; ++i) {
        if (data[sorted[i]] != data[sorted[i - 1]]) {
            rank++;
        }
        indices[sorted[i]] = rank;
    }

    int* temp = new int[n];
    for (int k = 1; (1 << k) < n; k <<= 1) {
        for (int i = 0; i < n; ++i) {
            temp[i] = (indices[i] << k) | (i + (1 << k) < n ? indices[i + (1 << k)] : 0);
        }

        for (int i = 0; i < n; ++i) {
            buckets[temp[i]] = 0;  // Reset buckets
        }

        for (int i = 0; i < n; ++i) {
            buckets[temp[i]]++;
        }

        for (int i = 1; i < n; ++i) {
            buckets[i] += buckets[i - 1];
        }

        for (int i = n - 1; i >= 0; --i) {
            suffixArray[--buckets[temp[i]]] = sorted[i];
        }

        for (int i = 0; i < n; ++i) {
            indices[suffixArray[i]] = temp[suffixArray[i]] == temp[suffixArray[i - 1]] && i > 0 ? indices[suffixArray[i - 1]] : i;
        }
    }

    delete[] indices;
    delete[] temp;
}

void runDivsufsortBenchmark() {
  std::vector<std::string> snapshotFiles;
  for (const auto& entry : std::filesystem::directory_iterator("tests")) {
    snapshotFiles.push_back(entry.path().string());
  }
  
  std::vector<std::chrono::duration<double>> times;
  std::vector<std::chrono::duration<double>> times2;
  std::vector<std::chrono::duration<double>> times3;
  std::vector<std::chrono::duration<double>> times4;
  std::vector<std::chrono::duration<double>> times5;

  int buckets[256];
  int sorted[MAX_LENGTH];

  
  int32_t *sa = reinterpret_cast<int32_t *>(malloc_huge_pages(MAX_LENGTH*sizeof(int32_t)));
  uint32_t *sa2 = reinterpret_cast<uint32_t *>(malloc_huge_pages(MAX_LENGTH*sizeof(uint32_t)));
  byte *buffer = reinterpret_cast<byte *>(malloc_huge_pages(MAX_LENGTH));
  int32_t *bA = reinterpret_cast<int32_t *>(malloc_huge_pages((256)*sizeof(int32_t)));;
  int32_t *bB = reinterpret_cast<int32_t *>(malloc_huge_pages((256*256)*sizeof(int32_t)));

  void * ctx = libsais_create_ctx();
  workerData *worker = new workerData;

  std::cout << "Testing divsufsort" << std::endl;
  for (const auto& file : snapshotFiles) {
    // printf("enter\n");
    // Load snapshot data from file
    std::ifstream ifs(file, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t size = ifs.tellg(); 
    ifs.seekg(0, ifs.beg);
    ifs.read(reinterpret_cast<char*>(buffer), size);
    ifs.close();
    
    // Run divsufsort
    auto start = std::chrono::steady_clock::now();
    divsufsort(buffer, sa, size, bA, bB); 
    auto end = std::chrono::steady_clock::now();
    
    std::chrono::duration<double> time = end - start;
    times.push_back(time);
  }

  memset(sa, 0, MAX_LENGTH);

  std::cout << "Testing sais" << std::endl;
  for (const auto& file : snapshotFiles) {
    // printf("enter\n");
    // Load snapshot data from file
    std::ifstream ifs(file, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t size = ifs.tellg(); 
    ifs.seekg(0, ifs.beg);
    byte* buffer = new byte[size];
    ifs.read(reinterpret_cast<char*>(buffer), size);
    ifs.close();
    
    // Run libcubwt
    prefetch(buffer, size, 3);
    auto start = std::chrono::steady_clock::now(); 
    libsais_ctx(ctx, buffer, sa, size, 0, NULL); 
    auto end = std::chrono::steady_clock::now();
    
    auto time = end - start;
    times2.push_back(time);
  }

  memset(sa, 0, MAX_LENGTH);

  std::cout << "Testing Archon3r3" << std::endl;
  for (const auto& file : snapshotFiles) {
    // Load snapshot data from file
    std::ifstream ifs(file, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t size = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    ifs.read(reinterpret_cast<char*>(worker->sData), size);
    ifs.close();

    worker->data_len = size;

    // Run Archon3r3
    auto start = std::chrono::steady_clock::now();
    encode(*worker);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> time = end - start;
    times3.push_back(time);
  }

  memset(sa, 0, MAX_LENGTH);

  std::cout << "Testing Archon" << std::endl;
  for (const auto& file : snapshotFiles) {
      // Load snapshot data from file
      FILE* fp = fopen(file.c_str(), "rb");
      if (!fp) {
          std::cout << "Failed to open file: " << file << std::endl;
          continue;
      }

      fseek(fp, 0, SEEK_END);
      size_t size = ftell(fp);
      fseek(fp, 0, SEEK_SET);

      Archon ar(size);
      ar.enRead(fp, size);
      fclose(fp);

      // Run Archon
      auto start = std::chrono::steady_clock::now();
      ar.enCompute();
      auto end = std::chrono::steady_clock::now();

      std::chrono::duration<double> time = end - start;
      times4.push_back(time);
  }

  memset(sa, 0, MAX_LENGTH);

  std::cout << "Testing DC3" << std::endl;
  for (const auto& file : snapshotFiles) {
    // printf("enter\n");
    // Load snapshot data from file
    std::ifstream ifs(file, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t size = ifs.tellg(); 
    ifs.seekg(0, ifs.beg);
    byte* buffer = new byte[size+3];
    for (int i = size-3; i < size; i++) buffer[i] = 0;
    ifs.read(reinterpret_cast<char*>(buffer), size);
    ifs.close();
    
    // Run libcubwt
    std::string str(reinterpret_cast<char const*>(buffer), size);
    auto start = std::chrono::steady_clock::now(); 
    std::vector<std::string::size_type> suffArr = dc3::suffixArray(str.cbegin(), str.cend());
    auto end = std::chrono::steady_clock::now();
    
    auto time = end - start;
    times5.push_back(time);
  }

  double divsufsortAverage = 0.0;
  double libsaisAverage = 0.0;
  double archon3r3Average = 0.0;
  double archon4r0Average = 0.0;
  double saislcpAverage = 0.0;

  for (const auto& time : times) {
    divsufsortAverage += time.count();
  }
  for (const auto& time : times2) {
    libsaisAverage += time.count(); 
  }
  for (const auto& time : times3) {
    archon3r3Average += time.count();
  }
  for (const auto& time : times4) {
    archon4r0Average += time.count();
  }
  for (const auto& time : times5) {
    saislcpAverage += time.count();
  }

  std::cout << "Total divsufsort time: " << std::setprecision (5) << divsufsortAverage << " seconds" << std::endl;
  std::cout << "Total sais time:       " << std::setprecision (5) << libsaisAverage << " seconds" << std::endl;
  std::cout << "Total Archon3r3 time:  " << std::setprecision(5) << archon3r3Average << " seconds" << std::endl;
  std::cout << "Total Archon4r0 time:  " << std::setprecision(5) << archon4r0Average << " seconds" << std::endl;
  std::cout << "Total DC3 time:  " << std::setprecision(5) << saislcpAverage << " seconds" << std::endl;

  divsufsortAverage /= times.size();
  libsaisAverage /= times2.size();
  archon3r3Average /= times3.size();
  archon4r0Average /= times4.size();
  saislcpAverage /= times5.size();

  std::cout << "Average divsufsort time: " << divsufsortAverage << " seconds" << std::endl;
  std::cout << "Average sais time:       " << libsaisAverage << " seconds" << std::endl;
  std::cout << "Average Archon3r3 time:       " << archon3r3Average << " seconds" << std::endl;
  std::cout << "Average Archon4r0 time:       " << archon4r0Average << " seconds" << std::endl;
  std::cout << "Average DC3 time:       " << saislcpAverage << " seconds" << std::endl;
}

#if defined(__AVX2__)

void computeByteFrequencyAVX2(const unsigned char* data, size_t dataSize, int frequencyTable[256]) {
    __m256i chunk;
    const size_t simdWidth = 32; // AVX2 SIMD register width in bytes

    // Zero-initialize a local frequency table to avoid read-modify-write AVX2 operations
    int localFrequencyTable[256] = {0};

    // Process chunks of 32 bytes
    for (size_t i = 0; i < dataSize; i += simdWidth) {
        if (i + simdWidth <= dataSize) { // Ensure we don't read past the end
            chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));

            // Update frequency table in a non-vectorized manner due to AVX2 limitations
            unsigned char temp[simdWidth];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), chunk);

            for (size_t j = 0; j < simdWidth; ++j) {
                ++localFrequencyTable[temp[j]];
            }
        } else {
            // Handle remainder bytes that don't fit into a full AVX2 register
            for (size_t j = i; j < dataSize; ++j) {
                ++localFrequencyTable[data[j]];
            }
            break; // Exit the loop after processing the remainder
        }
    }

    // Accumulate the results into the provided frequency table
    for (int i = 0; i < 256; ++i) {
        frequencyTable[i] += localFrequencyTable[i];
    }
}

#endif


void AstroBWTv3(byte *input, int inputLen, byte *outputhash, workerData &worker, bool lookupMine)
{
  // auto recoverFunc = [&outputhash](void *r)
  // {
  //   std::random_device rd;
  //   std::mt19937 gen(rd());
  //   std::uniform_int_distribution<uint8_t> dist(0, 255);
  //   std::array<uint8_t, 16> buf;
  //   std::generate(buf.begin(), buf.end(), [&dist, &gen]()
  //                 { return dist(gen); });
  //   std::memcpy(outputhash, buf.data(), buf.size());
  //   std::cout << "exception occured, returning random hash" << std::endl;
  // };
  // std::function<void(void *)> recover = recoverFunc;

  try
  {
    std::fill_n(worker.sData + 256, 64, 0);

    __builtin_prefetch(&worker.sData[256], 1, 3);
    __builtin_prefetch(&worker.sData[256+64], 1, 3);
    __builtin_prefetch(&worker.sData[256+128], 1, 3);
    __builtin_prefetch(&worker.sData[256+192], 1, 3);
    
    hashSHA256(worker.sha256, input, &worker.sData[320], inputLen);
    worker.salsa20.setKey(&worker.sData[320]);
    worker.salsa20.setIv(&worker.sData[256]);

    __builtin_prefetch(worker.sData, 1, 3);
    __builtin_prefetch(&worker.sData[64], 1, 3);
    __builtin_prefetch(&worker.sData[128], 1, 3);
    __builtin_prefetch(&worker.sData[192], 1, 3);

    worker.salsa20.processBytes(worker.salsaInput, worker.sData, 256);

    __builtin_prefetch(&worker.key + 8, 1, 3);
    __builtin_prefetch(&worker.key + 8+64, 1, 3);
    __builtin_prefetch(&worker.key + 8+128, 1, 3);
    __builtin_prefetch(&worker.key + 8+192, 1, 3);

    RC4_set_key(&worker.key, 256,  worker.sData);
    RC4(&worker.key, 256, worker.sData,  worker.sData);

    worker.lhash = hash_64_fnv1a_256(worker.sData);
    worker.prev_lhash = worker.lhash;

    worker.tries = 0;
    worker.isSame = false;

    // printf(hexStr(worker.chunk, 256).c_str());
    // printf("\n\n");

    
    // auto start = std::chrono::steady_clock::now();
    // auto end = std::chrono::steady_clock::now();

    if (lookupMine) {
      // start = std::chrono::steady_clock::now();
      lookupCompute(worker, false);
      // end = std::chrono::steady_clock::now();
    }
    else {
      // start = std::chrono::steady_clock::now();
      #if defined(__AVX2__)
      // branchComputeCPU_avx2(worker, false);
      branchComputeCPU_avx2_zOptimized(worker, false);
      #elif defined(__aarch64__)
      branchComputeCPU_aarch64(worker, false);
      #else
      branchComputeCPU(worker, false);
      #endif
      // end = std::chrono::steady_clock::now();
    }
    

    // auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    // if (!lookupMine) printf("AVX2: ");
    // else printf("Lookup: ");
    // printf("branched section took %dns\n", time.count());
    if (debugOpOrder) {
      if (lookupMine) {
        printf("Lookup Table:\n-----------\n");
        for (int i = 0; i < worker.opsB.size(); i++) {
          printf("%d, ", worker.opsB[i]);
        }
      } else {
        printf("Scalar:\n-----------\n");
        for (int i = 0; i < worker.opsA.size(); i++) {
          printf("%d, ", worker.opsA[i]);
        }
      }

      printf("\n");
      worker.opsA.clear();
      worker.opsB.clear();
    }
    // // worker.data_len = 70000;
    // saveBufferToFile("worker_sData_snapshot.bin", worker.sData, worker.data_len);
    // printf("data length: %d\n", worker.data_len);
    divsufsort(worker.sData, worker.sa, worker.data_len, worker.bA, worker.bB);
    // computeByteFrequencyAVX2(worker.sData, worker.data_len, worker.freq);
    // libsais_ctx(worker.ctx, worker.sData, worker.sa, worker.data_len, MAX_LENGTH-worker.data_len, NULL);

    if (littleEndian())
    {
      byte *B = reinterpret_cast<byte *>(worker.sa);
      hashSHA256(worker.sha256, B, worker.sHash, worker.data_len*4);
      // worker.sHash = nHash;
    }
    else
    {
      byte *s = new byte[MAX_LENGTH * 4];
      for (int i = 0; i < worker.data_len; i++)
      {
        s[i << 1] = htonl(worker.sa[i]);
      }
      hashSHA256(worker.sha256, s, worker.sHash, worker.data_len*4);
      // worker.sHash = nHash;
      delete[] s;
    }
    memcpy(outputhash, worker.sHash, 32);
  }
  catch (const std::exception &ex)
  {
    // recover(outputhash);
    std::cerr << ex.what() << std::endl;
  }
}


void branchComputeCPU(workerData &worker, bool isTest)
{
  while (true)
  {
    if(isTest) {

    } else {
      worker.tries++;
      if (debugOpOrder) printf("t: 0x%lx p: 0x%lx l: 0x%lx\n", worker.tries, worker.prev_lhash, worker.lhash);
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries;
      // __builtin_prefetch(&worker.random_switcher,0,3);
      // printf("%d worker.random_switcher %d %08jx\n", worker.tries, worker.random_switcher, worker.random_switcher);

      worker.op = static_cast<byte>(worker.random_switcher);
      //if (debugOpOrder) worker.opsA.push_back(worker.op);

      // printf("op: %d\n", worker.op);

      worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
      worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

      if (worker.pos1 > worker.pos2)
      {
        std::swap(worker.pos1, worker.pos2);
      }

      if (worker.pos2 - worker.pos1 > 32)
      {
        worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
      }

      worker.chunk = &worker.sData[(worker.tries - 1) * 256];
      if (debugOpOrder) printf("worker.op: %03d p1: %03d p2: %03d\n", worker.op, worker.pos1, worker.pos2);

      if (worker.tries == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[(worker.tries - 2) * 256];
      }

      if (debugOpOrder) {
        printf("tries: %03lu chunk_before[  0->%03d]: ", worker.tries, worker.pos2);
        for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
          printf("%02x", worker.chunk[x]);
        }
        printf("\n");
      }

      memcpy(worker.chunk, worker.prev_chunk, 256);
      if (debugOpOrder) {
        printf("tries: %03lu  chunk_fixed[  0->%03d]: ", worker.tries, worker.pos2);
        for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
          //printf("%d \n", x);
          printf("%02x", worker.chunk[x]);
        }
        printf("\n");
      }
    }

    switch (worker.op)
    {
    case 0:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random

        // INSERT_RANDOM_CODE_END
        worker.t1 = worker.chunk[worker.pos1];
        worker.t2 = worker.chunk[worker.pos2];
        worker.chunk[worker.pos1] = reverse8(worker.t2);
        worker.chunk[worker.pos2] = reverse8(worker.t1);
      }
      break;
    case 1:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 2:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 3:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 4:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 5:
    {
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {

        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right

        // INSERT_RANDOM_CODE_END
      }
    }
    break;
    case 6:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -

        // INSERT_RANDOM_CODE_END
      }
      break;
    case 7:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 8:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 10); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 5);// rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 9:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 10:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] *= worker.chunk[i];              // *
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 11:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 5);            // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 12:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 13:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 14:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 15:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 16:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 17:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 18:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 9);  // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 5);         // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 19:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 20:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 21:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 22:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 23:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 4); // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 1);                           // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 24:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 25:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 26:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                 // *
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 27:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 28:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 29:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 30:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 31:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 32:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 33:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 34:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 35:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 1); // rotate  bits by 1
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 36:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 37:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 38:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 39:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 40:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 41:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] -= (worker.chunk[i] ^ 97);        // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 42:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 4); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 43:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 44:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 45:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 10); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 5);                       // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 46:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 47:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 48:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        // worker.chunk[i] = ~worker.chunk[i];                    // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];                    // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 49:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 50:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);     // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = rl8(worker.chunk[i], 1); // rotate  bits by 1
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 51:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 52:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 53:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 54:

#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);  // reverse bits
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        // worker.chunk[i] = ~worker.chunk[i];    // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];    // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }

      break;
    case 55:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 56:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 57:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 8);                // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 58:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 59:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 60:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 61:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 8);             // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 5);// rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 62:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 63:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] += worker.chunk[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 64:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 65:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 8); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 66:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 67:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 68:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 69:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 70:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 71:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 72:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 73:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 74:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 75:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 76:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 77:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 78:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 79:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 80:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 81:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 82:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 83:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 84:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 85:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 86:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 87:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] += worker.chunk[i];               // +
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 88:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 89:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 90:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);     // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 91:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 92:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 93:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 94:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 95:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 10); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 96:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 97:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 98:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 99:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 100:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 101:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 102:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 103:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 104:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] += worker.chunk[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 105:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 106:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 107:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 6);             // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 108:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 109:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 110:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 111:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 112:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 113:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 1);                           // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 114:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 115:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 116:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 117:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 118:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 119:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 120:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 121:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 122:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 123:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 6);                // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 124:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 125:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 126:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 9); // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 1); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 127:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 128:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 129:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 130:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 131:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] *= worker.chunk[i];                 // *
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 132:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 133:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 134:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 135:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 136:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 137:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 138:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 139:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 8); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 140:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 141:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 142:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 143:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 144:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 145:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 146:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 147:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 148:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 149:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);  // reverse bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
        worker.chunk[i] += worker.chunk[i];           // +
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 150:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 151:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 152:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 153:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 4); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        // worker.chunk[i] = ~worker.chunk[i];     // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];     // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 154:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 155:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 156:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = rl8(worker.chunk[i], 4);             // rotate  bits by 3
        // worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 157:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 158:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);    // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 159:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 160:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 4);             // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 3);    // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 161:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 162:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] -= (worker.chunk[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 163:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 164:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                 // *
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 165:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 166:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 167:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 168:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 169:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 170:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);   // XOR and -
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);   // XOR and -
        worker.chunk[i] *= worker.chunk[i];          // *
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 171:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);    // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 172:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 173:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 174:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 175:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 176:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 177:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 178:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 179:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 180:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 181:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 182:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 5);         // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 183:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];        // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97); // XOR and -
        worker.chunk[i] -= (worker.chunk[i] ^ 97); // XOR and -
        worker.chunk[i] *= worker.chunk[i];        // *
                                                     // INSERT_RANDOM_CODE_END
      }
      break;
    case 184:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 185:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 186:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 187:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 188:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 189:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] -= (worker.chunk[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 190:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 191:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 192:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 193:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 194:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 195:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 196:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 197:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 198:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 199:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];           // binary NOT operator
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] *= worker.chunk[i];           // *
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 200:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 201:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 202:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 203:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], 1);                // rotate  bits by 1
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 204:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 205:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 206:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 207:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 8); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 3);                           // rotate  bits by 3
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 208:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 209:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 210:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 211:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 212:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 213:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 214:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 215:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 216:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 217:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 218:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
        worker.chunk[i] = ~worker.chunk[i];          // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];          // *
        worker.chunk[i] -= (worker.chunk[i] ^ 97);   // XOR and -
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 219:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 220:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 221:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5); // rotate  bits by 5
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] = reverse8(worker.chunk[i]);     // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 222:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 223:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 224:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 4);  // rotate  bits by 1
        // worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       //
      }
      break;
    case 225:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 226:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);  // reverse bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
        worker.chunk[i] *= worker.chunk[i];           // *
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 227:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 228:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 229:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 230:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 231:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 232:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 233:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);    // rotate  bits by 3
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 234:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 235:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 236:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 237:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 238:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = rl8(worker.chunk[i], 3); // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 239:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 6); // rotate  bits by 5
        // worker.chunk[i] = rl8(worker.chunk[i], 1); // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 240:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 241:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 242:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 243:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 244:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 245:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = rl8(worker.chunk[i], 5);             // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 246:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = rl8(worker.chunk[i], 1);             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 247:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 248:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 249:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 250:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 251:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 252:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] ^= rl8(worker.chunk[i], 4);            // rotate  bits by 4
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 253:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        // INSERT_RANDOM_CODE_END

        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
      }
      break;
    case 254:
    case 255:
      RC4_set_key(&worker.key, 256,  worker.chunk);
// worker.chunk = highwayhash.Sum(worker.chunk[:], worker.chunk[:])
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= static_cast<uint8_t>(std::bitset<8>(worker.chunk[i]).count()); // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 3);                                  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);                                 // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 3);                                  // rotate  bits by 3
                                                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    default:
      break;
    }

    if(isTest) {
      break;
    }

    // if (op == 53) {
    //   std::cout << hexStr(worker.chunk, 256) << std::endl << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos1], 1) << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos2], 1) << std::endl;
    // }

    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    if (debugOpOrder){printf("worker.A: %02X\n", worker.A);}

    if (worker.A < 0x10)
    { // 6.25 % probability
      if (debugOpOrder){printf("A\n");}
      __builtin_prefetch(worker.chunk, 0, 0);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
      // if (debugOpOrder) printf("A: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x20)
    { // 12.5 % probability
      if (debugOpOrder){printf("B\n");}
      __builtin_prefetch(worker.chunk, 0, 0);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);
      // if (debugOpOrder) printf("B: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x30)
    { // 18.75 % probability
      // std::copy(worker.chunk, worker.chunk + worker.pos2, s3);
        if (debugOpOrder){printf("C\n");}
      __builtin_prefetch(worker.chunk, 0, 0);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      HH_ALIGNAS(16)
      const highwayhash::HH_U64 key2[2] = {worker.tries, worker.prev_lhash};
      worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations
      // if (debugOpOrder) printf("C: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A <= 0x40)
    { // 25% probablility
      // if (debugOpOrder) {
      //   printf("D: RC4 key:\n");
      //   for (int i = 0; i < 256; i++) {
      //     printf("%d, ", worker.key.data[i]);
      //   }
      // }
        if (debugOpOrder){printf("D\n");}
      __builtin_prefetch(&worker.key, 0, 0);
      RC4(&worker.key, 256, worker.chunk,  worker.chunk);
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    prefetch(worker.chunk, 256, 1);
    memcpy(&worker.sData[(worker.tries - 1) * 256], worker.chunk, 256);

    
    if (debugOpOrder && worker.op == sus_op) {
      printf("op %d result:\n", worker.op);
      for (int i = 0; i < 256; i++) {
          printf("%02X ", worker.chunk[i]);
      } 
      printf("\n");
    }
    // std::copy(worker.chunk, worker.chunk + 256, &worker.sData[(worker.tries - 1) * 256]);

    // memcpy(&worker->data.data()[(worker.tries - 1) * 256], worker.chunk, 256);

    // std::cout << hexStr(worker.chunk, 256) << std::endl;

    if (worker.tries > 260 + 16 || (worker.chunk[255] >= 0xf0 && worker.tries > 260))
    {
      break;
    }
  }
  worker.data_len = static_cast<uint32_t>((worker.tries - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
}

#if defined(__AVX2__)

void branchComputeCPU_avx2(workerData &worker, bool isTest)
{
  while (true)
  {
    if(isTest) {

    } else {
      worker.tries++;
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries;
      // __builtin_prefetch(&worker.random_switcher,0,3);
      // printf("%d worker.random_switcher %d %08jx\n", worker.tries, worker.random_switcher, worker.random_switcher);

      worker.op = static_cast<byte>(worker.random_switcher);
      // if (debugOpOrder) worker.opsA.push_back(worker.op);

      // printf("op: %d\n", worker.op);

      worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
      worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

      if (worker.pos1 > worker.pos2)
      {
        std::swap(worker.pos1, worker.pos2);
      }

      if (worker.pos2 - worker.pos1 > 32)
      {
        worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
      }

      worker.chunk = &worker.sData[(worker.tries - 1) * 256];

      if (worker.tries == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[(worker.tries - 2) * 256];

        __builtin_prefetch(worker.prev_chunk,0,3);
        __builtin_prefetch(worker.prev_chunk+64,0,3);
        __builtin_prefetch(worker.prev_chunk+128,0,3);
        __builtin_prefetch(worker.prev_chunk+192,0,3);

        // Calculate the start and end blocks
        int start_block = 0;
        int end_block = worker.pos1 / 16;

        // Copy the blocks before worker.pos1
        for (int i = start_block; i < end_block; i++) {
            __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
            _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
        }

        // Copy the remaining bytes before worker.pos1
        for (int i = end_block * 16; i < worker.pos1; i++) {
            worker.chunk[i] = worker.prev_chunk[i];
        }

        // Calculate the start and end blocks
        start_block = (worker.pos2 + 15) / 16;
        end_block = 16;

        // Copy the blocks after worker.pos2
        for (int i = start_block; i < end_block; i++) {
            __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
            _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
        }

        // Copy the remaining bytes after worker.pos2
        for (int i = worker.pos2; i < start_block * 16; i++) {
          worker.chunk[i] = worker.prev_chunk[i];
        }
      }
    }

    __builtin_prefetch(&worker.chunk[worker.pos1],1,3);

    // if (debugOpOrder && worker.op == sus_op) {
    //   printf("SIMD pre op %d:\n", worker.op);
    //   for (int i = 0; i < 256; i++) {
    //       printf("%02X ", worker.prev_chunk[i]);
    //   } 
    //   printf("\n");
    // }
    // fmt::printf("op: %d, ", worker.op);
    // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

    switch(worker.op) {
      case 0:
        // #pragma GCC unroll 16
        {
          // Load 32 bytes of worker.prev_chunk starting from i into an AVX2 256-bit register
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          __m256i pop = popcnt256_epi8(data);
          
          data = _mm256_xor_si256(data,pop);

          // Rotate left by 5
          data = _mm256_rol_epi8(data, 5);

          // Full 16-bit multiplication
          data = _mm256_mul_epi8(data, data);
          data = _mm256_rolv_epi8(data, data);

          // Write results to workerData

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        if ((worker.pos2-worker.pos1)%2 == 1) {
          worker.t1 = worker.chunk[worker.pos1];
          worker.t2 = worker.chunk[worker.pos2];
          worker.chunk[worker.pos1] = reverse8(worker.t2);
          worker.chunk[worker.pos2] = reverse8(worker.t1);
        }
        break;
      case 1:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          __m256i shift = _mm256_and_si256(data, vec_3);
          data = _mm256_sllv_epi8(data, shift);
          data = _mm256_rol_epi8(data,1);
          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.prev_chunk[worker.pos2]));
          data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 2:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          __m256i pop = popcnt256_epi8(data);
          data = _mm256_xor_si256(data,pop);
          data = _mm256_reverse_epi8(data);

          __m256i shift = _mm256_and_si256(data, vec_3);
          data = _mm256_sllv_epi8(data, shift);

          pop = popcnt256_epi8(data);
          data = _mm256_xor_si256(data,pop);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 3:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rolv_epi8(data,_mm256_add_epi8(data,vec_3));
          data = _mm256_xor_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_rol_epi8(data,1);
          
          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 4:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
          data = _mm256_rolv_epi8(data,data);
          data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 5:
        {
          // Load 32 bytes of worker.prev_chunk starting from i into an AVX2 256-bit register
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          __m256i pop = popcnt256_epi8(data);
          data = _mm256_xor_si256(data,pop);
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
          data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        
        break;
      case 6:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
          data = _mm256_rol_epi8(data, 3);
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          __m256i x = _mm256_xor_si256(data,_mm256_set1_epi8(97));
          data = _mm256_sub_epi8(data,x);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 7:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_add_epi8(data, data);;
          data = _mm256_rolv_epi8(data, data);

          __m256i pop = popcnt256_epi8(data);
          data = _mm256_xor_si256(data,pop);
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 8:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_rol_epi8(data,2);
          data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 9:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data,4));
          data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 10:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_mul_epi8(data, data);
          data = _mm256_rol_epi8(data, 3);
          data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 11:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 6);
          data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_rolv_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 12:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
          data = _mm256_mul_epi8(data, data);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 13:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 1);
          data = _mm256_xor_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
          data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 14:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
          data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
          data = _mm256_mul_epi8(data, data);
          data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 15:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
          data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 16:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_rol_epi8(data,4));
          data = _mm256_mul_epi8(data, data);
          data = _mm256_rol_epi8(data,1);
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 17:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_mul_epi8(data, data);
          data = _mm256_rol_epi8(data,5);
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 18:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
          data = _mm256_rol_epi8(data, 1);
          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 19:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));
          data = _mm256_rol_epi8(data, 5);
          data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
          data = _mm256_add_epi8(data, data);;;

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 20:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_reverse_epi8(data);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 21:

        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 1);
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_add_epi8(data, data);
          data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
    break;
      case 22:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_reverse_epi8(data);
          data = _mm256_mul_epi8(data,data);
          data = _mm256_rol_epi8(data,1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 23:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 4);
          data = _mm256_xor_si256(data,popcnt256_epi8(data));
          data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
      break;
      case 24:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_add_epi8(data, data);
          data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
          data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 25:
  #pragma GCC unroll 32
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.chunk[i] = worker.prev_chunk[i] ^ (byte)bitTable[worker.prev_chunk[i]];             // ones count bits
          worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
          worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
          worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 26:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_mul_epi8(data, data);
          data = _mm256_xor_si256(data,popcnt256_epi8(data));
          data = _mm256_add_epi8(data, data);
          data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 27:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 5);
          data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
          data = _mm256_rol_epi8(data, 5);
          
          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 28:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_add_epi8(data, data);
          data = _mm256_add_epi8(data, data);
          data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 29:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_mul_epi8(data, data);
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 30:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
          data = _mm256_rol_epi8(data, 5);
          data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 31:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;
        
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 32:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_reverse_epi8(data);
          data = _mm256_rol_epi8(data, 3);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 33:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rolv_epi8(data, data);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
          data = _mm256_reverse_epi8(data);
          data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 34:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
          data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 35:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_add_epi8(data, data);
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_rol_epi8(data, 1);
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 36:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, popcnt256_epi8(data));
          data = _mm256_rol_epi8(data, 1);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 37:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rolv_epi8(data, data);
          data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 38:
  #pragma GCC unroll 32
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.chunk[i] = worker.prev_chunk[i] >> (worker.prev_chunk[i] & 3);    // shift right
          worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
          worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
          worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 39:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 40:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rolv_epi8(data, data);
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_xor_si256(data, popcnt256_epi8(data));
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 41:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 5);
          data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
          data = _mm256_rol_epi8(data, 3);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 42:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 4);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_rolv_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 43:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_add_epi8(data, data);
          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 44:
  #pragma GCC unroll 32
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.chunk[i] = worker.prev_chunk[i] ^ (byte)bitTable[worker.prev_chunk[i]];             // ones count bits
          worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
          worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
          worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 45:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;
        
          data = _mm256_rol_epi8(data, 2);
          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_xor_si256(data, popcnt256_epi8(data));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 46:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;
        
          data = _mm256_xor_si256(data, popcnt256_epi8(data));
          data = _mm256_add_epi8(data, data);
          data = _mm256_rol_epi8(data, 5);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 47:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 5);
          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_rol_epi8(data, 5);
          data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 48:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rolv_epi8(data, data);
          data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 49:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;
        
          data = _mm256_xor_si256(data, popcnt256_epi8(data));
          data = _mm256_add_epi8(data, data);
          data = _mm256_reverse_epi8(data);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 50:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;
        
          data = _mm256_reverse_epi8(data);
          data = _mm256_rol_epi8(data, 3);
          data = _mm256_add_epi8(data, data);
          data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 51:
  #pragma GCC unroll 32
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.chunk[i] = worker.prev_chunk[i] ^ worker.chunk[worker.pos2];     // XOR
          worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
          worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
          worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 52:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rolv_epi8(data, data);
          data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_xor_si256(data, popcnt256_epi8(data));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 53:
  #pragma GCC unroll 32
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.chunk[i] = worker.prev_chunk[i]*2;                 // +
          worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
          worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
          worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                                // INSERT_RANDOM_CODE_END
        }
        break;
      case 54:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;
        
          data = _mm256_reverse_epi8(data);
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }

        break;
      case 55:
  #pragma GCC unroll 32
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.chunk[i] = reverse8(worker.prev_chunk[i]);      // reverse bits
          worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
          worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
          worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 56:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_mul_epi8(data, data);
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 57:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rolv_epi8(data, data);
          data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 58:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;
        
          data = _mm256_reverse_epi8(data);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }  
        break;
      case 59:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 1);
          data = _mm256_mul_epi8(data, data);
          data = _mm256_rolv_epi8(data, data);
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 60:
        {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_rol_epi8(data, 3);

            #ifdef _WIN32
              data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            #else
              data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            #endif
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }

        break;
      case 61:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 5);
          data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 62:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 63:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 5);
          data = _mm256_xor_si256(data, popcnt256_epi8(data));
          data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
          data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }

        break;
      case 64:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_reverse_epi8(data);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
          data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 65:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;


          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 66:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_reverse_epi8(data);
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
          data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 67:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 1);
          data = _mm256_xor_si256(data, popcnt256_epi8(data));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
          data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 68:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 69:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_add_epi8(data, data);
          data = _mm256_mul_epi8(data, data);
          data = _mm256_reverse_epi8(data);
          data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 70:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_mul_epi8(data, data);
          data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 71:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_rol_epi8(data, 5);
          data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
          data = _mm256_mul_epi8(data, data);
          data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 72:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_reverse_epi8(data);
          data = _mm256_xor_si256(data, popcnt256_epi8(data));
          data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 73:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_xor_si256(data, popcnt256_epi8(data));
          data = _mm256_reverse_epi8(data);
          data = _mm256_rol_epi8(data, 5);
          data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 74:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_mul_epi8(data, data);
          data = _mm256_rol_epi8(data, 3);
          data = _mm256_reverse_epi8(data);
          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
      case 75:
        {
          __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
          __m256i old = data;

          data = _mm256_mul_epi8(data, data);
          data = _mm256_xor_si256(data, popcnt256_epi8(data));
          data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
          data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
          _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        }
        break;
        case 76:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 77:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_add_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, popcnt256_epi8(data));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 78:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rolv_epi8(data, data);
            data = _mm256_reverse_epi8(data);
            data = _mm256_mul_epi8(data, data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 79:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_add_epi8(data, data);
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 80:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rolv_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_add_epi8(data, data);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 81:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, popcnt256_epi8(data));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 82:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 83:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_reverse_epi8(data);
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 84:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_rol_epi8(data, 1);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 85:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 86:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 87:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 88:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rol_epi8(data, 1);
            data = _mm256_mul_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 89:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_mul_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 90:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_reverse_epi8(data);
            data = _mm256_rol_epi8(data, 6);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 91:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, popcnt256_epi8(data));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 92:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, popcnt256_epi8(data));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_xor_si256(data, popcnt256_epi8(data));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 93:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 94:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 95:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_rol_epi8(data, 2);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 96:
    #pragma GCC unroll 32
          for (int i = worker.pos1; i < worker.pos2; i++)
          {
            // INSERT_RANDOM_CODE_START
            worker.chunk[i] = worker.prev_chunk[i] ^ rl8(worker.prev_chunk[i], 2);   // rotate  bits by 2
            worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
            worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
            worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                                  // INSERT_RANDOM_CODE_END
          }
          break;
        case 97:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, popcnt256_epi8(data));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 98:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 99:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_reverse_epi8(data);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 100:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rolv_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, popcnt256_epi8(data));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 101:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, popcnt256_epi8(data));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 102:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_add_epi8(data, data);
            data = _mm256_rol_epi8(data, 3);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 103:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rolv_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 104:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, popcnt256_epi8(data));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 105:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 106:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_rol_epi8(data, 1);
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 107:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rol_epi8(data, 6);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 108:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 109:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_mul_epi8(data, data);
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 110:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 111:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_mul_epi8(data, data);
            data = _mm256_reverse_epi8(data);
            data = _mm256_mul_epi8(data, data);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 112:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 113:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 6);
            data = _mm256_xor_si256(data, popcnt256_epi8(data));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 114:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_reverse_epi8(data);
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 115:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rolv_epi8(data, data);
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 3);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 116:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, popcnt256_epi8(data));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 117:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 118:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_add_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 119:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 120:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 121:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_add_epi8(data, data);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 122:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 123:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_rol_epi8(data, 6);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 124:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 125:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_add_epi8(data, data);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 126:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 127:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 128:
    #pragma GCC unroll 32
          for (int i = worker.pos1; i < worker.pos2; i++)
          {
            // INSERT_RANDOM_CODE_START
            worker.chunk[i] = rl8(worker.prev_chunk[i], worker.prev_chunk[i]); // rotate  bits by random
            worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
            worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
            worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
                                                                              // INSERT_RANDOM_CODE_END
          }
          break;
        case 129:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 130:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_rol_epi8(data, 1);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 131:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_rol_epi8(data, 1);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 132:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_reverse_epi8(data);
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 133:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 134:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_rol_epi8(data, 1);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 135:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_add_epi8(data, data);
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 136:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 137:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_reverse_epi8(data);
            data = _mm256_rolv_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 138:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_add_epi8(data, data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 139:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rol_epi8(data, 3);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 140:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 141:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 142:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 143:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 144:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rolv_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_rolv_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 145:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 146:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 147:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 148:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 149:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_reverse_epi8(data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 150:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 151:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 152:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 153:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 4);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 154:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 155:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 156:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rol_epi8(data, 4);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 157:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 158:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_add_epi8(data, data);
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 159:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 160:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_reverse_epi8(data);
            data = _mm256_rol_epi8(data, 4);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 161:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_rolv_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 162:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_mul_epi8(data, data);
            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 163:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 164:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_mul_epi8(data, data);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 165:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 166:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_add_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 167:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_mul_epi8(data, data);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 168:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rolv_epi8(data, data);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 169:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 170:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_reverse_epi8(data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 171:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 172:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 173:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 174:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_rolv_epi8(data, data);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 175:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 176:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 177:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 178:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_add_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 179:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_add_epi8(data, data);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 180:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 181:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 182:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 6);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 183:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 184:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 185:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 186:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 187:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_add_epi8(data, data);
            data = _mm256_rol_epi8(data, 3);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 188:
    #pragma GCC unroll 32
          for (int i = worker.pos1; i < worker.pos2; i++)
          {
            // INSERT_RANDOM_CODE_START
            worker.chunk[i] ^= rl8(worker.prev_chunk[i], 4);   // rotate  bits by 4
            worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
            worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
            worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                                  // INSERT_RANDOM_CODE_END
          }
          break;
        case 189:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 190:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 191:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 192:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_add_epi8(data, data);
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 193:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 194:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 195:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 196:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_reverse_epi8(data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 197:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_mul_epi8(data, data);
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 198:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_reverse_epi8(data);
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 199:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_add_epi8(data, data);
            data = _mm256_mul_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 200:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_reverse_epi8(data);
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 201:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 202:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 203:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 1);
            data = _mm256_rolv_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 204:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 205:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 206:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_reverse_epi8(data);
            data = _mm256_reverse_epi8(data);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 207:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 208:
    #pragma GCC unroll 32
          for (int i = worker.pos1; i < worker.pos2; i++)
          {
            // INSERT_RANDOM_CODE_START
            worker.chunk[i] = worker.prev_chunk[i]*2;                          // +
            worker.chunk[i] += worker.chunk[i];                          // +
            worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
            worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
                                                                          // INSERT_RANDOM_CODE_END
          }
          break;
        case 209:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_reverse_epi8(data);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 210:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 211:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_add_epi8(data, data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_rolv_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 212:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 213:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 214:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 215:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 216:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 217:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_add_epi8(data, data);
            data = _mm256_rol_epi8(data, 1);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 218:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 219:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 220:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_reverse_epi8(data);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 221:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 222:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_mul_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 223:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 224:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rol_epi8(data, 4);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 225:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_reverse_epi8(data);
            data = _mm256_rol_epi8(data, 3);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 226:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_reverse_epi8(data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 227:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 228:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_add_epi8(data, data);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 229:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 230:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_mul_epi8(data, data);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rolv_epi8(data, data);
            data = _mm256_rolv_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 231:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 3);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_reverse_epi8(data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 232:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_mul_epi8(data, data);
            data = _mm256_mul_epi8(data, data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 233:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 1);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_rol_epi8(data, 3);
            pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 234:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 235:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_mul_epi8(data, data);
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 236:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_add_epi8(data, data);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 237:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rol_epi8(data, 3);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 238:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_add_epi8(data, data);
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 239:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 6);
            data = _mm256_mul_epi8(data, data);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 240:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_add_epi8(data, data);
            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 241:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 242:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_add_epi8(data, data);
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 243:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_rol_epi8(data, 1);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 244:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_reverse_epi8(data);
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 245:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 246:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            data = _mm256_rol_epi8(data, 1);
            data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
            data = _mm256_add_epi8(data, data);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 247:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rol_epi8(data, 5);
            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 248:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
            data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_rol_epi8(data, 5);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 249:
    #pragma GCC unroll 32
          for (int i = worker.pos1; i < worker.pos2; i++)
          {
            // INSERT_RANDOM_CODE_START
            worker.chunk[i] = reverse8(worker.prev_chunk[i]);                    // reverse bits
            worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
            worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
            worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                              // INSERT_RANDOM_CODE_END
          }
          break;
        case 250:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
            data = _mm256_rolv_epi8(data, data);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 251:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_add_epi8(data, data);
            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 252:
          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            data = _mm256_reverse_epi8(data);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        case 253:
          {
            std::copy(&worker.prev_chunk[worker.pos1], &worker.prev_chunk[worker.pos2], &worker.chunk[worker.pos1]);
    #pragma GCC unroll 32
            for (int i = worker.pos1; i < worker.pos2; i++)
            {
              // INSERT_RANDOM_CODE_START
              worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
              worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
              worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
              worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
              // INSERT_RANDOM_CODE_END

              worker.prev_lhash = worker.lhash + worker.prev_lhash;
              worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
            }
            break;
          }
        case 254:
        case 255:
          RC4_set_key(&worker.key, 256, worker.prev_chunk);

          {
            __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
            __m256i old = data;

            __m256i pop = popcnt256_epi8(data);
            data = _mm256_xor_si256(data, pop);
            data = _mm256_rol_epi8(data, 3);
            data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
            data = _mm256_rol_epi8(data, 3);

          data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
            _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
          }
          break;
        default:
          break;
      }

    if(isTest) {
      break;
    }
  
    // if (op == 53) {
    //   std::cout << hexStr(worker.chunk, 256) << std::endl << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos1], 1) << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos2], 1) << std::endl;
    // }

    __builtin_prefetch(worker.chunk,0,3);
    // __builtin_prefetch(worker.chunk+64,0,3);
    // __builtin_prefetch(worker.chunk+128,0,3);
    __builtin_prefetch(worker.chunk+192,0,3);

    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    if (worker.A < 0x10)
    { // 6.25 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

      // uint64_t test = XXHash64::hash(worker.chunk, worker.pos2, 0);
      if (worker.op == sus_op && debugOpOrder) printf("SIMD: A: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x20)
    { // 12.5 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);

      // uint64_t test = hash_64_fnv1a(worker.chunk, worker.pos2);
      if (worker.op == sus_op && debugOpOrder) printf("SIMD: B: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x30)
    { // 18.75 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      HH_ALIGNAS(16)
      const highwayhash::HH_U64 key2[2] = {worker.tries, worker.prev_lhash};
      worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations

      // uint64_t test = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations
      if (worker.op == sus_op && debugOpOrder) printf("SIMD: C: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A <= 0x40)
    { // 25% probablility
      // if (debugOpOrder && worker.op == sus_op) {
      //   printf("SIMD: D: RC4 key:\n");
      //   for (int i = 0; i < 256; i++) {
      //     printf("%d, ", worker.key.data[i]);
      //   }
      // }
      RC4(&worker.key, 256, worker.chunk, worker.chunk);
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    // if (debugOpOrder && worker.op == sus_op) {
    //   printf("SIMD op %d result:\n", worker.op);
    //   for (int i = 0; i < 256; i++) {
    //       printf("%02X ", worker.chunk[i]);
    //   } 
    //   printf("\n");
    // }

    // memcpy(&worker.sData[(worker.tries - 1) * 256], worker.chunk, 256);
    
    // std::copy(worker.chunk, worker.chunk + 256, &worker.sData[(worker.tries - 1) * 256]);

    // memcpy(&worker->data.data()[(worker.tries - 1) * 256], worker.chunk, 256);

    // std::cout << hexStr(worker.chunk, 256) << std::endl;

    if (worker.tries > 260 + 16 || (worker.sData[(worker.tries-1)*256+255] >= 0xf0 && worker.tries > 260))
    {
      break;
    }
  }
  worker.data_len = static_cast<uint32_t>((worker.tries - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
}

void branchComputeCPU_avx2_zOptimized(workerData &worker, bool isTest)
{
  byte prevOp;
  // int sameCount = 0;

  int repeatCounter = 0;
  byte lastChar = 0;

  int maxRepeat = 0;
  int reducedChars = 0;

  #pragma unroll 64
  for (int it = 0; it < 277; it++) 
  {
  // while(true) {
    if(isTest) {
      // __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
      // __m256i cmp = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(worker.prev_chunk[worker.pos1]));
      // uint32_t mask = (1 << (worker.pos2 - worker.pos1)) - 1;
      // int result = _mm256_movemask_epi8(cmp);

      // if ((result & mask) == mask) {
      //   worker.isSame = true;
      // }
    } else {
      worker.tries++;
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries;
      // __builtin_prefetch(&worker.random_switcher,0,3);
      // printf("%d worker.random_switcher %d %08jx\n", worker.tries, worker.random_switcher, worker.random_switcher);

      prevOp = worker.op;
      worker.op = static_cast<byte>(worker.random_switcher);
      // if (debugOpOrder) worker.opsA.push_back(worker.op);

      // printf("op: %d\n", worker.op);

      byte p1 = static_cast<byte>(worker.random_switcher >> 8);
      byte p2 = static_cast<byte>(worker.random_switcher >> 16);

      if (p1 > p2)
      {
        std::swap(p1, p2);
      }

      if (p2 - p1 > 32)
      {
        p2 = p1 + ((p2 - p1) & 0x1f);
      }

      if (p1 < worker.pos1 || p2 > worker.pos2) worker.isSame = false;

      worker.pos1 = p1;
      worker.pos2 = p2;

      worker.chunk = &worker.sData[(worker.tries - 1) * 256];

      if (worker.tries == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[(worker.tries - 2) * 256];

        __builtin_prefetch(worker.prev_chunk,0,3);
        __builtin_prefetch(worker.prev_chunk+64,0,3);
        __builtin_prefetch(worker.prev_chunk+128,0,3);
        __builtin_prefetch(worker.prev_chunk+192,0,3);

        // Calculate the start and end blocks
        int start_block = 0;
        int end_block = worker.pos1 / 16;

        // Copy the blocks before worker.pos1
        for (int i = start_block; i < end_block; i++) {
            __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
            _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
        }

        // Copy the remaining bytes before worker.pos1
        for (int i = end_block * 16; i < worker.pos1; i++) {
            worker.chunk[i] = worker.prev_chunk[i];
        }

        // Calculate the start and end blocks
        start_block = (worker.pos2 + 15) / 16;
        end_block = 16;

        // Copy the blocks after worker.pos2
        for (int i = start_block; i < end_block; i++) {
            __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
            _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
        }

        // Copy the remaining bytes after worker.pos2
        for (int i = worker.pos2; i < start_block * 16; i++) {
          worker.chunk[i] = worker.prev_chunk[i];
        }
      }
    }

    __builtin_prefetch(&worker.chunk[worker.pos1],1,3);

    // if (debugOpOrder && worker.op == sus_op) {
    //   printf("SIMD pre op %d:\n", worker.op);
    //   for (int i = 0; i < 256; i++) {
    //       printf("%02X ", worker.prev_chunk[i]);
    //   } 
    //   printf("\n");
    // }
    // fmt::printf("op: %d, ", worker.op);
    // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

    // if (worker.op > astro_branched_zOp::branchComputeSize) return;
    // if (!isTest) {
    //   printf("op: %d, Input State: ", worker.op);
    //   for (int i = worker.pos1; i < worker.pos2; i++) {
    //     printf("%02X ", worker.prev_chunk[i]);
    //   }
    //   printf(", same: %d", worker.isSame);
    //   printf("\n");
    // }

    worker.data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    worker.old = worker.data;
    if (!isTest && !worker.isSame) {
      __m256i cmp = _mm256_cmpeq_epi8(worker.data, _mm256_set1_epi8(worker.prev_chunk[worker.pos1]));
      uint32_t mask = (1 << (worker.pos2 - worker.pos1)) - 1;
      int result = _mm256_movemask_epi8(cmp);

      if ((result & mask) == mask) {
        worker.isSame = true;
        // if (worker.pos2 - worker.pos1 > 0) printf("same after op: %d, for op: %d\n", prevOp, worker.op);
      }
    } else {
      // sameCount++;
    }

    astro_branched_zOp::branchCompute[worker.op + (256*worker.isSame)](worker);

    if(isTest) {
      break;
    }
  
    // if (op == 53) {
    //   std::cout << hexStr(worker.chunk, 256) << std::endl << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos1], 1) << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos2], 1) << std::endl;
    // }

    __builtin_prefetch(worker.chunk,0,3);
    // __builtin_prefetch(worker.chunk+64,0,3);
    // __builtin_prefetch(worker.chunk+128,0,3);
    __builtin_prefetch(worker.chunk+192,0,3);

    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    if (worker.A < 0x10)
    { // 6.25 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

      // uint64_t test = XXHash64::hash(worker.chunk, worker.pos2, 0);
      if (worker.op == sus_op && debugOpOrder) printf("SIMD: A: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x20)
    { // 12.5 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);

      // uint64_t test = hash_64_fnv1a(worker.chunk, worker.pos2);
      if (worker.op == sus_op && debugOpOrder) printf("SIMD: B: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x30)
    { // 18.75 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      HH_ALIGNAS(16)
      const highwayhash::HH_U64 key2[2] = {worker.tries, worker.prev_lhash};
      worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations

      // uint64_t test = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations
      if (worker.op == sus_op && debugOpOrder) printf("SIMD: C: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A <= 0x40)
    { // 25% probablility
      // if (debugOpOrder && worker.op == sus_op) {
      //   printf("SIMD: D: RC4 key:\n");
      //   for (int i = 0; i < 256; i++) {
      //     printf("%d, ", worker.key.data[i]);
      //   }
      // }
      RC4(&worker.key, 256, worker.chunk, worker.chunk);
      worker.isSame = false;
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    // if (debugOpOrder && worker.op == sus_op) {
    //   printf("SIMD op %d result:\n", worker.op);
    //   for (int i = 0; i < 256; i++) {
    //       printf("%02X ", worker.chunk[i]);
    //   } 
    //   printf("\n");
    // }

    // memcpy(&worker.sData[(worker.tries - 1) * 256], worker.chunk, 256);
    
    // std::copy(worker.chunk, worker.chunk + 256, &worker.sData[(worker.tries - 1) * 256]);

    // memcpy(&worker->data.data()[(worker.tries - 1) * 256], worker.chunk, 256);

    // std::cout << hexStr(worker.chunk, 256) << std::endl;

    // if (worker.tries == 1) lastChar = worker.sData[0];
    // for (int i = 0; i < 256; i++) {
    //   if ((worker.tries > 1 || i > 0) && worker.chunk[i] != lastChar) {
    //     lastChar = worker.chunk[i];
    //     maxRepeat = std::max(maxRepeat, repeatCounter);
    //     repeatCounter = 1;
    //     reducedChars++;
    //   } else {
    //     repeatCounter++;
    //   }
    // }

    if (worker.tries > 260 + 16 || (worker.sData[(worker.tries-1)*256+255] >= 0xf0 && worker.tries > 260))
    {
      break;
    }
  }

  // if (!isTest) printf("longest repeated length: %d\n", maxRepeat);
  // if (!isTest) printf("%d out of %d ops had repeating char inputs\n", sameCount, worker.tries);
  worker.data_len = static_cast<uint32_t>((worker.tries - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
  // if (!isTest) printf("dataLen vs reducedChars: %d / %d\n", worker.data_len, reducedChars);
}

#endif

// Compute the new values for worker.chunk using layered lookup tables instead of
// branched computational operations

void lookupCompute(workerData &worker, bool isTest)
{
  while (true)
  {
    if(isTest) {

    } else {
      worker.tries++;
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries;
      // printf("%d worker.random_switcher %d %08jx\n", worker.tries, worker.random_switcher, worker.random_switcher);

      worker.op = static_cast<byte>(worker.random_switcher);
      if (debugOpOrder) worker.opsB.push_back(worker.op);

      // printf("op: %d\n", worker.op);

      worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
      worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

      // __builtin_prefetch(worker.chunk + worker.pos1, 0, 1);
      // __builtin_prefetch(worker.maskTable, 0, 0);

      if (worker.pos1 > worker.pos2)
      {
        std::swap(worker.pos1, worker.pos2);
      }

      if (worker.pos2 - worker.pos1 > 32)
      {
        worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
      }

      // int otherpos = std::find(branchedOps.begin(), branchedOps.end(), worker.op) == branchedOps.end() ? 0 : worker.chunk[worker.pos2];
      // __builtin_prefetch(&worker.chunk[worker.pos1], 0, 0);
      // __builtin_prefetch(&worker.lookup[lookupIndex(worker.op,0,otherpos)]);
      worker.chunk = &worker.sData[(worker.tries - 1) * 256];
      if (worker.tries == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[(worker.tries - 2) * 256];

        #if defined(__AVX2__)
          // Calculate the start and end blocks
          int start_block = 0;
          int end_block = worker.pos1 / 16;

          // Copy the blocks before worker.pos1
          for (int i = start_block; i < end_block; i++) {
              __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
              _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
          }

          // Copy the remaining bytes before worker.pos1
          for (int i = end_block * 16; i < worker.pos1; i++) {
              worker.chunk[i] = worker.prev_chunk[i];
          }

          // Calculate the start and end blocks
          start_block = (worker.pos2 + 15) / 16;
          end_block = 16;

          // Copy the blocks after worker.pos2
          for (int i = start_block; i < end_block; i++) {
              __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
              _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
          }

          // Copy the remaining bytes after worker.pos2
          for (int i = worker.pos2; i < start_block * 16; i++) {
            worker.chunk[i] = worker.prev_chunk[i];
          }
        #else
          memcpy(worker.chunk, worker.prev_chunk, 256);
        #endif
      }

      if (debugOpOrder && worker.op == sus_op) {
        printf("Lookup pre op %d, pos1: %d, pos2: %d::\n", worker.op, worker.pos1, worker.pos2);
        for (int i = 0; i < 256; i++) {
            printf("%02X ", worker.prev_chunk[i]);
        } 
        printf("\n");
      }
    }
    // fmt::printf("op: %d, ", worker.op);
    // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

    // printf("index: %d\n", lookupIndex(op, worker.chunk[worker.pos1], worker.chunk[worker.pos2]));

    if (worker.op == 253) {
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        worker.chunk[i] = worker.prev_chunk[i];
      }
      for (int i = worker.pos1; i < worker.pos2; i++)
      {

        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        // INSERT_RANDOM_CODE_END

        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
      }
      goto after;
    }
    if (worker.op >= 254) {
      RC4_set_key(&worker.key, 256,  worker.prev_chunk);
    }
    {
      bool use2D = std::find(worker.branchedOps, worker.branchedOps + branchedOps_size, worker.op) == worker.branchedOps + branchedOps_size;
      uint16_t *lookup2D = use2D ? &worker.lookup2D[0] : nullptr;
      byte *lookup3D = use2D ? nullptr : &worker.lookup3D[0];

      int firstIndex;
      __builtin_prefetch(&worker.prev_chunk[worker.pos1],0,3);
      __builtin_prefetch(&worker.prev_chunk[worker.pos1]+192,0,3);

      if (use2D) {
        firstIndex = worker.reg_idx[worker.op]*(256*256);
        int n = 0;

        // Manually unrolled loops for repetetive efficiency. Worst possible loop count for 2D
        // lookups is now 4, with less than 4 being pretty common.

        //TODO: ask AI if assignment would be faster below

        // Groups of 8
        for (int i = worker.pos1; i < worker.pos2-7; i += 8) {
          __builtin_prefetch(&lookup2D[firstIndex + 256*n++],0,3);
          uint32_t val1 = (lookup2D[(firstIndex + (worker.prev_chunk[i+1] << 8)) | worker.prev_chunk[i]]) |
            (lookup2D[(firstIndex + (worker.prev_chunk[i+3] << 8)) | worker.prev_chunk[i+2]] << 16);
          uint32_t val2 =(lookup2D[(firstIndex + (worker.prev_chunk[i+5] << 8)) | worker.prev_chunk[i+4]]) |
            (lookup2D[(firstIndex + (worker.prev_chunk[i+7] << 8)) | worker.prev_chunk[i+6]] << 16);

          *(uint64_t*)&worker.chunk[i] = val1 | ((uint64_t)val2 << 32);
        }

        // Groups of 4
        for (int i = worker.pos2-((worker.pos2-worker.pos1)%8); i < worker.pos2-3; i += 4) {
          __builtin_prefetch(&lookup2D[firstIndex + 256*n++],0,3);
          uint32_t val = lookup2D[(firstIndex + (worker.prev_chunk[i+1] << 8)) | worker.prev_chunk[i]] |
            (lookup2D[(firstIndex + (worker.prev_chunk[i+3] << 8)) | worker.prev_chunk[i+2]] << 16);
          *(uint32_t*)&worker.chunk[i] = val;
        }

        // Groups of 2
        for (int i = worker.pos2-((worker.pos2-worker.pos1)%4); i < worker.pos2-1; i += 2) {
          __builtin_prefetch(&lookup2D[firstIndex + 256*n++],0,3);
          uint16_t val = lookup2D[(firstIndex + (worker.prev_chunk[i+1] << 8)) | worker.prev_chunk[i]];
          *(uint16_t*)&worker.chunk[i] = val;
        }

        // Last if odd
        if ((worker.pos2-worker.pos1)%2 != 0) {
          uint16_t val = lookup2D[firstIndex + (worker.prev_chunk[worker.pos2-1] << 8)];
          worker.chunk[worker.pos2-1] = (val & 0xFF00) >> 8;
        }
      } else {
        firstIndex = worker.branched_idx[worker.op]*256*256 + worker.chunk[worker.pos2]*256;
        int n = 0;

        // Manually unrolled loops for repetetive efficiency. Worst possible loop count for 3D
        // lookups is now 4, with less than 4 being pretty common.

        // Groups of 16
        for(int i = worker.pos1; i < worker.pos2-15; i += 16) {
          __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
          worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
          worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
          worker.chunk[i+2] = lookup3D[firstIndex + worker.prev_chunk[i+2]];
          worker.chunk[i+3] = lookup3D[firstIndex + worker.prev_chunk[i+3]];
          worker.chunk[i+4] = lookup3D[firstIndex + worker.prev_chunk[i+4]];
          worker.chunk[i+5] = lookup3D[firstIndex + worker.prev_chunk[i+5]];
          worker.chunk[i+6] = lookup3D[firstIndex + worker.prev_chunk[i+6]];
          worker.chunk[i+7] = lookup3D[firstIndex + worker.prev_chunk[i+7]];

          worker.chunk[i+8] = lookup3D[firstIndex + worker.prev_chunk[i+8]];
          worker.chunk[i+9] = lookup3D[firstIndex + worker.prev_chunk[i+9]];
          worker.chunk[i+10] = lookup3D[firstIndex + worker.prev_chunk[i+10]];
          worker.chunk[i+11] = lookup3D[firstIndex + worker.prev_chunk[i+11]];
          worker.chunk[i+12] = lookup3D[firstIndex + worker.prev_chunk[i+12]];
          worker.chunk[i+13] = lookup3D[firstIndex + worker.prev_chunk[i+13]];
          worker.chunk[i+14] = lookup3D[firstIndex + worker.prev_chunk[i+14]];
          worker.chunk[i+15] = lookup3D[firstIndex + worker.prev_chunk[i+15]];
        }

        // Groups of 8
        for(int i = worker.pos2-((worker.pos2-worker.pos1)%16); i < worker.pos2-7; i += 8) {
          __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
          worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
          worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
          worker.chunk[i+2] = lookup3D[firstIndex + worker.prev_chunk[i+2]];
          worker.chunk[i+3] = lookup3D[firstIndex + worker.prev_chunk[i+3]];
          worker.chunk[i+4] = lookup3D[firstIndex + worker.prev_chunk[i+4]];
          worker.chunk[i+5] = lookup3D[firstIndex + worker.prev_chunk[i+5]];
          worker.chunk[i+6] = lookup3D[firstIndex + worker.prev_chunk[i+6]];
          worker.chunk[i+7] = lookup3D[firstIndex + worker.prev_chunk[i+7]];
        }

        // Groups of 4
        for(int i = worker.pos2-((worker.pos2-worker.pos1)%8); i < worker.pos2-3; i+= 4) {
          __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
          worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
          worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
          worker.chunk[i+2] = lookup3D[firstIndex + worker.prev_chunk[i+2]];
          worker.chunk[i+3] = lookup3D[firstIndex + worker.prev_chunk[i+3]];
        }

        // Groups of 2
        for(int i = worker.pos2-((worker.pos2-worker.pos1)%4); i < worker.pos2-1; i+= 2) {
          __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
          worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
          worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
        }

        // Last if odd
        if ((worker.pos2-worker.pos1)%2 != 0) {
          worker.chunk[worker.pos2-1] = lookup3D[firstIndex + worker.prev_chunk[worker.pos2-1]];
        }
      }
      if (worker.op == 0) {
        if ((worker.pos2-worker.pos1)%2 == 1) {
          worker.t1 = worker.chunk[worker.pos1];
          worker.t2 = worker.chunk[worker.pos2];
          worker.chunk[worker.pos1] = reverse8(worker.t2);
          worker.chunk[worker.pos2] = reverse8(worker.t1);
        }
      }
    }

after:

    if(isTest) {
      break;
    }
    // if (op == 53) {
    //   std::cout << hexStr(worker.chunk, 256) << std::endl << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos1], 1) << std::endl;
    //   std::cout << hexStr(&worker.chunk[worker.pos2], 1) << std::endl;
    // }

    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    if (worker.A < 0x10)
    { // 6.25 % probability
      // __builtin_prefetch(worker.chunk);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

      // uint64_t test = XXHash64::hash(worker.chunk, worker.pos2, 0);
      if (worker.op == sus_op && debugOpOrder)  printf("Lookup: A: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x20)
    { // 12.5 % probability
      // __builtin_prefetch(worker.chunk);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);

      // uint64_t test = hash_64_fnv1a(worker.chunk, worker.pos2);
      if (worker.op == sus_op && debugOpOrder)  printf("Lookup: B: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x30)
    { // 18.75 % probability
      // std::copy(worker.chunk, worker.chunk + worker.pos2, s3);
      // __builtin_prefetch(worker.chunk);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      HH_ALIGNAS(16)
      const highwayhash::HH_U64 key2[2] = {worker.tries, worker.prev_lhash};
      worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations

      // uint64_t test = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations
      if (worker.op == sus_op && debugOpOrder)  printf("Lookup: C: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A <= 0x40)
    { // 25% probablility
      // if (worker.op == sus_op && debugOpOrder) {
      //   printf("Lookup: D: RC4 key:\n");
      //   for (int i = 0; i < 256; i++) {
      //     printf("%d, ", worker.key.data[i]);
      //   }
      // }
      // prefetch(worker.chunk, 0, 1);
      RC4(&worker.key, 256, worker.chunk,  worker.chunk);
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    if (debugOpOrder && worker.op == sus_op) {
      printf("Lookup op %d result:\n", worker.op);
      for (int i = 0; i < 256; i++) {
          printf("%02X ", worker.chunk[i]);
      } 
      printf("\n");
    }

    // memcpy(&worker.sData[(worker.tries - 1) * 256], worker.chunk, 256);
    
    // std::copy(worker.chunk, worker.chunk + 256, &worker.sData[(worker.tries - 1) * 256]);

    // memcpy(&worker->data.data()[(worker.tries - 1) * 256], worker.chunk, 256);

    // std::cout << hexStr(worker.chunk, 256) << std::endl;

    if (worker.tries > 260 + 16 || (worker.sData[(worker.tries-1)*256+255] >= 0xf0 && worker.tries > 260))
    {
      break;
    }
  }
  worker.data_len = static_cast<uint32_t>((worker.tries - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
}


