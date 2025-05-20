#include <endian.hpp>
#include <inttypes.h>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

#include <boost/thread.hpp>
#include <algorithm>
#include <bitset>
#include <iostream>
#include <fstream>
#include <thread>

#include <fnv1a.h>
#include <xxhash64.h>
#include "astrobwtv3.h"
#include "tnn-hugepages.h"
#include "astrotest.hpp"
// #include "branched_AVX2.h"

#include <unordered_map>
#include <array>
#include <algorithm>

#if defined(__x86_64__)
  #include <xmmintrin.h>
#endif
#if defined(__aarch64__)
  #include "astro_aarch64.hpp"
#endif

#include <random>
#include <chrono>

#include <Salsa20.h>
#include <sodium.h>

// #include <alcp/digest.h>

#include <highwayhash/sip_hash.h>
#include <filesystem>
#include <functional>
#include "lookupcompute.h"
#if defined(USE_ASTRO_SPSA)
  #include "spsa.hpp"
#else
  #define MINPREFLEN 4
#endif

extern "C"
{
  #include "divsufsort_private.h"
  #include "divsufsort.h"
}

#include <utility>

#include <hex.h>
#include <openssl/rc4.h>

#include <fstream>

#include <bit>
// #include <libcubwt.cuh>
// #include <device_sa.cuh>
#include <lookup.h>
// #include <sacak-lcp.h>
#if defined(__x86_64__)
  #include <immintrin.h>
  #include <emmintrin.h>
#endif

using byte = unsigned char;

int ops[256];

uint16_t lookup2D[regOps_size*256*256];
byte lookup3D[branchedOps_size*256*256];

std::vector<byte> opsA;
std::vector<byte> opsB;

bool debugOpOrder = false;

void (*astroCompFunc)(workerData &worker, bool isTest, int wIndex) = wolfCompute;

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

// template <std::size_t N>
// inline void generateRandomBytesForTune(std::uint8_t (&iv_buff)[N])
// {
//   auto const hes = std::random_device{}();

//   using random_bytes_engine = std::independent_bits_engine<std::default_random_engine,
//                                                            CHAR_BIT, unsigned short>;

//   random_bytes_engine rbe;
//   rbe.seed(hes);

//   std::generate(std::begin(iv_buff), std::end(iv_buff), std::ref(rbe));
// }

// bool setAstroAlgo(std::string desiredAlgo) {
//   bool toRet = false;
//   for(int x = 0; x < numAstroFuncs; x++) {
//     if(desiredAlgo.compare(allAstroFuncs[x].funcName) == 0) {
//       printf("Setting AstroBWTv3 override: %s\n", allAstroFuncs[x].funcName.c_str());
//       astroCompFunc = allAstroFuncs[x].funcPtr;
//       toRet = true;
//       break;
//     }
//   }
//   if(!toRet) {
//     printf("Unrecognized AstroBWTv3 algo: %s\nAllowed options are: ", desiredAlgo.c_str());
//     for(int x = 0; x < numAstroFuncs; x++) {
//       printf("%s ", allAstroFuncs[x].funcName.c_str());
//     }
//     printf("\n");
//   }
//   return toRet;
// }

// void astroTune(int num_threads, int tuneWarmupSec, int tuneDurationSec) {
//   int64_t tuneWarmupMs = tuneWarmupSec * 1000;
//   int64_t tuneDurationMs = tuneDurationSec * 1000;

//   int totalTuneTime = numAstroFuncs * (tuneWarmupSec + tuneDurationSec);
//   printf("Tuning %zu AstroBWTv3 algos for %d seconds in total\n", numAstroFuncs, totalTuneTime);
//   fflush(stdout);

//   boost::mutex durLock;
//   std::vector<int64_t> durations[numAstroFuncs];
  
//   boost::mutex hashLock;
//   int64_t numHashes[numAstroFuncs];
//   for(int x = 0; x < numAstroFuncs; x++) {
//     numHashes[x] = 0;
//   }

//   int fastestCompIdx = 0;
//   void (*fastestComp)(workerData &worker, bool isTest, int wIndex) = branchComputeCPU;

//   try {
//     byte random_buffer[48];
//     generateRandomBytesForTune<48>(random_buffer);
//     byte res[32];

//     boost::thread tune_threads[num_threads];
//     for (int x = 0; x < numAstroFuncs; x++)
//     {
//       astroCompFunc = allAstroFuncs[x].funcPtr;

//       // Start each thread with an inline lambda function
//       for (int i = 0; i < num_threads; ++i) {
//         tune_threads[i] = boost::thread([&]() {
//           int tid = i;
//           workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
//           initWorker(*worker);
//           lookupGen(*worker, nullptr, nullptr);

//           auto warmupStart = std::chrono::steady_clock::now();
//           for(;;) {
//             AstroBWTv3(random_buffer, 48, res, *worker, false);
//             if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - warmupStart).count() > tuneWarmupMs) {
//               break;
//             }
//           }

//           int hashes = 0;
//           auto start = std::chrono::steady_clock::now();
//           for(;;) {
//             AstroBWTv3(random_buffer, 48, res, *worker, false);
//             hashes++;
//             if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() > tuneDurationMs) {
//               break;
//             }
//           }
//           hashLock.lock();
//           numHashes[x] += hashes;
//           hashLock.unlock();
//         });
//       }
//       // Wait for all threads to finish
//       for (int i = 0; i < num_threads; ++i) {
//           tune_threads[i].join();
//       }
//       printf("%s: %2.3f   ", allAstroFuncs[x].funcName.c_str(), (double)numHashes[x] / (double)tuneDurationMs);
//     }
//   } catch (const std::exception& e) {
//       std::cerr << "Exception: " << e.what() << "\n";
//   }

//   astroCompFunc = allAstroFuncs[0].funcPtr;
//   int64_t mostHashes = numHashes[0];
//   for (int x = 0; x < numAstroFuncs; x++)
//   {
//     // printf("%s = %d\n", astroCompFuncNames.data()[x].c_str(), numHashes[x]);
//     if (numHashes[x] > mostHashes)
//     {
//       astroCompFunc = allAstroFuncs[x].funcPtr;
//       mostHashes = numHashes[x];
//       fastestCompIdx = x;
//     }
//   }

//   printf("\nUsing %s\n", allAstroFuncs[fastestCompIdx].funcName.c_str());
// }

void hashSHA256(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize)
{
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, input, inputSize);
  SHA256_Final(digest, &sha256);
}

TNN_TARGETS
void AstroBWTv3(byte *input, int inputLen, byte *outputhash, workerData &worker, bool unused)
{
  try
  {
    constexpr int i = 0;
    memset(worker.sData + ASTRO_SCRATCH_SIZE*i + 256, 0, 64);

    __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 256], 1, 0);
    __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 256+64], 1, 0);
    __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 256+128], 1, 0);
    __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 256+192], 1, 0);
    
    hashSHA256(worker.sha256, &input[i*inputLen], &worker.sData[ASTRO_SCRATCH_SIZE*i + 320], inputLen);

    __builtin_prefetch(worker.sData + ASTRO_SCRATCH_SIZE*i, 1, 3);
    __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 64], 1, 3);
    __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 128], 1, 3);
    __builtin_prefetch(&worker.sData[ASTRO_SCRATCH_SIZE*i + 192], 1, 3);

    constexpr byte salsaInput[256] = {0};
    crypto_stream_salsa20_xor(
        &worker.sData[ASTRO_SCRATCH_SIZE*i],         // output
        salsaInput,                                  // input
        256,                                          // length
        &worker.sData[ASTRO_SCRATCH_SIZE*i + 256],   // 8-byte nonce/IV
        &worker.sData[ASTRO_SCRATCH_SIZE*i + 320]    // 32-byte key
    );

    __builtin_prefetch(&worker.key[i] + 8, 1, 2);
    __builtin_prefetch(&worker.key[i] + 8+64, 1, 2);
    __builtin_prefetch(&worker.key[i] + 8+128, 1, 2);
    __builtin_prefetch(&worker.key[i] + 8+192, 1, 2);

    RC4_set_key(&worker.key[i], 256,  &worker.sData[ASTRO_SCRATCH_SIZE*i]);
    RC4(&worker.key[i], 256, &worker.sData[ASTRO_SCRATCH_SIZE*i], &worker.sData[ASTRO_SCRATCH_SIZE*i]);

    worker.lhash = hash_64_fnv1a_256(&worker.sData[ASTRO_SCRATCH_SIZE*i]);
    worker.prev_lhash = worker.lhash;

    worker.tries[i] = 0;
    //worker.isSame = false;

    astroCompFunc(worker, false, i);

    #if defined(USE_ASTRO_SPSA)
      SPSA(&worker.sData[i * ASTRO_SCRATCH_SIZE], worker.data_len, worker);
      // byte *B = reinterpret_cast<byte *>(worker.sa);
      // hashSHA256(worker.sha256, B, (outputhash + 32*i), worker.data_len*4);
      memcpy(outputhash, worker.padding, 32);
    #else
      divsufsort(&worker.sData[i * ASTRO_SCRATCH_SIZE], worker.sa, worker.data_len, worker.bA, worker.bB);
      byte *B = reinterpret_cast<byte *>(worker.sa);
      hashSHA256(worker.sha256, B, (outputhash + 32*i), worker.data_len*4);
    #endif
  }
  catch (const std::exception &ex)
  {
    // recover(outputhash);
    std::cerr << ex.what() << std::endl;
  }
}

// SIMD chunk copy

#if defined(__x86_64__)
__attribute__ ((target("avx512f")))
// // Copy prev_chunk between start -> end to chunk (inclusive)
inline void copyChunkData(workerData &worker, uint8_t start, uint8_t end) {
  for (int i = start; i + 63 < end; i += 64) {
    __m512i prev_data = _mm512_loadu_si512((__m512i*)&worker.prev_chunk[i]);
    _mm512_storeu_si512((__m512i*)&worker.chunk[i], prev_data);
  }
}

__attribute__ ((target("avx2")))
// Copy prev_chunk between start -> end to chunk (inclusive)
void copyChunkData(workerData &worker, int start, int end) {
  for (int i = start; i < end; i += 32) {
    __m256i prev_data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[i]);
    _mm256_storeu_si256((__m256i*)&worker.chunk[i], prev_data);
  }
}
__attribute__ ((target("sse2")))
// Copy prev_chunk between start -> end to chunk (inclusive)
void copyChunkData(workerData &worker, int start, int end) {
  for (int i = start; i < end; i += 16) {
    __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i]);
    _mm_storeu_si128((__m128i*)&worker.chunk[i], prev_data);
  }
}
__attribute__ ((target("default")))
#endif

// Copy prev_chunk between start -> end to chunk (inclusive)
void copyChunkData(workerData &worker, int start, int end) {
  std::copy_n(&worker.prev_chunk[start], end - start, &worker.chunk[start]);
}

// WOLF CODE

void wolfCompute(workerData &worker, bool isTest, int wIndex)
{
  byte prevOp;
  int changeCount = 0;

  worker.templateIdx = 0;
  uint8_t chunkCount = 1;
  int firstChunk = 0;

  uint8_t lp1 = 0;
  uint8_t lp2 = 255;

  for (int it = 0; it < 278; ++it)
  {
      // TODO prefetch next chunk into L2
      worker.tries[wIndex]++;
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];

      prevOp = worker.op;
      worker.op = static_cast<byte>(worker.random_switcher);

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

      if (worker.tries[wIndex] > 0) {
        lp1 = std::min(lp1, p1);
        lp2 = std::max(lp2, p2);
      }

      worker.pos1 = p1;
      worker.pos2 = p2;

      worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];

      if (worker.tries[wIndex] == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];
        __builtin_prefetch(worker.chunk+p1,1,1);

        memcpy(worker.chunk, worker.prev_chunk, 256);
      }
    // }

    // TODO: Make below in all SIMD variants in a function, using FMV for architecture-agnostic calling
    // if FMV causes slowdown from overhead, use a live cached dispatch similar to wolfPermute

    if (worker.op == 253)
    {
      for (int i = worker.pos1; i < worker.pos2; i++)
      {

        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
        worker.chunk[i] ^= worker.prev_chunk[worker.pos2];     // XOR
        worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
        // INSERT_RANDOM_CODE_END

        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
      }

      goto after;
    }
    if (worker.op >= 254) {
      RC4_set_key(&worker.key[wIndex], 256,  worker.prev_chunk);
    }
    wolfPerms[0](worker.prev_chunk, worker.chunk, worker.op, worker.pos1, worker.pos2, worker);

    if (!worker.op) {
      if ((worker.pos2-worker.pos1)%2 == 1) {
        worker.t1 = worker.chunk[worker.pos1];
        worker.t2 = worker.chunk[worker.pos2];
        worker.chunk[worker.pos1] = reverse8(worker.t2);
        worker.chunk[worker.pos2] = reverse8(worker.t1);
        //worker.isSame = false;
      }
    }

after:
    uint8_t pushPos1 = lp1;
    uint8_t pushPos2 = lp2;

    if (worker.pos1 == worker.pos2) {
      pushPos1 = -1;
      pushPos2 = -1;
    }

    byte workerA = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    workerA = (256 + (workerA % 256)) % 256;

    if (workerA < 0x10)
    { // 6.25 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

      #ifdef DEBUG_OP_ORDER
      if (worker.op == sus_op && debugOpOrder)  printf("Wolf: A: new worker.lhash: %08jx\n", worker.lhash);
      #endif
    }

    if (workerA < 0x20)
    { // 12.5 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);

      #ifdef DEBUG_OP_ORDER
      if (worker.op == sus_op && debugOpOrder)  printf("Wolf: B: new worker.lhash: %08jx\n", worker.lhash);
      #endif
    }

    if (workerA < 0x30)
    { // 18.75 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      HH_ALIGNAS(16)
      const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
      worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations

      #ifdef DEBUG_OP_ORDER
      if (worker.op == sus_op && debugOpOrder)  printf("Wolf: C: new worker.lhash: %08jx\n", worker.lhash);
      #endif
    }

    if (workerA <= 0x40)
    { // 25% probablility
      RC4(&worker.key[wIndex], 256, worker.chunk,  worker.chunk);
      //worker.isSame = false;
      if (255 - pushPos2 < MINPREFLEN)
        pushPos2 = 255;
      if (pushPos1 < MINPREFLEN)
        pushPos1 = 0;


      if (pushPos1 == 255) pushPos1 = 0;
      
      worker.astroTemplate[worker.templateIdx] = templateMarker{
        (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
        (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
        (uint16_t)0,
        (uint16_t)0,
        (uint16_t)((firstChunk << 7) | chunkCount)
      };

      pushPos1 = 0;
      pushPos2 = 255;
      worker.templateIdx += (worker.tries[wIndex] > 1);
      firstChunk = worker.tries[wIndex]-1;
      lp1 = 255;
      lp2 = 0;
      chunkCount = 1;
    } else {
      chunkCount++;
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    if (255 - pushPos2 < MINPREFLEN)
      pushPos2 = 255;
    if (pushPos1 < MINPREFLEN)
      pushPos1 = 0;

    #ifdef DEBUG_OP_ORDER
    if (debugOpOrder && worker.op == sus_op) {
      printf("Wolf op %d result:\n", worker.op);
      for (int i = 0; i < 256; i++) {
        printf("%02X ", worker.chunk[i]);
      } 
      printf("\n");
    }
    #endif

    if (worker.tries[wIndex] > 260 + 16 || (worker.sData[(worker.tries[wIndex]-1)*256+255] >= 0xf0 && worker.tries[wIndex] > 260))
    {
      break;
    }
  }

  if (chunkCount > 0) {
    if (255 - lp2 < MINPREFLEN)
      lp2 = 255;
    if (lp1 < MINPREFLEN)
      lp1 = 0;
    worker.astroTemplate[worker.templateIdx] = templateMarker{
      (uint8_t)(chunkCount > 1 ? lp1 : 0),
      (uint8_t)(chunkCount > 1 ? lp2 : 255),
      (uint16_t)0,
      (uint16_t)0,
      (uint16_t)((firstChunk << 7) | chunkCount)
    };

    worker.templateIdx++;
  }

  // printf("%dc\n", changeCount);
  worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
}

// void decompressChunks(workerData &worker, int wIndex) {
//   // First pass: Copy all base chunks to their positions in sData
//   for (int tIdx = 0; tIdx < worker.templateIdx; tIdx++) {
//       const templateMarker& tpl = worker.astroTemplate[tIdx];
      
//       // Extract chunk count and first chunk from posData
//       uint16_t chunkCount = tpl.posData & 0x7F;
//       uint16_t firstChunk = tpl.posData >> 7;
      
//       // Get base chunk pointer
//       uint8_t* baseChunk = &worker.baseChunks()[tpl.baseIdx * ASTRO_CHUNK_SIZE];
//       uint8_t* destBase = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + firstChunk * 256];
      
//       // Prefetch the next template's base chunk
//       if (tIdx + 1 < worker.templateIdx) {
//           __builtin_prefetch(&worker.baseChunks()[worker.astroTemplate[tIdx + 1].baseIdx * ASTRO_CHUNK_SIZE], 0, 1);
//       }
      
//       // Copy base chunk to all chunks in this stamp (SIMD-optimized memcpy)
//       for (int c = 0; c < chunkCount; c++) {
//           memcpy(destBase + c * 256, baseChunk, 256);
//       }
//   }
  
//   // Second pass: Apply deltas
//   for (int tIdx = 0; tIdx < worker.templateIdx; tIdx++) {
//       const templateMarker& tpl = worker.astroTemplate[tIdx];
      
//       uint16_t chunkCount = tpl.posData & 0x7F;
//       uint16_t firstChunk = tpl.posData >> 7;
      
//       // Skip if only one chunk (no deltas)
//       if (chunkCount <= 1) continue;
      
//       // Prefetch delta data
//       __builtin_prefetch(&worker.chunkDeltas[tpl.deltaOffset], 0, 1);
      
//       // Apply deltas for each chunk after the first
//       for (int c = 1; c < chunkCount; c++) {
//           uint8_t* targetChunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + 
//                                                (firstChunk + c) * 256];
//           uint32_t deltaIdx = tpl.deltaOffset + (c - 1) * MAX_DELTA_SIZE;
          
//           // Prefetch next delta
//           if (c + 1 < chunkCount) {
//               __builtin_prefetch(&worker.chunkDeltas[deltaIdx + MAX_DELTA_SIZE], 0, 1);
//           }
          
//           // Apply modified bytes
//           int deltaSize = tpl.p2 - tpl.p1 + 1;
//           for (int i = 0; i < deltaSize; i++) {
//               targetChunk[tpl.p1 + i] = worker.chunkDeltas[deltaIdx + i];
//           }
          
//           // Apply byte 255
//           targetChunk[255] = worker.chunkDeltas[deltaIdx + deltaSize];
//       }
//   }
// }

// void wolfCompute_compressed(workerData &worker, bool isTest, int wIndex)
// {
//   byte prevOp;
//   worker.templateIdx = 0;
//   uint8_t chunkCount = 1;
//   int firstChunk = 0;

//   uint8_t lp1 = 255;
//   uint8_t lp2 = 0;

//   // Track indices for compressed data storage
//   uint16_t currentBaseIdx = 0;
//   uint32_t currentDeltaOffset = 0;
  
//   // Initialize the first base chunk in baseChunks[0]
//   memcpy(&worker.baseChunks()[0], worker.sData, 256);
  
//   // Start with prev_chunk at baseChunk[0] and chunk at baseChunk[1]
//   worker.prev_chunk = &worker.baseChunks()[currentBaseIdx * ASTRO_CHUNK_SIZE];
//   worker.chunk = &worker.baseChunks()[(currentBaseIdx * ASTRO_CHUNK_SIZE) + 256];

//   for (int it = 0; it < 278; ++it)
//   {
//       worker.tries[wIndex]++;
//       worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];

//       prevOp = worker.op;
//       worker.op = static_cast<byte>(worker.random_switcher);

//       byte p1 = static_cast<byte>(worker.random_switcher >> 8);
//       byte p2 = static_cast<byte>(worker.random_switcher >> 16);

//       // Branchless swap
//       byte shouldSwap = p1 > p2;
//       byte temp = p1;
//       p1 = shouldSwap ? p2 : p1;
//       p2 = shouldSwap ? temp : p2;

//       // Simplified clipping
//       if (p2 - p1 > 32)
//       {
//         p2 = p1 + ((p2 - p1) & 0x1f);
//       }

//       // Combined min/max update
//       if (worker.tries[wIndex] > 0) {
//         lp1 = (p1 < lp1) ? p1 : lp1;
//         lp2 = (p2 > lp2) ? p2 : lp2;
//       }

//       worker.pos1 = p1;
//       worker.pos2 = p2;

//       // Copy prev_chunk to chunk (maintains modifications)
//       memcpy(worker.chunk, worker.prev_chunk, 256);
      
//       // Prefetch hints
//       __builtin_prefetch(worker.chunk + p1, 1, 1);
//       __builtin_prefetch(worker.prev_chunk + p2, 0, 1);

//       // Apply operations exactly like the original
//       if (worker.op == 253)
//       {
//         for (int i = worker.pos1; i < worker.pos2; i++)
//         {
//           worker.chunk[i] = rl8(worker.chunk[i], 3);
//           worker.chunk[i] ^= rl8(worker.chunk[i], 2);
//           worker.chunk[i] ^= worker.prev_chunk[worker.pos2];
//           worker.chunk[i] = rl8(worker.chunk[i], 3);

//           worker.prev_lhash = worker.lhash + worker.prev_lhash;
//           worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
//         }

//         goto after;
//       }
      
//       if (worker.op >= 254) {
//         RC4_set_key(&worker.key[wIndex], 256, worker.prev_chunk);
//       }
//       wolfPerms[0](worker.prev_chunk, worker.chunk, worker.op, worker.pos1, worker.pos2, worker);

//       if (!worker.op) {
//         if ((worker.pos2-worker.pos1) & 1) { // Simpler odd check
//           worker.t1 = worker.chunk[worker.pos1];
//           worker.t2 = worker.chunk[worker.pos2];
//           worker.chunk[worker.pos1] = reverse8(worker.t2);
//           worker.chunk[worker.pos2] = reverse8(worker.t1);
//           worker.isSame = false;
//         }
//       }

// after:
//       // Combined position update
//       uint8_t pushPos1 = (worker.pos1 == worker.pos2) ? 255 : lp1;
//       uint8_t pushPos2 = (worker.pos1 == worker.pos2) ? 0 : lp2;

//       // Simplified modulus operation (eliminate redundant 256 operations)
//       int diff = worker.chunk[worker.pos1] - worker.chunk[worker.pos2];
//       worker.A = ((diff % 256) + 256) & 0xFF;

//       // Cascade through hash operations with early exit
//       if (worker.A < 0x30) { // Common case optimization
//         worker.prev_lhash = worker.lhash + worker.prev_lhash;
        
//         if (worker.A < 0x10) {
//           worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
//         } else if (worker.A < 0x20) {
//           worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);
//         } else {
//           HH_ALIGNAS(16)
//           const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
//           worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2);
//         }
//       }

//       if (worker.A <= 0x40)
//       { // Start of new stamp
//         RC4(&worker.key[wIndex], 256, worker.chunk, worker.chunk);
//         worker.isSame = false;
        
//         // Record the compression data for the previous stamp
//         if (worker.templateIdx < ASTRO_MAX_CHUNKS && worker.tries[wIndex] > 1) {
//           // Direct template initialization (avoid temporary)
//           templateMarker& tpl = worker.astroTemplate[worker.templateIdx];
//           tpl.p1 = (chunkCount > 1) ? lp1 : 0;
//           tpl.p2 = (chunkCount > 1) ? lp2 : 255;
//           tpl.keySpotA = 0;
//           tpl.keySpotB = 0;
//           tpl.posData = (firstChunk << 7) | chunkCount;
//           tpl.baseIdx = currentBaseIdx;
//           tpl.deltaOffset = currentDeltaOffset;
          
//           worker.templateIdx++;
//           currentBaseIdx++;
//           currentDeltaOffset += (chunkCount > 1) ? (chunkCount - 1) * MAX_DELTA_SIZE : 0;
//         }
        
//         // Start new stamp
//         firstChunk = worker.tries[wIndex] - 1;
//         lp1 = 255;
//         lp2 = 0;
//         chunkCount = 1;
        
//         // Move to next chunk (chunk becomes the new base)
//         worker.prev_chunk = worker.chunk;
//         worker.chunk += 256;
//       } else {
//         // Store delta for this chunk (optimized)
//         if (chunkCount > 1) {
//           uint32_t deltaIdx = currentDeltaOffset + (chunkCount - 2) * MAX_DELTA_SIZE;
          
//           // Check bounds once
//           if (deltaIdx + (lp2 - lp1 + 2) < sizeof(worker.chunkDeltas)) {
//             // Use memcpy for contiguous range (often optimized)
//             if (lp2 >= lp1 && lp2 < 255) {  // Sanity check
//               memcpy(&worker.chunkDeltas[deltaIdx], &worker.chunk[lp1], lp2 - lp1 + 1);
//               worker.chunkDeltas[deltaIdx + (lp2 - lp1 + 1)] = worker.chunk[255];
//             }
//           }
//         }
//         chunkCount++;
//       }

//       // Final byte update
//       worker.chunk[255] ^= worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

//       // Combined exit condition
//       if (worker.tries[wIndex] > 276 || 
//          (worker.chunk[255] >= 0xf0 && worker.tries[wIndex] > 260))
//       {
//         break;
//       }
//   }

//   // Handle the final stamp (simplified)
//   if (chunkCount > 0 && worker.templateIdx < ASTRO_MAX_CHUNKS) {
//     // Only adjust bounds if needed
//     if (255 - lp2 < MINPREFLEN) lp2 = 255;
//     if (lp1 < MINPREFLEN) lp1 = 0;
    
//     // Direct initialization
//     templateMarker& tpl = worker.astroTemplate[worker.templateIdx];
//     tpl.p1 = (chunkCount > 1) ? lp1 : 0;
//     tpl.p2 = (chunkCount > 1) ? lp2 : 255;
//     tpl.keySpotA = 0;
//     tpl.keySpotB = 0;
//     tpl.posData = (firstChunk << 7) | chunkCount;
//     tpl.baseIdx = currentBaseIdx;
//     tpl.deltaOffset = currentDeltaOffset;
    
//     worker.templateIdx++;
//   }

//   // Simplified data length calculation
//   worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + 
//     (((static_cast<uint64_t>(worker.chunk[253]) << 8) | 
//       static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
// }

// Compute the new values for worker.chunk using layered lookup tables instead of
// branched computational operations
// void lookupCompute(workerData &worker, bool isTest, int wIndex)
// {
//   worker.templateIdx = 0;
//   uint8_t chunkCount = 1;
//   int firstChunk = 0;

//   uint8_t lp1 = 0;
//   uint8_t lp2 = 255;
//   while (true)
//   {
//     if(isTest) {

//     } else {
//       worker.tries[wIndex]++;
//       worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];
//       // printf("%d worker.random_switcher %d %08jx\n", worker.tries[wIndex], worker.random_switcher, worker.random_switcher);

//       worker.op = static_cast<byte>(worker.random_switcher);
//       #ifdef DEBUG_OP_ORDER
//       if (debugOpOrder) worker.opsB.push_back(worker.op);
//       #endif

//       // printf("op: %d\n", worker.op);

//       worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
//       worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

//       // __builtin_prefetch(worker.chunk + worker.pos1, 0, 1);
//       // __builtin_prefetch(worker.maskTable, 0, 0);

//       if (worker.pos1 > worker.pos2)
//       {
//         std::swap(worker.pos1, worker.pos2);
//       }

//       if (worker.pos2 - worker.pos1 > 32)
//       {
//         worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
//       }

//       if (worker.tries[wIndex] > 0) {
//         lp1 = std::min(lp1, worker.pos1);
//         lp2 = std::max(lp2, worker.pos2);
//       }

//       // int otherpos = std::find(branchedOps.begin(), branchedOps.end(), worker.op) == branchedOps.end() ? 0 : worker.chunk[worker.pos2];
//       // __builtin_prefetch(&worker.chunk[worker.pos1], 0, 0);
//       // __builtin_prefetch(&worker.lookup[lookupIndex(worker.op,0,otherpos)]);
//       worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];
//       if (worker.tries[wIndex] == 1) {
//         worker.prev_chunk = worker.chunk;
//       } else {
//         worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];

//         #if defined(__AVX2__)
//           // Calculate the start and end blocks
//           int start_block = 0;
//           int end_block = worker.pos1 / 16;

//           // Copy the blocks before worker.pos1
//           for (int i = start_block; i < end_block; i++) {
//               __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
//               _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
//           }

//           // Copy the remaining bytes before worker.pos1
//           for (int i = end_block * 16; i < worker.pos1; i++) {
//               worker.chunk[i] = worker.prev_chunk[i];
//           }

//           // Calculate the start and end blocks
//           start_block = (worker.pos2 + 15) / 16;
//           end_block = 16;

//           // Copy the blocks after worker.pos2
//           for (int i = start_block; i < end_block; i++) {
//               __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
//               _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
//           }

//           // Copy the remaining bytes after worker.pos2
//           for (int i = worker.pos2; i < start_block * 16; i++) {
//             worker.chunk[i] = worker.prev_chunk[i];
//           }
//         #else
//           memcpy(worker.chunk, worker.prev_chunk, 256);
//         #endif
//       }

//       #ifdef DEBUG_OP_ORDER
//       if (debugOpOrder && worker.op == sus_op) {
//         printf("Lookup pre op %d, pos1: %d, pos2: %d::\n", worker.op, worker.pos1, worker.pos2);
//         for (int i = 0; i < 256; i++) {
//             printf("%02X ", worker.prev_chunk[i]);
//         } 
//         printf("\n");
//       }
//       #endif
//     }
//     // fmt::printf("op: %d, ", worker.op);
//     // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

//     // printf("index: %d\n", lookupIndex(op, worker.chunk[worker.pos1], worker.chunk[worker.pos2]));

//     if (worker.op == 253) {
// #pragma GCC unroll 32
//       for (int i = worker.pos1; i < worker.pos2; i++)
//       {
//         worker.chunk[i] = worker.prev_chunk[i];
//       }
//       for (int i = worker.pos1; i < worker.pos2; i++)
//       {

//         // INSERT_RANDOM_CODE_START
//         worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
//         worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
//         worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
//         worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
//         // INSERT_RANDOM_CODE_END

//         worker.prev_lhash = worker.lhash + worker.prev_lhash;
//         worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
//       }
//       goto after;
//     }
//     if (worker.op >= 254) {
//       RC4_set_key(&worker.key[wIndex], 256,  worker.prev_chunk);
//     }
//     {
//       bool use2D = std::find(worker.branchedOps, worker.branchedOps + branchedOps_size, worker.op) == worker.branchedOps + branchedOps_size;
//       uint16_t *lookup2D = use2D ? &worker.lookup2D[0] : nullptr;
//       byte *lookup3D = use2D ? nullptr : &worker.lookup3D[0];

//       int firstIndex;
//       __builtin_prefetch(&worker.prev_chunk[worker.pos1],0,3);
//       __builtin_prefetch(&worker.prev_chunk[worker.pos1]+192,0,3);

//       if (use2D) {
//         firstIndex = worker.reg_idx[worker.op]*(256*256);
//         int n = 0;

//         // Manually unrolled loops for repetetive efficiency. Worst possible loop count for 2D
//         // lookups is now 4, with less than 4 being pretty common.

//         //TODO: ask AI if assignment would be faster below

//         // Groups of 8
//         for (int i = worker.pos1; i < worker.pos2-7; i += 8) {
//           __builtin_prefetch(&lookup2D[firstIndex + 256*n++],0,3);
//           uint32_t val1 = (lookup2D[(firstIndex + (worker.prev_chunk[i+1] << 8)) | worker.prev_chunk[i]]) |
//             (lookup2D[(firstIndex + (worker.prev_chunk[i+3] << 8)) | worker.prev_chunk[i+2]] << 16);
//           uint32_t val2 =(lookup2D[(firstIndex + (worker.prev_chunk[i+5] << 8)) | worker.prev_chunk[i+4]]) |
//             (lookup2D[(firstIndex + (worker.prev_chunk[i+7] << 8)) | worker.prev_chunk[i+6]] << 16);

//           *(uint64_t*)&worker.chunk[i] = val1 | ((uint64_t)val2 << 32);
//         }

//         // Groups of 4
//         for (int i = worker.pos2-((worker.pos2-worker.pos1)%8); i < worker.pos2-3; i += 4) {
//           __builtin_prefetch(&lookup2D[firstIndex + 256*n++],0,3);
//           uint32_t val = lookup2D[(firstIndex + (worker.prev_chunk[i+1] << 8)) | worker.prev_chunk[i]] |
//             (lookup2D[(firstIndex + (worker.prev_chunk[i+3] << 8)) | worker.prev_chunk[i+2]] << 16);
//           *(uint32_t*)&worker.chunk[i] = val;
//         }

//         // Groups of 2
//         for (int i = worker.pos2-((worker.pos2-worker.pos1)%4); i < worker.pos2-1; i += 2) {
//           __builtin_prefetch(&lookup2D[firstIndex + 256*n++],0,3);
//           uint16_t val = lookup2D[(firstIndex + (worker.prev_chunk[i+1] << 8)) | worker.prev_chunk[i]];
//           *(uint16_t*)&worker.chunk[i] = val;
//         }

//         // Last if odd
//         if ((worker.pos2-worker.pos1)%2 != 0) {
//           uint16_t val = lookup2D[firstIndex + (worker.prev_chunk[worker.pos2-1] << 8)];
//           worker.chunk[worker.pos2-1] = (val & 0xFF00) >> 8;
//         }
//       } else {
//         firstIndex = worker.branched_idx[worker.op]*256*256 + worker.chunk[worker.pos2]*256;
//         int n = 0;

//         // Manually unrolled loops for repetetive efficiency. Worst possible loop count for 3D
//         // lookups is now 4, with less than 4 being pretty common.

//         // Groups of 16
//         for(int i = worker.pos1; i < worker.pos2-15; i += 16) {
//           __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
//           worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
//           worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
//           worker.chunk[i+2] = lookup3D[firstIndex + worker.prev_chunk[i+2]];
//           worker.chunk[i+3] = lookup3D[firstIndex + worker.prev_chunk[i+3]];
//           worker.chunk[i+4] = lookup3D[firstIndex + worker.prev_chunk[i+4]];
//           worker.chunk[i+5] = lookup3D[firstIndex + worker.prev_chunk[i+5]];
//           worker.chunk[i+6] = lookup3D[firstIndex + worker.prev_chunk[i+6]];
//           worker.chunk[i+7] = lookup3D[firstIndex + worker.prev_chunk[i+7]];

//           worker.chunk[i+8] = lookup3D[firstIndex + worker.prev_chunk[i+8]];
//           worker.chunk[i+9] = lookup3D[firstIndex + worker.prev_chunk[i+9]];
//           worker.chunk[i+10] = lookup3D[firstIndex + worker.prev_chunk[i+10]];
//           worker.chunk[i+11] = lookup3D[firstIndex + worker.prev_chunk[i+11]];
//           worker.chunk[i+12] = lookup3D[firstIndex + worker.prev_chunk[i+12]];
//           worker.chunk[i+13] = lookup3D[firstIndex + worker.prev_chunk[i+13]];
//           worker.chunk[i+14] = lookup3D[firstIndex + worker.prev_chunk[i+14]];
//           worker.chunk[i+15] = lookup3D[firstIndex + worker.prev_chunk[i+15]];
//         }

//         // Groups of 8
//         for(int i = worker.pos2-((worker.pos2-worker.pos1)%16); i < worker.pos2-7; i += 8) {
//           __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
//           worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
//           worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
//           worker.chunk[i+2] = lookup3D[firstIndex + worker.prev_chunk[i+2]];
//           worker.chunk[i+3] = lookup3D[firstIndex + worker.prev_chunk[i+3]];
//           worker.chunk[i+4] = lookup3D[firstIndex + worker.prev_chunk[i+4]];
//           worker.chunk[i+5] = lookup3D[firstIndex + worker.prev_chunk[i+5]];
//           worker.chunk[i+6] = lookup3D[firstIndex + worker.prev_chunk[i+6]];
//           worker.chunk[i+7] = lookup3D[firstIndex + worker.prev_chunk[i+7]];
//         }

//         // Groups of 4
//         for(int i = worker.pos2-((worker.pos2-worker.pos1)%8); i < worker.pos2-3; i+= 4) {
//           __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
//           worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
//           worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
//           worker.chunk[i+2] = lookup3D[firstIndex + worker.prev_chunk[i+2]];
//           worker.chunk[i+3] = lookup3D[firstIndex + worker.prev_chunk[i+3]];
//         }

//         // Groups of 2
//         for(int i = worker.pos2-((worker.pos2-worker.pos1)%4); i < worker.pos2-1; i+= 2) {
//           __builtin_prefetch(&lookup3D[firstIndex + 64*n++],0,3);
//           worker.chunk[i] = lookup3D[firstIndex + worker.prev_chunk[i]];
//           worker.chunk[i+1] = lookup3D[firstIndex + worker.prev_chunk[i+1]];
//         }

//         // Last if odd
//         if ((worker.pos2-worker.pos1)%2 != 0) {
//           worker.chunk[worker.pos2-1] = lookup3D[firstIndex + worker.prev_chunk[worker.pos2-1]];
//         }
//       }
//       if (worker.op == 0) {
//         if ((worker.pos2-worker.pos1)%2 == 1) {
//           worker.t1 = worker.chunk[worker.pos1];
//           worker.t2 = worker.chunk[worker.pos2];
//           worker.chunk[worker.pos1] = reverse8(worker.t2);
//           worker.chunk[worker.pos2] = reverse8(worker.t1);
//         }
//       }
//     }

// after:

//     if(isTest) {
//       break;
//     }
//     // if (op == 53) {
//     //   std::cout << hexStr(worker.chunk, 256) << std::endl << std::endl;
//     //   std::cout << hexStr(&worker.chunk[worker.pos1], 1) << std::endl;
//     //   std::cout << hexStr(&worker.chunk[worker.pos2], 1) << std::endl;
//     // }

//     uint8_t pushPos1 = lp1;
//     uint8_t pushPos2 = lp2;

//     if (worker.pos1 == worker.pos2) {
//       pushPos1 = -1;
//       pushPos2 = -1;
//     }

//     worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
//     worker.A = (256 + (worker.A % 256)) % 256;

//     if (worker.A < 0x10)
//     { // 6.25 % probability
//       // __builtin_prefetch(worker.chunk);
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);

//       // uint64_t test = XXHash64::hash(worker.chunk, worker.pos2, 0);
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder)  printf("Lookup: A: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A < 0x20)
//     { // 12.5 % probability
//       // __builtin_prefetch(worker.chunk);
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);

//       // uint64_t test = hash_64_fnv1a(worker.chunk, worker.pos2);
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder)  printf("Lookup: B: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A < 0x30)
//     { // 18.75 % probability
//       // std::copy(worker.chunk, worker.chunk + worker.pos2, s3);
//       // __builtin_prefetch(worker.chunk);
//       worker.prev_lhash = worker.lhash + worker.prev_lhash;
//       HH_ALIGNAS(16)
//       const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
//       worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations

//       // uint64_t test = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations
//       #ifdef DEBUG_OP_ORDER
//       if (worker.op == sus_op && debugOpOrder)  printf("Lookup: C: new worker.lhash: %08jx\n", worker.lhash);
//       #endif
//     }

//     if (worker.A <= 0x40)
//     { // 25% probablility
//       // if (worker.op == sus_op && debugOpOrder) {
//       //   printf("Lookup: D: RC4 key:\n");
//       //   for (int i = 0; i < 256; i++) {
//       //     printf("%d, ", worker.key.data[i]);
//       //   }
//       // }
//       // prefetch(worker.chunk, 0, 1);
//       RC4(&worker.key[wIndex], 256, worker.chunk,  worker.chunk);
//       if (255 - pushPos2 < MINPREFLEN)
//         pushPos2 = 255;
//       if (pushPos1 < MINPREFLEN)
//         pushPos1 = 0;


//       if (pushPos1 == 255) pushPos1 = 0;
      
//       worker.astroTemplate[worker.templateIdx] = templateMarker{
//         (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
//         (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
//         (uint16_t)0,
//         (uint16_t)0,
//         (uint16_t)((firstChunk << 7) | chunkCount)
//       };

//       pushPos1 = 0;
//       pushPos2 = 255;
//       worker.templateIdx += (worker.tries[wIndex] > 1);
//       firstChunk = worker.tries[wIndex]-1;
//       lp1 = 255;
//       lp2 = 0;
//       chunkCount = 1;
//     } else {
//       chunkCount++;
//     }

//     worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

//     if (255 - pushPos2 < MINPREFLEN)
//       pushPos2 = 255;
//     if (pushPos1 < MINPREFLEN)
//       pushPos1 = 0;

//     #ifdef DEBUG_OP_ORDER
//     if (debugOpOrder && worker.op == sus_op) {
//       printf("Lookup op %d result:\n", worker.op);
//       for (int i = 0; i < 256; i++) {
//           printf("%02X ", worker.chunk[i]);
//       } 
//       printf("\n");
//     }
//     #endif

//     // memcpy(&worker.sData[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);
    
//     // std::copy(worker.chunk, worker.chunk + 256, &worker.sData[(worker.tries[wIndex] - 1) * 256]);

//     // memcpy(&worker->data.data()[(worker.tries[wIndex] - 1) * 256], worker.chunk, 256);

//     // std::cout << hexStr(worker.chunk, 256) << std::endl;

//     if (worker.tries[wIndex] > 260 + 16 || (worker.sData[(worker.tries[wIndex]-1)*256+255] >= 0xf0 && worker.tries[wIndex] > 260))
//     {
//       break;
//     }
//   }

//   if (chunkCount > 0) {
//     if (255 - lp2 < MINPREFLEN)
//       lp2 = 255;
//     if (lp1 < MINPREFLEN)
//       lp1 = 0;
//     worker.astroTemplate[worker.templateIdx] = templateMarker{
//       (uint8_t)(chunkCount > 1 ? lp1 : 0),
//       (uint8_t)(chunkCount > 1 ? lp2 : 255),
//       (uint16_t)0,
//       (uint16_t)0,
//       (uint16_t)((firstChunk << 7) | chunkCount)
//     };
//     worker.templateIdx++;
//   }

//   worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
// }


