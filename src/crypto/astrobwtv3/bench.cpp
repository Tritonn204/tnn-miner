#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
// Include all your production headers
#include "astroworker.h"
#include "astrobwtv3.h"
#include <fnv1a.h>
#include <xxhash64.h>
#include <Salsa20.h>
#include <openssl/sha.h>
#include <openssl/rc4.h>
// ... other includes as needed

// Add this near the top of your bench.cpp file
#ifdef _WIN32
    #include <malloc.h>
    #define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #include <stdlib.h>
    #define aligned_free(ptr) free(ptr)
#endif

AstroFunc allAstroFuncs[] = {};
bool printHugepagesError = true;
size_t numAstroFuncs;

// Function prototypes
void wolfCompute(workerData &worker, bool isTest, int wIndex);
void wolfCompute_optimized(workerData &worker, bool isTest, int wIndex);
void wolfCompute_compressed(workerData &worker, bool isTest, int wIndex);

// Benchmark parameters
constexpr int NUM_ITERATIONS = 100000;
constexpr int WARMUP_ITERATIONS = 2000;
constexpr int INPUT_LEN = 80;  // Typical input length

inline void _hashSHA256(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize)
{
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, input, inputSize);
  SHA256_Final(digest, &sha256);
}

// Initialize worker data using the same process as AstroBWTv3
void initializeWorkerDataInPlace(workerData& worker, uint8_t* input, int inputLen) {
  uint8_t scratch[384] = {0};
  
  SHA256_Init(&worker.sha256);
  _hashSHA256(worker.sha256, input, &scratch[320], inputLen);
  worker.salsa20.setKey(&scratch[320]);
  worker.salsa20.setIv(&scratch[256]);
  worker.salsa20.processBytes(worker.salsaInput, scratch, 256);
  
  RC4_set_key(&worker.key[0], 256, scratch);
  RC4(&worker.key[0], 256, scratch, scratch);
  
  worker.lhash = hash_64_fnv1a_256(scratch);
  worker.prev_lhash = worker.lhash;
  worker.tries[0] = 0;
  worker.isSame = false;
  
  memcpy(worker.sData, scratch, 256);
}

int main() {
    // Allocate aligned worker data structures
    workerData* worker = (workerData*)aligned_alloc(64, sizeof(workerData));
    
    // Create random input data
    uint8_t input[INPUT_LEN];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        // Warmup v1 original
        for (int j = 0; j < INPUT_LEN; j++) {
          input[j] = static_cast<uint8_t>(dis(gen));
        }
        initializeWorkerDataInPlace(*worker, input, INPUT_LEN);
        wolfCompute(*worker, false, 0);
        
        // Warmup v1 optimized
        for (int j = 0; j < INPUT_LEN; j++) {
          input[j] = static_cast<uint8_t>(dis(gen));
        }
        initializeWorkerDataInPlace(*worker, input, INPUT_LEN);
        wolfCompute_optimized(*worker, false, 0);
        
        // Warmup compressed version
        for (int j = 0; j < INPUT_LEN; j++) {
          input[j] = static_cast<uint8_t>(dis(gen));
        }
        initializeWorkerDataInPlace(*worker, input, INPUT_LEN);
        wolfCompute_compressed(*worker, false, 0);
    }
    
    std::cout << "Starting benchmark..." << std::endl;
    
    // Benchmark v1 original
    auto start_v1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < INPUT_LEN; j++) {
          input[j] = static_cast<uint8_t>(dis(gen));
        }
        initializeWorkerDataInPlace(*worker, input, INPUT_LEN);
        wolfCompute(*worker, false, 0);
    }
    auto end_v1 = std::chrono::high_resolution_clock::now();
    
    // Benchmark v1 optimized
    auto start_v1_opt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < INPUT_LEN; j++) {
          input[j] = static_cast<uint8_t>(dis(gen));
        }
        initializeWorkerDataInPlace(*worker, input, INPUT_LEN);
        wolfCompute_optimized(*worker, false, 0);
    }
    auto end_v1_opt = std::chrono::high_resolution_clock::now();
    
    // Benchmark compressed version
    auto start_compressed = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < INPUT_LEN; j++) {
          input[j] = static_cast<uint8_t>(dis(gen));
        }
        initializeWorkerDataInPlace(*worker, input, INPUT_LEN);
        wolfCompute_compressed(*worker, false, 0);
    }
    auto end_compressed = std::chrono::high_resolution_clock::now();
    
    // Calculate durations
    auto duration_v1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_v1 - start_v1);
    auto duration_v1_opt = std::chrono::duration_cast<std::chrono::nanoseconds>(end_v1_opt - start_v1_opt);
    auto duration_compressed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_compressed - start_compressed);
    
    double avg_v1 = duration_v1.count() / (double)NUM_ITERATIONS;
    double avg_v1_opt = duration_v1_opt.count() / (double)NUM_ITERATIONS;
    double avg_compressed = duration_compressed.count() / (double)NUM_ITERATIONS;
    
    double speedup_v1_opt = avg_v1 / avg_v1_opt;
    double speedup_compressed = avg_v1 / avg_compressed;
    
    // Results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nBenchmark Results (" << NUM_ITERATIONS << " iterations):" << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "wolfCompute:           " << std::setw(10) << avg_v1/1000.0 << " μs/iteration (baseline)" << std::endl;
    std::cout << "wolfCompute_optimized: " << std::setw(10) << avg_v1_opt/1000.0 << " μs/iteration (" 
              << speedup_v1_opt << "x faster)" << std::endl;
    std::cout << "wolfCompute (compressed): " << std::setw(10) << avg_compressed/1000.0 << " μs/iteration (" 
              << speedup_compressed << "x faster)" << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    
    // Performance comparison
    std::cout << "\nPerformance Comparison:" << std::endl;
    std::cout << "optimized vs original: " << std::setw(6) << ((avg_v1 - avg_v1_opt)/avg_v1 * 100) << "% improvement" << std::endl;
    std::cout << "compressed vs original:   " << std::setw(6) << ((avg_v1 - avg_compressed)/avg_v1 * 100) << "% improvement" << std::endl;
    std::cout << "compressed vs optimized: " << std::setw(4) << ((avg_v1_opt > avg_compressed) ? 
              ((avg_v1_opt - avg_compressed)/avg_v1_opt * 100) : 
              -((avg_compressed - avg_v1_opt)/avg_compressed * 100)) 
              << "% " << ((avg_v1_opt > avg_compressed) ? "improvement" : "slower") << std::endl;
    
    // Timing breakdown
    std::cout << "\nTiming Breakdown:" << std::endl;
    std::cout << "Total original time:           " << std::setw(10) << duration_v1.count()/1000000.0 << " ms" << std::endl;
    std::cout << "Total optimized time: " << std::setw(10) << duration_v1_opt.count()/1000000.0 << " ms" << std::endl;
    std::cout << "Total compressed time:   " << std::setw(10) << duration_compressed.count()/1000000.0 << " ms" << std::endl;
    
    // Memory usage comparison
    size_t v1_memory = sizeof(workerData) + worker->data_len;
    size_t compressed_memory = sizeof(workerData); // Compressed in templates
    std::cout << "\nMemory Usage (approximate):" << std::endl;
    std::cout << "original/optimized memory: " << std::setw(10) << v1_memory << " bytes" << std::endl;
    std::cout << "compressed memory:      " << std::setw(10) << compressed_memory << " bytes" << std::endl;
    std::cout << "Memory savings:         " << std::setw(10) << (v1_memory - compressed_memory) << " bytes" << std::endl;
    
    aligned_free(worker);
    
    return 0;
}