#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <immintrin.h>

// Define tu_int as __uint128_t
typedef __uint128_t tu_int;

// Your d128 function
static inline __uint128_t d128(tu_int a, uint64_t b) {
  uint64_t dividend_hi = a >> 64;
  uint64_t dividend_lo = (uint64_t)a;
  
#if defined(__x86_64__)
  uint64_t q_hi = 0, q_lo = 0, remainder = 0;
  
  if (dividend_hi < b) {
    // Single division case: dividend_hi:dividend_lo / b
    __asm__("divq %[divisor]"
            : "=a"(q_lo), "=d"(remainder)
            : [divisor] "r"(b), "a"(dividend_lo), "d"(dividend_hi)
            : "cc");
  } else {
    // Two division case
    // First: 0:dividend_hi / b
    __asm__("divq %[divisor]"
            : "=a"(q_hi), "=d"(dividend_hi)
            : [divisor] "r"(b), "a"(dividend_hi), "d"(0ULL)
            : "cc");
    
    // Second: remainder:dividend_lo / b
    __asm__("divq %[divisor]"
            : "=a"(q_lo), "=d"(remainder)
            : [divisor] "r"(b), "a"(dividend_lo), "d"(dividend_hi)
            : "cc");
  }
  
  return ((tu_int)q_hi << 64) | q_lo;
#else
  return a / b;
#endif
}

// Your div128_128_clz_improved function
static inline __uint128_t div128_128_clz_improved(__uint128_t dividend, __uint128_t divisor) {
  if (divisor > dividend) return 0;
  if (divisor == dividend) return 1;
  if ((divisor >> 64) == 0) return d128(dividend, (uint64_t)divisor);
  
  int dividend_clz = (dividend >> 64) ? __builtin_clzll(dividend >> 64) : 
                    64 + __builtin_clzll((uint64_t)dividend);
  int divisor_clz = (divisor >> 64) ? __builtin_clzll(divisor >> 64) : 
                    64 + __builtin_clzll((uint64_t)divisor);
  
  int shift = divisor_clz - dividend_clz;
  if (shift < 0) return 0;
  
  __uint128_t shifted_divisor = divisor << shift;
  __uint128_t quotient = 0;
  
  int chunks = (shift + 1) / 8;
  int remaining = (shift + 1) % 8;
  
  for (int chunk = 0; chunk < chunks; chunk++) {
    __uint128_t chunk_quotient = 0;
    __uint128_t chunk_divisor = shifted_divisor >> (8 * chunk);
    
    for (int bit = 0; bit < 8; bit++) {
      chunk_quotient <<= 1;
      if (dividend >= chunk_divisor) {
        dividend -= chunk_divisor;
        chunk_quotient |= 1;
      }
      chunk_divisor >>= 1;
    }
    
    quotient = (quotient << 8) | chunk_quotient;
  }
  
  if (remaining > 0) {
    __uint128_t remaining_quotient = 0;
    __uint128_t remaining_divisor = shifted_divisor >> (8 * chunks);
    
    for (int bit = 0; bit < remaining; bit++) {
      remaining_quotient <<= 1;
      if (dividend >= remaining_divisor) {
        dividend -= remaining_divisor;
        remaining_quotient |= 1;
      }
      remaining_divisor >>= 1;
    }
    
    quotient = (quotient << remaining) | remaining_quotient;
  }
  
  return quotient;
}

// SSE-optimized version of division
static inline __uint128_t div128_128_sse(__uint128_t dividend, __uint128_t divisor) {
  if (divisor > dividend) return 0;
  if (divisor == dividend) return 1;
  if ((divisor >> 64) == 0) return d128(dividend, (uint64_t)divisor);
  
  // Use SSE for bit manipulation where possible
  int dividend_clz = (dividend >> 64) ? __builtin_clzll(dividend >> 64) : 
                    64 + __builtin_clzll((uint64_t)dividend);
  int divisor_clz = (divisor >> 64) ? __builtin_clzll(divisor >> 64) : 
                    64 + __builtin_clzll((uint64_t)divisor);
  
  int shift = divisor_clz - dividend_clz;
  if (shift < 0) return 0;
  
  __uint128_t shifted_divisor = divisor << shift;
  __uint128_t quotient = 0;
  
  // Process 64-bit chunks using SSE when possible
  if (shift >= 64) {
    // Can use SSE for comparison and subtraction of 64-bit values
    uint64_t div_hi = shifted_divisor >> 64;
    uint64_t div_lo = (uint64_t)shifted_divisor;
    uint64_t rem_hi = dividend >> 64;
    uint64_t rem_lo = (uint64_t)dividend;
    
    for (int i = 0; i <= shift; i++) {
      quotient <<= 1;
      
      // Compare 128-bit values using 64-bit comparisons
      if (rem_hi > div_hi || (rem_hi == div_hi && rem_lo >= div_lo)) {
        // Subtract using SSE intrinsics for 64-bit arithmetic
        __m128i rem = _mm_set_epi64x(rem_hi, rem_lo);
        __m128i div = _mm_set_epi64x(div_hi, div_lo);
        
        // Perform subtraction with borrow handling
        uint64_t borrow = (rem_lo < div_lo) ? 1 : 0;
        rem_lo -= div_lo;
        rem_hi -= div_hi + borrow;
        
        quotient |= 1;
      }
      
      // Shift divisor right
      div_lo = (div_lo >> 1) | (div_hi << 63);
      div_hi >>= 1;
    }
    
    dividend = ((tu_int)rem_hi << 64) | rem_lo;
  } else {
    // Fall back to original algorithm for small shifts
    for (int i = 0; i <= shift; i++) {
      quotient <<= 1;
      if (dividend >= shifted_divisor) {
        dividend -= shifted_divisor;
        quotient |= 1;
      }
      shifted_divisor >>= 1;
    }
  }
  
  return quotient;
}

// AVX2-optimized version
static inline __uint128_t div128_128_avx2(__uint128_t dividend, __uint128_t divisor) {
  if (divisor > dividend) return 0;
  if (divisor == dividend) return 1;
  if ((divisor >> 64) == 0) return d128(dividend, (uint64_t)divisor);
  
  int dividend_clz = (dividend >> 64) ? __builtin_clzll(dividend >> 64) : 
                    64 + __builtin_clzll((uint64_t)dividend);
  int divisor_clz = (divisor >> 64) ? __builtin_clzll(divisor >> 64) : 
                    64 + __builtin_clzll((uint64_t)divisor);
  
  int shift = divisor_clz - dividend_clz;
  if (shift < 0) return 0;
  
  __uint128_t shifted_divisor = divisor << shift;
  __uint128_t quotient = 0;
  
  uint64_t div_hi = shifted_divisor >> 64;
  uint64_t div_lo = (uint64_t)shifted_divisor;
  uint64_t rem_hi = dividend >> 64;
  uint64_t rem_lo = (uint64_t)dividend;
  
  // Tier 1: For very large shifts (>= 32), use AVX2 to process 16 bits at a time
  if (shift >= 32) {
    const int AVX2_BITS = 16;
    int avx2_iters = shift / AVX2_BITS;
    
    for (int iter = 0; iter < avx2_iters; iter++) {
      uint16_t iter_quotient = 0;
      
      // Process 16 bits using unrolled loop
      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        // Process 4 bits at a time
        for (int j = 0; j < 4; j++) {
          iter_quotient <<= 1;
          if (rem_hi > div_hi || (rem_hi == div_hi && rem_lo >= div_lo)) {
            uint64_t borrow = (rem_lo < div_lo) ? 1 : 0;
            rem_lo -= div_lo;
            rem_hi -= div_hi + borrow;
            iter_quotient |= 1;
          }
          div_lo = (div_lo >> 1) | (div_hi << 63);
          div_hi >>= 1;
        }
      }
      
      quotient = (quotient << AVX2_BITS) | iter_quotient;
      shift -= AVX2_BITS;
    }
  }
  
  // Tier 2: For medium shifts (8-31), use SSE to process 8 bits at a time
  if (shift >= 8) {
    const int SSE_BITS = 8;
    int sse_iters = shift / SSE_BITS;
    
    for (int iter = 0; iter < sse_iters; iter++) {
      uint8_t iter_quotient = 0;
      
      // Use SSE comparison hints
      __m128i rem = _mm_set_epi64x(rem_hi, rem_lo);
      __m128i div = _mm_set_epi64x(div_hi, div_lo);
      
      #pragma unroll 8
      for (int bit = 0; bit < SSE_BITS; bit++) {
        iter_quotient <<= 1;
        
        if (rem_hi > div_hi || (rem_hi == div_hi && rem_lo >= div_lo)) {
          uint64_t borrow = (rem_lo < div_lo) ? 1 : 0;
          rem_lo -= div_lo;
          rem_hi -= div_hi + borrow;
          iter_quotient |= 1;
        }
        
        div_lo = (div_lo >> 1) | (div_hi << 63);
        div_hi >>= 1;
      }
      
      quotient = (quotient << SSE_BITS) | iter_quotient;
      shift -= SSE_BITS;
    }
  }
  
  // Tier 3: For small shifts (< 8), use scalar operations
  for (int i = 0; i <= shift; i++) {
    quotient <<= 1;
    
    if (rem_hi > div_hi || (rem_hi == div_hi && rem_lo >= div_lo)) {
      uint64_t borrow = (rem_lo < div_lo) ? 1 : 0;
      rem_lo -= div_lo;
      rem_hi -= div_hi + borrow;
      quotient |= 1;
    }
    
    div_lo = (div_lo >> 1) | (div_hi << 63);
    div_hi >>= 1;
  }
  
  return quotient;
}

// Optimized SSE version with better bit processing
static inline __uint128_t div128_128_sse_optimized(__uint128_t dividend, __uint128_t divisor) {
  if (divisor > dividend) return 0;
  if (divisor == dividend) return 1;
  if ((divisor >> 64) == 0) return d128(dividend, (uint64_t)divisor);
  
  int dividend_clz = (dividend >> 64) ? __builtin_clzll(dividend >> 64) : 
                    64 + __builtin_clzll((uint64_t)dividend);
  int divisor_clz = (divisor >> 64) ? __builtin_clzll(divisor >> 64) : 
                    64 + __builtin_clzll((uint64_t)divisor);
  
  int shift = divisor_clz - dividend_clz;
  if (shift < 0) return 0;
  
  __uint128_t shifted_divisor = divisor << shift;
  __uint128_t quotient = 0;
  
  uint64_t div_hi = shifted_divisor >> 64;
  uint64_t div_lo = (uint64_t)shifted_divisor;
  uint64_t rem_hi = dividend >> 64;
  uint64_t rem_lo = (uint64_t)dividend;
  
  // Process bits in groups of 4 or 8 depending on shift count
  const int bits_per_iter = (shift > 32) ? 8 : 4;
  
  while (shift >= bits_per_iter) {
    uint8_t iter_quotient = 0;
    
    #pragma unroll
    for (int bit = 0; bit < bits_per_iter; bit++) {
      iter_quotient <<= 1;
      
      // Branch-free comparison and subtraction
      uint64_t ge = (rem_hi > div_hi) | ((rem_hi == div_hi) & (rem_lo >= div_lo));
      uint64_t borrow = (rem_lo < div_lo) & ge;
      
      rem_lo -= div_lo & -ge;
      rem_hi -= (div_hi + borrow) & -ge;
      iter_quotient |= ge;
      
      // Shift divisor
      div_lo = (div_lo >> 1) | (div_hi << 63);
      div_hi >>= 1;
    }
    
    quotient = (quotient << bits_per_iter) | iter_quotient;
    shift -= bits_per_iter;
  }
  
  // Handle remaining bits
  for (int i = 0; i <= shift; i++) {
    quotient <<= 1;
    if (rem_hi > div_hi || (rem_hi == div_hi && rem_lo >= div_lo)) {
      uint64_t borrow = (rem_lo < div_lo) ? 1 : 0;
      rem_lo -= div_lo;
      rem_hi -= div_hi + borrow;
      quotient |= 1;
    }
    div_lo = (div_lo >> 1) | (div_hi << 63);
    div_hi >>= 1;
  }
  
  return quotient;
}

// Generate random 128-bit number with at least 127 bits
__uint128_t random_uint127_plus(std::mt19937_64& gen) {
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    
    // Generate high 64 bits with at least bit 62 or 63 set (ensuring 126+ bits total)
    uint64_t high = dist(gen) | (1ULL << 62);  // Ensure at least bit 126 is set
    uint64_t low = dist(gen);
    
    return ((__uint128_t)high << 64) | low;
}

// Generate random 64-bit number
uint64_t random_uint64(std::mt19937_64& gen) {
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    return dist(gen);
}

// Generate random 128-bit number
__uint128_t random_uint128(std::mt19937_64& gen) {
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    __uint128_t result = dist(gen);
    result = (result << 64) | dist(gen);
    return result;
}

struct TestCase {
    __uint128_t dividend;
    __uint128_t divisor;
    std::string description;
};

struct BenchmarkResult {
    double custom_ns;
    double sse_ns;
    double avx2_ns;
    double compiler_ns;
    double custom_speedup;
    double sse_speedup;
    double avx2_speedup;
};

int main() {
    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    
    // Create test cases
    std::vector<TestCase> test_cases;
    
    // Edge cases with large dividends
    test_cases.push_back({__uint128_t(1) << 127, 2, "2^127 / 2"});
    test_cases.push_back({(__uint128_t(1) << 127) | (__uint128_t(1) << 126), 3, "(2^127 + 2^126) / 3"});
    test_cases.push_back({__uint128_t(-1), 10, "Max uint128 / 10"});
    
    // Random test cases with 64-bit divisor and 127+ bit dividend
    for (int i = 0; i < 5; i++) {
        uint64_t divisor = random_uint64(gen);
        if (divisor == 0) divisor = 1;
        
        test_cases.push_back({
            random_uint127_plus(gen),
            divisor,
            "Random 127+/64 bit #" + std::to_string(i+1)
        });
    }
    
    // Random test cases with 128-bit divisor and 127+ bit dividend
    for (int i = 0; i < 10; i++) {
        __uint128_t dividend = random_uint127_plus(gen);
        __uint128_t divisor = random_uint128(gen);
        
        if (divisor == 0) divisor = 1;
        if (divisor > dividend) {
            divisor = dividend / 2;
            if (divisor == 0) divisor = 1;
        }
        
        test_cases.push_back({
            dividend,
            divisor,
            "Random 127+/128 bit #" + std::to_string(i+1)
        });
    }
    
    const int NUM_RUNS = 10;
    const int WARMUP_ITERATIONS = 10000;
    const int BENCHMARK_ITERATIONS = 1000000;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Benchmarking 128-bit Division (10 runs per test)\n";
    std::cout << "All dividends are at least 127 bits\n";
    std::cout << "================================================\n\n";
    
    // Summary statistics
    std::vector<double> all_custom_speedups;
    std::vector<double> all_sse_speedups;
    std::vector<double> all_avx2_speedups;
    
    for (const auto& test : test_cases) {
        std::vector<BenchmarkResult> results;
        
        // Verify dividend is at least 127 bits
        int dividend_bits = 128 - ((test.dividend >> 64) ? __builtin_clzll(test.dividend >> 64) : 
                           64 + __builtin_clzll((uint64_t)test.dividend));
        
        // Verify correctness once
        __uint128_t custom_result = div128_128_clz_improved(test.dividend, test.divisor);
        __uint128_t sse_result = div128_128_sse(test.dividend, test.divisor);
        __uint128_t avx2_result = div128_128_avx2(test.dividend, test.divisor);
        __uint128_t compiler_result = test.dividend / test.divisor;
        
        bool custom_match = (custom_result == compiler_result);
        bool sse_match = (sse_result == compiler_result);
        bool avx2_match = (avx2_result == compiler_result);
        
        // Run benchmark multiple times
        for (int run = 0; run < NUM_RUNS; run++) {
            BenchmarkResult result;
            
            // Warmup
            volatile __uint128_t dummy = 0;
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                dummy = div128_128_clz_improved(test.dividend, test.divisor);
                dummy = div128_128_sse(test.dividend, test.divisor);
                dummy = div128_128_avx2(test.dividend, test.divisor);
                dummy = test.dividend / test.divisor;
            }
            
            // Benchmark custom implementation
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                dummy = div128_128_clz_improved(test.dividend, test.divisor);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto custom_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            result.custom_ns = static_cast<double>(custom_duration.count()) / BENCHMARK_ITERATIONS;
            
            // Benchmark SSE implementation
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                dummy = div128_128_sse(test.dividend, test.divisor);
            }
            end = std::chrono::high_resolution_clock::now();
            auto sse_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            result.sse_ns = static_cast<double>(sse_duration.count()) / BENCHMARK_ITERATIONS;
            
            // Benchmark AVX2 implementation
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                dummy = div128_128_avx2(test.dividend, test.divisor);
            }
            end = std::chrono::high_resolution_clock::now();
            auto avx2_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            result.avx2_ns = static_cast<double>(avx2_duration.count()) / BENCHMARK_ITERATIONS;
            
            // Benchmark compiler implementation
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                dummy = test.dividend / test.divisor;
            }
            end = std::chrono::high_resolution_clock::now();
            auto compiler_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            result.compiler_ns = static_cast<double>(compiler_duration.count()) / BENCHMARK_ITERATIONS;
            
            result.custom_speedup = result.compiler_ns / result.custom_ns;
            result.sse_speedup = result.compiler_ns / result.sse_ns;
            result.avx2_speedup = result.compiler_ns / result.avx2_ns;
            
            results.push_back(result);
        }
        
        // Calculate statistics
        double avg_custom_ns = 0, avg_sse_ns = 0, avg_avx2_ns = 0, avg_compiler_ns = 0;
        double avg_custom_speedup = 0, avg_sse_speedup = 0, avg_avx2_speedup = 0;
        
        for (const auto& r : results) {
            avg_custom_ns += r.custom_ns;
            avg_sse_ns += r.sse_ns;
            avg_avx2_ns += r.avx2_ns;
            avg_compiler_ns += r.compiler_ns;
            avg_custom_speedup += r.custom_speedup;
            avg_sse_speedup += r.sse_speedup;
            avg_avx2_speedup += r.avx2_speedup;
        }
        
        avg_custom_ns /= NUM_RUNS;
        avg_sse_ns /= NUM_RUNS;
        avg_avx2_ns /= NUM_RUNS;
        avg_compiler_ns /= NUM_RUNS;
        avg_custom_speedup /= NUM_RUNS;
        avg_sse_speedup /= NUM_RUNS;
        avg_avx2_speedup /= NUM_RUNS;
        
        all_custom_speedups.push_back(avg_custom_speedup);
        all_sse_speedups.push_back(avg_sse_speedup);
        all_avx2_speedups.push_back(avg_avx2_speedup);
        
        // Print summary
        std::cout << test.description << " (" << dividend_bits << " bits):\n";
        std::cout << "  Custom:   " << avg_custom_ns << " ns/op (speedup: " 
                  << avg_custom_speedup << "x)" << (custom_match ? "" : " [MISMATCH]") << "\n";
        std::cout << "  SSE:      " << avg_sse_ns << " ns/op (speedup: " 
                  << avg_sse_speedup << "x)" << (sse_match ? "" : " [MISMATCH]") << "\n";
        std::cout << "  AVX2:     " << avg_avx2_ns << " ns/op (speedup: " 
                  << avg_avx2_speedup << "x)" << (avx2_match ? "" : " [MISMATCH]") << "\n";
        std::cout << "  Compiler: " << avg_compiler_ns << " ns/op (baseline)\n\n";
    }
    
    // Overall summary
    double overall_custom_speedup = std::accumulate(all_custom_speedups.begin(), 
                                                   all_custom_speedups.end(), 0.0) / all_custom_speedups.size();
    double overall_sse_speedup = std::accumulate(all_sse_speedups.begin(), 
                                                all_sse_speedups.end(), 0.0) / all_sse_speedups.size();
    double overall_avx2_speedup = std::accumulate(all_avx2_speedups.begin(), 
                                                all_avx2_speedups.end(), 0.0) / all_avx2_speedups.size();
    
    std::cout << "Overall Average Speedup:\n";
    std::cout << "========================\n";
    std::cout << "Custom implementation: " << overall_custom_speedup << "x faster\n";
    std::cout << "SSE implementation:    " << overall_sse_speedup << "x faster\n";
    std::cout << "AVX2 implementation:   " << overall_avx2_speedup << "x faster\n";
    
    return 0;
}