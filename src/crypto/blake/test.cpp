#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include "blake_avx2.h"


void print_hash(const char* label, const uint8_t* hash, size_t len) {
    std::cout << label << ": ";
    for (size_t i = 0; i < len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (int)hash[i];
    }
    std::cout << std::dec << std::endl;
}

void test_correctness() {
    std::cout << "=== Correctness Test ===" << std::endl;
    
    // Test data
    uint8_t test_data[200];
    for (int i = 0; i < 200; i++) {
        test_data[i] = i;
    }
    
    // Our scalar implementation
    uint8_t scalar_output[32];
    do_blake_hash_scalar(test_data, scalar_output);
    
    // Our AVX2 implementation
    uint8_t avx2_output[32];
    do_blake_hash_avx2(test_data, avx2_output);
    
    // Print results
    print_hash("Scalar Blake-256", scalar_output, 32);
    print_hash("AVX2 Blake-256", avx2_output, 32);
    
    // Compare
    bool scalar_match = memcmp(avx2_output, scalar_output, 32) == 0;
    
    std::cout << "\nScalar matches SPH: " << (scalar_match ? "YES" : "NO") << std::endl;
}

void benchmark() {
    std::cout << "\n=== Performance Test ===" << std::endl;
    
    const int iterations = 1000000;
    uint8_t test_data[200];
    uint8_t output[32];
    
    // Initialize test data
    for (int i = 0; i < 200; i++) {
        test_data[i] = i;
    }
    
    // Benchmark scalar
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        do_blake_hash_scalar(test_data, output);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Benchmark AVX2
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        do_blake_hash_avx2(test_data, output);
    }
    end = std::chrono::high_resolution_clock::now();
    auto avx2_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Scalar Blake-256: " << scalar_time << " μs (" 
              << (iterations * 1000000.0 / scalar_time) << " hashes/sec)" << std::endl;
    std::cout << "AVX2 Blake-256: " << avx2_time << " μs (" 
              << (iterations * 1000000.0 / avx2_time) << " hashes/sec)" << std::endl;
    
    std::cout << "Speedup AVX2 vs scalar: " << (double)scalar_time / avx2_time << "x" << std::endl;
}

// Example of how to add inline assembly
void example_inline_asm() {
    // You can add the assembly from the article here
    // Example structure:
    /*
    __asm__ volatile (
        ".intel_syntax noprefix\n\t"
        
        // Your assembly code here
        "vpaddq ymm0, ymm0, ymm4\n\t"
        "vpaddq ymm0, ymm0, ymm1\n\t"
        // ... etc
        
        ".att_syntax prefix\n\t"
        : // outputs
        : // inputs
        : // clobbers
    );
    */
}

int main() {
    test_correctness();
    benchmark();
    return 0;
}