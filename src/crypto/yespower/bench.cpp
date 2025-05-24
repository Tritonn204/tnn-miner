/*
 * Yespower optimization comparison - standalone function
 */

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "yespower.h"

static uint64_t bench_time_us(void)
{
    struct timespec t;
#ifdef CLOCK_MONOTONIC_RAW
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t))
        return 0;
#else
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        return 0;
#endif
    return 1 + (uint64_t)t.tv_sec * 1000000 + t.tv_nsec / 1000;
}

#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <chrono>
#include <iostream>

// Thread data for benchmark
struct BenchThreadData {
    int thread_id;
    int (*hash_func)(const uint8_t *, size_t, const yespower_params_t *, yespower_binary_t *);
    const yespower_params_t *params;
    uint8_t *src_base;
    unsigned int iterations_per_thread;
    std::atomic<uint64_t> *thread_time;
    std::atomic<bool> *thread_error;
};

static void benchmark_thread_worker(BenchThreadData data) {
    yespower_binary_t dst;
    uint8_t src[80];
    
    // Copy base input
    memcpy(src, data.src_base, 80);
    
    uint64_t start_time = bench_time_us();
    
    for (unsigned int i = 0; i < data.iterations_per_thread; i++) {
        // Each thread gets unique nonces to avoid cache effects
        ((uint32_t*)src)[19] = (data.thread_id << 24) | i;
        
        if (data.hash_func(src, 80, data.params, &dst)) {
            data.thread_error->store(true);
            break;
        }
    }
    
    uint64_t end_time = bench_time_us();
    data.thread_time->store(end_time - start_time);
}

// Multithreaded benchmark for a single configuration
static int benchmark_config_multithreaded(const yespower_params_t *params,
                                         const char *config_name,
                                         yespower_bench_result_t *result) {
    // Get number of threads
    unsigned int num_threads = std::thread::hardware_concurrency() / 2;
    num_threads = 5;
    if (num_threads == 0) num_threads = 1;
    
    // Test input
    uint8_t src[80];
    for (unsigned int i = 0; i < sizeof(src); i++)
        src[i] = i * 3;
    
    result->config_name = config_name;
    result->passed_correctness = 1;
    
    // Correctness check first
    yespower_binary_t dst_ref, dst_opt;
    for (int i = 0; i < 10; i++) {
        ((uint32_t*)src)[19] = i * 0x6c078965U;
        
        if (yespower_ref_tls(src, sizeof(src), params, &dst_ref) ||
            yespower_tls(src, sizeof(src), params, &dst_opt)) {
            result->passed_correctness = 0;
            return -1;
        }
        
        if (memcmp(&dst_ref, &dst_opt, sizeof(dst_ref))) {
            result->passed_correctness = 0;
            return -1;
        }
    }
    
    // Determine total iterations based on algorithm cost
    unsigned int total_iterations = 25000;
    
    unsigned int iterations_per_thread = total_iterations / num_threads;
    if (iterations_per_thread < 10) iterations_per_thread = 10;
    
    // Allocate thread data
    std::vector<std::atomic<uint64_t>> ref_times(num_threads);
    std::vector<std::atomic<uint64_t>> opt_times(num_threads);
    std::vector<std::atomic<bool>> ref_errors(num_threads);
    std::vector<std::atomic<bool>> opt_errors(num_threads);
    
    // Initialize atomics
    for (unsigned int i = 0; i < num_threads; i++) {
        ref_times[i].store(0);
        opt_times[i].store(0);
        ref_errors[i].store(false);
        opt_errors[i].store(false);
    }
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        yespower_ref_tls(src, sizeof(src), params, &dst_ref);
        yespower_tls(src, sizeof(src), params, &dst_opt);
    }
    
    std::cout << "Benchmarking " << config_name 
              << " (" << num_threads << " threads, " 
              << iterations_per_thread << " iterations each)...\n";
    
    // Benchmark reference implementation
    std::cout << "  Testing REF..." << std::flush;
    
    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        for (unsigned int t = 0; t < num_threads; t++) {
            BenchThreadData data = {
                .thread_id = (int)t,
                .hash_func = yespower_ref_tls,
                .params = params,
                .src_base = src,
                .iterations_per_thread = iterations_per_thread,
                .thread_time = &ref_times[t],
                .thread_error = &ref_errors[t]
            };
            
            threads.emplace_back(benchmark_thread_worker, data);
        }
        
        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    // Check for errors
    for (unsigned int t = 0; t < num_threads; t++) {
        if (ref_errors[t].load()) {
            std::cout << " FAILED\n";
            return -1;
        }
    }
    
    // Calculate reference results
    uint64_t total_ref_time = 0;
    uint64_t max_ref_time = 0;
    for (unsigned int t = 0; t < num_threads; t++) {
        uint64_t time = ref_times[t].load();
        if (time > max_ref_time) max_ref_time = time;
        total_ref_time += time;
    }
    
    std::cout << " done\n";
    
    // Benchmark optimized implementation
    std::cout << "  Testing OPT..." << std::flush;
    
    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        for (unsigned int t = 0; t < num_threads; t++) {
            BenchThreadData data = {
                .thread_id = (int)t,
                .hash_func = yespower_tls,
                .params = params,
                .src_base = src,
                .iterations_per_thread = iterations_per_thread,
                .thread_time = &opt_times[t],
                .thread_error = &opt_errors[t]
            };
            
            threads.emplace_back(benchmark_thread_worker, data);
        }
        
        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    // Check for errors
    for (unsigned int t = 0; t < num_threads; t++) {
        if (opt_errors[t].load()) {
            std::cout << " FAILED\n";
            return -1;
        }
    }
    
    // Calculate optimized results
    uint64_t total_opt_time = 0;
    uint64_t max_opt_time = 0;
    for (unsigned int t = 0; t < num_threads; t++) {
        uint64_t time = opt_times[t].load();
        if (time > max_opt_time) max_opt_time = time;
        total_opt_time += time;
    }
    
    std::cout << " done\n";
    
    // Store results
    unsigned int total_hashes = iterations_per_thread * num_threads;
    result->ref_time_us = max_ref_time;  // Bottleneck time
    result->opt_time_us = max_opt_time;  // Bottleneck time
    result->ref_hash_rate = 1000000.0 * total_hashes / max_ref_time;
    result->opt_hash_rate = 1000000.0 * total_hashes / max_opt_time;
    result->speedup = result->opt_hash_rate / result->ref_hash_rate;
    
    return 0;
}

// Updated main benchmark function
int benchmark_yespower_comparison_mt(yespower_bench_result_t *results, size_t *num_results)
{
    struct {
        const char *name;
        yespower_params_t params;
    } configs[] = {
        // {"v0.5 N=2048 r=8", {YESPOWER_0_5, 2048, 8, nullptr, 0}},
        // {"v1.0 N=2048 r=8", {YESPOWER_1_0, 2048, 8, nullptr, 0}},
        // {"v1.0 N=2048 r=8 +pers", {YESPOWER_1_0, 2048, 8, (const uint8_t*)"Test", 4}},
        {"ADVC r=32", {YESPOWER_1_0, 2048, 32, (const uint8_t*)"Let the quest begin", 19}},
    };
    
    size_t num_configs = sizeof(configs) / sizeof(configs[0]);
    if (*num_results < num_configs) {
        *num_results = num_configs;
        return -1;
    }
    
    std::cout << "=== Multithreaded Yespower Benchmark ===\n";
    std::cout << "Using " << std::thread::hardware_concurrency()/2 << " threads\n\n";
    
    for (size_t i = 0; i < num_configs; i++) {
        benchmark_config_multithreaded(&configs[i].params, configs[i].name, &results[i]);
    }
    
    *num_results = num_configs;
    return 0;
}

// Optional: Simple convenience function for quick testing
void quick_benchmark() {
    yespower_bench_result_t results[10];
    size_t num_results = 10;
    
    if (benchmark_yespower_comparison_mt(results, &num_results) == 0) {
        print_yespower_benchmark_results(results, num_results);
    }
}

// Convenience function to print results
void print_yespower_benchmark_results(const yespower_bench_result_t *results, size_t num_results)
{
    printf("\n=== Yespower Optimization Results ===\n\n");
    
    for (size_t i = 0; i < num_results; i++) {
        const yespower_bench_result_t *r = &results[i];
        
        printf("Config: %s\n", r->config_name);
        
        if (!r->passed_correctness) {
            printf("  ❌ FAILED correctness check\n\n");
            continue;
        }
        
        printf("  ✅ Passed correctness\n");
        printf("  REF: %7.1f H/s (%6.2f ms/hash)\n", 
               r->ref_hash_rate, r->ref_time_us / 1000.0);
        printf("  OPT: %7.1f H/s (%6.2f ms/hash)\n", 
               r->opt_hash_rate, r->opt_time_us / 1000.0);
        printf("  Speedup: %.2fx", r->speedup);
        
        if (r->speedup > 1.15) {
            printf(" 🚀\n");
        } else if (r->speedup > 1.05) {
            printf(" ⬆️\n");
        } else if (r->speedup < 0.95) {
            printf(" ⚠️\n");
        } else {
            printf(" ≈\n");
        }
        printf("\n");
    }
    
    // Summary
    double total_speedup = 0;
    int valid_count = 0;
    for (size_t i = 0; i < num_results; i++) {
        if (results[i].passed_correctness) {
            total_speedup += results[i].speedup;
            valid_count++;
        }
    }
    
    if (valid_count > 0) {
        printf("Average speedup: %.2fx\n\n", total_speedup / valid_count);
    }
}