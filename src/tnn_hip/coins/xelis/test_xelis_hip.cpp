#ifdef TNN_HIP
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
#include <hip/hip_runtime.h>

#include "../../common/gpu_rtc.hpp"
#include "../../common/gpu_algo.hpp"
#include "../../../crypto/xelis-hash/xelis-hash.hpp"
#include <hex.h>

// Embedded kernel source
#include "xelis-hash-v3.hip.hpp"
#include "xelis_embedded_headers.hpp"
#include "tnn_hip_common_embedded.hpp"

// Test vector: Use a simple known input
static const uint8_t TEST_WORK[XELIS_TEMPLATE_SIZE] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
    0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f
};

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "[ERROR] HIP call failed: %s (error %d) at %s:%d\n", \
                hipGetErrorString(err), err, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

static uint64_t compute_checksum(const uint64_t *data, size_t count) {
    uint64_t checksum = 0;
    for (size_t i = 0; i < count; i++) {
        checksum ^= data[i];
    }
    return checksum;
}

// Run a single test case (shared by all test configurations)
static int run_test_case(
    const char *test_name,
    hipFunction_t stage1_func,
    hipFunction_t stage3_func,
    hipFunction_t blake3_func,
    uint64_t nonce_start,
    uint32_t batch_size,
    uint32_t grid_x,
    uint32_t block_x,
    uint32_t num_launches = 1  // Number of sequential kernel launches
) {
    printf("\n[TEST] === %s ===\n", test_name);
    printf("[TEST] Configuration: batch_size=%u, grid(%u,1,1), block(%u,1,1), launches=%u\n",
           batch_size, grid_x, block_x, num_launches);
    printf("[TEST] Total threads per launch: %u\n", grid_x * block_x);
    printf("[TEST] Total hashes (all launches): %u\n", batch_size);
    fflush(stdout);

    const uint32_t scratch_offset = 0;
    const uint32_t total_threads = grid_x * block_x;
    const uint32_t hashes_per_launch = total_threads;

    if (batch_size != hashes_per_launch * num_launches) {
        fprintf(stderr, "[ERROR] batch_size (%u) != hashes_per_launch (%u) * num_launches (%u)\n",
                batch_size, hashes_per_launch, num_launches);
        return 1;
    }

    // Allocate CPU scratch buffers per launch (not all batches)
    uint64_t *cpu_scratch_s1 = new uint64_t[XELIS_MEMORY_SIZE_V3 * hashes_per_launch];
    uint64_t *cpu_scratch_s3 = new uint64_t[XELIS_MEMORY_SIZE_V3 * hashes_per_launch];
    uint8_t *cpu_hashes = new uint8_t[32 * hashes_per_launch];

    // Allocate GPU memory (per launch, not all batches)
    uint8_t *d_input = nullptr;
    uint64_t *d_scratch = nullptr;
    uint8_t *d_output = nullptr;
    uint64_t *d_difficulty = nullptr;
    uint64_t *d_solutions = nullptr;

    HIP_CHECK(hipMalloc(&d_input, XELIS_TEMPLATE_SIZE));
    HIP_CHECK(hipMalloc(&d_scratch, XELIS_MEMORY_SIZE_V3 * sizeof(uint64_t) * hashes_per_launch));
    HIP_CHECK(hipMalloc(&d_output, 32 * hashes_per_launch));
    HIP_CHECK(hipMalloc(&d_difficulty, 32));
    HIP_CHECK(hipMalloc(&d_solutions, 64));

    HIP_CHECK(hipMemcpy(d_input, TEST_WORK, XELIS_TEMPLATE_SIZE, hipMemcpyHostToDevice));

    // Run Stage 1 - single block only (grid_x should always be 1)
    if (grid_x != 1) {
        fprintf(stderr, "[ERROR] Multi-block tests not supported (grid_x must be 1)\n");
        return 1;
    }

    bool all_pass = true;
    printf("[TEST] Processing %u launch(es), %u hashes per launch\n", num_launches, hashes_per_launch);
    fflush(stdout);

    for (uint32_t launch_idx = 0; launch_idx < num_launches; launch_idx++) {
        uint64_t launch_nonce_start = nonce_start + (launch_idx * hashes_per_launch);
        uint32_t launch_scratch_offset = 0;  // Each launch uses scratch from 0 (independent)
        uint32_t launch_batch_size = hashes_per_launch;

        printf("\n[TEST] === Launch %u/%u (nonces %llu-%llu) ===\n",
               launch_idx + 1, num_launches,
               (unsigned long long)launch_nonce_start,
               (unsigned long long)(launch_nonce_start + hashes_per_launch - 1));
        fflush(stdout);

        // 1. Compute CPU reference for this launch
        printf("[TEST]   Computing CPU reference...\n");
        fflush(stdout);
        for (uint32_t i = 0; i < hashes_per_launch; i++) {
            uint8_t cpu_input[XELIS_TEMPLATE_SIZE];
            memcpy(cpu_input, TEST_WORK, XELIS_TEMPLATE_SIZE);

            uint64_t counter = launch_nonce_start & 0xFFFFFFFFFFFFULL;
            uint64_t nonce = (launch_nonce_start & 0xFFFF000000000000ULL) | ((counter + i) & 0xFFFFFFFFFFFFULL);

            // Insert nonce at bytes 40-47
            for (int j = 0; j < 8; j++) {
                cpu_input[40 + j] = (nonce >> (j * 8)) & 0xFF;
            }

            uint64_t *batch_s1 = cpu_scratch_s1 + (i * XELIS_MEMORY_SIZE_V3);
            uint64_t *batch_s3 = cpu_scratch_s3 + (i * XELIS_MEMORY_SIZE_V3);

            xelis_stage1_v3(cpu_input, batch_s1, XELIS_TEMPLATE_SIZE);
            memcpy(batch_s3, batch_s1, XELIS_MEMORY_SIZE_V3 * sizeof(uint64_t));
            xelis_stage3_v3(batch_s3);
            xelis_blake3_v3((uint8_t*)batch_s3, cpu_hashes + (i * 32));
        }

        // 2. Run GPU kernels
        HIP_CHECK(hipMemset(d_scratch, 0, XELIS_MEMORY_SIZE_V3 * sizeof(uint64_t) * hashes_per_launch));

        printf("[TEST]   Running GPU Stage 1...\n");
        fflush(stdout);
        void *stage1_args[] = {&d_input, &d_scratch, (void*)&launch_nonce_start, (void*)&launch_batch_size, (void*)&launch_scratch_offset};
        HIP_CHECK(hipModuleLaunchKernel(stage1_func, 1, 1, 1, block_x, 1, 1, 0, nullptr, stage1_args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());

        printf("[TEST]   Running GPU Stage 3...\n");
        fflush(stdout);
        void *stage3_args[] = {&d_scratch, (void*)&launch_batch_size, (void*)&launch_scratch_offset, &d_difficulty, &d_solutions, (void*)&launch_nonce_start};
        HIP_CHECK(hipModuleLaunchKernel(stage3_func, 1, 1, 1, block_x, 1, 1, 0, nullptr, stage3_args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());

        printf("[TEST]   Running GPU Blake3...\n");
        fflush(stdout);
        void *blake3_args[] = {&d_scratch, &d_output, (void*)&launch_batch_size, (void*)&launch_scratch_offset, &d_difficulty, &d_solutions, (void*)&launch_nonce_start};
        HIP_CHECK(hipModuleLaunchKernel(blake3_func, 1, 1, 1, 256, 1, 1, 0, nullptr, blake3_args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());

        // 3. Download and validate
        uint8_t *gpu_hashes = new uint8_t[32 * hashes_per_launch];
        HIP_CHECK(hipMemcpy(gpu_hashes, d_output, 32 * hashes_per_launch, hipMemcpyDeviceToHost));

        printf("[TEST]   Validating hashes...\n");
        fflush(stdout);
        uint32_t mismatches = 0;
        for (uint32_t i = 0; i < hashes_per_launch; i++) {
            if (memcmp(cpu_hashes + (i * 32), gpu_hashes + (i * 32), 32) != 0) {
                if (mismatches < 5) {
                    printf("[TEST]     \033[31m✗ Mismatch at hash %u (nonce %llu)\033[0m\n",
                           i, (unsigned long long)(launch_nonce_start + i));
                    printf("[TEST]       CPU: %s\n", hexStr(cpu_hashes + (i * 32), 32).c_str());
                    printf("[TEST]       GPU: %s\n", hexStr(gpu_hashes + (i * 32), 32).c_str());
                }
                mismatches++;
            }
        }

        delete[] gpu_hashes;

        if (mismatches == 0) {
            printf("[TEST]   \033[32m✓ Launch %u: All %u hashes match\033[0m\n", launch_idx + 1, hashes_per_launch);
        } else {
            printf("[TEST]   \033[31m✗ Launch %u: %u/%u mismatches (%.2f%% failure)\033[0m\n",
                   launch_idx + 1, mismatches, hashes_per_launch, (100.0 * mismatches) / hashes_per_launch);
            all_pass = false;
        }
        fflush(stdout);
    }

    // Summary for this test case
    printf("\n[TEST] %s: %s\n", test_name, all_pass ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");

    // Cleanup
    delete[] cpu_scratch_s1;
    delete[] cpu_scratch_s3;
    delete[] cpu_hashes;
    hipFree(d_input);
    hipFree(d_scratch);
    hipFree(d_output);
    hipFree(d_difficulty);
    hipFree(d_solutions);

    return all_pass ? 0 : 1;
}

static int test_xelis_hip_impl() {

    // 1. Initialize HIP device
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "[ERROR] No HIP devices found\n");
        return 1;
    }

    printf("[TEST] Found %d HIP device(s)\n", deviceCount);

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    HIP_CHECK(hipSetDevice(0));

    printf("[TEST] Using device 0: %s\n", props.name);
    printf("[TEST] Compute capability: %d.%d\n", props.major, props.minor);

    // Detect platform at runtime based on gcnArchName
    // AMD uses "gfxXXXX", NVIDIA uses empty string (gcnArchName not available)
    #if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
    bool is_amd = false;
    #else
    bool is_amd = true;
    #endif

    // Set GPU architecture for RTCCompiler (same as precompile does)
    if (is_amd) {
        RTCCompiler::instance().set_gpu_arch(props.gcnArchName);
        printf("[TEST] Using arch=%s (AMD)\n\n", props.gcnArchName);
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "sm_%d%d", props.major, props.minor);
        RTCCompiler::instance().set_gpu_arch(buf);
        printf("[TEST] Using arch=%s (NVIDIA)\n\n", buf);
    }

    // 2. Compile GPU kernels using RTCCompiler (once for all tests)
    printf("[TEST] === Compiling GPU Kernels ===\n");

    auto& compiler = RTCCompiler::instance();

    // Register common headers
    auto rtc_headers = build_rtc_headers(
        hip_embedded::XELIS_SOURCES,
        hip_embedded::COMMON_HEADERS
    );

    printf("[TEST] Registering %zu manifest headers with RTCCompiler\n", rtc_headers.size());
    for (const auto& h : rtc_headers) {
        compiler.add_header_source(std::string(h.name), std::string(h.source));
    }

    const std::string xelis_source =
        std::string(hip_xelis_v3_source::SRC_TNN_HIP_CRYPTO_XELIS_HASH_XELIS_HASH_V3_HIP_SOURCE);

    // Build compile options (same as precompile does) - use runtime detection
    std::vector<std::string> compile_opts;
    if (is_amd) {
        compile_opts = {"-O3", "-mno-cumode", "-ffast-math"};
        compile_opts.push_back("-DXELIS_MIN_WG=64");
        compile_opts.push_back("-DXELIS_MAX_WG=256");
    } else {
        compile_opts = {"--dopt=on", "--use_fast_math"};
#ifdef __linux__
        compile_opts.push_back("--device-int128");
#endif
        compile_opts.push_back("-DXELIS_MIN_WG=32");
        compile_opts.push_back("-DXELIS_MAX_WG=128");
    }

    printf("[TEST] Compiling Xelis v3 module (once for all kernels)...\n");
    fflush(stdout);

    // Compile once - this creates a module with all kernels
    auto module_kernel = compiler.compile_from_source(
        xelis_source,
        "xelis-hash-v3.hip",
        "xelis_stage1_kernel",  // Primary kernel for compilation
        compile_opts
    );
    printf("[TEST]   ✓ Module compiled\n");
    fflush(stdout);

    // Load all kernel functions from the module
    hipFunction_t stage1_func = module_kernel.function;
    hipFunction_t stage3_func = nullptr;
    hipFunction_t blake3_func = nullptr;

    hipError_t err = hipModuleGetFunction(&stage3_func, module_kernel.module, "xelis_s3_efficient_noblake_kernel");
    if (err != hipSuccess) {
        fprintf(stderr, "[ERROR] Failed to get stage3 kernel: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("[TEST]   ✓ Loaded xelis_s3_efficient_noblake_kernel\n");
    fflush(stdout);

    err = hipModuleGetFunction(&blake3_func, module_kernel.module, "xelis_blake3_batch");
    if (err != hipSuccess) {
        fprintf(stderr, "[ERROR] Failed to get blake3 kernel: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("[TEST]   ✓ Loaded xelis_blake3_batch\n\n");
    fflush(stdout);

    // 3. Run multiple test cases (single-block only)
    printf("========================================\n");
    printf("[TEST] Running Single-Block Multi-Batch Validation\n");
    printf("========================================\n");
    fflush(stdout);

    int failures = 0;

    // Test 1: Single batch, single block (baseline)
    failures += run_test_case(
        "Test 1: Single Batch, Single Block",
        stage1_func, stage3_func, blake3_func,
        0x0000000000000000ULL,  // nonce_start
        1,      // batch_size
        1,      // grid_x
        1       // block_x
    );

    // Test 2: Small multi-batch, single block
    failures += run_test_case(
        "Test 2: Multi-Batch (32), Single Block",
        stage1_func, stage3_func, blake3_func,
        0x0000000000000000ULL,  // nonce_start
        32,     // batch_size
        1,      // grid_x
        32      // block_x
    );

    // Test 3: Large multi-batch, single block (256*20 = 5120 hashes in 20 launches)
    failures += run_test_case(
        "Test 3: Large Multi-Batch (5120), Single Block, 20 Launches",
        stage1_func, stage3_func, blake3_func,
        0x0000000000000000ULL,  // nonce_start
        5120,   // batch_size
        1,      // grid_x
        256,    // block_x
        20      // num_launches
    );

    // Test 4: Test with non-zero nonce_start (device ID bits set)
    uint64_t device_nonce = (1ULL << 59);  // Device ID = 1
    failures += run_test_case(
        "Test 4: Device ID Segmentation (device=1)",
        stage1_func, stage3_func, blake3_func,
        device_nonce,  // nonce_start with device ID
        32,     // batch_size
        1,      // grid_x
        32      // block_x
    );

    // Final summary
    printf("\n========================================\n");
    printf("[TEST] === FINAL SUMMARY ===\n");
    printf("[TEST] Total tests run: 4\n");
    printf("[TEST] Failures: %d\n", failures);
    printf("[TEST] Overall: %s\n", (failures == 0) ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
    printf("========================================\n\n");

    return (failures == 0) ? 0 : 1;
}


int test_xelis_hip() {
    printf("\n");
    printf("========================================\n");
    printf("[hip-test-xelis] Starting stage-by-stage GPU validation\n");
    printf("========================================\n\n");
    fflush(stdout);

    try {
        return test_xelis_hip_impl();
    } catch (const std::exception& e) {
        fprintf(stderr, "\n[ERROR] Test failed with exception: %s\n", e.what());
        fflush(stderr);
        return 1;
    } catch (...) {
        fprintf(stderr, "\n[ERROR] Test failed with unknown exception\n");
        fflush(stderr);
        return 1;
    }
}

#else

int test_xelis_hip() {
    fprintf(stderr, "[ERROR] TNN_HIP not enabled, cannot run GPU tests\n");
    return 1;
}

#endif
