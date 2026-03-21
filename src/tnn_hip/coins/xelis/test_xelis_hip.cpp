#ifdef TNN_HIP
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <tnn_hip/common/gpu_compat.hpp>

#include <boost/asio.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <core/rootcert.h>

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
    oroError_t err = call; \
    if (err != oroSuccess) { \
        fprintf(stderr, "[ERROR] GPU call failed: %s (error %d) at %s:%d\n", \
                tnn_error_string(err), err, __FILE__, __LINE__); \
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

// Run a single test case: serial stage1, production stage3, serial blake3
static int run_test_case(
    const char *test_name,
    oroFunction_t stage1_func,
    oroFunction_t stage3_func,
    oroFunction_t blake3_func,
    uint64_t nonce_start,
    uint32_t batch_size,
    uint32_t block_x,
    uint32_t num_launches = 1
) {
    const uint32_t hashes_per_launch = batch_size / num_launches;

    printf("\n[TEST] === %s ===\n", test_name);
    printf("[TEST] Configuration: batch_size=%u, block_x=%u, launches=%u\n",
           batch_size, block_x, num_launches);
    printf("[TEST] Hashes per launch: %u\n", hashes_per_launch);
    fflush(stdout);

    if (hashes_per_launch == 0 || hashes_per_launch * num_launches != batch_size) {
        fprintf(stderr, "[ERROR] batch_size (%u) not evenly divisible by num_launches (%u)\n",
                batch_size, num_launches);
        return 1;
    }

    // Allocate CPU scratch buffers per launch
    uint64_t *cpu_scratch_s1 = new uint64_t[XELIS_MEMORY_SIZE_V3 * hashes_per_launch];
    uint64_t *cpu_scratch_s3 = new uint64_t[XELIS_MEMORY_SIZE_V3 * hashes_per_launch];
    uint8_t *cpu_hashes = new uint8_t[32 * hashes_per_launch];

    // Allocate GPU memory
    uint8_t *d_input = nullptr;
    uint64_t *d_scratch = nullptr;
    uint8_t *d_output = nullptr;
    uint64_t *d_difficulty = nullptr;
    uint64_t *d_solutions = nullptr;

    HIP_CHECK(oroMalloc((oroDeviceptr*)&d_input, XELIS_TEMPLATE_SIZE));
    HIP_CHECK(oroMalloc((oroDeviceptr*)&d_scratch, XELIS_MEMORY_SIZE_V3 * sizeof(uint64_t) * hashes_per_launch));
    HIP_CHECK(oroMalloc((oroDeviceptr*)&d_output, 32 * hashes_per_launch));
    HIP_CHECK(oroMalloc((oroDeviceptr*)&d_difficulty, 32));
    HIP_CHECK(oroMalloc((oroDeviceptr*)&d_solutions, 64));

    HIP_CHECK(oroMemcpy(d_input, TEST_WORK, XELIS_TEMPLATE_SIZE, oroMemcpyHostToDevice));

    bool all_pass = true;
    printf("[TEST] Processing %u launch(es), %u hashes per launch\n", num_launches, hashes_per_launch);
    fflush(stdout);

    for (uint32_t launch_idx = 0; launch_idx < num_launches; launch_idx++) {
        uint64_t launch_nonce_start = nonce_start + (launch_idx * hashes_per_launch);
        uint32_t launch_scratch_offset = 0;
        uint32_t launch_batch_size = hashes_per_launch;

        printf("\n[TEST] === Launch %u/%u (nonces %llu-%llu) ===\n",
               launch_idx + 1, num_launches,
               (unsigned long long)launch_nonce_start,
               (unsigned long long)(launch_nonce_start + hashes_per_launch - 1));
        fflush(stdout);

        // 1. Compute CPU reference
        printf("[TEST]   Computing CPU reference...\n");
        fflush(stdout);
        for (uint32_t i = 0; i < hashes_per_launch; i++) {
            uint8_t cpu_input[XELIS_TEMPLATE_SIZE];
            memcpy(cpu_input, TEST_WORK, XELIS_TEMPLATE_SIZE);

            uint64_t counter = launch_nonce_start & 0xFFFFFFFFFFFFULL;
            uint64_t nonce = (launch_nonce_start & 0xFFFF000000000000ULL) | ((counter + i) & 0xFFFFFFFFFFFFULL);

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
        HIP_CHECK(oroMemset(d_scratch, 0, XELIS_MEMORY_SIZE_V3 * sizeof(uint64_t) * hashes_per_launch));

        // Stage 1: serial (xelis_stage1_kernel)
        printf("[TEST]   Running GPU Stage 1...\n");
        fflush(stdout);
        uint32_t s1_grid = (launch_batch_size + block_x - 1) / block_x;
        void *stage1_args[] = {&d_input, &d_scratch, (void*)&launch_nonce_start, (void*)&launch_batch_size, (void*)&launch_scratch_offset};
        HIP_CHECK(oroModuleLaunchKernel(stage1_func, s1_grid, 1, 1, block_x, 1, 1, 0, nullptr, stage1_args, nullptr));
        HIP_CHECK(oroDeviceSynchronize());

        // Stage 3: production s3_hybrid_v2_noblake
        printf("[TEST]   Running GPU Stage 3...\n");
        fflush(stdout);
        uint32_t s3_grid = (launch_batch_size + block_x - 1) / block_x;
        void *stage3_args[] = {&d_scratch, (void*)&launch_batch_size, (void*)&launch_scratch_offset, &d_difficulty, &d_solutions, (void*)&launch_nonce_start};
        HIP_CHECK(oroModuleLaunchKernel(stage3_func, s3_grid, 1, 1, block_x, 1, 1, 0, nullptr, stage3_args, nullptr));
        HIP_CHECK(oroDeviceSynchronize());

        // Blake3: serial (xelis_blake3_batch)
        printf("[TEST]   Running GPU Blake3...\n");
        fflush(stdout);
        uint32_t b3_grid = (launch_batch_size + 256 - 1) / 256;
        void *blake3_args[] = {&d_scratch, &d_output, (void*)&launch_batch_size, (void*)&launch_scratch_offset, &d_difficulty, &d_solutions, (void*)&launch_nonce_start};
        HIP_CHECK(oroModuleLaunchKernel(blake3_func, b3_grid, 1, 1, 256, 1, 1, 0, nullptr, blake3_args, nullptr));
        HIP_CHECK(oroDeviceSynchronize());

        // 3. Download and validate
        uint8_t *gpu_hashes = new uint8_t[32 * hashes_per_launch];
        HIP_CHECK(oroMemcpy(gpu_hashes, d_output, 32 * hashes_per_launch, oroMemcpyDeviceToHost));

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
    (void)oroFree((oroDeviceptr)d_input);
    (void)oroFree((oroDeviceptr)d_scratch);
    (void)oroFree((oroDeviceptr)d_output);
    (void)oroFree((oroDeviceptr)d_difficulty);
    (void)oroFree((oroDeviceptr)d_solutions);

    return all_pass ? 0 : 1;
}

static int test_xelis_hip_impl() {

    // 1. Initialize HIP device
    int deviceCount = 0;
    HIP_CHECK(oroGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "[ERROR] No GPU devices found\n");
        return 1;
    }

    printf("[TEST] Found %d GPU device(s)\n", deviceCount);

    oroDeviceProp_t props;
    HIP_CHECK(oroGetDeviceProperties(&props, tnn_get_device(0)));

    // Create GPU context on main thread — worker threads reuse via oroCtxSetCurrent
    oroCtx main_ctx;
    HIP_CHECK(oroCtxCreate(&main_ctx, 0, tnn_get_device(0)));

    printf("[TEST] Using device 0: %s\n", props.name);
    printf("[TEST] Compute capability: %d.%d\n", props.major, props.minor);

    bool is_amd = tnn_is_amd_device(0);

    if (is_amd) {
        printf("[TEST] Using arch=%s (AMD)\n\n", props.gcnArchName);
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "sm_%d%d", props.major, props.minor);
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
        if (props.gcnArchName[0] != '\0') {
            compile_opts.push_back(std::string("--gpu-architecture=") + props.gcnArchName);
        }
    } else {
        compile_opts = {"--dopt=on", "--use_fast_math"};
#ifdef __linux__
        compile_opts.push_back("--device-int128");
#endif
        compile_opts.push_back("-DXELIS_MIN_WG=32");
        compile_opts.push_back("-DXELIS_MAX_WG=128");
        {
            char arch_buf[32];
            std::snprintf(arch_buf, sizeof(arch_buf), "sm_%d%d", props.major, props.minor);
            compile_opts.push_back(std::string("--gpu-architecture=") + arch_buf);
        }

        // Arch-dependent stage3 register cap
        oroDeviceProp_t props;
        (void)oroGetDeviceProperties(&props, tnn_get_device(0));
        int s3_nreg = (props.major >= 9) ? 40 : (props.major >= 8) ? 64 : 80;
        compile_opts.push_back("-DXELIS_S3_NREG=" + std::to_string(s3_nreg));
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

    // Load kernel functions: serial stage1, production stage3, serial blake3
    oroFunction_t stage1_func = module_kernel.function;  // xelis_stage1_kernel
    oroFunction_t stage3_func = nullptr;
    oroFunction_t blake3_func = nullptr;

    oroError_t err = oroModuleGetFunction(&stage3_func, module_kernel.module, "xelis_s3_hybrid_v2_noblake_kernel");
    if (err != oroSuccess) {
        fprintf(stderr, "[ERROR] Failed to get stage3 kernel: %s\n", tnn_error_string(err));
        return 1;
    }
    printf("[TEST]   ✓ Loaded xelis_s3_hybrid_v2_noblake_kernel\n");
    fflush(stdout);

    err = oroModuleGetFunction(&blake3_func, module_kernel.module, "xelis_blake3_batch");
    if (err != oroSuccess) {
        fprintf(stderr, "[ERROR] Failed to get blake3 kernel: %s\n", tnn_error_string(err));
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

    // Test 1: Single hash (baseline)
    failures += run_test_case(
        "Test 1: Single Hash",
        stage1_func, stage3_func, blake3_func,
        0x0000000000000000ULL,  // nonce_start
        1,      // batch_size
        1       // s3_block_x
    );

    // Test 2: Small batch (32 hashes)
    failures += run_test_case(
        "Test 2: Small Batch (32)",
        stage1_func, stage3_func, blake3_func,
        0x0000000000000000ULL,  // nonce_start
        32,     // batch_size
        32      // s3_block_x
    );

    // Test 3: Medium batch with multiple launches (256 hashes × 20 launches)
    failures += run_test_case(
        "Test 3: Medium Batch (5120), 20 Launches",
        stage1_func, stage3_func, blake3_func,
        0x0000000000000000ULL,  // nonce_start
        5120,   // batch_size
        256,    // s3_block_x
        20      // num_launches
    );

    // Test 4: Non-zero nonce_start (device ID bits set)
    uint64_t device_nonce = (1ULL << 59);  // Device ID = 1
    failures += run_test_case(
        "Test 4: Device ID Segmentation (device=1)",
        stage1_func, stage3_func, blake3_func,
        device_nonce,  // nonce_start with device ID
        32,     // batch_size
        32      // s3_block_x
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


// ============================================================
// Threaded interference test: reproduce the real miner's environment
// Each level adds more of what getWork_v2 / miner main loop does
// ============================================================

// Level 1: GPU work runs on a spawned std::thread (not main thread)
static int test_threaded_level1(
    oroCtx gpu_ctx,
    oroFunction_t stage1_func,
    oroFunction_t stage3_func,
    oroFunction_t blake3_func)
{
    printf("\n[THREAD-TEST] Level 1: GPU kernels on std::thread\n");
    fflush(stdout);

    std::atomic<int> result{-1};
    std::thread gpu_thread([&]() {
        oroCtxSetCurrent(gpu_ctx);
        result = run_test_case(
            "Threaded: std::thread GPU",
            stage1_func, stage3_func, blake3_func,
            0x0000000000000000ULL, 32, 32);
    });
    gpu_thread.join();
    printf("[THREAD-TEST] Level 1: %s\n", result == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    return result;
}

// Level 2: boost::asio::io_context running on another thread alongside GPU
static int test_threaded_level2(
    oroCtx gpu_ctx,
    oroFunction_t stage1_func,
    oroFunction_t stage3_func,
    oroFunction_t blake3_func)
{
    printf("\n[THREAD-TEST] Level 2: GPU + boost::asio io_context on separate threads\n");
    fflush(stdout);

    std::atomic<bool> stop_asio{false};
    boost::asio::io_context ioc;
    auto work_guard = boost::asio::make_work_guard(ioc);

    // Simulate GETWORK thread: runs boost::asio io_context
    std::thread asio_thread([&]() {
        // Timer to keep io_context busy
        boost::asio::steady_timer timer(ioc);
        std::function<void(const boost::system::error_code&)> tick;
        tick = [&](const boost::system::error_code& ec) {
            if (!ec && !stop_asio.load()) {
                timer.expires_after(std::chrono::milliseconds(50));
                timer.async_wait(tick);
            }
        };
        timer.expires_after(std::chrono::milliseconds(50));
        timer.async_wait(tick);
        ioc.run();
    });

    // Give asio thread time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::atomic<int> result{-1};
    std::thread gpu_thread([&]() {
        oroCtxSetCurrent(gpu_ctx);
        result = run_test_case(
            "Threaded: GPU + asio io_context",
            stage1_func, stage3_func, blake3_func,
            0x0000000000000000ULL, 32, 32);
    });
    gpu_thread.join();

    stop_asio = true;
    work_guard.reset();
    ioc.stop();
    asio_thread.join();

    printf("[THREAD-TEST] Level 2: %s\n", result == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    return result;
}

// Level 3: boost::asio::spawn (Boost.Context coroutines) running alongside GPU
static int test_threaded_level3(
    oroCtx gpu_ctx,
    oroFunction_t stage1_func,
    oroFunction_t stage3_func,
    oroFunction_t blake3_func)
{
    printf("\n[THREAD-TEST] Level 3: GPU + boost::asio::spawn coroutine on separate threads\n");
    fflush(stdout);

    std::atomic<bool> stop_asio{false};
    boost::asio::io_context ioc;

    // Simulate getWork_v2: uses boost::asio::spawn (stackful coroutine via Boost.Context)
    std::thread asio_thread([&]() {
        boost::asio::spawn(ioc, [&](boost::asio::yield_context yield) {
            boost::asio::steady_timer timer(ioc);
            while (!stop_asio.load()) {
                timer.expires_after(std::chrono::milliseconds(50));
                boost::system::error_code ec;
                timer.async_wait(yield[ec]);
                if (ec) break;
            }
        },
        [](std::exception_ptr ex) {
            if (ex) {
                try { std::rethrow_exception(ex); }
                catch (const std::exception& e) {
                    fprintf(stderr, "[THREAD-TEST] spawn exception: %s\n", e.what());
                }
            }
        });
        ioc.run();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::atomic<int> result{-1};
    std::thread gpu_thread([&]() {
        oroCtxSetCurrent(gpu_ctx);
        result = run_test_case(
            "Threaded: GPU + boost::asio::spawn",
            stage1_func, stage3_func, blake3_func,
            0x0000000000000000ULL, 32, 32);
    });
    gpu_thread.join();

    stop_asio = true;
    ioc.stop();
    asio_thread.join();

    printf("[THREAD-TEST] Level 3: %s\n", result == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    return result;
}

// Level 4: SSL context + load_root_certificates alongside GPU
static int test_threaded_level4(
    oroCtx gpu_ctx,
    oroFunction_t stage1_func,
    oroFunction_t stage3_func,
    oroFunction_t blake3_func)
{
    printf("\n[THREAD-TEST] Level 4: GPU + SSL context + load_root_certificates\n");
    fflush(stdout);

    std::atomic<bool> stop_asio{false};
    boost::asio::io_context ioc;
    boost::asio::ssl::context ctx{boost::asio::ssl::context::tlsv12_client};
    load_root_certificates(ctx);

    std::thread asio_thread([&]() {
        boost::asio::spawn(ioc, [&](boost::asio::yield_context yield) {
            boost::asio::steady_timer timer(ioc);
            while (!stop_asio.load()) {
                timer.expires_after(std::chrono::milliseconds(50));
                boost::system::error_code ec;
                timer.async_wait(yield[ec]);
                if (ec) break;
            }
        },
        [](std::exception_ptr) {});
        ioc.run();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::atomic<int> result{-1};
    std::thread gpu_thread([&]() {
        oroCtxSetCurrent(gpu_ctx);
        result = run_test_case(
            "Threaded: GPU + SSL ctx",
            stage1_func, stage3_func, blake3_func,
            0x0000000000000000ULL, 32, 32);
    });
    gpu_thread.join();

    stop_asio = true;
    ioc.stop();
    asio_thread.join();

    printf("[THREAD-TEST] Level 4: %s\n", result == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    return result;
}

// Level 5: SSL + TCP resolver inside spawn coroutine (like do_session_v2)
static int test_threaded_level5(
    oroCtx gpu_ctx,
    oroFunction_t stage1_func,
    oroFunction_t stage3_func,
    oroFunction_t blake3_func)
{
    printf("\n[THREAD-TEST] Level 5: GPU + SSL + TCP resolve inside spawn\n");
    fflush(stdout);

    std::atomic<bool> stop_asio{false};
    boost::asio::io_context ioc;
    boost::asio::ssl::context ctx{boost::asio::ssl::context::tlsv12_client};
    load_root_certificates(ctx);

    std::thread asio_thread([&]() {
        boost::asio::spawn(ioc, [&](boost::asio::yield_context yield) {
            // Do a real DNS resolve like getWork_v2 -> do_session_v2 does
            boost::asio::ip::tcp::resolver resolver(ioc);
            boost::system::error_code ec;
            // Resolve localhost - always works, exercises the full async path
            auto results = resolver.async_resolve("127.0.0.1", "80", yield[ec]);
            // Don't care if it fails, we just want the side effects
            (void)results;

            boost::asio::steady_timer timer(ioc);
            while (!stop_asio.load()) {
                timer.expires_after(std::chrono::milliseconds(50));
                timer.async_wait(yield[ec]);
                if (ec) break;
            }
        },
        [](std::exception_ptr) {});
        ioc.run();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::atomic<int> result{-1};
    std::thread gpu_thread([&]() {
        oroCtxSetCurrent(gpu_ctx);
        result = run_test_case(
            "Threaded: GPU + SSL + resolve",
            stage1_func, stage3_func, blake3_func,
            0x0000000000000000ULL, 32, 32);
    });
    gpu_thread.join();

    stop_asio = true;
    ioc.stop();
    asio_thread.join();

    printf("[THREAD-TEST] Level 5: %s\n", result == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    return result;
}

// Level 6: Full miner sim - 2x getWork_v2-like threads + mutex/cv + SSL
static int test_threaded_level6(
    oroCtx gpu_ctx,
    oroFunction_t stage1_func,
    oroFunction_t stage3_func,
    oroFunction_t blake3_func)
{
    printf("\n[THREAD-TEST] Level 6: GPU + 2x SSL/spawn/resolve threads + mutex/cv\n");
    fflush(stdout);

    std::atomic<bool> stop_asio{false};
    std::mutex mtx;
    std::condition_variable cv_local;
    bool data_ready = false;

    auto make_getwork_thread = [&](boost::asio::io_context &ioc, const char *label) {
        boost::asio::ssl::context ctx{boost::asio::ssl::context::tlsv12_client};
        load_root_certificates(ctx);

        return std::thread([&ioc, &stop_asio, &mtx, &cv_local, &data_ready, label]() {
            boost::asio::spawn(ioc, [&](boost::asio::yield_context yield) {
                boost::asio::ip::tcp::resolver resolver(ioc);
                boost::system::error_code ec;
                resolver.async_resolve("127.0.0.1", "80", yield[ec]);

                boost::asio::steady_timer timer(ioc);
                while (!stop_asio.load()) {
                    timer.expires_after(std::chrono::milliseconds(50));
                    timer.async_wait(yield[ec]);
                    if (ec) break;
                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        data_ready = true;
                    }
                    cv_local.notify_all();
                }
            },
            [label](std::exception_ptr ex) {
                if (ex) {
                    try { std::rethrow_exception(ex); }
                    catch (const std::exception& e) {
                        fprintf(stderr, "[THREAD-TEST] %s spawn exception: %s\n", label, e.what());
                    }
                }
            });
            ioc.run();
        });
    };

    boost::asio::io_context ioc1, ioc2;
    auto getwork = make_getwork_thread(ioc1, "GETWORK");
    auto devwork = make_getwork_thread(ioc2, "DEVWORK");

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    std::atomic<int> result{-1};
    std::thread gpu_thread([&]() {
        oroCtxSetCurrent(gpu_ctx);
        result = run_test_case(
            "Threaded: full getWork sim",
            stage1_func, stage3_func, blake3_func,
            0x0000000000000000ULL, 32, 32);
    });
    gpu_thread.join();

    stop_asio = true;
    ioc1.stop();
    ioc2.stop();
    getwork.join();
    devwork.join();

    printf("[THREAD-TEST] Level 6: %s\n", result == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    return result;
}

// ============================================================
// Profile: execute_op cost breakdown (stubbed vs real)
// ============================================================

static int run_profile_ops() {
    printf("\n========================================\n");
    printf("[PROFILE] execute_op cost breakdown\n");
    printf("========================================\n\n");
    fflush(stdout);

    oroDeviceProp_t props;
    HIP_CHECK(oroGetDeviceProperties(&props, tnn_get_device(0)));

    bool is_amd = tnn_is_amd_device(0);

    // Compile with XELIS_PROFILE_OPS enabled
    auto& compiler = RTCCompiler::instance();
    const std::string xelis_source =
        std::string(hip_xelis_v3_source::SRC_TNN_HIP_CRYPTO_XELIS_HASH_XELIS_HASH_V3_HIP_SOURCE);

    auto rtc_headers = build_rtc_headers(
        hip_embedded::XELIS_SOURCES,
        hip_embedded::COMMON_HEADERS
    );
    for (const auto& h : rtc_headers) {
        compiler.add_header_source(std::string(h.name), std::string(h.source));
    }

    std::vector<std::string> compile_opts;
    if (is_amd) {
        compile_opts = {"-O3", "-mno-cumode", "-ffast-math"};
        compile_opts.push_back("-DXELIS_MIN_WG=64");
        compile_opts.push_back("-DXELIS_MAX_WG=256");
        if (props.gcnArchName[0] != '\0')
            compile_opts.push_back(std::string("--gpu-architecture=") + props.gcnArchName);
    } else {
        compile_opts = {"--dopt=on", "--use_fast_math"};
#ifdef __linux__
        compile_opts.push_back("--device-int128");
#endif
        compile_opts.push_back("-DXELIS_MIN_WG=32");
        compile_opts.push_back("-DXELIS_MAX_WG=128");
        {
            char arch_buf[32];
            std::snprintf(arch_buf, sizeof(arch_buf), "sm_%d%d", props.major, props.minor);
            compile_opts.push_back(std::string("--gpu-architecture=") + arch_buf);
        }
        int s3_nreg = (props.major >= 9) ? 40 : (props.major >= 8) ? 64 : 80;
        compile_opts.push_back("-DXELIS_S3_NREG=" + std::to_string(s3_nreg));
    }

    // Enable the profiling kernel
    compile_opts.push_back("-DXELIS_PROFILE_OPS");

    printf("[PROFILE] Compiling with XELIS_PROFILE_OPS...\n");
    fflush(stdout);

    auto module_kernel = compiler.compile_from_source(
        xelis_source, "xelis-hash-v3.hip", "xelis_stage1_kernel", compile_opts);

    oroFunction_t stage1_func = module_kernel.function;
    oroFunction_t profile_func = nullptr;

    oroError_t err = oroModuleGetFunction(&profile_func, module_kernel.module, "xelis_profile_ops_kernel");
    if (err != oroSuccess) {
        fprintf(stderr, "[ERROR] Failed to get xelis_profile_ops_kernel: %s\n", tnn_error_string(err));
        return 1;
    }
    printf("[PROFILE] Kernels loaded\n\n");
    fflush(stdout);

    // Allocate GPU memory
    const uint32_t block_x = is_amd ? 64 : 128;
    const uint32_t batch_size = block_x;  // 1 block

    uint8_t *d_input = nullptr;
    uint64_t *d_scratch = nullptr;
    uint64_t *d_profile = nullptr;

    HIP_CHECK(oroMalloc((oroDeviceptr*)&d_input, XELIS_TEMPLATE_SIZE));
    HIP_CHECK(oroMalloc((oroDeviceptr*)&d_scratch, XELIS_MEMORY_SIZE_V3 * sizeof(uint64_t) * batch_size));
    HIP_CHECK(oroMalloc((oroDeviceptr*)&d_profile, 4 * sizeof(uint64_t)));

    HIP_CHECK(oroMemcpy(d_input, TEST_WORK, XELIS_TEMPLATE_SIZE, oroMemcpyHostToDevice));
    HIP_CHECK(oroMemset(d_scratch, 0, XELIS_MEMORY_SIZE_V3 * sizeof(uint64_t) * batch_size));
    HIP_CHECK(oroMemset(d_profile, 0, 4 * sizeof(uint64_t)));

    // Run stage1 to populate scratchpad
    printf("[PROFILE] Running Stage 1 (populate scratchpad)...\n");
    fflush(stdout);
    uint64_t nonce_start = 0x0000000000000000ULL;
    uint32_t scratch_offset = 0;
    void *s1_args[] = {&d_input, &d_scratch, (void*)&nonce_start, (void*)&batch_size, (void*)&scratch_offset};
    HIP_CHECK(oroModuleLaunchKernel(stage1_func, 1, 1, 1, block_x, 1, 1, 0, nullptr, s1_args, nullptr));
    HIP_CHECK(oroDeviceSynchronize());

    // Run profile kernel
    printf("[PROFILE] Running profile kernel (%u threads, 1 block)...\n", block_x);
    fflush(stdout);
    void *prof_args[] = {&d_scratch, &d_profile, (void*)&scratch_offset};
    HIP_CHECK(oroModuleLaunchKernel(profile_func, 1, 1, 1, block_x, 1, 1, 0, nullptr, prof_args, nullptr));
    HIP_CHECK(oroDeviceSynchronize());

    // Read results
    uint64_t results[4];
    HIP_CHECK(oroMemcpy(results, d_profile, 4 * sizeof(uint64_t), oroMemcpyDeviceToHost));

    uint64_t stub_cycles = results[0];
    uint64_t real_cycles = results[1];
    uint64_t expensive_count = results[2];
    uint64_t total_count = results[3];

    uint64_t delta = real_cycles - stub_cycles;
    double cheap_pct = 100.0 * (total_count - expensive_count) / total_count;
    double expensive_pct = 100.0 * expensive_count / total_count;

    printf("\n========================================\n");
    printf("[PROFILE] execute_op cost breakdown (per-thread avg, warp 0)\n");
    printf("========================================\n");
    printf("  Total ops:            %llu\n", (unsigned long long)total_count);
    printf("  Cheap ops:            %llu (%.1f%%)\n",
           (unsigned long long)(total_count - expensive_count), cheap_pct);
    printf("  Expensive ops:        %llu (%.1f%%)\n",
           (unsigned long long)expensive_count, expensive_pct);
    printf("\n");
    printf("  Stub cycles (total):  %llu\n", (unsigned long long)stub_cycles);
    printf("  Real cycles (total):  %llu\n", (unsigned long long)real_cycles);
    printf("  Delta (expensive):    %llu cycles\n", (unsigned long long)delta);
    printf("\n");
    if (expensive_count > 0)
        printf("  Avg per expensive op: %llu cycles\n", (unsigned long long)(delta / expensive_count));
    if (total_count > 0) {
        printf("  Avg per op (stub):    %llu cycles\n", (unsigned long long)(stub_cycles / total_count));
        printf("  Avg per op (real):    %llu cycles\n", (unsigned long long)(real_cycles / total_count));
    }
    printf("\n");
    printf("  Expensive ops cost:   %.1f%% of total execute_op time\n",
           100.0 * delta / real_cycles);
    printf("  Cheap ops cost:       %.1f%% of total execute_op time\n",
           100.0 * stub_cycles / real_cycles);
    printf("========================================\n\n");
    fflush(stdout);

    (void)oroFree((oroDeviceptr)d_input);
    (void)oroFree((oroDeviceptr)d_scratch);
    (void)oroFree((oroDeviceptr)d_profile);

    return 0;
}

int test_xelis_hip() {
    printf("\n");
    printf("========================================\n");
    printf("[hip-test-xelis] Starting stage-by-stage GPU validation\n");
    printf("========================================\n\n");
    fflush(stdout);

    try {
        int ret = test_xelis_hip_impl();
        if (ret != 0) return ret;

        // Now run threaded interference tests with the same compiled kernels
        // Re-init for threaded tests (reuse the compiled module)
        printf("\n\n");
        printf("========================================\n");
        printf("[THREAD-TEST] Threaded Interference Tests\n");
        printf("========================================\n");
        printf("Each level adds more miner-like concurrency.\n");
        printf("If a level crashes, that's our interference source.\n\n");
        fflush(stdout);

        // Re-compile (or we could refactor to share, but this is a test)
        auto& compiler = RTCCompiler::instance();
        const std::string xelis_source =
            std::string(hip_xelis_v3_source::SRC_TNN_HIP_CRYPTO_XELIS_HASH_XELIS_HASH_V3_HIP_SOURCE);

        oroDeviceProp_t props;
        (void)oroGetDeviceProperties(&props, tnn_get_device(0));

        bool is_amd = tnn_is_amd_device(0);

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
            int s3_nreg = (props.major >= 9) ? 40 : (props.major >= 8) ? 64 : 80;
            compile_opts.push_back("-DXELIS_S3_NREG=" + std::to_string(s3_nreg));
        }

        auto module_kernel = compiler.compile_from_source(
            xelis_source, "xelis-hash-v3.hip", "xelis_stage1_kernel", compile_opts);

        // Get current context for worker threads to reuse
        oroCtx main_ctx;
        oroCtxGetCurrent(&main_ctx);

        oroFunction_t stage1_func = module_kernel.function;  // xelis_stage1_kernel
        oroFunction_t stage3_func = nullptr, blake3_func = nullptr;
        (void)oroModuleGetFunction(&stage3_func, module_kernel.module, "xelis_s3_hybrid_v2_noblake_kernel");
        (void)oroModuleGetFunction(&blake3_func, module_kernel.module, "xelis_blake3_batch");

        int failures = 0;
        failures += test_threaded_level1(main_ctx, stage1_func, stage3_func, blake3_func);
        if (failures) { printf("\n[THREAD-TEST] STOPPED at Level 1\n"); return 1; }

        failures += test_threaded_level2(main_ctx, stage1_func, stage3_func, blake3_func);
        if (failures) { printf("\n[THREAD-TEST] STOPPED at Level 2\n"); return 1; }

        failures += test_threaded_level3(main_ctx, stage1_func, stage3_func, blake3_func);
        if (failures) { printf("\n[THREAD-TEST] STOPPED at Level 3\n"); return 1; }

        failures += test_threaded_level4(main_ctx, stage1_func, stage3_func, blake3_func);
        if (failures) { printf("\n[THREAD-TEST] STOPPED at Level 4\n"); return 1; }

        failures += test_threaded_level5(main_ctx, stage1_func, stage3_func, blake3_func);
        if (failures) { printf("\n[THREAD-TEST] STOPPED at Level 5\n"); return 1; }

        failures += test_threaded_level6(main_ctx, stage1_func, stage3_func, blake3_func);
        if (failures) { printf("\n[THREAD-TEST] STOPPED at Level 6\n"); return 1; }

        printf("\n========================================\n");
        printf("[THREAD-TEST] All 6 levels passed!\n");
        printf("========================================\n");

        // Run execute_op profiling
        run_profile_ops();

        return 0;

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
