#ifdef TNN_KAWPOW

#include <coins/miners.hpp>
#include <net/net.hpp>
#include <stratum/kawpow-stratum.h>
#include <hex.h>
#include <algo_definitions.h>
#include <tnn_log.hpp>

#include "../../common/gpu_compat.hpp"
#include "../../common/gpu_rtc.hpp"
#include "../../common/gpu_miner.hpp"
#include "../../common/hip_algo_registry.hpp"
#include "../../common/gpu_submit_queue.hpp"
#include "../../common/gpu_device_filter.hpp"
#include <job_safe.hpp>
#include "../../crypto/kawpow/kawpow_proggen.hpp"
#include <kawpow.hip.hpp>
#include <kawpow_embedded_headers.hpp>
#include <ethash-dag-gen.hip.hpp>
#include <ethash/ethash.hpp>
#include <ethash/progpow.hpp>
#include <ethash/ethash-internal.hpp>
#include <ethash/kawpow_coins.h>

#include <chrono>
#include <vector>
#include <cstring>
#include <cstdio>
#include <string>
#include <thread>

// ============================================================================
// Helpers
// ============================================================================

#define KP_CHECK(expr)                                                       \
    do {                                                                      \
        oroError_t _e = (expr);                                               \
        if (_e != oroSuccess) {                                               \
            fprintf(stderr, "[KawPow] GPU error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, tnn_error_string(_e));                 \
            return -1;                                                        \
        }                                                                     \
    } while (0)

#define KP_CHECK_VOID(expr)                                                  \
    do {                                                                      \
        oroError_t _e = (expr);                                               \
        if (_e != oroSuccess) {                                               \
            fprintf(stderr, "[KawPow] GPU error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, tnn_error_string(_e));                 \
            return;                                                           \
        }                                                                     \
    } while (0)

static std::string hash256_hex(const ethash::hash256& h)
{
    static const char hx[] = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (int i = 0; i < 32; i++) {
        s.push_back(hx[h.bytes[i] >> 4]);
        s.push_back(hx[h.bytes[i] & 0xf]);
    }
    return s;
}

static std::string words_hex(const uint32_t* w, int n)
{
    static const char hx[] = "0123456789abcdef";
    std::string s;
    s.reserve(n * 8);
    for (int i = 0; i < n; i++) {
        uint32_t v = w[i];
        for (int b = 0; b < 4; b++) {
            uint8_t byte = (uint8_t)(v & 0xFF);
            s.push_back(hx[byte >> 4]);
            s.push_back(hx[byte & 0xf]);
            v >>= 8;
        }
    }
    return s;
}

// ============================================================================
// Compile KawPow kernel with program + padding injection
// ============================================================================

static int compile_kawpow_kernel(
    int block_number,
    const kawpow_coin_padding_t& padding,
    bool is_amd,
    const oroDeviceProp_t& props,
    oroFunction_t& out_func,
    oroModule_t* out_module = nullptr,
    uint32_t dag_num_items_div = 1,
    uint32_t barrett_m = 1,
    uint32_t barrett_shift = 0)
{
    std::string src(hip_kawpow_source::SRC_TNN_HIP_CRYPTO_KAWPOW_KAWPOW_HIP_SOURCE);
    src = kawpow_proggen::inject_coin_padding(src, padding);
    src = kawpow_proggen::inject_dag_constants(src, dag_num_items_div, barrett_m, barrett_shift);
    src = kawpow_proggen::inject_program(src, block_number);

    auto& compiler = RTCCompiler::instance();

    std::vector<std::string> opts;
    if (is_amd) {
        opts = {"-O3", "-mno-cumode", "-ffast-math"};
        if (props.gcnArchName[0] != '\0')
            opts.push_back(std::string("--gpu-architecture=") + props.gcnArchName);
    } else {
        opts = {"--dopt=on", "--use_fast_math"};
        char buf[32];
        std::snprintf(buf, sizeof(buf), "sm_%d%d", props.major, props.minor);
        opts.push_back(std::string("--gpu-architecture=") + buf);
    }

    uint64_t period = (uint64_t)block_number / 3;
    std::string name = "kawpow_p" + std::to_string(period) + ".hip";

    auto ck = compiler.compile_from_source(src, name, "kawpow_hash_kernel", opts);
    out_func = ck.function;
    if (out_module) *out_module = ck.module;
    return 0;
}

// ============================================================================
// GPU Test: compile kernel, generate DAG on GPU, compare GPU vs CPU reference
// ============================================================================

struct KawPowDAG {
    uint32_t* d_dag = nullptr;        // device pointer
    uint32_t* d_l1_cache = nullptr;   // device pointer (first 16KB of DAG)
    uint32_t  num_items_2048;
    uint32_t  dag_num_items_div;
    size_t    size_bytes;
};

struct KPTestVec {
    int         block_number;
    const char* header_hex;
    const char* nonce_hex;
};

static const KPTestVec gpu_test_vecs[] = {
    {0,  "0000000000000000000000000000000000000000000000000000000000000000", "0000000000000000"},
    {49, "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b", "0000000007073c07"},
    {50, "9e7248f20914913a73d80a70174c331b1d34f260535ac3631d770e656b5dd922", "00000000076e482e"},
    {99, "de37e1824c86d35d154cf65a88de6d9286aec4f7f10c3fc9f0fa1bcc2687188d", "000000003917afab"},
};
static const int NUM_GPU_TESTS = sizeof(gpu_test_vecs) / sizeof(gpu_test_vecs[0]);

static ethash::hash256 hex_to_h256(const char* hex)
{
    ethash::hash256 h = {};
    for (size_t i = 0; i < 64 && hex[i] && hex[i + 1]; i += 2) {
        auto nib = [](char c) -> int { return c <= '9' ? (c - '0') : (c - 'a' + 10); };
        h.bytes[i / 2] = (uint8_t)((nib(hex[i]) << 4) | nib(hex[i + 1]));
    }
    return h;
}

static uint64_t hex_to_u64(const char* hex)
{
    uint64_t v = 0;
    for (int i = 0; hex[i]; i++) {
        v <<= 4;
        v |= (hex[i] <= '9') ? (hex[i] - '0') : (hex[i] - 'a' + 10);
    }
    return v;
}

int kawpow_gpu_test()
{
    printf("\n========================================\n");
    printf("[KawPow] GPU Kernel Validation\n");
    printf("========================================\n\n");
    fflush(stdout);

    // ---- Init GPU ----
    int devCount = 0;
    KP_CHECK(oroGetDeviceCount(&devCount));
    if (devCount == 0) { fprintf(stderr, "No GPU found\n"); return 1; }

    oroDeviceProp_t props;
    KP_CHECK(oroGetDeviceProperties(&props, tnn_get_device(0)));

    oroCtx gpu_ctx;
    KP_CHECK(oroCtxCreate(&gpu_ctx, 0, tnn_get_device(0)));

    bool is_amd = tnn_is_amd_device(0);
    printf("[KawPow] Device: %s (%s)\n\n", props.name,
           is_amd ? "AMD" : "NVIDIA");
    fflush(stdout);

    // ---- Generate DAG on GPU (epoch 0) ----
    const int epoch = 0;
    auto ctx = ethash::create_epoch_context(epoch);
    if (!ctx) { fprintf(stderr, "Failed to create epoch context\n"); return 1; }

    KawPowDAG dag;
    dag.num_items_2048    = (uint32_t)(ctx->full_dataset_num_items / 2);
    dag.dag_num_items_div = dag.num_items_2048;
    size_t dag_words      = (size_t)dag.num_items_2048 * 64;
    dag.size_bytes        = dag_words * sizeof(uint32_t);

    printf("[KawPow] Generating DAG on GPU for epoch %d: %u items (%.1f MB)...\n",
           epoch, dag.num_items_2048, dag.size_bytes / (1024.0 * 1024.0));
    fflush(stdout);

    // Allocate DAG on GPU
    KP_CHECK(oroMalloc((oroDeviceptr*)&dag.d_dag, dag.size_bytes));

    // Upload light cache
    uint32_t num_cache_items = (uint32_t)ctx->light_cache_num_items;
    size_t cache_bytes = (size_t)num_cache_items * 64;
    uint32_t* d_light_cache = nullptr;
    KP_CHECK(oroMalloc((oroDeviceptr*)&d_light_cache, cache_bytes));
    KP_CHECK(oroMemcpy(d_light_cache, ctx->light_cache, cache_bytes, oroMemcpyHostToDevice));

    printf("[KawPow] Light cache uploaded (%.1f MB, %u items)\n",
           cache_bytes / (1024.0 * 1024.0), num_cache_items);
    fflush(stdout);

    // Compile DAG gen kernel
    auto& compiler = RTCCompiler::instance();
    for (const auto& h : hip_embedded::KAWPOW_HEADERS)
        compiler.add_header_source(std::string(h.path), std::string(h.source));
    for (const auto& h : hip_embedded::COMMON_HEADERS)
        compiler.add_header_source(std::string(h.path), std::string(h.source));

    std::vector<std::string> dag_opts;
    if (is_amd) {
        dag_opts = {"-O3", "-mno-cumode", "-ffast-math"};
        if (props.gcnArchName[0] != '\0')
            dag_opts.push_back(std::string("--gpu-architecture=") + props.gcnArchName);
    } else {
        dag_opts = {"--dopt=on", "--use_fast_math"};
        char buf[32];
        std::snprintf(buf, sizeof(buf), "sm_%d%d", props.major, props.minor);
        dag_opts.push_back(std::string("--gpu-architecture=") + buf);
    }

    std::string dag_src(hip_ethash_dag_source::SRC_TNN_HIP_CRYPTO_KAWPOW_ETHASH_DAG_GEN_HIP_SOURCE);
    auto dag_ck = compiler.compile_from_source(dag_src, "ethash-dag-gen.hip",
                                                "ethash_dag_gen_kernel", dag_opts);
    if (!dag_ck.function) {
        fprintf(stderr, "[KawPow] DAG gen kernel compile failed\n");
        oroFree((oroDeviceptr)d_light_cache);
        oroFree((oroDeviceptr)dag.d_dag);
        return 1;
    }

    // Launch DAG gen kernel
    auto t0 = std::chrono::steady_clock::now();
    {
        uint32_t dag_items = dag.num_items_2048;
        int block_size = 256;
        int grid_size = (dag_items + block_size - 1) / block_size;
        void* args[] = { &d_light_cache, &dag.d_dag, &num_cache_items, &dag_items };
        KP_CHECK(oroModuleLaunchKernel(dag_ck.function, grid_size, 1, 1, block_size, 1, 1,
                                        0, nullptr, args, nullptr));
        KP_CHECK(oroDeviceSynchronize());
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("[KawPow] DAG generated on GPU in %.2fs\n",
           std::chrono::duration<double>(t1 - t0).count());
    fflush(stdout);

    oroFree((oroDeviceptr)d_light_cache);

    // ---- Validate GPU DAG against CPU reference ----
    printf("\n[KawPow] Phase 1: DAG Validation (GPU vs CPU reference)\n");
    printf("----------------------------------------\n");
    fflush(stdout);

    // Spot-check items at various positions
    const uint32_t check_indices[] = { 0, 1, 2, 100, 1000, dag.num_items_2048 / 2, dag.num_items_2048 - 1 };
    const int num_checks = sizeof(check_indices) / sizeof(check_indices[0]);
    int dag_pass = 0, dag_fail = 0;

    for (int c = 0; c < num_checks; c++) {
        uint32_t item_idx = check_indices[c];
        if (item_idx >= dag.num_items_2048) continue;

        // GPU: download this 2048-bit item (64 uint32)
        uint32_t gpu_item[64];
        KP_CHECK(oroMemcpy(gpu_item, (uint8_t*)dag.d_dag + (size_t)item_idx * 256,
                            256, oroMemcpyDeviceToHost));

        // CPU: compute reference
        ethash::hash2048 cpu_item = ethash::calculate_dataset_item_2048(*ctx, item_idx);

        if (memcmp(gpu_item, &cpu_item, 256) == 0) {
            printf("  [DAG %u] \033[32mPASS\033[0m\n", item_idx);
            dag_pass++;
        } else {
            printf("  [DAG %u] \033[31mFAIL\033[0m\n", item_idx);
            printf("    GPU: %s...\n", words_hex(gpu_item, 4).c_str());
            printf("    CPU: %s...\n", words_hex((const uint32_t*)&cpu_item, 4).c_str());
            dag_fail++;
        }
    }
    fflush(stdout);

    if (dag_fail > 0) {
        fprintf(stderr, "\n[KawPow] DAG validation FAILED (%d/%d), aborting kernel tests\n",
                dag_fail, dag_pass + dag_fail);
        oroFree((oroDeviceptr)dag.d_dag);
        oroCtxDestroy(gpu_ctx);
        return 1;
    }
    printf("  DAG validation: %d/%d \033[32mpassed\033[0m\n\n", dag_pass, dag_pass);
    fflush(stdout);

    // ---- L1 cache: first 16KB of DAG ----
    KP_CHECK(oroMalloc((oroDeviceptr*)&dag.d_l1_cache, 16384));
    KP_CHECK(oroMemcpy(dag.d_l1_cache, dag.d_dag, 16384, oroMemcpyDeviceToDevice));

    uint32_t* d_dag = dag.d_dag;
    uint32_t* d_l1_cache = dag.d_l1_cache;

    // ---- Allocate GPU buffers ----
    uint32_t* d_header    = nullptr;
    uint64_t* d_target    = nullptr;
    uint64_t* d_solutions = nullptr;
    uint32_t* d_results   = nullptr;

    KP_CHECK(oroMalloc((oroDeviceptr*)&d_header,    32));
    KP_CHECK(oroMalloc((oroDeviceptr*)&d_target,    32));
    KP_CHECK(oroMalloc((oroDeviceptr*)&d_solutions, 8 + 1024 * 40));
    KP_CHECK(oroMalloc((oroDeviceptr*)&d_results,   16 * sizeof(uint32_t)));

    // Set target to all-1s (accept everything)
    uint64_t max_target[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    KP_CHECK(oroMemcpy(d_target, max_target, 32, oroMemcpyHostToDevice));

    // ---- Phase 2: Kernel tests (mono + split) ----
    printf("[KawPow] Phase 2: ProgPoW Kernel Validation\n");
    printf("----------------------------------------\n");
    fflush(stdout);

    // Barrett constants for compile-time injection
    uint32_t dag_div = dag.dag_num_items_div;
    uint32_t barrett_m = (uint32_t)((1ULL << 32) / dag_div);
    uint32_t barrett_s = 0;

    int pass = 0, fail = 0;
    int last_period = -1;
    oroFunction_t kernel_func = nullptr;

    for (int t = 0; t < NUM_GPU_TESTS; t++) {
        const auto& tv = gpu_test_vecs[t];
        int period = tv.block_number / 3;

        printf("[GPU %d] block=%d period=%d nonce=%s\n",
               t + 1, tv.block_number, period, tv.nonce_hex);
        fflush(stdout);

        // Recompile if period changed
        if (period != last_period) {
            printf("         Compiling kernel for period %d...\n", period);
            fflush(stdout);
            if (compile_kawpow_kernel(tv.block_number, *currentKawpowPadding,
                                      is_amd, props, kernel_func, nullptr,
                                      dag_div, barrett_m, barrett_s) != 0) {
                fprintf(stderr, "         Compile failed!\n");
                fail++;
                continue;
            }
            last_period = period;
        }

        // CPU reference
        ethash::hash256 header = hex_to_h256(tv.header_hex);
        uint64_t nonce = hex_to_u64(tv.nonce_hex);
        auto cpu_result = progpow::hash(*ctx, tv.block_number, header, nonce);

        // Upload header
        KP_CHECK(oroMemcpy(d_header, header.word32s, 32, oroMemcpyHostToDevice));

        // Clear solutions + results
        uint64_t zero = 0;
        KP_CHECK(oroMemcpy(d_solutions, &zero, 8, oroMemcpyHostToDevice));
        uint32_t zeros[16] = {};
        KP_CHECK(oroMemcpy(d_results, zeros, 64, oroMemcpyHostToDevice));

        // Launch: 1 hash = 16 threads
        uint32_t block_size  = 16;
        uint32_t grid_size   = 1;
        uint32_t batch_size  = 1;
        uint32_t block_num   = (uint32_t)tv.block_number;

        void* args[] = {
            &d_header, &d_dag, &nonce, &batch_size,
            &d_target, &d_solutions, &block_num, &d_results,
            &d_l1_cache,
        };

        size_t smem = kawpow_shared_mem(block_size);
        KP_CHECK(oroModuleLaunchKernel(
            kernel_func, grid_size, 1, 1, block_size, 1, 1,
            smem, nullptr, args, nullptr));
        KP_CHECK(oroDeviceSynchronize());

        // Download results (16 uint32: 8 mix + 8 final)
        uint32_t gpu_out[16];
        KP_CHECK(oroMemcpy(gpu_out, d_results, 64, oroMemcpyDeviceToHost));

        // Compare
        std::string cpu_mix_hex   = hash256_hex(cpu_result.mix_hash);
        std::string cpu_final_hex = hash256_hex(cpu_result.final_hash);
        std::string gpu_mix_hex   = words_hex(gpu_out, 8);
        std::string gpu_final_hex = words_hex(gpu_out + 8, 8);

        bool mix_ok   = (cpu_mix_hex == gpu_mix_hex);
        bool final_ok = (cpu_final_hex == gpu_final_hex);

        if (mix_ok && final_ok) {
            printf("         \033[32mPASS\033[0m mix=%s\n", gpu_mix_hex.c_str());
            pass++;
        } else {
            printf("         \033[31mFAIL\033[0m\n");
            if (!mix_ok) {
                printf("           mix cpu: %s\n", cpu_mix_hex.c_str());
                printf("           mix gpu: %s\n", gpu_mix_hex.c_str());
            }
            if (!final_ok) {
                printf("           fin cpu: %s\n", cpu_final_hex.c_str());
                printf("           fin gpu: %s\n", gpu_final_hex.c_str());
            }
            fail++;
        }
        fflush(stdout);
    }

    // ================================================================
    // Split strategy tests — same test vectors, 3-kernel pipeline
    // ================================================================
    printf("\n[KawPow] Split strategy validation\n");
    printf("----------------------------------------\n");
    fflush(stdout);

    int split_pass = 0, split_fail = 0;
    last_period = -1;
    oroFunction_t seed_fn = nullptr, progpow_fn = nullptr, final_fn = nullptr;
    oroModule_t split_module = nullptr;

    // Intermediate buffer: 16 uint32 per hash
    uint32_t* d_intermediate = nullptr;
    KP_CHECK(oroMalloc((oroDeviceptr*)&d_intermediate, 16 * sizeof(uint32_t)));

    for (int t = 0; t < NUM_GPU_TESTS; t++) {
        const auto& tv = gpu_test_vecs[t];
        int period = tv.block_number / 3;

        printf("[Split %d] block=%d period=%d nonce=%s\n",
               t + 1, tv.block_number, period, tv.nonce_hex);
        fflush(stdout);

        // Recompile if period changed
        if (period != last_period) {
            oroFunction_t mono_fn = nullptr;
            printf("         Compiling kernel for period %d...\n", period);
            fflush(stdout);
            if (compile_kawpow_kernel(tv.block_number, *currentKawpowPadding,
                                      is_amd, props, mono_fn, &split_module,
                                      dag_div, barrett_m, barrett_s) != 0) {
                fprintf(stderr, "         Compile failed!\n");
                split_fail++;
                continue;
            }
            // Extract split kernels from the same module
            oroModuleGetFunction(&seed_fn,    split_module, "kawpow_seed_kernel");
            oroModuleGetFunction(&progpow_fn, split_module, "kawpow_progpow_kernel");
            oroModuleGetFunction(&final_fn,   split_module, "kawpow_final_kernel");

            if (!seed_fn || !progpow_fn || !final_fn) {
                fprintf(stderr, "         Failed to extract split kernels!\n");
                split_fail++;
                continue;
            }
            last_period = period;
        }

        // CPU reference
        ethash::hash256 header = hex_to_h256(tv.header_hex);
        uint64_t nonce = hex_to_u64(tv.nonce_hex);
        auto cpu_result = progpow::hash(*ctx, tv.block_number, header, nonce);

        // Upload header
        KP_CHECK(oroMemcpy(d_header, header.word32s, 32, oroMemcpyHostToDevice));

        // Clear solutions + results + intermediate
        uint64_t zero = 0;
        KP_CHECK(oroMemcpy(d_solutions, &zero, 8, oroMemcpyHostToDevice));
        uint32_t zeros[16] = {};
        KP_CHECK(oroMemcpy(d_results, zeros, 64, oroMemcpyHostToDevice));
        KP_CHECK(oroMemset((oroDeviceptr)d_intermediate, 0, 64));

        uint32_t batch_size = 1;

        // Seed kernel: 1 thread per hash
        {
            void* args[] = { &d_header, &nonce, &batch_size, &d_intermediate };
            KP_CHECK(oroModuleLaunchKernel(seed_fn, 1, 1, 1, 1, 1, 1,
                                           0, nullptr, args, nullptr));
            KP_CHECK(oroDeviceSynchronize());
        }

        // Progpow kernel: 16 threads per hash (1 block of 16)
        {
            void* args[] = { &d_dag, &batch_size, &d_l1_cache, &d_intermediate };
            KP_CHECK(oroModuleLaunchKernel(progpow_fn, 1, 1, 1, 16, 1, 1,
                                           kawpow_shared_mem_split(16), nullptr, args, nullptr));
            KP_CHECK(oroDeviceSynchronize());
        }

        // Final kernel: 1 thread per hash
        {
            void* args[] = { &nonce, &batch_size, &d_target, &d_solutions,
                             &d_results, &d_intermediate };
            KP_CHECK(oroModuleLaunchKernel(final_fn, 1, 1, 1, 1, 1, 1,
                                           0, nullptr, args, nullptr));
            KP_CHECK(oroDeviceSynchronize());
        }

        // Download results (16 uint32: 8 mix + 8 final)
        uint32_t gpu_out[16];
        KP_CHECK(oroMemcpy(gpu_out, d_results, 64, oroMemcpyDeviceToHost));

        // Compare
        std::string cpu_mix_hex   = hash256_hex(cpu_result.mix_hash);
        std::string cpu_final_hex = hash256_hex(cpu_result.final_hash);
        std::string gpu_mix_hex   = words_hex(gpu_out, 8);
        std::string gpu_final_hex = words_hex(gpu_out + 8, 8);

        bool mix_ok   = (cpu_mix_hex == gpu_mix_hex);
        bool final_ok = (cpu_final_hex == gpu_final_hex);

        if (mix_ok && final_ok) {
            printf("         \033[32mPASS\033[0m mix=%s\n", gpu_mix_hex.c_str());
            split_pass++;
        } else {
            printf("         \033[31mFAIL\033[0m\n");
            if (!mix_ok) {
                printf("           mix cpu: %s\n", cpu_mix_hex.c_str());
                printf("           mix gpu: %s\n", gpu_mix_hex.c_str());
            }
            if (!final_ok) {
                printf("           fin cpu: %s\n", cpu_final_hex.c_str());
                printf("           fin gpu: %s\n", gpu_final_hex.c_str());
            }
            split_fail++;
        }
        fflush(stdout);
    }

    pass += split_pass;
    fail += split_fail;
    int total_kernel_tests = NUM_GPU_TESTS * 2; // mono + split

    // Cleanup
    oroFree((oroDeviceptr)d_intermediate);
    oroFree((oroDeviceptr)d_dag);
    oroFree((oroDeviceptr)d_l1_cache);
    oroFree((oroDeviceptr)d_header);
    oroFree((oroDeviceptr)d_target);
    oroFree((oroDeviceptr)d_solutions);
    oroFree((oroDeviceptr)d_results);
    oroCtxDestroy(gpu_ctx);

    printf("\n========================================\n");
    printf("[GPU] DAG: %d/%d, Kernel: %d/%d (mono: %d/%d, split: %d/%d)",
           dag_pass, dag_pass, pass, total_kernel_tests,
           pass - split_pass, NUM_GPU_TESTS,
           split_pass, NUM_GPU_TESTS);
    if (fail > 0) printf(", \033[31m%d FAILED\033[0m", fail);
    else          printf(", \033[32mall passed\033[0m");
    printf("\n========================================\n\n");
    fflush(stdout);

    return (fail == 0) ? 0 : 1;
}

// ============================================================================
// GPU Benchmark — uses generic autotune (compile + DAG setup + block size sweep)
// ============================================================================

void kawpow_bench(int block_height)
{
    if (block_height > 0) {
        kawpow_bench_block_override() = block_height;
        int epoch = ethash::get_epoch_number(block_height);
        printf("\n========================================\n");
        printf("[KawPow] GPU Benchmark (block %d, epoch %d)\n", block_height, epoch);
        printf("========================================\n\n");
    } else {
        printf("\n========================================\n");
        printf("[KawPow] GPU Benchmark (autotune)\n");
        printf("========================================\n\n");
    }
    fflush(stdout);

    // Use cached tune if available; --gpu-retune forces a fresh sweep

    auto algo = AlgoRegistry::instance().create("kawpow");
    if (!algo) {
        fprintf(stderr, "[KawPow] Failed to create algorithm instance\n");
        return;
    }

    if (!algo->initialize(0)) {
        fprintf(stderr, "[KawPow] Initialization / autotune failed\n");
        return;
    }

    auto result = algo->get_tuning_result();
    printf("\n========================================\n");
    printf("[KawPow] Autotune complete:\n");
    printf("  %s\n", result.describe().c_str());
    printf("========================================\n\n");
    fflush(stdout);

    algo->cleanup();
}

// ============================================================================
// Dump generated program body for a given block number
// ============================================================================

void kawpow_dump_program(int block_number)
{
    uint64_t period = (uint64_t)block_number / 3;
    int epoch = ethash::get_epoch_number(block_number);

    printf("\n========================================\n");
    printf("[KawPow] Program dump: block %d, period %llu, epoch %d\n",
           block_number, (unsigned long long)period, epoch);
    printf("========================================\n\n");

    // Expanded body (for reading)
    std::string prog = kawpow_proggen::generate_program(block_number);
    // Macro body (what RTC compiles)
    std::string prog_macro = kawpow_proggen::generate_program_macro(block_number);
    // GLC body
    std::string prog_glc = kawpow_proggen::generate_program_glc(block_number);

    // Write expanded body
    {
        char path[128];
        snprintf(path, sizeof(path), "kawpow_program_b%d.hip", block_number);
        FILE* f = fopen(path, "w");
        if (f) {
            fprintf(f, "// ProgPoW program for block %d (period %llu, epoch %d)\n",
                    block_number, (unsigned long long)period, epoch);
            fprintf(f, "// Expanded body (for reading)\n\n");
            fwrite(prog.data(), 1, prog.size(), f);
            fclose(f);
            printf("  Expanded body   → %s\n", path);
        }
    }

    // Write macro definition
    {
        char path[128];
        snprintf(path, sizeof(path), "kawpow_macro_b%d.hip", block_number);
        FILE* f = fopen(path, "w");
        if (f) {
            fprintf(f, "// ProgPoW macro for block %d (period %llu, epoch %d)\n",
                    block_number, (unsigned long long)period, epoch);
            fprintf(f, "// This is the actual #define injected into the RTC kernel\n\n");
            fwrite(prog_macro.data(), 1, prog_macro.size(), f);

            // Also show example 1-way, 2-way, 4-way loop usage with ping-pong
            fprintf(f, "\n// ============================================================\n");
            fprintf(f, "// Example: 1-way main loop (ping-pong BODY_PIPE)\n");
            fprintf(f, "// ============================================================\n");
            fprintf(f, "//\n");
            fprintf(f, "// uint4 _dg0, _dg1;\n");
            fprintf(f, "// PROGPOW_ISSUE_DAG(mix, _dg0, 0u);\n");
            fprintf(f, "// for (uint32_t loop = 0; loop < 62; loop += 2) {\n");
            fprintf(f, "//     PROGPOW_BODY_PIPE(mix, _dg0, _dg1, loop + 1);\n");
            fprintf(f, "//     PROGPOW_BODY_PIPE(mix, _dg1, _dg0, loop + 2);\n");
            fprintf(f, "// }\n");
            fprintf(f, "// PROGPOW_BODY_PIPE(mix, _dg0, _dg1, 63);\n");
            fprintf(f, "// PROGPOW_BODY(mix, _dg1);\n");
            fprintf(f, "\n// ============================================================\n");
            fprintf(f, "// Example: 2-way interleaved loop (ping-pong)\n");
            fprintf(f, "// ============================================================\n");
            fprintf(f, "//\n");
            fprintf(f, "// uint4 _da0, _da1, _db0, _db1;\n");
            fprintf(f, "// PROGPOW_ISSUE_DAG(m0, _da0, 0u);\n");
            fprintf(f, "// PROGPOW_ISSUE_DAG(m1, _db0, 0u);\n");
            fprintf(f, "// for (uint32_t loop = 0; loop < 62; loop += 2) {\n");
            fprintf(f, "//     PROGPOW_BODY_PIPE(m0, _da0, _da1, loop + 1);\n");
            fprintf(f, "//     PROGPOW_BODY_PIPE(m1, _db0, _db1, loop + 1);\n");
            fprintf(f, "//     PROGPOW_BODY_PIPE(m0, _da1, _da0, loop + 2);\n");
            fprintf(f, "//     PROGPOW_BODY_PIPE(m1, _db1, _db0, loop + 2);\n");
            fprintf(f, "// }\n");
            fprintf(f, "// PROGPOW_BODY_PIPE(m0, _da0, _da1, 63);\n");
            fprintf(f, "// PROGPOW_BODY_PIPE(m1, _db0, _db1, 63);\n");
            fprintf(f, "// PROGPOW_BODY(m0, _da1); PROGPOW_BODY(m1, _db1);\n");
            fprintf(f, "\n// ============================================================\n");
            fprintf(f, "// Example: 4-way interleaved loop (ping-pong)\n");
            fprintf(f, "// ============================================================\n");
            fprintf(f, "//\n");
            fprintf(f, "// uint4 _da0,_da1, _db0,_db1, _dc0,_dc1, _dd0,_dd1;\n");
            fprintf(f, "// PROGPOW_ISSUE_DAG(m0, _da0, 0u); ...\n");
            fprintf(f, "// for (uint32_t loop = 0; loop < 62; loop += 2) {\n");
            fprintf(f, "//     PROGPOW_BODY_PIPE(m0, _da0, _da1, loop+1); ...\n");
            fprintf(f, "//     PROGPOW_BODY_PIPE(m0, _da1, _da0, loop+2); ...\n");
            fprintf(f, "// }\n");
            fprintf(f, "// PROGPOW_BODY_PIPE(m0, _da0, _da1, 63); ...\n");
            fprintf(f, "// PROGPOW_BODY(m0, _da1); ...\n");

            fclose(f);
            printf("  Macro def       → %s\n", path);
        }
    }

    // Write GLC body
    {
        char path[128];
        snprintf(path, sizeof(path), "kawpow_program_glc_b%d.hip", block_number);
        FILE* f = fopen(path, "w");
        if (f) {
            fprintf(f, "// ProgPoW GLC program for block %d (period %llu, epoch %d)\n",
                    block_number, (unsigned long long)period, epoch);
            fprintf(f, "// GLC variant (inline asm DAG load with GLC flag)\n\n");
            fwrite(prog_glc.data(), 1, prog_glc.size(), f);
            fclose(f);
            printf("  GLC program     → %s\n", path);
        }
    }

    // Dump full injected kernel source (with placeholder DAG constants)
    {
        std::string src(hip_kawpow_source::SRC_TNN_HIP_CRYPTO_KAWPOW_KAWPOW_HIP_SOURCE);
        src = kawpow_proggen::inject_coin_padding(src, *currentKawpowPadding);
        src = kawpow_proggen::inject_dag_constants(src, 1, 1, 0); // placeholder
        src = kawpow_proggen::inject_program(src, block_number);

        char path[128];
        snprintf(path, sizeof(path), "kawpow_full_b%d.hip", block_number);
        FILE* f = fopen(path, "w");
        if (f) {
            fwrite(src.data(), 1, src.size(), f);
            fclose(f);
            printf("  Full kernel     → %s  (includes 1-way, 2-way, 4-way kernels)\n", path);
        }
    }

    printf("\n");
    fflush(stdout);
}

// ============================================================================
// Helper: parse 64-char hex target string into 32-byte LE array
// ============================================================================
static void parse_hex_target(const std::string& hex, uint8_t out[32])
{
    memset(out, 0, 32);
    auto nib = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    // hex is big-endian (MSB first), GPU target is LE
    size_t len = std::min(hex.size(), (size_t)64);
    for (size_t i = 0; i + 1 < len; i += 2) {
        uint8_t byte = (nib(hex[i]) << 4) | nib(hex[i + 1]);
        // hex[0..1] = MSB = out[31], hex[62..63] = LSB = out[0]
        out[31 - i / 2] = byte;
    }
}

// ============================================================================
// KawPow solution builder
// ============================================================================
static std::optional<GPUSubmitEntry> kawpow_build_solution(
    const uint8_t* hash, uint64_t nonce, int gpu_id,
    const JobSnapshot& job_snapshot, bool devMine)
{
    // hash = 32 bytes of mix_hash from GPU
    // job_snapshot.work_template = 32 bytes of header_hash

    // CPU verification: recompute using progpow reference
    // We need epoch context + block number, stored in job_snapshot extras
    // For now, skip CPU verification — the GPU kernel already does target check
    // TODO: add CPU verification with progpow::hash()

    // Format nonce as 16-char hex (LE bytes → hex, with 0x prefix)
    uint8_t nonce_bytes[8];
    for (int i = 0; i < 8; i++) nonce_bytes[i] = (nonce >> (i * 8)) & 0xFF;
    std::string nonce_hex = "0x" + hexStr(nonce_bytes, 8);

    // header_hash from job snapshot
    std::string header_hex = "0x" + hexStr(job_snapshot.work_template.data(), 32);

    // mix_hash from GPU result
    std::string mix_hex = "0x" + hexStr(hash, 32);

    auto& profile = devMine ? devMiningProfile : miningProfile;
    (void)profile;

    // KawPow stratum submit: [worker, jobId, nonce, header_hash, mix_hash]
    boost::json::object payload = {
        {"id", submitTracker.nextId(gpu_id)},
        {"method", KawPowStratum::submit.method},
        {"params", boost::json::array{
            devMine ? devWorkerName : workerName,
            job_snapshot.job_id_str,
            nonce_hex,
            header_hex,
            mix_hex
        }}
    };

    return GPUSubmitEntry{std::move(payload), devMine, job_snapshot.job_id};
}

// ============================================================================
// Mining entry point
// ============================================================================

void mineKawPow_hip(int tid)
{
    (void)tid;

    std::vector<std::unique_ptr<GPUMiner>> miners;
    int gpuCount;
    (void)oroGetDeviceCount(&gpuCount);

    // Initialize GPUs in parallel
    {
        std::vector<std::thread> init_threads;
        std::vector<std::unique_ptr<GPUMiner>> per_gpu(gpuCount);
        std::vector<bool> gpu_ok(gpuCount, false);

        for (int d = 0; d < gpuCount; d++) {
            if (!shouldUseDevice(d)) continue;

            init_threads.emplace_back([&, d]() {
                try {
                    auto miner = std::make_unique<GPUMiner>("kawpow", d);
                    if (miner->initialize()) {
                        per_gpu[d] = std::move(miner);
                        gpu_ok[d] = true;
                    } else {
                        setcolor(RED);
                        fprintf(stderr, "Failed to initialize GPU %d for KawPow mining\n", d);
                        setcolor(BRIGHT_WHITE);
                    }
                } catch (const std::exception& e) {
                    setcolor(RED);
                    fprintf(stderr, "GPU %d init error: %s\n", d, e.what());
                    setcolor(BRIGHT_WHITE);
                }
            });
        }
        for (auto& t : init_threads) t.join();
        for (int d = 0; d < gpuCount; d++) {
            if (gpu_ok[d]) miners.push_back(std::move(per_gpu[d]));
        }
    }

    if (miners.empty()) {
        setcolor(RED);
        fprintf(stderr, "No GPUs available for KawPow mining\n");
        setcolor(BRIGHT_WHITE);
        return;
    }

    TNN_LOG_INFO_COLOR(BRIGHT_YELLOW, "[KawPow] All GPUs initialized, ready to mine\n");

    int64_t localOurHeight = 0;
    int64_t localDevHeight = 0;

    std::atomic<int64_t> current_job_height{0};
    std::atomic<int64_t> current_dev_job_height{0};

    bool miners_started = false;

waitForJob:
    GPUSubmitQueue::instance().start(
        &share, &devShare,
        &submitting, &submittingDev,
        &data_ready, &cv,
        [&](int64_t job_id, bool is_dev) -> bool {
            int64_t current = is_dev ? current_dev_job_height.load() : current_job_height.load();
            // Allow 2 jobs of staleness for stratum
            return job_id >= (current - 2) && job_id <= current;
        }
    );

    while (!isConnected) {
        CHECK_CLOSE;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    while (!ABORT_MINER) {
        std::this_thread::yield();
        try {
            boost::json::value myJob;
            boost::json::value myJobDev;
            TNN_SNAPSHOT_JOBS(myJob, myJobDev);

            if (!myJob.is_object() || !myJob.as_object().contains("header_hash"))
                continue;
            if (ourHeight == 0 && devHeight == 0)
                continue;

            // Update main work
            if (ourHeight == 0 || localOurHeight != ourHeight) {
                current_job_height.store(ourHeight);

                // Parse header hash (32 bytes) as work template
                std::string hdr_hex = std::string(myJob.at("header_hash").as_string());
                uint8_t header[32] = {};
                hexstrToBytes(hdr_hex, header);

                // Parse 256-bit target from stratum
                std::string target_hex = std::string(myJob.at("target").as_string());
                uint8_t target_bytes[32] = {};
                parse_hex_target(target_hex, target_bytes);

                // Use height as difficulty proxy (for display only)
                uint64_t diff_display = (uint64_t)difficulty;

                std::string job_id_str;
                if (myJob.as_object().contains("jobId"))
                    job_id_str = std::string(myJob.at("jobId").as_string());

                // Extract block height for period tracking
                int64_t block_height = 0;
                if (myJob.as_object().contains("height"))
                    block_height = myJob.at("height").to_number<int64_t>();

                for (auto& miner : miners) {
                    miner->set_work(header, diff_display);
                    miner->set_raw_target(target_bytes);
                    miner->set_job_id(ourHeight, job_id_str);
                    miner->set_dep("period", block_height / 3);
                }

                if (!miners_started) {
                    TNN_LOG_INFO("[KawPow] Starting all GPU miners\n");
                    for (auto& miner : miners) {
                        miner->set_dev_fee(devFee);
                        miner->start([&](const uint8_t* hash, uint64_t nonce, int gpu_id,
                                         const JobSnapshot& job_snapshot) -> std::optional<GPUSubmitEntry> {
                            printf("\n");
                            if (job_snapshot.is_dev) {
                                setcolor(CYAN);
                                printf("DEV | ");
                            } else {
                                setcolor(BRIGHT_YELLOW);
                            }
                            printf("GPU #%d found a share (nonce %016llx)\n", gpu_id, (unsigned long long)nonce);
                            fflush(stdout);
                            setcolor(BRIGHT_WHITE);
                            return kawpow_build_solution(hash, nonce, gpu_id, job_snapshot, job_snapshot.is_dev);
                        });
                    }
                    miners_started = true;
                    TNN_LOG_INFO_COLOR(BRIGHT_YELLOW, "[KawPow] All GPU miners started\n");
                }

                localOurHeight = ourHeight;
            }

            // Update dev work
            if (devConnected && myJobDev.is_object() && myJobDev.as_object().contains("header_hash")) {
                if (devHeight == 0 || localDevHeight != devHeight) {
                    current_dev_job_height.store(devHeight);

                    std::string hdr_hex = std::string(myJobDev.at("header_hash").as_string());
                    uint8_t header[32] = {};
                    hexstrToBytes(hdr_hex, header);

                    // Parse dev target
                    uint8_t dev_target_bytes[32] = {};
                    if (myJobDev.as_object().contains("target")) {
                        std::string dev_target_hex = std::string(myJobDev.at("target").as_string());
                        parse_hex_target(dev_target_hex, dev_target_bytes);
                    }

                    std::string dev_job_id_str;
                    if (myJobDev.as_object().contains("jobId"))
                        dev_job_id_str = std::string(myJobDev.at("jobId").as_string());

                    // Extract dev block height for period tracking
                    int64_t dev_block_height = 0;
                    if (myJobDev.as_object().contains("height"))
                        dev_block_height = myJobDev.at("height").to_number<int64_t>();

                    for (auto& miner : miners) {
                        miner->set_dev_work(header, (uint64_t)difficulty);
                        miner->set_dev_raw_target(dev_target_bytes);
                        miner->set_dev_job_id(devHeight, dev_job_id_str);
                        miner->set_dev_dep("period", dev_block_height / 3);
                    }

                    localDevHeight = devHeight;
                }
            }

            if (!isConnected) break;
        } catch (std::exception& e) {
            setcolor(RED);
            fprintf(stderr, "KawPow mining error: %s\n", e.what());
            setcolor(BRIGHT_WHITE);
            localOurHeight = -1;
            localDevHeight = -1;
        }

        if (!isConnected) {
            data_ready = true;
            cv.notify_all();
            break;
        }
    }

    for (auto& miner : miners) miner->stop();
    GPUSubmitQueue::instance().stop();

    if (!isConnected) {
        miners_started = false;
        localOurHeight = 0;
        localDevHeight = 0;
        goto waitForJob;
    }
}

#endif // TNN_KAWPOW
