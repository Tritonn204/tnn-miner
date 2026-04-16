#ifdef TNN_KAWPOW

#include "../../common/gpu_compat.hpp"
#include "../../common/gpu_rtc.hpp"
#include "../../common/hip_algo_registry.hpp"
#include "../../crypto/kawpow/kawpow_proggen.hpp"
#include <kawpow.hip.hpp>
#include <kawpow_embedded_headers.hpp>
#include <ethash-dag-gen.hip.hpp>
#include <ethash/ethash.hpp>
#include <ethash/progpow.hpp>
#include <ethash/ethash-internal.hpp>
#include <ethash/kawpow_coins.h>
#include <algo_definitions.h>
#include <tnn_log.hpp>

#include <chrono>
#include <vector>
#include <cstring>
#include <cstdio>
#include <string>

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
    oroModule_t* out_module = nullptr)
{
    std::string src(hip_kawpow_source::SRC_TNN_HIP_CRYPTO_KAWPOW_KAWPOW_HIP_SOURCE);
    src = kawpow_proggen::inject_coin_padding(src, padding);
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
    uint32_t* d_solutions = nullptr;
    uint32_t* d_results   = nullptr;

    KP_CHECK(oroMalloc((oroDeviceptr*)&d_header,    32));
    KP_CHECK(oroMalloc((oroDeviceptr*)&d_target,    32));
    KP_CHECK(oroMalloc((oroDeviceptr*)&d_solutions, 320));
    KP_CHECK(oroMalloc((oroDeviceptr*)&d_results,   16 * sizeof(uint32_t)));

    // Set target to all-1s (accept everything)
    uint64_t max_target[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    KP_CHECK(oroMemcpy(d_target, max_target, 32, oroMemcpyHostToDevice));

    // ---- Phase 2: Kernel tests (mono + split) ----
    printf("[KawPow] Phase 2: ProgPoW Kernel Validation\n");
    printf("----------------------------------------\n");
    fflush(stdout);

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
            if (compile_kawpow_kernel(tv.block_number, KAWPOW_PADDING_RVN,
                                      is_amd, props, kernel_func) != 0) {
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
        uint32_t zero = 0;
        KP_CHECK(oroMemcpy(d_solutions, &zero, 4, oroMemcpyHostToDevice));
        uint32_t zeros[16] = {};
        KP_CHECK(oroMemcpy(d_results, zeros, 64, oroMemcpyHostToDevice));

        // Launch: 1 hash = 16 threads
        uint32_t block_size  = 16;
        uint32_t grid_size   = 1;
        uint32_t batch_size  = 1;
        uint32_t block_num   = (uint32_t)tv.block_number;
        uint32_t dagdiv      = dag.dag_num_items_div;
        uint32_t b_m = (uint32_t)((1ULL << 32) / dagdiv); // floor-based Barrett
        uint32_t b_s = 0;

        void* args[] = {
            &d_header, &d_dag, &nonce, &batch_size,
            &d_target, &d_solutions, &block_num, &dagdiv, &d_results,
            &d_l1_cache, &b_m, &b_s,
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
            if (compile_kawpow_kernel(tv.block_number, KAWPOW_PADDING_RVN,
                                      is_amd, props, mono_fn, &split_module) != 0) {
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
        uint32_t zero = 0;
        KP_CHECK(oroMemcpy(d_solutions, &zero, 4, oroMemcpyHostToDevice));
        uint32_t zeros[16] = {};
        KP_CHECK(oroMemcpy(d_results, zeros, 64, oroMemcpyHostToDevice));
        KP_CHECK(oroMemset((oroDeviceptr)d_intermediate, 0, 64));

        uint32_t batch_size = 1;
        uint32_t dagdiv = dag.dag_num_items_div;
        uint32_t b_m = (uint32_t)((1ULL << 32) / dagdiv);
        uint32_t b_s = 0;

        // Seed kernel: 1 thread per hash
        {
            void* args[] = { &d_header, &nonce, &batch_size, &d_intermediate };
            KP_CHECK(oroModuleLaunchKernel(seed_fn, 1, 1, 1, 1, 1, 1,
                                           0, nullptr, args, nullptr));
            KP_CHECK(oroDeviceSynchronize());
        }

        // Progpow kernel: 16 threads per hash (1 block of 16)
        {
            void* args[] = { &d_dag, &batch_size, &dagdiv, &d_l1_cache,
                             &b_m, &b_s, &d_intermediate };
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

void kawpow_bench()
{
    printf("\n========================================\n");
    printf("[KawPow] GPU Benchmark (autotune)\n");
    printf("========================================\n\n");
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
// Mining entry point (stub)
// ============================================================================

void mineKawPow_hip(int tid)
{
    TNN_LOG_INFO("[KawPow] GPU mining not yet implemented\n");
    (void)tid;
}

#endif // TNN_KAWPOW
