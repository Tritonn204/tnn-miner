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
    uint32_t barrett_shift = 0,
    const uint32_t* l1_words = nullptr)
{
    std::string src(hip_kawpow_source::SRC_TNN_HIP_CRYPTO_KAWPOW_KAWPOW_HIP_SOURCE);
    src = kawpow_proggen::inject_coin_padding(src, padding);
    src = kawpow_proggen::inject_dag_constants(src, dag_num_items_div, barrett_m, barrett_shift);

    if (l1_words) {
        src = kawpow_proggen::inject_constant_l1_table(src, l1_words, 4096);
    }

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
    printf("[KawPow] GPU Kernel Validation (batched)\n");
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
    printf("[KawPow] Device: %s (%s)\n", props.name, is_amd ? "AMD" : "NVIDIA");
    printf("[KawPow] sharedMemPerBlock = %zu\n\n", props.sharedMemPerBlock);
    fflush(stdout);

    // ---- Compile helpers (same for all epochs) ----
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

    // ================================================================
    // Launch-config constants — respect __launch_bounds__ on each kernel
    // ================================================================
    constexpr uint32_t SEED_FINAL_TPB     = 128;
    constexpr uint32_t PROGPOW_TPB        = 256;
    constexpr uint32_t MONO_TPB           = 256;
    constexpr uint32_t LANES_PER_HASH     = 16;

    constexpr uint32_t HASHES_PER_BLOCK_2WAY = (PROGPOW_TPB / LANES_PER_HASH) * 2; // 32
    constexpr uint32_t HASHES_PER_BLOCK_4WAY = (PROGPOW_TPB / LANES_PER_HASH) * 4; // 64
    constexpr uint32_t HASHES_PER_BLOCK_MONO = (MONO_TPB / LANES_PER_HASH);         // 16

    constexpr uint32_t NUM_BLOCKS_PP = 32;
    constexpr uint32_t BATCH_SIZE_MONO  = NUM_BLOCKS_PP * HASHES_PER_BLOCK_MONO;     // 512
    constexpr uint32_t BATCH_SIZE_SPLIT = NUM_BLOCKS_PP * HASHES_PER_BLOCK_2WAY;     // 1024

    const uint32_t SEED_FINAL_GRID_MONO  = (BATCH_SIZE_MONO  + SEED_FINAL_TPB - 1) / SEED_FINAL_TPB;
    const uint32_t SEED_FINAL_GRID_SPLIT = (BATCH_SIZE_SPLIT + SEED_FINAL_TPB - 1) / SEED_FINAL_TPB;

    printf("[KawPow] Mono:  TPB=%u, grid=%u blocks, batch=%u hashes\n",
           MONO_TPB, NUM_BLOCKS_PP, BATCH_SIZE_MONO);
    printf("[KawPow] Split: seed/final TPB=%u, progpow TPB=%u, grid=%u blocks, batch=%u hashes\n",
           SEED_FINAL_TPB, PROGPOW_TPB, NUM_BLOCKS_PP, BATCH_SIZE_SPLIT);
    fflush(stdout);

    // Allocate GPU buffers sized for the larger of the two batches
    constexpr uint32_t MAX_BATCH = (BATCH_SIZE_MONO > BATCH_SIZE_SPLIT)
                                   ? BATCH_SIZE_MONO : BATCH_SIZE_SPLIT;
    // Sizes per hash
    const size_t inter_per_hash_u32  = 8;   // d_intermediate: digest only (z,w → digest)
    const size_t result_per_hash_u32 = 16;  // d_results: mix[0..7] + final[0..7]

    const size_t intermediate_bytes = (size_t)MAX_BATCH * inter_per_hash_u32  * sizeof(uint32_t);
    const size_t results_bytes      = (size_t)MAX_BATCH * result_per_hash_u32 * sizeof(uint32_t);

    AlignedDevAlloc header_alloc{}, target_alloc{}, solutions_alloc{}, results_alloc{}, intermediate_alloc{};
    AlignedDevAlloc dag_alloc{};

    uint32_t* d_header       = nullptr;
    uint64_t* d_target       = nullptr;
    uint64_t* d_solutions    = nullptr;
    uint32_t* d_results      = nullptr;
    uint32_t* d_intermediate = nullptr;

    KP_CHECK(oroMallocAligned(header_alloc,       32,            64));
    KP_CHECK(oroMallocAligned(target_alloc,       32,            64));
    KP_CHECK(oroMallocAligned(solutions_alloc,    8 + 1024 * 40, 64));
    KP_CHECK(oroMallocAligned(results_alloc,      results_bytes,      64));
    KP_CHECK(oroMallocAligned(intermediate_alloc, intermediate_bytes, 64));

    d_header       = reinterpret_cast<uint32_t*>(header_alloc.aligned);
    d_target       = reinterpret_cast<uint64_t*>(target_alloc.aligned);
    d_solutions    = reinterpret_cast<uint64_t*>(solutions_alloc.aligned);
    d_results      = reinterpret_cast<uint32_t*>(results_alloc.aligned);
    d_intermediate = reinterpret_cast<uint32_t*>(intermediate_alloc.aligned);

    uint64_t max_target[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    KP_CHECK(oroMemcpy(d_target, max_target, 32, oroMemcpyHostToDevice));

    std::vector<uint32_t> host_results(MAX_BATCH * result_per_hash_u32);

    auto build_samples = [&](uint32_t batch_size, uint32_t hashes_per_block) {
        std::set<uint32_t> s;
        uint32_t nblocks = (batch_size + hashes_per_block - 1) / hashes_per_block;
        for (uint32_t b = 0; b < nblocks; b++) {
            uint32_t base = b * hashes_per_block;
            s.insert(base);
            s.insert(std::min(base + 1, batch_size - 1));
            s.insert(std::min(base + 2, batch_size - 1));
            s.insert(std::min(base + hashes_per_block / 2, batch_size - 1));
            s.insert(std::min(base + hashes_per_block - 1, batch_size - 1));
        }
        s.insert(batch_size - 1);
        return std::vector<uint32_t>(s.begin(), s.end());
    };

    auto samples_mono  = build_samples(BATCH_SIZE_MONO,  HASHES_PER_BLOCK_MONO);
    auto samples_split = build_samples(BATCH_SIZE_SPLIT, HASHES_PER_BLOCK_2WAY);

    auto validate_batch = [&](const char* tag, int tnum,
                              const ethash::epoch_context& ctx,
                              const ethash::hash256& header,
                              uint64_t nonce_start, int block_number,
                              uint32_t batch_size,
                              const std::vector<uint32_t>& samples,
                              int& pass_ctr, int& fail_ctr)
    {
        size_t bytes = (size_t)batch_size * result_per_hash_u32 * sizeof(uint32_t);
        KP_CHECK(oroMemcpy(host_results.data(), d_results, bytes, oroMemcpyDeviceToHost));

        int local_fail = 0;
        for (uint32_t hi : samples) {
            if (hi >= batch_size) continue;
            uint64_t n = nonce_start + (uint64_t)hi;
            auto cpu = progpow::hash(ctx, block_number, header, n);

            const uint32_t* gpu = &host_results[(size_t)hi * result_per_hash_u32];
            std::string cpu_mix   = hash256_hex(cpu.mix_hash);
            std::string cpu_final = hash256_hex(cpu.final_hash);
            std::string gpu_mix   = words_hex(gpu, 8);
            std::string gpu_final = words_hex(gpu + 8, 8);

            bool ok = (cpu_mix == gpu_mix) && (cpu_final == gpu_final);
            if (!ok) {
                if (local_fail == 0)
                    printf("[%s %d] \033[31mFAIL\033[0m at sampled indices:\n", tag, tnum);
                uint32_t blk = hi / (batch_size / NUM_BLOCKS_PP);
                printf("    hash_id=%4u (block~%2u, nonce=+%u)\n", hi, blk, hi);
                if (cpu_mix != gpu_mix)
                    printf("      mix cpu: %s\n      mix gpu: %s\n",
                           cpu_mix.c_str(), gpu_mix.c_str());
                if (cpu_final != gpu_final)
                    printf("      fin cpu: %s\n      fin gpu: %s\n",
                           cpu_final.c_str(), gpu_final.c_str());
                local_fail++;
                if (local_fail >= 10) { printf("    ... (truncated)\n"); break; }
            }
        }
        if (local_fail == 0) {
            printf("[%s %d] \033[32mPASS\033[0m (%zu samples across %u hashes)\n",
                   tag, tnum, samples.size(), batch_size);
            pass_ctr++;
        } else {
            fail_ctr++;
        }
        fflush(stdout);
        return 0;
    };

    struct SynthTV {
        int epoch;
        int block_number;
        const char* header_hex;
        uint64_t nonce;
    };

    static const SynthTV test_vecs[] = {
        { 0,     0, "0000000000000000000000000000000000000000000000000000000000000000", 0x0ULL },
        { 0,     1, "1111111111111111111111111111111111111111111111111111111111111111", 0x100ULL },
        { 0,     2, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 0xdeadbeefULL },
        { 500, 2250000, "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff", 0x0ULL },
        { 500, 2250001, "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100", 0x123456789abcdef0ULL },
        { 500, 2253750, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef", 0x0f0f0f0f0f0f0f0fULL },
        { 500, 2257499, "0000000000000000000000000000000000000000000000000000000000000001", 0xffffffffffffff00ULL },
    };
    constexpr int NUM_VECS = sizeof(test_vecs) / sizeof(test_vecs[0]);

    int total_pass = 0, total_fail = 0;
    int dag_total_pass = 0, dag_total_fail = 0;

    int prev_epoch = -1;
    KawPowDAG dag{};

    for (int vi = 0; vi < NUM_VECS; ) {
        int epoch = test_vecs[vi].epoch;
        if (epoch == prev_epoch) { vi++; continue; }

        printf("\n========================================\n");
        printf("[KawPow] Epoch %d\n", epoch);
        printf("========================================\n\n");
        fflush(stdout);

        auto ctx = ethash::create_epoch_context(epoch);
        if (!ctx) {
            fprintf(stderr, "[KawPow] Failed to create epoch %d context\n", epoch);
            return 1;
        }

        // ---- DAG generation on GPU ----
        oroFreeAligned(dag_alloc);
        dag.d_dag = nullptr;
        dag.d_l1_cache = nullptr;

        dag.num_items_2048    = (uint32_t)(ctx->full_dataset_num_items / 2);
        dag.dag_num_items_div = dag.num_items_2048;

        constexpr size_t DAG_WORDS_PER_ITEM    = 64; // original 2048-bit item
        constexpr size_t DAG_W0_ROWS_PER_ITEM  = 16; // one w0 per vec4 row

        size_t dag_base_words = (size_t)dag.num_items_2048 * DAG_WORDS_PER_ITEM;
        size_t dag_w0_words   = (size_t)dag.num_items_2048 * DAG_W0_ROWS_PER_ITEM;
        size_t dag_words      = dag_base_words + dag_w0_words;

        dag.size_bytes = dag_words * sizeof(uint32_t);

        printf("[KawPow] Generating DAG: %u items (%.1f MB)...\n",
               dag.num_items_2048, dag.size_bytes / (1024.0 * 1024.0));
        fflush(stdout);

        KP_CHECK(oroMallocAligned(dag_alloc, dag.size_bytes, 256));
        dag.d_dag = reinterpret_cast<uint32_t*>(dag_alloc.aligned);

        uint32_t num_cache_items = (uint32_t)ctx->light_cache_num_items;
        size_t cache_bytes = (size_t)num_cache_items * 64;
        uint32_t* d_light_cache = nullptr;
        KP_CHECK(oroMalloc((oroDeviceptr*)&d_light_cache, cache_bytes));
        KP_CHECK(oroMemcpy(d_light_cache, ctx->light_cache, cache_bytes, oroMemcpyHostToDevice));

        std::string dag_src(hip_ethash_dag_source::SRC_TNN_HIP_CRYPTO_KAWPOW_ETHASH_DAG_GEN_HIP_SOURCE);
        auto dag_ck = compiler.compile_from_source(dag_src, "ethash-dag-gen.hip",
                                                   "ethash_dag_gen_kernel", dag_opts);
        if (!dag_ck.function) {
            fprintf(stderr, "[KawPow] DAG gen kernel compile failed\n");
            oroFree((oroDeviceptr)d_light_cache);
            oroFreeAligned(dag_alloc);
            return 1;
        }

        auto t0 = std::chrono::steady_clock::now();
        {
            uint32_t di = dag.num_items_2048;
            int bs = 256, gs = (di + bs - 1) / bs;
            void* args[] = { &d_light_cache, &dag.d_dag, &num_cache_items, &di };
            KP_CHECK(oroModuleLaunchKernel(dag_ck.function, gs, 1, 1, bs, 1, 1,
                                           0, nullptr, args, nullptr));
            KP_CHECK(oroDeviceSynchronize());
        }
        auto t1 = std::chrono::steady_clock::now();
        printf("[KawPow] DAG generated in %.2fs\n",
               std::chrono::duration<double>(t1 - t0).count());
        oroFree((oroDeviceptr)d_light_cache);

        // ---- DAG spot check ----
        printf("\n[DAG] Spot check:\n");
        const uint32_t chk[] = { 0, 1, 100, 1000, dag.num_items_2048/2, dag.num_items_2048-1 };
        for (uint32_t idx : chk) {
            if (idx >= dag.num_items_2048) continue;
            uint32_t gpu_item[64];
            KP_CHECK(oroMemcpy(gpu_item, (uint8_t*)dag.d_dag + (size_t)idx*256,
                               256, oroMemcpyDeviceToHost));
            ethash::hash2048 cpu_item = ethash::calculate_dataset_item_2048(*ctx, idx);
            bool ok = (memcmp(gpu_item, &cpu_item, 256) == 0);
            printf("  [%u] %s\n", idx, ok ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
            if (ok) dag_total_pass++; else dag_total_fail++;
        }
        // if (dag_total_fail > 0) {
        //     fprintf(stderr, "\n[KawPow] DAG validation FAILED, aborting\n");
        //     return 1;
        // }

        dag.d_l1_cache = dag.d_dag;

        uint32_t* d_dag      = dag.d_dag;
        uint32_t* d_l1_cache = dag.d_l1_cache;

        uint32_t dag_div   = dag.dag_num_items_div;
        uint32_t barrett_m = (uint32_t)((1ULL << 32) / dag_div);
        uint32_t barrett_s = 0;

        int last_period = -1;
        oroFunction_t mono_func = nullptr;
        oroFunction_t seed_fn = nullptr, progpow_fn = nullptr, final_fn = nullptr;
        oroModule_t split_module = nullptr;

        for (int t = vi; t < NUM_VECS && test_vecs[t].epoch == epoch; t++) {
            const auto& tv = test_vecs[t];
            int period = tv.block_number / 3;

            ethash::hash256 header = hex_to_h256(tv.header_hex);
            uint64_t nonce_start   = tv.nonce;
            uint32_t block_num     = (uint32_t)tv.block_number;

            if (period != last_period) {
                printf("\n  Compiling period %d (block %d)...\n", period, tv.block_number);
                fflush(stdout);

                oroFunction_t mono_fn_tmp = nullptr;
                if (compile_kawpow_kernel(tv.block_number, *currentKawpowPadding,
                                          is_amd, props, mono_fn_tmp, &split_module,
                                          dag_div, barrett_m, barrett_s, reinterpret_cast<const uint32_t*>(ctx->light_cache)
                                        ) != 0) {
                    fprintf(stderr, "  Compile FAILED\n");
                    total_fail += 2;
                    continue;
                }
                mono_func = mono_fn_tmp;
                oroError_t e1 = oroModuleGetFunction(&seed_fn,    split_module, "kawpow_seed_kernel_seed64");
                oroError_t e2 = oroModuleGetFunction(&progpow_fn, split_module, "kawpow_progpow_kernel_2way_seed64_digest");
                oroError_t e3 = oroModuleGetFunction(&final_fn,   split_module, "kawpow_final_kernel_reseed_seed64_digest");

                if (e1 != oroSuccess || e2 != oroSuccess || e3 != oroSuccess ||
                    !seed_fn || !progpow_fn || !final_fn) {
                    fprintf(stderr, "  Failed to extract split kernels (e1=%d e2=%d e3=%d)\n",
                            e1, e2, e3);
                    seed_fn = progpow_fn = final_fn = nullptr;
                }
                last_period = period;
            }

            // ==== MONO test ====
            // {
            //     uint32_t batch_size = BATCH_SIZE_MONO;
            //     KP_CHECK(oroMemcpy(d_header, header.word32s, 32, oroMemcpyHostToDevice));
            //     uint64_t zero = 0;
            //     KP_CHECK(oroMemcpy(d_solutions, &zero, 8, oroMemcpyHostToDevice));
            //     KP_CHECK(oroMemset((oroDeviceptr)d_results, 0, results_bytes));
            //     uint32_t* d_solution_flag = nullptr;

            //     void* args[] = {
            //         &d_header, &d_dag, &nonce_start, &batch_size,
            //         &d_target, &d_solutions, &d_solution_flag, &block_num, &d_results,
            //         &d_l1_cache,
            //     };
            //     size_t smem = kawpow_shared_mem(MONO_TPB);
            //     KP_CHECK(oroModuleLaunchKernel(mono_func,
            //         NUM_BLOCKS_PP, 1, 1, MONO_TPB, 1, 1,
            //         smem, nullptr, args, nullptr));
            //     KP_CHECK(oroDeviceSynchronize());

            //     validate_batch("Mono", t+1, *ctx, header, nonce_start, tv.block_number,
            //                    batch_size, samples_mono, total_pass, total_fail);
            // }

            // ==== SPLIT 2-WAY test ====
            if (seed_fn && progpow_fn && final_fn) {
                uint32_t batch_size = BATCH_SIZE_SPLIT;

                KP_CHECK(oroMemcpy(d_header, header.word32s, 32, oroMemcpyHostToDevice));
                uint64_t zero = 0;
                KP_CHECK(oroMemcpy(d_solutions, &zero, 8, oroMemcpyHostToDevice));
                KP_CHECK(oroMemset((oroDeviceptr)d_results,      0, results_bytes));
                KP_CHECK(oroMemset((oroDeviceptr)d_intermediate, 0, intermediate_bytes));

                {
                    void* args[] = { &d_header, &nonce_start, &batch_size, &d_intermediate };
                    KP_CHECK(oroModuleLaunchKernel(seed_fn,
                        SEED_FINAL_GRID_SPLIT, 1, 1,
                        SEED_FINAL_TPB, 1, 1,
                        0, nullptr, args, nullptr));
                    KP_CHECK(oroDeviceSynchronize());
                    printf("  [Split %d] seed kernel OK\n", t+1);
                }

                {
                    uint32_t tmp[16];
                    KP_CHECK(oroMemcpy(tmp, d_intermediate, 64, oroMemcpyDeviceToHost));
                    printf("    seed[0] z,w: %s\n", words_hex(tmp, 2).c_str());       // [0..1]
                    printf("    seed[1] z,w: %s\n", words_hex(tmp + 8, 2).c_str());   // [8..9]
                    bool all_zero = !tmp[0] && !tmp[1];

                    // After progpow: digest overwrites to [0..7] per hash
                    KP_CHECK(oroMemcpy(tmp, d_intermediate, 64, oroMemcpyDeviceToHost));
                    printf("    digest[0]: %s\n", words_hex(tmp, 8).c_str());          // [0..7]
                    printf("    digest[1]: %s\n", words_hex(tmp + 8, 8).c_str());      // [8..15]
                    all_zero = true;
                    for (int i = 0; i < 8; i++) if (tmp[i]) all_zero = false;
                    if (all_zero)
                        printf("    \033[33mWARNING: seed output is all zeros!\033[0m\n");
                }

                {
                    size_t smem = kawpow_shared_mem_split(PROGPOW_TPB);
                    printf("    progpow: grid=%u, TPB=%u, smem=%zu, batch=%u\n",
                           NUM_BLOCKS_PP, PROGPOW_TPB, smem, batch_size);
                    fflush(stdout);

                    void* args[] = { &d_dag, &batch_size, &d_dag, &d_intermediate };
                    KP_CHECK(oroModuleLaunchKernel(progpow_fn,
                        NUM_BLOCKS_PP, 1, 1,
                        PROGPOW_TPB, 1, 1,
                        smem, nullptr, args, nullptr));
                    KP_CHECK(oroDeviceSynchronize());
                    printf("  [Split %d] progpow kernel OK\n", t+1);
                }

                {
                    uint32_t tmp[32];
                    KP_CHECK(oroMemcpy(tmp, d_intermediate, 128, oroMemcpyDeviceToHost));
                    printf("    digest[0]: %s\n", words_hex(tmp + 8, 8).c_str());
                    printf("    digest[1]: %s\n", words_hex(tmp + 24, 8).c_str());
                    bool all_zero = true;
                    for (int i = 8; i < 16; i++) if (tmp[i]) all_zero = false;
                    if (all_zero)
                        printf("    \033[33mWARNING: progpow digest is all zeros!\033[0m\n");
                }

                {
                    uint32_t* d_solution_flag = nullptr;
                    void* args[] = { &d_header, &nonce_start, &batch_size, &d_target, &d_solutions,
                                     &d_solution_flag, &d_results, &d_intermediate };
                    KP_CHECK(oroModuleLaunchKernel(final_fn,
                        SEED_FINAL_GRID_SPLIT, 1, 1,
                        SEED_FINAL_TPB, 1, 1,
                        0, nullptr, args, nullptr));
                    KP_CHECK(oroDeviceSynchronize());
                    printf("  [Split %d] final kernel OK\n", t+1);
                }

                validate_batch("Split", t+1, *ctx, header, nonce_start, tv.block_number,
                               batch_size, samples_split, total_pass, total_fail);
            } else {
                printf("  [Split %d] SKIPPED (kernel extraction failed)\n", t+1);
                total_fail++;
            }

            {
                uint64_t sol_count = 0;
                KP_CHECK(oroMemcpy(&sol_count, d_solutions, 8, oroMemcpyDeviceToHost));
                printf("  [Split %d] solutions found: %llu\n", t+1, (unsigned long long)sol_count);

                if (sol_count > 0 && sol_count <= 1024) {
                    uint32_t num_check = (uint32_t)std::min(sol_count, (uint64_t)8);
                    std::vector<uint64_t> sols(1 + num_check * 5);
                    KP_CHECK(oroMemcpy(sols.data(), d_solutions,
                                      (1 + num_check * 5) * sizeof(uint64_t),
                                      oroMemcpyDeviceToHost));

                    for (uint32_t si = 0; si < num_check; si++) {
                        uint64_t sol_nonce = sols[1 + si * 5 + 0];
                        uint32_t hash_id   = (uint32_t)(sol_nonce - nonce_start);

                        const uint64_t* sol_mix64 = &sols[1 + si * 5 + 1];
                        const uint8_t*  sol_mix_bytes = (const uint8_t*)sol_mix64;

                        const uint32_t* res_mix32 = &host_results[(size_t)hash_id * 16];
                        const uint8_t*  res_mix_bytes = (const uint8_t*)res_mix32;

                        auto cpu = progpow::hash(*ctx, tv.block_number, header, sol_nonce);

                        printf("    sol[%u] nonce=%016llx hash_id=%u\n",
                              si, (unsigned long long)sol_nonce, hash_id);

                        bool sol_vs_res = (memcmp(sol_mix_bytes, res_mix_bytes, 32) == 0);
                        printf("      d_solutions vs d_results mix: %s\n",
                              sol_vs_res ? "\033[32mMATCH\033[0m"
                                          : "\033[31mMISMATCH\033[0m");

                        if (!sol_vs_res) {
                            printf("      d_solutions mix: ");
                            for (int b = 0; b < 32; b++) printf("%02x", sol_mix_bytes[b]);
                            printf("\n");
                            printf("      d_results   mix: ");
                            for (int b = 0; b < 32; b++) printf("%02x", res_mix_bytes[b]);
                            printf("\n");
                        }

                        bool sol_vs_cpu = (memcmp(sol_mix_bytes, cpu.mix_hash.bytes, 32) == 0);
                        printf("      d_solutions vs CPU mix:      %s\n",
                              sol_vs_cpu ? "\033[32mMATCH\033[0m"
                                          : "\033[31mMISMATCH\033[0m");

                        if (!sol_vs_cpu) {
                            printf("      d_solutions mix: ");
                            for (int b = 0; b < 32; b++) printf("%02x", sol_mix_bytes[b]);
                            printf("\n");
                            printf("      CPU         mix: ");
                            for (int b = 0; b < 32; b++) printf("%02x", cpu.mix_hash.bytes[b]);
                            printf("\n");
                        }
                    }
                }
            }

            fflush(stdout);
        }

        while (vi < NUM_VECS && test_vecs[vi].epoch == epoch) vi++;
        prev_epoch = epoch;
    }

    // ---- Cleanup ----
    oroFreeAligned(dag_alloc);
    oroFreeAligned(header_alloc);
    oroFreeAligned(target_alloc);
    oroFreeAligned(solutions_alloc);
    oroFreeAligned(results_alloc);
    oroFreeAligned(intermediate_alloc);
    oroCtxDestroy(gpu_ctx);

    printf("\n========================================\n");
    printf("[KawPow] DAG: %d passed", dag_total_pass);
    printf(", Kernel: %d/%d", total_pass, total_pass + total_fail);
    if (total_fail > 0) printf(", \033[31m%d FAILED\033[0m", total_fail);
    else                printf(", \033[32mall passed\033[0m");
    printf("\n========================================\n\n");
    fflush(stdout);

    return (total_fail == 0) ? 0 : 1;
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
            fprintf(f, "// dag128_u64 _dg0, _dg1;\n");
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
            fprintf(f, "// dag128_u64 _da0, _da1, _db0, _db1;\n");
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
            fprintf(f, "// dag128_u64 _da0,_da1, _db0,_db1, _dc0,_dc1, _dd0,_dd1;\n");
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

    // Format nonce as 16-char hex integer text (big-endian text form).
    // Stratum server parses nonce with _toBufferLE(), so this is correct.
    char nonce_buf[19];
    snprintf(nonce_buf, sizeof(nonce_buf), "0x%016llx",
             (unsigned long long)nonce);
    std::string nonce_hex(nonce_buf);

    // header_hash from job snapshot
    // Sent as raw hex bytes; server parses with _toBuffer() (no byte swap)
    std::string header_hex = "0x" + hexStr(job_snapshot.work_template.data(), 32);

    // mix_hash from GPU result
    // Sent as raw hex bytes; server parses with _toBuffer() (no byte swap)
    std::string mix_hex = "0x" + hexStr(hash, 32);

    auto& profile = devMine ? devMiningProfile : miningProfile;
    (void)profile;

    // Optional local debug: show nonce in both interpretations
    uint8_t nonce_le[8];
    for (int i = 0; i < 8; i++) {
        nonce_le[i] = (uint8_t)((nonce >> (i * 8)) & 0xFF);
    }

    // Optional local target check if JobSnapshot carries the raw target
    // Assumes raw_target is stored as LE bytes for GPU compare
    bool have_target = !job_snapshot.raw_target.empty();

    TNN_LOG_TRACE("[TRACE] [KP BUILD] GPU %d, Nonce: 0x%016llx\n",
                  gpu_id, (unsigned long long)nonce);
    TNN_LOG_TRACE("[TRACE] [KP BUILD]   job_id: %s\n", job_snapshot.job_id_str.c_str());
    TNN_LOG_TRACE("[TRACE] [KP BUILD]   nonce(be text): %s\n", nonce_hex.c_str());
    TNN_LOG_TRACE("[TRACE] [KP BUILD]   nonce(le bytes): %s\n", hexStr(nonce_le, 8).c_str());
    TNN_LOG_TRACE("[TRACE] [KP BUILD]   header_hash: %s\n", header_hex.c_str());
    TNN_LOG_TRACE("[TRACE] [KP BUILD]   mix_hash:    %s\n", mix_hex.c_str());
    if (have_target) {
        TNN_LOG_TRACE("[TRACE] [KP BUILD]   target(le):   %s\n", hexStr(job_snapshot.raw_target.data(), 32).c_str());
    }

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

    TNN_LOG_TRACE("[TRACE] [KP SUBMIT]   worker: %s\n", (devMine ? devWorkerName : workerName).c_str());
    TNN_LOG_TRACE("[TRACE] [KP SUBMIT]   jobId:  %s\n", job_snapshot.job_id_str.c_str());
    TNN_LOG_TRACE("[TRACE] [KP SUBMIT]   nonce:  %s\n", nonce_hex.c_str());
    TNN_LOG_TRACE("[TRACE] [KP SUBMIT]   header: %s\n", header_hex.c_str());
    TNN_LOG_TRACE("[TRACE] [KP SUBMIT]   mix:    %s\n", mix_hex.c_str());

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
        std::vector<std::unique_ptr<GPUMiner>> per_gpu(gpuCount);
        std::vector<bool> gpu_ok(gpuCount, false);
        std::vector<std::thread> init_threads;

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

        // Service any pending main-thread GPU work (e.g. DAG rebuild)
        if (miners_started) {
            for (auto& miner : miners) {
                if (miner->needs_main_thread_service()) {
                    miner->service_main_thread();
                }
            }
        }

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

                setcolor(CYAN);
                TNN_LOG_DEBUG("[KP WORK] job=%s height=%lld\n",
                      job_id_str.c_str(), (long long)block_height);

                TNN_LOG_DEBUG("[KP WORK] header_hash = %s\n", hdr_hex.c_str());
                TNN_LOG_DEBUG("[KP WORK] target_hex  = %s\n", target_hex.c_str());
                TNN_LOG_DEBUG("[KP WORK] target_le   = %s\n", hexStr(target_bytes, 32).c_str());

                fflush(stdout);
                setcolor(BRIGHT_WHITE);

                if (!miners_started) {
                    // Build live DAG on init thread BEFORE starting worker threads.
                    // Worker threads poison the GPU context on AMD HIP Windows —
                    // heavy GPU work (DAG gen) must happen before they exist.
                    TNN_LOG_INFO("[KawPow] Building live DAG on init thread...\n");
                    for (auto& miner : miners) {
                        if (!miner->flush_deps()) {
                            TNN_LOG_ERROR("[KawPow] GPU %d: flush_deps failed, skipping\n", miner->get_device_id());
                        }
                    }

                    TNN_LOG_INFO("[KawPow] Starting all GPU miners\n");
                    for (auto& miner : miners) {
                        miner->set_dev_fee(devFee);
                        miner->start([&](const uint8_t* hash, uint64_t nonce, int gpu_id,
                                        const JobSnapshot& job_snapshot)
                            -> std::optional<GPUSubmitEntry>
                        {
                            printf("\n");

                            if (job_snapshot.is_dev) {
                                setcolor(CYAN);
                                printf("DEV | ");
                            } else {
                                setcolor(BRIGHT_YELLOW);
                            }

                            printf("GPU #%d found a share (nonce %016llx)\n",
                                  gpu_id, (unsigned long long)nonce);

                            TNN_LOG_DEBUG("[KP SHARE] job_id      = %lld\n",
                                  (long long)job_snapshot.job_id);

                            TNN_LOG_DEBUG("[KP SHARE] nonce(be)   = %016llx\n",
                                  (unsigned long long)nonce);

                            uint8_t nbytes[8];
                            for (int i = 0; i < 8; i++)
                                nbytes[i] = (nonce >> (i * 8)) & 0xFF;

                            TNN_LOG_DEBUG("[KP SHARE] nonce(le)   = %s\n", hexStr(nbytes, 8).c_str());

                            TNN_LOG_DEBUG("[KP SHARE] header_hash = %s\n",
                                  hexStr(job_snapshot.work_template.data(), 32).c_str());

                            TNN_LOG_DEBUG("[KP SHARE] mix_hash    = %s\n",
                                  hexStr(hash, 32).c_str());

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
