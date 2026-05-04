#include "xelis-hash-v3-internal.hpp"
#include "xelis_v3_fmv.gen.h"
#include <tnn-hugepages.hpp>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>



// ============================================================================
// ARM stage_1 (not part of spec/SIMD tier system)
// On x86, stage_1/stage_1_nt/stage_1_serial come from generated simd_sources/ TUs.
// ============================================================================

#if !defined(__x86_64__)

static void stage_1(const uint8_t *input, uint64_t *sp, size_t input_len)
{
    constexpr size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3;
    uint8_t *t = reinterpret_cast<uint8_t *>(sp);
    uint8_t K2_values[4][32];
    uint8_t nonces[4][12];

    stage_1_derive(input, input_len, K2_values, nonces);

    for (int i = 0; i < 4; i++) {
        uint8_t state[48] = {0};
        ChaCha20SetKey(state, K2_values[i]);
        ChaCha20SetNonce(state, nonces[i]);
        ChaCha20EncryptBytes(state, NULL, t + i * bytes_per_chunk, bytes_per_chunk, 8);
    }
}

#endif // !defined(__x86_64__)

// ============================================================================
// STAGE 3 - ARM (not part of spec/SIMD tier system)
// On x86, stage_3/stage_1 come from the generated simd_sources/ TUs.
// ============================================================================

// On ARM, stage_3 + modpow stubs are defined here directly.
// On x86, they come from the generated simd_sources/ TUs.
#if !defined(__x86_64__)

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_CRYPTO)
static inline uint64x2_t aes_round_v3_register(uint64_t mem_a, uint64_t mem_b, uint8x16_t key_vec)
{
    uint64x2_t input_u64 = {mem_b, mem_a};
    uint8x16_t block = vreinterpretq_u8_u64(input_u64);
    block = vaeseq_u8(block, vdupq_n_u8(0));
    block = vaesmcq_u8(block);
    block = veorq_u8(block, key_vec);
    return vreinterpretq_u64_u8(block);
}
#endif

static void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
    constexpr uint8_t key[17] = "xelishash-pow-v3";
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_CRYPTO)
    uint8x16_t key_vec = vld1q_u8(key);
#endif
    uint64_t *__restrict mem_buffer_a = scratch_pad;
    uint64_t *__restrict mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V3;
    uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
    uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;
    size_t r = 0;

    for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; ++i)
    {
        uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
        uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];
        uint64_t hash1, hash2;

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_CRYPTO)
        { uint64x2_t ru = aes_round_v3_register(mem_a, mem_b, key_vec);
          hash1 = vgetq_lane_u64(ru, 0); hash2 = vgetq_lane_u64(ru, 1); }
#else
        uint8_t bb[16];
        uint64_to_le_bytes(mem_b, bb); uint64_to_le_bytes(mem_a, bb+8);
        aes_round(bb, key);
        hash1 = le_bytes_to_uint64(bb); hash2 = le_bytes_to_uint64(bb+8);
#endif
        uint64_t result = ~(hash1 ^ hash2);

        uint64_t next_a_idx = map_index(result);
        prefetch_L1(&mem_buffer_a[next_a_idx]);

        for (size_t j = 0; j < XELIS_BUFFER_SIZE_V3; ++j)
        {
            uint64_t a = mem_buffer_a[next_a_idx];
            uint64_t b = mem_buffer_b[map_index(a ^ ~ROTR(result, r))];
            uint64_t c = scratch_pad[r];
            r = (r + 1) % XELIS_MEMORY_SIZE_V3;

uint32_t op_raw = ROTL(result, (uint32_t)c);
uint64_t v      = execute_operation(op_raw, a, b, c, r, result, i, j);

            uint64_t idx_seed = v ^ result;
            result            = ROTL(idx_seed, r);

            MapIndexPair read_idx = map_index_pair(result, idx_seed);
            next_a_idx = read_idx.first;
            prefetch_L1(&mem_buffer_a[next_a_idx]);

            int      use_b = pick_half(v);
            uint64_t idx_t = read_idx.second;
            uint64_t t     = (use_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

            uint64_t idx_a = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
            uint64_t idx_b = map_index(idx_a ^ ~result ^ XELIS_SCATTER_CONST);

            uint64_t mem_a_tmp = mem_buffer_a[idx_a];
            mem_buffer_a[idx_a] = t;
            mem_buffer_b[idx_b] ^= mem_a_tmp ^ ROTR(t, i + j);
        }

        addr_a = modular_power_fast(addr_a, addr_b, result);
        uint64_t sq_r, sq_a;
        isqrt_pair(result, addr_a, &sq_r, &sq_a);
        addr_b = sq_r * (r + 1) * sq_a;
    }
}

static void stage_3_l2(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
    // ARM L2 variant — same as stage_3 but with L2 prefetch hint
    // (On ARM this is a minimal difference; included for API symmetry)
    XELIS_STAGE3_SOFT_AES_BODY_PF(prefetch_L2)
}

static void stage_3_sw(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
    XELIS_STAGE3_SOFT_AES_BODY_SWITCH
}

static void stage_3_merged(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
    XELIS_STAGE3_SOFT_AES_BODY_MERGED
}

#endif // !defined(__x86_64__)

// ============================================================================
// VARIANT REGISTRY
// ============================================================================

using Stage1Fn = void(*)(const uint8_t *, uint64_t *, size_t);
using Stage3Fn = void(*)(uint64_t *, workerData_xelis_v3 &);

// On ARM, dispatchers don't exist — wire directly to the local statics
#if !defined(__x86_64__)
static Stage1Fn get_stage_1() { return stage_1; }
static Stage1Fn get_stage_1_nt() { return stage_1; }     // NT is a no-op on ARM
static Stage1Fn get_stage_1_serial() { return stage_1; } // ARM stage_1 is already serial
#endif

// Stage 1 SIMD override — set by xelis_set_simd_override() from CLI flag.
// When set, xelis_hash_v3 / _nt use this instead of get_stage_1().
#if defined(__x86_64__)
static Stage1Fn g_stage1_override    = nullptr;
static Stage1Fn g_stage1_nt_override = nullptr;

bool xelis_set_simd_override(const char *level)
{
    std::string s(level);
    if (s == "avx512") {
        g_stage1_override    = stage_1_avx512;
        g_stage1_nt_override = stage_1_nt_avx512;
    } else if (s == "avx2") {
        g_stage1_override    = stage_1_avx2;
        g_stage1_nt_override = stage_1_nt_avx2;
    } else if (s == "aes") {
        g_stage1_override    = stage_1_aes;
        g_stage1_nt_override = stage_1_nt_aes;
    } else {
        g_stage1_override    = stage_1_fallback;
        g_stage1_nt_override = stage_1_nt_fallback;
    }
    return true;
}
#else
static Stage1Fn g_stage1_override    = nullptr;
static Stage1Fn g_stage1_nt_override = nullptr;
bool xelis_set_simd_override(const char *) { return false; }
#endif

// Stage 3 dispatch: two-dimensional (SIMD level x AES-NI).
// AES-NI availability is independent of SIMD level — a CPU can have
// AVX512 without AES-NI or vice versa.  Resolved once, cached in a
// static function pointer so every subsequent call is a single indirect.
#if defined(__x86_64__)
static Stage3Fn pick_stage_3() {
    static Stage3Fn fn = [] {
        bool has_aes = __builtin_cpu_supports("aes") && __builtin_cpu_supports("sse4.1");
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")
            && __builtin_cpu_supports("avx512bw"))
            return (Stage3Fn)(has_aes ? stage_3_hw_avx512 : stage_3_avx512);
        if (__builtin_cpu_supports("avx2"))
            return (Stage3Fn)(has_aes ? stage_3_hw_avx2 : stage_3_avx2);
        if (has_aes)
            return (Stage3Fn)stage_3_aes;
        return (Stage3Fn)stage_3_fallback;
    }();
    return fn;
}

static Stage3Fn pick_stage_3_l2() {
    static Stage3Fn fn = [] {
        bool has_aes = __builtin_cpu_supports("aes") && __builtin_cpu_supports("sse4.1");
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")
            && __builtin_cpu_supports("avx512bw"))
            return (Stage3Fn)(has_aes ? stage_3_hw_l2_avx512 : stage_3_l2_avx512);
        if (__builtin_cpu_supports("avx2"))
            return (Stage3Fn)(has_aes ? stage_3_hw_l2_avx2 : stage_3_l2_avx2);
        if (has_aes)
            return (Stage3Fn)stage_3_l2_aes;
        return (Stage3Fn)stage_3_l2_fallback;
    }();
    return fn;
}

static Stage3Fn pick_stage_3_switch() {
    static Stage3Fn fn = [] {
        bool has_aes = __builtin_cpu_supports("aes") && __builtin_cpu_supports("sse4.1");
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")
            && __builtin_cpu_supports("avx512bw"))
            return (Stage3Fn)(has_aes ? stage_3_hw_sw_avx512 : stage_3_sw_avx512);
        if (__builtin_cpu_supports("avx2"))
            return (Stage3Fn)(has_aes ? stage_3_hw_sw_avx2 : stage_3_sw_avx2);
        if (has_aes)
            return (Stage3Fn)stage_3_sw_aes;
        return (Stage3Fn)stage_3_sw_fallback;
    }();
    return fn;
}

static Stage3Fn pick_stage_3_merged() {
    static Stage3Fn fn = [] {
        bool has_aes = __builtin_cpu_supports("aes") && __builtin_cpu_supports("sse4.1");
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")
            && __builtin_cpu_supports("avx512bw"))
            return (Stage3Fn)(has_aes ? stage_3_hw_merged_avx512 : stage_3_merged_avx512);
        if (__builtin_cpu_supports("avx2"))
            return (Stage3Fn)(has_aes ? stage_3_hw_merged_avx2 : stage_3_merged_avx2);
        if (has_aes)
            return (Stage3Fn)stage_3_merged_aes;
        return (Stage3Fn)stage_3_merged_fallback;
    }();
    return fn;
}

#else
static Stage3Fn pick_stage_3() { return stage_3; }
static Stage3Fn pick_stage_3_l2() { return stage_3_l2; }
static Stage3Fn pick_stage_3_switch() { return stage_3_sw; }
static Stage3Fn pick_stage_3_merged() { return stage_3_merged; }
#endif

struct HashVariant {
    const char *id;
    const char *description;
    Stage1Fn    stage1;
    Stage3Fn    stage3;
    bool        enabled;
};

static HashVariant HASH_VARIANTS[] = {
    // id              description                               s1                    s3              enabled
    { "baseline",      "S1:pipelined + S3:baseline",            get_stage_1(),        pick_stage_3(),  true },
    { "s1_serial",     "S1:serial    + S3:baseline",            get_stage_1_serial(), pick_stage_3(),  true },
    { "s3_merged",     "S1:pipelined + S3:merged-div",          get_stage_1(),        pick_stage_3_merged(), true },
};

static constexpr size_t NUM_VARIANTS = sizeof(HASH_VARIANTS) / sizeof(HASH_VARIANTS[0]);

// ============================================================================
// CANONICAL HASH (uses baseline)
// ============================================================================

TNN_SECTION("xelis_hot") void xelis_hash_v3(byte *input, workerData_xelis_v3 &worker, byte *hashResult)
{
    auto s1 = g_stage1_override ? g_stage1_override : get_stage_1();
    s1(input, worker.scratchPad, 112);
    pick_stage_3()(worker.scratchPad, worker);
    blake3((uint8_t *)worker.scratchPad, XELIS_OUTPUT_SIZE_V3, hashResult);
}

TNN_SECTION("xelis_hot") void xelis_hash_v3_nt(byte *input, workerData_xelis_v3 &worker, byte *hashResult)
{
    auto s1 = g_stage1_nt_override ? g_stage1_nt_override : get_stage_1_nt();
    s1(input, worker.scratchPad, 112);
    pick_stage_3()(worker.scratchPad, worker);
    blake3((uint8_t *)worker.scratchPad, XELIS_OUTPUT_SIZE_V3, hashResult);
}

TNN_SECTION("xelis_hot") void xelis_hash_v3_switch(byte *input, workerData_xelis_v3 &worker, byte *hashResult)
{
    auto s1 = g_stage1_override ? g_stage1_override : get_stage_1();
    s1(input, worker.scratchPad, 112);
    pick_stage_3_switch()(worker.scratchPad, worker);
    blake3((uint8_t *)worker.scratchPad, XELIS_OUTPUT_SIZE_V3, hashResult);
}

TNN_SECTION("xelis_hot") void xelis_hash_v3_switch_nt(byte *input, workerData_xelis_v3 &worker, byte *hashResult)
{
    auto s1 = g_stage1_nt_override ? g_stage1_nt_override : get_stage_1_nt();
    s1(input, worker.scratchPad, 112);
    pick_stage_3_switch()(worker.scratchPad, worker);
    blake3((uint8_t *)worker.scratchPad, XELIS_OUTPUT_SIZE_V3, hashResult);
}

TNN_SECTION("xelis_hot") void xelis_hash_v3_merged(byte *input, workerData_xelis_v3 &worker, byte *hashResult)
{
    auto s1 = g_stage1_override ? g_stage1_override : get_stage_1();
    s1(input, worker.scratchPad, 112);
    pick_stage_3_merged()(worker.scratchPad, worker);
    blake3((uint8_t *)worker.scratchPad, XELIS_OUTPUT_SIZE_V3, hashResult);
}

TNN_SECTION("xelis_hot") void xelis_hash_v3_merged_nt(byte *input, workerData_xelis_v3 &worker, byte *hashResult)
{
    auto s1 = g_stage1_nt_override ? g_stage1_nt_override : get_stage_1_nt();
    s1(input, worker.scratchPad, 112);
    pick_stage_3_merged()(worker.scratchPad, worker);
    blake3((uint8_t *)worker.scratchPad, XELIS_OUTPUT_SIZE_V3, hashResult);
}

// ============================================================================
// BENCHMARK
// ============================================================================

struct BenchResult {
    const char *id;
    const char *description;
    double      hs;
    double      ms;
    double      speedup_pct;
    bool        correct;
    byte        hash[XELIS_HASH_SIZE];
};

__attribute__((cold)) void xelis_benchmark_all_variants()
{
    constexpr uint32_t WARMUP     = 100;
    constexpr uint32_t ITERATIONS = 5000;

    byte input[112];
    for (int i = 0; i < 112; i++) input[i] = (byte)((i * 17 + 31) & 0xFF);

    auto *worker = (workerData_xelis_v3 *)malloc_huge_pages(sizeof(workerData_xelis_v3));
    byte                scratch[XELIS_HASH_SIZE];

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            XELIS HASH V3 - FULL VARIANT BENCHMARK                    ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Buffer: " << std::setw(4) << (XELIS_MEMORY_SIZE_V3 * 8 / 1024)
              << " KB  │  Outer iters: " << XELIS_SCRATCHPAD_ITERS_V3
              << "  │  Inner iters: " << XELIS_BUFFER_SIZE_V3
              << "  │  Bench: " << ITERATIONS << "  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    std::vector<BenchResult> results;
    results.reserve(NUM_VARIANTS);
    byte baseline_hash[XELIS_HASH_SIZE] = {0};
    double baseline_hs = 1.0;

    for (size_t v = 0; v < NUM_VARIANTS; v++)
    {
        if (!HASH_VARIANTS[v].enabled) continue;

        BenchResult r;
        r.id          = HASH_VARIANTS[v].id;
        r.description = HASH_VARIANTS[v].description;

        auto run = [&](uint32_t n) {
            for (uint32_t k = 0; k < n; k++) {
                HASH_VARIANTS[v].stage1(input, worker->scratchPad, 112);
                HASH_VARIANTS[v].stage3(worker->scratchPad, *worker);
                blake3((uint8_t *)worker->scratchPad, XELIS_OUTPUT_SIZE_V3, r.hash);
            }
        };

        run(WARMUP);

        auto t0 = std::chrono::steady_clock::now();
        run(ITERATIONS);
        auto t1 = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::milli> elapsed = t1 - t0;
        r.hs = (ITERATIONS * 1000.0) / elapsed.count();
        r.ms = elapsed.count() / ITERATIONS;

        if (v == 0) {
            memcpy(baseline_hash, r.hash, XELIS_HASH_SIZE);
            baseline_hs   = r.hs;
            r.speedup_pct = 0.0;
            r.correct     = true;
        } else {
            r.correct     = (memcmp(r.hash, baseline_hash, XELIS_HASH_SIZE) == 0);
            r.speedup_pct = ((r.hs / baseline_hs) - 1.0) * 100.0;
        }

        results.push_back(r);
    }

    // ---- Print table ----
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "┌────────────────┬────────────────────────────────────┬───────────┬──────────┬──────────┬─────────┐\n";
    std::cout << "│ ID             │ Description                        │    H/s    │ ms/hash  │ vs Base  │ Correct │\n";
    std::cout << "├────────────────┼────────────────────────────────────┼───────────┼──────────┼──────────┼─────────┤\n";

    for (const auto &r : results) {
        std::cout << "│ " << std::left  << std::setw(14) << r.id
                  << " │ " << std::left  << std::setw(34) << r.description
                  << " │ " << std::right << std::setw(9)  << r.hs
                  << " │ " << std::right << std::setw(8)  << r.ms
                  << " │ ";
        if (r.speedup_pct == 0.0 && r.correct)
            std::cout << "  base    │";
        else
            std::cout << (r.speedup_pct >= 0 ? " +" : " ")
                      << std::setw(6) << r.speedup_pct << "% │";
        std::cout << "    " << (r.correct ? "✓" : "✗") << "    │\n";
    }

    std::cout << "└────────────────┴────────────────────────────────────┴───────────┴──────────┴──────────┴─────────┘\n";

    std::cout << "\nBaseline hash: ";
    for (int i = 0; i < 32; i++) printf("%02x", baseline_hash[i]);
    std::cout << "\n";

    for (const auto &r : results) {
        if (!r.correct) {
            std::cout << "  ⚠ MISMATCH [" << r.id << "]: ";
            for (int i = 0; i < 32; i++) printf("%02x", r.hash[i]);
            std::cout << "\n";
        }
    }
    std::cout << "\n";
    fflush(stdout);
    free_huge_pages(worker);
}

// ============================================================================
// PUBLIC API
// ============================================================================

TNN_SECTION("xelis_hot") void xelis_stage1_v3(const uint8_t *input, uint64_t *scratch_pad, size_t input_len)
{
    get_stage_1()(input, scratch_pad, input_len);
}

TNN_SECTION("xelis_hot") void xelis_stage3_v3(uint64_t *scratch_pad)
{
    static thread_local auto *worker = new workerData_xelis_v3();
    pick_stage_3()(scratch_pad, *worker);
}

TNN_SECTION("xelis_hot") void xelis_blake3_v3(const uint8_t *scratch_pad, byte *hashResult)
{
    blake3(scratch_pad, XELIS_OUTPUT_SIZE_V3, hashResult);
}

static void xelis_benchmark_multithread();

__attribute__((cold)) void xelis_benchmark_cpu_hash_v3()
{
    constexpr uint32_t ITERATIONS = 10000;
    byte input[112] = {0};
    auto *worker = (workerData_xelis_v3 *)malloc_huge_pages(sizeof(workerData_xelis_v3));
    byte hash_result[XELIS_HASH_SIZE] = {0};

    printf("v3 bench\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < ITERATIONS; ++i)
        xelis_hash_v3(input, *worker, hash_result);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time:    " << elapsed.count() << " ms\n";
    std::cout << "H/s:     " << ((ITERATIONS * 1000.0) / elapsed.count()) << "\n";
    std::cout << "ms/hash: " << (elapsed.count() / ITERATIONS) << "\n";
    for (int i = 0; i < 32; i++) printf("%02x", hash_result[i]);
    printf("\n");
    fflush(stdout);

    free_huge_pages(worker);

    xelis_benchmark_all_variants();
    fflush(stdout);
    xelis_benchmark_multithread();
    fflush(stdout);
}

// ============================================================================
// TESTS
// ============================================================================

namespace xelis_tests_v3
{
    using Hash = std::array<byte, XELIS_HASH_SIZE>;

    __attribute__((cold)) bool test_input(const char *name, byte *input, size_t, const Hash &expected)
    {
        auto worker = std::make_unique<workerData_xelis_v3>();
        byte got[XELIS_HASH_SIZE] = {0};
        xelis_hash_v3(input, *worker, got);

        if (std::memcmp(got, expected.data(), XELIS_HASH_SIZE) != 0) {
            std::cout << "FAIL '" << name << "'\n  expected: ";
            for (auto b : expected) printf("%02x", b);
            std::cout << "\n  got:      ";
            for (int i = 0; i < 32; i++) printf("%02x", got[i]);
            std::cout << "\n";
            fflush(stdout);
            return false;
        }
        return true;
    }

    __attribute__((cold)) bool test_reused_scratchpad()
    {
        auto worker = std::make_unique<workerData_xelis_v3>();
        byte input[112];
        srand(time(NULL));
        for (int i = 0; i < 112; i++) input[i] = rand() & 0xFF;

        byte h1[XELIS_HASH_SIZE], h2[XELIS_HASH_SIZE];
        xelis_hash_v3(input, *worker, h1);
        xelis_hash_v3(input, *worker, h2);

        bool ok = std::memcmp(h1, h2, XELIS_HASH_SIZE) == 0;
        std::cout << (ok ? "PASS" : "FAIL") << " test_reused_scratchpad\n";
        return ok;
    }

    __attribute__((cold)) bool test_zero_hash()
    {
        alignas(32) byte input[112] = {0};
        Hash expected = {
            105,172,103, 40, 94,253, 92,162,
             42,252,  5,196,236,238, 91,218,
             22,157,228,233,239,  8,250, 57,
            212,166,121,132,148,205,103,163};
        return test_input("test_zero_hash", input, sizeof(input), expected);
    }

    __attribute__((cold)) bool test_verify_output()
    {
        byte input[] = {
            172,236,108,212,181, 31,109, 45, 44,242, 54,225,143,133,
             89, 44,179,108, 39,191, 32,116,229, 33, 63,130, 33,120,185, 89,
            146,141, 10, 79,183,107,238,122, 92,222, 25,134, 90,107,116,
            110,236, 53,255,  5,214,126, 24,216, 97,199,148,239,253,102,
            199,184,232,253,158,145, 86,187,112, 81, 78, 70, 80,110, 33,
             37,159,233,198,  1,178,108,210,100,109,155,106,124,124, 83,
             89, 50,197,115,231, 32, 74,  2, 92, 47, 25,220,135,249,122,
            172,220,137,143,234, 68,188};
        Hash expected = {
            242,  8,176,222,203, 27,104,
            187, 22, 40, 68, 73, 79, 79, 65,
             83,138,101, 10,116,194, 41,153,
             21, 92,163, 12,206,231,156, 70, 83};
        return test_input("test_verify_output", input, sizeof(input), expected);
    }

    __attribute__((cold)) bool test_pick_half()
    {
        int ones = 0, zeros = 0;
        for (uint64_t i = 0; i < 1000000; i++)
            (pick_half(i) ? ones : zeros)++;
        double ratio = (double)ones / 1000000.0;
        bool ok = std::abs(ratio - 0.5) < 0.01;
        std::cout << (ok ? "PASS" : "FAIL") << " test_pick_half ratio=" << ratio << "\n";
        return ok;
    }

    __attribute__((cold)) bool test_map_index()
    {
        for (int i = 0; i < 1000000; i++) {
            uint64_t x = ((uint64_t)rand() << 32) | rand();
            uint64_t y = ((uint64_t)rand() << 32) | rand();
            uint64_t idx_x = map_index(x);
            uint64_t idx_y = map_index(y);
            MapIndexPair pair = map_index_pair(x, y);

            if (pair.first != idx_x || pair.second != idx_y) {
                printf("FAIL test_map_index_pair: (%llu,%llu) != (%llu,%llu)\n",
                       (unsigned long long)pair.first,
                       (unsigned long long)pair.second,
                       (unsigned long long)idx_x,
                       (unsigned long long)idx_y);
                return false;
            }
            if (idx_x >= XELIS_BUFFER_SIZE_V3) {
                printf("FAIL test_map_index: %llu >= %zu\n",
                       (unsigned long long)idx_x, XELIS_BUFFER_SIZE_V3);
                return false;
            }
        }
        if (map_index(0) >= XELIS_BUFFER_SIZE_V3 ||
            map_index(UINT64_MAX) >= XELIS_BUFFER_SIZE_V3) {
            std::cout << "FAIL test_map_index edge case\n"; return false;
        }
        MapIndexPair edge_pair = map_index_pair(0, UINT64_MAX);
        if (edge_pair.first != map_index(0) ||
            edge_pair.second != map_index(UINT64_MAX)) {
            std::cout << "FAIL test_map_index_pair edge case\n"; return false;
        }
        std::cout << "PASS test_map_index\n";
        return true;
    }

    __attribute__((cold)) bool test_isqrt_correctness()
    {
        if (isqrt(0)!=0||isqrt(1)!=1||isqrt(2)!=1||isqrt(3)!=1||isqrt(4)!=2) return false;
        for (uint64_t i = 0; i < 100000; i++) {
            uint64_t sq = i * i;
            if (isqrt(sq) != i) {
                printf("FAIL isqrt(%llu) = %llu expected %llu\n",
                       (unsigned long long)sq, (unsigned long long)isqrt(sq),
                       (unsigned long long)i);
                return false;
            }
        }
        for (uint64_t i = 1; i < 10000; i++) {
            uint64_t n = i*i + i/2, root = isqrt(n);
            if (root*root > n || (root+1)*(root+1) <= n) {
                printf("FAIL isqrt(%llu) = %llu\n",
                       (unsigned long long)n, (unsigned long long)root);
                return false;
            }
        }
        std::cout << "PASS test_isqrt_correctness\n";
        return true;
    }

    __attribute__((cold)) bool test_modular_power()
    {
        if (modular_power(2,3,5)!=3) return false;
        if (modular_power(3,4,7)!=4) return false;
        if (modular_power(5,0,13)!=1) return false;
        if (modular_power(0,5,13)!=0) return false;
        if (modular_power(7,1,13)!=7) return false;
        modular_power(2,3,0);
        std::cout << "PASS test_modular_power\n";
        return true;
    }

    __attribute__((cold)) bool test_stage3_single_iteration()
    {
        byte input[112] = {0};
        auto worker = std::make_unique<workerData_xelis_v3>();
        memset(worker->scratchPad, 0, XELIS_MEMORY_SIZE_V3 * 8);
        get_stage_1()(input, worker->scratchPad, 112);

        uint64_t *mem_buffer_a = worker->scratchPad;
        uint64_t *mem_buffer_b = worker->scratchPad + XELIS_BUFFER_SIZE_V3;
        uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
        uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;

        printf("\nStage3 single iteration trace:\n");
        printf("  addr_a=0x%016llx addr_b=0x%016llx\n",
               (unsigned long long)addr_a, (unsigned long long)addr_b);

        uint64_t mem_a = mem_buffer_a[addr_a % XELIS_BUFFER_SIZE_V3];
        uint64_t mem_b = mem_buffer_b[addr_b % XELIS_BUFFER_SIZE_V3];

        uint8_t key[17] = "xelishash-pow-v3";
        uint8_t block[16];
        uint64_to_le_bytes(mem_b, block);
        uint64_to_le_bytes(mem_a, block + 8);
        aes_round(block, key);
        uint64_t hash1  = le_bytes_to_uint64(block);
        uint64_t hash2  = le_bytes_to_uint64(block + 8);
        uint64_t result = ~(hash1 ^ hash2);

        printf("  mem_a=0x%016llx mem_b=0x%016llx result=0x%016llx\n",
               (unsigned long long)mem_a, (unsigned long long)mem_b,
               (unsigned long long)result);

        size_t r = 0;
        for (size_t j = 0; j < 3; ++j) {
            uint64_t a = mem_buffer_a[map_index(result)];
            uint64_t b = mem_buffer_b[map_index(~ROTR(result, r))];
            uint64_t c = (r < XELIS_BUFFER_SIZE_V3) ? mem_buffer_a[r] : mem_buffer_b[r - XELIS_BUFFER_SIZE_V3];
            size_t r_before = r++;
uint32_t op_raw = ROTL(result, (uint32_t)c);
uint64_t v      = execute_operation(op_raw, a, b, c, r, result, 0, j);
            uint64_t idx_seed = v ^ result;
            result = ROTL(idx_seed, r);
            printf("  j=%zu r=%zu op=%u v=0x%016llx result=0x%016llx\n",
                   j, r_before, op_raw,
                   (unsigned long long)v, (unsigned long long)result);
        }
        return true;
    }
}

#include <thread>
#include <atomic>
#include <barrier>
#include <map>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <tnn-hugepages.hpp>
#include <tnn-common.hpp>
#include <tnn_affinity.h>


__attribute__((cold)) static unsigned computeHTThreshold(unsigned num_threads)
{
    // Read L3 size from sysfs
    // /sys/devices/system/cpu/cpu0/cache/index3/size
    // Returns KB as integer
    auto readL3KB = []() -> size_t {
        for (unsigned cpu = 0; cpu < 16; cpu++) {
            for (unsigned idx = 0; idx < 8; idx++) {
                std::string level_path = "/sys/devices/system/cpu/cpu"
                    + std::to_string(cpu) + "/cache/index"
                    + std::to_string(idx) + "/level";
                std::string type_path  = "/sys/devices/system/cpu/cpu"
                    + std::to_string(cpu) + "/cache/index"
                    + std::to_string(idx) + "/size";

                std::ifstream f_level(level_path);
                if (!f_level.is_open()) continue;
                int level = 0;
                f_level >> level;
                if (level != 3) continue;

                std::ifstream f_size(type_path);
                if (!f_size.is_open()) continue;
                std::string size_str;
                f_size >> size_str;

                // Parse "32768K" or "32M"
                size_t val = std::stoull(size_str);
                if (size_str.back() == 'M') val *= 1024;
                return val;  // already in KB
            }
        }
        return 32768;  // fallback: assume 32MB
    };

    static size_t l3_kb = readL3KB();
    constexpr size_t scratchpad_kb = (XELIS_MEMORY_SIZE_V3 * 8) / 1024;  // 531KB

    // How many scratchpads fit in L3 with headroom
    // Use 80% of L3 to leave room for code, stack, other data
    size_t usable_l3_kb    = (l3_kb * 8) / 10;
    size_t fits_in_l3      = usable_l3_kb / scratchpad_kb;

    // Get physical core count
    unsigned phys_cores    = std::thread::hardware_concurrency();

    // NT kicks in when adding more threads would overflow L3
    // i.e., when thread_index >= fits_in_l3 AND thread_index >= phys_cores
    // The second condition ensures we never NT a physical-only thread
    unsigned threshold     = (unsigned)(std::min)(
                                 fits_in_l3,
                                 (size_t)phys_cores
                             );

    // Clamp to actual thread count
    return (std::min)(threshold, num_threads);
}

// ============================================================================
// MULTI-THREADED BENCHMARK
// ============================================================================

struct MTBenchConfig {
    const char* id;
    const char* description;
    Stage1Fn    stage1;
    Stage1Fn    nt_stage1;
    Stage3Fn    stage3;
};

#if defined(__x86_64__)
static MTBenchConfig MT_CONFIGS[] = {
    { "L1+S",           "S3:L1 lookahead + scratch (default)",   get_stage_1(),       nullptr,           pick_stage_3() },
    { "hybrid",         "S3:L1+S + S1:pipelined P, NT HT",      get_stage_1(),       get_stage_1_nt(),  pick_stage_3() },
};
#else
static MTBenchConfig MT_CONFIGS[] = {
    { "baseline",       "S3:baseline",                           get_stage_1(),       nullptr,           pick_stage_3() },
};
#endif

static constexpr size_t NUM_MT_CONFIGS = sizeof(MT_CONFIGS) / sizeof(MT_CONFIGS[0]);

struct MTWorkerResult {
    uint64_t hashes_completed;
    double   elapsed_ms;
};

// Worker function for multi-threaded benchmark
__attribute__((cold)) static void mt_worker_fn(
    int thread_id,
    const MTBenchConfig* config,
    std::atomic<bool>* start_flag,
    std::atomic<bool>* stop_flag,
    MTWorkerResult* result,
    std::barrier<>* ready_barrier,
    bool use_affinity,
    unsigned ht_threshold)   // ← new param
{
    workerData_xelis_v3* worker =
        (workerData_xelis_v3*)malloc_huge_pages(sizeof(workerData_xelis_v3));

    // Select stage1 based on whether this thread is an HT sibling
    bool is_ht_sibling = use_affinity && ((unsigned)thread_id >= ht_threshold);
    Stage1Fn effective_stage1 = (is_ht_sibling && config->nt_stage1)
                                ? config->nt_stage1
                                : config->stage1;

    alignas(64) byte work[XELIS_TEMPLATE_SIZE] = {0};
    alignas(64) byte finalwork[XELIS_TEMPLATE_SIZE] = {0};
    byte hash[XELIS_HASH_SIZE];

    for (int i = 0; i < 112; i++)
        work[i] = (byte)((i * 17 + 31 + thread_id * 7) & 0xFF);

    uint64_t nonce = 0;

    ready_barrier->arrive_and_wait();
    while (!start_flag->load(std::memory_order_acquire)) {
#ifdef __x86_64__
        _mm_pause();
#endif
    }

    auto t0 = std::chrono::steady_clock::now();
    uint64_t count = 0;

    while (true) {
        nonce++;
        uint64_t n = ((thread_id) % (256 * 256)) | ((nonce & 0xFF) << 16) | (nonce << 24);
        memcpy(&work[40], &n, 8);
        memcpy(finalwork, work, XELIS_TEMPLATE_SIZE);

        effective_stage1(finalwork, worker->scratchPad, 112);
        config->stage3(worker->scratchPad, *worker);
        blake3((uint8_t*)worker->scratchPad, XELIS_OUTPUT_SIZE_V3, hash);

        std::reverse(hash, hash + 32);
        count++;

        if ((count & 63) == 0) {
            if (stop_flag->load(std::memory_order_relaxed)) break;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    result->hashes_completed = count;
    result->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    free_huge_pages(worker);
}

// ---------------------------------------------------------------------------
// Inner MT benchmark pass — runs all configs x thread counts, returns results
// ---------------------------------------------------------------------------

struct MTBenchResult {
    const char* config_id;
    unsigned int threads;
    double total_hs;
    double per_thread_hs;
    double scaling_efficiency;
};

__attribute__((cold)) static std::vector<MTBenchResult> xelis_mt_bench_pass(
    const char* label,
    const std::vector<unsigned int>& thread_counts,
    double duration_sec,
    unsigned ht_threshold)
{
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  XELIS V3 MT BENCH  --  " << std::left << std::setw(43) << label << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Duration: " << std::fixed << std::setprecision(1) << duration_sec
              << "s per config  │  Scratchpad: "
              << (XELIS_MEMORY_SIZE_V3 * 8 / 1024) << " KB"
              << std::string(22, ' ') << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    std::vector<MTBenchResult> all_results;
    std::map<const char*, double> single_thread_hs;

    for (const auto& config : MT_CONFIGS) {
        std::cout << "Testing config: " << config.id << " (" << config.description << ")\n";
        std::cout << std::string(70, '-') << "\n";

        for (unsigned int num_threads : thread_counts) {
            std::cout << "  " << std::setw(2) << num_threads << " thread(s): " << std::flush;

            std::vector<std::thread> bench_threads(num_threads);
            std::vector<MTWorkerResult> results(num_threads);

            std::atomic<bool> start_flag{false};
            std::atomic<bool> stop_flag{false};
            std::barrier ready_barrier(num_threads + 1);

            for (unsigned int t = 0; t < num_threads; t++) {
                bench_threads[t] = std::thread(mt_worker_fn, t, &config,
                                                &start_flag, &stop_flag, &results[t],
                                                &ready_barrier, lockThreads, ht_threshold);

                if (lockThreads) {
                    setAffinity(bench_threads[t].native_handle(), t);
                }
            }

            ready_barrier.arrive_and_wait();
            start_flag.store(true, std::memory_order_release);

            std::this_thread::sleep_for(
                std::chrono::milliseconds((int)(duration_sec * 1000)));

            stop_flag.store(true, std::memory_order_release);

            for (auto& t : bench_threads) {
                t.join();
            }

            uint64_t total_hashes = 0;
            double max_elapsed = 0;
            for (const auto& r : results) {
                total_hashes += r.hashes_completed;
                max_elapsed = (std::max)(max_elapsed, r.elapsed_ms);
            }

            double total_hs = (total_hashes * 1000.0) / max_elapsed;
            double per_thread_hs = total_hs / num_threads;

            double efficiency = 100.0;
            if (num_threads == 1) {
                single_thread_hs[config.id] = total_hs;
            } else {
                double ideal = single_thread_hs[config.id] * num_threads;
                efficiency = (total_hs / ideal) * 100.0;
            }

            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(8) << total_hs << " H/s total, "
                      << std::setw(8) << per_thread_hs << " H/s/thread, "
                      << std::setw(5) << efficiency << "% scaling\n";

            all_results.push_back({config.id, num_threads, total_hs, per_thread_hs, efficiency});
        }
        std::cout << "\n";
    }

    // Summary table
    std::cout << "┌───────────┬";
    for (size_t i = 0; i < NUM_MT_CONFIGS; i++) std::cout << "────────────────┬";
    std::cout << "────────────┐\n";
    std::cout << "│  Threads  │";
    for (const auto& config : MT_CONFIGS)
        std::cout << "  " << std::setw(12) << config.id << "  │";
    std::cout << "   Winner   │\n";
    std::cout << "├───────────┼";
    for (size_t i = 0; i < NUM_MT_CONFIGS; i++) std::cout << "────────────────┼";
    std::cout << "────────────┤\n";

    for (unsigned int tc : thread_counts) {
        std::cout << "│     " << std::setw(2) << tc << "    │";
        double best_hs = 0;
        const char* best_id = "";
        for (const auto& config : MT_CONFIGS) {
            for (const auto& r : all_results) {
                if (r.threads == tc && strcmp(r.config_id, config.id) == 0) {
                    std::cout << "  " << std::setw(10) << r.total_hs << " H/s│";
                    if (r.total_hs > best_hs) { best_hs = r.total_hs; best_id = r.config_id; }
                    break;
                }
            }
        }
        std::cout << "  " << std::setw(10) << best_id << "│\n";
    }
    std::cout << "└───────────┴";
    for (size_t i = 0; i < NUM_MT_CONFIGS; i++) std::cout << "────────────────┴";
    std::cout << "────────────┘\n\n";
    fflush(stdout);

    return all_results;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

__attribute__((cold)) static void xelis_benchmark_multithread()
{
    unsigned hw = std::thread::hardware_concurrency();

    // 1, 2, 4, 4+4, 4+4+4, ... up to 32, skip if > hw
    std::vector<unsigned int> thread_counts;
    for (unsigned t : {1u, 2u, 4u})
        if (t <= hw) thread_counts.push_back(t);
    for (unsigned t = 8; t <= 32 && t <= hw; t += 4)
        thread_counts.push_back(t);

    constexpr double BENCH_DURATION_SEC = 10.0;
    unsigned ht_threshold = computeHTThreshold(hw);

    xelis_mt_bench_pass("Prefetch A/B", thread_counts, BENCH_DURATION_SEC, ht_threshold);
}

// ============================================================================
// STARTUP TUNE — sweeps S1 (pipelined/serial) × S3 (goto/switch/merged) × hybrid
// ============================================================================

bool xelis_v3_use_hybrid = false;
bool xelis_v3_use_switch = false;
bool xelis_v3_use_merged = false;
unsigned xelis_v3_ht_threshold = UINT_MAX;

struct XelisTuneConfig {
    const char *name;
    Stage1Fn    s1;         // stage1 for physical-core threads
    Stage1Fn    s1_ht;      // stage1 for HT-sibling threads (nullptr = same as s1)
    Stage3Fn    s3;
    bool        is_serial;  // true if s1 is serial variant
    bool        is_switch;
    bool        is_merged;
    bool        is_hybrid;
};

__attribute__((cold)) void xelis_tune_v3(int num_threads)
{
    constexpr double WARMUP_SEC  = 2.0;
    constexpr double BENCH_SEC   = 8.0;

    unsigned phys = std::thread::hardware_concurrency();
    unsigned ht_thresh = (std::min)(phys, (unsigned)num_threads);

    Stage1Fn s1_pipe   = g_stage1_override    ? g_stage1_override    : get_stage_1();
    Stage1Fn s1_nt     = g_stage1_nt_override ? g_stage1_nt_override : get_stage_1_nt();
    Stage1Fn s1_serial = get_stage_1_serial();
    Stage3Fn s3_goto   = pick_stage_3();
    Stage3Fn s3_sw     = pick_stage_3_switch();
    Stage3Fn s3_merge  = pick_stage_3_merged();

    //                    name                    s1          s1_ht    s3        serial  switch  merged  hybrid
    XelisTuneConfig configs[] = {
        { "pipe+goto",           s1_pipe,   nullptr,  s3_goto,   false, false, false, false },
        { "pipe+goto+hyb",      s1_pipe,   s1_nt,    s3_goto,   false, false, false, true  },
        // { "pipe+switch",        s1_pipe,   nullptr,  s3_sw,     false, true,  false, false },
        // { "pipe+switch+hyb",    s1_pipe,   s1_nt,    s3_sw,     false, true,  false, true  },
        // { "pipe+merged",        s1_pipe,   nullptr,  s3_merge,  false, false, true,  false },
        // { "pipe+merged+hyb",    s1_pipe,   s1_nt,    s3_merge,  false, false, true,  true  },
        { "ser+goto",           s1_serial, nullptr,  s3_goto,   true,  false, false, false },
        { "ser+goto+hyb",       s1_serial, s1_nt,    s3_goto,   true,  false, false, true  },
        // { "ser+switch",         s1_serial, nullptr,  s3_sw,     true,  true,  false, false },
        // { "ser+switch+hyb",     s1_serial, s1_nt,    s3_sw,     true,  true,  false, true  },
        // { "ser+merged",         s1_serial, nullptr,  s3_merge,  true,  false, true,  false },
        // { "ser+merged+hyb",     s1_serial, s1_nt,    s3_merge,  true,  false, true,  true  },
    };
    constexpr size_t NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

    printf("Xelis v3 tune: %d threads, %u physical cores, %.0fs warmup + %.0fs bench x %zu configs\n",
           num_threads, phys, WARMUP_SEC, BENCH_SEC, NUM_CONFIGS);
    fflush(stdout);

    double best_hs = 0;
    size_t best_idx = 0;

    for (size_t ci = 0; ci < NUM_CONFIGS; ci++)
    {
        auto &cfg = configs[ci];
        std::atomic<uint64_t> total_hashes{0};
        std::atomic<bool> stop_flag{false};
        std::barrier ready_barrier(num_threads + 1);

        std::vector<std::thread> threads(num_threads);
        for (int t = 0; t < num_threads; t++)
        {
            threads[t] = std::thread([&, t]() {
                auto *worker = (workerData_xelis_v3 *)malloc_huge_pages(sizeof(workerData_xelis_v3));
                alignas(64) byte input[XELIS_TEMPLATE_SIZE] = {0};
                byte hash[XELIS_HASH_SIZE];

                for (int k = 0; k < 112; k++)
                    input[k] = (byte)((k * 17 + 31 + t * 7) & 0xFF);

                auto write_nonce = [&](uint64_t nonce) {
                    uint64_t n = ((uint64_t)t % (256 * 256))
                               | ((nonce & 0xFFULL) << 16)
                               | (nonce << 24);
                    memcpy(input + 40, &n, sizeof(n));
                };

                bool is_ht = ((unsigned)t >= ht_thresh);
                Stage1Fn s1_fn = (is_ht && cfg.s1_ht) ? cfg.s1_ht : cfg.s1;
                Stage3Fn s3_fn = cfg.s3;

                ready_barrier.arrive_and_wait();

                // Warmup
                uint64_t warmup_count = 0;
                auto warmup_end = std::chrono::steady_clock::now()
                    + std::chrono::milliseconds((int)(WARMUP_SEC * 1000));
                while (std::chrono::steady_clock::now() < warmup_end) {
                    write_nonce(++warmup_count);
                    s1_fn(input, worker->scratchPad, 112);
                    s3_fn(worker->scratchPad, *worker);
                    blake3((uint8_t *)worker->scratchPad, XELIS_OUTPUT_SIZE_V3, hash);
                    if ((warmup_count & 63) == 0)
                        std::this_thread::yield();
                }

                // Bench
                uint64_t count = 0;
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    write_nonce(++count);
                    s1_fn(input, worker->scratchPad, 112);
                    s3_fn(worker->scratchPad, *worker);
                    blake3((uint8_t *)worker->scratchPad, XELIS_OUTPUT_SIZE_V3, hash);
                    if ((count & 63) == 0)
                        std::this_thread::yield();
                }
                total_hashes.fetch_add(count, std::memory_order_relaxed);
                free_huge_pages(worker);
            });

            if (lockThreads)
                setAffinity(threads[t].native_handle(), t);
        }

        ready_barrier.arrive_and_wait();

        // Wait for warmup + bench duration
        std::this_thread::sleep_for(
            std::chrono::milliseconds((int)((WARMUP_SEC + BENCH_SEC) * 1000)));
        stop_flag.store(true, std::memory_order_release);

        for (auto &th : threads) th.join();

        double hs = (double)total_hashes.load() / BENCH_SEC;
        printf("  %-20s: %.2f H/s\n", cfg.name, hs);
        fflush(stdout);

        if (hs > best_hs) {
            best_hs = hs;
            best_idx = ci;
        }
    }

    auto &winner = configs[best_idx];
    xelis_v3_use_hybrid   = winner.is_hybrid;
    xelis_v3_use_switch   = winner.is_switch;
    xelis_v3_use_merged   = winner.is_merged;
    xelis_v3_ht_threshold = ht_thresh;

    // If serial S1 won, override the global stage1 dispatch so all hash functions use it
    if (winner.is_serial) {
        g_stage1_override    = get_stage_1_serial();
        g_stage1_nt_override = get_stage_1_serial();
    }

    printf("Selected: %s (%.2f H/s)\n\n", winner.name, best_hs);
    fflush(stdout);
}

// ============================================================================
// TEST RUNNER
// ============================================================================

__attribute__((cold)) void xelis_benchmark_chacha_variants()
{
    constexpr uint32_t ITERATIONS = 50000;
    constexpr size_t bytes = XELIS_BYTES_PER_CHUNK_V3;
    
    uint8_t key[32]   = {0};
    uint8_t nonce[12] = {0};
    
    // Allocate output buffers
    auto buf = std::make_unique<uint8_t[]>(bytes * 4);
    uint8_t* outputs[4];
    for (int i = 0; i < 4; i++) outputs[i] = buf.get() + i * bytes;

    // Single stream
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t k = 0; k < ITERATIONS; k++) {
        uint8_t state[48] = {0};
        ChaCha20SetKey(state, key);
        ChaCha20SetNonce(state, nonce);
        ChaCha20EncryptBytes(state, NULL, outputs[0], bytes, 8);
    }
    auto t1 = std::chrono::steady_clock::now();
    double single_ms = std::chrono::duration<double,std::milli>(t1-t0).count()/ITERATIONS;

    // 4-stream Xelis
    uint8_t keys[4][32]   = {};
    uint8_t nonces[4][12] = {};
    auto t2 = std::chrono::steady_clock::now();
    for (uint32_t k = 0; k < ITERATIONS; k++) {
        ChaCha20EncryptXelis(keys, nonces, outputs, bytes, 8);
    }
    auto t3 = std::chrono::steady_clock::now();
    double xelis_ms = std::chrono::duration<double,std::milli>(t3-t2).count()/ITERATIONS;

    // NT variant
    auto t4 = std::chrono::steady_clock::now();
    for (uint32_t k = 0; k < ITERATIONS; k++) {
        ChaCha20EncryptXelisNT(keys, nonces, outputs, bytes, 8);
    }
    auto t5 = std::chrono::steady_clock::now();
    double nt_ms = std::chrono::duration<double,std::milli>(t5-t4).count()/ITERATIONS;

    double chunk_kb = bytes / 1024.0;
    std::cout << "\n=== ChaCha20 Variant Throughput ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Chunk size: " << chunk_kb << " KB  x4 = " 
              << chunk_kb*4 << " KB\n\n";
    auto gb = [&](double ms) {
        return (bytes * 4) / (ms / 1000.0) / (1024.0*1024.0*1024.0);
    };
    std::cout << "Single stream: " << std::setw(7) << single_ms 
              << " ms  " << std::setw(6) << gb(single_ms) << " GB/s\n";
    std::cout << "4-stream Xelis:" << std::setw(6) << xelis_ms 
              << " ms  " << std::setw(6) << gb(xelis_ms) << " GB/s\n";
    std::cout << "4-stream NT:   " << std::setw(7) << nt_ms    
              << " ms  " << std::setw(6) << gb(nt_ms)    << " GB/s\n";
    std::cout << "Xelis vs single: " 
              << ((single_ms/xelis_ms)-1.0)*100 << "% faster\n";
    std::cout << "NT vs Xelis:     " 
              << ((xelis_ms/nt_ms)-1.0)*100     << "% faster\n";
}

__attribute__((cold)) int xelis_runTests_v3()
{
    std::cout << "\n=== Running Xelis Hash v3 Tests ===\n";
    fflush(stdout);

    bool ok = true;
    ok &= xelis_tests_v3::test_zero_hash();
    fflush(stdout);
    ok &= xelis_tests_v3::test_verify_output();
    fflush(stdout);
    ok &= xelis_tests_v3::test_reused_scratchpad();
    fflush(stdout);
    ok &= xelis_tests_v3::test_map_index();
    fflush(stdout);
    ok &= xelis_tests_v3::test_pick_half();
    fflush(stdout);
    ok &= xelis_tests_v3::test_isqrt_correctness();
    fflush(stdout);
    ok &= xelis_tests_v3::test_modular_power();
    fflush(stdout);
    ok &= xelis_tests_v3::test_stage3_single_iteration();
    fflush(stdout);

    if (ok) {
        std::cout << "\nXELIS-HASH-V3: All tests passed! ✓\n\n";
        fflush(stdout);
        return 0;
    }

    std::cout << "\nXELIS-HASH-V3: Some tests failed! ✗\n";
    fflush(stdout);
    return 1;
}
