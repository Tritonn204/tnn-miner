#include "test_pearl_hip.h"
#include "pearl_mining.hpp"
#include "pearl_tuning.hpp"
#include <tnn_hip/common/gpu_algo_impl.hpp>
#include <tnn_log.hpp>
#include <chrono>
#include <stdexcept>

namespace tnn::pearl {
namespace {

JobSnapshot offline_job(native::Shape shape, bool easy) {
    native::Header header{};
    header[0] = 1;
    header[68] = 17;
    JobSnapshot job(header.data(), header.size(), 1, 1, ALGO_PEARL_POUW, "offline-pearl", false);
    job.connection_generation = 1;
    job.pearl_cert_version = 3;
    job.block_height = 1;
    job.raw_target[0] = 1;
    if (easy) {
        // Largest wire target whose configuration-adjusted bound fits U256.
        const auto factor = native::jackpot_work(shape.k);
        uint64_t remainder = 0;
        for (int i = 31; i >= 0; --i) {
            const uint64_t digit = remainder * 256 + 255;
            job.raw_target[i] = uint8_t(digit / factor);
            remainder = digit % factor;
        }
    }
    return job;
}

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void test_tune_cache_contract() {
    auto rejected_configuration = [](ExecutionOptions options) {
        bool rejected = false;
        try { (void)pearl_gpu_config(options); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Unqualified experimental configuration accepted");
    };
    ExecutionOptions experimental;
    experimental.backend = Backend::ExperimentalGfx1201;
    rejected_configuration(experimental); // No foreign production mining.
    experimental.mode = ExecutionMode::Benchmark;
    experimental.recipe = 8;
    rejected_configuration(experimental);
    ExecutionOptions prepared;
    prepared.test_workload = TestWorkload::PreparedRaw;
    rejected_configuration(prepared); // Reused inputs cannot become mining work.

    oroDeviceProp_t properties{};
    require(oroGetDeviceProperties(&properties, tnn_get_device(0)) == oroSuccess,
            "Cannot query tune validation device");
    const auto config = pearl_gpu_config();
    TuningResult valid{};
    valid.valid = true;
    valid.block_size = 128;
    valid.batch_size = ExecutionOptions{}.batch_size;
    valid.num_blocks = 4096;
    valid.hashrate = 52e12;
    valid.batch_time_ms = 84;
    valid.tune_keys = {{"m", 8192}, {"n", 8192}, {"k", 4096},
                       {"pearl_version", tuning::version}, {"pearl_backend", 1100}};
    require(config.custom_tune_validate_fn(valid, properties, 0), "Valid tune rejected");

    // Exercise the production validator without editing the user's tune cache
    // or launching any sweep. Each corruption must independently fail closed.
    auto rejected = [&](auto corrupt) {
        auto result = valid;
        corrupt(result);
        require(!config.custom_tune_validate_fn(result, properties, 0),
                "Invalid cached tune accepted");
    };
    rejected([](auto& result) { result.batch_size = 32; });
    rejected([](auto& result) { result.tune_keys["pearl_version"] = tuning::version - 1; });
    rejected([](auto& result) { result.tune_keys["pearl_backend"] = 1200; });
    rejected([](auto& result) { result.tune_keys["k"] = 16384; });
    rejected([](auto& result) { result.tune_keys.erase("m"); });
    rejected([](auto& result) { result.num_blocks = 1; });
    rejected([](auto& result) { result.hashrate = -1; });
    TNN_LOG_INFO("[PEARL-HIP-TEST] Cached tune validation passed\n");
}

} // namespace

int test_pearl_hip(bool experimental_gfx12, unsigned recipe) {
    try {
        if (experimental_gfx12) {
            oroDeviceProp_t props{};
            require(oroGetDeviceProperties(&props, tnn_get_device(0)) == oroSuccess &&
                    tnn_is_amd_device(0) && parse_gfx_number(props.gcnArchName) == 1201,
                    "Experimental test requires gfx1201; no kernels launched");
        }
        TNN_LOG_INFO("\n[PEARL-HIP-TEST] Production preparation, jackpots and proofs (offline)\n");
        const auto candidates = tuning::coarse();
        require(candidates.size() == 24, "Incorrect coarse tuning coverage");
        for (const auto shape : candidates)
            require(tuning::supported(shape) && shape.m % 1024 == 0 && shape.n % 1024 == 0,
                    "Invalid tuning grid");
        const auto refinement = tuning::refine({tuning::default_shape});
        require(std::any_of(refinement.begin(), refinement.end(),
                            [](auto shape) { return shape.m == 6144 || shape.n == 6144; }),
                "Refinement lost intermediate shapes");
        require(!tuning::supported({8192, 8192, 16384, native::CandidateLayout::native_4x32}),
                "Unqualified K accepted by tuner");
        // Two independent A operands exercise batch isolation without turning
        // the exhaustive CPU proof check into a full mining-size workload.
        ExecutionOptions options{ExecutionMode::Validation, {256, 256, 2048}, 512, 2};
        options.backend = experimental_gfx12 ? Backend::ExperimentalGfx1201 : Backend::QualifiedGfx1100;
        options.recipe = recipe;
        GPUAlgorithm algorithm(pearl_gpu_config(options));
        require(algorithm.initialize(0), "Pearl validation initialization failed");
        if (!experimental_gfx12) test_tune_cache_contract();
        auto job = offline_job(options.shape, true);
        for (unsigned attempt = 0; attempt < 4; ++attempt) {
            // Exercise independent attempts, a target update and the dev cache.
            if (attempt == 2) {
                job.raw_target[29] /= 2;
                job.job_id_str = "offline-retarget";
            }
            if (attempt == 3) {
                job.is_dev = true;
                job.connection_generation = 2;
                job.work_template[0] ^= 1;
                job.job_id_str = "offline-dev";
            }
            algorithm.set_job_snapshot(job);
            const auto batch = algorithm.mine_batch(attempt, options.batch_size);
            require(batch.count == options.batch_size &&
                        batch.work_multiplier == options.shape.macs(),
                    "Incorrect work accounting");
            const auto winners = pearl_validate_batch(batch, true);
            require(winners > 0, "Validation fixture found no winners");
            TNN_LOG_INFO("[PEARL-HIP-TEST] batch=%u candidates=1024 winners=%zu CPU/GPU/proof "
                         "checks passed\n",
                         attempt, winners);
        }
        algorithm.cleanup();

        // Cover the production K and a non-power-of-two Merkle tree. Both
        // certificate versions must agree with the independent CPU reference.
        for (unsigned depth : {4096u, 8192u}) {
          for (unsigned version : {2u, 3u}) {
            ExecutionOptions rectangular{ExecutionMode::Validation, {384, 256, depth}, 768, 1};
            rectangular.backend = options.backend;
            rectangular.recipe = recipe;
            GPUAlgorithm check(pearl_gpu_config(rectangular));
            require(check.initialize(0), "Pearl rectangular initialization failed");

            auto rectangular_job = offline_job(rectangular.shape, true);
            rectangular_job.pearl_cert_version = version;
            check.set_job_snapshot(rectangular_job);

            const auto batch = check.mine_batch(0, rectangular.batch_size);
            require(pearl_validate_batch(batch, true) == 768,
                    "Incomplete rectangular candidate coverage");
            TNN_LOG_INFO("[PEARL-HIP-TEST] 384x256x%u cert=%u candidates=768 passed\n", depth, version);
          }
        }

        // Same production collector must fail explicitly when capacity is exceeded.
        auto overflow_options = options;
        overflow_options.winner_capacity = 7;
        GPUAlgorithm overflow(pearl_gpu_config(overflow_options));
        require(overflow.initialize(0), "Pearl overflow initialization failed");
        overflow.set_job_snapshot(offline_job(options.shape, true));
        bool rejected = false;
        try {
            (void)overflow.mine_batch(0, options.batch_size);
        } catch (const std::exception &error) {
            rejected =
                std::string(error.what()).find("winner capacity exceeded") != std::string::npos;
        }
        require(rejected, "Winner overflow was not rejected");
        TNN_LOG_INFO("[PEARL-HIP-TEST] PASS: preparation, full candidate coverage, proof "
                     "snapshots, guards and overflow\n");
        return 0;
    } catch (const std::exception &error) {
        fflush(stdout);
        TNN_LOG_ERROR("\n[PEARL-HIP-TEST] %s\n", error.what());
        return 1;
    }
}

int tune_pearl_hip() {
    try {
        GPUAlgorithm algorithm(pearl_gpu_config());
        require(algorithm.initialize(0), "Pearl tuning initialization failed");
        const auto tune = algorithm.get_tuning_result();
        native::Shape shape{uint32_t(tune.tune_keys.at("m")), uint32_t(tune.tune_keys.at("n")),
                            uint32_t(tune.tune_keys.at("k")), native::CandidateLayout::native_4x32};
        algorithm.set_job_snapshot(offline_job(shape, false));
        const auto batch = algorithm.mine_batch(0, tune.batch_size);
        require(batch.count == tune.batch_size && batch.work_multiplier == shape.macs(),
                "Selected tuning result was not applied to the mining adapter");
        TNN_LOG_INFO("[PEARL-TUNE] PASS: selected configuration applied; no network connection\n");
        return 0;
    } catch (const std::exception& error) {
        TNN_LOG_ERROR("[PEARL-TUNE] %s\n", error.what());
        return 1;
    }
}

int bench_pearl_hip(uint32_t m, uint32_t n, uint32_t k, uint32_t seconds_requested,
                    bool experimental_gfx12, unsigned recipe, unsigned workload) {
    try {
        if (experimental_gfx12) {
            oroDeviceProp_t props{};
            require(oroGetDeviceProperties(&props, tnn_get_device(0)) == oroSuccess &&
                    tnn_is_amd_device(0) && parse_gfx_number(props.gcnArchName) == 1201,
                    "Experimental benchmark requires gfx1201; no kernels launched");
        }
        require(workload < 3, "Invalid tester workload");
        require(seconds_requested >= 1 && seconds_requested <= 600,
                "Pearl benchmark duration must be between 1 and 600 seconds");
        ExecutionOptions options{ExecutionMode::Benchmark, {m, n, k}, 256};
        options.backend = experimental_gfx12 ? Backend::ExperimentalGfx1201 : Backend::QualifiedGfx1100;
        options.recipe = recipe;
        options.test_workload = static_cast<TestWorkload>(workload);
        options.shape.validate();
        if (experimental_gfx12) {
            auto qualification = options;
            qualification.test_workload = TestWorkload::FreshFused;
            GPUAlgorithm check(pearl_gpu_config(qualification));
            require(check.initialize(0), "Shape qualification initialization failed");
            auto job = offline_job(options.shape, false);
            // Keep proof collection bounded: expect about eight winners per
            // batch regardless of shape, and verify them before any timing.
            const uint64_t candidates = uint64_t(m) * n / 128 * options.batch_size;
            const uint64_t divisor = native::jackpot_work(k, native::CandidateLayout::native_4x32) *
                                     std::max<uint64_t>(1, candidates / 8);
            uint64_t remainder = 0;
            for (int i = 31; i >= 0; --i) {
                const uint64_t digit = remainder * 256 + 255;
                job.raw_target[i] = uint8_t(digit / divisor);
                remainder = digit % divisor;
            }
            check.set_job_snapshot(job);
            size_t winners = 0;
            for (unsigned attempt = 0; attempt < 4 && !winners; ++attempt)
                winners = pearl_validate_batch(check.mine_batch(attempt, options.batch_size), false);
            require(winners > 0, "Shape qualification found no checkable winners");
            TNN_LOG_INFO("[PEARL-GFX12] Shape proof qualification passed (%zu winners)\n", winners);
        }
        TNN_LOG_INFO("[PEARL-HIP-BENCH] backend=%s recipe=%u workload=%u batch=%u\n",
                     experimental_gfx12 ? "gfx1201-experimental" : "gfx1100", recipe,
                     workload, options.batch_size);
        const char* labels[] = {"fresh preparation + fused jackpots + readback",
                                "prepared operands + fused jackpots + readback",
                                "prepared operands + D-free raw GEMM + readback"};
        TNN_LOG_INFO("\n[PEARL-HIP-BENCH] %ux%ux%u %s; offline\n", m, n, k, labels[workload]);
        GPUAlgorithm algorithm(pearl_gpu_config(options));
        require(algorithm.initialize(0), "Pearl benchmark initialization failed");
        algorithm.set_job_snapshot(offline_job(options.shape, false));
        for (unsigned i = 0; i < 3; ++i)
            (void)algorithm.mine_batch(i, options.batch_size);

        using Clock = std::chrono::steady_clock;
        uint64_t completed = 0;
        const auto started = Clock::now();
        do {
            const auto batch = algorithm.mine_batch(completed + 3, options.batch_size);
            require(batch.count == options.batch_size, "Incomplete Pearl benchmark batch");
            completed += batch.count;
        } while (Clock::now() - started < std::chrono::seconds(seconds_requested));
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        const double macs = double(completed) * options.shape.macs();
        TNN_LOG_INFO("[PEARL-HIP-BENCH] completed=%llu wall_s=%.6f TMAC/s=%.3f (network/proof "
                     "submission excluded)\n",
                     static_cast<unsigned long long>(completed), seconds, macs / (seconds * 1e12));
        return 0;
    } catch (const std::exception &error) {
        fflush(stdout);
        TNN_LOG_ERROR("\n[PEARL-HIP-BENCH] %s\n", error.what());
        return 1;
    }
}

} // namespace tnn::pearl
