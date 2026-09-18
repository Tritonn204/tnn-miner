#include "test_pearl_hip.h"
#include "pearl_mining.hpp"
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

} // namespace

int test_pearl_hip() {
    try {
        TNN_LOG_INFO("\n[PEARL-HIP-TEST] Production preparation, jackpots and proofs (offline)\n");
        // Two independent A operands exercise batch isolation without turning
        // the exhaustive CPU proof check into a full mining-size workload.
        const ExecutionOptions options{ExecutionMode::Validation, {256, 256, 2048}, 512, 2};
        GPUAlgorithm algorithm(pearl_gpu_config(options));
        require(algorithm.initialize(0), "Pearl validation initialization failed");
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
        for (unsigned version : {2u, 3u}) {
            const ExecutionOptions rectangular{ExecutionMode::Validation, {384, 256, 4096}, 768, 1};
            GPUAlgorithm check(pearl_gpu_config(rectangular));
            require(check.initialize(0), "Pearl rectangular initialization failed");

            auto rectangular_job = offline_job(rectangular.shape, true);
            rectangular_job.pearl_cert_version = version;
            check.set_job_snapshot(rectangular_job);

            const auto batch = check.mine_batch(0, rectangular.batch_size);
            require(pearl_validate_batch(batch, true) == 768,
                    "Incomplete rectangular candidate coverage");
            TNN_LOG_INFO("[PEARL-HIP-TEST] 384x256x4096 cert=%u candidates=768 passed\n", version);
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

int bench_pearl_hip(uint32_t m, uint32_t n, uint32_t k, uint32_t seconds_requested) {
    try {
        require(seconds_requested >= 1 && seconds_requested <= 600,
                "Pearl benchmark duration must be between 1 and 600 seconds");
        const ExecutionOptions options{ExecutionMode::Benchmark, {m, n, k}, 256};
        options.shape.validate();
        TNN_LOG_INFO("\n[PEARL-HIP-BENCH] %ux%ux%u production preparation + fused jackpot + "
                     "readback; offline\n",
                     m, n, k);
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
