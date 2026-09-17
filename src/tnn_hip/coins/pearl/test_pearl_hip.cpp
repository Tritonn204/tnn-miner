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
        const ExecutionOptions options{ExecutionMode::Validation, {256, 256, 2048}, 512};
        GPUAlgorithm algorithm(pearl_gpu_config(options));
        require(algorithm.initialize(0), "Pearl validation initialization failed");
        auto job = offline_job(options.shape, true);
        for (unsigned attempt = 0; attempt < 4; ++attempt) {
            // Exercise both owned operand slots, a target update and dev cache.
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
            const auto batch = algorithm.mine_batch(attempt, 1);
            require(batch.count == 1 && batch.work_multiplier == options.shape.macs(),
                    "Incorrect work accounting");
            const auto winners = pearl_validate_batch(batch, true);
            require(winners > 0, "Validation fixture found no winners");
            TNN_LOG_INFO("[PEARL-HIP-TEST] attempt=%u candidates=512 winners=%zu CPU/GPU/proof "
                         "checks passed\n",
                         attempt, winners);
        }
        algorithm.cleanup();

        // Same production collector must fail explicitly when capacity is exceeded.
        auto overflow_options = options;
        overflow_options.winner_capacity = 7;
        GPUAlgorithm overflow(pearl_gpu_config(overflow_options));
        require(overflow.initialize(0), "Pearl overflow initialization failed");
        overflow.set_job_snapshot(offline_job(options.shape, true));
        bool rejected = false;
        try {
            (void)overflow.mine_batch(0, 1);
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

int bench_pearl_hip(uint32_t m, uint32_t n, uint32_t k) {
    try {
        const ExecutionOptions options{ExecutionMode::Benchmark, {m, n, k}, 256};
        options.shape.validate();
        TNN_LOG_INFO("\n[PEARL-HIP-BENCH] %ux%ux%u production preparation + fused jackpot + "
                     "readback; offline\n",
                     m, n, k);
        GPUAlgorithm algorithm(pearl_gpu_config(options));
        require(algorithm.initialize(0), "Pearl benchmark initialization failed");
        algorithm.set_job_snapshot(offline_job(options.shape, false));
        for (unsigned i = 0; i < 3; ++i)
            (void)algorithm.mine_batch(i, 1);

        using Clock = std::chrono::steady_clock;
        uint64_t completed = 0;
        const auto started = Clock::now();
        do {
            const auto batch = algorithm.mine_batch(completed + 3, 1);
            require(batch.count == 1, "Incomplete Pearl benchmark batch");
            ++completed;
        } while (Clock::now() - started < std::chrono::seconds(5));
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
