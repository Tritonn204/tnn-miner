#include "pearl_tuning.hpp"
#include <tnn_hip/common/gpu_algo_impl.hpp>
#include <chrono>
#include <cmath>
#include <set>
#include <tuple>

namespace tnn::pearl {
namespace {

using Clock = std::chrono::steady_clock;

void check_cancelled() {
    if (g_autotune_stop.load(std::memory_order_relaxed))
        throw std::runtime_error("Pearl tuning cancelled; no result saved");
}

bool fits(native::Shape shape, const ExecutionOptions& options) {
    size_t free = 0, total = 0;
    if (oroMemGetInfo(&free, &total) != oroSuccess)
        throw std::runtime_error("Cannot query Pearl tuning memory budget");
    return tuning::fits_memory(shape, options.batch_size, options.winner_capacity, free, total);
}

JobSnapshot tune_job(native::Shape shape, unsigned batch, bool qualification) {
    native::Header header{};
    header[0] = 1;
    header[68] = 17;
    JobSnapshot job(header.data(), header.size(), 1, 1, ALGO_PEARL_POUW, "pearl-tune", false);
    job.connection_generation = 1;
    job.pearl_cert_version = 3;
    job.raw_target[0] = 1;
    if (qualification) {
        // About eight winners per batch, independent of matrix dimensions.
        const uint64_t candidates = uint64_t(shape.m) * shape.n / 128 * batch;
        const uint64_t divisor = native::jackpot_work(shape.k, shape.layout) *
                                 std::max<uint64_t>(1, candidates / 8);
        uint64_t remainder = 0;
        for (int i = 31; i >= 0; --i) {
            const uint64_t digit = remainder * 256 + 255;
            job.raw_target[i] = uint8_t(digit / divisor);
            remainder = digit % divisor;
        }
    }
    return job;
}

struct Measurement {
    native::Shape shape;
    double rate = 0;
    double batch_ms = 0;
};

Measurement measure(native::Shape shape, ExecutionOptions options, int device,
                    double seconds, bool qualify) {
    check_cancelled();
    options.mode = ExecutionMode::Benchmark; // Explicit shape: no recursive tune.
    options.shape = shape;
    GPUAlgorithm algorithm(pearl_gpu_config(options));
    if (!algorithm.initialize(device))
        throw std::runtime_error("Pearl tune candidate initialization failed");

    if (qualify) {
        algorithm.set_job_snapshot(tune_job(shape, options.batch_size, true));
        size_t winners = 0;
        for (unsigned i = 0; i < 4 && !winners; ++i) {
            check_cancelled();
            winners = pearl_validate_batch(algorithm.mine_batch(i, options.batch_size), false);
        }
        if (!winners)
            throw std::runtime_error("Pearl tune qualification found no checkable winners");
    }

    algorithm.set_job_snapshot(tune_job(shape, options.batch_size, false));
    for (unsigned i = 0; i < 3; ++i) {
        check_cancelled();
        (void)algorithm.mine_batch(i, options.batch_size);
    }
    uint64_t completed = 0, batches = 0;
    const auto start = Clock::now();
    do {
        check_cancelled();
        const auto result = algorithm.mine_batch(completed, options.batch_size);
        if (result.count != options.batch_size || result.work_multiplier != shape.macs())
            throw std::runtime_error("Pearl tune work accounting mismatch");
        completed += result.count;
        ++batches;
    } while (std::chrono::duration<double>(Clock::now() - start).count() < seconds);
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    return {shape, double(completed) * shape.macs() / elapsed, elapsed * 1000 / batches};
}

void result_for(const Measurement& winner, const ExecutionOptions& options, TuningResult& out) {
    out = {};
    out.valid = true;
    out.block_size = 128;
    out.batch_size = options.batch_size;
    out.num_blocks = int((winner.shape.m / 128) * (winner.shape.n / pearl_tile_n(options)));
    out.hashrate = winner.rate;
    out.batch_time_ms = winner.batch_ms;
    const bool retained = options.architecture == "gfx1100" && options.backend == Backend::Rdna3;
    out.tune_keys = {{"pearl_version", tuning::cache_version(options)},
                    {"pearl_backend", tuning::architecture_code(options)},
                    {"m", winner.shape.m}, {"n", winner.shape.n}, {"k", winner.shape.k}};
    if (!retained) {
        out.tune_keys["pearl_recipe"] = options.recipe;
        out.tune_keys["pearl_engine"] = tuning::engine_code(options);
    }
}

native::Shape shape_from(const TuningResult& result) {
    auto value = [&](const char* key) -> uint32_t {
        auto it = result.tune_keys.find(key);
        if (it == result.tune_keys.end() || it->second <= 0 || it->second > UINT32_MAX)
            return 0;
        return uint32_t(it->second);
    };
    return {value("m"), value("n"), value("k"), native::CandidateLayout::native_4x32};
}

bool tune(ExecutionOptions options, int device, bool enabled, TuningResult& out) {
    const auto baseline_shape = tuning::baseline(options, [&](auto shape) { return fits(shape, options); });
    if (!enabled) {
        result_for({baseline_shape}, options, out);
        return true;
    }
    if (options.backend == Backend::PortableSimt) {
        TNN_LOG_INFO("[PEARL-TUNE] Screening %zu SIMT recipes, then up to %u shapes; batch=%u\n",
                     tuning::recipes(options).size(), tuning::candidate_budget, options.batch_size);
        // Recipe screening is separate from the 32 unique matrix shapes.
        // Every optional kernel earns a process-local proof qualification.
        (void)measure(baseline_shape, options, device, 2.0, true);
        auto selected = options;
        for (unsigned recipe : tuning::recipes(options)) {
            if (recipe == options.recipe) continue;
            check_cancelled();
            auto trial = options;
            trial.recipe = recipe;
            try {
                pearl_qualify_recipe(device, trial);
            } catch (const std::exception& error) {
                check_cancelled();
                TNN_LOG_INFO("[PEARL-TUNE] Recipe %u unavailable: %s\n", recipe, error.what());
                // A device error is not a recoverable candidate rejection.
                if (oroDeviceSynchronize() != oroSuccess) throw;
                continue;
            }
            const auto sample = measure(baseline_shape, trial, device, 2.0, true);
            TNN_LOG_INFO("[PEARL-TUNE] Recipe %u: %.3f TMAC/s, %.1f ms/batch\n",
                         recipe, sample.rate / 1e12, sample.batch_ms);
            // Confirm against a fresh incumbent measurement, not a cold
            // baseline from before several CPU-heavy qualification suites.
            const auto control = measure(baseline_shape, selected, device, 2.0, false);
            const auto repeat = measure(baseline_shape, trial, device, 2.0, false);
            if (sample.rate > control.rate * 1.01 && repeat.rate > control.rate * 1.01) {
                selected = trial;
            }
        }
        options = selected;
        TNN_LOG_INFO("[PEARL-TUNE] Using recipe %u for shape screening\n", options.recipe);
    }
    TNN_LOG_INFO("[PEARL-TUNE] Up to %u shapes; fresh prep + fused jackpots + readback, batch=%u\n",
                 tuning::candidate_budget, options.batch_size);
    std::vector<Measurement> results;
    std::set<std::tuple<unsigned, unsigned, unsigned>> visited;
    auto screen = [&](native::Shape shape) {
        if (results.size() >= tuning::candidate_budget ||
            !visited.emplace(shape.m, shape.n, shape.k).second)
            return;
        check_cancelled();
        if (!fits(shape, options)) {
            TNN_LOG_INFO("[PEARL-TUNE] Skip %ux%ux%u: memory reserve\n", shape.m, shape.n, shape.k);
            return;
        }
        if (tuning::bringup_backend(options) && !results.empty()) {
            double projected = 0;
            for (const auto& prior : results)
                projected = std::max(projected, tuning::projected_batch_ms(
                    prior.shape, shape, prior.batch_ms));
            // Bound *growth* using measured fresh-prep batches. Never pretend
            // a host timeout can preempt a submitted GPU kernel.
            if (projected * 2 > 500) {
                TNN_LOG_INFO("[PEARL-TUNE] Skip %ux%ux%u: conservative batch estimate %.1f ms\n",
                             shape.m, shape.n, shape.k, projected * 2);
                return;
            }
        }
        auto sample = measure(shape, options, device, 2.0, true);
        results.push_back(sample);
        TNN_LOG_INFO("[PEARL-TUNE] %zu/%u %ux%ux%u %.3f TMAC/s, %.1f ms/batch\n",
                     results.size(), tuning::candidate_budget, shape.m, shape.n, shape.k,
                     sample.rate / 1e12, sample.batch_ms);
    };
    screen(baseline_shape);
    const auto coarse = tuning::coarse(options);
    for (auto shape : coarse) {
        if (tuning::bringup_backend(options) && results.size() >= 24) break;
        screen(shape);
    }
    auto rank = [&] {
        std::stable_sort(results.begin(), results.end(),
                         [](auto a, auto b) { return a.rate > b.rate; });
    };
    rank();
    std::vector<native::Shape> leaders;
    for (size_t i = 0; i < std::min<size_t>(2, results.size()); ++i)
        leaders.push_back(results[i].shape);
    for (auto shape : tuning::refine(leaders, tuning::domain_for(options)))
        screen(shape);
    // Boundary clipping/deduplication can leave refinement slots unfilled.
    // Continue around ranked coarse runners-up rather than remeasure a shape.
    const auto ranked = results;
    for (auto sample : ranked)
        for (auto shape : tuning::refine({sample.shape}, tuning::domain_for(options)))
            screen(shape);
    rank();

    std::vector<native::Shape> finalists{baseline_shape};
    for (auto sample : results) {
        if (!tuning::same(sample.shape, baseline_shape))
            finalists.push_back(sample.shape);
        if (finalists.size() == 3)
            break;
    }
    std::vector<std::array<Measurement, 3>> confirmation(finalists.size());
    for (unsigned round = 0; round < 3; ++round) {
        for (size_t step = 0; step < finalists.size(); ++step) {
            const size_t index = round % 2 ? finalists.size() - 1 - step : step;
            confirmation[index][round] = measure(finalists[index], options, device, 5.0, false);
            const auto& sample = confirmation[index][round];
            TNN_LOG_INFO("[PEARL-TUNE] Confirm %u %ux%ux%u %.3f TMAC/s\n", round + 1,
                         sample.shape.m, sample.shape.n, sample.shape.k, sample.rate / 1e12);
        }
    }
    auto median = [](auto samples) {
        std::sort(samples.begin(), samples.end(), [](auto a, auto b) { return a.rate < b.rate; });
        return samples[1];
    };
    auto winner = median(confirmation[0]);
    const double baseline = winner.rate;
    for (size_t i = 1; i < confirmation.size(); ++i) {
        const auto candidate = median(confirmation[i]);
        unsigned wins = 0;
        for (unsigned round = 0; round < 3; ++round)
            wins += confirmation[i][round].rate > confirmation[0][round].rate;
        if (candidate.rate >= baseline * 1.01 && candidate.rate > winner.rate && wins >= 2)
            winner = candidate;
    }
    check_cancelled();
    result_for(winner, options, out);
    return true;
}

} // namespace

void pearl_configure_tuning(AlgoConfig& config, ExecutionOptions options) {
    // Allocation must follow selection; the probe owns temporary explicit-shape
    // adapters and releases each before trying the next candidate.
    config.pre_tune_fn = nullptr;
    config.fixed_launch.reset();
    if (options.backend == Backend::PortableSimt) {
        config.custom_tune_source_fn = [options, active_recipe = options.recipe]
            (const TuningResult& result, AlgoConfig& worker) mutable {
            const auto selected = unsigned(result.tune_keys.at("pearl_recipe"));
            if (!tuning::selectable_recipe(options, selected))
                throw std::runtime_error("Invalid selected Pearl recipe");
            if (selected == active_recipe) return false;
            auto explicit_options = options;
            explicit_options.recipe = selected;
            explicit_options.mode = ExecutionMode::Benchmark;
            const auto specialized = pearl_gpu_config(explicit_options);
            worker.source_transform_fn = specialized.source_transform_fn;
            worker.compiler_opts_amd = specialized.compiler_opts_amd;
            active_recipe = selected;
            return true;
        };
    }
    config.custom_tune_fn = [options](const KernelMap&, const oroDeviceProp_t& props, int device,
                                     bool enabled, TuningResult& result) {
        if (!tnn_is_amd_device(device) || architecture_name(props.gcnArchName) != options.architecture)
            throw std::runtime_error("Pearl tuning backend/device mismatch");
        (void)tuning::architecture_code(options);
        return tune(options, device, enabled, result);
    };
    config.custom_tune_validate_fn = [options](const TuningResult& result,
                                               const oroDeviceProp_t& props, int device) {
        const auto shape = shape_from(result);
        auto identity = options;
        if (tuning::bringup_backend(options)) {
            const auto found = result.tune_keys.find("pearl_recipe");
            if (found == result.tune_keys.end() || found->second < 0 || found->second > 127 ||
                !tuning::selectable_recipe(options, unsigned(found->second))) return false;
            identity.recipe = unsigned(found->second);
        }
        return tnn_is_amd_device(device) && architecture_name(props.gcnArchName) == options.architecture &&
               result.valid && result.block_size == 128 && result.batch_size == options.batch_size &&
               tuning::valid_measurement(result.hashrate, result.batch_time_ms) &&
               tuning::supported(shape, tuning::domain_for(identity)) && tuning::matches_identity(result, identity) &&
               result.num_blocks == int((shape.m / 128) * (shape.n / pearl_tile_n(identity))) && fits(shape, options);
    };
    config.custom_tune_apply_fn = [options](const TuningResult& result,
                                           const oroDeviceProp_t& props, int device, void** data) mutable {
        if (*data)
            throw std::runtime_error("Pearl shape cannot change while a worker owns buffers");
        if (tuning::bringup_backend(options)) {
            options.recipe = unsigned(result.tune_keys.at("pearl_recipe"));
            pearl_qualify_recipe(device, options);
        }
        if (options.backend == Backend::PortableSimt) {
            TNN_LOG_INFO("[PEARL-TUNE] SIMT recipe=%u tile=128x%u K-step=%u LDS banks=%u load=%uB\n",
                         options.recipe, pearl_tile_n(options), (options.recipe & 4) ? 64u : 32u,
                         (options.recipe & 32) ? 2u : 1u,
                         (options.recipe & 16) ? 16u : (options.recipe & 8) ? 8u : 1u);
        }
        options.shape = shape_from(result);
        options.mode = ExecutionMode::Benchmark;
        const auto explicit_config = pearl_gpu_config(options);
        if (!explicit_config.pre_tune_fn({}, props, device, data))
            return false;
        if (result.hashrate > 0) {
            TNN_LOG_INFO("[PEARL-TUNE] Selected %ux%ux%u, batch=%u, %.3f TMAC/s, %.1f ms/batch\n",
                         options.shape.m, options.shape.n, options.shape.k, options.batch_size,
                         result.hashrate / 1e12, result.batch_time_ms);
        } else {
            TNN_LOG_INFO("[PEARL-TUNE] Default %ux%ux%u, batch=%u (not timed)\n",
                         options.shape.m, options.shape.n, options.shape.k, options.batch_size);
        }
        return true;
    };
}

} // namespace tnn::pearl
