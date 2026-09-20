#include <tnn_hip/coins/pearl/pearl_tuning.hpp>
#include <tnn_hip/common/gpu_tune_guard.hpp>
#include <cassert>
#include <future>
#include <limits>
#include <set>
#include <tuple>
#include <map>

using namespace tnn::pearl;

int main() {
    struct Result { std::map<std::string, int64_t> tune_keys; };
    for (const char* arch : {"gfx900", "gfx906", "gfx1010", "gfx1011", "gfx1012",
                            "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034",
                            "gfx908", "gfx90a", "gfx942"}) {
        ExecutionOptions options;
        options.architecture = arch;
        options.backend = cdna_target(arch) ? Backend::Cdna : Backend::PortableSimt;
        Result result{{{"pearl_version", tuning::portable_version},
                       {"pearl_backend", tuning::architecture_code(options)},
                       {"pearl_engine", tuning::engine_code(options)}, {"pearl_recipe", 0}}};
        assert(tuning::matches_identity(result, options));
        ++result.tune_keys["pearl_engine"];
        assert(!tuning::matches_identity(result, options));
        const auto baseline = tuning::baseline(options, [](auto) { return true; });
        assert(baseline.m == 2048 && baseline.n == 2048 && baseline.k == 2048);
        assert(tuning::supported({6144, 3072, 4096, native::CandidateLayout::native_4x32}));
        assert(tuning::supported(baseline));
        const auto candidates = tuning::coarse(options);
        assert(candidates.size() == 24);
        for (unsigned k : native::qualified_depths) {
            assert(std::any_of(candidates.begin(), candidates.end(), [k](auto s) {
                return s.m == 16384 && s.n == 16384 && s.k == k;
            }));
        }
        assert(tuning::refine({baseline}).size() > 0);
        if (options.backend == Backend::PortableSimt) {
            const bool rdna1 = std::string_view(arch).starts_with("gfx101");
            assert(tuning::recipes(options).size() == (rdna1 ? 5 : 6));
            assert(tuning::selectable_recipe(options, 116));
            assert(!tuning::selectable_recipe(options, 117));
            options.recipe = 116;
            assert(pearl_tile_n(options) == 64);
        }
        assert(tuning::projected_batch_ms(baseline, {4096, 4096, 2048}, 10) == 40);
    }
    for (const char* arch : {"gfx1101", "gfx1102", "gfx1200", "gfx1201"}) {
        ExecutionOptions options;
        options.architecture = arch;
        options.backend = rdna3_target(arch) ? Backend::Rdna3 : Backend::Rdna4;
        Result result{{{"pearl_version", tuning::multiarch_version},
                       {"pearl_backend", tuning::architecture_code(options)},
                       {"pearl_engine", options.backend == Backend::Rdna3 ? 3 : 4}, {"pearl_recipe", 0}}};
        assert(tuning::matches_identity(result, options));
        for (auto key : {"pearl_version", "pearl_backend", "pearl_engine", "pearl_recipe"}) {
            auto bad = result;
            ++bad.tune_keys[key];
            assert(!tuning::matches_identity(bad, options));
            bad.tune_keys.erase(key);
            assert(!tuning::matches_identity(bad, options));
        }
        assert(tuning::baseline(options, [](auto s) { return s.m <= 4096; }).m == 4096);
        assert(tuning::baseline(options, [](auto s) { return s.m <= 2048; }).k == 2048);
        bool failed = false;
        try { (void)tuning::baseline(options, [](auto) { return false; }); }
        catch (const std::runtime_error&) { failed = true; }
        assert(failed);
    }
    ExecutionOptions retained;
    Result old{{{"pearl_version", 3}, {"pearl_backend", 1100}}};
    assert(tuning::matches_identity(old, retained));
    const auto initial = tuning::coarse();
    assert(initial.size() == 24 && tuning::same(initial.front(), tuning::default_shape));
    std::vector<native::Shape> combined = initial;
    for (auto leader : initial) {
        for (auto shape : tuning::refine({leader})) {
            if (combined.size() < tuning::candidate_budget)
                tuning::append(combined, shape);
        }
    }
    assert(combined.size() == 32);
    std::set<std::tuple<unsigned, unsigned, unsigned>> unique;
    for (auto shape : combined) {
        assert(tuning::supported(shape));
        assert(unique.emplace(shape.m, shape.n, shape.k).second);
    }
    auto smaller = tuning::backend;
    smaller.maximum_m = smaller.maximum_n = 8192;
    smaller.maximum_k = 4096;
    const auto small = tuning::coarse(smaller);
    assert(small.size() == 16);
    for (auto shape : small) assert(tuning::supported(shape, smaller));
    const auto refined = tuning::refine({tuning::default_shape});
    assert(std::any_of(refined.begin(), refined.end(), [](auto s) { return s.m == 6144; }));
    assert(!tuning::supported({8192, 8192, 16384, native::CandidateLayout::native_4x32}));

    assert(!tuning::fits_memory(tuning::default_shape, 32, 4096, 0, 24ull << 30));
    assert(tuning::fits_memory(tuning::default_shape, 32, 4096, 20ull << 30, 24ull << 30));
    assert(tuning::required_bytes({16384, 16384, 8192}, 32, 4096) > (4ull << 30));
    bool invalid = false;
    try { (void)tuning::required_bytes(tuning::default_shape, 33, 1); }
    catch (const std::invalid_argument&) { invalid = true; }
    assert(invalid);
    assert(tuning::valid_measurement(0, 0));
    assert(tuning::valid_measurement(52e12, 170));
    assert(!tuning::valid_measurement(0, 170));
    assert(!tuning::valid_measurement(-1, 170));
    assert(!tuning::valid_measurement(std::numeric_limits<double>::infinity(), 170));
    assert(!tuning::valid_measurement(std::numeric_limits<double>::quiet_NaN(), 170));

    std::timed_mutex mutex;
    std::atomic<bool> cancelled{false};
    std::unique_lock held(mutex);
    auto waiter = std::async(std::launch::async, [&] {
        std::unique_lock<std::timed_mutex> lock(mutex, std::defer_lock);
        return acquire_gpu_tune_lock(lock, cancelled);
    });
    cancelled = true;
    assert(waiter.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    assert(!waiter.get());
    held.unlock();
    cancelled = false;
    std::unique_lock<std::timed_mutex> lock(mutex, std::defer_lock);
    assert(acquire_gpu_tune_lock(lock, cancelled));
}
