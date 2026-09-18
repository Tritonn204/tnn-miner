#include <tnn_hip/coins/pearl/pearl_tuning.hpp>
#include <tnn_hip/common/gpu_tune_guard.hpp>
#include <cassert>
#include <future>
#include <limits>
#include <set>
#include <tuple>

using namespace tnn::pearl;

int main() {
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
