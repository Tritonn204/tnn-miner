#include <boost/json/src.hpp>
#include <array>
#include <cassert>
#include <chrono>
#include <thread>
#include "tnn_hip/common/gpu_submit_queue.hpp"

using namespace std::chrono_literals;

int main()
{
    tnn_set_log_level(TnnLogLevel::Off);
    auto& queue = GPUSubmitQueue::instance();
    boost::json::object user, dev;
    bool busy = true, dev_busy = false, ready = false;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> current{true};
    std::array<std::atomic<unsigned>, 4> counts{};

    auto start = [&](bool retain) {
        queue.start(&user, &dev, &busy, &dev_busy, &ready, &cv,
            [&](int64_t, bool) { return current.load(); }, &mutex,
            [&](const GPUSubmitEntry&, SubmitDisposition outcome) {
                ++counts[static_cast<unsigned>(outcome)];
            }, retain, 20ms);
    };
    auto wait_for = [&](SubmitDisposition outcome, unsigned expected) {
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (counts[static_cast<unsigned>(outcome)] != expected) {
            assert(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(1ms);
        }
    };

    start(true);
    queue.push({{{"candidate", 1}}, false, 1});
    std::this_thread::sleep_for(100ms);
    for (const auto& count : counts) assert(count == 0);
    {
        std::lock_guard lock(mutex);
        busy = false;
    }
    wait_for(SubmitDisposition::HandedOff, 1);
    {
        std::lock_guard lock(mutex);
        assert(user.at("candidate").as_int64() == 1 && ready);
    }

    queue.push({{}, false, 1});
    std::this_thread::sleep_for(30ms);
    current = false;
    wait_for(SubmitDisposition::Stale, 1);
    current = true;
    queue.push({{}, false, 1});
    queue.push({{}, false, 1});
    queue.stop();
    assert(counts[static_cast<unsigned>(SubmitDisposition::Shutdown)] == 2);

    // Existing callers retain the opt-out timeout behavior.
    start(false);
    queue.push({{}, false, 1});
    wait_for(SubmitDisposition::Timeout, 1);
    queue.stop();
}
