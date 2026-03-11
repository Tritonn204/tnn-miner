#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <boost/json.hpp>
#include "tnn_log.hpp"

// ============================================================================
// GPU Submit Queue
// ============================================================================
// Decouples GPU mining threads from network I/O. GPU callbacks push pre-built
// share payloads into the queue and return immediately so the GPU keeps hashing.
// A dedicated drain thread pops entries one at a time and feeds them into the
// existing single-slot submit mechanism (share/submitting/data_ready/cv).

struct GPUSubmitEntry {
    boost::json::object payload;
    bool is_dev;
    int64_t job_id;   // for stale check at drain time
};

class GPUSubmitQueue {
public:
    static GPUSubmitQueue& instance() {
        static GPUSubmitQueue inst;
        return inst;
    }

    // Start the drain thread. Call once after network globals are available.
    // stale_check: returns true if job_id is still current (not stale)
    void start(
        boost::json::object* share_ptr,
        boost::json::object* dev_share_ptr,
        bool* submitting_ptr,
        bool* submitting_dev_ptr,
        bool* data_ready_ptr,
        boost::condition_variable* cv_ptr,
        std::function<bool(int64_t job_id, bool is_dev)> stale_check)
    {
        stop();  // clean up any previous drain thread

        running_.store(true);
        share_ptr_ = share_ptr;
        dev_share_ptr_ = dev_share_ptr;
        submitting_ptr_ = submitting_ptr;
        submitting_dev_ptr_ = submitting_dev_ptr;
        data_ready_ptr_ = data_ready_ptr;
        cv_ptr_ = cv_ptr;
        stale_check_ = std::move(stale_check);

        drain_thread_ = std::thread([this]() { drain_loop(); });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        queue_cv_.notify_all();
        if (drain_thread_.joinable()) drain_thread_.join();
    }

    // Push a submit entry (non-blocking, called from GPU threads)
    void push(GPUSubmitEntry entry) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_.push(std::move(entry));
        }
        queue_cv_.notify_one();
    }

    size_t pending() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_.size();
    }

private:
    GPUSubmitQueue() = default;
    ~GPUSubmitQueue() { stop(); }

    void drain_loop() {
        while (running_.load()) {
            GPUSubmitEntry entry;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [&] {
                    return !queue_.empty() || !running_.load();
                });
                if (!running_.load() && queue_.empty()) break;
                if (queue_.empty()) continue;
                entry = std::move(queue_.front());
                queue_.pop();
            }

            // Stale check before waiting for submit slot
            if (stale_check_ && !stale_check_(entry.job_id, entry.is_dev)) {
                TNN_LOG_DEBUG("[DEBUG] Submit queue: dropping stale solution (job_id=%ld, dev=%d)\n",
                    (long)entry.job_id, entry.is_dev);
                continue;
            }

            // Wait for the single-slot submit mechanism to be free
            bool* flag = entry.is_dev ? submitting_dev_ptr_ : submitting_ptr_;
            int wait_ms = 0;
            while (*flag && running_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (++wait_ms > 5000) {
                    TNN_LOG_ERROR("[WARN] Submit queue: timeout waiting for slot, dropping solution\n");
                    break;
                }
                // Re-check staleness while waiting
                if (stale_check_ && !stale_check_(entry.job_id, entry.is_dev)) {
                    TNN_LOG_DEBUG("[DEBUG] Submit queue: solution went stale during wait\n");
                    wait_ms = -1; // sentinel
                    break;
                }
            }
            if (wait_ms >= 5000 || wait_ms == -1) continue;
            if (!running_.load()) break;

            // Write to the shared slot and signal
            *flag = true;
            if (entry.is_dev)
                *dev_share_ptr_ = std::move(entry.payload);
            else
                *share_ptr_ = std::move(entry.payload);

            *data_ready_ptr_ = true;
            cv_ptr_->notify_all();
        }
    }

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<GPUSubmitEntry> queue_;
    std::atomic<bool> running_{false};
    std::thread drain_thread_;

    // Pointers into the existing net globals (set once at start)
    boost::json::object* share_ptr_ = nullptr;
    boost::json::object* dev_share_ptr_ = nullptr;
    bool* submitting_ptr_ = nullptr;
    bool* submitting_dev_ptr_ = nullptr;
    bool* data_ready_ptr_ = nullptr;
    boost::condition_variable* cv_ptr_ = nullptr;
    std::function<bool(int64_t, bool)> stale_check_;
};
