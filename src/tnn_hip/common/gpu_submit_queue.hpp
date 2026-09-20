#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
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

enum class SubmitDisposition { HandedOff, Stale, Timeout, Shutdown };

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
        std::condition_variable* cv_ptr,
        std::function<bool(int64_t job_id, bool is_dev)> stale_check,
        std::mutex* slot_mutex = nullptr,
        std::function<void(const GPUSubmitEntry&, SubmitDisposition)> disposition = {},
        bool retain_valid_on_timeout = false,
        std::chrono::milliseconds slot_wait_timeout = std::chrono::seconds(5))
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
        slot_mutex_ = slot_mutex;
        disposition_ = std::move(disposition);
        retain_valid_on_timeout_ = retain_valid_on_timeout;
        slot_wait_timeout_ = slot_wait_timeout;

        drain_thread_ = std::thread([this]() { drain_loop(); });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        queue_cv_.notify_all();
        if (drain_thread_.joinable()) drain_thread_.join();
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!queue_.empty()) {
            report(queue_.front(), SubmitDisposition::Shutdown);
            queue_.pop();
        }
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
    void report(const GPUSubmitEntry& entry, SubmitDisposition outcome) {
        if (disposition_) disposition_(entry, outcome);
    }
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
                report(entry, SubmitDisposition::Stale);
                continue;
            }

            // Wait for the single-slot submit mechanism to be free
            bool* flag = entry.is_dev ? submitting_dev_ptr_ : submitting_ptr_;
            bool discarded = false;
            auto deadline = std::chrono::steady_clock::now() + slot_wait_timeout_;
            auto slot_busy = [&] {
                std::unique_lock<std::mutex> lock;
                if (slot_mutex_) lock = std::unique_lock<std::mutex>(*slot_mutex_);
                return *flag;
            };
            while (slot_busy() && running_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (std::chrono::steady_clock::now() >= deadline) {
                    if (retain_valid_on_timeout_) {
                        deadline = std::chrono::steady_clock::now() + slot_wait_timeout_;
                        TNN_LOG_DEBUG("[DEBUG] Submit queue: retaining valid solution under backpressure\n");
                        continue;
                    }
                    TNN_LOG_ERROR("[WARN] Submit queue: timeout waiting for slot, dropping solution\n");
                    report(entry, SubmitDisposition::Timeout);
                    discarded = true;
                    break;
                }
                // Re-check staleness while waiting
                if (stale_check_ && !stale_check_(entry.job_id, entry.is_dev)) {
                    TNN_LOG_DEBUG("[DEBUG] Submit queue: solution went stale during wait\n");
                    report(entry, SubmitDisposition::Stale);
                    discarded = true;
                    break;
                }
            }
            if (discarded) continue;
            if (!running_.load()) {
                report(entry, SubmitDisposition::Shutdown);
                break;
            }
            if (stale_check_ && !stale_check_(entry.job_id, entry.is_dev)) {
                report(entry, SubmitDisposition::Stale);
                continue;
            }

            // Write to the shared slot and signal
            std::unique_lock<std::mutex> slot_lock;
            if (slot_mutex_) slot_lock = std::unique_lock<std::mutex>(*slot_mutex_);
            if (*flag) {
                // Another producer filled the slot after our unlocked check.
                // Keep this candidate; never silently discard it.
                std::lock_guard queue_lock(queue_mutex_);
                queue_.push(std::move(entry));
                continue;
            }
            report(entry, SubmitDisposition::HandedOff);
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
    std::condition_variable* cv_ptr_ = nullptr;
    std::function<bool(int64_t, bool)> stale_check_;
    std::mutex* slot_mutex_ = nullptr;
    std::function<void(const GPUSubmitEntry&, SubmitDisposition)> disposition_;
    bool retain_valid_on_timeout_ = false;
    std::chrono::milliseconds slot_wait_timeout_{5000};
};
