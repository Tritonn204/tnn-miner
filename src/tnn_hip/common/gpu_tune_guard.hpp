#pragma once

#include <atomic>
#include <chrono>
#include <mutex>

// A waiting worker must observe shutdown too, not only the measuring worker.
inline bool acquire_gpu_tune_lock(std::unique_lock<std::timed_mutex>& lock,
                                   const std::atomic<bool>& cancelled) {
    while (!cancelled.load(std::memory_order_relaxed)) {
        if (lock.try_lock_for(std::chrono::milliseconds(100)))
            return !cancelled.load(std::memory_order_relaxed);
    }
    return false;
}
