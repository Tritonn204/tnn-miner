#pragma once

#include <mutex>
#include <optional>
#include <utility>

// Host-only: publish/copy one complete job, with separate user and dev slots.
// Snapshot owns its data and provides an is_dev field. No GPU API dependency.
template<class Snapshot>
class GPUJobMailbox {
public:
    void publish(Snapshot snapshot) {
        std::lock_guard<std::mutex> lock(mutex_);
        (snapshot.is_dev ? dev_ : user_) = std::move(snapshot);
    }

    std::optional<Snapshot> read(bool is_dev) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_dev ? dev_ : user_;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        user_.reset();
        dev_.reset();
    }

private:
    mutable std::mutex mutex_;
    std::optional<Snapshot> user_;
    std::optional<Snapshot> dev_;
};
