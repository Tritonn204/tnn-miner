#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>

enum class RateUnit {
    Hashes,
    Solutions,
    MultiplyAccumulates,
};

struct RateInfo {
    RateUnit unit = RateUnit::Hashes;
};

constexpr const char* rate_suffix(RateUnit unit) {
    switch (unit) {
    case RateUnit::Solutions: return "Sol/s";
    case RateUnit::MultiplyAccumulates: return "MAC/s";
    default: return "H/s";
    }
}

constexpr const char* efficiency_suffix(RateUnit unit) {
    switch (unit) {
    case RateUnit::Solutions: return "Sol/J";
    case RateUnit::MultiplyAccumulates: return "MAC/J";
    default: return "H/J";
    }
}

inline uint64_t completed_work(uint32_t count, uint64_t multiplier = 1) {
    if (count != 0 && multiplier == 0) {
        throw std::invalid_argument("Completed work multiplier must be positive");
    }
    if (count != 0 && multiplier > std::numeric_limits<uint64_t>::max() / count) {
        throw std::overflow_error("Completed work multiplication overflow");
    }
    return uint64_t(count) * multiplier;
}

inline void add_completed_work(std::atomic<uint64_t>& counter, uint64_t work) {
    auto previous = counter.load(std::memory_order_relaxed);
    do {
        if (work > std::numeric_limits<uint64_t>::max() - previous) {
            throw std::overflow_error("Completed work counter overflow");
        }
    } while (!counter.compare_exchange_weak(previous, previous + work,
                                            std::memory_order_relaxed));
}
