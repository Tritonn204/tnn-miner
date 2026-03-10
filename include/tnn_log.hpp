#pragma once
#include <cstdio>
#include <cstdint>
#include <atomic>

// ============================================================================
// TNN Log Levels
// ============================================================================
// OFF   = 0  - Only errors (fprintf to stderr, always on)
// INFO  = 1  - Normal operational messages (default)
// DEBUG = 2  - Moderate verbosity (batch details, solution info)
// TRACE = 3  - Maximum verbosity (every function entry/exit)

enum class TnnLogLevel : uint8_t {
    Off   = 0,
    Info  = 1,
    Debug = 2,
    Trace = 3,
};

inline std::atomic<uint8_t>& tnn_log_level_ref() {
    static std::atomic<uint8_t> level{(uint8_t)TnnLogLevel::Info};
    return level;
}

inline void tnn_set_log_level(TnnLogLevel lvl) {
    tnn_log_level_ref().store((uint8_t)lvl, std::memory_order_relaxed);
}

inline TnnLogLevel tnn_get_log_level() {
    return (TnnLogLevel)tnn_log_level_ref().load(std::memory_order_relaxed);
}

inline bool tnn_log_enabled(TnnLogLevel lvl) {
    return (uint8_t)lvl <= tnn_log_level_ref().load(std::memory_order_relaxed);
}

// ============================================================================
// Log Macros
// ============================================================================
// Use these instead of raw printf/fprintf for gated output.
// TNN_LOG_ERROR always prints (to stderr).
// TNN_LOG_INFO / TNN_LOG_DEBUG / TNN_LOG_TRACE are gated by runtime level.

#define TNN_LOG_ERROR(fmt, ...) \
    fprintf(stderr, fmt, ##__VA_ARGS__)

#define TNN_LOG_INFO(fmt, ...) \
    do { if (tnn_log_enabled(TnnLogLevel::Info)) printf(fmt, ##__VA_ARGS__); } while(0)

#define TNN_LOG_DEBUG(fmt, ...) \
    do { if (tnn_log_enabled(TnnLogLevel::Debug)) printf(fmt, ##__VA_ARGS__); } while(0)

#define TNN_LOG_TRACE(fmt, ...) \
    do { if (tnn_log_enabled(TnnLogLevel::Trace)) printf(fmt, ##__VA_ARGS__); } while(0)
