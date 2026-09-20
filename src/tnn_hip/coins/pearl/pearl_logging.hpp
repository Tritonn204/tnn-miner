#pragma once

#include <tnn_log.hpp>
#include <cstdint>
#include <string>

namespace tnn::pearl {

inline void log_found(bool is_dev, int device, uint64_t attempt, unsigned row, unsigned col) {
    TNN_LOG_INFO_COLOR(is_dev ? CYAN : BRIGHT_YELLOW,
                       "\n%sGPU #%d found a solution: attempt %llu, row %u, col %u\n",
                       is_dev ? "DEV | " : "", device,
                       static_cast<unsigned long long>(attempt), row, col);
}

inline void log_share(bool is_dev, int, bool accepted, const std::string &reason) {
    TNN_LOG_INFO_COLOR(is_dev ? CYAN : (accepted ? BRIGHT_WHITE : RED),
                       "\n%sStratum: share %s%s%s\n", is_dev ? "DEV | " : "",
                       accepted ? "accepted" : "rejected", reason.empty() ? "" : ": ",
                       reason.c_str());
}

inline void log_stratum_error(const std::string &message) {
    // The reporter may have left an unterminated status line on stdout.
    fflush(stdout);
    TNN_LOG_ERROR("\nStratum: %s\n", message.c_str());
}

} // namespace tnn::pearl
