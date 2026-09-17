#pragma once

#include <tnn_log.hpp>
#include <string>

namespace tnn::pearl {

inline void log_share(bool is_dev, int device, bool accepted, const std::string &reason) {
    TNN_LOG_INFO_COLOR(is_dev ? CYAN : (accepted ? BRIGHT_WHITE : RED),
                       "\n%s[PEARL-STRATUM] GPU %d share %s%s%s\n", is_dev ? "DEV | " : "", device,
                       accepted ? "accepted" : "rejected", reason.empty() ? "" : ": ",
                       reason.c_str());
}

inline void log_stratum_error(const std::string &message) {
    // The reporter may have left an unterminated status line on stdout.
    fflush(stdout);
    TNN_LOG_ERROR("\n[PEARL-STRATUM] %s\n", message.c_str());
}

} // namespace tnn::pearl
