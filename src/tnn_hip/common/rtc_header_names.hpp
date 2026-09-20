#pragma once

#include <string_view>

namespace tnn::gpu {

// Embedded manifests retain repository-relative paths. RTC sources may use
// those paths, intermediate include roots, or the historical basename.
template<class Register>
void for_each_rtc_header_name(std::string_view name, Register&& register_name) {
    while (!name.empty()) {
        register_name(name);
        const auto slash = name.find('/');
        if (slash == std::string_view::npos) break;
        name.remove_prefix(slash + 1);
    }
}

} // namespace tnn::gpu
