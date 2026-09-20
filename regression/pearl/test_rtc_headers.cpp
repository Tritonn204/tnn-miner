#include <tnn_hip/common/rtc_header_names.hpp>

#include <cassert>
#include <string>
#include <vector>

int main() {
    auto names = [](std::string_view path) {
        std::vector<std::string> result;
        tnn::gpu::for_each_rtc_header_name(path, [&](std::string_view name) {
            result.emplace_back(name);
        });
        return result;
    };

    assert((names("src/tnn_hip/crypto/blake3-inline.hip.inc") == std::vector<std::string>{
        "src/tnn_hip/crypto/blake3-inline.hip.inc",
        "tnn_hip/crypto/blake3-inline.hip.inc",
        "crypto/blake3-inline.hip.inc", "blake3-inline.hip.inc"}));
    assert((names("internal/accessors.hpp") == std::vector<std::string>{
        "internal/accessors.hpp", "accessors.hpp"}));
    assert((names("hiprtc_types.hip.h") == std::vector<std::string>{"hiprtc_types.hip.h"}));
    assert(names("").empty());
}
