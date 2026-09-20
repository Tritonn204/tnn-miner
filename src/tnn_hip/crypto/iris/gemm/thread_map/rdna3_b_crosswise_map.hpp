#pragma once

namespace tnn::hip::iris::gemm::thread_map {

struct Rdna3BCrosswiseMap {
    static constexpr int kGroups = 2;
};

} // namespace tnn::hip::iris::gemm::thread_map
