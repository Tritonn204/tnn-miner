#pragma once

namespace tnn::hip::iris::gemm::collective {

struct DenseSerialMainloop {
    static constexpr bool kPipelined = false;
};

} // namespace tnn::hip::iris::gemm::collective
