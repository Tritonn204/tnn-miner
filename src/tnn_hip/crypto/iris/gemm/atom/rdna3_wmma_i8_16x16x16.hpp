#pragma once

#include "../iris_include.hpp"

namespace tnn::hip::iris::gemm::atom {

template <bool ARowMajor = true, bool BColMajor = true, bool Saturate = false>
using Rdna3WmmaI8_16x16x16 =
    ::iris::hip::WmmaI8I8I32_16x16x16<ARowMajor, BColMajor, Saturate>;

} // namespace tnn::hip::iris::gemm::atom
