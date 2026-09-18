#pragma once

#include <cstdint>

namespace tnn::hip::iris::gemm {

// Computational limits only. A consumer must intersect these with its own
// semantic/proof requirements before exposing a shape to an autotuner.
struct ShapeDomain {
    uint32_t minimum_m, maximum_m, alignment_m;
    uint32_t minimum_n, maximum_n, alignment_n;
    uint32_t minimum_k, maximum_k, alignment_k;

    constexpr bool contains(uint32_t m, uint32_t n, uint32_t k) const {
        return m >= minimum_m && m <= maximum_m && m % alignment_m == 0 &&
               n >= minimum_n && n <= maximum_n && n % alignment_n == 0 &&
               k >= minimum_k && k <= maximum_k && k % alignment_k == 0;
    }
};

// The packed buffer descriptors address less than 4 GiB per operand.
inline constexpr ShapeDomain native128_domain{128, 16384, 128, 128, 16384, 128,
                                               32, 8192, 32};

} // namespace tnn::hip::iris::gemm
