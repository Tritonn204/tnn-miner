#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tnn::pearl {

// Upstream zk-pow/api/sanity_checks.rs: penalized_adjustment_factor.
// This is work per jackpot, not a Stratum difficulty conversion.
inline uint64_t jackpot_work(uint32_t rows, uint32_t columns, uint32_t k, uint32_t rank) {
    if (!rows || !columns || rank < 128 || rank > 1024 || k / rank < 16) {
        throw std::invalid_argument("Invalid Pearl jackpot configuration");
    }

    uint64_t factor = rows;
    for (uint64_t value : {uint64_t(columns), uint64_t(k / rank), uint64_t(128)}) {
        if (factor > std::numeric_limits<uint64_t>::max() / value) {
            throw std::overflow_error("Pearl jackpot work overflow");
        }
        factor *= value;
    }
    return factor;
}

// Checked little-endian U256 multiplication, matching the miner-facing
// penalized_target_bound (not consensus saturation). No floating point.
inline std::array<uint8_t, 32> scale_jackpot_target(
    const std::array<uint8_t, 32>& wire_target, uint64_t factor) {
    if (!factor || wire_target == std::array<uint8_t, 32>{}) {
        throw std::invalid_argument("Zero Pearl target or jackpot work");
    }

    std::array<uint8_t, 40> product{};
    for (unsigned j = 0; j < 8; ++j) {
        unsigned carry = 0;
        const unsigned digit = unsigned((factor >> (8 * j)) & 255);
        for (unsigned i = 0; i < 32; ++i) {
            const unsigned value = product[i + j] + wire_target[i] * digit + carry;
            product[i + j] = uint8_t(value);
            carry = value >> 8;
        }
        product[32 + j] = uint8_t(carry);
    }
    for (unsigned i = 32; i < product.size(); ++i) {
        if (product[i]) throw std::overflow_error("Pearl jackpot target overflow; increase pool difficulty");
    }

    std::array<uint8_t, 32> result{};
    for (unsigned i = 0; i < result.size(); ++i) result[i] = product[i];
    return result;
}

} // namespace tnn::pearl
