#pragma once

#include "pearl_mining.hpp"
#include <tnn_hip/crypto/iris/gemm/shape_domain.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <vector>

namespace tnn::pearl::tuning {

inline constexpr int64_t version = 3;
inline constexpr unsigned candidate_budget = 32;
inline constexpr unsigned grid = 1024;
inline constexpr native::Shape default_shape{8192, 8192, 4096,
                                             native::CandidateLayout::native_4x32};
using Domain = tnn::hip::iris::gemm::ShapeDomain;
inline constexpr Domain backend = tnn::hip::iris::gemm::native128_domain;

inline bool supported(native::Shape s, Domain domain = backend) {
    return domain.contains(s.m, s.n, s.k) &&
           s.m >= grid && s.n >= grid && s.m % grid == 0 && s.n % grid == 0 &&
           s.m <= native::maximum_dimension && s.n <= native::maximum_dimension &&
           native::qualified_depth(s.k) && s.layout == native::CandidateLayout::native_4x32;
}

inline bool same(native::Shape a, native::Shape b) {
    return a.m == b.m && a.n == b.n && a.k == b.k && a.layout == b.layout;
}

inline void append(std::vector<native::Shape>& out, native::Shape shape, Domain domain = backend) {
    shape.layout = native::CandidateLayout::native_4x32;
    if (supported(shape, domain) && std::none_of(out.begin(), out.end(),
                                      [&](auto other) { return same(shape, other); }))
        out.push_back(shape);
}

inline std::vector<unsigned> axis(unsigned minimum, unsigned maximum, unsigned alignment) {
    std::vector<unsigned> values;
    const unsigned step = std::lcm(grid, alignment);
    maximum = std::min(maximum, native::maximum_dimension) / step * step;
    for (unsigned quarter = 1; quarter <= 4; ++quarter) {
        const unsigned value = unsigned(uint64_t(maximum) * quarter / 4) / step * step;
        if (value >= std::max(grid, minimum) &&
            std::find(values.begin(), values.end(), value) == values.end())
            values.push_back(value);
    }
    return values;
}

inline std::vector<native::Shape> coarse(Domain domain = backend) {
    std::vector<native::Shape> result;
    append(result, default_shape, domain);
    const auto rows = axis(domain.minimum_m, domain.maximum_m, domain.alignment_m);
    const auto columns = axis(domain.minimum_n, domain.maximum_n, domain.alignment_n);

    // Sample both diagonals of the coarse grid: square and rectangular shapes
    // across the full range, at every qualified K. On gfx1100 this reserves
    // eight of the 32 candidates for local refinement instead of truncating
    // a Cartesian sweep before it reaches the largest dimensions.
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < columns.size(); ++j) {
            if (j != i && j != columns.size() - 1 - i)
                continue;
            for (unsigned k : native::qualified_depths)
                append(result, {rows[i], columns[j], k}, domain);
        }
    return result;
}

inline std::vector<native::Shape> refine(const std::vector<native::Shape>& leaders,
                                        Domain domain = backend) {
    std::vector<native::Shape> result;
    for (unsigned distance : {2u, 1u})
        for (bool change_n : {false, true})
            for (int sign : {-1, 1})
                for (auto shape : leaders) {
                    const unsigned step = std::lcm(grid, change_n ? domain.alignment_n : domain.alignment_m);
                    const int64_t value = int64_t(change_n ? shape.n : shape.m) + sign * int64_t(distance * step);
                    const unsigned minimum = std::max(grid, change_n ? domain.minimum_n : domain.minimum_m);
                    const unsigned maximum = std::min(native::maximum_dimension,
                        change_n ? domain.maximum_n : domain.maximum_m);
                    if (value < minimum || value > maximum)
                        continue;
                    (change_n ? shape.n : shape.m) = unsigned(value);
                    append(result, shape, domain);
                }
    return result;
}

inline bool valid_measurement(double rate, double batch_ms) {
    // Do not rely on isfinite under the miner's -ffast-math build flags.
    auto finite = [](double value) {
        return (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ull) != 0x7ff0000000000000ull;
    };
    return finite(rate) && finite(batch_ms) && rate >= 0 && batch_ms >= 0 &&
           ((rate == 0) == (batch_ms == 0)); // Both zero only for an unmeasured default.
}

inline uint64_t required_bytes(native::Shape shape, unsigned batch, unsigned capacity) {
    return native::allocation_budget(shape, batch, capacity);
}

inline bool fits_memory(native::Shape shape, unsigned batch, unsigned capacity,
                         uint64_t free, uint64_t total) {
    const uint64_t reserve = std::max<uint64_t>(512ull << 20, total / 10);
    return free > reserve && required_bytes(shape, batch, capacity) <= free - reserve;
}

} // namespace tnn::pearl::tuning
