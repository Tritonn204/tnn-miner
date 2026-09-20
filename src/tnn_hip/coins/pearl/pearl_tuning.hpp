#pragma once

#include "pearl_mining.hpp"
#include "pearl_arch.hpp"
#include <tnn_hip/crypto/iris/gemm/shape_domain.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

namespace tnn::pearl::tuning {

inline constexpr int64_t version = 3;
inline constexpr int64_t multiarch_version = 4;
inline constexpr int64_t portable_version = 5;
inline constexpr unsigned candidate_budget = 32;
inline constexpr unsigned grid = 1024;
inline constexpr native::Shape default_shape{8192, 8192, 4096,
                                             native::CandidateLayout::native_4x32};
using Domain = tnn::hip::iris::gemm::ShapeDomain;
inline constexpr Domain backend = tnn::hip::iris::gemm::native128_domain;

inline Domain domain_for(const ExecutionOptions& options) {
    using namespace tnn::hip::iris::gemm;
    if (options.backend == Backend::Cdna) return cdna128_domain;
    if (options.backend == Backend::PortableSimt) {
        auto domain = pearl_tile_n(options) == 64 ? simt64_domain : simt128_domain;
        domain.minimum_k = domain.alignment_k = (options.recipe & 4) ? 64 : 32;
        return domain;
    }
    return native128_domain;
}

inline bool bringup_backend(const ExecutionOptions& options) {
    return options.backend == Backend::PortableSimt || options.backend == Backend::Cdna;
}

inline int engine_code(const ExecutionOptions& options) {
    switch (options.backend) {
    case Backend::Rdna3: return 3;
    case Backend::Rdna4: return 4;
    case Backend::PortableSimt: return 1;
    case Backend::Cdna: return 2;
    }
    throw std::invalid_argument("Unknown Pearl engine");
}

inline int64_t cache_version(const ExecutionOptions& options) {
    return bringup_backend(options) ? portable_version :
           options.architecture == "gfx1100" ? version : multiarch_version;
}

inline std::vector<unsigned> recipes(const ExecutionOptions& options) {
    if (options.backend != Backend::PortableSimt) return {options.recipe};
    const unsigned arithmetic = options.recipe & 2;
    std::vector<unsigned> result{arithmetic, arithmetic | 8, arithmetic | 4 | 16};
    // HIP 6.4 full-body audits find scratch in the wide, double-bank scalar
    // recipe on RDNA1. Keep it available to studies, not automatic mining.
    const auto arch = architecture_name(options.architecture);
    if (arch != "gfx1010" && arch != "gfx1011" && arch != "gfx1012")
        result.push_back(arithmetic | 4 | 16 | 32);
    result.push_back(arithmetic | 4 | 16 | 64);
    result.push_back(arithmetic | 4 | 16 | 32 | 64);
    return result;
}

inline bool selectable_recipe(const ExecutionOptions& options, unsigned recipe) {
    const auto allowed = recipes(options);
    return std::find(allowed.begin(), allowed.end(), recipe) != allowed.end();
}

inline int architecture_code(const ExecutionOptions& options) {
    const auto name = architecture_name(options.architecture);
    if (name == "gfx90a" && options.backend == Backend::Cdna) return 0x90a;
    if (name.size() < 4 || name.size() > 7 || name.substr(0, 3) != "gfx")
        throw std::invalid_argument("Invalid Pearl tuning architecture");
    int code = 0;
    for (char digit : name.substr(3)) {
        if (digit < '0' || digit > '9')
            throw std::invalid_argument("Unknown Pearl tuning architecture");
        code = code * 10 + digit - '0';
    }
    const bool supported = options.backend == Backend::PortableSimt ? portable_target(code) :
        options.backend == Backend::Cdna ? cdna_target(name) :
        options.backend == Backend::Rdna3 ? rdna3_target(name) : rdna4_target(name);
    if (!supported) throw std::invalid_argument("Pearl tuning architecture/engine mismatch");
    return code;
}

template<class Fits>
native::Shape baseline(const ExecutionOptions& options, Fits fits) {
    if (bringup_backend(options)) {
        for (native::Shape shape : {native::Shape{2048, 2048, 2048}, native::Shape{1024, 1024, 2048}}) {
            shape.layout = native::CandidateLayout::native_4x32;
            if (fits(shape)) return shape;
        }
        throw std::runtime_error("Pearl baseline cannot fit with the required memory reserve");
    }
    if (fits(default_shape)) return default_shape;
    if (options.architecture != "gfx1100")
        for (native::Shape shape : {native::Shape{4096, 4096, 4096}, native::Shape{2048, 2048, 2048}}) {
            shape.layout = native::CandidateLayout::native_4x32;
            if (fits(shape)) return shape;
        }
    throw std::runtime_error("Pearl default batch cannot fit with the required memory reserve");
}

template<class Result>
bool matches_identity(const Result& result, const ExecutionOptions& options) {
    auto matches = [&](const char* key, int64_t value) {
        auto it = result.tune_keys.find(key);
        return it != result.tune_keys.end() && it->second == value;
    };
    const bool retained = options.architecture == "gfx1100" && options.backend == Backend::Rdna3;
    return matches("pearl_backend", architecture_code(options)) &&
           matches("pearl_version", cache_version(options)) &&
           (retained || (matches("pearl_recipe", options.recipe) &&
                         matches("pearl_engine", engine_code(options))));
}

inline bool supported(native::Shape s, Domain domain = backend) {
    return domain.contains(s.m, s.n, s.k) &&
           s.m >= grid && s.n >= grid && s.m % grid == 0 && s.n % grid == 0 &&
           s.m <= native::maximum_dimension && s.n <= native::maximum_dimension &&
           native::qualified_depth(s.k) && s.layout == native::CandidateLayout::native_4x32;
}

inline bool same(native::Shape a, native::Shape b) {
    return a.m == b.m && a.n == b.n && a.k == b.k && a.layout == b.layout;
}

// This is a conservative extrapolation, not a device watchdog guarantee.
// Include operand traffic as well as arithmetic: thin shapes can be memory-bound.
inline double projected_batch_ms(native::Shape previous, native::Shape next, double ms) {
    const double work = double(next.macs()) / previous.macs();
    const double bytes = (double(next.m) + next.n) * next.k /
                         ((double(previous.m) + previous.n) * previous.k);
    return ms * std::max(work, bytes);
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

inline std::vector<native::Shape> coarse(const ExecutionOptions& options) {
    if (!bringup_backend(options)) return coarse();
    std::vector<native::Shape> result;
    // Twenty-four slots cover small, square and strongly rectangular cases at
    // every supported K. Eight remain for local 1024-grid refinement.
    for (auto axes : {std::pair{2048u, 2048u}, {4096u, 4096u},
                      {2048u, 8192u}, {8192u, 2048u}, {8192u, 8192u},
                      {4096u, 16384u}, {16384u, 4096u}, {16384u, 16384u}})
        for (unsigned k : native::qualified_depths)
            append(result, {axes.first, axes.second, k}, domain_for(options));
    std::stable_sort(result.begin(), result.end(), [](auto a, auto b) {
        return a.macs() < b.macs();
    });
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
