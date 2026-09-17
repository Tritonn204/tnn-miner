#pragma once

#include "qualified/api.hpp"
#include <string_view>

namespace tnn::hip::iris::gemm {

using qualified::Mode;
using qualified::NativeResults;

enum class Backend { Automatic, QualifiedGfx1100, PortableSimt };

inline Backend select_backend(std::string_view architecture, Backend requested) {
    // HIP architecture strings can carry feature suffixes, e.g. gfx90a:xnack-.
    const auto name = architecture.substr(0, architecture.find(':'));
    if (requested != Backend::Automatic) {
        return requested;
    }
    return name == "gfx1100" ? Backend::QualifiedGfx1100 : Backend::PortableSimt;
}

struct Problem {
    const int8_t *a;
    const int8_t *b;
    int32_t *d;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldd;
    unsigned rank;
    const uint32_t *key;
    const uint32_t *target;
    NativeResults results;
    uint32_t *diagnostic;
};

inline bool valid_problem(const Problem &p, Mode mode) {
    return qualified::valid_mode(mode, p.diagnostic) &&
           qualified::valid_inputs(p.a, p.b, p.d, p.m, p.n, p.k, p.lda, p.ldb, p.ldd, p.rank, p.key,
                                   p.target, p.results, p.diagnostic);
}

// Initializes no result state. A caller resets state once per complete problem,
// then can dispatch disjoint full-K tile ranges without losing winner counts.
hipError_t launch_portable_range(const Problem &problem, Mode mode, hipStream_t stream,
                                 unsigned first_tile, unsigned tile_count);

// The portable path enqueues bounded ranges in one stream. Call range directly
// when per-range timing/cancellation is required. No state reset between ranges.
hipError_t launch(const Problem &problem, Mode mode, hipStream_t stream,
                  Backend requested = Backend::Automatic);

} // namespace tnn::hip::iris::gemm
