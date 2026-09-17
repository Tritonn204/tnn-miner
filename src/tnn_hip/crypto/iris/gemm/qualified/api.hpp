#pragma once

#if defined(__HIPCC_RTC__)
#include "hiprtc_types.hip.h"
#else
#include <cstdint>
#include <hip/hip_runtime.h>
#endif

namespace tnn::hip::iris::gemm::qualified {

#if defined(__HIPCC_RTC__)
using uintptr_t = __UINTPTR_TYPE__;
#endif

struct NativeWinner {
    uint32_t row;
    uint32_t col;
    uint32_t digest[8];
};

struct NativeState {
    uint32_t total_hits;
    uint32_t overflow;
};

struct NativeResults {
    NativeWinner *winners;
    NativeState *state;
    uint32_t capacity;
};

static_assert(sizeof(NativeWinner) == 40 && sizeof(NativeState) == 8);
static_assert(sizeof(NativeResults) == 24);

inline bool valid_inputs(const int8_t *a, const int8_t *b, int32_t *d, uint32_t m, uint32_t n,
                         uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldd, unsigned rank,
                         const uint32_t *key, const uint32_t *target, NativeResults results,
                         uint32_t *diagnostic) {
    auto aligned = [](const void *p, unsigned align) {
        return p && reinterpret_cast<uintptr_t>(p) % align == 0;
    };

    const bool shape_valid = rank == 128 && m >= 256 && n >= 256 && m <= 8192 && n <= 8192 &&
                             m % 128 == 0 && n % 256 == 0 && (k == 2048 || k == 4096);
    const bool strides_valid = lda >= m && ldb >= k && ldd >= m && lda <= m + 16 && ldb <= k + 16 &&
                               ldd <= m + 16 && lda % 16 == 0 && ldb % 16 == 0 && ldd % 16 == 0;
    const bool buffers_valid = aligned(a, 16) && aligned(b, 16) && aligned(d, 16) &&
                               aligned(key, 4) && aligned(target, 4) && aligned(results.state, 4);
    const bool results_valid = (!results.capacity || aligned(results.winners, 4)) &&
                               results.capacity <= size_t(m) * n / 128 &&
                               (!diagnostic || aligned(diagnostic, 4));

    return shape_valid && strides_valid && buffers_valid && results_valid;
}
enum class Mode { Raw, Fused, Diagnostic };

inline bool valid_mode(Mode mode, const uint32_t *diagnostic) {
    return mode == Mode::Raw || mode == Mode::Fused ||
           (mode == Mode::Diagnostic && diagnostic != nullptr);
}

// Full D required. Caller resets state; mode explicitly controls diagnostics.
// Scheduling is compile-time: there is no ignored runtime "streamed" argument.
#if !defined(__HIPCC_RTC__)
hipError_t launch(const int8_t *a, const int8_t *b, int32_t *d, uint32_t m, uint32_t n, uint32_t k,
                  uint32_t lda, uint32_t ldb, uint32_t ldd, unsigned rank, const uint32_t *key,
                  const uint32_t *target, NativeResults results, uint32_t *diagnostic,
                  hipStream_t stream, Mode mode);
#endif

} // namespace tnn::hip::iris::gemm::qualified
