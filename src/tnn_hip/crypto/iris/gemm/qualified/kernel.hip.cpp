// Derived from frozen gemm_next/tune.hip.cpp; local provenance is archived in study/gemm_next/core/provenance.json.
// Selected in-place checkpoint variant. Existing Iris defaults are not changed.
#include "api.hpp"
#include "ops.hpp"
#include "../../../blake3-inline.hip.inc"

namespace tnn::hip::iris::gemm::qualified {
namespace {

template <bool Diagnostic, bool Raw, bool NoD, class Configuration = Gfx1100Pearl>
__global__ __launch_bounds__(Configuration::Scheduling::threads) void tune_kernel(
    const int8_t *__restrict__ a, const int8_t *__restrict__ b, int32_t *__restrict__ d, unsigned m,
    unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldd,
    const uint32_t *__restrict__ key, const uint32_t *__restrict__ target, NativeResults results,
    uint32_t *diagnostic) {
#include "kernel_body.inc"
}

} // anonymous namespace

using Schedule = Gfx1100Pearl::Scheduling;

hipError_t launch(const int8_t *a, const int8_t *b, int32_t *d, uint32_t m, uint32_t n, uint32_t k,
                  uint32_t lda, uint32_t ldb, uint32_t ldd, unsigned rank, const uint32_t *key,
                  const uint32_t *target, NativeResults results, uint32_t *diagnostic,
                  hipStream_t stream, Mode mode) {
    if (!valid_mode(mode, diagnostic) ||
        !valid_inputs(a, b, d, m, n, k, lda, ldb, ldd, rank, key, target, results, diagnostic)) {
        return hipErrorInvalidValue;
    }

    dim3 grid(size_t(m / Schedule::tile_m) * (n / Schedule::tile_n));
#define RUN(D, R, N)                                                                               \
    hipLaunchKernelGGL((tune_kernel<D, R, N>), grid, dim3(Schedule::threads), 0, stream, a, b, d,  \
                       m, n, k, lda, ldb, ldd, key, target, results, diagnostic)
    switch (mode) {
    case Mode::Raw:
        RUN(false, true, false);
        break;
    case Mode::Fused:
        RUN(false, false, false);
        break;
    case Mode::Diagnostic:
        RUN(true, false, false);
        break;
    }
#undef RUN
    return hipGetLastError();
}
} // namespace tnn::hip::iris::gemm::qualified
