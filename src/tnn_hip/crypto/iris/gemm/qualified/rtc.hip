// Stable RTC entry points over the same body used by the retained AOT launcher.
#include "api.hpp"
#include "ops.hpp"
#include "blake3-inline.hip.inc"

namespace tnn::hip::iris::gemm::qualified {

extern "C" __global__ __launch_bounds__(Gfx1100Pearl::Scheduling::threads)
void pearl_iris_raw(
    const int8_t *__restrict__ a, const int8_t *__restrict__ b, int32_t *__restrict__ d, unsigned m,
    unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldd,
    const uint32_t *__restrict__ key, const uint32_t *__restrict__ target, NativeResults results,
    uint32_t *diagnostic) {
    using Configuration = Gfx1100Pearl;
    constexpr bool Raw = true;
    constexpr bool Diagnostic = false;
    constexpr bool NoD = false;
#include "kernel_body.inc"
}

extern "C" __global__ __launch_bounds__(Gfx1100Pearl::Scheduling::threads)
void pearl_iris_fused(
    const int8_t *__restrict__ a, const int8_t *__restrict__ b, int32_t *__restrict__ d, unsigned m,
    unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldd,
    const uint32_t *__restrict__ key, const uint32_t *__restrict__ target, NativeResults results,
    uint32_t *diagnostic) {
    using Configuration = Gfx1100Pearl;
    constexpr bool Raw = false;
    constexpr bool Diagnostic = false;
    constexpr bool NoD = false;
#include "kernel_body.inc"
}

extern "C" __global__ __launch_bounds__(Gfx1100Pearl::Scheduling::threads)
void pearl_iris_diagnostic(
    const int8_t *__restrict__ a, const int8_t *__restrict__ b, int32_t *__restrict__ d, unsigned m,
    unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldd,
    const uint32_t *__restrict__ key, const uint32_t *__restrict__ target, NativeResults results,
    uint32_t *diagnostic) {
    using Configuration = Gfx1100Pearl;
    constexpr bool Raw = false;
    constexpr bool Diagnostic = true;
    constexpr bool NoD = false;
#include "kernel_body.inc"
}

} // namespace tnn::hip::iris::gemm::qualified
