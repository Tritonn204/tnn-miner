#include "dispatch.hpp"
#include "portable/recipe.hpp"

namespace tnn::hip::iris::gemm {

hipError_t launch(const Problem &p, Mode mode, hipStream_t stream, Backend requested) {
    if (!valid_problem(p, mode)) {
        return hipErrorInvalidValue;
    }

    int device = 0;
    hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) {
        return status;
    }
    hipDeviceProp_t properties{};
    status = hipGetDeviceProperties(&properties, device);
    if (status != hipSuccess) {
        return status;
    }

    const auto backend = select_backend(properties.gcnArchName, requested);
    if (backend == Backend::QualifiedGfx1100) {
        if (select_backend(properties.gcnArchName, Backend::Automatic) != backend) {
            return hipErrorNoBinaryForGpu;
        }
#if defined(IRIS_DISPATCH_WITH_GFX1100)
        return qualified::launch(p.a, p.b, p.d, p.m, p.n, p.k, p.lda, p.ldb, p.ldd, p.rank, p.key,
                                 p.target, p.results, p.diagnostic, stream, mode);
#else
        // Fallback-only builds must not pretend the qualified object was linked.
        if (requested != Backend::Automatic) {
            return hipErrorNoBinaryForGpu;
        }
#endif
    } else if (backend != Backend::PortableSimt) {
        return hipErrorInvalidValue;
    }

    using Layout = portable::SimtPearl::Layout;
    using Schedule = portable::SimtPearl::Schedule;
    const unsigned total = (p.m / Layout::rows) * (p.n / Layout::columns);
    for (unsigned first = 0; first < total; first += Schedule::max_tiles_per_launch) {
        const unsigned remaining = total - first;
        const unsigned count =
            remaining < Schedule::max_tiles_per_launch ? remaining : Schedule::max_tiles_per_launch;
        status = launch_portable_range(p, mode, stream, first, count);
        if (status != hipSuccess) {
            return status;
        }
    }
    return hipSuccess;
}

} // namespace tnn::hip::iris::gemm
