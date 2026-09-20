#pragma once

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
#include "arch/gfx1100.hpp"
namespace tnn::hip::iris::gemm {
using DefaultBackend = arch::gfx1100::Backend;
namespace default_arch = arch::gfx1100;
} // namespace tnn::hip::iris::gemm
#else
#error "Unsupported HIP GEMM backend architecture"
#endif
