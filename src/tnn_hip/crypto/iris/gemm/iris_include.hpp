#pragma once

// GEMM backend include shim:
// - RTC/header-flattened builds typically resolve "iris.hpp" directly.
// - Normal source builds can fall back to the repo-relative path.
#if __has_include("iris.hpp")
#include "iris.hpp"
#elif __has_include("../../../iris/rt/iris.hpp")
#include "../../../iris/rt/iris.hpp"
#else
#error "Unable to locate iris.hpp for iris::gemm backend"
#endif
