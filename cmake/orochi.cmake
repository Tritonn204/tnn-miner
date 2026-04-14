if (WITH_OROCHI)
    add_definitions(-DTNN_HIP)
    add_definitions(-DWITH_OROCHI)
    message(STATUS "Building with Orochi (unified AMD+NVIDIA GPU support)")

    # Legacy HIP algos not yet ported to Orochi RTC
    set(WITH_ASTRIXHASH OFF CACHE BOOL "" FORCE)
    set(WITH_NXLHASH OFF CACHE BOOL "" FORCE)
    set(WITH_WALAHASH OFF CACHE BOOL "" FORCE)

    # ---- Orochi core sources ----
    set(OROCHI_DIR "${PROJECT_SOURCE_DIR}/contrib/Orochi")
    set(OROCHI_SOURCES
        "${OROCHI_DIR}/Orochi/Orochi.cpp"
        "${OROCHI_DIR}/contrib/hipew/src/hipew.cpp"
        "${OROCHI_DIR}/contrib/cuew/src/cuew.cpp"
    )

    # ---- Include paths ----
    # Orochi.h does #include "../contrib/hipew/include/hipew.h" so it
    # needs the Orochi/ directory as an include root.
    set(OROCHI_INCLUDE_DIRS
        "${OROCHI_DIR}"
        "${OROCHI_DIR}/contrib/hipew/include"
        "${OROCHI_DIR}/contrib/cuew/include"
    )

    # ---- CUDA headers (for cuew compilation) ----
    # Try to find CUDA include path for OROCHI_ENABLE_CUEW.
    # Without this, the NVIDIA backend won't compile, but the AMD backend still works.
    set(_oro_cuda_path "")
    if (DEFINED ENV{CUDA_PATH})
        set(_oro_cuda_path "$ENV{CUDA_PATH}")
    elseif (WIN32 AND EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        # Find newest CUDA version
        file(GLOB _cuda_versions "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
        if (_cuda_versions)
            list(SORT _cuda_versions)
            list(GET _cuda_versions -1 _oro_cuda_path)
        endif()
    elseif (EXISTS "/usr/local/cuda")
        set(_oro_cuda_path "/usr/local/cuda")
    endif()

    if (_oro_cuda_path)
        message(STATUS "Orochi: CUDA SDK found at ${_oro_cuda_path}")
        list(APPEND OROCHI_INCLUDE_DIRS "${_oro_cuda_path}/include")
        add_definitions(-DOROCHI_ENABLE_CUEW)
    else()
        message(STATUS "Orochi: CUDA SDK not found — NVIDIA backend disabled (AMD-only)")
    endif()

    # ---- Embed HIP sources (same as WITH_HIP path) ----
    include(cmake/embed_hip_sources.cmake)

    # ---- GPU source list (compiled as plain C++ under Orochi) ----
    # NOTE: Legacy .hip kernel files (hello-world, astrix, nxl, wala) are excluded —
    # they use HIP language extensions (__global__, <<<>>>) and need hipcc.
    # Only RTC-compiled algos (xelis) work under Orochi for now.
    list(APPEND TNN_HIP_SOURCES
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/main_hip.cpp"
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/test_hiprtc_isolation.cpp"
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/gpu_rtc_precompile.cpp"
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/devInfo.cpp"
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/gpuOC.hip"
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/gpuOCArgs.cpp"
        "${PROJECT_SOURCE_DIR}/src/core/hipkill.hip"
    )

    # Embedded kernel headers (same as WITH_HIP)
    embed_hip_sources(
        OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/tnn_hip_common_embedded.hpp"
        SOURCES
            "${PROJECT_SOURCE_DIR}/src/tnn_hip/common/stdint-jit.hip.inc"
            "${PROJECT_SOURCE_DIR}/src/tnn_hip/common/uint128-compat.hip.inc"
        MANIFEST_NAME COMMON_HEADERS
    )

    # Per-algo HIP crypto sources
    include(cmake/hip-crypto/astrix-hash/astrix-hash-hip.cmake)
    include(cmake/hip-crypto/nxl-hash/nxl-hash-hip.cmake)
    include(cmake/hip-crypto/wala-hash/wala-hash-hip.cmake)
    include(cmake/hip-crypto/xelis-hash/xelis-hash-hip.cmake)
    include(cmake/hip-crypto/kawpow/kawpow-hip.cmake)

    # NOTE: We do NOT define __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__ at
    # compile time. Platform detection happens at runtime via oroGetCurAPI().
    # The kernel RTC compiler will pass the appropriate -D flag based on the
    # detected backend.

endif()
