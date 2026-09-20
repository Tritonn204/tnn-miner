if (WITH_HIP)
  include(cmake/embed_hip_sources.cmake)

  add_definitions(/DTNN_HIP)
  message(STATUS "Building with HIP GPU support")

  if(NOT DEFINED HIP_PLATFORM) 
    set(HIP_PLATFORM "amd")
  endif()

  if (HIP_PLATFORM MATCHES "nvidia" OR HIP_PLATFORM MATCHES "nvcc")
    # set(TNN_RDC "-rdc=false")
  else()
    # set(TNN_RDC "-fno-gpu-rdc")
  endif()

  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${TNN_RDC}")
  unset(TNN_RDC CACHE)

  # Global HIP source list for the whole project
  # (this will be visible in the top-level CMake)
  list(APPEND TNN_HIP_SOURCES
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/main_hip.cpp"
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/test_hiprtc_isolation.cpp"
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/gpu_rtc_precompile.cpp"
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/hello-world.hip"
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/devInfo.cpp"
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/gpuOC.hip"
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/gpuOCArgs.cpp"
    "${PROJECT_SOURCE_DIR}/src/core/hipkill.hip"
  )

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/tnn_hip_common_embedded.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/common/stdint-jit.hip.inc"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/common/hiprtc_types.hip.h"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/common/uint128-compat.hip.inc"
      MANIFEST_NAME COMMON_HEADERS
  )

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/iris_embedded_headers.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/arch_traits.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/block_reduce.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/buffer_view.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/coordinate.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/copy.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/iris.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/lds_view.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/load_store_traits.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/space_filling_curve.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/static_distributed_tensor.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/sweep_tile.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/tensor_adaptor.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/tensor_coordinate.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/tensor_desc.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/tensor_view.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/thread_map.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/tile_distribution.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/tile_window.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/transform.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/vec_store.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/waitcnt.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/warp_primitives.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/warp_wmma.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/wmma_i8.hpp"
      MANIFEST_NAME IRIS_HEADERS
  )

  # These included cmakes should also do list(APPEND TNN_HIP_SOURCES ...)
  include(cmake/hip-crypto/astrix-hash/astrix-hash-hip.cmake)
  include(cmake/hip-crypto/nxl-hash/nxl-hash-hip.cmake)
  include(cmake/hip-crypto/wala-hash/wala-hash-hip.cmake)
  include(cmake/hip-crypto/xelis-hash/xelis-hash-hip.cmake)
  include(cmake/hip-crypto/kawpow/kawpow-hip.cmake)
  if (WITH_QHASH)
    include(cmake/hip-crypto/qhash/qhash-hip.cmake)
  endif()
  include(cmake/hip-crypto/pearl-pouw/pearl-pouw-hip.cmake)

  if (HIP_PLATFORM MATCHES "nvidia")
    add_compile_definitions(__HIP_PLATFORM_NVIDIA__)
  else()
    add_compile_definitions(__HIP_PLATFORM_AMD__)
  endif()
else()
  remove_definitions(/DTNN_HIP)
endif()
