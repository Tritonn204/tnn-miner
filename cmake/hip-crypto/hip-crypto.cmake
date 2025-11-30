if (WITH_HIP)
  add_definitions(/DTNN_HIP)
  message(STATUS "Building with HIP GPU support")

  if(NOT DEFINED HIP_PLATFORM)
    set(HIP_PLATFORM "amd")
  endif()

  if (HIP_PLATFORM MATCHES "nvidia" OR HIP_PLATFORM MATCHES "nvcc")
    # set(TNN_RDC "-rdc=false")
  else()
    set(TNN_RDC "-fno-gpu-rdc")
  endif()

  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${TNN_RDC}")
  unset(TNN_RDC CACHE)

  # Global HIP source list for the whole project
  # (this will be visible in the top-level CMake)
  list(APPEND TNN_HIP_SOURCES
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/main_hip.cpp"
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/hello-world.hip"
    "${PROJECT_SOURCE_DIR}/src/tnn_hip/core/devInfo.hip"
    "${PROJECT_SOURCE_DIR}/src/core/hipkill.hip"
  )

  # These included cmakes should also do list(APPEND TNN_HIP_SOURCES ...)
  include(cmake/hip-crypto/astrix-hash/astrix-hash-hip.cmake)
  include(cmake/hip-crypto/nxl-hash/nxl-hash-hip.cmake)
  include(cmake/hip-crypto/wala-hash/wala-hash-hip.cmake)

  if (HIP_PLATFORM MATCHES "nvidia")
    add_compile_definitions(__HIP_PLATFORM_NVIDIA__)
  else()
    add_compile_definitions(__HIP_PLATFORM_AMD__)
  endif()
else()
  remove_definitions(/DTNN_HIP)
endif()
