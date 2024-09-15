if (WITH_HIP)
  add_definitions(/DTNN_HIP)
  message(STATUS "Building with HIP GPU support")
  cmake_minimum_required(VERSION 3.28) # TNN HIP language support requires 3.28

  # set(HIP_PLATFORM "amd" CACHE STRING "Specify HIP platform (amd or nvidia)")

  # if(HIP_PLATFORM MATCHES "amd")
  #     add_definitions(-D__HIP_PLATFORM_AMD__)
  #     message(STATUS "Building for AMD platform")
  # elseif(HIP_PLATFORM MATCHES "nvidia")
  #     add_definitions(-D__HIP_PLATFORM_NVIDIA__)
  #     message(STATUS "Building for NVIDIA platform")
  # else()
  #     message(FATAL_ERROR "Invalid HIP platform specified. Must be 'amd' or 'nvidia'.")
  # endif()

  # if(WIN32)
  #   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target x86_64-pc-windows-gnu")
  #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target x86_64-pc-windows-gnu")
  #   set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -target x86_64-pc-windows-gnu")
  # endif()

  # # set(CMAKE_HIP_COMPILER_WORKS 1)

  # # enable_language(HIP)
  # # set(CMAKE_HIP_COMPILER ${HIP_PATH}/bin/clang.exe)

  file(GLOB_RECURSE hipSources
    src/tnn_hip/hello-world.hip
    # src/tnn_hip/crypto/cshake/cshake256.hip
  )
  
  include(cmake/hip-crypto/astrix-hash/astrix-hash-hip.cmake)

  # # Add HIP sources and libraries
  # add_library(tnn_hip STATIC ${hipSources})
  # # add_executable(tnn_hip_run ${hipSources})

  # install(TARGETS tnn_hip
  #   DESTINATION ${CMAKE_INSTALL_PREFIX}
  # )

  if (CMAKE_HIP_PLATFORM MATCHES nvidia OR CMAKE_HIP_PLATFORM MATCHES nvcc)
    # set(TNN_RDC "-rdc=false")
  else()
    set(TNN_RDC "-fno-gpu-rdc")
  endif()

  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${TNN_RDC}")
  list(APPEND SOURCES_CRYPTO
    ${hipSources}
    src/tnn_hip/crypto/cshake/ref-cshake.cpp
  )

  # # Create a static archive incrementally for large object file counts.
  # if(NOT DEFINED CMAKE_HIP_ARCHIVE_CREATE)
  #   set(CMAKE_HIP_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  # endif()
  # if(NOT DEFINED CMAKE_HIP_ARCHIVE_APPEND)
  #   set(CMAKE_HIP_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
  # endif()
  # if(NOT DEFINED CMAKE_HIP_ARCHIVE_FINISH)
  #   set(CMAKE_HIP_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
  # endif()

  # add_library(tnn_hip STATIC ${hipSources})
  # set_target_properties(tnn_hip PROPERTIES HIP_ARCHITECTURES gfx1100)
  # set_target_properties(tnn_hip PROPERTIES LANGUAGES HIP)
  # set_target_properties(tnn_hip PROPERTIES LINKER_LANGUAGE HIP)

  # set(CMAKE_C_COMPILER ${HIP_PATH}/bin/clang.exe)
  # set(CMAKE_CXX_COMPILER ${HIP_PATH}/bin/clang++.exe)

  # set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -target x86_64-pc-windows-gnu")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target x86_64-pc-windows-gnu")
else()
  remove_definitions(/DTNN_HIP)
endif()