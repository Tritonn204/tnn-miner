if (WITH_HIP)
  add_definitions(/DTNN_HIP)
  message(STATUS "Building with HIP GPU support")
  cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21

  set(CMAKE_HIP_CREATE_STATIC_LIBRARY ON)

  file(GLOB_RECURSE hipSources
    src/crypto/hip/hello-world.hip
  )

  add_library(tnn_hip STATIC ${hipSources})
  set_target_properties(tnn_hip PROPERTIES LANGUAGES HIP)
  set_target_properties(tnn_hip PROPERTIES LINKER_LANGUAGE HIP)
endif()
  remove_definitions(/DTNN_HIP)
unset(WITH_HIP CACHE)