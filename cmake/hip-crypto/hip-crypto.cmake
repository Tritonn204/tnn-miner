if (WITH_HIP)
  add_definitions(/DTNN_HIP)
  message(STATUS "Building with HIP GPU support")
  cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21

  # set(CMAKE_HIP_CREATE_STATIC_LIBRARY ON)

  set(CMAKE_HIP_COMPILER_WORKS 1)

  file(GLOB_RECURSE hipSources
    src/crypto/hip/hello-world.hip
  )

  # Create a static archive incrementally for large object file counts.
  if(NOT DEFINED CMAKE_HIP_ARCHIVE_CREATE)
    set(CMAKE_HIP_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  endif()
  if(NOT DEFINED CMAKE_HIP_ARCHIVE_APPEND)
    set(CMAKE_HIP_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
  endif()
  if(NOT DEFINED CMAKE_HIP_ARCHIVE_FINISH)
    set(CMAKE_HIP_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
  endif()

  # # compile a HIP file into an object file
  # if(NOT CMAKE_HIP_COMPILE_OBJECT)
  #   set(CMAKE_HIP_COMPILE_OBJECT
  #     "<CMAKE_HIP_COMPILER> ${_CMAKE_HIP_EXTRA_FLAGS} <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> ${_CMAKE_COMPILE_AS_HIP_FLAG} -c <SOURCE>")
  # endif()


  add_library(tnn_hip ${hipSources})
  set_target_properties(tnn_hip PROPERTIES HIP_ARCHITECTURES gfx1100)
  set_target_properties(tnn_hip PROPERTIES LANGUAGES HIP)
  set_target_properties(tnn_hip PROPERTIES LINKER_LANGUAGE HIP)

endif()
  remove_definitions(/DTNN_HIP)
unset(WITH_HIP CACHE)