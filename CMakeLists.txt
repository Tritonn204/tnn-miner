# HIP support was added in CMake 3.21
cmake_minimum_required(VERSION 3.16.3)

include(CMakePrintHelpers)

#project(
	#  Tnn-miner
  #  VERSION 0.3.2
  #  DESCRIPTION "Open Source Astrobwt Miner"
  #)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -static -O3 -fPIE -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -static -O3 -pthread -Wuninitialized -DNDEBUG")

if(USE_GCC)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated-declarations")
else()
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
  #set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE STRING "C++ compiler" FORCE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated")
endif()

project(
  Tnn-miner
  VERSION 0.3.4
  DESCRIPTION "Open Source Astrobwt Miner"
)

# This CMAKE variable isn't defined until 'project' is called.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flax-vector-conversions -march=armv8-a")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flax-vector-conversions -march=armv8-a")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -m64")
endif()

# if(WIN32)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

# if(UNIX)
# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fstack-protector")

set (CMAKE_STANDARD_REQUIRED TRUE)
set (CMAKE_CXX_STANDARD 20)

enable_language(ASM)

include(CMakePrintHelpers)

if(WIN32)
  include_directories(
    "C:/mingw64/include"
    "C:/mingw64/x86_64-w64-mingw32/include"
    "C:/mingw64/include/c++"
  )
  link_directories(
    "C:/mingw64/lib"
  )
endif()

# This stores downloaded dependencies so we do not need to re-download them
set(CPM_SOURCE_CACHE "${PROJECT_SOURCE_DIR}/.cache/CPM")

set(CPM_DOWNLOAD_VERSION 0.39.0)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage("gh:nlohmann/json@3.10.5")

set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)

find_program(LSB_RELEASE_EXEC lsb_release)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -cs
    OUTPUT_VARIABLE LSB_RELEASE_CODENAME_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_NO_WARN_NEW_VERSIONS ON)
find_package(Boost 1.80 COMPONENTS
  chrono
  context
  json
  thread
  atomic
  program_options
)

if(Boost_FOUND)
  message("OS-provided Boost found")
  set(TNN_BOOST_INCLUDES ${Boost_INCLUDE_DIRS})
else()
  message("Boost not found. Using CPM to add it")
  CPMAddPackage(
    NAME Boost
    VERSION 1.85.0
    URL https://github.com/boostorg/boost/releases/download/boost-1.85.0/boost-1.85.0-cmake.tar.xz
    URL_HASH SHA256=0a9cc56ceae46986f5f4d43fe0311d90cf6d2fa9028258a95cab49ffdacf92ad
    OPTIONS "Boost_ENABLE_CMAKE ON" "BOOST_INCLUDE_LIBRARIES chrono\\\;context\\\;json\\\;thread\\\;atomic\\\;program_options" # Note the escapes!
      # Maintain build compatibility between find_package and CMakeLists.txt variants
  )

  if(Boost_ADDED)
    file(GLOB files "${Boost_SOURCE_DIR}/libs/*/include")
    foreach(filename ${files})
      include_directories(${filename})
    endforeach()

    file(GLOB files "${Boost_SOURCE_DIR}/libs/*/*/include")
    foreach(filename ${files})
      include_directories(${filename})
    endforeach()
  else()
    message(FATAL_ERROR "Boost was neither found nor added.")
  endif()

  set(TNN_BOOST_INCLUDES ${Boost_SOURCE_DIR})
endif()

# This is universal regardless of how wheter we compiled or found Boost
set(TNN_BOOST_LIBS Boost::chrono Boost::context Boost::json Boost::thread Boost::atomic Boost::program_options)

#cmake_print_variables(TNN_BOOST_INCLUDES)
#cmake_print_variables(TNN_BOOST_LIBS)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/highwayhash
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/src/headers"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/"
  "${PROJECT_SOURCE_DIR}/src/spectrex/"
  "${PROJECT_SOURCE_DIR}/src/xatum/"
  "${PROJECT_SOURCE_DIR}/src/stratum/"
  "${PROJECT_SOURCE_DIR}/src/miner/"
  "${TNN_BOOST_INCLUDES}"
  "${PROJECT_SOURCE_DIR}/src/broadcast/"
)

file(GLOB all_SRCS
  "${PROJECT_SOURCE_DIR}/include/*.h"
  "${PROJECT_SOURCE_DIR}/include/*.hpp"
  "${PROJECT_SOURCE_DIR}/include/libkeccak/*.c"
  # "${PROJECT_SOURCE_DIR}/include/*.cuh"
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.c"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.c"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/*.c"
  "${PROJECT_SOURCE_DIR}/src/spectrex/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/spectrex/*.c"
  "${PROJECT_SOURCE_DIR}/src/broadcast/*.cpp"
  # "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cu"
  # "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cuh"
  "${PROJECT_SOURCE_DIR}/src/miner/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/miner/*.c"
)

# Remove aarch files from all_SRCS
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  list(FILTER all_SRCS EXCLUDE REGEX ".*/.*/*aarch64*")
endif()


add_executable(${PROJECT_NAME} ${all_SRCS})

if(WIN32)
  target_link_libraries(${PROJECT_NAME} mingw32 wsock32 ws2_32 winpthread)
endif()
if(UNIX)
  target_link_libraries(${PROJECT_NAME} udns pthread)
endif()
target_link_libraries(${PROJECT_NAME} Threads::Threads nlohmann_json::nlohmann_json ${TNN_BOOST_LIBS} OpenSSL::SSL OpenSSL::Crypto)
