cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.27)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-fansi-escape-codes -fcolor-diagnostics -static -fuse-ld=lld -O3 -ftree-vectorize")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fansi-escape-codes -fcolor-diagnostics -static -static-libgcc -fuse-ld=lld -m64 -O3 -Wuninitialized -ftree-vectorize -march=native")
set(CMAKE_HIP_FLAGS "-static -fansi-escape-codes -fcolor-diagnostics -fgpu-rdc -fuse-ld=lld -Wno-deprecated-register")

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

if(WIN32)
set(CMAKE_C_COMPILER "${HIP_PATH}/bin/clang.exe")
set(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/clang++.exe")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_USE_WINDOWS_H -Wa,-mbig-obj")
# add_definitions( -DBOOST_ALL_NO_LIB )
endif()

if(UNIX)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wl,--whole-archive -Wl,--no-whole-archive")
endif()

if(WIN32)
    set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
    set(CMAKE_IGNORE_PREFIX_PATH "C:/mingw64")
else()
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_hip_STANDARD 20)

project(
  Tnn-miner-ROCm
  VERSION 0.1.0
  DESCRIPTION "Open Source Astrobwt Miner"
  LANGUAGES C CXX
)

#accounting for case sensitivity issues with ROCm
if(WIN32)
find_package(HIP2 REQUIRED)
else()
find_package(HIP REQUIRED)
endif()

enable_language(HIP)
include_directories(${hip_INCLUDE_DIRS})

# On ROCm rocThrust requires rocPRIM
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

# "/opt/rocm" - default install prefix
find_package(rocthrust REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

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
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
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

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_NO_WARN_NEW_VERSIONS ON)
find_package(Boost 1.80 REQUIRED
  system
  context
  coroutine
  json 
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
    OPTIONS "Boost_ENABLE_CMAKE ON" "BOOST_INCLUDE_LIBRARIES json\\\;context\\\;coroutine\\\;program_options" # Note the escapes!
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
set(TNN_BOOST_LIBS Boost::json Boost::coroutine Boost::context Boost::program_options)

include_directories( ${Boost_INCLUDE_DIRS} )

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/highwayhash
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/src/headers"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/"
  "${PROJECT_SOURCE_DIR}/src/xatum/"
  "${PROJECT_SOURCE_DIR}/src/miner/"
  "${TNN_BOOST_INCLUDES}"
  "${PROJECT_SOURCE_DIR}/src/broadcast/"
  "${HIP_PATH}/include"
)

link_directories(
  "${HIP_PATH}/lib"
)

file(GLOB all_SRCS
  "${PROJECT_SOURCE_DIR}/include/*.h"
  "${PROJECT_SOURCE_DIR}/include/*.hpp"
  "${PROJECT_SOURCE_DIR}/include/*.hip"
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.c"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.c"
  # "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.hip"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/*.c"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/*.hip"
  "${PROJECT_SOURCE_DIR}/src/broadcast/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/miner/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/miner/*.c"
)

add_definitions(${GTK3_CFLAGS_OTHER})
add_executable(${PROJECT_NAME} ${all_SRCS})

if(WIN32)
  target_link_libraries(${PROJECT_NAME} wsock32 ws2_32)
endif()
if(UNIX)
  target_link_libraries(${PROJECT_NAME} udns pthread)
endif()

target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json ${TNN_BOOST_LIBS} OpenSSL::SSL OpenSSL::Crypto hip::host roc::rocprim roc::rocthrust)

set_property(TARGET ${PROJECT_NAME} PROPERTY HIP_ARCHITECTURES gfx1100)
set(CMAKE_EXE_LINKER_FLAGS "-fgpu-rdc --hip-link")