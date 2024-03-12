cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.27)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-fansi-escape-codes -fcolor-diagnostics -static -O3 -ftree-vectorize")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fansi-escape-codes -fcolor-diagnostics -static -m64 -g -Wuninitialized -ftree-vectorize")
set(CMAKE_HIP_FLAGS "-static -fansi-escape-codes -fcolor-diagnostics -g -fgpu-rdc --hip-link -fuse-ld=lld -Wno-deprecated-register")

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

set(CMAKE_EXE_LINKER_FLAGS "-fstack-protector")

if(WIN32)
    set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_hip_STANDARD 17)

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


set(CPM_DOWNLOAD_VERSION 0.34.0)

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

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.80 REQUIRED
  system
  coroutine
  chrono
  json
  context
  atomic
)
include_directories( ${Boost_INCLUDE_DIRS} )

if(WIN32)
  include_directories(
    "C:/mingw64/include"
  )
  link_directories(
    "C:/mingw64/lib"
  )
endif()


include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/highwayhash
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/"
  "${PROJECT_SOURCE_DIR}/src/miner/"
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
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.hip"
  "${PROJECT_SOURCE_DIR}/src/miner/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/miner/*.c"
)

add_definitions(${GTK3_CFLAGS_OTHER})
add_executable(${PROJECT_NAME} ${all_SRCS})

if(WIN32)
  target_link_libraries(${PROJECT_NAME} wsock32 ws2_32)
endif()
if(UNIX)
  target_link_libraries(${PROJECT_NAME} udns)
endif()

target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json OpenSSL::SSL ${Boost_LIBRARIES} OpenSSL::Crypto hip::host roc::rocprim roc::rocthrust)


set_property(TARGET ${PROJECT_NAME} PROPERTY hip_ARCHITECTURES gfx1100)
set(CMAKE_EXE_LINKER_FLAGS "-fgpu-rdc -lC:/Boost/lib/libboost_coroutine-clangw17-mt-x64-1_82.lib")