cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-static -O3 -fPIE -ftree-vectorize -mtune=znver4")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp -pthread -m64 -Wall -Wuninitialized -funroll-all-loops -ftree-vectorize -march=x86-64-v3 -mtune=znver4")

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER g++)

if(WIN32)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wa,-mbig-obj")
endif()

if(UNIX)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive")
endif()

set(CMAKE_EXE_LINKER_FLAGS "-fstack-protector")

set (CMAKE_CXX_STANDARD 23)
# set (CMAKE_CUDA_STANDARD 17)

project(
  Tnn-miner
  VERSION 0.2.0
  DESCRIPTION "Open Source Astrobwt Miner"
)

# enable_language(CUDA)
# find_package(CUDAToolkit REQUIRED)

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

include_directories(${CUDAToolkit_INCLUDE_DIRS})

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
  chrono
  context
  json
  thread
  atomic
)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/highwayhash
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/src/headers"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/"
  "${PROJECT_SOURCE_DIR}/src/miner/"
)

file(GLOB all_SRCS
  "${PROJECT_SOURCE_DIR}/include/*.h"
  "${PROJECT_SOURCE_DIR}/include/*.hpp"
  # "${PROJECT_SOURCE_DIR}/include/*.cuh"
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.c"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.c"
  # "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cu"
  # "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cuh"
  "${PROJECT_SOURCE_DIR}/src/miner/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/miner/*.c"
)

add_definitions(${GTK3_CFLAGS_OTHER})
add_executable(${PROJECT_NAME} ${all_SRCS})

if(WIN32)
  target_link_libraries(${PROJECT_NAME} mingw32 wsock32 ws2_32 winpthread)
endif()
if(UNIX)
  target_link_libraries(${PROJECT_NAME} udns pthread)
endif()
target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json OpenSSL::SSL ${Boost_LIBRARIES} OpenSSL::Crypto gmp gmpxx ${CUDA_LIBRARIES})

# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;86" CUDA_SEPARABLE_COMPILATION ON)
# # set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

# target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#   -Xptxas -O3,-v;
#   --extended-lambda;
#   --use_fast_math;
# >)