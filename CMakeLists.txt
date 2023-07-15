cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -static -Wall -O3 -fopenmp -march=native -funroll-all-loops -ftree-vectorize -std=c++2a")

if(WIN32)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wa,-mbig-obj")
endif()

project(
  Tnn-miner
  VERSION 0.1.3
  DESCRIPTION "Open Source Astrobwt Miner"
)

if(WIN32)
  include_directories(
    PUBLIC
    "C:/mingw64/include"
    "C:/mingw64/x86_64-w64-mingw32/include"
    "C:/mingw64/include/c++"
  )
  link_directories(
    "C:/mingw64/lib"
  )
  set(OPENSSL_USE_STATIC_LIBS TRUE)
  find_package(OpenSSL REQUIRED)
endif()

include_directories(
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/highwayhash
  PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/src/headers"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/"
  "${PROJECT_SOURCE_DIR}/src/miner/"
)

file(GLOB all_SRCS
  "${PROJECT_SOURCE_DIR}/include/*.h"
  "${PROJECT_SOURCE_DIR}/include/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.c"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.c"
  "${PROJECT_SOURCE_DIR}/src/miner/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/miner/*.c"
)

add_executable(${PROJECT_NAME} ${all_SRCS})

if(WIN32)
  target_link_libraries(${PROJECT_NAME} mingw32 boost_context-mt boost_json-mt boost_thread-mt boost_atomic-mt wsock32 ws2_32 ssl OpenSSL::Crypto)
else()
  target_link_libraries(${PROJECT_NAME} ssl atomic crypto pthread boost_chrono boost_context boost_json boost_thread boost_atomic)
endif()

target_link_libraries(${PROJECT_NAME} gmp gmpxx divsufsort)