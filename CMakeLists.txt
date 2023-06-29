cmake_minimum_required(VERSION 3.26)

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

project(
  Tnn-miner
  VERSION 0.1
  DESCRIPTION "Open Source Astrobwt Miner"
)

include_directories(
  PUBLIC
  "C:/mingw64/include"
  "C:/mingw64/x86_64-w64-mingw32/include"
  "C:/mingw64/include/c++"
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

link_directories(
  "C:/ming64/lib"
)

add_executable(${PROJECT_NAME} ${all_SRCS})

if(WIN32)
  target_link_libraries(${PROJECT_NAME} wsock32 ws2_32)
endif()

target_link_libraries(${PROJECT_NAME} mingw32 fmt libtomcrypt boost_context-mt boost_thread-mt boost_atomic-mt ssl crypto gmp gmpxx)