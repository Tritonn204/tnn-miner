cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -pthread -std=c++2a -march=native -m64 -Wall -Wuninitialized -funroll-all-loops -ftree-vectorize")

if(WIN32)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wa,-mbig-obj")
endif()

# if(UNIX)
# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive")
# endif()

project(
  Tnn-miner
  VERSION 0.1.4
  DESCRIPTION "Open Source Astrobwt Miner"
)

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

set(OPENSSL_USE_STATIC_LIBS ON)
find_package(OpenSSL REQUIRED)

set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.82 REQUIRED
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
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.c"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.c"
  "${PROJECT_SOURCE_DIR}/src/miner/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/miner/*.c"
)

add_executable(${PROJECT_NAME} ${all_SRCS})

if(WIN32)
  target_link_libraries(${PROJECT_NAME} mingw32 wsock32 ws2_32)
endif()
target_link_libraries(${PROJECT_NAME} ssl ${Boost_LIBRARIES} OpenSSL::Crypto gmp gmpxx divsufsort)