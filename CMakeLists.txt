cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -static -m64 -Wall -O3 -fopenmp -funroll-all-loops -march=x86-64 -ftree-vectorize -std=c++2a")

if(WIN32)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wa,-mbig-obj")
endif()

project(
  Tnn-miner
  VERSION 0.1
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
    "${PROJECT_SOURCE_DIR}/lib/Win32"
  )
  set(OPENSSL_USE_STATIC_LIBS TRUE)
  find_package(OpenSSL REQUIRED)
endif()


if(UNIX)
  set(Boost_NO_SYSTEM_PATHS TRUE)
  if (Boost_NO_SYSTEM_PATHS)
    set(Boost_NO_BOOST_CMAKE TRUE)
    set(BOOST_ROOT "/opt/boost")
    set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")
    set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/lib")
  endif (Boost_NO_SYSTEM_PATHS)
  set(Boost_USE_STATIC_LIBS   ON)
  set(Boost_USE_MULTITHREADED ON)
  find_package( 
    Boost 1.82 REQUIRED 
    COMPONENTS context json thread atomic
  )
  link_directories(
    "/opt/boost/lib"
    "${PROJECT_SOURCE_DIR}/lib/Linux"
  )
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
  target_link_libraries(${PROJECT_NAME} ssl crypto pthread Boost::context Boost::json Boost::thread Boost::atomic)
endif()

target_link_libraries(${PROJECT_NAME} fmt gmp gmpxx divsufsort)