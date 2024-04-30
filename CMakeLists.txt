cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -static -O3 -g -fPIE -ftree-vectorize -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -static -O3 -g -pthread -m64 -Wuninitialized -march=native -DNDEBUG")

if(USE_GCC)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated-declarations")
else()
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated")
endif()

project(
  Tnn-miner
  VERSION 0.3.1
  DESCRIPTION "Open Source Astrobwt Miner"
)

# if(WIN32)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

# if(UNIX)
# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fstack-protector")

set (CMAKE_STANDARD_REQUIRED TRUE)
set (CMAKE_CXX_STANDARD 23)



enable_language(ASM)

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

find_program(LSB_RELEASE_EXEC lsb_release)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -cs
    OUTPUT_VARIABLE LSB_RELEASE_CODENAME_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if("${LSB_RELEASE_CODENAME_SHORT}" STREQUAL "jammy")
  set(Boost_NO_SYSTEM_PATHS TRUE)
  if (Boost_NO_SYSTEM_PATHS)
    set(Boost_NO_BOOST_CMAKE TRUE)
    set(BOOST_ROOT "${PROJECT_SOURCE_DIR}/boost-1.82.0")
    set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}")
    set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/libs")
  endif (Boost_NO_SYSTEM_PATHS)
endif()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_NO_WARN_NEW_VERSIONS ON)
find_package(Boost 1.80 REQUIRED
  chrono
  context
  json
  thread
  atomic
  program_options
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(BOOST_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/highwayhash
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/src/headers"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/"
  "${PROJECT_SOURCE_DIR}/src/xatum/"
  "${PROJECT_SOURCE_DIR}/src/miner/"
  "${BOOST_INCLUDE_DIRS}"
)

set(ASM_SOURCES 
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/*.s"
)

file(GLOB all_SRCS
  "${PROJECT_SOURCE_DIR}/include/*.h"
  "${PROJECT_SOURCE_DIR}/include/*.hpp"
  # "${PROJECT_SOURCE_DIR}/include/*.cuh"
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.c"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/*.c"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/*.c"
  "${ASM_SOURCES}"
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
target_link_libraries(${PROJECT_NAME} Threads::Threads nlohmann_json::nlohmann_json ${Boost_LIBRARIES} OpenSSL::SSL OpenSSL::Crypto)

