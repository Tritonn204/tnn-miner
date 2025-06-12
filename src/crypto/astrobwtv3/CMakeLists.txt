cmake_minimum_required(VERSION 3.18)
project(astrobwt_benchmark)

# Inherit parent project settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_COLOR_DIAGNOSTICS ON)
# force C colored diagnostic output
set(CFLAGS "${CFLAGS} -fdiagnostics-color=always")
# force C++ colored diagnostic output
set(CXXFLAGS "${CXXFLAGS} -fdiagnostics-color=always")
set(CMAKE_BUILD_TYPE Release)

# Get parent directory paths - if benchmark is in src/crypto/astrobwtv3/
# Going up 3 levels gets us to the project root
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
message(STATUS "Project root: ${PROJECT_ROOT}")

# Set compiler flags to match main project
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -Wuninitialized -DNDEBUG")

# Architecture specific flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flax-vector-conversions -march=armv8-a+aes")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=x86-64-v3 -m64")
endif()

# CPM setup - use the same cache directory as the main project
set(CPM_SOURCE_CACHE "${PROJECT_ROOT}/_deps/CPM")
set(CPM_DOWNLOAD_VERSION 0.40.8)
set(CPM_DOWNLOAD_LOCATION "${PROJECT_ROOT}/_deps/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

# Include directories
include_directories(
    "${PROJECT_ROOT}/include"
    "${PROJECT_ROOT}/include/highwayhash"
    "${PROJECT_ROOT}/src"
    "${PROJECT_ROOT}/src/astrobwtv3"
    "${PROJECT_ROOT}/src/crypto"
    "${PROJECT_ROOT}/lib"
    "${PROJECT_ROOT}/extern"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# For Windows/MinGW
if(WIN32)
    include_directories(
        "C:/mingw64/include"
        "C:/mingw64/x86_64-w64-mingw32/include"
    )
    link_directories(
        "C:/mingw64/lib"
        "C:/mingw64/x86_64-w64-mingw32/lib"
    )
endif()

# Find required packages
set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)

# Boost setup
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.80 COMPONENTS chrono thread)

if(Boost_FOUND)
    message("OS-provided Boost found")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
else()
    message("Boost not found. Using CPM to add it")
    
    file(DOWNLOAD https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
         "${PROJECT_ROOT}/_deps/boost-1.86.0-cmake.tar.xz"
         EXPECTED_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57)
    
    if(WIN32)
        set(CMAKE_ASM_FLAGS -Wno-unused-command-line-argument)
    endif()
    
    CPMAddPackage(
        NAME Boost
        VERSION 1.86.0
        URL "${PROJECT_ROOT}/_deps/boost-1.86.0-cmake.tar.xz"
        URL_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
        OPTIONS 
            "BOOST_ENABLE_CMAKE ON"
            "BOOST_SKIP_INSTALL_RULES ON"
            "BOOST_USE_STATIC_LIBS ON"
            "BOOST_INSTALL_LAYOUT system"
            "BOOST_INCLUDE_LIBRARIES chrono\\\;thread"
    )

    if(Boost_ADDED)
        file(GLOB boost_includes "${Boost_SOURCE_DIR}/libs/*/include")
        foreach(dir ${boost_includes})
            include_directories(${dir})
        endforeach()

        file(GLOB boost_includes "${Boost_SOURCE_DIR}/libs/*/*/include")
        foreach(dir ${boost_includes})
            include_directories(${dir})
        endforeach()
    else()
        message(FATAL_ERROR "Boost was neither found nor added.")
    endif()
endif()

# Source files for benchmark
set(BENCHMARK_SOURCES
    bench.cpp
    astrobwtv3.h
    astrobwtv3.cpp

    # Add wolfPermute implementations
    ${PROJECT_ROOT}/src/crypto/astrobwtv3/wolfbranching.cpp
    
    # Add divsufsort
    ${PROJECT_ROOT}/src/crypto/astrobwtv3/divsufsort.c
    ${PROJECT_ROOT}/src/crypto/astrobwtv3/sssort.c
    ${PROJECT_ROOT}/src/crypto/astrobwtv3/trsort.c
    ${PROJECT_ROOT}/src/crypto/astrobwtv3/utils.c
    
    # Add other required source files
    ${PROJECT_ROOT}/src/crypto/astrobwtv3/lookupcompute.cpp
)

# Create benchmark executable
add_executable(astro_benchmark ${BENCHMARK_SOURCES})

# Link libraries
target_link_libraries(astro_benchmark
    OpenSSL::Crypto
    Boost::chrono
    Boost::thread
)

# Platform specific linking
if(WIN32)
    target_link_libraries(astro_benchmark ws2_32 crypt32 winpthread)
elseif(UNIX)
    target_link_libraries(astro_benchmark pthread)
endif()

# Set output directory
set_target_properties(astro_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Add optimization flags for the benchmark
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(astro_benchmark PRIVATE -march=native -mtune=native)
endif()