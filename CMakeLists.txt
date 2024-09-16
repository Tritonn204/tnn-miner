# HIP support was added in CMake 3.21
cmake_minimum_required(VERSION 3.16.3)

include(CMakePrintHelpers)
cmake_print_variables(CMAKE_HOST_SYSTEM_NAME)

set(TARGET_ARCH "amd64")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -Wuninitialized -DNDEBUG")

option(USE_GCC        "Use GCC instead of Clang" OFF)
option(USE_ASTRO_SPSA "Use Astro Stamped Permutation Suffix Array library" ON)

# if (WIN32)
#   if(WITH_HIP)
#     message(STATUS "HIP compilation is not currently support for TNN Miner on Windows.")
#   endif()
#   set(WITH_HIP OFF)
# endif()

if(USE_GCC)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated-declarations")
else()
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated")
endif()

# set(CMAKE_C_COMPILER_WORKS 1)
# set(CMAKE_CXX_COMPILER_WORKS 1)

set(CMAKE_COLOR_DIAGNOSTICS ON)
# force C colored diagnostic output
set(CFLAGS "${CFLAGS} -fdiagnostics-color=always")
# force C++ colored diagnostic output
set(CXXFLAGS "${CXXFLAGS} -fdiagnostics-color=always")

# This looks weird, but we prefer the -D variable over the ENV var
string(TIMESTAMP VERSION_DEF "%Y.%m.%d.%H-dev")
set(INTERNAL_TNN_VERSION ${VERSION_DEF})
if(DEFINED ENV{TNN_VERSION})
  set(INTERNAL_TNN_VERSION $ENV{TNN_VERSION})
endif()
if(DEFINED TNN_VERSION)
  set(INTERNAL_TNN_VERSION ${TNN_VERSION})
endif()

string(REGEX REPLACE "^[a-zA-Z]" ""  INTERNAL_TNN_VERSION ${INTERNAL_TNN_VERSION})

# TNN Options
# option(WITH_CN_LITE         "Enable CryptoNight-Lite algorithms family" ON)
# option(WITH_CN_HEAVY        "Enable CryptoNight-Heavy algorithms family" ON)
# option(WITH_CN_PICO         "Enable CryptoNight-Pico algorithm" ON)
# option(WITH_CN_FEMTO        "Enable CryptoNight-UPX2 algorithm" ON)
option(WITH_ASTROBWTV3      "Enable AstroBWTv3 algorithms family" ON)
option(WITH_XELISHASH       "Enable XelisHash algorithms family" ON)
option(WITH_RANDOMX         "Enable RandomX algorithms family" ON)
option(WITH_VERUSHASH        "Enable VerusHash algorithms family" ON)
option(WITH_ASTRIXHASH        "Enable AstrixHash algorithms family" ON)
# option(WITH_ARGON2          "Enable Argon2 algorithms family" ON)
# option(WITH_KAWPOW          "Enable KawPow algorithms family" ON)
# option(WITH_GHOSTRIDER      "Enable GhostRider algorithm" ON)
# option(WITH_EMBEDDED_CONFIG "Enable internal embedded JSON config" OFF)
# option(WITH_OPENCL          "Enable OpenCL backend" ON)
set(WITH_OPENCL_VERSION 200 CACHE STRING "Target OpenCL version")
set_property(CACHE WITH_OPENCL_VERSION PROPERTY STRINGS 120 200 210 220)
option(WITH_HIP             "Enable GPU support through the HIP SDK" ON)
# option(WITH_CUDA            "Enable CUDA backend" ON)
# option(WITH_NVML            "Enable NVML (NVIDIA Management Library) support (only if CUDA backend enabled)" ON)
# option(WITH_ADL             "Enable ADL (AMD Display Library) or sysfs support (only if OpenCL backend enabled)" ON)
# option(WITH_STRICT_CACHE    "Enable strict checks for OpenCL cache" ON)
option(WITH_INTERLEAVE_DEBUG_LOG "Enable debug log for threads interleave" OFF)
option(WITH_PROFILING       "Enable profiling for developers" OFF)
option(WITH_SSE4_1          "Enable SSE 4.1" ON)
option(WITH_AVX2            "Enable AVX2" ON)
# option(WITH_BENCHMARK       "Enable builtin RandomX benchmark and stress test" ON)
option(WITH_SECURE_JIT      "Enable secure access to JIT memory" OFF)
option(WITH_DMI             "Enable DMI/SMBIOS reader" ON)

option(BUILD_STATIC         "Build static binary" ON)
option(ARM_V8               "Force ARMv8 (64 bit) architecture, use with caution if automatic detection fails, but you sure it may work" OFF)
option(ARM_V7               "Force ARMv7 (32 bit) architecture, use with caution if automatic detection fails, but you sure it may work" OFF)
# End of XMRig options

if (WITH_HIP)
  # if (WIN32)
  #   set(CMAKE_C_COMPILER clang-cl)
  #   set(CMAKE_CXX_COMPILER clang-cl)
  # endif()
  # set(CMAKE_C_COMPILER "C:/Program Files/LLVM/bin/clang-cl.exe")  # Adjust path as needed
  # set(CMAKE_CXX_COMPILER "C:/Program Files/LLVM/bin/clang-cl.exe")  # Adjust path as needed

  if (WIN32)
    add_definitions(-DBOOST_ALL_NO_LIB)
    set(CMAKE_C_COMPILER $ENV{HIP_PATH}/bin/clang.exe)
    set(CMAKE_CXX_COMPILER $ENV{HIP_PATH}/bin/clang++.exe)

    # if (HIP_PLATFORM MATCHES "nvidia")
    #   set(CMAKE_C_COMPILER "cl")  # Adjust path as needed
    #   set(CMAKE_CXX_COMPILER "cl")  # Adjust path as needed

    #   link_directories($ENV{LIB})
    # else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target x86_64-pc-windows-msvc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target x86_64-pc-windows-msvc")
    # endif()
  endif()
  file(GLOB HIP_MODULES 
    "${HIP_PATH}/lib/cmake/*"
  )
  list(APPEND CMAKE_MODULE_PATH ${HIP_MODULES})
  set(HIP_LANG HIP)

  link_directories(
    $ENV{HIP_PATH}/lib
  )

  # set(HIP_PLATFORM $ENV{HIP_PLATFORM})
endif()

project(
  Tnn-miner
  LANGUAGES C CXX ${HIP_LANG}
  DESCRIPTION "Open Source Dero/Spectre/Xelis Miner"
)

set(PROJECT_VERSION ${INTERNAL_TNN_VERSION})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include (CheckIncludeFile)
include (cmake/cpu.cmake)
include (cmake/os.cmake)

cmake_print_variables($ENV{GITLAB_CI})
cmake_print_variables(INTERNAL_TNN_VERSION)
cmake_print_variables(CMAKE_SYSTEM_PROCESSOR)

# This CMAKE variable isn't defined until 'project' is called.
set(DO_STATIC " -static ")
if(CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
  # We cannot use static linking on OSX.
  set(DO_STATIC " ")
endif()

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin" OR USE_GCC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -fuse-ld=ld")
else()
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(WITH_HIP OFF)
  set(TARGET_ARCH "aarch64")
  # AES needed for Xelisv2
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flax-vector-conversions -march=armv8-a+aes")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flax-vector-conversions -march=armv8-a+aes")
elseif($ENV{GITLAB_CI} MATCHES "true")
  set(WITH_HIP OFF)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(TARGET_ARCH "aarch64")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -g -flax-vector-conversions -march=armv8-a+aes")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -flax-vector-conversions -march=armv8-a+aes")
  else()
    set(TARGET_ARCH "amd64")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ")
    
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=$ENV{AMD64_TARGET_LEVEL}")
  endif()
else()
  set(TARGET_ARCH "amd64")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=x86-64-v3 -m64")
endif()

set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ${DO_STATIC} -DTNN_VERSION=${INTERNAL_TNN_VERSION}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${DO_STATIC} -DTNN_VERSION=${INTERNAL_TNN_VERSION}")

cmake_print_variables(CMAKE_C_FLAGS_RELEASE)
cmake_print_variables(CMAKE_CXX_FLAGS_RELEASE)

# if(WIN32)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

# if(UNIX)
# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fstack-protector")

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_STANDARD_REQUIRED TRUE)

enable_language(ASM)

if(WIN32)
  if(NOT WITH_HIP)
    set(OPENSSL_ROOT_DIR "C:/openssl/clang64")
    include_directories(
      "C:/mingw64/include"
      "C:/mingw64/x86_64-w64-mingw32/include"
      "C:/mingw64/include/c++"
    )
    link_directories(
      "C:/mingw64/lib"
      "C:/mingw64/x86_64-w64-mingw32/lib"
    )
  else()
    set(OPENSSL_ROOT_DIR "C:/openssl_msvc")
    add_library(OpenSSL::Crypto UNKNOWN IMPORTED)
    set_target_properties(OpenSSL::Crypto PROPERTIES IMPORTED_LOCATION "${OPENSSL_ROOT_DIR}/lib/libcrypto.lib")
  
    add_library(OpenSSL::SSL UNKNOWN IMPORTED)
    set_target_properties(OpenSSL::SSL PROPERTIES IMPORTED_LOCATION "${OPENSSL_ROOT_DIR}/lib/libssl.lib")
  endif()
  include_directories(
    ${OPENSSL_ROOT_DIR}/include
  )
  link_directories(
    ${OPENSSL_ROOT_DIR}/lib
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

set(OPENSSL_USE_STATIC_LIBS TRUE)

if (NOT WIN32 OR NOT WITH_HIP)
  find_package(OpenSSL REQUIRED)
endif()

link_directories(OpenSSL_LIBRARY_DIRS)

find_program(LSB_RELEASE_EXEC lsb_release)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -cs
    OUTPUT_VARIABLE LSB_RELEASE_CODENAME_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


if (UNIX)
  set(THREAD_LIB pthread)
elseif(NOT WITH_HIP)
  set(THREAD_LIB winpthread)
endif()


set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_NO_WARN_NEW_VERSIONS ON)

if (NOT USE_CPM_BOOST)
find_package(Boost 1.80 COMPONENTS
  chrono
  context
  coroutine
  thread
  json
  atomic
  program_options
)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads REQUIRED)

if(Boost_FOUND AND NOT USE_CPM_BOOST)
  message("OS-provided Boost found")
  set(TNN_BOOST_INCLUDES ${Boost_INCLUDE_DIRS})
  include_directories(${TNN_BOOST_INCLUDES})
  set(TNN_BOOST_LIBDIRS ${Boost_LIBRARY_DIRS})

  message(STATUS "Boost library directories: ${TNN_BOOST_LIBDIRS}")

  link_directories(${TNN_BOOST_LIBDIRS})
  message(${Boost_VERSION})
else()
  message("Boost not found. Using CPM to add it")
  # if (WIN32 AND WITH_HIP)
  #   set(CPM_COMPILER 
  #     "CMAKE_C_COMPILER clang-cl"
  #     "CMAKE_CXX_COMPILER clang-cl"
  #   )
  # endif()
  CPMAddPackage(
    NAME Boost
    VERSION 1.85.0
    URL https://github.com/boostorg/boost/releases/download/boost-1.85.0/boost-1.85.0-cmake.tar.xz
    URL_HASH SHA256=0a9cc56ceae46986f5f4d43fe0311d90cf6d2fa9028258a95cab49ffdacf92ad
    OPTIONS 
      "Boost_USE_STATIC_LIBS ON" 
      "Boost_ENABLE_CMAKE ON" 
      "BOOST_INCLUDE_LIBRARIES chrono\\\;context\\\;coroutine\\\;json\\\;thread\\\;atomic\\\;program_options" 
      # "BOOST_TOOLSET clangw"  # Specify Windows Clang toolset
      # ${CPM_COMPILER}  
      # "CMAKE_CXX_FLAGS /DCLANG_COMPILER=1 /fms-compatibility"
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

endif()

# This is universal regardless of how wheter we compiled or found Boost
set(TNN_BOOST_LIBS Boost::chrono Boost::context Boost::coroutine Boost::thread Boost::json Boost::atomic Boost::program_options)

# # if(APPLE)
CPMAddPackage(
  NAME sodium
  GITHUB_REPOSITORY robinlinden/libsodium-cmake
  GIT_TAG f568ff02f1bed155ea598c0e803ef3c9db2703d2
  OPTIONS 
    "CMAKE_C_FLAGS_RELEASE -msse3 -mssse3 -maes -mpclmul -mavx2" 
    "CMAKE_CXX_FLAGS_RELEASE -msse3 -mssse3 -maes -mpclmul -mavx2"
)

# if (TARGET sodium)
#   # Add SSSE3 and PCLMUL flags for sodium target
#   target_compile_options(sodium PRIVATE -Ofast -mssse3 -mpclmul -maes -mavx2)
# endif()

include_directories(
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/include/highwayhash"
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/"
  "${PROJECT_SOURCE_DIR}/src/xelis-hash/"
  "${PROJECT_SOURCE_DIR}/src/spectrex/"
  "${PROJECT_SOURCE_DIR}/src/xatum/"
  "${PROJECT_SOURCE_DIR}/src/stratum/"
  "${PROJECT_SOURCE_DIR}/src/net/"
  "${PROJECT_SOURCE_DIR}/src/crypto/"
  "${PROJECT_SOURCE_DIR}/src/broadcast/"
  "${PROJECT_SOURCE_DIR}/lib"
  "${PROJECT_SOURCE_DIR}/extern"
)

file(GLOB_RECURSE SOURCES_CORE
  "${PROJECT_SOURCE_DIR}/include/libkeccak/*.c"
  "${PROJECT_SOURCE_DIR}/src/core/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/core/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/cshake/*.c"
  "${PROJECT_SOURCE_DIR}/src/broadcast/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/coins/dummy_miners.cpp"
)

file(GLOB SOURCES_NET
  "${PROJECT_SOURCE_DIR}/src/net/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/net/*.c"
)

if(USE_ASTRO_SPSA)
  set(SPSA_OS_PREFIX "linux")
  if(WIN32)
    set(SPSA_OS_PREFIX "win")
  endif()
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flto -DUSE_ASTRO_SPSA=ON")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto -DUSE_ASTRO_SPSA=ON")

  if(EXISTS ${PROJECT_SOURCE_DIR}/lib/astrospsa/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}.a)
    set(SPSA_LIB_DIR ${PROJECT_SOURCE_DIR}/lib/astrospsa)
  else()
    include(FetchContent)
    ## Fetch the static library
    FetchContent_Declare(
        astrospsa
        GIT_REPOSITORY https://gitlab.com/Tritonn204/astro-spsa.git
        GIT_TAG        b525e71ded6bdd997f3c21612de1dc57f08e6b55
    )
    FetchContent_MakeAvailable(astrospsa)
    set(SPSA_LIB_DIR ${astrospsa_SOURCE_DIR})
  endif()
  cmake_print_variables(SPSA_LIB_DIR)
  include_directories(${SPSA_LIB_DIR})
endif()

if (NOT WITH_HIP)
  include(cmake/astrobwtv3.cmake)
  include(cmake/xelis-hash.cmake)
  include(cmake/randomx.cmake)
  include(cmake/verus-hash.cmake)
  include(cmake/astrix-hash.cmake)

  include(cmake/blake2.cmake)
  include(cmake/chacha20.cmake)
else()
  include(cmake/hip-crypto/hip-crypto.cmake)
endif()

# enable_language(HIP)
# set(CMAKE_HIP_COMPILER_WORKS 1)


if (WITH_BLAKE3)
  add_subdirectory(${PROJECT_SOURCE_DIR}/extern/BLAKE3/c)
  set(BLAKE3_LIBRARIES blake3)
endif()

# Remove aarch files from SOURCES_CORE
#if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND NOT CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
#  list(FILTER SOURCES_CORE EXCLUDE REGEX ".*/.*/*aarch64*")
#endif()

add_executable(${PROJECT_NAME}-${PROJECT_VERSION} ${SOURCES_CORE} ${SOURCES_NET} ${SOURCES_CRYPTO} ${HEADERS_CRYPTO})

if (WITH_RANDOMX AND NOT WITH_HIP)
  target_link_libraries(${PROJECT_NAME}-${PROJECT_VERSION} randomx)
endif()

if (WITH_HIP)
  if (HIP_PLATFORM MATCHES amd)
   set(TNN_OUTPUT_NAME "tnn-miner-${INTERNAL_TNN_VERSION}_rocm")
   set_target_properties(${PROJECT_NAME}-${PROJECT_VERSION} PROPERTIES
    HIP_ARCHITECTURES "gfx900;gfx1010;gfx1030;gfx1100"   # For AMD GPUs
    HIP_PLATFORM "amd"
  )
  else ()
    set(TNN_OUTPUT_NAME "tnn-miner-${INTERNAL_TNN_VERSION}_cuda")
  endif ()
  
  # include(ExternalProject)
  # set(hip_DIR "${HIP_PATH}/cmake")
  # message(STATUS "hip_DIR = ${hip_DIR}")
  # include_directories(${hip_INCLUDE_DIRS})
  # link_directories(${hip_LIBRARY_DIRS})
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -ffast-math")

  # ExternalProject_Add(tnn_hip_project
  #   SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/tnn_hip
  #   CMAKE_ARGS -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_HIP_COMPILER_WORKS=1 -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
  #   BUILD_ALWAYS TRUE
  # )

  # add_dependencies(${PROJECT_NAME}-${PROJECT_VERSION} tnn_hip_project)
  # target_link_libraries(${PROJECT_NAME}-${PROJECT_VERSION} ${CMAKE_BINARY_DIR}/tnn_hip.lib)
  # target_link_libraries(${PROJECT_NAME}-${PROJECT_VERSION} ${CMAKE_BINARY_DIR}/libtnn_hip.a)
else() 
  set(TNN_OUTPUT_NAME "tnn-miner-${INTERNAL_TNN_VERSION}_cpu")
endif()

target_link_libraries(${PROJECT_NAME}-${PROJECT_VERSION} ${TNN_BOOST_LIBS})

set_target_properties(${PROJECT_NAME}-${PROJECT_VERSION} PROPERTIES OUTPUT_NAME ${TNN_OUTPUT_NAME})

if(WIN32)
  if (NOT WITH_HIP)
    target_link_libraries(${PROJECT_NAME}-${PROJECT_VERSION} mingw32)
  endif()
  target_link_libraries(${PROJECT_NAME}-${PROJECT_VERSION} wsock32 ws2_32 kernel32)
endif()

if(NOT APPLE AND NOT WIN32)
  target_link_libraries(${PROJECT_NAME}-${PROJECT_VERSION} udns)
endif()
if(USE_ASTRO_SPSA)
  target_link_libraries(${PROJECT_NAME}-${PROJECT_VERSION} ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}.a)
endif()
target_link_libraries(${PROJECT_NAME}-${PROJECT_VERSION} ${THREAD_LIB} OpenSSL::SSL OpenSSL::Crypto sodium ${BLAKE3_LIBRARIES})

set_target_properties(${PROJECT_NAME}-${PROJECT_VERSION} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)