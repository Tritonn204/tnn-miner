# HIP support was added in CMake 3.21
cmake_minimum_required(VERSION 3.18)

#cmake_policy(SET CMP0156 NEW)

include(CMakePrintHelpers)
cmake_print_variables(CMAKE_HOST_SYSTEM_NAME)

# TODO: Rename WIN_CROSS to something more intelligent.  And maybe use it to specific the -march value?

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
  message(STATUS "Target system is Windows!")
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
  set(BOOST_TOOLSET_OPTS "BOOST_TOOLSET clang")
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -Wno-unused-command-line-argument")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-unused-command-line-argument")
endif()

set(TARGET_ARCH "amd64")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -Ofast -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto  -Wuninitialized -DNDEBUG")

option(USE_GCC        "Use GCC instead of Clang" OFF)
option(USE_ASTRO_SPSA "Use Astro Stamped Permutation Suffix Array library" ON)
option(WIN_CROSS      "Whether we are cross-compiling on Linux for Windows" OFF)

# if (WIN32)
#   if(WITH_HIP)
#     message(STATUS "HIP compilation is not currently support for TNN Miner on Windows.")
#   endif()
#   set(WITH_HIP OFF)
# endif()

if(USE_GCC)
  if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER gcc)
  endif()
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++)
  endif()
else()
  if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER clang)
  endif()
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
  endif()
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
option(WITH_VERUSHASH        "Enable VerusHash algorithms family" OFF)
option(WITH_ASTRIXHASH        "Enable AstrixHash algorithms family" ON)
option(WITH_NXLHASH        "Enable NxlHash algorithms family" ON)
option(WITH_HOOHASH        "Enable Hoohash algorithms family" ON)
option(WITH_WALAHASH        "Enable Walahash algorithms family" ON)
option(WITH_SHAIHIVE        "Enable ShaiHive algorithms family" ON)
option(WITH_YESPOWER        "Enable YesPower algorithms family" ON)
option(WITH_RINHASH        "Enable RinHash algorithms family" ON)
# option(WITH_ARGON2          "Enable Argon2 algorithms family" ON)
# option(WITH_KAWPOW          "Enable KawPow algorithms family" ON)
# option(WITH_GHOSTRIDER      "Enable GhostRider algorithm" ON)
# option(WITH_EMBEDDED_CONFIG "Enable internal embedded JSON config" OFF)
# option(WITH_OPENCL          "Enable OpenCL backend" ON)
set(WITH_OPENCL_VERSION 200 CACHE STRING "Target OpenCL version")
set_property(CACHE WITH_OPENCL_VERSION PROPERTY STRINGS 120 200 210 220)
option(WITH_HIP             "Enable GPU support through the HIP SDK" OFF)
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
  set(USE_ASTRO_SPSA OFF)
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
  LANGUAGES C CXX
  DESCRIPTION "Open Source Dero/Spectre/Xelis Miner"
)

set(PROJECT_VERSION ${INTERNAL_TNN_VERSION})

if(USE_GCC)
  if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER gcc)
  endif()
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++)
  endif()

  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated-declarations")
else()
  if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER clang)
  endif()
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
  endif()

  # TODO: Rename this to SPSA_LIB_SUFFIX
  set(CMAKE_LIB_SUFFIX "")
  set(CMAKE_DO_OPAQUE "")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    string(REGEX MATCH "^([0-9]+)" CXX_MAJOR_VERSION "${CMAKE_CXX_COMPILER_VERSION}") 
    set(CMAKE_LIB_SUFFIX "_clang_${CXX_MAJOR_VERSION}")
    if(NOT CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
      if(APPLE AND EXISTS "/opt/homebrew/opt/llvm@${CXX_MAJOR_VERSION}/bin/llvm-ar")
        set(CMAKE_AR "/opt/homebrew/opt/llvm@${CXX_MAJOR_VERSION}/bin/llvm-ar")
        set(CMAKE_RANLIB "/opt/homebrew/opt/llvm@${CXX_MAJOR_VERSION}/bin/llvm-ranlib")
      else()
        set(CMAKE_AR "llvm-ar-${CXX_MAJOR_VERSION}")
        set(CMAKE_RANLIB "llvm-ranlib-${CXX_MAJOR_VERSION}")
      endif()
    endif()
  endif()

  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated -Wno-nan-infinity-disabled ${CMAKE_DO_OPAQUE}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated -Wno-nan-infinity-disabled ${CMAKE_DO_OPAQUE}")
endif()

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


if(DEFINED ENV{CPU_ARCHTARGET})
  set(CPU_ARCHTARGET "$ENV{CPU_ARCHTARGET}")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  set(WITH_HIP OFF)
  set(TARGET_ARCH "aarch64")
  if (NOT CPU_ARCHTARGET)
    set(CPU_ARCHTARGET "armv8-a+aes")
  endif()
  # AES needed for Xelisv2
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flax-vector-conversions -march=${CPU_ARCHTARGET}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flax-vector-conversions -march=${CPU_ARCHTARGET}")
else()
  set(TARGET_ARCH "amd64")
  if (WITH_HIP)
    set(CPU_ARCHTARGET "x86-64")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mtune=generic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=generic")
  endif()
  if (NOT CPU_ARCHTARGET)
    set(CPU_ARCHTARGET "x86-64-v3")
  endif()
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=${CPU_ARCHTARGET} -m64")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=${CPU_ARCHTARGET} -m64")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D\"CPU_ARCHTARGET=${CPU_ARCHTARGET}\"")
#if(WIN_CROSS)
#  set(WIN_CROSS_OPTS "${WIN_CROSS_OPTS} --target=x86_64-pc-windows-msvc")
#  set(WIN_CROSS_OPTS "${WIN_CROSS_OPTS} \
#  -I/src/clang64/include \
#  -I/src/winsdk/crt/include \
#  -I/src/winsdk/sdk/Include/ucrt \
#  -I/src/winsdk/sdk/Include/shared \
#  -I/src/winsdk/sdk/Include/um")
#endif()
#set(CMAKE_LIB_SUFFIX "${CMAKE_LIB_SUFFIX}_${CPU_ARCHTARGET}")

if(TARGET_ARCH MATCHES "amd64")
  message(STATUS "Target architecture is amd64: enabling -msha")
  
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:-msha>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-msha>")
endif()

set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ${DO_STATIC} -DTNN_VERSION=${INTERNAL_TNN_VERSION}") # ${WIN_CROSS_OPTS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${DO_STATIC} -DTNN_VERSION=${INTERNAL_TNN_VERSION}") # ${WIN_CROSS_OPTS}")

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

#set (CMAKE_CXX_STANDARD 20)
#set (CMAKE_STANDARD_REQUIRED TRUE)

enable_language(ASM)

set(OPENSSL_USE_STATIC_LIBS TRUE)

if(WIN32)
  message(STATUS "WIN32!")
  if(NOT WITH_HIP)
    set(OPENSSL_ROOT_DIR "C:/openssl/clang64")
    set(SODIUM_ROOT_DIR "C:/sodium/clang64")
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
    set_target_properties(OpenSSL::Crypto PROPERTIES IMPORTED_LOCATION "${OPENSSL_ROOT_DIR}/lib/libcrypto_static.lib")
  
    add_library(OpenSSL::SSL UNKNOWN IMPORTED)
    set_target_properties(OpenSSL::SSL PROPERTIES IMPORTED_LOCATION "${OPENSSL_ROOT_DIR}/lib/libssl_static.lib")
  endif()
  include_directories(
    ${OPENSSL_ROOT_DIR}/include
    ${SODIUM_ROOT_DIR}/include
  )
  link_directories(
    ${OPENSSL_ROOT_DIR}/lib
    ${SODIUM_ROOT_DIR}/lib
  )
endif()

# This stores downloaded dependencies so we do not need to re-download them
set(CPM_SOURCE_CACHE "${PROJECT_SOURCE_DIR}/_deps/CPM")
cmake_print_variables(CPM_SOURCE_CACHE)

set(CPM_DOWNLOAD_VERSION 0.40.8)

set(CPM_DOWNLOAD_LOCATION "${PROJECT_SOURCE_DIR}/_deps/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

cmake_print_variables(CPM_DOWNLOAD_LOCATION)
include(${CPM_DOWNLOAD_LOCATION})

#set(OPENSSL_USE_STATIC_LIBS TRUE)

if (WIN_CROSS)
  set(OPENSSL_ROOT "/src/clang64")
endif()
if (NOT WIN32 OR NOT WITH_HIP)
  find_package(OpenSSL REQUIRED)
endif()

if (UNIX)
  set(THREAD_LIB pthread)
elseif(NOT WITH_HIP)
  set(THREAD_LIB winpthread)
endif()

if (WIN_CROSS)
  set(THREAD_LIB winpthread)
endif()
cmake_print_variables(THREAD_LIB)

if (WIN_CROSS)
  set(Boost_ROOT "${PROJECT_SOURCE_DIR}/export/boost-x64")
  set(COMMON_FLAGS "-DBOOST_HAS_THREADS=1 -Wno-nonportable-include-path -Wno-pragma-pack -Wno-ignored-attributes -Wno-ignored-pragma-intrinsic -Wno-comment -Wno-microsoft-anon-tag")
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ${COMMON_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMMON_FLAGS}")
  #set(BOOST_HAS_THREADS "1")
  #set(BOOST_THREADAPI "win32")
endif()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_NO_WARN_NEW_VERSIONS ON)

if (NOT USE_CPM_BOOST)
find_package(Boost 1.80 COMPONENTS
  chrono
  context
  thread
  json
  atomic
  program_options
)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads REQUIRED)

if (WIN_CROSS AND NOT Boost_FOUND)
  message(FATAL_ERROR "Boost was not found: ${Boost_ROOT}")
endif()

if(Boost_FOUND AND NOT USE_CPM_BOOST)
  message("OS-provided Boost found")
  set(TNN_BOOST_INCLUDES ${BOOST_INCLUDE_DIRS})
  include_directories(${TNN_BOOST_INCLUDES})
  set(TNN_BOOST_LIBDIRS ${BOOST_LIBRARY_DIRS})

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

  file(DOWNLOAD https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
       "${PROJECT_SOURCE_DIR}/_deps/boost-1.86.0-cmake.tar.xz"
       EXPECTED_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57)
  if(WIN32)
    set(CMAKE_ASM_FLAGS -Wno-unused-command-line-argument) # adding this flag as an example
  endif()
  CPMAddPackage(
    NAME Boost
    VERSION 1.86.0 # Versions less than 1.85.0 may need patches for installation targets.
    URL "${PROJECT_SOURCE_DIR}/_deps/boost-1.86.0-cmake.tar.xz"
    URL_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
    OPTIONS 
      "BOOST_ENABLE_CMAKE ON"
      "BOOST_SKIP_INSTALL_RULES ON" # Set `OFF` for installation
      "BOOST_USE_STATIC_LIBS ON"
      "BOOST_INSTALL_LAYOUT system"
      "BOOST_INCLUDE_LIBRARIES chrono\\\;context\\\;coroutine\\\;multiprecision\\\;json\\\;thread\\\;atomic\\\;program_options"
      "${BOOST_TOOLSET_OPTS}"  # Specify Clang toolset
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
#set(TNN_BOOST_LIBS Boost::chrono Boost::context Boost::coroutine Boost::thread Boost::json Boost::atomic Boost::program_options Boost::asio)
set(TNN_BOOST_LIBS Boost::atomic Boost::context Boost::chrono Boost::json Boost::program_options Boost::thread) #@Boost::beast Boost::bind Boost::json Boost::chrono Boost::thread Boost::atomic Boost::program_options )

set(WINSDK_DIR "${PROJECT_SOURCE_DIR}/wincross/winsdk")
if(WIN_CROSS)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target x86_64-pc-windows-msvc")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target x86_64-pc-windows-msvc")
  set(CMAKE_SKIP_RPATH TRUE)
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=ld.lld-${CXX_MAJOR_VERSION} -target x86_64-pc-windows-msvc") # -Wl,-machine:x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld-link-${CXX_MAJOR_VERSION} -v -target x86_64-pc-windows-msvc") # -Wl,-machine:x64")
  #link_directories("${WINSDK_DIR}/crt/lib/x64" "${WINSDK_DIR}/sdk/lib/ucrt/x64" "${WINSDK_DIR}/sdk/lib/um/x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${WINSDK_DIR}/../clang64/lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${WINSDK_DIR}/crt/lib/x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${WINSDK_DIR}/sdk/lib/ucrt/x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${WINSDK_DIR}/sdk/lib/um/x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${TNN_BOOST_LIBDIRS}")
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xlinker /force")
  set(LINK_OPTIONS "${CMAKE_EXE_LINKER_FLAGS}")
elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin" OR USE_GCC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -fuse-ld=ld")
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v -fuse-ld=lld-${CXX_MAJOR_VERSION}")
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=C:/mingw64/bin/ld.lld.exe")
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld-link")
endif()
cmake_print_variables(CMAKE_EXE_LINKER_FLAGS)

if(WITH_HIP)
CPMAddPackage(
  NAME sodium
  GITHUB_REPOSITORY robinlinden/libsodium-cmake
  GIT_TAG f568ff02f1bed155ea598c0e803ef3c9db2703d2
  OPTIONS 
    "CMAKE_C_FLAGS_RELEASE -msse3 -mssse3 -maes -mpclmul -mavx2" 
    "CMAKE_CXX_FLAGS_RELEASE -msse3 -mssse3 -maes -mpclmul -mavx2"
)
endif()

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
if(WIN_CROSS)
  include_directories(
    "${WINSDK_DIR}/../clang64/include"
    "${WINSDK_DIR}/crt/include"
    "${WINSDK_DIR}/sdk/Include/ucrt"
    "${WINSDK_DIR}/sdk/Include/shared"
    "${WINSDK_DIR}/sdk/Include/um"
  )
endif()

file(GLOB_RECURSE SOURCES_CORE
  "${PROJECT_SOURCE_DIR}/include/libkeccak/*.c"
  "${PROJECT_SOURCE_DIR}/src/core/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/core/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/cshake/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/skein/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/salsa-simd/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/sha3/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/argon2/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/argon2/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/broadcast/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/coins/dummy_miners.cpp"
  "${PROJECT_SOURCE_DIR}/src/util/*.c"
  "${PROJECT_SOURCE_DIR}/src/util/*.cpp"
)

file(GLOB SOURCES_NET
  "${PROJECT_SOURCE_DIR}/src/net/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/net/*.c"
  "${PROJECT_SOURCE_DIR}/src/net/kas/*.cpp"
)

include(cmake/spsa.cmake)

if (NOT WITH_HIP)
  include(cmake/astrobwtv3.cmake)
  include(cmake/xelis-hash.cmake)
  include(cmake/randomx.cmake)
  include(cmake/verus-hash.cmake)
  include(cmake/astrix-hash.cmake)
  include(cmake/nxl-hash.cmake)
  include(cmake/hoohash.cmake)
  include(cmake/wala-hash.cmake)
  include(cmake/shai-hive.cmake)
  include(cmake/yespower.cmake)
  include(cmake/rinhash.cmake)

  include(cmake/blake2.cmake)
  include(cmake/chacha20.cmake)
else()
  include(cmake/astrix-hash.cmake)
  include(cmake/nxl-hash.cmake)
  include(cmake/wala-hash.cmake)
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

if (WITH_RANDOMX)
  include(cmake/winring0.cmake)
endif()

include(cmake/setup.cmake)

if (WITH_HIP)
  enable_language(${HIP_LANG})
  if (HIP_PLATFORM MATCHES amd)
    set(CMAKE_HIP_ARCHITECTURES 
    "gfx900;gfx902;gfx904;gfx906;"
    "gfx1010;gfx1011;gfx1012;gfx1013;"
    "gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;"
    "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151"
    )

    set(TNN_OUTPUT_NAME "tnn-miner-rocm")
    add_executable(${TNN_OUTPUT_NAME} ${SOURCES_CORE} ${SOURCES_NET} ${SOURCES_CRYPTO} ${HEADERS_CRYPTO})
    setup_target_libraries(${TNN_OUTPUT_NAME})
  else ()
    set(TNN_OUTPUT_NAME "tnn-miner-cuda")
    set(CMAKE_HIP_ARCHITECTURES "61;70;75;80;89;90")
    add_executable(${TNN_OUTPUT_NAME} ${SOURCES_CORE} ${SOURCES_NET} ${SOURCES_CRYPTO} ${HEADERS_CRYPTO})
    setup_target_libraries(${TNN_OUTPUT_NAME})
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
  # set(TNN_OUTPUT_NAME "tnn-miner-${INTERNAL_TNN_VERSION}_cpu")
  set(TNN_OUTPUT_NAME "tnn-miner-cpu")
  add_executable(${TNN_OUTPUT_NAME} ${SOURCES_CORE} ${SOURCES_NET} ${SOURCES_CRYPTO} ${HEADERS_CRYPTO})

  setup_target_libraries(${TNN_OUTPUT_NAME})

  if (WITH_RANDOMX)
    target_link_libraries(${TNN_OUTPUT_NAME} randomx)
  endif()

  # Set output name
  set_target_properties(${TNN_OUTPUT_NAME} PROPERTIES OUTPUT_NAME ${TNN_OUTPUT_NAME})

endif()

unset(HIP_LANG CACHE)
