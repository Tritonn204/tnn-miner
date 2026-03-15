# HIP support was added in CMake 3.21
cmake_minimum_required(VERSION 3.30)

#cmake_policy(SET CMP0156 NEW)

include(CMakePrintHelpers)
cmake_print_variables(CMAKE_HOST_SYSTEM_NAME)

# TODO: Rename WIN_CROSS to something more intelligent.  And maybe use it to specific the -march value?

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(TARGET_ARCH "amd64")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -Wuninitialized -DNDEBUG")

option(USE_GCC        "Use GCC instead of Clang" OFF)
option(USE_ASTRO_SPSA "Use Astro Stamped Permutation Suffix Array library" ON)
option(WIN_CROSS      "Whether we are cross-compiling on Linux for Windows" OFF)

# HIP_PLATFORM: "" (none), "amd", or "nvidia"
set(HIP_PLATFORM "${HIP_PLATFORM}" CACHE STRING "HIP platform: amd or nvidia")
set_property(CACHE HIP_PLATFORM PROPERTY STRINGS "" "amd" "nvidia")

# ----------------- Select C / C++ / HIP compilers -----------------
if (WIN32)

  # ---------- Windows + HIP (MSVC / clang-cl toolchain) ----------
  if (WITH_HIP)
    # Resolve HIP_PATH from CMake var or environment
      if (NOT DEFINED HIP_PATH AND DEFINED ENV{HIP_PATH})
        set(HIP_PATH "$ENV{HIP_PATH}" CACHE PATH "HIP root" FORCE)
      endif()

      if (NOT DEFINED HIP_PATH)
        message(FATAL_ERROR "WITH_HIP=ON on Windows but HIP_PATH is not set (CMake or ENV).")
      endif()

      if (HIP_PLATFORM STREQUAL "")
        set(HIP_PLATFORM "amd" CACHE STRING "HIP platform" FORCE)
      endif()

      # HIP CMake modules + ROCm libs
      file(GLOB _hip_modules "${HIP_PATH}/lib/cmake/*")
      list(APPEND CMAKE_MODULE_PATH ${_hip_modules})
      link_directories("${HIP_PATH}/lib")

      # NVIDIA HIP on Windows: use clang-cl for host code (MSVC ABI required by NVCC)
      if (HIP_PLATFORM MATCHES "nvidia")
        find_program(_CLANG_CL clang-cl PATHS "${HIP_PATH}/bin" NO_DEFAULT_PATH)
        if (NOT _CLANG_CL)
          find_program(_CLANG_CL clang-cl)
        endif()
        if (_CLANG_CL)
          set(CMAKE_C_COMPILER   "${_CLANG_CL}" CACHE FILEPATH "" FORCE)
          set(CMAKE_CXX_COMPILER "${_CLANG_CL}" CACHE FILEPATH "" FORCE)
          message(STATUS "NVIDIA HIP on Windows: using clang-cl at ${_CLANG_CL}")
        else()
          message(WARNING "clang-cl not found — NVIDIA HIP build may fail with MSVC-style flags")
        endif()

        # clang-cl needs compiler-rt builtins for FMV (__cpu_model) and
        # 128-bit integer ops (__udivti3/__umodti3) that MSVC doesn't provide
        file(GLOB _clang_rt_builtins "${HIP_PATH}/lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib")
        if (_clang_rt_builtins)
          list(GET _clang_rt_builtins -1 _clang_rt_builtins_lib)  # use newest
          link_libraries("${_clang_rt_builtins_lib}")
          message(STATUS "Linking clang_rt.builtins: ${_clang_rt_builtins_lib}")
        else()
          message(WARNING "clang_rt.builtins-x86_64.lib not found — link errors likely")
        endif()
      endif()

# ---------- Windows CPU-only ----------
  else()
    message(STATUS "Target system is Windows (CPU-only build)")

    if (USE_GCC)
      set(CMAKE_C_COMPILER   gcc   CACHE FILEPATH "" FORCE)
      set(CMAKE_CXX_COMPILER g++   CACHE FILEPATH "" FORCE)
      set(CMAKE_ASM_COMPILER gcc   CACHE FILEPATH "" FORCE)
    else()
      # MinGW/Clang flow
      set(CMAKE_C_COMPILER   clang   CACHE FILEPATH "" FORCE)
      set(CMAKE_CXX_COMPILER clang++ CACHE FILEPATH "" FORCE)
      set(CMAKE_ASM_COMPILER clang   CACHE FILEPATH "" FORCE)
    endif()

    # Use MinGW lld if available
    if (EXISTS "C:/mingw64/bin/ld.lld.exe")
      set(CMAKE_LINKER "C:/mingw64/bin/ld.lld.exe" CACHE FILEPATH "" FORCE)
    endif()
  endif()

# ----------------- Non-Windows platforms -----------------
else()
  if (USE_GCC)
    set(CMAKE_C_COMPILER   gcc   CACHE FILEPATH "" FORCE)
    set(CMAKE_CXX_COMPILER g++   CACHE FILEPATH "" FORCE)
    set(CMAKE_ASM_COMPILER gcc   CACHE FILEPATH "" FORCE)
  else()
    set(CMAKE_C_COMPILER   clang   CACHE FILEPATH "" FORCE)
    set(CMAKE_CXX_COMPILER clang++ CACHE FILEPATH "" FORCE)
    set(CMAKE_ASM_COMPILER clang   CACHE FILEPATH "" FORCE)
  endif()
endif()

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

option(WITH_LTO "Enable Link-Time Optimization" ON)

if (NOT WITH_HIP)
  if (WITH_LTO)
    message(STATUS "LTO enabled (WITH_LTO=ON)")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flto -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto -fomit-frame-pointer")
  else()
    message(STATUS "LTO disabled (pass -DWITH_LTO=ON to enable)")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer")
  endif()

else()
  set(USE_ASTRO_SPSA OFF)

  # ----- HIP on Windows: use clang-cl / MSVC-style toolchain -----
  if (WIN32)
    if (NOT CMAKE_GENERATOR MATCHES "Ninja")
      message(FATAL_ERROR
        "WITH_HIP=ON on Windows requires the Ninja generator.\n"
        "Current generator: '${CMAKE_GENERATOR}'.\n"
        "Run CMake like:\n"
        "  cmake -S . -B hip-build/win32 -G \"Ninja\" -DWITH_HIP=ON -DHIP_PATH=\"C:/Program Files/AMD/ROCm/6.x\""
      )
    endif()

    add_definitions(-DBOOST_ALL_NO_LIB)
    add_definitions(-D_WIN32_WINNT=0x0A00)
  endif()

  # ----- Common HIP CMake modules / library dirs (all platforms) -----
  if (DEFINED HIP_PATH)
    file(GLOB HIP_MODULES
      "${HIP_PATH}/lib/cmake/*"
    )
    list(APPEND CMAKE_MODULE_PATH ${HIP_MODULES})

    link_directories("${HIP_PATH}/lib")
  endif()

  set(HIP_LANG HIP)
endif()

if(APPLE)
  set(_LLVM_BIN "/opt/homebrew/opt/llvm@18/bin")

  find_program(_HB_LLVM_AR     NAMES llvm-ar     PATHS "${_LLVM_BIN}" NO_DEFAULT_PATH)
  find_program(_HB_LLVM_RANLIB NAMES llvm-ranlib PATHS "${_LLVM_BIN}" NO_DEFAULT_PATH)
  find_program(_HB_LLVM_NM     NAMES llvm-nm     PATHS "${_LLVM_BIN}" NO_DEFAULT_PATH)

  if(_HB_LLVM_AR AND _HB_LLVM_RANLIB)
    set(CMAKE_AR     "${_HB_LLVM_AR}"     CACHE FILEPATH "Archiver" FORCE)
    set(CMAKE_RANLIB "${_HB_LLVM_RANLIB}" CACHE FILEPATH "Ranlib"   FORCE)
  endif()

  if(_HB_LLVM_NM)
    set(CMAKE_NM "${_HB_LLVM_NM}" CACHE FILEPATH "NM" FORCE)
  endif()
endif()

project(
  Tnn-miner
  LANGUAGES C CXX ASM
  DESCRIPTION "Open Source Dero/Spectre/Xelis Miner"
)

if (WIN32 AND (MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
             AND WITH_HIP AND HIP_PLATFORM MATCHES "nvidia")))

  message(STATUS "MSVC WHY?... ${MSVC}, ${CMAKE_CXX_SIMULATE_ID}, ${HIP_PLATFORM}")

  add_compile_options(
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/EHa>
    $<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:/EHsc>
  )

  add_compile_options(
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:/MT>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/MT>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:C>>:/MTd>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/MTd>
  )

  if (NOT WITH_HIP)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedStatic$<$<CONFIG:Debug>:Debug>")
  else()
    unset(CMAKE_MSVC_RUNTIME_LIBRARY CACHE)
    unset(CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "")
  endif()
endif()

set(PROJECT_VERSION ${INTERNAL_TNN_VERSION})

if(NOT USE_GCC)
  set(CMAKE_LIB_SUFFIX "")
  set(CMAKE_DO_OPAQUE "")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    string(REGEX MATCH "^([0-9]+)" CXX_MAJOR_VERSION "${CMAKE_CXX_COMPILER_VERSION}")

    # Only try to (re)set if not already set by the Apple/Homebrew block
    if(NOT DEFINED CMAKE_AR OR CMAKE_AR STREQUAL "" OR CMAKE_AR MATCHES "XcodeDefault")
      # Prefer Homebrew location first, then fallback to PATH
      find_program(LLVM_AR_PROG
        NAMES llvm-ar-${CXX_MAJOR_VERSION} llvm-ar
        PATHS /opt/homebrew/opt/llvm@18/bin
      )
      if(LLVM_AR_PROG)
        set(CMAKE_AR "${LLVM_AR_PROG}" CACHE FILEPATH "Archiver" FORCE)
      endif()
    endif()

    if(NOT DEFINED CMAKE_RANLIB OR CMAKE_RANLIB STREQUAL "" OR CMAKE_RANLIB MATCHES "XcodeDefault")
      find_program(LLVM_RANLIB_PROG
        NAMES llvm-ranlib-${CXX_MAJOR_VERSION} llvm-ranlib
        PATHS /opt/homebrew/opt/llvm@18/bin
      )
      if(LLVM_RANLIB_PROG)
        set(CMAKE_RANLIB "${LLVM_RANLIB_PROG}" CACHE FILEPATH "Ranlib" FORCE)
      endif()
    endif()
  endif()

  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated -Wno-nan-infinity-disabled ${CMAKE_DO_OPAQUE}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated -Wno-nan-infinity-disabled ${CMAKE_DO_OPAQUE}")
else()
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated-declarations")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include (CheckIncludeFile)
include (cmake/cpu.cmake)
include (cmake/os.cmake)

cmake_print_variables($ENV{GITLAB_CI})
cmake_print_variables(INTERNAL_TNN_VERSION)
cmake_print_variables(CMAKE_SYSTEM_PROCESSOR)

# This CMAKE variable isn't defined until 'project' is called.
set(DO_STATIC " ")

# No static linking on macOS, ever
if (NOT CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
  # If this is NOT an MSVC-style compiler, we can use -static
  if (NOT (MSVC
           OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
               AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")))
    set(DO_STATIC " -static ")
  endif()
endif()

message(STATUS "DO_STATIC='${DO_STATIC}'")

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

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_STANDARD_REQUIRED TRUE)

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
    if (HIP_PLATFORM MATCHES "nvidia")
      add_compile_definitions(BOOST_HAS_THREADS=1)
    endif()

    if (HIP_PLATFORM MATCHES "nvidia")
      # NVIDIA clang-cl uses /MT (static CRT) — dynamic OpenSSL avoids CRT mismatch
      set(TNN_NEED_DYNAMIC_OPENSSL ON)
    else()
      # AMD HIP uses lld-link (MSVC linker) — static MSVC-ABI OpenSSL
      set(TNN_NEED_STATIC_OPENSSL ON)
    endif()
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
if(CMAKE_CXX_SIMULATE_ID)
  string(MAKE_C_IDENTIFIER "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_SIMULATE_ID}-${CMAKE_CXX_COMPILER_VERSION}" COMPILER_TAG)
else()
  string(MAKE_C_IDENTIFIER "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}" COMPILER_TAG)
endif()

if(UNIX)
  if(DEFINED ENV{XDG_CACHE_HOME})
    set(_cache_root "$ENV{XDG_CACHE_HOME}")
  else()
    set(_cache_root "$ENV{HOME}/.cache")
  endif()

  if (WITH_HIP)
    set(CPM_SOURCE_CACHE
        "${_cache_root}/Tnn-miner/CPM-hip-${CMAKE_SYSTEM_NAME}-${COMPILER_TAG}")
  else()
    set(CPM_SOURCE_CACHE
        "${_cache_root}/Tnn-miner/CPM-cpu-${CMAKE_SYSTEM_NAME}-${COMPILER_TAG}")
  endif()

else()
  if (WITH_HIP)
    set(CPM_SOURCE_CACHE
        "${PROJECT_SOURCE_DIR}/_deps/CPM-hip-${CMAKE_SYSTEM_NAME}-${COMPILER_TAG}")
  else()
    set(CPM_SOURCE_CACHE
        "${PROJECT_SOURCE_DIR}/_deps/CPM-cpu-${CMAKE_SYSTEM_NAME}-${COMPILER_TAG}")
  endif()
endif()
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

set(_OSSL_ROOT "C:/openssl_msvc")

if (TNN_NEED_DYNAMIC_OPENSSL)
  # NVIDIA HIP (clang-cl /MT) — dynamic OpenSSL to avoid CRT mismatch
  # Ship libcrypto-3-x64.dll + libssl-3-x64.dll alongside the executable
  if(NOT EXISTS "${_OSSL_ROOT}/lib/libssl.lib")
    message(FATAL_ERROR
      "OpenSSL MSVC not found at ${_OSSL_ROOT}. Run scripts/prereqs.bat first.")
  endif()
  message(STATUS "Using prebuilt OpenSSL (MSVC/dynamic) at ${_OSSL_ROOT}")

  add_library(OpenSSL::Crypto SHARED IMPORTED GLOBAL)
  set_target_properties(OpenSSL::Crypto PROPERTIES
    IMPORTED_IMPLIB "${_OSSL_ROOT}/lib/libcrypto.lib"
    IMPORTED_LOCATION "${_OSSL_ROOT}/bin/libcrypto-3-x64.dll"
  )

  add_library(OpenSSL::SSL SHARED IMPORTED GLOBAL)
  set_target_properties(OpenSSL::SSL PROPERTIES
    IMPORTED_IMPLIB "${_OSSL_ROOT}/lib/libssl.lib"
    IMPORTED_LOCATION "${_OSSL_ROOT}/bin/libssl-3-x64.dll"
  )
  target_link_libraries(OpenSSL::SSL INTERFACE OpenSSL::Crypto)

  include_directories("${_OSSL_ROOT}/include")
  link_directories("${_OSSL_ROOT}/lib")

elseif (TNN_NEED_STATIC_OPENSSL)
  # AMD HIP (lld-link) — static MSVC-ABI OpenSSL
  if(NOT EXISTS "${_OSSL_ROOT}/lib/libssl_static.lib")
    message(FATAL_ERROR
      "OpenSSL MSVC not found at ${_OSSL_ROOT}. Run scripts/prereqs.bat first.")
  endif()
  message(STATUS "Using prebuilt OpenSSL (MSVC/static) at ${_OSSL_ROOT}")

  add_library(OpenSSL::Crypto STATIC IMPORTED GLOBAL)
  set_target_properties(OpenSSL::Crypto PROPERTIES
    IMPORTED_LOCATION "${_OSSL_ROOT}/lib/libcrypto_static.lib"
  )

  add_library(OpenSSL::SSL STATIC IMPORTED GLOBAL)
  set_target_properties(OpenSSL::SSL PROPERTIES
    IMPORTED_LOCATION "${_OSSL_ROOT}/lib/libssl_static.lib"
  )
  target_link_libraries(OpenSSL::SSL INTERFACE OpenSSL::Crypto)

  include_directories("${_OSSL_ROOT}/include")
  link_directories("${_OSSL_ROOT}/lib")

else()
  if (WIN_CROSS)
    set(OPENSSL_ROOT "/src/clang64")
  endif()
  find_package(OpenSSL REQUIRED)
  if(OPENSSL_INCLUDE_DIR)
    include_directories(${OPENSSL_INCLUDE_DIR})
  endif()
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
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_NO_WARN_NEW_VERSIONS ON)

if(APPLE)
  set(USE_CPM_BOOST ON CACHE BOOL "" FORCE)
endif()

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
  set(TNN_BOOST_INCLUDES ${Boost_INCLUDE_DIRS})
  include_directories(${TNN_BOOST_INCLUDES})
  set(TNN_BOOST_LIBDIRS ${Boost_LIBRARY_DIRS})

  message(STATUS "Boost library directories: ${TNN_BOOST_LIBDIRS}")

  link_directories(${TNN_BOOST_LIBDIRS})
  message(${Boost_VERSION})
else()
  message("Boost not found. Using CPM to add it")

  set(BOOST_TARBALL "${PROJECT_SOURCE_DIR}/_deps/boost-1.86.0-cmake.tar.xz")
  if(NOT EXISTS "${BOOST_TARBALL}")
    message(STATUS "Downloading Boost archive to ${BOOST_TARBALL}")
    file(DOWNLOAD
        https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
        "${BOOST_TARBALL}"
        EXPECTED_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57)
    message(STATUS "Boost 1.86.0 has been downloaded successfully in the CPM cache directory!")
  else()
    message(STATUS "Using cached Boost archive at ${BOOST_TARBALL}")
  endif()

  # Only use MSVC compatibility for HIP builds on Windows
  # CPU builds use GNU/MinGW which is incompatible with -fms-compatibility
  if(WITH_HIP AND WIN32)
    set(BOOST_MS_COMPAT "CMAKE_CXX_FLAGS -fms-compatibility")
  else()
    set(BOOST_MS_COMPAT "")
  endif()

  CPMAddPackage(
    NAME tnn-prism
    GITHUB_REPOSITORY Tritonn204/tnn-prism
    GIT_TAG b029466
  )
  list(APPEND CMAKE_MODULE_PATH "${tnn-prism_SOURCE_DIR}/cmake")
  include(SimdTiers)
  include_directories("${tnn-prism_SOURCE_DIR}/include")

  CPMAddPackage(
    NAME Boost
    VERSION 1.86.0 # Versions less than 1.85.0 may need patches for installation targets.
    URL "${PROJECT_SOURCE_DIR}/_deps/boost-1.86.0-cmake.tar.xz"
    URL_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
    OPTIONS
      "BOOST_ENABLE_CMAKE ON"
      "BOOST_SKIP_INSTALL_RULES ON" # Set `OFF` for installation
      "BOOST_USE_STATIC_LIBS ON"
      "BOOST_USE_STATIC_RUNTIME ON"
      "BOOST_USE_MULTITHREADED ON"
      "BOOST_INSTALL_LAYOUT system"
      "BOOST_INCLUDE_LIBRARIES chrono\\\;context\\\;coroutine\\\;multiprecision\\\;json\\\;atomic\\\;program_options"
      "${BOOST_TOOLSET_OPTS}"  # Specify Clang toolset
      "${BOOST_MS_COMPAT}"
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

    file(GLOB_RECURSE BOOST_CMAKELISTS
      "${Boost_SOURCE_DIR}/CMakeLists.txt"
      "${Boost_SOURCE_DIR}/libs/*/CMakeLists.txt"
    )
  else()
    message(FATAL_ERROR "Boost was neither found nor added.")
  endif()
endif()

# This is universal regardless of how wheter we compiled or found Boost
#set(TNN_BOOST_LIBS Boost::chrono Boost::context Boost::coroutine Boost::thread Boost::json Boost::atomic Boost::program_options Boost::asio)
set(TNN_BOOST_LIBS Boost::atomic Boost::context Boost::chrono Boost::json Boost::program_options) #@Boost::beast Boost::bind Boost::json Boost::chrono Boost::atomic Boost::program_options )

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
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin" OR USE_GCC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -fuse-ld=ld")
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
  if (NOT WITH_HIP OR HIP_PLATFORM MATCHES "amd")
    find_program(LLD_PROG NAMES ld.lld-${CXX_MAJOR_VERSION} ld.lld lld)
    if (LLD_PROG)
      get_filename_component(LLD_DIR "${LLD_PROG}" DIRECTORY)
      message(STATUS "Using lld at ${LLD_PROG}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v -B${LLD_DIR} -fuse-ld=lld -Wl,--sort-section=name")
    else()
      message(STATUS "lld not found, falling back to default linker")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v")
    endif()
  else()
    # HIP + NVIDIA
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v")
  endif()
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
  if (NOT WITH_HIP)
    # CPU builds: use MinGW's linker
    set(CMAKE_LINKER "C:/mingw64/bin/ld.lld.exe")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
    if (WITH_LTO)
      # LLD 21 LTO backend needs explicit +aes for FMV ifunc resolvers (Clang 20 bitcode compat)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-plugin-opt=-mattr=+aes")
    endif()
  endif()
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v")
endif()
cmake_print_variables(CMAKE_EXE_LINKER_FLAGS)

# if(WITH_HIP)
# CPMAddPackage(
#   NAME sodium
#   GITHUB_REPOSITORY robinlinden/libsodium-cmake
#   GIT_TAG f568ff02f1bed155ea598c0e803ef3c9db2703d2
#   OPTIONS 
#     "CMAKE_C_FLAGS_RELEASE -msse3 -mssse3 -maes -mpclmul -mavx2" 
#     "CMAKE_CXX_FLAGS_RELEASE -msse3 -mssse3 -maes -mpclmul -mavx2"
# )
# endif()

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
  "${PROJECT_SOURCE_DIR}/src/crypto/sha3/*.c"
  "${PROJECT_SOURCE_DIR}/src/broadcast/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/coins/dummy_miners.cpp"
  "${PROJECT_SOURCE_DIR}/src/util/*.c"
  "${PROJECT_SOURCE_DIR}/src/util/*.cpp"
)

if (NOT WITH_HIP)
  file(GLOB SOURCES_CORE_CRYPTO_SALSA
       "${PROJECT_SOURCE_DIR}/src/crypto/salsa-simd/*.c")

  file(GLOB SOURCES_CORE_ARGON2
       "${PROJECT_SOURCE_DIR}/src/crypto/argon2/*.c"
       "${PROJECT_SOURCE_DIR}/src/crypto/argon2/*.cpp")

  list(APPEND SOURCES_CORE
       ${SOURCES_CORE_CRYPTO_SALSA}
       ${SOURCES_CORE_ARGON2})
endif()

file(GLOB SOURCES_NET
  "${PROJECT_SOURCE_DIR}/src/net/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/net/*.c"
  "${PROJECT_SOURCE_DIR}/src/net/kas/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/net/xelis/*.cpp"
)

include(cmake/spsa.cmake)
include(cmake/xelis-hash.cmake)
include(cmake/astrix-hash.cmake)
include(cmake/nxl-hash.cmake)
include(cmake/wala-hash.cmake)
include(cmake/chacha20.cmake)

if (NOT WITH_HIP)
  include(cmake/astrobwtv3.cmake)
  include(cmake/randomx.cmake)
  include(cmake/verus-hash.cmake)
  include(cmake/hoohash.cmake)
  include(cmake/shai-hive.cmake)
  include(cmake/yespower.cmake)
  include(cmake/rinhash.cmake)

  include(cmake/blake2.cmake)
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

if (WITH_RANDOMX)
  include(cmake/winring0.cmake)
endif()

include(cmake/setup.cmake)

if (WITH_HIP)
  # STAGE 1: Create static library with all regular C++ code and dependencies using normal linking
  set(TNN_CORE_LIB "tnn-core-lib")
  add_library(${TNN_CORE_LIB} STATIC
    ${SOURCES_CORE}
    ${SOURCES_NET}
    ${SOURCES_CRYPTO}
    ${HEADERS_CRYPTO}
  )

  # Apply normal C++ linking to the static library (no HIP linking issues here)
  setup_target_libraries(${TNN_CORE_LIB})

  #
  # Windows + NVIDIA HIP: manual hipcc compilation of HIP sources
  #
  find_package(hip REQUIRED)

  # if (WIN32 AND HIP_PLATFORM MATCHES "nvidia")
  if (FALSE)
    message(STATUS "Configuring HIP (NVIDIA backend) on Windows using manual hipcc compilation")

    # Path to hipcc – can be overridden on the CMake command line
    if(WIN32)
      set(HIPCC_EXT ".exe")
    endif()

    if (NOT DEFINED HIPCC_EXECUTABLE)
      set(HIPCC_EXECUTABLE "$ENV{HIP_PATH}/bin/hipcc${HIPCC_EXT}")
    endif()

    if (NOT EXISTS "${HIPCC_EXECUTABLE}")
      message(FATAL_ERROR "HIPCC_EXECUTABLE not found: ${HIPCC_EXECUTABLE}")
    endif()

    # Output name for CUDA backend via HIP
    set(TNN_OUTPUT_NAME "tnn-miner-cuda")

    # Adjust architectures as desired

    set(TNN_HIP_OBJS "")

    set(HIPCC_INCLUDES_RSP
        "${CMAKE_CURRENT_BINARY_DIR}/hipcc_includes_$<CONFIG>.rsp")

    get_target_property(TNN_CORE_INCS ${TNN_CORE_LIB} INCLUDE_DIRECTORIES)

    set(HIPCC_INCLUDES_RSP "${CMAKE_CURRENT_BINARY_DIR}/hipcc_includes_$<CONFIG>.rsp")

    set(HIPCC_INCLUDES_CONTENT "")
    foreach(inc IN LISTS TNN_CORE_INCS)
      string(APPEND HIPCC_INCLUDES_CONTENT "-I\"${inc}\"\n")
    endforeach()

    file(GENERATE
      OUTPUT  "${HIPCC_INCLUDES_RSP}"
      CONTENT "${HIPCC_INCLUDES_CONTENT}"
    )

    foreach(src ${TNN_HIP_SOURCES})
      get_filename_component(src_name "${src}" NAME_WE)
      set(obj "${CMAKE_CURRENT_BINARY_DIR}/${src_name}.obj")

      set(HIPCC_ARCH_FLAGS
        "-gencode=arch=compute_90,code=sm_90"
      )

      add_custom_command(
        OUTPUT ${obj}
        COMMAND ${HIPCC_EXECUTABLE}
                -std=c++20
                -O3
                -x cu
                ${HIPCC_ARCH_FLAGS}
                -I"${HIP_PATH}/include"
                --options-file="${HIPCC_INCLUDES_RSP}"
                -c "${src}" -o "${obj}"
        DEPENDS "${src}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
      )

      list(APPEND TNN_HIP_OBJS "${obj}")
    endforeach()

    add_executable(${TNN_OUTPUT_NAME}
      ${TNN_HIP_OBJS}
    )

    if (NOT DEFINED ENV{CUDA_PATH})
      message(FATAL_ERROR
        "===============================================\n"
        " CUDA backend requires CUDA_PATH to be defined.\n"
        " On Windows this is normally set by the installer:\n"
        "   setx CUDA_PATH \"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.X\"\n"
        "==============================================="
      )
    endif()

    # Link with core C++ library and HIP runtime
    target_link_libraries(${TNN_OUTPUT_NAME}
      PRIVATE
        ${TNN_CORE_LIB}
        "$ENV{CUDA_PATH}/lib/x64/cudart_static.lib"
        "$ENV{CUDA_PATH}/lib/x64/cuda.lib"
        cuda hiprtc nvrtc
    )

  #
  # All other HIP platforms: use proper HIP language integration
  #
  else()
    if (HIP_PLATFORM MATCHES "amd")
      set(CMAKE_HIP_ARCHITECTURES
        "gfx1100"
      )
      set(HIPRTC_LIB hiprtc)
      set(TNN_OUTPUT_NAME "tnn-miner-rocm")
      set(HIPRT_LIB hip::host)
    else()
      if(WIN32)
        if (NOT DEFINED ENV{CUDA_PATH})
          message(FATAL_ERROR
            "HIP_PLATFORM=nvidia requires CUDA_PATH environment variable "
            "pointing to the CUDA Toolkit root (e.g. C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1)."
          )
        endif()

        set(CMAKE_HIP_ARCHITECTURES "90")
        set(TNN_OUTPUT_NAME "tnn-miner-cuda")

        set(CUDA_LIB_DIR "$ENV{CUDA_PATH}/lib/x64")

        set(HIPRT_LIB
          "${CUDA_LIB_DIR}/cudart_static.lib"
          "${CUDA_LIB_DIR}/cudadevrt.lib"
          "${CUDA_LIB_DIR}/cuda.lib"
        )

        set(HIPRTC_LIB 
          "${CUDA_LIB_DIR}/nvrtc.lib" 
        )
      else()
        set(CMAKE_HIP_ARCHITECTURES "90")
        set(TNN_OUTPUT_NAME "tnn-miner-cuda")

        set(HIPRT_LIB  cudart_static cudadevrt cuda)
        set(HIPRTC_LIB nvrtc)
      endif()
    endif()

    set(HIP_LANG HIP)
    set(CMAKE_HIP_COMPILER_WORKS 1)

    enable_language(${HIP_LANG})

    add_executable(${TNN_OUTPUT_NAME}
      ${TNN_HIP_SOURCES}
    )

    # Per-platform stats API port: CPU=8989, AMD=8990, NVIDIA=8991, Intel=8992
    if (HIP_PLATFORM MATCHES "amd")
      target_compile_definitions(${TNN_OUTPUT_NAME} PRIVATE TNN_BROADCAST_PORT=8990)
    elseif (HIP_PLATFORM MATCHES "nvidia")
      target_compile_definitions(${TNN_OUTPUT_NAME} PRIVATE TNN_BROADCAST_PORT=8991)
    elseif (HIP_PLATFORM MATCHES "intel")
      target_compile_definitions(${TNN_OUTPUT_NAME} PRIVATE TNN_BROADCAST_PORT=8992)
    endif()

    if (WIN32 AND HIP_PLATFORM MATCHES "nvidia")
      set(CMAKE_HIP_LINK_EXECUTABLE
        "<CMAKE_LINKER> /machine:x64 /INCREMENTAL:NO <LINK_FLAGS> <OBJECTS> /OUT:<TARGET> <LINK_LIBRARIES>"
      )
    endif()

    # Collect HIP (and optionally CUDA) include dirs in one variable
    set(_tnn_hip_includes "")

    # HIP includes (needed on both AMD and NVIDIA HIP backends)
    if(DEFINED ENV{HIP_PATH})
      list(APPEND _tnn_hip_includes "$ENV{HIP_PATH}/include")
    else()
      message(WARNING "HIP_PATH is not set; HIP headers may not be found.")
    endif()

    # On NVIDIA backend, also add CUDA includes
    if(HIP_PLATFORM MATCHES "nvidia")
      if(DEFINED ENV{CUDA_PATH})
        list(APPEND _tnn_hip_includes "$ENV{CUDA_PATH}/include")
      else()
        message(FATAL_ERROR
          "HIP_PLATFORM=nvidia but CUDA_PATH is not set. "
          "CUDA headers are required for the NVIDIA HIP backend."
        )
      endif()
      set(HIPRTC_DLL_DIR "${CMAKE_SOURCE_DIR}/lib/nvrtc")
    else()
      set(HIPRTC_DLL_DIR "${CMAKE_SOURCE_DIR}/lib/hiprtc")
    endif()

    # Single include directive, works for both AMD and NVIDIA
    target_include_directories(${TNN_CORE_LIB}
      PRIVATE
        ${_tnn_hip_includes}
    )

    target_include_directories(${TNN_OUTPUT_NAME}
      PRIVATE
        ${_tnn_hip_includes}
    )

    if (WIN32)
      target_link_libraries(${TNN_OUTPUT_NAME}
        PRIVATE
          ${TNN_CORE_LIB}
          ${HIPRT_LIB}
          ${HIPRTC_LIB}
          user32.lib
      )
    else()
      target_link_libraries(${TNN_OUTPUT_NAME}
        PRIVATE
          ${TNN_CORE_LIB}
          ${HIPRT_LIB}
          ${HIPRTC_LIB}
          -Wl,-Bstatic udns numa -Wl,-Bdynamic
      )
      # Bake RPATH so the binary finds bundled NVRTC .so files
      if (HIP_PLATFORM MATCHES "nvidia")
        set_target_properties(${TNN_OUTPUT_NAME} PROPERTIES
          BUILD_RPATH_USE_ORIGIN TRUE
          INSTALL_RPATH "$ORIGIN/nvrtc_libs"
          BUILD_RPATH "$ORIGIN/nvrtc_libs"
        )
      endif()
    endif()

    # Global HIP flags
    if (HIP_PLATFORM MATCHES "amd")
      # ROCm/clang backend
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -ffast-math")
    elseif (HIP_PLATFORM MATCHES "nvidia")
      # CUDA/nvcc backend
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 --use_fast_math")
    else()
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
    endif()
  endif()

  if (WIN32)
    add_custom_command(
      TARGET ${TNN_OUTPUT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${HIPRTC_DLL_DIR}"
        "$<TARGET_FILE_DIR:${TNN_OUTPUT_NAME}>"
    )
    if (TNN_NEED_DYNAMIC_OPENSSL)
      add_custom_command(
        TARGET ${TNN_OUTPUT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_OSSL_ROOT}/bin/libcrypto-3-x64.dll"
          "${_OSSL_ROOT}/bin/libssl-3-x64.dll"
          "$<TARGET_FILE_DIR:${TNN_OUTPUT_NAME}>"
      )
    endif()
  endif()

else()
  # CPU-only build path
  set(TNN_OUTPUT_NAME "tnn-miner-cpu")
  add_executable(${TNN_OUTPUT_NAME}
    ${SOURCES_CORE}
    ${SOURCES_NET}
    ${SOURCES_CRYPTO}
    ${HEADERS_CRYPTO}
  )

  setup_target_libraries(${TNN_OUTPUT_NAME})

  if (WITH_RANDOMX)
    target_link_libraries(${TNN_OUTPUT_NAME} randomx)
  endif()

  # Set output name
  set_target_properties(${TNN_OUTPUT_NAME} PROPERTIES OUTPUT_NAME ${TNN_OUTPUT_NAME})
endif()

# Copy default-tunes → tunes next to the binary
add_custom_command(
  TARGET ${TNN_OUTPUT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/default-tunes"
    "$<TARGET_FILE_DIR:${TNN_OUTPUT_NAME}>/tunes"
)

unset(HIP_LANG CACHE)
