if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(TNN_64_BIT ON)
    add_definitions(-DTNN_64_BIT)
else()
    set(TNN_64_BIT OFF)
endif()

if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(WARNING "CMAKE_SYSTEM_PROCESSOR not defined")
endif()

include(CheckCXXCompilerFlag)

if (CMAKE_CXX_COMPILER_ID MATCHES MSVC 
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(VAES_SUPPORTED ON)
else()
    CHECK_CXX_COMPILER_FLAG("-mavx2 -mvaes" VAES_SUPPORTED)
endif()

if (NOT VAES_SUPPORTED)
    set(WITH_VAES OFF)
endif()

if (TNN_64_BIT AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    add_definitions(-DRAPIDJSON_SSE2)
else()
    set(WITH_SSE4_1 OFF)
    set(WITH_AVX2 OFF)
    set(WITH_VAES OFF)
endif()

if (ARM_V8)
    set(ARM_TARGET 8)
elseif (ARM_V7)
    set(ARM_TARGET 7)
endif()

if (NOT ARM_TARGET)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|armv8-a)$")
        set(ARM_TARGET 8)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armv7f|armv7s|armv7k|armv7-a|armv7l|armv7ve)$")
        set(ARM_TARGET 7)
    endif()
endif()

if (ARM_TARGET AND ARM_TARGET GREATER 6)
    set(TNN_ARM ON)
    add_definitions(-DTNN_ARM=${ARM_TARGET})

    message(STATUS "Use ARM_TARGET=${ARM_TARGET} (${CMAKE_SYSTEM_PROCESSOR})")

    if (ARM_TARGET EQUAL 8)
        CHECK_CXX_COMPILER_FLAG(-march=armv8-a+crypto TNN_ARM_CRYPTO)

        if (TNN_ARM_CRYPTO)
            add_definitions(-DTNN_ARM_CRYPTO)
            set(ARM8_CXX_FLAGS "-march=armv8-a+crypto")
        else()
            set(ARM8_CXX_FLAGS "-march=armv8-a")
        endif()
    endif()
endif()

if (WITH_SSE4_1)
    add_definitions(-DTNN_FEATURE_SSE4_1)
endif()

if (WITH_AVX2)
    add_definitions(-DTNN_FEATURE_AVX2)
endif()
