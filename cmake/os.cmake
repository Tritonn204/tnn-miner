if (WIN32)
    set(TNN_OS_WIN ON)
elseif (APPLE)
    set(TNN_OS_APPLE ON)

    if (IOS OR CMAKE_SYSTEM_NAME STREQUAL iOS)
        set(TNN_OS_IOS ON)
    else()
        set(TNN_OS_MACOS ON)
    endif()
else()
    set(TNN_OS_UNIX ON)

    if (ANDROID OR CMAKE_SYSTEM_NAME MATCHES "Android")
        set(TNN_OS_ANDROID ON)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(TNN_OS_LINUX ON)
    elseif(CMAKE_SYSTEM_NAME STREQUAL FreeBSD OR CMAKE_SYSTEM_NAME STREQUAL DragonFly)
        set(TNN_OS_FREEBSD ON)
    endif()
endif()


if (TNN_OS_WIN)
    add_definitions(-DWIN32 -DTNN_OS_WIN)
elseif(TNN_OS_APPLE)
    add_definitions(-DTNN_OS_APPLE)

    if (TNN_OS_IOS)
        add_definitions(-DTNN_OS_IOS)
    else()
        add_definitions(-DTNN_OS_MACOS)
    endif()

    if (TNN_ARM)
        set(WITH_SECURE_JIT ON)
    endif()
elseif(TNN_OS_UNIX)
    add_definitions(-DTNN_OS_UNIX)

    if (TNN_OS_ANDROID)
        add_definitions(-DTNN_OS_ANDROID)
    elseif (TNN_OS_LINUX)
        add_definitions(-DTNN_OS_LINUX)
    elseif (TNN_OS_FREEBSD)
        add_definitions(-DTNN_OS_FREEBSD)
    endif()
endif()

set(WINSDK_DIR "${PROJECT_SOURCE_DIR}/wincross/winsdk")
# TODO: Rename this to something else
if(WIN_CROSS)
  set(WIN_CROSS_OPTS "${WIN_CROSS_OPTS} --target=x86_64-pc-windows-msvc")
  set(WIN_CROSS_OPTS "${WIN_CROSS_OPTS} \
  -I${WINSDK_DIR}/../clang64/include \
  -I${WINSDK_DIR}/crt/include \
  -I${WINSDK_DIR}/sdk/Include/ucrt \
  -I${WINSDK_DIR}/sdk/Include/shared \
  -I${WINSDK_DIR}/sdk/Include/um")
endif()

set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ${WIN_CROSS_OPTS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${WIN_CROSS_OPTS}")

if (WITH_SECURE_JIT)
    add_definitions(-DTNN_SECURE_JIT)
endif()
