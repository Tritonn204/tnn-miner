# cmake/winring0.cmake
# WinRing0 driver for MSR access on Windows platforms

if(NOT WIN32)
  return()
endif()

# Define WinRing0 paths
set(WINRING0_ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/winring0")
set(WINRING0_INCLUDE_DIR "${WINRING0_ROOT_DIR}/include")
set(WINRING0_LIB_DIR "${WINRING0_ROOT_DIR}/lib")
set(WINRING0_BIN_DIR "${WINRING0_ROOT_DIR}/bin")

# Platform-specific settings
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  # 64-bit
  set(WINRING0_LIB_NAME "WinRing0x64")
  set(WINRING0_DLL "${WINRING0_BIN_DIR}/x64/${WINRING0_LIB_NAME}.dll")
  set(WINRING0_SYS "${WINRING0_BIN_DIR}/x64/${WINRING0_LIB_NAME}.sys")
  set(WINRING0_LIB "${WINRING0_LIB_DIR}/x64/${WINRING0_LIB_NAME}.lib")
else()
  # 32-bit
  set(WINRING0_LIB_NAME "WinRing0")
  set(WINRING0_DLL "${WINRING0_BIN_DIR}/x86/${WINRING0_LIB_NAME}.dll")
  set(WINRING0_SYS "${WINRING0_BIN_DIR}/x86/${WINRING0_LIB_NAME}.sys")
  set(WINRING0_LIB "${WINRING0_LIB_DIR}/x86/${WINRING0_LIB_NAME}.lib")
endif()

# Verify files exist
if(NOT EXISTS "${WINRING0_INCLUDE_DIR}/OlsApi.h")
  message(WARNING "WinRing0 headers not found at ${WINRING0_INCLUDE_DIR}")
  set(WINRING0_FOUND FALSE)
  return()
endif()

if(NOT EXISTS "${WINRING0_LIB}")
  message(WARNING "WinRing0 library not found at ${WINRING0_LIB}")
  set(WINRING0_FOUND FALSE)
  return()
endif()

# Create imported target
add_library(WinRing0 STATIC IMPORTED)
set_target_properties(WinRing0 PROPERTIES
  IMPORTED_LOCATION "${WINRING0_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${WINRING0_INCLUDE_DIR}"
)

# Define function to copy WinRing0 runtime files to target output directory
function(target_link_winring0 target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "Invalid target: ${target_name}")
  endif()
  
  target_include_directories(${target_name} PRIVATE "${WINRING0_INCLUDE_DIR}")
  target_link_libraries(${target_name} PRIVATE WinRing0)
  
  add_custom_command(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${WINRING0_DLL}" $<TARGET_FILE_DIR:${target_name}>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${WINRING0_SYS}" $<TARGET_FILE_DIR:${target_name}>
    COMMENT "Copying WinRing0 runtime files to output directory"
  )
  
  target_compile_definitions(${target_name} PRIVATE TNN_HAS_WINRING0=1)
endfunction()

set(WINRING0_FOUND TRUE)