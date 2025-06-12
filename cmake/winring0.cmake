if(NOT WIN32)
  return()
endif()

set(WINRING0_ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/winring0")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(WINRING0_DLL "${WINRING0_ROOT_DIR}/WinRing0x64.dll")
  set(WINRING0_SYS "${WINRING0_ROOT_DIR}/WinRing0x64.sys")
  set(WINRING0_IMPLIB "${WINRING0_ROOT_DIR}/libWinRing0x64.a")
else()
  set(WINRING0_DLL "${WINRING0_ROOT_DIR}/WinRing0.dll")
  set(WINRING0_SYS "${WINRING0_ROOT_DIR}/WinRing0.sys")
  set(WINRING0_IMPLIB "${WINRING0_ROOT_DIR}/libWinRing0.a")
endif()

set(WINRING0_FOUND TRUE)

# # Check architecture
# if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#   set(WINRING0_DLL "${WINRING0_ROOT_DIR}/WinRing0x64.dll")
#   set(WINRING0_SYS "${WINRING0_ROOT_DIR}/WinRing0x64.sys")
#   set(WINRING0_IMPLIB "${CMAKE_BINARY_DIR}/libWinRing0x64.a")
# else()
#   set(WINRING0_DLL "${WINRING0_ROOT_DIR}/WinRing0.dll")
#   set(WINRING0_SYS "${WINRING0_ROOT_DIR}/WinRing0.sys")
#   set(WINRING0_IMPLIB "${CMAKE_BINARY_DIR}/libWinRing0.a")
# endif()

# # Check files
# if(NOT EXISTS "${WINRING0_ROOT_DIR}/OlsApi.h")
#   message(FATAL_ERROR "WinRing0 header not found: ${WINRING0_ROOT_DIR}/OlsApi.h")
# endif()

# if(NOT EXISTS "${WINRING0_DLL}")
#   message(FATAL_ERROR "WinRing0 DLL not found: ${WINRING0_DLL}")
# endif()

# # Create .def file for import library generation
# if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#   set(WINRING0_DEF "${CMAKE_BINARY_DIR}/WinRing0x64.def")
# else()
#   set(WINRING0_DEF "${CMAKE_BINARY_DIR}/WinRing0.def")
# endif()

# # Write .def file with all the exports
# file(WRITE ${WINRING0_DEF}
# "EXPORTS
# InitializeOls
# DeinitializeOls
# GetDllStatus
# GetDllVersion
# GetDriverVersion
# GetDriverType
# IsCpuid
# IsMsr
# IsTsc
# Rdmsr
# RdmsrTx
# RdmsrPx
# Wrmsr
# WrmsrTx
# WrmsrPx
# Cpuid
# CpuidTx
# CpuidPx
# Rdtsc
# RdtscTx
# RdtscPx
# ReadIoPortByte
# ReadIoPortWord
# ReadIoPortDword
# WriteIoPortByte
# WriteIoPortWord
# WriteIoPortDword
# ")

# # Find dlltool (should be available with MinGW/clang)
# find_program(DLLTOOL_EXECUTABLE NAMES 
#   llvm-dlltool 
#   dlltool
#   x86_64-w64-mingw32-dlltool  # Common MinGW path
# )

# if(NOT DLLTOOL_EXECUTABLE)
#   message(FATAL_ERROR "dlltool not found. Please install MinGW development tools.")
# endif()

# # Generate import library if it doesn't exist
# if(NOT EXISTS ${WINRING0_IMPLIB})
#   message(STATUS "Creating MinGW-compatible import library for WinRing0...")
  
#   execute_process(
#     COMMAND ${DLLTOOL_EXECUTABLE} 
#       --def ${WINRING0_DEF}
#       --dllname $<IF:$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>,WinRing0x64.dll,WinRing0.dll>
#       --output-lib ${WINRING0_IMPLIB}
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#     RESULT_VARIABLE DLLTOOL_RESULT
#     OUTPUT_VARIABLE DLLTOOL_OUTPUT
#     ERROR_VARIABLE DLLTOOL_ERROR
#   )
  
#   if(NOT DLLTOOL_RESULT EQUAL 0)
#     message(FATAL_ERROR "Failed to create import library: ${DLLTOOL_ERROR}")
#   endif()
  
#   message(STATUS "Successfully created import library: ${WINRING0_IMPLIB}")
# endif()

# function(target_link_winring0 target_name)
#   if(NOT TARGET ${target_name})
#     message(FATAL_ERROR "Invalid target: ${target_name}")
#   endif()
  
#   message(STATUS "Linking WinRing0 to ${target_name} using ${WINRING0_IMPLIB}")
  
#   # Add include directory
#   target_include_directories(${target_name} PRIVATE "${WINRING0_ROOT_DIR}")
  
#   # Link the MinGW-compatible import library
#   target_link_libraries(${target_name} ${WINRING0_IMPLIB})
  
#   # Copy runtime files to output directory  
#   add_custom_command(TARGET ${target_name} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${WINRING0_DLL}" $<TARGET_FILE_DIR:${target_name}>
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${WINRING0_SYS}" $<TARGET_FILE_DIR:${target_name}>
#     COMMENT "Copying WinRing0 runtime files"
#   )
  
#   target_compile_definitions(${target_name} PRIVATE TNN_HAS_WINRING0=1)
# endfunction()

# set(WINRING0_FOUND TRUE)