if(NOT WIN32)
  return()
endif()

set(WINRING0_ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/winring0")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(WINRING0_SYS "${WINRING0_ROOT_DIR}/WinRing0x64.sys")
else()
  set(WINRING0_SYS "${WINRING0_ROOT_DIR}/WinRing0.sys")
endif()

set(WINRING0_FOUND TRUE)

if(NOT EXISTS "${WINRING0_SYS}")
  message(WARNING "WinRing0 driver not found: ${WINRING0_SYS} — MSR optimization will not be available")
  set(WINRING0_FOUND FALSE)
endif()

function(target_link_winring0 target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "Invalid target: ${target_name}")
  endif()

  if(NOT WINRING0_FOUND)
    return()
  endif()

  # Copy .sys driver to output directory (no DLL or import lib needed)
  add_custom_command(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${WINRING0_SYS}" $<TARGET_FILE_DIR:${target_name}>
    COMMENT "Copying WinRing0x64.sys driver"
  )

  target_compile_definitions(${target_name} PRIVATE TNN_HAS_WINRING0=1)
endfunction()
