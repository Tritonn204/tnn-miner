# Define a function to set properties and link libraries for a target
function(setup_target_libraries target_name)
  # Link Boost libraries
  target_link_libraries(${target_name} ${TNN_BOOST_LIBS})

  # Link Windows-specific libraries
  if(WIN32)
    if (NOT WITH_HIP)
      target_link_libraries(${target_name} mingw32)
    endif()
    target_link_libraries(${target_name} wsock32 ws2_32 kernel32)
  endif()

  # Link libraries for non-Apple, non-Windows systems (likely Linux)
  if(NOT APPLE AND NOT WIN32)
    target_link_libraries(${target_name} udns)
  endif()

  # Link AstroSPSA library if enabled
  if(USE_ASTRO_SPSA)
    target_link_libraries(${target_name} ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${TNN_SPSA_CLANG_VER}.a)
  endif()

  # Link threading libraries, OpenSSL, and other required libraries
  target_link_libraries(${target_name} ${THREAD_LIB} OpenSSL::SSL OpenSSL::Crypto sodium ${BLAKE3_LIBRARIES})

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endfunction()
