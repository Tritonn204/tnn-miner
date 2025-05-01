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
  if(NOT APPLE AND NOT WIN32 AND NOT WIN_CROSS)
    target_link_libraries(${target_name} udns)
  endif()

  # Link AstroSPSA library if enabled
  if(USE_ASTRO_SPSA)
    cmake_print_variables(SPSA_FULL_LIB_PATH)
    target_link_libraries(${target_name} ${SPSA_FULL_LIB_PATH})
  endif()

  if (WIN_CROSS)
    add_library(sodium STATIC IMPORTED)
    set(LibSodium_ROOT "${PROJECT_SOURCE_DIR}/wincross/clang64/lib")
    #target_link_libraries(${target_name} PRIVATE ${LibSodium_ROOT})
    #find_package(LibSodium REQUIRED)
    link_directories(LibSodium_ROOT)
    link_directories("${PROJECT_SOURCE_DIR}/export/boost-x64/lib")
  endif()

  # Link threading libraries, OpenSSL, and other required libraries
  if (WIN_CROSS)
    #add_library(${THREAD_LIB} STATIC IMPORTED)
    #add_library(sodium STATIC IMPORTED)
    #target_link_libraries(${target_name} OpenSSL::SSL OpenSSL::Crypto)
    #target_link_libraries(${target_name} ${BLAKE3_LIBRARIES})
    #target_link_libraries(${target_name} ${THREAD_LIB} OpenSSL::SSL OpenSSL::Crypto sodium ${BLAKE3_LIBRARIES})
  else()
    #target_link_libraries(${target_name} ${THREAD_LIB} OpenSSL::SSL OpenSSL::Crypto sodium ${BLAKE3_LIBRARIES})
    target_link_libraries(${target_name} ${THREAD_LIB} OpenSSL::SSL OpenSSL::Crypto ${BLAKE3_LIBRARIES})
  endif()

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endfunction()
