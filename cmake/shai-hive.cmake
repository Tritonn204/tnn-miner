if (WITH_SHAIHIVE)
  add_definitions(/DTNN_SHAIHIVE)

  message(STATUS "Building with ShaiHive support")

  file(GLOB_RECURSE shaiHeaders
    src/crypto/shai/*.h
  )

  file(GLOB_RECURSE shaiSources
    src/crypto/shai/*.cpp
    src/crypto/shai/*.c
    src/net/shai/*.cpp
    src/coins/mine_shai.cpp
  )

  #set_source_files_properties(src/crypto/shai/haraka.c COMPILE_FLAGS -maes)
  #set_source_files_properties(src/crypto/shai/verus_clhash.cpp COMPILE_FLAGS "-maes -mpclmul")
  

  # list(APPEND xelisSources
  #   src/coins/mine_xelis.cpp
  # )

  list(APPEND HEADERS_CRYPTO
    ${shaiHeaders}
  )

  list(APPEND SOURCES_CRYPTO
    ${shaiSources}
  )
else()
  remove_definitions(/DTNN_SHAIHIVE)
endif()