if (ARM_TARGET)
  message(STATUS "Removing VerusHash support due to arm build")
  unset(WITH_VERUSHASH CACHE)
endif()
if (WITH_VERUSHASH)
    add_definitions(/DTNN_VERUSHASH)

    message(STATUS "Building with VerusHash support")

    file(GLOB_RECURSE verusHeaders
      src/crypto/verus/*.h
    )

    file(GLOB_RECURSE verusSources
      src/crypto/verus/*.cpp
      src/crypto/verus/*.c
      src/net/verus/*.cpp
      src/coins/mine_verus.cpp
    )

    set_source_files_properties(src/crypto/verus/haraka.c COMPILE_FLAGS -maes)
    set_source_files_properties(src/crypto/verus/verus_clhash.cpp COMPILE_FLAGS "-maes -mpclmul")
    

    # list(APPEND xelisSources
    #   src/coins/mine_xelis.cpp
    # )

    list(APPEND HEADERS_CRYPTO
      ${verusHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${verusSources}
    )
else()
    remove_definitions(/DTNN_ALGO_VERUSHASH)
endif()
unset(WITH_VERUSHASH CACHE)
