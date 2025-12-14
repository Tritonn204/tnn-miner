if (WITH_NXLHASH)
    add_definitions(/DTNN_NXLHASH)

    message(STATUS "Building with NxlHash support")

    set(WITH_BLAKE3 ON)

    file(GLOB_RECURSE nxlHashHeaders
      src/crypto/nxl-hash/*.h
    )

    file(GLOB_RECURSE nxlHashSources
      src/crypto/nxl-hash/*.cpp
      src/crypto/nxl-hash/*.c
      src/net/nxl/*.cpp
    )
    
    # New unified CPU mining infrastructure
    list(APPEND nxlHashSources
      src/tnn_cpu/mine_kas_unified.cpp
    )

    list(APPEND HEADERS_CRYPTO
      ${nxlHashHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${nxlHashSources}
    )
else()
    remove_definitions(/DTNN_NXLHASH)
endif()
