if (WITH_ASTRIXHASH)
    add_definitions(/DTNN_ASTRIXHASH)

    message(STATUS "Building with AstrixHash support")

    set(WITH_BLAKE3 ON)

    file(GLOB_RECURSE astrixHashHeaders
      src/crypto/astrix-hash/*.h
    )

    file(GLOB_RECURSE astrixHashSources
      src/crypto/astrix-hash/*.cpp
      src/crypto/astrix-hash/*.c
      src/net/astrix/*.cpp
    )
    
    # New unified CPU mining infrastructure
    list(APPEND astrixHashSources
      src/tnn_cpu/mine_kas_unified.cpp
    )

    list(APPEND HEADERS_CRYPTO
      ${astrixHashHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${astrixHashSources}
    )
else()
    remove_definitions(/DTNN_ASTRIXHASH)
endif()
