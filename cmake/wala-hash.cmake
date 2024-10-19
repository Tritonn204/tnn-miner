if (WITH_ASTRIXHASH)
    add_definitions(/DTNN_WALAHASH)

    message(STATUS "Building with WalaHash support")

    set(WITH_BLAKE3 ON)

    file(GLOB_RECURSE walaHashHeaders
      src/crypto/wala-hash/*.h
    )

    file(GLOB_RECURSE walaHashSources
      src/crypto/wala-hash/*.cpp
      src/crypto/wala-hash/*.c
      src/net/wala/*.cpp
    )
    
    list(APPEND walaHashSources
      src/coins/mine_waglayla.cpp
    )

    list(APPEND HEADERS_CRYPTO
      ${walaHashHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${walaHashSources}
    )
else()
    remove_definitions(/DTNN_WALAHASH)
endif()
