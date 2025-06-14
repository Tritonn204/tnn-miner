if (WITH_BLAKE2)
    add_definitions(/DTNN_BLAKE2)

    file(GLOB blake2_sources
      src/crypto/blake2/blake2b.cpp
    )

    list(APPEND SOURCES_CRYPTO
      ${blake2_sources}
    )
else()
    remove_definitions(/DTNN_BLAKE2)
endif()
unset(WITH_BLAKE2 CACHE)
