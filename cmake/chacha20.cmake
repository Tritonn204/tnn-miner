if (WITH_CHACHA)
    add_definitions(/DTNN_CHACHA)

    file(GLOB chacha_sources
      "src/crypto/chacha/*.c"
    )

    list(APPEND SOURCES_CRYPTO
      ${chacha_sources}
    )
else()
    remove_definitions(/DTNN_ALGO_CHACHA)
endif()
unset(WITH_CHACHA CACHE)