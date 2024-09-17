if (WITH_CHACHA)
    add_definitions(/DTNN_CHACHA)

    if (TNN_ARM)
      file(GLOB chacha_sources
        "src/crypto/chacha/chacha20.c"
      )
    else()
      file(GLOB chacha_sources
        "src/crypto/chacha/*.c"
      )
    endif()

    set_source_files_properties(src/crypto/chacha/chacha20_sse2.c COMPILE_FLAGS -msse2)
    set_source_files_properties(src/crypto/chacha/chacha20_avx2.c COMPILE_FLAGS -mavx2)
    set_source_files_properties(src/crypto/chacha/chacha20_avx512.c COMPILE_FLAGS -mavx512f)

    list(APPEND SOURCES_CRYPTO
      ${chacha_sources}
    )
else()
    remove_definitions(/DTNN_CHACHA)
endif()
unset(WITH_CHACHA CACHE)
