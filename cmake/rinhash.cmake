if (WITH_RINHASH)
    add_definitions(/DTNN_RINHASH)

    message(STATUS "Building with RinHash support")

    file(GLOB_RECURSE rinHeaders
      src/crypto/rinhash/*.h
    )

    file(GLOB_RECURSE rinSources
      src/crypto/rinhash/*.cpp
      src/crypto/rinhash/*.c
      src/net/btc/*.cpp
    )
    
    list(APPEND rinSources
      src/coins/mine_rinhash.cpp
    )

    list(APPEND HEADERS_CRYPTO
      ${rinHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${rinSources}
    )

else()
    remove_definitions(/DTNN_RINHASH)
endif()
