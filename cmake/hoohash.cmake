if (WITH_HOOHASH)
    add_definitions(/DTNN_HOOHASH)

    message(STATUS "Building with HooHash support")

    set(WITH_BLAKE3 ON)

    file(GLOB_RECURSE HooHashHeaders
      src/crypto/hoohash/*.h
    )

    file(GLOB_RECURSE HooHashSources
      src/crypto/hoohash/*.cpp
      src/crypto/hoohash/*.c
      src/net/hoohash/*.cpp
    )
    
    list(APPEND HooHashSources
      src/coins/mine_hoosat.cpp
    )

    list(APPEND HEADERS_CRYPTO
      ${HooHashHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${HooHashSources}
    )
else()
    remove_definitions(/DTNN_HOOHASH)
endif()
