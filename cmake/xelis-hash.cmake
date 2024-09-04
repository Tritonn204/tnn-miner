if (WITH_XELISHASH)
    add_definitions(/DTNN_XELISHASH)

    message(STATUS "Building with XelisHash support")

    set(WITH_BLAKE3 ON)
    set(WITH_CHACHA ON)

    file(GLOB_RECURSE xelisHeaders
      src/crypto/xelis-hash/*.h
    )

    file(GLOB_RECURSE xelisSources
      src/crypto/xelis-hash/*.cpp
      src/crypto/xelis-hash/*.c
      src/net/xelis/*.cpp
    )

    list(APPEND xelisSources
      src/coins/mine_xelis.cpp
    )

    list(FILTER xelisSources EXCLUDE REGEX src/xelis-hash/target/*)

    list(APPEND HEADERS_CRYPTO
      ${xelisHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${xelisSources}
    )
else()
    remove_definitions(/DTNN_ALGO_XELISHASH)
endif()
unset(WITH_XELISHASH CACHE)
