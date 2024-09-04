if (WITH_XELISHASH)
    add_definitions(/DTNN_ALGO_XELISHASH)

    message(STATUS "Building with XelisHash support")

    file(GLOB_RECURSE xelisHeaders
      src/crypto/xelis-hash/*.h
    )

    file(GLOB_RECURSE xelisSources
      src/crypto/xelis-hash/*.cpp
      src/crypto/xelis-hash/*.c
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
