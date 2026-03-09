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
    )

    # New unified CPU mining infrastructure
    list(APPEND xelisSources
      src/tnn_cpu/mine_xelis_unified.cpp
    )

    list(FILTER xelisSources EXCLUDE REGEX src/xelis-hash/target/*)
    list(FILTER xelisSources EXCLUDE REGEX "\\.spec\\.cpp$")

    # Generate per-SIMD-tier TUs from spec files
    include(cmake/simd_tiers.cmake)
    set(xelis_simd_generated "")
    simd_tiers_generate(
      SPEC   src/crypto/xelis-hash/xelis-hash-v3-fmv.spec.cpp
      OUTDIR src/crypto/xelis-hash/simd_sources
      OUTPUT_SOURCES xelis_simd_generated
    )
    include_directories("${PROJECT_SOURCE_DIR}/src/crypto/xelis-hash/simd_sources")

    list(APPEND HEADERS_CRYPTO
      ${xelisHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${xelisSources}
      ${xelis_simd_generated}
    )
else()
    remove_definitions(/DTNN_XELISHASH)
endif()
