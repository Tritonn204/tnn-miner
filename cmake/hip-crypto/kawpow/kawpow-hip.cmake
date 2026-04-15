if (WITH_KAWPOW)
  add_definitions(/DTNN_KAWPOW)

  message(STATUS "Building with KawPow support")

  # Embed dependency headers for HIPRTC include resolution
  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/kawpow_embedded_headers.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/bitselect.hip.h"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/util/uint2ops.h"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/keccak-amd.hip.inc"
      MANIFEST_NAME KAWPOW_HEADERS
  )

  # Embed the KawPow GPU kernel source for HIPRTC
  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/kawpow.hip.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/kawpow/kawpow.hip"
      NO_MANIFEST
      NAMESPACE hip_kawpow_source
  )

  include_directories("${PROJECT_BINARY_DIR}/generated")

  # cpp-kawpow reference implementation (CPU)
  # Used for test vector validation and light-verify
  set(KAWPOW_REF_DIR "${PROJECT_SOURCE_DIR}/src/crypto/kawpow")

  list(APPEND KAWPOW_REF_SOURCES
    "${KAWPOW_REF_DIR}/ethash/ethash.cpp"
    "${KAWPOW_REF_DIR}/ethash/managed.cpp"
    "${KAWPOW_REF_DIR}/ethash/progpow.cpp"
    "${KAWPOW_REF_DIR}/ethash/primes.c"
    "${KAWPOW_REF_DIR}/keccak/keccak.c"
    "${KAWPOW_REF_DIR}/keccak/keccakf800.c"
    "${KAWPOW_REF_DIR}/keccak/keccakf1600.c"
  )

  # The ethash headers use #include <ethash/...> so we need the parent dir as include root
  include_directories("${KAWPOW_REF_DIR}")

  list(APPEND TNN_HIP_SOURCES
    src/tnn_hip/coins/kawpow/test_kawpow_hip.cpp
    src/tnn_hip/coins/kawpow/mine_kawpow.hip.cpp
    ${KAWPOW_REF_SOURCES}
  )
else()
  remove_definitions(/DTNN_KAWPOW)
endif()
