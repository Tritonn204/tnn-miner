if (WITH_PEARL)
  add_definitions(/DTNN_PEARL)

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl_embedded_headers.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/blake3-inline.hip.inc"
      MANIFEST_NAME PEARL_HEADERS
  )

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl-noise-dense-test.hip.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/pearl/noise_generation_dense_test.hip"
      NO_MANIFEST
      NAMESPACE hip_pearl_noise_dense_source
  )

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl-expand-jackpot.hip.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/pearl/pearl_expand_jackpot.hip"
      NO_MANIFEST
      NAMESPACE hip_pearl_expand_jackpot_source
  )

  include_directories("${PROJECT_BINARY_DIR}/generated")

  list(APPEND TNN_HIP_SOURCES
    src/tnn_hip/coins/pearl/test_pearl_hip.cpp
  )
else()
  remove_definitions(/DTNN_PEARL)
endif()
