if (WITH_PEARL)
  add_definitions(/DTNN_PEARL)

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl_embedded_headers.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/blake3-inline.hip.inc"
      MANIFEST_NAME PEARL_HEADERS
  )

  # Embed rocwmma cooperative GEMM kernel source
  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl_rocwmma_kernel.hip.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/gemm/rocwmma_kernel.hip"
      NO_MANIFEST
      NAMESPACE hip_pearl_rocwmma_source
  )

  # Embed pearl_gemm_simple kernel source (1:1 analogue of CPU ref GEMM)
  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl_gemm_simple.hip.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/pearl/pearl_gemm_simple.hip"
      NO_MANIFEST
      NAMESPACE hip_pearl_gemm_simple_source
  )

  # Embed rocwmma header sources for HIPRTC
  file(GLOB_RECURSE ROCWMMA_INTERNAL_HEADERS
      "${PROJECT_SOURCE_DIR}/include/rocwmma/internal/*.hpp"
  )
  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/rocwmma_headers.hip.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/include/rocwmma/rocwmma.hpp"
          "${PROJECT_SOURCE_DIR}/include/rocwmma/rocwmma_transforms.hpp"
          "${PROJECT_SOURCE_DIR}/include/rocwmma/rocwmma_impl.hpp"
          "${PROJECT_SOURCE_DIR}/include/rocwmma/rocwmma_transforms_impl.hpp"
          ${ROCWMMA_INTERNAL_HEADERS}
      MANIFEST_NAME ROCWMMA_HEADERS
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
    src/tnn_hip/crypto/pearl/pearl_pouw_defs.cpp
    src/tnn_hip/coins/pearl/test_pearl_hip.cpp
  )
else()
  remove_definitions(/DTNN_PEARL)
endif()
