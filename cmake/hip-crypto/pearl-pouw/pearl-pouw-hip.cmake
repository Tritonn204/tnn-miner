if (WITH_PEARL)
  add_definitions(/DTNN_PEARL)

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl_prepare_source.hpp"
      SOURCES "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/pearl/pearl_prepare.hip"
      NO_MANIFEST
      NAMESPACE hip_pearl_prepare_source
  )

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl_batch_prepare_source.hpp"
      SOURCES "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/pearl/pearl_batch_prepare.hip"
      NO_MANIFEST
      NAMESPACE hip_pearl_batch_prepare_source
  )

  # Production Iris recipe, shared by mining, validation and benchmark.
  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl_iris_qualified_headers.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/iris/gemm/native128/recipe.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/iris/gemm/native128/asm_templates.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/pearl/native128/layout.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/pearl/native128/slots.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/iris/gemm/native128/raw_slots.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/iris/gemm/native128/raw_kernel.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/iris/gemm/native128/buffer_inputs.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/bounded_buffer_load.hpp"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/iris/rt/gemm_schedule.hpp"
      MANIFEST_NAME PEARL_IRIS_QUALIFIED_HEADERS
  )

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl_iris_qualified_kernel.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/pearl/native128/rtc.hip"
      NO_MANIFEST
      NAMESPACE hip_pearl_iris_qualified_source
  )

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/pearl_embedded_headers.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/blake3-inline.hip.inc"
      MANIFEST_NAME PEARL_HEADERS
  )

  include_directories("${PROJECT_BINARY_DIR}/generated")

  list(APPEND TNN_HIP_SOURCES
    src/tnn_hip/crypto/pearl/pearl_native.cpp
    src/tnn_hip/coins/pearl/pearl_gpu.cpp
    src/tnn_hip/coins/pearl/pearl_tuning.cpp
    src/tnn_hip/coins/pearl/mine_pearl.hip.cpp
    src/tnn_hip/coins/pearl/test_pearl_hip.cpp
  )
else()
  remove_definitions(/DTNN_PEARL)
endif()
