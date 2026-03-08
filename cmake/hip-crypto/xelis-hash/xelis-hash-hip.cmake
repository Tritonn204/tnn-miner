if (WITH_XELISHASH)
  add_definitions(/DTNN_XELISHASH)
  # Embed HIP headers needed for HIPRTC compilation
  # These will be passed to hiprtcCreateProgram as header sources
  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/xelis_embedded_headers.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/blake3-inline.hip.inc"
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/aes-inline.hip.inc"
      MANIFEST_NAME XELIS_SOURCES
  )

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/xelis-hash-v3.hip.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/xelis-hash/xelis-hash-v3.hip"
      NO_MANIFEST
      NAMESPACE hip_xelis_v3_source
  )

  # Add generated directory to include path so code can find the embedded headers
  include_directories("${PROJECT_BINARY_DIR}/generated")

  # Add the generated header file so it gets regenerated when sources change
  set(XELIS_EMBEDDED_HEADERS "${PROJECT_BINARY_DIR}/generated/xelis_embedded_headers.hpp")

  list(APPEND TNN_HIP_SOURCES
    src/tnn_hip/coins/xelis/mine_xelis.hip.cpp
    src/tnn_hip/coins/xelis/test_xelis_hip.cpp
    ${XELIS_EMBEDDED_HEADERS}  # CMake will regenerate this when .hip/.inc files change
  )
else()
  remove_definitions(/DTNN_XELISHASH)
endif()