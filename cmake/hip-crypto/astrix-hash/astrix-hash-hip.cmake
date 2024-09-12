if (WITH_ASTRIXHASH)
  file(GLOB_RECURSE hipAstrixHashSources
    # src/tnn_hip/crypto/astrix-hash/*.hip
    src/tnn_hip/crypto/cshake/*.cpp
    src/tnn_hip/crypto/cshake/*.hip
  )

  list(APPEND hipSources ${hipAstrixHashSources})
endif()