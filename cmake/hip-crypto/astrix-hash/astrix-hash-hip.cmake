if (WITH_ASTRIXHASH)
  file(GLOB_RECURSE hipAstrixHashSources
    # src/tnn_hip/crypto/astrix-hash/*.hip
    crypto/cshake/*.cpp
    crypto/cshake/*.hip
  )

  list(APPEND hipSources ${hipAstrixHashSources})
endif()