if (WITH_ASTRIXHASH)
  file(GLOB_RECURSE hipAstrixHashSources
    src/tnn_hip/crypto/astrix-hash/*.hip
    src/tnn_hip/coins/astrix/mine_astrix.hip
    src/tnn_hip/coins/astrix/mine_astrix.hip.cpp
  )

  list(APPEND hipSources ${hipAstrixHashSources})
endif()