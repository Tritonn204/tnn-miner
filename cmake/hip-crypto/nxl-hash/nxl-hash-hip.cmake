if (WITH_NXLHASH)
  file(GLOB_RECURSE hipNxlHashSources
    src/tnn_hip/crypto/nxl-hash/*.hip
    src/tnn_hip/coins/nexellia/mine_nexellia.hip
    src/tnn_hip/coins/nexellia/mine_nexellia.hip.cpp
  )

  list(APPEND hipSources ${hipNxlHashSources})
endif()