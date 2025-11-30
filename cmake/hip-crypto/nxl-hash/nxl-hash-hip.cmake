if (WITH_NXLHASH)
  file(GLOB_RECURSE hipNxlHashSources
    src/tnn_hip/crypto/nxl-hash/*.hip
    src/tnn_hip/coins/nexellia/mine_nexellia.hip
  )

  list(APPEND SOURCES_CRYPTO 
    src/tnn_hip/coins/nexellia/mine_nexellia.hip.cpp
  )

  list(APPEND TNN_HIP_SOURCES ${hipNxlHashSources})
endif()