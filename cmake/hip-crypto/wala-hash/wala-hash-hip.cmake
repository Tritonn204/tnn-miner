if (WITH_WALAHASH)
  add_definitions(/DTNN_WALAHASH)

  file(GLOB_RECURSE hipWalaHashSources
    src/tnn_hip/crypto/wala-hash/*.hip
    src/tnn_hip/coins/waglayla/mine_waglayla.hip
  )

  list(APPEND SOURCES_CRYPTO 
    src/tnn_hip/coins/waglayla/mine_waglayla.hip.cpp
  )

  list(APPEND TNN_HIP_SOURCES ${hipWalaHashSources})
else()
  remove_definitions(/DTNN_WALAHASH)
endif()
