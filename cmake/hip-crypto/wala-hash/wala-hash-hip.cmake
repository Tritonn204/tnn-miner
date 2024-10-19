if (WITH_WALAHASH)
  add_definitions(/DTNN_WALAHASH)

  file(GLOB_RECURSE hipWalaHashSources
    src/tnn_hip/crypto/wala-hash/*.hip
    src/tnn_hip/coins/waglayla/mine_waglayla.hip
    src/tnn_hip/coins/waglayla/mine_waglayla.hip.cpp
  )

  list(APPEND hipSources ${hipWalaHashSources})
else()
  remove_definitions(/DTNN_WALAHASH)
endif()
