if (WITH_PEARL)
  add_definitions(/DTNN_PEARL)

  list(APPEND TNN_HIP_SOURCES
    src/tnn_hip/coins/pearl/test_pearl_hip.cpp
  )
else()
  remove_definitions(/DTNN_PEARL)
endif()
