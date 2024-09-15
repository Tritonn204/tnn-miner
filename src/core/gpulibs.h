#pragma once

#include <tnn_hip/hello.hpp>
#include <tnn_hip/crypto/cshake/cshake256.h>

inline int GPUTest() {
  #ifdef TNN_HIP
    if (is_hip_supported()) {
      helloTest();
      test_cshake256_hip();
      test_cshake256_comparison();
    }
  #endif
  return 0;
}