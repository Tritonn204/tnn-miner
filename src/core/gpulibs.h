#pragma once

#include <tnn_hip/hello.hpp>
#include <tnn_hip/crypto/astrix-hash/test_hip_astrix.h>

inline int GPUTest() {
  #ifdef TNN_HIP
    if (is_hip_supported()) {
      helloTest();
      benchAstrixHip();
    }
  #endif
  return 0;
}