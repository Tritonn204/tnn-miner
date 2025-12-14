#pragma once

#include <tnn_hip/hello.hpp>

#include <tnn_hip/crypto/astrix-hash/test_hip_astrix.h>
#include <tnn_hip/crypto/nxl-hash/test_hip_nxl.h>
#include <tnn_hip/crypto/wala-hash/test_hip_wala.h>

#include <crypto/astrix-hash/astrix-hash.h>
#include <crypto/nxl-hash/nxl-hash.h>

#include <tnn_hip/core/test_hiprtc_isolation.hpp>

inline int GPUTest() {
  #ifdef TNN_HIP
    // Run HIPRTC isolation test FIRST, before any other GPU work
    test_hiprtc_isolation();

    // if (is_hip_supported()) {
      // helloTest();
      // benchAstrixHip();
      // benchWalaHip();
      // AstrixHash::test();

      // benchNxlHip();
      // NxlHash::test();
    // }
  #endif
  return 0;
}