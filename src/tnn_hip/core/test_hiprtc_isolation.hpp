#pragma once

// HIPRTC isolation test - tests if HIPRTC crashes are intrinsic or from external interference
// Implementation is in src/tnn_hip/core/test_hiprtc_isolation.cpp (compiled with HIP support)
// This runs early in main() before threads/GPU kernels/network to isolate HIPRTC behavior

extern "C" void test_hiprtc_isolation();
