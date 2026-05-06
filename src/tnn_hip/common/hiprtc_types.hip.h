#pragma once
// ============================================================================
// hiprtc_types.hip.h — Standard integer types for RTC-compiled GPU kernels
//
// hipRTC 7.01+ wraps stdint types in __hip_internal:: namespace.
// This header imports them into the global namespace so kernel sources
// work across hipRTC versions.  For CUDA RTC, falls back to manual typedefs.
// For offline (hipcc/nvcc) compilation, just includes the normal headers.
//
// Include this at the top of any .hip file that may be compiled via RTC.
// ============================================================================

#if !defined(__HIPCC_RTC__) && !defined(__CUDACC_RTC__)
  // Offline compilation — normal headers available
  #include <stdint.h>
#elif defined(__HIP_PLATFORM_AMD__)
  // AMD hipRTC — import from __hip_internal:: (7.01+) with fallback
  #ifndef __HIPRTC__INTTYPES_DEFINED
  #define __HIPRTC__INTTYPES_DEFINED
  using uint8_t  = __hip_internal::uint8_t;
  using uint16_t = __hip_internal::uint16_t;
  using uint32_t = __hip_internal::uint32_t;
  using uint64_t = __hip_internal::uint64_t;
  using int8_t   = __hip_internal::int8_t;
  using int16_t  = __hip_internal::int16_t;
  using int32_t  = __hip_internal::int32_t;
  using int64_t  = __hip_internal::int64_t;
  // hipRTC 6.x pre-defines size_t globally; 7.01+ moves it to __hip_internal::
  #ifndef __SIZE_TYPE__
  using size_t   = __hip_internal::size_t;
  #endif
  #endif
#elif defined(__CUDACC_RTC__)
  // NVIDIA nvRTC — manual typedefs
  #include "stdint-jit.hip.inc"
#endif
