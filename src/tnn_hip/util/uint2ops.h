#pragma once

#if !defined(__HIPCC_RTC__) && !defined(__CUDACC_RTC__)
#include <hip/hip_runtime.h>
#else
// hipRTC: stdint types live in __hip_internal:: on AMD
#ifndef __HIPRTC__INTTYPES_DEFINED
#define __HIPRTC__INTTYPES_DEFINED
#ifdef __HIP_PLATFORM_AMD__
using uint8_t  = __hip_internal::uint8_t;
using uint16_t = __hip_internal::uint16_t;
#elif defined(__CUDACC_RTC__)
#include "stdint-jit.hip.inc"
#endif
#endif
#endif
static __device__ __forceinline__ uint2 rol64(const uint2 a, const uint8_t r)
{
  uint2 result;
#ifdef __HIP_PLATFORM_AMD__
  if (r <= 32)
  {
    result.x = __builtin_amdgcn_alignbit(a.x, a.y, 32 - r);
    result.y = __builtin_amdgcn_alignbit(a.y, a.x, 32 - r);
  }
  else
  {
    result.x = __builtin_amdgcn_alignbit(a.y, a.x, 64 - r);
    result.y = __builtin_amdgcn_alignbit(a.x, a.y, 64 - r);
  }
#else
	if (r >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"((uint32_t)r));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"((uint32_t)r));
	}
	else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"((uint32_t)r));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"((uint32_t)r));
	}
#endif
  return result;
}

static __device__ __forceinline__ uint2 operator^(uint2 a, uint2 b)
{
  return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

static __device__ __forceinline__ void operator^=(uint2 &a, uint2 b)
{
  a.x = a.x ^ b.x;
  a.y = a.y ^ b.y;
}