#pragma once

#include <hip/hip_runtime.h>

__device__ static __forceinline__ uint64_t bitselect(const uint64_t a, const uint64_t b, const uint64_t c)
{
#ifdef __HIP_PLATFORM_AMD__
  uint32_t a_lo = (uint32_t)(a & 0xFFFFFFFF);
  uint32_t a_hi = (uint32_t)(a >> 32);

  uint32_t b_lo = (uint32_t)(b & 0xFFFFFFFF);
  uint32_t b_hi = (uint32_t)(b >> 32);

  uint32_t c_lo = (uint32_t)(c & 0xFFFFFFFF);
  uint32_t c_hi = (uint32_t)(c >> 32);

  uint32_t result_lo = (a_lo & ~c_lo) | (b_lo & c_lo);
  uint32_t result_hi = (a_hi & ~c_hi) | (b_hi & c_hi);

  return ((uint64_t)result_hi << 32) | result_lo;
#else
  uint32_t result_lo, result_hi;

  asm volatile(
      "lop3.b32 %0, %2, %3, %4, 0x56;\n\t"
      "lop3.b32 %1, %5, %6, %7, 0x56;\n\t"
      : "=r"(result_lo), "=r"(result_hi)
      : "r"((uint32_t)(a & 0xFFFFFFFF)), "r"((uint32_t)(b & 0xFFFFFFFF)), "r"((uint32_t)(c & 0xFFFFFFFF)),
        "r"((uint32_t)(a >> 32)), "r"((uint32_t)(b >> 32)), "r"((uint32_t)(c >> 32)));

  return ((uint64_t)result_hi << 32) | result_lo;
#endif
}

__device__ static __forceinline__ uint2 bitselect(const uint2 a, const uint2 b, const uint2 c)
{
#ifdef __HIP_PLATFORM_AMD__
  return make_uint2((a.x & ~c.x) | (b.x & c.x), (a.y & ~c.y) | (b.y & c.y));
#else
  uint2 result;

  asm volatile(
      "lop3.b32 %0, %2, %3, %4, 0x56;\n\t"
      "lop3.b32 %1, %5, %6, %7, 0x56;\n\t"
      : "=r"(result.x), "=r"(result.y)
      : "r"(a.x), "r"(b.x), "r"(c.x),  // 0x96 = 0xF0 ^ ((~0xCC) & 0xAA)
        "r"(a.y), "r"(b.y), "r"(c.y)); // 0x96 = 0xF0 ^ ((~0xCC) & 0xAA)

  return result;
#endif
}

__device__ static __forceinline__ void bitselect(const uint2 a, const uint2 b, const uint2 c, uint2 &result)
{
#ifdef __HIP_PLATFORM_AMD__
  result.x = (a.x & ~c.x) | (b.x & c.x);
  result.y = (a.y & ~c.y) | (b.y & c.y);
#else
  asm volatile(
      "lop3.b32 %0, %2, %3, %4, 0x56;\n\t"
      "lop3.b32 %1, %5, %6, %7, 0x56;\n\t"
      : "=r"(result.x), "=r"(result.y)
      : "r"(a.x), "r"(b.x), "r"(c.x),  // 0x96 = 0xF0 ^ ((~0xCC) & 0xAA)
        "r"(a.y), "r"(b.y), "r"(c.y)); // 0x96 = 0xF0 ^ ((~0xCC) & 0xAA)
#endif
}