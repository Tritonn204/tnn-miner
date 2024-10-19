// Copyright (c) 2014-2023, The Monero Project
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other
//    materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors may be
//    used to endorse or promote products derived from this software without specific
//    prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
// THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _SKEIN_PORT_H_
#define _SKEIN_PORT_H_

#include <limits.h>
#include <stdint.h>
#include <hip/hip_runtime.h>

#ifndef RETURN_VALUES
#  define RETURN_VALUES
#  if defined( DLL_EXPORT )
#    if defined( _MSC_VER ) || defined ( __INTEL_COMPILER )
#      define VOID_RETURN    __declspec( dllexport ) void __stdcall
#      define INT_RETURN     __declspec( dllexport ) int  __stdcall
#    elif defined( __GNUC__ )
#      define VOID_RETURN    __declspec( __dllexport__ ) void
#      define INT_RETURN     __declspec( __dllexport__ ) int
#    else
#      error Use of the DLL is only available on the Microsoft, Intel and GCC compilers
#    endif
#  elif defined( DLL_IMPORT )
#    if defined( _MSC_VER ) || defined ( __INTEL_COMPILER )
#      define VOID_RETURN    __declspec( dllimport ) void __stdcall
#      define INT_RETURN     __declspec( dllimport ) int  __stdcall
#    elif defined( __GNUC__ )
#      define VOID_RETURN    __declspec( __dllimport__ ) void
#      define INT_RETURN     __declspec( __dllimport__ ) int
#    else
#      error Use of the DLL is only available on the Microsoft, Intel and GCC compilers
#    endif
#  elif defined( __WATCOMC__ )
#    define VOID_RETURN  void __cdecl
#    define INT_RETURN   int  __cdecl
#  else
#    define VOID_RETURN  void
#    define INT_RETURN   int
#  endif
#endif

/*  These defines are used to declare buffers in a way that allows
    faster operations on longer variables to be used.  In all these
    defines 'size' must be a power of 2 and >= 8

    dec_unit_type(size,x)       declares a variable 'x' of length 
                                'size' bits

    dec_bufr_type(size,bsize,x) declares a buffer 'x' of length 'bsize' 
                                bytes defined as an array of variables
                                each of 'size' bits (bsize must be a 
                                multiple of size / 8)

    ptr_cast(x,size)            casts a pointer to a pointer to a 
                                varaiable of length 'size' bits
*/

#define ui_type(size)               uint##size##_t
#define dec_unit_type(size,x)       typedef ui_type(size) x
#define dec_bufr_type(size,bsize,x) typedef ui_type(size) x[bsize / (size >> 3)]
#define ptr_cast(x,size)            ((ui_type(size)*)(x))

typedef unsigned int    uint_t;             /* native unsigned integer */
typedef uint8_t         u08b_t;             /*  8-bit unsigned integer */
typedef uint64_t        u64b_t;             /* 64-bit unsigned integer */

#ifndef RotL_64
#define RotL_64(x,N)    (((x) << (N)) | ((x) >> (64-(N))))
#endif

/*
 * Skein is "natively" little-endian (unlike SHA-xxx), for optimal
 * performance on x86 CPUs.  The Skein code requires the following
 * definitions for dealing with endianness:
 *
 *    SKEIN_HIP_NEED_SWAP:  0 for little-endian, 1 for big-endian
 *    Skein_Put64_LSB_First
 *    Skein_Get64_LSB_First
 *    Skein_Swap64
 *
 * If SKEIN_HIP_NEED_SWAP is defined at compile time, it is used here
 * along with the portable versions of Put64/Get64/Swap64, which 
 * are slow in general.
 *
 * Otherwise, an "auto-detect" of endianness is attempted below.
 * If the default handling doesn't work well, the user may insert
 * platform-specific code instead (e.g., for big-endian CPUs).
 *
 */
#ifndef SKEIN_HIP_NEED_SWAP /* compile-time "override" for endianness? */


// #include "int-util.h"

// Copyright (c) 2012-2013 The Cryptonote developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#pragma once
#ifndef INT_UTILS_H_
#define INT_UTILS_H_

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#ifndef _MSC_VER
#include <sys/param.h>
#else
#define inline __inline
#endif

#ifndef LITTLE_ENDIAN
#define LITTLE_ENDIAN 0x1234
#define BIG_ENDIAN 0x4321
#endif

#if !defined(BYTE_ORDER) && (defined(__LITTLE_ENDIAN__) || defined(__arm__) || defined(WIN32))
#define BYTE_ORDER LITTLE_ENDIAN
#endif

static __device__ __forceinline__ uint32_t rol32(uint32_t x, int r) {
  return (x << (r & 31)) | (x >> (-r & 31));
}

static __device__ __forceinline__ uint64_t rol64(const uint64_t vv, const uint8_t r)
{
  // Extract the lower 32 bits and higher 32 bits of the 64-bit number
  uint32_t lo = (uint32_t)(vv & 0xFFFFFFFF); // Lower 32 bits
  uint32_t hi = (uint32_t)(vv >> 32);        // Higher 32 bits
#ifdef __HIP_PLATFORM_AMD__
  if (r <= 32)
  {
    // Rotate within the 64-bit number using the lower and higher 32-bit parts
    uint32_t result_lo = __builtin_amdgcn_alignbit(lo, hi, 32 - r);
    uint32_t result_hi = __builtin_amdgcn_alignbit(hi, lo, 32 - r);
    return ((uint64_t)result_hi << 32) | result_lo;
  }
  else
  {
    // If r > 32, rotate the opposite way (flip the roles of lo and hi)
    uint32_t result_lo = __builtin_amdgcn_alignbit(hi, lo, 64 - r);
    uint32_t result_hi = __builtin_amdgcn_alignbit(lo, hi, 64 - r);
    return ((uint64_t)result_hi << 32) | result_lo;
  }
#else
  uint32_t result_lo, result_hi;

  if (r <= 32)
  {
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, 0xC8;\n\t" // Rotate lo by r (C8 is ((~A & B) | C), simulates shift and rotate)
        "lop3.b32 %0, %0, %4, %5, 0xC8;\n\t" // Rotate hi by (32 - r)
        : "=r"(result_lo)
        : "r"(lo), "r"(hi), "r"(32 - r), "r"(hi), "r"(32 - r));

    asm volatile(
        "lop3.b32 %0, %1, %2, %3, 0xC8;\n\t" // Rotate hi by r
        "lop3.b32 %0, %0, %4, %5, 0xC8;\n\t" // Rotate lo by (32 - r)
        : "=r"(result_hi)
        : "r"(hi), "r"(lo), "r"(32 - r), "r"(lo), "r"(32 - r));

    return (static_cast<uint64_t>(result_hi) << 32) | result_lo;
  }
  else
  {
    uint32_t shift_r = r - 32;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, 0xC8;\n\t" // Rotate hi by shift_r
        "lop3.b32 %0, %0, %4, %5, 0xC8;\n\t" // Rotate lo by (64 - r)
        : "=r"(result_lo)
        : "r"(hi), "r"(lo), "r"(shift_r), "r"(lo), "r"(shift_r));

    asm volatile(
        "lop3.b32 %0, %1, %2, %3, 0xC8;\n\t" // Rotate lo by shift_r
        "lop3.b32 %0, %0, %4, %5, 0xC8;\n\t" // Rotate hi by (64 - r)
        : "=r"(result_hi)
        : "r"(lo), "r"(hi), "r"(shift_r), "r"(hi), "r"(shift_r));

    return (static_cast<uint64_t>(result_hi) << 32) | result_lo;
  }
#endif
}

static __device__ __forceinline__ uint64_t hi_dword(uint64_t val) {
  return val >> 32;
}

static __device__ __forceinline__ uint64_t lo_dword(uint64_t val) {
  return val & 0xFFFFFFFF;
}

static __device__ __forceinline__ uint64_t div_with_reminder(uint64_t dividend, uint32_t divisor, uint32_t* remainder) {
  dividend |= ((uint64_t)*remainder) << 32;
  *remainder = dividend % divisor;
  return dividend / divisor;
}

// Long division with 2^32 base
static __device__ __forceinline__ uint32_t div128_32(uint64_t dividend_hi, uint64_t dividend_lo, uint32_t divisor, uint64_t* quotient_hi, uint64_t* quotient_lo) {
  uint64_t dividend_dwords[4];
  uint32_t remainder = 0;

  dividend_dwords[3] = hi_dword(dividend_hi);
  dividend_dwords[2] = lo_dword(dividend_hi);
  dividend_dwords[1] = hi_dword(dividend_lo);
  dividend_dwords[0] = lo_dword(dividend_lo);

  *quotient_hi  = div_with_reminder(dividend_dwords[3], divisor, &remainder) << 32;
  *quotient_hi |= div_with_reminder(dividend_dwords[2], divisor, &remainder);
  *quotient_lo  = div_with_reminder(dividend_dwords[1], divisor, &remainder) << 32;
  *quotient_lo |= div_with_reminder(dividend_dwords[0], divisor, &remainder);

  return remainder;
}

#define IDENT32(x) ((uint32_t) (x))
#define IDENT64(x) ((uint64_t) (x))

#define SWAP32(x) ((((uint32_t) (x) & 0x000000ff) << 24) | \
  (((uint32_t) (x) & 0x0000ff00) <<  8) | \
  (((uint32_t) (x) & 0x00ff0000) >>  8) | \
  (((uint32_t) (x) & 0xff000000) >> 24))
#define SWAP64(x) ((((uint64_t) (x) & 0x00000000000000ff) << 56) | \
  (((uint64_t) (x) & 0x000000000000ff00) << 40) | \
  (((uint64_t) (x) & 0x0000000000ff0000) << 24) | \
  (((uint64_t) (x) & 0x00000000ff000000) <<  8) | \
  (((uint64_t) (x) & 0x000000ff00000000) >>  8) | \
  (((uint64_t) (x) & 0x0000ff0000000000) >> 24) | \
  (((uint64_t) (x) & 0x00ff000000000000) >> 40) | \
  (((uint64_t) (x) & 0xff00000000000000) >> 56))

static __device__ __forceinline__ uint32_t ident32(uint32_t x) { return x; }
static __device__ __forceinline__ uint64_t ident64(uint64_t x) { return x; }

static __device__ __forceinline__ uint32_t swap32(uint32_t x) {
  x = ((x & 0x00ff00ff) << 8) | ((x & 0xff00ff00) >> 8);
  return (x << 16) | (x >> 16);
}
static __device__ __forceinline__ uint64_t swap64(uint64_t x) {
  x = ((x & 0x00ff00ff00ff00ff) <<  8) | ((x & 0xff00ff00ff00ff00) >>  8);
  x = ((x & 0x0000ffff0000ffff) << 16) | ((x & 0xffff0000ffff0000) >> 16);
  return (x << 32) | (x >> 32);
}

#if defined(__GNUC__)
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif
static __device__ __forceinline__ void mem_inplace_ident(void *mem UNUSED, size_t n UNUSED) { }
#undef UNUSED

static __device__ __forceinline__ void mem_inplace_swap32(void *mem, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    ((uint32_t *) mem)[i] = swap32(((const uint32_t *) mem)[i]);
  }
}
static __device__ __forceinline__ void mem_inplace_swap64(void *mem, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    ((uint64_t *) mem)[i] = swap64(((const uint64_t *) mem)[i]);
  }
}

static __device__ __forceinline__ void memcpy_ident32(void *dst, const void *src, size_t n) {
  memcpy(dst, src, 4 * n);
}
static __device__ __forceinline__ void memcpy_ident64(void *dst, const void *src, size_t n) {
  memcpy(dst, src, 8 * n);
}

static __device__ __forceinline__ void memcpy_swap32(void *dst, const void *src, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    ((uint32_t *) dst)[i] = swap32(((const uint32_t *) src)[i]);
  }
}
static __device__ __forceinline__ void memcpy_swap64(void *dst, const void *src, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    ((uint64_t *) dst)[i] = swap64(((const uint64_t *) src)[i]);
  }
}

#if !defined(BYTE_ORDER) || !defined(LITTLE_ENDIAN) || !defined(BIG_ENDIAN)
static_assert(false, "BYTE_ORDER is undefined. Perhaps, GNU extensions are not enabled");
#endif

#if BYTE_ORDER == LITTLE_ENDIAN
#define SWAP32LE IDENT32
#define SWAP32BE SWAP32
#define swap32le ident32
#define swap32be swap32
#define mem_inplace_swap32le mem_inplace_ident
#define mem_inplace_swap32be mem_inplace_swap32
#define memcpy_swap32le memcpy_ident32
#define memcpy_swap32be memcpy_swap32
#define SWAP64LE IDENT64
#define SWAP64BE SWAP64
#define swap64le ident64
#define swap64be swap64
#define mem_inplace_swap64le mem_inplace_ident
#define mem_inplace_swap64be mem_inplace_swap64
#define memcpy_swap64le memcpy_ident64
#define memcpy_swap64be memcpy_swap64
#endif

#if BYTE_ORDER == BIG_ENDIAN
#define SWAP32BE IDENT32
#define SWAP32LE SWAP32
#define swap32be ident32
#define swap32le swap32
#define mem_inplace_swap32be mem_inplace_ident
#define mem_inplace_swap32le mem_inplace_swap32
#define memcpy_swap32be memcpy_ident32
#define memcpy_swap32le memcpy_swap32
#define SWAP64BE IDENT64
#define SWAP64LE SWAP64
#define swap64be ident64
#define swap64le swap64
#define mem_inplace_swap64be mem_inplace_ident
#define mem_inplace_swap64le mem_inplace_swap64
#define memcpy_swap64be memcpy_ident64
#define memcpy_swap64le memcpy_swap64
#endif

#endif /* INT_UTILS_H_ */

#define IS_BIG_ENDIAN      4321 /* byte 0 is most significant (mc68k) */
#define IS_LITTLE_ENDIAN   1234 /* byte 0 is least significant (i386) */

#if BYTE_ORDER == LITTLE_ENDIAN
#  define PLATFORM_BYTE_ORDER IS_LITTLE_ENDIAN
#endif

#if BYTE_ORDER == BIG_ENDIAN
#  define PLATFORM_BYTE_ORDER IS_BIG_ENDIAN
#endif

/* special handler for IA64, which may be either endianness (?)  */
/* here we assume little-endian, but this may need to be changed */
#if defined(__ia64) || defined(__ia64__) || defined(_M_IA64)
#  define PLATFORM_MUST_ALIGN (1)
#ifndef PLATFORM_BYTE_ORDER
#  define PLATFORM_BYTE_ORDER IS_LITTLE_ENDIAN
#endif
#endif

#ifndef   PLATFORM_MUST_ALIGN
#  define PLATFORM_MUST_ALIGN (0)
#endif


#if   PLATFORM_BYTE_ORDER == IS_BIG_ENDIAN
    /* here for big-endian CPUs */
#define SKEIN_HIP_NEED_SWAP   (1)
#elif PLATFORM_BYTE_ORDER == IS_LITTLE_ENDIAN
    /* here for x86 and x86-64 CPUs (and other detected little-endian CPUs) */
#define SKEIN_HIP_NEED_SWAP   (0)
#if   PLATFORM_MUST_ALIGN == 0              /* ok to use "fast" versions? */
#define Skein_Put64_LSB_First(dst08,src64,bCnt) memcpy(dst08,src64,bCnt)
#define Skein_Get64_LSB_First(dst64,src08,wCnt) memcpy(dst64,src08,8*(wCnt))
#endif
#else
#error "Skein needs endianness setting!"
#endif

#endif /* ifndef SKEIN_HIP_NEED_SWAP */

/*
 ******************************************************************
 *      Provide any definitions still needed.
 ******************************************************************
 */
#ifndef Skein_Swap64  /* swap for big-endian, nop for little-endian */
#if     SKEIN_HIP_NEED_SWAP
#define Skein_Swap64(w64)                       \
  ( (( ((u64b_t)(w64))       & 0xFF) << 56) |   \
    (((((u64b_t)(w64)) >> 8) & 0xFF) << 48) |   \
    (((((u64b_t)(w64)) >>16) & 0xFF) << 40) |   \
    (((((u64b_t)(w64)) >>24) & 0xFF) << 32) |   \
    (((((u64b_t)(w64)) >>32) & 0xFF) << 24) |   \
    (((((u64b_t)(w64)) >>40) & 0xFF) << 16) |   \
    (((((u64b_t)(w64)) >>48) & 0xFF) <<  8) |   \
    (((((u64b_t)(w64)) >>56) & 0xFF)      ) )
#else
#define Skein_Swap64(w64)  (w64)
#endif
#endif  /* ifndef Skein_Swap64 */


#ifndef Skein_Put64_LSB_First
void    Skein_Put64_LSB_First(u08b_t *dst,const u64b_t *src,size_t bCnt)
#ifdef  SKEIN_HIP_PORT_CODE /* instantiate the function code here? */
    { /* this version is fully portable (big-endian or little-endian), but slow */
    size_t n;

    for (n=0;n<bCnt;n++)
        dst[n] = (u08b_t) (src[n>>3] >> (8*(n&7)));
    }
#else
    ;    /* output only the function prototype */
#endif
#endif   /* ifndef Skein_Put64_LSB_First */


#ifndef Skein_Get64_LSB_First
void    Skein_Get64_LSB_First(u64b_t *dst,const u08b_t *src,size_t wCnt)
#ifdef  SKEIN_HIP_PORT_CODE /* instantiate the function code here? */
    { /* this version is fully portable (big-endian or little-endian), but slow */
    size_t n;

    for (n=0;n<8*wCnt;n+=8)
        dst[n/8] = (((u64b_t) src[n  ])      ) +
                   (((u64b_t) src[n+1]) <<  8) +
                   (((u64b_t) src[n+2]) << 16) +
                   (((u64b_t) src[n+3]) << 24) +
                   (((u64b_t) src[n+4]) << 32) +
                   (((u64b_t) src[n+5]) << 40) +
                   (((u64b_t) src[n+6]) << 48) +
                   (((u64b_t) src[n+7]) << 56) ;
    }
#else
    ;    /* output only the function prototype */
#endif
#endif   /* ifndef Skein_Get64_LSB_First */

#endif   /* ifndef _SKEIN_PORT_H_ */
