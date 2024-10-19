/** libkeccak-tiny
 *
 * A single-file implementation of SHA-3 and SHAKE.
 *
 * Implementor: David Leon Gil
 * License: CC0, attribution kindly requested. Blame taken too,
 * but not liability.
 */
#define __STDC_WANT_LIB_EXT1__ 1

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitselect.hip.h"

#define DEV_INLINE __device__ static __forceinline__

/******** The Keccak-f[1600] permutation ********/

/*** Constants. ***/
__device__ static constexpr uint8_t rho[24] =
    {1, 3, 6, 10, 15, 21,
     28, 36, 45, 55, 2, 14,
     27, 41, 56, 8, 25, 43,
     62, 18, 39, 61, 20, 44};
__device__ static constexpr uint8_t pi[24] =
    {10, 7, 11, 17, 18, 3,
     5, 16, 8, 21, 24, 4,
     15, 23, 19, 13, 12, 2,
     20, 14, 22, 9, 6, 1};
__device__ static const uint64_t RC[24] =
    {1ULL, 0x8082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
     0x808bULL, 0x80000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
     0x8aULL, 0x88ULL, 0x80008009ULL, 0x8000000aULL,
     0x8000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
     0x8000000000008002ULL, 0x8000000000000080ULL, 0x800aULL, 0x800000008000000aULL,
     0x8000000080008081ULL, 0x8000000000008080ULL, 0x80000001ULL, 0x8000000080008008ULL};

#ifdef __HIP_PLATFORM_AMD__
DEV_INLINE uint64_t xor5(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d, const uint64_t e)
{
  return a ^ b ^ c ^ d ^ e;
}
#else
DEV_INLINE uint64_t xor5(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d, const uint64_t e)
{
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
  uint32_t result_lo, result_hi;
  
  asm volatile (
      "// xor5\n\t"
      "lop3.b32 %0, %2, %3, %4, 0x96;\n\t"
      "lop3.b32 %0, %0, %5, %6, 0x96;\n\t"
      "lop3.b32 %1, %7, %8, %9, 0x96;\n\t"
      "lop3.b32 %1, %1, %10, %11, 0x96;"
      : "=r"(result_lo), "=r"(result_hi)
      : "r"(static_cast<uint32_t>(a & 0xFFFFFFFF)), 
        "r"(static_cast<uint32_t>(b & 0xFFFFFFFF)), 
        "r"(static_cast<uint32_t>(c & 0xFFFFFFFF)),
        "r"(static_cast<uint32_t>(d & 0xFFFFFFFF)),
        "r"(static_cast<uint32_t>(e & 0xFFFFFFFF)),
        "r"(static_cast<uint32_t>(a >> 32)), 
        "r"(static_cast<uint32_t>(b >> 32)), 
        "r"(static_cast<uint32_t>(c >> 32)),
        "r"(static_cast<uint32_t>(d >> 32)),
        "r"(static_cast<uint32_t>(e >> 32)));

  return (static_cast<uint64_t>(result_hi) << 32) | result_lo;
#else
  return a ^ b ^ c ^ d ^ e;
#endif
}
#endif

/*** Helper macros to unroll the permutation. ***/
DEV_INLINE uint64_t rol(const uint64_t vv, const uint8_t r)
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

#define REPEAT6(e) e e e e e e
#define REPEAT24(e) REPEAT6(e e e e)
#define REPEAT23(e) REPEAT6(e e e) e e e e e
#define REPEAT5(e) e e e e e
#define FOR5(v, s, e) \
  v = 0;              \
  REPEAT5(e; v += s;)

/*** Keccak-f[1600] ***/
DEV_INLINE void keccakf(void *state)
{
  uint64_t *a = (uint64_t *)state;
  uint64_t b[5] = {0};
  uint64_t t = 0, v = 0;
  uint8_t x, y;

#pragma unroll
  for (int i = 0; i < 23; i++)
  {
    // Theta
    FOR5(x, 1,
         b[x] = xor5(a[x], a[x + 5], a[x + 10], a[x + 15], a[x + 20]);)

    v = b[4];
    t = b[0];
    b[4] = b[4] ^ rol(b[1], 1);
    b[0] = b[0] ^ rol(b[2], 1);
    b[1] = b[1] ^ rol(b[3], 1);
    b[2] = b[2] ^ rol(v, 1);
    b[3] = b[3] ^ rol(t, 1);

    FOR5(x, 1,
         FOR5(y, 5, a[y + x] ^= b[(x + 4) % 5];))

    // Rho and pi
    t = a[1];
    x = 23;
    REPEAT23(a[pi[x]] = rol(a[pi[x - 1]], rho[x]); x--;)
    a[pi[0]] = rol(t, rho[0]);

    FOR5(y, 5,
         v = a[y];
         t = a[y + 1];
         a[y] = bitselect(a[y] ^ a[y + 2], a[y], a[y + 1]);
         a[y + 1] = bitselect(a[y + 1] ^ a[y + 3], a[y + 1], a[y + 2]);
         a[y + 2] = bitselect(a[y + 2] ^ a[y + 4], a[y + 2], a[y + 3]);
         a[y + 3] = bitselect(a[y + 3] ^ v, a[y + 3], a[y + 4]);
         a[y + 4] = bitselect(a[y + 4] ^ t, a[y + 4], v);)

    // Iota
    a[0] ^= RC[i];
  }
  // Theta
  FOR5(x, 1,
       b[x] = xor5(a[x], a[x + 5], a[x + 10], a[x + 15], a[x + 20]);)

  v = b[4];
  t = b[0];
  b[4] = b[4] ^ rol(b[1], 1);
  b[0] = b[0] ^ rol(b[2], 1);
  b[1] = b[1] ^ rol(b[3], 1);
  b[2] = b[2] ^ rol(v, 1);
  b[3] = b[3] ^ rol(t, 1);

  a[0] ^= b[4];
  a[1] ^= b[0];
  a[6] ^= b[0];
  a[2] ^= b[1];
  a[12] ^= b[1];
  a[3] ^= b[2];
  a[18] ^= b[2];
  a[4] ^= b[3];
  a[24] ^= b[3];

  // Rho and pi
  a[1] = rol(a[pi[22]], rho[23]);
  a[2] = rol(a[pi[16]], rho[17]);
  a[4] = rol(a[pi[10]], rho[11]);
  a[3] = rol(a[pi[4]], rho[5]);

  // Chi
  v = a[0];

  a[0] = bitselect(a[0] ^ a[2], a[0], a[1]);
  a[1] = bitselect(a[1] ^ a[3], a[1], a[2]);
  a[2] = bitselect(a[2] ^ a[4], a[2], a[3]);
  a[3] = bitselect(a[3] ^ v, a[3], a[4]);

  // Iota
  a[0] ^= RC[23];
}

/******** The FIPS202-defined functions. ********/

/*** Some helper macros. ***/

// #define _(S) do { S } while (0)
// #define FOR(i, ST, L, S) \
//   _(for (size_t i = 0; i < L; i += ST) { S; })
// #define mkapply_ds(NAME, S)                                          \
//   DEV_INLINE void NAME(uint8_t* dst,                              \
//                           const uint8_t* src,                        \
//                           size_t len) {                              \
//     FOR(i, 1, len, S);                                               \
//   }
// #define mkapply_sd(NAME, S)                                          \
//   DEV_INLINE void NAME(const uint8_t* src,                        \
//                           uint8_t* dst,                              \
//                           size_t len) {                              \
//     FOR(i, 1, len, S);                                               \
//   }

// mkapply_ds(xorin, dst[i] ^= src[i])  // xorin
// mkapply_sd(setout, dst[i] = src[i])  // setout

#define P keccakf
#define Plen 200

// #define foldP(I, L, F) \
//   while (L >= rate) {  \
//     F(a, I, rate);     \
//     P(a);              \
//     I += rate;         \
//     L -= rate;         \
//   }

// /** The sponge-based hash construction. **/
// __device__ __forceinline__ static void hash(uint8_t* out, size_t outlen,
//                        const uint8_t* in, size_t inlen,
//                        size_t rate, uint8_t delim) {
//   uint8_t a[Plen] = {0};
//   // Absorb input.
//   foldP(in, inlen, xorin);
//   // Xor in the DS and pad frame.
//   a[inlen] ^= delim;
//   a[rate - 1] ^= 0x80;
//   // Xor in the last block.
//   xorin(a, in, inlen);
//   // Apply P
//   P(a);
//   // Squeeze output.
//   foldP(out, outlen, setout);
//   setout(a, out, outlen);
// }

// __device__ __forceinline__ static void shake32_heavy(
//                        const uint8_t initP[Plen],
//                        uint8_t* out,
//                        const uint8_t* in) {
//   uint8_t a[Plen];

//   #pragma unroll
//   for (int i=0; i<10; i++) ((uint64_t *)a)[i] = ((uint64_t *)initP)[i] ^ ((uint64_t *)in)[i];
//   // ((ulonglong4 *)a)[0] = ((ulonglong4 *)initP)[0] ^ ((ulonglong4 *)in)[0];
//   // ((ulonglong4 *)a)[1] = ((ulonglong4 *)initP)[1] ^ ((ulonglong4 *)in)[1];
//   // ((ulonglong2 *)a)[4] = ((ulonglong2 *)initP)[4] ^ ((ulonglong4 *)in)[4];

//   #pragma unroll
//   for (int i=10; i<25; i++) ((uint64_t *)a)[i] = ((uint64_t *)initP)[i];
//   // ((ulonglong2 *)a)[5] = ((ulonglong2 *)initP)[5];

//   // #pragma unroll
//   // for (int i = 3; i < 6; i++) ((ulonglong4 *)a)[i] = ((ulonglong4 *)initP)[i];

//   // ((ulonglong2 *)a)[12] = ((ulonglong2 *)initP)[12];
//   // ((uint64_t *)a)[24] = ((uint64_t *)initP)[24];

//   P(a);
//   // ((ulonglong4*)out)[0] = ((ulonglong4*)a)[0];
//   memcpy(out, a, 32);
// }

// /*** Helper macros to define SHA3 and SHAKE instances. ***/
// #define defsha3(bits)                                             \
//   __device__ __forceinline__ static void sha3_##bits(uint8_t* out, size_t outlen,                    \
//                   const uint8_t* in, size_t inlen) {              \
//     hash(out, outlen, in, inlen, 200 - (bits / 4), 0x06);  \
//   }

// /*** FIPS202 SHA3 FOFs ***/
// defsha3(224)
// defsha3(256)
// defsha3(384)
// defsha3(512)