#include "bitselect.hip.h"
#if defined(__HIPCC_RTC__) || defined(__CUDACC_RTC__)
#include "uint2ops.h"
#else
#include <tnn_hip/util/uint2ops.h>
#endif

__device__ __constant__ uint2 const keccak_round_constants[24] = {
    {0x00000001, 0x00000000}, {0x00008082, 0x00000000}, {0x0000808a, 0x80000000}, {0x80008000, 0x80000000}, {0x0000808b, 0x00000000}, {0x80000001, 0x00000000}, {0x80008081, 0x80000000}, {0x00008009, 0x80000000}, {0x0000008a, 0x00000000}, {0x00000088, 0x00000000}, {0x80008009, 0x00000000}, {0x8000000a, 0x00000000}, {0x8000808b, 0x00000000}, {0x0000008b, 0x80000000}, {0x00008089, 0x80000000}, {0x00008003, 0x80000000}, {0x00008002, 0x80000000}, {0x00000080, 0x80000000}, {0x0000800a, 0x00000000}, {0x8000000a, 0x80000000}, {0x80008081, 0x80000000}, {0x00008080, 0x80000000}, {0x80000001, 0x00000000}, {0x80008008, 0x80000000}};

static __device__ __forceinline__ void xor5(const uint2 a, const uint2 b, const uint2 c, const uint2 d, const uint2 e, uint2 &result)
{
  result.x = a.x ^ b.x ^ c.x ^ d.x ^ e.x;
  result.y = a.y ^ b.y ^ c.y ^ d.y ^ e.y;
}

static __device__ __forceinline__ void xor3(const uint2 a, const uint2 b, const uint2 c, uint2 &result)
{
  result.x = a.x ^ b.x ^ c.x;
  result.y = a.y ^ b.y ^ c.y;
}

#define REPEAT6(e) e e e e e e
#define REPEAT24(e) REPEAT6(e e e e)
#define REPEAT23(e) REPEAT6(e e e) e e e e e
#define REPEAT5(e) e e e e e
#define FOR5(v, s, e) \
  v = 0;              \
  REPEAT5(e; v += s;)

static __device__ __forceinline__ void keccakf(void *state)
{
  uint2 *a = (uint2 *)state;
  uint2 s[25];
  uint2 b[5];
  uint2 u;

  memset(b, 0, sizeof(b));

  uint2 t = make_uint2(0,0), v = make_uint2(0,0);
  uint8_t x, y;

  // Copy the input state directly without re-initializing it
  #pragma unroll
  for (int i = 0; i < 25; ++i)
    s[i] = a[i]; // Assuming state is already fully initialized with padding and length bits

    // Perform 24 rounds of the Keccak permutation
#pragma unroll
  for (int i = 0; i < 23; i++)
  {
    // Theta
    xor5(s[0], s[5], s[10], s[15], s[20], b[0]);
    xor5(s[1], s[6], s[11], s[16], s[21], b[1]);
    xor5(s[2], s[7], s[12], s[17], s[22], b[2]);
    xor5(s[3], s[8], s[13], s[18], s[23], b[3]);
    xor5(s[4], s[9], s[14], s[19], s[24], b[4]);

    u = b[4] ^ rol64(b[1], 1);
    s[0] ^= u;
    s[5] ^= u;
    s[10] ^= u;
    s[15] ^= u;
    s[20] ^= u;

    u = b[0] ^ rol64(b[2], 1);
    s[1] ^= u;
    s[6] ^= u;
    s[11] ^= u;
    s[16] ^= u;
    s[21] ^= u;

    u = b[1] ^ rol64(b[3], 1);
    s[2] ^= u;
    s[7] ^= u;
    s[12] ^= u;
    s[17] ^= u;
    s[22] ^= u;

    u = b[2] ^ rol64(b[4], 1);
    s[3] ^= u;
    s[8] ^= u;
    s[13] ^= u;
    s[18] ^= u;
    s[23] ^= u;

    u = b[3] ^ rol64(b[0], 1);
    s[4] ^= u;
    s[9] ^= u;
    s[14] ^= u;
    s[19] ^= u;
    s[24] ^= u;

    // Rho and pi
    u = s[1];
    s[1] = rol64(s[6], 44);
    s[6] = rol64(s[9], 20);
    s[9] = rol64(s[22], 61);
    s[22] = rol64(s[14], 39);
    s[14] = rol64(s[20], 18);
    s[20] = rol64(s[2], 62);
    s[2] = rol64(s[12], 43);
    s[12] = rol64(s[13], 25);
    s[13] = rol64(s[19], 8);
    s[19] = rol64(s[23], 56);
    s[23] = rol64(s[15], 41);
    s[15] = rol64(s[4], 27);
    s[4] = rol64(s[24], 14);
    s[24] = rol64(s[21], 2);
    s[21] = rol64(s[8], 55);
    s[8] = rol64(s[16], 45);
    s[16] = rol64(s[5], 36);
    s[5] = rol64(s[3], 28);
    s[3] = rol64(s[18], 21);
    s[18] = rol64(s[17], 15);
    s[17] = rol64(s[11], 10);
    s[11] = rol64(s[7], 6);
    s[7] = rol64(s[10], 3);
    s[10] = rol64(u, 1);

    // Chi
    FOR5(y, 5,
         v = s[y];
         t = s[y + 1];
         bitselect(s[y] ^ s[y + 2], s[y], s[y + 1], s[y]);
         bitselect(s[y + 1] ^ s[y + 3], s[y + 1], s[y + 2], s[y + 1]);
         bitselect(s[y + 2] ^ s[y + 4], s[y + 2], s[y + 3], s[y + 2]);
         bitselect(s[y + 3] ^ v, s[y + 3], s[y + 4], s[y + 3]);
         bitselect(s[y + 4] ^ t, s[y + 4], v, s[y + 4]);)

    // Iota
    s[0] ^= keccak_round_constants[i];
  }
  // Theta
  xor5(s[0], s[5], s[10], s[15], s[20], b[0]);
  xor5(s[1], s[6], s[11], s[16], s[21], b[1]);
  xor5(s[2], s[7], s[12], s[17], s[22], b[2]);
  xor5(s[3], s[8], s[13], s[18], s[23], b[3]);
  xor5(s[4], s[9], s[14], s[19], s[24], b[4]);

  v = b[4];
  t = b[0];
  b[4] = b[4] ^ rol64(b[1], 1);
  b[0] = b[0] ^ rol64(b[2], 1);
  b[1] = b[1] ^ rol64(b[3], 1);
  b[2] = b[2] ^ rol64(v, 1);
  b[3] = b[3] ^ rol64(t, 1);

  s[0] ^= b[4];
  s[1] ^= b[0];
  s[6] ^= b[0];
  s[2] ^= b[1];
  s[12] ^= b[1];
  s[3] ^= b[2];
  s[18] ^= b[2];
  s[4] ^= b[3];
  s[24] ^= b[3];

  // Rho and pi
  s[1] = rol64(s[6], 44);
  s[2] = rol64(s[12], 43);
  s[4] = rol64(s[24], 14);
  s[3] = rol64(s[18], 21);

  // Chi
  v = s[0];

  bitselect(s[0] ^ s[2], s[0], s[1], s[0]);
  bitselect(s[1] ^ s[3], s[1], s[2], s[1]);
  bitselect(s[2] ^ s[4], s[2], s[3], s[2]);
  bitselect(s[3] ^ v, s[3], s[4], s[3]);

  // Iota
  s[0] ^= keccak_round_constants[23];

  // Finalize state back into `a`
  for (int i = 0; i < 25; ++i)
    a[i] = s[i]; // Copy the transformed state back to `a`
}

// ============================================================================
// keccak_f800 — Optimized Keccak-f[800] (25 × 32-bit, 22 rounds)
//
// Same techniques as the f1600: in-place ρ/π chain (1 temp, not 25),
// BFI-pattern χ, and last-round pruning for 8-word output.
// ============================================================================

__device__ __constant__ static const uint32_t keccak_f800_rc[22] = {
    0x00000001u, 0x00008082u, 0x0000808au, 0x80008000u,
    0x0000808bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008au, 0x00000088u, 0x80008009u, 0x8000000au,
    0x8000808bu, 0x0000008bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800au, 0x8000000au,
    0x80008081u, 0x00008080u
};

static __device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n)
{
    return (x << n) | (x >> (32u - n));
}

static __device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32u - n));
}

// χ step: a ^ (~b & c)
// AMD:    v_xor + v_bfi_b32 (2 instructions)
// NVIDIA: lop3.b32 0xD2     (1 instruction)
static __device__ __forceinline__ uint32_t chi32(uint32_t a, uint32_t b, uint32_t c)
{
#ifdef __HIP_PLATFORM_AMD__
    uint32_t result;
    asm volatile("v_bfi_b32 %0, %1, %2, %3"
                 : "=v"(result) : "v"(b), "v"(a), "v"(a ^ c));
    return result;
#else
    uint32_t result;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xD2;"
                 : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
#endif
}

// 5-way XOR for θ column parity
// AMD:    4 v_xor (compiler handles)
// NVIDIA: 2 lop3.b32 0x96 (3-input XOR truth table)
static __device__ __forceinline__ uint32_t xor5_32(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e)
{
#ifndef __HIP_PLATFORM_AMD__
    uint32_t result;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, 0x96;\n\t"
        "lop3.b32 %0, %0, %4, %5, 0x96;"
        : "=r"(result) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e));
    return result;
#else
    return a ^ b ^ c ^ d ^ e;
#endif
}

static __device__ void keccak_f800(uint32_t* __restrict__ st)
{
    uint32_t s[25], b[5], u;

    #pragma unroll
    for (int i = 0; i < 25; i++) s[i] = st[i];

    // ---- Rounds 0..20 (21 full rounds) ----
    #pragma unroll
    for (int round = 0; round < 21; round++)
    {
        // θ — column parity
        b[0] = xor5_32(s[0], s[5], s[10], s[15], s[20]);
        b[1] = xor5_32(s[1], s[6], s[11], s[16], s[21]);
        b[2] = xor5_32(s[2], s[7], s[12], s[17], s[22]);
        b[3] = xor5_32(s[3], s[8], s[13], s[18], s[23]);
        b[4] = xor5_32(s[4], s[9], s[14], s[19], s[24]);

        u = b[4] ^ rotl32(b[1], 1);
        s[0]  ^= u; s[5]  ^= u; s[10] ^= u; s[15] ^= u; s[20] ^= u;
        u = b[0] ^ rotl32(b[2], 1);
        s[1]  ^= u; s[6]  ^= u; s[11] ^= u; s[16] ^= u; s[21] ^= u;
        u = b[1] ^ rotl32(b[3], 1);
        s[2]  ^= u; s[7]  ^= u; s[12] ^= u; s[17] ^= u; s[22] ^= u;
        u = b[2] ^ rotl32(b[4], 1);
        s[3]  ^= u; s[8]  ^= u; s[13] ^= u; s[18] ^= u; s[23] ^= u;
        u = b[3] ^ rotl32(b[0], 1);
        s[4]  ^= u; s[9]  ^= u; s[14] ^= u; s[19] ^= u; s[24] ^= u;

        // ρ/π in-place chain (1 temp instead of 25)
        u     = s[1];
        s[1]  = rotl32(s[6],  12);
        s[6]  = rotl32(s[9],  20);
        s[9]  = rotl32(s[22], 29);
        s[22] = rotl32(s[14],  7);
        s[14] = rotl32(s[20], 18);
        s[20] = rotl32(s[2],  30);
        s[2]  = rotl32(s[12], 11);
        s[12] = rotl32(s[13], 25);
        s[13] = rotl32(s[19],  8);
        s[19] = rotl32(s[23], 24);
        s[23] = rotl32(s[15],  9);
        s[15] = rotl32(s[4],  27);
        s[4]  = rotl32(s[24], 14);
        s[24] = rotl32(s[21],  2);
        s[21] = rotl32(s[8],  23);
        s[8]  = rotl32(s[16], 13);
        s[16] = rotl32(s[5],   4);
        s[5]  = rotl32(s[3],  28);
        s[3]  = rotl32(s[18], 21);
        s[18] = rotl32(s[17], 15);
        s[17] = rotl32(s[11], 10);
        s[11] = rotl32(s[7],   6);
        s[7]  = rotl32(s[10],  3);
        s[10] = rotl32(u,      1);

        // χ — platform-optimized (LOP3 on NVIDIA, BFI on AMD)
        #pragma unroll
        for (int y = 0; y < 25; y += 5)
        {
            uint32_t v = s[y], t = s[y + 1];
            s[y]     = chi32(s[y],     s[y + 1], s[y + 2]);
            s[y + 1] = chi32(s[y + 1], s[y + 2], s[y + 3]);
            s[y + 2] = chi32(s[y + 2], s[y + 3], s[y + 4]);
            s[y + 3] = chi32(s[y + 3], s[y + 4], v);
            s[y + 4] = chi32(s[y + 4], v,        t);
        }

        // ι
        s[0] ^= keccak_f800_rc[round];
    }

    // ---- Round 21 (last) — pruned for 8-word output ----
    // θ: full column parity (needs all 25 words)
    b[0] = xor5_32(s[0], s[5], s[10], s[15], s[20]);
    b[1] = xor5_32(s[1], s[6], s[11], s[16], s[21]);
    b[2] = xor5_32(s[2], s[7], s[12], s[17], s[22]);
    b[3] = xor5_32(s[3], s[8], s[13], s[18], s[23]);
    b[4] = xor5_32(s[4], s[9], s[14], s[19], s[24]);

    uint32_t d0 = b[4] ^ rotl32(b[1], 1);
    uint32_t d1 = b[0] ^ rotl32(b[2], 1);
    uint32_t d2 = b[1] ^ rotl32(b[3], 1);
    uint32_t d3 = b[2] ^ rotl32(b[4], 1);
    uint32_t d4 = b[3] ^ rotl32(b[0], 1);

    // Fused θ + ρ/π — only the 10 positions feeding output rows 0–1
    uint32_t t0 = s[0]  ^ d0;
    uint32_t t1 = rotl32(s[6]  ^ d1, 12);
    uint32_t t2 = rotl32(s[12] ^ d2, 11);
    uint32_t t3 = rotl32(s[18] ^ d3, 21);
    uint32_t t4 = rotl32(s[24] ^ d4, 14);
    uint32_t t5 = rotl32(s[3]  ^ d3, 28);
    uint32_t t6 = rotl32(s[9]  ^ d4, 20);
    uint32_t t7 = rotl32(s[10] ^ d0,  3);
    uint32_t t8 = rotl32(s[16] ^ d1, 13);
    uint32_t t9 = rotl32(s[22] ^ d2, 29);

    // χ rows 0–1 only (positions 0–7 are the output)
    st[0] = chi32(t0, t1, t2) ^ keccak_f800_rc[21];  // fused ι
    st[1] = chi32(t1, t2, t3);
    st[2] = chi32(t2, t3, t4);
    st[3] = chi32(t3, t4, t0);
    st[4] = chi32(t4, t0, t1);
    st[5] = chi32(t5, t6, t7);
    st[6] = chi32(t6, t7, t8);
    st[7] = chi32(t7, t8, t9);
}