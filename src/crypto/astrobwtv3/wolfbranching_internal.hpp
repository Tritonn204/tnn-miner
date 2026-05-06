#pragma once
/*
 * wolfbranching_internal.hpp - Shared macros for spec-generated wolf tier TUs.
 *
 * Contains:
 *   - WOLF_BRANCH_SIMD          per-byte SIMD branch macro (parameterised popcnt)
 *   - WOLF_COMPUTE_BODY         full wolfCompute loop body (calls wolfPermute)
 *
 * The spec system's cross-reference rewriter turns bare `wolfPermute(...)` calls
 * inside tier blocks into `wolfPermute_<tier>(...)`, so the macro body does not
 * need to know which tier it lives in.
 */

#include "astrobwtv3.h"
#include "simd_util.hpp"

/* ---- includes needed by WOLF_COMPUTE_BODY ---- */
#include <fnv1a.h>
#include <xxhash64.h>
#include <highwayhash/sip_hash.h>

/* ---- constants used in WOLF_COMPUTE_BODY ---- */
#ifndef MINPREFLEN
#define MINPREFLEN 4
#endif

/* ---- external data defined in wolfbranching.cpp ---- */
extern uint32_t CodeLUT[257];
extern uint16_t *CodeLUT_16;
extern uint8_t wolfBranch(uint8_t val, uint8_t pos2val, uint32_t opcode);

/* ---- copyChunkData is defined per-tier in the spec (or in astrobwtv3.cpp
 *      for non-x86 builds).  Each TU that expands WOLF_COMPUTE_BODY must
 *      have a definition visible — the spec system handles this. ---- */

/* -----------------------------------------------------------------------
 * WOLF_POPCNT256_EPI8 -- override BEFORE including this header (or before
 * expanding WOLF_BRANCH_SIMD) to use a native instruction on AVX-512 BITALG.
 * Default: LUT-based popcnt256_epi8 from simd_util.hpp.
 * ----------------------------------------------------------------------- */
#ifndef WOLF_POPCNT256_EPI8
#define WOLF_POPCNT256_EPI8(x) popcnt256_epi8(x)
#endif

/* -----------------------------------------------------------------------
 * WOLF_BRANCH_SIMD -- SIMD branch ops on a __m256i value.
 * Same as WOLF_BRANCH_AVX2 but uses WOLF_POPCNT256_EPI8 so each tier can
 * override the popcount implementation.
 * ----------------------------------------------------------------------- */
#if defined(__x86_64__)

#define WOLF_BRANCH_SIMD(IN, POS2VAL, OPCODE)                                   \
  {                                                                              \
    const __m256i _wb_vec3 = _mm256_set1_epi8(3);                                \
    for (int _wb_i = 3; _wb_i >= 0; --_wb_i) {                                  \
      uint8_t _wb_insn = ((OPCODE) >> (_wb_i << 2)) & 0xF;                      \
      switch (_wb_insn) {                                                        \
        case 0:  (IN) = _mm256_add_epi8((IN), (IN)); break;                      \
        case 1:  (IN) = _mm256_sub_epi8((IN),                                    \
                     _mm256_xor_si256((IN), _mm256_set1_epi8(97))); break;       \
        case 2:  (IN) = _mm256_mul_epi8((IN), (IN)); break;                      \
        case 3:  (IN) = _mm256_xor_si256((IN),                                   \
                     _mm256_set1_epi8((POS2VAL))); break;                        \
        case 4:  (IN) = _mm256_xor_si256((IN),                                   \
                     _mm256_set1_epi64x(-1LL)); break;                           \
        case 5:  (IN) = _mm256_and_si256((IN),                                   \
                     _mm256_set1_epi8((POS2VAL))); break;                        \
        case 6:  (IN) = _mm256_sllv_epi8((IN),                                   \
                     _mm256_and_si256((IN), _wb_vec3)); break;                   \
        case 7:  (IN) = _mm256_srlv_epi8((IN),                                   \
                     _mm256_and_si256((IN), _wb_vec3)); break;                   \
        case 8:  (IN) = _mm256_reverse_epi8((IN)); break;                        \
        case 9:  (IN) = _mm256_xor_si256((IN),                                   \
                     WOLF_POPCNT256_EPI8((IN))); break;                          \
        case 10: (IN) = _mm256_rolv_epi8((IN), (IN)); break;                     \
        case 11: (IN) = _mm256_rol_epi8((IN), 1); break;                         \
        case 12: (IN) = _mm256_xor_si256((IN),                                   \
                     _mm256_rol_epi8((IN), 2)); break;                           \
        case 13: (IN) = _mm256_rol_epi8((IN), 3); break;                         \
        case 14: (IN) = _mm256_xor_si256((IN),                                   \
                     _mm256_rol_epi8((IN), 4)); break;                           \
        case 15: (IN) = _mm256_rol_epi8((IN), 5); break;                         \
      }                                                                          \
    }                                                                            \
  }

#endif /* __x86_64__ */

/* -----------------------------------------------------------------------
 * WOLF_COMPUTE_BODY -- the full wolfCompute hot loop.
 *
 * Expands inside:
 *   void wolfCompute(...) { WOLF_COMPUTE_BODY(wolfPermute) }
 *
 * The PERMUTE_FN parameter is the wolfPermute function to call.
 * In the spec file, gen_tiers.py's cross-ref rewriter sees the bare
 * `wolfPermute` token in `WOLF_COMPUTE_BODY(wolfPermute)` and rewrites
 * it to `wolfPermute_<tier>`, so the macro ends up calling the tier-local
 * variant at compile time.
 *
 * Relies on the enclosing TU providing:
 *   - PERMUTE_FN (wolfPermute or tier-suffixed variant)
 *   - COPY_FN    (copyChunkData or tier-suffixed variant)
 *   - XXHash64, hash_64_fnv1a, highwayhash::SipHash, RC4, RC4_set_key
 * ----------------------------------------------------------------------- */
#define WOLF_COMPUTE_BODY(PERMUTE_FN, COPY_FN)                                   \
  byte prevOp;                                                                   \
  int changeCount = 0;                                                           \
                                                                                 \
  worker.templateIdx = 0;                                                        \
  uint8_t chunkCount = 1;                                                        \
  int firstChunk = 0;                                                            \
                                                                                 \
  uint8_t lp1 = 0;                                                               \
  uint8_t lp2 = 255;                                                             \
                                                                                 \
  worker.tries[wIndex] = 0;                                                      \
  for (int it = 0; it < 278; ++it)                                               \
  {                                                                              \
      worker.tries[wIndex]++;                                                    \
      worker.random_switcher = worker.prev_lhash ^ worker.lhash                  \
                             ^ worker.tries[wIndex];                             \
                                                                                 \
      prevOp = worker.op;                                                        \
      worker.op = static_cast<byte>(worker.random_switcher);                     \
                                                                                 \
      byte p1 = static_cast<byte>(worker.random_switcher >> 8);                  \
      byte p2 = static_cast<byte>(worker.random_switcher >> 16);                 \
                                                                                 \
      if (p1 > p2)                                                               \
      {                                                                          \
        std::swap(p1, p2);                                                       \
      }                                                                          \
                                                                                 \
      if (p2 - p1 > 32)                                                          \
      {                                                                          \
        p2 = p1 + ((p2 - p1) & 0x1f);                                            \
      }                                                                          \
                                                                                 \
      if (worker.tries[wIndex] > 0) {                                            \
        lp1 = std::min(lp1, p1);                                                 \
        lp2 = std::max(lp2, p2);                                                 \
      }                                                                          \
                                                                                 \
      if (p1 < worker.pos1 || p2 > worker.pos2)                                  \
        { worker.isSame = false; changeCount++; }                                \
                                                                                 \
      worker.pos1 = p1;                                                          \
      worker.pos2 = p2;                                                          \
                                                                                 \
      worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE                   \
                                   + (worker.tries[wIndex] - 1) * 256];          \
                                                                                 \
      if (worker.tries[wIndex] == 1) {                                           \
        worker.prev_chunk = worker.chunk;                                        \
      } else {                                                                   \
        worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE            \
                                          + (worker.tries[wIndex] - 2) * 256];   \
        memcpy(worker.chunk, worker.prev_chunk, 256);                            \
      }                                                                          \
                                                                                 \
    if (worker.op == 253)                                                         \
    {                                                                            \
      COPY_FN(worker, worker.pos1, worker.pos2);                                 \
      for (int i = worker.pos1; i < worker.pos2; i++)                            \
      {                                                                          \
        worker.chunk[i] = rl8(worker.chunk[i], 3);                               \
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);                              \
        worker.chunk[i] ^= worker.prev_chunk[worker.pos2];                       \
        worker.chunk[i] = rl8(worker.chunk[i], 3);                               \
                                                                                 \
        worker.prev_lhash = worker.lhash + worker.prev_lhash;                    \
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);             \
      }                                                                          \
                                                                                 \
      goto _wolf_after;                                                          \
    }                                                                            \
    if (worker.op >= 254) {                                                       \
      RC4_set_key(&worker.key[wIndex], 256, worker.prev_chunk);                  \
    }                                                                            \
    PERMUTE_FN(worker.prev_chunk, worker.chunk, worker.op,                       \
               worker.pos1, worker.pos2, worker);                               \
                                                                                 \
    if (!worker.op) {                                                             \
      if ((worker.pos2 - worker.pos1) % 2 == 1) {                                \
        worker.t1 = worker.chunk[worker.pos1];                                   \
        worker.t2 = worker.chunk[worker.pos2];                                   \
        worker.chunk[worker.pos1] = reverse8(worker.t2);                         \
        worker.chunk[worker.pos2] = reverse8(worker.t1);                         \
        worker.isSame = false;                                                   \
      }                                                                          \
    }                                                                            \
                                                                                 \
_wolf_after:                                                                     \
    {                                                                            \
    uint8_t pushPos1 = lp1;                                                      \
    uint8_t pushPos2 = lp2;                                                      \
                                                                                 \
    if (worker.pos1 == worker.pos2) {                                             \
      pushPos1 = (uint8_t)-1;                                                    \
      pushPos2 = (uint8_t)-1;                                                    \
    }                                                                            \
                                                                                 \
    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);           \
    worker.A = (256 + (worker.A % 256)) % 256;                                   \
                                                                                 \
    if (worker.A < 0x10)                                                          \
    {                                                                            \
      worker.prev_lhash = worker.lhash + worker.prev_lhash;                      \
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);               \
    }                                                                            \
                                                                                 \
    if (worker.A < 0x20)                                                          \
    {                                                                            \
      worker.prev_lhash = worker.lhash + worker.prev_lhash;                      \
      worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);                   \
    }                                                                            \
                                                                                 \
    if (worker.A < 0x30)                                                          \
    {                                                                            \
      worker.prev_lhash = worker.lhash + worker.prev_lhash;                      \
      HH_ALIGNAS(16)                                                             \
      const highwayhash::HH_U64 key2[2] =                                        \
          {worker.tries[wIndex], worker.prev_lhash};                             \
      worker.lhash = highwayhash::SipHash(                                        \
          key2, (char*)worker.chunk, worker.pos2);                               \
    }                                                                            \
                                                                                 \
    if (worker.A <= 0x40)                                                         \
    {                                                                            \
      RC4(&worker.key[wIndex], 256, worker.chunk, worker.chunk);                  \
      worker.isSame = false;                                                     \
      if (255 - pushPos2 < MINPREFLEN) pushPos2 = 255;                           \
      if (pushPos1 < MINPREFLEN)       pushPos1 = 0;                             \
      if (pushPos1 == 255)             pushPos1 = 0;                             \
                                                                                 \
      worker.astroTemplate[worker.templateIdx] = templateMarker{                  \
        (uint8_t)(chunkCount > 1 ? pushPos1 : 0),                                \
        (uint8_t)(chunkCount > 1 ? pushPos2 : 255),                              \
        (uint16_t)0,                                                             \
        (uint16_t)0,                                                             \
        (uint16_t)((firstChunk << 7) | chunkCount)                               \
      };                                                                         \
                                                                                 \
      pushPos1 = 0;                                                              \
      pushPos2 = 255;                                                            \
      worker.templateIdx += (worker.tries[wIndex] > 1);                          \
      firstChunk = worker.tries[wIndex] - 1;                                     \
      lp1 = 255;                                                                 \
      lp2 = 0;                                                                   \
      chunkCount = 1;                                                            \
    } else {                                                                     \
      chunkCount++;                                                              \
    }                                                                            \
                                                                                 \
    worker.chunk[255] = worker.chunk[255]                                         \
                      ^ worker.chunk[worker.pos1]                                \
                      ^ worker.chunk[worker.pos2];                               \
                                                                                 \
    if (255 - pushPos2 < MINPREFLEN) pushPos2 = 255;                             \
    if (pushPos1 < MINPREFLEN)       pushPos1 = 0;                               \
    }                                                                            \
                                                                                 \
    if (worker.tries[wIndex] > 260 + 16                                           \
        || (worker.sData[(worker.tries[wIndex]-1)*256+255] >= 0xf0               \
            && worker.tries[wIndex] > 260))                                      \
    {                                                                            \
      break;                                                                     \
    }                                                                            \
  }                                                                              \
                                                                                 \
  if (chunkCount > 0) {                                                           \
    if (255 - lp2 < MINPREFLEN) lp2 = 255;                                       \
    if (lp1 < MINPREFLEN)       lp1 = 0;                                         \
    worker.astroTemplate[worker.templateIdx] = templateMarker{                    \
      (uint8_t)(chunkCount > 1 ? lp1 : 0),                                       \
      (uint8_t)(chunkCount > 1 ? lp2 : 255),                                     \
      (uint16_t)0,                                                               \
      (uint16_t)0,                                                               \
      (uint16_t)((firstChunk << 7) | chunkCount)                                 \
    };                                                                           \
    worker.templateIdx++;                                                         \
  }                                                                              \
                                                                                 \
  worker.data_len = static_cast<uint32_t>(                                        \
      (worker.tries[wIndex] - 4) * 256                                           \
      + (((static_cast<uint64_t>(worker.chunk[253]) << 8)                        \
          | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
