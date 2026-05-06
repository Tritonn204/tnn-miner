#pragma once

#include "tnn-common.hpp"

#include <net.hpp>

#include <num.h>
#include <hex.h>
#include <endian.hpp>
#include <terminal.hpp>

using byte = unsigned char;

extern bool rx_hugePages;

inline Num ConvertDifficultyToBig(Num d, int algo)
{
  switch(algo) {
    case ALGO_ASTROBWTV3:
      return oneLsh256 / d;
    case ALGO_XELISV2:
    case ALGO_XELISV3:
      return maxU256 / d;
    case ALGO_SPECTRE_X:
      return oneLsh256 / (d+1);
    case ALGO_KAWPOW:
      return maxU256 / d;
    default:
      return 0;
  }
}

inline bool CheckHash(unsigned char *hash, int64_t diff, int algo)
{
  if (littleEndian()) std::reverse(hash, hash+32);
  bool cmp = Num(hexStr(hash, 32).c_str(), 16) <= ConvertDifficultyToBig(diff, algo);
  if (littleEndian()) std::reverse(hash, hash+32);
  return (cmp);
}

inline bool CheckHash(unsigned char *hash, Num diff, int algo)
{
  if (littleEndian()) std::reverse(hash, hash+32);
  bool cmp = Num(hexStr(hash, 32).c_str(), 16) <= diff;
  if (littleEndian()) std::reverse(hash, hash+32);
  return (cmp);
}

// Fast LE hash-vs-target comparison (no bignum, no hex conversion)
// Both hash and target are 32 bytes in little-endian byte order
static inline bool hashMeetsTarget_le(const uint8_t *hash, const uint8_t *target) {
    const uint64_t *h = (const uint64_t *)hash;
    const uint64_t *t = (const uint64_t *)target;
    if (h[3] != t[3]) return h[3] < t[3];
    if (h[2] != t[2]) return h[2] < t[2];
    if (h[1] != t[1]) return h[1] < t[1];
    return h[0] <= t[0];
}

static inline uint64_t load_be64(const uint8_t *p) {
    return ((uint64_t)p[0] << 56) |
           ((uint64_t)p[1] << 48) |
           ((uint64_t)p[2] << 40) |
           ((uint64_t)p[3] << 32) |
           ((uint64_t)p[4] << 24) |
           ((uint64_t)p[5] << 16) |
           ((uint64_t)p[6] << 8) |
           (uint64_t)p[7];
}

// Xelis hashes are emitted in canonical/display byte order, while cached
// targets are stored as LE bytes. Compare without reversing the hot hash buffer.
static inline bool hashMeetsTarget_be_hash_le_target(const uint8_t *hash, const uint8_t *target) {
    const uint64_t *t = (const uint64_t *)target;
    const uint64_t h3 = load_be64(hash);
    if (h3 != t[3]) return h3 < t[3];
    const uint64_t h2 = load_be64(hash + 8);
    if (h2 != t[2]) return h2 < t[2];
    const uint64_t h1 = load_be64(hash + 16);
    if (h1 != t[1]) return h1 < t[1];
    return load_be64(hash + 24) <= t[0];
}

inline std::string uint32ToHex(uint32_t value) {
  std::stringstream ss;
  ss << std::hex << std::setw(8) << std::setfill('0') << value;
  return ss.str();
}

static inline void unsupportedCPU(int tid) {
  printf("This coin is not supported on CPUs\n");
}

static inline void unsupportedGpu(int tid) {
  printf("This coin is not supported on GPUs\n");
}

void mineDero(int tid);

#if defined(TNN_XELISHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
void mineXelis_unified(int tid);
#endif

void mineSpectre(int tid);

uint32_t rx_targetToDifficulty(const char* target);
void randomx_init_extern();
void randomx_init_intern(int threads);
void randomx_set_flags(bool autoFlags);
int rxRPCTest();
void mineRx0(int tid);

void mineVerus(int tid);

#if defined(TNN_ASTRIXHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
void mineAstrix_unified(int tid);
#endif

#if defined(TNN_NXLHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
void mineNexellia_unified(int tid);
#endif

#if defined(TNN_HOOHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
void mineHoosat_unified(int tid);
#endif

#if defined(TNN_WALAHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
void mineWaglayla_unified(int tid);
#endif

void mineShai(int tid);

void mineYespower(int tid);

void mineRinhash(int tid);

void mineAstrix_hip(int tid);
void mineNexellia_hip(int tid);
void mineWaglayla_hip(int tid);
void mineXelis_hip(int tid);
void mineKawPow_hip(int tid);

typedef void (*mineFunc)(int);
inline mineFunc getMiningFunc(int algoNum, bool gpu) {

  #ifdef TNN_HIP
  if(gpu) {
    switch(algoNum) {
#ifdef TNN_XELISHASH
      case ALGO_XELISV2:
      case ALGO_XELISV3:
        return mineXelis_hip;
        break;
#endif
#ifdef TNN_ASTRIXHASH
      case ALGO_ASTRIX_HASH:
        return mineAstrix_hip;
        break;
#endif
#ifdef TNN_NXLHASH
      case ALGO_NXL_HASH:
        return mineNexellia_hip;
        break;
#endif
#ifdef TNN_WALAHASH
      case ALGO_WALA_HASH:
        return mineWaglayla_hip;
        break;
#endif
#ifdef TNN_KAWPOW
      case ALGO_KAWPOW:
        return mineKawPow_hip;
        break;
#endif
      default:
        return unsupportedGpu;
        break;
    }
  }
  #endif

  switch(algoNum) {
    case ALGO_ASTROBWTV3:
      return mineDero;
      break;
    // case ALGO_XELISV1:
    //   return mineXelis_v1;
#if defined(TNN_XELISHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
    case ALGO_XELISV2:
    case ALGO_XELISV3:
      return mineXelis_unified;
      break;
#endif
    case ALGO_SPECTRE_X:
      return mineSpectre;
      break;
    case ALGO_RX0:
      return mineRx0;
      break;
    case ALGO_VERUS:
      return mineVerus;
      break;
#if defined(TNN_ASTRIXHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
    case ALGO_ASTRIX_HASH:
      return mineAstrix_unified;
      break;
#endif
#if defined(TNN_NXLHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
    case ALGO_NXL_HASH:
      return mineNexellia_unified;
      break;
#endif
#if defined(TNN_HOOHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
    case ALGO_HOOHASH:
      return mineHoosat_unified;
      break;
#endif
#if defined(TNN_WALAHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
    case ALGO_WALA_HASH:
      return mineWaglayla_unified;
      break;
#endif
    case ALGO_SHAI_HIVE:
      return mineShai;
      break;
    case ALGO_YESPOWER:
      return mineYespower;
      break;
    case ALGO_RINHASH:
      return mineRinhash;
      break;
    default:
      return unsupportedCPU;
      break;
  }
}
