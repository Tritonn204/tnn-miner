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
