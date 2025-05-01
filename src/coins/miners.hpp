#pragma once

#include "tnn-common.hpp"
#include <net.hpp>
#include <num.h>
#include <hex.h>
#include <endian.hpp>
#include <terminal.h>

using byte = unsigned char;

extern bool rx_hugePages;

inline Num ConvertDifficultyToBig(Num d, int algo)
{
  switch(algo) {
    case ALGO_ASTROBWTV3:
      return oneLsh256 / d;
    case ALGO_XELISV2:
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

void mineXelis(int tid);

void mineSpectre(int tid);

uint32_t rx_targetToDifficulty(const char* target);
void randomx_init_extern();
void randomx_init_intern(int threads);
void randomx_set_flags(bool autoFlags);
int rxRPCTest();
void mineRx0(int tid);

void mineVerus(int tid);

void mineAstrix(int tid);

void mineNexellia(int tid);

void mineHoosat(int tid);

void mineWaglayla(int tid);

void mineShai(int tid);

void mineAstrix_hip(int tid);
void mineNexellia_hip(int tid);
void mineWaglayla_hip(int tid);

typedef void (*mineFunc)(int);
inline mineFunc getMiningFunc(int algoNum, bool gpu) {
  if(gpu) {
    switch(algoNum) {
      case ALGO_ASTRIX_HASH:
        return mineAstrix_hip;
        break;
      case ALGO_NXL_HASH:
        return mineNexellia_hip;
        break;
      case ALGO_WALA_HASH:
        return mineWaglayla_hip;
        break;
      default:
        return unsupportedGpu;
        break;
    }
  }
  switch(algoNum) {
    case ALGO_ASTROBWTV3:
      return mineDero;
      break;
    case ALGO_XELISV2:
      return mineXelis;
      break;
    case ALGO_SPECTRE_X:
      return mineSpectre;
      break;
    case ALGO_RX0:
      return mineRx0;
      break;
    case ALGO_VERUS:
      return mineVerus;
      break;
    case ALGO_ASTRIX_HASH:
      return mineAstrix;
      break;
    case ALGO_NXL_HASH:
      return mineNexellia;
      break;
    case ALGO_HOOHASH:
      return mineHoosat;
      break;
    case ALGO_WALA_HASH:
      return mineWaglayla;
      break;
    case ALGO_SHAI_HIVE:
      return mineShai;
      break;
    default:
      return unsupportedCPU;
      break;
  }
}
