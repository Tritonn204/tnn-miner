#pragma once

#include <tnn-common.hpp>
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
    case DERO_HASH:
      return oneLsh256 / d;
    case XELIS_HASH:
      return maxU256 / d;
    case SPECTRE_X:
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

void mineDero(int tid);

void mineXelis(int tid);

void mineSpectre(int tid);

uint32_t rx_targetToDifficulty(const char* target);
void randomx_init_extern();
void randomx_init_intern(int threads);
void randomx_set_flags(bool autoFlags);
void rxRPCTest();
void mineRx0(int tid);

void mineVerus(int tid);

void mineAstrix(int tid);

void mineNexellia(int tid);

void mineHoosat(int tid);

void mineWaglayla(int tid);

void mineShai(int tid);

typedef void (*mineFunc)(int);
const mineFunc POW[] = {
  mineDero, // 0
  mineXelis, 
  mineXelis, 
  mineSpectre,
  mineRx0,
  mineRx0,
  mineRx0, // 5
  mineRx0,
  mineRx0,
  mineRx0,
  mineRx0,
  mineVerus, // 10
  mineVerus,
  mineAstrix,
  mineNexellia,
  mineHoosat,
  mineWaglayla,
  mineShai
};

