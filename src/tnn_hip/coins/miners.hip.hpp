#pragma once

#include <tnn-common.hpp>
#include <net.hpp>
#include <num.h>
#include <hex.h>
#include <endian.hpp>
#include <terminal.h>

using byte = unsigned char;

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

void mineAstrix_hip();

static inline void unsupported() {
  printf("This coin is not supported on GPUs\n");
}

typedef void (*mineFunc_hip)();
const mineFunc_hip POW[] = {
  unsupported, // 0
  unsupported, 
  unsupported, 
  unsupported,
  unsupported,
  unsupported,
  unsupported, // 5
  unsupported,
  unsupported,
  unsupported,
  unsupported,
  unsupported, // 10
  unsupported,
  mineAstrix_hip
};

