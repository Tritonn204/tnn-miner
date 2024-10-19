#pragma once

#include <tnn-common.hpp>
#include <net.hpp>
#include <num.h>
#include <hex.h>
#include <endian.hpp>
#include <terminal.h>

using byte = unsigned char;

inline Num ConvertDifficultyToBig_hip(Num d, int algo)
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

void mineAstrix_hip();
void mineNexellia_hip();
void mineWaglayla_hip();

static inline void unsupported() {
  printf("This coin is not supported on GPUs\n");
}

typedef void (*mineFunc_hip)();
const mineFunc_hip POW_HIP[] = {
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
  mineAstrix_hip,
  mineNexellia_hip,
  unsupported,
  mineWaglayla_hip // 15
};

