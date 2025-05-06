#pragma once

#include <inttypes.h>
#include "astroworker.h"
#include "compile.h"

#define PACK_MASK 0x1FFFF
#define MINPREFLEN 4
#define MPOFFSET (MINPREFLEN-1)

#define ISO(x, y) (((x >> y*8) & 0xFF))

#define getHeadStart(x) (( \
  (( \
    (x[0] << 16) | (x[1] << 8) | ((x[2] & 0xFE)) \
  ) >> 1) \
))

#define getHeadIdx(x) (( \
  (( \
    ((x[2] & 0x1) << 16) | (x[3] << 8) | (x[4]) \
  ) & PACK_MASK) \
))

#define setHeadStart(x, y) { \
  x[0] = (ISO(y << 1,2)); \
  x[1] = (ISO(y << 1,1)); \
  x[2] = (x[2] & 0x1) | ((ISO(y << 1,0))); \
}

#define setHeadIdx(x, y) { \
  x[2] = (x[2] & 0xFE) | (ISO(y, 2)); \
  x[3] = (ISO(y,1)); \
  x[4] = (ISO(y,0)); \
}

TNN_TARGETS
bool SPSA(const uint8_t* data, int dataSize, workerData &ctx);
void initSPSA();