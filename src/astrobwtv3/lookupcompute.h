#pragma once

#include "astrobwtv3.h"
#include <vector>
#include <numeric>

void lookupGen(workerData &worker, uint16_t *lookup2D, byte *lookup3D);

inline int lookupIndex(int op, int val, int v2) {
  return (v2 * 256 * 256) + (op * 256) + val;
}

void branchResult(byte &val, int op, byte v2);