#pragma once

#include "pow_hip.h"

inline int lookupIndex_hip(int op, int val, int v2) {
  return (v2 * 256 * 256) + (op * 256) + val;
}

__device__ void branchResult_hip(byte &val, int op, byte v2);