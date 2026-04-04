#pragma once
// Newton LUT configuration — shared between host (LUT generation) and device (LUT lookup)
// Change NEWTON_LUT_BITS to resize the table; everything else derives automatically.

#ifndef NEWTON_LUT_BITS
#define NEWTON_LUT_BITS  10
#endif

#define NEWTON_LUT_N     (1 << (NEWTON_LUT_BITS - 1))   // 512 @ 10 bits
#define NEWTON_LUT_SHIFT (32 - NEWTON_LUT_BITS)          // 22  @ 10 bits
#define NEWTON_LUT_BASE  NEWTON_LUT_N                     // 512 @ 10 bits
