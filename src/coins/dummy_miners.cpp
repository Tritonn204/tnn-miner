#include "miners.hpp"

#define blankfunc(S) void S(int tid){}

// AstroBWTv3 coins
#ifndef TNN_ASTROBWTV3
blankfunc(mineDero);
blankfunc(mineSpectre);
#endif

// XelisHash v1/v2 coins
#ifndef TNN_XELISHASH
blankfunc(mineXelis);
#endif

// RandomX coins
#ifndef TNN_RANDOMX
blankfunc(mineRx0);
#endif

// Verus 
#ifndef TNN_VERUSHASH
blankfunc(mineVerus);
#endif

// Astrix
#ifndef TNN_ASTRIXHASH
blankfunc(mineAstrix);
#endif

// Nexellia
#ifndef TNN_NXLHASH
blankfunc(mineNexellia);
#endif

// Hoosat
#ifndef TNN_HOOHASH
blankfunc(mineHoosat);
#endif

// Hoosat
#ifndef TNN_WALAHASH
blankfunc(mineWaglayla);
#endif

// Shai
#ifndef TNN_SHAIHIVE
blankfunc(mineShai);
#endif

#ifndef TNN_HIP
blankfunc(mineAstrix_hip);
blankfunc(mineNexellia_hip);
blankfunc(mineWaglayla_hip);
#endif
