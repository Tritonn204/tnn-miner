#include "miners.hpp"

// AstroBWTv3 coins
#ifndef TNN_ASTROBWTV3
void mineDero(int tid){}
void mineSpectre(int tid){}
#endif

// XelisHash v1/v2 coins
#ifndef TNN_XELISHASH
void mineXelis(int tid){}
#endif

// RandomX coins
#ifndef TNN_RANDOMX
void mineRx0(int tid){}
#endif

// Verus 
#ifndef TNN_VERUSHASH
void mineVerus(int tid){}
#endif

// Astrix
#ifndef TNN_ASTRIXHASH
void mineAstrix(int tid){}
#endif
