#pragma once

#include <stdint.h>
#include <vector>
#include <string>

#define DERO_HASH 0
#define XELIS_HASH 1
#define SPECTRE_X 2

#define DERO_SOLO 0

#define XELIS_SOLO 10
#define XELIS_XATUM 11
#define XELIS_STRATUM 12

#define SPECTRE_SOLO 20
#define SPECTRE_STRATUM 21

static const char *nullArg = "NULL";

extern int protocol;

extern std::string host;
extern std::string wallet;

// Dev fee config
// Dev fee is a % of hashrate
extern int batchSize;
extern double minFee;
extern double devFee;
extern const char *devPool;

extern int jobCounter;

extern int blockCounter;
extern int miniBlockCounter;
extern int rejected;
extern int accepted;
//static int firstRejected;

extern uint64_t hashrate;
extern uint64_t ourHeight;
extern uint64_t devHeight;

extern uint64_t difficulty;
extern uint64_t difficultyDev;

extern double doubleDiff;
extern double doubleDiffDev;

extern std::vector<int64_t> rate5min;
extern std::vector<int64_t> rate1min;
extern std::vector<int64_t> rate30sec;

extern bool isConnected;
extern bool devConnected;
