#pragma once

#include <stdint.h>
#include <vector>
#include <string>

#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <boost/json.hpp>

#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <boost/tokenizer.hpp>

#include <num.h>

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
static const char* devWorkerName = "tnn-dev";                                      

extern std::string workerName;
extern std::string workerNameFromWallet;
extern bool useLookupMine;

extern int protocol;

extern std::string host;
extern std::string wallet;

extern Num oneLsh256;      
extern Num maxU256;             

extern int miningAlgo;

// Dev fee config
// Dev fee is a % of hashrate
extern int batchSize;
extern double minFee;
extern double devFee;

extern int jobCounter;
extern int reportCounter;
extern int reportInterval;

extern int blockCounter;
extern int miniBlockCounter;
extern int rejected;
extern int accepted;
//static int firstRejected;

//extern uint64_t hashrate;
extern int64_t ourHeight;
extern int64_t devHeight;

extern int64_t difficulty;
extern int64_t difficultyDev;

extern double doubleDiff;
extern double doubleDiffDev;

extern std::vector<int64_t> rate5min;
extern std::vector<int64_t> rate1min;
extern std::vector<int64_t> rate30sec;

extern std::atomic<int64_t> counter;
extern std::atomic<int64_t> benchCounter;

extern bool isConnected;
extern bool devConnected;

extern bool beQuiet;

extern boost::asio::io_context my_context;
// Construct a timer without setting an expiry time.
extern boost::asio::steady_timer update_timer;
extern std::chrono::time_point<std::chrono::steady_clock> g_start_time;