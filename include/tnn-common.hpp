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

#include <boost/multiprecision/cpp_int.hpp>

#include <num.h>

#include "algo_definitions.h"

#define CMP_LT_U256(X, Y) (X[3] != Y[3] ? X[3] < Y[3] : X[2] != Y[2] ? X[2] < Y[2] \
                                                                            : X[1] != Y[1]   ? X[1] < Y[1] \
                                                                                                           : X[0] < Y[0])

extern bool ABORT_MINER;

#define CHECK_CLOSE if (ABORT_MINER) return;
#define CHECK_CLOSE_RET(s) if (ABORT_MINER) return s;

static const char *nullArg = "NULL";
static const char* devWorkerName = "tnn-dev";                                      

extern bool devTurn;

extern std::string workerName;
extern std::string workerNameFromWallet;
extern bool useLookupMine;

extern int protocol;
extern bool gpuMine;

extern std::string host;
extern std::string wallet;
extern std::string devWallet;

extern Num oneLsh256;      
extern Num maxU256;     

extern boost::multiprecision::uint256_t bigDiff;
extern boost::multiprecision::uint256_t bigDiff_dev;

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

extern uint64_t nonce0;
extern uint64_t nonce0_dev;

extern double doubleDiff;
extern double doubleDiffDev;

extern int HIP_deviceCount;
extern std::string HIP_names[32];
extern std::string HIP_pcieID[32];
extern uint64_t HIP_kIndex[32];
extern uint64_t HIP_kIndex_dev[32];
extern std::vector<std::atomic<uint64_t>> HIP_counters;
extern std::vector<std::vector<int64_t>> HIP_rates5min;
extern std::vector<std::vector<int64_t>> HIP_rates1min;
extern std::vector<std::vector<int64_t>> HIP_rates30sec;

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

inline std::string cpp_int_toHex(boost::multiprecision::cpp_int in) {
  std::ostringstream oss;
  oss << std::hex << in;
  std::string hex_string = oss.str();

  return hex_string;
}

inline void cpp_int_to_byte_array(const boost::multiprecision::uint256_t &num, uint8_t *out) {  
  for (size_t i = 0; i < 32; ++i) {
    out[i] = static_cast<uint8_t>(num >> (i * 8) & 0xFF);
  }
}

inline void cpp_int_to_be_byte_array(const boost::multiprecision::uint256_t &num, uint8_t *out) {  
  for (size_t i = 0; i < 32; ++i) {
    out[i] = static_cast<uint8_t>(num >> ((32 - i - 1) * 8) & 0xFF);
  }
}