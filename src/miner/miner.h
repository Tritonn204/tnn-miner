#ifndef MINER
#define MINER

#include <bigint.h>
#include <stdint.h>
#include <chrono>
#include <inttypes.h>
#include <hex.h>
#include <endian.h>
#include <gmpxx.h>
#include <boost/thread.hpp>

const char *consoleLine = " TNN-MINER V1 || ";

const int workerThreads = 2;

const int reportInterval = 4;

const char *host;
const char *port;
const char *wallet;
int threads;

// Dev fee config
// Dev fee is a % of hashrate
const double devFee = 2.5;
const char *devPool = "community-pools.mysrv.cloud";
const char *devPort = "10300";
// @ tritonn on Dero Name Service
const char *devWallet = "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7";

const int MINIBLOCK_SIZE = 48;
mpz_class oneLsh256;

void getWork(bool isDev);
void sendWork();
void devWork();

void mineBlock(int i);
void benchmark(int i);
void logSeconds(int duration);

inline mpz_class ConvertDifficultyToBig(int64_t d)
{
  // (1 << 256) / (difficultyNum )c
  mpz_class difficulty = mpz_class(std::to_string(d), 10);
  mpz_class res = oneLsh256 / difficulty;
  return res; 
}

inline mpz_class ConvertDifficultyToBig(mpz_class d)
{
  // (1 << 256) / (difficultyNum )
  mpz_class res = oneLsh256 / d;
  return res;
}

inline mpz_class bytesToMpz(unsigned char *data, int len)
{
  return mpz_class(hexStr(data, len).c_str(), 16);
}

inline bool CheckHash(unsigned char *hash, int64_t diff)
{
  int cmp = mpz_cmp(bytesToMpz(hash, 32).get_mpz_t(), ConvertDifficultyToBig(diff).get_mpz_t());
  return (cmp < 0);
}

void setPriorityClass(boost::thread::native_handle_type t, int priority);

void setPriority(boost::thread::native_handle_type t, int priority);

void setAffinity(boost::thread::native_handle_type t, int core);

void update(std::chrono::_V2::system_clock::time_point startTime);

#endif