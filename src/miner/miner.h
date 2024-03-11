#ifndef MINER
#define MINER

#include <bigint.h>
#include <stdint.h>
#include <chrono>
#include <inttypes.h>
#include <hex.h>
#include <endian.hpp>
#include <boost/thread.hpp>
#include <vector>
#include <terminal.h>
#include <string>
#include <num.h>

#ifdef _WIN32
#include <windows.h>
#endif

const int workerThreads = 2;

const int reportInterval = 1;

const char *nullArg = "NULL";
std::string host = nullArg;
std::string port = nullArg;
std::string wallet = nullArg;
int threads = 0;
int testOp = -1;
int testLen = -1;
bool gpuMine = false;

int cudaMemNumerator = 1000;
int cudaMemDenominator = 750; //Kilobytes per worker in VRAM

// Dev fee config
// Dev fee is a % of hashrate
int batchSize = 5000;
double minFee = 1;
double devFee = 2.5;
const char *devPool = "dero.rabidmining.com";
const char *devPort = "10300";
// @ tritonn on Dero Name Service
const char *devWallet = "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7";

const int MINIBLOCK_SIZE = 48;
Num oneLsh256;                                                   

void getWork(bool isDev);
void sendWork();
void devWork();

void mineBlock(int i);
void cudaMine();

void benchmark(int i);
void logSeconds(std::chrono::_V2::system_clock::time_point start_time, int duration, bool *stop);

inline Num ConvertDifficultyToBig(int64_t d)
{
  // (1 << 256) / (difficultyNum )c
  Num difficulty = Num(std::to_string(d).c_str(), 10);
  Num res = oneLsh256 / difficulty;
  return res; 
}

std::vector<std::string> supportCheck = {
  "sse","sse2","sse3","avx","avx2","avx512bw"
};

inline void pSupport(const char *check, bool res)
{
  const char* nod = res ? "yes" : "no";
  std::cout << check << ": " << nod << std::endl;
}

inline void printSupported()
{
#if defined(__aarch64__) || defined(_M_ARM64)
  //do nothing
#else
  setcolor(BRIGHT_WHITE);
  printf("Supported SIMD Suites\n\n");
  setcolor(CYAN);
  pSupport("SSE", __builtin_cpu_supports("sse"));
  pSupport("SSE2", __builtin_cpu_supports("sse2"));
  pSupport("SSE3", __builtin_cpu_supports("sse3"));
  pSupport("SSE4.1", __builtin_cpu_supports("sse4.1"));
  pSupport("SSE4.2", __builtin_cpu_supports("sse4.2"));
  pSupport("AVX", __builtin_cpu_supports("avx"));
  pSupport("AVX2", __builtin_cpu_supports("avx2"));
  pSupport("AVX512", __builtin_cpu_supports("avx512f"));
  setcolor(BRIGHT_WHITE);
  printf("\n");
#endif
}

inline Num ConvertDifficultyToBig(Num d)
{
  // (1 << 256) / (difficultyNum )
  Num res = oneLsh256 / d;
  return res;
}

inline bool CheckHash(unsigned char *hash, int64_t diff)
{
  if (littleEndian()) std::reverse(hash, hash+32);
  bool cmp = Num(hexStr(hash, 32).c_str(), 16) < ConvertDifficultyToBig(diff);
  if (littleEndian()) std::reverse(hash, hash+32);
  return (cmp);
}

inline bool CheckHash(unsigned char *hash, Num diff)
{
  if (littleEndian()) std::reverse(hash, hash+32);
  bool cmp = Num(hexStr(hash, 32).c_str(), 16) < diff;
  if (littleEndian()) std::reverse(hash, hash+32);
  return (cmp);
}

void setPriorityClass(boost::thread::native_handle_type t, int priority);

void setPriority(boost::thread::native_handle_type t, int priority);

void setAffinity(boost::thread::native_handle_type t, int core);

void update(std::chrono::_V2::system_clock::time_point startTime);

#endif