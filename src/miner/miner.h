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
#include <xatum.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define DERO_HASH 0
#define XELIS_HASH 1
#define SPECTRE_X 2

#define DERO_SOLO 0

#define XELIS_SOLO 10
#define XELIS_XATUM 11
#define XELIS_STRATUM 12

#define SPECTRE_SOLO 20
#define SPECTRE_STRATUM 21

const int workerThreads = 2;

const int reportInterval = 1;

const char *nullArg = "NULL";

std::string symbol = nullArg;
std::string host = nullArg;
std::string port = nullArg;
std::string wallet = nullArg;
std::string workerName = "default";

bool useStratum = false;

int miningAlgo = DERO_HASH;
int protocol = XELIS_SOLO;

int threads = 0;
int testOp = -1;
int testLen = -1;
bool gpuMine = false;
bool useLookupMine = false;
bool broadcastStats = false;

int cudaMemNumerator = 1000;
int cudaMemDenominator = 750; //Kilobytes per worker in VRAM

// Dev fee config
// Dev fee is a % of hashrate
int batchSize = 5000;
double minFee = 1;
double devFee = 2.5;
const char *devPool = "dero.rabidmining.com";

std::string defaultHost[] = {
  "dero.rabidmining.com",
  "127.0.0.1",
  "127.0.0.1"
};

std::string devPort[] = {
  "10300",
  "8080",
  "5555"
};
// @ tritonn on Dero Name Service
std::string devWallet[] = {
  "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7",
  "xel:xz9574c80c4xegnvurazpmxhw5dlg2n0g9qm60uwgt75uqyx3pcsqzzra9m",
  "spectre:qr5l7q4s6mrfs9r7n0l090nhxrjdkxwacyxgk8lt2wt57ka6xr0ucvr0cmgnf"
};

std::string testDevWallet[] = {
  "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7",
  "xet:5zwxjesmz6gtpg3c6zt20n9nevsyeewavpx6nwmv08z2hu2dpp3sq8w8ue6",
  "spectre:qr5l7q4s6mrfs9r7n0l090nhxrjdkxwacyxgk8lt2wt57ka6xr0ucvr0cmgnf"
};

std::string *devSelection = devWallet;

std::unordered_map<std::string, int> coinSelector = {
  {"dero", DERO_HASH},
  {"DERO", DERO_HASH},
  {"xel", XELIS_HASH},
  {"XEL", XELIS_HASH},
  {"spr", SPECTRE_X},
  {"SPR", SPECTRE_X}
};

const char* devWorkerName = "tnn-dev";

Num oneLsh256;      
Num maxU256;                                                   

void getWork(bool isDev, int algo);
void sendWork();
void devWork();

int handleXatumPacket(Xatum::packet xPacket, bool isDev);

void mine(int tid, int algo = DERO_HASH);
void cudaMine();

void benchmark(int i);
void logSeconds(std::chrono::_V2::steady_clock::time_point start_time, int duration, bool *stop);

inline Num ConvertDifficultyToBig(int64_t d, int algo)
{
  Num difficulty = Num(std::to_string(d).c_str(), 10);
  switch(algo) {
    case DERO_HASH:
      return oneLsh256 / difficulty;
    case XELIS_HASH:
      return maxU256 / difficulty;
    case SPECTRE_X:
      return oneLsh256 / (difficulty+1);
    default:
      return 0;
  }
}

std::vector<std::string> supportCheck = {
  "sse","sse2","sse3","avx","avx2","avx512f"
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
  printf("Supported CPU Intrinsics\n\n");
  setcolor(CYAN);
  pSupport("SSE", __builtin_cpu_supports("sse"));
  pSupport("SSE2", __builtin_cpu_supports("sse2"));
  pSupport("SSE3", __builtin_cpu_supports("sse3"));
  pSupport("SSE4.1", __builtin_cpu_supports("sse4.1"));
  pSupport("SSE4.2", __builtin_cpu_supports("sse4.2"));
  pSupport("AVX", __builtin_cpu_supports("avx"));
  pSupport("AVX2", __builtin_cpu_supports("avx2"));
  pSupport("AVX512", __builtin_cpu_supports("avx512f"));
  bool sha = false;
  #if defined(__SHA__)
  sha = true;
  #endif
  pSupport("SHA", sha);
  setcolor(BRIGHT_WHITE);
  printf("\n");
#endif
}

inline Num ConvertDifficultyToBig(Num d, int algo)
{
  switch(algo) {
    case DERO_HASH:
      return oneLsh256 / d;
    case XELIS_HASH:
      return maxU256 / d;
    case SPECTRE_X:
      return oneLsh256 / (d+1);
    default:
      return 0;
  }
}

inline bool CheckHash(unsigned char *hash, int64_t diff, int algo)
{
  if (littleEndian()) std::reverse(hash, hash+32);
  bool cmp = Num(hexStr(hash, 32).c_str(), 16) < ConvertDifficultyToBig(diff, algo);
  if (littleEndian()) std::reverse(hash, hash+32);
  return (cmp);
}

inline bool CheckHash(unsigned char *hash, Num diff, int algo)
{
  if (littleEndian()) std::reverse(hash, hash+32);
  bool cmp = Num(hexStr(hash, 32).c_str(), 16) < diff;
  if (littleEndian()) std::reverse(hash, hash+32);
  return (cmp);
}

void setPriorityClass(boost::thread::native_handle_type t, int priority);

void setPriority(boost::thread::native_handle_type t, int priority);

void setAffinity(boost::thread::native_handle_type t, int core);

void update(std::chrono::_V2::steady_clock::time_point startTime);

#endif