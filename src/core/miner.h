#pragma once

#ifndef MINER
#define MINER

#include <tnn-common.hpp>

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

#if defined(_WIN32)
#include <windows.h>
LPTSTR lpNxtPage;  // Address of the next page to ask for
DWORD dwPages = 0; // Count of pages gotten so far
DWORD dwPageSize;  // Page size on this computer

HANDLE hInput;
DWORD prev_mode;
#else
#include <sched.h>
#define THREAD_PRIORITY_ABOVE_NORMAL -5
#define THREAD_PRIORITY_HIGHEST -20
#define THREAD_PRIORITY_TIME_CRITICAL -20
#endif

const int workerThreads = 2;

std::string symbol = nullArg;
std::string port = nullArg;
std::string password = "x";

bool useStratum = false;

int threads = 0;
int testOp = -1;
int testLen = -1;
int processPriority = 0;
bool gpuMine = false;
bool broadcastStats = false;
bool checkWallet = true;

int tuneWarmupSec;
int tuneDurationSec;

int cudaMemNumerator = 1000;
int cudaMemDenominator = 750; //Kilobytes per worker in VRAM

std::string defaultHost[] = {
  "dero-node-sk.mysrv.cloud",
  "127.0.0.1", // xel
  "127.0.0.1",
  "127.0.0.1", // spr
  "monerohash.com", // rx0
  "monerohash.com",
  "monerohash.com",
  "monerohash.com",
  "monerohash.com",
  "monerohash.com",
  "monerohash.com",
  "", // verus
  "",
  "127.0.0.1" // aix
};

std::string devPort[] = {
  "10300", // dero
  "8080", // xel
  "8080",
  "5555", // spr
  "2222", // rx0
  "2222",
  "2222",
  "2222",
  "2222",
  "2222",
  "2222",
  "", // verus
  "",
  "5555" // aix
};
// @ tritonn on Dero Name Service
std::string devWallet[] = {
  "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7",
  "xel:xz9574c80c4xegnvurazpmxhw5dlg2n0g9qm60uwgt75uqyx3pcsqzzra9m",
  "xel:xz9574c80c4xegnvurazpmxhw5dlg2n0g9qm60uwgt75uqyx3pcsqzzra9m",
#if defined(__x86_64__)
  "spectre:qr5l7q4s6mrfs9r7n0l090nhxrjdkxwacyxgk8lt2wt57ka6xr0ucvr0cmgnf",
#else
  "spectre:qqty6rrlsxwzcwdx7ge60256cw7r2adu7c8nqtsqxjmkt2c83h3kss3uqeay0",
#endif
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "", //verus
  "", //verus
  "astrix:qz2mzpga6qv9uvnpueau7gs29vgu3ynj80xmd2dmja2kelzh6cssymsk3shjx"
};

std::string testDevWallet[] = {
  "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7",
  "xet:5zwxjesmz6gtpg3c6zt20n9nevsyeewavpx6nwmv08z2hu2dpp3sq8w8ue6",
  "xet:5zwxjesmz6gtpg3c6zt20n9nevsyeewavpx6nwmv08z2hu2dpp3sq8w8ue6",
  "spectredev:qqhh8ul66g7t6aj5ggzl473cpan25tv6yjm0cl4hffprgtqfvmyaq8q28m4z8",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe",
  "", //verus
  "", //verus
  "astrix:qz2mzpga6qv9uvnpueau7gs29vgu3ynj80xmd2dmja2kelzh6cssymsk3shjx"
};

std::string *devSelection = devWallet;

std::unordered_map<std::string, int> coinSelector = {
  {"dero", DERO_HASH},
  {"DERO", DERO_HASH},
  {"xel", XELIS_HASH},
  {"XEL", XELIS_HASH},
  {"spr", SPECTRE_X},
  {"SPR", SPECTRE_X},
  {"rx0", RX0},
  {"xmr", RX0},
  {"XMR", RX0},
  {"sal", RX0},
  {"SAL", RX0},
  {"zeph", RX0},
  {"ZEPH", RX0},
  {"aix", ASTRIX_HASH},
  {"AIX", ASTRIX_HASH}
};

void getWork(bool isDev, int algo);
void sendWork();
void devWork();

void mine(int tid, int algo = DERO_HASH);
void cudaMine();

void benchmark(int i);
void logSeconds(std::chrono::steady_clock::time_point start_time, int duration, bool *stop);

Num CompactToBig(uint32_t compact) {
    // Extract the mantissa, sign bit, and exponent
    uint32_t mantissa = compact & 0x007fffff;
    bool isNegative = (compact & 0x00800000) != 0;
    uint32_t exponent = compact >> 24;

    Num target;

    if (exponent <= 3) {
        mantissa >>= 8 * (3 - exponent);
        target = Num(mantissa);
    } else {
        target = Num(mantissa);
        target <<= 8 * (exponent - 3);
    }

    if (isNegative) {
        target = -target;
    }

    return target;
}

inline Num ConvertDifficultyToBig(int64_t d, int algo)
{
  Num difficulty = Num(std::to_string(d).c_str(), 10);
  switch(algo) {
    case DERO_HASH:
      return oneLsh256 / difficulty;
    case XELIS_HASH:
      return maxU256 / difficulty;
    case SPECTRE_X:
      return (oneLsh256-1) / difficulty;
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
  pSupport(" SSE", __builtin_cpu_supports("sse"));
  pSupport(" SSE2", __builtin_cpu_supports("sse2"));
  pSupport(" SSE3", __builtin_cpu_supports("sse3"));
  pSupport(" SSE4.1", __builtin_cpu_supports("sse4.1"));
  pSupport(" SSE4.2", __builtin_cpu_supports("sse4.2"));
  pSupport(" AES", __builtin_cpu_supports("aes"));
  pSupport(" AVX", __builtin_cpu_supports("avx"));
  pSupport(" AVX2", __builtin_cpu_supports("avx2"));
  pSupport(" AVX512", __builtin_cpu_supports("avx512f"));
  bool sha = false;
  #if defined(__SHA__)
  sha = true;
  #endif
  pSupport(" SHA", sha);
  setcolor(BRIGHT_WHITE);
  printf("\n");
#endif
}

void setPriorityClass(boost::thread::native_handle_type t, int priority);

void setPriority(boost::thread::native_handle_type t, int priority);

void setAffinity(boost::thread::native_handle_type t, uint64_t core);

void update(std::chrono::steady_clock::time_point startTime);

#endif