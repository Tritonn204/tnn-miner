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

std::string password = "x";

int testOp = -1;
int testLen = -1;
int processPriority = 0;
bool broadcastStats = false;
bool checkWallet = true;

int tuneWarmupSec;
int tuneDurationSec;

int cudaMemNumerator = 1000;
int cudaMemDenominator = 750; //Kilobytes per worker in VRAM

void getWork_v2(MiningProfile *miningProf);
void sendWork();
void devWork();

void mine(int tid, int algo = ALGO_ASTROBWTV3);
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
    case ALGO_ASTROBWTV3:
      return oneLsh256 / difficulty;
    case ALGO_XELISV2:
      return maxU256 / difficulty;
    case ALGO_SPECTRE_X:
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