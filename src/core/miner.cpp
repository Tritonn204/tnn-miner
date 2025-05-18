//
// Copyright (c) 2016-2019 Vinnie Falco (vinnie dot falco at gmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Official repository: https://github.com/boostorg/beast
//

//------------------------------------------------------------------------------
//
// Example: WebSocket SSL client, coroutine
//
//------------------------------------------------------------------------------

#include "tnn-common.hpp"
#include "tnn-hugepages.h"
#include "gpulibs.h"
#include "hipkill.h"

#include "rootcert.h"
#include <DNSResolver.hpp>
#include "net.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <numeric>

#include "miner.h"

#include <random>

#include <hex.h>
#include "algos.h"
#include <thread>

#include <chrono>

#include <future>
#include <limits>
#include <libcubwt.cuh>

#include <openssl/err.h>
#include <openssl/ssl.h>
#include <base64.hpp>

#include <bit>
#include <broadcastServer.hpp>
#include <stratum/stratum.h>

#include <exception>

#include "reporter.hpp"

#include <coins/miners.hpp>
#include <tnn_hip/core/devInfo.hip.h>
#include <boost/algorithm/string.hpp>

#if defined(USE_ASTRO_SPSA)
  #include "spsa.hpp"
#endif

// INITIALIZE COMMON STUFF
int reportCounter = 0;
int reportInterval = 3;

bool ABORT_MINER = false;
const char *tnnTargetArch = XSTR(CPU_ARCHTARGET);
double latest_hashrate = 0.0;

bool gpuMine = false;
bool printHashrateOnExit = false;
std::string wallet = "NULL";
std::string devWallet = "NULL";

int HIP_deviceCount = 0;

uint256_t bigDiff(0);
uint256_t bigDiff_dev(0);

uint64_t nonce0 = 0;
uint64_t nonce0_dev = 0;

std::string HIP_names[32];
std::string HIP_pcieID[32];
uint64_t HIP_kIndex[32] = {0};
uint64_t HIP_kIndex_dev[32] = {0};
std::vector<std::atomic<uint64_t>> HIP_counters(32);
std::vector<std::vector<int64_t>> HIP_rates5min(32);
std::vector<std::vector<int64_t>> HIP_rates1min(32);
std::vector<std::vector<int64_t>> HIP_rates30sec(32);

std::atomic<int64_t> counter = 0;
std::atomic<int64_t> benchCounter = 0;
boost::asio::io_context my_context;
boost::asio::steady_timer update_timer = boost::asio::steady_timer(my_context);
std::chrono::time_point<std::chrono::steady_clock> g_start_time = std::chrono::steady_clock::now();
int mine_time = 0;
boost::asio::steady_timer mine_duration_timer = boost::asio::steady_timer(my_context);
bool printHugepagesError = true;

Num oneLsh256 = Num(1) << 256;
Num maxU256 = Num(2).pow(256) - 1;

const auto processor_count = std::thread::hardware_concurrency();

/* Start definitions from tnn-common.hpp */

MiningProfile miningProfile = MiningProfile();
MiningProfile devMiningProfile = MiningProfile();

// Dev fee config
// Dev fee is a % of hashrate
int batchSize = 5000;
double minFee = 1.0;
double devFee = 2.5;

int jobCounter;

int blockCounter;
int miniBlockCounter;
int rejected;
int accepted;


//static int firstRejected;

//uint64_t hashrate;
int64_t ourHeight;
int64_t devHeight;

int nonceLen;
int nonceLenDev;

int64_t difficulty;
int64_t difficultyDev;

double doubleDiff = 0;
double doubleDiffDev = 0;

bool useLookupMine = false;

std::vector<int64_t> rate5min;
std::vector<int64_t> rate1min;
std::vector<int64_t> rate30sec;

std::string workerName = "default";
std::string workerNameFromWallet = "";

bool isConnected = false;
bool devConnected = false;

bool devTurn = false;
bool beQuiet = false;
/* End definitions from tnn-common.hpp */

/* Start definitions from astrobwtv3.h */
#if defined(TNN_ASTROBWTV3)
AstroFunc allAstroFuncs[] = {
  // {"branch", branchComputeCPU},
  // {"lookup", lookupCompute},
  {"wolf", wolfCompute},
// #if defined(__AVX2__)
//   {"avx2z", branchComputeCPU_avx2_zOptimized}
// #elif defined(__aarch64__)
//   {"aarch64", branchComputeCPU_aarch64}
// #endif
};
size_t numAstroFuncs;
#endif
/* End definitions from astrobwtv3.h */

// #include <cuda_runtime.h>

namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
namespace po = boost::program_options;  // from <boost/program_options.hpp>

boost::mutex mutex;
boost::mutex devMutex;
boost::mutex userMutex;
boost::mutex reportMutex;

uint16_t *lookup2D_global; // Storage for computed values of 2-byte chunks
byte *lookup3D_global;     // Storage for deterministically computed values of 1-byte chunks

using byte = unsigned char;
int bench_duration = -1;
bool startBenchmark = false;
bool stopBenchmark = false;
//------------------------------------------------------------------------------

void openssl_log_callback(const SSL *ssl, int where, int ret)
{
  if (ret <= 0)
  {
    int error = SSL_get_error(ssl, ret);
    char errbuf[256];
    ERR_error_string_n(error, errbuf, sizeof(errbuf));
    std::cerr << "OpenSSL Error: " << errbuf << std::endl;
  }
}

//------------------------------------------------------------------------------

#if defined(TNN_ASTROBWTV3)
void initializeExterns() {
  numAstroFuncs = std::size(allAstroFuncs); //sizeof(allAstroFuncs)/sizeof(allAstroFuncs[0]);
}
#endif

int enhanceWallet(MiningProfile *currentProfile, bool checkWallet) {
  if(checkWallet) {
    if (currentProfile->coin.miningAlgo == ALGO_ASTROBWTV3 && !(currentProfile->wallet.find("der") == std::string::npos || currentProfile->wallet.find("det") == std::string::npos))
    {
      std::cout << "Provided wallet address is not valid for Dero" << std::endl;
      return EXIT_FAILURE;
    }
    if (currentProfile->coin.miningAlgo == ALGO_XELISV2 && !(currentProfile->wallet.find("xel") == std::string::npos || currentProfile->wallet.find("xet") == std::string::npos || currentProfile->wallet.find("Kr") == std::string::npos))
    {
      std::cout << "Provided wallet address is not valid for Xelis" << std::endl;
      return EXIT_FAILURE;
    }
    if (currentProfile->coin.miningAlgo == ALGO_SHAI_HIVE && !(currentProfile->wallet.find("sh1") == std::string::npos))
    {
      std::cout << "Provided wallet address is not valid for Shai" << std::endl;
      return EXIT_FAILURE;
    }
  }

  if(currentProfile->wallet.find("dero", 0) != std::string::npos) {
    currentProfile->coin = coins[COIN_DERO];
  }
  if(currentProfile->wallet.find("xel:", 0) != std::string::npos || currentProfile->wallet.find("xet:", 0) != std::string::npos) {
    currentProfile->coin = coins[COIN_XELIS];
  }
  if(currentProfile->wallet.find("spectre", 0) != std::string::npos) {
    currentProfile->coin = coins[COIN_SPECTRE];
    currentProfile->protocol = PROTO_SPECTRE_STRATUM;
  }
  return EXIT_SUCCESS;
}


void hipKill() {
  #ifdef TNN_HIP
  hipDeviceReset_wrapper();
  #endif
}

void onExit() {
  hipKill();
  ABORT_MINER = true;
  setcolor(BRIGHT_WHITE);
  if(printHashrateOnExit) {
    printf("\n\n%s: %d threads @ %2.2f with %d shares accepted (built with ", miningProfile.coin.coinPrettyName.c_str(), threads, latest_hashrate, accepted);
#ifdef __clang__
    std::cout << "Clang "
              << __clang_major__ << "."
              << __clang_minor__ << "."
              << __clang_patchlevel__ << ")" << std::endl;
#elif defined(__GNUC__)
    std::cout << "GCC "
              << __GNUC__ << "."
              << __GNUC_MINOR__ << "."
              << __GNUC_PATCHLEVEL__ << ")" << std::endl;
#else
    std::cout << "Unknown compiler" << std::endl;
#endif
  }

  printf("\nExiting Miner...\n");
  fflush(stdout);

  //boost::this_thread::sleep_for(boost::chrono::seconds(1));
  //fflush(stdout);

#if defined(_WIN32)
  SetConsoleMode(hInput, ENABLE_EXTENDED_FLAGS | (prev_mode));
#endif
}

void sigterm(int signum) {
  std::cout << "\n\nTerminate signal (" << signum << ") received." << std::flush;
  exit(signum);
}

void sigint(int signum) {
  std::cout << "\n\nInterrupt signal (" << signum << ") received." << std::flush;
  exit(signum);
}

int main(int argc, char **argv)
{
  // test_cshake256();

  // GPUTest();
  //printf("pre test\n");
  #ifdef TNN_HIP
  GPUTest();
  if (reportInterval == 3) reportInterval = 5;
  HIP_deviceCount = getGPUCount();
  for (int i = 0; i < HIP_deviceCount; i++) {
    HIP_names[i] = getDeviceName(i);
    HIP_pcieID[i] = getPCIBusId(i);
  }
  #endif
  //printf("post test\n");

  std::atexit(onExit);
  signal(SIGTERM, sigterm);
  signal(SIGINT, sigint);
  alignas(64) char buf[65536];
  setvbuf(stdout, buf, _IOFBF, 65536);
  srand(time(NULL)); // Placing higher here to ensure the effect cascades through the entire program

  #if defined(TNN_ASTROBWTV3)
  initWolfLUT();
  initializeExterns();
  #endif
  //printf("post wolf\n");

  // Check command line arguments.
  lookup2D_global = (uint16_t *)malloc_huge_pages(regOps_size * (256 * 256) * sizeof(uint16_t));
  lookup3D_global = (byte *)malloc_huge_pages(branchedOps_size * (256 * 256) * sizeof(byte));

  // default values
  bool lockThreads = true;
  devFee = 2.5;

  po::variables_map vm;
  po::options_description opts = get_prog_opts();
  try
  {
    int style = get_prog_style();
    po::store(po::command_line_parser(argc, argv).options(opts).style(style).run(), vm);
    po::notify(vm);
  }
  catch (std::exception &e)
  {
    printf("%s v%s %s\n", consoleLine, versionString, targetArch);
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Remember: Long options now use a double-dash -- instead of a single-dash -\n";
    return -1;
  }
  catch (...)
  {
    std::cerr << "Unknown error!" << "\n";
    return -1;
  }

#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  hInput = GetStdHandle(STD_INPUT_HANDLE);
  GetConsoleMode(hInput, &prev_mode); 
  SetConsoleMode(hInput, ENABLE_EXTENDED_FLAGS | (prev_mode & ~ENABLE_QUICK_EDIT_MODE));
#endif
  setcolor(BRIGHT_WHITE);
  printf("%s v%s %s\n", consoleLine, versionString, targetArch);
  printf("Compiled with %s\n", __VERSION__);
  if(vm.count("quiet")) {
    beQuiet = true;
  } else {
    printf("%s", TNN);
  }
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  HANDLE hSelfToken = NULL;

  ::OpenProcessToken(::GetCurrentProcess(), TOKEN_ALL_ACCESS, &hSelfToken);
  if (SetPrivilege(hSelfToken, SE_LOCK_MEMORY_NAME, true))
    std::cout << "Permission Granted for Huge Pages!" << std::endl;
  else
    std::cout << "Huge Pages: Permission Failed..." << std::endl;

  // SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
#endif

  if (vm.count("help"))
  {
    std::cout << opts << std::endl;
    boost::this_thread::sleep_for(boost::chrono::seconds(1));
    return 0;
  }

  if (vm.count("dero"))
  {
    #if defined(TNN_ASTROBWTV3)
    miningProfile.coin = coins[COIN_DERO];
    #else
    setcolor(RED);
    printf("%s", unsupported_astro);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("xelis"))
  {
    #if defined(TNN_XELISHASH)
    miningProfile.coin = coins[COIN_XELIS];
    #else
    setcolor(RED);
    printf("%s", unsupported_xelishash);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("spectre"))
  {
    #if defined(TNN_ASTROBWTV3)
    miningProfile.coin = coins[COIN_SPECTRE];
    miningProfile.protocol = PROTO_SPECTRE_STRATUM;
    #else
    setcolor(RED);
    printf("%s", unsupported_astro);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("astrix"))
  {
    #if defined(TNN_ASTRIXHASH)
    miningProfile.coin = coins[COIN_AIX];
    miningProfile.protocol = PROTO_KAS_STRATUM;
    #else
    setcolor(RED);
    printf("%s", unsupported_astrix);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("nexellia"))
  {
    #if defined(TNN_ASTRIXHASH)
    miningProfile.coin = coins[COIN_NXL];
    miningProfile.protocol = PROTO_KAS_STRATUM;
    #else
    setcolor(RED);
    printf("%s", unsupported_astrix);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("hoosat"))
  {
    #if defined(TNN_HOOHASH)
    miningProfile.coin = coins[COIN_HTN];
    miningProfile.protocol = PROTO_KAS_STRATUM;
    #else
    setcolor(RED);
    printf("%s", unsupported_hoohash);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("waglayla"))
  {
    #if defined(TNN_WALAHASH)
    miningProfile.coin = coins[COIN_WALA];
    miningProfile.protocol = PROTO_KAS_STRATUM;
    #else
    setcolor(RED);
    printf("%s", unsupported_waglayla);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("randomx"))
  {
    fflush(stdout);
    #if defined(TNN_RANDOMX)
    miningProfile.coin = coins[COIN_RX0];
    miningProfile.protocol = PROTO_RX0_SOLO; // Solo minin unsupported for now, so default to stratum instead
    #else
    setcolor(RED);
    printf("%s", unsupported_randomx);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  miningProfile.protocol = vm.count("xatum") ? PROTO_XELIS_XATUM : miningProfile.protocol;

  miningProfile.useStratum |= vm.count("stratum");

  if (vm.count("test-spectre"))
  {
    #if defined(TNN_ASTROBWTV3)
    #if defined(USE_ASTRO_SPSA)
      initSPSA();
    #endif
    return SpectreX::test();
    #else
    setcolor(RED);
    printf("%s", unsupported_astro);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("test-xelis"))
  {
    #if defined(TNN_XELISHASH)
    int rc = xelis_runTests_v2();
    return rc;
    #else
    setcolor(RED);
    printf("%s", unsupported_xelishash);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("test-randomx"))
  {
    #if defined(TNN_RANDOMX)
    int rc = RandomXTest();
    rc += rxRPCTest();
    return rc;
    #else
    setcolor(RED);
    printf("%s", unsupported_randomx);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("test-astrix"))
  {
    #if defined(TNN_RANDOMX)
    return AstrixHash::test();
    #else
    setcolor(RED);
    printf("%s", unsupported_astrix);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("test-nexellia"))
  {
    #if defined(TNN_RANDOMX)
    return NxlHash::test();
    #else
    setcolor(RED);
    printf("%s", unsupported_nexellia);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("test-hoosat"))
  {
    #if defined(TNN_RANDOMX)
    return HooHash::test();
    #else
    setcolor(RED);
    printf("%s", unsupported_hoohash);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("test-waglayla"))
  {
    #if defined(TNN_RANDOMX)
    return WalaHash::test();
    #else
    setcolor(RED);
    printf("%s", unsupported_waglayla);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("test-shai"))
  {
    #if defined(TNN_SHAIHIVE)
    return ShaiHive::test();
    #else
    setcolor(RED);
    printf("%s", unsupported_shai);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("xelis-bench"))
  {
    #if defined(TNN_XELISHASH)
    boost::thread t(xelis_benchmark_cpu_hash_v2);
    setPriority(t.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
    t.join();
    return 0;
    #else
    setcolor(RED);
    printf("%s", unsupported_xelishash);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }

  if (vm.count("sabench"))
  {
    #if defined(TNN_ASTROBWTV3)
    runDivsufsortBenchmark();
    return 0;
    #else
    setcolor(RED);
    printf("%s", unsupported_astro);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    #endif
  }

  if (vm.count("daemon-address"))
  {
    miningProfile.setPoolAddress(vm["daemon-address"].as<std::string>());
  }

  if (vm.count("port"))
  {
    miningProfile.port = std::to_string(vm["port"].as<int>());
    try {
      const int i{std::stoi(miningProfile.port)};
    } catch (...) {
      printf("ERROR: provided port is invalid: %s\n", miningProfile.port.c_str());
      return 1;
    }
  }
  if (vm.count("wallet"))
  {
    miningProfile.wallet = vm["wallet"].as<std::string>();
    if(miningProfile.wallet.find("dero", 0) != std::string::npos) {
      miningProfile.coin = coins[COIN_DERO];
    }
    if(miningProfile.wallet.find("xel:", 0) != std::string::npos || miningProfile.wallet.find("xet:", 0) != std::string::npos) {
      miningProfile.coin = coins[COIN_XELIS];
    }
    if(miningProfile.wallet.find("spectre", 0) != std::string::npos || miningProfile.wallet.find("spectretest", 0) != std::string::npos) {
      miningProfile.coin = coins[COIN_SPECTRE];
      miningProfile.protocol = PROTO_SPECTRE_STRATUM;
    }
    if(miningProfile.wallet.find("astrix", 0) != std::string::npos || miningProfile.wallet.find("astrixtest", 0) != std::string::npos) {
      miningProfile.coin = coins[COIN_AIX];
      miningProfile.protocol = PROTO_KAS_STRATUM;
    }
    if(miningProfile.wallet.find("nexellia", 0) != std::string::npos || miningProfile.wallet.find("nexelliatest", 0) != std::string::npos) {
      miningProfile.coin = coins[COIN_NXL];
      miningProfile.protocol = PROTO_KAS_STRATUM;
    }
    if(miningProfile.wallet.find("hoosat", 0) != std::string::npos || miningProfile.wallet.find("hoosattest", 0) != std::string::npos) {
      miningProfile.coin = coins[COIN_HTN];
      miningProfile.protocol = PROTO_KAS_STRATUM;
    }
    if(miningProfile.wallet.find("ZEPHYR", 0) != std::string::npos) {
      miningProfile.coin = coins[COIN_ZEPH];
      miningProfile.protocol = PROTO_RX0_SOLO;
    }

    boost::char_separator<char> sep(".");
    boost::tokenizer<boost::char_separator<char>> tok(miningProfile.wallet, sep);
    std::vector<std::string> tokens;
    std::copy(tok.begin(), tok.end(), std::back_inserter<std::vector<std::string> >(tokens));
    if(tokens.size() == 2) {
      miningProfile.wallet = tokens[0];
      workerNameFromWallet = tokens[1];
    }
  }
  if (vm.count("ignore-wallet"))
  {
    checkWallet = false;
  }
  if (vm.count("worker-name"))
  {
    workerName = vm["worker-name"].as<std::string>();
  }
  else
  {
    if(workerNameFromWallet != "") {
      workerName = workerNameFromWallet;
    } else {
      workerName = boost::asio::ip::host_name();
    }
  }
  miningProfile.workerName = workerName;
  if (vm.count("threads"))
  {
    threads = vm["threads"].as<int>();
  }
  if (vm.count("report-interval"))
  {
    reportInterval = vm["report-interval"].as<int>();
  }
  if (vm.count("dev-fee"))
  {
    try
    {
      devFee = vm["dev-fee"].as<double>();
      if (devFee < minFee)
      {
        setcolor(RED);
        printf("ERROR: dev fee must be at least %.2f", minFee);
        fflush(stdout);

        setcolor(BRIGHT_WHITE);
        boost::this_thread::sleep_for(boost::chrono::seconds(1));
        return 1;
      }
    }
    catch (...)
    {
      printf("ERROR: invalid dev fee parameter... format should be for example '1.0'");
      boost::this_thread::sleep_for(boost::chrono::seconds(1));
      return 1;
    }
  }
  if (vm.count("no-lock"))
  {
    setcolor(CYAN);
    printf("CPU affinity has been disabled\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    lockThreads = false;
  }
  if (vm.count("gpu"))
  {
    gpuMine = true;
  }

  if (vm.count("broadcast"))
  {
    broadcastStats = true;
  }
  // GPU-specific
  if (vm.count("batch-size"))
  {
    batchSize = vm["batch-size"].as<int>();
  }

  // Test-specific
  if (vm.count("op"))
  {
    testOp = vm["op"].as<int>();
  }
  if (vm.count("len"))
  {
    testLen = vm["len"].as<int>();
  }
  if (vm.count("lookup"))
  {
    printf("Use Lookup\n");
    useLookupMine = true;
  }

  // We can do this because we've set default in terminal.h
  // tuneWarmupSec = vm["tune-warmup"].as<int>();
  // tuneDurationSec = vm["tune-duration"].as<int>();

  mine_time = vm["mine-time"].as<int>();

  // Ensure we capture *all* of the other options before we start using goto
  if (vm.count("test-dero"))
  {
    #if defined(TNN_ASTROBWTV3)
    // temporary for optimization fishing:
    mapZeroes();
    // end of temporary section

    #if defined(USE_ASTRO_SPSA)
      initSPSA();
    #endif
    int rc = DeroTesting(testOp, testLen, useLookupMine);
    if(rc > 255) {
      rc = 1;
    }
    return rc;
    #else 
    setcolor(RED);
    printf("%s", unsupported_astro);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return 1;
    #endif
  }
  if (vm.count("dero-benchmark"))
  {
    bench_duration = vm["dero-benchmark"].as<int>();
    if (bench_duration <= 0)
    {
      printf("ERROR: Invalid benchmark arguments. Use -h for assistance\n");
      return 1;
    }
  }

fillBlanks:
{
  std::string localSymbol;
  if (miningProfile.coin.coinId == unknownCoin.coinId)
  {
    setcolor(CYAN);
    printf("%s\n", coinPrompt);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    std::string cmdLine;
    std::getline(std::cin, cmdLine);
    if (cmdLine != "" && cmdLine.find_first_not_of(' ') != std::string::npos)
    {
      localSymbol = cmdLine;
      std::transform(localSymbol.begin(), localSymbol.end(), localSymbol.begin(), ::toupper);
    }
    else
    {
      localSymbol = "DERO";
      setcolor(BRIGHT_YELLOW);
      printf("Default value will be used: %s\n\n", "DERO");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }

  for(int x = 0; x < COIN_COUNT; x++) {
    if(boost::iequals(coins[x].coinSymbol, localSymbol)) {
      miningProfile.coin = coins[x];
    }
  }
  if(miningProfile.coin.coinId == unknownCoin.coinId)
  {
    setcolor(RED);
    std::cout << "ERROR: Invalid coin symbol: " << localSymbol << std::endl << std::flush;
    setcolor(BRIGHT_YELLOW);
    printf("Supported symbols are:\n");
    for(int x = 0; x < COIN_COUNT; x++) {
      printf("%s\n", coins[x].coinSymbol.c_str());
    }
    printf("\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    miningProfile.coin = unknownCoin;
    goto fillBlanks;
  }

  // necessary as long as the bridge is a thing
  if (miningProfile.coin.miningAlgo == ALGO_SPECTRE_X) miningProfile.useStratum = true;

  int i = 0;
  std::vector<std::string *> stringParams = {&miningProfile.host, &miningProfile.port, &miningProfile.wallet};
  std::vector<const char *> stringDefaults = {devInfo[miningProfile.coin.coinId].devHost.c_str(),
                                              devInfo[miningProfile.coin.coinId].devPort.c_str(),
                                              devInfo[miningProfile.coin.coinId].devWallet.c_str()};
  std::vector<const char *> stringPrompts = {daemonPrompt, portPrompt, walletPrompt};
  for (std::string *param : stringParams)
  {
    if (param->empty())
    {
      setcolor(CYAN);
      printf("%s\n", stringPrompts[i]);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);

      std::string cmdLine;
      std::getline(std::cin, cmdLine);
      if (cmdLine != "" && cmdLine.find_first_not_of(' ') != std::string::npos)
      {
        *param = cmdLine;
      }
      else
      {
        *param = stringDefaults[i];
        setcolor(BRIGHT_YELLOW);
        printf("Default value will be used: %s\n\n", (*param).c_str());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }

      if (param == &miningProfile.host) {
        miningProfile.setPoolAddress(miningProfile.host);
      }
    }
    i++;
  }

  if (miningProfile.useStratum)
  {
    switch (miningProfile.coin.miningAlgo)
    {
      case ALGO_XELISV2:
        miningProfile.protocol = PROTO_XELIS_STRATUM;
        break;
      case ALGO_SPECTRE_X:
        miningProfile.protocol = PROTO_SPECTRE_STRATUM;
        break;
      case ALGO_RX0:
        miningProfile.protocol = PROTO_RX0_STRATUM;
        break;
    }
  }

  if (threads == 0)
  {
    threads = processor_count;
  }

  #if defined(_WIN32)
    if (threads > 32) lockThreads = false;
  #endif

  setcolor(BRIGHT_YELLOW);

  // #ifdef TNN_ASTROBWTV3
  // if (miningProfile.coin.miningAlgo == ALGO_ASTROBWTV3 || miningProfile.coin.miningAlgo == ALGO_SPECTRE_X) {
  //   if (vm.count("no-tune")) {
  //     std::string noTune = vm["no-tune"].as<std::string>();
  //     if(!setAstroAlgo(noTune)) {
  //       throw po::validation_error(po::validation_error::invalid_option_value, "no-tune");
  //     }
  //   } else {
  //     astroTune(threads, tuneWarmupSec, tuneDurationSec);
  //   }
  // }
  // fflush(stdout);
  // setcolor(BRIGHT_WHITE);
  // #endif

  #ifdef TNN_SHAIHIVE
  if (miningProfile.coin.miningAlgo == ALGO_SHAI_HIVE) {
    ShaiHive::tuneTimeLimit();
  }
  fflush(stdout);
  setcolor(BRIGHT_WHITE);
  #endif

  printf("\n");
}

  goto Mining;

// Benchmarking:
// {
//   if (threads <= 0)
//   {
//     threads = 1;
//   }

//   unsigned int n = std::thread::hardware_concurrency();
//   int winMask = 0;
//   for (int i = 0; i < n - 1; i++)
//   {
//     winMask += 1 << i;
//   }

//   host = defaultHost[miningAlgo];
//   port = devPort[miningAlgo];
//   wallet = devSelection[miningAlgo];

//   boost::thread GETWORK(getWork, false, miningAlgo);
//   // setPriority(GETWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

//   winMask = std::max(1, winMask);

//   // Create worker threads and set CPU affinity
//   for (int i = 0; i < threads; i++)
//   {
//     boost::thread t(benchmark, i + 1);

//     if (lockThreads)
//     {
//       setAffinity(t.native_handle(), (i % n));
//     }

//     // setPriority(t.native_handle(), THREAD_PRIORITY_HIGHEST);

//    //  mutex.lock();
//     std::cout << "(Benchmark) Worker " << i + 1 << " created" << std::endl;
//    //  mutex.unlock();
//   }

//   while (!isConnected)
//   {
//     boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
//   }
//   auto start_time = std::chrono::steady_clock::now();
//   startBenchmark = true;

//   boost::thread t2(logSeconds, start_time, bench_duration, &stopBenchmark);
//   setPriority(t2.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

//   while (true)
//   {
//     auto now = std::chrono::steady_clock::now();
//     auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
//     if (milliseconds >= bench_duration * 1000)
//     {
//       stopBenchmark = true;
//       break;
//     }
//     boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
//   }

//   auto now = std::chrono::steady_clock::now();
//   auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
//   int64_t hashrate = counter / bench_duration;
//   std::cout << "Mined for " << seconds << " seconds, average rate of " << std::flush;

//   std::string rateSuffix = " H/s";
//   double rate = (double)hashrate;
//   if (hashrate >= 1000000)
//   {
//     rate = (double)(hashrate / 1000000.0);
//     rateSuffix = " MH/s";
//   }
//   else if (hashrate >= 1000)
//   {
//     rate = (double)(hashrate / 1000.0);
//     rateSuffix = " KH/s";
//   }
//   std::cout << std::setprecision(3) << rate << rateSuffix << std::endl;
//   boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
//   return 0;
// }

Mining:
{
  printHashrateOnExit = true;
 //  mutex.lock();
  #ifndef TNN_HIP
    printSupported();
  #else
    gpuMine = true;
  #endif
 //  mutex.unlock();1
  int rc = enhanceWallet(&miningProfile, checkWallet);
  if(rc != 0) {
    return rc;
  }
  #if defined(USE_ASTRO_SPSA)
    initSPSA();
  #endif

  boost::thread GETWORK(getWork_v2, &miningProfile);
  // setPriority(GETWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  devMiningProfile = miningProfile;
  devMiningProfile.setDev(vm.count("testnet"));
  boost::thread DEVWORK(getWork_v2, &devMiningProfile);
  // setPriority(DEVWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  unsigned int n = std::thread::hardware_concurrency();

  #ifdef TNN_RANDOMX

  if (miningProfile.coin.miningAlgo == ALGO_RX0) {
    rx_hugePages = vm.count("rx-hugepages");
    randomx_set_flags(true);
    fflush(stdout);
    randomx_init_intern(n);
  }
  #endif

  // Create worker threads and set CPU affinity
 //  mutex.lock();
  boost::thread minerThreads[threads];
  if (gpuMine)
  {
    threads = 0;
    #ifdef TNN_HIP
    std::cout << "Starting GPU worker.." << std::endl;
    boost::thread t(getMiningFunc(miningProfile.coin.miningAlgo, true), 0);
    #else
    printf("Please use a GPU TNN Miner binary...\n");
    return -1;
    #endif
  } else {
    std::cout << "Starting threads: ";
    for (int i = 0; i < threads; i++)
    {
      minerThreads[i] = boost::thread(getMiningFunc(miningProfile.coin.miningAlgo, false), i + 1);

      if (lockThreads)
      {
        setAffinity(minerThreads[i].native_handle(), i);
      }
      // if (threads == 1 || (n > 2 && i <= n - 2))
      // setPriority(t.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

      std::cout << i + 1;
      if(i+1 != threads)
        std::cout << ", ";
    }
    std::cout << std::endl;
  }
 //  mutex.unlock();

  g_start_time = std::chrono::steady_clock::now();
  if (broadcastStats)
  {
    boost::thread BROADCAST(BroadcastServer::serverThread, &rate30sec, &accepted, &rejected, miningProfile.coin.coinPrettyName.c_str(), versionString, reportInterval);
  }

  while (!isConnected)
  {
    boost::this_thread::yield();
  }

  if(mine_time > 5) {
    mine_duration_timer.expires_after(std::chrono::seconds(mine_time));
    std::cout << "Will mine for " << mine_time << " seconds" << std::endl;
    mine_duration_timer.async_wait([&](const boost::system::error_code &ec)
      {
        ABORT_MINER = true;
        std::cout << std::endl << "Mined for " << mine_time << " seconds" << std::endl;
        update_timer.cancel();
        mine_duration_timer.cancel();
        // Stop all the io_context. So we can actually leave!
        my_context.stop();
        CHECK_CLOSE;
      });
  }

  // boost::thread reportThread([&]() {
    // Set an expiry time relative to now.
    update_timer.expires_after(std::chrono::seconds(1));

    // Start an asynchronous wait.
    update_timer.async_wait(update_handler);
    my_context.run();
  // });
  // setPriority(reportThread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);

  while(!ABORT_MINER) {
    std::this_thread::yield();
  }
  //ioc.reset();
  GETWORK.interrupt();
  DEVWORK.interrupt();
  std::cout << "Interrupting all threads...\n";
  for (unsigned i = 0; i < threads; ++i) {
    minerThreads[i].interrupt();
    minerThreads[i].join();
  }

  return EXIT_SUCCESS;
}
}

void logSeconds(std::chrono::steady_clock::time_point start_time, int duration, bool *stop)
{
  int i = 0;
  while (!(*stop))
  {
    auto now = std::chrono::steady_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    if (milliseconds >= 1000)
    {
      start_time = now;
     //  mutex.lock();
      // std::cout << "\n" << std::flush;
      printf("\rBENCHMARKING: %d/%d seconds elapsed...", i, duration);
      std::cout << std::flush;
     //  mutex.unlock();
      i++;
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(250));
  }
}

#if defined(_WIN32)
DWORD_PTR SetThreadAffinityWithGroups(HANDLE threadHandle, DWORD_PTR coreIndex)
{
  DWORD numGroups = GetActiveProcessorGroupCount();

  // Calculate group and processor within the group
  DWORD group = static_cast<DWORD>(coreIndex / 64);
  DWORD numProcessorsInGroup = GetMaximumProcessorCount(group);
  DWORD processorInGroup = static_cast<DWORD>(coreIndex % numProcessorsInGroup);

  if (group < numGroups)
  {
    GROUP_AFFINITY groupAffinity = {};
    groupAffinity.Group = static_cast<WORD>(group);
    groupAffinity.Mask = static_cast<KAFFINITY>(1ULL << processorInGroup);

    GROUP_AFFINITY previousGroupAffinity;
    if (!SetThreadGroupAffinity(threadHandle, &groupAffinity, &previousGroupAffinity))
    {
      return 0; // Fail case, return 0 like SetThreadAffinityMask
    }

    // Return the previous affinity mask for compatibility with your code
    return previousGroupAffinity.Mask;
  }

  return 0; // If out of bounds
}
#endif

void setAffinity(boost::thread::native_handle_type t, uint64_t core)
{
#if defined(_WIN32)
  HANDLE threadHandle = t;
  DWORD_PTR affinityMask = core;
  DWORD_PTR previousAffinityMask = SetThreadAffinityWithGroups(threadHandle, affinityMask);
  if (previousAffinityMask == 0)
  {
    DWORD error = GetLastError();
    std::cerr << "Failed to set CPU affinity for thread. Error code: " << error << std::endl;
  }

#elif !defined(__APPLE__)
  // Get the native handle of the thread
  pthread_t threadHandle = t;

  // Create a CPU set with a single core
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset); // Set core 0

  // Set the CPU affinity of the thread
  if (pthread_setaffinity_np(threadHandle, sizeof(cpu_set_t), &cpuset) != 0)
  {
    std::cerr << "Failed to set CPU affinity for thread" << std::endl;
  }

#endif
}

void setPriority(boost::thread::native_handle_type t, int priority)
{
#if defined(_WIN32)

  HANDLE threadHandle = t;

  // Set the thread priority
  int threadPriority = priority;
  BOOL success = SetThreadPriority(threadHandle, threadPriority);
  if (!success)
  {
    DWORD error = GetLastError();
    std::cerr << "Failed to set thread priority. Error code: " << error << std::endl;
  }

#else
  // Get the native handle of the thread
  pthread_t threadHandle = t;

  // Set the thread priority
  int threadPriority = priority;
  // do nothing

#endif
}

void getWork_v2(MiningProfile *miningProf)
{
  net::io_context ioc;
  ssl::context ctx = ssl::context{ssl::context::tlsv12_client};
  load_root_certificates(ctx);

  bool caughtDisconnect = false;

connectionAttempt:
  CHECK_CLOSE;
  bool *B = miningProf->isDev ? &devConnected : &isConnected;
  *B = false;
  setcolor(BRIGHT_YELLOW);
  std::cout << (miningProf->isDev ? "Dev " : "") << "Connecting...\n";
  setcolor(BRIGHT_WHITE);
  try
  {
    // Launch the asynchronous operation
    bool err = false;
    if (miningProf->isDev)
    {
      switch (miningProf->coin.miningAlgo)
      {
        case ALGO_ASTROBWTV3:
        {
          miningProf->workerName = workerName;
          break;
        }
        case ALGO_XELISV2:
        case ALGO_SPECTRE_X:
        {
          miningProf->workerName = "tnn-dev";
          break;
        }
        case ALGO_RX0:
        case ALGO_VERUS:
        case ALGO_ASTRIX_HASH:
        case ALGO_NXL_HASH:
        case ALGO_HOOHASH:
        case ALGO_WALA_HASH:
        {
          miningProf->workerName = devWorkerName;
          break;
        }
      }
      boost::asio::spawn(ioc, std::bind(&do_session_v2, miningProf, std::ref(ioc), std::ref(ctx), std::placeholders::_1),
                         // on completion, spawn will call this function
                         [&](std::exception_ptr ex)
                         {
                           if (ex)
                           {
                             std::rethrow_exception(ex);
                             err = true;
                           }
                         });
    }
    else
    {
      boost::asio::spawn(ioc, std::bind(&do_session_v2, miningProf, std::ref(ioc), std::ref(ctx), std::placeholders::_1),
                         // on completion, spawn will call this function
                         [&](std::exception_ptr ex)
                         {
                           if (ex)
                           {
                             std::rethrow_exception(ex);
                             err = true;
                           }
                         });
    }
    ioc.run();

    if (err)
    {
      if (!miningProf->isDev)
      {
       //  mutex.lock();
        setcolor(RED);
        std::cerr << "\nError establishing connections" << std::endl
                  << "Will try again in about 10 seconds...\n\n" << std::flush;
        setcolor(BRIGHT_WHITE);
       //  mutex.unlock();
      }
      boost::this_thread::sleep_for(boost::chrono::milliseconds(randomSleepTimeMs()));
      ioc.restart();
      goto connectionAttempt;
    }
    else
    {
      caughtDisconnect = false;
    }
  }
  catch (boost::thread_interrupted&) {
    //std::cout << "Thread was interrupted!" << std::endl;
    ioc.restart();
    return;
  }
  catch (...)
  {
    CHECK_CLOSE;
    // std::cerr << boost::current_exception_diagnostic_information() << std::endl;
    if (!miningProf->isDev)
    {
     //  mutex.lock();
      setcolor(RED);
      std::cerr << "\nError establishing connections" << std::endl
                << "Will try again in about 10 seconds...\n\n" << std::flush;
      setcolor(BRIGHT_WHITE);
     //  mutex.unlock();
    }
    else
    {
     //  mutex.lock();
      setcolor(RED);
      std::cerr << "Dev connection error\n" << std::flush;
      setcolor(BRIGHT_WHITE);
     //  mutex.unlock();
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(randomSleepTimeMs()));
    ioc.restart();
    goto connectionAttempt;
  }
  while (*B)
  {
    caughtDisconnect = false;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  }
  CHECK_CLOSE;
  if (!miningProf->isDev)
  {
   //  mutex.lock();
    setcolor(RED);
    if (!caughtDisconnect)
      std::cerr << "\nERROR: lost connection" << std::endl
                << "Will try to reconnect in about 10 seconds...\n\n";
    else
      std::cerr << "\nError establishing connection" << std::endl
                << "Will try again in about 10 seconds...\n\n";

    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    rate30sec.clear();
   //  mutex.unlock();
  }
  else
  {
   //  mutex.lock();
    setcolor(RED);
    if (!caughtDisconnect)
      std::cerr << "\nERROR: lost connection to dev node (mining will continue)" << std::endl
                << "Will try to reconnect in about 10 seconds...\n\n";
    else
      std::cerr << "\nError establishing connection to dev node" << std::endl
                << "Will try again in about 10 seconds...\n\n";

    fflush(stdout);
    setcolor(BRIGHT_WHITE);
   //  mutex.unlock();
  }
  caughtDisconnect = true;
  CHECK_CLOSE;
  boost::this_thread::sleep_for(boost::chrono::milliseconds(randomSleepTimeMs()));
  ioc.restart();
  goto connectionAttempt;
}
