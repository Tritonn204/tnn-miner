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
#include "rootcert.h"
#include <DNSResolver.hpp>
#include "net.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include "miner.h"

#include <random>

#include <hex.h>
#include "algos.h"
#include <thread>

#include <chrono>

#include <hugepages.h>
#include <future>
#include <limits>
#include <libcubwt.cuh>

#include <openssl/err.h>
#include <openssl/ssl.h>
#include <base64.hpp>

#include <bit>
#include <broadcastServer.hpp>
#include <stratum.h>

#include <exception>

#include "reporter.hpp"
#include "coins/miners.hpp"

// INITIALIZE COMMON STUFF
int miningAlgo = DERO_HASH;
int reportCounter = 0;
int reportInterval = 3;
std::atomic<int64_t> counter = 0;
std::atomic<int64_t> benchCounter = 0;
boost::asio::io_context my_context;
boost::asio::steady_timer update_timer = boost::asio::steady_timer(my_context);
std::chrono::time_point<std::chrono::steady_clock> g_start_time = std::chrono::steady_clock::now();

Num oneLsh256 = Num(1) << 256;
Num maxU256 = Num(2).pow(256) - 1;

const auto processor_count = std::thread::hardware_concurrency();

/* Start definitions from tnn-common.hpp */
int protocol = XELIS_SOLO;

std::string daemonProto = "";
std::string host = "NULL";
std::string wallet = "NULL";

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

int64_t difficulty;
int64_t difficultyDev;

double doubleDiff;
double doubleDiffDev;

bool useLookupMine = false;

std::vector<int64_t> rate5min;
std::vector<int64_t> rate1min;
std::vector<int64_t> rate30sec;

std::string workerName = "default";
std::string workerNameFromWallet = "";

bool isConnected = false;
bool devConnected = false;

bool beQuiet = false;
/* End definitions from tnn-common.hpp */

/* Start definitions from astrobwtv3.h */
AstroFunc allAstroFuncs[] = {
  {"branch", branchComputeCPU},
  {"lookup", lookupCompute},
  {"wolf", wolfCompute},
#if defined(__AVX2__)
  {"avx2z", branchComputeCPU_avx2_zOptimized}
#elif defined(__aarch64__)
  {"aarch64", branchComputeCPU_aarch64}
#endif
};
size_t numAstroFuncs;
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

void initializeExterns() {
  numAstroFuncs = std::size(allAstroFuncs); //sizeof(allAstroFuncs)/sizeof(allAstroFuncs[0]);
}

void onExit() {
  setcolor(BRIGHT_WHITE);
  printf("\n\nExiting Miner...");
  fflush(stdout);

#if defined(_WIN32)
  SetConsoleMode(hInput, ENABLE_EXTENDED_FLAGS | (prev_mode & ~ENABLE_QUICK_EDIT_MODE));
#endif
}

void sigterm(int signum) {
  std::cout << "\n\nInterrupt signal (" << signum << ") received.\n" << std::flush;
  exit(0);
}

int main(int argc, char **argv)
{
  std::atexit(onExit);
  signal(SIGTERM, sigterm);
  alignas(64) char buf[65536];
  setvbuf(stdout, buf, _IOFBF, 65536);
  srand(time(NULL)); // Placing higher here to ensure the effect cascades through the entire program

  initWolfLUT();
  initializeExterns();
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
    printf("%s v%s\n", consoleLine, versionString);
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
  printf("%s v%s\n", consoleLine, versionString);
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
    symbol = "DERO";
  }

  if (vm.count("xelis"))
  {
    symbol = "XEL";
  }

  if (vm.count("spectre"))
  {
    symbol = "SPR";
    protocol = SPECTRE_STRATUM;
  }

  if (vm.count("xatum"))
  {
    protocol = XELIS_XATUM;
  }

  if (vm.count("stratum"))
  {
    useStratum = true;
  }

  if (vm.count("testnet"))
  {
    devSelection = testDevWallet;
  }

  if (vm.count("test-spectre"))
  {
    return SpectreX::test();
  }

  if (vm.count("test-xelis"))
  {
    int rc = xelis_runTests_v2();
    return rc;
  }

  if (vm.count("xelis-bench"))
  {
    boost::thread t(xelis_benchmark_cpu_hash_v2);
    setPriority(t.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
    t.join();
    return 0;
  }

  if (vm.count("sabench"))
  {
    runDivsufsortBenchmark();
    return 0;
  }

  if (vm.count("daemon-address"))
  {
    host = vm["daemon-address"].as<std::string>();
    boost::char_separator<char> sep(":");
    boost::tokenizer<boost::char_separator<char>> tok(host, sep);
    std::vector<std::string> tokens;
    std::copy(tok.begin(), tok.end(), std::back_inserter<std::vector<std::string> >(tokens));
    if(tokens.size() == 2) {
      host = tokens[0];
      try
      {
        // given host:port
        const int i{std::stoi(tokens[1])};
        port = tokens[1];
      }
      catch (...)
      {
        // protocol:host
        daemonProto = tokens[0];
        host = tokens[1];
      }
    } else if(tokens.size() == 3) {
      daemonProto = tokens[0];  // wss, stratum+tcp, stratum+ssl, et al
      host = tokens[1];
      port = tokens[2];
    }
    boost::replace_all(host, "/", "");
    if (daemonProto.size() > 0) {
      if (daemonProto.find("stratum") != std::string::npos) useStratum = true;
      if (daemonProto.find("xatum") != std::string::npos) protocol = XELIS_XATUM;
    }
  }

  if (vm.count("port"))
  {
    port = std::to_string(vm["port"].as<int>());
    try {
      const int i{std::stoi(port)};
    } catch (...) {
      printf("ERROR: provided port is invalid: %s\n", port.c_str());
      return 1;
    }
  }
  if (vm.count("wallet"))
  {
    wallet = vm["wallet"].as<std::string>();
    if(wallet.find("dero", 0) != std::string::npos) {
      symbol = "DERO";
    }
    if(wallet.find("xel:", 0) != std::string::npos || wallet.find("xet:", 0) != std::string::npos) {
      symbol = "XEL";
    }
    if(wallet.find("spectre", 0) != std::string::npos) {
      symbol = "SPR";
      protocol = SPECTRE_STRATUM;
    }
    boost::char_separator<char> sep(".");
    boost::tokenizer<boost::char_separator<char>> tok(wallet, sep);
    std::vector<std::string> tokens;
    std::copy(tok.begin(), tok.end(), std::back_inserter<std::vector<std::string> >(tokens));
    if(tokens.size() == 2) {
      wallet = tokens[0];
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
        std::cout << "%" << std::endl;
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
  tuneWarmupSec = vm["tune-warmup"].as<int>();
  tuneDurationSec = vm["tune-duration"].as<int>();

  // Ensure we capture *all* of the other options before we start using goto
  if (vm.count("test-dero"))
  {
    // temporary for optimization fishing:
    mapZeroes();
    // end of temporary section

    int rc = DeroTesting(testOp, testLen, useLookupMine);
    if(rc > 255) {
      rc = 1;
    }
    return rc;
  }
  if (vm.count("dero-benchmark"))
  {
    bench_duration = vm["dero-benchmark"].as<int>();
    if (bench_duration <= 0)
    {
      printf("ERROR: Invalid benchmark arguments. Use -h for assistance\n");
      return 1;
    }
    goto Benchmarking;
  }

fillBlanks:
{
  if (symbol == nullArg)
  {
    setcolor(CYAN);
    printf("%s\n", coinPrompt);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    std::string cmdLine;
    std::getline(std::cin, cmdLine);
    if (cmdLine != "" && cmdLine.find_first_not_of(' ') != std::string::npos)
    {
      symbol = cmdLine;
    }
    else
    {
      symbol = "DERO";
      setcolor(BRIGHT_YELLOW);
      printf("Default value will be used: %s\n\n", "DERO");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }

  auto it = coinSelector.find(symbol);
  if (it != coinSelector.end())
  {
    miningAlgo = it->second;
  }
  else
  {
    setcolor(RED);
    std::cout << "ERROR: Invalid coin symbol: " << symbol << std::endl << std::flush;
    setcolor(BRIGHT_YELLOW);
    it = coinSelector.begin();
    printf("Supported symbols are:\n");
    while (it != coinSelector.end())
    {
      printf("%s\n", it->first.c_str());
      it++;
    }
    printf("\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    symbol = nullArg;
    goto fillBlanks;
  }

  // necessary as long as the bridge is a thing
  if (miningAlgo == SPECTRE_X) useStratum = true;

  int i = 0;
  std::vector<std::string *> stringParams = {&host, &port, &wallet};
  std::vector<const char *> stringDefaults = {defaultHost[miningAlgo].c_str(), devPort[miningAlgo].c_str(), devSelection[miningAlgo].c_str()};
  std::vector<const char *> stringPrompts = {daemonPrompt, portPrompt, walletPrompt};
  for (std::string *param : stringParams)
  {
    if (*param == nullArg)
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

      if (param == &host) {
        boost::char_separator<char> sep(":");
        boost::tokenizer<boost::char_separator<char>> tok(host, sep);
        std::vector<std::string> tokens;
        std::copy(tok.begin(), tok.end(), std::back_inserter<std::vector<std::string> >(tokens));
        if(tokens.size() == 2) {
          host = tokens[0];
          try
          {
            // given host:port
            const int i{std::stoi(tokens[1])};
            port = tokens[1];
          }
          catch (...)
          {
            // protocol:host
            daemonProto = tokens[0];
            host = tokens[1];
          }
        } else if(tokens.size() == 3) {
          daemonProto = tokens[0];  // wss, stratum+tcp, stratum+ssl, et al
          host = tokens[1];
          port = tokens[2];
        }
        boost::replace_all(host, "/", "");
        if (daemonProto.size() > 0) {
          if (daemonProto.find("stratum") != std::string::npos) useStratum = true;
          if (daemonProto.find("xatum") != std::string::npos) protocol = XELIS_XATUM;
        }
      }
    }
    i++;
  }

  if (useStratum)
  {
    switch (miningAlgo)
    {
      case XELIS_HASH:
        protocol = XELIS_STRATUM;
        break;
      case SPECTRE_X:
        protocol = SPECTRE_STRATUM;
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
  if (miningAlgo == DERO_HASH || miningAlgo == SPECTRE_X) {
    if (vm.count("no-tune")) {
      std::string noTune = vm["no-tune"].as<std::string>();
      if(!setAstroAlgo(noTune)) {
        throw po::validation_error(po::validation_error::invalid_option_value, "no-tune");
      }
    } else {
      astroTune(threads, tuneWarmupSec, tuneDurationSec);
    }
  }
  fflush(stdout);
  setcolor(BRIGHT_WHITE);

  printf("\n");
}

  goto Mining;

Benchmarking:
{
  if (threads <= 0)
  {
    threads = 1;
  }

  unsigned int n = std::thread::hardware_concurrency();
  int winMask = 0;
  for (int i = 0; i < n - 1; i++)
  {
    winMask += 1 << i;
  }

  host = defaultHost[miningAlgo];
  port = devPort[miningAlgo];
  wallet = devSelection[miningAlgo];

  boost::thread GETWORK(getWork, false, miningAlgo);
  // setPriority(GETWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  winMask = std::max(1, winMask);

  // Create worker threads and set CPU affinity
  for (int i = 0; i < threads; i++)
  {
    boost::thread t(benchmark, i + 1);

    if (lockThreads)
    {
#if defined(_WIN32)
      setAffinity(t.native_handle(), 1LL << (i % n));
#else
      setAffinity(t.native_handle(), (i % n));
#endif
    }

    // setPriority(t.native_handle(), THREAD_PRIORITY_HIGHEST);

   //  mutex.lock();
    std::cout << "(Benchmark) Worker " << i + 1 << " created" << std::endl;
   //  mutex.unlock();
  }

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }
  auto start_time = std::chrono::steady_clock::now();
  startBenchmark = true;

  boost::thread t2(logSeconds, start_time, bench_duration, &stopBenchmark);
  setPriority(t2.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  while (true)
  {
    auto now = std::chrono::steady_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    if (milliseconds >= bench_duration * 1000)
    {
      stopBenchmark = true;
      break;
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
  }

  auto now = std::chrono::steady_clock::now();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
  int64_t hashrate = counter / bench_duration;
  std::cout << "Mined for " << seconds << " seconds, average rate of " << std::flush;

  std::string rateSuffix = " H/s";
  double rate = (double)hashrate;
  if (hashrate >= 1000000)
  {
    rate = (double)(hashrate / 1000000.0);
    rateSuffix = " MH/s";
  }
  else if (hashrate >= 1000)
  {
    rate = (double)(hashrate / 1000.0);
    rateSuffix = " KH/s";
  }
  std::cout << std::setprecision(3) << rate << rateSuffix << std::endl;
  boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  return 0;
}

Mining:
{
 //  mutex.lock();
  printSupported();
 //  mutex.unlock();

  if (miningAlgo == DERO_HASH && (wallet.find("der", 0) == std::string::npos && wallet.find("det", 0) == std::string::npos))
  {
    std::cout << "Provided wallet address is not valid for Dero" << std::endl;
    return EXIT_FAILURE;
  }
  if (miningAlgo == XELIS_HASH && (wallet.find("xel", 0) == std::string::npos && wallet.find("xet") == std::string::npos && wallet.find("Kr", 0) == std::string::npos))
  {
    std::cout << "Provided wallet address is not valid for Xelis" << std::endl;
    return EXIT_FAILURE;
  }
  if (miningAlgo == SPECTRE_X && (wallet.find("spectre", 0) == std::string::npos)) {
    std::cout << "Provided wallet address is not valid for Spectre" << std::endl;
    return EXIT_FAILURE;
  }
  boost::thread GETWORK(getWork, false, miningAlgo);
  // setPriority(GETWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  boost::thread DEVWORK(getWork, true, miningAlgo);
  // setPriority(DEVWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  unsigned int n = std::thread::hardware_concurrency();
  int winMask = 0;
  for (int i = 0; i < n - 1; i++)
  {
    winMask += 1 << i;
  }

  winMask = std::max(1, winMask);

  // Create worker threads and set CPU affinity
 //  mutex.lock();
  if (false /*gpuMine*/)
  {
    // boost::thread t(cudaMine);
    // setPriority(t.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
    // continue;
  }
  else
    std::cout << "Starting threads: ";
    for (int i = 0; i < threads; i++)
    {

      boost::thread t(POW[miningAlgo], i + 1);

      if (lockThreads)
      {
#if defined(_WIN32)
        setAffinity(t.native_handle(), 1 << (i % n));
#else
        setAffinity(t.native_handle(), i);
#endif
      }
      // if (threads == 1 || (n > 2 && i <= n - 2))
      // setPriority(t.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

      std::cout << i + 1;
      if(i+1 != threads)
        std::cout << ", ";
    }
    std::cout << std::endl;
 //  mutex.unlock();

  g_start_time = std::chrono::steady_clock::now();
  if (broadcastStats)
  {
    boost::thread BROADCAST(BroadcastServer::serverThread, &rate30sec, &accepted, &rejected, versionString, reportInterval);
  }

  while (!isConnected)
  {
    boost::this_thread::yield();
  }

  // boost::thread reportThread([&]() {
    // Set an expiry time relative to now.
    update_timer.expires_after(std::chrono::seconds(1));

    // Start an asynchronous wait.
    update_timer.async_wait(update_handler);
    my_context.run();
  // });
  // setPriority(reportThread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);

  for(;;) {
    std::this_thread::yield();
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
DWORD_PTR SetThreadAffinityWithGroups(HANDLE threadHandle, DWORD_PTR coreIndex) {
    DWORD numGroups = GetActiveProcessorGroupCount();
    DWORD numProcessorsInGroup = GetMaximumProcessorCount(0); // Assume uniform group sizes

    // Calculate group and processor within the group
    DWORD group = static_cast<DWORD>(coreIndex / numProcessorsInGroup);
    DWORD processorInGroup = static_cast<DWORD>(coreIndex % numProcessorsInGroup);

    if (group < numGroups) {
        GROUP_AFFINITY groupAffinity = {};
        groupAffinity.Group = static_cast<WORD>(group);
        groupAffinity.Mask = static_cast<KAFFINITY>(1ULL << processorInGroup);

        GROUP_AFFINITY previousGroupAffinity;
        if (!SetThreadGroupAffinity(threadHandle, &groupAffinity, &previousGroupAffinity)) {
            return 0; // Fail case, return 0 like SetThreadAffinityMask
        }

        // Return the previous affinity mask for compatibility with your code
        return previousGroupAffinity.Mask;
    }

    return 0; // If out of bounds
};
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

void getWork(bool isDev, int algo)
{
  net::io_context ioc;
  ssl::context ctx = ssl::context{ssl::context::tlsv12_client};
  load_root_certificates(ctx);

  bool caughtDisconnect = false;

connectionAttempt:
  bool *B = isDev ? &devConnected : &isConnected;
  *B = false;
 //  mutex.lock();
  setcolor(BRIGHT_YELLOW);
  std::cout << "Connecting...\n" << std::flush;
  setcolor(BRIGHT_WHITE);
 //  mutex.unlock();
  try
  {
    // Launch the asynchronous operation
    bool err = false;
    if (isDev)
    {
      std::string DAEMONPROTO, HOST, WORKER, PORT;
      switch (algo)
      {
        case DERO_HASH:
        {
          DAEMONPROTO = "";
          HOST = defaultHost[DERO_HASH];
          WORKER = devWorkerName;
          PORT = devPort[DERO_HASH];
          break;
        }
        case XELIS_HASH:
        {
          DAEMONPROTO = daemonProto;
          HOST = host;
          WORKER = devWorkerName;
          PORT = port;
          break;
        }
        case SPECTRE_X:
        {
          DAEMONPROTO = daemonProto;
          HOST = host;
          WORKER = devWorkerName;
          PORT = port;
          break;
        }
      }
      boost::asio::spawn(ioc, std::bind(&do_session, DAEMONPROTO, HOST, PORT, devSelection[algo], WORKER, algo, std::ref(ioc), std::ref(ctx), std::placeholders::_1, true),
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
      boost::asio::spawn(ioc, std::bind(&do_session, daemonProto, host, port, wallet, workerName, algo, std::ref(ioc), std::ref(ctx), std::placeholders::_1, false),
                         // on completion, spawn will call this function
                         [&](std::exception_ptr ex)
                         {
                           if (ex)
                           {
                             std::rethrow_exception(ex);
                             err = true;
                           }
                         });
    ioc.run();
    if (err)
    {
      if (!isDev)
      {
       //  mutex.lock();
        setcolor(RED);
        std::cerr << "\nError establishing connections" << std::endl
                  << "Will try again in 10 seconds...\n\n" << std::flush;
        setcolor(BRIGHT_WHITE);
       //  mutex.unlock();
      }
      boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
      ioc.reset();
      goto connectionAttempt;
    }
    else
    {
      caughtDisconnect = false;
    }
  }
  catch (...)
  {
    if (!isDev)
    {
     //  mutex.lock();
      setcolor(RED);
      std::cerr << "\nError establishing connections" << std::endl
                << "Will try again in 10 seconds...\n\n" << std::flush;
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
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
    ioc.reset();
    goto connectionAttempt;
  }
  while (*B)
  {
    caughtDisconnect = false;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  }
  if (!isDev)
  {
   //  mutex.lock();
    setcolor(RED);
    if (!caughtDisconnect)
      std::cerr << "\nERROR: lost connection" << std::endl
                << "Will try to reconnect in 10 seconds...\n\n";
    else
      std::cerr << "\nError establishing connection" << std::endl
                << "Will try again in 10 seconds...\n\n";

    fflush(stdout);
    setcolor(BRIGHT_WHITE);
   //  mutex.unlock();
  }
  else
  {
   //  mutex.lock();
    setcolor(RED);
    if (!caughtDisconnect)
      std::cerr << "\nERROR: lost connection to dev node (mining will continue)" << std::endl
                << "Will try to reconnect in 10 seconds...\n\n";
    else
      std::cerr << "\nError establishing connection to dev node" << std::endl
                << "Will try again in 10 seconds...\n\n";

    fflush(stdout);
    setcolor(BRIGHT_WHITE);
   //  mutex.unlock();
  }
  caughtDisconnect = true;
  boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
  ioc.reset();
  goto connectionAttempt;
}

void benchmark(int tid)
{

  byte work[MINIBLOCK_SIZE];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::array<uint8_t, 48> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                { return dist(gen); });
  std::memcpy(work, buf.data(), buf.size());

  boost::this_thread::sleep_for(boost::chrono::milliseconds(125));

  int64_t localJobCounter;

  int32_t i = 0;

  byte powHash[32];
  // byte powHash2[32];
  workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*worker);
  lookupGen(*worker, lookup2D_global, lookup3D_global);

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  while (!startBenchmark)
  {
  }

  work[MINIBLOCK_SIZE - 1] = (byte)tid;
  while (true)
  {
    boost::json::value myJob;
    {
      std::scoped_lock<boost::mutex> lockGuard(mutex);
      boost::json::value myJob = job;
      localJobCounter = jobCounter;
    }
    byte *b2 = new byte[MINIBLOCK_SIZE];
    hexstrToBytes(std::string(myJob.at("blockhashing_blob").as_string()), b2);
    memcpy(work, b2, MINIBLOCK_SIZE);
    delete[] b2;

    while (localJobCounter == jobCounter)
    {
      i++;
      // double which = (double)(rand() % 10000);
      // bool devMine = (devConnected && which < devFee * 100.0);
      std::memcpy(&work[MINIBLOCK_SIZE - 5], &i, sizeof(i));
      // swap endianness
      if (littleEndian())
      {
        std::swap(work[MINIBLOCK_SIZE - 5], work[MINIBLOCK_SIZE - 2]);
        std::swap(work[MINIBLOCK_SIZE - 4], work[MINIBLOCK_SIZE - 3]);
      }
      AstroBWTv3(work, MINIBLOCK_SIZE, powHash, *worker, useLookupMine);

      counter.fetch_add(1);
      benchCounter.fetch_add(1);
      if (stopBenchmark)
        break;
    }
    if (stopBenchmark)
      break;
  }
}
