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
#include "DNSResolver.hpp"
#include "net.hpp"
#include "astrobwtv3.h"

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

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <miner.h>

#include <random>

#include <hex.h>
#include <astrobwtv3.h>
#include <xelis-hash.hpp>
#include <spectrex.h>

#include <astrotest.hpp>
#include <thread>

#include <chrono>

#include <hugepages.h>
#include <future>
#include <limits>
#include <libcubwt.cuh>
#include <lookupcompute.h>

#include <openssl/err.h>
#include <openssl/ssl.h>
#include <base64.hpp>

#include <bit>
#include <broadcastServer.hpp>
#include <stratum.h>

#include <exception>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <sched.h>
#define THREAD_PRIORITY_ABOVE_NORMAL -5
#define THREAD_PRIORITY_HIGHEST -20
#define THREAD_PRIORITY_TIME_CRITICAL -20
#endif

#if defined(_WIN32)
LPTSTR lpNxtPage;  // Address of the next page to ask for
DWORD dwPages = 0; // Count of pages gotten so far
DWORD dwPageSize;  // Page size on this computer
#endif


boost::asio::io_context my_context;
// Construct a timer without setting an expiry time.
boost::asio::steady_timer update_timer(my_context);
std::chrono::time_point g_start_time = std::chrono::steady_clock::now();

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

int reportCounter = 0;

//static int firstRejected;

//uint64_t hashrate;
int64_t ourHeight;
int64_t devHeight;

int64_t difficulty;
int64_t difficultyDev;

double doubleDiff;
double doubleDiffDev;

std::vector<int64_t> rate5min;
std::vector<int64_t> rate1min;
std::vector<int64_t> rate30sec;

bool isConnected = false;
bool devConnected = false;

bool beQuiet = false;
/* End definitions from tnn-common.hpp */

/* Start definitions from astrobwtv3.hpp */
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
/* End definitions from astrobwtv3.hpp */

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

std::atomic<int64_t> counter = 0;
std::atomic<int> astroSync = 0;
std::atomic<int64_t> benchCounter = 0;

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

int update_handler(const boost::system::error_code& error)
{
  if (error == boost::asio::error::operation_aborted) {
    return 1;
  }

  // Set an expiry time relative to now.
  update_timer.expires_after(std::chrono::seconds(1));

  // Start an asynchronous wait.
  update_timer.async_wait(update_handler);

  if (!isConnected) {
    return 1;
  }

  reportCounter++;

  auto now = std::chrono::steady_clock::now();
  //auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count();

  auto daysUp = std::chrono::duration_cast<std::chrono::hours>(now - g_start_time).count() / 24;
  auto hoursUp = std::chrono::duration_cast<std::chrono::hours>(now - g_start_time).count() % 24;
  auto minutesUp = std::chrono::duration_cast<std::chrono::minutes>(now - g_start_time).count() % 60;
  auto secondsUp = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time).count() % 60;

  std::scoped_lock<boost::mutex> lockGuard(mutex);
  int64_t currentHashes = counter.load();
  counter.store(0);

  double ratio = 1.0 * 1;
  if (rate30sec.size() <= (30 / 1))
  {
    rate30sec.push_back((int64_t)(currentHashes * ratio));
  }
  else
  {
    rate30sec.erase(rate30sec.begin());
    rate30sec.push_back((int64_t)(currentHashes * ratio));
  }

  int64_t hashrate = 1.0 * std::accumulate(rate30sec.begin(), rate30sec.end(), 0LL) / (rate30sec.size() * 1);
  hashrate = (hashrate * 1.0) / (double)1;

  if (reportCounter >= reportInterval) {
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


    setcolor(BRIGHT_WHITE);
    std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << versionString << " ";
    setcolor(CYAN);
    std::cout << std::setw(2) << std::setprecision(3) << "HASHRATE " << rate << rateSuffix << " | " << std::flush;

    std::string uptime = std::to_string(daysUp) + "d-" +
                  std::to_string(hoursUp) + "h-" +
                  std::to_string(minutesUp) + "m-" +
                  std::to_string(secondsUp) + "s >> ";

    double dPrint;

    switch(miningAlgo) {
      case DERO_HASH:
        dPrint = difficulty;
        break;
      case XELIS_HASH:
        dPrint = difficulty;
        break;
      case SPECTRE_X:
        dPrint = doubleDiff;
        break;
    }

    std::cout << std::setw(2) << "ACCEPTED " << accepted << std::setw(2) << " | REJECTED " << rejected
              << std::setw(2) << " | DIFFICULTY " << dPrint << std::setw(2) << " | UPTIME " << uptime << std::flush;
    setcolor(BRIGHT_WHITE); 
    reportCounter = 0;
  }

  return 0;
}

void initializeExterns() {
  numAstroFuncs = std::size(allAstroFuncs); //sizeof(allAstroFuncs)/sizeof(allAstroFuncs[0]);
}

int main(int argc, char **argv)
{
  srand(time(NULL)); // Placing higher here to ensure the effect cascades through the entire program

  initWolfLUT();
  initializeExterns();
  // Check command line arguments.
  lookup2D_global = (uint16_t *)malloc_huge_pages(regOps_size * (256 * 256) * sizeof(uint16_t));
  lookup3D_global = (byte *)malloc_huge_pages(branchedOps_size * (256 * 256) * sizeof(byte));

  oneLsh256 = Num(1) << 256;
  maxU256 = Num(2).pow(256) - 1;

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

  SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
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

  if (vm.count("spectre-test"))
  {
    return SpectreX::test();
  }

  if (vm.count("xelis-test"))
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
  if (vm.count("dero-test"))
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
    std::cout << "ERROR: Invalid coin symbol: " << symbol << std::endl;
    setcolor(BRIGHT_YELLOW);
    it = coinSelector.begin();
    printf("Supported symbols are:\n");
    while (it != coinSelector.end())
    {
      printf("%s\n", it->first.c_str());
      it++;
    }
    printf("\n");
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

  if (wallet.find_last_of(".") != std::string::npos) {
    workerName = wallet.substr(wallet.find_last_of(".") + 1);
  }

  if (threads == 0)
  {
    if (gpuMine)
      threads = 1;
    else
    {
      while (true)
      {
        setcolor(CYAN);
        printf("%s\n", threadPrompt);
        setcolor(BRIGHT_WHITE);

        std::string cmdLine;
        std::getline(std::cin, cmdLine);
        if (cmdLine != "" && cmdLine.find_first_not_of(' ') != std::string::npos)
        {
          try
          {
            threads = std::stoi(cmdLine.c_str());
            break;
          }
          catch (...)
          {
            printf("ERROR: invalid threads parameter... must be an integer\n");
            continue;
          }
        }
        else
        {
          setcolor(BRIGHT_YELLOW);
          printf("Default value will be used: 1\n\n");
          setcolor(BRIGHT_WHITE);
          threads = 1;
          break;
        }

        if (threads == 0)
          threads = 1;
        break;
      }
    }
  }

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
      setAffinity(t.native_handle(), 1 << (i % n));
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

      boost::thread t(mine, i + 1, miningAlgo);

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

  // Set an expiry time relative to now.
  update_timer.expires_after(std::chrono::seconds(1));

  // Start an asynchronous wait.
  update_timer.async_wait(update_handler);
  my_context.run();

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

/*
void update(std::chrono::steady_clock::time_point start_time)
{
  auto beginning = start_time;
  boost::this_thread::yield();

startReporting:
  while (true)
  {
    if (!isConnected)
      break;

    auto now = std::chrono::steady_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

    auto daysUp = std::chrono::duration_cast<std::chrono::hours>(now - beginning).count() / 24;
    auto hoursUp = std::chrono::duration_cast<std::chrono::hours>(now - beginning).count() % 24;
    auto minutesUp = std::chrono::duration_cast<std::chrono::minutes>(now - beginning).count() % 60;
    auto secondsUp = std::chrono::duration_cast<std::chrono::seconds>(now - beginning).count() % 60;

    if (milliseconds >= reportInterval * 1000)
    {
      std::scoped_lock<boost::mutex> lockGuard(mutex);
      start_time = now;
      int64_t currentHashes = counter.load();
      counter.store(0);

      // if (rate1min.size() <= 60 / reportInterval)
      // {
      //   rate1min.push_back(currentHashes);
      // }
      // else
      // {
      //   rate1min.erase(rate1min.begin());
      //   rate1min.push_back(currentHashes);
      // }

      float ratio = (1000.0f / milliseconds) * reportInterval;
      if (rate30sec.size() <= 30 / reportInterval)
      {
        rate30sec.push_back((int64_t)(currentHashes * ratio));
      }
      else
      {
        rate30sec.erase(rate30sec.begin());
        rate30sec.push_back((int64_t)(currentHashes * ratio));
      }

      int64_t hashrate = 1.0 * std::accumulate(rate30sec.begin(), rate30sec.end(), 0LL) / (rate30sec.size() * reportInterval);

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

      setcolor(BRIGHT_WHITE);
      std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << versionString << " ";
      setcolor(CYAN);
      std::cout << std::setw(2) << std::setprecision(3) << "HASHRATE " << rate << rateSuffix << " | " << std::flush;

      std::string uptime = std::to_string(daysUp) + "d-" +
                     std::to_string(hoursUp) + "h-" +
                     std::to_string(minutesUp) + "m-" +
                     std::to_string(secondsUp) + "s >> ";

      double dPrint;

      switch(miningAlgo) {
        case DERO_HASH:
          dPrint = difficulty;
          break;
        case XELIS_HASH:
          dPrint = difficulty;
          break;
        case SPECTRE_X:
          dPrint = doubleDiff;
          break;
      }

      std::cout << std::setw(2) << "ACCEPTED " << accepted << std::setw(2) << " | REJECTED " << rejected
                << std::setw(2) << " | DIFFICULTY " << dPrint << std::setw(2) << " | UPTIME " << uptime << std::flush;
      setcolor(BRIGHT_WHITE);
     //  mutex.unlock();
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }
  while (true)
  {
    if (isConnected)
    {
      rate30sec.clear();
      counter.store(0);
      start_time = std::chrono::steady_clock::now();
      beginning = start_time;
      break;
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
  }
  goto startReporting;
}
*/

void setAffinity(boost::thread::native_handle_type t, int core)
{
#if defined(_WIN32)

  HANDLE threadHandle = t;

  // Affinity on Windows makes hashing slower atm
  // Set the CPU affinity mask to the first processor (core 0)
  DWORD_PTR affinityMask = core; // Set to the first processor
  DWORD_PTR previousAffinityMask = SetThreadAffinityMask(threadHandle, affinityMask);
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
  std::cout << "Connecting...\n";
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
                  << "Will try again in 10 seconds...\n\n";
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
                << "Will try again in 10 seconds...\n\n";
      setcolor(BRIGHT_WHITE);
     //  mutex.unlock();
    }
    else
    {
     //  mutex.lock();
      setcolor(RED);
      std::cerr << "Dev connection error\n";
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
    boost::json::value myJob = job;
    boost::json::value myJobDev = devJob;
    localJobCounter = jobCounter;

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

void mine(int tid, int algo)
{
  switch (algo)
  {
  case DERO_HASH:
    mineDero(tid);
  case XELIS_HASH:
    mineXelis(tid);
  case SPECTRE_X:
    mineSpectre(tid);
  }
}

void mineDero(int tid)
{

  byte random_buf[12];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::array<uint8_t, 12> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                { return dist(gen); });
  std::memcpy(random_buf, buf.data(), buf.size());

  boost::this_thread::sleep_for(boost::chrono::milliseconds(125));


  int64_t localJobCounter;
  byte powHash[32];
  // byte powHash2[32];
  byte devWork[MINIBLOCK_SIZE*DERO_BATCH];
  byte work[MINIBLOCK_SIZE*DERO_BATCH];

  workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*worker);
  lookupGen(*worker, nullptr, nullptr);

  // std::cout << *worker << std::endl;

waitForJob:

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  while (true)
  {
    try
    {
     //  mutex.lock();
      boost::json::value myJob = job;
      boost::json::value myJobDev = devJob;
      localJobCounter = jobCounter;
     //  mutex.unlock();

      byte *b2 = new byte[MINIBLOCK_SIZE];
      hexstrToBytes(std::string(myJob.at("blockhashing_blob").as_string()), b2);
      for (int i = 0; i < DERO_BATCH; i++) {
        memcpy(work + i*MINIBLOCK_SIZE, b2, MINIBLOCK_SIZE);
      }
      delete[] b2;

      if (devConnected)
      {
        byte *b2d = new byte[MINIBLOCK_SIZE];
        hexstrToBytes(std::string(myJobDev.at("blockhashing_blob").as_string()), b2d);
        for (int i = 0; i < DERO_BATCH; i++) {
          memcpy(devWork + i*MINIBLOCK_SIZE, b2d, MINIBLOCK_SIZE);
        }
        delete[] b2d;
      }

      for (int i = 0; i < DERO_BATCH; i++) {
        memcpy(&work[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 12], random_buf, 12);
        memcpy(&devWork[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 12], random_buf, 12);

        work[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 1] = (byte)tid;
        devWork[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 1] = (byte)tid;
      }

      if ((work[0] & 0xf) != 1)
      { // check  version
       //  mutex.lock();
        std::cerr << "Unknown version, please check for updates: "
                  << "version" << (work[0] & 0x1f) << std::endl;
       //  mutex.unlock();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        continue;
      }
      double which;
      bool devMine = false;
      bool submit = false;
      int64_t DIFF;
      Num cmpDiff;
      // DIFF = 5000;

      std::string hex;
      int32_t nonce = 0;
      while (localJobCounter == jobCounter)
      {
        which = (double)(rand() % 10000);
        devMine = (devConnected && which < devFee * 100.0);
        DIFF = devMine ? difficultyDev : difficulty;

        // printf("Difficulty: %" PRIx64 "\n", DIFF);

        cmpDiff = ConvertDifficultyToBig(DIFF, DERO_HASH);
        nonce += DERO_BATCH;
        byte *WORK = devMine ? &devWork[0] : &work[0];

        for (int i = 0; i < DERO_BATCH; i++) {
          int N = nonce + i;
          memcpy(&WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 5], &N, sizeof(N));
        }

        // swap endianness
        if (littleEndian())
        {
          for (int i = 0; i < DERO_BATCH; i++) {
            std::swap(WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 5], WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 2]);
            std::swap(WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 4], WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 3]);
          }
        }

        // for (int i = 0; i < MINIBLOCK_SIZE; i++) {
        //   printf("%02x", WORK[i]);
        // }
        // printf("\n");
        AstroBWTv3_batch(WORK, MINIBLOCK_SIZE, powHash, *worker, useLookupMine);
        // AstroBWTv3_batch((byte*)"b", 1, powHash, *worker, useLookupMine);
        // for (int i = 0; i < 32; i++) {
        //   printf("%02x", powHash[i]);
        // }
        // printf("\n");
        // AstroBWTv3(&WORK[0], MINIBLOCK_SIZE, powHash, *worker, useLookupMine);
        // AstroBWTv3((byte*)("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\0"), MINIBLOCK_SIZE, powHash, *worker, useLookupMine);

        counter.fetch_add(DERO_BATCH);
        submit = devMine ? !submittingDev : !submitting;

        for (int i = 0; i < DERO_BATCH; i++) {
          byte *currHash = &powHash[32*i];
          if (CheckHash(currHash, cmpDiff, DERO_HASH))
          {
            if (!submit) {
              for(;;) {
                if (submit || localJobCounter != jobCounter)
                  break;
                boost::this_thread::yield();
              }
            }
            if (localJobCounter != jobCounter)
                  break;
            // printf("work: %s, hash: %s\n", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str(), hexStr(powHash, 32).c_str());
            boost::lock_guard<boost::mutex> lock(mutex);
            if (devMine)
            {
              setcolor(CYAN);
              std::cout << "\n(DEV) Thread " << tid << " found a dev share\n";
              setcolor(BRIGHT_WHITE);
              devShare = {
                  {"jobid", myJobDev.at("jobid").as_string().c_str()},
                  {"mbl_blob", hexStr(&WORK[MINIBLOCK_SIZE*i], MINIBLOCK_SIZE).c_str()}};
              submittingDev = true;
              data_ready = true;
            }
            else
            {
              setcolor(BRIGHT_YELLOW);
              std::cout << "\nThread " << tid << " found a nonce!\n";
              setcolor(BRIGHT_WHITE);
              share = {
                  {"jobid", myJob.at("jobid").as_string().c_str()},
                  {"mbl_blob", hexStr(&WORK[MINIBLOCK_SIZE*i], MINIBLOCK_SIZE).c_str()}};
              submitting = true;
              data_ready = true;
            }
            cv.notify_all();
          }
        }

        if (!isConnected)
          break;
      }
      if (!isConnected)
        break;
    }
    catch (std::exception& e)
    {
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl;
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}

void mineXelis_v1(int tid)
{
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint64_t i = 0;
  uint64_t i_dev = 0;

  byte powHash[32];
  alignas(64) byte work[XELIS_BYTES_ARRAY_INPUT] = {0};
  alignas(64) byte devWork[XELIS_BYTES_ARRAY_INPUT] = {0};
  alignas(64) byte FINALWORK[XELIS_BYTES_ARRAY_INPUT] = {0};

  alignas(64) workerData_xelis *worker = (workerData_xelis *)malloc_huge_pages(sizeof(workerData_xelis));

waitForJob:

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  while (true)
  {
    try
    {
     //  mutex.lock();
      boost::json::value myJob = job;
      boost::json::value myJobDev = devJob;
      localJobCounter = jobCounter;

     //  mutex.unlock();

      if (!myJob.at("template").is_string())
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[XELIS_TEMPLATE_SIZE];
        switch (protocol)
        {
        case XELIS_SOLO:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        case XELIS_XATUM:
        {
          std::string b64 = base64::from_base64(std::string(myJob.at("template").as_string().c_str()));
          memcpy(b2, b64.data(), b64.size());
          break;
        }
        case XELIS_STRATUM:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        }
        memcpy(work, b2, XELIS_TEMPLATE_SIZE);
        delete[] b2;
        localOurHeight = ourHeight;
        i = 0;
      }

      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          byte *b2d = new byte[XELIS_TEMPLATE_SIZE];
          switch (protocol)
          {
          case XELIS_SOLO:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          case XELIS_XATUM:
          {
            std::string b64 = base64::from_base64(std::string(myJobDev.at("template").as_string().c_str()));
            memcpy(b2d, b64.data(), b64.size());
            break;
          }
          case XELIS_STRATUM:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          }
          memcpy(devWork, b2d, XELIS_TEMPLATE_SIZE);
          delete[] b2d;
          localDevHeight = devHeight;
          i_dev = 0;
        }
      }

      bool devMine = false;
      double which;
      bool submit = false;
      uint64_t DIFF;
      Num cmpDiff;

      while (localJobCounter == jobCounter)
      {
        which = (double)(rand() % 10000);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        DIFF = devMine ? difficultyDev : difficulty;
        if (DIFF == 0)
          continue;
        cmpDiff = ConvertDifficultyToBig(DIFF, XELIS_HASH);

        uint64_t *nonce = devMine ? &i_dev : &i;
        (*nonce)++;

        // printf("nonce = %llu\n", *nonce);

        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        byte *nonceBytes = &WORK[40];
        uint64_t n = ((tid - 1) % (256 * 256)) | ((rand()%256) << 16) | ((*nonce) << 24);
        memcpy(nonceBytes, (byte *)&n, 8);

        // if (littleEndian())
        // {
        //   std::swap(nonceBytes[7], nonceBytes[0]);
        //   std::swap(nonceBytes[6], nonceBytes[1]);
        //   std::swap(nonceBytes[5], nonceBytes[2]);
        //   std::swap(nonceBytes[4], nonceBytes[3]);
        // }

        if (localJobCounter != jobCounter)
          break;

        // std::copy(WORK, WORK + XELIS_TEMPLATE_SIZE, FINALWORK);
        memcpy(FINALWORK, WORK, XELIS_BYTES_ARRAY_INPUT);
        
        xelis_hash(FINALWORK, *worker, powHash);

        if (littleEndian())
        {
          std::reverse(powHash, powHash + 32);
        }

        counter.fetch_add(1);
        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight)
          break;

        if (CheckHash(powHash, cmpDiff, XELIS_HASH))
        {
          if (!submit) {
            for(;;) {
              if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
                break;
              boost::this_thread::yield();
            }
          }
          if (protocol == XELIS_XATUM && littleEndian())
          {
            std::reverse(powHash, powHash + 32);
          }
          // if (protocol == XELIS_STRATUM && littleEndian())
          // {
          //   std::reverse((byte*)&n, (byte*)n + 8);
          // }

          std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          std::string foundBlob = hexStr(&WORK[0], XELIS_TEMPLATE_SIZE);
          if (devMine)
          {
           //  mutex.lock();
            if (localJobCounter != jobCounter || localDevHeight != devHeight)
            {
             //  mutex.unlock();
              break;
            }
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n";
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case XELIS_SOLO:
              devShare = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case XELIS_XATUM:
              devShare = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case XELIS_STRATUM:
              devShare = {{{"id", XelisStratum::submitID},
                           {"method", XelisStratum::submit.method.c_str()},
                           {"params", {devWorkerName,                                 // WORKER
                                       myJobDev.at("jobId").as_string().c_str(), // JOB ID
                                       hexStr((byte *)&n, 8).c_str()}}}};
              break;
            }
            submittingDev = true;
           //  mutex.unlock();
          }
          else
          {
           //  mutex.lock();
            if (localJobCounter != jobCounter || localOurHeight != ourHeight)
            {
             //  mutex.unlock();
              break;
            }
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n";
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case XELIS_SOLO:
              share = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case XELIS_XATUM:
              share = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case XELIS_STRATUM:
              share = {{{"id", XelisStratum::submitID},
                        {"method", XelisStratum::submit.method.c_str()},
                        {"params", {workerName,                                   // WORKER
                                    myJob.at("jobId").as_string().c_str(), // JOB ID
                                    hexStr((byte *)&n, 8).c_str()}}}};

              // std::cout << "blob: " << hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str() << std::endl;
              // std::cout << "hash: " << hexStr(&powHash[0], 32) << std::endl;
              std::vector<char> diffHex;
              cmpDiff.print(diffHex, 16);
              // std::cout << "difficulty (LE): " << std::string(diffHex.data()).c_str() << std::endl;
              // printf("blob: %s\n", foundBlob.c_str());
              // printf("hash (BE): %s\n", hexStr(&powHash[0], 32).c_str());
              // printf("nonce (Full bytes for injection): %s\n", hexStr((byte *)&n, 8).c_str());

              break;
            }
            submitting = true;
           //  mutex.unlock();
          }
        }

        if (!isConnected)
          break;
      }
      if (!isConnected)
        break;
    }
    catch (std::exception& e)
    {
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl;
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}

void mineXelis(int tid)
{
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint64_t i = 0;
  uint64_t i_dev = 0;

  byte powHash[32];
  alignas(64) byte work[XELIS_TEMPLATE_SIZE] = {0};
  alignas(64) byte devWork[XELIS_TEMPLATE_SIZE] = {0};
  alignas(64) byte FINALWORK[XELIS_TEMPLATE_SIZE] = {0};

  alignas(64) workerData_xelis_v2 *worker = (workerData_xelis_v2 *)malloc_huge_pages(sizeof(workerData_xelis));

waitForJob:

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  while (true)
  {
    try
    {
     //  mutex.lock();
      boost::json::value myJob = job;
      boost::json::value myJobDev = devJob;
      localJobCounter = jobCounter;

     //  mutex.unlock();

      if (!myJob.at("miner_work").is_string())
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[XELIS_TEMPLATE_SIZE];
        switch (protocol)
        {
        case XELIS_SOLO:
          hexstrToBytes(std::string(myJob.at("miner_work").as_string()), b2);
          break;
        case XELIS_XATUM:
        {
          std::string b64 = base64::from_base64(std::string(myJob.at("miner_work").as_string().c_str()));
          memcpy(b2, b64.data(), b64.size());
          break;
        }
        case XELIS_STRATUM:
          hexstrToBytes(std::string(myJob.at("miner_work").as_string()), b2);
          break;
        }
        memcpy(work, b2, XELIS_TEMPLATE_SIZE);
        delete[] b2;
        localOurHeight = ourHeight;
        i = 0;
      }

      if (devConnected && myJobDev.at("miner_work").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          byte *b2d = new byte[XELIS_TEMPLATE_SIZE];
          switch (protocol)
          {
          case XELIS_SOLO:
            hexstrToBytes(std::string(myJobDev.at("miner_work").as_string()), b2d);
            break;
          case XELIS_XATUM:
          {
            std::string b64 = base64::from_base64(std::string(myJobDev.at("miner_work").as_string().c_str()));
            memcpy(b2d, b64.data(), b64.size());
            break;
          }
          case XELIS_STRATUM:
            hexstrToBytes(std::string(myJobDev.at("miner_work").as_string()), b2d);
            break;
          }
          memcpy(devWork, b2d, XELIS_TEMPLATE_SIZE);
          delete[] b2d;
          localDevHeight = devHeight;
          i_dev = 0;
        }
      }

      bool devMine = false;
      double which;
      bool submit = false;
      uint64_t DIFF;
      Num cmpDiff;

      while (localJobCounter == jobCounter)
      {
        which = (double)(rand() % 10000);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        DIFF = devMine ? difficultyDev : difficulty;
        if (DIFF == 0)
          continue;
        cmpDiff = ConvertDifficultyToBig(DIFF, XELIS_HASH);

        uint64_t *nonce = devMine ? &i_dev : &i;
        (*nonce)++;

        // printf("nonce = %llu\n", *nonce);

        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        byte *nonceBytes = &WORK[40];
        uint64_t n = ((tid - 1) % (256 * 256)) | ((rand()%256) << 16) | ((*nonce) << 24);
        memcpy(nonceBytes, (byte *)&n, 8);

        // if (littleEndian())
        // {
        //   std::swap(nonceBytes[7], nonceBytes[0]);
        //   std::swap(nonceBytes[6], nonceBytes[1]);
        //   std::swap(nonceBytes[5], nonceBytes[2]);
        //   std::swap(nonceBytes[4], nonceBytes[3]);
        // }

        if (localJobCounter != jobCounter)
          break;

        memcpy(FINALWORK, WORK, XELIS_TEMPLATE_SIZE);
        
        xelis_hash_v2(FINALWORK, *worker, powHash);

        if (littleEndian())
        {
          std::reverse(powHash, powHash + 32);
        }

        counter.fetch_add(1);
        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight)
          break;

        if (CheckHash(powHash, cmpDiff, XELIS_HASH))
        {
          if (!submit) {
            for(;;) {
              if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
                break;
              boost::this_thread::yield();
            }
          }
          if (protocol == XELIS_XATUM && littleEndian())
          {
            std::reverse(powHash, powHash + 32);
          }
          // if (protocol == XELIS_STRATUM && littleEndian())
          // {
          //   std::reverse((byte*)&n, (byte*)n + 8);
          // }

          std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          std::string foundBlob = hexStr(&WORK[0], XELIS_TEMPLATE_SIZE);
          if (devMine)
          {
           //  mutex.lock();
            if (localJobCounter != jobCounter || localDevHeight != devHeight)
            {
             //  mutex.unlock();
              break;
            }
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n";
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case XELIS_SOLO:
              devShare = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case XELIS_XATUM:
              devShare = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case XELIS_STRATUM:
              devShare = {{{"id", XelisStratum::submitID},
                           {"method", XelisStratum::submit.method.c_str()},
                           {"params", {devWorkerName,                                 // WORKER
                                       myJobDev.at("jobId").as_string().c_str(), // JOB ID
                                       hexStr((byte *)&n, 8).c_str()}}}};
              break;
            }
            submittingDev = true;
            data_ready = true;
           //  mutex.unlock();
          }
          else
          {
           //  mutex.lock();
            if (localJobCounter != jobCounter || localOurHeight != ourHeight)
            {
             //  mutex.unlock();
              break;
            }
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n";
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case XELIS_SOLO:
              share = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case XELIS_XATUM:
              share = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case XELIS_STRATUM:
              share = {{{"id", XelisStratum::submitID},
                        {"method", XelisStratum::submit.method.c_str()},
                        {"params", {workerName,                                   // WORKER
                                    myJob.at("jobId").as_string().c_str(), // JOB ID
                                    hexStr((byte *)&n, 8).c_str()}}}};

              // std::cout << "blob: " << hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str() << std::endl;
              // std::cout << "hash: " << hexStr(&powHash[0], 32) << std::endl;
              std::vector<char> diffHex;
              cmpDiff.print(diffHex, 16);
              // std::cout << "difficulty (LE): " << std::string(diffHex.data()).c_str() << std::endl;
              // printf("blob: %s\n", foundBlob.c_str());
              // printf("hash (BE): %s\n", hexStr(&powHash[0], 32).c_str());
              // printf("nonce (Full bytes for injection): %s\n", hexStr((byte *)&n, 8).c_str());

              break;
            }
            submitting = true;
            data_ready = true;
           //  mutex.unlock();
          }
          cv.notify_all();
        }

        if (!isConnected) {
          data_ready = true;
          cv.notify_all();
          break;
        }
      }
      if (!isConnected) {
        data_ready = true;
        cv.notify_all();
        break;
      }
    }
    catch (std::exception& e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}




void mineSpectre(int tid)
{
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint64_t i = 0;
  uint64_t i_dev = 0;

  byte powHash[32];
  byte work[SpectreX::INPUT_SIZE] = {0};
  byte devWork[SpectreX::INPUT_SIZE] = {0};

  workerData *astroWorker = (workerData *)malloc_huge_pages(sizeof(workerData));
  SpectreX::worker *worker = (SpectreX::worker *)malloc_huge_pages(sizeof(SpectreX::worker));
  initWorker(*astroWorker);
  lookupGen(*astroWorker, nullptr, nullptr);
  worker->astroWorker = astroWorker;

  workerData *devAstroWorker = (workerData *)malloc_huge_pages(sizeof(workerData));
  SpectreX::worker *devWorker = (SpectreX::worker *)malloc_huge_pages(sizeof(SpectreX::worker));
  initWorker(*devAstroWorker);
  lookupGen(*devAstroWorker, nullptr, nullptr);
  devWorker->astroWorker = devAstroWorker;

waitForJob:

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  while (true)
  {
    try
    {
     //  mutex.lock();
      boost::json::value myJob = job;
      boost::json::value myJobDev = devJob;
      localJobCounter = jobCounter;
     //  mutex.unlock();

      // printf("looping somewhere\n");

      if (!myJob.at("template").is_string()) {
        continue;
      }
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[SpectreX::INPUT_SIZE];
        switch (protocol)
        {
        case SPECTRE_SOLO:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        case SPECTRE_STRATUM:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        }
        memcpy(work, b2, SpectreX::INPUT_SIZE);
        // SpectreX::genPrePowHash(b2, *worker);/
        // SpectreX::newMatrix(b2, worker->mat);
        delete[] b2;
        localOurHeight = ourHeight;
        i = 0;
      }

      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          byte *b2d = new byte[SpectreX::INPUT_SIZE];
          switch (protocol)
          {
          case SPECTRE_SOLO:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          case SPECTRE_STRATUM:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          }
          memcpy(devWork, b2d, SpectreX::INPUT_SIZE);
          // SpectreX::genPrePowHash(b2d, *devWorker);
          // SpectreX::newMatrix(b2d, devWorker->mat);
          delete[] b2d;
          localDevHeight = devHeight;
          i_dev = 0;
        }
      }

      bool devMine = false;
      double which;
      bool submit = false;
      double DIFF = 1;
      Num cmpDiff;

      // printf("end of job application\n");
      while (localJobCounter == jobCounter)
      {
        which = (double)(rand() % 10000);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        DIFF = devMine ? doubleDiffDev : doubleDiff;
        if (DIFF == 0)
          continue;

        // cmpDiff = ConvertDifficultyToBig(DIFF, SPECTRE_X);
        cmpDiff = SpectreX::diffToTarget(DIFF);

        uint64_t *nonce = devMine ? &i_dev : &i;
        (*nonce)++;

        // printf("nonce = %llu\n", *nonce);

        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        byte *nonceBytes = &WORK[72];
        uint64_t n;
        
        int enLen = 0;
        
        boost::json::value &J = devMine ? myJobDev : myJob;
        if (!J.as_object().if_contains("extraNonce") || J.at("extraNonce").as_string().size() == 0)
          n = ((tid - 1) % (256 * 256)) | ((rand() % 256) << 16) | ((*nonce) << 24);
        else {
          uint64_t eN = std::stoull(std::string(J.at("extraNonce").as_string().c_str()), NULL, 16);
          enLen = std::string(J.at("extraNonce").as_string()).size()/2;
          n = ((tid - 1) % (256 * 256)) | ((*nonce) << 16) | (eN << 56);
        }
        memcpy(nonceBytes, (byte *)&n, 8);

        // printf("after nonce: %s\n", hexStr(WORK, SpectreX::INPUT_SIZE).c_str());

        if (localJobCounter != jobCounter)
          break;

        SpectreX::worker &usedWorker = devMine ? *devWorker : *worker;
        SpectreX::hash(usedWorker, WORK, SpectreX::INPUT_SIZE, powHash);

        // if (littleEndian())
        // {
        //   std::reverse(powHash, powHash + 32);
        // }

        counter.fetch_add(1);
        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight)
          break;


        if (Num(hexStr(powHash, 32).c_str(), 16) <= cmpDiff)
        {
          if (!submit) {
            for(;;) {
              if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
                break;
              boost::this_thread::yield();
            }
          }
          // if (littleEndian())
          // {
          //   std::reverse(powHash, powHash + 32);
          // }
        //   std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          boost::lock_guard<boost::mutex> lock(mutex);
          if (devMine)
          {
            // std::scoped_lock<boost::mutex> lockGuard(devMutex);
            if (localJobCounter != jobCounter || localDevHeight != devHeight)
            {
              break;
            }
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n";
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case SPECTRE_SOLO:
              devShare = {{"block_template", hexStr(&WORK[0], SpectreX::INPUT_SIZE).c_str()}};
              break;
            case SPECTRE_STRATUM:
              std::vector<char> nonceStr;
              // Num(std::to_string((n << enLen*8) >> enLen*8).c_str(),10).print(nonceStr, 16);
              Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
              devShare = {{{"id", SpectreStratum::submitID},
                        {"method", SpectreStratum::submit.method.c_str()},
                        {"params", {devWorkerName,                                   // WORKER
                                    myJobDev.at("jobId").as_string().c_str(), // JOB ID
                                    std::string(nonceStr.data()).c_str()}}}};

              break;
            }
            submittingDev = true;
            data_ready = true;
          }
          else
          {
            // std::scoped_lock<boost::mutex> lockGuard(userMutex);
            if (localJobCounter != jobCounter || localOurHeight != ourHeight)
            {
              break;
            }
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n";
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case SPECTRE_SOLO:
              share = {{"block_template", hexStr(&WORK[0], SpectreX::INPUT_SIZE).c_str()}};
              break;
            case SPECTRE_STRATUM:
              std::vector<char> nonceStr;
              // Num(std::to_string((n << enLen*8) >> enLen*8).c_str(),10).print(nonceStr, 16);
              Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
              share = {{{"id", SpectreStratum::submitID},
                        {"method", SpectreStratum::submit.method.c_str()},
                        {"params", {workerName,                                   // WORKER
                                    myJob.at("jobId").as_string().c_str(), // JOB ID
                                    std::string(nonceStr.data()).c_str()}}}};

              // std::cout << "blob: " << hexStr(&WORK[0], SpectreX::INPUT_SIZE).c_str() << std::endl;
              // std::cout << "nonce: " << nonceStr.data() << std::endl;
              // std::cout << "extraNonce: " << hexStr(&WORK[SpectreX::INPUT_SIZE - 48], enLen).c_str() << std::endl;
              // std::cout << "hash: " << hexStr(&powHash[0], 32) << std::endl;
              // std::vector<char> diffHex;
              // cmpDiff.print(diffHex, 16);
              // std::cout << "difficulty (LE): " << std::string(diffHex.data()).c_str() << std::endl;
              // std::cout << "powValue: " << Num(hexStr(powHash, 32).c_str(), 16) << std::endl;
              // std::cout << "target (decimal): " << cmpDiff << std::endl;

              // printf("blob: %s\n", foundBlob.c_str());
              // printf("hash (BE): %s\n", hexStr(&powHash[0], 32).c_str());
              // printf("nonce (Full bytes for injection): %s\n", hexStr((byte *)&n, 8).c_str());

              break;
            }
            submitting = true;
            data_ready = true;
          }
          cv.notify_all();
        }

        if (!isConnected) {
          data_ready = true;
          cv.notify_all();
          break;
        }
      }
      if (!isConnected) {
        data_ready = true;
        cv.notify_all();
        break;
      }
    }
    catch (std::exception& e)
    {
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl;
    }
    if (!isConnected) {
      data_ready = true;
      cv.notify_all();
      break;
    }
  }
  goto waitForJob;
}
