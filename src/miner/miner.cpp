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

#define FMT_HEADER_ONLY

#include "rootcert.h"

#include <boost/program_options.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/json.hpp>

#include <boost/thread.hpp>
#include <boost/atomic.hpp>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <miner.h>

#include <random>

#include <hex.h>
#include <pow.h>
// #include <astrobwtv3_cuda.cuh>
#include <powtest.h>
#include <thread>

#include <chrono>
#include <fmt/format.h>
#include <fmt/printf.h>

#include <hugepages.h>
#include <future>
#include <limits>
#include <libcubwt.cuh>
#include <lookupcompute.h>
#include <xelis-hash.hpp>

#include <bit>

#if defined(_WIN32)
#include <Windows.h>
#else
#include "cpp-dns.hpp"
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

// #include <cuda_runtime.h>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
namespace po = boost::program_options;  // from <boost/program_options.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

boost::mutex mutex;
boost::mutex wsMutex;

boost::json::value job;
boost::json::value devJob;
boost::json::object share;
boost::json::object devShare;

std::string currentBlob;
std::string devBlob;

bool submitting = false;
bool submittingDev = false;

uint16_t *lookup2D_global; // Storage for computed values of 2-byte chunks
byte *lookup3D_global; // Storage for deterministically computed values of 1-byte chunks

int jobCounter;
std::atomic<int64_t> counter = 0;
std::atomic<int64_t> benchCounter = 0;

int blockCounter;
int miniBlockCounter;
int rejected;
int accepted;
int firstRejected;

int64_t ourHeight;
int64_t devHeight;
int64_t difficulty;
int64_t difficultyDev;

std::vector<int64_t> rate5min;
std::vector<int64_t> rate1min;
std::vector<int64_t> rate30sec;

bool isConnected = false;
bool devConnected = false;

using byte = unsigned char;
int bench_duration = -1;
bool stopBenchmark = false;
bool startBenchmark = false;
//------------------------------------------------------------------------------

// Report a failure
void fail(beast::error_code ec, char const *what) noexcept
{
  mutex.lock();
  setcolor(RED);
  std::cerr << what << ": " << ec.message() << "\n";
  setcolor(BRIGHT_WHITE);
  mutex.unlock();
}

// Sends a WebSocket message and prints the response
void do_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev,
    int fixedDiff = 0)
{
  boost::json::error_code jsonEc;
  beast::error_code ec;
  std::string fixedDiffStr = "";
  if(fixedDiff > 0) {
    fixedDiffStr = "#" + std::to_string(fixedDiff);
  }

  // These objects perform our I/O
  int addrCount = 0;
  //bool resolved = false;

  net::ip::address ip_address;

  websocket::stream<
      beast::ssl_stream<beast::tcp_stream>>
      ws(ioc, ctx);

  // If the specified host/pool is not in IP address form, resolve to acquire the IP address
#ifndef _WIN32
  boost::asio::ip::address::from_string(host, ec);
  if (ec)
  {
    // Using cpp-dns to circumvent the issues cause by combining static linking and getaddrinfo()
    // A second io_context is used to enable std::promise
    net::io_context ioc2;
    std::string ip;
    std::promise<void> p;

    YukiWorkshop::DNSResolver d(ioc2);
    d.resolve_a4(host, [&](int err, auto &addrs, auto &qname, auto &cname, uint ttl)
                 {
  if (!err) {
      mutex.lock();
      for (auto &it : addrs) {
        addrCount++;
        ip = it.to_string();
      }
      p.set_value();
  } else {
    p.set_value();
  } });
    ioc2.run();

    std::future<void> f = p.get_future();
    f.get();
    mutex.unlock();

    if (addrCount == 0)
    {
      mutex.lock();
      setcolor(RED);
      std::cerr << "ERROR: Could not resolve " << host << std::endl;
      setcolor(BRIGHT_WHITE);
      mutex.unlock();
      return;
    }

    ip_address = net::ip::address::from_string(ip.c_str(), ec);
  }
  else
  {
    ip_address = net::ip::address::from_string(host, ec);
  }

  tcp::endpoint daemon(ip_address, (uint_least16_t)std::stoi(port.c_str()));
  // Set a timeout on the operation
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(ws).connect(daemon);

#else
  // Look up the domain name
  tcp::resolver resolver(ioc);
  auto const results = resolver.async_resolve(host, port, yield[ec]);
  if (ec)
    return fail(ec, "resolve");

  // Set a timeout on the operation
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  auto daemon = beast::get_lowest_layer(ws).connect(results);
#endif

  // Set SNI Hostname (many hosts need this to handshake successfully)
  if (!SSL_set_tlsext_host_name(
          ws.next_layer().native_handle(),
          host.c_str()))
  {
    ec = beast::error_code(static_cast<int>(::ERR_get_error()),
                           net::error::get_ssl_category());
    return fail(ec, "connect");
  }

  // Update the host string. This will provide the value of the
  // Host HTTP header during the WebSocket handshake.
  // See https://tools.ietf.org/html/rfc7230#section-5.4
  host += ':' + std::to_string(daemon.port());

  // Set a timeout on the operation
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));

  // Set a decorator to change the User-Agent of the handshake
  ws.set_option(websocket::stream_base::decorator(
      [](websocket::request_type &req)
      {
        req.set(http::field::user_agent,
                std::string(BOOST_BEAST_VERSION_STRING) +
                    " websocket-client-coro");
      }));

  // Perform the SSL handshake
  ws.next_layer().async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec)
    return fail(ec, "ssl_handshake");

  // Turn off the timeout on the tcp_stream, because
  // the websocket stream has its own timeout system.
  beast::get_lowest_layer(ws).expires_never();

  // Set suggested timeout settings for the websocket
  ws.set_option(
      websocket::stream_base::timeout::suggested(
          beast::role_type::client));

  // Perform the websocket handshake
  std::stringstream ss;
  ss << "/ws/" << wallet << fixedDiffStr;
  ws.async_handshake(host, ss.str().c_str(), yield[ec]);
  if (ec)
    return fail(ec, "handshake");

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;
  boost::json::value workData;

  while (true)
  {
    try
    {
      buffer.clear();
      workInfo.str("");
      workInfo.clear();

      bool *B = isDev ? &submittingDev : &submitting;

      if (*B)
      {
        boost::json::object *S = isDev ? &devShare : &share;
        std::string msg = boost::json::serialize(*S);
        // mutex.lock();
        // std::cout << msg;
        // mutex.unlock();
        ws.async_write(boost::asio::buffer(msg), yield[ec]);
        if (ec)
        {
          return fail(ec, "async_write");
        }
        *B = false;
      }

      beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(5));
      ws.async_read(buffer, yield[ec]);
      if (!ec)
      {
        // handle getwork feed

        beast::get_lowest_layer(ws).expires_never();
        workInfo << beast::make_printable(buffer.data());

        workData = boost::json::parse(workInfo.str(), jsonEc);
        if (!jsonEc)
        {
          //workData = boost::json::parse(workInfo.str());
          if ((isDev ? (workData.at("height") != devHeight) : (workData.at("height") != ourHeight)))
          {
            // mutex.lock();
            if (isDev)
              devJob = workData;
            else
              job = workData;
            boost::json::value *J = isDev ? &devJob : &job;
            // mutex.unlock();

            if ((*J).at("lasterror") != "")
            {
              std::cerr << "received error: " << (*J).at("lasterror") << std::endl
                        << consoleLine;
            }

            if (!isDev)
            {
              currentBlob = (*J).at("blockhashing_blob").as_string();
              //blockCounter = (*J).at("blocks");
              //miniBlockCounter = (*J).at("miniblocks");
              //rejected = (*J).at("rejected");
              //hashrate = (*J).at("difficultyuint64");
              ourHeight = (*J).at("height").as_int64();
              difficulty = (*J).at("difficultyuint64").as_int64();
              // printf("NEW JOB RECEIVED | Height: %d | Difficulty %" PRIu64 "\n", ourHeight, difficulty);
              accepted = (*J).at("miniblocks").as_int64();
              rejected = (*J).at("rejected").as_int64();
              if (!isConnected)
              {
                mutex.lock();
                setcolor(BRIGHT_YELLOW);
                printf("Mining at: %s/ws/%s%s\n", host.c_str(), wallet.c_str(), fixedDiffStr.c_str());
                setcolor(CYAN);
                printf("Dev fee: %.2f", devFee);
                std::cout << "%" << std::endl;
                setcolor(BRIGHT_WHITE);
                mutex.unlock();
              }
              isConnected = isConnected || true;
              jobCounter++;
            }
            else
            {
              difficultyDev = (*J).at("difficultyuint64").as_int64();
              devBlob = (*J).at("blockhashing_blob").as_string();
              devHeight = (*J).at("height").as_int64();
              if (!devConnected)
              {
                mutex.lock();
                setcolor(CYAN);
                printf("Connected to dev node: %s\n", devPool);
                setcolor(BRIGHT_WHITE);
                mutex.unlock();
              }
              devConnected = devConnected || true;
              jobCounter++;
            }
          }
        }
      }
      else
      {
        bool *B = isDev ? &devConnected : &isConnected;
        (*B) = false;
        fail(ec, "async_read");
        return;
      }
    }
    catch (...)
    {
      std::cout << "ws error\n";
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // // Close the WebSocket connection
  // ws.async_close(websocket::close_code::normal, yield[ec]);
  // if (ec)
  //   return fail(ec, "close");

  // If we get here then the connection is closed gracefully

  // The make_printable() function helps print a ConstBufferSequence
  // std::cout << beast::make_printable(buffer.data()) << std::endl;
}

//------------------------------------------------------------------------------

int main(int argc, char **argv)
{
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
#endif
  setcolor(BRIGHT_WHITE);
  printf("%s", TNN);
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
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
  // Check command line arguments.
  lookup2D_global = (uint16_t *)malloc_huge_pages(regOps_size*(256*256)*sizeof(uint16_t));
  lookup3D_global = (byte *)malloc_huge_pages(branchedOps_size*(256*256)*sizeof(byte));
  oneLsh256 = Num(1) << 256;

  // default values
  bool lockThreads = true;
  devFee = 2.5;

  po::variables_map vm;
  po::options_description opts = get_prog_opts();
  try {
    int style = get_prog_style();
    po::store(po::command_line_parser(argc, argv).options(opts).style(style).run(), vm);
    po::notify(vm);
  }
  catch(std::exception& e)
  {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Remember: Long options now use a double-dash -- instead of a single-dash -\n";
    return -1;
  }
  catch(...)
  {
    std::cerr << "Unknown error!" << "\n";
    return -1;
  }

  if (vm.count("help")) {
    std::cout << opts << std::endl;
    boost::this_thread::sleep_for(boost::chrono::seconds(1));
    return 0;
  }

  if (vm.count("xelis")) {
    std::string input = vm["xelis"].as<std::string>();

    // Prepare the input and scratch pad
    alignas(64) byte input_bytes[BYTES_ARRAY_INPUT];
    std::fill_n(input_bytes, BYTES_ARRAY_INPUT, 0);
    std::copy_n(input.c_str(), std::min(input.size(), static_cast<size_t>(BYTES_ARRAY_INPUT)), input_bytes);

    uint64_t scratch_pad[MEMORY_SIZE];
    std::fill_n(scratch_pad, MEMORY_SIZE, 0);

    byte hash_result[HASH_SIZE];
    std::fill_n(hash_result, HASH_SIZE, 0);

    // Compute the hash
    workerData_xelis worker;
    xelis_hash(input_bytes, worker, hash_result);

    // Print the hash result as a continuous hex string
    std::cout << "C++ hash result: ";
    for (byte b : hash_result) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    std::cout << std::endl;

    // Return from the function after printing
    return 0;
  }

  if (vm.count("xelis-test")) {
    xelis_runTests();
    return 0;
  }

  if (vm.count("xelis-bench")) {
    boost::thread t(xelis_benchmark_cpu_hash);
    setPriority(t.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
    t.join();
    return 0;
  }

  if (vm.count("sabench")) {
    runDivsufsortBenchmark();
    return 0;
  }

  if (vm.count("daemon-address")) {
    host = vm["daemon-address"].as<std::string>();
    // TODO: Check if this contains a host:port... and then parse accordingly
  }
  if (vm.count("port")) {
    port = std::to_string(vm["port"].as<int>());
  }
  if (vm.count("wallet")) {
    wallet = vm["wallet"].as<std::string>();
  }
  if (vm.count("threads")) {
    threads = vm["threads"].as<int>();
  }
  if (vm.count("dev-fee")) {
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
  if (vm.count("no-lock")) {
    setcolor(CYAN);
    printf("CPU affinity has been disabled\n");
    setcolor(BRIGHT_WHITE);
    lockThreads = false;
  }
  if (vm.count("gpu")) {
    gpuMine = true;
  }
  // GPU-specific
  if (vm.count("batch-size")) {
    batchSize = vm["batch-size"].as<int>();
  }

  // Test-specific
  if (vm.count("op")) {
    testOp = vm["op"].as<int>();
  }
  if (vm.count("len")) {
    testLen = vm["len"].as<int>();
  }
  if (vm.count("lookup")) {
    printf("Use Lookup\n");
    useLookupMine = true;
  }
  
  // Ensure we capture *all* of the other options before we start using goto
  if (vm.count("dero-test")) {
    int rc = Testing();
    return rc;
  }
  if (vm.count("dero-benchmark")) {
    bench_duration = vm["dero-benchmark"].as<int>();
    if(bench_duration <= 0) {
      printf("ERROR: Invalid benchmark arguments. Use -h for assistance\n");
      return 1;
    }
    goto Benchmarking;
  }

fillBlanks:
{
  printf("%s\n", inputIntro);
  std::vector<std::string *> stringParams = {&host, &port, &wallet};
  std::vector<const char *> stringDefaults = {devPool, "10300", devWallet};
  std::vector<const char *> stringPrompts = {daemonPrompt, portPrompt, walletPrompt};
  int i = 0;
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
    }
    i++;
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
}
  printf("\n");
  goto Mining;

Benchmarking:
{
  if(threads <= 0) {
    threads = 1;
  }

  unsigned int n = std::thread::hardware_concurrency();
  int winMask = 0;
  for (int i = 0; i < n - 1; i++)
  {
    winMask += 1 << i;
  }

  host = devPool;
  port = devPort;
  wallet = devWallet;

  boost::thread GETWORK(getWork, false, 0);
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

    mutex.lock();
    std::cout << "(Benchmark) Worker " << i + 1 << " created" << std::endl;
    mutex.unlock();
  }

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  auto start_time = std::chrono::steady_clock::now();
  startBenchmark = true;
  boost::thread t2(logSeconds, start_time, bench_duration, &stopBenchmark);
  setPriority(t2.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);

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
  std::string intro = fmt::sprintf("Mined for %d seconds, average rate of ", seconds);
  std::cout << intro << std::flush;
  if (hashrate >= 1000000)
  {
    double rate = (double)(hashrate / 1000000.0);
    std::string hrate = fmt::sprintf("%.3f MH/s", rate);
    std::cout << hrate << std::endl;
  }
  else if (hashrate >= 1000)
  {
    double rate = (double)(hashrate / 1000.0);
    std::string hrate = fmt::sprintf("%.3f KH/s", rate);
    std::cout << hrate << std::endl;
  }
  else
  {
    std::string hrate = fmt::sprintf("%.3f H/s", (double)hashrate);
    std::cout << hrate << std::endl;
  }
  boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  return 0;
}

Mining:
{
  mutex.lock();
  printSupported();
  mutex.unlock();

  boost::thread GETWORK(getWork, false, 0);
  // setPriority(GETWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  boost::thread DEVWORK(getWork, true, 0);
  // setPriority(DEVWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  unsigned int n = std::thread::hardware_concurrency();
  int winMask = 0;
  for (int i = 0; i < n - 1; i++)
  {
    winMask += 1 << i;
  }

  winMask = std::max(1, winMask);

  // Create worker threads and set CPU affinity
  mutex.lock();
  if (false /*gpuMine*/ )
  {
    // boost::thread t(cudaMine);
    // setPriority(t.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
    // continue;
  }
  else
    for (int i = 0; i < threads; i++)
    {
      boost::thread t(mineBlock, i + 1);

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

      std::cout << "Thread " << i + 1 << " started" << std::endl;
    }
  mutex.unlock();

  auto start_time = std::chrono::steady_clock::now();
  // update(start_time);

  while (!isConnected)
  {
    boost::this_thread::yield();
  }

  boost::thread reporter(update, start_time);
  setPriority(reporter.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  while (true)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
  }

  return EXIT_SUCCESS;
  }
}

int Testing() {
  int failedTests = 0;
  Num diffTest("1234567890123456789", 10);

  if (testOp >= 0) {
    if (testLen >= 0) {
      failedTests += runOpTests(testOp, testLen);
    } else {
      failedTests += runOpTests(testOp);
    }
  }
  failedTests += TestAstroBWTv3();
  // TestAstroBWTv3_cuda();
  // TestAstroBWTv3repeattest();
  boost::this_thread::sleep_for(boost::chrono::seconds(1));
  return failedTests;
}

void logSeconds(std::chrono::_V2::steady_clock::time_point start_time, int duration, bool *stop)
{
  int i = 0;
  while (true)
  {
    auto now = std::chrono::steady_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    if (milliseconds >= 1000)
    {
      start_time = now;
      mutex.lock();
      // std::cout << "\n" << std::flush;
      printf("\rBENCHMARKING: %d/%d seconds elapsed...", i, duration);
      std::cout << std::flush;
      mutex.unlock();
      if (i == duration || *stop)
        break;
      i++;
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(250));
  }
}

void update(std::chrono::_V2::steady_clock::time_point start_time)
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

      float ratio = 1000.0f/milliseconds;
      if (rate30sec.size() <= 30 / reportInterval)
      {
        rate30sec.push_back((int64_t)(currentHashes*ratio));
      }
      else
      {
        rate30sec.erase(rate30sec.begin());
        rate30sec.push_back((int64_t)(currentHashes*ratio));
      }

      int64_t hashrate = 1.0 * std::accumulate(rate30sec.begin(), rate30sec.end(), 0LL) / (rate30sec.size() * reportInterval);

      if (hashrate >= 1000000)
      {
        double rate = (double)(hashrate / 1000000.0);
        std::string hrate = fmt::sprintf("HASHRATE %.3f MH/s", rate);
        mutex.lock();
        setcolor(BRIGHT_WHITE);
        std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine;
        setcolor(CYAN);
        std::cout << std::setw(2) << hrate << " | " << std::flush;
      }
      else if (hashrate >= 1000)
      {
        double rate = (double)(hashrate / 1000.0);
        std::string hrate = fmt::sprintf("HASHRATE %.3f KH/s", rate);
        mutex.lock();
        setcolor(BRIGHT_WHITE);
        std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine;
        setcolor(CYAN);
        std::cout << std::setw(2) << hrate << " | " << std::flush;
      }
      else
      {
        std::string hrate = fmt::sprintf("HASHRATE %.0f H/s", (double)hashrate, hrate);
        mutex.lock();
        setcolor(BRIGHT_WHITE);
        std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine;
        setcolor(CYAN);
        std::cout << std::setw(2) << hrate << " | " << std::flush;
      }

      std::string uptime = fmt::sprintf("%dd-%dh-%dm-%ds >> ", daysUp, hoursUp, minutesUp, secondsUp);
      std::cout << std::setw(2) << "ACCEPTED " << accepted << std::setw(2) << " | REJECTED " << rejected
                << std::setw(2) << " | DIFFICULTY " << (difficulty) << std::setw(2) << " | UPTIME " << uptime << std::flush;
      setcolor(BRIGHT_WHITE);
      mutex.unlock();
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

#else
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
  //pthread_t threadHandle = t;

  // Set the thread priority
  //int threadPriority = priority;
  // do nothing

#endif
}

void getWork(bool isDev, int fixedDiff)
{
  net::io_context ioc;
  ssl::context ctx = ssl::context{ssl::context::tlsv12_client};
  load_root_certificates(ctx);

  bool caughtDisconnect = false;

connectionAttempt:
  bool *B = isDev ? &devConnected : &isConnected;
  *B = false;
  mutex.lock();
  setcolor(BRIGHT_YELLOW);
  std::cout << "Connecting...\n";
  setcolor(BRIGHT_WHITE);
  mutex.unlock();
  try
  {
    // Launch the asynchronous operation
    bool err = false;
    if (isDev)
      boost::asio::spawn(ioc, std::bind(&do_session, std::string(devPool), std::string(devPort), std::string(devWallet), std::ref(ioc), std::ref(ctx), std::placeholders::_1, true, fixedDiff),
                         // on completion, spawn will call this function
                         [&](std::exception_ptr ex)
                         {
                           if (ex)
                           {
                             std::rethrow_exception(ex);
                             err = true;
                           }
                         });
    else
      boost::asio::spawn(ioc, std::bind(&do_session, host, port, wallet, std::ref(ioc), std::ref(ctx), std::placeholders::_1, false, fixedDiff),
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
        mutex.lock();
        setcolor(RED);
        std::cerr << "\nError establishing connections" << std::endl
                  << "Will try again in 10 seconds...\n\n";
        setcolor(BRIGHT_WHITE);
        mutex.unlock();
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
      mutex.lock();
      setcolor(RED);
      std::cerr << "\nError establishing connections" << std::endl
                << "Will try again in 10 seconds...\n\n";
      setcolor(BRIGHT_WHITE);
      mutex.unlock();
    }
    else
    {
      mutex.lock();
      setcolor(RED);
      std::cerr << "Dev connection error\n";
      setcolor(BRIGHT_WHITE);
      mutex.unlock();
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
    mutex.lock();
    setcolor(RED);
    if (!caughtDisconnect)
      std::cerr << "\nERROR: lost connection" << std::endl
                << "Will try to reconnect in 10 seconds...\n\n";
    else
      std::cerr << "\nError establishing connection" << std::endl
                << "Will try again in 10 seconds...\n\n";
    setcolor(BRIGHT_WHITE);
    mutex.unlock();
  }
  else
  {
    mutex.lock();
    setcolor(RED);
    if (!caughtDisconnect)
      std::cerr << "\nERROR: lost connection to dev node (mining will continue)" << std::endl
                << "Will try to reconnect in 10 seconds...\n\n";
    else
      std::cerr << "\nError establishing connection to dev node" << std::endl
                << "Will try again in 10 seconds...\n\n";
    setcolor(BRIGHT_WHITE);
    mutex.unlock();
  }
  caughtDisconnect = true;
  boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
  ioc.reset();
  goto connectionAttempt;
}

void benchmark(int tid)
{

  byte work[MINIBLOCK_SIZE];
  //byte work_fast[MINIBLOCK_SIZE];
  //byte work_fast_fast[MINIBLOCK_SIZE];

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
  //byte powHash2[32];
  workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*worker);
  lookupGen(*worker, lookup2D_global, lookup3D_global);

  workerData *worker2 = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*worker2);
  lookupGen(*worker2, lookup2D_global, lookup3D_global);
  // workerData *worker = new workerData();

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  work[MINIBLOCK_SIZE - 1] = (byte)tid;
  while(!startBenchmark) {

  }
  while (true)
  {
    boost::json::value myJob = job;
    boost::json::value myJobDev = devJob;
    localJobCounter = jobCounter;

    byte *b2 = new byte[MINIBLOCK_SIZE];
    hexstr_to_bytes(std::string(myJob.at("blockhashing_blob").as_string()), b2);
    memcpy(work, b2, MINIBLOCK_SIZE);
    delete[] b2;

    while (localJobCounter == jobCounter)
    {
      i++;
      //double which = (double)(rand() % 10000);
      //bool devMine = (devConnected && which < devFee * 100.0);
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

void mineBlock(int tid)
{
  byte work[MINIBLOCK_SIZE];

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
  //byte powHash2[32];
  byte devWork[MINIBLOCK_SIZE];

  workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*worker);
  lookupGen(*worker, lookup2D_global, lookup3D_global);

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
      mutex.lock();
      boost::json::value myJob = job;
      boost::json::value myJobDev = devJob;
      localJobCounter = jobCounter;
      mutex.unlock();

      byte *b2 = new byte[MINIBLOCK_SIZE];
      hexstr_to_bytes(std::string(myJob.at("blockhashing_blob").as_string()), b2);
      memcpy(work, b2, MINIBLOCK_SIZE);
      delete[] b2;
      //hexstr_to_bytes_direct(std::string(myJob.at("blockhashing_blob").as_string()), work);

      if (devConnected)
      {
        byte *b2d = new byte[MINIBLOCK_SIZE];
        hexstr_to_bytes(std::string(myJobDev.at("blockhashing_blob").as_string()), b2d);
        memcpy(devWork, b2d, MINIBLOCK_SIZE);
        delete[] b2d;
        //hexstr_to_bytes_direct(std::string(myJobDev.at("blockhashing_blob").as_string()), devWork);
      }

      memcpy(&work[MINIBLOCK_SIZE - 12], random_buf, 12);
      memcpy(&devWork[MINIBLOCK_SIZE - 12], random_buf, 12);

      work[MINIBLOCK_SIZE - 1] = (byte)tid;
      devWork[MINIBLOCK_SIZE - 1] = (byte)tid;

      if ((work[0] & 0xf) != 1)
      { // check  version
        mutex.lock();
        std::cerr << "Unknown version, please check for updates: "
                  << "version" << (work[0] & 0x1f) << std::endl;
        mutex.unlock();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        continue;
      }
      double which;
      bool devMine = false;
      bool submit = false;
      uint64_t DIFF;
      Num cmpDiff;
      // DIFF = 5000;

      std::string hex;
      int32_t i = 0;
      while (localJobCounter == jobCounter)
      {
        which = (double)(rand() % 10000);
        devMine = (devConnected && which < devFee * 100.0);
        DIFF = devMine ? difficultyDev : difficulty;

        // printf("Difficulty: %" PRIx64 "\n", DIFF);

        cmpDiff = ConvertDifficultyToBig(DIFF);
        i++;
        byte *WORK = devMine ? &devWork[0] : &work[0];
        memcpy(&WORK[MINIBLOCK_SIZE - 5], &i, sizeof(i));

        // swap endianness
        if (littleEndian())
        {
          std::swap(WORK[MINIBLOCK_SIZE - 5], WORK[MINIBLOCK_SIZE - 2]);
          std::swap(WORK[MINIBLOCK_SIZE - 4], WORK[MINIBLOCK_SIZE - 3]);
        }
        AstroBWTv3(&WORK[0], MINIBLOCK_SIZE, powHash, *worker, useLookupMine);
        // AstroBWTv3((byte*)("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\0"), MINIBLOCK_SIZE, powHash, *worker, useLookupMine);
        
        counter.fetch_add(1);
        submit = devMine ? !submittingDev : !submitting;

        if (submit && CheckHash(&powHash[0], cmpDiff))
        {
          // printf("work: %s, hash: %s\n", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str(), hexStr(powHash, 32).c_str());
          if (devMine)
          {
            mutex.lock();
            submittingDev = true;
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n";
            setcolor(BRIGHT_WHITE);
            mutex.unlock();
            devShare = {
                {"jobid", myJobDev.at("jobid")},
                {"mbl_blob", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str()}};
          }
          else
          {
            mutex.lock();
            submitting = true;
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n";
            setcolor(BRIGHT_WHITE);
            mutex.unlock();
            share = {
                {"jobid", myJob.at("jobid")},
                {"mbl_blob", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str()}};
          }
        }

        if (!isConnected)
          break;
      }
      if (!isConnected)
        break;
    }
    catch (...)
    {
      std::cerr << "Error in POW Function" << std::endl;
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}
