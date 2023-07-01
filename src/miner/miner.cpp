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

#include "rootcert.h"

#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/spawn.hpp>

#include <boost/thread.hpp>
#include <boost/atomic.hpp>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <miner.h>
#include <nlohmann/json.hpp>

#include <bigint.h>
#include <random>

#include <hex.h>
#include <pow.h>
#include <powtest.h>
#include <thread>

#include <chrono>
#include <fmt/format.h>
#include <fmt/printf.h>

#if defined(_WIN32)
#include <Windows.h>
#elif defined(linux)
#include <sched.h>
#endif

#if defined(_WIN32)
LPTSTR lpNxtPage;  // Address of the next page to ask for
DWORD dwPages = 0; // Count of pages gotten so far
DWORD dwPageSize;  // Page size on this computer
#endif

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

using json = nlohmann::json;

boost::mutex mutex;
boost::mutex wsMutex;

json job;
json devJob;
json share;
json devShare;

bool submitting = false;
bool submittingDev = false;

int jobCounter;
boost::atomic<int64_t> counter = 0;

int blockCounter;
int miniBlockCounter;
int rejected;
int accepted;
int firstRejected;

uint64_t hashrate;
int64_t ourHeight;
int64_t devHeight;
int64_t difficulty;
int64_t difficultyDev;

std::vector<int64_t> rate5min;
std::vector<int64_t> rate1min;

bool isConnected = false;
bool devConnected = false;

using byte = unsigned char;
bool stopBenchmark = false;
//------------------------------------------------------------------------------

// Report a failure
void fail(beast::error_code ec, char const *what)
{
  std::cerr << what << ": " << ec.message() << "\n";
}

// Sends a WebSocket message and prints the response
void do_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  beast::error_code ec;

  // These objects perform our I/O
  tcp::resolver resolver(ioc);
  websocket::stream<
      beast::ssl_stream<beast::tcp_stream>>
      ws(ioc, ctx);

  // Look up the domain name
  auto const results = resolver.async_resolve(host, port, yield[ec]);
  if (ec)
    return fail(ec, "resolve");

  // Set a timeout on the operation
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  auto ep = beast::get_lowest_layer(ws).async_connect(results, yield[ec]);
  if (ec)
    return fail(ec, "connect");

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
  host += ':' + std::to_string(ep.port());

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
  ss << "/ws/" << wallet;
  ws.async_handshake(host, ss.str().c_str(), yield[ec]);
  if (ec)
    return fail(ec, "handshake");

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  while (true)
  {
    buffer.clear();
    workInfo.str("");
    workInfo.clear();
    ws.read(buffer);

    // hand getwork feed

    workInfo << beast::make_printable(buffer.data());
    json workData = json::parse(workInfo.str().c_str());
    if ((isDev ? (workData.at("height") != devHeight) : (workData.at("height") != ourHeight)))
    {
      json *J = isDev ? &devJob : &job;
      *J = workData;
      if (!isDev)
        jobCounter++;

      if ((*J).at("lasterror") != "")
      {
        std::cerr << "received error: " << (*J).at("lasterror") << std::endl
                  << consoleLine;
      }

      if (!isDev)
      {
        mutex.lock();
        blockCounter = (*J).at("blocks");
        miniBlockCounter = (*J).at("miniblocks");
        rejected = (*J).at("rejected");
        hashrate = (*J).at("difficultyuint64");
        ourHeight = (*J).at("height");
        difficulty = (*J).at("difficultyuint64");
        printf("NEW JOB RECEIVED | Height: %d | Difficulty %" PRIu64 "\n", ourHeight, difficulty);
        rejected = (*J).at("rejected");
        if (!isConnected)
          firstRejected = rejected;
        isConnected = isConnected || true;
        mutex.unlock();
      }
      else
      {
        mutex.lock();
        difficultyDev = (*J).at("difficultyuint64");
        devHeight = (*J).at("height");
        if (!devConnected)
          std::cout << "Connected to dev pool\n";
        devConnected = devConnected || true;
        mutex.unlock();
      }
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

void rpc_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    net::io_context &ioc,
    net::yield_context yield,
    bool isDev)
{
  beast::error_code ec;
  // These objects perform our I/O
  tcp::resolver resolver(ioc);
  beast::tcp_stream stream(ioc);

  // Look up the domain name
  auto const results = resolver.resolve(host, port);

  // Make the connection on the IP address we get from a lookup
  stream.connect(results);

  std::cout << "RPC connect" << std::endl;

  while (true)
  {
    if (isDev ? submittingDev : submitting)
    {
      std::cout << "submitting" << std::endl;
      // Set up an HTTP GET request message
      std::stringstream ss;
      
      json *SHARE = isDev ? &devShare : &share;
      json params = {{"jobid", (*SHARE).at("jobid")},
               {"mbl_blob", (*SHARE).at("mbl_blob")}};

      json BODY = {
          {"jsonrpc", "2.0"},
          {"id", "1"},
          {"method", "DERO.SubmitBlock"},
          {"params:", params},
        };

      ss << std::quoted(BODY.dump(-1, ' ', false));
      std::string d = ss.str().c_str();

      http::request<http::string_body> req{http::verb::post, "/json_rpc", 11};
      req.set(http::field::host, host);
      req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
      req.body() = d.c_str();
      req.prepare_payload();
      req.set(http::field::content_type, "application/json");

      std::cout << req.body() << std::endl;

      // Send the HTTP request to the remote host
      http::write(stream, req);

      bool *flag = isDev ? &submittingDev : &submitting;
      *flag = false;

      // This buffer is used for reading and must be persisted
      beast::flat_buffer buffer;

      // Declare a container to hold the response
      http::response<http::dynamic_body> res;
      // Receive the HTTP response
      http::read(stream, buffer, res);

      // Write the message to standard out
      mutex.lock();
      std::cout << res << " " << beast::make_printable(buffer.data()) << std::endl;
      mutex.unlock();
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(25));
  }
}
//------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  // Check command line arguments.
  mpz_pow_ui(oneLsh256.get_mpz_t(), mpz_class(2).get_mpz_t(), 255);
  if (argc < 5)
  {
    std::string command(argv[1]);
    if (command == "test")
    {
      TestAstroBWTv3();
      boost::this_thread::sleep_for(boost::chrono::seconds(30));
      return 0;
    }
    if (command == "benchmark")
    {
      threads = 1;
      threads = std::stoi(argv[2]);
      int duration = std::stoi(argv[3]);

      unsigned int n = std::thread::hardware_concurrency();
      int winMask = 0;
      for (int i = 0; i < n - 1; i++)
      {
        winMask += 1 << i;
      }

      winMask = std::max(1, winMask);

      // Create worker threads and set CPU affinity
      for (int i = 0; i < threads; i++)
      {
        boost::thread t(benchmark, i + 1);
        setAffinity(t.native_handle(), 1 << (i % n));
        if (threads == 1 || (n > 2 && i < n - 2))
          setPriority(t.native_handle(), THREAD_PRIORITY_HIGHEST);

        mutex.lock();
        std::cout << "(Benchmark) Worker " << i + 1 << " created" << std::endl;
        mutex.unlock();
      }

      boost::thread t2(logSeconds, duration);
      setPriority(t2.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

      boost::this_thread::sleep_for(boost::chrono::seconds(duration));
      stopBenchmark = true;

      int64_t hashrate = counter / duration;
      std::string intro = fmt::sprintf("Mined for %d seconds, average rate of ", duration);
      std::cout << intro << std::flush;
      if (hashrate >= 1000000)
      {
        double rate = (double)(hashrate / 1000000.0);
        std::string hrate = fmt::sprintf("%.2f MH/s", rate);
        std::cout << hrate << std::endl;
      }
      else if (hashrate >= 1000)
      {
        double rate = (double)(hashrate / 1000.0);
        std::string hrate = fmt::sprintf("%.2f KH/s", rate);
        std::cout << hrate << std::endl;
      }
      else
      {
        std::string hrate = fmt::sprintf("%.2f H/s", (double)hashrate);
        std::cout << hrate << std::endl;
      }
      boost::this_thread::sleep_for(boost::chrono::seconds(30));
      return 0;
    }
  }
  if (argc < 5)
  {
    std::cerr << "Usage: websocket-client-coro-ssl <host> <port> <text> <threads>\n"
              << "Example:\n"
              << "    Tnn-miner 0.0.0.0 10100 walletAddress 1\n";
    return EXIT_FAILURE;
  }
  host = argv[1];
  port = argv[2];
  wallet = argv[3];

  // #if defined(_WIN32)
  // if(!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
  // {
  //   printf("Failed to end background mode (%d)\n", GetLastError());
  // }
  // #endif

  boost::thread GETWORK(getWork, false);
  setPriority(GETWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  boost::thread SENDWORK(sendWork);
  setPriority(SENDWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  boost::thread DEVWORK(devWork);
  setPriority(DEVWORK.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  threads = 1;
  if (argc >= 5 && argv[4] != NULL)
  {
    threads = std::stoi(argv[4]);
  }

  unsigned int n = std::thread::hardware_concurrency();
  int winMask = 0;
  for (int i = 0; i < n - 1; i++)
  {
    winMask += 1 << i;
  }

  winMask = std::max(1, winMask);

  // Create worker threads and set CPU affinity
  for (int i = 0; i < threads; i++)
  {
    boost::thread t(mineBlock, i + 1);
    setAffinity(t.native_handle(), 1 << (i % n));
    if (threads == 1 || (n > 2 && i < n - 2))
      setPriority(t.native_handle(), THREAD_PRIORITY_HIGHEST);

    mutex.lock();
    std::cout << "Worker " << i + 1 << " created" << std::endl;
    mutex.unlock();
  }

  boost::thread reporter(update, start_time);
  setPriority(reporter.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

  std::string input;
  while (getline(std::cin, input) && input != "quit")
  {
    if (input == "hello")
      std::cout << "Hello world!\n";
    else
      std::cout << "Unrecognized command: " << input << "\n";
    std::cout << consoleLine;
  }

  return EXIT_SUCCESS;
}

void logSeconds(int duration)
{
  int i = 0;
  while (true)
  {
    if (i == duration)
      break;
    mutex.lock();
    std::cout << "\r" << std::flush;
    printf("BENCHMARKING: %d/%d seconds elapsed...", i, duration);
    mutex.unlock();
    boost::this_thread::sleep_for(boost::chrono::seconds(1));
    i++;
  }
}

void update(std::chrono::_V2::system_clock::time_point start_time)
{
  boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
  while (true)
  {
    mutex.lock();
    int64_t currentHashes = counter.load();
    counter = 0;
    mutex.unlock();

    if (rate1min.size() <= 60 / reportInterval)
    {
      rate1min.push_back(currentHashes);
    }
    else
    {
      rate1min.erase(rate1min.begin());
      rate1min.push_back(currentHashes);
    }

    int64_t hashrate = 1.0 * std::accumulate(rate1min.begin(), rate1min.end(), 0LL) / (rate1min.size() * 5);

    mutex.lock();
    if (hashrate >= 1000000)
    {
      double rate = (double)(hashrate / 1000000.0);
      std::string hrate = fmt::sprintf("HASHRATE (1 min) | %.2f MH/s", rate);
      std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << std::setw(2) << hrate << " | " << std::flush;
    }
    else if (hashrate >= 1000)
    {
      double rate = (double)(hashrate / 1000.0);
      std::string hrate = fmt::sprintf("HASHRATE (1 min) | %.2f KH/s", rate);
      std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << std::setw(2) << hrate << " | " << std::flush;
    }
    else
    {
      std::string hrate = fmt::sprintf("HASHRATE (1 min) | %.2f H/s", (double)hashrate, hrate);
      std::cout << "\r" << std::setw(2) << std::setfill('0') << std::setw(2) << consoleLine << std::setw(2) << hrate << " | " << std::flush;
    }

    std::cout << std::setw(2) << "ACCEPTED " << accepted << std::setw(2) << " | REJECTED " << (rejected - firstRejected) << " >> " << std::flush;
    mutex.unlock();
    boost::this_thread::sleep_for(boost::chrono::seconds(reportInterval));
  }
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

#elif defined(linux)
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

#elif defined(linux)
  // Get the native handle of the thread
  pthread_t threadHandle = t;

  // Set the thread priority
  int threadPriority = priority;
  BOOL success = SetThreadPriority(threadHandle, threadPriority);
  if (!success)
  {
    DWORD error = GetLastError();
    std::cerr << "Failed to set thread priority. Error code: " << error << std::endl;
  }

#endif
}

void sendWork()
{
  net::io_context ioc;
  ssl::context ctx = ssl::context{ssl::context::tlsv12_client};
  load_root_certificates(ctx);

connectionAttempt:
  // Launch the asynchronous operation
  bool err = false;
  boost::asio::spawn(ioc, std::bind(&rpc_session, std::string(host), std::string("10102"), std::string(wallet), std::ref(ioc), std::placeholders::_1, false),
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
    if (err)
    {
      std::cerr << "RPC connection error" << std::endl
                << "Will try again in 10 seconds";
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
    ioc.reset();
    goto connectionAttempt;
  }
}

void devWork()
{
  net::io_context ioc;
  ssl::context ctx = ssl::context{ssl::context::tlsv12_client};
  load_root_certificates(ctx);

connectionAttempt:
  // Launch the asynchronous operation
  bool err = false;
  boost::asio::spawn(ioc, std::bind(&rpc_session, std::string(devPool), std::string("10102"), std::string(devWallet), std::ref(ioc), std::placeholders::_1, true),
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
    if (err)
    {
      std::cerr << "(DEV) RPC connection error" << std::endl
                << "Will try again in 10 seconds";
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
    ioc.reset();
    goto connectionAttempt;
  }
}

void getWork(bool isDev)
{
  net::io_context ioc;
  ssl::context ctx = ssl::context{ssl::context::tlsv12_client};
  load_root_certificates(ctx);

connectionAttempt:
  // Launch the asynchronous operation
  bool err = false;
  boost::asio::spawn(ioc, std::bind(&do_session, std::string(devPool), std::string(devPort), std::string(devWallet), std::ref(ioc), std::ref(ctx), std::placeholders::_1, true),
                     // on completion, spawn will call this function
                     [&](std::exception_ptr ex)
                     {
                       if (ex)
                       {
                         std::rethrow_exception(ex);
                         err = true;
                       }
                     });
  boost::asio::spawn(ioc, std::bind(&do_session, std::string(host), std::string(port), std::string(wallet), std::ref(ioc), std::ref(ctx), std::placeholders::_1, false),
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
    if (err)
    {
      std::cerr << "Error establishing connections" << std::endl
                << "Will try again in 10 seconds";
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
    ioc.reset();
    goto connectionAttempt;
  }
}

void benchmark(int tid)
{
  bigint diff;
  byte work[MINIBLOCK_SIZE];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::array<uint8_t, 48> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                { return dist(gen); });
  std::memcpy(work, buf.data(), buf.size());

  boost::this_thread::sleep_for(boost::chrono::milliseconds(50));

  int32_t i = 0;

  byte powHash[32];
  workerData *worker = new workerData();

#if defined(_WIN32)
  LPVOID lpvBase;       // Base address of the test memory
  LPTSTR lpPtr;         // Generic character pointer
  BOOL bSuccess;        // Flag
  DWORD iCount;         // Generic counter
  SYSTEM_INFO sSysInfo; // Useful information about the system
  GetSystemInfo(&sSysInfo);
  dwPageSize = sSysInfo.dwPageSize;

  void *DATA = VirtualAlloc(NULL, 2048 * dwPageSize, MEM_LARGE_PAGES | MEM_COMMIT, PAGE_READWRITE);
#endif

  work[MINIBLOCK_SIZE - 1] = (byte)tid;
  while (true)
  {
    i++;
    std::memcpy(&work[MINIBLOCK_SIZE - 5], &i, sizeof(i));
    // swap endianness
    if (littleEndian)
    {
      std::swap(work[MINIBLOCK_SIZE - 5], work[MINIBLOCK_SIZE - 2]);
      std::swap(work[MINIBLOCK_SIZE - 4], work[MINIBLOCK_SIZE - 3]);
    }
    AstroBWTv3(work, MINIBLOCK_SIZE, powHash, *worker);
    counter.store(counter + 1);
    if (stopBenchmark)
      break;
  }
}

void mineBlock(int tid)
{
  bigint diff;
  byte work[MINIBLOCK_SIZE];

  byte random_buf[12];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::array<uint8_t, 12> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                { return dist(gen); });
  std::memcpy(random_buf, buf.data(), buf.size());

  boost::this_thread::sleep_for(boost::chrono::milliseconds(50));

  int64_t localJobCounter;
  int32_t i = 0;

  byte powHash[32];
  workerData *worker = new workerData();
  byte devWork[MINIBLOCK_SIZE];

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
  }

#if defined(_WIN32)
  LPVOID lpvBase;       // Base address of the test memory
  LPTSTR lpPtr;         // Generic character pointer
  BOOL bSuccess;        // Flag
  DWORD iCount;         // Generic counter
  SYSTEM_INFO sSysInfo; // Useful information about the system
  GetSystemInfo(&sSysInfo);

  printf("This computer has page size %d.\n", sSysInfo.dwPageSize);
  dwPageSize = sSysInfo.dwPageSize;

  void *DATA = VirtualAlloc(NULL, 2048 * dwPageSize, MEM_LARGE_PAGES | MEM_COMMIT, PAGE_READWRITE);
#endif

  while (true)
  {
    json myJob = job;
    json myJobDev = devJob;
    localJobCounter = jobCounter;

    byte *b2 = new byte[MINIBLOCK_SIZE];
    hexstr_to_bytes(myJob.at("blockhashing_blob"), b2);
    std::copy(b2, b2 + MINIBLOCK_SIZE, work);

    if (devConnected)
    {
      byte *b2d = new byte[MINIBLOCK_SIZE];
      hexstr_to_bytes(myJobDev.at("blockhashing_blob"), b2d);
      std::copy(b2d, b2d + MINIBLOCK_SIZE, devWork);
    }

    std::copy(random_buf, random_buf + 12, work + MINIBLOCK_SIZE - 12);
    std::copy(random_buf, random_buf + 12, devWork + MINIBLOCK_SIZE - 12);

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

    auto submit_share = [&]()
    {
      try
      {
        if (devMine)
        {
          mutex.lock();
          devShare = {
              {"jobid", myJobDev.at("jobid")},
              {"mbl_blob", hexStr(devWork, MINIBLOCK_SIZE).c_str()}};
          mutex.unlock();
        }
        else
        {
          mutex.lock();
          share = {
              {"jobid", myJob.at("jobid")},
              {"mbl_blob", hexStr(work, MINIBLOCK_SIZE).c_str()}};
          mutex.unlock();
        }
        // if (auto err = c.stratum.SubmitShare(share); err != nullptr)
        // {
        //   c.logger.Error(err, "Failed to submit share");
        // }
      }
      catch (...)
      {
        std::cout << "failed to submit share" << std::endl;
      }
    };

    uint64_t DIFF = devMine ? difficultyDev : difficulty;
    // DIFF = 5000;

    while (localJobCounter == jobCounter)
    {
      which = (double)(rand() % 1000);
      devMine = (devConnected && which < devFee * 10.0);
      i++;
      byte *WORK = devMine ? &devWork[0] : &work[0];
      std::memcpy(&WORK[MINIBLOCK_SIZE - 5], &i, sizeof(i));
      // swap endianness
      if (littleEndian)
      {
        std::swap(WORK[MINIBLOCK_SIZE - 5], WORK[MINIBLOCK_SIZE - 2]);
        std::swap(WORK[MINIBLOCK_SIZE - 4], WORK[MINIBLOCK_SIZE - 3]);
      }
      AstroBWTv3(WORK, MINIBLOCK_SIZE, powHash, *worker);
      counter.store(counter + 1);
      submit = devMine ? !submittingDev : !submitting;
      if (submit && CheckHash(powHash, DIFF))
      { // note we are doing a local, NW might have moved meanwhile
        if (devMine)
        {
          mutex.lock();
          submittingDev = true;
          std::cout << "\nFound dev share... ";
          mutex.unlock();
        }
        else
        {
          mutex.lock();
          submitting = true;
          std::cout << "\nThread " << tid << " found a nonce... ";
          mutex.unlock();
        }
        submit_share();
      }
    }
  }
}