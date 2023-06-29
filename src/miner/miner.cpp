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

  if (isDev ? submittingDev : submitting)
  {
    std::stringstream shareStream;
    shareStream << isDev ? devShare : share;
    ws.async_write(net::buffer(shareStream.str()), yield[ec]);
    if (ec)
      return fail(ec, "failed to submit share");
  }

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  ws.async_read(buffer, yield[ec]);
  if (ec)
    return fail(ec, "read");

  // hand getwork feed
  if (isDev ? !submittingDev : !submitting)
  {
    workInfo << beast::make_printable(buffer.data());

    json workData = json::parse(workInfo.str().c_str());
    if (isDev ? (workData.at("height") != devHeight) : (workData.at("height") != ourHeight))
    {
      mutex.lock();
      json *J = isDev ? &devJob : &job;
      *J = workData;
      if (!isDev)
        jobCounter++;
      mutex.unlock();

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
  }
  // submit shares
  else if (isDev ? submittingDev : submitting)
  {
    std::stringstream result;
    result << beast::make_printable(buffer.data());
    std::string error = json::parse(result.str().c_str()).at("lasterror");
    if (error != "")
    {
      if (isDev)
        std::cerr << "(Dev Share)";
      std::cerr << "REJECTED: " << job.at("lasterror");
    }
    else
    {
      if (isDev)
        std::cerr << "(Dev Share) ";
      std::cout << "ACCEPTED!\n";
    }
    if (isDev)
      submittingDev = false;
    else
      submitting = false;
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
  auto start_time = std::chrono::high_resolution_clock::now();

  // Check command line arguments.
  mpz_pow_ui(oneLsh256.get_mpz_t(), mpz_class(2).get_mpz_t(), 255);
  if (argc == 2)
  {
    std::string command(argv[1]);
    if (command == "test")
    {
      TestAstroBWTv3();
      boost::this_thread::sleep_for(boost::chrono::seconds(30));
    }
    return 0;
  }
  else if (argc < 4)
  {
    std::cerr << "Usage: websocket-client-coro-ssl <host> <port> <text> <threads(optional)\n"
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

  boost::thread polling(getWork, false);
  setPriority(polling.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

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
    if (threads == 1 || (n > 2 && i < threads - 2))
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

void update(std::chrono::_V2::system_clock::time_point start_time)
{
  boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
  while (true)
  {
    mutex.lock();
    auto current_time = std::chrono::high_resolution_clock::now();
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
      std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << std::setw(2) << hrate << " >> " << std::flush;
    }
    else if (hashrate >= 1000)
    {
      double rate = (double)(hashrate / 1000.0);
      std::string hrate = fmt::sprintf("HASHRATE (1 min) | %.2f KH/s", rate);
      std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << std::setw(2) << hrate << " >> " << std::flush;
    }
    else
    {
      std::string hrate = fmt::sprintf("HASHRATE (1 min) | %.2f H/s", (double)hashrate, hrate);
      std::cout << "\r" << std::setw(2) << std::setfill('0') << std::setw(2) << consoleLine << std::setw(2) << hrate << " >> " << std::flush;
    }
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

void getWork(bool isDev)
{
  net::io_context ioc;
  ssl::context ctx = ssl::context{ssl::context::tlsv12_client};
  load_root_certificates(ctx);

  while (true)
  {
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
    bool err2 = false;
    boost::asio::spawn(ioc, std::bind(&do_session, std::string(host), std::string(port), std::string(wallet), std::ref(ioc), std::ref(ctx), std::placeholders::_1, false),
                       // on completion, spawn will call this function
                       [&](std::exception_ptr ex)
                       {
                         if (ex)
                         {
                           std::rethrow_exception(ex);
                           err2 = true;
                         }
                       });
    ioc.run();
    if (err || err2)
    {
      if (err)
      {
        std::cerr << "Error connecting to dev server: " << devPool << ":" << devPort << std::endl
                  << "Will try again in 10 seconds";
      }

      if (err2)
      {
        std::cerr << "Error connecting to server: " << host << ":" << port << std::endl
                  << "Will try again in 10 seconds";
      }

      boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
      continue;
    }
    ioc.reset();
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
    while (localJobCounter == jobCounter)
    {
      double which = (double)(rand() % 1000);
      bool devMine = (devConnected && which < devFee * 10.0);
      i++;

      if (devMine)
      {
        std::memcpy(&devWork[MINIBLOCK_SIZE - 5], &i, sizeof(i));
        // swap endianness
        if (littleEndian)
        {
          std::swap(devWork[MINIBLOCK_SIZE - 5], devWork[MINIBLOCK_SIZE - 2]);
          std::swap(devWork[MINIBLOCK_SIZE - 4], devWork[MINIBLOCK_SIZE - 3]);
        }
        AstroBWTv3(devWork, MINIBLOCK_SIZE, powHash, *worker);
        counter.store(counter + 1);
      }
      else
      {
        std::memcpy(&work[MINIBLOCK_SIZE - 5], &i, sizeof(i));
        // swap endianness
        if (littleEndian)
        {
          std::swap(work[MINIBLOCK_SIZE - 5], work[MINIBLOCK_SIZE - 2]);
          std::swap(work[MINIBLOCK_SIZE - 4], work[MINIBLOCK_SIZE - 3]);
        }
        AstroBWTv3(work, MINIBLOCK_SIZE, powHash, *worker);
        counter.store(counter + 1);
      }

      bool submit = devMine ? !submittingDev : !submitting;
      if (submit && CheckHash(powHash, (devMine ? difficultyDev : difficulty)))
      { // note we are doing a local, NW might have moved meanwhile
        mutex.lock();
        if (devMine)
        {
          std::cout << "Found dev share... ";
        }
        else
        {
          std::cout << "Thread " << tid << " found a nonce... ";
        }
        auto submit_share = [&]()
        {
          try
          {
            if (devMine)
            {
              devShare = {
                  {"jobid", myJobDev.at("jobid")},
                  {"mbl_blob", hexStr(devWork, MINIBLOCK_SIZE).c_str()}};
              submittingDev = true;
            }
            else
            {
              share = {
                  {"jobid", myJob.at("jobid")},
                  {"mbl_blob", hexStr(work, MINIBLOCK_SIZE).c_str()}};
              submitting = true;
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
        submit_share();
        mutex.unlock();
      }
    }
  }
}