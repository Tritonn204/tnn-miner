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
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <boost/json.hpp>

#include <boost/thread.hpp>
#include <boost/atomic.hpp>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <miner.h>
#include <nlohmann/json.hpp>

#include <random>

#include <hex.h>
#include <astrobwtv3.h>
#include <xelis-hash.hpp>
#include <spectrex.h>

#include <astrotest.hpp>
#include <thread>

#include <chrono>
#include <fmt/format.h>
#include <fmt/printf.h>

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

using json = nlohmann::json;

boost::mutex mutex;
boost::mutex wsMutex;
boost::mutex reportMutex;

json job = json({});
json devJob = json({});

boost::json::object share = {};
boost::json::object devShare = {};

std::string currentBlob;
std::string devBlob;

bool submitting = false;
bool submittingDev = false;

uint16_t *lookup2D_global; // Storage for computed values of 2-byte chunks
byte *lookup3D_global;     // Storage for deterministically computed values of 1-byte chunks

int jobCounter;
std::atomic<int64_t> counter = 0;
std::atomic<int64_t> benchCounter = 0;

int blockCounter;
int miniBlockCounter;
int rejected;
int accepted;
int firstRejected;

uint64_t hashrate;
uint64_t ourHeight = 0;
uint64_t devHeight = 0;

uint64_t difficulty;
uint64_t difficultyDev;

double doubleDiff;
double doubleDiffDev;

std::vector<int64_t> rate5min;
std::vector<int64_t> rate1min;
std::vector<int64_t> rate30sec;

bool isConnected = false;
bool devConnected = false;

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
// Report a failure
void fail(beast::error_code ec, char const *what) noexcept
{
  wsMutex.lock();
  setcolor(RED);
  std::cerr << '\n'
            << what << ": " << ec.message() << "\n";
  setcolor(BRIGHT_WHITE);
  wsMutex.unlock();
}

tcp::endpoint resolve_host(net::io_context &ioc, net::yield_context yield, std::string host, std::string port) {
  beast::error_code ec;

  int addrCount = 0;
  net::ip::address ip_address;

  // If the specified host/pool is not in IP address form, resolve to acquire the IP address
#if !defined(_WIN32) && !defined(__APPLE__)
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
          wsMutex.lock();
          for (auto &it : addrs) {
            addrCount++;
            ip = it.to_string();
          }
          p.set_value();
          wsMutex.unlock();
      } else {
        p.set_value();
      }
    });
    ioc2.run();

    std::future<void> f = p.get_future();
    f.get();

    if (addrCount == 0)
    {
      wsMutex.lock();
      setcolor(RED);
      std::cerr << "ERROR: Could not resolve " << host << std::endl;
      setcolor(BRIGHT_WHITE);
      wsMutex.unlock();
      //return stream;
      // FIXME: what do?
    }

    ip_address = net::ip::address::from_string(ip.c_str(), ec);
  }
  else
  {
    ip_address = net::ip::address::from_string(host, ec);
  }

  tcp::endpoint result(ip_address, (uint_least16_t)std::stoi(port.c_str()));
  return result;
#else
  // Look up the domain name
  tcp::resolver resolver(ioc);
  auto const results = resolver.async_resolve(host, port, yield[ec]);
  if (ec)
    fail(ec, "resolve");

  return results.begin()->endpoint();
#endif
}


void dero_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  beast::error_code ec;

  websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(ioc, ctx);
  auto endpoint = resolve_host(ioc, yield, host, port);

  // Set a timeout on the operation
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(ws).connect(endpoint);

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
  host += ':' + port;

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
  // Perform the SSL/TLS handshake
  ws.next_layer().async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec)
    return fail(ec, "tls_handshake");

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
  {
    ws.async_close(websocket::close_code::normal, yield[ec]);
    return fail(ec, "handshake");
  }
  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

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
        // wsMutex.lock();
        // std::cout << msg;
        // wsMutex.unlock();
        ws.async_write(boost::asio::buffer(msg), yield[ec]);
        if (ec)
        {
          return fail(ec, "Submission Error");
        }
        *B = false;
      }

      beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(60));
      ws.async_read(buffer, yield[ec]);
      if (!ec)
      {
        // handle getwork feed

        beast::get_lowest_layer(ws).expires_never();
        workInfo << beast::make_printable(buffer.data());
        if (json::accept(workInfo.str()))
        {
          json workData = json::parse(workInfo.str());
          if ((isDev ? (workData.at("height") != devHeight) : (workData.at("height") != ourHeight)))
          {
            // wsMutex.lock();
            if (isDev)
              devJob = workData;
            else
              job = workData;
            json *J = isDev ? &devJob : &job;
            // wsMutex.unlock();

            if ((*J).at("lasterror") != "")
            {
              std::cerr << "received error: " << (*J).at("lasterror") << std::endl
                        << consoleLine << versionString << " ";
            }

            if (!isDev)
            {
              currentBlob = std::string((*J).at("blockhashing_blob"));
              blockCounter = (*J).at("blocks");
              miniBlockCounter = (*J).at("miniblocks");
              rejected = (*J).at("rejected");
              hashrate = (*J).at("difficultyuint64");
              ourHeight = (*J).at("height");
              difficulty = (*J).at("difficultyuint64");
              // printf("NEW JOB RECEIVED | Height: %d | Difficulty %" PRIu64 "\n", ourHeight, difficulty);
              accepted = (*J).at("miniblocks");
              rejected = (*J).at("rejected");
              if (!isConnected)
              {
                wsMutex.lock();
                setcolor(BRIGHT_YELLOW);
                printf("Mining at: %s/ws/%s\n", host.c_str(), wallet.c_str());
                setcolor(CYAN);
                printf("Dev fee: %.2f", devFee);
                std::cout << "%" << std::endl;
                setcolor(BRIGHT_WHITE);
                wsMutex.unlock();
              }
              isConnected = isConnected || true;
              jobCounter++;
            }
            else
            {
              difficultyDev = (*J).at("difficultyuint64");
              devBlob = std::string((*J).at("blockhashing_blob"));
              devHeight = (*J).at("height");
              if (!devConnected)
              {
                wsMutex.lock();
                setcolor(CYAN);
                printf("Connected to dev node: %s\n", devPool);
                setcolor(BRIGHT_WHITE);
                wsMutex.unlock();
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
        return fail(ec, "async_read");
      }
    }
    catch (...)
    {
      setcolor(RED);
      std::cout << "ws error\n";
      setcolor(BRIGHT_WHITE);
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // // Close the WebSocket connection
  ws.async_close(websocket::close_code::normal, yield[ec]);
  if (ec)
    return fail(ec, "close");

  // If we get here then the connection is closed gracefully

  // The make_printable() function helps print a ConstBufferSequence
  // std::cout << beast::make_printable(buffer.data()) << std::endl;
}

void xelis_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    net::yield_context yield,
    bool isDev)
{
  beast::error_code ec;
  auto endpoint = resolve_host(ioc, yield, host, port);
  websocket::stream<beast::tcp_stream> ws(ioc);

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(ws).connect(endpoint);

  // Update the host string. This will provide the value of the
  // Host HTTP header during the WebSocket handshake.
  // See https://tools.ietf.org/html/rfc7230#section-5.4
  host += ':' + port; //std::to_string(daemon.port());

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

  // Turn off the timeout on the tcp_stream, because
  // the websocket stream has its own timeout system.
  beast::get_lowest_layer(ws).expires_never();

  // Set suggested timeout settings for the websocket
  ws.set_option(
      websocket::stream_base::timeout::suggested(
          beast::role_type::client));

  // Perform the websocket handshake
  std::stringstream ss;
  ss << "/getwork/" << wallet << "/" << worker;
  ws.async_handshake(host, ss.str().c_str(), yield[ec]);
  if (ec)
  {
    return fail(ec, "handshake");
  }

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  bool subStart = false;

  while (true)
  {
    try
    {
      bool *B = isDev ? &submittingDev : &submitting;
      if (*B)
      {
        boost::json::object *S = isDev ? &devShare : &share;
        std::string msg = boost::json::serialize(*S);

        // Acquire a lock before writing to the WebSocket
        ws.async_write(boost::asio::buffer(msg), [&](const boost::system::error_code &ec, std::size_t)
                       {
            if (ec) {
                setcolor(RED);
                printf("\nasync_write: submission error\n");
                setcolor(BRIGHT_WHITE);
            } });
        (*B) = false;
      }

      buffer.clear();
      workInfo.str("");
      workInfo.clear();

      beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(180));
      ws.async_read(buffer, yield[ec]);
      if (!ec)
      {
        // handle getwork feed
        beast::get_lowest_layer(ws).expires_never();
        workInfo << beast::make_printable(buffer.data());

        // std::cout << "Received data: " << workInfo.str() << std::endl;
        if (json::accept(workInfo.str()))
        {
          json response = json::parse(workInfo.str());
          json workData;
          if (response.contains("block_rejected"))
          {
            rejected++;
          }
          else if (response.contains("new_job") || response.contains("template"))
          {

            if (response.contains("new_job"))
              workData = response.at("new_job");
            else if (response.contains("template"))
              workData = response;

            if ((isDev ? (workData.at("height") != devHeight) : (workData.at("height") != ourHeight)))
            {
              if (isDev)
                devJob = workData;
              else
                job = workData;
              json *J = isDev ? &devJob : &job;

              if ((*J).contains("lasterror") && (*J).at("lasterror") != "")
              {
                std::cerr << "received error: " << (*J).at("lasterror") << std::endl
                          << consoleLine << "v" << versionString << " ";
              }

              if (!isDev)
              {
                currentBlob = (*J).at("template").get<std::string>();
                ourHeight++;
                difficulty = std::stoull((*J).at("difficulty").get<std::string>());

                if (!isConnected)
                {
                  wsMutex.lock();
                  setcolor(BRIGHT_YELLOW);
                  printf("Mining at: %s/getwork/%s/%s\n", host.c_str(), wallet.c_str(), worker.c_str());
                  setcolor(CYAN);
                  printf("Dev fee: %.2f", devFee);
                  std::cout << "%" << std::endl;
                  setcolor(BRIGHT_WHITE);
                  wsMutex.unlock();
                }
                isConnected = true;
                jobCounter++;
              }
              else
              {
                devBlob = (*J).at("template").get<std::string>();
                devHeight++;
                difficultyDev = std::stoull((*J).at("difficulty").get<std::string>());

                if (!devConnected)
                {
                  wsMutex.lock();
                  setcolor(CYAN);
                  printf("Connected to dev node: %s\n", host.c_str());
                  setcolor(BRIGHT_WHITE);
                  wsMutex.unlock();
                }
                devConnected = true;
                jobCounter++;
              }
            }
          }
          else
          {
            accepted++;
          }
        }
      }
      else
      {
        bool *B = isDev ? &devConnected : &isConnected;
        (*B) = false;
        return fail(ec, "async_read");
      }
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      std::cout << "ws error: " << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);
      // submission_thread.interrupt();
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // Close the WebSocket connection
  // submission_thread.interrupt();
  ws.async_close(websocket::close_code::normal, yield[ec]);
  printf("loop broken\n");
  if (ec)
    return fail(ec, "close");
}

void xatumFailure(bool isDev) noexcept
{
  setcolor(RED);
  if (isDev)
    printf("DEV | ");
  printf("Xatum Disconnect\n");
  setcolor(BRIGHT_WHITE);
}

void xatum_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  ctx.set_options(boost::asio::ssl::context::default_workarounds |
                  boost::asio::ssl::context::no_sslv2 |
                  boost::asio::ssl::context::no_sslv3 |
                  boost::asio::ssl::context::no_tlsv1 |
                  boost::asio::ssl::context::no_tlsv1_1);

  beast::error_code ec;
  ctx.set_verify_mode(ssl::verify_none); // Accept self-signed certificates
  tcp::socket socket(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);
  boost::asio::deadline_timer deadline(ioc, boost::posix_time::seconds(1));

  auto endpoint = resolve_host(ioc, yield, host, port);
  // Set a timeout on the operation
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec)
    return fail(ec, "connect");

  // Set the SNI hostname
  if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
  {
    throw beast::system_error{
        static_cast<int>(::ERR_get_error()),
        boost::asio::error::get_ssl_category()};
  }

  // Perform the SSL handshake
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(300));
  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec)
    return fail(ec, "handshake");

  boost::json::object handshake_packet = {
      {"addr", wallet.c_str()},
      {"work", worker.c_str()},
      {"agent", (std::string("tnn-miner ") + versionString).c_str()},
      {"algos", boost::json::array{
                    "xel/0",
                }}};

  // std::string handshakeStr = handshake_packet.serialize();
  std::string handshakeStr = "shake~" + boost::json::serialize(handshake_packet) + "\n";

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // stream.async_write_some(boost::asio::buffer(handshakeStr, 1024), yield[ec]);
  // if (ec)
  //     return fail(ec, "write");

  size_t trans = boost::asio::async_write(stream, boost::asio::buffer(handshakeStr), yield[ec]);
  if (ec)
    return fail(ec, "Xatum C2S handshake");

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  Xatum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

  while (true)
  {
    try
    {
      if (
          Xatum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - Xatum::lastReceivedJobTime > Xatum::jobTimeout)
      {
        bool *C = isDev ? &devConnected : &isConnected;
        (*C) = false;
        return fail(ec, "Xatum session timed out");
      }
      bool *B = isDev ? &submittingDev : &submitting;
      if (*B)
      {
        boost::json::object *S = &share;
        if (isDev)
          S = &devShare;
        std::string msg = Xatum::submission + boost::json::serialize(*S) + "\n";
        // if (lastHash.compare((*S).at("hash").get_string()) == 0) continue;
        // lastHash = (*S).at("hash").get_string();

        // printf("submitting share: %s\n", msg.c_str());
        // Acquire a lock before writing to the WebSocket
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(10));
        boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
        if (ec)
        {
          bool *C = isDev ? &devConnected : &isConnected;
          (*C) = false;
          return fail(ec, "Xatum submission");
        }
        (*B) = false;
      }
      boost::asio::streambuf response;
      std::stringstream workInfo;
      beast::get_lowest_layer(stream).expires_after(std::chrono::milliseconds(45000));

      deadline.expires_from_now(boost::posix_time::seconds(1));
      deadline.async_wait([&](beast::error_code ec)
                          {
          if (!ec) {
              beast::get_lowest_layer(stream).cancel();
          } });
      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      // if (ec && trans > 0)
      //   return fail(ec, "Xatum async_read_until");

      if (trans > 0)
      {
        std::string data = beast::buffers_to_string(response.data());
        // Consume the data from the buffer after processing it
        response.consume(trans);

        if (data.compare(Xatum::pingPacket) == 0)
        {
          // printf("pinged\n");
          boost::asio::async_write(stream, boost::asio::buffer(Xatum::pongPacket), yield[ec]);
        }
        else
        {
          Xatum::packet xPacket = Xatum::parsePacket(data, "~");
          int r = handleXatumPacket(xPacket, isDev);
          // if (r == -1) {
          //   bool *B = isDev ? &devConnected : &isConnected;
          //   (*B) = false;
          //   // return xatumFailure(isDev);
          // }
        }
      }
    }
    catch (const std::exception &e)
    {
      bool *C = isDev ? &devConnected : &isConnected;
      (*C) = false;
      return fail(ec, "Xatum session error");
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // submission_thread.interrupt();
  stream.async_shutdown(yield[ec]);
}

int handleXatumPacket(Xatum::packet xPacket, bool isDev)
{
  std::string command = xPacket.command;
  json data = xPacket.data;
  int res = 0;

  if (command == Xatum::print)
  {
    if (Xatum::accepted.compare(data.at("msg").get<std::string>()) == 0)
      accepted++;

    if (Xatum::stale.compare(data.at("msg").get<std::string>()) == 0)
      rejected++;

    int msgLevel = data.at("lvl").get<int>();
    if (msgLevel < Xatum::logLevel)
      return 0;

    printf("\n");
    if (isDev)
    {
      setcolor(CYAN);
      printf("DEV | ");
    }

    wsMutex.lock();
    switch (msgLevel)
    {
    case Xatum::ERROR_MSG:
      if (!isDev)
        setcolor(RED);
      res = -1;
      printf("Xatum ERROR: ");
      break;
    case Xatum::WARN_MSG:
      if (!isDev)
        setcolor(BRIGHT_YELLOW);
      printf("Xatum WARNING: ");
      break;
    case Xatum::INFO_MSG:
      if (!isDev)
        setcolor(BRIGHT_WHITE);
      printf("Xatum INFO: ");
      break;
    case Xatum::VERBOSE_MSG:
      if (!isDev)
        setcolor(BRIGHT_WHITE);
      printf("Xatum VERBOSE: ");
      break;
    }

    printf("%s\n", data.at("msg").get<std::string>().c_str());

    setcolor(BRIGHT_WHITE);
    wsMutex.unlock();
  }

  else if (command == Xatum::newJob)
  {
    uint64_t *diff = isDev ? &difficultyDev : &difficulty;
    json *J = isDev ? &devJob : &job;
    uint64_t *h = isDev ? &devHeight : &ourHeight;

    std::string *B = isDev ? &devBlob : &currentBlob;

    if (data.at("blob").get<std::string>().compare(*B) == 0)
      return 0;
    *B = data.at("blob").get<std::string>();

    Xatum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    // std::cout << data << std::endl;
    if (!isDev)
    {
      setcolor(CYAN);
      printf("\nNew Xatum job received\n");
      setcolor(BRIGHT_WHITE);
    }
    *diff = data.at("diff").get<uint64_t>();

    if ((*J).contains("template"))
      (*J).at("template") = (*B).c_str();
    else
      (*J).emplace("template", (*B).c_str());

    bool *C = isDev ? &devConnected : &isConnected;

    if (!*C)
    {
      if (!isDev)
      {
        wsMutex.lock();
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
        setcolor(CYAN);
        printf("Dev fee: %.2f", devFee);
        std::cout << "%" << std::endl;
        setcolor(BRIGHT_WHITE);
        wsMutex.unlock();
      }
      else
      {
        wsMutex.lock();
        setcolor(CYAN);
        printf("Connected to dev node: %s\n", host.c_str());
        setcolor(BRIGHT_WHITE);
        wsMutex.unlock();
      }
    }

    *C = true;

    (*h)++;
    jobCounter++;
  }

  else if (!isDev && command == Xatum::success)
  {
    // std::cout << data << std::endl;
    if (data.at("msg").get<std::string>() == "ok")
    {
      printf("Xatum: share accepted!");
      accepted++;
    }
    else
    {
      rejected++;
      setcolor(RED);
      printf("\nXatum Share Rejected: %s\n", data.at("msg").get<std::string>().c_str());
      setcolor(BRIGHT_WHITE);
    }
  }
  else
  {
    printf("unknown command: %s\n", command.c_str());
  }

  return res;
}

void xelis_stratum_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  ctx.set_options(boost::asio::ssl::context::default_workarounds |
                  boost::asio::ssl::context::no_sslv2 |
                  boost::asio::ssl::context::no_sslv3 |
                  boost::asio::ssl::context::no_tlsv1 |
                  boost::asio::ssl::context::no_tlsv1_1);

  beast::error_code ec;
  boost::system::error_code jsonEc;
  boost::asio::deadline_timer deadline(ioc, boost::posix_time::seconds(1));

  auto endpoint = resolve_host(ioc, yield, host, port);

  // Create a TCP socket
  ctx.set_verify_mode(ssl::verify_none); // Accept self-signed certificates
  //tcp::socket socket(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);
  // Set a timeout on the operation
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec)
    return fail(ec, "connect");

  // Set the SNI hostname
  if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
  {
    throw beast::system_error{
        static_cast<int>(::ERR_get_error()),
        boost::asio::error::get_ssl_category()};
  }

  // Perform the SSL handshake
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(300));
  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec)
    return fail(ec, "handshake");

  boost::json::object packet = XelisStratum::stratumCall;
  packet.at("id") = XelisStratum::subscribe.id;
  packet.at("method") = XelisStratum::subscribe.method;
  std::string minerName = "tnn-miner/" + std::string(versionString);
  packet.at("params") = boost::json::array({minerName, boost::json::array({"xel/0"})});
  std::string subscription = boost::json::serialize(packet) + "\n";

  // std::cout << subscription << std::endl;

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  size_t trans = boost::asio::async_write(stream, boost::asio::buffer(subscription), yield[ec]);
  if (ec)
    return fail(ec, "Stratum subscribe");

  // Make sure subscription is successful
  boost::asio::streambuf subRes;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::read_until(stream, subRes, "\n");

  std::string subResString = beast::buffers_to_string(subRes.data());
  subRes.consume(trans);
  boost::json::object subResJson = boost::json::parse(subResString.c_str(), jsonEc).as_object();
  if (jsonEc)
  {
    std::cerr << jsonEc.message() << std::endl;
  }

  // std::cout << boost::json::serialize(subResJson).c_str() << std::endl;
  handleXStratumResponse(subResJson, isDev);

  // Authorize Stratum Worker
  packet = XelisStratum::stratumCall;
  packet.at("id") = XelisStratum::authorize.id;
  packet.at("method") = XelisStratum::authorize.method;
  packet.at("params") = boost::json::array({wallet, worker, "x"});
  std::string authorization = boost::json::serialize(packet) + "\n";

  // std::cout << authorization << std::endl;

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::async_write(stream, boost::asio::buffer(authorization), yield[ec]);
  if (ec)
    return fail(ec, "Stratum authorize");

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  XelisStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

  while (true)
  {
    try
    {
      if (
          XelisStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - XelisStratum::lastReceivedJobTime > XelisStratum::jobTimeout)
      {
        bool *C = isDev ? &devConnected : &isConnected;
        (*C) = false;
        return fail(ec, "Stratum session timed out");
      }
      bool *B = isDev ? &submittingDev : &submitting;
      if (*B)
      {
        boost::json::object *S = &share;
        if (isDev)
          S = &devShare;

        std::string msg = boost::json::serialize((*S)) + "\n";
        // if (lastHash.compare((*S).at("hash").get_string()) == 0) continue;
        // lastHash = (*S).at("hash").get_string();

        // printf("submitting share: %s\n", msg.c_str());
        // Acquire a lock before writing to the WebSocket

        // std::cout << "sending in: " << msg << std::endl;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(10));
        boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
        if (ec)
        {
          bool *C = isDev ? &devConnected : &isConnected;
          (*C) = false;
          return fail(ec, "Stratum submit");
        }
        (*B) = false;
      }

      boost::asio::streambuf response;
      std::stringstream workInfo;
      beast::get_lowest_layer(stream).expires_after(std::chrono::milliseconds(60000));

      deadline.expires_from_now(boost::posix_time::seconds(1));
      deadline.async_wait([&](beast::error_code ec)
                          {
          if (!ec) {
              beast::get_lowest_layer(stream).cancel();
          } });
      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec && trans > 0)
        return fail(ec, "Stratum async_read");

      if (trans > 0)
      {
        std::scoped_lock<boost::mutex> lockGuard(wsMutex);
        std::string data = beast::buffers_to_string(response.data());
        // Consume the data from the buffer after processing it
        response.consume(trans);

        // std::cout << data << std::endl;

        if (data.compare(XelisStratum::k1ping) == 0)
        {
          trans = boost::asio::async_write(
              stream,
              boost::asio::buffer(XelisStratum::k1pong),
              yield[ec]);
          if (ec && trans > 0)
            return fail(ec, "Stratum pong (K1 style)");
        }
        else
        {
          boost::json::object sRPC = boost::json::parse(data.c_str()).as_object();
          if (sRPC.contains("method"))
          {
            if (std::string(sRPC.at("method").as_string().c_str()).compare(XelisStratum::s_ping) == 0)
            {
              boost::json::object pong({{"id", sRPC.at("id").get_uint64()},
                                        {"method", XelisStratum::pong.method}});
              std::string pongPacket = std::string(boost::json::serialize(pong).c_str()) + "\n";
              trans = boost::asio::async_write(
                  stream,
                  boost::asio::buffer(pongPacket),
                  yield[ec]);
              if (ec && trans > 0)
                return fail(ec, "Stratum pong");
            }
            else
              handleXStratumPacket(sRPC, isDev);
          }
          else
          {
            handleXStratumResponse(sRPC, isDev);
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      bool *C = isDev ? &devConnected : &isConnected;
      (*C) = false;
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);
      return fail(ec, "Stratum session error");
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // submission_thread.interrupt();
  stream.async_shutdown(yield[ec]);
}

int handleXStratumPacket(boost::json::object packet, bool isDev)
{
  std::string M = packet.at("method").get_string().c_str();
  if (M.compare(XelisStratum::s_notify) == 0)
  {

    if (ourHeight > 0 && packet.at("params").as_array()[4].get_bool() != true)
      return 0;

    setcolor(CYAN);
    if (!isDev)
      printf("\nStratum: new job received\n");
    setcolor(BRIGHT_WHITE);

    XelisStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    json *J = isDev ? &devJob : &job;
    uint64_t *h = isDev ? &devHeight : &ourHeight;

    std::string bs = (*J).at("template").get<std::string>();
    char *blob = (char *)bs.c_str();

    const char *jobId = packet.at("params").as_array()[0].get_string().c_str();
    const char *ts = packet.at("params").as_array()[1].get_string().c_str();
    int tsLen = packet.at("params").as_array()[1].get_string().size();
    const char *header = packet.at("params").as_array()[2].get_string().c_str();

    memset(&blob[64], '0', 16);
    memcpy(&blob[64 + 16 - tsLen], ts, tsLen);
    memcpy(blob, header, 64);

    (*J).at("template") = std::string(blob);
    (*J)["jobId"] = jobId;

    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
        setcolor(CYAN);
        printf("Dev fee: %.2f", devFee);
        std::cout << "%" << std::endl;
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        setcolor(CYAN);
        printf("Connected to dev node: %s\n", host.c_str());
        setcolor(BRIGHT_WHITE);
      }
    }

    *C = true;
    (*h)++;
    jobCounter++;
  }
  else if (M.compare(XelisStratum::s_setDifficulty) == 0)
  {

    uint64_t *d = isDev ? &difficultyDev : &difficulty;
    (*d) = packet.at("params").as_array()[0].get_double();
    if ((*d) == 0)
      (*d) = packet.at("params").as_array()[0].get_uint64();
  }
  else if (M.compare(XelisStratum::s_setExtraNonce) == 0)
  {

    json *J = isDev ? &devJob : &job;
    uint64_t *h = isDev ? &devHeight : &ourHeight;

    std::string bs = (*J).at("template").get<std::string>();
    char *blob = (char *)bs.c_str();
    const char *en = packet.at("params").as_array()[0].as_string().c_str();
    int enLen = packet.at("params").as_array()[0].as_string().size();

    memset(&blob[48], '0', 64);
    memcpy(&blob[48], en, enLen);

    (*J).at("template") = std::string(blob).c_str();

    (*h)++;
    jobCounter++;
  }
  else if (M.compare(XelisStratum::s_print) == 0)
  {

    int lLevel = packet.at("params").as_array()[0].as_int64();
    if (lLevel != XelisStratum::STRATUM_DEBUG)
    {
      int res = 0;
      printf("\n");
      if (isDev)
      {
        setcolor(CYAN);
        printf("DEV | ");
      }

      switch (lLevel)
      {
      case XelisStratum::STRATUM_INFO:
        if (!isDev)
          setcolor(BRIGHT_WHITE);
        printf("Stratum INFO: ");
        break;
      case XelisStratum::STRATUM_WARN:
        if (!isDev)
          setcolor(BRIGHT_YELLOW);
        printf("Stratum WARNING: ");
        break;
      case XelisStratum::STRATUM_ERROR:
        if (!isDev)
          setcolor(RED);
        printf("Stratum ERROR: ");
        res = -1;
        break;
      case XelisStratum::STRATUM_DEBUG:
        break;
      }
      printf("%s\n", packet.at("params").as_array()[1].as_string().c_str());

      setcolor(BRIGHT_WHITE);

      return res;
    }
  }
  return 0;
}

int handleXStratumResponse(boost::json::object packet, bool isDev)
{
  // if (!isDev) {
  // if (!packet.contains("id")) return 0;
  int64_t id = packet["id"].as_int64();

  switch (id)
  {
  case XelisStratum::subscribeID:
  {
    json *J = isDev ? &devJob : &job;
    if (!(*J).contains("template"))
    {
      byte blankBlob[XELIS_TEMPLATE_SIZE * 2];
      memset(blankBlob, '0', XELIS_TEMPLATE_SIZE * 2);
      std::string base = std::string((char *)blankBlob);
      base.resize(XELIS_TEMPLATE_SIZE * 2);
      (*J).emplace("template", base.c_str());
    }

    std::string bs = (*J).at("template").get<std::string>();
    char *blob = (char *)bs.c_str();
    const char *extraNonce = packet.at("result").get_array()[1].get_string().c_str();
    int enLen = packet.at("result").get_array()[2].get_int64();
    const char *pubKey = packet.at("result").get_array()[3].get_string().c_str();

    memset(&blob[96], '0', 64);
    memcpy(&blob[96], extraNonce, enLen * 2);
    memcpy(&blob[160], pubKey, 64);

    (*J).at("template") = std::string(blob).c_str();
  }
  break;
  case XelisStratum::submitID:
  {
    printf("\n");
    if (isDev)
    {
      setcolor(CYAN);
      printf("DEV | ");
    }
    if (!packet["result"].is_null() && packet.at("result").get_bool())
    {
      accepted++;
      std::cout << "Stratum: share accepted" << std::endl;
      setcolor(BRIGHT_WHITE);
    }
    else
    {
      rejected++;
      if (!isDev)
        setcolor(RED);
      std::cout << "Stratum: share rejected: " << packet.at("error").get_object()["message"].get_string() << std::endl;
      setcolor(BRIGHT_WHITE);
    }
    break;
  }
  }
  return 0;
}

void spectre_stratum_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  ctx.set_options(boost::asio::ssl::context::default_workarounds |
                  boost::asio::ssl::context::no_sslv2 |
                  boost::asio::ssl::context::no_sslv3 |
                  boost::asio::ssl::context::no_tlsv1 |
                  boost::asio::ssl::context::no_tlsv1_1);

  beast::error_code ec;
  boost::system::error_code jsonEc;
  boost::asio::deadline_timer deadline(ioc, boost::posix_time::seconds(1));

  auto endpoint = resolve_host(ioc, yield, host, port);
  boost::beast::tcp_stream stream(ioc);

  // Set a timeout on the operation
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec)
    return fail(ec, "connect");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  boost::json::object packet;

  // Authorize Stratum Worker
  packet = XelisStratum::stratumCall;
  packet.at("id") = XelisStratum::authorize.id;
  packet.at("method") = XelisStratum::authorize.method;
  packet.at("params") = boost::json::array({wallet + "." + worker});

  std::string authorization = boost::json::serialize(packet) + "\n";

  // // std::cout << authorization << std::endl;

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  boost::asio::async_write(stream, boost::asio::buffer(authorization), yield[ec]);
  if (ec)
    return fail(ec, "Stratum authorize");

  packet = XelisStratum::stratumCall;

  packet["id"] = SpectreStratum::subscribe.id;
  packet["method"] = SpectreStratum::subscribe.method;
  packet["params"] = boost::json::array({
    minerName
  });

  boost::asio::streambuf subRes;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  size_t trans = boost::asio::read_until(stream, subRes, "\n");

  std::string authResString = beast::buffers_to_string(subRes.data());
  subRes.consume(trans);
  boost::json::object authResJson = boost::json::parse(authResString.c_str(), jsonEc).as_object();
  if (jsonEc)
  {
    std::cerr << jsonEc.message() << std::endl;
  }

  // SpectreStratum::stratumCall;
  // packet.at("id") = SpectreStratum::subscribe.id;
  // packet.at("method") = SpectreStratum::subscribe.method;
  // packet.at("params") = {minerName};
  std::string subscription = boost::json::serialize(packet) + "\n";

  // std::cout << authResString << std::endl;

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::async_write(stream, boost::asio::buffer(subscription), yield[ec]);
  if (ec)
    return fail(ec, "Stratum subscribe");

  // beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // trans = boost::asio::read_until(stream, subRes, "\n");

  // std::string subResString = beast::buffers_to_string(subRes.data());
  // subRes.consume(trans);

  // wsMutex.lock();
  // printf("before packet\n");
  // std::cout << subResString << std::endl;

  // printf("before parse\n");
  // wsMutex.unlock();
  // boost::json::object subResJson = boost::json::parse(subResString.c_str(), jsonEc).as_object();
  // if (jsonEc)
  // {
  //   std::cerr << jsonEc.message() << std::endl;
  // }

  // printf("after parse\n");

  // handleXStratumPacket(subResJson, isDev);

  // // This buffer will hold the incoming message
  // beast::flat_buffer buffer;
  // std::stringstream workInfo;

  SpectreStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

  SpectreStratum::jobCache jobCache;

  while (true)
  {
    try
{
      if (
        !isDev &&
        SpectreStratum::lastShareSubmissionTime > 0 &&
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - SpectreStratum::lastShareSubmissionTime > SpectreStratum::shareSubmitTimeout) {
          rate30sec.clear();
          rate30sec.push_back(0);
        }

      if (
          SpectreStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - SpectreStratum::lastReceivedJobTime > SpectreStratum::jobTimeout)
      {
        bool *C = isDev ? &devConnected : &isConnected;
        (*C) = false;
        return fail(ec, "Stratum session timed out");
      }
      bool *B = isDev ? &submittingDev : &submitting;
      if (*B)
      {
        boost::json::object *S = &share;
        if (isDev)
          S = &devShare;

        std::string msg = boost::json::serialize((*S)) + "\n";
        // if (lastHash.compare((*S).at("hash").get_string()) == 0) continue;
        // lastHash = (*S).at("hash").get_string();

        // printf("submitting share: %s\n", msg.c_str());
        // Acquire a lock before writing to the WebSocket

        // std::cout << "sending in: " << msg << std::endl;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(10));
        boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
        if (ec)
        {
          bool *C = isDev ? &devConnected : &isConnected;
          (*C) = false;
          return fail(ec, "Stratum submit");
        }
        if (!isDev) SpectreStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        (*B) = false;
      }

      boost::asio::streambuf response;
      std::stringstream workInfo;
      beast::get_lowest_layer(stream).expires_after(std::chrono::milliseconds(60000));

      deadline.expires_from_now(boost::posix_time::seconds(1));
      deadline.async_wait([&](beast::error_code ec)
                          {
          if (!ec) {
              beast::get_lowest_layer(stream).cancel();
          } });
      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec && trans > 0)
        return fail(ec, "Stratum async_read");

      if (trans > 0)
      {
        std::scoped_lock<boost::mutex> lockGuard(wsMutex);
        std::vector<std::string> packets;
        std::string data = beast::buffers_to_string(response.data());
        // Consume the data from the buffer after processing it
        response.consume(trans);

        // std::cout << data << std::endl;

        std::stringstream  jsonStream(data);

        std::string line;
        while(std::getline(jsonStream,line,'\n'))
        {
            packets.push_back(line);
        }

        for (std::string packet : packets) {
          try {
            boost::json::object sRPC = boost::json::parse(packet.c_str()).as_object();
            if (sRPC.contains("method"))
            {
              if (std::string(sRPC.at("method").as_string().c_str()).compare(SpectreStratum::s_ping) == 0)
              {
                boost::json::object pong({{"id", sRPC.at("id").get_uint64()},
                                          {"method", SpectreStratum::pong.method}});
                std::string pongPacket = std::string(boost::json::serialize(pong).c_str()) + "\n";
                trans = boost::asio::async_write(
                    stream,
                    boost::asio::buffer(pongPacket),
                    yield[ec]);
                if (ec && trans > 0)
                  return fail(ec, "Stratum pong");
              }
              else
                handleSpectreStratumPacket(sRPC, &jobCache, isDev);
            }
            else
            {
              handleSpectreStratumResponse(sRPC, isDev);
            } 
          } catch(const std::exception &e){
            setcolor(RED);
            printf("BEFORE PACKET\n");
            std::cout << packet << std::endl;
            printf("AFTER PACKET\n");
            std::cerr << e.what() << std::endl;
            setcolor(BRIGHT_WHITE);
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      bool *C = isDev ? &devConnected : &isConnected;
      (*C) = false;
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);
      return fail(ec, "Stratum session error");
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // submission_thread.interrupt();
  printf("\n\n\nflagged connection loss\n");
  stream.close();
}

void do_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    int algo,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  switch (algo)
  {
  case DERO_HASH:
    dero_session(host, port, wallet, ioc, ctx, yield, isDev);
    break;
  case XELIS_HASH:
  {
    switch (protocol)
    {
    case XELIS_SOLO:
      xelis_session(host, port, wallet, worker, ioc, yield, isDev);
      break;
    case XELIS_XATUM:
      xatum_session(host, port, wallet, worker, ioc, ctx, yield, isDev);
      break;
    case XELIS_STRATUM:
      xelis_stratum_session(host, port, wallet, worker, ioc, ctx, yield, isDev);
      break;
    }
    break;
  }
  case SPECTRE_X:
    spectre_stratum_session(host, port, wallet, worker, ioc, ctx, yield, isDev);
    break;
  }
}

int handleSpectreStratumPacket(boost::json::object packet, SpectreStratum::jobCache *cache, bool isDev)
{
  std::string M = packet.at("method").get_string().c_str();
  if (M.compare(SpectreStratum::s_notify) == 0)
  {
    json *J = isDev ? &devJob : &job;
    uint64_t *h = isDev ? &devHeight : &ourHeight;

    uint64_t id = std::stoull(packet["params"].as_array()[0].get_string().c_str());

    uint64_t h1 = packet["params"].as_array()[1].as_array()[0].get_uint64();
    uint64_t h2 = packet["params"].as_array()[1].as_array()[1].get_uint64();
    uint64_t h3 = packet["params"].as_array()[1].as_array()[2].get_uint64();
    uint64_t h4 = packet["params"].as_array()[1].as_array()[3].get_uint64();

    uint64_t comboHeader[4] = {h1, h2, h3, h4};

    bool isEqual = true;
    for (int i = 0; i < 4; i++) {
      isEqual &= comboHeader[i] == cache->header[i];
    }
    if (isEqual) return 0;

    for (int i = 0; i < 4; i++) {
      cache->header[i] = comboHeader[i];
    }

    uint64_t ts = packet["params"].as_array()[2].get_uint64();

    std::string h1Str = hexStr((byte*)&h1, 8);
    std::string h2Str = hexStr((byte*)&h2, 8);
    std::string h3Str = hexStr((byte*)&h3, 8);
    std::string h4Str = hexStr((byte*)&h4, 8);

    std::string tsStr = hexStr((byte*)&ts, 8);

    char newTemplate[160];
    memset(newTemplate, '0', 160);

    memcpy(newTemplate + 16 - h1Str.size(), h1Str.data(), h1Str.size());
    memcpy(newTemplate + 16 + 16 - h2Str.size(), h2Str.data(), h2Str.size());
    memcpy(newTemplate + 32 + 16 - h3Str.size(), h3Str.data(), h3Str.size());
    memcpy(newTemplate + 48 + 16 - h4Str.size(), h4Str.data(), h4Str.size());
    memcpy(newTemplate + 64 + 16 - tsStr.size(), tsStr.data(), tsStr.size());

    setcolor(CYAN);
    if (!isDev)
      printf("\nStratum: new job received\n");
    setcolor(BRIGHT_WHITE);

    SpectreStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();


    (*J)["template"] = std::string(newTemplate, SpectreX::INPUT_SIZE*2);
    // std::string testPrint = (*J)["template"].get<std::string>();

    // byte testOut[160];
    // memcpy(testOut, testPrint.data(), 160);
    // for (int i = 0; i < 160; i++) {
    //   std::cout << testOut[i];
    // }
    // printf("\n");
    // std::cout << testPrint;
    // printf("\n");

    // std::string bs = (*J).at("template").get<std::string>();
    // char *blob = (char *)bs.c_str();

    // const char *jobId = packet.at("params").as_array()[0].get_string().c_str();
    // const char *ts = packet.at("params").as_array()[1].get_string().c_str();
    // int tsLen = packet.at("params").as_array()[1].get_string().size();
    // const char *header = packet.at("params").as_array()[2].get_string().c_str();

    // memset(&blob[64], '0', 16);
    // memcpy(&blob[64 + 16 - tsLen], ts, tsLen);
    // memcpy(blob, header, 64);

    // (*J).at("template") = std::string(blob);
    (*J)["jobId"] = id;

    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
        setcolor(CYAN);
        printf("Dev fee: %.2f", devFee);
        std::cout << "%" << std::endl;
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        setcolor(CYAN);
        printf("Connected to dev node: %s\n", host.c_str());
        setcolor(BRIGHT_WHITE);
      }
    }

    *C = true;
    (*h)++;
    jobCounter++;
  }
  else if (M.compare(SpectreStratum::s_setDifficulty) == 0)
  {
    // std::cout << boost::json::serialize(packet).c_str() << std::endl;
    double *d = isDev ? &doubleDiffDev : &doubleDiff;
    (*d) = packet.at("params").as_array()[0].get_double();
    if ((*d) < 0.00000000001) (*d) = packet.at("params").as_array()[0].get_uint64();

    // printf("%f\n", (*d));
  }
  else if (M.compare(SpectreStratum::s_setExtraNonce) == 0)
  {
    // std::cout << boost::json::serialize(packet).c_str() << std::endl;
    // json *J = isDev ? &devJob : &job;
    // uint64_t *h = isDev ? &devHeight : &ourHeight;

    // std::string bs = (*J).at("template").get<std::string>();
    // char *blob = (char *)bs.c_str();
    // const char *en = packet.at("params").as_array()[0].as_string().c_str();
    // int enLen = packet.at("params").as_array()[0].as_string().size();

    // memset(&blob[48], '0', 64);
    // memcpy(&blob[48], en, enLen);

    // (*J).at("template") = std::string(blob).c_str();

    // (*h)++;
    // jobCounter++;
  }
  else if (M.compare(SpectreStratum::s_print) == 0)
  {

    int lLevel = packet.at("params").as_array()[0].as_int64();
    if (lLevel != SpectreStratum::STRATUM_DEBUG)
    {
      int res = 0;
      printf("\n");
      if (isDev)
      {
        setcolor(CYAN);
        printf("DEV | ");
      }

      switch (lLevel)
      {
      case SpectreStratum::STRATUM_INFO:
        if (!isDev)
          setcolor(BRIGHT_WHITE);
        printf("Stratum INFO: ");
        break;
      case SpectreStratum::STRATUM_WARN:
        if (!isDev)
          setcolor(BRIGHT_YELLOW);
        printf("Stratum WARNING: ");
        break;
      case SpectreStratum::STRATUM_ERROR:
        if (!isDev)
          setcolor(RED);
        printf("Stratum ERROR: ");
        res = -1;
        break;
      case SpectreStratum::STRATUM_DEBUG:
        break;
      }
      printf("%s\n", packet.at("params").as_array()[1].as_string().c_str());

      setcolor(BRIGHT_WHITE);

      return res;
    }
  }
  return 0;
}

int handleSpectreStratumResponse(boost::json::object packet, bool isDev)
{
  // if (!isDev) {
  // if (!packet.contains("id")) return 0;
  int64_t id = packet["id"].as_int64();

  switch (id)
  {
    case SpectreStratum::subscribeID:
    {
      std::cout << boost::json::serialize(packet).c_str() << std::endl;
      if (packet["error"].is_null()) return 0;
      else {
        const char *errorMsg = packet["error"].get_string().c_str();
        setcolor(RED);
        printf("\n");
        if (isDev) {
          setcolor(CYAN);
          printf("DEV | ");
        }
        printf("Stratum ERROR: %s\n", errorMsg);
        return -1;
      }
    }
    break;
    case SpectreStratum::submitID:
    {
      printf("\n");
      if (isDev)
      {
        setcolor(CYAN);
        printf("DEV | ");
      }
      if (!packet["result"].is_null() && packet.at("result").get_bool())
      {
        if (!isDev) accepted++;
        std::cout << "Stratum: share accepted" << std::endl;
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        if (!isDev) rejected++;
        if (!isDev)
          setcolor(RED);
        std::cout << "Stratum: share rejected: " << packet.at("error").get_array()[1].get_string() << std::endl;
        setcolor(BRIGHT_WHITE);
      }
      break;
    }
  }
  return 0;
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
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Remember: Long options now use a double-dash -- instead of a single-dash -\n";
    return -1;
  }
  catch (...)
  {
    std::cerr << "Unknown error!"
              << "\n";
    return -1;
  }

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
    SpectreX::test();
    return 0;
  }

  if (vm.count("xelis-test"))
  {
    xelis_runTests();
    return 0;
  }

  if (vm.count("xelis-bench"))
  {
    boost::thread t(xelis_benchmark_cpu_hash);
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
    // TODO: Check if this contains a host:port... and then parse accordingly
  }
  if (vm.count("port"))
  {
    port = std::to_string(vm["port"].as<int>());
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
  }
  if (vm.count("worker-name"))
  {
    workerName = vm["worker-name"].as<std::string>();
  }
  else
  {
    workerName = boost::asio::ip::host_name();
  }
  if (vm.count("threads"))
  {
    threads = vm["threads"].as<int>();
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

  // Ensure we capture *all* of the other options before we start using goto
  if (vm.count("dero-test"))
  {
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

      if (i == 0)
      {
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

  host = devPool;
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
 //  mutex.lock();
  printSupported();
 //  mutex.unlock();

  if (miningAlgo == DERO_HASH && !(wallet.substr(0, 3) == "der" || wallet.substr(0, 3) == "det"))
  {
    std::cout << "Provided wallet address is not valid for Dero" << std::endl;
    return EXIT_FAILURE;
  }
  if (miningAlgo == XELIS_HASH && !(wallet.substr(0, 3) == "xel" || wallet.substr(0, 3) == "xet"))
  {
    std::cout << "Provided wallet address is not valid for Xelis" << std::endl;
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

      std::cout << "Thread " << i + 1 << " started" << std::endl;
    }
 //  mutex.unlock();

  auto start_time = std::chrono::steady_clock::now();
  if (broadcastStats)
  {
    boost::thread BROADCAST(BroadcastServer::serverThread, &rate30sec, &accepted, &rejected, versionString);
  }

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

      float ratio = 1000.0f / milliseconds;
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

      if (hashrate >= 1000000)
      {
        double rate = (double)(hashrate / 1000000.0);
        std::string hrate = fmt::sprintf("HASHRATE %.3f MH/s", rate);
       //  mutex.lock();
        setcolor(BRIGHT_WHITE);
        std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << versionString << " ";
        setcolor(CYAN);
        std::cout << std::setw(2) << hrate << " | " << std::flush;
      }
      else if (hashrate >= 1000)
      {
        double rate = (double)(hashrate / 1000.0);
        std::string hrate = fmt::sprintf("HASHRATE %.3f KH/s", rate);
       //  mutex.lock();
        setcolor(BRIGHT_WHITE);
        std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << versionString << " ";
        setcolor(CYAN);
        std::cout << std::setw(2) << hrate << " | " << std::flush;
      }
      else
      {
        std::string hrate = fmt::sprintf("HASHRATE %.0f H/s", (double)hashrate, hrate);
       //  mutex.lock();
        setcolor(BRIGHT_WHITE);
        std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << versionString << " ";
        setcolor(CYAN);
        std::cout << std::setw(2) << hrate << " | " << std::flush;
      }

      std::string uptime = fmt::sprintf("%dd-%dh-%dm-%ds >> ", daysUp, hoursUp, minutesUp, secondsUp);

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
      std::string HOST, WORKER, PORT;
      switch (algo)
      {
        case DERO_HASH:
        {
          HOST = devPool;
          WORKER = workerName;
          PORT = devPort[DERO_HASH];
          break;
        }
        case XELIS_HASH:
        {
          HOST = host;
          WORKER = devWorkerName;
          PORT = port;
          break;
        }
        case SPECTRE_X:
        {
          HOST = host;
          WORKER = devWorkerName;
          PORT = port;
          break;
        }
      }
      boost::asio::spawn(ioc, std::bind(&do_session, HOST, PORT, devSelection[algo], WORKER, algo, std::ref(ioc), std::ref(ctx), std::placeholders::_1, true),
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
      boost::asio::spawn(ioc, std::bind(&do_session, host, port, wallet, workerName, algo, std::ref(ioc), std::ref(ctx), std::placeholders::_1, false),
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
    json myJob = job;
    json myJobDev = devJob;
    localJobCounter = jobCounter;

    byte *b2 = new byte[MINIBLOCK_SIZE];
    hexstrToBytes(myJob.at("blockhashing_blob"), b2);
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
  // byte powHash2[32];
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
     //  mutex.lock();
      json myJob = job;
      json myJobDev = devJob;
      localJobCounter = jobCounter;
     //  mutex.unlock();

      byte *b2 = new byte[MINIBLOCK_SIZE];
      hexstrToBytes(myJob.at("blockhashing_blob"), b2);
      memcpy(work, b2, MINIBLOCK_SIZE);
      delete[] b2;

      if (devConnected)
      {
        byte *b2d = new byte[MINIBLOCK_SIZE];
        hexstrToBytes(myJobDev.at("blockhashing_blob"), b2d);
        memcpy(devWork, b2d, MINIBLOCK_SIZE);
        delete[] b2d;
      }

      memcpy(&work[MINIBLOCK_SIZE - 12], random_buf, 12);
      memcpy(&devWork[MINIBLOCK_SIZE - 12], random_buf, 12);

      work[MINIBLOCK_SIZE - 1] = (byte)tid;
      devWork[MINIBLOCK_SIZE - 1] = (byte)tid;

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

        cmpDiff = ConvertDifficultyToBig(DIFF, DERO_HASH);
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

        if (submit && CheckHash(&powHash[0], cmpDiff, DERO_HASH))
        {
          // printf("work: %s, hash: %s\n", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str(), hexStr(powHash, 32).c_str());
          if (devMine)
          {
           //  mutex.lock();
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n";
            setcolor(BRIGHT_WHITE);
            devShare = {
                {"jobid", myJobDev.at("jobid")},
                {"mbl_blob", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str()}};
            submittingDev = true;
           //  mutex.unlock();
          }
          else
          {
           //  mutex.lock();
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n";
            setcolor(BRIGHT_WHITE);
            share = {
                {"jobid", myJob.at("jobid")},
                {"mbl_blob", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str()}};
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
    catch (...)
    {
      std::cerr << "Error in POW Function" << std::endl;
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
      json myJob = job;
      json myJobDev = devJob;
      localJobCounter = jobCounter;

     //  mutex.unlock();

      if (!myJob.contains("template"))
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[XELIS_TEMPLATE_SIZE];
        switch (protocol)
        {
        case XELIS_SOLO:
          hexstrToBytes(myJob.at("template"), b2);
          break;
        case XELIS_XATUM:
        {
          std::string b64 = base64::from_base64(myJob.at("template").get<std::string>());
          memcpy(b2, b64.data(), b64.size());
          break;
        }
        case XELIS_STRATUM:
          hexstrToBytes(myJob.at("template"), b2);
          break;
        }
        memcpy(work, b2, XELIS_TEMPLATE_SIZE);
        delete[] b2;
        localOurHeight = ourHeight;
        i = 0;
      }

      if (devConnected && myJobDev.contains("template"))
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          byte *b2d = new byte[XELIS_TEMPLATE_SIZE];
          switch (protocol)
          {
          case XELIS_SOLO:
            hexstrToBytes(myJobDev.at("template"), b2d);
            break;
          case XELIS_XATUM:
          {
            std::string b64 = base64::from_base64(myJobDev.at("template").get<std::string>().c_str());
            memcpy(b2d, b64.data(), b64.size());
            break;
          }
          case XELIS_STRATUM:
            hexstrToBytes(myJobDev.at("template"), b2d);
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

        if (submit && CheckHash(powHash, cmpDiff, XELIS_HASH))
        {
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
                                       devJob.at("jobId").get<std::string>().c_str(), // JOB ID
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
                                    myJob.at("jobId").get<std::string>().c_str(), // JOB ID
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
    catch (...)
    {
     //  mutex.lock();
      std::cerr << "Error in POW Function" << std::endl;
     //  mutex.unlock();
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
  alignas(64) byte work[SpectreX::INPUT_SIZE] = {0};
  alignas(64) byte devWork[SpectreX::INPUT_SIZE] = {0};

  alignas(64) workerData *astroWorker = (workerData *)malloc_huge_pages(sizeof(workerData));
  alignas(64) SpectreX::worker *worker = (SpectreX::worker *)malloc_huge_pages(sizeof(SpectreX::worker));
  worker->astroWorker = astroWorker;

  alignas(64) workerData *devAstroWorker = (workerData *)malloc_huge_pages(sizeof(workerData));
  alignas(64) SpectreX::worker *devWorker = (SpectreX::worker *)malloc_huge_pages(sizeof(SpectreX::worker));
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
      json myJob = job;
      json myJobDev = devJob;
      localJobCounter = jobCounter;
     //  mutex.unlock();

      // printf("looping somewhere\n");

      if (!myJob.contains("template"))
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[SpectreX::INPUT_SIZE];
        switch (protocol)
        {
        case SPECTRE_SOLO:
          hexstrToBytes(myJob.at("template").get<std::string>(), b2);
          break;
        case SPECTRE_STRATUM:
          hexstrToBytes(myJob.at("template").get<std::string>(), b2);
          break;
        }
        memcpy(work, b2, SpectreX::INPUT_SIZE);
        // SpectreX::genPrePowHash(b2, *worker);/
        // SpectreX::newMatrix(b2, worker->mat);
        delete[] b2;
        localOurHeight = ourHeight;
        i = 0;
      }

      if (devConnected && myJobDev.contains("template"))
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          byte *b2d = new byte[SpectreX::INPUT_SIZE];
          switch (protocol)
          {
          case SPECTRE_SOLO:
            hexstrToBytes(myJobDev.at("template").get<std::string>(), b2d);
            break;
          case SPECTRE_STRATUM:
            hexstrToBytes(myJobDev.at("template").get<std::string>(), b2d);
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
        
        json &J = devMine ? myJobDev : myJob;
        if (J["extraNonce"].is_null() || J["extraNonce"].get<std::string>().size() == 0)
          n = ((tid - 1) % (256 * 256)) | ((rand() % 256) << 16) | ((*nonce) << 24);
        else {
          int eN = J["extraNonce"].get<uint32_t>();
          n = eN | (((tid - 1) % (256 * 256)) << 24) | ((*nonce) << 40);
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

        if (submit && Num(hexStr(powHash, 32).c_str(), 16) <= cmpDiff)
        {
          std::scoped_lock<boost::mutex> lockGuard(mutex);
          // if (littleEndian())
          // {
          //   std::reverse(powHash, powHash + 32);
          // }
        //   std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          if (devMine)
          {
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
              devShare = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case SPECTRE_STRATUM:
              std::vector<char> nonceStr;
              Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
              devShare = {{{"id", SpectreStratum::submitID},
                        {"method", SpectreStratum::submit.method.c_str()},
                        {"params", {devWorkerName,                                   // WORKER
                                    std::to_string(devJob["jobId"].get<uint64_t>()).c_str(), // JOB ID
                                    std::string(nonceStr.data()).c_str()}}}};

              break;
            }
            submittingDev = true;
          }
          else
          {
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
              share = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case SPECTRE_STRATUM:
              std::vector<char> nonceStr;
              Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
              share = {{{"id", SpectreStratum::submitID},
                        {"method", SpectreStratum::submit.method.c_str()},
                        {"params", {workerName,                                   // WORKER
                                    std::to_string(myJob["jobId"].get<uint64_t>()).c_str(), // JOB ID
                                    std::string(nonceStr.data()).c_str()}}}};

              // std::cout << "blob: " << hexStr(&WORK[0], SpectreX::INPUT_SIZE).c_str() << std::endl;
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
      ////  mutex.lock();
      std::cerr << "Error in POW Function" << std::endl;
      ////  mutex.unlock();
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}