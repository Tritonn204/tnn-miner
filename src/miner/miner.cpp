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

#if defined(_WIN32)
#include <Windows.h>
#else
#include "cpp-dns.hpp"
#include <sched.h>
#define THREAD_PRIORITY_ABOVE_NORMAL -5
#define THREAD_PRIORITY_HIGHEST -20
#define THREAD_PRIORITY_TIME_CRITICAL -20
#endif

#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/json.hpp>

#include <thread>
#include <mutex>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <miner.h>
#include <nlohmann/json.hpp>

#include <random>

#include <hex.h>
#include <pow.h>
// #include <astrobwtv3_cuda.cuh>
#include <powtest.h>
#include <thread>

#include <xelis-hash.hpp>

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

#include <pow_hip.h>

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
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

using json = nlohmann::json;

std::mutex mutex;
std::mutex wsMutex;

json job;
json devJob;
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

uint64_t hashrate;
uint64_t ourHeight = 0;
uint64_t devHeight = 0;
uint64_t difficulty;
uint64_t difficultyDev;

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

// Sends a WebSocket message and prints the response
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
  mutex.lock();
  setcolor(RED);
  std::cerr << '\n'
            << what << ": " << ec.message() << "\n";
  setcolor(BRIGHT_WHITE);
  mutex.unlock();
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

  // These objects perform our I/O
  int addrCount = 0;
  // bool resolved = false;

  net::ip::address ip_address;

  websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(ioc, ctx);

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

  std::thread submission_thread([&]
                                  {
      bool *C = isDev ? &isConnected : &devConnected;
      bool *B = isDev ? &submittingDev : &submitting;
      while (true) {
        try{
          if (!(*C)) break;
          if (*B) {
              boost::json::object *S = isDev ? &devShare : &share;
              std::string msg = boost::json::serialize(*S);

              // Acquire a lock before writing to the WebSocket
              ws.async_write(boost::asio::buffer(msg), [&](const boost::system::error_code& ec, std::size_t) {
                  if (ec) {
                      setcolor(RED);
                      printf("submission error\n");
                      setcolor(BRIGHT_WHITE);
                  }
              });
              (*B) = false;
          }
        } catch(...) {}

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      } });

  while (true)
  {
    try
    {
      buffer.clear();
      workInfo.str("");
      workInfo.clear();

      bool *B = isDev ? &submittingDev : &submitting;

      // if (*B)
      // {
      //   boost::json::object *S = isDev ? &devShare : &share;
      //   std::string msg = boost::json::serialize(*S);
      //   // mutex.lock();
      //   // std::cout << msg;
      //   // mutex.unlock();
      //   ws.async_write(boost::asio::buffer(msg), yield[ec]);
      //   if (ec)
      //   {
      //     return fail(ec, "async_write");
      //   }
      //   *B = false;
      // }

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
            // mutex.lock();
            if (isDev)
              devJob = workData;
            else
              job = workData;
            json *J = isDev ? &devJob : &job;
            // mutex.unlock();

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
                mutex.lock();
                setcolor(BRIGHT_YELLOW);
                printf("Mining at: %s/ws/%s\n", host.c_str(), wallet.c_str());
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
              difficultyDev = (*J).at("difficultyuint64");
              devBlob = std::string((*J).at("blockhashing_blob"));
              devHeight = (*J).at("height");
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
        return fail(ec, "async_read");
      }
    }
    catch (...)
    {
      setcolor(RED);
      std::cout << "ws error\n";
      setcolor(BRIGHT_WHITE);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(125));
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

  // These objects perform our I/O
  int addrCount = 0;

  net::ip::address ip_address;

  websocket::stream<beast::tcp_stream> ws(ioc);

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
          if (response.contains("new_job"))
          {
            json workData = response.at("new_job");
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
                  mutex.lock();
                  setcolor(BRIGHT_YELLOW);
                  printf("Mining at: %s/getwork/%s/%s\n", host.c_str(), wallet.c_str(), worker.c_str());
                  setcolor(CYAN);
                  printf("Dev fee: %.2f", devFee);
                  std::cout << "%" << std::endl;
                  setcolor(BRIGHT_WHITE);
                  mutex.unlock();
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
                  mutex.lock();
                  setcolor(CYAN);
                  printf("Connected to dev node: %s\n", host.c_str());
                  setcolor(BRIGHT_WHITE);
                  mutex.unlock();
                }
                devConnected = true;
                jobCounter++;
              }
            }
          }
          else
          {
            if (response.contains("block_rejected"))
            {
              rejected++;
            }
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
    std::this_thread::sleep_for(std::chrono::milliseconds(125));
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

  // SSL_CTX_set_info_callback(ctx.native_handle(), openssl_log_callback);

  // These objects perform our I/O
  int addrCount = 0;
  // bool resolved = false;
  // Create a TCP socket
  ctx.set_verify_mode(ssl::verify_none); // Accept self-signed certificates
  tcp::socket socket(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);

  net::ip::address ip_address;

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
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(stream).async_connect(daemon, yield[ec]);
  if (ec)
    return fail(ec, "connect");

#else
  // Look up the domain name
  tcp::resolver resolver(ioc);
  auto const results = resolver.async_resolve(host, port, yield[ec]);
  if (ec)
    return fail(ec, "resolve");

  // Set a timeout on the operation
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  auto daemon = beast::get_lowest_layer(stream).async_connect(results, yield[ec]);
  if (ec)
    return fail(ec, "connect");
#endif

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
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() 
        - Xatum::lastReceivedJobTime > Xatum::jobTimeout) {
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
        boost::asio::async_write(stream, boost::asio::buffer(msg), [&](const boost::system::error_code &ec, std::size_t)
                                 {
                      if (ec) {
                          bool *C = isDev ? &devConnected : &isConnected;
                          (*C) = false;
                          return fail(ec, "Xatum submission error");
                      } });
        (*B) = false;
      }
      boost::asio::streambuf response;
      std::stringstream workInfo;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec && trans > 0)
        return fail(ec, "Xatum async_read_until");

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
    std::this_thread::sleep_for(std::chrono::milliseconds(125));
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
    mutex.lock();
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
      printf("Xatum INFO: ");
      break;
    }

    printf("%s\n", data.at("msg").get<std::string>().c_str());

    setcolor(BRIGHT_WHITE);
    mutex.unlock();
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
        mutex.lock();
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
        setcolor(CYAN);
        printf("Dev fee: %.2f", devFee);
        std::cout << "%" << std::endl;
        setcolor(BRIGHT_WHITE);
        mutex.unlock();
      }
      else
      {
        mutex.lock();
        setcolor(CYAN);
        printf("Connected to dev node: %s\n", host.c_str());
        setcolor(BRIGHT_WHITE);
        mutex.unlock();
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
      printf("accepted!");
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
      // TODO
      break;
    }
    break;
  }
  }
}

//------------------------------------------------------------------------------

int main(int argc, char **argv)
{
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
#endif
  setcolor(RED);
  printf(TNN);
  setcolor(BRIGHT_WHITE);
  std::this_thread::sleep_for(std::chrono::seconds(1));
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  HANDLE hSelfToken = NULL;

  ::OpenProcessToken(::GetCurrentProcess(), TOKEN_ALL_ACCESS, &hSelfToken);
  if (SetPrivilege(hSelfToken, SE_LOCK_MEMORY_NAME, true))
    std::cout << "Permission Granted for Huge Pages!" << std::endl;
  else
    std::cout << "Huge Pages: Permission Failed..." << std::endl;

  SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
  // Check command line arguments.
  lookup2D_global = (uint16_t *)malloc_huge_pages(regOps_size*(256*256)*sizeof(uint16_t));
  lookup3D_global = (byte *)malloc_huge_pages(branchedOps_size*(256*256)*sizeof(byte));

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
    std::this_thread::sleep_for(std::chrono::seconds(1));
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

  if (vm.count("xatum"))
  {
    protocol = XELIS_XATUM;
  }

  if (vm.count("testnet"))
  {
    devSelection = testDevWallet;
  }

  if (vm.count("xelis-test"))
  {
    xelis_runTests();
    return 0;
  }

  if (vm.count("xelis-bench"))
  {
    std::thread t(xelis_benchmark_cpu_hash);
    t.join();
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
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return 1;
      }
    }
    catch (...)
    {
      printf("ERROR: invalid dev fee parameter... format should be for example '1.0'");
      std::this_thread::sleep_for(std::chrono::seconds(1));
      return 1;
    }
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
    goto Testing;
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
Testing:
{
  Num diffTest("20000", 10);

  if (testOp >= 0)
  {
    if (testLen >= 0)
    {
      runOpTests(testOp, testLen);
    }
    else
    {
      runOpTests(testOp);
    }
  }
  TestAstroBWTv3();
  TestAstroBWTv3_hip();
  // TestAstroBWTv3repeattest();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  return 0;
}
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

  std::thread GETWORK(getWork, false, miningAlgo);

  winMask = std::max(1, winMask);

  // Create worker threads and set CPU affinity
  for (int i = 0; i < threads; i++)
  {
    std::thread t(benchmark, i + 1);

    mutex.lock();
    std::cout << "(Benchmark) Worker " << i + 1 << " created" << std::endl;
    mutex.unlock();
  }

  while (!isConnected)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  auto start_time = std::chrono::steady_clock::now();
  startBenchmark = true;

  std::thread t2(logSeconds, start_time, bench_duration, &stopBenchmark);

  while (true)
  {
    auto now = std::chrono::steady_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    if (milliseconds >= bench_duration * 1000)
    {
      stopBenchmark = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
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
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return 0;
}

Mining:
{
  mutex.lock();
  printSupported();
  mutex.unlock();

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
  std::thread GETWORK(getWork, false, miningAlgo);

  std::thread DEVWORK(getWork, true, miningAlgo);

  unsigned int n = std::thread::hardware_concurrency();
  int winMask = 0;
  for (int i = 0; i < n - 1; i++)
  {
    winMask += 1 << i;
  }

  winMask = std::max(1, winMask);

  // Create worker threads and set CPU affinity
  mutex.lock();
  if (gpuMine)
  {
    std::thread t(hipMine);
    t.detach();
    // continue;
  }
  else
  {
    for (int i = 0; i < threads; i++)
    {
      std::thread t(mine, i+1, miningAlgo);
      t.detach();
      
      std::cout << "Thread " << i + 1 << " started" << std::endl;
    }
    printf("after loop\n");
  }
  mutex.unlock();

  auto start_time = std::chrono::high_resolution_clock::now();
  if (broadcastStats) {
    std::thread BROADCAST(BroadcastServer::serverThread, &rate30sec, &accepted, &rejected, versionString);
  }

  while (!isConnected)
  {
    std::this_thread::yield();
  }

  std::thread reporter(update, start_time);

  while (true)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  return EXIT_SUCCESS;
}
}

void logSeconds(std::chrono::steady_clock::time_point start_time, int duration, bool *stop)
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
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
}

void update(std::chrono::steady_clock::time_point start_time)
{
  auto beginning = start_time;
  std::this_thread::yield();

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
    std::this_thread::sleep_for(std::chrono::milliseconds(125));
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
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  goto startReporting;
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
        WORKER = "tnn-dev";
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
        mutex.lock();
        setcolor(RED);
        std::cerr << "\nError establishing connections" << std::endl
                  << "Will try again in 10 seconds...\n\n";
        setcolor(BRIGHT_WHITE);
        mutex.unlock();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10000));
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
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    ioc.reset();
    goto connectionAttempt;
  }
  while (*B)
  {
    caughtDisconnect = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
  std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  ioc.reset();
  goto connectionAttempt;
}

void benchmark(int tid)
{

  byte work[MINIBLOCK_SIZE];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned short> dist(0, 255);
  std::array<uint8_t, 48> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                { return dist(gen); });
  std::memcpy(work, buf.data(), buf.size());

  std::this_thread::sleep_for(std::chrono::milliseconds(125));

  int64_t localJobCounter;

  int32_t i = 0;

  byte powHash[32];
  // byte powHash2[32];
  workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*worker);
  lookupGen(*worker, lookup2D_global, lookup3D_global);

  while (!isConnected)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
  }
}

void mineGPU(int tid, int algo)
{
  switch (algo)
  {
  case DERO_HASH:
    mineDero(tid);
  case XELIS_HASH:
    mineXelis(tid);
  }
}


void mineDero(int tid)
{
  byte work[MINIBLOCK_SIZE];

  byte random_buf[12];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned short> dist(0, 255);
  std::array<uint8_t, 12> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                { return dist(gen); });
  std::memcpy(random_buf, buf.data(), buf.size());

  std::this_thread::sleep_for(std::chrono::milliseconds(125));

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
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  while (true)
  {
    try
    {
      mutex.lock();
      json myJob = job;
      json myJobDev = devJob;
      localJobCounter = jobCounter;
      mutex.unlock();

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
        mutex.lock();
        std::cerr << "Unknown version, please check for updates: "
                  << "version" << (work[0] & 0x1f) << std::endl;
        mutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
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
            mutex.lock();
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n";
            setcolor(BRIGHT_WHITE);
            devShare = {
                {"jobid", myJobDev.at("jobid")},
                {"mbl_blob", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str()}};
            submittingDev = true;
            mutex.unlock();
          }
          else
          {
            mutex.lock();
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n";
            setcolor(BRIGHT_WHITE);
            share = {
                {"jobid", myJob.at("jobid")},
                {"mbl_blob", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str()}};
            submitting = true;
            mutex.unlock();
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
  alignas(32) byte work[XELIS_BYTES_ARRAY_INPUT] = {0};
  alignas(32) byte devWork[XELIS_BYTES_ARRAY_INPUT] = {0};
  alignas(32) byte FINALWORK[XELIS_BYTES_ARRAY_INPUT] = {0};

  alignas(32) workerData_xelis *worker = (workerData_xelis *)malloc_huge_pages(sizeof(workerData_xelis));
waitForJob:

  while (!isConnected)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  while (true)
  {
    try
    {
      mutex.lock();
      json myJob = job;
      json myJobDev = devJob;
      localJobCounter = jobCounter;

      mutex.unlock();

      // if (!myJob.contains("template")) continue;
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
          std::string b64 = base64::from_base64(myJob.at("template").get<std::string>());
          memcpy(b2, b64.data(), b64.size());
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
            std::string b64 = base64::from_base64(myJobDev.at("template").get<std::string>().c_str());
            memcpy(b2d, b64.data(), b64.size());
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
        cmpDiff = ConvertDifficultyToBig(DIFF, XELIS_HASH);

        uint64_t *nonce = devMine ? &i_dev : &i;
        (*nonce)++;

        // printf("nonce = %llu\n", *nonce);

        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        byte *nonceBytes = &WORK[40];
        uint64_t n = ((tid - 1) % (256 * 256 * 256)) | (*nonce << 24);
        memcpy(nonceBytes, &n, 8);

        if (littleEndian())
        {
          std::swap(nonceBytes[7], nonceBytes[0]);
          std::swap(nonceBytes[6], nonceBytes[1]);
          std::swap(nonceBytes[5], nonceBytes[2]);
          std::swap(nonceBytes[4], nonceBytes[3]);
        }

        if (localJobCounter != jobCounter || localOurHeight != ourHeight)
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

          std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          if (devMine)
          {
            mutex.lock();
            if (localJobCounter != jobCounter || localDevHeight != devHeight)
            {
              mutex.unlock();
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
            }
            submittingDev = true;
            mutex.unlock();
          }
          else
          {
            mutex.lock();
            if (localJobCounter != jobCounter || localOurHeight != ourHeight)
            {
              mutex.unlock();
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
            }
            submitting = true;
            mutex.unlock();
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
      mutex.lock();
      std::cerr << "Error in POW Function" << std::endl;
      mutex.unlock();
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}

void hipMine()
{
  printf("made it in\n");
  int GPUCount = 0;
  hipGetDeviceCount(&GPUCount);
  int GPUbound = GPUCount;

  if (GPUbound == 0)
  {
    setcolor(RED);
    std::cerr << "ERROR: No GPU with ROCm nor HIP compute capability could be found\n";
    setcolor(BRIGHT_WHITE);
    std::this_thread::sleep_for(std::chrono::seconds(20));
    return;
  }

  // checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpyToSymbol(bitTable_d, bitTable, sizeof(bitTable), 0, cudaMemcpyHostToDevice));

  byte random_buf[12];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned short> dist(0, 255);
  std::array<uint8_t, 12> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                { return dist(gen); });
  memcpy(random_buf, buf.data(), buf.size());

  int64_t localJobCounter;

  int *batchSizes = new int[GPUCount];

  hipError_t err;
  int HHA;
  err = hipMalloc((void **)&HHA, sizeof(int));

  if (err != hipSuccess)
  {
    printf("hipMalloc went wrong!\n");
    printf("Error: %s\n", hipGetErrorString(err));
  }

  for (int i = 0; i < GPUbound; i++)
  {
    printf("device: %d\n", i);
    hipSetDevice(i);
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    err = hipMemGetInfo(&freeBytes, &totalBytes);

    if (err != hipSuccess)
    {
      printf("hipMemGetInfo went wrong!\n");
      printf("Error: %s\n", hipGetErrorString(err));
    }

    batchSizes[i] = batchSize;

    if (batchSizes[i] == 0) batchSizes[i] = ((freeBytes / sizeof(workerData_hip)) * cudaMemNumerator) / cudaMemDenominator;
    // batchSizes[i] = std::min(batchSizes[i], 8912);
    printf("Free memory on GPU #%d: %ld MB\n", i, freeBytes/1000000);
    printf("batchSize on GPU #%d: %d\n", i, batchSizes[i]);
  }

  int workerArraySize = 0;
  int blobArraySize = 0;

  workerData_hip *workers_h;
  workerData_hip *hip_workers;

//   class workerData_hip_aos
// {
// public:
//   unsigned char sHash[32];
//   unsigned char sha_key[32];
//   unsigned char sData[MAXX + 64];

//   unsigned char counter[64];

//   SHA256_CTX_HIP sha256;
//   rc4_state key;

//   int32_t sa[MAXX];
  
//   alignas(32) byte branchedOps[branchedOps_size_hip];
//   alignas(32) byte regularOps[regOps_size_hip];

//   alignas(32) byte branched_idx[256];
//   alignas(32) byte reg_idx[256];

//   // Salsa20_cuda salsa20;

//   int bucket_A[256];
//   int bucket_B[256*256];
//   int M;

//   unsigned char step_3[256];

//   unsigned char *lookup3D;
//   uint16_t *lookup2D;

//   uint64_t random_switcher;

//   uint64_t lhash;
//   uint64_t prev_lhash;
//   uint64_t tries;

//   unsigned char op;
//   unsigned char pos1;
//   unsigned char pos2;
//   unsigned char t1;
//   unsigned char t2;

//   // Vars for split sais kernels
//   int *sais_C, *sais_B, *sais_D, *sais_RA, *sais_b;
//   int sais_i, sais_j, sais_m, sais_p, sais_q, sais_t_var, sais_name, sais_pidx = 0, sais_newfs;
//   int sais_c0, sais_c1;
//   unsigned int sais_flags;

//   unsigned char A;
//   uint32_t data_len;
// };

  byte *hip_output;
  byte *hip_work;
  byte *hip_devWork;

  for (int i = 0; i < GPUbound; i++)
  {
    workerArraySize += batchSizes[i];
  }

  workers_h = new workerData_hip[workerArraySize];

  byte *work = new byte[workerArraySize * MINIBLOCK_SIZE];
  byte *devWork = new byte[workerArraySize * MINIBLOCK_SIZE];
  byte *outputHashes = new byte[workerArraySize * 32];

  workerData *worker = (workerData *)malloc(sizeof(workerData));
  initWorker(*worker);
  lookupGen(*worker, lookup2D_global, lookup3D_global);
  
  for (int d = 0; d < GPUCount; d++) {
    hipSetDevice(d);
  }

  for (int d = 0; d < GPUbound; d++)
  {
    printf("device: %d\n", d);
    hipSetDevice(d);
    hipMalloc((void **)&hip_output, workerArraySize * 32);
    hipMalloc((void **)&hip_work, workerArraySize * MINIBLOCK_SIZE);
    hipMalloc((void **)&hip_devWork, workerArraySize * MINIBLOCK_SIZE);
    hipMalloc((void **)&hip_workers, sizeof(workerData_hip) * workerArraySize);
    // ASTRO_LOOKUPGEN_HIP(d, batchSizes[d], &hip_workers[hipWorkerStartIndexes[d]]);
    hipDeviceSynchronize();
  }

waitForJob:
  printf("After Lookupgen\n");
  while (!isConnected)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }

  std::string blobString;

  while (true)
  {
    mutex.lock();
    json myJob = job;
    json myJobDev = devJob;
    localJobCounter = jobCounter;
    mutex.unlock();

    byte *b2 = new byte[MINIBLOCK_SIZE];
    blobString = myJob.at("blockhashing_blob");
    hexstrToBytes(blobString, b2);
    for (int i = 0; i < workerArraySize; i++)
    {
      memcpy(&work[i * MINIBLOCK_SIZE], b2, MINIBLOCK_SIZE);
      memcpy(&work[i * MINIBLOCK_SIZE + MINIBLOCK_SIZE - 12], random_buf, 12);
    }
    delete[] b2;

    if (devConnected)
    {
      byte *b2d = new byte[MINIBLOCK_SIZE];
      blobString = myJobDev.at("blockhashing_blob");
      hexstrToBytes(blobString, b2d);
      for (int i = 0; i < workerArraySize; i++)
      {
        memcpy(&devWork[i * MINIBLOCK_SIZE], b2d, MINIBLOCK_SIZE);
        memcpy(&devWork[i * MINIBLOCK_SIZE + MINIBLOCK_SIZE - 12], random_buf, 12);
      }
      delete[] b2d;
    }

    for (int d = 0; d < GPUbound; d++)
    {
      hipSetDevice(d);
      hipMemset(hip_output, 0, workerArraySize * 32);
    }

    double which;
    bool devMine = false;
    bool submit = false;
    uint64_t DIFF;

    int nonce = 0;
    for (int d = 0; d < GPUbound; d++)
    {
      hipSetDevice(d);
      hipMemcpy(hip_work, work, workerArraySize * MINIBLOCK_SIZE, hipMemcpyHostToDevice);
      hipMemcpy(hip_devWork, devWork, workerArraySize * MINIBLOCK_SIZE, hipMemcpyHostToDevice);
    }

    Num cmpDiff;

    while (localJobCounter == jobCounter)
    {
      which = (double)(rand() % 10000);
      devMine = (devConnected && which < devFee * 100.0);
      DIFF = devMine ? difficultyDev : difficulty;

      // printf("Difficulty: %" PRIx64 "\n", DIFF);

      cmpDiff = ConvertDifficultyToBig(DIFF, DERO_HASH);

      byte *WORK = devMine ? hip_devWork : hip_work;

      for (int d = 0; d < GPUbound; d++)
      {
        hipSetDevice(d);
        ASTRO_INIT_HIP(d, WORK, batchSizes[d], 0, nonce);
        nonce += batchSizes[d];
      }

      for (int d = 0; d < GPUbound; d++)
      {
        hipSetDevice(d);
        hipDeviceSynchronize();
        if (localJobCounter != jobCounter)
          break;
      }
      if (localJobCounter != jobCounter)
        break;

      for (int d = 0; d < GPUbound; d++)
      {
        hipSetDevice(d);
        ASTRO_HIP(WORK, hip_output, hip_workers, MINIBLOCK_SIZE, batchSizes[d], 0, 0);
      }

      for (int d = 0; d < GPUbound; d++)
      {
        hipSetDevice(d);
        hipDeviceSynchronize();
        if (localJobCounter != jobCounter)
          break;
      }

      counter.store(counter + workerArraySize);

      if (localJobCounter != jobCounter)
        break;

      int dupes = 0;

      // for (int d = 0; d < GPUbound; d++)
      // {
      //   hipSetDevice(d);
      //   hipMemcpy(outputHashes, hip_output, workerArraySize * 32, hipMemcpyDeviceToHost);
      //   for (int i = 0; i < batchSizes[d]; i++) {
      //     byte* ref = &outputHashes[i*32];
      //     int refIndex = i;
      //     for (int j = 0; j < batchSizes[d]; j++) {
      //       if (j == refIndex) continue;
      //       byte* comp = &outputHashes[j*32];
      //       bool same = true;
      //       for (int k = 0; k < 32; k++) {
      //         if (ref[k] != comp[k]) {
      //           same = false;
      //           break;
      //         }
      //       }
      //       if (same) {
      //         dupes++;
      //         printf("Duplicate hash found!\n index A: %d, index B: %d\n, hash: %s\n", refIndex, j, hexStr(ref, 32).c_str());
      //         printf("work A: %s, work B: %s\n", hexStr(&work[refIndex*MINIBLOCK_SIZE], 48).c_str(), hexStr(&work[j*MINIBLOCK_SIZE], 48).c_str());

      //         workerData *W = (workerData *)malloc(sizeof(workerData));
      //         initWorker(*W);
      //         lookupGen(*W, lookup2D_global, lookup3D_global);
      //         byte res[32];
      //         byte res2[32];
      //         AstroBWTv3(&work[refIndex*MINIBLOCK_SIZE], MINIBLOCK_SIZE, res, *W, false);
      //         AstroBWTv3(&work[j*MINIBLOCK_SIZE], MINIBLOCK_SIZE, res2, *W, false);
      //         printf("hash validation A: %s, hash validation B: %s\n", hexStr(res, 32).c_str(), hexStr(res2, 32).c_str());
      //         break;
      //       }
      //     }
      //   }
      // }

      // for (int d = 0; d < GPUbound; d++)
      // {
      //   hipSetDevice(d);
      //   hipMemcpy(work, WORK, workerArraySize * MINIBLOCK_SIZE, hipMemcpyDeviceToHost);
      //   hipMemcpy(outputHashes, hip_output, workerArraySize * 32, hipMemcpyDeviceToHost);

      //   for (int i = 0; i < batchSizes[d]; i++)
      //   {
      //     workerData *W = (workerData *)malloc(sizeof(workerData));
      //     initWorker(*W);
      //     lookupGen(*W, lookup2D_global, lookup3D_global);
      //     byte *ref = &work[i * MINIBLOCK_SIZE];
      //     byte comp[32];

      //     printf("\n");

      //     AstroBWTv3(ref, MINIBLOCK_SIZE, comp, *W, false);

      //     bool same = true;
      //     for (int j = 0; j < 32; j++)
      //     {
      //       if (outputHashes[i * 32 + j] != comp[j])
      //       {
      //         same = false;
      //         break;
      //       }
      //     }
      //     if (!same)
      //       printf("invalid POW at index %d\n GPU: %s, CPU: %s\n", i, hexStr(&outputHashes[i * 32], 32).c_str(), hexStr(comp, 32).c_str());
      //   }
      // }

    //   if (!isConnected)
    //     break;

    //   counter.store(counter + workerArraySize - dupes);
      submit = devMine ? !submittingDev : !submitting;

      std::vector<std::string> existing(workerArraySize);

      for (int d = 0; d < GPUCount; d++)
      {
        hipSetDevice(d);
        hipMemcpy(work, WORK, workerArraySize * MINIBLOCK_SIZE, hipMemcpyDeviceToHost);
        hipMemcpy(outputHashes, hip_output, workerArraySize * 32, hipMemcpyDeviceToHost);
        // printf("demo: %s\n", hexStr(outputHashes, 128).c_str());
        for (int h = 0; h < batchSizes[d]; h++)
        {
          // byte *tester = &outputHashes[hipWorkerStartIndexes[d] * 32 + h * 32];
          // if (tester[31] == 0 && tester[30] <= 1)
          //   printf("should be valid\n hash: %s", hexStr(tester, 32).c_str());
          // std::string H = hexStr(&outputHashes[h*32], 32);
          // std::string W = hexStr(&work[h*MINIBLOCK_SIZE], MINIBLOCK_SIZE);
          // if (std::find(existing.begin(), existing.end(), H) != existing.end()) printf("DUPLICATE at pos: %d\n", h);
          // else existing.push_back(H);
          // if (submit && CheckHash(&outputHashes[h*32], cmpDiff))
          // {
          //   // printf("work: %s, hash: %s\n", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str(), hexStr(powHash, 32).c_str());
          //   if (devMine)
          //   {
          //     mutex.lock();
          //     submittingDev = true;
          //     setcolor(CYAN);
          //     std::cout << "\n(DEV) GPU " << d << " found a dev share\n";
          //     setcolor(BRIGHT_WHITE);
          //     mutex.unlock();
          //     devShare = {
          //         {"jobid", myJobDev.at("jobid")},
          //         {"mbl_blob", hexStr(&work[h*MINIBLOCK_SIZE], MINIBLOCK_SIZE).c_str()}};
          //   }
          //   else
          //   {
          //     mutex.lock();
          //     submitting = true;
          //     setcolor(BRIGHT_YELLOW);
          //     std::cout << "\nGPU " << d << " found a nonce!\n";
          //     setcolor(BRIGHT_WHITE);
          //     mutex.unlock();
          //     share = {
          //         {"jobid", myJob.at("jobid")},
          //         {"mbl_blob", hexStr(&work[h*MINIBLOCK_SIZE], MINIBLOCK_SIZE).c_str()}};
          //   }
          //   for(;;) {
          //     if(((devMine && !submittingDev) || (!devMine && !submitting)) || localJobCounter != jobCounter) {
          //       break;
          //     }
          //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
          //   }
          //   if (localJobCounter != jobCounter) break;
          // }
        }
        if (localJobCounter != jobCounter) break;
      }
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}