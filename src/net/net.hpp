#pragma once

#include "tnn-common.hpp"

#include <boost/asio.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <boost/beast.hpp>
#include <boost/beast/core.hpp>
#include <boost/json.hpp>

#include <boost/thread.hpp>
#include <iostream>

#include "sessions.hpp"

//#include "miner.h"

#include "terminal.h"
#include "DNSResolver.hpp"

//namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>
namespace beast = boost::beast;         // from <boost/beast.hpp>
//namespace http = beast::http;           // from <boost/beast/http.hpp>
//namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>

extern boost::json::value job;
extern boost::json::value devJob;

extern std::string currentBlob;
extern std::string devBlob;

extern boost::json::object share;
extern boost::json::object devShare;

extern bool submitting;
extern bool submittingDev;

extern boost::condition_variable cv;
extern boost::mutex mutex;
extern bool data_ready;

extern boost::mutex wsMutex;

tcp::endpoint resolve_host(boost::mutex &wsMutex, net::io_context &ioc, net::yield_context yield, std::string host, std::string port);

extern void do_session(std::string hostProto, std::string host, std::string const &port, std::string const &wallet, std::string const &worker, int algo, net::io_context &ioc, ssl::context &ctx, net::yield_context yield, bool isDev);

inline void setForDisconnected(bool *connectedPtr, bool *submitPtr, bool *abortPtr, bool *dataReadyPtr, boost::condition_variable *cvPtr) {
  if (connectedPtr != nullptr) *connectedPtr = false;
  if (submitPtr != nullptr)    *submitPtr = false;
  if (abortPtr != nullptr)     *abortPtr = true;
  if (dataReadyPtr != nullptr) *dataReadyPtr = true;
  if (cvPtr != nullptr)        cvPtr->notify_all();
}

// Report a failure
inline void fail(beast::error_code ec, char const *what) noexcept
{
  // mutex.lock();
  setcolor(RED);
  std::cerr << '\n'
            << what << ": " << ec.message() << "\n";
  setcolor(BRIGHT_WHITE);
  // mutex.unlock();
}

inline tcp::endpoint resolve_host(boost::mutex &wsMutex, net::io_context &ioc, net::yield_context yield, std::string host, std::string port) {
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
          // mutex.lock();
          for (auto &it : addrs) {
            addrCount++;
            ip = it.to_string();
          }
          p.set_value();
          // mutex.unlock();
      } else {
        p.set_value();
      }
    });
    ioc2.run();

    std::future<void> f = p.get_future();
    f.get();

    if (addrCount == 0)
    {
      // mutex.lock();
      setcolor(RED);
      std::cerr << "ERROR: Could not resolve " << host << std::endl;
      setcolor(BRIGHT_WHITE);
      // mutex.unlock();
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

// Session selector
inline void do_session(
    std::string hostProto,
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
  bool use_ssl = (hostProto.find("ssl") != std::string::npos);
  switch (algo)
  {
  case DERO_HASH:
    dero_session(hostProto, host, port, wallet, worker, ioc, ctx, yield, isDev);
    break;
  case XELIS_HASH:
  {
    switch (protocol)
    {
    case XELIS_SOLO:
      xelis_session(hostProto, host, port, wallet, worker, ioc, yield, isDev);
      break;
    case XELIS_XATUM:
      xatum_session(hostProto, host, port, wallet, worker, ioc, ctx, yield, isDev);
      break;
    case XELIS_STRATUM:
    {
      if(use_ssl) {
        xelis_stratum_session(hostProto, host, port, wallet, worker, ioc, ctx, yield, isDev);
      } else {
        xelis_stratum_session_nossl(hostProto, host, port, wallet, worker, ioc, ctx, yield, isDev);
      }
      break;
    }
    }
    break;
  }
  case SPECTRE_X:
    spectre_stratum_session(hostProto, host, port, wallet, worker, ioc, ctx, yield, isDev);
    break;
  }
}