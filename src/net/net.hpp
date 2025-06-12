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
extern boost::mutex devMutex;
extern boost::mutex userMutex;
extern bool data_ready;

extern boost::mutex wsMutex;

tcp::endpoint resolve_host(boost::mutex &wsMutex, net::io_context &ioc, net::yield_context yield, std::string host, std::string port);

inline void setForDisconnected(bool *connectedPtr, bool *submitPtr, bool *abortPtr, bool *dataReadyPtr, boost::condition_variable *cvPtr) {
  if (connectedPtr != nullptr) *connectedPtr = false;
  if (submitPtr != nullptr)    *submitPtr = false;
  if (abortPtr != nullptr)     *abortPtr = true;
  if (dataReadyPtr != nullptr) *dataReadyPtr = true;
  if (cvPtr != nullptr)        cvPtr->notify_all();
}

// Report a failure
inline void fail(beast::error_code ec, char const *where) noexcept
{
  // mutex.lock();
  setcolor(RED);
  std::cerr << '\n'
            << where << ": " << ec.message() << "\n";
  setcolor(BRIGHT_WHITE);
  // mutex.unlock();
}

inline void fail(char const *where, char const *why) noexcept
{
  // mutex.lock();
  setcolor(RED);
  std::cerr << '\n'
            << where << ": " << why << "\n";
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

inline void do_session_v2(
  MiningProfile *miningProf,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield)
{
  // Dirker TODO: Should this be switching based off Coin first?  Then protocol?
  // Is algo even the right thing to begin with?!?
  bool use_ssl = miningProf->transportLayer.find("wss", 0) != std::string::npos;
  use_ssl |= miningProf->transportLayer.find("ssl", 0) != std::string::npos;
  switch (miningProf->coin.miningAlgo)
  {
  #ifdef TNN_ASTROBWTV3
  case ALGO_ASTROBWTV3:
    dero_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
    break;
  #endif
  #ifdef TNN_XELISHASH
  case ALGO_XELISV2:
  {
    switch (miningProf->protocol)
    {
    case PROTO_XELIS_SOLO:
      xelis_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, yield, miningProf->isDev);
      break;
    case PROTO_XELIS_XATUM:
      xatum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
      break;
    case PROTO_XELIS_STRATUM:
    {
      if(use_ssl) {
        xelis_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
      } else {
        xelis_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
      }
      break;
    }
    }
    break;
  }
  #endif
  #ifdef TNN_ASTROBWTV3
  case ALGO_SPECTRE_X:
    switch (miningProf->protocol)
    {
      case PROTO_SPECTRE_SOLO:
        break;
      case PROTO_SPECTRE_STRATUM:
        spectre_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        break;
    }
    break;
  #endif
  #ifdef TNN_RANDOMX
  case ALGO_RX0:
  {
    switch (miningProf->protocol)
    {
      case PROTO_RX0_SOLO:
        rx0_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->isDev);
        break;
      case PROTO_RX0_STRATUM:
      {
        if(use_ssl) {
          rx0_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        } else {
          rx0_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        }
        break;
      }
    }
    break;
  }
  #endif
  #ifdef TNN_VERUSHASH
  case ALGO_VERUS:
  {
    switch (miningProf->protocol)
    {
      case PROTO_VERUS_SOLO:
        break;
      case PROTO_VERUS_STRATUM:
      {
        // if (use_ssl) {

        // } else {
          verus_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        // }
        break;
      }
    }
    break;
  }
  #endif
  #ifdef TNN_ASTRIXHASH
  case ALGO_ASTRIX_HASH:
    switch (miningProf->protocol)
    {
      case PROTO_KAS_SOLO:
        kas_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->isDev);
        break;
      case PROTO_KAS_STRATUM:
        kas_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        break;
    }
  #endif
  #ifdef TNN_NXLHASH
  case ALGO_NXL_HASH:
    switch (miningProf->protocol)
    {
      case PROTO_KAS_SOLO:
        kas_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->isDev);
        break;
      case PROTO_KAS_STRATUM:
        kas_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        break;
    }
  #endif
  #ifdef TNN_HOOHASH
  case ALGO_HOOHASH:
    switch (miningProf->protocol)
    {
      case PROTO_KAS_SOLO:
        kas_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->isDev);
        break;
      case PROTO_KAS_STRATUM:
        kas_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        break;
    }
  #endif
  #ifdef TNN_WALAHASH
  case ALGO_WALA_HASH:
    switch (miningProf->protocol)
    {
      case PROTO_KAS_SOLO:
        kas_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->isDev);
        break;
      case PROTO_KAS_STRATUM:
        kas_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        break;
    }
  #endif
  #ifdef TNN_SHAIHIVE
  case ALGO_SHAI_HIVE:
    shai_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
    break;
  #endif
  #ifdef TNN_YESPOWER
  case ALGO_YESPOWER:
    switch (miningProf->protocol)
    {
      case PROTO_BTC_STRATUM:
        if(use_ssl) {
          btc_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        } else {
          btc_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        }
        break;
    }
    break;
  #endif
  #ifdef TNN_RINHASH
  case ALGO_RINHASH:
    switch (miningProf->protocol)
    {
      case PROTO_BTC_STRATUM:
        if(use_ssl) {
          btc_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        } else {
          btc_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        }
        break;
    }
    break;
  #endif
  }
}