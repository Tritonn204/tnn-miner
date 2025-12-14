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
#include <atomic>

#include "sessions.hpp"

#include "terminal.hpp"
#include "DNSResolver.hpp"

using tcp = boost::asio::ip::tcp;
namespace beast = boost::beast;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;

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

// Helper trait to set values uniformly for both bool and atomic<bool>
namespace detail {
    template<typename T>
    inline void safe_set(T* ptr, bool value) {
        if (ptr != nullptr) *ptr = value;
    }
    
    template<>
    inline void safe_set<std::atomic<bool>>(std::atomic<bool>* ptr, bool value) {
        if (ptr != nullptr) ptr->store(value, std::memory_order_release);
    }
}

// Templated version supporting both bool* and std::atomic<bool>* for abort
template<typename AbortType = bool>
inline void setForDisconnected(
    bool *connectedPtr, 
    bool *submitPtr, 
    AbortType *abortPtr, 
    bool *dataReadyPtr, 
    boost::condition_variable *cvPtr) 
{
    if (connectedPtr != nullptr) *connectedPtr = false;
    if (submitPtr != nullptr)    *submitPtr = false;
    detail::safe_set(abortPtr, true);
    if (dataReadyPtr != nullptr) *dataReadyPtr = true;
    if (cvPtr != nullptr)        cvPtr->notify_all();
}

// Overload for when abort is handled separately (pass nullptr)
inline void setForDisconnectedNoAbort(
    bool *connectedPtr, 
    bool *submitPtr,
    bool *dataReadyPtr, 
    boost::condition_variable *cvPtr) 
{
    if (connectedPtr != nullptr) *connectedPtr = false;
    if (submitPtr != nullptr)    *submitPtr = false;
    if (dataReadyPtr != nullptr) *dataReadyPtr = true;
    if (cvPtr != nullptr)        cvPtr->notify_all();
}

// Report a failure
inline void fail(beast::error_code ec, char const *where) noexcept
{
    setcolor(RED);
    std::cerr << '\n' << where << ": " << ec.message() << "\n";
    setcolor(BRIGHT_WHITE);
    std::cerr.flush();
}

inline void fail(char const *where, char const *why) noexcept
{
    setcolor(RED);
    std::cerr << '\n' << where << ": " << why << "\n";
    setcolor(BRIGHT_WHITE);
    std::cerr.flush();
}

inline tcp::endpoint resolve_host(
    boost::mutex &wsMutex, 
    net::io_context &ioc, 
    net::yield_context yield, 
    std::string host, 
    std::string port) 
{
    beast::error_code ec;
    int addrCount = 0;
    net::ip::address ip_address;

#if !defined(_WIN32) && !defined(__APPLE__)
    // Check if host is already an IP address
    boost::asio::ip::address::from_string(host, ec);
    if (ec)
    {
        // Need to resolve hostname
        // Using cpp-dns to circumvent issues with static linking and getaddrinfo()
        net::io_context ioc2;
        std::string ip;
        std::promise<void> p;

        YukiWorkshop::DNSResolver d(ioc2);
        d.resolve_a4(host, [&](int err, auto &addrs, auto &qname, auto &cname, uint ttl)
        {
            if (!err) {
                for (auto &it : addrs) {
                    addrCount++;
                    ip = it.to_string();
                }
            }
            p.set_value();
        });
        ioc2.run();

        std::future<void> f = p.get_future();
        f.get();

        if (addrCount == 0)
        {
            setcolor(RED);
            std::cerr << "ERROR: Could not resolve " << host << std::endl;
            setcolor(BRIGHT_WHITE);
            // Return a default endpoint - caller should check for errors
            return tcp::endpoint();
        }

        ip_address = net::ip::address::from_string(ip, ec);
        if (ec) {
            setcolor(RED);
            std::cerr << "ERROR: Invalid IP address from DNS: " << ip << std::endl;
            setcolor(BRIGHT_WHITE);
            return tcp::endpoint();
        }
    }
    else
    {
        ip_address = net::ip::address::from_string(host, ec);
    }

    tcp::endpoint result(ip_address, static_cast<uint16_t>(std::stoi(port)));
    return result;
#else
    // Use standard resolver on Windows and macOS
    tcp::resolver resolver(ioc);
    auto const results = resolver.async_resolve(host, port, yield[ec]);
    if (ec) {
        fail(ec, "resolve");
        return tcp::endpoint();
    }

    return results.begin()->endpoint();
#endif
}

inline void do_session_v2(
    MiningProfile *miningProf,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield)
{
    bool use_ssl = miningProf->transportLayer.find("wss", 0) != std::string::npos;
    use_ssl |= miningProf->transportLayer.find("ssl", 0) != std::string::npos;
    
    switch (miningProf->coin.miningAlgo)
    {
#ifdef TNN_ASTROBWTV3
    case ALGO_ASTROBWTV3:
        dero_session(miningProf->host, miningProf->port, miningProf->wallet, 
                     miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        break;
#endif

#ifdef TNN_XELISHASH
    case ALGO_XELISV2:
    case ALGO_XELISV3:
    {
        switch (miningProf->protocol)
        {
        case PROTO_XELIS_SOLO:
            xelis_session(miningProf->host, miningProf->port, miningProf->wallet, 
                          miningProf->workerName, ioc, yield, miningProf->isDev);
            break;
        case PROTO_XELIS_XATUM:
            xatum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                          miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            break;
        case PROTO_XELIS_STRATUM:
            if (use_ssl) {
                xelis_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                      miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            } else {
                xelis_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, 
                                            miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            }
            break;
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
            if (use_ssl) {
                spectre_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                        miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            } else {
                spectre_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, 
                                              miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            }
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
            if (use_ssl) {
                rx0_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                    miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            } else {
                rx0_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, 
                                          miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            }
            break;
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
            verus_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                  miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            break;
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
            if (use_ssl) {
                kas_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                    miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            } else {
                kas_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, 
                                          miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            }
            break;
        }
        break;
#endif

#ifdef TNN_NXLHASH
    case ALGO_NXL_HASH:
        switch (miningProf->protocol)
        {
        case PROTO_KAS_SOLO:
            kas_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->isDev);
            break;
        case PROTO_KAS_STRATUM:
            if (use_ssl) {
                kas_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                    miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            } else {
                kas_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, 
                                          miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            }
            break;
        }
        break;
#endif

#ifdef TNN_HOOHASH
    case ALGO_HOOHASH:
        switch (miningProf->protocol)
        {
        case PROTO_KAS_SOLO:
            kas_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->isDev);
            break;
        case PROTO_KAS_STRATUM:
            if (use_ssl) {
                kas_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                    miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            } else {
                kas_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, 
                                          miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            }
            break;
        }
        break;
#endif

#ifdef TNN_WALAHASH
    case ALGO_WALA_HASH:
        switch (miningProf->protocol)
        {
        case PROTO_KAS_SOLO:
            kas_session(miningProf->host, miningProf->port, miningProf->wallet, miningProf->isDev);
            break;
        case PROTO_KAS_STRATUM:
            if (use_ssl) {
                kas_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                    miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            } else {
                kas_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, 
                                          miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            }
            break;
        }
        break;
#endif

#ifdef TNN_SHAIHIVE
    case ALGO_SHAI_HIVE:
        shai_session(miningProf->host, miningProf->port, miningProf->wallet, 
                     miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
        break;
#endif

#ifdef TNN_YESPOWER
    case ALGO_YESPOWER:
        switch (miningProf->protocol)
        {
        case PROTO_BTC_STRATUM:
            if (use_ssl) {
                btc_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                    miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            } else {
                btc_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, 
                                          miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
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
            if (use_ssl) {
                btc_stratum_session(miningProf->host, miningProf->port, miningProf->wallet, 
                                    miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            } else {
                btc_stratum_session_nossl(miningProf->host, miningProf->port, miningProf->wallet, 
                                          miningProf->workerName, ioc, ctx, yield, miningProf->isDev);
            }
            break;
        }
        break;
#endif

    default:
        setcolor(RED);
        std::cerr << "Unknown mining algorithm: " << miningProf->coin.miningAlgo << std::endl;
        setcolor(BRIGHT_WHITE);
        break;
    }
}