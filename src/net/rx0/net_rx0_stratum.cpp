#include "../net.hpp"
#include <hex.h>

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

#include <stratum/stratum.h>
#include "rx0_jobCache.hpp"

#include <randomx/randomx.h>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

std::atomic<bool> randomx_ready = false;
std::atomic<bool> randomx_ready_dev = false;

std::atomic<bool> needsDatasetUpdate = false;

boost::mutex dsMutex;

uint64_t diff_numerator = boost_swap_impl::stoull("0x100000001", nullptr, 16);

static uint64_t rx_targetToDifficulty(const char* target) {
  uint32_t targetInt = boost_swap_impl::stoul(target, nullptr, 16);
  targetInt = __builtin_bswap32(targetInt);
  uint64_t diff = diff_numerator / targetInt;

  fflush(stdout);
  return diff;
}

int handleRandomXStratumPacket(boost::json::object packet, bool isDev) {
  std::string M = packet["method"].as_string().c_str();
  if (M.compare(rx0Stratum::s_job) == 0) {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    if (!packet["error"].is_null()) return 1;

    boost::json::object newJob = packet["params"].as_object();

    setcolor(isDev ? CYAN : BRIGHT_WHITE);
    if (!isDev)
      printf("\nStratum: new job received\n");
    else
      printf("\nDEV Stratum: new job received\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    rx0Stratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    boost::json::value *JV = isDev ? &devJob : &job;

    int64_t *h = isDev ? &devHeight : &ourHeight;

    if (!isDev) difficulty = rx_targetToDifficulty(newJob.at("target").as_string().c_str());
    else difficultyDev = rx_targetToDifficulty(newJob.at("target").as_string().c_str());
    
    // No need to reference different keys - updateVM handles cache management internally
    updateVM(newJob, isDev);

    (*JV) = newJob;

    (*h)++;
    jobCounter++;
  }
  else if (M.compare(rx0Stratum::s_print) == 0) {
    int lLevel = packet.at("params").as_array()[0].to_number<int64_t>();
    if (lLevel != rx0Stratum::STRATUM_DEBUG) {
      int res = 0;
      printf("\n");
      if (isDev) {
        setcolor(CYAN);
        printf("DEV | ");
      }

      switch (lLevel) {
      case rx0Stratum::STRATUM_INFO:
        if (!isDev)
          setcolor(BRIGHT_WHITE);
        printf("Stratum INFO: ");
        break;
      case rx0Stratum::STRATUM_WARN:
        if (!isDev)
          setcolor(BRIGHT_YELLOW);
        printf("Stratum WARNING: ");
        break;
      case rx0Stratum::STRATUM_ERROR:
        if (!isDev)
          setcolor(RED);
        printf("Stratum ERROR: ");
        res = -1;
        break;
      case rx0Stratum::STRATUM_DEBUG:
        break;
      }
      printf("%s\n", packet.at("params").as_array()[1].as_string().c_str());

      fflush(stdout);
      setcolor(BRIGHT_WHITE);

      return res;
    }
  }
  return 0;
}

int handleRandomXStratumResponse(boost::json::object packet, bool isDev)
{
  // if (!isDev) {
  // if (!packet.contains("id")) return 0;
  int64_t id = packet["id"].to_number<int64_t>();
  switch (id)
  {
  case rx0Stratum::loginID:
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    boost::json::value *JV = isDev ? &devJob : &job;

    if (!packet["error"].is_null())
    {
      setcolor(RED);
      printf("Stratum Error: %s\n", packet["error"].as_object()["message"].as_string().c_str());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return 1;
    }

    boost::json::object res = packet["result"].as_object();
    boost::json::object newJob = res["job"].as_object();

    std::string &l_ID = isDev ? randomx_login_dev : randomx_login;
    l_ID = res.at("id").as_string().c_str();

    (*JV) = newJob;

    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
        difficulty = rx_targetToDifficulty(newJob.at("target").as_string().c_str());
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", miningProfile.host.c_str(), miningProfile.wallet.c_str());
        fflush(stdout);
        setcolor(CYAN);
        printf("Dev fee: %.2f%% of your total hashrate\n", devFee);

        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        setcolor(CYAN);
        printf("Connected to dev node");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }

    updateVM(newJob, isDev);

    *C = true;
  }
  break;
  case rx0Stratum::submitID:
  {
    printf("\n");
    if (isDev)
    {
      setcolor(CYAN);
      printf("DEV | ");
    }

    bool shareAccepted = false;
    std::string errorMessage;

    // Check if there's an error first
    if (!packet["error"].is_null())
    {
      // Share was rejected due to error
      shareAccepted = false;
      auto errorObj = packet["error"].as_object();
      if (errorObj.contains("message"))
      {
        errorMessage = errorObj["message"].as_string().c_str();
      }
      else
      {
        errorMessage = "Unknown error";
      }
    }
    else if (!packet["result"].is_null())
    {
      // Check what type the result is
      auto &result = packet["result"];

      if (result.is_bool())
      {
        // Standard boolean response
        shareAccepted = result.as_bool();
      }
      else if (result.is_object())
      {
        // Object response like {"status":"OK"}
        auto resultObj = result.as_object();
        if (resultObj.contains("status"))
        {
          std::string status = resultObj["status"].as_string().c_str();
          shareAccepted = (status == "OK" || status == "ok" || status == "accepted");
        }
        else
        {
          // If it's an object without error, assume accepted
          shareAccepted = true;
        }
      }
      else if (result.is_string())
      {
        // Some pools might return string responses
        std::string resultStr = result.as_string().c_str();
        shareAccepted = (resultStr == "OK" || resultStr == "ok" || resultStr == "accepted");
      }
      else
      {
        shareAccepted = true;
      }
    }
    else
    {
      // Both result and error are null - this shouldn't happen, treat as rejected
      shareAccepted = false;
      errorMessage = "Invalid response: both result and error are null";
    }

    // Handle the result
    if (shareAccepted)
    {
      accepted += !isDev;
      if (!isDev) setcolor(BRIGHT_YELLOW);
      std::cout << "Stratum share accepted" << std::endl;
    }
    else
    {
      rejected += !isDev;
      if (!isDev)
        setcolor(RED);
      std::cout << "Stratum share rejected";
      if (!errorMessage.empty())
      {
        std::cout << ": " << errorMessage;
      }
      std::cout << std::endl;
    }

    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    break;
  }
  }
  return 0;
}

void rx0_stratum_session(
    std::string sessionHost,
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

    auto endpoint = resolve_host(wsMutex, ioc, yield, sessionHost, port);
    
    // SSL stream instead of TCP stream
    boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);

    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
    if (ec)
        return fail(ec, "connect");

    // SSL handshake
    stream.async_handshake(ssl::stream_base::client, yield[ec]);
    if (ec)
        return fail(ec, "handshake");

    boost::json::object packet = rx0Stratum::stratumCall;
    packet.at("id") = rx0Stratum::login.id;
    packet.at("method") = rx0Stratum::login.method;

    std::string userAgent = "tnn-miner/" + std::string(versionString);

    boost::json::object loginParams = {
        {"login", wallet},
        {"pass", stratumPassword},
        {"rigid", worker},
        {"agent", userAgent.c_str()}
    };

    packet.at("params") = loginParams;
    std::string login = boost::json::serialize(packet) + "\n";

    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    size_t trans = boost::asio::async_write(stream, boost::asio::buffer(login), yield[ec]);
    if (ec)
        return fail(ec, "Stratum login");

    rx0Stratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Add persistent packet buffer for handling split packets
    std::string packetBuffer;

    bool submitThread = false;
    bool abort = false;

    // Fixed submit thread pattern
    boost::thread subThread([&](){
        submitThread = true;
        while(!abort) {
            boost::unique_lock<boost::mutex> lock(mutex);
            bool *B = isDev ? &submittingDev : &submitting;
            cv.wait(lock, [&]{ return (data_ready && (*B)) || abort; });
            if (abort) break;
            
            try {
                boost::json::object *S = isDev ? &devShare : &share;
                std::string msg = boost::json::serialize((*S)) + "\n";
                
                beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(5));
                boost::asio::async_write(stream, boost::asio::buffer(msg), 
                    [&](const boost::system::error_code& error, std::size_t bytes_transferred) {
                        if (error) {
                            printf("error on write: %s\n", error.message().c_str());
                            fflush(stdout);
                            abort = true;
                        } else if (!isDev) {
                            rx0Stratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
                        }
                    });
                
                (*B) = false;
                data_ready = false;
            } catch (const std::exception &e) {
                setcolor(RED);
                printf("\nSubmit thread error: %s\n", e.what());
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                break;
            }
            boost::this_thread::yield();
        }
        submitThread = false;
    });

    boost::thread cacheThread([&]() {
        while(!abort) {
            boost::unique_lock<boost::mutex> lock(dsMutex);
            
            // Wait for dataset update signal OR abort
            cv.wait(lock, [&]{ 
                return needsDatasetUpdate.load() || abort; 
            });
            
            if (abort) break;
            printf("needs update");
            fflush(stdout);
            
            if (globalInDevBatch.load() == isDev) {
              try {
                needsDatasetUpdate.exchange(false);
                checkAndUpdateDatasetIfNeeded(isDev);
              } catch (const std::exception &e) {
                setcolor(RED);
                printf("\nDataset update error: %s\n", e.what());
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
              }
            }
            
            boost::this_thread::yield();
        }
    });

    while (!ABORT_MINER)
    {
        bool *C = isDev ? &devConnected : &isConnected;
        bool *B = isDev ? &submittingDev : &submitting;
        
        try
        {
            // Timeout check
            if (rx0Stratum::lastReceivedJobTime > 0 &&
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - 
                rx0Stratum::lastReceivedJobTime > rx0Stratum::jobTimeout)
            {
                setcolor(RED);
                printf("Stratum timeout\n");
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                setForDisconnected(C, B, &abort, &data_ready, &cv);

                // Wait for submit thread to finish
                for (;;) {
                    if (!submitThread) break;
                    boost::this_thread::yield();
                }
                
                // SSL shutdown
                stream.async_shutdown(yield[ec]);
                beast::get_lowest_layer(stream).close();
                return fail(ec, "Stratum session timed out");
            }

            // Read incoming data
            boost::asio::streambuf response;
            beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
            trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
            
            if (ec)
            {
                setcolor(RED);
                printf("Failed to read: %s\n", isDev ? "dev" : "user");
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                setForDisconnected(C, B, &abort, &data_ready, &cv);
                cv.notify_all();

                for (;;) {
                    if (!submitThread) break;
                    boost::this_thread::yield();
                }
                
                // SSL shutdown
                stream.async_shutdown(yield[ec]);
                beast::get_lowest_layer(stream).close();
                return fail(ec, "async_read");
            }

            if (trans > 0)
            {
                std::scoped_lock<boost::mutex> lockGuard(wsMutex);
                
                // Add new data to persistent buffer
                std::string newData = beast::buffers_to_string(response.data());
                response.consume(trans);
                packetBuffer += newData;

                // Process all complete packets (lines ending with \n)
                size_t pos = 0;
                while ((pos = packetBuffer.find('\n')) != std::string::npos)
                {
                    std::string completePacket = packetBuffer.substr(0, pos);
                    packetBuffer.erase(0, pos + 1); // Remove processed packet including \n

                    if (!completePacket.empty())
                    {
                        try
                        {
                            boost::json::object sRPC = boost::json::parse(completePacket).as_object();
                            
                            if (sRPC.contains("method"))
                            {
                                std::string method = sRPC.at("method").as_string().c_str();
                                if (method.compare(rx0Stratum::s_ping) == 0)
                                {
                                    boost::json::object pong({
                                        {"id", sRPC.at("id").get_uint64()},
                                        {"method", rx0Stratum::pong.method}
                                    });
                                    std::string pongPacket = boost::json::serialize(pong) + "\n";
                                    trans = boost::asio::async_write(stream, boost::asio::buffer(pongPacket), yield[ec]);
                                    if (ec)
                                    {
                                        setcolor(RED);
                                        printf("Failed to send pong\n");
                                        fflush(stdout);
                                        setcolor(BRIGHT_WHITE);
                                        setForDisconnected(C, B, &abort, &data_ready, &cv);

                                        for (;;) {
                                            if (!submitThread) break;
                                            boost::this_thread::yield();
                                        }
                                        
                                        // SSL shutdown
                                        stream.async_shutdown(yield[ec]);
                                        beast::get_lowest_layer(stream).close();
                                        return fail(ec, "Stratum pong");
                                    }
                                }
                                else
                                {
                                    handleRandomXStratumPacket(sRPC, isDev);
                                }
                            }
                            else
                            {
                                handleRandomXStratumResponse(sRPC, isDev);
                            }
                        }
                        catch (const std::exception &e)
                        {
                            setcolor(RED);
                            printf("Parse error: %s\nPacket: %s\n", e.what(), completePacket.c_str());
                            fflush(stdout);
                            setcolor(BRIGHT_WHITE);
                        }
                    }
                }

                // Prevent buffer from growing indefinitely
                if (packetBuffer.length() > 65536)
                {
                    setcolor(RED);
                    printf("Packet buffer overflow, clearing\n");
                    fflush(stdout);
                    setcolor(BRIGHT_WHITE);
                    packetBuffer.clear();
                }
            }
        }
        catch (const std::exception &e)
        {
            printf("Session exception: %s\n", e.what());
            fflush(stdout);
            setForDisconnected(C, B, &abort, &data_ready, &cv);

            for (;;) {
                if (!submitThread) break;
                boost::this_thread::yield();
            }
            
            // SSL shutdown
            stream.async_shutdown(yield[ec]);
            beast::get_lowest_layer(stream).close();
            setcolor(RED);
            std::cerr << e.what() << std::endl;
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
            return;
        }

        boost::this_thread::yield();
        
        if(ABORT_MINER) {
            bool *connPtr = isDev ? &devConnected : &isConnected;
            bool *submitPtr = isDev ? &submittingDev : &submitting;
            setForDisconnected(connPtr, submitPtr, &abort, &data_ready, &cv);
            ioc.stop();
        }
    }

    // Clean shutdown
    abort = true;
    cv.notify_all();

    cacheThread.interrupt();
    cacheThread.join();

    subThread.interrupt();
    subThread.join();

    // SSL shutdown
    beast::error_code shutdown_ec;
    stream.async_shutdown(yield[shutdown_ec]);
    beast::get_lowest_layer(stream).close();
}

void rx0_stratum_session_nossl(
    std::string sessionHost,
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

    auto endpoint = resolve_host(wsMutex, ioc, yield, sessionHost, port);
    boost::beast::tcp_stream stream(ioc);

    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
    if (ec)
        return fail(ec, "connect");

    boost::json::object packet = rx0Stratum::stratumCall;
    packet.at("id") = rx0Stratum::login.id;
    packet.at("method") = rx0Stratum::login.method;

    std::string userAgent = "tnn-miner/" + std::string(versionString);

    boost::json::object loginParams = {
        {"login", wallet},
        {"pass", stratumPassword},
        {"rigid", worker},
        {"agent", userAgent.c_str()}
    };

    packet.at("params") = loginParams;
    std::string login = boost::json::serialize(packet) + "\n";

    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    size_t trans = boost::asio::async_write(stream, boost::asio::buffer(login), yield[ec]);
    if (ec)
        return fail(ec, "Stratum login");

    rx0Stratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Add persistent packet buffer for handling split packets
    std::string packetBuffer;

    bool submitThread = false;
    bool abort = false;

    boost::thread subThread([&](){
        submitThread = true;
        while(!abort) {
            boost::unique_lock<boost::mutex> lock(mutex);
            bool *B = isDev ? &submittingDev : &submitting;
            cv.wait(lock, [&]{ return (data_ready && (*B)) || abort; });
            if (abort) break;
            
            try {
                boost::json::object *S = isDev ? &devShare : &share;
                std::string msg = boost::json::serialize((*S)) + "\n";
                
                beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(5));
                boost::asio::async_write(stream, boost::asio::buffer(msg), 
                    [&](const boost::system::error_code& error, std::size_t bytes_transferred) {
                        if (error) {
                            printf("error on write: %s\n", error.message().c_str());
                            fflush(stdout);
                            abort = true;
                        } else if (!isDev) {
                            rx0Stratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
                        }
                    });
                
                (*B) = false;
                data_ready = false;
            } catch (const std::exception &e) {
                setcolor(RED);
                printf("\nSubmit thread error: %s\n", e.what());
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                break;
            }
            boost::this_thread::yield();
        }
        submitThread = false;
    });

    boost::thread cacheThread([&]() {
        while(!abort) {
            boost::unique_lock<boost::mutex> lock(mutex);
            
            // Wait for dataset update signal OR abort
            cv.wait(lock, [&]{ 
                return needsDatasetUpdate.load() || abort; 
            });
            
            if (abort) break;
            bool isActiveMiningMode = (isDev == globalInDevBatch.load());
            if (isActiveMiningMode) {
              try {
                checkAndUpdateDatasetIfNeeded(isDev);
                needsDatasetUpdate.exchange(false);
                lock.unlock();
              } catch (const std::exception &e) {
                setcolor(RED);
                printf("\nDataset update error: %s\n", e.what());
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
              }
            }
            
            boost::this_thread::yield();
        }
    });

    while (!ABORT_MINER)
    {
        bool *C = isDev ? &devConnected : &isConnected;
        bool *B = isDev ? &submittingDev : &submitting;
        
        try
        {
            // Timeout check
            if (rx0Stratum::lastReceivedJobTime > 0 &&
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - 
                rx0Stratum::lastReceivedJobTime > rx0Stratum::jobTimeout)
            {
                setcolor(RED);
                printf("Stratum timeout\n");
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                setForDisconnected(C, B, &abort, &data_ready, &cv);

                // Wait for submit thread to finish
                for (;;) {
                    if (!submitThread) break;
                    boost::this_thread::yield();
                }
                stream.close();
                return fail(ec, "Stratum session timed out");
            }

            // Read incoming data
            boost::asio::streambuf response;
            beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
            trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
            
            if (ec)
            {
                setcolor(RED);
                printf("Failed to read: %s\n", isDev ? "dev" : "user");
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                setForDisconnected(C, B, &abort, &data_ready, &cv);
                boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
                cv.notify_all();

                for (;;) {
                    if (!submitThread) break;
                    boost::this_thread::yield();
                }
                
                stream.close();
                return fail(ec, "async_read");
            }

            if (trans > 0)
            {
                std::scoped_lock<boost::mutex> lockGuard(wsMutex);
                
                // Add new data to persistent buffer
                std::string newData = beast::buffers_to_string(response.data());
                response.consume(trans);
                packetBuffer += newData;

                // Process all complete packets (lines ending with \n)
                size_t pos = 0;
                while ((pos = packetBuffer.find('\n')) != std::string::npos)
                {
                    std::string completePacket = packetBuffer.substr(0, pos);
                    packetBuffer.erase(0, pos + 1); // Remove processed packet including \n

                    if (!completePacket.empty())
                    {
                        try
                        {
                            boost::json::object sRPC = boost::json::parse(completePacket).as_object();
                            
                            if (sRPC.contains("method"))
                            {
                                std::string method = sRPC.at("method").as_string().c_str();
                                if (method.compare(rx0Stratum::s_ping) == 0)
                                {
                                    boost::json::object pong({
                                        {"id", sRPC.at("id").get_uint64()},
                                        {"method", rx0Stratum::pong.method}
                                    });
                                    std::string pongPacket = boost::json::serialize(pong) + "\n";
                                    trans = boost::asio::async_write(stream, boost::asio::buffer(pongPacket), yield[ec]);
                                    if (ec)
                                    {
                                        setcolor(RED);
                                        printf("Failed to send pong\n");
                                        fflush(stdout);
                                        setcolor(BRIGHT_WHITE);
                                        setForDisconnected(C, B, &abort, &data_ready, &cv);

                                        for (;;) {
                                            if (!submitThread) break;
                                            boost::this_thread::yield();
                                        }
                                        stream.close();
                                        return fail(ec, "Stratum pong");
                                    }
                                }
                                else
                                {
                                    handleRandomXStratumPacket(sRPC, isDev);
                                }
                            }
                            else
                            {
                                handleRandomXStratumResponse(sRPC, isDev);
                            }
                        }
                        catch (const std::exception &e)
                        {
                            setcolor(RED);
                            printf("Parse error: %s\nPacket: %s\n", e.what(), completePacket.c_str());
                            fflush(stdout);
                            setcolor(BRIGHT_WHITE);
                        }
                    }
                }

                // Prevent buffer from growing indefinitely
                if (packetBuffer.length() > 65536)
                {
                    setcolor(RED);
                    printf("Packet buffer overflow, clearing\n");
                    fflush(stdout);
                    setcolor(BRIGHT_WHITE);
                    packetBuffer.clear();
                }
            }
        }
        catch (const std::exception &e)
        {
            printf("Session exception: %s\n", e.what());
            fflush(stdout);
            setForDisconnected(C, B, &abort, &data_ready, &cv);

            for (;;) {
                if (!submitThread) break;
                boost::this_thread::yield();
            }
            stream.close();
            setcolor(RED);
            std::cerr << e.what() << std::endl;
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
            return;
        }

        boost::this_thread::yield();
        
        if(ABORT_MINER) {
            bool *connPtr = isDev ? &devConnected : &isConnected;
            bool *submitPtr = isDev ? &submittingDev : &submitting;
            setForDisconnected(connPtr, submitPtr, &abort, &data_ready, &cv);
            ioc.stop();
        }
    }

    // Clean shutdown
    abort = true;
    cv.notify_all();

    cacheThread.interrupt();
    cacheThread.join();

    subThread.joinable();
    subThread.join();
}