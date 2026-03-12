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
#include <spectrex/spectrex.h>

#include <atomic>
#include <queue>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

int handleSpectreStratumPacket(boost::json::object packet, SpectreStratum::jobCache *cache, bool isDev)
{
  std::string M = std::string(packet.at("method").get_string());
  
  if (isDev) {
    nonceLenDev = 0;
  } else {
    nonceLen = 0;
  }
  
  if (M == SpectreStratum::s_notify)
  {
    std::scoped_lock<std::mutex> lockGuard(mutex);
    boost::json::value *J = isDev ? &devJob : &job;
    int64_t *h = isDev ? &devHeight : &ourHeight;

    uint64_t h1 = packet["params"].as_array()[1].as_array()[0].get_uint64();
    uint64_t h2 = packet["params"].as_array()[1].as_array()[1].get_uint64();
    uint64_t h3 = packet["params"].as_array()[1].as_array()[2].get_uint64();
    uint64_t h4 = packet["params"].as_array()[1].as_array()[3].get_uint64();
    uint64_t ts = packet["params"].as_array()[2].get_uint64();

    std::string jobIdStr = std::string(packet["params"].as_array()[0].get_string());
    (*J).as_object()["jobId"] = jobIdStr;

    bool isEqual;
    if (isDev) {
      isEqual = cache->devHeader[0] == h1 && cache->devHeader[1] == h2 && 
                cache->devHeader[2] == h3 && cache->devHeader[3] == h4 && 
                cache->devHeader[4] == ts;
    } else {
      isEqual = cache->header[0] == h1 && cache->header[1] == h2 && 
                cache->header[2] == h3 && cache->header[3] == h4 && 
                cache->header[4] == ts;
    }

    if (!isEqual) {
      uint64_t &N = isDev ? nonce0_dev : nonce0;
      N = 0;

      if (isDev) {
        cache->devHeader[0] = h1;
        cache->devHeader[1] = h2;
        cache->devHeader[2] = h3;
        cache->devHeader[3] = h4;
        cache->devHeader[4] = ts;
      } else {
        cache->header[0] = h1;
        cache->header[1] = h2;
        cache->header[2] = h3;
        cache->header[3] = h4;
        cache->header[4] = ts;
      }

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

      SpectreStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      (*J).as_object()["template"] = std::string(newTemplate, SpectreX::INPUT_SIZE * 2);
      (*h)++;
      jobCounter++;
    }

    if (!isEqual && !beQuiet) {
      setcolor(CYAN);
      printf("\n");
      if (isDev) printf("DEV | ");
      printf("Stratum: new job received\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
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
        printf("Connected to dev node\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }

    *C = true;
  }
  else if (M == SpectreStratum::s_setDifficulty)
  {
    double *d = isDev ? &doubleDiffDev : &doubleDiff;
    (*d) = packet.at("params").as_array()[0].get_double();
    if ((*d) < 0.00000000001) 
      (*d) = packet.at("params").as_array()[0].get_uint64();

    uint256_t *dRef = isDev ? &bigDiff_dev : &bigDiff;
    *dRef = SpectreX::diffToTarget(*d);

    if (!beQuiet) {
      setcolor(CYAN);
      if (isDev) printf("DEV | ");
      printf("Difficulty set to: %.8f\n", *d);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    jobCounter++;
  }
  else if (M == SpectreStratum::s_setExtraNonce)
  {
    std::scoped_lock<std::mutex> lockGuard(mutex);
    boost::json::value *J = isDev ? &devJob : &job;

    std::string enStr = std::string(packet.at("params").as_array()[0].as_string());
    (*J).as_object()["extraNonce"] = enStr;
  }
  else if (M == SpectreStratum::s_print)
  {
    int lLevel = packet.at("params").as_array()[0].to_number<int64_t>();
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
        if (!isDev) setcolor(BRIGHT_WHITE);
        printf("Stratum INFO: ");
        break;
      case SpectreStratum::STRATUM_WARN:
        if (!isDev) setcolor(BRIGHT_YELLOW);
        printf("Stratum WARNING: ");
        break;
      case SpectreStratum::STRATUM_ERROR:
        if (!isDev) setcolor(RED);
        printf("Stratum ERROR: ");
        res = -1;
        break;
      case SpectreStratum::STRATUM_DEBUG:
        break;
      }
      
      std::string msgStr = std::string(packet.at("params").as_array()[1].as_string());
      printf("%s\n", msgStr.c_str());

      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return res;
    }
  }
  else
  {
    std::string packetStr = boost::json::serialize(packet);
    std::cout << "Stratum: unrecognized packet: " << packetStr << std::endl;
  }
  return 0;
}

int handleSpectreStratumResponse(boost::json::object packet, bool isDev)
{
  if (!packet.contains("id")) return 0;
  
  int64_t id = packet["id"].to_number<int64_t>();

  switch (id)
  {
    case SpectreStratum::subscribeID:
    {
      if (!beQuiet) {
        std::string packetStr = boost::json::serialize(packet);
        std::cout << packetStr << std::endl;
      }
      
      if (!packet["error"].is_null()) {
        std::string errorMsg = "Unknown error";
        if (packet["error"].is_string()) {
          errorMsg = std::string(packet["error"].get_string());
        }
        setcolor(RED);
        printf("\n");
        if (isDev) {
          setcolor(CYAN);
          printf("DEV | ");
        }
        printf("Stratum ERROR: %s\n", errorMsg.c_str());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        return -1;
      }
      return 0;
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
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        if (!isDev) rejected++;
        if (!isDev) setcolor(RED);

        std::string errStr = "Unknown error";
        if (packet.contains("error")) {
          if (packet["error"].is_array() && packet["error"].as_array().size() > 1) {
            errStr = std::string(packet.at("error").as_array()[1].as_string());
          } else if (packet["error"].is_object() && packet["error"].as_object().contains("message")) {
            errStr = std::string(packet.at("error").at("message").get_string());
          } else if (packet["error"].is_string()) {
            errStr = std::string(packet["error"].get_string());
          }
        }
        
        std::cout << "Stratum: share rejected: " << errStr << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
      break;
    }
  }
  return 0;
}

// Helper to wait for submit thread
inline void waitForSubmitThread(bool &submitThread) {
  while (submitThread) {
    std::this_thread::yield();
  }
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
  SpectreStratum::lastReceivedJobTime = 0;
  ctx.set_options(boost::asio::ssl::context::default_workarounds |
                  boost::asio::ssl::context::no_sslv2 |
                  boost::asio::ssl::context::no_sslv3 |
                  boost::asio::ssl::context::no_tlsv1 |
                  boost::asio::ssl::context::no_tlsv1_1);

  beast::error_code ec;
  ctx.set_verify_mode(ssl::verify_none);

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  // Simple stream without strand
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-spectre-ssl");

  if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
  {
    throw beast::system_error{
        static_cast<int>(::ERR_get_error()),
        boost::asio::error::get_ssl_category()};
  }

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec) return fail(ec, "handshake-spectre-ssl");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  boost::json::object packet;
  SpectreStratum::jobCache jobCache;

  // Subscribe
  packet = SpectreStratum::stratumCall;
  packet["id"] = SpectreStratum::subscribe.id;
  packet["method"] = SpectreStratum::subscribe.method;
  packet["params"] = boost::json::array({minerName});
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum subscribe");

    boost::asio::streambuf buf;
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    size_t n = boost::asio::async_read_until(stream, buf, "\n", yield[ec]);
    if (ec) return fail(ec, "Stratum subscribe response");

    std::string s = beast::buffers_to_string(buf.data());
    buf.consume(n);
    
    std::stringstream ss(s);
    std::string line;
    while (std::getline(ss, line)) {
      if (line.empty()) continue;
      try {
        auto rpc = boost::json::parse(line).as_object();
        if (rpc.contains("method"))
          handleSpectreStratumPacket(rpc, &jobCache, isDev);
        else
          handleSpectreStratumResponse(rpc, isDev);
      } catch (const std::exception &e) {
        printf("Subscribe response parse error: %s\n", e.what());
      }
    }
  }

  // Authorize
  packet = SpectreStratum::stratumCall;
  packet["id"] = SpectreStratum::authorize.id;
  packet["method"] = SpectreStratum::authorize.method;
  if (isDev) {
    packet["params"] = boost::json::array({
        devWallet + "." + worker + "-" + tnnTargetArch,
        stratumPassword
    });
  } else {
    packet["params"] = boost::json::array({
        wallet + "." + worker,
        stratumPassword
    });
  }
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum authorize");

    boost::asio::streambuf buf;
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    size_t n = boost::asio::async_read_until(stream, buf, "\n", yield[ec]);
    if (ec) return fail(ec, "Stratum authorize response");

    std::string s = beast::buffers_to_string(buf.data());
    buf.consume(n);
    
    std::stringstream ss(s);
    std::string line;
    while (std::getline(ss, line)) {
      if (line.empty()) continue;
      try {
        auto rpc = boost::json::parse(line).as_object();
        if (rpc.contains("method"))
          handleSpectreStratumPacket(rpc, &jobCache, isDev);
        else
          handleSpectreStratumResponse(rpc, isDev);
      } catch (const std::exception &e) {
        printf("Authorize response parse error: %s\n", e.what());
      }
    }
  }

  SpectreStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  std::string packetBuffer;
  std::queue<std::string> submitQueue;
  std::mutex submitMutex;
  bool submitThread = false;
  bool abort = false;

  // Submit thread with queue
  std::thread subThread([&]() {
    submitThread = true;
    while (!abort) {
      std::unique_lock<std::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&] { return (data_ready && (*B)) || abort; });
      if (abort) break;

      try {
        boost::json::object *S = isDev ? &devShare : &share;
        std::string msg = boost::json::serialize(*S) + "\n";
        
        {
          std::lock_guard<std::mutex> qlock(submitMutex);
          submitQueue.push(std::move(msg));
        }
        
        // Process queue
        while (!abort) {
          std::string qmsg;
          {
            std::lock_guard<std::mutex> qlock(submitMutex);
            if (submitQueue.empty()) break;
            qmsg = std::move(submitQueue.front());
            submitQueue.pop();
          }
          
          beast::error_code wec;
          beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
          boost::asio::write(stream, boost::asio::buffer(qmsg), wec);
          
          if (wec) {
            printf("error on write: %s\n", wec.message().c_str());
            fflush(stdout);
            abort = true;
            break;
          }
          
          if (!isDev) {
            SpectreStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
          }
        }
        
        *B = false;
        data_ready = false;
      } catch (const std::exception &e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        break;
      }
      std::this_thread::yield();
    }
    submitThread = false;
  });

  // Main message loop
  while (!ABORT_MINER && !abort) {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try {
      if (SpectreStratum::lastReceivedJobTime > 0 &&
          (std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() -
           SpectreStratum::lastReceivedJobTime) > SpectreStratum::jobTimeout)
      {
        setcolor(RED);
        printf("\nStratum session timed out\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        waitForSubmitThread(submitThread);
        beast::get_lowest_layer(stream).close();
        return fail(ec, "Stratum session timed out");
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t n = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        waitForSubmitThread(submitThread);
        beast::get_lowest_layer(stream).close();
        return fail(ec, "async_read");
      }

      if (n > 0) {
        std::string newData = beast::buffers_to_string(response.data());
        response.consume(n);
        packetBuffer += newData;

        if (packetBuffer.size() > 1024 * 1024) {
          setcolor(RED);
          printf("\nPacket buffer overflow, disconnecting\n");
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
          setForDisconnected(C, B, &abort, &data_ready, &cv);
          waitForSubmitThread(submitThread);
          beast::get_lowest_layer(stream).close();
          return;
        }

        size_t pos;
        while ((pos = packetBuffer.find('\n')) != std::string::npos) {
          std::string line = packetBuffer.substr(0, pos);
          packetBuffer.erase(0, pos + 1);
          
          if (line.empty()) continue;

          try {
            auto rpc = boost::json::parse(line).as_object();
            if (rpc.contains("method")) {
              std::string method = std::string(rpc["method"].as_string());
              if (method == SpectreStratum::s_ping) {
                boost::json::object pong = {
                    {"id", rpc["id"].get_uint64()},
                    {"method", SpectreStratum::pong.method}
                };
                std::string pongMsg = boost::json::serialize(pong) + "\n";
                boost::asio::async_write(stream, boost::asio::buffer(pongMsg), yield[ec]);
                if (ec) {
                  setForDisconnected(C, B, &abort, &data_ready, &cv);
                  waitForSubmitThread(submitThread);
                  beast::get_lowest_layer(stream).close();
                  return fail(ec, "Stratum pong");
                }
              } else {
                handleSpectreStratumPacket(rpc, &jobCache, isDev);
              }
            } else {
              handleSpectreStratumResponse(rpc, isDev);
            }
          } catch (const std::exception &e) {
            setcolor(RED);
            printf("\nParse error: %s\n", e.what());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
        }
      }
    } catch (const std::exception &e) {
      setcolor(RED);
      printf("\nSpectre stratum error: %s\n", e.what());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      waitForSubmitThread(submitThread);
      beast::get_lowest_layer(stream).close();
      return;
    }

    std::this_thread::yield();

    if (ABORT_MINER) {
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      ioc.stop();
    }
  }

  // Cleanup
  abort = true;
  cv.notify_all();
  if (subThread.joinable()) {
    subThread.join();
  }

  beast::error_code shutdown_ec;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(5));
  stream.async_shutdown(yield[shutdown_ec]);
  beast::get_lowest_layer(stream).close();
}

void spectre_stratum_session_nossl(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  SpectreStratum::lastReceivedJobTime = 0;
  beast::error_code ec;

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  // Simple stream without strand
  boost::beast::tcp_stream stream(ioc);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-spectre-nossl");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  boost::json::object packet;
  SpectreStratum::jobCache jobCache;

  // Subscribe
  packet = SpectreStratum::stratumCall;
  packet["id"] = SpectreStratum::subscribe.id;
  packet["method"] = SpectreStratum::subscribe.method;
  packet["params"] = boost::json::array({minerName});
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum subscribe");

    boost::asio::streambuf buf;
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    size_t n = boost::asio::async_read_until(stream, buf, "\n", yield[ec]);
    if (ec) return fail(ec, "Stratum subscribe response");

    std::string s = beast::buffers_to_string(buf.data());
    buf.consume(n);
    
    std::stringstream ss(s);
    std::string line;
    while (std::getline(ss, line)) {
      if (line.empty()) continue;
      try {
        auto rpc = boost::json::parse(line).as_object();
        if (rpc.contains("method"))
          handleSpectreStratumPacket(rpc, &jobCache, isDev);
        else
          handleSpectreStratumResponse(rpc, isDev);
      } catch (const std::exception &e) {
        printf("Subscribe response parse error: %s\n", e.what());
      }
    }
  }

  // Authorize
  packet = SpectreStratum::stratumCall;
  packet["id"] = SpectreStratum::authorize.id;
  packet["method"] = SpectreStratum::authorize.method;
  if (isDev) {
    packet["params"] = boost::json::array({
        devWallet + "." + worker + "-" + tnnTargetArch,
        stratumPassword
    });
  } else {
    packet["params"] = boost::json::array({
        wallet + "." + worker,
        stratumPassword
    });
  }
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum authorize");

    boost::asio::streambuf buf;
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    size_t n = boost::asio::async_read_until(stream, buf, "\n", yield[ec]);
    if (ec) return fail(ec, "Stratum authorize response");

    std::string s = beast::buffers_to_string(buf.data());
    buf.consume(n);
    
    std::stringstream ss(s);
    std::string line;
    while (std::getline(ss, line)) {
      if (line.empty()) continue;
      try {
        auto rpc = boost::json::parse(line).as_object();
        if (rpc.contains("method"))
          handleSpectreStratumPacket(rpc, &jobCache, isDev);
        else
          handleSpectreStratumResponse(rpc, isDev);
      } catch (const std::exception &e) {
        printf("Authorize response parse error: %s\n", e.what());
      }
    }
  }

  SpectreStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  std::string packetBuffer;
  std::queue<std::string> submitQueue;
  std::mutex submitMutex;
  bool submitThread = false;
  bool abort = false;

  // Submit thread with queue
  std::thread subThread([&]() {
    submitThread = true;
    while (!abort) {
      std::unique_lock<std::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&] { return (data_ready && (*B)) || abort; });
      if (abort) break;

      try {
        boost::json::object *S = isDev ? &devShare : &share;
        std::string msg = boost::json::serialize(*S) + "\n";
        
        {
          std::lock_guard<std::mutex> qlock(submitMutex);
          submitQueue.push(std::move(msg));
        }
        
        // Process queue
        while (!abort) {
          std::string qmsg;
          {
            std::lock_guard<std::mutex> qlock(submitMutex);
            if (submitQueue.empty()) break;
            qmsg = std::move(submitQueue.front());
            submitQueue.pop();
          }
          
          beast::error_code wec;
          beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
          boost::asio::write(stream, boost::asio::buffer(qmsg), wec);
          
          if (wec) {
            printf("error on write: %s\n", wec.message().c_str());
            fflush(stdout);
            abort = true;
            break;
          }
          
          if (!isDev) {
            SpectreStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
          }
        }
        
        *B = false;
        data_ready = false;
      } catch (const std::exception &e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        break;
      }
      std::this_thread::yield();
    }
    submitThread = false;
  });

  // Main message loop
  while (!ABORT_MINER && !abort) {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try {
      if (SpectreStratum::lastReceivedJobTime > 0 &&
          (std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() -
           SpectreStratum::lastReceivedJobTime) > SpectreStratum::jobTimeout)
      {
        setcolor(RED);
        printf("\nStratum session timed out\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        waitForSubmitThread(submitThread);
        stream.close();
        return fail(ec, "Stratum session timed out");
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t n = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        waitForSubmitThread(submitThread);
        stream.close();
        return fail(ec, "async_read");
      }

      if (n > 0) {
        std::string newData = beast::buffers_to_string(response.data());
        response.consume(n);
        packetBuffer += newData;

        if (packetBuffer.size() > 1024 * 1024) {
          setcolor(RED);
          printf("\nPacket buffer overflow, disconnecting\n");
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
          setForDisconnected(C, B, &abort, &data_ready, &cv);
          waitForSubmitThread(submitThread);
          stream.close();
          return;
        }

        size_t pos;
        while ((pos = packetBuffer.find('\n')) != std::string::npos) {
          std::string line = packetBuffer.substr(0, pos);
          packetBuffer.erase(0, pos + 1);
          
          if (line.empty()) continue;

          try {
            auto rpc = boost::json::parse(line).as_object();
            if (rpc.contains("method")) {
              std::string method = std::string(rpc["method"].as_string());
              if (method == SpectreStratum::s_ping) {
                boost::json::object pong = {
                    {"id", rpc["id"].get_uint64()},
                    {"method", SpectreStratum::pong.method}
                };
                std::string pongMsg = boost::json::serialize(pong) + "\n";
                boost::asio::async_write(stream, boost::asio::buffer(pongMsg), yield[ec]);
                if (ec) {
                  setForDisconnected(C, B, &abort, &data_ready, &cv);
                  waitForSubmitThread(submitThread);
                  stream.close();
                  return fail(ec, "Stratum pong");
                }
              } else {
                handleSpectreStratumPacket(rpc, &jobCache, isDev);
              }
            } else {
              handleSpectreStratumResponse(rpc, isDev);
            }
          } catch (const std::exception &e) {
            setcolor(RED);
            printf("\nParse error: %s\n", e.what());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
        }
      }
    } catch (const std::exception &e) {
      setcolor(RED);
      printf("\nSpectre stratum error: %s\n", e.what());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      waitForSubmitThread(submitThread);
      stream.close();
      return;
    }

    std::this_thread::yield();

    if (ABORT_MINER) {
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      ioc.stop();
    }
  }

  // Cleanup
  abort = true;
  cv.notify_all();
  if (subThread.joinable()) {
    subThread.join();
  }

  stream.close();
}