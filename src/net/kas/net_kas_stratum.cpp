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

#include <atomic>
#include <queue>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

using uint256_t = boost::multiprecision::uint256_t;
using cpp_dec_float_50 = boost::multiprecision::cpp_dec_float_50;

int handleKasStratumPacket(boost::json::object packet, KasStratum::jobCache *cache, bool isDev)
{
  std::string M = std::string(packet.at("method").get_string());
  
  if (M == KasStratum::s_notify)
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    boost::json::value *J = isDev ? &devJob : &job;
    int64_t *h = isDev ? &devHeight : &ourHeight;

    uint64_t h1 = packet["params"].as_array()[1].as_array()[0].get_uint64();
    uint64_t h2 = packet["params"].as_array()[1].as_array()[1].get_uint64();
    uint64_t h3 = packet["params"].as_array()[1].as_array()[2].get_uint64();
    uint64_t h4 = packet["params"].as_array()[1].as_array()[3].get_uint64();

    uint64_t comboHeader[4] = {h1, h2, h3, h4};
    uint64_t ts = packet["params"].as_array()[2].get_uint64();

    bool isEqual = true;
    for (int i = 0; i < 4; i++) {
      isEqual &= comboHeader[i] == cache->header[i];
    }
    isEqual &= ts == cache->ts;

    if (!isEqual) {
      uint64_t &N = isDev ? nonce0_dev : nonce0;
      N = 0;

      if (gpuMine) {
        for (int i = 0; i < HIP_deviceCount; i++) {
          uint64_t &K = isDev ? HIP_kIndex_dev[i] : HIP_kIndex[i];
        }
      }
    }

    for (int i = 0; i < 4; i++) {
      cache->header[i] = comboHeader[i];
    }
    cache->ts = ts;

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

    if (!isEqual && !beQuiet) {
      setcolor(CYAN);
      if (!isDev)
        printf("\nStratum: new job received\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    KasStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    std::string jobIdStr = std::string(packet["params"].as_array()[0].get_string());
    (*J).as_object()["template"] = std::string(newTemplate, KasStratum::INPUT_SIZE * 2);
    (*J).as_object()["jobId"] = jobIdStr;

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
    (*h)++;
    jobCounter++;
  }
  else if (M == KasStratum::s_setDifficulty)
  {
    double *d = isDev ? &doubleDiffDev : &doubleDiff;
    (*d) = packet.at("params").as_array()[0].get_double();
    if ((*d) < 0.00000000001) 
      (*d) = packet.at("params").as_array()[0].get_uint64();

    uint256_t *dRef = isDev ? &bigDiff_dev : &bigDiff;
    *dRef = KasStratum::diffToTarget(*d);

    jobCounter++;
  }
  else if (M == KasStratum::s_setExtraNonce)
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    boost::json::value *J = isDev ? &devJob : &job;

    std::string enStr = std::string(packet.at("params").as_array()[0].as_string());
    (*J).as_object()["extraNonce"] = enStr;
  }
  else if (M == KasStratum::s_print)
  {
    int lLevel = packet.at("params").as_array()[0].to_number<int64_t>();
    if (lLevel != KasStratum::STRATUM_DEBUG)
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
      case KasStratum::STRATUM_INFO:
        if (!isDev) setcolor(BRIGHT_WHITE);
        printf("Stratum INFO: ");
        break;
      case KasStratum::STRATUM_WARN:
        if (!isDev) setcolor(BRIGHT_YELLOW);
        printf("Stratum WARNING: ");
        break;
      case KasStratum::STRATUM_ERROR:
        if (!isDev) setcolor(RED);
        printf("Stratum ERROR: ");
        res = -1;
        break;
      case KasStratum::STRATUM_DEBUG:
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

int handleKasStratumResponse(boost::json::object packet, bool isDev)
{
  if (!packet.contains("id")) return 0;
  
  int64_t id = packet["id"].to_number<int64_t>();

  switch (id)
  {
    case KasStratum::subscribeID:
    {
      std::string packetStr = boost::json::serialize(packet);
      std::cout << packetStr << std::endl;
      
      if (packet["error"].is_null()) return 0;
      else {
        std::string errorMsg = std::string(packet["error"].get_string());
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
    }
    break;
    case KasStratum::submitID:
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

        std::string errStr;
        if (packet.contains("error")) {
          if (packet["error"].is_array()) {
            errStr = std::string(packet.at("error").as_array()[1].as_string());
          } else if (packet["error"].is_object() && packet["error"].as_object().contains("message")) {
            errStr = std::string(packet.at("error").at("message").get_string());
          } else if (packet["error"].is_string()) {
            errStr = std::string(packet["error"].get_string());
          } else {
            errStr = "Unknown error";
          }
        } else {
          errStr = "Unknown error";
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

void kas_stratum_session(
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
  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  auto strand = net::make_strand(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(strand, ctx);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-kas-ssl");

  if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
  {
    throw beast::system_error{
        static_cast<int>(::ERR_get_error()),
        boost::asio::error::get_ssl_category()};
  }

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec) return fail(ec, "handshake-kas-ssl");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  boost::json::object packet;
  KasStratum::jobCache jobCache;

  // Subscribe
  packet = KasStratum::stratumCall;
  packet["id"] = KasStratum::subscribe.id;
  packet["method"] = KasStratum::subscribe.method;
  packet["params"] = boost::json::array({minerName});
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum subscribe");
  }

  // Authorize
  packet = KasStratum::stratumCall;
  packet["id"] = KasStratum::authorize.id;
  packet["method"] = KasStratum::authorize.method;
  packet["params"] = boost::json::array({
      wallet + "." + worker,
      stratumPassword
  });
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum authorize");
  }

  KasStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  std::string packetBuffer;
  std::queue<std::string> submitQueue;
  boost::mutex submitMutex;
  std::atomic<bool> abort{false};

  auto process_write_queue = [&]() {
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          if (submitQueue.empty()) return;
          msg = std::move(submitQueue.front());
          submitQueue.pop();
        }
        
        beast::error_code wec;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
        boost::asio::write(stream, boost::asio::buffer(msg), wec);
        
        if (wec) {
          printf("error on write: %s\n", wec.message().c_str());
          fflush(stdout);
          abort = true;
          return;
        }
        
        if (!isDev) {
          KasStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
        }
      }
    });
  };

  bool submitThreadRunning = true;

  boost::thread subThread([&]() {
    while (!abort.load()) {
      boost::unique_lock<boost::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&] { return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;

      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          submitQueue.push(std::move(msg));
        }
        process_write_queue();
        
      } catch (const std::exception &e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        break;
      }
      *B = false;
      data_ready = false;
      boost::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  while (!ABORT_MINER && !abort.load()) {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try {
      if (KasStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count()
          - KasStratum::lastReceivedJobTime > KasStratum::jobTimeout)
      {
        setcolor(RED);
        printf("\nStratum session timed out\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t n = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      std::string newData(
          boost::asio::buffers_begin(response.data()),
          boost::asio::buffers_begin(response.data()) + n
      );
      response.consume(n);
      packetBuffer += newData;

      if (packetBuffer.size() > 1024 * 1024) {
        setcolor(RED);
        printf("\nPacket buffer overflow, disconnecting\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
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
            if (method == KasStratum::s_ping)
            {
              boost::json::object pong = {
                  {"id", rpc["id"].get_uint64()},
                  {"method", KasStratum::pong.method}
              };
              std::string pongMsg = boost::json::serialize(pong) + "\n";
              boost::asio::async_write(stream, boost::asio::buffer(pongMsg), yield[ec]);
              if (ec) {
                setForDisconnected(C, B, &abort, &data_ready, &cv);
                break;
              }
            }
            else {
              handleKasStratumPacket(rpc, &jobCache, isDev);
            }
          }
          else {
            handleKasStratumResponse(rpc, isDev);
          }
        }
        catch (const std::exception &e) {
          setcolor(RED);
          printf("\nParse error: %s\n", e.what());
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
        }
      }
    }
    catch (const std::exception &e) {
      setcolor(RED);
      printf("\nSession exception: %s\n", e.what());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }

    boost::this_thread::yield();
  }

  abort = true;
  cv.notify_all();
  
  if (submitThreadRunning) {
    subThread.interrupt();
    if (subThread.joinable()) {
      subThread.join();
    }
  }

  beast::error_code shutdown_ec;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(5));
  stream.async_shutdown(yield[shutdown_ec]);
  beast::get_lowest_layer(stream).close();
}

void kas_stratum_session_nossl(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  beast::error_code ec;
  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  auto strand = net::make_strand(ioc);
  boost::beast::tcp_stream stream(strand);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-kas-nossl");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  boost::json::object packet;
  KasStratum::jobCache jobCache;

  // Subscribe
  packet = KasStratum::stratumCall;
  packet["id"] = KasStratum::subscribe.id;
  packet["method"] = KasStratum::subscribe.method;
  packet["params"] = boost::json::array({minerName});
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum subscribe");
  }

  // Authorize
  packet = KasStratum::stratumCall;
  packet["id"] = KasStratum::authorize.id;
  packet["method"] = KasStratum::authorize.method;
  packet["params"] = boost::json::array({
      wallet + "." + worker,
      stratumPassword
  });
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum authorize");
  }

  KasStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  std::string packetBuffer;
  std::queue<std::string> submitQueue;
  boost::mutex submitMutex;
  std::atomic<bool> abort{false};

  auto process_write_queue = [&]() {
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          if (submitQueue.empty()) return;
          msg = std::move(submitQueue.front());
          submitQueue.pop();
        }
        
        beast::error_code wec;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
        boost::asio::write(stream, boost::asio::buffer(msg), wec);
        
        if (wec) {
          printf("error on write: %s\n", wec.message().c_str());
          fflush(stdout);
          abort = true;
          return;
        }
        
        if (!isDev) {
          KasStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
        }
      }
    });
  };

  bool submitThreadRunning = true;

  boost::thread subThread([&]() {
    while (!abort.load()) {
      boost::unique_lock<boost::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&] { return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;

      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          submitQueue.push(std::move(msg));
        }
        process_write_queue();
        
      } catch (const std::exception &e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        break;
      }
      *B = false;
      data_ready = false;
      boost::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  while (!ABORT_MINER && !abort.load()) {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try {
      if (KasStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count()
          - KasStratum::lastReceivedJobTime > KasStratum::jobTimeout)
      {
        setcolor(RED);
        printf("\nStratum session timed out\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t n = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      std::string newData(
          boost::asio::buffers_begin(response.data()),
          boost::asio::buffers_begin(response.data()) + n
      );
      response.consume(n);
      packetBuffer += newData;

      if (packetBuffer.size() > 1024 * 1024) {
        setcolor(RED);
        printf("\nPacket buffer overflow, disconnecting\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
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
            if (method == KasStratum::s_ping)
            {
              boost::json::object pong = {
                  {"id", rpc["id"].get_uint64()},
                  {"method", KasStratum::pong.method}
              };
              std::string pongMsg = boost::json::serialize(pong) + "\n";
              boost::asio::async_write(stream, boost::asio::buffer(pongMsg), yield[ec]);
              if (ec) {
                setForDisconnected(C, B, &abort, &data_ready, &cv);
                break;
              }
            }
            else {
              handleKasStratumPacket(rpc, &jobCache, isDev);
            }
          }
          else {
            handleKasStratumResponse(rpc, isDev);
          }
        }
        catch (const std::exception &e) {
          setcolor(RED);
          printf("\nParse error: %s\n", e.what());
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
        }
      }
    }
    catch (const std::exception &e) {
      setcolor(RED);
      printf("\nSession exception: %s\n", e.what());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }

    boost::this_thread::yield();
  }

  abort = true;
  cv.notify_all();
  
  if (submitThreadRunning) {
    subThread.interrupt();
    if (subThread.joinable()) {
      subThread.join();
    }
  }

  beast::error_code close_ec;
  stream.close();
}