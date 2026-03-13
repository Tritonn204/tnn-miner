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

#include <atomic>
#include <queue>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

std::atomic<bool> randomx_ready = false;
std::atomic<bool> randomx_ready_dev = false;

std::atomic<bool> needsDatasetUpdate = false;

std::mutex dsMutex;

uint64_t diff_numerator = boost_swap_impl::stoull("0x100000001", nullptr, 16);

static uint64_t rx_targetToDifficulty(const char* target) {
  uint32_t targetInt = boost_swap_impl::stoul(target, nullptr, 16);
  targetInt = __builtin_bswap32(targetInt);
  uint64_t diff = diff_numerator / targetInt;
  return diff;
}

int handleRandomXStratumPacket(boost::json::object packet, bool isDev) {
  std::string M = std::string(packet["method"].as_string());
  
  if (M == rx0Stratum::s_job) {
    std::scoped_lock<std::mutex> lockGuard(mutex);
    if (!packet["error"].is_null()) return 1;

    boost::json::object newJob = packet["params"].as_object();

    if (!datasetInitInProgress.load()) {
      setcolor(isDev ? CYAN : BRIGHT_WHITE);
      if (!isDev)
        printf("\nStratum: new job received\n");
      else
        printf("\nDEV Stratum: new job received\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    rx0Stratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    boost::json::value *JV = isDev ? &devJob : &job;
    int64_t *h = isDev ? &devHeight : &ourHeight;

    std::string targetStr = std::string(newJob.at("target").as_string());
    if (!isDev) 
      difficulty = rx_targetToDifficulty(targetStr.c_str());
    else 
      difficultyDev = rx_targetToDifficulty(targetStr.c_str());
    
    updateVM(newJob, isDev);

    (*JV) = newJob;

    (*h)++;
    jobCounter++;
  }
  else if (M == rx0Stratum::s_print) {
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
        if (!isDev) setcolor(BRIGHT_WHITE);
        printf("Stratum INFO: ");
        break;
      case rx0Stratum::STRATUM_WARN:
        if (!isDev) setcolor(BRIGHT_YELLOW);
        printf("Stratum WARNING: ");
        break;
      case rx0Stratum::STRATUM_ERROR:
        if (!isDev) setcolor(RED);
        printf("Stratum ERROR: ");
        res = -1;
        break;
      case rx0Stratum::STRATUM_DEBUG:
        break;
      }
      
      std::string msgStr = std::string(packet.at("params").as_array()[1].as_string());
      printf("%s\n", msgStr.c_str());

      fflush(stdout);
      setcolor(BRIGHT_WHITE);

      return res;
    }
  }
  return 0;
}

int handleRandomXStratumResponse(boost::json::object packet, bool isDev)
{
  if (!packet.contains("id")) return 0;
  
  int64_t id = packet["id"].to_number<int64_t>();
  
  switch (id)
  {
  case rx0Stratum::loginID:
  {
    std::scoped_lock<std::mutex> lockGuard(mutex);
    boost::json::value *JV = isDev ? &devJob : &job;

    if (!packet["error"].is_null())
    {
      std::string errorMsg = std::string(packet["error"].as_object()["message"].as_string());
      setcolor(RED);
      printf("Stratum Error: %s\n", errorMsg.c_str());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return 1;
    }

    boost::json::object res = packet["result"].as_object();
    boost::json::object newJob = res["job"].as_object();

    std::string &l_ID = isDev ? randomx_login_dev : randomx_login;
    l_ID = std::string(res.at("id").as_string());

    (*JV) = newJob;

    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
        std::string targetStr = std::string(newJob.at("target").as_string());
        difficulty = rx_targetToDifficulty(targetStr.c_str());
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

    updateVM(newJob, isDev);

    *C = true;
  }
  break;
  default:
  {
    if (!SubmitTracker::isSubmitId(id)) break;
    int submitDevice = submitTracker.resolve(id);
    printf("\n");
    if (isDev)
    {
      setcolor(CYAN);
      printf("DEV | ");
    }

    bool shareAccepted = false;
    std::string errorMessage;

    if (!packet["error"].is_null())
    {
      shareAccepted = false;
      auto errorObj = packet["error"].as_object();
      if (errorObj.contains("message"))
      {
        errorMessage = std::string(errorObj["message"].as_string());
      }
      else
      {
        errorMessage = "Unknown error";
      }
    }
    else if (!packet["result"].is_null())
    {
      auto &result = packet["result"];

      if (result.is_bool())
      {
        shareAccepted = result.as_bool();
      }
      else if (result.is_object())
      {
        auto resultObj = result.as_object();
        if (resultObj.contains("status"))
        {
          std::string status = std::string(resultObj["status"].as_string());
          shareAccepted = (status == "OK" || status == "ok" || status == "accepted");
        }
        else
        {
          shareAccepted = true;
        }
      }
      else if (result.is_string())
      {
        std::string resultStr = std::string(result.as_string());
        shareAccepted = (resultStr == "OK" || resultStr == "ok" || resultStr == "accepted");
      }
      else
      {
        shareAccepted = true;
      }
    }
    else
    {
      shareAccepted = false;
      errorMessage = "Invalid response: both result and error are null";
    }

    if (shareAccepted)
    {
      accepted += !isDev;
      if (!isDev) recordDeviceShare(submitDevice, true);
      if (!isDev) setcolor(BRIGHT_YELLOW);
      std::cout << "Stratum share accepted" << std::endl;
    }
    else
    {
      rejected += !isDev;
      if (!isDev) recordDeviceShare(submitDevice, false);
      if (!isDev) setcolor(RED);
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
  rx0Stratum::lastReceivedJobTime = 0;
  ctx.set_options(boost::asio::ssl::context::default_workarounds |
                  boost::asio::ssl::context::no_sslv2 |
                  boost::asio::ssl::context::no_sslv3 |
                  boost::asio::ssl::context::no_tlsv1 |
                  boost::asio::ssl::context::no_tlsv1_1);

  beast::error_code ec;

  auto endpoint = resolve_host(wsMutex, ioc, yield, sessionHost, port);
  
  // Strand for thread-safe I/O
  auto strand = net::make_strand(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(strand, ctx);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-rx0-ssl");

  if (!SSL_set_tlsext_host_name(stream.native_handle(), sessionHost.c_str()))
  {
    throw beast::system_error{
        static_cast<int>(::ERR_get_error()),
        boost::asio::error::get_ssl_category()};
  }

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec) return fail(ec, "handshake-rx0-ssl");

  boost::json::object packet = rx0Stratum::stratumCall;
  packet["id"] = rx0Stratum::login.id;
  packet["method"] = rx0Stratum::login.method;

  std::string userAgent = "tnn-miner/" + std::string(versionString);

  boost::json::object loginParams = {
      {"login", wallet},
      {"pass", stratumPassword},
      {"rigid", worker},
      {"agent", userAgent}
  };

  packet["params"] = loginParams;
  std::string login = boost::json::serialize(packet) + "\n";

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  boost::asio::async_write(stream, boost::asio::buffer(login), yield[ec]);
  if (ec) return fail(ec, "Stratum login");

  rx0Stratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  std::string packetBuffer;
  
  // Thread-safe submit queue
  std::queue<std::string> submitQueue;
  std::mutex submitMutex;
  std::atomic<bool> abort{false};

  auto process_write_queue = [&]() {
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        {
          std::lock_guard<std::mutex> qlock(submitMutex);
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
          rx0Stratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
        }
      }
    });
  };

  bool submitThreadRunning = true;

  std::thread subThread([&](){
    while (!abort.load()) {
      std::unique_lock<std::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&]{ return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;
      
      try {
        boost::json::object &S = isDev ? devShare : share;
        hoist_rpc_id(S);
        std::string msg = boost::json::serialize(S) + "\n";
        
        {
          std::lock_guard<std::mutex> qlock(submitMutex);
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
      std::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  // ORIGINAL cache thread synchronization - unchanged
  std::thread cacheThread([&]() {
    while(!abort.load()) {
      std::unique_lock<std::mutex> lock(dsMutex);
      
      cv.wait(lock, [&]{ 
        return needsDatasetUpdate.load() || abort.load(); 
      });
      
      if (abort.load()) break;
      
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
      
      std::this_thread::yield();
    }
  });

  while (!ABORT_MINER && !abort.load())
  {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;
    
    try
    {
      if (rx0Stratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() - 
          rx0Stratum::lastReceivedJobTime > rx0Stratum::jobTimeout)
      {
        setcolor(RED);
        printf("\nStratum timeout\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      
      if (ec)
      {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      if (trans > 0)
      {
        // Extract exact bytes read
        std::string newData(
            boost::asio::buffers_begin(response.data()),
            boost::asio::buffers_begin(response.data()) + trans
        );
        response.consume(trans);
        packetBuffer += newData;

        // Buffer overflow protection
        if (packetBuffer.size() > 1024 * 1024) {
          setcolor(RED);
          printf("\nPacket buffer overflow, disconnecting\n");
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
          setForDisconnected(C, B, &abort, &data_ready, &cv);
          break;
        }

        size_t pos;
        while ((pos = packetBuffer.find('\n')) != std::string::npos)
        {
          std::string line = packetBuffer.substr(0, pos);
          packetBuffer.erase(0, pos + 1);

          if (line.empty()) continue;

          try
          {
            boost::json::object sRPC = boost::json::parse(line).as_object();
            
            if (sRPC.contains("method"))
            {
              std::string method = std::string(sRPC.at("method").as_string());
              if (method == rx0Stratum::s_ping)
              {
                boost::json::object pong = {
                    {"id", sRPC.at("id").get_uint64()},
                    {"method", rx0Stratum::pong.method}
                };
                std::string pongPacket = boost::json::serialize(pong) + "\n";
                boost::asio::async_write(stream, boost::asio::buffer(pongPacket), yield[ec]);
                if (ec)
                {
                  setForDisconnected(C, B, &abort, &data_ready, &cv);
                  break;
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
            printf("\nParse error: %s\n", e.what());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      printf("\nSession exception: %s\n", e.what());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }

    std::this_thread::yield();
  }

  // Clean shutdown
  abort = true;
  cv.notify_all();

  if (cacheThread.joinable()) {
    cacheThread.join();
  }

  if (submitThreadRunning) {
    if (subThread.joinable()) {
      subThread.join();
    }
  }

  beast::error_code shutdown_ec;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(5));
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
  rx0Stratum::lastReceivedJobTime = 0;
  beast::error_code ec;

  auto endpoint = resolve_host(wsMutex, ioc, yield, sessionHost, port);
  
  // Strand for thread-safe I/O
  auto strand = net::make_strand(ioc);
  boost::beast::tcp_stream stream(strand);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-rx0-nossl");

  boost::json::object packet = rx0Stratum::stratumCall;
  packet["id"] = rx0Stratum::login.id;
  packet["method"] = rx0Stratum::login.method;

  std::string userAgent = "tnn-miner/" + std::string(versionString);

  boost::json::object loginParams = {
      {"login", wallet},
      {"pass", stratumPassword},
      {"rigid", worker},
      {"agent", userAgent}
  };

  packet["params"] = loginParams;
  std::string login = boost::json::serialize(packet) + "\n";

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  boost::asio::async_write(stream, boost::asio::buffer(login), yield[ec]);
  if (ec) return fail(ec, "Stratum login");

  rx0Stratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  std::string packetBuffer;
  
  // Thread-safe submit queue
  std::queue<std::string> submitQueue;
  std::mutex submitMutex;
  std::atomic<bool> abort{false};

  auto process_write_queue = [&]() {
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        {
          std::lock_guard<std::mutex> qlock(submitMutex);
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
          rx0Stratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
        }
      }
    });
  };

  bool submitThreadRunning = true;

  std::thread subThread([&](){
    while (!abort.load()) {
      std::unique_lock<std::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&]{ return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;
      
      try {
        boost::json::object &S = isDev ? devShare : share;
        hoist_rpc_id(S);
        std::string msg = boost::json::serialize(S) + "\n";
        
        {
          std::lock_guard<std::mutex> qlock(submitMutex);
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
      std::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  // ORIGINAL cache thread synchronization - unchanged (uses mutex, not dsMutex)
  std::thread cacheThread([&]() {
    while(!abort.load()) {
      std::unique_lock<std::mutex> lock(mutex);
      
      cv.wait(lock, [&]{ 
        return needsDatasetUpdate.load() || abort.load(); 
      });
      
      if (abort.load()) break;
      
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
      
      std::this_thread::yield();
    }
  });

  while (!ABORT_MINER && !abort.load())
  {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;
    
    try
    {
      if (rx0Stratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() - 
          rx0Stratum::lastReceivedJobTime > rx0Stratum::jobTimeout)
      {
        setcolor(RED);
        printf("\nStratum timeout\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      
      if (ec)
      {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      if (trans > 0)
      {
        // Extract exact bytes read
        std::string newData(
            boost::asio::buffers_begin(response.data()),
            boost::asio::buffers_begin(response.data()) + trans
        );
        response.consume(trans);
        packetBuffer += newData;

        // Buffer overflow protection
        if (packetBuffer.size() > 1024 * 1024) {
          setcolor(RED);
          printf("\nPacket buffer overflow, disconnecting\n");
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
          setForDisconnected(C, B, &abort, &data_ready, &cv);
          break;
        }

        size_t pos;
        while ((pos = packetBuffer.find('\n')) != std::string::npos)
        {
          std::string line = packetBuffer.substr(0, pos);
          packetBuffer.erase(0, pos + 1);

          if (line.empty()) continue;

          try
          {
            boost::json::object sRPC = boost::json::parse(line).as_object();
            
            if (sRPC.contains("method"))
            {
              std::string method = std::string(sRPC.at("method").as_string());
              if (method == rx0Stratum::s_ping)
              {
                boost::json::object pong = {
                    {"id", sRPC.at("id").get_uint64()},
                    {"method", rx0Stratum::pong.method}
                };
                std::string pongPacket = boost::json::serialize(pong) + "\n";
                boost::asio::async_write(stream, boost::asio::buffer(pongPacket), yield[ec]);
                if (ec)
                {
                  setForDisconnected(C, B, &abort, &data_ready, &cv);
                  break;
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
            printf("\nParse error: %s\n", e.what());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      printf("\nSession exception: %s\n", e.what());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }

    std::this_thread::yield();
  }

  // Clean shutdown
  abort = true;
  cv.notify_all();

  if (cacheThread.joinable()) {
    cacheThread.join();
  }

  if (submitThreadRunning) {
    if (subThread.joinable()) {
      subThread.join();
    }
  }

  beast::error_code close_ec;
  stream.close();
}