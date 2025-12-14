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
#include <xelis-hash/xelis-hash.hpp>

#include <queue>
#include <atomic>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

int handleXStratumPacket(boost::json::object packet, bool isDev)
{
  std::string M = std::string(packet["method"].as_string());
  if (M == XelisStratum::s_notify)
  {
    if (ourHeight > 0 && packet["params"].as_array()[4].get_bool() != true) {
      XelisStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
      return 0;
    }

    setcolor(CYAN);
    if (!isDev)
      printf("\nStratum: new job received\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    XelisStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    boost::json::value *JV = isDev ? &devJob : &job;
    boost::json::object J = (*JV).as_object();

    int64_t *h = isDev ? &devHeight : &ourHeight;

    // Safely extract strings to prevent dangling pointers
    std::string blobStr;
    if (!J["miner_work"].is_null()) {
      blobStr = std::string(J["miner_work"].as_string());
    }
    blobStr.resize(XELIS_TEMPLATE_SIZE * 2, '0');
    
    std::string jobIdStr = std::string(packet["params"].as_array()[0].as_string());
    std::string tsStr = std::string(packet["params"].as_array()[1].as_string());
    std::string headerStr = std::string(packet["params"].as_array()[2].as_string());
    
    int tsLen = tsStr.size();

    // Safely manipulate the blob string
    std::fill(blobStr.begin() + 64, blobStr.begin() + 80, '0');
    if (tsLen > 0 && tsLen <= 16) {
      std::copy(tsStr.begin(), tsStr.end(), blobStr.begin() + 64 + 16 - tsLen);
    }
    if (headerStr.size() >= 64) {
      std::copy(headerStr.begin(), headerStr.begin() + 64, blobStr.begin());
    }

    J["miner_work"] = blobStr;
    J["jobId"] = jobIdStr;

    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", miningProfile.host.c_str(), miningProfile.wallet.c_str());
        fflush(stdout);
        setcolor(CYAN);
        printf("Dev fee: %.2f%% of your hashrate\n", devFee);

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

    // Acquire lock BEFORE modifying job/devJob to prevent race condition
    {
      std::scoped_lock<boost::mutex> lockGuard(mutex);
      (*JV) = J;
      *C = true;
      (*h)++;
      jobCounter++;
    }
  }
  else if (M == XelisStratum::s_setDifficulty)
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    int64_t *d = isDev ? &difficultyDev : &difficulty;
    (*d) = packet["params"].as_array()[0].get_double();
    if ((*d) == 0)
      (*d) = packet["params"].as_array()[0].get_uint64();
  }
  else if (M == XelisStratum::s_setExtraNonce)
  {
    boost::json::value *JV = isDev ? &devJob : &job;
    boost::json::object J = (*JV).as_object();

    int64_t *h = isDev ? &devHeight : &ourHeight;

    // Safely extract strings
    std::string blobStr;
    if (!J["miner_work"].is_null()) {
      blobStr = std::string(J["miner_work"].as_string());
    }
    blobStr.resize(XELIS_TEMPLATE_SIZE * 2, '0');

    std::string enStr = std::string(packet["params"].as_array()[0].as_string());
    int enLen = enStr.size();

    // Safely manipulate the blob string
    std::fill(blobStr.begin() + 48, blobStr.begin() + 112, '0');
    if (enLen > 0 && enLen <= 64) {
      std::copy(enStr.begin(), enStr.end(), blobStr.begin() + 48);
    }

    J["miner_work"] = blobStr;

    // Acquire lock BEFORE modifying job/devJob to prevent race condition
    {
      std::scoped_lock<boost::mutex> lockGuard(mutex);
      (*JV) = J;
      (*h)++;
      jobCounter++;
    }
  }
  else if (M == XelisStratum::s_print)
  {
    int lLevel = packet.at("params").as_array()[0].to_number<int64_t>();
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
      
      std::string msgStr = std::string(packet.at("params").as_array()[1].as_string());
      printf("%s\n", msgStr.c_str());

      fflush(stdout);
      setcolor(BRIGHT_WHITE);

      return res;
    }
  }
  return 0;
}

int handleXStratumResponse(boost::json::object packet, bool isDev)
{
  if (!packet.contains("id")) return 0;
  
  int64_t id = packet["id"].to_number<int64_t>();

  switch (id)
  {
  case XelisStratum::subscribeID:
  {
    boost::json::value *JV = isDev ? &devJob : &job;
    boost::json::object J = (*JV).as_object();
    
    // Initialize blob if needed
    std::string blobStr;
    if (J["miner_work"].is_null())
    {
      blobStr.resize(XELIS_TEMPLATE_SIZE * 2, '0');
    }
    else
    {
      blobStr = std::string(J["miner_work"].as_string());
      blobStr.resize(XELIS_TEMPLATE_SIZE * 2, '0');
    }

    // Safely extract response data
    std::string extraNonceStr = std::string(packet["result"].get_array()[1].get_string());
    int enLen = packet["result"].get_array()[2].get_int64();
    std::string pubKeyStr = std::string(packet["result"].get_array()[3].get_string());

    // Safely manipulate the blob string
    std::fill(blobStr.begin() + 96, blobStr.begin() + 160, '0');
    if (enLen > 0 && extraNonceStr.size() >= static_cast<size_t>(enLen * 2)) {
      std::copy(extraNonceStr.begin(), extraNonceStr.begin() + (enLen * 2), blobStr.begin() + 96);
    }
    if (pubKeyStr.size() >= 64) {
      std::copy(pubKeyStr.begin(), pubKeyStr.begin() + 64, blobStr.begin() + 160);
    }

    J["miner_work"] = blobStr;

    // Acquire lock BEFORE modifying job/devJob to prevent race condition
    {
      std::scoped_lock<boost::mutex> lockGuard(mutex);
      (*JV) = J;
    }
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
      accepted += !isDev;
      std::cout << "Stratum: share accepted" << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    else
    {
      rejected += !isDev;
      if (!isDev)
        setcolor(RED);
      
      std::string errorMsg = "Unknown error";
      if (packet.contains("error") && packet.at("error").is_object()) {
        if (packet.at("error").get_object().contains("message")) {
          errorMsg = std::string(packet.at("error").get_object()["message"].get_string());
        }
      }
      std::cout << "Stratum: share rejected: " << errorMsg << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    break;
  }
  }
  return 0;
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
  XelisStratum::lastReceivedJobTime = 0;
  beast::error_code ec;
  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  // Create strand for thread-safe operations
  auto strand = net::make_strand(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(strand, ctx);

  ctx.set_verify_mode(ssl::verify_none);
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect");

  if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
    throw beast::system_error(static_cast<int>(::ERR_get_error()), boost::asio::error::get_ssl_category());

  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec) return fail(ec, "handshake-xelis-strat");

  // Thread-safe queue for submissions
  std::queue<std::string> submitQueue;
  boost::mutex submitMutex;
  std::atomic<bool> abort{false};
  std::atomic<bool> writeInProgress{false};

  auto send_json = [&](const boost::json::object &obj) {
    std::string msg = boost::json::serialize(obj) + "\n";
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) fail(ec, "send_json");
  };

  auto send_and_recv_line = [&]() -> std::string {
    boost::asio::streambuf response;
    boost::asio::async_read_until(stream, response, "\n", yield[ec]);
    if (ec) return "";
    std::string result = beast::buffers_to_string(response.data());
    response.consume(result.size());
    return result;
  };

  // Process write queue (called from strand)
  auto process_write_queue = [&]() {
    // Try to acquire write lock
    bool expected = false;
    if (!writeInProgress.compare_exchange_strong(expected, true)) {
      return; // Already processing
    }
    
    // Post to strand to process queue
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        
        // Get next message
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          if (submitQueue.empty()) {
            writeInProgress = false;
            return;
          }
          msg = std::move(submitQueue.front());
          submitQueue.pop();
        }
        
        // Synchronous write within the strand (safe because strand serializes)
        beast::error_code ec;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
        boost::asio::write(stream, boost::asio::buffer(msg), ec);
        
        if (ec) {
          printf("error on write: %s\n", ec.message().c_str());
          fflush(stdout);
          abort = true;
          writeInProgress = false;
          return;
        }
      }
      
      writeInProgress = false;
    });
  };

  // Subscribe
  boost::json::object subscribe = XelisStratum::stratumCall;
  subscribe["id"] = XelisStratum::subscribe.id;
  subscribe["method"] = XelisStratum::subscribe.method;
  subscribe["params"] = { "tnn-miner/" + std::string(versionString), {"xel/v3"} };
  send_json(subscribe);
  auto subResStr = send_and_recv_line();
  if (!subResStr.empty()) {
    auto subResJson = boost::json::parse(subResStr).as_object();
    handleXStratumResponse(subResJson, isDev);
  }

  // Authorize
  boost::json::object auth = XelisStratum::stratumCall;
  auth["id"] = XelisStratum::authorize.id;
  auth["method"] = XelisStratum::authorize.method;
  auth["params"] = { wallet, worker, isDev ? "d=10000" : stratumPassword };
  send_json(auth);

  XelisStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

  std::string packetBuffer;
  bool submitThreadRunning = true;

  boost::thread submitThread([&](){
    while (!abort.load()) {
      boost::unique_lock<boost::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&]{ return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;

      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        
        // Queue the message for thread-safe sending
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          submitQueue.push(std::move(msg));
        }
        
        // Trigger write processing
        process_write_queue();
        
      } catch (const std::exception &e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        setcolor(BRIGHT_WHITE);
        fflush(stdout);
        break;
      } catch (...) {
        setcolor(RED);
        printf("\nSubmit thread unknown error\n");
        setcolor(BRIGHT_WHITE);
        fflush(stdout);
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
      if (XelisStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() -
          XelisStratum::lastReceivedJobTime > XelisStratum::jobTimeout) {
        setcolor(RED); 
        printf("\nStratum timeout\n"); 
        setcolor(BRIGHT_WHITE);
        fflush(stdout);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t bytes = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      // Fix: Only get the exact bytes read, not all buffer data
      std::string newData(
        boost::asio::buffers_begin(response.data()),
        boost::asio::buffers_begin(response.data()) + bytes
      );
      response.consume(bytes);
      packetBuffer += newData;

      // Prevent unbounded buffer growth
      if (packetBuffer.size() > 1024 * 1024) {  // 1MB limit
        setcolor(RED);
        printf("\nPacket buffer overflow, disconnecting\n");
        setcolor(BRIGHT_WHITE);
        fflush(stdout);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      size_t pos;
      while ((pos = packetBuffer.find('\n')) != std::string::npos) {
        std::string line = packetBuffer.substr(0, pos);
        packetBuffer.erase(0, pos + 1);
        
        if (line.empty()) continue;
        
        if (line == XelisStratum::k1ping) {
          std::string pongMsg = XelisStratum::k1pong;
          boost::asio::async_write(stream, boost::asio::buffer(pongMsg), yield[ec]);
          if (ec) {
            setForDisconnected(C, B, &abort, &data_ready, &cv);
            break;
          }
        } else {
          try {
            auto sRPC = boost::json::parse(line).as_object();
            if (sRPC.contains("method")) {
              std::string method = std::string(sRPC["method"].as_string());
              if (method == XelisStratum::s_ping) {
                boost::json::object pong = {
                  {"id", sRPC["id"].get_uint64()}, 
                  {"method", XelisStratum::pong.method}
                };
                send_json(pong);
              } else {
                handleXStratumPacket(sRPC, isDev);
              }
            } else {
              handleXStratumResponse(sRPC, isDev);
            }
          } catch (const std::exception &e) {
            setcolor(RED);
            printf("\nJSON parse error: %s\n", e.what());
            setcolor(BRIGHT_WHITE);
            fflush(stdout);
          }
        }
      }

    } catch (const std::exception &e) {
      setcolor(RED); 
      std::cerr << "\nException in stratum loop: " << e.what() << std::endl; 
      setcolor(BRIGHT_WHITE);
      fflush(stdout);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }
    boost::this_thread::yield();
  }

  // Clean shutdown
  abort = true;
  cv.notify_all();
  
  if (submitThreadRunning) {
    submitThread.interrupt();
    if (submitThread.joinable()) {
      submitThread.join();
    }
  }
  
  beast::error_code shutdown_ec;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(5));
  stream.async_shutdown(yield[shutdown_ec]);
  // Ignore shutdown errors - connection may already be closed
  beast::get_lowest_layer(stream).close();
}

void xelis_stratum_session_nossl(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  XelisStratum::lastReceivedJobTime = 0;
  beast::error_code ec;
  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  // Create strand for thread-safe operations
  auto strand = net::make_strand(ioc);
  boost::beast::tcp_stream stream(strand);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect");

  // Thread-safe queue for submissions
  std::queue<std::string> submitQueue;
  boost::mutex submitMutex;
  std::atomic<bool> abort{false};
  std::atomic<bool> writeInProgress{false};

  auto send_json = [&](const boost::json::object &obj) {
    std::string msg = boost::json::serialize(obj) + "\n";
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) fail(ec, "send_json");
  };

  auto send_and_recv_line = [&]() -> std::string {
    boost::asio::streambuf response;
    boost::asio::async_read_until(stream, response, "\n", yield[ec]);
    if (ec) return "";
    std::string result = beast::buffers_to_string(response.data());
    response.consume(result.size());
    return result;
  };

  // Process write queue (called from strand)
  auto process_write_queue = [&]() {
    // Try to acquire write lock
    bool expected = false;
    if (!writeInProgress.compare_exchange_strong(expected, true)) {
      return; // Already processing
    }
    
    // Post to strand to process queue
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        
        // Get next message
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          if (submitQueue.empty()) {
            writeInProgress = false;
            return;
          }
          msg = std::move(submitQueue.front());
          submitQueue.pop();
        }
        
        // Synchronous write within the strand (safe because strand serializes)
        beast::error_code ec;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
        boost::asio::write(stream, boost::asio::buffer(msg), ec);
        
        if (ec) {
          printf("error on write: %s\n", ec.message().c_str());
          fflush(stdout);
          abort = true;
          writeInProgress = false;
          return;
        }
      }
      
      writeInProgress = false;
    });
  };

  // Subscribe
  boost::json::object subscribe = XelisStratum::stratumCall;
  subscribe["id"] = XelisStratum::subscribe.id;
  subscribe["method"] = XelisStratum::subscribe.method;
  subscribe["params"] = { "tnn-miner/" + std::string(versionString), {"xel/v3"} };
  send_json(subscribe);

  auto subResStr = send_and_recv_line();
  if (!subResStr.empty()) {
    auto subResJson = boost::json::parse(subResStr).as_object();
    handleXStratumResponse(subResJson, isDev);
  }

  // Authorize
  boost::json::object auth = XelisStratum::stratumCall;
  auth["id"] = XelisStratum::authorize.id;
  auth["method"] = XelisStratum::authorize.method;
  auth["params"] = { wallet, worker, isDev ? "d=10000" : stratumPassword };
  send_json(auth);

  XelisStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

  std::string packetBuffer;
  bool submitThreadRunning = true;

  boost::thread submitThread([&](){
    while (!abort.load()) {
      boost::unique_lock<boost::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&]{ return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;

      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        
        // Queue the message for thread-safe sending
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          submitQueue.push(std::move(msg));
        }
        
        // Trigger write processing
        process_write_queue();
        
      } catch (const std::exception &e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        setcolor(BRIGHT_WHITE);
        fflush(stdout);
        break;
      } catch (...) {
        setcolor(RED);
        printf("\nSubmit thread unknown error\n");
        setcolor(BRIGHT_WHITE);
        fflush(stdout);
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
      if (XelisStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() -
          XelisStratum::lastReceivedJobTime > XelisStratum::jobTimeout) {
        setcolor(RED); 
        printf("\nStratum timeout\n"); 
        setcolor(BRIGHT_WHITE);
        fflush(stdout);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t bytes = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      // Fix: Only get the exact bytes read, not all buffer data
      std::string newData(
        boost::asio::buffers_begin(response.data()),
        boost::asio::buffers_begin(response.data()) + bytes
      );
      response.consume(bytes);
      packetBuffer += newData;

      // Prevent unbounded buffer growth
      if (packetBuffer.size() > 1024 * 1024) {  // 1MB limit
        setcolor(RED);
        printf("\nPacket buffer overflow, disconnecting\n");
        setcolor(BRIGHT_WHITE);
        fflush(stdout);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      size_t pos;
      while ((pos = packetBuffer.find('\n')) != std::string::npos) {
        std::string line = packetBuffer.substr(0, pos);
        packetBuffer.erase(0, pos + 1);

        if (line.empty()) continue;

        if (line == XelisStratum::k1ping) {
          std::string pongMsg = XelisStratum::k1pong;
          boost::asio::async_write(stream, boost::asio::buffer(pongMsg), yield[ec]);
          if (ec) {
            setForDisconnected(C, B, &abort, &data_ready, &cv);
            break;
          }
        } else {
          try {
            auto sRPC = boost::json::parse(line).as_object();
            if (sRPC.contains("method")) {
              std::string method = std::string(sRPC["method"].as_string());
              if (method == XelisStratum::s_ping) {
                boost::json::object pong = {
                  {"id", sRPC["id"].get_uint64()}, 
                  {"method", XelisStratum::pong.method}
                };
                send_json(pong);
              } else {
                handleXStratumPacket(sRPC, isDev);
              }
            } else {
              handleXStratumResponse(sRPC, isDev);
            }
          } catch (const std::exception &e) {
            setcolor(RED);
            printf("\nJSON parse error: %s\n", e.what());
            setcolor(BRIGHT_WHITE);
            fflush(stdout);
          }
        }
      }

    } catch (const std::exception &e) {
      setcolor(RED); 
      std::cerr << "\nException in stratum loop: " << e.what() << std::endl; 
      setcolor(BRIGHT_WHITE);
      fflush(stdout);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }

    boost::this_thread::yield();

    if (ABORT_MINER) {
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      ioc.stop();
      break;
    }
  }

  // Clean shutdown
  abort = true;
  cv.notify_all();
  
  if (submitThreadRunning) {
    submitThread.interrupt();
    if (submitThread.joinable()) {
      submitThread.join();
    }
  }
  
  beast::error_code shutdown_ec;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(5));
  beast::get_lowest_layer(stream).close();
}