#include "net.hpp"
#include "hex.h"

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

#include <xatum.h>
#include <xelis-hash/xelis-hash.hpp>

#include <atomic>
#include <queue>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

void xatumFailure(bool isDev) noexcept
{
  setcolor(RED);
  if (isDev)
    printf("DEV | ");
  printf("Xatum Disconnect\n");
  fflush(stdout);
  setcolor(BRIGHT_WHITE);
}

int handleXatumPacket(Xatum::packet xPacket, bool isDev)
{
  std::string command = xPacket.command;
  boost::json::value data = xPacket.data;
  int res = 0;

  if (command == Xatum::print)
  {
    // Safely compare strings
    std::string msgStr = std::string(data.at("msg").as_string());
    
    if (Xatum::accepted_msg == msgStr)
      accepted += !isDev;

    if (Xatum::stale_msg == msgStr)
      rejected += !isDev;

    int msgLevel = data.at("lvl").to_number<int64_t>();
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
      printf("Xatum VERBOSE: ");
      break;
    }

    printf("%s\n", msgStr.c_str());

    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }

  else if (command == Xatum::newJob)
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    int64_t *diff = isDev ? &difficultyDev : &difficulty;
    boost::json::value *J = isDev ? &devJob : &job;
    int64_t *h = isDev ? &devHeight : &ourHeight;

    std::string *B = isDev ? &devBlob : &currentBlob;

    // Safely extract blob string
    std::string newBlob = std::string(data.at("blob").as_string());
    
    if (newBlob == *B)
      return 0;
    
    *B = newBlob;

    Xatum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    if (!isDev)
    {
      setcolor(CYAN);
      printf("\nNew Xatum job received\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    *diff = data.at("diff").to_number<uint64_t>();

    (*J).as_object().emplace("miner_work", *B);

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
        printf("Connected to dev node: %s\n", devMiningProfile.host.c_str());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }

    *C = true;

    (*h)++;
    jobCounter++;
  }

  else if (!isDev && command == Xatum::success)
  {
    std::string msgStr = std::string(data.at("msg").as_string());
    if (msgStr == "ok")
    {
      printf("\nXatum: share accepted!\n");
      fflush(stdout);
      accepted++;
    }
    else
    {
      rejected++;
      setcolor(RED);
      printf("\nXatum Share Rejected: %s\n", msgStr.c_str());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
  else
  {
    printf("unknown command: %s\n", command.c_str());
    fflush(stdout);
  }

  return res;
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
  Xatum::lastReceivedJobTime = 0;
  ctx.set_options(boost::asio::ssl::context::default_workarounds |
                  boost::asio::ssl::context::no_sslv2 |
                  boost::asio::ssl::context::no_sslv3 |
                  boost::asio::ssl::context::no_tlsv1 |
                  boost::asio::ssl::context::no_tlsv1_1);

  beast::error_code ec;
  ctx.set_verify_mode(ssl::verify_none); // Accept self-signed certificates
  
  // Create strand for thread-safe operations
  auto strand = net::make_strand(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(strand, ctx);

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  // Set a timeout on the operation
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec)
    return fail(ec, "connect-xatum");

  // Set the SNI hostname
  if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
  {
    throw beast::system_error{
        static_cast<int>(::ERR_get_error()),
        boost::asio::error::get_ssl_category()};
  }

  // Perform the SSL handshake
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec)
    return fail(ec, "handshake-xatum");

  boost::json::object handshake_packet = {
      {"addr", wallet},
      {"work", worker},
      {"agent", std::string("tnn-miner ") + versionString},
      {"algos", boost::json::array{"xel/1"}}};

  std::string handshakeStr = "shake~" + boost::json::serialize(handshake_packet) + "\n";

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  size_t trans = boost::asio::async_write(stream, boost::asio::buffer(handshakeStr), yield[ec]);
  if (ec)
    return fail(ec, "Xatum C2S handshake");

  Xatum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  // Thread-safe queue for submissions
  std::queue<std::string> submitQueue;
  boost::mutex submitMutex;
  std::atomic<bool> abort{false};

  auto process_write_queue = [&]() {
    // Post to strand to process entire queue
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        
        // Get next message
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          if (submitQueue.empty()) {
            return;
          }
          msg = std::move(submitQueue.front());
          submitQueue.pop();
        }
        
        // Synchronous write (safe because we're in the strand)
        beast::error_code wec;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
        boost::asio::write(stream, boost::asio::buffer(msg), wec);
        
        if (wec) {
          printf("error on write: %s\n", wec.message().c_str());
          fflush(stdout);
          abort = true;
          return;
        }
      }
    });
  };

  bool submitThreadRunning = true;

  boost::thread subThread([&](){
    while(!abort.load()) {
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

  // Buffer for receiving data
  std::string packetBuffer;

  while (!ABORT_MINER && !abort.load())
  {
    bool *B = isDev ? &submittingDev : &submitting;
    try
    {
      if (Xatum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() - 
          Xatum::lastReceivedJobTime > Xatum::jobTimeout)
      {
        setcolor(RED);
        printf("\nXatum session timed out\n");
        setcolor(BRIGHT_WHITE);
        fflush(stdout);
        
        bool *C = isDev ? &devConnected : &isConnected;
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));

      size_t bytes = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec) {
        bool *C = isDev ? &devConnected : &isConnected;
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      if (bytes > 0)
      {
        // Extract exactly the bytes read
        std::string data(
            boost::asio::buffers_begin(response.data()),
            boost::asio::buffers_begin(response.data()) + bytes
        );
        response.consume(bytes);

        // Add to packet buffer
        packetBuffer += data;

        // Prevent unbounded buffer growth
        if (packetBuffer.size() > 1024 * 1024) {  // 1MB limit
          setcolor(RED);
          printf("\nPacket buffer overflow in xatum_session, disconnecting\n");
          setcolor(BRIGHT_WHITE);
          fflush(stdout);
          bool *C = isDev ? &devConnected : &isConnected;
          setForDisconnected(C, B, &abort, &data_ready, &cv);
          break;
        }

        // Process all complete lines
        size_t pos;
        while ((pos = packetBuffer.find('\n')) != std::string::npos) {
          std::string line = packetBuffer.substr(0, pos);
          packetBuffer.erase(0, pos + 1);

          if (line.empty()) continue;

          // Remove trailing carriage return if present
          if (!line.empty() && line.back() == '\r') {
            line.pop_back();
          }

          if (line == Xatum::pingPacket)
          {
            std::string pongMsg = Xatum::pongPacket;
            boost::asio::async_write(stream, boost::asio::buffer(pongMsg), yield[ec]);
            if (ec) {
              bool *C = isDev ? &devConnected : &isConnected;
              setForDisconnected(C, B, &abort, &data_ready, &cv);
              break;
            }
          }
          else
          {
            try {
              Xatum::packet xPacket = Xatum::parsePacket(line, "~");
              int r = handleXatumPacket(xPacket, isDev);
              // Error handling could go here if needed
            } catch (const std::exception &e) {
              setcolor(RED);
              printf("\nError parsing Xatum packet: %s\n", e.what());
              setcolor(BRIGHT_WHITE);
              fflush(stdout);
            }
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      printf("\nXatum session error: %s\n", e.what());
      setcolor(BRIGHT_WHITE);
      fflush(stdout);
      
      bool *C = isDev ? &devConnected : &isConnected;
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }
    
    boost::this_thread::yield();
  }

  // Clean shutdown
  abort = true;
  cv.notify_all();
  
  if (submitThreadRunning) {
    subThread.interrupt();
    if (subThread.joinable()) {
      subThread.join();
    }
  }

  // Graceful shutdown
  beast::error_code shutdown_ec;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(5));
  stream.async_shutdown(yield[shutdown_ec]);
  // Ignore shutdown errors
}