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

#include <atomic>
#include <queue>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

void dero_session(
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
  
  // Create strand for thread-safe operations
  auto strand = net::make_strand(ioc);
  websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(strand, ctx);

  // Set a timeout on the operation
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(ws).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-dero");

  // Set SNI Hostname (many hosts need this to handshake successfully)
  if (!SSL_set_tlsext_host_name(
          ws.next_layer().native_handle(),
          host.c_str()))
  {
    ec = beast::error_code(static_cast<int>(::ERR_get_error()),
                          net::error::get_ssl_category());
    return fail(ec, "sni-hostname");
  }

  // Update the host string. This will provide the value of the
  // Host HTTP header during the WebSocket handshake.
  // See https://tools.ietf.org/html/rfc7230#section-5.4
  host += ':' + port;

  // Set a decorator to change the User-Agent of the handshake
  ws.set_option(websocket::stream_base::decorator(
      [](websocket::request_type &req)
      {
        req.set(http::field::user_agent,
                std::string(BOOST_BEAST_VERSION_STRING) +
                    " websocket-client-coro");
      }));

  // Perform the SSL/TLS handshake
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
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
  std::string url("/ws/" + wallet);
  if (isDev) {
    url += ".";
    if(!worker.empty()) {
      url += worker + "-tnn-dev";
    } else {
      url += "tnn-dev";
    }
  } else {
    if(!worker.empty()) {
      url += "." + worker;
    }
  }

  ws.async_handshake(host, url.c_str(), yield[ec]);
  if (ec)
  {
    // Try again without the 'worker' being appended. This is what local nodes require
    url = "/ws/" + wallet;
    ws.async_handshake(host, url.c_str(), yield[ec]);
    if (ec) {
      beast::error_code close_ec;
      ws.async_close(websocket::close_code::normal, yield[close_ec]);
      fail(ec, "handshake-dero");
      return;
    }
  }

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  // Thread-safe queue for submissions
  std::queue<std::string> submitQueue;
  std::mutex submitMutex;
  std::atomic<bool> abort{false};

  auto process_write_queue = [&]() {
    // Post to strand to process entire queue
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        
        // Get next message
        {
          std::lock_guard<std::mutex> qlock(submitMutex);
          if (submitQueue.empty()) {
            return;
          }
          msg = std::move(submitQueue.front());
          submitQueue.pop();
        }
        
        // Synchronous write (safe because we're in the strand)
        beast::error_code wec;
        beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
        ws.write(boost::asio::buffer(msg), wec);
        
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

  std::thread subThread([&](){
    while(!abort.load()) {
      std::unique_lock<std::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&]{ return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;

      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        
        // Queue the message for thread-safe sending
        {
          std::lock_guard<std::mutex> qlock(submitMutex);
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
      std::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  while (!ABORT_MINER && !abort.load())
  {
    bool *B = isDev ? &submittingDev : &submitting;
    try
    {
      buffer.clear();
      workInfo.str("");
      workInfo.clear();

      beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(60));
      ws.async_read(buffer, yield[ec]);
      if (!ec)
      {
        // handle getwork feed
        workInfo << beast::make_printable(buffer.data());

        boost::system::error_code jsonEc;
        boost::json::value workData = boost::json::parse(workInfo.str(), jsonEc);
        if (!jsonEc)
        {
          // Acquire lock BEFORE modifying job/devJob to prevent race condition
          std::scoped_lock<std::mutex> lockGuard(mutex);

          if (isDev)
            devJob = workData;
          else
            job = workData;

          boost::json::value *J = isDev ? &devJob : &job;

          // Safely extract lasterror
          if ((*J).as_object().contains("lasterror")) {
            std::string lastError = std::string((*J).at("lasterror").as_string());
            if (!lastError.empty())
            {
              std::cerr << "received error: " << lastError << std::endl
                        << consoleLine << versionString << " ";
            }
          }

          if (!isDev)
          {
            // Safely copy strings
            currentBlob = std::string((*J).at("blockhashing_blob").as_string());
            ourHeight = (*J).at("height").to_number<int64_t>();
            difficulty = (*J).at("difficultyuint64").to_number<int64_t>();
            accepted = (*J).at("miniblocks").to_number<int64_t>();
            rejected = (*J).at("rejected").to_number<int64_t>();

            if (!isConnected)
            {
              setcolor(BRIGHT_YELLOW);
              printf("Mining at: %s%s\n", host.c_str(), url.c_str());
              fflush(stdout);
              setcolor(CYAN);
              printf("Dev fee: %.2f%% of your total hashrate\n", devFee);
              fflush(stdout);
              setcolor(BRIGHT_WHITE);
            }
            isConnected = true;
            jobCounter++;
          }
          else
          {
            // Safely copy strings
            devBlob = std::string((*J).at("blockhashing_blob").as_string());
            devHeight = (*J).at("height").to_number<int64_t>();
            difficultyDev = (*J).at("difficultyuint64").to_number<int64_t>();

            if (!devConnected)
            {
              setcolor(CYAN);
              printf("Connected to dev node: %s\n", host.c_str());
              fflush(stdout);
              setcolor(BRIGHT_WHITE);
            }
            devConnected = true;
            jobCounter++;
          }
        }
        else
        {
          setcolor(RED);
          printf("\nJSON parse error in dero_session: %s\n", jsonEc.message().c_str());
          setcolor(BRIGHT_WHITE);
          fflush(stdout);
        }
      }
      else
      {
        bool *C = isDev ? &devConnected : &isConnected;
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      std::cout << "\nws error: " << e.what() << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      
      bool *C = isDev ? &devConnected : &isConnected;
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }
    std::this_thread::yield();
  }

  // Clean shutdown
  abort = true;
  cv.notify_all();
  
  if (submitThreadRunning) {
    if (subThread.joinable()) {
      subThread.join();
    }
  }

  // Close websocket gracefully
  beast::error_code close_ec;
  ws.async_close(websocket::close_code::normal, yield[close_ec]);
  // Ignore close errors
}