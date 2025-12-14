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

#include <xelis-hash/xelis-hash.hpp>

#include <atomic>
#include <queue>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

void xelis_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    net::yield_context yield,
    bool isDev)
{
  beast::error_code ec;
  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  // Create strand for thread-safe operations
  auto strand = net::make_strand(ioc);
  websocket::stream<beast::tcp_stream> ws(strand);

  fflush(stdout);

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(ws).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-xelis");

  fflush(stdout);

  // Update the host string
  host += ':' + port;

  // Set a decorator to change the User-Agent of the handshake
  ws.set_option(websocket::stream_base::decorator(
      [](websocket::request_type &req)
      {
        req.set(http::field::user_agent,
                std::string(BOOST_BEAST_VERSION_STRING) +
                    " websocket-client-coro");
      }));

  // Turn off the timeout on the tcp_stream
  beast::get_lowest_layer(ws).expires_never();

  // Set suggested timeout settings for the websocket
  ws.set_option(
      websocket::stream_base::timeout::suggested(
          beast::role_type::client));

  // Perform the websocket handshake
  std::stringstream ss;
  ss << "/getwork/" << wallet << "/" << worker;
  ws.async_handshake(host, ss.str().c_str(), yield[ec]);
  if (ec)
  {
    return fail(ec, "handshake-xelis");
  }

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

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
        beast::error_code ec;
        beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));
        ws.write(boost::asio::buffer(msg), ec);
        
        if (ec) {
          printf("error on write: %s\n", ec.message().c_str());
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

      printf("\nSubmit thread triggered");
      fflush(stdout);

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

  fflush(stdout);

  while (!ABORT_MINER && !abort.load())
  {
    bool *B = isDev ? &submittingDev : &submitting;
    try
    {
      buffer.clear();
      workInfo.str("");
      workInfo.clear();

      beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(180));
      ws.async_read(buffer, yield[ec]);
      if (!ec)
      {
        // handle getwork feed
        workInfo << beast::make_printable(buffer.data());

        boost::system::error_code jsonEc;
        boost::json::value response = boost::json::parse(workInfo.str(), jsonEc);
        if (!jsonEc)
        {
          if(response.is_string()) {
            std::string resp = std::string(response.as_string());
            if(resp == "block_accepted") {
              accepted++;
              setcolor(BRIGHT_YELLOW);
              if (!isDev) {
                printf("Block Accepted!\n");
                fflush(stdout);
              }
              setcolor(BRIGHT_WHITE);
            }
          } 
          else if(response.is_object() && response.as_object().contains("block_rejected")) {
            rejected++;
            setcolor(RED);
            if (!isDev) {
              std::string rejectReason = std::string(response.as_object()["block_rejected"].as_string());
              printf("Block Rejected: %s\n", rejectReason.c_str());
              fflush(stdout);
            }
            setcolor(BRIGHT_WHITE);
          }
          else if (response.is_object() && 
                   (response.as_object().contains("new_job") || 
                    response.as_object().contains("miner_work")))
          {
            boost::json::value workData;
            if (response.as_object().contains("new_job")) {
              workData = response.at("new_job");
            }

            int64_t newHeight = workData.at("height").to_number<int64_t>();
            int64_t currentHeight = isDev ? devHeight : ourHeight;

            if (newHeight != currentHeight)
            {
              if (isDev)
                devJob = workData;
              else
                job = workData;
              
              boost::json::value *J = isDev ? &devJob : &job;

              auto lasterror = (*J).as_object().if_contains("lasterror");
              if (nullptr != lasterror)
              {
                std::string errorStr = std::string(lasterror->as_string());
                std::cerr << "received error: " << errorStr << std::endl
                          << consoleLine << "v" << versionString << " ";
              }

              std::scoped_lock<boost::mutex> lockGuard(mutex);
              if (!isDev)
              {
                currentBlob = std::string((*J).at("miner_work").as_string());
                ourHeight++;
                
                std::string diffStr = std::string((*J).at("difficulty").as_string());
                difficulty = std::stoull(diffStr);

                if (!isConnected)
                {
                  setcolor(BRIGHT_YELLOW);
                  printf("Mining at: %s/getwork/%s/%s\n", host.c_str(), wallet.c_str(), worker.c_str());
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
                devBlob = std::string((*J).at("miner_work").as_string());
                devHeight++;
                
                std::string diffStr = std::string((*J).at("difficulty").as_string());
                difficultyDev = std::stoull(diffStr);

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
          }
        }
        else
        {
          setcolor(RED);
          printf("\nJSON parse error in xelis_session: %s\n", jsonEc.message().c_str());
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

  // Close websocket gracefully
  beast::error_code close_ec;
  ws.async_close(websocket::close_code::normal, yield[close_ec]);
  // Ignore close errors
}