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

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

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
  websocket::stream<beast::tcp_stream> ws(ioc);

  fflush(stdout);

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(ws).connect(endpoint);
  fflush(stdout);


  // Update the host string. This will provide the value of the
  // Host HTTP header during the WebSocket handshake.
  // See https://tools.ietf.org/html/rfc7230#section-5.4
  host += ':' + port; //std::to_string(daemon.port());

  // Set a timeout on the operation
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));

  // Set a decorator to change the User-Agent of the handshake
  ws.set_option(websocket::stream_base::decorator(
      [](websocket::request_type &req)
      {
        req.set(http::field::user_agent,
                std::string(BOOST_BEAST_VERSION_STRING) +
                    " websocket-client-coro");
      }));

  // Turn off the timeout on the tcp_stream, because
  // the websocket stream has its own timeout system.
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
        bool err = false;
        boost::json::object *S = &share;
        if (isDev)
          S = &devShare;

        std::string msg = boost::json::serialize((*S)) + "\n";
        // std::cout << "sending in: " << msg << std::endl;
        beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(1));
        ws.async_write(boost::asio::buffer(msg), [&](const boost::system::error_code& error, std::size_t bytes_transferred) {
          if (error) {
            printf("error on write: %s\n", error.message().c_str());
            fflush(stdout);
            abort = true;
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

  fflush(stdout);

  while (!ABORT_MINER)
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

        // std::cout << "Received data: " << workInfo.str() << std::endl;
        boost::system::error_code jsonEc;
        boost::json::value response = boost::json::parse(workInfo.str(), jsonEc);
        if (!jsonEc)
        {
          if(response.is_string()) {
            std::string resp = std::string(response.as_string());
            if(resp.compare("block_accepted") == 0) {
              accepted++;
              setcolor(BRIGHT_YELLOW);
              if (!isDev) printf("Block Accepted!\n");
              fflush(stdout);
              setcolor(BRIGHT_WHITE);
            }
          } 
          else if(!response.as_object()["block_rejected"].is_null()) {
            rejected++;
            setcolor(RED);
            if (!isDev) printf("Block Rejected: %s\n", response.as_object()["block_rejected"].as_string().c_str());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
          else if (response.as_object().contains("new_job") || response.as_object().contains("miner_work"))
          {
            boost::json::value workData;
            if (response.as_object().contains("new_job")) {
              workData = response.at("new_job");
            //} else if (response.as_object().contains("miner_work")) {
            //  workData = response;
            }

            if ((isDev ? (workData.at("height").to_number<int64_t>() != devHeight) : (workData.at("height").to_number<int64_t>() != ourHeight)))
            {
              if (isDev)
                devJob = workData;
              else
                job = workData;
              boost::json::value *J = isDev ? &devJob : &job;

              auto lasterror = (*J).as_object().if_contains("lasterror");
              if (nullptr != lasterror)
              {
                std::cerr << "received error: " << (*lasterror).as_string() << std::endl
                          << consoleLine << "v" << versionString << " ";
              }

              std::scoped_lock<boost::mutex> lockGuard(mutex);
              if (!isDev)
              {
                currentBlob = (*J).at("miner_work").as_string();
                ourHeight++;
                difficulty = std::stoull(std::string((*J).at("difficulty").as_string().c_str()));

                if (!isConnected)
                {
                  // mutex.lock();
                  setcolor(BRIGHT_YELLOW);
                  printf("Mining at: %s/getwork/%s/%s\n", host.c_str(), wallet.c_str(), worker.c_str());
                  fflush(stdout);
                  setcolor(CYAN);
                  printf("Dev fee: %.2f%% of your total hashrate\n", devFee);
          
                  fflush(stdout);
                  setcolor(BRIGHT_WHITE);
                  // mutex.unlock();
                }
                isConnected = true;
                jobCounter++;
              }
              else
              {
                devBlob = (*J).at("miner_work").as_string();
                devHeight++;
                difficultyDev = std::stoull(std::string((*J).at("difficulty").as_string().c_str()));

                if (!devConnected)
                {
                  // mutex.lock();
                  setcolor(CYAN);
                  printf("Connected to dev node: %s\n", host.c_str());
                  fflush(stdout);
                  setcolor(BRIGHT_WHITE);
                  // mutex.unlock();
                }
                devConnected = true;
                jobCounter++;
              }
            }
          }
        }
      }
      else
      {
        bool *C = isDev ? &devConnected : &isConnected;
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        
        for (;;) {
          if (!submitThread) break;
          boost::this_thread::yield();
        }
        
        return fail(ec, "async_read");
      }
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      std::cout << "ws error: " << e.what() << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      // submission_thread.interrupt();
    }
    boost::this_thread::yield();
    if(ABORT_MINER) {
      bool *connPtr = isDev ? &devConnected : &isConnected;
      bool *submitPtr = isDev ? &submittingDev : &submitting;
      setForDisconnected(connPtr, submitPtr, &abort, &data_ready, &cv);
      ioc.stop();
    }
  }
  cv.notify_all();

  subThread.interrupt();
  subThread.join();
}