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

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

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
  websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(ioc, ctx);

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(ws).connect(endpoint);

  // Set SNI Hostname (many hosts need this to handshake successfully)
  if (!SSL_set_tlsext_host_name(
          ws.next_layer().native_handle(),
          host.c_str()))
  {
    ec = beast::error_code(static_cast<int>(::ERR_get_error()),
                          net::error::get_ssl_category());
    return fail(ec, "connect");
  }

  // Update the host string. This will provide the value of the
  // Host HTTP header during the WebSocket handshake.
  // See https://tools.ietf.org/html/rfc7230#section-5.4
  host += ':' + port;

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

  // Perform the SSL/TLS handshake
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
  if(!worker.empty()) {
    url += "." + worker;
  }

  ws.async_handshake(host, url.c_str(), yield[ec]);
  if (ec)
  {
    // Try again without the 'worker' being appended. This is what local nodes require
    url = "/ws/" + wallet;
    ws.async_handshake(host, url.c_str(), yield[ec]);
    if (ec) {
      ws.async_close(websocket::close_code::normal, yield[ec]);
      return fail(ec, "handshake");
    }
  }
  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;
  boost::json::error_code jsonEc;
  boost::json::value workData;

  bool submitThread = false;
  bool abort = false;

  boost::thread([&](){
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

  while (true)
  {
    try
    {
      bool *B = isDev ? &submittingDev : &submitting;
      buffer.clear();
      workInfo.str("");
      workInfo.clear();

      beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(60));
      ws.async_read(buffer, yield[ec]);
      if (!ec)
      {
        // handle getwork feed

        workInfo << beast::make_printable(buffer.data());

        // std::cout << workInfo.str() << std::endl;

        workData = boost::json::parse(workInfo.str(), jsonEc);
        if (!jsonEc)
        {
          // if ((isDev ? (workData.at("height") != devHeight) : (workData.at("height") != ourHeight)))
          // {
            // mutex.lock();
            if (isDev)
              devJob = workData;
            else
              job = workData;
            boost::json::value *J = isDev ? &devJob : &job;
            // mutex.unlock();

            if ((*J).at("lasterror") != "")
            {
              std::cerr << "received error: " << (*J).at("lasterror") << std::endl
                        << consoleLine << versionString << " ";
            }

            if (!isDev)
            {
              currentBlob = (*J).at("blockhashing_blob").as_string();
              //blockCounter = (*J).at("blocks");
              //miniBlockCounter = (*J).at("miniblocks");
              //rejected = (*J).at("rejected");
              //hashrate = (*J).at("difficultyuint64");
              ourHeight = (*J).at("height").to_number<int64_t>();
              difficulty = (*J).at("difficultyuint64").to_number<int64_t>();
              // printf("NEW JOB RECEIVED | Height: %d | Difficulty %" PRIu64 "\n", ourHeight, difficulty);
              accepted = (*J).at("miniblocks").to_number<int64_t>();
              rejected = (*J).at("rejected").to_number<int64_t>();
              if (!isConnected)
              {
                // mutex.lock();
                setcolor(BRIGHT_YELLOW);
                printf("Mining at: %s%s\n", host.c_str(), url.c_str());
                fflush(stdout);
                setcolor(CYAN);
                printf("Dev fee: %.2f", devFee);
                std::cout << "%" << std::endl;
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                // mutex.unlock();
              }
              isConnected = isConnected || true;
              jobCounter++;
            }
            else
            {
              difficultyDev = (*J).at("difficultyuint64").to_number<int64_t>();
              devBlob = (*J).at("blockhashing_blob").as_string();
              devHeight = (*J).at("height").to_number<int64_t>();
              if (!devConnected)
              {
                // mutex.lock();
                setcolor(CYAN);
                printf("Connected to dev node: %s\n", host.c_str());
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                // mutex.unlock();
              }
              devConnected = devConnected || true;
              jobCounter++;
            }
          // }
        }
      }
      else
      {
        bool *C = isDev ? &devConnected : &isConnected;
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        // printf("DISCONNECT at read\n");
        // fflush(stdout);

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
    }
    boost::this_thread::yield();
  }
}