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

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

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
    if (Xatum::accepted_msg.compare(data.at("msg").as_string()) == 0)
      accepted++;

    if (Xatum::stale_msg.compare(data.at("msg").as_string()) == 0)
      rejected++;

    int msgLevel = data.at("lvl").as_int64();
    if (msgLevel < Xatum::logLevel)
      return 0;

    printf("\n");
    if (isDev)
    {
      setcolor(CYAN);
      printf("DEV | ");
    }

    // mutex.lock();
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

    printf("%s\n", data.at("msg").as_string().c_str());

    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    // mutex.unlock();
  }

  else if (command == Xatum::newJob)
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    int64_t *diff = isDev ? &difficultyDev : &difficulty;
    boost::json::value *J = isDev ? &devJob : &job;
    int64_t *h = isDev ? &devHeight : &ourHeight;

    std::string *B = isDev ? &devBlob : &currentBlob;

    if (data.at("blob").as_string().compare(*B) == 0)
      return 0;
    *B = data.at("blob").as_string();

    Xatum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    // std::cout << data << std::endl;
    if (!isDev)
    {
      setcolor(CYAN);
      printf("\nNew Xatum job received\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    *diff = data.at("diff").as_uint64();

    (*J).as_object().emplace("miner_work", (*B).c_str());

    bool *C = isDev ? &devConnected : &isConnected;

    if (!*C)
    {
      if (!isDev)
      {
        // mutex.lock();
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
        fflush(stdout);
        setcolor(CYAN);
        printf("Dev fee: %.2f", devFee);
        std::cout << "%" << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        // mutex.unlock();
      }
      else
      {
        // mutex.lock();
        setcolor(CYAN);
        printf("Connected to dev node: %s\n", host.c_str());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        // mutex.unlock();
      }
    }

    *C = true;

    (*h)++;
    jobCounter++;
  }

  else if (!isDev && command == Xatum::success)
  {
    // std::cout << data << std::endl;
    if (data.at("msg").as_string() == "ok")
    {
      printf("Xatum: share accepted!");
      fflush(stdout);
      accepted++;
    }
    else
    {
      rejected++;
      setcolor(RED);
      printf("\nXatum Share Rejected: %s\n", data.at("msg").as_string().c_str());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
  else
  {
    printf("unknown command: %s\n", command.c_str());
  }

  return res;
}

void xatum_session(
    std::string hostProto,
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
  ctx.set_verify_mode(ssl::verify_none); // Accept self-signed certificates
  tcp::socket socket(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  // Set a timeout on the operation
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec)
    return fail(ec, "connect");

  // Set the SNI hostname
  if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
  {
    throw beast::system_error{
        static_cast<int>(::ERR_get_error()),
        boost::asio::error::get_ssl_category()};
  }

  // Perform the SSL handshake
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(300));
  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec)
    return fail(ec, "handshake-xatum");

  boost::json::object handshake_packet = {
      {"addr", wallet.c_str()},
      {"work", worker.c_str()},
      {"agent", (std::string("tnn-miner ") + versionString).c_str()},
      {"algos", boost::json::array{
                    "xel/1",
                }}};

  // std::string handshakeStr = handshake_packet.serialize();
  std::string handshakeStr = "shake~" + boost::json::serialize(handshake_packet) + "\n";

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // stream.async_write_some(boost::asio::buffer(handshakeStr, 1024), yield[ec]);
  // if (ec)
  //     return fail(ec, "write");

  size_t trans = boost::asio::async_write(stream, boost::asio::buffer(handshakeStr), yield[ec]);
  if (ec)
    return fail(ec, "Xatum C2S handshake");

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  Xatum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

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
        boost::json::object *S = &share;
        if (isDev)
          S = &devShare;

        boost::system::error_code ec;
        std::string msg = boost::json::serialize((*S)) + "\n";
        // std::cout << "sending in: " << msg << std::endl;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(1));
        boost::asio::async_write(stream, boost::asio::buffer(msg), [&](const boost::system::error_code& error, std::size_t bytes_transferred) {
          if (error) {
            printf("error on write: %s\n", error.message().c_str());
            fflush(stdout);
            abort = true;
          }
          // (*B) = false;
          // data_ready = false;
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
      //boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
      boost::this_thread::yield();
    }
    submitThread = false;
  });

  while (true)
  {
    bool *B = isDev ? &submittingDev : &submitting;
    try
    {
      if (
          Xatum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - Xatum::lastReceivedJobTime > Xatum::jobTimeout)
      {
        bool *C = isDev ? &devConnected : &isConnected;
        setForDisconnected(C, B, &abort, &data_ready, &cv);

        for (;;) {
          if (!submitThread) break;
          boost::this_thread::yield();
        }
        
        stream.shutdown();
        return fail(ec, "Xatum session timed out");
      }
      boost::asio::streambuf response;
      std::stringstream workInfo;
      beast::get_lowest_layer(stream).expires_after(std::chrono::milliseconds(45000));

      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      // if (ec && trans > 0)
      //   return fail(ec, "Xatum async_read_until");

      if (trans > 0)
      {
        std::string data = beast::buffers_to_string(response.data());
        // Consume the data from the buffer after processing it
        response.consume(trans);

        if (data.compare(Xatum::pingPacket) == 0)
        {
          // printf("pinged\n");
          boost::asio::async_write(stream, boost::asio::buffer(Xatum::pongPacket), yield[ec]);
        }
        else
        {
          Xatum::packet xPacket = Xatum::parsePacket(data, "~");
          int r = handleXatumPacket(xPacket, isDev);
          // if (r == -1) {
          //   bool *B = isDev ? &devConnected : &isConnected;
          //   (*B) = false;
          //   // return xatumFailure(isDev);
          // }
        }
      }
    }
    catch (const std::exception &e)
    {
      bool *C = isDev ? &devConnected : &isConnected;
      setForDisconnected(C, B, &abort, &data_ready, &cv);

      for (;;) {
        if (!submitThread) break;
        boost::this_thread::yield();
      }
      
      stream.shutdown();
      return fail(ec, "Xatum session error");
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  }

  // submission_thread.interrupt();
  stream.async_shutdown(yield[ec]);
}
