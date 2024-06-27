#include "tnn-common.hpp"
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

#include <nlohmann/json.hpp>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

using json = nlohmann::json;

boost::mutex wsMutex;

/* Start definitions from net.hpp */
json job = json({});
json devJob = json({});

std::string currentBlob;
std::string devBlob;

boost::json::object share = {};
boost::json::object devShare = {};

bool submitting = false;
bool submittingDev = false;
/* End definitions from net.hpp */

// Report a failure
void fail(beast::error_code ec, char const *what) noexcept
{
  wsMutex.lock();
  setcolor(RED);
  std::cerr << '\n'
            << what << ": " << ec.message() << "\n";
  setcolor(BRIGHT_WHITE);
  wsMutex.unlock();
}

tcp::endpoint resolve_host(boost::mutex &wsMutex, net::io_context &ioc, net::yield_context yield, std::string host, std::string port) {
  beast::error_code ec;

  int addrCount = 0;
  net::ip::address ip_address;

  // If the specified host/pool is not in IP address form, resolve to acquire the IP address
#if !defined(_WIN32) && !defined(__APPLE__)
  boost::asio::ip::address::from_string(host, ec);
  if (ec)
  {
    // Using cpp-dns to circumvent the issues cause by combining static linking and getaddrinfo()
    // A second io_context is used to enable std::promise
    net::io_context ioc2;
    std::string ip;
    std::promise<void> p;

    YukiWorkshop::DNSResolver d(ioc2);
    d.resolve_a4(host, [&](int err, auto &addrs, auto &qname, auto &cname, uint ttl)
    {
      if (!err) {
          wsMutex.lock();
          for (auto &it : addrs) {
            addrCount++;
            ip = it.to_string();
          }
          p.set_value();
          wsMutex.unlock();
      } else {
        p.set_value();
      }
    });
    ioc2.run();

    std::future<void> f = p.get_future();
    f.get();

    if (addrCount == 0)
    {
      wsMutex.lock();
      setcolor(RED);
      std::cerr << "ERROR: Could not resolve " << host << std::endl;
      setcolor(BRIGHT_WHITE);
      wsMutex.unlock();
      //return stream;
      // FIXME: what do?
    }

    ip_address = net::ip::address::from_string(ip.c_str(), ec);
  }
  else
  {
    ip_address = net::ip::address::from_string(host, ec);
  }

  tcp::endpoint result(ip_address, (uint_least16_t)std::stoi(port.c_str()));
  return result;
#else
  // Look up the domain name
  tcp::resolver resolver(ioc);
  auto const results = resolver.async_resolve(host, port, yield[ec]);
  if (ec)
    fail(ec, "resolve");

  return results.begin()->endpoint();
#endif
}

void dero_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  beast::error_code ec;

  websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(ioc, ctx);
  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);

  // Set a timeout on the operation
  beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(30));

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
  std::stringstream ss;
  ss << "/ws/" << wallet;

  ws.async_handshake(host, ss.str().c_str(), yield[ec]);
  if (ec)
  {
    ws.async_close(websocket::close_code::normal, yield[ec]);
    return fail(ec, "handshake");
  }
  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  while (true)
  {
    try
    {
      buffer.clear();
      workInfo.str("");
      workInfo.clear();

      bool *B = isDev ? &submittingDev : &submitting;

      if (*B)
      {
        boost::json::object *S = isDev ? &devShare : &share;
        std::string msg = boost::json::serialize(*S);
        // wsMutex.lock();
        // std::cout << msg;
        // wsMutex.unlock();
        ws.async_write(boost::asio::buffer(msg), yield[ec]);
        if (ec)
        {
          return fail(ec, "Submission Error");
        }
        *B = false;
      }

      beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(60));
      ws.async_read(buffer, yield[ec]);
      if (!ec)
      {
        // handle getwork feed

        beast::get_lowest_layer(ws).expires_never();
        workInfo << beast::make_printable(buffer.data());
        if (json::accept(workInfo.str()))
        {
          json workData = json::parse(workInfo.str());
          if ((isDev ? (workData.at("height") != devHeight) : (workData.at("height") != ourHeight)))
          {
            // wsMutex.lock();
            if (isDev)
              devJob = workData;
            else
              job = workData;
            json *J = isDev ? &devJob : &job;
            // wsMutex.unlock();

            if ((*J).at("lasterror") != "")
            {
              std::cerr << "received error: " << (*J).at("lasterror") << std::endl
                        << consoleLine << versionString << " ";
            }

            if (!isDev)
            {
              currentBlob = std::string((*J).at("blockhashing_blob"));
              blockCounter = (*J).at("blocks");
              miniBlockCounter = (*J).at("miniblocks");
              rejected = (*J).at("rejected");
              hashrate = (*J).at("difficultyuint64");
              ourHeight = (*J).at("height");
              difficulty = (*J).at("difficultyuint64");
              // printf("NEW JOB RECEIVED | Height: %d | Difficulty %" PRIu64 "\n", ourHeight, difficulty);
              accepted = (*J).at("miniblocks");
              rejected = (*J).at("rejected");
              if (!isConnected)
              {
                wsMutex.lock();
                setcolor(BRIGHT_YELLOW);
                printf("Mining at: %s/ws/%s\n", host.c_str(), wallet.c_str());
                setcolor(CYAN);
                printf("Dev fee: %.2f", devFee);
                std::cout << "%" << std::endl;
                setcolor(BRIGHT_WHITE);
                wsMutex.unlock();
              }
              isConnected = isConnected || true;
              jobCounter++;
            }
            else
            {
              difficultyDev = (*J).at("difficultyuint64");
              devBlob = std::string((*J).at("blockhashing_blob"));
              devHeight = (*J).at("height");
              if (!devConnected)
              {
                wsMutex.lock();
                setcolor(CYAN);
                printf("Connected to dev node: %s\n", devPool);
                setcolor(BRIGHT_WHITE);
                wsMutex.unlock();
              }
              devConnected = devConnected || true;
              jobCounter++;
            }
          }
        }
      }
      else
      {
        bool *B = isDev ? &devConnected : &isConnected;
        (*B) = false;
        return fail(ec, "async_read");
      }
    }
    catch (...)
    {
      setcolor(RED);
      std::cout << "ws error\n";
      setcolor(BRIGHT_WHITE);
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // // Close the WebSocket connection
  ws.async_close(websocket::close_code::normal, yield[ec]);
  if (ec)
    return fail(ec, "close");

  // If we get here then the connection is closed gracefully

  // The make_printable() function helps print a ConstBufferSequence
  // std::cout << beast::make_printable(buffer.data()) << std::endl;
}

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

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(ws).connect(endpoint);

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
    return fail(ec, "handshake");
  }

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  bool subStart = false;

  while (true)
  {
    try
    {
      bool *B = isDev ? &submittingDev : &submitting;
      if (*B)
      {
        boost::json::object *S = isDev ? &devShare : &share;
        std::string msg = boost::json::serialize(*S);

        // Acquire a lock before writing to the WebSocket
        ws.async_write(boost::asio::buffer(msg), [&](const boost::system::error_code &ec, std::size_t)
                       {
            if (ec) {
                setcolor(RED);
                printf("\nasync_write: submission error\n");
                setcolor(BRIGHT_WHITE);
            } });
        (*B) = false;
      }

      buffer.clear();
      workInfo.str("");
      workInfo.clear();

      beast::get_lowest_layer(ws).expires_after(std::chrono::seconds(180));
      ws.async_read(buffer, yield[ec]);
      if (!ec)
      {
        // handle getwork feed
        beast::get_lowest_layer(ws).expires_never();
        workInfo << beast::make_printable(buffer.data());

        // std::cout << "Received data: " << workInfo.str() << std::endl;
        if (json::accept(workInfo.str()))
        {
          json response = json::parse(workInfo.str());
          json workData;
          if (response.contains("block_rejected"))
          {
            rejected++;
          }
          else if (response.contains("new_job") || response.contains("template"))
          {

            if (response.contains("new_job"))
              workData = response.at("new_job");
            else if (response.contains("template"))
              workData = response;

            if ((isDev ? (workData.at("height") != devHeight) : (workData.at("height") != ourHeight)))
            {
              if (isDev)
                devJob = workData;
              else
                job = workData;
              json *J = isDev ? &devJob : &job;

              if ((*J).contains("lasterror") && (*J).at("lasterror") != "")
              {
                std::cerr << "received error: " << (*J).at("lasterror") << std::endl
                          << consoleLine << "v" << versionString << " ";
              }

              if (!isDev)
              {
                currentBlob = (*J).at("template").get<std::string>();
                ourHeight++;
                difficulty = std::stoull((*J).at("difficulty").get<std::string>());

                if (!isConnected)
                {
                  wsMutex.lock();
                  setcolor(BRIGHT_YELLOW);
                  printf("Mining at: %s/getwork/%s/%s\n", host.c_str(), wallet.c_str(), worker.c_str());
                  setcolor(CYAN);
                  printf("Dev fee: %.2f", devFee);
                  std::cout << "%" << std::endl;
                  setcolor(BRIGHT_WHITE);
                  wsMutex.unlock();
                }
                isConnected = true;
                jobCounter++;
              }
              else
              {
                devBlob = (*J).at("template").get<std::string>();
                devHeight++;
                difficultyDev = std::stoull((*J).at("difficulty").get<std::string>());

                if (!devConnected)
                {
                  wsMutex.lock();
                  setcolor(CYAN);
                  printf("Connected to dev node: %s\n", host.c_str());
                  setcolor(BRIGHT_WHITE);
                  wsMutex.unlock();
                }
                devConnected = true;
                jobCounter++;
              }
            }
          }
          else
          {
            accepted++;
          }
        }
      }
      else
      {
        bool *B = isDev ? &devConnected : &isConnected;
        (*B) = false;
        return fail(ec, "async_read");
      }
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      std::cout << "ws error: " << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);
      // submission_thread.interrupt();
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // Close the WebSocket connection
  // submission_thread.interrupt();
  ws.async_close(websocket::close_code::normal, yield[ec]);
  printf("loop broken\n");
  if (ec)
    return fail(ec, "close");
}

void xatumFailure(bool isDev) noexcept
{
  setcolor(RED);
  if (isDev)
    printf("DEV | ");
  printf("Xatum Disconnect\n");
  setcolor(BRIGHT_WHITE);
}

int handleXatumPacket(Xatum::packet xPacket, bool isDev);

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
  ctx.set_options(boost::asio::ssl::context::default_workarounds |
                  boost::asio::ssl::context::no_sslv2 |
                  boost::asio::ssl::context::no_sslv3 |
                  boost::asio::ssl::context::no_tlsv1 |
                  boost::asio::ssl::context::no_tlsv1_1);

  beast::error_code ec;
  ctx.set_verify_mode(ssl::verify_none); // Accept self-signed certificates
  tcp::socket socket(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);
  boost::asio::deadline_timer deadline(ioc, boost::posix_time::seconds(1));

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
    return fail(ec, "handshake");

  boost::json::object handshake_packet = {
      {"addr", wallet.c_str()},
      {"work", worker.c_str()},
      {"agent", (std::string("tnn-miner ") + versionString).c_str()},
      {"algos", boost::json::array{
                    "xel/0",
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

  while (true)
  {
    try
    {
      if (
          Xatum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - Xatum::lastReceivedJobTime > Xatum::jobTimeout)
      {
        bool *C = isDev ? &devConnected : &isConnected;
        (*C) = false;
        return fail(ec, "Xatum session timed out");
      }
      bool *B = isDev ? &submittingDev : &submitting;
      if (*B)
      {
        boost::json::object *S = &share;
        if (isDev)
          S = &devShare;
        std::string msg = Xatum::submission + boost::json::serialize(*S) + "\n";
        // if (lastHash.compare((*S).at("hash").get_string()) == 0) continue;
        // lastHash = (*S).at("hash").get_string();

        // printf("submitting share: %s\n", msg.c_str());
        // Acquire a lock before writing to the WebSocket
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(10));
        boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
        if (ec)
        {
          bool *C = isDev ? &devConnected : &isConnected;
          (*C) = false;
          return fail(ec, "Xatum submission");
        }
        (*B) = false;
      }
      boost::asio::streambuf response;
      std::stringstream workInfo;
      beast::get_lowest_layer(stream).expires_after(std::chrono::milliseconds(45000));

      deadline.expires_from_now(boost::posix_time::seconds(1));
      deadline.async_wait([&](beast::error_code ec)
                          {
          if (!ec) {
              beast::get_lowest_layer(stream).cancel();
          } });
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
      (*C) = false;
      return fail(ec, "Xatum session error");
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // submission_thread.interrupt();
  stream.async_shutdown(yield[ec]);
}

int handleXatumPacket(Xatum::packet xPacket, bool isDev)
{
  std::string command = xPacket.command;
  json data = xPacket.data;
  int res = 0;

  if (command == Xatum::print)
  {
    if (Xatum::accepted_msg.compare(data.at("msg").get<std::string>()) == 0)
      accepted++;

    if (Xatum::stale_msg.compare(data.at("msg").get<std::string>()) == 0)
      rejected++;

    int msgLevel = data.at("lvl").get<int>();
    if (msgLevel < Xatum::logLevel)
      return 0;

    printf("\n");
    if (isDev)
    {
      setcolor(CYAN);
      printf("DEV | ");
    }

    wsMutex.lock();
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

    printf("%s\n", data.at("msg").get<std::string>().c_str());

    setcolor(BRIGHT_WHITE);
    wsMutex.unlock();
  }

  else if (command == Xatum::newJob)
  {
    uint64_t *diff = isDev ? &difficultyDev : &difficulty;
    json *J = isDev ? &devJob : &job;
    uint64_t *h = isDev ? &devHeight : &ourHeight;

    std::string *B = isDev ? &devBlob : &currentBlob;

    if (data.at("blob").get<std::string>().compare(*B) == 0)
      return 0;
    *B = data.at("blob").get<std::string>();

    Xatum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    // std::cout << data << std::endl;
    if (!isDev)
    {
      setcolor(CYAN);
      printf("\nNew Xatum job received\n");
      setcolor(BRIGHT_WHITE);
    }
    *diff = data.at("diff").get<uint64_t>();

    if ((*J).contains("template"))
      (*J).at("template") = (*B).c_str();
    else
      (*J).emplace("template", (*B).c_str());

    bool *C = isDev ? &devConnected : &isConnected;

    if (!*C)
    {
      if (!isDev)
      {
        wsMutex.lock();
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
        setcolor(CYAN);
        printf("Dev fee: %.2f", devFee);
        std::cout << "%" << std::endl;
        setcolor(BRIGHT_WHITE);
        wsMutex.unlock();
      }
      else
      {
        wsMutex.lock();
        setcolor(CYAN);
        printf("Connected to dev node: %s\n", host.c_str());
        setcolor(BRIGHT_WHITE);
        wsMutex.unlock();
      }
    }

    *C = true;

    (*h)++;
    jobCounter++;
  }

  else if (!isDev && command == Xatum::success)
  {
    // std::cout << data << std::endl;
    if (data.at("msg").get<std::string>() == "ok")
    {
      printf("Xatum: share accepted!");
      accepted++;
    }
    else
    {
      rejected++;
      setcolor(RED);
      printf("\nXatum Share Rejected: %s\n", data.at("msg").get<std::string>().c_str());
      setcolor(BRIGHT_WHITE);
    }
  }
  else
  {
    printf("unknown command: %s\n", command.c_str());
  }

  return res;
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
  ctx.set_options(boost::asio::ssl::context::default_workarounds |
                  boost::asio::ssl::context::no_sslv2 |
                  boost::asio::ssl::context::no_sslv3 |
                  boost::asio::ssl::context::no_tlsv1 |
                  boost::asio::ssl::context::no_tlsv1_1);

  beast::error_code ec;
  boost::system::error_code jsonEc;
  boost::asio::deadline_timer deadline(ioc, boost::posix_time::seconds(1));

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);

  // Create a TCP socket
  ctx.set_verify_mode(ssl::verify_none); // Accept self-signed certificates
  //tcp::socket socket(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);
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
    return fail(ec, "handshake");

  boost::json::object packet = XelisStratum::stratumCall;
  packet.at("id") = XelisStratum::subscribe.id;
  packet.at("method") = XelisStratum::subscribe.method;
  std::string minerName = "tnn-miner/" + std::string(versionString);
  packet.at("params") = boost::json::array({minerName, boost::json::array({"xel/0"})});
  std::string subscription = boost::json::serialize(packet) + "\n";

  // std::cout << subscription << std::endl;

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  size_t trans = boost::asio::async_write(stream, boost::asio::buffer(subscription), yield[ec]);
  if (ec)
    return fail(ec, "Stratum subscribe");

  // Make sure subscription is successful
  boost::asio::streambuf subRes;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::read_until(stream, subRes, "\n");

  std::string subResString = beast::buffers_to_string(subRes.data());
  subRes.consume(trans);
  boost::json::object subResJson = boost::json::parse(subResString.c_str(), jsonEc).as_object();
  if (jsonEc)
  {
    std::cerr << jsonEc.message() << std::endl;
  }

  // std::cout << boost::json::serialize(subResJson).c_str() << std::endl;
  handleXStratumResponse(subResJson, isDev);

  // Authorize Stratum Worker
  packet = XelisStratum::stratumCall;
  packet.at("id") = XelisStratum::authorize.id;
  packet.at("method") = XelisStratum::authorize.method;
  packet.at("params") = boost::json::array({wallet, worker, "x"});
  std::string authorization = boost::json::serialize(packet) + "\n";

  // std::cout << authorization << std::endl;

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::async_write(stream, boost::asio::buffer(authorization), yield[ec]);
  if (ec)
    return fail(ec, "Stratum authorize");

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  XelisStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

  while (true)
  {
    try
    {
      if (
          XelisStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - XelisStratum::lastReceivedJobTime > XelisStratum::jobTimeout)
      {
        bool *C = isDev ? &devConnected : &isConnected;
        (*C) = false;
        return fail(ec, "Stratum session timed out");
      }
      bool *B = isDev ? &submittingDev : &submitting;
      if (*B)
      {
        boost::json::object *S = &share;
        if (isDev)
          S = &devShare;

        std::string msg = boost::json::serialize((*S)) + "\n";
        // if (lastHash.compare((*S).at("hash").get_string()) == 0) continue;
        // lastHash = (*S).at("hash").get_string();

        // printf("submitting share: %s\n", msg.c_str());
        // Acquire a lock before writing to the WebSocket

        // std::cout << "sending in: " << msg << std::endl;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(10));
        boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
        if (ec)
        {
          bool *C = isDev ? &devConnected : &isConnected;
          (*C) = false;
          return fail(ec, "Stratum submit");
        }
        (*B) = false;
      }

      boost::asio::streambuf response;
      std::stringstream workInfo;
      beast::get_lowest_layer(stream).expires_after(std::chrono::milliseconds(60000));

      deadline.expires_from_now(boost::posix_time::seconds(1));
      deadline.async_wait([&](beast::error_code ec)
                          {
          if (!ec) {
              beast::get_lowest_layer(stream).cancel();
          } });
      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec && trans > 0)
        return fail(ec, "Stratum async_read");

      if (trans > 0)
      {
        std::scoped_lock<boost::mutex> lockGuard(wsMutex);
        std::string data = beast::buffers_to_string(response.data());
        // Consume the data from the buffer after processing it
        response.consume(trans);

        // std::cout << data << std::endl;

        if (data.compare(XelisStratum::k1ping) == 0)
        {
          trans = boost::asio::async_write(
              stream,
              boost::asio::buffer(XelisStratum::k1pong),
              yield[ec]);
          if (ec && trans > 0)
            return fail(ec, "Stratum pong (K1 style)");
        }
        else
        {
          boost::json::object sRPC = boost::json::parse(data.c_str()).as_object();
          if (sRPC.contains("method"))
          {
            if (std::string(sRPC.at("method").as_string().c_str()).compare(XelisStratum::s_ping) == 0)
            {
              boost::json::object pong({{"id", sRPC.at("id").get_uint64()},
                                        {"method", XelisStratum::pong.method}});
              std::string pongPacket = std::string(boost::json::serialize(pong).c_str()) + "\n";
              trans = boost::asio::async_write(
                  stream,
                  boost::asio::buffer(pongPacket),
                  yield[ec]);
              if (ec && trans > 0)
                return fail(ec, "Stratum pong");
            }
            else
              handleXStratumPacket(sRPC, isDev);
          }
          else
          {
            handleXStratumResponse(sRPC, isDev);
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      bool *C = isDev ? &devConnected : &isConnected;
      (*C) = false;
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);
      return fail(ec, "Stratum session error");
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // submission_thread.interrupt();
  stream.async_shutdown(yield[ec]);
}

int handleXStratumPacket(boost::json::object packet, bool isDev)
{
  std::string M = packet.at("method").get_string().c_str();
  if (M.compare(XelisStratum::s_notify) == 0)
  {

    if (ourHeight > 0 && packet.at("params").as_array()[4].get_bool() != true)
      return 0;

    setcolor(CYAN);
    if (!isDev)
      printf("\nStratum: new job received\n");
    setcolor(BRIGHT_WHITE);

    XelisStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    json *J = isDev ? &devJob : &job;
    uint64_t *h = isDev ? &devHeight : &ourHeight;

    std::string bs = (*J).at("template").get<std::string>();
    char *blob = (char *)bs.c_str();

    const char *jobId = packet.at("params").as_array()[0].get_string().c_str();
    const char *ts = packet.at("params").as_array()[1].get_string().c_str();
    int tsLen = packet.at("params").as_array()[1].get_string().size();
    const char *header = packet.at("params").as_array()[2].get_string().c_str();

    memset(&blob[64], '0', 16);
    memcpy(&blob[64 + 16 - tsLen], ts, tsLen);
    memcpy(blob, header, 64);

    (*J).at("template") = std::string(blob);
    (*J)["jobId"] = jobId;

    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
        setcolor(CYAN);
        printf("Dev fee: %.2f", devFee);
        std::cout << "%" << std::endl;
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        setcolor(CYAN);
        printf("Connected to dev node: %s\n", host.c_str());
        setcolor(BRIGHT_WHITE);
      }
    }

    *C = true;
    (*h)++;
    jobCounter++;
  }
  else if (M.compare(XelisStratum::s_setDifficulty) == 0)
  {

    uint64_t *d = isDev ? &difficultyDev : &difficulty;
    (*d) = packet.at("params").as_array()[0].get_double();
    if ((*d) == 0)
      (*d) = packet.at("params").as_array()[0].get_uint64();
  }
  else if (M.compare(XelisStratum::s_setExtraNonce) == 0)
  {

    json *J = isDev ? &devJob : &job;
    uint64_t *h = isDev ? &devHeight : &ourHeight;

    std::string bs = (*J).at("template").get<std::string>();
    char *blob = (char *)bs.c_str();
    const char *en = packet.at("params").as_array()[0].as_string().c_str();
    int enLen = packet.at("params").as_array()[0].as_string().size();

    memset(&blob[48], '0', 64);
    memcpy(&blob[48], en, enLen);

    (*J).at("template") = std::string(blob).c_str();

    (*h)++;
    jobCounter++;
  }
  else if (M.compare(XelisStratum::s_print) == 0)
  {

    int lLevel = packet.at("params").as_array()[0].as_int64();
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
      printf("%s\n", packet.at("params").as_array()[1].as_string().c_str());

      setcolor(BRIGHT_WHITE);

      return res;
    }
  }
  return 0;
}

int handleXStratumResponse(boost::json::object packet, bool isDev)
{
  // if (!isDev) {
  // if (!packet.contains("id")) return 0;
  int64_t id = packet["id"].as_int64();

  switch (id)
  {
  case XelisStratum::subscribeID:
  {
    json *J = isDev ? &devJob : &job;
    if (!(*J).contains("template"))
    {
      byte blankBlob[XELIS_TEMPLATE_SIZE * 2];
      memset(blankBlob, '0', XELIS_TEMPLATE_SIZE * 2);
      std::string base = std::string((char *)blankBlob);
      base.resize(XELIS_TEMPLATE_SIZE * 2);
      (*J).emplace("template", base.c_str());
    }

    std::string bs = (*J).at("template").get<std::string>();
    char *blob = (char *)bs.c_str();
    const char *extraNonce = packet.at("result").get_array()[1].get_string().c_str();
    int enLen = packet.at("result").get_array()[2].get_int64();
    const char *pubKey = packet.at("result").get_array()[3].get_string().c_str();

    memset(&blob[96], '0', 64);
    memcpy(&blob[96], extraNonce, enLen * 2);
    memcpy(&blob[160], pubKey, 64);

    (*J).at("template") = std::string(blob).c_str();
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
      accepted++;
      std::cout << "Stratum: share accepted" << std::endl;
      setcolor(BRIGHT_WHITE);
    }
    else
    {
      rejected++;
      if (!isDev)
        setcolor(RED);
      std::cout << "Stratum: share rejected: " << packet.at("error").get_object()["message"].get_string() << std::endl;
      setcolor(BRIGHT_WHITE);
    }
    break;
  }
  }
  return 0;
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
  ctx.set_options(boost::asio::ssl::context::default_workarounds |
                  boost::asio::ssl::context::no_sslv2 |
                  boost::asio::ssl::context::no_sslv3 |
                  boost::asio::ssl::context::no_tlsv1 |
                  boost::asio::ssl::context::no_tlsv1_1);

  beast::error_code ec;
  boost::system::error_code jsonEc;
  boost::asio::deadline_timer deadline(ioc, boost::posix_time::seconds(1));

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  boost::beast::tcp_stream stream(ioc);

  // Set a timeout on the operation
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec)
    return fail(ec, "connect");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  boost::json::object packet;

  // Authorize Stratum Worker
  packet = XelisStratum::stratumCall;
  packet.at("id") = XelisStratum::authorize.id;
  packet.at("method") = XelisStratum::authorize.method;
  packet.at("params") = boost::json::array({wallet + "." + worker});

  std::string authorization = boost::json::serialize(packet) + "\n";

  // // std::cout << authorization << std::endl;

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  boost::asio::async_write(stream, boost::asio::buffer(authorization), yield[ec]);
  if (ec)
    return fail(ec, "Stratum authorize");

  packet = XelisStratum::stratumCall;

  packet["id"] = SpectreStratum::subscribe.id;
  packet["method"] = SpectreStratum::subscribe.method;
  packet["params"] = boost::json::array({
    minerName
  });

  boost::asio::streambuf subRes;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  size_t trans = boost::asio::read_until(stream, subRes, "\n");

  std::string authResString = beast::buffers_to_string(subRes.data());
  subRes.consume(trans);
  boost::json::object authResJson = boost::json::parse(authResString.c_str(), jsonEc).as_object();
  if (jsonEc)
  {
    std::cerr << jsonEc.message() << std::endl;
  }

  // SpectreStratum::stratumCall;
  // packet.at("id") = SpectreStratum::subscribe.id;
  // packet.at("method") = SpectreStratum::subscribe.method;
  // packet.at("params") = {minerName};
  std::string subscription = boost::json::serialize(packet) + "\n";

  // std::cout << authResString << std::endl;

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::async_write(stream, boost::asio::buffer(subscription), yield[ec]);
  if (ec)
    return fail(ec, "Stratum subscribe");

  // beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // trans = boost::asio::read_until(stream, subRes, "\n");

  // std::string subResString = beast::buffers_to_string(subRes.data());
  // subRes.consume(trans);

  // wsMutex.lock();
  // printf("before packet\n");
  // std::cout << subResString << std::endl;

  // printf("before parse\n");
  // wsMutex.unlock();
  // boost::json::object subResJson = boost::json::parse(subResString.c_str(), jsonEc).as_object();
  // if (jsonEc)
  // {
  //   std::cerr << jsonEc.message() << std::endl;
  // }

  // printf("after parse\n");

  // handleXStratumPacket(subResJson, isDev);

  // // This buffer will hold the incoming message
  // beast::flat_buffer buffer;
  // std::stringstream workInfo;

  SpectreStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

  SpectreStratum::jobCache jobCache;

  std::string chopQueue = "NULL";

  while (true)
  {
    try
{
      if (
        !isDev &&
        SpectreStratum::lastShareSubmissionTime > 0 &&
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - SpectreStratum::lastShareSubmissionTime > SpectreStratum::shareSubmitTimeout) {
          rate30sec.clear();
          rate30sec.push_back(0);
        }

      if (
          SpectreStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - SpectreStratum::lastReceivedJobTime > SpectreStratum::jobTimeout)
      {
        bool *C = isDev ? &devConnected : &isConnected;
        (*C) = false;
        return fail(ec, "Stratum session timed out");
      }
      bool *B = isDev ? &submittingDev : &submitting;
      if (*B)
      {
        boost::json::object *S = &share;
        if (isDev)
          S = &devShare;

        std::string msg = boost::json::serialize((*S)) + "\n";
        // if (lastHash.compare((*S).at("hash").get_string()) == 0) continue;
        // lastHash = (*S).at("hash").get_string();

        // printf("submitting share: %s\n", msg.c_str());
        // Acquire a lock before writing to the WebSocket

        // std::cout << "sending in: " << msg << std::endl;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(10));
        boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
        if (ec)
        {
          bool *C = isDev ? &devConnected : &isConnected;
          (*C) = false;
          return fail(ec, "Stratum submit");
        }
        if (!isDev) SpectreStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        (*B) = false;
      }

      boost::asio::streambuf response;
      std::stringstream workInfo;
      beast::get_lowest_layer(stream).expires_after(std::chrono::milliseconds(60000));

      deadline.expires_from_now(boost::posix_time::seconds(1));
      deadline.async_wait([&](beast::error_code ec)
                          {
          if (!ec) {
              beast::get_lowest_layer(stream).cancel();
          } });
      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec && trans > 0)
        return fail(ec, "Stratum async_read");

      if (trans > 0)
      {
        std::scoped_lock<boost::mutex> lockGuard(wsMutex);
        std::vector<std::string> packets;
        std::string data = beast::buffers_to_string(response.data());
        // Consume the data from the buffer after processing it
        response.consume(trans);

        // std::cout << data << std::endl;

        std::stringstream  jsonStream(data);

        std::string line;
        while(std::getline(jsonStream,line,'\n'))
        {
            packets.push_back(line);
        }

        for (std::string packet : packets) {
          try {
            boost::json::object sRPC = boost::json::parse(packet.c_str()).as_object();
            if (sRPC.contains("method"))
            {
              if (std::string(sRPC.at("method").as_string().c_str()).compare(SpectreStratum::s_ping) == 0)
              {
                boost::json::object pong({{"id", sRPC.at("id").get_uint64()},
                                          {"method", SpectreStratum::pong.method}});
                std::string pongPacket = std::string(boost::json::serialize(pong).c_str()) + "\n";
                trans = boost::asio::async_write(
                    stream,
                    boost::asio::buffer(pongPacket),
                    yield[ec]);
                if (ec && trans > 0)
                  return fail(ec, "Stratum pong");
              }
              else
                handleSpectreStratumPacket(sRPC, &jobCache, isDev);
            }
            else
            {
              handleSpectreStratumResponse(sRPC, isDev);
            } 
          } catch(const std::exception &e){
            // printf("\n\n packet count: %d, msg size: %llu\n\n", packets.size(), trans);

            // setcolor(RED);
            // printf("BEFORE PACKET\n");
            // std::cout << packet << std::endl;
            // printf("AFTER PACKET\n");
            // std::cerr << e.what() << std::endl;
            // setcolor(BRIGHT_WHITE);
            bool tryParse = (chopQueue.compare("NULL") != 0);

            if (tryParse) {
              chopQueue += packet;
              // printf("resulting json string: %s\n\n", chopQueue.c_str());
              try
              {
                boost::json::object sRPC = boost::json::parse(chopQueue.c_str()).as_object();
                if (sRPC.contains("method"))
                {
                  if (std::string(sRPC.at("method").as_string().c_str()).compare(SpectreStratum::s_ping) == 0)
                  {
                    boost::json::object pong({{"id", sRPC.at("id").get_uint64()},
                                              {"method", SpectreStratum::pong.method}});
                    std::string pongPacket = std::string(boost::json::serialize(pong).c_str()) + "\n";
                    trans = boost::asio::async_write(
                        stream,
                        boost::asio::buffer(pongPacket),
                        yield[ec]);
                    if (ec && trans > 0)
                      return fail(ec, "Stratum pong");
                  }
                  else
                    handleSpectreStratumPacket(sRPC, &jobCache, isDev);
                }
                else
                {
                  handleSpectreStratumResponse(sRPC, isDev);
                }
                chopQueue = "NULL";
                // printf("COMBINE WORKED!\n\n");
              }
              catch (const std::exception &e)
              {
                setcolor(RED);
                printf("COMBINE FAILED\n\nBEFORE PACKET\n");
                std::cout << chopQueue << std::endl;
                printf("AFTER PACKET\n");
                std::cerr << e.what() << std::endl;
                setcolor(BRIGHT_WHITE);
              }
            } else {
              chopQueue = packet;
              // printf("partial json start = %s\n", chopQueue.c_str());
            } 
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      bool *C = isDev ? &devConnected : &isConnected;
      (*C) = false;
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);
      return fail(ec, "Stratum session error");
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(125));
  }

  // submission_thread.interrupt();
  printf("\n\n\nflagged connection loss\n");
  stream.close();
}

void do_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    int algo,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  switch (algo)
  {
  case DERO_HASH:
    dero_session(host, port, wallet, ioc, ctx, yield, isDev);
    break;
  case XELIS_HASH:
  {
    switch (protocol)
    {
    case XELIS_SOLO:
      xelis_session(host, port, wallet, worker, ioc, yield, isDev);
      break;
    case XELIS_XATUM:
      xatum_session(host, port, wallet, worker, ioc, ctx, yield, isDev);
      break;
    case XELIS_STRATUM:
      xelis_stratum_session(host, port, wallet, worker, ioc, ctx, yield, isDev);
      break;
    }
    break;
  }
  case SPECTRE_X:
    spectre_stratum_session(host, port, wallet, worker, ioc, ctx, yield, isDev);
    break;
  }
}

int handleSpectreStratumPacket(boost::json::object packet, SpectreStratum::jobCache *cache, bool isDev)
{
  std::string M = packet.at("method").get_string().c_str();
  if (M.compare(SpectreStratum::s_notify) == 0)
  {
    json *J = isDev ? &devJob : &job;
    uint64_t *h = isDev ? &devHeight : &ourHeight;

    uint64_t id = std::stoull(packet["params"].as_array()[0].get_string().c_str());

    uint64_t h1 = packet["params"].as_array()[1].as_array()[0].get_uint64();
    uint64_t h2 = packet["params"].as_array()[1].as_array()[1].get_uint64();
    uint64_t h3 = packet["params"].as_array()[1].as_array()[2].get_uint64();
    uint64_t h4 = packet["params"].as_array()[1].as_array()[3].get_uint64();

    uint64_t comboHeader[4] = {h1, h2, h3, h4};

    bool isEqual = true;
    for (int i = 0; i < 4; i++) {
      isEqual &= comboHeader[i] == cache->header[i];
    }
    if (isEqual) return 0;

    for (int i = 0; i < 4; i++) {
      cache->header[i] = comboHeader[i];
    }

    uint64_t ts = packet["params"].as_array()[2].get_uint64();

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

    setcolor(CYAN);
    if (!isDev)
      printf("\nStratum: new job received\n");
    setcolor(BRIGHT_WHITE);

    SpectreStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();


    (*J)["template"] = std::string(newTemplate, SpectreX::INPUT_SIZE*2);
    // std::string testPrint = (*J)["template"].get<std::string>();

    // byte testOut[160];
    // memcpy(testOut, testPrint.data(), 160);
    // for (int i = 0; i < 160; i++) {
    //   std::cout << testOut[i];
    // }
    // printf("\n");
    // std::cout << testPrint;
    // printf("\n");

    // std::string bs = (*J).at("template").get<std::string>();
    // char *blob = (char *)bs.c_str();

    // const char *jobId = packet.at("params").as_array()[0].get_string().c_str();
    // const char *ts = packet.at("params").as_array()[1].get_string().c_str();
    // int tsLen = packet.at("params").as_array()[1].get_string().size();
    // const char *header = packet.at("params").as_array()[2].get_string().c_str();

    // memset(&blob[64], '0', 16);
    // memcpy(&blob[64 + 16 - tsLen], ts, tsLen);
    // memcpy(blob, header, 64);

    // (*J).at("template") = std::string(blob);
    (*J)["jobId"] = id;

    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
        setcolor(CYAN);
        printf("Dev fee: %.2f", devFee);
        std::cout << "%" << std::endl;
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        setcolor(CYAN);
        printf("Connected to dev node: %s\n", host.c_str());
        setcolor(BRIGHT_WHITE);
      }
    }

    *C = true;
    (*h)++;
    jobCounter++;
  }
  else if (M.compare(SpectreStratum::s_setDifficulty) == 0)
  {
    // std::cout << boost::json::serialize(packet).c_str() << std::endl;
    double *d = isDev ? &doubleDiffDev : &doubleDiff;
    (*d) = packet.at("params").as_array()[0].get_double();
    if ((*d) < 0.00000000001) (*d) = packet.at("params").as_array()[0].get_uint64();

    // printf("%f\n", (*d));
  }
  else if (M.compare(SpectreStratum::s_setExtraNonce) == 0)
  {
    // std::cout << boost::json::serialize(packet).c_str() << std::endl;
    // json *J = isDev ? &devJob : &job;
    // uint64_t *h = isDev ? &devHeight : &ourHeight;

    // std::string bs = (*J).at("template").get<std::string>();
    // char *blob = (char *)bs.c_str();
    // const char *en = packet.at("params").as_array()[0].as_string().c_str();
    // int enLen = packet.at("params").as_array()[0].as_string().size();

    // memset(&blob[48], '0', 64);
    // memcpy(&blob[48], en, enLen);

    // (*J).at("template") = std::string(blob).c_str();

    // (*h)++;
    // jobCounter++;
  }
  else if (M.compare(SpectreStratum::s_print) == 0)
  {

    int lLevel = packet.at("params").as_array()[0].as_int64();
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
        if (!isDev)
          setcolor(BRIGHT_WHITE);
        printf("Stratum INFO: ");
        break;
      case SpectreStratum::STRATUM_WARN:
        if (!isDev)
          setcolor(BRIGHT_YELLOW);
        printf("Stratum WARNING: ");
        break;
      case SpectreStratum::STRATUM_ERROR:
        if (!isDev)
          setcolor(RED);
        printf("Stratum ERROR: ");
        res = -1;
        break;
      case SpectreStratum::STRATUM_DEBUG:
        break;
      }
      printf("%s\n", packet.at("params").as_array()[1].as_string().c_str());

      setcolor(BRIGHT_WHITE);

      return res;
    }
  }
  return 0;
}

int handleSpectreStratumResponse(boost::json::object packet, bool isDev)
{
  // if (!isDev) {
  // if (!packet.contains("id")) return 0;
  int64_t id = packet["id"].as_int64();

  switch (id)
  {
    case SpectreStratum::subscribeID:
    {
      std::cout << boost::json::serialize(packet).c_str() << std::endl;
      if (packet["error"].is_null()) return 0;
      else {
        const char *errorMsg = packet["error"].get_string().c_str();
        setcolor(RED);
        printf("\n");
        if (isDev) {
          setcolor(CYAN);
          printf("DEV | ");
        }
        printf("Stratum ERROR: %s\n", errorMsg);
        return -1;
      }
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
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        if (!isDev) rejected++;
        if (!isDev)
          setcolor(RED);
        std::cout << "Stratum: share rejected: " << packet.at("error").get_array()[1].get_string() << std::endl;
        setcolor(BRIGHT_WHITE);
      }
      break;
    }
  }
  return 0;
}
