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

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

int handleXStratumPacket(boost::json::object packet, bool isDev)
{
  std::string M = packet["method"].as_string().c_str();
  if (M.compare(XelisStratum::s_notify) == 0)
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

    char *blob = (char *)J["miner_work"].as_string().c_str();
    const char *jobId = packet["params"].as_array()[0].as_string().c_str();
    const char *ts = packet["params"].as_array()[1].as_string().c_str();
    int tsLen = packet["params"].as_array()[1].as_string().size();
    const char *header = packet["params"].as_array()[2].as_string().c_str();

    memset(&blob[64], '0', 16);
    memcpy(&blob[64 + 16 - tsLen], ts, tsLen);
    memcpy(blob, header, 64);

    J["miner_work"] = std::string(blob);
    J["jobId"] = std::string(jobId);

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
        printf("Connected to dev node");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }

    (*JV) = J;

    *C = true;
    (*h)++;
    jobCounter++;
  }
  else if (M.compare(XelisStratum::s_setDifficulty) == 0)
  {

    int64_t *d = isDev ? &difficultyDev : &difficulty;
    (*d) = packet["params"].as_array()[0].get_double();
    if ((*d) == 0)
      (*d) = packet["params"].as_array()[0].get_uint64();
  }
  else if (M.compare(XelisStratum::s_setExtraNonce) == 0)
  {

    boost::json::value *JV = isDev ? &devJob : &job;
    boost::json::object J = (*JV).as_object();

    int64_t *h = isDev ? &devHeight : &ourHeight;

    char *blob = (char *)J["miner_work"].as_string().c_str();
    const char *en = packet["params"].as_array()[0].as_string().c_str();
    int enLen = packet["params"].as_array()[0].as_string().size();

    memset(&blob[48], '0', 64);
    memcpy(&blob[48], en, enLen);

    J["miner_work"] = std::string(blob).c_str();

    (*JV) = J;

    (*h)++;
    jobCounter++;
  }
  else if (M.compare(XelisStratum::s_print) == 0)
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
      printf("%s\n", packet.at("params").as_array()[1].as_string().c_str());

      fflush(stdout);
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
  int64_t id = packet["id"].to_number<int64_t>();

  switch (id)
  {
  case XelisStratum::subscribeID:
  {
    boost::json::value *JV = isDev ? &devJob : &job;
    boost::json::object J = (*JV).as_object();
    if (J["miner_work"].is_null())
    {
      byte blankBlob[XELIS_TEMPLATE_SIZE * 2];
      memset(blankBlob, '0', XELIS_TEMPLATE_SIZE * 2);
      std::string base = std::string((char *)blankBlob);
      base.resize(XELIS_TEMPLATE_SIZE * 2);
      J["miner_work"] = base.c_str();
    }

    char *blob = (char *)J["miner_work"].as_string().c_str();
    const char *extraNonce = packet["result"].get_array()[1].get_string().c_str();
    int enLen = packet["result"].get_array()[2].get_int64();
    const char *pubKey = packet["result"].get_array()[3].get_string().c_str();

    memset(&blob[96], '0', 64);
    memcpy(&blob[96], extraNonce, enLen * 2);
    memcpy(&blob[160], pubKey, 64);

    J["miner_work"] = std::string(blob).c_str();
    (*JV) = J;
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
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    else
    {
      rejected++;
      if (!isDev)
        setcolor(RED);
      std::cout << "Stratum: share rejected: " << packet.at("error").get_object()["message"].get_string() << std::endl;
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
  beast::error_code ec;
  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);

  ctx.set_verify_mode(ssl::verify_none);
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect");

  if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
    throw beast::system_error(static_cast<int>(::ERR_get_error()), boost::asio::error::get_ssl_category());

  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec) return fail(ec, "handshake-xelis-strat");

  auto send_json = [&](const boost::json::object &obj) {
    std::string msg = boost::json::serialize(obj) + "\n";
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) fail(ec, "send_json");
  };

  auto send_and_recv_line = [&]() -> std::string {
    boost::asio::streambuf response;
    boost::asio::read_until(stream, response, "\n", ec);
    if (ec) return "";
    std::string result = beast::buffers_to_string(response.data());
    response.consume(result.size());
    return result;
  };

  // Subscribe
  boost::json::object subscribe = XelisStratum::stratumCall;
  subscribe["id"] = XelisStratum::subscribe.id;
  subscribe["method"] = XelisStratum::subscribe.method;
  subscribe["params"] = { "tnn-miner/" + std::string(versionString), {"xel/1"} };
  send_json(subscribe);
  auto subResStr = send_and_recv_line();
  auto subResJson = boost::json::parse(subResStr).as_object();
  handleXStratumResponse(subResJson, isDev);

  // Authorize
  boost::json::object auth = XelisStratum::stratumCall;
  auth["id"] = XelisStratum::authorize.id;
  auth["method"] = XelisStratum::authorize.method;
  auth["params"] = { wallet, worker, isDev ? "d=10000" : stratumPassword };
  send_json(auth);

  std::string packetBuffer;
  bool submitThreadRunning = true, abort = false;

  boost::thread submitThread([&](){
    while (!abort) {
      boost::unique_lock<boost::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&]{ return (data_ready && (*B)) || abort; });
      if (abort) break;

      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(1));
        boost::asio::async_write(stream, boost::asio::buffer(msg), [&](const boost::system::error_code& error, std::size_t bytes_transferred) {
          if (error) {
            printf("error on write: %s\n", error.message().c_str());
            fflush(stdout);
            abort = true;
          }
        });
      } catch (...) {
        setcolor(RED);
        printf("\nSubmit thread error\n");
        setcolor(BRIGHT_WHITE);
        break;
      }
      *B = false;
      data_ready = false;
      boost::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  while (!ABORT_MINER) {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try {
      if (XelisStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() -
          XelisStratum::lastReceivedJobTime > XelisStratum::jobTimeout) {
        setcolor(RED); printf("timeout\n"); setcolor(BRIGHT_WHITE);
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

      std::string newData = beast::buffers_to_string(response.data());
      response.consume(bytes);
      packetBuffer += newData;

      size_t pos;
      while ((pos = packetBuffer.find('\n')) != std::string::npos) {
        std::string line = packetBuffer.substr(0, pos);
        packetBuffer.erase(0, pos + 1);
        if (line == XelisStratum::k1ping) {
          boost::asio::async_write(stream, boost::asio::buffer(XelisStratum::k1pong), yield[ec]);
        } else {
          auto sRPC = boost::json::parse(line).as_object();
          if (sRPC.contains("method")) {
            if (sRPC["method"].as_string() == XelisStratum::s_ping) {
              boost::json::object pong = {{"id", sRPC["id"].get_uint64()}, {"method", XelisStratum::pong.method}};
              send_json(pong);
            } else {
              handleXStratumPacket(sRPC, isDev);
            }
          } else {
            handleXStratumResponse(sRPC, isDev);
          }
        }
      }

    } catch (const std::exception &e) {
      setcolor(RED); std::cerr << e.what() << std::endl; setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }
    boost::this_thread::yield();
  }

  cv.notify_all();
  if (submitThreadRunning) {
    submitThread.interrupt();
    submitThread.join();
  }
  stream.shutdown();
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
  beast::error_code ec;
  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  boost::beast::tcp_stream stream(ioc);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect");

  auto send_json = [&](const boost::json::object &obj) {
    std::string msg = boost::json::serialize(obj) + "\n";
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) fail(ec, "send_json");
  };

  auto send_and_recv_line = [&]() -> std::string {
    boost::asio::streambuf response;
    boost::asio::read_until(stream, response, "\n", ec);
    if (ec) return "";
    std::string result = beast::buffers_to_string(response.data());
    response.consume(result.size());
    return result;
  };

  // Subscribe
  boost::json::object subscribe = XelisStratum::stratumCall;
  subscribe["id"] = XelisStratum::subscribe.id;
  subscribe["method"] = XelisStratum::subscribe.method;
  subscribe["params"] = { "tnn-miner/" + std::string(versionString), {"xel/1"} };
  send_json(subscribe);

  auto subResStr = send_and_recv_line();
  auto subResJson = boost::json::parse(subResStr).as_object();
  handleXStratumResponse(subResJson, isDev);

  // Authorize
  boost::json::object auth = XelisStratum::stratumCall;
  auth["id"] = XelisStratum::authorize.id;
  auth["method"] = XelisStratum::authorize.method;
  auth["params"] = { wallet, worker, isDev ? "d=10000" : stratumPassword };
  send_json(auth);

  std::string packetBuffer;
  bool submitThreadRunning = true, abort = false;

  boost::thread submitThread([&](){
    while (!abort) {
      boost::unique_lock<boost::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&]{ return (data_ready && (*B)) || abort; });
      if (abort) break;

      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(1));
        boost::asio::async_write(stream, boost::asio::buffer(msg), [&](const boost::system::error_code& error, std::size_t) {
          if (error) {
            printf("error on write: %s\n", error.message().c_str());
            fflush(stdout);
            abort = true;
          }
        });
      } catch (...) {
        setcolor(RED);
        printf("\nSubmit thread error\n");
        setcolor(BRIGHT_WHITE);
        break;
      }
      *B = false;
      data_ready = false;
      boost::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  while (!ABORT_MINER) {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try {
      if (XelisStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() -
          XelisStratum::lastReceivedJobTime > XelisStratum::jobTimeout) {
        setcolor(RED); printf("timeout\n"); setcolor(BRIGHT_WHITE);
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

      std::string newData = beast::buffers_to_string(response.data());
      response.consume(bytes);
      packetBuffer += newData;

      size_t pos;
      while ((pos = packetBuffer.find('\n')) != std::string::npos) {
        std::string line = packetBuffer.substr(0, pos);
        packetBuffer.erase(0, pos + 1);

        if (line == XelisStratum::k1ping) {
          boost::asio::async_write(stream, boost::asio::buffer(XelisStratum::k1pong), yield[ec]);
        } else {
          auto sRPC = boost::json::parse(line).as_object();
          if (sRPC.contains("method")) {
            if (sRPC["method"].as_string() == XelisStratum::s_ping) {
              boost::json::object pong = {{"id", sRPC["id"].get_uint64()}, {"method", XelisStratum::pong.method}};
              send_json(pong);
            } else {
              handleXStratumPacket(sRPC, isDev);
            }
          } else {
            handleXStratumResponse(sRPC, isDev);
          }
        }
      }

    } catch (const std::exception &e) {
      setcolor(RED); std::cerr << e.what() << std::endl; setcolor(BRIGHT_WHITE);
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

  cv.notify_all();
  if (submitThreadRunning) {
    submitThread.interrupt();
    submitThread.join();
  }
  stream.close();
}
