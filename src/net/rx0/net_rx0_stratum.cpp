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

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

bool randomx_ready = false;
bool randomx_ready_dev = false;

uint64_t diff_numerator = boost_swap_impl::stoull("0x100000001", nullptr, 16);

static uint64_t rx_targetToDifficulty(const char* target) {
  uint32_t targetInt = boost_swap_impl::stoul(target, nullptr, 16);
  targetInt = __builtin_bswap32(targetInt);
  uint64_t diff = diff_numerator / targetInt;

  fflush(stdout);
  return diff;
}

int handleRandomXStratumPacket(boost::json::object packet, bool isDev)
{
  std::string M = packet["method"].as_string().c_str();
  if (M.compare(rx0Stratum::s_job) == 0)
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    if (!packet["error"].is_null()) return 1;

    boost::json::object newJob = packet["params"].as_object();

    setcolor(CYAN);
    if (!isDev)
      printf("\nStratum: new job received\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    rx0Stratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    boost::json::value *JV = isDev ? &devJob : &job;

    int64_t *h = isDev ? &devHeight : &ourHeight;

    if (!isDev) {
      difficulty = rx_targetToDifficulty(newJob.at("target").as_string().c_str());
    }

    std::string &refKey = isDev ? randomx_cacheKey_dev : randomx_cacheKey;

    updateVM(newJob, isDev);

    (*JV) = newJob;

    (*h)++;
    jobCounter++;
  }
  else if (M.compare(rx0Stratum::s_print) == 0)
  {

    int lLevel = packet.at("params").as_array()[0].to_number<int64_t>();
    if (lLevel != rx0Stratum::STRATUM_DEBUG)
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
      case rx0Stratum::STRATUM_INFO:
        if (!isDev)
          setcolor(BRIGHT_WHITE);
        printf("Stratum INFO: ");
        break;
      case rx0Stratum::STRATUM_WARN:
        if (!isDev)
          setcolor(BRIGHT_YELLOW);
        printf("Stratum WARNING: ");
        break;
      case rx0Stratum::STRATUM_ERROR:
        if (!isDev)
          setcolor(RED);
        printf("Stratum ERROR: ");
        res = -1;
        break;
      case rx0Stratum::STRATUM_DEBUG:
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

int handleRandomXStratumResponse(boost::json::object packet, bool isDev)
{
  // if (!isDev) {
  // if (!packet.contains("id")) return 0;
  int64_t id = packet["id"].to_number<int64_t>();
  switch (id)
  {
  case rx0Stratum::loginID:
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    boost::json::value *JV = isDev ? &devJob : &job;

    if (!packet["error"].is_null()) {
      setcolor(RED);
      printf("Stratum Error: %s\n", packet["error"].as_object()["message"].as_string().c_str());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return 1;
    }

    boost::json::object res = packet["result"].as_object();
    boost::json::object newJob = res["job"].as_object();

    std::string &l_ID = isDev ? randomx_login_dev : randomx_login;
    l_ID = res.at("id").as_string().c_str();

    (*JV) = newJob;
    
    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
        difficulty = rx_targetToDifficulty(newJob.at("target").as_string().c_str());
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
        fflush(stdout);
        setcolor(CYAN);
        printf("Dev fee: %.2f%% of your total hashrate\n", devFee);

        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        setcolor(CYAN);
        printf("Connected to dev node: %s\n", host.c_str());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }

    updateVM(newJob, isDev);

    *C = true;
  }
  break;
  case rx0Stratum::submitID:
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
      std::cout << "Stratum share accepted" << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    else
    {
      rejected++;
      if (!isDev)
        setcolor(RED);
      std::cout << "Stratum share rejected: " << packet.at("error").get_object()["message"].get_string().c_str() << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    break;
  }
  }
  return 0;
}

void rx0_stratum_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  // ctx.set_options(boost::asio::ssl::context::default_workarounds |
  //                 boost::asio::ssl::context::no_sslv2 |
  //                 boost::asio::ssl::context::no_sslv3 |
  //                 boost::asio::ssl::context::no_tlsv1 |
  //                 boost::asio::ssl::context::no_tlsv1_1);

  // beast::error_code ec;
  // boost::system::error_code jsonEc;

  // auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);

  // // Create a TCP socket
  // ctx.set_verify_mode(ssl::verify_none); // Accept self-signed certificates
  // //tcp::socket socket(ioc);
  // boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);
  // // Set a timeout on the operation
  // beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));

  // // Make the connection on the IP address we get from a lookup
  // beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  // if (ec)
  //   return fail(ec, "connect");

  // // Set the SNI hostname
  // if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
  // {
  //   throw beast::system_error{
  //       static_cast<int>(::ERR_get_error()),
  //       boost::asio::error::get_ssl_category()};
  // }

  // // Perform the SSL handshake
  // beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(300));
  // stream.async_handshake(ssl::stream_base::client, yield[ec]);
  // if (ec)
  //   return fail(ec, "handshake-randomx-strat");

  // boost::json::object packet = RandomXStratum::stratumCall;
  // packet.at("id") = RandomXStratum::subscribe.id;
  // packet.at("method") = RandomXStratum::subscribe.method;
  // std::string minerName = "tnn-miner/" + std::string(versionString);
  // packet.at("params") = boost::json::array({minerName, boost::json::array({"xel/1"})});
  // std::string login = boost::json::serialize(packet) + "\n";

  // // std::cout << login << std::endl;

  // beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // size_t trans = boost::asio::async_write(stream, boost::asio::buffer(login), yield[ec]);
  // if (ec)
  //   return fail(ec, "Stratum subscribe");

  // // Make sure login is successful
  // boost::asio::streambuf subRes;
  // beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // trans = boost::asio::read_until(stream, subRes, "\n");

  // std::string subResString = beast::buffers_to_string(subRes.data());
  // subRes.consume(trans);
  // boost::json::object subResJson = boost::json::parse(subResString.c_str(), jsonEc).as_object();
  // if (jsonEc)
  // {
  //   std::cerr << jsonEc.message() << std::endl;
  // }

  // // std::cout << boost::json::serialize(subResJson).c_str() << std::endl;

  // try {
  //   handleRandomXStratumResponse(subResJson, isDev);
  // } catch (const std::exception &e) {setcolor(RED);printf("%s", e.what());setcolor(BRIGHT_WHITE);}

  // // Authorize Stratum Worker
  // packet = RandomXStratum::stratumCall;
  // packet.at("id") = RandomXStratum::authorize.id;
  // packet.at("method") = RandomXStratum::authorize.method;
  // packet.at("params") = boost::json::array({wallet, worker, "x"});
  // std::string authorization = boost::json::serialize(packet) + "\n";

  // // std::cout << authorization << std::endl;

  // beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // trans = boost::asio::async_write(stream, boost::asio::buffer(authorization), yield[ec]);
  // if (ec)
  //   return fail(ec, "Stratum authorize");

  // // This buffer will hold the incoming message
  // beast::flat_buffer buffer;
  // std::stringstream workInfo;

  // RandomXStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

  // bool submitThread = false;
  // bool abort = false;

  // // boost::thread([&](){
  // //   submitThread = true;
  // //   while(true) {
  // //     if (abort) {
  // //       break;
  // //     }
  // //     try {
  // //       bool *B = isDev ? &submittingDev : &submitting;
  // //       if (*B)
  // //       {
  // //         bool err = false;
  // //         boost::json::object *S = &share;
  // //         if (isDev)
  // //           S = &devShare;

  // //         std::string msg = boost::json::serialize((*S)) + "\n";
  // //         // std::cout << "sending in: " << msg << std::endl;
  // //         beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(1));
  // //         boost::asio::write(stream, boost::asio::buffer(msg));
  // //         (*B) = false;
  // //         if (err) break;
  // //       }
  // //     } catch (const std::exception &e) {
  // //       setcolor(RED);
  // //       printf("\nSubmit thread error: %s\n", e.what());
  // //       setcolor(BRIGHT_WHITE);
  // //       break;
  // //     }
  // //     boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  // //   }
  // //   submitThread = false;
  // // });


  // boost::thread([&](){
  //   submitThread = true;
  //   while(!abort) {
  //     boost::unique_lock<boost::mutex> lock(mutex);
  //     bool *B = isDev ? &submittingDev : &submitting;
  //     cv.wait(lock, [&]{ return (data_ready && (*B)) || abort; });
  //     if (abort) break;
  //     try {
  //       boost::json::object *S = &share;
  //       if (isDev)
  //         S = &devShare;

  //       boost::system::error_code ec;
  //       std::string msg = boost::json::serialize((*S)) + "\n";
  //       // std::cout << "sending in: " << msg << std::endl;
  //       beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(1));
  //       boost::asio::async_write(stream, boost::asio::buffer(msg), [&](const boost::system::error_code& error, std::size_t bytes_transferred) {
  //         if (error) {
  //           printf("error on write: %s\n", error.message().c_str());
  //           fflush(stdout);
  //           abort = true;
  //         }
  //         if (!isDev) SpectreStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  //         // (*B) = false;
  //         // data_ready = false;
  //       });
  //       (*B) = false;
  //       data_ready = false;
  //     } catch (const std::exception &e) {
  //       setcolor(RED);
  //       printf("\nSubmit thread error: %s\n", e.what());
  //       fflush(stdout);
  //       setcolor(BRIGHT_WHITE);
  //       break;
  //     }
  //     //boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  //     boost::this_thread::yield();
  //   }
  //   submitThread = false;
  // });

  // while (true)
  // {
  //   bool *C = isDev ? &devConnected : &isConnected;
  //   bool *B = isDev ? &submittingDev : &submitting;
  //   try
  //   {
  //     if (
  //         RandomXStratum::lastReceivedJobTime > 0 &&
  //         std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - RandomXStratum::lastReceivedJobTime > RandomXStratum::jobTimeout)
  //     {
  //       setcolor(RED);
  //       printf("timeout\n");
  //       fflush(stdout);
  //       fflush(stdout);
  //       setcolor(BRIGHT_WHITE);
  //       setForDisconnected(C, B, &abort, &data_ready, &cv);

  //       for (;;) {
  //         if (!submitThread) break;
  //         boost::this_thread::yield();
  //       }
  //       stream.shutdown();
  //       return fail(ec, "Stratum session timed out");
  //     }

  //     boost::asio::streambuf response;
  //     std::stringstream workInfo;
  //     beast::get_lowest_layer(stream).expires_after(std::chrono::milliseconds(60000));
  //     trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
  //     if (ec && trans > 0)
  //     {
  //       setcolor(RED);
  //       printf("failed to read: %s\n", isDev ? "dev" : "user");
  //       fflush(stdout);
  //       setcolor(BRIGHT_WHITE);
  //       setForDisconnected(C, B, &abort, &data_ready, &cv);
  //       boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  //       cv.notify_all();

  //       for (;;) {
  //         if (!submitThread) {
  //           break;
  //         }
  //         boost::this_thread::yield();
  //       }
        
  //       stream.shutdown();
  //       return fail(ec, "async_read");
  //     }

  //     if (trans > 0)
  //     {
  //       std::scoped_lock<boost::mutex> lockGuard(wsMutex);
  //       std::vector<std::string> packets;
  //       std::string data = beast::buffers_to_string(response.data());
  //       // Consume the data from the buffer after processing it
  //       response.consume(trans);

  //       std::cout << data << std::endl;

  //       std::stringstream jsonStream(data);

  //       std::string line;
  //       while (std::getline(jsonStream, line, '\n'))
  //       {
  //         packets.push_back(line);
  //       }

  //       // for (std::string packet : packets)
  //       // {
  //       //   try
  //       //   {
  //       //     boost::json::object sRPC = boost::json::parse(packet).as_object();
  //       //     if (sRPC.contains("method"))
  //       //     {
  //       //       if (std::string(sRPC.at("method").as_string().c_str()).compare(RandomXStratum::s_ping) == 0)
  //       //       {
  //       //         boost::json::object pong({{"id", sRPC.at("id").get_uint64()},
  //       //                                   {"method", RandomXStratum::pong.method}});
  //       //         std::string pongPacket = std::string(boost::json::serialize(pong).c_str()) + "\n";
  //       //         trans = boost::asio::async_write(
  //       //             stream,
  //       //             boost::asio::buffer(pongPacket),
  //       //             yield[ec]);
  //       //         if (ec && trans > 0)
  //       //         {
  //       //           setcolor(RED);
  //       //           printf("ec && trans > 0\n");
  //       //           fflush(stdout);
  //       //           setcolor(BRIGHT_WHITE);
  //       //           setForDisconnected(C, B, &abort, &data_ready, &cv);

  //       //           for (;;)
  //       //           {
  //       //             if (!submitThread)
  //       //               break;
  //       //             boost::this_thread::yield();
  //       //           }
  //       //           stream.shutdown();
  //       //           return fail(ec, "Stratum pong");
  //       //         }
  //       //       }
  //       //       else
  //       //         handleRandomXStratumPacket(sRPC, isDev);
  //       //     }
  //       //     else
  //       //     {
  //       //       handleRandomXStratumResponse(sRPC, isDev);
  //       //     }
  //       //   }
  //       //   catch (const std::exception &e)
  //       //   {
  //       //     setcolor(RED);
  //       //     printf("%s\n", e.what());
  //       //     fflush(stdout);
  //       //     setcolor(BRIGHT_WHITE);
  //       //   }
  //       // }
  //     }
  //   }
  //   catch (const std::exception &e)
  //   {
  //     bool *C = isDev ? &devConnected : &isConnected;
  //     printf("exception\n");
  //     fflush(stdout);
  //     setForDisconnected(C, B, &abort, &data_ready, &cv);

  //     for (;;) {
  //       if (!submitThread) break;
  //       boost::this_thread::yield();
  //     }
  //     stream.shutdown();
  //     setcolor(RED);
  //     std::cerr << e.what() << std::endl;
  //     fflush(stdout);
  //     setcolor(BRIGHT_WHITE);
  //   }
  //   boost::this_thread::yield();
  // }
}

void rx0_stratum_session_nossl(
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

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  boost::beast::tcp_stream stream(ioc);

  // Set a timeout on the operation
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));

  // Make the connection on the IP address we get from a lookup
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec)
    return fail(ec, "connect");

  boost::json::object packet = rx0Stratum::stratumCall;
  packet.at("id") = rx0Stratum::login.id;
  packet.at("method") = rx0Stratum::login.method;

  std::string userAgent = "tnn-miner/" + std::string(versionString);


  boost::json::object loginParams = {
    {"login", wallet},
    {"pass", "x"},
    {"rigid", worker},
    {"agent", userAgent.c_str()}
  };

  packet.at("params") = loginParams;
  std::string login = boost::json::serialize(packet) + "\n";

  // std::cout << login << std::endl;
  // fflush(stdout);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  size_t trans = boost::asio::async_write(stream, boost::asio::buffer(login), yield[ec]);
  if (ec)
    return fail(ec, "Stratum login");

  // try {
  //   handleRandomXStratumResponse(subResJson, isDev);
  // } catch (const std::exception &e) {setcolor(RED);printf("%s", e.what());fflush(stdout);setcolor(BRIGHT_WHITE);}

  // This buffer will hold the incoming message
  beast::flat_buffer buffer;
  std::stringstream workInfo;

  rx0Stratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

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
          if (!isDev) SpectreStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
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
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;
    try
    {
      if (
          rx0Stratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - rx0Stratum::lastReceivedJobTime > rx0Stratum::jobTimeout)
      {
        setcolor(RED);
        printf("timeout\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);

        for (;;) {
          if (!submitThread) break;
          boost::this_thread::yield();
        }
        stream.close();
        return fail(ec, "Stratum session timed out");
      }

      boost::asio::streambuf response;
      std::stringstream workInfo;
      beast::get_lowest_layer(stream).expires_after(std::chrono::minutes(5));
      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec && trans > 0)
      {
        setcolor(RED);
        printf("failed to read: %s\n", isDev ? "dev" : "user");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
        cv.notify_all();

        for (;;) {
          if (!submitThread) {
            break;
          }
          boost::this_thread::yield();
        }
        
        stream.close();
        return fail(ec, "async_read");
      }

      if (trans > 0)
      {
        std::scoped_lock<boost::mutex> lockGuard(wsMutex);
        std::vector<std::string> packets;
        std::string data = beast::buffers_to_string(response.data());
        // Consume the data from the buffer after processing it
        response.consume(trans);

        // std::cout << data << std::endl;
        fflush(stdout);

        std::stringstream jsonStream(data);

        std::string line;
        while (std::getline(jsonStream, line, '\n'))
        {
          packets.push_back(line);
        }

        for (std::string packet : packets)
        {
          try
          {
            boost::json::object sRPC = boost::json::parse(packet).as_object();
            if (sRPC.contains("method"))
            {
              if (std::string(sRPC.at("method").as_string().c_str()).compare(rx0Stratum::s_ping) == 0)
              {
                boost::json::object pong({{"id", sRPC.at("id").get_uint64()},
                                          {"method", rx0Stratum::pong.method}});
                std::string pongPacket = std::string(boost::json::serialize(pong).c_str()) + "\n";
                trans = boost::asio::async_write(
                    stream,
                    boost::asio::buffer(pongPacket),
                    yield[ec]);
                if (ec && trans > 0)
                {
                  setcolor(RED);
                  printf("ec && trans > 0\n");
                  fflush(stdout);
                  setcolor(BRIGHT_WHITE);
                  setForDisconnected(C, B, &abort, &data_ready, &cv);

                  for (;;)
                  {
                    if (!submitThread)
                      break;
                    boost::this_thread::yield();
                  }
                  stream.close();
                  return fail(ec, "Stratum pong");
                }
              }
              else
                handleRandomXStratumPacket(sRPC, isDev);
            }
            else
            {
              handleRandomXStratumResponse(sRPC, isDev);
            }
          }
          catch (const std::exception &e)
          {
            setcolor(RED);
            printf("%s\n", e.what());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      bool *C = isDev ? &devConnected : &isConnected;
      printf("exception\n");
      fflush(stdout);
      setForDisconnected(C, B, &abort, &data_ready, &cv);

      for (;;) {
        if (!submitThread) break;
        boost::this_thread::yield();
      }
      stream.close();
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    boost::this_thread::yield();
  }
  // submission_thread.interrupt();
}
