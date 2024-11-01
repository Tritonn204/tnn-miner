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
#include <spectrex/spectrex.h>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

int handleSpectreStratumPacket(boost::json::object packet, SpectreStratum::jobCache *cache, bool isDev)
{
  std::string M = packet.at("method").get_string().c_str();
  if (M.compare(SpectreStratum::s_notify) == 0)
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    boost::json::value *J = isDev ? &devJob : &job;
    int64_t *h = isDev ? &devHeight : &ourHeight;

    uint64_t h1 = packet["params"].as_array()[1].as_array()[0].get_uint64();
    uint64_t h2 = packet["params"].as_array()[1].as_array()[1].get_uint64();
    uint64_t h3 = packet["params"].as_array()[1].as_array()[2].get_uint64();
    uint64_t h4 = packet["params"].as_array()[1].as_array()[3].get_uint64();

    uint64_t comboHeader[4] = {h1, h2, h3, h4};

    (*J).as_object()["jobId"] = packet["params"].as_array()[0].get_string().c_str();

    bool isEqual = true;
    for (int i = 0; i < 4; i++) {
      isEqual &= comboHeader[i] == cache->header[i];
    }
    if (!isEqual) {
      uint64_t &N = isDev ? nonce0_dev : nonce0;
      N = 0;
    }

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

    if(!isEqual && !beQuiet) {
      setcolor(CYAN);
      // if (!isDev)
      printf("\n");
      if (isDev) printf("DEV | ");
      printf("\nStratum: new job received\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    SpectreStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();


    (*J).as_object()["template"] = std::string(newTemplate, SpectreX::INPUT_SIZE*2);
    (*J).as_object()["jobId"] = packet["params"].as_array()[0].get_string().c_str();

    if (!isDev) std::cout << "job packet: " << boost::json::serialize(packet).c_str() << std::endl;
    if (!isDev) std::cout << "stored job: " << boost::json::serialize(*J).c_str() << std::endl;

    if (isDev) std::cout << "Dev job packet: " << boost::json::serialize(packet).c_str() << std::endl;
    if (isDev) std::cout << "Devstored job: " << boost::json::serialize(*J).c_str() << std::endl;

    bool *C = isDev ? &devConnected : &isConnected;
    if (!*C)
    {
      if (!isDev)
      {
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
        printf("Connected to dev node\n");
        fflush(stdout);
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

    uint256_t *dRef = isDev ? &bigDiff_dev : &bigDiff;
    *dRef = SpectreX::diffToTarget(*d);

    jobCounter++;
    // printf("%f\n", (*d));
  }
  else if (M.compare(SpectreStratum::s_setExtraNonce) == 0)
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    boost::json::value *J = isDev ? &devJob : &job;
    // uint64_t *h = isDev ? &devHeight : &ourHeight;

    // std::string bs = (*J).at("template").get<std::string>();
    // char *blob = (char *)bs.c_str();
    const char *en = packet.at("params").as_array()[0].as_string().c_str();
    // char *c = NULL;
    int enLen = packet.at("params").as_array()[0].as_string().size();

    // uint32_t EN = strtoul(en, &c, 16);


    // memset(&blob[48], '0', 64);
    // memcpy(&blob[48], en, enLen);

    (*J).as_object()["extraNonce"] = std::string(en);

    // (*h)++;
    // jobCounter++;
  }
  else if (M.compare(SpectreStratum::s_print) == 0)
  {

    int lLevel = packet.at("params").as_array()[0].to_number<int64_t>();
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

      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return res;
    }
  } else {
    std::cout << "Stratum: unrecognized packet: " << boost::json::serialize(packet).c_str() << std::endl;
  }
  return 0;
}

int handleSpectreStratumResponse(boost::json::object packet, bool isDev)
{
  // if (!isDev) {
  // if (!packet.contains("id")) return 0;
  int64_t id = packet["id"].to_number<int64_t>();
  // std::cout << "Stratum packet: " << boost::json::serialize(packet).c_str() << std::endl;

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
        fflush(stdout);
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
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        if (!isDev) rejected++;
        if (!isDev)
          setcolor(RED);

        boost::json::string ERR;
        if (packet["error"].is_array()) {
          ERR = packet.at("error").as_array()[1].as_string();
        } else {
          ERR = packet.at("error").at("message").get_string();
        }
        std::cout << "Stratum: share rejected: " << ERR.c_str() << std::endl;
        
        fflush(stdout);
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

  SpectreStratum::jobCache jobCache;

  // Subscribe to Stratum
  packet = SpectreStratum::stratumCall;
  packet["id"] = SpectreStratum::subscribe.id;
  packet["method"] = SpectreStratum::subscribe.method;
  packet["params"] = boost::json::array({
    minerName
  });
  std::string subscription = boost::json::serialize(packet) + "\n";

  // std::cout << authResString << std::endl;
  size_t trans;

  try {
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    trans = boost::asio::async_write(stream, boost::asio::buffer(subscription), yield[ec]);
    if (ec)
      return fail(ec, "Stratum subscribe");

    boost::asio::streambuf subRes;
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    trans = boost::asio::read_until(stream, subRes, "\n");

    std::string subResString = beast::buffers_to_string(subRes.data());
    subRes.consume(trans);
    if (jsonEc)
    {
      std::cerr << jsonEc.message() << std::endl;
    }

    // std::cout << "sub result: " << subResString << std::endl << std::flush;

    std::stringstream  jsonStream(subResString);
    std::vector<std::string> packets;

    std::string line;
    while(std::getline(jsonStream,line,'\n'))
    {
      packets.push_back(line);
    }

    for (std::string packet : packets) {
      boost::json::object subRPC = boost::json::parse(packet.c_str()).as_object();
      if (subRPC.contains("method"))
      {
        handleSpectreStratumPacket(subRPC, &jobCache, isDev);
      } 
    }
  } catch (const std::exception &e) {
    setcolor(RED);
    printf("\nStratum Subscribe error: %s\n", e.what());
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }
  // Authorize Stratum Worker
  packet = SpectreStratum::stratumCall;
  packet.at("id") = SpectreStratum::authorize.id;
  packet.at("method") = SpectreStratum::authorize.method;
  packet.at("params") = boost::json::array({wallet + "." + worker});

  std::string authorization = boost::json::serialize(packet) + "\n";

  // std::cout << authorization << std::endl;
  try {
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(authorization), yield[ec]);
    if (ec)
      return fail(ec, "Stratum authorize");

    boost::asio::streambuf authRes;
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    trans = boost::asio::read_until(stream, authRes, "\n");

    std::string authResString = beast::buffers_to_string(authRes.data());
    authRes.consume(trans);
    if (jsonEc)
    {
      std::cerr << jsonEc.message() << std::endl;
    }

    // std::cout << "auth result: " << authResString << std::endl << std::flush;

    std::stringstream  jsonStream(authResString);
    std::vector<std::string> packets;

    std::string line;
    while(std::getline(jsonStream,line,'\n'))
    {
      packets.push_back(line);
    }

    for (std::string packet : packets) {
      boost::json::object authRPC = boost::json::parse(packet.c_str()).as_object();
      if (authRPC.contains("method"))
      {
        handleSpectreStratumPacket(authRPC, &jobCache, isDev);
      }
    }
  } catch (const std::exception &e) {
    setcolor(RED);
    printf("\nStratum Authorize error: %s\n", e.what());
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }
  // SpectreStratum::stratumCall;
  // packet.at("id") = SpectreStratum::subscribe.id;
  // packet.at("method") = SpectreStratum::subscribe.method;
  // packet.at("params") = {minerName};

  // beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  // trans = boost::asio::read_until(stream, subRes, "\n");

  // std::string subResString = beast::buffers_to_string(subRes.data());
  // subRes.consume(trans);

  // mutex.lock();
  // printf("before packet\n");
  // std::cout << subResString << std::endl;

  // printf("before parse\n");
  // mutex.unlock();
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

  std::string chopQueue = "NULL";

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
          SpectreStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - SpectreStratum::lastReceivedJobTime > SpectreStratum::jobTimeout)
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
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));

      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec) {
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
        // std::scoped_lock<boost::mutex> lockGuard(wsMutex);
        std::vector<std::string> packets;
        std::string data = beast::buffers_to_string(response.data());
        // Consume the data from the buffer after processing it
        response.consume(trans);

        // std::cout << "received: " << data << std::endl << std::flush;
        // printf("received data\n");
        fflush(stdout);

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
                if(ec) {
                  printf("error on write(%zu): %s\n", trans, ec.message().c_str());
                  fflush(stdout);
                }
                if (ec && trans > 0) {
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
                handleSpectreStratumPacket(sRPC, &jobCache, isDev);
            }
            else
            {
              handleSpectreStratumResponse(sRPC, isDev);
            } 
          } catch(const std::exception &e){
            // printf("\n\n packet count: %d, msg size: %llu\n\n", packets.size(), trans);
            setcolor(RED);
            // printf("BEFORE PACKET\n");
            // std::cout << "BAD PACKET: " << packet << std::endl;
            // printf("AFTER PACKET\n");
            // std::cerr << e.what() << std::endl;
            setcolor(BRIGHT_WHITE);
            bool tryParse = (chopQueue.compare("NULL") != 0);

            if (tryParse) {
              chopQueue += packet;
              printf("resulting json string: %s\n\n", chopQueue.c_str());
              try
              {
                packets.clear();
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
                    if(ec) {
                      printf("error on write(%zu): %s\n", trans, ec.message().c_str());
                      fflush(stdout);
                    }
                    if (ec && trans > 0) {
                      printf("ec && trans > 0\n");
                      fflush(stdout);
                      setForDisconnected(C, B, &abort, &data_ready, &cv);
  
                      for (;;)
                      {
                        if (!submitThread) {
                          break;
                        }
                        boost::this_thread::yield();
                      }
                      stream.close();
                      return fail(ec, "Stratum pong");
                    }
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
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
              }
            } else {
              chopQueue = packet;
              printf("partial json start = %s\n", chopQueue.c_str());
            } 
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
      return fail(ec, "Stratum session error");
    }
    boost::this_thread::yield();
  }

  // submission_thread.interrupt();
  // printf("\n\n\nflagged connection loss\n");
  // stream.close();
}