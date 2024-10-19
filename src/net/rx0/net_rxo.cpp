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

#include <cpp-http/httplib.h>

#include <num.h>

#include "rx0_jobCache.hpp"
#include <randomx/randomx.h>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

static const char *gtID = "0";
static const char *submitID = "7";

static const char *jsonType = "application/json";

Num maxTarget = Num(2).pow(256);

void updateVM(boost::json::object &newJob, bool isDev) {
  std::string &refKey = isDev ? randomx_cacheKey_dev : randomx_cacheKey;

  if (std::string(newJob.at("seed_hash").as_string().c_str()).compare(refKey) != 0) {
    setcolor(isDev ? CYAN : BRIGHT_YELLOW);
    printf("\n");
    if (isDev) printf("DEV | ");
    printf("Reinitializing RandomX cache...\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    randomx_cache *refCache = isDev ? rxCache_dev : rxCache;
    randomx_dataset *refDataset = isDev ? rxDataset_dev : rxDataset;

    bool &status = isDev ? randomx_ready_dev : randomx_ready;
    status = false;

    unsigned char *newSeed = (unsigned char *)malloc(32);
    hexstrToBytes(newJob.at("seed_hash").as_string().c_str(), newSeed);

    // printf("%s | seedHash\n%s | seedBuffer\n", newJob.at("seed_hash").as_string().c_str(), hexStr(newSeed, 32).c_str());

    randomx_update_data(refCache, refDataset, newSeed, 32, std::thread::hardware_concurrency());

    delete[] newSeed;

    refKey = newJob.at("seed_hash").as_string().c_str();

    setcolor(isDev ? CYAN : BRIGHT_YELLOW);
    printf("\n");
    if (isDev) printf("DEV | ");
    printf("RandomX cache updated successfully\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    status = true;
  }
}


void rx0_session(
    std::string sessionHost,
    std::string const &port,
    std::string const &wallet,
    bool isDev)
{
  httplib::Client daemon(sessionHost, stoul(port));

  // submit thread here

  // job thread here
  uint64_t chainHeight = 0;

  boost::json::object get_block_temnplate = {
    {"id", gtID},
    {"jsonrpc", "2.0"},
    {"method", "get_block_template"},
    {"params", {
      {"wallet_address", wallet.c_str()},
      {"reserve_size", 60}
    }}
  };
  std::string gbtReq = boost::json::serialize(get_block_temnplate);

  bool submitThread = false;
  bool abort = false;

  auto rx0_getTemplate = [&]() -> int {
    auto res = daemon.Post("/json_rpc", gbtReq, jsonType);
    if (res && res->status == 200)
    {
      std::string response = res->body;
      boost::json::object resJson = boost::json::parse(response).as_object();

      if (resJson["error"].is_null()) {
        boost::json::object newJob = resJson["result"].as_object();

        if ((isDev ? devJob : job).as_object()["template"].is_null() ||
          std::string(newJob["blocktemplate_blob"].as_string().c_str()).compare(
          (isDev ? devJob : job).as_object()["template"].as_string().c_str()) != 0)
        {
          chainHeight = newJob.at("height").to_number<uint64_t>();
          boost::json::value &J = isDev ? devJob : job;

          Num newTarget = maxTarget / Num(newJob["difficulty"].to_number<uint64_t>());
          std::vector<char> tmp;
          newTarget.print(tmp, 16);

          std::string tString = (const char *)tmp.data();
          if (!isDev) difficulty = newJob["difficulty"].to_number<uint64_t>();
          else difficultyDev = newJob["difficulty"].to_number<uint64_t>();
    

          J = {
              {"blob", newJob["blockhashing_blob"].as_string().c_str()},
              {"template", newJob["blocktemplate_blob"].as_string().c_str()},
              {"target", tString.c_str()},
              {"seed_hash", newJob["seed_hash"].as_string().c_str()}
          };

          // std::cout << "Received template: " << response << std::endl;
        }

        // std::cout << "difficulty: " << newJob.at("difficulty").to_number<uint64_t>() << std::endl;

        bool *C = isDev ? &devConnected : &isConnected;
        if (!*C)
        {
          if (!isDev)
          {
            difficulty = newJob.at("difficulty").to_number<uint64_t>();
            setcolor(BRIGHT_YELLOW);
            printf("Mining at: %s to wallet %s\n", sessionHost.c_str(), wallet.c_str());
            fflush(stdout);
            setcolor(CYAN);
            printf("Dev fee: %.2f%% of your total hashrate\n", devFee);
    
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
          else
          {
            setcolor(CYAN);
            printf("Connected to dev node: %s\n", sessionHost.c_str());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
        }

        updateVM(newJob, isDev);
        jobCounter++;

        *C = true;
        return 0;
      } else {
        setcolor(RED);
        // printf("get_block_template: %s\n", resJson.at("error").as_object().at("message").as_string().c_str());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        return 1;
      }
    }
    else
    {
      fail("get_block_template", (res ? std::to_string(res->status).c_str() : "No response"));
      return 1;
    }
  };


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

        std::string msg = boost::json::serialize((*S)) + "\n";
        // std::cout << "sending in: " << msg << std::endl;
        auto res = daemon.Post("/json_rpc", msg, jsonType);
        if (res && res->status == 200)
        {
          boost::json::object result = boost::json::parse(res->body).as_object();
          if (!result["error"].is_null()) {
            setcolor(isDev ? CYAN : RED);
            printf("%s\n", result["error"].as_object()["message"].as_string().c_str());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);

            rejected++;
          } else {
            // std::cout << boost::json::serialize(result) << std::endl << std::flush;
            setcolor(isDev ? CYAN : BRIGHT_YELLOW);
            printf("\n");
            if (isDev) printf("DEV | ");
            printf("Block accepted!\n");
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
            accepted++;

            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
            rx0_getTemplate();
          }
        } else {
          fail("submit_block", (res ? std::to_string(res->status).c_str() : "No response"));
        }
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


  for (;;)
  {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;
    try
    {
      if (rx0_getTemplate()) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);

        for (;;)
        {
          if (!submitThread)
            break;
          boost::this_thread::yield();
        }       
        return;
      }
      boost::this_thread::sleep_for(boost::chrono::seconds(5));
    }
    catch (const std::exception &e)
    {
      bool *C = isDev ? &devConnected : &isConnected;
      printf("exception\n");
      fflush(stdout);
      setForDisconnected(C, B, &abort, &data_ready, &cv);

      for (;;)
      {
        if (!submitThread)
          break;
        boost::this_thread::yield();
      }
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return;
    }
    boost::this_thread::yield();
  }
}