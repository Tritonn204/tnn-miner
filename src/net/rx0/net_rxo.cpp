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
#include "numa_optimizer.hpp"

#include <atomic>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

static const char *gtID = "0";
static const char *submitID = "7";

static const char *jsonType = "application/json";

Num maxTarget = Num(2).pow(256);

std::atomic<bool> sharedDatasetMode{true};
std::string currentDatasetSeedHash = "";
std::mutex datasetMutex;

// Progress tracking for dataset initialization
std::atomic<int> datasetInitProgress{0};
std::atomic<bool> datasetInitInProgress{false};

void updateDataset(randomx_cache* cache, std::string seedHash, bool isDev) {
    printf("\n");
    fflush(stdout);
    
    datasetInitProgress.store(0);
    datasetInitInProgress.store(true);
    
    unsigned char *seed = (unsigned char *)malloc(32);
    if (!seed) {
        setcolor(RED);
        printf("\nFailed to allocate memory for seed\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        datasetInitInProgress.store(false);
        return;
    }
    
    hexstrToBytes(seedHash.c_str(), seed);
    
    randomx_update_data(cache, rxDataset, seed, 32, std::thread::hardware_concurrency(), isDev);
    
    free(seed);
    
    currentDatasetSeedHash = seedHash;
    
    datasetInitInProgress.store(false);
    
    setcolor(isDev ? CYAN : BRIGHT_YELLOW);
    printf("\n");
    if (isDev) printf("DEV | ");
    printf("RandomX dataset initialized successfully\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
}

void updateVM(boost::json::object &newJob, bool isDev) {
    std::string newSeedHash = std::string(newJob.at("seed_hash").as_string());
    
    randomx_cache* &targetCache = isDev ? rxCache_dev : rxCache;
    std::string &targetCacheKey = isDev ? randomx_cacheKey_dev : randomx_cacheKey;
    std::atomic<bool> &targetReady = isDev ? randomx_ready_dev : randomx_ready;
    
    if (newSeedHash == targetCacheKey) {
        return;
    }
    
    setcolor(isDev ? CYAN : BRIGHT_YELLOW);
    printf("\n");
    if (isDev) printf("DEV | ");
    printf("Reinitializing RandomX cache...\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    
    unsigned char *newSeed = (unsigned char *)malloc(32);
    if (!newSeed) {
        setcolor(RED);
        printf("\nFailed to allocate memory for seed\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        return;
    }
    
    hexstrToBytes(newSeedHash.c_str(), newSeed);
    randomx_init_cache(targetCache, newSeed, 32);
    free(newSeed);
    
    targetCacheKey = newSeedHash;
    targetReady.store(false);  // ← Gate mining threads
    
    needsDatasetUpdate.store(true);
    cv.notify_all();
    
    setcolor(isDev ? CYAN : BRIGHT_YELLOW);
    printf("\n");
    if (isDev) printf("DEV | ");
    printf("RandomX cache updated successfully\n");
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
}

void rx0_session(
    std::string sessionHost,
    std::string const &port,
    std::string const &wallet,
    bool isDev)
{
  // Mutex for HTTP client thread safety
  std::mutex httpMutex;
  httplib::Client daemon(sessionHost, stoul(port));
  
  // Set reasonable timeouts
  daemon.set_connection_timeout(30);
  daemon.set_read_timeout(30);
  daemon.set_write_timeout(30);

  uint64_t chainHeight = 0;

  boost::json::object get_block_template = {
    {"id", gtID},
    {"jsonrpc", "2.0"},
    {"method", "get_block_template"},
    {"params", {
      {"wallet_address", wallet},
      {"reserve_size", 60}
    }}
  };
  std::string gbtReq = boost::json::serialize(get_block_template);

  std::atomic<bool> abort{false};
  bool submitThreadRunning = true;
  bool cacheThreadRunning = true;

  auto rx0_getTemplate = [&]() -> int {
    std::lock_guard<std::mutex> httpLock(httpMutex);
    
    auto res = daemon.Post("/json_rpc", gbtReq, jsonType);
    if (res && res->status == 200)
    {
      std::string response = res->body;
      boost::json::object resJson = boost::json::parse(response).as_object();

      if (resJson["error"].is_null()) {
        boost::json::object newJob = resJson["result"].as_object();

        // Safe string extraction
        std::string newTemplateBlob = std::string(newJob["blocktemplate_blob"].as_string());
        
        boost::json::value &J = isDev ? devJob : job;
        bool isNewJob = J.as_object()["template"].is_null();
        
        if (!isNewJob) {
          std::string currentTemplate = std::string(J.as_object()["template"].as_string());
          isNewJob = (newTemplateBlob != currentTemplate);
        }

        if (isNewJob)
        {
          chainHeight = newJob.at("height").to_number<uint64_t>();

          Num newTarget = maxTarget / Num(newJob["difficulty"].to_number<uint64_t>());
          std::vector<char> tmp;
          newTarget.print(tmp, 16);

          std::string tString(tmp.data(), tmp.size());
          
          if (!isDev) 
            difficulty = newJob["difficulty"].to_number<uint64_t>();
          else 
            difficultyDev = newJob["difficulty"].to_number<uint64_t>();

          // Safe string extraction for all fields
          std::string blobStr = std::string(newJob["blockhashing_blob"].as_string());
          std::string seedHashStr = std::string(newJob["seed_hash"].as_string());

          J = {
              {"blob", blobStr},
              {"template", newTemplateBlob},
              {"target", tString},
              {"seed_hash", seedHashStr}
          };

          updateVM(newJob, isDev);
          jobCounter++;
        }

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

        *C = true;
        return 0;
      } else {
        setcolor(RED);
        if (resJson.contains("error") && resJson["error"].is_object()) {
          auto errObj = resJson["error"].as_object();
          if (errObj.contains("message")) {
            std::string errMsg = std::string(errObj["message"].as_string());
            printf("get_block_template: %s\n", errMsg.c_str());
          }
        }
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

  std::thread subThread([&](){
    while (!abort.load()) {
      std::unique_lock<std::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&]{ return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;
      
      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        
        // Thread-safe HTTP access
        httplib::Result res;
        {
          std::lock_guard<std::mutex> httpLock(httpMutex);
          res = daemon.Post("/json_rpc", msg, jsonType);
        }
        
        if (res && res->status == 200)
        {
          boost::json::object result = boost::json::parse(res->body).as_object();
          if (!result["error"].is_null()) {
            setcolor(isDev ? CYAN : RED);
            if (result["error"].is_object() && result["error"].as_object().contains("message")) {
              std::string errMsg = std::string(result["error"].as_object()["message"].as_string());
              printf("%s\n", errMsg.c_str());
            }
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
            rejected++;
          } else {
            setcolor(isDev ? CYAN : BRIGHT_YELLOW);
            printf("\n");
            if (isDev) printf("DEV | ");
            printf("Block accepted!\n");
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
            accepted++;

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            rx0_getTemplate();
          }
        } else {
          fail("submit_block", (res ? std::to_string(res->status).c_str() : "No response"));
        }
        
        *B = false;
        data_ready = false;
      } catch (const std::exception &e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        break;
      }
      std::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  // ORIGINAL cache thread synchronization preserved
  std::thread cacheThread([&]() {
    while (!abort.load()) {
      std::unique_lock<std::mutex> lock(mutex);
      
      cv.wait(lock, [&]{ 
        return needsDatasetUpdate.load() || abort.load(); 
      });
      
      if (abort.load()) break;
      
      if (globalInDevBatch.load() == isDev) {
        try {
          needsDatasetUpdate.exchange(false);
          checkAndUpdateDatasetIfNeeded(isDev);
        } catch (const std::exception &e) {
          setcolor(RED);
          printf("\nDataset update error: %s\n", e.what());
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
        }
      }
      
      std::this_thread::yield();
    }
    cacheThreadRunning = false;
  });

  while (!ABORT_MINER && !abort.load())
  {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;
    
    try
    {
      if (rx0_getTemplate()) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      printf("\nSession exception: %s\n", e.what());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }
    
    std::this_thread::yield();
  }

  // Clean shutdown
  abort = true;
  cv.notify_all();

  if (cacheThreadRunning) {
    if (cacheThread.joinable()) {
      cacheThread.join();
    }
  }

  if (submitThreadRunning) {
    if (subThread.joinable()) {
      subThread.join();
    }
  }
}