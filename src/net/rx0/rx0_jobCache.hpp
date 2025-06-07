#pragma once
#include <randomx/randomx.h>
#include <string>

extern std::atomic<bool> randomx_ready;
extern std::atomic<bool> randomx_ready_dev;

extern std::string randomx_cacheKey;
extern std::string randomx_cacheKey_dev;

extern std::string randomx_login;
extern std::string randomx_login_dev;

extern std::atomic<bool> globalInDevBatch;
extern std::atomic<bool> isDevOnActiveCache; 
extern std::mutex cacheSwitchMutex; 

extern std::atomic<bool> sharedDatasetMode;
extern std::string currentDatasetSeedHash;
extern std::mutex datasetMutex;

extern std::atomic<bool> needsDatasetUpdate;
extern std::atomic<bool> inUpdate;

extern std::atomic<bool> firstBatch;

inline Num rx0_calcTarget(boost::json::value &job) {
  int tLen = job.at("target").as_string().size();
  Num cmpTarget;

  if (tLen <= 16)
  {
    uint32_t cmpTargetInt = boost_swap_impl::stoul(job.at("target").as_string().c_str(), nullptr, 16);

    cmpTargetInt = __builtin_bswap32(cmpTargetInt);
    cmpTarget = Num(cmpTargetInt) << 224;
  }
  else if (tLen <= 32)
  {
    uint64_t cmpTargetInt = boost_swap_impl::stoull(job.at("target").as_string().c_str(), nullptr, 32);

    cmpTargetInt = __builtin_bswap64(cmpTargetInt);
    cmpTarget = Num(cmpTargetInt) << 192;
  }
  else
  {
    cmpTarget = Num(job.at("target").as_string().c_str(), 16);
    // NOTE still needs the have 32-byte reversal implemented probably
  }
  return cmpTarget;
}

void randomx_update_data(randomx_cache* rc, randomx_dataset* rd, void *seed, 
                        size_t seedSize, int threadCount, bool isDev);

void updateVM(boost::json::object &newJob, bool isDev);
void updateDataset(randomx_cache* cache, std::string seedHash, bool isDev);

inline bool checkAndUpdateDatasetIfNeeded(bool isDev) {
  // Get references to relevant variables based on isDev
  std::atomic<bool> &myReady = isDev ? randomx_ready_dev : randomx_ready;
  std::atomic<bool> &altReady = (!isDev) ? randomx_ready_dev : randomx_ready;
  std::string &myCacheKey = isDev ? randomx_cacheKey_dev : randomx_cacheKey;
  randomx_cache* &myCache = isDev ? rxCache_dev : rxCache;
  
  // Fast check - only proceed if not ready and cache key exists
  if (!myReady.load() && !myCacheKey.empty()) {
    fflush(stdout);
    // Check if our cache key differs from current dataset
    std::scoped_lock<std::mutex> lock(datasetMutex);
    if (currentDatasetSeedHash != myCacheKey) {
      // Thread-safe check if dataset needs updating
      updateDataset(myCache, myCacheKey, isDev);
      altReady.store(false);
    }
    myReady.store(true);

    return true;  // We did some work
  }
  
  return false;  // No work needed
}