#pragma once
#include <randomx/randomx.h>
#include <string>

extern bool randomx_ready;
extern bool randomx_ready_dev;

extern std::string randomx_cacheKey;
extern std::string randomx_cacheKey_dev;

extern std::string randomx_login;
extern std::string randomx_login_dev;

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

void randomx_update_data(randomx_cache* rc, randomx_dataset* rd, void *seed, size_t seedSize, int threadCount);
void randomx_update_data_numa(randomx_cache* rc, randomx_dataset **datasets, 
                             void *seed, size_t seedSize, int threadCount);

void updateVM(boost::json::object &newJob, bool isDev);