#pragma once
#include <randomx/randomx.h>
#include <string>

extern bool randomx_ready;
extern bool randomx_ready_dev;

extern std::string randomx_cacheKey;
extern std::string randomx_cacheKey_dev;

void randomx_update_data(randomx_cache* rc, randomx_dataset* rd, void *seed, size_t seedSize, int threadCount);