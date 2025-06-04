#include "miners.hpp"
#include "numa_optimizer.h"  // Add this

#include <net/rx0/rx0_jobCache.hpp>
#include <randomx/randomx.h>
#include <randomx/dataset.hpp>
#include <randomx/common.hpp>
#include <randomx/jit_compiler.hpp>
#include <stratum/stratum.h>
#include <thread>
#include <vector>
#include <map>

// Change from single datasets to per-NUMA-node datasets
randomx_dataset* rxDatasets_numa[256];      // One per NUMA node

// Fallback for non-NUMA systems
randomx_dataset* rxDataset;
randomx_dataset* rxDataset_dev;

randomx_cache* rxCache;
randomx_cache* rxCache_dev;

randomx_flags rxFlags;
bool rx_hugePages;
bool rx_numa_enabled = false;
int numa_nodes = 1;

// Map thread ID to NUMA node
std::map<int, int> thread_numa_map;
std::mutex numa_map_mutex;

std::string randomx_cacheKey = "0000000000000000000000000000000000000000000000000000000000000000";
std::string randomx_cacheKey_dev = "0000000000000000000000000000000000000000000000000000000000000000";

std::string randomx_login;
std::string randomx_login_dev;

std::mutex cacheSwitchMutex;
std::atomic<bool> isDevOnActiveCache{false};

struct NUMADatasetInfo {
    randomx_dataset* dataset;
    size_t size;
    int numa_node;
};

std::map<randomx_dataset*, NUMADatasetInfo> numa_dataset_map;

void randomx_init_intern(int threadCount) {
  if (nullptr == randomx::selectArgonImpl(rxFlags)) {
    throw std::runtime_error("Unsupported Argon2 implementation");
  }
  if ((rxFlags & RANDOMX_FLAG_JIT) && !RANDOMX_HAVE_COMPILER) {
    throw std::runtime_error("JIT compilation is not supported on this platform. Try without --jit");
  }
  if (!(rxFlags & RANDOMX_FLAG_JIT) && RANDOMX_HAVE_COMPILER) {
    std::cout << "WARNING: You are using the interpreter mode. Use --jit for optimal performance." << std::endl;
  }

  // Allocate both caches for quick switching
  rxCache = randomx_alloc_cache(rxFlags);
  rxCache_dev = randomx_alloc_cache(rxFlags);

  if (rxCache == nullptr || rxCache_dev == nullptr) {
    throw std::runtime_error("RandomX Cache Alloc Failed");
  }

  // Initialize NUMA if available
  if (NUMAOptimizer::initialize()) {
    numa_nodes = NUMAOptimizer::getMemoryNodes();
    
    if (numa_nodes < 2 || threads < 4 || !lockThreads) {
      rx_numa_enabled = false;
      numa_nodes = 1;
      rxDataset = randomx_alloc_dataset(rxFlags);
      
      if (!rxDataset) {
        throw std::runtime_error("Failed to allocate dataset");
      }
    } else {
      rx_numa_enabled = true;

      setcolor(BRIGHT_YELLOW);
      std::cout << " NUMA enabled: " << numa_nodes << " memory nodes detected" << std::endl;
      if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
        std::cout << " Large pages enabled - will allocate on NUMA nodes" << std::endl;
      }
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      
      for (int node = 0; node < numa_nodes; node++) {
        // Set memory policy for this NUMA node
        {
          NUMAOptimizer::ScopedMemoryPolicy policy(node);
          // Allocate only one dataset per NUMA node
          rxDatasets_numa[node] = randomx_alloc_dataset(rxFlags);
          if (!rxDatasets_numa[node]) {
            throw std::runtime_error("Failed to allocate dataset for NUMA node " + std::to_string(node));
          }
        }
        
        // Optimize the allocated memory for mining
        if (rxDatasets_numa[node]->memory) {
          size_t dataset_size = randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE;
          NUMAOptimizer::optimizeMemoryForMining(rxDatasets_numa[node]->memory, dataset_size);
        }
        
        setcolor(BRIGHT_YELLOW);
        std::cout << " Allocated dataset on NUMA node " << node << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
      
      // Restore default memory policy for the rest of the program
      NUMAOptimizer::restoreMemoryPolicy();
    }
  } else {
    // Fallback to single dataset for non-NUMA systems
    rx_numa_enabled = false;
    numa_nodes = 1;
    rxDataset = randomx_alloc_dataset(rxFlags);
    
    if (!rxDataset) {
      throw std::runtime_error("Failed to allocate dataset");
    }
  }
}


// Modified to update all NUMA datasets
void randomx_update_data_numa(randomx_cache* rc, randomx_dataset **datasets, 
                             void *seed, size_t seedSize, int threadCount) {
  randomx_init_cache(rc, seed, seedSize);
  
  uint32_t datasetItemCount = randomx_dataset_item_count();
  
  if (!rx_numa_enabled) {
    randomx_init_dataset(datasets[0], rc, 0, datasetItemCount);
    return;
  }

  // Update each NUMA node's dataset
  for (int node = 0; node < numa_nodes; node++) {
    std::vector<std::thread> threads;
    
    // Calculate threads per node - distribute total threads across nodes
    int threadsPerNode = threadCount / numa_nodes;
    if (node < (threadCount % numa_nodes)) {
      threadsPerNode++; // Distribute remainder threads to first few nodes
    }
    
    if (threadsPerNode > 1) {
      auto perThread = datasetItemCount / threadsPerNode;
      auto remainder = datasetItemCount % threadsPerNode;
      uint32_t startItem = 0;

      for (int i = 0; i < threadsPerNode; ++i) {
        auto count = perThread + (i == threadsPerNode - 1 ? remainder : 0);
        
        threads.push_back(std::thread([=, &rc, &datasets]() {
          // Set memory policy to ensure initialization happens on correct node
          {
            NUMAOptimizer::ScopedMemoryPolicy policy(node);
            randomx_init_dataset(datasets[node], rc, startItem, count);
          }
        }));
        startItem += count;
      }
      
      for (auto& t : threads) {
        t.join();
      }
    } else {
      // Single-threaded initialization for this node
      NUMAOptimizer::ScopedMemoryPolicy policy(node);
      randomx_init_dataset(datasets[node], rc, 0, datasetItemCount);
    }
  }
}

// Wrapper that handles both NUMA and non-NUMA cases
void randomx_update_data(randomx_cache* rc, randomx_dataset* rd, void *seed, 
                        size_t seedSize, int threadCount) {
  if (rx_numa_enabled) {
    randomx_update_data_numa(rc, rxDatasets_numa, seed, seedSize, threadCount);
  } else {
    randomx_init_cache(rc, seed, seedSize);
    
    uint32_t datasetItemCount = randomx_dataset_item_count();
    std::vector<std::thread> threads;


    if (threadCount > 1) {
      auto perThread = datasetItemCount / threadCount;
      auto remainder = datasetItemCount % threadCount;
      uint32_t startItem = 0;

      for (int i = 0; i < threadCount; ++i) {
        auto count = perThread + (i == threadCount - 1 ? remainder : 0);
        threads.push_back(std::thread(&randomx_init_dataset, rd, rc, startItem, count));
        startItem += count;
      }
      for (unsigned i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
    } else {
      randomx_init_dataset(rd, rc, 0, datasetItemCount);
    }
    threads.clear();
  }
}

// Helper to get dataset for current thread
randomx_dataset* getDatasetForThread(int tid) {
  if (!rx_numa_enabled) {
    return rxDataset;
  }
  
  int numa_node = 0;
  
#ifdef __linux__
  int current_cpu = sched_getcpu();
  if (current_cpu >= 0) {
    numa_node = numa_node_of_cpu(current_cpu);
  }
#elif defined(_WIN32)
  PROCESSOR_NUMBER proc_num = {0};
  proc_num.Group = 0xFFFF;
  proc_num.Number = 0xFF;
  GetCurrentProcessorNumberEx(&proc_num);

  if (proc_num.Group != 0xFFFF && proc_num.Number != 0xFF) {
    USHORT node_number = 0;
    if (GetNumaProcessorNodeEx(&proc_num, &node_number)) {
      numa_node = node_number;
    } else {
      numa_node = (tid - 1) % numa_nodes;
    }
  } else {
    numa_node = (tid - 1) % numa_nodes;
  }
#endif
  
  // Validate and use fallback if needed
  if (numa_node < 0 || numa_node >= numa_nodes) {
    numa_node = (tid - 1) % numa_nodes;
  }
  
  return rxDatasets_numa[numa_node];
}

void randomx_set_flags(bool autoFlags) {
  if (autoFlags) {
    rxFlags = randomx_get_flags();
  }

  rxFlags |= RANDOMX_FLAG_FULL_MEM;
  if (rx_hugePages) rxFlags |= RANDOMX_FLAG_LARGE_PAGES;

  setcolor(BRIGHT_YELLOW);
  if (rxFlags & RANDOMX_FLAG_ARGON2_AVX2) {
    std::cout << " Argon2 implementation: AVX2" << std::endl;
  }
  else if (rxFlags & RANDOMX_FLAG_ARGON2_SSSE3) {
    std::cout << " Argon2 implementation: SSSE3" << std::endl;
  }
  else {
    std::cout << " Argon2 implementation: reference" << std::endl;
  }

  if (rxFlags & RANDOMX_FLAG_FULL_MEM) {
    std::cout << " full memory mode (2080 MiB)" << std::endl;
  }
  else {
    std::cout << " light memory mode (256 MiB)" << std::endl;
  }

  if (rxFlags & RANDOMX_FLAG_JIT) {
    std::cout << " JIT compiled mode ";
    if (rxFlags & RANDOMX_FLAG_SECURE) {
      std::cout << "(secure)";
    }
    std::cout << std::endl;
  }
  else {
    std::cout << " interpreted mode" << std::endl;
  }

  if (rxFlags & RANDOMX_FLAG_HARD_AES) {
    std::cout << " hardware AES mode" << std::endl;
  }
  else {
    std::cout << " software AES mode" << std::endl;
  }

  if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
    std::cout << " large pages mode" << std::endl;
  }
  else {
    std::cout << " small pages mode" << std::endl;
  }
  printf("\n");
  fflush(stdout);
  setcolor(BRIGHT_WHITE);
}

int rxRPCTest() {
  int toRet = 0;
  const char* expected = "2b21c0534efc99db149d3fdb6e2f2938958f85229157f7da1586da641c041004";
  const char* seedHash = "78a2c61c1cac2577a9a65287a27e933ab56929c1bf428a6e1768636bdfdc9989";

  const char* blob = "0103bdefedb6065feba13815bf0a1ce2de984dcc449feaac5dc953cbc66de0fab9c082d4321c4"
  "300000000000000000000000000000000000000000000000000000000c8e633f791e0084871c36dd329edf7ab5e42eb6"
  "cef27d9136b1732b9dc01e6a501";

  const char* daemonBlob = "0103bdefedb6065feba13815bf0a1ce2de984dcc449feaac5dc953cbc66de0fab9c082d"
  "4321c430608b900000000000000000000000000000000000000000000000000c8e633f791e0084871c36dd329edf7ab5"
  "e42eb6cef27d9136b1732b9dc01e6a501";
  
  const char* fullTemplate = "0103bdefedb6065feba13815bf0a1ce2de984dcc449feaac5dc953cbc66de0fab9c08"
  "2d4321c4300000000000000000000000000000000000000000000000000000000024601ff0a02908188929bd90302f56"
  "a855c3ddbbb45aa2de64ff2e61f3569f398daa904b541049e9c82095966a8045a455048a19baf9bfaf31802f5fd6ee52"
  "d8f869455ca565e0c34755f9abc2b2ad6e9168689e92808526bee5e045a45504867800101f14963fa44a05664d1b904a"
  "87fd9e8e984a74378cb40ba552a11f7fba2cfc633023c000000000000000000000000000000000000000000000000000"
  "000000000000000000000000000000000000000000000000000000000000000000000012c7be86ab07488ba43e8e03d8"
  "5a67625cfbf98c8544de4c877241b7aaafc7fe30000000000";

  const char *nonceHex = "0608b900";

  int th = std::thread::hardware_concurrency();

  byte seedBuffer[32];
  hexstrToBytes(seedHash, seedBuffer);

  printf("%s | seedHash\n%s | seedBuffer\n", seedHash, hexStr(seedBuffer, 32).c_str());

  randomx_set_flags(true);
  randomx_init_intern(th);
  
  randomx_vm *vm = randomx_create_vm(rxFlags, rxCache, rxDataset);
  randomx_update_data(rxCache, rxDataset, seedBuffer, 32, th);
  randomx_vm_set_cache(vm, rxCache);

  // hashing

  byte work[RANDOMX_TEMPLATE_SIZE];
  byte powHash[32];

  hexstrToBytes(blob, work);
  hexstrToBytes(nonceHex, &work[39]);

  printf("%s | work\n%s | daemonWork\n", 
    hexStr(work, std::string(blob).size()/2).c_str(), daemonBlob
  );
  if(memcmp(hexStr(work, std::string(blob).size()/2).c_str(), daemonBlob, 200) != 0) {
    toRet += 1;
  }

  randomx_calculate_hash(vm, work, std::string(blob).size()/2, powHash);
  randomx_calculate_hash(vm, work, std::string(blob).size()/2, powHash);

  printf("%s | powHash\n%s | expected\n", hexStr(powHash, 32).c_str(), expected);
  if(memcmp(hexStr(powHash, 32).c_str(), expected, 64) != 0) {
    toRet += 1;
  }
  return toRet;
}

// Global variables for synchronized batch scheduling
std::atomic<bool> globalInDevBatch(false);
std::atomic<int64_t> globalNextDevBatchTime(0);
std::atomic<int64_t> globalDevBatchDuration(0);
std::atomic<uint32_t> globalBatchSalt(0);  // Random salt for batch scheduling

void initGlobalBatchScheduler() {
  // Initialize once at program start
  globalBatchSalt = rand() & 0xFFFFFF;
  auto currentTime = std::chrono::steady_clock::now();
  globalNextDevBatchTime = std::chrono::duration_cast<std::chrono::milliseconds>(
    currentTime.time_since_epoch()).count() + (rand() % 60000);
}

void mineRx0(int tid) {
  boost::this_thread::sleep_for(boost::chrono::milliseconds(125));

  int64_t localJobCounter;
  int64_t localUserHeight = 0;
  int64_t localDevHeight = 0;
 
  std::string localCacheKey = "";
 
  byte powHash[32];
  byte devWork[RANDOMX_TEMPLATE_SIZE];
  byte work[RANDOMX_TEMPLATE_SIZE];

  randomx_vm *vm = nullptr;
  
  // Batched dev fee constants
  const int64_t BATCH_WINDOW_MS = 120000; // 2-minute window
  const int64_t MIN_BATCH_DURATION_MS = 5000;  // Minimum 5 seconds per batch
  const int64_t MAX_BATCH_DURATION_MS = 30000; // Maximum 30 seconds per batch

  // Initialize batch scheduling if this is the first thread
  if (tid == 0) {
    initGlobalBatchScheduler();
  }

waitForJob:
  while (!isConnected) {
    CHECK_CLOSE;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  bool devMine = false;

  while (!ABORT_MINER) {
    fflush(stdout);
    
    try {
      boost::json::value myJob;
      boost::json::value myJobDev;
      {
        std::scoped_lock<boost::mutex> lockGuard(mutex);
        myJob = job;
        myJobDev = devJob;
        localJobCounter = jobCounter;
      }
      
      if (devMine != globalInDevBatch.load()) {
        printf("devMine in thread %d changing to %d\n", tid, globalInDevBatch.load());
        fflush(stdout);
      } else {
        printf("devMine of %d is correct in thread %d\n", devMine, tid);
        fflush(stdout);
      }
      devMine = globalInDevBatch.load();

      if (devMine) {
        while(!randomx_ready_dev.load()) {
          boost::this_thread::yield();
        }
      } else {
        while(!randomx_ready.load()) {
          boost::this_thread::yield();
        }
      }

      std::string &tKey = devMine ? randomx_cacheKey_dev : randomx_cacheKey;

      if (tKey != localCacheKey ) {
        if (vm) {
          randomx_destroy_vm(vm);
          vm = nullptr;
        }

        randomx_cache* tCache = devMine ? rxCache_dev : rxCache;

        randomx_dataset* RXD = getDatasetForThread(tid);
        vm = randomx_create_vm(rxFlags, rxCache_dev, RXD);

        if (vm == nullptr) {
          if ((rxFlags & RANDOMX_FLAG_HARD_AES)) {
            throw std::runtime_error("Cannot create user VM with the selected options. Try using --softAes");
          }
          if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
            throw std::runtime_error("Cannot create user VM with the selected options. Try without --largePages");
          }
          throw std::runtime_error("Cannot create user VM");
        }
        localCacheKey = tKey;
      }
      
      if (ourHeight == 0 || localUserHeight != ourHeight) {
        byte *b2 = new byte[RANDOMX_TEMPLATE_SIZE];
        memset(b2, 0, RANDOMX_TEMPLATE_SIZE);
        hexstrToBytes(std::string(myJob.at("blob").as_string()), b2);
        memcpy(work, b2, std::string(myJob.at("blob").as_string().c_str()).size() / 2);
        delete[] b2;
        localUserHeight = ourHeight;
      }
      
      if (devConnected && (devHeight == 0 || localDevHeight != devHeight)) {
        byte *b2d = new byte[RANDOMX_TEMPLATE_SIZE];
        memset(b2d, 0, RANDOMX_TEMPLATE_SIZE);
        hexstrToBytes(std::string(myJobDev.at("blob").as_string()), b2d);
        memcpy(devWork, b2d, std::string(myJobDev.at("blob").as_string().c_str()).size() / 2);
        delete[] b2d;
        localDevHeight = devHeight;
      }

      bool submit = false;

      int userLen = std::string(myJob.at("blob").as_string().c_str()).size() / 2;
      int devLen = std::string(myJobDev.at("blob").as_string().c_str()).size() / 2;

      while (localJobCounter == jobCounter) {
        CHECK_CLOSE;
        
        auto now = std::chrono::steady_clock::now();
        int64_t currentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        uint32_t scheduleSeed = localJobCounter + globalBatchSalt;

        if (vm == nullptr) continue;

        if (!globalInDevBatch.load() && currentTimeMs >= globalNextDevBatchTime.load() && 
            devConnected) {

          if (!globalInDevBatch.exchange(true)) {
            std::mt19937 rng(scheduleSeed);
            std::uniform_int_distribution<int64_t> durationDist(
                MIN_BATCH_DURATION_MS, MAX_BATCH_DURATION_MS);
              
            int64_t rawDuration = durationDist(rng);
            globalDevBatchDuration.store(rawDuration * (devFee / 100.0));

            randomx_vm_set_cache(vm, rxCache_dev);

            needsDatasetUpdate = true;
            cv.notify_all();

            printf("switched to dev mode\n");
            fflush(stdout);
          }
        } 
        else if (globalInDevBatch.load() && 
                 currentTimeMs >= globalNextDevBatchTime.load() + globalDevBatchDuration.load()) {
          
          if (globalInDevBatch.exchange(false)) {
            std::mt19937 rng(scheduleSeed + 1);
            std::uniform_int_distribution<int64_t> timingDist(30000, BATCH_WINDOW_MS - 30000);
            
            int64_t nextInterval = timingDist(rng);
            globalNextDevBatchTime.store(currentTimeMs + nextInterval);

            randomx_vm_set_cache(vm, rxCache);

            needsDatasetUpdate = true;
            cv.notify_all();

            printf("switched to user mode\n");
            fflush(stdout);
          }
        }

        if (globalInDevBatch.load() != devMine) break;
        
        if (vm == nullptr) continue; // Skip if VM not ready

        uint64_t *nonce = devMine ? &nonce0_dev : &nonce0;
        (*nonce)++;

        byte *WORK = devMine ? &devWork[0] : &work[0];
        boost::json::value &J = devMine ? myJobDev : myJob;

        uint32_t N = (*nonce) << 11 | tid;
        memcpy(&WORK[39], &N, 4);
        
        if (localJobCounter != jobCounter) {
          break;
        }
        
        if ((!devMine && localUserHeight != ourHeight) || 
            (devMine && localDevHeight != devHeight)) {
          break;
        }

        if(!randomx_ready.load() && !randomx_ready_dev.load()) {
          continue;
        }

        randomx_calculate_hash(vm, WORK, devMine ? devLen : userLen, powHash);
        counter.fetch_add(1);
        submit = devMine ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || 
            (!devMine && localUserHeight != ourHeight) ||
            (devMine && localDevHeight != devHeight)) {
          break;
        }

        Num cmpTarget = rx0_calcTarget(devMine ? myJobDev : myJob);
        
        std::reverse(powHash, powHash + 32);
        if (Num(hexStr(powHash, 32).c_str(), 16) < cmpTarget) {
          std::reverse(powHash, powHash + 32);
          
          if (!submit) {
            for(;;) {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              int64_t &rH = devMine ? devHeight : ourHeight;
              int64_t &lH = devMine ? localDevHeight : localUserHeight;
              if (submit || localJobCounter != jobCounter || rH != lH)
                break;
              boost::this_thread::yield();
            }
          }
          
          int64_t &rH = devMine ? devHeight : ourHeight;
          int64_t &lH = devMine ? localDevHeight : localUserHeight;
          if (localJobCounter != jobCounter || rH != lH) {
            break;
          }

          if (devMine) {
            submittingDev = true;

            if (localJobCounter != jobCounter || localDevHeight != devHeight) {
              submittingDev = false;
              break;
            }

            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);

            N = __builtin_bswap32(N);
            devShare = {
              {"method", rx0Stratum::submit.method.c_str()},
              {"id", rx0Stratum::submit.id},
              {"params", {
                {"id", randomx_login_dev.c_str()},
                {"job_id", myJobDev.at("job_id").as_string().c_str()},
                {"nonce", uint32ToHex(N).c_str()},
                {"result", hexStr(powHash, 32).c_str()}
              }}
            };
            data_ready = true;
          }
          else {
            submitting = true;

            if (localJobCounter != jobCounter || localUserHeight  != ourHeight) {
              submitting = false;
              break;
            }

            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
                      
            switch (miningProfile.protocol) {
              case PROTO_RX0_SOLO:
              {
                int fbSize = myJob.at("template").as_string().size() / 2;
                byte *fullBlob = new byte[fbSize];
                hexstrToBytes(std::string(myJob.at("template").as_string()), fullBlob);
                memcpy(&fullBlob[39], &N, 4);

                share = {
                  {"jsonrpc", "2.0"},
                  {"method", "submit_block"},
                  {"id", 7},
                  {"params", {hexStr(fullBlob, fbSize).c_str()}}
                };
                delete[] fullBlob;
                break;
              }
              case PROTO_RX0_STRATUM:
              {
                N = __builtin_bswap32(N);
                share = {
                  {"method", rx0Stratum::submit.method.c_str()},
                  {"id", rx0Stratum::submit.id},
                  {"params", {
                    {"id", randomx_login.c_str()},
                    {"job_id", myJob.at("job_id").as_string().c_str()},
                    {"nonce", uint32ToHex(N).c_str()},
                    {"result", hexStr(powHash, 32).c_str()}
                  }}
                };
                break;                
              }
            }
            data_ready = true;
          }
          cv.notify_all();
        }

        if (!isConnected)
          break;
      }
      if (!isConnected)
        break;
      
      if (localJobCounter != jobCounter) {
        if (tid == 0) {
          globalBatchSalt = (globalBatchSalt + jobCounter) ^ (rand() & 0xFFFFFF);
        }
      }
    }
    catch (std::exception& e) {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);

      localJobCounter = -1;
      localUserHeight = -1;
      localDevHeight = -1;
      
      // Cleanup VMs on error
      if (vm) {
        randomx_destroy_vm(vm);
        vm = nullptr;
      }
      localCacheKey = "";
      
      // Reset batch state
      globalInDevBatch.store(false);
    }
    if (!isConnected)
      break;
  }
 
  // Cleanup before waiting for job
  if (vm) {
    randomx_destroy_vm(vm);
    vm = nullptr;
  }
  localCacheKey = "";
 
  goto waitForJob;
}