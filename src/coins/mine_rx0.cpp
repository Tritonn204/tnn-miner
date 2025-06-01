#include "miners.hpp"

#include <net/rx0/rx0_jobCache.hpp>

#include <randomx/randomx.h>
#include <randomx/dataset.hpp>
#include <randomx/common.hpp>
#include <randomx/jit_compiler.hpp>

#include <stratum/stratum.h>
#include <thread>

randomx_dataset* rxDataset;
randomx_cache* rxCache;

randomx_dataset* rxDataset_dev;
randomx_cache* rxCache_dev;

randomx_flags rxFlags;

bool rx_hugePages;

std::string randomx_cacheKey = "0000000000000000000000000000000000000000000000000000000000000000";
std::string randomx_cacheKey_dev = "0000000000000000000000000000000000000000000000000000000000000000";

std::string randomx_login;
std::string randomx_login_dev;

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

  rxCache = randomx_alloc_cache(rxFlags);
  rxCache_dev = randomx_alloc_cache(rxFlags);

  rxDataset = randomx_alloc_dataset(rxFlags);
  rxDataset_dev = randomx_alloc_dataset(rxFlags);

  if (rxCache == nullptr || rxCache_dev == nullptr) {
    throw std::runtime_error("RandomX Cache Alloc Failed");
  }
}

void randomx_update_data(randomx_cache* rc, randomx_dataset* rd, void *seed, size_t seedSize, int threadCount) {
  randomx_init_cache(rc, seed, seedSize);
  
  uint32_t datasetItemCount = randomx_dataset_item_count();
  std::vector<std::thread> threads;

  if (threadCount > 1) {
    auto perThread = datasetItemCount / threadCount;
    auto remainder = datasetItemCount % threadCount;
    uint32_t startItem = 0;

    for (int i = 0; i < threadCount; ++i)
    {
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

void randomx_set_flags(bool autoFlags) {
  if (autoFlags) {
    rxFlags = randomx_get_flags();
  }

  rxFlags |= RANDOMX_FLAG_FULL_MEM;
  if (rx_hugePages) rxFlags |= RANDOMX_FLAG_LARGE_PAGES; // TODO: Make this a toggle from CLI
  fflush(stdout);

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
 
  std::string localUserCacheKey = "";
  std::string localDevCacheKey = "";
 
  byte powHash[32];
  byte devWork[RANDOMX_TEMPLATE_SIZE];
  byte work[RANDOMX_TEMPLATE_SIZE];

  randomx_vm *vm = nullptr;
  randomx_vm *vmDev = nullptr;
  
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

  while (!ABORT_MINER) {
    while(!randomx_ready && !randomx_ready_dev) {
      boost::this_thread::yield();
    }
    
    try {
      boost::json::value myJob;
      boost::json::value myJobDev;
      {
        std::scoped_lock<boost::mutex> lockGuard(mutex);
        myJob = job;
        myJobDev = devJob;
        localJobCounter = jobCounter;
      }
      
      // VM recreation logic (unchanged)
      if (randomx_ready && randomx_cacheKey != localUserCacheKey) {
        if (vm) {
          randomx_destroy_vm(vm);
          vm = nullptr;
        }
        vm = randomx_create_vm(rxFlags, rxCache, rxDataset);
        if (vm == nullptr) {
          if ((rxFlags & RANDOMX_FLAG_HARD_AES)) {
            throw std::runtime_error("Cannot create user VM with the selected options. Try using --softAes");
          }
          if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
            throw std::runtime_error("Cannot create user VM with the selected options. Try without --largePages");
          }
          throw std::runtime_error("Cannot create user VM");
        }
        localUserCacheKey = randomx_cacheKey;
      }
      
      if (randomx_ready_dev && randomx_cacheKey_dev != localDevCacheKey) {
        if (vmDev) {
          randomx_destroy_vm(vmDev);
          vmDev = nullptr;
        }
        vmDev = randomx_create_vm(rxFlags, rxCache_dev, rxDataset_dev);
        if (vmDev == nullptr) {
          if ((rxFlags & RANDOMX_FLAG_HARD_AES)) {
            throw std::runtime_error("DEV: Cannot create VM with the selected options. Try using --softAes");
          }
          if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
            throw std::runtime_error("DEV: Cannot create VM with the selected options. Try without --largePages");
          }
          throw std::runtime_error("DEV: Cannot create VM");
        }
        localDevCacheKey = randomx_cacheKey_dev;
      }
      
      // Update work blobs (unchanged)
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
      bool devMine = false;

      int userLen = std::string(myJob.at("blob").as_string().c_str()).size() / 2;
      int devLen = std::string(myJobDev.at("blob").as_string().c_str()).size() / 2;

      while (localJobCounter == jobCounter) {
        CHECK_CLOSE;
        
        // Synchronized batch scheduling based on jobCounter + salt
        auto now = std::chrono::steady_clock::now();
        int64_t currentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        // Deterministic batch scheduling - all threads see the same decision
        uint32_t scheduleSeed = localJobCounter + globalBatchSalt;
        
        if (!globalInDevBatch.load() && currentTimeMs >= globalNextDevBatchTime.load() && 
            devConnected && randomx_ready_dev && vmDev != nullptr) {
          
          // First thread that detects this condition sets up the batch
          if (!globalInDevBatch.exchange(true)) {
            // Use jobCounter + salt to seed the random batch duration
            std::mt19937 rng(scheduleSeed);
            std::uniform_int_distribution<int64_t> durationDist(
                MIN_BATCH_DURATION_MS, MAX_BATCH_DURATION_MS);
              
            int64_t rawDuration = durationDist(rng);
            int64_t scaledDuration = static_cast<int64_t>(rawDuration * devFee);
            globalDevBatchDuration.store(scaledDuration);
            
            if (tid == 0) { // Only log from thread 0
              setcolor(CYAN);
              std::cout << "\nAll threads entering dev batch for " 
                        << (scaledDuration/1000.0) << " seconds" << std::endl;
              setcolor(BRIGHT_WHITE);
            }
          }
        } 
        else if (globalInDevBatch.load() && 
                 currentTimeMs >= globalNextDevBatchTime.load() + globalDevBatchDuration.load()) {
          
          // First thread that detects this condition ends the batch
          if (globalInDevBatch.exchange(false)) {
            // Use jobCounter + salt + 1 to seed the next batch timing
            std::mt19937 rng(scheduleSeed + 1);
            std::uniform_int_distribution<int64_t> timingDist(30000, BATCH_WINDOW_MS - 30000);
            
            int64_t nextInterval = timingDist(rng);
            globalNextDevBatchTime.store(currentTimeMs + nextInterval);
            
            if (tid == 0) { // Only log from thread 0
              setcolor(CYAN);
              std::cout << "\nAll threads exiting dev batch, next batch in ~" 
                        << (nextInterval/1000.0) << " seconds" << std::endl;
              setcolor(BRIGHT_WHITE);
            }
          }
        }

        // All threads use the same global batch state
        devMine = globalInDevBatch.load() || (randomx_ready_dev && !randomx_ready && vmDev != nullptr);
        
        // Use appropriate VM
        randomx_vm *chosenVm = devMine ? vmDev : vm;
        if (!chosenVm) continue; // Skip if VM not ready

        uint64_t *nonce = devMine ? &nonce0_dev : &nonce0;
        (*nonce)++;

        byte *WORK = devMine ? &devWork[0] : &work[0];
        boost::json::value &J = devMine ? myJobDev : myJob;

        uint32_t N = (*nonce) << 11 | tid;
        memcpy(&WORK[39], &N, 4);
        
        // Check for job updates before hashing
        if (localJobCounter != jobCounter) {
          break;
        }
        
        // Check for height updates before hashing
        if ((!devMine && localUserHeight != ourHeight) || 
            (devMine && localDevHeight != devHeight)) {
          break;
        }

        randomx_calculate_hash(chosenVm, WORK, devMine ? devLen : userLen, powHash);
        counter.fetch_add(1);
        submit = devMine ? !submittingDev : !submitting;

        // Check for updates after hashing
        if (localJobCounter != jobCounter || 
            (!devMine && localUserHeight != ourHeight) ||
            (devMine && localDevHeight != devHeight)) {
          break;
        }

        Num cmpTarget = rx0_calcTarget(devMine ? myJobDev : myJob);
        
        std::reverse(powHash, powHash + 32);
        if (Num(hexStr(powHash, 32).c_str(), 16) < cmpTarget) {
          std::reverse(powHash, powHash + 32);
          
          // Rest of the submission logic (unchanged)
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
          
          // Final check before submission
          int64_t &rH = devMine ? devHeight : ourHeight;
          int64_t &lH = devMine ? localDevHeight : localUserHeight;
          if (localJobCounter != jobCounter || rH != lH) {
            break;
          }

          if (devMine) {
            submittingDev = true;
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
            // User share submission (unchanged)
            submitting = true;
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
      
      // Reset batch state when job changes
      if (localJobCounter != jobCounter) {
        globalInDevBatch.store(false);
        globalNextDevBatchTime.store(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() + (rand() % 30000));
        
        // Update the salt on job changes to prevent predictability
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
      if (vmDev) {
        randomx_destroy_vm(vmDev);
        vmDev = nullptr;
      }
      localUserCacheKey = "";
      localDevCacheKey = "";
      
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
  if (vmDev) {
    randomx_destroy_vm(vmDev);
    vmDev = nullptr;
  }
  localUserCacheKey = "";
  localDevCacheKey = "";
 
  goto waitForJob;
}