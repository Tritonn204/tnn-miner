#include "miners.hpp"

#include <net/rx0/rx0_jobCache.hpp>

#include <randomx/randomx.h>
#include <randomx/dataset.hpp>
#include <randomx/common.hpp>
#include <randomx/jit_compiler.hpp>

randomx_dataset* rxDataset;
randomx_cache* rxCache;

randomx_dataset* rxDataset_dev;
randomx_cache* rxCache_dev;

randomx_flags rxFlags;

bool rx_hugePages;

std::string randomx_cacheKey = "0000000000000000000000000000000000000000000000000000000000000000";
std::string randomx_cacheKey_dev = "0000000000000000000000000000000000000000000000000000000000000000";

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
  randomx_init_cache(rc, &seed, seedSize);
  
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

void mineRandomX(int tid){
  boost::this_thread::sleep_for(boost::chrono::milliseconds(125));

  int64_t localJobCounter;
  byte powHash[32];
  byte devWork[RANDOMX_BLOB_SIZE];
  byte work[RANDOMX_BLOB_SIZE];

  randomx_vm *vm = randomx_create_vm(rxFlags, rxCache, rxDataset);
  randomx_vm *vmDev = randomx_create_vm(rxFlags, rxCache, rxDataset);

  if (vm == nullptr) {
    if ((rxFlags & RANDOMX_FLAG_HARD_AES)) {
      throw std::runtime_error("Cannot create VM with the selected options. Try using --softAes");
    }
    if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
      throw std::runtime_error("Cannot create VM with the selected options. Try without --largePages");
    }
    throw std::runtime_error("Cannot create VM");
  }

  if (vmDev == nullptr) {
    if ((rxFlags & RANDOMX_FLAG_HARD_AES)) {
      throw std::runtime_error("DEV: Cannot create VM with the selected options. Try using --softAes");
    }
    if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
      throw std::runtime_error("DEV: Cannot create VM with the selected options. Try without --largePages");
    }
    throw std::runtime_error("DEV: Cannot create VM");
  }

waitForJob:

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }


  while (true)
  {
    while(!randomx_ready && !randomx_ready_dev) {
      boost::this_thread::yield();
    }
    try
    {
      boost::json::value myJob;
      boost::json::value myJobDev;
      {
        std::scoped_lock<boost::mutex> lockGuard(mutex);
        myJob = job;
        myJobDev = devJob;
        localJobCounter = jobCounter;
      }

      byte *b2 = new byte[RANDOMX_BLOB_SIZE];
      hexstrToBytes(std::string(myJob.at("blob").as_string()), b2);
      memcpy(work, b2, RANDOMX_BLOB_SIZE);
      delete[] b2;

      if (devConnected)
      {
        byte *b2d = new byte[RANDOMX_BLOB_SIZE];
        hexstrToBytes(std::string(myJobDev.at("blob").as_string()), b2d);
        memcpy(devWork, b2d, RANDOMX_BLOB_SIZE);
        delete[] b2d;
      }

      double which;
      bool devMine = false;
      bool submit = false;
      Num cmpDiff;
      // DIFF = 5000;

      std::string hex;

      uint32_t userNonce = 0;
      uint32_t devNonce = 0;

      while (localJobCounter == jobCounter)
      {
        which = (double)(rand() % 10000);
        devMine = (devConnected && which < devFee * 100.0 && randomx_ready_dev) || (randomx_ready_dev && !randomx_ready);

        uint32_t &nonce = devMine ? devNonce : userNonce;

        // printf("Difficulty: %" PRIx64 "\n", DIFF);

        uint32_t cmpTargetInt = devMine ? 
          boost_swap_impl::stoul(myJobDev.at("target").as_string().c_str(), nullptr, 16) :
          boost_swap_impl::stoul(myJob.at("target").as_string().c_str(), nullptr, 16);

        cmpTargetInt = __builtin_bswap32(cmpTargetInt);
        Num cmpTarget = Num(cmpTargetInt) << 192;

        nonce ++;

        byte *WORK = devMine ? &devWork[0] : &work[0];

        uint32_t N = nonce << 11 | tid;
        memcpy(&WORK[39], &N, 4);
        if (littleEndian())
        {
          std::swap(WORK[39], WORK[42]);
          std::swap(WORK[40], WORK[41]);
        }

        if (!randomx_ready_dev && !randomx_ready) break;
        randomx_calculate_hash(devMine ? vmDev : vm, WORK, RANDOMX_BLOB_SIZE, powHash);
        counter.fetch_add(1);
        submit = devMine ? !submittingDev : !submitting;

        // std::cout << Num(hexStr(powHash, 32).c_str(), 16) << "\n" << cmpTarget << std::endl << std::flush;

        if (Num(hexStr(powHash, 32).c_str(), 16) < cmpTarget)
        {
          if (!submit) {
            for(;;) {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              if (submit || localJobCounter != jobCounter)
                break;
              boost::this_thread::yield();
            }
          }
          if (localJobCounter != jobCounter)
                break;
          // printf("work: %s, hash: %s\n", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str(), hexStr(powHash, 32).c_str());
          // boost::lock_guard<boost::mutex> lock(mutex);
          if (devMine)
          {
            // submittingDev = true;
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            // devShare = {
            //     {"jobid", myJobDev.at("jobid").as_string().c_str()},
            //     {"mbl_blob", hexStr(&WORK[MINIBLOCK_SIZE*i], MINIBLOCK_SIZE).c_str()}};
            // data_ready = true;
          }
          else
          {
            // submitting = true;
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            // share = {
            //     {"jobid", myJob.at("jobid").as_string().c_str()},
            //     {"mbl_blob", hexStr(&WORK[MINIBLOCK_SIZE*i], MINIBLOCK_SIZE).c_str()}};
            // data_ready = true;
          }
          cv.notify_all();
        }

        if (!isConnected)
          break;
      }
      if (!isConnected)
        break;
    }
    catch (std::exception& e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);

      localJobCounter = -1;
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}