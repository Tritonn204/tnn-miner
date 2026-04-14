// mine_yespower.cpp
#include "miners.hpp"
#include "tnn-hugepages.hpp"
#include "numa_optimizer.hpp"

#include <yespower/yespower.h>
#include <yespower/yespower_algo.h>
#include <stratum/btc-stratum.h>

#include <endian.hpp>
#include <openssl/sha.h>

void mineYespower(int tid)
{
  thread_local std::random_device rd;
  thread_local std::mt19937 rng(rd());
  thread_local std::uniform_real_distribution<double> dist(0, 10000);
  
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint32_t nonce = 0;
  uint32_t nonce_dev = 0;

  byte powHash[32];
  alignas(64) byte work[80] = {0};
  alignas(64) byte devWork[80] = {0};
  alignas(64) byte FINALWORK[80] = {0};

  uint32_t targetWords[8];
  uint32_t targetWords_dev[8];

  yespower_local_t yespower_local;
  yespower_local_t yespower_dev_local;
  yespower_binary_t result;
  
  if (yespower_init_local(&yespower_local) != 0) {
    setcolor(RED);
    std::cerr << "Failed to initialize yespower context for thread " << tid << std::endl;
    setcolor(BRIGHT_WHITE);
    return;
  }
  
  if (yespower_init_local(&yespower_dev_local) != 0) {
    setcolor(RED);
    std::cerr << "Failed to initialize yespower dev context for thread " << tid << std::endl;
    setcolor(BRIGHT_WHITE);
    yespower_free_local(&yespower_local);
    return;
  }

  if (yespower_local.aligned) {
    NUMAOptimizer::optimizeMemoryForMining(yespower_local.aligned, yespower_local.aligned_size);
  }
  if (yespower_dev_local.aligned) {
    NUMAOptimizer::optimizeMemoryForMining(yespower_dev_local.aligned, yespower_dev_local.aligned_size);
  }

  thread_local uint64_t localCount = 0;

waitForJob:
  while (!isConnected)
  {
    CHECK_CLOSE;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  while (!ABORT_MINER)
  {
    try
    {
      boost::json::value myJob;
      boost::json::value myJobDev;
      {
        std::scoped_lock<std::mutex> lockGuard(mutex);
        myJob = job;
        myJobDev = devJob;
        localJobCounter = jobCounter;
      }

      if (!myJob.at("template").is_string())
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        hexstrToBytes(std::string(myJob.at("template").as_string()), work);
        localOurHeight = ourHeight;
        nonce = 0;

        uint32_t *work_words = (uint32_t *)work;
        for (int i = 0; i < 19; i++) {
          work_words[i] = __builtin_bswap32(work_words[i]);
        }
      }

      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          hexstrToBytes(std::string(myJobDev.at("template").as_string()), devWork);
          localDevHeight = devHeight;
          nonce_dev = 0;

          uint32_t *work_words = (uint32_t *)devWork;
          for (int i = 0; i < 19; i++) {
            work_words[i] = __builtin_bswap32(work_words[i]);
          }
        }
      }

      bool devMine = false;
      double which;
      bool submit = false;

      BTCStratum::diffToWords(doubleDiff / 65536.0, targetWords);
      BTCStratum::diffToWords(doubleDiffDev / 65536.0, targetWords_dev);

      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;
        which = dist(rng);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);

        uint32_t *noncePtr = devMine ? &nonce_dev : &nonce;
        (*noncePtr)++;

        byte *WORK = (devMine && devConnected) ? devWork : work;
        memcpy(FINALWORK, WORK, 80);

        uint32_t n = ((tid - 1) % (256 * 256)) | ((*noncePtr) << 16);
        be32enc(FINALWORK + 76, n);

        if (localJobCounter != jobCounter) {
          if (localCount) { cpu_counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        yespower_local_t *local = devMine ? &yespower_dev_local : &yespower_local;
        const yespower_params_t *params = devMine ? &devYespowerParams : &currentYespowerParams;
        
        if (yespower(local, FINALWORK, 80, params, &result) != 0) {
          setcolor(RED);
          std::cerr << "yespower computation failed for thread " << tid << std::endl;
          setcolor(BRIGHT_WHITE);
          if (localCount) { cpu_counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        uint32_t *currentTarget = devMine ? targetWords_dev : targetWords;
        if (++localCount >= 1024) { cpu_counter.fetch_add(localCount); localCount = 0; }

        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight) {
          if (localCount) { cpu_counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        if (checkYespowerHash(result.uc, currentTarget))
        {
          if (!submit)
          {
            for (;;)
            {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
                break;
              std::this_thread::yield();
            }
          }

          memcpy(powHash, result.uc, 32);
          uint32_t baseNTime = std::stoul(std::string((devMine ? myJobDev : myJob).at("nTime").as_string()), nullptr, 16);

          if (devMine)
          {
            submittingDev = true;
            if (localJobCounter != jobCounter || localDevHeight != devHeight) {
              if (localCount) { cpu_counter.fetch_add(localCount); localCount = 0; }
              break;
            }
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);

            BTCStratum::formatShare(devShare, myJobDev, devWorkerName, n, baseNTime,
                                  (uint32_t)myJobDev.at("extraNonce2").get_uint64());
            data_ready = true;
          }
          else
          {
            submitting = true;
            if (localJobCounter != jobCounter || localOurHeight != ourHeight) {
              if (localCount) { cpu_counter.fetch_add(localCount); localCount = 0; }
              break;
            }
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);

            BTCStratum::formatShare(share, myJob, workerName, n, baseNTime,
                                  (uint32_t)myJob.at("extraNonce2").get_uint64());
            data_ready = true;
          }
          cv.notify_all();
        }

        if (!isConnected)
        {
          data_ready = true;
          cv.notify_all();
          if (localCount) { cpu_counter.fetch_add(localCount); localCount = 0; }
          break;
        }
      }
      if (localCount) { cpu_counter.fetch_add(localCount); localCount = 0; }
    }
    catch (std::exception &e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function: " << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);

      if (localCount) { cpu_counter.fetch_add(localCount); localCount = 0; }

      localJobCounter = -1;
      localOurHeight = -1;
      localDevHeight = -1;
    }
  }

  yespower_free_local(&yespower_local);
  yespower_free_local(&yespower_dev_local);
  
  goto waitForJob;
}
