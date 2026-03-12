#include "miners.hpp"
#include "numa_optimizer.hpp"

#include <rinhash/rinhash.h>
#include <stratum/btc-stratum.h>

#include <endian.hpp>
#include <openssl/sha.h>

void mineRinhash(int tid)
{
  int numa_nodes = NUMAOptimizer::getMemoryNodes();
  if (numa_nodes > 0) {
      int node = tid % numa_nodes;
      NUMAOptimizer::setMemoryPolicy(node);
      NUMAOptimizer::printThreadBinding(tid);
  }

  thread_local std::random_device rd;
  thread_local std::mt19937 rng(rd());
  thread_local std::uniform_real_distribution<double> dist(0, 10000);
  
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint32_t nonce = 0;
  uint32_t nonce_dev = 0;

  thread_local byte powHash[32];
  alignas(64) thread_local byte work[80] = {0};
  alignas(64) thread_local byte devWork[80] = {0};
  alignas(64) thread_local byte FINALWORK[80] = {0};

  thread_local uint32_t targetWords[8];
  thread_local uint32_t targetWords_dev[8];

  thread_local blake3_hasher blake3_prefix_main;
  thread_local blake3_hasher blake3_prefix_dev;
  thread_local bool blake3_main_ready = false;
  thread_local bool blake3_dev_ready = false;

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

      if (ourHeight <= 0 || localOurHeight != ourHeight)
      {
        hexstrToBytes(std::string(myJob.at("template").as_string()), work);
        localOurHeight = ourHeight;
        nonce = 0;

        blake3_hasher_init(&blake3_prefix_main);
        blake3_hasher_update(&blake3_prefix_main, work, 64);
        blake3_main_ready = true;
      }

      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight <= 0 || localDevHeight != devHeight)
        {
          hexstrToBytes(std::string(myJobDev.at("template").as_string()), devWork);
          localDevHeight = devHeight;
          nonce_dev = 0;

          blake3_hasher_init(&blake3_prefix_dev);
          blake3_hasher_update(&blake3_prefix_dev, devWork, 64);
          blake3_dev_ready = true;
        }
      }

      bool devMine = false;
      double which;
      bool submit = false;

      BTCStratum::diffToWords(doubleDiff, targetWords);
      BTCStratum::diffToWords(doubleDiffDev, targetWords_dev);

      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;
        which = dist(rng);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);

        uint32_t *noncePtr = devMine ? &nonce_dev : &nonce;
        (*noncePtr)++;

        byte *WORK = (devMine && devConnected) ? devWork : work;
        memcpy(FINALWORK, WORK, 80);

        uint32_t n = ((tid - 1) % (512)) | ((*noncePtr) << 9);
        be32enc(FINALWORK + 76, n);

        if (localJobCounter != jobCounter) {
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        RinHash::hash(powHash, FINALWORK, devMine ? &blake3_prefix_dev : &blake3_prefix_main);

        uint32_t *currentTarget = devMine ? targetWords_dev : targetWords;
        if (++localCount >= 512) { counter.fetch_add(localCount); localCount = 0; }

        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight) {
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        if (RinHash::checkHash(powHash, currentTarget))
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

          uint32_t baseNTime = std::stoul(std::string((devMine ? myJobDev : myJob).at("nTime").as_string()), nullptr, 16);

          if (devMine)
          {
            submittingDev = true;
            if (localJobCounter != jobCounter || localDevHeight != devHeight) {
              if (localCount) { counter.fetch_add(localCount); localCount = 0; }
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
              if (localCount) { counter.fetch_add(localCount); localCount = 0; }
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
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }
      }
      if (localCount) { counter.fetch_add(localCount); localCount = 0; }
    }
    catch (std::exception &e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function: " << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);

      if (localCount) { counter.fetch_add(localCount); localCount = 0; }

      localJobCounter = -1;
      localOurHeight = -1;
      localDevHeight = -1;
    }
  }

  // ⭐ NUMA: restore thread policy when mining thread ends
  NUMAOptimizer::restoreMemoryPolicy();

  goto waitForJob;
}
