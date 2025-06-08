// mine_yespower.cpp
#include "miners.hpp"

#include <rinhash/rinhash.h>
#include <stratum/btc-stratum.h>

#include <endian.hpp>
#include <openssl/sha.h>

void mineRinhash(int tid)
{
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

waitForJob:
  while (!isConnected)
  {
    CHECK_CLOSE;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  while (!ABORT_MINER)
  {
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

      if (!myJob.at("template").is_string())
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;

      // Handle new job data
      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        // Use original hexstrToBytes format
        hexstrToBytes(std::string(myJob.at("template").as_string()), work);
        localOurHeight = ourHeight;
        nonce = 0;

        // Optimize endian conversion
        uint32_t *work_words = (uint32_t *)work;
        for (int i = 0; i < 19; i++) {
          work_words[i] = __builtin_bswap32(work_words[i]);
        }

        blake3_hasher_init(&blake3_prefix_main);
        blake3_hasher_update(&blake3_prefix_main, work, 64);
        blake3_main_ready = true;
      }

      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          // Use original hexstrToBytes format
          hexstrToBytes(std::string(myJobDev.at("template").as_string()), devWork);
          localDevHeight = devHeight;
          nonce_dev = 0;

          uint32_t *work_words = (uint32_t *)devWork;
          for (int i = 0; i < 19; i++) {
            work_words[i] = __builtin_bswap32(work_words[i]);
          }

          blake3_hasher_init(&blake3_prefix_dev);
          blake3_hasher_update(&blake3_prefix_dev, devWork, 64);
          blake3_dev_ready = true;
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

        // Put nonce in correct position (76-79)
        uint32_t n = ((tid - 1) % (256 * 256)) | ((*noncePtr) << 16);
        be32enc(FINALWORK + 76, n);

        if (localJobCounter != jobCounter)
          break;

        RinHash::hash(powHash, FINALWORK, devMine ? &blake3_prefix_dev : &blake3_prefix_main);

        uint32_t *currentTarget = devMine ? targetWords_dev : targetWords;
        counter.fetch_add(1);

        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight)
          break;

        if (RinHash::checkHash(powHash, currentTarget))
        {
          // if (!submit)
          // {
          //   for (;;)
          //   {
          //     submit = (devMine && devConnected) ? !submittingDev : !submitting;
          //     if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
          //       break;
          //     boost::this_thread::yield();
          //   }
          // }

          // memcpy(powHash, powHash, 32);
          // uint32_t baseNTime = std::stoul(std::string((devMine ? myJobDev : myJob).at("nTime").as_string()), nullptr, 16);

          // if (devMine)
          // {
          //   submittingDev = true;
          //   if (localJobCounter != jobCounter || localDevHeight != devHeight)
          //     break;
          //   setcolor(CYAN);
          //   std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
          //   setcolor(BRIGHT_WHITE);

          //   BTCStratum::formatShare(devShare, myJobDev, devWorkerName, n, baseNTime,
          //                         (uint32_t)myJobDev.at("extraNonce2").get_uint64());
          //   data_ready = true;
          // }
          // else
          // {
          //   submitting = true;
          //   if (localJobCounter != jobCounter || localOurHeight != ourHeight)
          //     break;
          //   setcolor(BRIGHT_YELLOW);
          //   std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
          //   setcolor(BRIGHT_WHITE);

          //   BTCStratum::formatShare(share, myJob, workerName, n, baseNTime,
          //                         (uint32_t)myJob.at("extraNonce2").get_uint64());
          //   data_ready = true;
          // }
          // cv.notify_all();
        }

        if (!isConnected)
        {
          data_ready = true;
          cv.notify_all();
          break;
        }
      }
    }
    catch (std::exception &e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function: " << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);

      localJobCounter = -1;
      localOurHeight = -1;
      localDevHeight = -1;
    }
  }

  goto waitForJob;
}