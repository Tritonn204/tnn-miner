#include <tnn_hip/coins/miners.hip.hpp>
#include "tnn-hugepages.h"
#include <stratum/stratum.h>

#include "mine_astrix.hip.h"

void jobThread(
  int d, 
  Astrix_HIP_Worker::workerCtx ctx, 
  int64_t localJobCounter,
  int64_t localOurHeight,
  uint64_t n,
  uint64_t nonceMask,
  bool devMine
) {
  Astrix_HIP_Worker::setDevice(d);
  Astrix_HIP::copyDiff(ctx.cmpDiff);

  int kernelIndex = 0;

  uint16_t h_nonceCount = 0;
  uint64_t h_nonceBuffer[Astrix_HIP::MAX_NONCES];

  // printf("GPU config sanity:\nblocks: %d, batchSize: %d\n", ctx.blocks[d],ctx.batchSizes[d]);

  while (localJobCounter == jobCounter)
  {
    if (localJobCounter != jobCounter)
    {
      break;
    }

    Astrix_HIP::astrixHash_wrapper(
        ctx.blocks[d],
        nonceMask,
        n,
        ctx.d_nonceBuffer,
        ctx.d_nonceCount,
        ctx.d_hashBuffer,
        kernelIndex,
        ctx.batchSizes[d],
        d,
        devMine);

    HIP_counters[d].fetch_add(ctx.batchSizes[d]);
    counter.fetch_add(ctx.batchSizes[d]);
    // printf("%d loops\n", kernelIndex);

    kernelIndex++;
    // bool submit = (devMine && devConnected) ? !submittingDev : !submitting;

    if (localJobCounter != jobCounter || localOurHeight != ourHeight)
    {
      // printf("thread %d updating job after hash\n", tid);
      break;
    }

    if (!isConnected)
    {
      break;
    }

    std::this_thread::yield();
  }
}

void mineAstrix_hip()
{
  int d = 0;

  std::string diffHex;
  std::string diffHex_dev;

  byte diffBytes[32];
  byte diffBytes_dev[32];

  Astrix_HIP_Worker::workerCtx ctx;
  Astrix_HIP_Worker::newCtx(ctx);

  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  byte powHash[32];
  byte work[Astrix_HIP::INPUT_SIZE] = {0};
  byte devWork[Astrix_HIP::INPUT_SIZE] = {0};

waitForJob:
  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  uint64_t *nonceCache = (uint64_t*)malloc(sizeof(uint64_t)*ctx.GPUCount);

  Astrix_HIP_Worker::ctxMalloc(ctx);

  while (true)
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
      
      if (!myJob.at("template").is_string()) {
        continue;
      }
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[Astrix_HIP::INPUT_SIZE];
        switch (protocol)
        {
        case ASTRIX_SOLO:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        case ASTRIX_STRATUM:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        }
        memcpy(work, b2, Astrix_HIP::HASH_HEADER_SIZE);
        delete[] b2;

        localOurHeight = ourHeight;
        nonce0 = 0;
      }

      for (d = 0; d < ctx.GPUCount; d++) {
        Astrix_HIP_Worker::setDevice(d);
        Astrix_HIP::copyWork<false>(devWork);
        Astrix_HIP::newMatrix(work, false);
        Astrix_HIP::absorbPow(work, false);
      }

      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          byte *b2d = new byte[Astrix_HIP::INPUT_SIZE];
          switch (protocol)
          {
          case ASTRIX_SOLO:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          case ASTRIX_STRATUM:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          }
          memcpy(devWork, b2d, Astrix_HIP::HASH_HEADER_SIZE);
          delete[] b2d;

          localDevHeight = devHeight;
          nonce0_dev = 0;
        }

        for (d = 0; d < ctx.GPUCount; d++) {
          Astrix_HIP_Worker::setDevice(d);
          Astrix_HIP::copyWork<true>(devWork);
          Astrix_HIP::newMatrix(devWork, true);
          Astrix_HIP::absorbPow(work, true);
        }
      }

      bool devMine = false;
      double which;
      bool submit = false;
      double DIFF = 1;

      diffHex.clear();
      diffHex_dev.clear();

      diffHex = cpp_int_toHex(bigDiff);
      diffHex_dev = cpp_int_toHex(bigDiff_dev);

      cpp_int_to_byte_array(bigDiff, diffBytes);
      cpp_int_to_byte_array(bigDiff_dev, diffBytes_dev);

      std::vector<std::thread> workers;
      uint64_t n;
      
      int enLen = 0;
      uint64_t nonceMask = -1ULL;
      
      boost::json::value &J = devMine ? myJobDev : myJob;
      if (!J.as_object().if_contains("extraNonce") || J.at("extraNonce").as_string().size() == 0) {
        n = (uint64_t)(rand() % 65536) << 48;
        nonceMask >>= 16;
      } else {
        uint64_t eN = std::stoull(std::string(J.at("extraNonce").as_string().c_str()), NULL, 16);
        enLen = (std::string(J.at("extraNonce").as_string()).size()+1)/2;
        int offset = (64 - enLen*8);
        n = (eN << offset);
        nonceMask >>= enLen*8;
      }

      which = (double)(rand() % 10000);
      devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);

      ctx.cmpDiff = devMine ? diffBytes_dev : diffBytes;

      // printf("end of job application\n");
      for(d = 0; d < ctx.GPUCount; d++) {
        workers.emplace_back(jobThread, d, ctx, localJobCounter, localOurHeight, n, nonceMask, devMine);
      }

      for (auto& th : workers) {
        th.join();
      }

      workers.clear();
    }
    catch (std::exception& e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);

      localJobCounter = -1;
      localOurHeight = -1;
      localDevHeight = -1;
    }
    if (!isConnected) {
      data_ready = true;
      cv.notify_all();
      break;
    }
  }
  goto waitForJob;
}

