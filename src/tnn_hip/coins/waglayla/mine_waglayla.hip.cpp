// The order is important here for WIN32 builds: https://stackoverflow.com/a/16288859
#include <coins/miners.hpp>
#include "tnn-hugepages.h"
#include <stratum/stratum.h>

#include "mine_waglayla.hip.h"

#include <algo_definitions.h>

void jobThread(
  int d, 
  Wala_HIP_Worker::workerCtx ctx, 
  int64_t localJobCounter,
  int64_t localOurHeight,
  uint64_t n,
  uint64_t nonceMask,
  uint8_t *work,
  std::string jobId,
  bool devMine
) {
  Wala_HIP_Worker::setDevice(d);
  Wala_HIP::copyDiff(ctx.cmpDiff);

  int h_nonceCount = 0;
  uint64_t h_nonceBuffer[Wala_HIP::MAX_NONCES];

  while (doubleDiff == 0) {
    std::this_thread::yield();
  }

  // printf("GPU config sanity:\nblocks: %d, batchSize: %d\n", ctx.blocks[d],ctx.batchSizes[d]);

  while (localJobCounter == jobCounter)
  {
    CHECK_CLOSE;
    if (localJobCounter != jobCounter)
    {
      break;
    }

    uint64_t &N = devMine ? HIP_kIndex_dev[d] : HIP_kIndex[d];

    Wala_HIP::walaHash_wrapper(
        ctx.blocks[d],
        nonceMask,
        n,
        ctx.d_nonceBuffer,
        ctx.d_nonceCount,
        ctx.d_hashBuffer,
        N,
        ctx.batchSizes[d],
        d,
        devMine);

    HIP_counters[d].fetch_add(ctx.batchSizes[d]);
    counter.fetch_add(ctx.batchSizes[d]);
    // printf("%d loops\n", kernelIndex);

    N++; 
    // bool submit = (devMine && devConnected) ? !submittingDev : !submitting;

    if (localJobCounter != jobCounter || localOurHeight != ourHeight)
    {
      // printf("thread %d updating job after hash\n", tid);
      return;
    }

    Wala_HIP::nonceCounter(ctx.d_nonceCount, &h_nonceCount, ctx.d_nonceBuffer, h_nonceBuffer);
    if (h_nonceCount > 0) {
      for (int i = 0; i < h_nonceCount; i++) {
        uint64_t nonce = h_nonceBuffer[i];
        printf("\n");
        if (devMine) {
          setcolor(CYAN);
          printf("DEV | ");
        } else {
          setcolor(BRIGHT_YELLOW);
        }
        printf("GPU #%d found a nonce!", d);
        fflush(stdout);
        setcolor(BRIGHT_WHITE);

        bool submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (!submit) {
          for(;;) {
            submit = (devMine && devConnected) ? !submittingDev : !submitting;
            int64_t &rH = devMine ? devHeight : ourHeight;
            if (submit || localJobCounter != jobCounter || rH != localOurHeight)
              return;
            boost::this_thread::yield();
          }
        }

        int64_t &rH = devMine ? devHeight : ourHeight;
        if (localJobCounter != jobCounter || rH != localOurHeight) {
          return;
        }

        if (devMine)
        {
          submittingDev = true;
          switch (devMiningProfile.protocol)
          {
          case PROTO_KAS_SOLO:
            break;
          case PROTO_KAS_STRATUM:
            std::vector<char> nonceStr;
            Num(std::to_string(nonce).c_str(),10).print(nonceStr, 16);
            devShare = {{{"id", SpectreStratum::submitID},
                      {"method", SpectreStratum::submit.method.c_str()},
                      {"params", {devWorkerName,                                   // WORKER
                                  jobId.c_str(), // JOB ID
                                  std::string(nonceStr.data()).c_str()}}}};


            break;
          }
          data_ready = true;
        }
        else
        {
          submitting = true;
          switch (miningProfile.protocol)
          {
          case PROTO_KAS_SOLO:
            break;
          case PROTO_KAS_STRATUM:
            std::vector<char> nonceStr;
            Num(std::to_string(nonce).c_str(),10).print(nonceStr, 16);
            share = {{{"id", SpectreStratum::submitID},
                      {"method", SpectreStratum::submit.method.c_str()},
                      {"params", {workerName,                                   // WORKER
                                  jobId.c_str(), // JOB ID
                                  std::string(nonceStr.data()).c_str()}}}};
            break;
          }
          data_ready = true;
        }
        cv.notify_all();
      }
    }

    if (!isConnected)
    {
      return;
    }

    std::this_thread::yield();
  }
}

void mineWaglayla_hip(int tid)
{
  int d = 0;

  std::string diffHex;
  std::string diffHex_dev;

  byte diffBytes[32];
  byte diffBytes_dev[32];

  Wala_HIP_Worker::workerCtx ctx;
  Wala_HIP_Worker::newCtx(ctx);

  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  byte powHash[32];
  byte work[Wala_HIP::INPUT_SIZE] = {0};
  byte devWork[Wala_HIP::INPUT_SIZE] = {0};

waitForJob:
  while (!isConnected)
  {
    CHECK_CLOSE;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  uint64_t *nonceCache = (uint64_t*)malloc(sizeof(uint64_t)*ctx.GPUCount);

  Wala_HIP_Worker::ctxMalloc(ctx);

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
      
      if (!myJob.at("template").is_string()) {
        continue;
      }
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[Wala_HIP::INPUT_SIZE];
        switch (miningProfile.protocol)
        {
        case PROTO_KAS_SOLO:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        case PROTO_KAS_STRATUM:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        }
        memcpy(work, b2, Wala_HIP::HASH_HEADER_SIZE);
        delete[] b2;

        localOurHeight = ourHeight;
      }

      for (d = 0; d < ctx.GPUCount; d++) {
        Wala_HIP_Worker::setDevice(d);
        Wala_HIP::copyWork<false>(devWork);
        Wala_HIP::newMatrix(work, false);
        Wala_HIP::absorbPow(work, false);
      }

      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          byte *b2d = new byte[Wala_HIP::INPUT_SIZE];
          switch (devMiningProfile.protocol)
          {
          case PROTO_KAS_SOLO:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          case PROTO_KAS_STRATUM:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          }
          memcpy(devWork, b2d, Wala_HIP::HASH_HEADER_SIZE);
          delete[] b2d;

          localDevHeight = devHeight;
        }

        for (d = 0; d < ctx.GPUCount; d++) {
          Wala_HIP_Worker::setDevice(d);
          Wala_HIP::copyWork<true>(devWork);
          Wala_HIP::newMatrix(devWork, true);
          Wala_HIP::absorbPow(work, true);
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

      // printf("target: ");

      // for (int i = 0; i < 32; i++) {
      //   printf("%02x", diffBytes[i]);
      // }

      // printf("\n");

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

      CHECK_CLOSE;

      // printf("end of job application\n");
      for(d = 0; d < ctx.GPUCount; d++) {
        workers.emplace_back(jobThread, 
          d, 
          ctx, 
          localJobCounter, 
          devMine ? localDevHeight : localOurHeight, 
          n, 
          nonceMask,
          work, 
          devMine ? myJobDev.at("jobId").as_string().c_str() : myJob.at("jobId").as_string().c_str(),
          devMine
        );
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

