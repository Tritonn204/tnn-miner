#include "miners.hpp"
#include "numa_optimizer.hpp"
#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>
#include <spectrex/spectrex.h>
#include <stratum/stratum.h>

void mineSpectre(int tid)
{
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  thread_local byte powHash[32];
  thread_local byte work[SpectreX::INPUT_SIZE] = {0};
  thread_local byte devWork[SpectreX::INPUT_SIZE] = {0};

  thread_local byte diffBytes[32];
  thread_local byte diffBytes_dev[32];

  thread_local workerData *astroWorker =
      (workerData *)malloc_huge_pages(sizeof(workerData));
  thread_local SpectreX::worker *worker =
      (SpectreX::worker *)malloc_huge_pages(sizeof(SpectreX::worker));
  initWorker(*astroWorker);
  lookupGen(*astroWorker, nullptr, nullptr);
  worker->astroWorker = astroWorker;

  thread_local workerData *devAstroWorker =
      (workerData *)malloc_huge_pages(sizeof(workerData));
  thread_local SpectreX::worker *devWorker =
      (SpectreX::worker *)malloc_huge_pages(sizeof(SpectreX::worker));
  initWorker(*devAstroWorker);
  lookupGen(*devAstroWorker, nullptr, nullptr);
  devWorker->astroWorker = devAstroWorker;

  thread_local std::random_device rd;
  thread_local std::mt19937 rng(rd());
  thread_local std::uniform_real_distribution<double> dist(0, 10000);

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
      bool assigned = false;
      boost::json::value myJob;
      boost::json::value myJobDev;
      {
        std::scoped_lock<std::mutex> lockGuard(mutex);
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
        hexstrToBytes(std::string(myJob.at("template").as_string()), work);
        SpectreX::newMatrix(work, worker->matBuffer, *worker);
        localOurHeight = ourHeight;
      }
      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          hexstrToBytes(std::string(myJobDev.at("template").as_string()), devWork);
          SpectreX::newMatrix(devWork, devWorker->matBuffer, *devWorker);
          localDevHeight = devHeight;
        }
      }

      bool devMine = false;
      double which;
      bool submit = false;
      double DIFF = 1;

      cpp_int_to_byte_array(bigDiff, diffBytes);
      cpp_int_to_byte_array(bigDiff_dev, diffBytes_dev);

      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;
        which = dist(rng);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        DIFF = devMine ? doubleDiffDev : doubleDiff;
        if (DIFF == 0)
          continue;

        byte* cmpDiff = devMine ? diffBytes_dev : diffBytes;

        uint64_t *nonce = devMine ? &nonce0_dev : &nonce0;
        (*nonce)++;

        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        byte *nonceBytes = &WORK[72];
        uint64_t n;
        
        int enLen = 0;
        
        boost::json::value &J = devMine ? myJobDev : myJob;
        if (!J.as_object().if_contains("extraNonce") || J.at("extraNonce").as_string().size() == 0)
          n = ((tid - 1) % (256 * 256)) | ((rand() % 256) << 16) | ((*nonce) << 24);
        else {
          uint64_t eN = std::stoull(std::string(J.at("extraNonce").as_string().c_str()), NULL, 16);
          enLen = (std::string(J.at("extraNonce").as_string()).size()+1) / 2;
          int offset = (64 - enLen*8);
          n = ((tid - 1) % (256 * 256)) | (((*nonce) << 16) & ((1ULL << offset)-1)) | (eN << offset);
        }
        memcpy(nonceBytes, (byte *)&n, 8);

        if (localJobCounter != jobCounter) {
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        SpectreX::worker &usedWorker = devMine ? *devWorker : *worker;
        SpectreX::hash(usedWorker, WORK, SpectreX::INPUT_SIZE, powHash);

        if (++localCount >= 512) { counter.fetch_add(localCount); localCount = 0; }
        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight) {
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        if (SpectreX::checkNonce(((uint64_t*)usedWorker.scratchData),((uint64_t*)cmpDiff)))
        {
          if (!submit) {
            for(;;) {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              int64_t &rH = devMine ? devHeight : ourHeight;
              int64_t &oH = devMine ? localDevHeight : localOurHeight;
              if (submit || localJobCounter != jobCounter || rH != oH)
                break;
              std::this_thread::yield();
            }
          }

          int64_t &rH = devMine ? devHeight : ourHeight;
          int64_t &oH = devMine ? localDevHeight : localOurHeight;
          if (localJobCounter != jobCounter || rH != oH) {
            if (localCount) { counter.fetch_add(localCount); localCount = 0; }
            break;
          }

          if (devMine)
          {
            submittingDev = true;
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (devMiningProfile.protocol)
            {
            case PROTO_SPECTRE_SOLO:
              submitTracker.pushSoloDevice(-1, true);
              devShare = {{"block_template", hexStr(&WORK[0], SpectreX::INPUT_SIZE).c_str()}};
              break;
            case PROTO_SPECTRE_STRATUM:
              std::vector<char> nonceStr;
              Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
              std::string fullWorkerName = std::string(devWorkerName);
              fullWorkerName += "-" + std::string(tnnTargetArch);
              devShare = {{{"rpc_id", submitTracker.nextId(-1)},
                        {"method", SpectreStratum::submit.method.c_str()},
                        {"params", {fullWorkerName.c_str(),
                                    myJobDev.at("jobId").as_string().c_str(),
                                    std::string(nonceStr.data()).c_str()}}}};
              break;
            }
            data_ready = true;
          }
          else
          {
            submitting = true;
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (miningProfile.protocol)
            {
            case PROTO_SPECTRE_SOLO:
              submitTracker.pushSoloDevice(-1, false);
              share = {{"block_template", hexStr(&WORK[0], SpectreX::INPUT_SIZE).c_str()}};
              break;
            case PROTO_SPECTRE_STRATUM:
              std::vector<char> nonceStr;
              Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
              share = {{{"rpc_id", submitTracker.nextId(-1)},
                        {"method", SpectreStratum::submit.method.c_str()},
                        {"params", {workerName,
                                    myJob.at("jobId").as_string().c_str(),
                                    std::string(nonceStr.data()).c_str()}}}};
              break;
            }
            data_ready = true;
          }
          cv.notify_all();
        }

        if (!isConnected) {
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }
      }

      if (localCount) { counter.fetch_add(localCount); localCount = 0; }

      if (!isConnected) {
        break;
      }
    }
    catch (std::exception& e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);

      if (localCount) { counter.fetch_add(localCount); localCount = 0; }

      localJobCounter = -1;
      localOurHeight = -1;
      localDevHeight = -1;
    }
    if (!isConnected) {
      break;
    }
  }
  goto waitForJob;
}
