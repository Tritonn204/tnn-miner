#include "miners.hpp"
#include "tnn-hugepages.h"
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

  thread_local workerData *astroWorker = (workerData *)malloc_huge_pages(sizeof(workerData));
  thread_local SpectreX::worker *worker = (SpectreX::worker *)malloc_huge_pages(sizeof(SpectreX::worker));
  initWorker(*astroWorker);
  lookupGen(*astroWorker, nullptr, nullptr);
  worker->astroWorker = astroWorker;

  thread_local workerData *devAstroWorker = (workerData *)malloc_huge_pages(sizeof(workerData));
  thread_local SpectreX::worker *devWorker = (SpectreX::worker *)malloc_huge_pages(sizeof(SpectreX::worker));
  initWorker(*devAstroWorker);
  lookupGen(*devAstroWorker, nullptr, nullptr);
  devWorker->astroWorker = devAstroWorker;

    thread_local std::random_device rd;
    thread_local std::mt19937 rng(rd());
  thread_local std::uniform_real_distribution<double> dist(0, 10000);

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
      bool assigned = false;
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

      // printf("end of job application\n");
      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;
        which = dist(rng);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        DIFF = devMine ? doubleDiffDev : doubleDiff;
        if (DIFF == 0)
          continue;

        // cmpDiff = ConvertDifficultyToBig(DIFF, SPECTRE_X);
        byte* cmpDiff = devMine ? diffBytes_dev : diffBytes;

        uint64_t *nonce = devMine ? &nonce0_dev : &nonce0;
        (*nonce)++;

        // printf("nonce = %llu\n", *nonce);

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

        // printf("after nonce: %s\n", hexStr(WORK, SpectreX::INPUT_SIZE).c_str());

        if (localJobCounter != jobCounter) {
          // printf("thread %d updating job before hash\n", tid);
          break;
        }

        /*
        double which = (double)(rand() % 10000);
        bool devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
  
        byte* cmpDiff = devMine ? diffBytes_dev : diffBytes;
        uint64_t *nonce = devMine ? &nonce0_dev : &nonce0;
        (*nonce)++;
        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        uint64_t *n = (uint64_t*)&WORK[72];

        int enLen = devMine ? nonceLenDev : nonceLen;
        if(enLen <= 0) {
          (*n) = ((tid - 1) % (256 * 256)) | ((rand() % 256) << 16) | ((*nonce) << 24);
          //nonceBytes[0] = (byte)&n; //((tid - 1) % (256 * 256)) | ((rand() % 256) << 16) | ((*nonce) << 24);
          /
          //memcpy(nonceBytes, (byte *)&n, 8);
          nonceBytes[0] = ((tid - 1) % (256 * 256)) & 0xFF;
          nonceBytes[1] = ((tid - 1) % (256 * 256) >> 8) & 0xFF;
          
          nonceBytes[2] = (myRand % 256) & 0xFF;
          //nonceBytes[3] = (myRand >> 8) & 0xFF;
          
          nonceBytes[3] = (*nonce) & 0xFF;
          nonceBytes[4] = ((*nonce) >> 8) & 0xFF;
          nonceBytes[5] = ((*nonce) >> 16) & 0xFF;
          nonceBytes[6] = ((*nonce) >> 24) & 0xFF;
          nonceBytes[7] = ((*nonce) >> 32) & 0xFF;
          /
        } else {
          uint64_t &eN = devMine ? nonce0_dev : nonce0;
          int offset = (64 - enLen*8);
          (*n) = ((tid - 1) % (256 * 256)) | (((*nonce) << 16) & ((1ULL << offset)-1)) | (eN << offset);
        }
        //memcpy(nonceBytes, (byte *)&n, 8);

        // printf("after nonce: %s\n", hexStr(WORK, SpectreX::INPUT_SIZE).c_str());

        //if (localJobCounter != jobCounter) {
        //  // printf("thread %d updating job before hash\n", tid);
        //  break;
        //}
        */

        SpectreX::worker &usedWorker = devMine ? *devWorker : *worker;
        SpectreX::hash(usedWorker, WORK, SpectreX::INPUT_SIZE, powHash);

        // if (littleEndian())
        // {
        //   std::reverse(powHash, powHash + 32);
        // }

        counter.fetch_add(1);
        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight) {
          // printf("thread %d updating job after hash\n", tid);
          break;
        }


        if (SpectreX::checkNonce(((uint64_t*)usedWorker.scratchData),((uint64_t*)cmpDiff)))
        {
          // printf("thread %d entered submission process\n", tid);
          if (!submit) {
            for(;;) {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              int64_t &rH = devMine ? devHeight : ourHeight;
              int64_t &oH = devMine ? localDevHeight : localOurHeight;
              if (submit || localJobCounter != jobCounter || rH != oH)
                break;
              boost::this_thread::yield();
            }
          }

          int64_t &rH = devMine ? devHeight : ourHeight;
          int64_t &oH = devMine ? localDevHeight : localOurHeight;
          if (localJobCounter != jobCounter || rH != oH) {
            // printf("thread %d updating job after check\n", tid);
            break;
          }
          // if (littleEndian())
          // {
          //   std::reverse(powHash, powHash + 32);
          // }
        //   std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          // boost::lock_guard<boost::mutex> lock(mutex);
          if (devMine)
          {
            submittingDev = true;
            // std::scoped_lock<boost::mutex> lockGuard(devMutex);
            // if (localJobCounter != jobCounter || localDevHeight != devHeight)
            // {
            //   break;
            // }
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (devMiningProfile.protocol)
            {
            case PROTO_SPECTRE_SOLO:
              devShare = {{"block_template", hexStr(&WORK[0], SpectreX::INPUT_SIZE).c_str()}};
              break;
            case PROTO_SPECTRE_STRATUM:
              std::vector<char> nonceStr;
              // Num(std::to_string((n << enLen*8) >> enLen*8).c_str(),10).print(nonceStr, 16);
              Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
              std::string fullWorkerName = std::string(devWorkerName);
              fullWorkerName += "-" + std::string(tnnTargetArch);
              devShare = {{{"id", SpectreStratum::submitID},
                        {"method", SpectreStratum::submit.method.c_str()},
                        {"params", {fullWorkerName.c_str(), // WORKER
                                    myJobDev.at("jobId").as_string().c_str(), // JOB ID
                                    std::string(nonceStr.data()).c_str()}}}};

              break;
            }
            data_ready = true;
          }
          else
          {
            submitting = true;
            // std::scoped_lock<boost::mutex> lockGuard(userMutex);
            // if (localJobCounter != jobCounter || localOurHeight != ourHeight)
            // {
            //   break;
            // }
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (miningProfile.protocol)
            {
            case PROTO_SPECTRE_SOLO:
              share = {{"block_template", hexStr(&WORK[0], SpectreX::INPUT_SIZE).c_str()}};
              break;
            case PROTO_SPECTRE_STRATUM:
              std::vector<char> nonceStr;
              // Num(std::to_string((n << enLen*8) >> enLen*8).c_str(),10).print(nonceStr, 16);
              Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
              share = {{{"id", SpectreStratum::submitID},
                        {"method", SpectreStratum::submit.method.c_str()},
                        {"params", {workerName,                                   // WORKER
                                    myJob.at("jobId").as_string().c_str(), // JOB ID
                                    std::string(nonceStr.data()).c_str()}}}};

              // std::cout << "blob: " << hexStr(&WORK[0], SpectreX::INPUT_SIZE).c_str() << std::endl;
              // std::cout << "nonce: " << nonceStr.data() << std::endl;
              // std::cout << "extraNonce: " << hexStr(&WORK[SpectreX::INPUT_SIZE - 48], enLen).c_str() << std::endl;
              // std::cout << "hash: " << hexStr(&powHash[0], 32) << std::endl;
              // std::vector<char> diffHex;
              // cmpDiff.print(diffHex, 16);
              // std::cout << "difficulty (LE): " << std::string(diffHex.data()).c_str() << std::endl;
              // std::cout << "powValue: " << Num(hexStr(powHash, 32).c_str(), 16) << std::endl;
              // std::cout << "target (decimal): " << cmpDiff << std::endl;

              // printf("blob: %s\n", foundBlob.c_str());
              // printf("hash (BE): %s\n", hexStr(&powHash[0], 32).c_str());
              // printf("nonce (Full bytes for injection): %s\n", hexStr((byte *)&n, 8).c_str());

              break;
            }
            data_ready = true;
          }
          // printf("thread %d finished submission process\n", tid);
          cv.notify_all();
        }

        if (!isConnected) {
          break;
        }
      }
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
