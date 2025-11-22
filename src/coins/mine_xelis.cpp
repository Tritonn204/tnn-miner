#include "miners.hpp"
#include "tnn-hugepages.h"
#include <xelis-hash/xelis-hash.hpp>
#include <base64.hpp>
#include <stratum/stratum.h>

void mineXelis_v1(int tid)
{
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint64_t i = 0;
  uint64_t i_dev = 0;

  byte powHash[32];
  alignas(64) thread_local byte work[XELIS_BYTES_ARRAY_INPUT] = {0};
  alignas(64) thread_local byte devWork[XELIS_BYTES_ARRAY_INPUT] = {0};
  alignas(64) thread_local byte FINALWORK[XELIS_BYTES_ARRAY_INPUT] = {0};

  alignas(64) thread_local workerData_xelis *worker = (workerData_xelis *)malloc_huge_pages(sizeof(workerData_xelis));

  thread_local std::random_device rd;
  thread_local std::mt19937 rng(rd());
  thread_local std::uniform_real_distribution<double> dist(0, 10000);

  thread_local uint64_t localCount = 0;

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

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[XELIS_TEMPLATE_SIZE];
        switch (miningProfile.protocol)
        {
        case PROTO_XELIS_SOLO:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        case PROTO_XELIS_XATUM:
        {
          std::string b64 = base64::from_base64(std::string(myJob.at("template").as_string().c_str()));
          memcpy(b2, b64.data(), b64.size());
          break;
        }
        case PROTO_XELIS_STRATUM:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        }
        memcpy(work, b2, XELIS_TEMPLATE_SIZE);
        delete[] b2;
        localOurHeight = ourHeight;
        i = 0;
      }

      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          byte *b2d = new byte[XELIS_TEMPLATE_SIZE];
          switch (devMiningProfile.protocol)
          {
          case PROTO_XELIS_SOLO:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          case PROTO_XELIS_XATUM:
          {
            std::string b64 = base64::from_base64(std::string(myJobDev.at("template").as_string().c_str()));
            memcpy(b2d, b64.data(), b64.size());
            break;
          }
          case PROTO_XELIS_STRATUM:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          }
          memcpy(devWork, b2d, XELIS_TEMPLATE_SIZE);
          delete[] b2d;
          localDevHeight = devHeight;
          i_dev = 0;
        }
      }

      bool devMine = false;
      double which;
      bool submit = false;
      uint64_t DIFF;
      Num cmpDiff;

      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;
        which = dist(rng);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        DIFF = devMine ? difficultyDev : difficulty;
        if (DIFF == 0)
          continue;
        cmpDiff = ConvertDifficultyToBig(DIFF, ALGO_XELISV3);

        uint64_t *nonce = devMine ? &i_dev : &i;
        (*nonce)++;

        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        byte *nonceBytes = &WORK[40];
        uint64_t n = ((tid - 1) % (256 * 256)) | ((rand()%256) << 16) | ((*nonce) << 24);
        memcpy(nonceBytes, (byte *)&n, 8);

        if (localJobCounter != jobCounter) {
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        memcpy(FINALWORK, WORK, XELIS_BYTES_ARRAY_INPUT);
        xelis_hash(FINALWORK, *worker, powHash);

        if (littleEndian())
        {
          std::reverse(powHash, powHash + 32);
        }

        if (++localCount >= 1024) { counter.fetch_add(localCount); localCount = 0; }
        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight) {
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        if (CheckHash(powHash, cmpDiff, ALGO_XELISV3))
        {
          if (!submit) {
            for(;;) {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
                break;
              boost::this_thread::yield();
            }
          }
          if (miningProfile.protocol == PROTO_XELIS_XATUM && littleEndian())
          {
            std::reverse(powHash, powHash + 32);
          }

          std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          std::string foundBlob = hexStr(&WORK[0], XELIS_TEMPLATE_SIZE);
          if (devMine)
          {
            submittingDev = true;
            if (localJobCounter != jobCounter || localDevHeight != devHeight)
            {
              if (localCount) { counter.fetch_add(localCount); localCount = 0; }
              break;
            }
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (devMiningProfile.protocol)
            {
            case PROTO_XELIS_SOLO:
              devShare = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case PROTO_XELIS_XATUM:
              devShare = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case PROTO_XELIS_STRATUM:
              devShare = {{{"id", XelisStratum::submitID},
                           {"method", XelisStratum::submit.method.c_str()},
                           {"params", {devWorkerName,
                                       myJobDev.at("jobId").as_string().c_str(),
                                       hexStr((byte *)&n, 8).c_str()}}}};
              break;
            }
            data_ready = true;
          }
          else
          {
            submitting = true;
            if (localJobCounter != jobCounter || localOurHeight != ourHeight)
            {
              if (localCount) { counter.fetch_add(localCount); localCount = 0; }
              break;
            }
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (miningProfile.protocol)
            {
            case PROTO_XELIS_SOLO:
              share = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case PROTO_XELIS_XATUM:
              share = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case PROTO_XELIS_STRATUM:
              share = {{{"id", XelisStratum::submitID},
                        {"method", XelisStratum::submit.method.c_str()},
                        {"params", {workerName,
                                    myJob.at("jobId").as_string().c_str(),
                                    hexStr((byte *)&n, 8).c_str()}}}};
              std::vector<char> diffHex;
              cmpDiff.print(diffHex, 16);
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
      if (!isConnected)
        break;
    }
    catch (std::exception& e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);
      if (localCount) { counter.fetch_add(localCount); localCount = 0; }
      localJobCounter = -1;
      localOurHeight = -1;
      localDevHeight = -1;
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}

void mineXelis(int tid)
{
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint64_t i = 0;
  uint64_t i_dev = 0;

  thread_local byte powHash[32];
  alignas(64) thread_local byte work[XELIS_TEMPLATE_SIZE] = {0};
  alignas(64) thread_local byte devWork[XELIS_TEMPLATE_SIZE] = {0};
  alignas(64) thread_local byte FINALWORK[XELIS_TEMPLATE_SIZE] = {0};

  alignas(64) thread_local workerData_xelis_v3 *worker = (workerData_xelis_v3 *)malloc_huge_pages(sizeof(workerData_xelis_v3));

  thread_local std::random_device rd;
  thread_local std::mt19937 rng(rd());
  thread_local std::uniform_real_distribution<double> dist(0, 10000);
  thread_local std::uniform_real_distribution<double> n2(0, 256);

  thread_local uint64_t localCount = 0;

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

      if (!myJob.at("miner_work").is_string())
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[XELIS_TEMPLATE_SIZE];
        switch (miningProfile.protocol)
        {
        case PROTO_XELIS_SOLO:
          hexstrToBytes(std::string(myJob.at("miner_work").as_string()), b2);
          break;
        case PROTO_XELIS_XATUM:
        {
          std::string b64 = base64::from_base64(std::string(myJob.at("miner_work").as_string().c_str()));
          memcpy(b2, b64.data(), b64.size());
          break;
        }
        case PROTO_XELIS_STRATUM:
          hexstrToBytes(std::string(myJob.at("miner_work").as_string()), b2);
          break;
        }
        memcpy(work, b2, XELIS_TEMPLATE_SIZE);
        delete[] b2;
        localOurHeight = ourHeight;
        i = 0;
      }

      if (devConnected && myJobDev.at("miner_work").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          byte *b2d = new byte[XELIS_TEMPLATE_SIZE];
          switch (devMiningProfile.protocol)
          {
          case PROTO_XELIS_SOLO:
            hexstrToBytes(std::string(myJobDev.at("miner_work").as_string()), b2d);
            break;
          case PROTO_XELIS_XATUM:
          {
            std::string b64 = base64::from_base64(std::string(myJobDev.at("miner_work").as_string().c_str()));
            memcpy(b2d, b64.data(), b64.size());
            break;
          }
          case PROTO_XELIS_STRATUM:
            hexstrToBytes(std::string(myJobDev.at("miner_work").as_string()), b2d);
            break;
          }
          memcpy(devWork, b2d, XELIS_TEMPLATE_SIZE);
          delete[] b2d;
          localDevHeight = devHeight;
          i_dev = 0;
        }
      }

      bool devMine = false;
      double which;
      bool submit = false;
      uint64_t DIFF;
      Num cmpDiff;

      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;
        which = dist(rng);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        DIFF = devMine ? difficultyDev : difficulty;
        if (DIFF == 0)
          continue;
        cmpDiff = ConvertDifficultyToBig(DIFF, ALGO_XELISV3);

        uint64_t *nonce = devMine ? &i_dev : &i;
        (*nonce)++;

        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        byte *nonceBytes = &WORK[40];
        uint64_t n = ((tid - 1) % (256 * 256)) | ((int)(n2(rng)) << 16) | ((*nonce) << 24);
        memcpy(nonceBytes, (byte *)&n, 8);

        if (localJobCounter != jobCounter) {
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        memcpy(FINALWORK, WORK, XELIS_TEMPLATE_SIZE);
        xelis_hash_v3(FINALWORK, *worker, powHash);

        if (littleEndian())
        {
          std::reverse(powHash, powHash + 32);
        }

        if (++localCount >= 1024) { counter.fetch_add(localCount); localCount = 0; }
        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight) {
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }

        if (CheckHash(powHash, cmpDiff, ALGO_XELISV3))
        {
          if (!submit) {
            for(;;) {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
                break;
              boost::this_thread::yield();
            }
          }
          if (miningProfile.protocol == PROTO_XELIS_XATUM && littleEndian())
          {
            std::reverse(powHash, powHash + 32);
          }

          std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          std::string foundBlob = hexStr(&WORK[0], XELIS_TEMPLATE_SIZE);
          if (devMine)
          {
            if (localJobCounter != jobCounter || localDevHeight != devHeight)
            {
              if (localCount) { counter.fetch_add(localCount); localCount = 0; }
              break;
            }
            submittingDev = true;
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (devMiningProfile.protocol)
            {
            case PROTO_XELIS_SOLO:
              devShare = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case PROTO_XELIS_XATUM:
              devShare = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case PROTO_XELIS_STRATUM:
              devShare = {{{"id", XelisStratum::submitID},
                           {"method", XelisStratum::submit.method.c_str()},
                           {"params", {devWorkerName,
                                       myJobDev.at("jobId").as_string().c_str(),
                                       hexStr((byte *)&n, 8).c_str()}}}};
              break;
            }
            data_ready = true;
          }
          else
          {
            if (localJobCounter != jobCounter || localOurHeight != ourHeight)
            {
              if (localCount) { counter.fetch_add(localCount); localCount = 0; }
              break;
            }
            submitting = true;
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (miningProfile.protocol)
            {
            case PROTO_XELIS_SOLO:
              share = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case PROTO_XELIS_XATUM:
              share = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case PROTO_XELIS_STRATUM:
              share = {{{"id", XelisStratum::submitID},
                        {"method", XelisStratum::submit.method.c_str()},
                        {"params", {workerName,
                                    myJob.at("jobId").as_string().c_str(),
                                    hexStr((byte *)&n, 8).c_str()}}}};
              std::vector<char> diffHex;
              cmpDiff.print(diffHex, 16);
              break;
            }
            data_ready = true;
          }
          cv.notify_all();
        }

        if (!isConnected) {
          data_ready = true;
          cv.notify_all();
          if (localCount) { counter.fetch_add(localCount); localCount = 0; }
          break;
        }
      }
      if (localCount) { counter.fetch_add(localCount); localCount = 0; }
      if (!isConnected) {
        data_ready = true;
        cv.notify_all();
        break;
      }
    }
    catch (std::exception& e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);

      if (localCount) { counter.fetch_add(localCount); localCount = 0; }

      localJobCounter = -1;
      localOurHeight = -1;
      localDevHeight = -1;
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}
