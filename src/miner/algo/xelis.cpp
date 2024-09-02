#include <algos.hpp>
#include <xelis-hash.hpp>
#include <hugepages.h>
#include <base64.hpp>
#include <stratum.h>

void mineXelis_v1(int tid)
{
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint64_t i = 0;
  uint64_t i_dev = 0;

  byte powHash[32];
  alignas(64) byte work[XELIS_BYTES_ARRAY_INPUT] = {0};
  alignas(64) byte devWork[XELIS_BYTES_ARRAY_INPUT] = {0};
  alignas(64) byte FINALWORK[XELIS_BYTES_ARRAY_INPUT] = {0};

  alignas(64) workerData_xelis *worker = (workerData_xelis *)malloc_huge_pages(sizeof(workerData_xelis));

waitForJob:

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

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

      if (!myJob.at("template").is_string())
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[XELIS_TEMPLATE_SIZE];
        switch (protocol)
        {
        case XELIS_SOLO:
          hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
          break;
        case XELIS_XATUM:
        {
          std::string b64 = base64::from_base64(std::string(myJob.at("template").as_string().c_str()));
          memcpy(b2, b64.data(), b64.size());
          break;
        }
        case XELIS_STRATUM:
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
          switch (protocol)
          {
          case XELIS_SOLO:
            hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
            break;
          case XELIS_XATUM:
          {
            std::string b64 = base64::from_base64(std::string(myJobDev.at("template").as_string().c_str()));
            memcpy(b2d, b64.data(), b64.size());
            break;
          }
          case XELIS_STRATUM:
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
        which = (double)(rand() % 10000);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        DIFF = devMine ? difficultyDev : difficulty;
        if (DIFF == 0)
          continue;
        cmpDiff = ConvertDifficultyToBig(DIFF, XELIS_HASH);

        uint64_t *nonce = devMine ? &i_dev : &i;
        (*nonce)++;

        // printf("nonce = %llu\n", *nonce);

        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        byte *nonceBytes = &WORK[40];
        uint64_t n = ((tid - 1) % (256 * 256)) | ((rand()%256) << 16) | ((*nonce) << 24);
        memcpy(nonceBytes, (byte *)&n, 8);

        // if (littleEndian())
        // {
        //   std::swap(nonceBytes[7], nonceBytes[0]);
        //   std::swap(nonceBytes[6], nonceBytes[1]);
        //   std::swap(nonceBytes[5], nonceBytes[2]);
        //   std::swap(nonceBytes[4], nonceBytes[3]);
        // }

        if (localJobCounter != jobCounter)
          break;

        // std::copy(WORK, WORK + XELIS_TEMPLATE_SIZE, FINALWORK);
        memcpy(FINALWORK, WORK, XELIS_BYTES_ARRAY_INPUT);
        
        xelis_hash(FINALWORK, *worker, powHash);

        if (littleEndian())
        {
          std::reverse(powHash, powHash + 32);
        }

        counter.fetch_add(1);
        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight)
          break;

        if (CheckHash(powHash, cmpDiff, XELIS_HASH))
        {
          if (!submit) {
            for(;;) {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
                break;
              boost::this_thread::yield();
            }
          }
          if (protocol == XELIS_XATUM && littleEndian())
          {
            std::reverse(powHash, powHash + 32);
          }
          // if (protocol == XELIS_STRATUM && littleEndian())
          // {
          //   std::reverse((byte*)&n, (byte*)n + 8);
          // }

          std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          std::string foundBlob = hexStr(&WORK[0], XELIS_TEMPLATE_SIZE);
          // boost::lock_guard<boost::mutex> lock(mutex);
          if (devMine)
          {
            submittingDev = true;
           //  mutex.lock();
            if (localJobCounter != jobCounter || localDevHeight != devHeight)
            {
             //  mutex.unlock();
              break;
            }
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case XELIS_SOLO:
              devShare = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case XELIS_XATUM:
              devShare = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case XELIS_STRATUM:
              devShare = {{{"id", XelisStratum::submitID},
                           {"method", XelisStratum::submit.method.c_str()},
                           {"params", {devWorkerName,                                 // WORKER
                                       myJobDev.at("jobId").as_string().c_str(), // JOB ID
                                       hexStr((byte *)&n, 8).c_str()}}}};
              break;
            }
            data_ready = true;
           //  mutex.unlock();
          }
          else
          {
            submitting = true;
           //  mutex.lock();
            if (localJobCounter != jobCounter || localOurHeight != ourHeight)
            {
             //  mutex.unlock();
              break;
            }
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case XELIS_SOLO:
              share = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case XELIS_XATUM:
              share = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case XELIS_STRATUM:
              share = {{{"id", XelisStratum::submitID},
                        {"method", XelisStratum::submit.method.c_str()},
                        {"params", {workerName,                                   // WORKER
                                    myJob.at("jobId").as_string().c_str(), // JOB ID
                                    hexStr((byte *)&n, 8).c_str()}}}};

              // std::cout << "blob: " << hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str() << std::endl;
              // std::cout << "hash: " << hexStr(&powHash[0], 32) << std::endl;
              std::vector<char> diffHex;
              cmpDiff.print(diffHex, 16);
              // std::cout << "difficulty (LE): " << std::string(diffHex.data()).c_str() << std::endl;
              // printf("blob: %s\n", foundBlob.c_str());
              // printf("hash (BE): %s\n", hexStr(&powHash[0], 32).c_str());
              // printf("nonce (Full bytes for injection): %s\n", hexStr((byte *)&n, 8).c_str());

              break;
            }
            data_ready = true;
           //  mutex.unlock();
          }
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

  byte powHash[32];
  alignas(64) byte work[XELIS_TEMPLATE_SIZE] = {0};
  alignas(64) byte devWork[XELIS_TEMPLATE_SIZE] = {0};
  alignas(64) byte FINALWORK[XELIS_TEMPLATE_SIZE] = {0};

  alignas(64) workerData_xelis_v2 *worker = (workerData_xelis_v2 *)malloc_huge_pages(sizeof(workerData_xelis));

waitForJob:

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

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

      if (!myJob.at("miner_work").is_string())
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;

      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        byte *b2 = new byte[XELIS_TEMPLATE_SIZE];
        switch (protocol)
        {
        case XELIS_SOLO:
          hexstrToBytes(std::string(myJob.at("miner_work").as_string()), b2);
          break;
        case XELIS_XATUM:
        {
          std::string b64 = base64::from_base64(std::string(myJob.at("miner_work").as_string().c_str()));
          memcpy(b2, b64.data(), b64.size());
          break;
        }
        case XELIS_STRATUM:
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
          switch (protocol)
          {
          case XELIS_SOLO:
            hexstrToBytes(std::string(myJobDev.at("miner_work").as_string()), b2d);
            break;
          case XELIS_XATUM:
          {
            std::string b64 = base64::from_base64(std::string(myJobDev.at("miner_work").as_string().c_str()));
            memcpy(b2d, b64.data(), b64.size());
            break;
          }
          case XELIS_STRATUM:
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
        which = (double)(rand() % 10000);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        DIFF = devMine ? difficultyDev : difficulty;
        if (DIFF == 0)
          continue;
        cmpDiff = ConvertDifficultyToBig(DIFF, XELIS_HASH);

        uint64_t *nonce = devMine ? &i_dev : &i;
        (*nonce)++;

        // printf("nonce = %llu\n", *nonce);

        byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
        byte *nonceBytes = &WORK[40];
        uint64_t n = ((tid - 1) % (256 * 256)) | ((rand()%256) << 16) | ((*nonce) << 24);
        memcpy(nonceBytes, (byte *)&n, 8);

        // if (littleEndian())
        // {
        //   std::swap(nonceBytes[7], nonceBytes[0]);
        //   std::swap(nonceBytes[6], nonceBytes[1]);
        //   std::swap(nonceBytes[5], nonceBytes[2]);
        //   std::swap(nonceBytes[4], nonceBytes[3]);
        // }

        if (localJobCounter != jobCounter)
          break;

        memcpy(FINALWORK, WORK, XELIS_TEMPLATE_SIZE);
        
        xelis_hash_v2(FINALWORK, *worker, powHash);

        if (littleEndian())
        {
          std::reverse(powHash, powHash + 32);
        }

        counter.fetch_add(1);
        submit = (devMine && devConnected) ? !submittingDev : !submitting;

        if (localJobCounter != jobCounter || localOurHeight != ourHeight)
          break;

        if (CheckHash(powHash, cmpDiff, XELIS_HASH))
        {
          if (!submit) {
            for(;;) {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
                break;
              boost::this_thread::yield();
            }
          }
          if (protocol == XELIS_XATUM && littleEndian())
          {
            std::reverse(powHash, powHash + 32);
          }
          // if (protocol == XELIS_STRATUM && littleEndian())
          // {
          //   std::reverse((byte*)&n, (byte*)n + 8);
          // }

          std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
          std::string foundBlob = hexStr(&WORK[0], XELIS_TEMPLATE_SIZE);
          // boost::lock_guard<boost::mutex> lock(mutex);
          if (devMine)
          {
            submittingDev = true;
           //  mutex.lock();
            if (localJobCounter != jobCounter || localDevHeight != devHeight)
            {
             //  mutex.unlock();
              break;
            }
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case XELIS_SOLO:
              devShare = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case XELIS_XATUM:
              devShare = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case XELIS_STRATUM:
              devShare = {{{"id", XelisStratum::submitID},
                           {"method", XelisStratum::submit.method.c_str()},
                           {"params", {devWorkerName,                                 // WORKER
                                       myJobDev.at("jobId").as_string().c_str(), // JOB ID
                                       hexStr((byte *)&n, 8).c_str()}}}};
              break;
            }
            data_ready = true;
           //  mutex.unlock();
          }
          else
          {
            submitting = true;
           //  mutex.lock();
            if (localJobCounter != jobCounter || localOurHeight != ourHeight)
            {
             //  mutex.unlock();
              break;
            }
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            switch (protocol)
            {
            case XELIS_SOLO:
              share = {{"block_template", hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str()}};
              break;
            case XELIS_XATUM:
              share = {
                  {"data", b64.c_str()},
                  {"hash", hexStr(&powHash[0], 32).c_str()},
              };
              break;
            case XELIS_STRATUM:
              share = {{{"id", XelisStratum::submitID},
                        {"method", XelisStratum::submit.method.c_str()},
                        {"params", {workerName,                                   // WORKER
                                    myJob.at("jobId").as_string().c_str(), // JOB ID
                                    hexStr((byte *)&n, 8).c_str()}}}};

              // std::cout << "blob: " << hexStr(&WORK[0], XELIS_TEMPLATE_SIZE).c_str() << std::endl;
              // std::cout << "hash: " << hexStr(&powHash[0], 32) << std::endl;
              std::vector<char> diffHex;
              cmpDiff.print(diffHex, 16);
              // std::cout << "difficulty (LE): " << std::string(diffHex.data()).c_str() << std::endl;
              // printf("blob: %s\n", foundBlob.c_str());
              // printf("hash (BE): %s\n", hexStr(&powHash[0], 32).c_str());
              // printf("nonce (Full bytes for injection): %s\n", hexStr((byte *)&n, 8).c_str());

              break;
            }
            data_ready = true;
           //  mutex.unlock();
          }
          cv.notify_all();
        }

        if (!isConnected) {
          data_ready = true;
          cv.notify_all();
          break;
        }
      }
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

      localJobCounter = -1;
      localOurHeight = -1;
      localDevHeight = -1;
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}