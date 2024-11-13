#include "miners.hpp"
#include "tnn-hugepages.h"
#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>

void mineDero(int tid)
{
  byte random_buf[12];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  std::array<int, 12> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                { return dist(gen); });
  std::memcpy(random_buf, buf.data(), buf.size());

  boost::this_thread::sleep_for(boost::chrono::milliseconds(125));


  int64_t localJobCounter;
  byte powHash[32];
  // byte powHash2[32];
  byte devWork[MINIBLOCK_SIZE*DERO_BATCH];
  byte work[MINIBLOCK_SIZE*DERO_BATCH];

  workerData *userWorker = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*userWorker);
  lookupGen(*userWorker, nullptr, nullptr);

  workerData *devWorker = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*devWorker);
  lookupGen(*devWorker, nullptr, nullptr);

  // std::cout << *worker << std::endl;

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

      byte *b2 = new byte[MINIBLOCK_SIZE];
      hexstrToBytes(std::string(myJob.at("blockhashing_blob").as_string()), b2);
      for (int i = 0; i < DERO_BATCH; i++) {
        memcpy(work + i*MINIBLOCK_SIZE, b2, MINIBLOCK_SIZE);
      }
      delete[] b2;

      if (devConnected)
      {
        byte *b2d = new byte[MINIBLOCK_SIZE];
        hexstrToBytes(std::string(myJobDev.at("blockhashing_blob").as_string()), b2d);
        for (int i = 0; i < DERO_BATCH; i++) {
          memcpy(devWork + i*MINIBLOCK_SIZE, b2d, MINIBLOCK_SIZE);
        }
        delete[] b2d;
      }

      for (int i = 0; i < DERO_BATCH; i++) {
        memcpy(&work[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 12], random_buf, 12);
        memcpy(&devWork[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 12], random_buf, 12);

        work[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 1] = (byte)tid;
        devWork[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 1] = (byte)tid;
      }

      if ((work[0] & 0xf) != 1)
      { // check  version
       //  mutex.lock();
        std::cerr << "Unknown version, please check for updates: "
                  << "version" << (work[0] & 0x1f) << std::endl;
       //  mutex.unlock();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        continue;
      }
      double which;
      bool devMine = false;
      bool submit = false;
      int64_t DIFF;
      Num cmpDiff;
      workerData *worker;
      // DIFF = 5000;

      std::string hex;
      int32_t nonce = 0;
      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;
        which = (double)(rand() % 10000);
        devMine = (devConnected && which < devFee * 100.0);
        DIFF = devMine ? difficultyDev : difficulty;
        worker = devMine ? devWorker : userWorker;
        // printf("Difficulty: %" PRIx64 "\n", DIFF);

        cmpDiff = ConvertDifficultyToBig(DIFF, DERO_HASH);
        nonce += DERO_BATCH;
        byte *WORK = devMine ? &devWork[0] : &work[0];

        for (int i = 0; i < DERO_BATCH; i++) {
          int N = nonce + i;
          memcpy(&WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 5], &N, sizeof(N));
        }

        // swap endianness
        if (littleEndian())
        {
          for (int i = 0; i < DERO_BATCH; i++) {
            std::swap(WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 5], WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 2]);
            std::swap(WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 4], WORK[MINIBLOCK_SIZE*i + MINIBLOCK_SIZE - 3]);
          }
        }

        // for (int i = 0; i < MINIBLOCK_SIZE; i++) {
        //   printf("%02x", WORK[i]);
        // }
        // printf("\n");
        AstroBWTv3_batch(WORK, MINIBLOCK_SIZE, powHash, *worker, useLookupMine);
        // AstroBWTv3_batch((byte*)"b", 1, powHash, *worker, useLookupMine);
        // for (int i = 0; i < 32; i++) {
        //   printf("%02x", powHash[i]);
        // }
        // printf("\n");
        // AstroBWTv3(&WORK[0], MINIBLOCK_SIZE, powHash, *worker, useLookupMine);
        // AstroBWTv3((byte*)("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\0"), MINIBLOCK_SIZE, powHash, *worker, useLookupMine);

        counter.fetch_add(DERO_BATCH);
        submit = devMine ? !submittingDev : !submitting;

        for (int i = 0; i < DERO_BATCH; i++) {
          byte *currHash = &powHash[32*i];
          if (CheckHash(currHash, cmpDiff, DERO_HASH))
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
              submittingDev = true;
              setcolor(CYAN);
              std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
              setcolor(BRIGHT_WHITE);
              devShare = {
                  {"jobid", myJobDev.at("jobid").as_string().c_str()},
                  {"mbl_blob", hexStr(&WORK[MINIBLOCK_SIZE*i], MINIBLOCK_SIZE).c_str()}};
              data_ready = true;
            }
            else
            {
              submitting = true;
              setcolor(BRIGHT_YELLOW);
              std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
              setcolor(BRIGHT_WHITE);
              share = {
                  {"jobid", myJob.at("jobid").as_string().c_str()},
                  {"mbl_blob", hexStr(&WORK[MINIBLOCK_SIZE*i], MINIBLOCK_SIZE).c_str()}};
              data_ready = true;
            }
            cv.notify_all();
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
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}
