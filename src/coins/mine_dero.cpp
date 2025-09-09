#include "miners.hpp"
#include "tnn-hugepages.h"
#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>

#include <array>
#include <atomic>
#include <boost/json.hpp>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <cstdint>
#include <cstring>
#include <random>

namespace {

struct ThreadCtx {
  byte   work[MINIBLOCK_SIZE];
  byte   devWork[MINIBLOCK_SIZE];
  byte   powHash[32];
  workerData* worker;
  uint64_t localCount;
};

static inline void write_nonce_be(byte* WORK, uint32_t N) {
  WORK[MINIBLOCK_SIZE - 5] = (byte)((N >> 24) & 0xFF);
  WORK[MINIBLOCK_SIZE - 4] = (byte)((N >> 16) & 0xFF);
  WORK[MINIBLOCK_SIZE - 3] = (byte)((N >>  8) & 0xFF);
  WORK[MINIBLOCK_SIZE - 2] = (byte)((N      ) & 0xFF);
}

}

void mineDero(int tid)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);

  thread_local std::mt19937 gen_local(rd());
  thread_local std::uniform_real_distribution<double> dev_dist(0, 10000);

  thread_local ThreadCtx* ctx = nullptr;
  if (!ctx) {
    ctx = (ThreadCtx*)malloc_huge_pages(sizeof(ThreadCtx));
    std::memset(ctx, 0, sizeof(ThreadCtx));
    ctx->worker = (workerData*)malloc_huge_pages(sizeof(workerData));
    initWorker(*ctx->worker);
    lookupGen(*ctx->worker, nullptr, nullptr);
    ctx->localCount = 0;
  }

  byte random_tail[12];
  for (int i = 0; i < 12; ++i) random_tail[i] = (byte)dist(gen);

  boost::this_thread::sleep_for(boost::chrono::milliseconds(125));

  int64_t localJobCounter = -1;

  Num cmpDiffUser, cmpDiffDev;

waitForJob:

  while (!isConnected) {
    CHECK_CLOSE;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  while (!ABORT_MINER) {
    try {
      boost::json::value myJob;
      boost::json::value myJobDev;
      int64_t snapshotCounter;
      int64_t diffUserSnapshot;
      int64_t diffDevSnapshot;

      {
        std::scoped_lock<boost::mutex> lockGuard(mutex);
        myJob            = job;
        myJobDev         = devJob;
        snapshotCounter  = jobCounter;
        diffUserSnapshot = difficulty;
        diffDevSnapshot  = difficultyDev;
      }

      hexstrToBytes(std::string(myJob.at("blockhashing_blob").as_string()), ctx->work);
      if (devConnected) {
        hexstrToBytes(std::string(myJobDev.at("blockhashing_blob").as_string()), ctx->devWork);
      }

      std::memcpy(&ctx->work[MINIBLOCK_SIZE - 12], random_tail, 12);
      ctx->work[MINIBLOCK_SIZE - 1] = (byte)tid;
      if (devConnected) {
        std::memcpy(&ctx->devWork[MINIBLOCK_SIZE - 12], random_tail, 12);
        ctx->devWork[MINIBLOCK_SIZE - 1] = (byte)tid;
      }

      if ((ctx->work[0] & 0x0F) != 1) {
        std::cerr << "Unknown version, please check for updates: version"
                  << (ctx->work[0] & 0x1F) << std::endl;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        continue;
      }

      cmpDiffUser = ConvertDifficultyToBig(diffUserSnapshot, ALGO_ASTROBWTV3);
      cmpDiffDev  = ConvertDifficultyToBig(diffDevSnapshot,  ALGO_ASTROBWTV3);

      localJobCounter = snapshotCounter;
      uint32_t nonce  = 0;

      while (localJobCounter == jobCounter) {
        CHECK_CLOSE;

        const bool devMine = (devConnected && (dev_dist(gen_local) < devFee * 100.0));
        byte*       WORK   = devMine ? ctx->devWork : ctx->work;
        const Num&  cmp    = devMine ? cmpDiffDev   : cmpDiffUser;

        ++nonce;
        write_nonce_be(WORK, nonce);

        AstroBWTv3(WORK, MINIBLOCK_SIZE, ctx->powHash, *ctx->worker, useLookupMine);

        if (++ctx->localCount >= 1024) {
          counter.fetch_add(ctx->localCount);
          ctx->localCount = 0;
        }
        // counter.fetch_add(1);

        if (CheckHash(ctx->powHash, cmp, ALGO_ASTROBWTV3)) {
          bool submit = devMine ? !submittingDev : !submitting;
          if (!submit) {
            for (;;) {
              submit = devMine ? !submittingDev : !submitting;
              if (submit || localJobCounter != jobCounter) break;
              boost::this_thread::yield();
            }
          }
          if (localJobCounter != jobCounter) break;

          if (devMine) {
            submittingDev = true;
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            devShare = {
              {"jobid",    myJobDev.at("jobid").as_string().c_str()},
              {"mbl_blob", hexStr(WORK, MINIBLOCK_SIZE).c_str()}
            };
            data_ready = true;
          } else {
            submitting = true;
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
            share = {
              {"jobid",    myJob.at("jobid").as_string().c_str()},
              {"mbl_blob", hexStr(WORK, MINIBLOCK_SIZE).c_str()}
            };
            data_ready = true;
          }
          cv.notify_all();
        }

        if (!isConnected) break;
      }

      if (ctx->localCount) {
        counter.fetch_add(ctx->localCount);
        ctx->localCount = 0;
      }

      if (!isConnected) break;
    }
    catch (const std::exception& e) {
      setcolor(RED);
      std::cerr << "Error in POW Function\n" << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);

      localJobCounter = -1;
      if (ctx && ctx->localCount) {
        counter.fetch_add(ctx->localCount);
        ctx->localCount = 0;
      }
    }

    if (!isConnected) break;
  }

  goto waitForJob;
}