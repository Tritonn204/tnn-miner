#include "miners.hpp"
#include "numa_optimizer.hpp"
#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/lookupcompute.h>

#include <array>
#include <atomic>
#include <boost/json.hpp>
#include <boost/chrono.hpp>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <random>
#include <cstdlib>

static std::once_flag s_wolfLUTOnce;
void ensureWolfLUT() { std::call_once(s_wolfLUTOnce, initWolfLUT); }

// target = (2^256) / diff, stored as LE bytes
static inline void deroTargetFromDiff(uint64_t diff, uint8_t *targetOut) {
    static const boost::multiprecision::uint256_t kMaxU256 =
        (boost::multiprecision::uint256_t(1) << 256) - 1;
    if (diff == 0) { memset(targetOut, 0xFF, 32); return; }
    boost::multiprecision::uint256_t target = kMaxU256 / diff;
    boost::multiprecision::uint256_t rem = kMaxU256 % diff;
    if (rem + 1 >= diff) target += 1;
    cpp_int_to_byte_array(target, targetOut);
}

namespace {

static inline void write_nonce_be(byte* WORK, uint32_t N) {
  WORK[MINIBLOCK_SIZE - 5] = (byte)((N >> 24) & 0xFF);
  WORK[MINIBLOCK_SIZE - 4] = (byte)((N >> 16) & 0xFF);
  WORK[MINIBLOCK_SIZE - 3] = (byte)((N >>  8) & 0xFF);
  WORK[MINIBLOCK_SIZE - 2] = (byte)((N      ) & 0xFF);
}

} // namespace

void mineDero(int tid)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);

  thread_local std::mt19937 gen_local(rd());
  thread_local std::uniform_real_distribution<double> dev_dist(0, 10000);

  // NUMA-aware worker context and counters
  thread_local workerData* worker = nullptr;
  thread_local uint64_t localCount = 0;

  // Per-thread mining buffers
  thread_local std::array<byte, MINIBLOCK_SIZE> work;
  thread_local std::array<byte, MINIBLOCK_SIZE> devWork;
  thread_local std::array<byte, 32> powHash;

  if (!worker) {
    worker = static_cast<workerData*>(NUMAOptimizer::allocateLocal(sizeof(workerData)));
    if (!worker) {
      worker = static_cast<workerData*>(std::malloc(sizeof(workerData)));
    }
    if (!worker) {
      setcolor(RED);
      std::cerr << "Failed to allocate workerData for DERO miner" << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);
      return;
    }

    ensureWolfLUT();
    NUMAOptimizer::optimizeMemoryForMining(worker, sizeof(workerData));
    initWorker(*worker);
    lookupGen(*worker, nullptr, nullptr);
    localCount = 0;
  }

  byte random_tail[12];
  for (int i = 0; i < 12; ++i) random_tail[i] = (byte)dist(gen);

  std::this_thread::sleep_for(std::chrono::milliseconds(125));

  int64_t localJobCounter = -1;

  alignas(8) byte diffBytesUser[32] = {0};
  alignas(8) byte diffBytesDev[32] = {0};
  uint64_t cachedDiffUser = 0;
  uint64_t cachedDiffDev = 0;

waitForJob:

  while (!isConnected) {
    CHECK_CLOSE;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  while (!ABORT_MINER) {
    try {
      boost::json::value myJob;
      boost::json::value myJobDev;
      int64_t snapshotCounter;
      int64_t diffUserSnapshot;
      int64_t diffDevSnapshot;

      {
        std::scoped_lock<std::mutex> lockGuard(mutex);
        myJob            = job;
        myJobDev         = devJob;
        snapshotCounter  = jobCounter;
        diffUserSnapshot = difficulty;
        diffDevSnapshot  = difficultyDev;
      }

      hexstrToBytes(std::string(myJob.at("blockhashing_blob").as_string()), work.data());
      if (devConnected) {
        hexstrToBytes(std::string(myJobDev.at("blockhashing_blob").as_string()), devWork.data());
      }

      std::memcpy(&work[MINIBLOCK_SIZE - 12], random_tail, 12);
      work[MINIBLOCK_SIZE - 1] = (byte)tid;
      if (devConnected) {
        std::memcpy(&devWork[MINIBLOCK_SIZE - 12], random_tail, 12);
        devWork[MINIBLOCK_SIZE - 1] = (byte)tid;
      }

      if ((work[0] & 0x0F) != 1) {
        std::cerr << "Unknown version, please check for updates: version"
                  << (work[0] & 0x1F) << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }

      if (diffUserSnapshot != cachedDiffUser) {
        deroTargetFromDiff(diffUserSnapshot, diffBytesUser);
        cachedDiffUser = diffUserSnapshot;
      }
      if (diffDevSnapshot != cachedDiffDev) {
        deroTargetFromDiff(diffDevSnapshot, diffBytesDev);
        cachedDiffDev = diffDevSnapshot;
      }

      localJobCounter = snapshotCounter;
      uint32_t nonce  = 0;

      while (localJobCounter == jobCounter) {
        CHECK_CLOSE;

        const bool devMine = (devConnected && (dev_dist(gen_local) < devFee * 100.0));
        byte*      WORK    = devMine ? devWork.data() : work.data();
        byte*      cmpTarget = devMine ? diffBytesDev : diffBytesUser;

        ++nonce;
        write_nonce_be(WORK, nonce);

        AstroBWTv3(WORK, MINIBLOCK_SIZE, powHash.data(), *worker, useLookupMine);

        if (++localCount >= 512) {
          cpu_counter.fetch_add(localCount);
          localCount = 0;
        }
        // counter.fetch_add(1);

        if (hashMeetsTarget_le(powHash.data(), cmpTarget)) {
          bool submit = devMine ? !submittingDev : !submitting;
          if (!submit) {
            for (;;) {
              submit = devMine ? !submittingDev : !submitting;
              if (submit || localJobCounter != jobCounter) break;
              std::this_thread::yield();
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

      if (localCount) {
        cpu_counter.fetch_add(localCount);
        localCount = 0;
      }

      if (!isConnected) break;
    }
    catch (const std::exception& e) {
      setcolor(RED);
      std::cerr << "Error in POW Function\n" << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);

      localJobCounter = -1;
      if (localCount) {
        cpu_counter.fetch_add(localCount);
        localCount = 0;
      }
    }

    if (!isConnected) break;
  }

  goto waitForJob;
}
