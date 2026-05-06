#include <string>
#include "miners.hpp"
#include "tnn-hugepages.hpp"
#include "hex.h"
#include "numa_optimizer.hpp"   // ⭐ NUMA

#include <crypto/shai/shai-hive.h>

std::string convertPathToHexString(const std::vector<uint16_t> &path)
{
  std::ostringstream oss;
  for (const auto &val : path)
  {
    uint8_t byte1 = static_cast<uint8_t>(val & 0xFF);
    uint8_t byte2 = static_cast<uint8_t>((val >> 8) & 0xFF);
    oss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byte1);
    oss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byte2);
  }
  return oss.str();
}

std::string convertPathToHexString(uint16_t *path)
{
  std::ostringstream oss;
  for (int i = 0; i < 2008; i++)
  {
    uint16_t val = path[i];
    uint8_t byte1 = static_cast<uint8_t>(val & 0xFF);
    uint8_t byte2 = static_cast<uint8_t>((val >> 8) & 0xFF);
    oss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byte1);
    oss << std::setw(2) << std::setfill('0') << static_cast<int>(byte2);
  }
  return oss.str();
}

std::string byteArrayToHexString(const uint8_t *byteArray, size_t length)
{
  std::ostringstream oss;
  for (size_t i = 0; i < length; ++i) {
    oss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byteArray[i]);
  }
  return oss.str();
}

bool meets_target(std::string hash, std::string target)
{
  Num target_int = Num(target.c_str(), 16);
  Num hash_int = Num(hash.c_str(), 16);
  return hash_int < target_int;
}

uint32_t getLeastSignificant32Bits(uint64_t value)
{
  return static_cast<uint32_t>(value & 0xFFFFFFFF);
}

void mineShai(int tid)
{
  // ⭐ NUMA: Thread-local binding
  int numa_nodes = NUMAOptimizer::getMemoryNodes();
  if (numa_nodes > 0) {
      int node = tid % numa_nodes;
      NUMAOptimizer::setMemoryPolicy(node);
      NUMAOptimizer::printThreadBinding(tid);
  }

  byte random_buf[12];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  std::array<int, 12> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]() { return dist(gen); });
  std::memcpy(random_buf, buf.data(), buf.size());

  std::this_thread::sleep_for(std::chrono::milliseconds(125));

  int64_t localJobCounter;
  byte powHash[10000];
  byte work[ShaiHive::SHAI_DATA_SIZE];
  byte devWork[ShaiHive::SHAI_DATA_SIZE];

  ShaiHive::ShaiCtx workCtx;
  ShaiHive::ShaiCtx workCtxDev;
  uint64_t yieldCount = 0;

  // self-test block omitted…

waitForJob:

  while (!isConnected) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  srand(time(NULL));

  while (true)
  {
    try
    {
      boost::json::value myJob, myJobDev;
      {
        std::scoped_lock<std::mutex> lockGuard(mutex);
        myJob = job;
        myJobDev = devJob;
        localJobCounter = jobCounter;
      }

      hexstrToBytes(std::string(myJobDev.at("data").as_string()), workCtx.data);
      if (devConnected)
        hexstrToBytes(std::string(myJobDev.at("data").as_string()), workCtxDev.data);

      double which;
      bool devMine = false;
      bool submit = false;

      std::string target = std::string(myJob.at("target").as_string());
      std::string target_dev;
      if (devConnected) target_dev = std::string(myJobDev.at("target").as_string());

      uint8_t target_bytes[32];
      uint8_t target_bytes_dev[32];
      hexstrToBytes(target, target_bytes);
      if (devConnected) hexstrToBytes(target_dev, target_bytes_dev);

      uint8_t *TT;
      int32_t nonce = 0;

      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;

        which = (double)(rand() % 10000);
        devMine = (devConnected && which < devFee * 100.0);

        boost::json::value* mineJob = devMine ? &myJobDev : &myJob;

        byte *WORK = devMine ? &workCtxDev.data[0] : &workCtx.data[0];

        uint64_t *noncePtr = devMine ? &nonce0_dev : &nonce0;
        (*noncePtr)++;

        uint64_t N = (uint64_t)*noncePtr;
        uint32_t trueN = (N << 18) | ((tid & 511) << 10) | (rand() & 1023);
        memcpy(&WORK[76], &trueN, sizeof(trueN));

        if (ShaiHive::hash(workCtx, WORK))
        {
          cpu_counter.fetch_add(1);

          submit = devMine ? !submittingDev : !submitting;
          TT = devMine ? target_bytes_dev : target_bytes;

          if (submit && ShaiHive::checkNonce((uint32_t*)workCtx.sha, (uint32_t*)TT))
          {
            uint32_t sN = __builtin_bswap32(trueN);
            std::string job_id = std::string(mineJob->at("job_id").as_string());
            std::string pathHex = convertPathToHexString(workCtx.path);

            if (devMine)
            {
              submittingDev = true;
              submitTracker.pushSoloDevice(-1, true);
              devShare = { {"type","submit"},{"miner_id",devMiningProfile.wallet.c_str()},
                        {"nonce",uint32ToHex(sN).c_str()},
                        {"job_id",job_id},
                        {"path",pathHex.c_str()} };
              data_ready = true;
            }
            else
            {
              submitting = true;
              submitTracker.pushSoloDevice(-1, false);
              share = { {"type","submit"},{"miner_id",miningProfile.wallet.c_str()},
                        {"nonce",uint32ToHex(sN).c_str()},
                        {"job_id",job_id},
                        {"path",pathHex.c_str()} };
              data_ready = true;
            }
            cv.notify_all();
          }
        }

        if (!isConnected) break;
        if ((++yieldCount & 127) == 0)
          std::this_thread::yield();
      }
      if (!isConnected) break;

    } catch (std::exception &e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function\n" << e.what() << std::endl;
      setcolor(BRIGHT_WHITE);
      localJobCounter = -1;
    }

    if (!isConnected) break;
  }

  // ⭐ NUMA: Restore policy when exiting thread
  NUMAOptimizer::restoreMemoryPolicy();

  goto waitForJob;
}
