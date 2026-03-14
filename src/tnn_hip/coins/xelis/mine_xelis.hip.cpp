#include <coins/miners.hpp>
#include <net/net.hpp>
#include "tnn-hugepages.hpp"
#include <stratum/stratum.h>
#include <base64.hpp>

#include "../../common/gpu_miner.hpp"
#include "../../common/hip_algo_registry.hpp"
#include "../../common/gpu_submit_queue.hpp"
#include "../../common/tnn_log.hpp"
#include <algo_definitions.h>

#include "mine_xelis.hip.h"
#include <crypto/xelis-hash/xelis-hash.hpp>

// ============================================================================
// Xelis solution builder — called from GPU thread, must be non-blocking.
// Does CPU verification + payload construction, returns entry for queue.
// ============================================================================
static std::optional<GPUSubmitEntry> xelis_build_solution(
    const uint8_t *hash, uint64_t nonce, int gpu_id,
    const JobSnapshot& job_snapshot, bool devMine)
{
    // Build the complete work template with the winning nonce
    byte finalWork[XELIS_TEMPLATE_SIZE];
    memcpy(finalWork, job_snapshot.work_template.data(), XELIS_TEMPLATE_SIZE);

    // Insert nonce at correct position (bytes 40-47, little-endian)
    for (int i = 0; i < 8; i++) {
      finalWork[40 + i] = (nonce >> (i * 8)) & 0xFF;
    }

    // CPU verification before submission (detect hash computation errors)
    {
      static thread_local uint64_t *cpu_scratch = nullptr;
      if (!cpu_scratch) {
        cpu_scratch = new uint64_t[XELIS_MEMORY_SIZE_V3];
      }

      xelis_stage1_v3(finalWork, cpu_scratch, 112);
      xelis_stage3_v3(cpu_scratch);

      byte cpu_hash[32];
      xelis_blake3_v3((uint8_t*)cpu_scratch, cpu_hash);

      if (memcmp(cpu_hash, hash, 32) != 0) {
        printf("\033[31m[ERROR] GPU/CPU hash mismatch! Same input, different output.\033[0m\n");
        uint64_t device_id_extracted = (nonce >> 59) & 0x1F;
        uint64_t random_extracted = (nonce >> 48) & 0x7FF;
        uint64_t counter_extracted = nonce & 0xFFFFFFFFFFFFULL;
        printf("  GPU %d, Nonce: 0x%016llx (device=%llu, random=%llu, counter=%llu)\n",
               gpu_id, (unsigned long long)nonce,
               (unsigned long long)device_id_extracted,
               (unsigned long long)random_extracted,
               (unsigned long long)counter_extracted);
        printf("  Work template (first 48 bytes):\n    ");
        for (int i = 0; i < 48; i++) {
          printf("%02x", finalWork[i]);
          if ((i+1) % 32 == 0) printf("\n    ");
        }
        printf("\n  Nonce at bytes 40-47: ");
        for (int i = 40; i < 48; i++) printf("%02x", finalWork[i]);
        printf("\n  GPU hash: %s\n", hexStr(hash, 32).c_str());
        printf("  CPU hash: %s\n", hexStr(cpu_hash, 32).c_str());
        printf("  \033[31mNOT SUBMITTING - hash computation error!\033[0m\n");
        fflush(stdout);
        return std::nullopt;
      }
    }

    // Hash reversal for Xelis canonical format
    byte hash_reversed[32];
    memcpy(hash_reversed, hash, 32);
    std::reverse(hash_reversed, hash_reversed + 32);

    // For XATUM protocol, reverse back to raw format
    byte hash_for_submit[32];
    memcpy(hash_for_submit, hash_reversed, 32);
    auto& profile = devMine ? devMiningProfile : miningProfile;
    if (profile.protocol == PROTO_XELIS_XATUM && littleEndian()) {
      std::reverse(hash_for_submit, hash_for_submit + 32);
    }

    std::string b64 = base64::to_base64(std::string((char *)finalWork, XELIS_TEMPLATE_SIZE));

    // Build the protocol-specific payload
    // NOTE: All job metadata comes from job_snapshot (captured at batch start),
    // NOT from the live job/devJob globals which may have moved to a new height.
    boost::json::object payload;
    switch (profile.protocol)
    {
    case PROTO_XELIS_SOLO:
      submitTracker.pushSoloDevice(gpu_id, devMine);
      payload = {{"block_template", hexStr(finalWork, XELIS_TEMPLATE_SIZE).c_str()}};
      break;
    case PROTO_XELIS_XATUM:
      submitTracker.pushSoloDevice(gpu_id, devMine);
      payload = {
          {"data", b64.c_str()},
          {"hash", hexStr(hash_for_submit, 32).c_str()},
      };
      break;
    case PROTO_XELIS_STRATUM:
      payload = {{{"rpc_id", submitTracker.nextId(gpu_id)},
                   {"method", XelisStratum::submit.method.c_str()},
                   {"params", {devMine ? devWorkerName : workerName,
                               job_snapshot.job_id_str.c_str(),
                               hexStr((byte *)&nonce, 8).c_str()}}}};
      break;
    }

    return GPUSubmitEntry{std::move(payload), devMine, job_snapshot.job_id};
}

void mineXelis_hip(int tid)
{
  TNN_LOG_TRACE("[TRACE] mineXelis_hip: Entry, tid=%d\n", tid);

  // Bind CUDA context to this thread before any other HIP calls
  hipSetDevice(0);

  std::vector<std::unique_ptr<GPUMiner>> miners;
  int gpuCount;
  hipGetDeviceCount(&gpuCount);

  TNN_LOG_TRACE("[TRACE] mineXelis_hip: Found %d GPU(s)\n", gpuCount);

  // Determine algorithm version
  std::string algo_name = (miningProfile.coin.miningAlgo == ALGO_XELISV3) ? "xelis_v3" : "xelis_v2";
  TNN_LOG_TRACE("[TRACE] mineXelis_hip: Using algorithm '%s'\n", algo_name.c_str());

  // Initialize one miner per GPU
  for (int d = 0; d < gpuCount; d++)
  {
    TNN_LOG_TRACE("[TRACE] mineXelis_hip: Initializing GPU %d...\n", d);

    try
    {
      TNN_LOG_TRACE("[TRACE] mineXelis_hip: Creating GPUMiner for device %d\n", d);

      auto miner = std::make_unique<GPUMiner>(algo_name, d);

      TNN_LOG_TRACE("[TRACE] mineXelis_hip: GPUMiner created, calling initialize()\n");

      if (!miner->initialize())
      {
        setcolor(RED);
        std::cerr << "Failed to initialize GPU " << d << " for Xelis mining\n";
        setcolor(BRIGHT_WHITE);
        continue;
      }

      TNN_LOG_TRACE("[TRACE] mineXelis_hip: GPU %d initialized successfully\n", d);

      miners.push_back(std::move(miner));
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      std::cerr << "GPU " << d << " init error: " << e.what() << "\n";
      setcolor(BRIGHT_WHITE);
    }
  }

  TNN_LOG_TRACE("[TRACE] mineXelis_hip: Initialized %zu GPU miner(s)\n", miners.size());

  if (miners.empty())
  {
    setcolor(RED);
    std::cerr << "No GPUs available for mining\n";
    setcolor(BRIGHT_WHITE);
    return;
  }

  TNN_LOG_INFO("[INFO] All GPUs initialized, ready to start mining\n");

  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint64_t i = 0;
  uint64_t i_dev = 0;

  byte work[XELIS_TEMPLATE_SIZE] = {0};
  byte devWork[XELIS_TEMPLATE_SIZE] = {0};

  // Track job heights for stale detection in submit queue
  std::atomic<int64_t> current_job_height{0};
  std::atomic<int64_t> current_dev_job_height{0};

  uint64_t current_difficulty = 0;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(0, 10000);

  TNN_LOG_TRACE("[TRACE] mineXelis_hip: Setup locals\n");

  bool solo_mode = (miningProfile.protocol == PROTO_XELIS_SOLO);
  bool miners_started = false;

waitForJob:
  // (Re)start submit queue — safe to call after stop or on first entry
  GPUSubmitQueue::instance().start(
      &share, &devShare,
      &submitting, &submittingDev,
      &data_ready, &cv,
      [&, solo_mode](int64_t job_id, bool is_dev) -> bool {
        int64_t current = is_dev ? current_dev_job_height.load() : current_job_height.load();
        if (solo_mode)
          return job_id == current;
        return job_id >= (current - 2) && job_id <= current;
      }
  );
  while (!isConnected)
  {
    CHECK_CLOSE;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  while (!ABORT_MINER)
  {
    try
    {
      boost::json::value myJob;
      boost::json::value myJobDev;

      myJob = job;
      myJobDev = devJob;

      if (!myJob.is_object() || !myJob.as_object().contains("miner_work") || !myJob.at("miner_work").is_string())
      {
        continue;
      }
      if (ourHeight == 0 && devHeight == 0)
        continue;

      // Update main work
      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        // Update current job height (for stale detection)
        current_job_height.store(ourHeight);

        TNN_LOG_TRACE("[TRACE] Decoding work template\n");
        byte *b2 = new byte[XELIS_TEMPLATE_SIZE];
        switch (miningProfile.protocol)
        {
        case PROTO_XELIS_SOLO:
          hexstrToBytes(std::string(myJob.at("miner_work").as_string()), b2);
          break;
        case PROTO_XELIS_XATUM:
        {
          std::string b64 = base64::from_base64(
              std::string(myJob.at("miner_work").as_string().c_str()));
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

        // Update all GPU miners with new work
        current_difficulty = difficulty;
        std::string stratum_job_id;
        if (miningProfile.protocol == PROTO_XELIS_STRATUM && myJob.as_object().contains("jobId")) {
          stratum_job_id = std::string(myJob.at("jobId").as_string());
        }
        for (auto &miner : miners)
        {
          TNN_LOG_TRACE("[TRACE] Calling set_work on miner device %d\n", miner->get_device_id());
          miner->set_work(work, difficulty);
          miner->set_job_id(ourHeight, stratum_job_id);
        }

        // Start miners on first job (ensures full initialization before mining begins)
        if (!miners_started)
        {
          TNN_LOG_INFO("[INFO] Starting all GPU miners (first job received)\n");

          for (auto &miner : miners)
          {
            miner->set_dev_fee(devFee);
            miner->start([&](const uint8_t *hash, uint64_t nonce, int gpu_id, const JobSnapshot& job_snapshot)
                -> std::optional<GPUSubmitEntry>
            {
                printf("\n");
                if (job_snapshot.is_dev) {
                  setcolor(CYAN);
                  printf("DEV | ");
                } else {
                  setcolor(BRIGHT_YELLOW);
                }
                printf("GPU #%d found a nonce: %llu\n", gpu_id, (unsigned long long)nonce);
                fflush(stdout);
                setcolor(BRIGHT_WHITE);

                return xelis_build_solution(hash, nonce, gpu_id, job_snapshot, job_snapshot.is_dev);
            });
          }

          miners_started = true;
          TNN_LOG_INFO("[INFO] All GPU miners started successfully\n");
        }

        TNN_LOG_TRACE("[TRACE] All miners updated, continuing\n");
      }

      // Update dev work if needed
      if (devConnected && myJobDev.is_object() && myJobDev.as_object().contains("miner_work") && myJobDev.at("miner_work").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          current_dev_job_height.store(devHeight);

          byte *b2d = new byte[XELIS_TEMPLATE_SIZE];
          switch (devMiningProfile.protocol)
          {
          case PROTO_XELIS_SOLO:
            hexstrToBytes(std::string(myJobDev.at("miner_work").as_string()), b2d);
            break;
          case PROTO_XELIS_XATUM:
          {
            std::string b64 = base64::from_base64(
                std::string(myJobDev.at("miner_work").as_string().c_str()));
            memcpy(b2d, b64.data(), b64.size());
            break;
          }
          case PROTO_XELIS_STRATUM:
            hexstrToBytes(std::string(myJobDev.at("miner_work").as_string()), b2d);
            break;
          }
          memcpy(devWork, b2d, XELIS_TEMPLATE_SIZE);
          delete[] b2d;

          std::string dev_stratum_job_id;
          if (devMiningProfile.protocol == PROTO_XELIS_STRATUM && myJobDev.as_object().contains("jobId")) {
            dev_stratum_job_id = std::string(myJobDev.at("jobId").as_string());
          }
          for (auto &miner : miners)
          {
            miner->set_dev_work(devWork, difficulty);
            miner->set_dev_job_id(devHeight, dev_stratum_job_id);
          }

          localDevHeight = devHeight;
          i_dev = 0;
        }
      }

      // Check if disconnected
      if (!isConnected)
      {
        TNN_LOG_TRACE("[TRACE] Loop break: disconnected\n");
        break;
      }

      std::this_thread::yield();
    }
    catch (std::exception &e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);
      localOurHeight = -1;
      localDevHeight = -1;
    }

    if (!isConnected)
    {
      data_ready = true;
      cv.notify_all();
      break;
    }
  }

  // Stop all miners, then stop the submit queue
  for (auto &miner : miners)
  {
    miner->stop();
  }
  GPUSubmitQueue::instance().stop();

  if (!isConnected)
  {
    miners_started = false;
    localOurHeight = 0;
    localDevHeight = 0;
    goto waitForJob;
  }
}
