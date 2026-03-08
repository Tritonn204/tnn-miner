#include <coins/miners.hpp>
#include <net/net.hpp>
#include "tnn-hugepages.hpp"
#include <stratum/stratum.h>
#include <base64.hpp>

#include "../../common/gpu_miner.hpp"
#include "../../common/hip_algo_registry.hpp"
#include <algo_definitions.h>

#include "mine_xelis.hip.h"
#include <crypto/xelis-hash/xelis-hash.hpp>

void mineXelis_hip(int tid)
{
  printf("[TRACE] mineXelis_hip: Entry, tid=%d\n", tid);
  fflush(stdout);

  // Initialize GPU miner for all devices
  std::vector<std::unique_ptr<GPUMiner>> miners;
  int gpuCount;
  hipGetDeviceCount(&gpuCount);

  printf("[TRACE] mineXelis_hip: Found %d GPU(s)\n", gpuCount);
  fflush(stdout);

  // Determine algorithm version
  std::string algo_name = (miningProfile.coin.miningAlgo == ALGO_XELISV3) ? "xelis_v3" : "xelis_v2";
  printf("[TRACE] mineXelis_hip: Using algorithm '%s'\n", algo_name.c_str());
  fflush(stdout);

  // Initialize one miner per GPU
  for (int d = 0; d < gpuCount; d++)
  {
    printf("[TRACE] mineXelis_hip: Initializing GPU %d...\n", d);
    fflush(stdout);

    try
    {
      printf("[TRACE] mineXelis_hip: Creating GPUMiner for device %d\n", d);
      fflush(stdout);

      auto miner = std::make_unique<GPUMiner>(algo_name, d);

      printf("[TRACE] mineXelis_hip: GPUMiner created, calling initialize()\n");
      fflush(stdout);

      if (!miner->initialize())
      {
        setcolor(RED);
        std::cerr << "Failed to initialize GPU " << d << " for Xelis mining\n";
        setcolor(BRIGHT_WHITE);
        continue;
      }

      printf("[TRACE] mineXelis_hip: GPU %d initialized successfully\n", d);
      fflush(stdout);

      miners.push_back(std::move(miner));
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      std::cerr << "GPU " << d << " init error: " << e.what() << "\n";
      setcolor(BRIGHT_WHITE);
    }
  }

  printf("[TRACE] mineXelis_hip: Initialized %zu GPU miner(s)\n", miners.size());
  fflush(stdout);

  if (miners.empty())
  {
    setcolor(RED);
    std::cerr << "No GPUs available for mining\n";
    setcolor(BRIGHT_WHITE);
    return;
  }

  printf("[INFO] All GPUs initialized, ready to start mining\n");
  fflush(stdout);

  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;

  uint64_t i = 0;
  uint64_t i_dev = 0;

  byte work[XELIS_TEMPLATE_SIZE] = {0};
  byte devWork[XELIS_TEMPLATE_SIZE] = {0};

  // Saved work templates for solution submission (prevents race with job updates)
  byte saved_work[XELIS_TEMPLATE_SIZE] = {0};
  byte saved_devWork[XELIS_TEMPLATE_SIZE] = {0};
  std::mutex saved_work_mutex; // Protects saved_work, saved_devWork, saved_*_job_height

  // Track job heights to detect stale solutions
  std::atomic<int64_t> current_job_height{0};
  std::atomic<int64_t> current_dev_job_height{0};
  int64_t saved_job_height = 0;     // Protected by saved_work_mutex
  int64_t saved_dev_job_height = 0; // Protected by saved_work_mutex

  // Track if we've submitted a solution for current job (SOLO mode only)
  std::atomic<int64_t> last_submitted_height{-1};
  std::atomic<int64_t> last_submitted_dev_height{-1};

  uint64_t current_difficulty = 0;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(0, 10000);
  std::uniform_real_distribution<double> n2(0, 256);

  printf("[TRACE] mineXelis_hip: Setup locals\n");
  fflush(stdout);

  // Solution callback - receives RAW hash from GPU (already difficulty-checked)
  auto on_solution = [&](const uint8_t *hash, uint64_t nonce, int gpu_id, bool devMine, const JobSnapshot& job_snapshot)
  {
    // JobSnapshot contains copies of template, job_id, difficulty captured before batch execution
    // All data is guaranteed to match what the GPU processed (safe from concurrent job updates)
    // Let the pool decide if solution is stale - don't pre-filter here

    // // In SOLO mode, skip if we already submitted a solution for this job
    // // (prevents multiple submissions from same batch competing)
    // auto &protocol = devMine ? devMiningProfile.protocol : miningProfile.protocol;
    // if (protocol == PROTO_XELIS_SOLO)
    // {
    //   int64_t our_height = devMine ? devHeight : ourHeight;
    //   if (our_height > solution_height)
    //   {
    //     // Already moved to next job, solution is stale
    //     printf("[DEBUG] Skipping stale SOLO solution (height %ld, current %ld)\n", solution_height, our_height);
    //     fflush(stdout);
    //     return;
    //   }
    // }

    // For Xelis: reverse hash to canonical format for submission
    byte hash_reversed[32];
    memcpy(hash_reversed, hash, 32);
    std::reverse(hash_reversed, hash_reversed + 32);

    printf("\n");
    if (devMine)
    {
      setcolor(CYAN);
      printf("DEV | ");
    }
    else
    {
      setcolor(BRIGHT_YELLOW);
    }
    printf("GPU #%d found a nonce: %llu\n", gpu_id, nonce);
    fflush(stdout);
    setcolor(BRIGHT_WHITE);

    // Wait for previous submission to complete before submitting new one
    bool submit = (devMine && devConnected) ? !submittingDev : !submitting;

    if (!submit) {
        // Wait for previous submission with timeout
        int wait_count = 0;
        const int MAX_WAIT = 5000; // 5 second max wait (network can be slow)
        for(;;) {
            submit = (devMine && devConnected) ? !submittingDev : !submitting;

            // Check if job changed (solution became stale)
            int64_t current_height = devMine ? current_dev_job_height.load() : current_job_height.load();
            if (submit || job_snapshot.job_id != current_height) {
                if (job_snapshot.job_id != current_height) {
                    printf("[INFO] Job changed during submission wait (height %ld -> %ld), aborting solution\n",
                           job_snapshot.job_id, current_height);
                    fflush(stdout);
                }
                break;
            }

            if (++wait_count > MAX_WAIT) {
                printf("[WARN] Submission timeout after 5s, dropping solution to prevent GPU stall\n");
                fflush(stdout);
                return; // Drop this solution to keep GPU mining
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // If job changed, don't submit stale solution
        if (job_snapshot.job_id != (devMine ? current_dev_job_height.load() : current_job_height.load())) {
            return;
        }
    }

    // Build the complete work template with the winning nonce
    // Use the work template from JobSnapshot (captured before mine_batch() call)
    // This guarantees it matches exactly what the GPU processed
    byte finalWork[XELIS_TEMPLATE_SIZE];
    memcpy(finalWork, job_snapshot.work_template.data(), XELIS_TEMPLATE_SIZE);

    // Insert nonce at correct position (bytes 40-47, little-endian)
    for (int i = 0; i < 8; i++) {
      finalWork[40 + i] = (nonce >> (i * 8)) & 0xFF;
    }

    // CPU stage-by-stage verification before submission (detect hash computation errors)
    {
      static thread_local uint64_t *cpu_scratch = nullptr;
      if (!cpu_scratch) {
        cpu_scratch = new uint64_t[XELIS_MEMORY_SIZE_V3];
      }

      // Run CPU stages
      xelis_stage1_v3(finalWork, cpu_scratch, 112);

      // Optional: Checksum after stage1 for debugging
      // uint64_t cpu_s1_checksum = 0;
      // for (size_t i = 0; i < XELIS_MEMORY_SIZE_V3; i++) cpu_s1_checksum ^= cpu_scratch[i];
      // printf("  CPU Stage1 checksum: 0x%016llx\n", cpu_s1_checksum);

      xelis_stage3_v3(cpu_scratch);

      // Optional: Checksum after stage3 for debugging
      // uint64_t cpu_s3_checksum = 0;
      // for (size_t i = 0; i < XELIS_MEMORY_SIZE_V3; i++) cpu_s3_checksum ^= cpu_scratch[i];
      // printf("  CPU Stage3 checksum: 0x%016llx\n", cpu_s3_checksum);

      byte cpu_hash[32];
      xelis_blake3_v3((uint8_t*)cpu_scratch, cpu_hash);

      // GPU hash is raw blake3 output (not reversed)
      byte gpu_hash_raw[32];
      memcpy(gpu_hash_raw, hash, 32);  // Use original 'hash' parameter, NOT hash_reversed

      if (memcmp(cpu_hash, gpu_hash_raw, 32) != 0) {
        printf("\033[31m[ERROR] GPU/CPU hash mismatch! Same input, different output.\033[0m\n");
        // Extract nonce fields using NEW format: bits 59-63=device, 48-58=random, 0-47=counter
        uint64_t device_id_extracted = (nonce >> 59) & 0x1F;        // Top 5 bits
        uint64_t random_extracted = (nonce >> 48) & 0x7FF;          // Next 11 bits
        uint64_t counter_extracted = nonce & 0xFFFFFFFFFFFFULL;     // Bottom 48 bits
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
        printf("\n  GPU hash: %s\n", hexStr(gpu_hash_raw, 32).c_str());
        printf("  CPU hash: %s\n", hexStr(cpu_hash, 32).c_str());
        printf("  \033[31mNOT SUBMITTING - hash computation error!\033[0m\n");
        fflush(stdout);
        return;  // Don't submit invalid hash
      }
    }

    // Hash is currently reversed (Xelis canonical format)
    // For XATUM protocol, reverse it back to raw format (matches CPU behavior line 161-164)
    byte hash_for_submit[32];
    memcpy(hash_for_submit, hash_reversed, 32);
    if ((devMine ? devMiningProfile.protocol : miningProfile.protocol) == PROTO_XELIS_XATUM && littleEndian())
    {
      std::reverse(hash_for_submit, hash_for_submit + 32);
    }

    boost::json::value &myJob = devMine ? devJob : job;
    std::string b64 = base64::to_base64(std::string((char *)finalWork, XELIS_TEMPLATE_SIZE));

    {
      if (devMine)
      {
        submittingDev = true;
        switch (devMiningProfile.protocol)
        {
        case PROTO_XELIS_SOLO:
          devShare = {{"block_template", hexStr(finalWork, XELIS_TEMPLATE_SIZE).c_str()}};
          break;
        case PROTO_XELIS_XATUM:
          devShare = {
              {"data", b64.c_str()},
              {"hash", hexStr(hash_for_submit, 32).c_str()},
          };
          break;
        case PROTO_XELIS_STRATUM:
          devShare = {{{"id", XelisStratum::submitID},
                       {"method", XelisStratum::submit.method.c_str()},
                       {"params", {devWorkerName, myJob.at("jobId").as_string().c_str(), hexStr((byte *)&nonce, 8).c_str()}}}};
          break;
        }
        data_ready = true;
      }
      else
      {
        submitting = true;
        switch (miningProfile.protocol)
        {
        case PROTO_XELIS_SOLO:
          share = {{"block_template", hexStr(finalWork, XELIS_TEMPLATE_SIZE).c_str()}};
          break;
        case PROTO_XELIS_XATUM:
          share = {
              {"data", b64.c_str()},
              {"hash", hexStr(hash_for_submit, 32).c_str()},
          };
          break;
        case PROTO_XELIS_STRATUM:
          share = {{{"id", XelisStratum::submitID},
                    {"method", XelisStratum::submit.method.c_str()},
                    {"params", {workerName, myJob.at("jobId").as_string().c_str(), hexStr((byte *)&nonce, 8).c_str()}}}};
          break;
        }
        data_ready = true;
      }

      cv.notify_all();
    }
  };

  printf("[TRACE] on_solution defined\n");
  fflush(stdout);

  // Track if miners have been started
  std::atomic<bool> miners_started{false};

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
      // printf("[TRACE] Top of main loop\n"); fflush(stdout);
      boost::json::value myJob;
      boost::json::value myJobDev;

      myJob = job;
      myJobDev = devJob;

      if (!myJob.at("miner_work").is_string())
      {
        printf("NO WORK\n");
        fflush(stdout);
        continue;
      }
      if (ourHeight == 0 && devHeight == 0)
        continue;

      // Update main work
      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        printf("[INFO] New job received (height: %ld)\n", ourHeight);
        fflush(stdout);

        printf("[TRACE] Step 1: Updating job height atomic\n");
        fflush(stdout);
        // Update current job height (for stale detection)
        current_job_height.store(ourHeight);

        printf("[TRACE] Step 2: Decoding work template\n");
        fflush(stdout);
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

        printf("[TRACE] Step 3: Copying to work buffer\n");
        fflush(stdout);
        memcpy(work, b2, XELIS_TEMPLATE_SIZE);
        delete[] b2;

        localOurHeight = ourHeight;
        i = 0;

        printf("[TRACE] Step 4: Saving work snapshot\n");
        fflush(stdout);
        // Save work template snapshot before mining (prevents race with job updates)
        {
          std::lock_guard<std::mutex> lock(saved_work_mutex);
          memcpy(saved_work, work, XELIS_TEMPLATE_SIZE);
          saved_job_height = ourHeight;
        }
        printf("[DEBUG] Saved work template snapshot for height %ld\n", ourHeight);
        fflush(stdout);

        printf("[TRACE] Step 5: Calling set_work on %zu miners\n", miners.size());
        fflush(stdout);
        // Update all GPU miners with new work
        current_difficulty = difficulty;
        for (auto &miner : miners)
        {
          printf("[TRACE]   Calling set_work on miner device %d\n", miner->get_device_id());
          fflush(stdout);
          miner->set_work(work, difficulty);
          miner->set_job_id(ourHeight);  // Capture job height alongside template
          printf("[TRACE]   set_work returned\n");
          fflush(stdout);
        }

        // Start miners on first job (ensures full initialization before mining begins)
        if (!miners_started.load())
        {
          printf("[INFO] Starting all GPU miners (first job received)\n");
          fflush(stdout);

          for (auto &miner : miners)
          {
            int gpu_id = miner->get_device_id();
            miner->start([&, gpu_id](const uint8_t *hash, uint64_t nonce, const JobSnapshot& job_snapshot)
                         {
                // Determine if this is a dev share based on current mining state
                double which = dist(rng);
                bool devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);

                // JobSnapshot contains all job state captured before batch execution
                // (template, job_id, difficulty) - all guaranteed to match what GPU processed
                on_solution(hash, nonce, gpu_id, devMine, job_snapshot); });
          }

          miners_started.store(true);
          printf("[INFO] All GPU miners started successfully\n");
          fflush(stdout);
        }

        printf("[TRACE] Step 6: All miners updated, continuing\n");
        fflush(stdout);
      }

      // Update dev work if needed
      if (devConnected && myJobDev.at("miner_work").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          printf("[INFO] New dev job received (height: %ld)\n", devHeight);
          fflush(stdout);

          // Update current dev job height (for stale detection)
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

          // Save dev work template snapshot
          {
            std::lock_guard<std::mutex> lock(saved_work_mutex);
            memcpy(saved_devWork, devWork, XELIS_TEMPLATE_SIZE);
            saved_dev_job_height = devHeight;
          }

          localDevHeight = devHeight;
          i_dev = 0;
        }
      }

      // Check if disconnected (job changes are handled at top of loop)
      if (!isConnected)
      {
        printf("[TRACE] Loop break: disconnected\n");
        fflush(stdout);
        break;
      }

      // printf("[TRACE] End of loop iteration, sleeping 10ms\n"); fflush(stdout);
      std::this_thread::yield();
    }
    catch (std::exception &e)
    {
      setcolor(RED);
      std::cerr << "Error in GPU POW Function: " << e.what() << "\n";
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

  // Stop all miners
  for (auto &miner : miners)
  {
    miner->stop();
  }

  if (!isConnected)
  {
    goto waitForJob;
  }
}
