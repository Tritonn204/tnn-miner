#include <coins/miners.hpp>
#include <net/net.hpp>
#include "tnn-hugepages.hpp"
#include <stratum/stratum.h>
#include <base64.hpp>

#include "../../common/gpu_compat.hpp"
#include "../../common/gpu_miner.hpp"
#include "../../common/hip_algo_registry.hpp"
#include "../../common/gpu_submit_queue.hpp"
#include <job_safe.hpp>
#include "../../common/tnn_log.hpp"
#include <algo_definitions.h>
#include <crypto/qhash/qhash.hpp>

#include <thread>

// ============================================================================
// QHash solution builder (QubitCoin) — BTC stratum
// ============================================================================
// Called from GPU thread, must be non-blocking.
// Does CPU verification + payload construction, returns entry for queue.
// ============================================================================
static std::optional<GPUSubmitEntry> qhash_build_solution(
    const uint8_t *hash, uint64_t nonce, int gpu_id,
    const JobSnapshot& job_snapshot, bool devMine)
{
    // Build the complete block header with the winning nonce
    uint8_t final_work[QHASH_INPUT_SIZE];
    std::memcpy(final_work, job_snapshot.work_template.data(), QHASH_INPUT_SIZE);

    // Insert nonce at bytes 76-79 (4-byte LE nonce)
    std::memcpy(final_work + 76, &nonce, 4);

    // CPU verification before submission
    {
        static thread_local QHashWorkspace* cpu_ws = nullptr;
        if (!cpu_ws) cpu_ws = new QHashWorkspace;

        uint8_t cpu_hash[32];
        qhash_compute(final_work, cpu_hash, cpu_ws);

        if (std::memcmp(cpu_hash, hash, 32) != 0) {
            TNN_LOG_ERROR("[ERROR] GPU/CPU hash mismatch! GPU %d nonce 0x%llx\n",
                          gpu_id, (unsigned long long)nonce);
            return std::nullopt;
        }
    }

    // BTC stratum submit: [worker, job_id, extranonce2, ntime, nonce]
    boost::json::object payload = {
        {"id", 0},
        {"method", "mining.submit"},
        {"params", boost::json::array{
            devMine ? tnn::devWorkerName : tnn::workerName,
            job_snapshot.job_id_str,
            "",  // extranonce2 (empty for now)
            "",  // ntime (empty for now)
            hexStr((byte*)&nonce, 4).c_str()
        }}
    };

    return GPUSubmitEntry{std::move(payload), devMine, job_snapshot.job_id};
}

// ============================================================================
// QubitCoin GPU Mining Entry Point
// ============================================================================
void mineQubit_hip(int tid)
{
    TNN_LOG_TRACE("[TRACE] mineQubit_hip: Entry, tid=%d\n", tid);

    std::vector<std::unique_ptr<GPUMiner>> miners;
    int gpuCount;
    (void)oroGetDeviceCount(&gpuCount);

    TNN_LOG_TRACE("[TRACE] mineQubit_hip: Found %d GPU(s)\n", gpuCount);

    const std::string algo_name = "qhash";

    // Initialize GPUs in parallel
    {
        std::vector<std::thread> init_threads;
        std::vector<std::unique_ptr<GPUMiner>> per_gpu(gpuCount);
        std::vector<bool> gpu_ok(gpuCount, false);

        for (int d = 0; d < gpuCount; d++)
        {
            if (!shouldUseDevice(d)) continue;

            init_threads.emplace_back([&, d]() {
                TNN_LOG_TRACE("[TRACE] mineQubit_hip: Initializing GPU %d...\n", d);
                try
                {
                    auto miner = std::make_unique<GPUMiner>(algo_name, d);
                    if (miner->initialize())
                    {
                        TNN_LOG_TRACE("[TRACE] mineQubit_hip: GPU %d initialized successfully\n", d);
                        per_gpu[d] = std::move(miner);
                        gpu_ok[d] = true;
                    }
                    else
                    {
                        setcolor(RED);
                        std::cerr << "Failed to initialize GPU " << d << " for QHash mining\n";
                        setcolor(BRIGHT_WHITE);
                    }
                }
                catch (const std::exception &e)
                {
                    setcolor(RED);
                    std::cerr << "GPU " << d << " init error: " << e.what() << "\n";
                    setcolor(BRIGHT_WHITE);
                }
            });
        }

        for (auto& t : init_threads) t.join();

        for (int d = 0; d < gpuCount; d++)
        {
            if (gpu_ok[d])
                miners.push_back(std::move(per_gpu[d]));
        }
    }

    TNN_LOG_TRACE("[TRACE] mineQubit_hip: Initialized %zu GPU miner(s)\n", miners.size());

    if (miners.empty())
    {
        setcolor(RED);
        std::cerr << "No GPUs available for QHash mining\n";
        setcolor(BRIGHT_WHITE);
        return;
    }

    TNN_LOG_INFO_COLOR(BRIGHT_YELLOW, "[INFO] All GPUs initialized, ready to start QHash mining\n");

    int64_t localOurHeight = 0;
    int64_t localDevHeight = 0;

    uint64_t i = 0;
    uint64_t i_dev = 0;

    // TODO: Adjust template size for QubitCoin
    byte work[QHASH_INPUT_SIZE] = {0};
    byte devWork[QHASH_INPUT_SIZE] = {0};

    std::atomic<int64_t> current_job_height{0};
    std::atomic<int64_t> current_dev_job_height{0};

    uint64_t current_difficulty = 0;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0, 10000);

    TNN_LOG_TRACE("[TRACE] mineQubit_hip: Setup locals\n");

    bool miners_started = false;

waitForJob:
    // Start submit queue
    GPUSubmitQueue::instance().start(
        &share, &devShare,
        &submitting, &submittingDev,
        &data_ready, &cv,
        [&](int64_t job_id, bool is_dev) -> bool {
            int64_t current = is_dev ? current_dev_job_height.load() : current_job_height.load();
            return job_id >= (current - 2) && job_id <= current;
        }
    );

    // Main mining loop
    while (!ABORT_MINER)
    {
        // TODO: Parse stratum jobs and feed work to miners
        // This loop should:
        // 1. Receive new jobs from stratum
        // 2. Call miner->set_work() / miner->set_dev_work()
        // 3. Call miner->start() with qhash_build_solution callback

        // Placeholder sleep to avoid busy-wait
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Shutdown
    for (auto& miner : miners)
    {
        miner->stop();
    }
    GPUSubmitQueue::instance().stop();
}
