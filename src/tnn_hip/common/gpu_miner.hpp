#pragma once
#include "hip_algo_registry.hpp"
#include "gpu_submit_queue.hpp"
#include "tnn_log.hpp"
#include "../../coins/miners.hpp"
#include <atomic>
#include <thread>
#include <mutex>
#include <random>
#include <optional>

// Callback type for algo-specific solution processing.
// Receives (hash, nonce, gpu_id, job_snapshot) and returns a GPUSubmitEntry
// to queue for submission, or std::nullopt to skip (e.g. CPU verification failed).
using SolutionBuilder = std::function<
    std::optional<GPUSubmitEntry>(const uint8_t* hash, uint64_t nonce, int gpu_id, const JobSnapshot& job_snapshot)
>;

class GPUMiner {
public:
    GPUMiner(const std::string& algo_name, int device_id = 0)
        : device_id_(device_id), running_(false) {

        TNN_LOG_TRACE("[TRACE] GPUMiner: Constructor for algo='%s', device=%d\n", algo_name.c_str(), device_id);

        algo_ = AlgoRegistry::instance().create(algo_name);
        if (!algo_) {
            throw std::runtime_error("Unknown algorithm: " + algo_name);
        }

        TNN_LOG_TRACE("[TRACE] GPUMiner: Algorithm instance created\n");
    }

    bool initialize() {
        TNN_LOG_TRACE("[TRACE] GPUMiner: Calling algo_->initialize()\n");

        bool result = algo_->initialize(device_id_);

        TNN_LOG_TRACE("[TRACE] GPUMiner: algo_->initialize() returned %s\n", result ? "true" : "false");

        return result;
    }

    void set_work(const uint8_t* work_template, uint64_t difficulty) {
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            memcpy(current_work_, work_template, algo_->get_config().template_size);
        }
        current_difficulty_.store(difficulty, std::memory_order_release);
        work_updated_.store(true, std::memory_order_release);

        static bool logged_template = false;
        if (!logged_template) {
            if (tnn_log_enabled(TnnLogLevel::Debug)) {
                printf("\n[DEBUG] Work template set (GPU %d):\n  ", device_id_);
                for (int i = 0; i < 112; i++) {
                    printf("%02x", work_template[i]);
                    if ((i+1) % 32 == 0) printf("\n  ");
                }
                printf("\n  Difficulty: %llu\n", (unsigned long long)difficulty);
                fflush(stdout);
            }
            logged_template = true;
        }
    }

    void set_dev_work(const uint8_t* work_template, uint64_t difficulty) {
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            memcpy(dev_work_, work_template, algo_->get_config().template_size);
        }
        dev_difficulty_.store(difficulty, std::memory_order_release);
        dev_work_valid_.store(true, std::memory_order_release);
        work_updated_.store(true, std::memory_order_release);
    }

    void set_job_id(int64_t job_id, const std::string& job_id_str = "") {
        {
            std::lock_guard<std::mutex> lock(job_str_mutex_);
            current_job_id_str_ = job_id_str;
        }
        current_job_id_.store(job_id, std::memory_order_release);
    }

    void set_dev_job_id(int64_t job_id, const std::string& job_id_str = "") {
        {
            std::lock_guard<std::mutex> lock(job_str_mutex_);
            dev_job_id_str_ = job_id_str;
        }
        dev_job_id_.store(job_id, std::memory_order_release);
    }

    // Store a raw 256-bit target (for algos like KawPow where the pool
    // sends the target, not a scalar difficulty).  Uploaded to GPU per-batch
    // in mine_loop so dev and regular targets stay separate.
    void set_raw_target(const uint8_t* target_32bytes) {
        memcpy(raw_target_, target_32bytes, 32);
        raw_target_valid_.store(true, std::memory_order_release);
    }

    void set_dev_raw_target(const uint8_t* target_32bytes) {
        memcpy(dev_raw_target_, target_32bytes, 32);
        dev_raw_target_valid_.store(true, std::memory_order_release);
    }

    void set_dev_fee(double fee) {
        dev_fee_ = fee;
    }

    int64_t get_current_job_id() const {
        return current_job_id_.load(std::memory_order_acquire);
    }

    const uint8_t* get_current_work_template() const {
        return algo_->get_current_work_template();
    }

    // Start mining with a solution builder callback.
    // The builder does algo-specific work (CPU verification, payload construction)
    // and returns a GPUSubmitEntry to queue, or nullopt to skip.
    // Solutions are pushed to GPUSubmitQueue — the GPU thread never blocks on I/O.
    // Safe to call multiple times — stops any existing thread first.
    void start(SolutionBuilder builder) {
        stop();  // no-op if not running
        TNN_LOG_TRACE("[TRACE] Miner %d starting\n", device_id_);
        running_ = true;
        solution_builder_ = std::move(builder);
        miner_thread_ = std::thread([this]() {
            TNN_LOG_TRACE("[TRACE] Miner %d thread started\n", device_id_);
            mine_loop();
        });
    }

    void stop() {
        running_ = false;
        if (miner_thread_.joinable()) {
            miner_thread_.join();
        }
        // Reset work flags so a subsequent start() waits for fresh work
        work_updated_.store(false, std::memory_order_relaxed);
        dev_work_valid_.store(false, std::memory_order_relaxed);
    }

    double get_hashrate() const {
        return algo_->get_hashrate();
    }

    uint64_t get_total_hashes() const {
        return total_hashes_.load();
    }

    int get_device_id() const { return device_id_; }

    void set_solution_builder(SolutionBuilder builder) {
        solution_builder_ = std::move(builder);
    }

    // Forward a named dependency to the algo (thread-safe).
    // mine_batch fires the algo's deps_changed_fn hook when values change.
    void set_dep(const std::string& key, int64_t value) {
        algo_->set_dep(key, value);
    }

    void set_dev_dep(const std::string& key, int64_t value) {
        algo_->set_dev_dep(key, value);
    }

private:
    void mine_loop() {
        // Reuse the GPU context created during initialize().
        // This sets the thread-local s_api and binds the context (with its
        // compiled modules/kernels) to this worker thread.
        oroCtx ctx = algo_->get_ctx();
        if (!ctx) {
            TNN_LOG_ERROR("[ERROR] GPU %d: mine_loop has no context (initialize not called?)\n", device_id_);
            return;
        }
        oroError_t ctx_err = oroCtxSetCurrent(ctx);
        if (ctx_err != oroSuccess) {
            TNN_LOG_ERROR("[ERROR] GPU %d: mine_loop oroCtxSetCurrent failed: %s\n",
                    device_id_, tnn_error_string(ctx_err));
            return;
        }

        // Nonce segmentation:
        // - Bits 59-63 (top 5 bits): device ID (supports up to 32 GPUs)
        // - Bits 48-58 (next 11 bits): random int (0-2047, randomized per mining session)
        // - Bits 0-47 (bottom 48 bits): counter (incremented by batch_size each kernel launch)

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<uint64_t> random_dist(0, 0x7FF);  // 11-bit random value (0-2047)

        uint64_t device_id_bits = ((uint64_t)(device_id_ & 0x1F)) << 59;  // Top 5 bits
        uint64_t random_bits = random_dist(rng) << 48;                     // Next 11 bits
        uint64_t nonce = device_id_bits | random_bits;                     // Counter starts at 0

        const uint32_t batch_size = algo_->get_batch_size();
        const size_t hash_size = algo_->get_config().hash_size;

        TNN_LOG_DEBUG("[DEBUG] GPU %d: Nonce segmentation - device_id=%d in bits[59:63], random=%llu in bits[48:58], counter in bits[0:47]\n",
               device_id_, device_id_, (unsigned long long)(random_bits >> 48));
        TNN_LOG_DEBUG("[DEBUG] GPU %d: device_id_=%d, device_id_bits=0x%016llx, random_bits=0x%016llx, nonce=0x%016llx\n",
               device_id_, device_id_, (unsigned long long)device_id_bits, (unsigned long long)random_bits, (unsigned long long)nonce);

        std::uniform_real_distribution<double> fee_dist(0.0, 10000.0);
        uint64_t batch_counter = 0;

        // Wait for initial work to be set
        while (running_ && !work_updated_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        while (running_) {
            try {
                work_updated_.store(false, std::memory_order_relaxed);

                // Decide dev or user for this batch
                bool is_dev_batch = (dev_work_valid_.load(std::memory_order_acquire)
                                     && dev_fee_ > 0.0
                                     && fee_dist(rng) < dev_fee_ * 100.0);

                // Load the appropriate template into the algo
                const size_t tmpl_size = algo_->get_config().template_size;
                uint8_t local_work[512];  // large enough for any template
                {
                    std::lock_guard<std::mutex> lock(work_mutex_);
                    if (is_dev_batch)
                        memcpy(local_work, dev_work_, tmpl_size);
                    else
                        memcpy(local_work, current_work_, tmpl_size);
                }

                uint64_t batch_diff = is_dev_batch
                    ? dev_difficulty_.load(std::memory_order_acquire)
                    : current_difficulty_.load(std::memory_order_acquire);

                algo_->set_work(local_work, batch_diff);

                // Select correct dep set (regular vs dev) for this batch
                algo_->use_dev_deps(is_dev_batch);

                // Upload dev raw target if needed
                if (is_dev_batch && dev_raw_target_valid_.load(std::memory_order_acquire)) {
                    algo_->set_raw_target(dev_raw_target_);
                } else if (!is_dev_batch && raw_target_valid_.load(std::memory_order_acquire)) {
                    algo_->set_raw_target(raw_target_);
                }

                // Capture complete job state BEFORE mining exec
                const uint8_t* template_ptr = algo_->get_current_work_template();
                std::string id_str;
                {
                    std::lock_guard<std::mutex> lock(job_str_mutex_);
                    id_str = is_dev_batch ? dev_job_id_str_ : current_job_id_str_;
                }
                int64_t snap_job_id = is_dev_batch
                    ? dev_job_id_.load(std::memory_order_acquire)
                    : current_job_id_.load(std::memory_order_acquire);

                JobSnapshot job_snapshot(
                    template_ptr,
                    tmpl_size,
                    snap_job_id,
                    batch_diff,
                    algo_->get_config().algo_id,
                    id_str,
                    is_dev_batch
                );

                TNN_LOG_TRACE("[TRACE] GPU%d calling mine_batch: nonce=0x%016llx (dev=%llu, rand=%llu, ctr=%llu)\n",
                       device_id_, (unsigned long long)nonce,
                       (unsigned long long)((nonce >> 59) & 0x1F),
                       (unsigned long long)((nonce >> 48) & 0x7FF),
                       (unsigned long long)(nonce & 0xFFFFFFFFFFFFULL));
                auto result = algo_->mine_batch(nonce, batch_size);
                TNN_LOG_TRACE("[TRACE] GPU%d mine_batch END (checked %u hashes, found %u valid)\n",
                       device_id_, result.count, result.num_valid);

                for (uint32_t i = 0; i < result.num_valid; i++) {
                    uint64_t winning_nonce = result.valid_nonces[i];
                    const uint8_t* hash = result.valid_hashes.data() + i * hash_size;

                    TNN_LOG_TRACE("[TRACE] GPU%d SOLUTION FOUND (nonce=%llu), calling solution builder\n",
                          device_id_, winning_nonce);

                    if (solution_builder_) {
                        auto entry = solution_builder_(hash, winning_nonce, device_id_, job_snapshot);
                        if (entry.has_value()) {
                            GPUSubmitQueue::instance().push(std::move(entry.value()));
                        }
                    }
                }

                total_hashes_ += result.count;
                uint64_t counter_loc = nonce & 0xFFFFFFFFFFFFULL;
                counter_loc += batch_size;
                nonce = (nonce & 0xFFFF000000000000ULL) | (counter_loc & 0xFFFFFFFFFFFFULL);

                HIP_counters[device_id_].fetch_add(result.count);

            } catch (const std::exception& e) {
                TNN_LOG_ERROR("GPU mining error on device %d: %s\n", device_id_, e.what());
                running_ = false;
                break;
            }
        }
    }

    bool check_difficulty(const uint8_t* hash, uint64_t difficulty) {
        if (difficulty == 0) return false;

        uint8_t hash_copy[32];
        memcpy(hash_copy, hash, 32);
        std::reverse(hash_copy, hash_copy + 32);

        bool result = CheckHash(hash_copy, difficulty, algo_->get_config().algo_id);
        return result;
    }

    int device_id_;
    std::unique_ptr<IGPUAlgorithm> algo_;

    std::atomic<bool> running_;
    std::atomic<uint64_t> total_hashes_{0};

    // User work
    std::atomic<uint64_t> current_difficulty_{0};
    std::atomic<int64_t> current_job_id_{0};
    std::atomic<bool> work_updated_{false};
    uint8_t current_work_[512] = {};
    mutable std::mutex work_mutex_;

    // Dev work
    std::atomic<uint64_t> dev_difficulty_{0};
    std::atomic<int64_t> dev_job_id_{0};
    std::atomic<bool> dev_work_valid_{false};
    uint8_t dev_work_[512] = {};
    double dev_fee_ = 0.0;

    // Raw target storage (uploaded to GPU per-batch in mine_loop)
    uint8_t raw_target_[32] = {};
    uint8_t dev_raw_target_[32] = {};
    std::atomic<bool> raw_target_valid_{false};
    std::atomic<bool> dev_raw_target_valid_{false};

    mutable std::mutex job_str_mutex_;
    std::string current_job_id_str_;
    std::string dev_job_id_str_;

    std::thread miner_thread_;
    SolutionBuilder solution_builder_;
};
