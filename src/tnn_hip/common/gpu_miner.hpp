#pragma once
#include "hip_algo_registry.hpp"
#include "../../coins/miners.hpp"
#include <atomic>
#include <thread>
#include <mutex>

class GPUMiner {
public:
    GPUMiner(const std::string& algo_name, int device_id = 0)
        : device_id_(device_id), running_(false) {

        printf("[TRACE] GPUMiner: Constructor for algo='%s', device=%d\n", algo_name.c_str(), device_id);
        fflush(stdout);

        algo_ = AlgoRegistry::instance().create(algo_name);
        if (!algo_) {
            throw std::runtime_error("Unknown algorithm: " + algo_name);
        }

        printf("[TRACE] GPUMiner: Algorithm instance created\n");
        fflush(stdout);
    }

    bool initialize() {
        printf("[TRACE] GPUMiner: Calling algo_->initialize()\n");
        fflush(stdout);

        bool result = algo_->initialize(device_id_);

        printf("[TRACE] GPUMiner: algo_->initialize() returned %s\n", result ? "true" : "false");
        fflush(stdout);

        return result;
    }
    
    void set_work(const uint8_t* work_template, uint64_t difficulty) {
        algo_->set_work(work_template, difficulty);
        current_difficulty_.store(difficulty, std::memory_order_release);
        work_updated_.store(true, std::memory_order_release);

        // DEBUG: Log work template (first time only to avoid spam)
        static bool logged_template = false;
        if (!logged_template) {
            printf("\n[DEBUG] Work template set (GPU %d):\n  ", device_id_);
            for (int i = 0; i < 112; i++) {
                printf("%02x", work_template[i]);
                if ((i+1) % 32 == 0) printf("\n  ");
            }
            printf("\n  Difficulty: %lu\n", difficulty);
            fflush(stdout);
            logged_template = true;
        }
    }
    
    void start(std::function<void(const uint8_t*, uint64_t)> on_solution) {
        printf("[TRACE] Miner %d starting\n", device_id_);
        fflush(stdout);
        running_ = true;
        on_solution_ = on_solution;
        
        printf("[TRACE] Miner %d vars assigned\n", device_id_);
        fflush(stdout);
        miner_thread_ = std::thread([this]() {
            printf("[TRACE] Miner %d thread started\n", device_id_);
            fflush(stdout);
            mine_loop();
        });
    }
    
    void stop() {
        running_ = false;
        if (miner_thread_.joinable()) {
            miner_thread_.join();
        }
    }
    
    double get_hashrate() const {
        return algo_->get_hashrate();
    }
    
    uint64_t get_total_hashes() const {
        return total_hashes_.load();
    }
    
    int get_device_id() const { return device_id_; }
    
    void set_solution_callback(
        std::function<void(const uint8_t*, uint64_t)> callback
    ) {
        on_solution_ = callback;
    }

private:
    std::timed_mutex send_mutex;
    void mine_loop() {
        printf("[TRACE] mine_loop top of loop\n");
        fflush(stdout);

        uint64_t nonce = 0;
        const uint32_t batch_size = algo_->get_batch_size();
        const size_t hash_size = algo_->get_config().hash_size;

        printf("[TRACE] mine_loop config read\n");
        fflush(stdout);

        // Wait for initial work to be set
        while (running_ && !work_updated_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        printf("[TRACE] mine_loop work received\n");
        fflush(stdout);

        while (running_) {
            try {
                printf("[TRACE] mine_loop running_ is true\n");
                fflush(stdout);

                work_updated_.store(false, std::memory_order_relaxed);

                printf("[TRACE] GPU%d mine_batch START (nonce=%lu)\n", device_id_, nonce);
                fflush(stdout);
                auto result = algo_->mine_batch(nonce, batch_size);
                printf("[TRACE] GPU%d mine_batch END (checked %u hashes, found %u valid)\n",
                       device_id_, result.count, result.num_valid);
                fflush(stdout);

                for (uint32_t i = 0; i < result.num_valid; i++) {
                    uint64_t winning_nonce = result.valid_nonces[i];
                    const uint8_t* hash = result.valid_hashes.data() + i * hash_size;

                    printf("[TRACE] GPU%d SOLUTION FOUND (nonce=%lu), attempting callback with 50ms lock\n",
                          device_id_, winning_nonce);
                    fflush(stdout);

                    if (on_solution_) {
                        // 50ms timed lock around the send / callback
                        std::unique_lock<std::timed_mutex> lock(send_mutex, std::defer_lock);
                        if (lock.try_lock_for(std::chrono::milliseconds(50))) {
                            on_solution_(hash, winning_nonce);
                            // lock released automatically when going out of scope
                        } else {
                            // Couldn’t get the lock in time - optional logging
                            fprintf(stderr,
                                    "[WARN] GPU%d: could not acquire send_mutex within 50ms, "
                                    "skipping solution callback for nonce=%lu\n",
                                    device_id_, winning_nonce);
                        }

                        break; // FOR NOW, as you already had
                    }

                    printf("[TRACE] GPU%d callback pointer was null\n", device_id_);
                    fflush(stdout);
                }

                if (result.num_valid > 0) {
                    printf("[TRACE] GPU%d batch had %u solution(s)\n", device_id_, result.num_valid);
                    fflush(stdout);
                }

                total_hashes_ += result.count;
                nonce += batch_size;

                HIP_counters[device_id_].fetch_add(result.count);
                counter.fetch_add(result.count);

                // Check if work was updated
                if (work_updated_.load(std::memory_order_acquire)) {
                    nonce = 0;  // Reset nonce for new work
                }
            } catch (const std::exception& e) {
                fprintf(stderr, "GPU mining error on device %d: %s\n", device_id_, e.what());
                running_ = false;
                break;
            }
        }
    }

    bool check_difficulty(const uint8_t* hash, uint64_t difficulty) {
        if (difficulty == 0) return false;

        // For Xelis: hash needs to be reversed before CheckHash
        // CheckHash modifies the hash, so make a copy
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

    std::atomic<uint64_t> current_difficulty_{0};
    std::atomic<bool> work_updated_{false};

    std::thread miner_thread_;
    std::function<void(const uint8_t*, uint64_t)> on_solution_;
};