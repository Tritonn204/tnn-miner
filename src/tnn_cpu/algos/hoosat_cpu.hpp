#pragma once
#include "../common/cpu_algo.hpp"
#include "../common/cpu_algo_registry.hpp"
#include <hoohash/hoohash.h>
#include <algorithm>
#include <endian.hpp>
#include "tnn-common.hpp"

// Hoosat CPU Algorithm Implementation
class HoosatCPU : public ICPUAlgorithm {
public:
    HoosatCPU() : worker_(nullptr), dev_worker_(nullptr), thread_id_(0) {}

    ~HoosatCPU() override {
        cleanup();
    }

    bool initialize(int thread_id) override {
        thread_id_ = thread_id;

        // Allocate worker data
        worker_ = (HooHash::worker*)malloc(sizeof(HooHash::worker));
        dev_worker_ = (HooHash::worker*)malloc(sizeof(HooHash::worker));

        if (!worker_ || !dev_worker_) {
            cleanup();
            return false;
        }

        return true;
    }

    void cleanup() override {
        if (worker_) {
            free(worker_);
            worker_ = nullptr;
        }
        if (dev_worker_) {
            free(dev_worker_);
            dev_worker_ = nullptr;
        }
    }

    bool set_work(const uint8_t* work_template, size_t size) override {
        auto& config = get_config();
        if (size != config.template_size) {
            return false;
        }

        std::memcpy(work_buffer_, work_template, size);

        // Preprocess matrix for this work (Hoosat-specific)
        HooHash::newMatrix(work_buffer_, worker_->matBuffer, *worker_);

        return true;
    }

    bool compute_hash(uint64_t nonce, uint8_t* output) override {
        if (!worker_) return false;

        auto& config = get_config();

        // Copy work to local buffer
        uint8_t local_work[HooHash::INPUT_SIZE];
        std::memcpy(local_work, work_buffer_, HooHash::INPUT_SIZE);

        // Write nonce (already encoded with thread ID)
        std::memcpy(local_work + config.nonce_offset, &nonce, config.nonce_size);

        // Compute hash
        HooHash::hash(*worker_, local_work, HooHash::INPUT_SIZE, output);

        return true;
    }

    const CPUAlgoConfig& get_config() const override {
        static const CPUAlgoConfig config = {
            .name = "hoosat",
            .template_size = HooHash::INPUT_SIZE,
            .hash_size = 32,
            .nonce_offset = 72,
            .nonce_size = 8,
            .needs_hugepages = false,
            .needs_preprocessing = true,  // Uses newMatrix
            .algo_id = ALGO_HOOHASH
        };
        return config;
    }

private:
    HooHash::worker* worker_;
    HooHash::worker* dev_worker_;
    int thread_id_;
    uint8_t work_buffer_[HooHash::INPUT_SIZE];
};

// Register with the registry
REGISTER_CPU_ALGORITHM("hoosat", HoosatCPU)
