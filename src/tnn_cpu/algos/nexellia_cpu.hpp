#pragma once
#include "../common/cpu_algo.hpp"
#include "../common/cpu_algo_registry.hpp"
#include <nxl-hash/nxl-hash.h>
#include <algorithm>
#include <endian.hpp>
#include "tnn-common.hpp"

// Nexellia CPU Algorithm Implementation
class NexelliaCPU : public ICPUAlgorithm {
public:
    NexelliaCPU() : worker_(nullptr), dev_worker_(nullptr), thread_id_(0) {}

    ~NexelliaCPU() override {
        cleanup();
    }

    bool initialize(int thread_id) override {
        thread_id_ = thread_id;

        // Allocate worker data
        worker_ = (NxlHash::worker*)malloc(sizeof(NxlHash::worker));
        dev_worker_ = (NxlHash::worker*)malloc(sizeof(NxlHash::worker));

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

        // Preprocess matrix for this work (Nexellia-specific)
        NxlHash::newMatrix(work_buffer_, worker_->matBuffer, *worker_);

        return true;
    }

    bool compute_hash(uint64_t nonce, uint8_t* output) override {
        if (!worker_) return false;

        auto& config = get_config();

        // Copy work to local buffer
        uint8_t local_work[NxlHash::INPUT_SIZE];
        std::memcpy(local_work, work_buffer_, NxlHash::INPUT_SIZE);

        // Write nonce (already encoded with thread ID)
        std::memcpy(local_work + config.nonce_offset, &nonce, config.nonce_size);

        // Compute hash
        NxlHash::hash(*worker_, local_work, NxlHash::INPUT_SIZE, output);

        return true;
    }

    const CPUAlgoConfig& get_config() const override {
        static const CPUAlgoConfig config = {
            .name = "nexellia",
            .template_size = NxlHash::INPUT_SIZE,
            .hash_size = 32,
            .nonce_offset = 72,
            .nonce_size = 8,
            .needs_hugepages = false,
            .needs_preprocessing = true,  // Uses newMatrix
            .algo_id = ALGO_NXL_HASH
        };
        return config;
    }

private:
    NxlHash::worker* worker_;
    NxlHash::worker* dev_worker_;
    int thread_id_;
    uint8_t work_buffer_[NxlHash::INPUT_SIZE];
};

// Register with the registry
REGISTER_CPU_ALGORITHM("nexellia", NexelliaCPU)
