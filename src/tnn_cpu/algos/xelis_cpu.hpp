#pragma once
#include "../common/cpu_algo.hpp"
#include "../common/cpu_algo_registry.hpp"
#include "tnn-hugepages.hpp"
#include <xelis-hash/xelis-hash.hpp>
#include <algorithm>
#include <endian.hpp>
#include "tnn-common.hpp"

// Xelis V2 CPU Algorithm Implementation
class XelisV2CPU : public ICPUAlgorithm {
public:
    XelisV2CPU() : worker_(nullptr), thread_id_(0) {}

    ~XelisV2CPU() override {
        cleanup();
    }

    bool initialize(int thread_id) override {
        thread_id_ = thread_id;

        // Allocate worker data with huge pages
        worker_ = (workerData_xelis_v2*)malloc_huge_pages(sizeof(workerData_xelis_v2));
        if (!worker_) {
            return false;
        }

        return true;
    }

    void cleanup() override {
        if (worker_) {
            free_huge_pages(worker_);
            worker_ = nullptr;
        }
    }

    bool set_work(const uint8_t* work_template, size_t size) override {
        auto& config = get_config();
        if (size != config.template_size) {
            return false;
        }

        std::memcpy(work_buffer_, work_template, size);
        return true;
    }

    bool compute_hash(uint64_t nonce, uint8_t* output) override {
        if (!worker_) return false;

        auto& config = get_config();

        // Copy work to local buffer for this hash
        uint8_t local_work[XELIS_TEMPLATE_SIZE];
        std::memcpy(local_work, work_buffer_, XELIS_TEMPLATE_SIZE);

        // Nonce is already encoded with thread ID by CPUMiner
        // Just write it to the template
        std::memcpy(local_work + config.nonce_offset, &nonce, config.nonce_size);

        // Compute hash
        xelis_hash_v2(local_work, *worker_, output);

        // Handle endianness
        if (littleEndian()) {
            std::reverse(output, output + 32);
        }

        return true;
    }

    const CPUAlgoConfig& get_config() const override {
        static const CPUAlgoConfig config = {
            .name = "xelis_v2",
            .template_size = XELIS_TEMPLATE_SIZE,
            .hash_size = 32,
            .nonce_offset = 40,
            .nonce_size = 8,
            .needs_hugepages = true,
            .needs_preprocessing = false,
            .algo_id = ALGO_XELISV2
        };
        return config;
    }

private:
    workerData_xelis_v2* worker_;
    int thread_id_;
    uint8_t work_buffer_[XELIS_TEMPLATE_SIZE];
};

// Xelis V3 CPU Algorithm Implementation
class XelisV3CPU : public ICPUAlgorithm {
public:
    XelisV3CPU() : worker_(nullptr), thread_id_(0) {}

    ~XelisV3CPU() override {
        cleanup();
    }

    bool initialize(int thread_id) override {
        thread_id_ = thread_id;

        // Allocate worker data with huge pages
        worker_ = (workerData_xelis_v3*)malloc_huge_pages(sizeof(workerData_xelis_v3));
        if (!worker_) {
            return false;
        }

        return true;
    }

    void cleanup() override {
        if (worker_) {
            free_huge_pages(worker_);
            worker_ = nullptr;
        }
    }

    bool set_work(const uint8_t* work_template, size_t size) override {
        auto& config = get_config();
        if (size != config.template_size) {
            return false;
        }

        std::memcpy(work_buffer_, work_template, size);
        return true;
    }

    bool compute_hash(uint64_t nonce, uint8_t* output) override {
        if (!worker_) return false;

        auto& config = get_config();

        // Copy work to local buffer for this hash
        uint8_t local_work[XELIS_TEMPLATE_SIZE];
        std::memcpy(local_work, work_buffer_, XELIS_TEMPLATE_SIZE);

        // Write nonce (already encoded with thread ID)
        std::memcpy(local_work + config.nonce_offset, &nonce, config.nonce_size);

        // Compute hash
        xelis_hash_v3(local_work, *worker_, output);

        // Handle endianness
        if (littleEndian()) {
            std::reverse(output, output + 32);
        }

        return true;
    }

    const CPUAlgoConfig& get_config() const override {
        static const CPUAlgoConfig config = {
            .name = "xelis_v3",
            .template_size = XELIS_TEMPLATE_SIZE,
            .hash_size = 32,
            .nonce_offset = 40,
            .nonce_size = 8,
            .needs_hugepages = true,
            .needs_preprocessing = false,
            .algo_id = ALGO_XELISV3
        };
        return config;
    }

private:
    workerData_xelis_v3* worker_;
    int thread_id_;
    uint8_t work_buffer_[XELIS_TEMPLATE_SIZE];
};

// Register both versions with the registry
REGISTER_CPU_ALGORITHM("xelis_v2", XelisV2CPU)
REGISTER_CPU_ALGORITHM("xelis_v3", XelisV3CPU)
