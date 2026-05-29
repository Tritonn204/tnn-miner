#pragma once
#include "../common/cpu_algo.hpp"
#include "../common/cpu_algo_registry.hpp"
#include "tnn-common.hpp"
#include <crypto/qhash/qhash.hpp>

class QHashCPU : public ICPUAlgorithm {
public:
    QHashCPU() : ws_(nullptr), thread_id_(0) {}

    ~QHashCPU() override {
        cleanup();
    }

    bool initialize(int thread_id) override {
        thread_id_ = thread_id;
        ws_ = new QHashWorkspace;
        return ws_ != nullptr;
    }

    void cleanup() override {
        delete ws_;
        ws_ = nullptr;
    }

    bool set_work(const uint8_t* work_template, size_t size) override {
        if (size != QHASH_INPUT_SIZE) {
            return false;
        }
        std::memcpy(work_buffer_, work_template, size);
        return true;
    }

    bool compute_hash(uint64_t nonce, uint8_t* output) override {
        if (!ws_) return false;

        // Copy work template and insert nonce
        uint8_t local_work[QHASH_INPUT_SIZE];
        std::memcpy(local_work, work_buffer_, QHASH_INPUT_SIZE);

        // Insert nonce at bytes 76-79 (standard Bitcoin-style block header)
        std::memcpy(local_work + 76, &nonce, sizeof(nonce));

        qhash_compute(local_work, output, ws_);
        return true;
    }

    bool compute_hash_prepared(const uint8_t* prepared_work, uint8_t* output) override {
        if (!ws_) return false;
        qhash_compute(prepared_work, output, ws_);
        return true;
    }

    const CPUAlgoConfig& get_config() const override {
        static const CPUAlgoConfig config = {
            .name = "qhash",
            .template_size = QHASH_INPUT_SIZE,
            .hash_size = QHASH_SHA256_BYTES,
            .nonce_offset = 76,   // bytes 76-79 in 80-byte Bitcoin-style header
            .nonce_size = 4,
            .needs_hugepages = false,
            .needs_preprocessing = false,
            .algo_id = ALGO_QHASH
        };
        return config;
    }

private:
    QHashWorkspace* ws_;  // 1 MiB state vector, reused across hashes
    int thread_id_;
    uint8_t work_buffer_[QHASH_INPUT_SIZE];
};

REGISTER_CPU_ALGORITHM("qhash", QHashCPU)
