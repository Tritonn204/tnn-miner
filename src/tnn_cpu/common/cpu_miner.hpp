#pragma once
#include "cpu_algo.hpp"
#include "cpu_algo_registry.hpp"
#include <atomic>
#include <functional>
#include <cstring>
#include <vector>

// Include miners.hpp for ConvertDifficultyToBig and other utilities
// This is the same as legacy CPU miners do
#include "../../coins/miners.hpp"

// Simplified CPUMiner - no internal threading
// The mining thread calls compute_hash() directly in a loop
class CPUMiner {
public:
    CPUMiner(const std::string& algo_name, int thread_id)
        : thread_id_(thread_id), nonce_counter_(0) {

        algo_ = CPUAlgoRegistry::instance().create(algo_name);
        if (!algo_) {
            throw std::runtime_error("Unknown CPU algorithm: " + algo_name);
        }

        auto& config = algo_->get_config();
        work_buffer_.resize(config.template_size);
        current_difficulty_ = 0;
        total_hashes_ = 0;
    }

    ~CPUMiner() {
        if (algo_) {
            algo_->cleanup();
        }
    }

    bool initialize() {
        return algo_->initialize(thread_id_);
    }

    void set_work(const uint8_t* work_template, size_t size) {
        auto& config = algo_->get_config();
        if (size != config.template_size) {
            return;
        }

        std::memcpy(work_buffer_.data(), work_template, config.template_size);

        // Let algorithm do any preprocessing (e.g., matrix computation)
        algo_->set_work(work_buffer_.data(), config.template_size);

        nonce_counter_ = 0;  // Reset nonce for new work
    }

    void set_difficulty(uint64_t difficulty) {
        current_difficulty_ = difficulty;
    }

    // Compute one hash with automatic nonce encoding
    // Returns true if hash meets current difficulty
    bool mine_one(uint8_t* hash_output, uint64_t* found_nonce, uint8_t* work_output) {
        auto& config = algo_->get_config();

        // Encode nonce with thread ID
        uint64_t encoded_nonce = encode_nonce(nonce_counter_++, thread_id_);

        // Copy work to output buffer (for submission)
        std::memcpy(work_output, work_buffer_.data(), config.template_size);

        // Write nonce to work
        std::memcpy(work_output + config.nonce_offset,
                   &encoded_nonce,
                   config.nonce_size);

        // Compute hash (algorithm will write to hash_output)
        if (!algo_->compute_hash(encoded_nonce, hash_output)) {
            return false;
        }

        total_hashes_++;

        // Check difficulty using existing CheckHash function
        if (current_difficulty_ > 0 && check_difficulty(hash_output, current_difficulty_, config.algo_id)) {
            *found_nonce = encoded_nonce;
            return true;
        }

        return false;
    }

    uint64_t get_total_hashes() const {
        return total_hashes_;
    }

    int get_thread_id() const {
        return thread_id_;
    }

    const CPUAlgoConfig& get_config() const {
        return algo_->get_config();
    }

    uint64_t get_nonce_counter() const {
        return nonce_counter_;
    }

    void reset_nonce() {
        nonce_counter_ = 0;
    }

private:
    // Difficulty check using existing infrastructure
    bool check_difficulty(uint8_t* hash, uint64_t difficulty, int algo_id) {
        // Use existing ConvertDifficultyToBig and CheckHash from miners.hpp
        Num cmpDiff = ConvertDifficultyToBig(difficulty, algo_id);

        // CheckHash expects big-endian, handles reversing internally
        if (littleEndian()) {
            std::reverse(hash, hash + 32);
        }

        bool result = (Num(hexStr(hash, 32).c_str(), 16) <= cmpDiff);

        if (littleEndian()) {
            std::reverse(hash, hash + 32);
        }

        return result;
    }

    int thread_id_;
    std::unique_ptr<ICPUAlgorithm> algo_;

    uint64_t total_hashes_;
    uint64_t nonce_counter_;

    std::vector<uint8_t> work_buffer_;
    uint64_t current_difficulty_;
};
