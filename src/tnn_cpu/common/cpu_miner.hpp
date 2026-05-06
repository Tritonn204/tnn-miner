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
        if (difficulty != current_difficulty_) {
            current_difficulty_ = difficulty;
            recomputeTarget(difficulty, algo_->get_config().algo_id);
        }
    }

    // Compute one hash with automatic nonce encoding
    // Returns true if hash meets current difficulty
    bool mine_one(uint8_t* hash_output, uint64_t* found_nonce, uint8_t* work_output) {
        auto& config = algo_->get_config();

        // Encode nonce with thread ID
        uint64_t encoded_nonce = encode_nonce(nonce_counter_++, thread_id_);

        // Write nonce into this thread's prepared work buffer.
        std::memcpy(work_buffer_.data() + config.nonce_offset,
                   &encoded_nonce,
                   config.nonce_size);

        // Compute hash from the nonce-filled work buffer when supported.
        if (!algo_->compute_hash_prepared(work_buffer_.data(), hash_output) &&
            !algo_->compute_hash(encoded_nonce, hash_output)) {
            return false;
        }

        total_hashes_++;

        bool meets_target = false;
        if (current_difficulty_ > 0) {
            if (config.algo_id == ALGO_XELISV2 || config.algo_id == ALGO_XELISV3) {
                meets_target = hashMeetsTarget_be_hash_le_target(hash_output, cached_target_);
            } else {
                meets_target = hashMeetsTarget_le(hash_output, cached_target_);
            }
        }

        // Fast target comparison (no bignum, no hex, no hot-buffer reverse)
        if (meets_target) {
            *found_nonce = encoded_nonce;
            std::memcpy(work_output, work_buffer_.data(), config.template_size);
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
    void recomputeTarget(uint64_t diff, int algo_id) {
        static const boost::multiprecision::uint256_t kMaxU256 =
            (boost::multiprecision::uint256_t(1) << 256) - 1;
        if (diff == 0) { memset(cached_target_, 0xFF, 32); return; }
        boost::multiprecision::uint256_t target;
        switch (algo_id) {
            case ALGO_ASTROBWTV3: {
                // (2^256) / d
                target = kMaxU256 / diff;
                boost::multiprecision::uint256_t rem = kMaxU256 % diff;
                if (rem + 1 >= diff) target += 1;
                break;
            }
            case ALGO_XELISV2:
            case ALGO_XELISV3:
            default:
                // (2^256 - 1) / d
                target = kMaxU256 / diff;
                break;
        }
        cpp_int_to_byte_array(target, cached_target_);
    }

    int thread_id_;
    std::unique_ptr<ICPUAlgorithm> algo_;

    uint64_t total_hashes_;
    uint64_t nonce_counter_;

    std::vector<uint8_t> work_buffer_;
    uint64_t current_difficulty_;
    alignas(8) uint8_t cached_target_[32] = {0};
};
