#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

struct CPUAlgoConfig {
    std::string name;
    size_t template_size;        // Size of work template
    size_t hash_size;            // Output hash size (usually 32)
    size_t nonce_offset;         // Where to write nonce in template
    size_t nonce_size;           // Size of nonce field (usually 8)
    bool needs_hugepages;        // Whether worker data needs huge pages
    bool needs_preprocessing;    // Whether work needs preprocessing (e.g., matrix computation)
    int algo_id;                 // ALGO_* constant for difficulty conversion
};

struct HashResult {
    uint8_t hash[32];
    uint64_t nonce;
    bool valid;
};

// Abstract base class for CPU mining algorithms
class ICPUAlgorithm {
public:
    virtual ~ICPUAlgorithm() = default;

    // Initialize algorithm-specific resources for this thread
    // thread_id: The mining thread ID (for nonce space partitioning)
    virtual bool initialize(int thread_id) = 0;

    // Clean up algorithm resources
    virtual void cleanup() = 0;

    // Set new work template (called when job changes)
    // work_template: Pointer to work data
    // size: Size of work data
    // Returns: true if preprocessing succeeded (if needed)
    virtual bool set_work(const uint8_t* work_template, size_t size) = 0;

    // Compute a single hash
    // nonce: Nonce value to use
    // output: Buffer to write hash result (must be at least hash_size bytes)
    // Returns: true on success
    virtual bool compute_hash(uint64_t nonce, uint8_t* output) = 0;

    // Compute from a nonce-filled work template when the miner already had to
    // build one for possible submission. Algorithms that do not override this
    // keep the existing nonce-based path.
    virtual bool compute_hash_prepared(const uint8_t* prepared_work, uint8_t* output) {
        (void)prepared_work;
        (void)output;
        return false;
    }

    // Get algorithm configuration
    virtual const CPUAlgoConfig& get_config() const = 0;

    // Get current hashrate (optional, can return 0)
    virtual double get_hashrate() const { return 0.0; }
};

// Helper function to encode thread ID into nonce
inline uint64_t encode_nonce(uint64_t base_nonce, int thread_id) {
    // Match the Xelis live/tune nonce layout: thread id in the low 16 bits,
    // a varying byte in bits 16-23, and the monotonic counter above it.
    return ((thread_id - 1) % (256 * 256)) |
           ((base_nonce & 0xFFULL) << 16) |
           (base_nonce << 24);
}
