#pragma once

#include <tnn_hip/crypto/pearl/pearl_native.hpp>
#include <atomic>
#include <vector>

struct AlgoConfig;
struct BatchResult;
struct JobSnapshot;
struct GPUSubmitEntry;

namespace tnn::pearl {

enum class ExecutionMode { Mining, Validation, Benchmark };

// Passed into the algorithm factory; test settings never mutate a live worker.
struct ExecutionOptions {
    ExecutionMode mode = ExecutionMode::Mining;
    native::Shape shape{8192, 8192, 4096};
    uint32_t winner_capacity = 4096;
    uint32_t batch_size = 16;
};

inline bool mining_enabled = false;
inline unsigned cert_version_fallback = 0;
inline std::atomic<bool> worker_failed{false};
inline void configure_mining() {
    mining_enabled = true;
}

AlgoConfig pearl_gpu_config(ExecutionOptions options = {});
void pearl_configure_tuning(AlgoConfig& config, ExecutionOptions options);
void pearl_start_proofs();
void pearl_stop_proofs();
std::vector<GPUSubmitEntry> pearl_build_batch(const BatchResult &, int device, const JobSnapshot &);
// Offline validation of immutable evidence captured by the production path.
size_t pearl_validate_batch(const BatchResult &, bool verify_all_positions);

} // namespace tnn::pearl

void minePearl_hip(int tid);
