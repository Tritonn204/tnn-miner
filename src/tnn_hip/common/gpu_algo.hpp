#pragma once
#include <hip/hip_runtime.h>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <condition_variable>

enum class AlgoCategory {
    Simple,      // small state, mostly compute
    Mixed,       // moderate memory + compute
    MemoryHard,   // big scratchpad, random access (Xelis, RandomX-style)
};

// Structure for HIPRTC embedded headers
struct RTCHeader {
    std::string_view name;
    std::string_view source;
};

// ============================================================================
// Helper to build rtc_headers from embedded source manifests
// ============================================================================
namespace detail {
    template<typename T, size_t N>
    constexpr size_t manifest_size(const T (&)[N]) { return N; }
    
    template<typename T, size_t N>
    inline void append_manifest_impl(std::vector<RTCHeader>& result, const T (&manifest)[N]) {
        for (size_t i = 0; i < N; ++i) {
            result.push_back({manifest[i].path.data(), manifest[i].source.data()});
        }
    }
}

template<typename... Manifests>
inline std::vector<RTCHeader> build_rtc_headers(const Manifests&... manifests) {
    std::vector<RTCHeader> result;
    size_t total = (detail::manifest_size(manifests) + ...);
    result.reserve(total);
    (detail::append_manifest_impl(result, manifests), ...);
    return result;
}

// ============================================================================
// Block size limits for auto-tuning (per platform)
// ============================================================================
struct BlockSizeLimits {
    int block_min = 64;
    int block_max = 256;
    int step = 64;
    
    constexpr BlockSizeLimits() = default;
    constexpr BlockSizeLimits(int min_, int max_, int step_) 
        : block_min(min_), block_max(max_), step(step_) {}
};

// ============================================================================
// Tuning configuration result
// ============================================================================
struct TuningResult {
    int block_size = 64;
    int num_blocks = 0;
    uint32_t batch_size = 0;
    double hashrate = 0.0;
    double batch_time_ms = 0.0;
    bool valid = false;
    
    std::string describe() const {
        char buf[256];
        snprintf(buf, sizeof(buf), 
            "block_size=%d, num_blocks=%d, batch_size=%u, hashrate=%.2f H/s, time=%.2fms",
            block_size, num_blocks, batch_size, hashrate, batch_time_ms);
        return buf;
    }
};

// ============================================================================
// Kernel Launch Context - passed to execution strategy
// ============================================================================
struct KernelLaunchContext {
    // Device buffers
    uint8_t* d_input;
    uint8_t* d_outputs;
    uint64_t* d_scratch;
    uint64_t* d_difficulty_target;
    uint64_t* d_solutions;
    
    // Launch parameters
    uint64_t nonce_start;
    uint32_t batch_size;
    int block_size;
    int num_blocks;
    
    // Config reference
    const struct AlgoConfig* config;
    
    // Stream (nullptr = default stream)
    hipStream_t stream = nullptr;
};

// Forward declaration
struct AlgoConfig;

// ============================================================================
// Kernel Execution Strategy
// ============================================================================
using KernelMap = std::unordered_map<std::string, hipFunction_t>;

// Execution function signature
// Returns true on success
using KernelExecuteFn = std::function<bool(
    const KernelMap& kernels,
    const KernelLaunchContext& ctx
)>;

// ============================================================================
// Tune Coordinator (unchanged)
// ============================================================================
class TuneCoordinator {
public:
    static TuneCoordinator& instance() {
        static TuneCoordinator inst;
        return inst;
    }
    
    static std::string make_tune_key(const hipDeviceProp_t& props, 
                                      const std::string& algo_name) {
        std::string vendor;
        std::string arch;
        
#ifdef __HIP_PLATFORM_AMD__
        vendor = "amd";
        arch = props.gcnArchName;
#else
        vendor = "nvidia";
        char buf[32];
        snprintf(buf, sizeof(buf), "compute_%d%d", props.major, props.minor);
        arch = buf;
#endif
        
        size_t vram_gb = (props.totalGlobalMem + 512ULL * 1024 * 1024) / (1024ULL * 1024 * 1024);
        
        for (char& c : arch) {
            if (!isalnum(c) && c != '_' && c != '-') c = '_';
        }
        
        char key[256];
        snprintf(key, sizeof(key), "%s_%s_%zugb_%s", 
                 vendor.c_str(), arch.c_str(), vram_gb, algo_name.c_str());
        return key;
    }
    
    enum class TuneStatus {
        SHOULD_TUNE,
        WAIT_FOR_OTHER,
        ALREADY_DONE
    };
    
    TuneStatus begin_tune(const std::string& tune_key, int device_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (completed_.count(tune_key)) {
            return TuneStatus::ALREADY_DONE;
        }
        
        auto it = in_progress_.find(tune_key);
        if (it != in_progress_.end()) {
            waiters_[tune_key].push_back(device_id);
            return TuneStatus::WAIT_FOR_OTHER;
        }
        
        in_progress_[tune_key] = device_id;
        return TuneStatus::SHOULD_TUNE;
    }
    
    bool wait_for_tune(const std::string& tune_key, 
                       std::chrono::milliseconds timeout = std::chrono::milliseconds(300000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        auto deadline = std::chrono::steady_clock::now() + timeout;
        
        while (in_progress_.count(tune_key)) {
            if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                return false;
            }
        }
        
        return completed_.count(tune_key) > 0;
    }
    
    void end_tune(const std::string& tune_key, bool success) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        in_progress_.erase(tune_key);
        
        if (success) {
            completed_.insert(tune_key);
        }
        
        waiters_.erase(tune_key);
        
        lock.unlock();
        cv_.notify_all();
    }
    
    std::vector<int> get_waiters(const std::string& tune_key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = waiters_.find(tune_key);
        if (it != waiters_.end()) {
            return it->second;
        }
        return {};
    }
    
    bool is_completed(const std::string& tune_key) {
        std::lock_guard<std::mutex> lock(mutex_);
        return completed_.count(tune_key) > 0;
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        in_progress_.clear();
        completed_.clear();
        waiters_.clear();
    }
    
private:
    TuneCoordinator() = default;
    
    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<std::string, int> in_progress_;
    std::unordered_set<std::string> completed_;
    std::unordered_map<std::string, std::vector<int>> waiters_;
};

// ============================================================================
// Algorithm configuration
// ============================================================================
struct AlgoConfig {
    std::string name;
    std::string source_path;
    std::string_view source;
    
    // Multiple kernel names (first is used for tuning/fallback)
    std::vector<std::string> kernel_names;
    
    // Legacy single kernel name (converted to kernel_names if set)
    std::string kernel_name;
    
    std::vector<std::string> compiler_opts_amd;
    std::vector<std::string> compiler_opts_nvidia;
    std::vector<RTCHeader> rtc_headers;
    size_t template_size;
    size_t hash_size;
    size_t nonce_size;
    size_t scratch_per_hash;
    int preferred_block_size;
    int algo_id;

    size_t (*calc_shared_mem)(int block_size);
    
    AlgoCategory category = AlgoCategory::Mixed;
    bool enable_reg_tuning = true;

    // Block size limits per platform
    BlockSizeLimits amd_blocks{64, 1024, 64};
    BlockSizeLimits nvidia_blocks{32, 1024, 32};
    
    // Batch timing targets (milliseconds)
    double target_batch_time_ms = 100.0;
    double max_batch_time_ms = 500.0;
    double min_batch_time_ms = 10.0;
    
    // Auto-tune settings
    bool enable_autotune = true;
    int autotune_warmup = 2;
    int autotune_iterations = 3;
    
    // Memory overhead factor
    double memory_reserve_mb = 128.0;
    double memory_usage_factor = 0.9;
    
    // ========== Execution Strategy ==========
    
    // Custom execution function (nullptr = use default monolithic)
    KernelExecuteFn execute_fn = nullptr;
    
    // Helper to get all kernel names (handles legacy single name)
    std::vector<std::string> get_kernel_names() const {
        if (!kernel_names.empty()) {
            return kernel_names;
        }
        if (!kernel_name.empty()) {
            return {kernel_name};
        }
        return {};
    }
    
    // Get primary kernel name (for tuning)
    std::string get_primary_kernel() const {
        auto names = get_kernel_names();
        return names.empty() ? "" : names[0];
    }
};

// ============================================================================
// Batch result
// ============================================================================
struct BatchResult {
    std::vector<uint8_t> hashes;         // Deprecated
    std::vector<uint64_t> valid_nonces;
    std::vector<uint8_t> valid_hashes;
    uint32_t num_valid;
    uint64_t nonce_start;
    uint32_t count;
};

// ============================================================================
// GPU Algorithm Interface
// ============================================================================
class IGPUAlgorithm {
public:
    virtual ~IGPUAlgorithm() = default;
    
    virtual bool initialize(int device_id = 0) = 0;
    virtual void cleanup() = 0;

    virtual void set_work(const uint8_t* work_template, uint64_t difficulty) = 0;
    virtual BatchResult mine_batch(uint64_t nonce_start, uint32_t batch_size = 0) = 0;
    
    virtual uint32_t get_batch_size() const = 0;
    virtual double get_hashrate() const = 0;
    
    virtual const AlgoConfig& get_config() const = 0;
    
    virtual TuningResult get_tuning_result() const = 0;
    virtual bool set_batch_size_override(uint32_t batch_size) = 0;
};

// ============================================================================
// Global tuning overrides
// ============================================================================
struct GPUTuningOverrides {
    std::vector<uint32_t> gpu_batch_sizes;
    std::vector<int> gpu_block_sizes;
    bool disable_autotune = false;
    
    std::optional<uint32_t> get_batch_override(int device_id) const {
        if (device_id >= 0 && device_id < (int)gpu_batch_sizes.size()) {
            if (gpu_batch_sizes[device_id] > 0) {
                return gpu_batch_sizes[device_id];
            }
        }
        return std::nullopt;
    }
    
    std::optional<int> get_block_override(int device_id) const {
        if (device_id >= 0 && device_id < (int)gpu_block_sizes.size()) {
            if (gpu_block_sizes[device_id] > 0) {
                return gpu_block_sizes[device_id];
            }
        }
        return std::nullopt;
    }
};

extern GPUTuningOverrides g_tuning_overrides;
extern bool g_isTuning;
extern std::atomic<bool> g_mining_started;

inline std::optional<int> choose_maxregcount(
    const AlgoConfig& cfg,
    const hipDeviceProp_t& props)
{
#ifndef __HIP_PLATFORM_NVIDIA__
    return std::nullopt;
#else
    if (!cfg.enable_reg_tuning)
        return std::nullopt;

    const int major = props.major;
    const int minor = props.minor;
    (void)minor;

    switch (cfg.category) {
        case AlgoCategory::MemoryHard:
            if (major >= 9)       return 64;
            else if (major >= 8)  return 64;
            else                  return 64;
        case AlgoCategory::Mixed:
            if (major >= 9)       return 96;
            else if (major >= 8)  return 96;
            else                  return 80;
        case AlgoCategory::Simple:
        default:
            return std::nullopt;
    }
#endif
}

// ============================================================================
// Default Monolithic Execution Strategy
// ============================================================================
inline bool default_monolithic_execute(
    const KernelMap& kernels,
    const KernelLaunchContext& ctx
) {
    if (kernels.empty()) return false;

    // Get primary kernel (explicitly by name, not by iteration order!)
    std::string primary_name = ctx.config->get_primary_kernel();
    auto it = kernels.find(primary_name);

    if (it == kernels.end()) {
        // Fallback to first kernel if primary not found
        fprintf(stderr, "[WARN] Primary kernel '%s' not found, using first available\n",
                primary_name.c_str());
        it = kernels.begin();
    }

    hipFunction_t kernel = it->second;
    
    size_t shared_mem = ctx.config->calc_shared_mem(ctx.block_size);
    
    void* args[] = {
        (void*)&ctx.d_input,
        (void*)&ctx.d_outputs,
        (void*)&ctx.d_scratch,
        (void*)&ctx.nonce_start,
        (void*)&ctx.batch_size,
        (void*)&ctx.d_difficulty_target,
        (void*)&ctx.d_solutions
    };
    
    hipError_t err = hipModuleLaunchKernel(
        kernel,
        ctx.num_blocks, 1, 1,
        ctx.block_size, 1, 1,
        shared_mem, ctx.stream,
        args, nullptr
    );

    if (err != hipSuccess) {
        fprintf(stderr, "[KERNEL LAUNCH] hipModuleLaunchKernel failed: %s (blocks=%d, threads=%d, shared=%zu)\n",
                hipGetErrorString(err), ctx.num_blocks, ctx.block_size, shared_mem);
    }

    return err == hipSuccess;
}