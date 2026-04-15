#pragma once
#include <tnn_hip/common/gpu_compat.hpp>
#include "oro_seh_wrappers.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <condition_variable>
#include <tnn_log.hpp>
#include <atomic>

inline int parse_gfx_number(const char *gcnArchName)
{
  if (!gcnArchName)
    return 0;
  const char *p = std::strstr(gcnArchName, "gfx");
  if (!p)
    return 0;
  p += 3;

  int n = 0;
  while (*p >= '0' && *p <= '9')
  {
    n = n * 10 + (*p - '0');
    ++p;
  }
  return n;
}

inline bool is_amd_rdna_plus(int device_id) {
    if (!tnn_is_amd_device(device_id)) return false;
    oroDeviceProp_t props{};
    if (oroGetDeviceProperties(&props, tnn_get_device(device_id)) != oroSuccess) return false;
    const int gfx = parse_gfx_number(props.gcnArchName);
    return gfx >= 1010;
}

inline bool is_nvidia_ampere_plus(int device_id) {
    if (!tnn_is_nvidia_device(device_id)) return false;
    oroDeviceProp_t props{};
    if (oroGetDeviceProperties(&props, tnn_get_device(device_id)) != oroSuccess) return false;
    return (props.major >= 8);
}

inline std::atomic<bool> g_autotune_stop{false};

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
    uint8_t strategy = 0;      // Algo-defined strategy index (0 = default/first)
    bool valid = false;

    // Algo-specific tune keys (e.g., "s1_knee_batch" for xelis).
    // Probed after main autotune sweep, cached to disk as "tune.<key>=<value>".
    std::unordered_map<std::string, int64_t> tune_keys;

    int64_t get_tune_key(const std::string& key, int64_t default_val = 0) const {
        auto it = tune_keys.find(key);
        return (it != tune_keys.end()) ? it->second : default_val;
    }

    std::string describe() const {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "strategy=%u, block_size=%d, num_blocks=%d, batch_size=%u, hashrate=%.2f H/s, time=%.2fms",
            strategy, block_size, num_blocks, batch_size, hashrate, batch_time_ms);
        std::string s(buf);
        for (const auto& [k, v] : tune_keys) {
            char kbuf[128];
            snprintf(kbuf, sizeof(kbuf), ", %s=%lld", k.c_str(), (long long)v);
            s += kbuf;
        }
        return s;
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

    // Algo-defined strategy index (from autotune)
    uint8_t strategy = 0;

    // Config reference
    const struct AlgoConfig* config;

    // Stream (nullptr = default stream)
    oroStream_t stream = nullptr;

    // Module handle (for oroModuleGetGlobal — e.g., texture object setup)
    oroModule_t module = nullptr;

    // Algo-specific opaque state (e.g., KawPow DAG pointer + metadata)
    void* algo_data = nullptr;

    // Algo-specific tune keys (pointer to TuningResult's map, non-owning)
    const std::unordered_map<std::string, int64_t>* tune_keys = nullptr;

    int64_t get_tune_key(const std::string& key, int64_t default_val = 0) const {
        if (!tune_keys) return default_val;
        auto it = tune_keys->find(key);
        return (it != tune_keys->end()) ? it->second : default_val;
    }
};

// Forward declaration
struct AlgoConfig;

// ============================================================================
// Kernel Execution Strategy
// ============================================================================
using KernelMap = std::unordered_map<std::string, oroFunction_t>;

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
    
    static std::string make_tune_key(const oroDeviceProp_t& props,
                                      const std::string& algo_name,
                                      int device_id = 0) {
        std::string vendor;
        std::string arch;

        if (tnn_is_amd_device(device_id)) {
            vendor = "amd";
            arch = props.gcnArchName;
        } else {
            vendor = "nvidia";
            char buf[32];
            snprintf(buf, sizeof(buf), "compute_%d%d", props.major, props.minor);
            arch = buf;
        }
        
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
// Tune key probe callback — runs after main autotune sweep to discover
// algo-specific parameters (e.g., bandwidth knee, optimal sub-batch sizes).
// Writes results into result.tune_keys. Returns true on success.
// ============================================================================
using TuneKeyProbeFn = std::function<bool(
    const KernelMap& kernels,
    const oroDeviceProp_t& device_props,
    int compute_units,
    oroStream_t stream,
    const struct AlgoConfig& config,
    TuningResult& result,
    int device_id
)>;

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
    int batch_step_denom = 2;  // batch multiplier denominator: 2=half-steps (1x,1.5x,2x), 4=quarter-steps (1x,1.25x,1.5x,...)
    int variance_warmup = 2;      // warmup launches per variant in post-tune variance bench
    int variance_iterations = 20; // timed launches per variant in post-tune variance bench

    // Memory overhead factor
    double memory_reserve_mb = 128.0;
    double memory_usage_factor = 0.9;
    
    // ========== Execution Strategy ==========

    // Custom execution function (nullptr = use default monolithic)
    // The execute_fn receives ctx.strategy to know which variant to dispatch.
    KernelExecuteFn execute_fn = nullptr;

    // Strategy variants for autotune sweep.
    // If non-empty, autotune tries each strategy × block_size and picks the best.
    // Values are opaque uint8_t indices interpreted by execute_fn.
    // Empty = no strategy sweep (single strategy, strategy=0).
    std::vector<uint8_t> strategy_variants;

    // Human-readable names for each strategy (for logging). Same order as strategy_variants.
    std::vector<std::string> strategy_names;

    // Bottleneck kernel name per strategy for occupancy queries.
    // Maps strategy index (same order as strategy_variants) to kernel name.
    // If empty or missing entry, falls back to primary kernel.
    std::vector<std::string> strategy_bottleneck_kernels;

    // Combined occupancy score threshold for block size filtering.
    // Score = sqrt((threads/peak_achievable) × (threads/hw_max_threads)).
    // Geometric mean of register-aware and hardware-anchored occupancy.
    // Portable across kernel complexity without per-algo hand-tuning.
    double occupancy_threshold = 0.70;

    // Bottleneck-only execution for autotune sweep (nullptr = use execute_fn).
    // When set, autotune times only the bottleneck kernel (e.g. s3 for Sep)
    // instead of the full pipeline. Tune probes then handle edge kernels (s1, b3).
    KernelExecuteFn bottleneck_execute_fn = nullptr;

    // Pre-timing setup for bottleneck autotune (nullptr = no setup needed).
    // Runs before the timer starts each iteration — e.g. fills scratchpad via s1
    // so the bottleneck kernel sees realistic data-dependent access patterns.
    // Must sync the stream before returning.
    KernelExecuteFn bottleneck_setup_fn = nullptr;

    // Post-sweep tune key probe (nullptr = no extra probing)
    TuneKeyProbeFn tune_key_probe_fn = nullptr;

    // Source transformation — called before RTC compile to modify kernel source.
    // Used by KawPow to inject the random program + coin padding.
    using SourceTransformFn = std::function<std::string(const std::string& source, int device_id)>;
    SourceTransformFn source_transform_fn = nullptr;

    // Pre-tune setup — called after compile, before autotune sweep.
    // Allocates algo-specific resources (e.g., KawPow DAG) and sets *algo_data.
    using PreTuneFn = std::function<bool(const KernelMap& kernels, const oroDeviceProp_t& props,
                                          int device_id, void** algo_data)>;
    PreTuneFn pre_tune_fn = nullptr;

    // Occupancy-based tune — when set, skips the autotune sweep entirely.
    // Returns {block_size, batch_size} via the TuningResult. Called after pre_tune_fn.
    // If it returns true, configure_batch uses the result directly (no sweep).
    using OccupancyTuneFn = std::function<bool(TuningResult& result,
                                               const oroDeviceProp_t& props,
                                               int device_id, void* algo_data,
                                               double memory_reserve_mb,
                                               double memory_usage_factor)>;
    OccupancyTuneFn occupancy_tune_fn = nullptr;

    // Post-tune — called after autotune completes with the winning result.
    // Use for variance measurement, diagnostics, etc.
    using PostTuneFn = std::function<void(const TuningResult& result,
                                          const oroDeviceProp_t& props,
                                          int device_id, void* algo_data,
                                          int variance_warmup, int variance_iters)>;
    PostTuneFn post_tune_fn = nullptr;

    // Cleanup for algo_data (called during GPUAlgorithm::cleanup)
    using AlgoDataCleanupFn = std::function<void(void* algo_data)>;
    AlgoDataCleanupFn algo_data_cleanup_fn = nullptr;

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
// Job snapshot - captures all job state at batch start
// ============================================================================
struct JobSnapshot {
    std::vector<uint8_t> work_template;  // Copy of template data
    int64_t job_id;                      // Job ID/height
    uint64_t difficulty;                 // Difficulty for this job
    int algo_id;                         // Algorithm identifier (e.g., ALGO_XELISV3)
    std::string job_id_str;             // Protocol-assigned job ID (stratum)
    bool is_dev;                         // Whether this batch was mined for dev fee

    JobSnapshot() : job_id(0), difficulty(0), algo_id(0), is_dev(false) {}

    JobSnapshot(const uint8_t* template_data, size_t template_size,
                int64_t id, uint64_t diff, int algo,
                const std::string& id_str = "",
                bool dev = false)
        : work_template(template_data, template_data + template_size)
        , job_id(id)
        , difficulty(diff)
        , algo_id(algo)
        , job_id_str(id_str)
        , is_dev(dev)
    {}
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

    // GPU context created during initialize(), needed by worker threads
    virtual oroCtx get_ctx() const = 0;

    virtual void set_work(const uint8_t* work_template, uint64_t difficulty) = 0;
    virtual const uint8_t* get_current_work_template() const = 0;
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
    bool force_retune = false;
    std::set<int> retune_devices;  // empty = retune all (when force_retune is set)

    bool should_retune(int device_id) const {
        if (!force_retune) return false;
        if (retune_devices.empty()) return true;
        return retune_devices.count(device_id) > 0;
    }
    
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
    const oroDeviceProp_t& props,
    int device_id = 0)
{
    if (!tnn_is_nvidia_device(device_id) || !cfg.enable_reg_tuning)
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

    oroFunction_t kernel = it->second;

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

    TNN_LOG_TRACE("[LAUNCH] kernel=%p, grid=%d, block=%d, shared=%zu, stream=%p\n",
                  (void*)kernel, ctx.num_blocks, ctx.block_size, shared_mem, (void*)ctx.stream);
    TNN_LOG_TRACE("[LAUNCH] d_input=%p, d_outputs=%p, d_scratch=%p, d_target=%p, d_solutions=%p\n",
                  (void*)ctx.d_input, (void*)ctx.d_outputs, (void*)ctx.d_scratch,
                  (void*)ctx.d_difficulty_target, (void*)ctx.d_solutions);
    TNN_LOG_TRACE("[LAUNCH] nonce_start=%llu, batch_size=%u\n",
                  (unsigned long long)ctx.nonce_start, ctx.batch_size);
    fflush(stdout);

    oroError_t err = oro_safe_launch(
        kernel,
        ctx.num_blocks, 1, 1,
        ctx.block_size, 1, 1,
        shared_mem, ctx.stream,
        args, nullptr
    );

    TNN_LOG_TRACE("[LAUNCH] oroModuleLaunchKernel returned %d (%s)\n",
                  (int)err, tnn_error_string(err));
    fflush(stdout);

    if (err != oroSuccess) {
        TNN_LOG_ERROR("[KERNEL LAUNCH] oroModuleLaunchKernel failed: %s (blocks=%d, threads=%d, shared=%zu)\n",
                tnn_error_string(err), ctx.num_blocks, ctx.block_size, shared_mem);
    }

    return err == oroSuccess;
}