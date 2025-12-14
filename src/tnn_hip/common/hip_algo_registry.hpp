#pragma once
#include "gpu_algo_impl.hpp"
#include <memory>
#include <functional>

#include "tnn_hip_common_embedded.hpp"
#ifdef TNN_XELISHASH
#include "xelis_embedded_headers.hpp"
#include "xelis-hash-v3.hip.hpp"
#endif

// ============================================================================
// Xelis V3 Shared Memory Calculator
// ============================================================================
inline size_t xelis_v3_shared_mem(int block_size) {
    // return block_size * (4 * 32 + 4 * 12);  // K2 + nonces
    return 0;
}

static inline int parse_gfx_number(const char* gcnArchName) {
    if (!gcnArchName) return 0;
    // Examples: "gfx1100", "gfx1030:sramecc+:xnack-"
    const char* p = std::strstr(gcnArchName, "gfx");
    if (!p) return 0;
    p += 3;

    int n = 0;
    while (*p >= '0' && *p <= '9') {
        n = n * 10 + (*p - '0');
        ++p;
    }
    return n; // 1100, 1030, ...
}

static inline bool is_amd_rdna_plus(int device_id) {
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
    (void)device_id;
    return false;
#else
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device_id) != hipSuccess) return false;

    const int gfx = parse_gfx_number(props.gcnArchName);
    return gfx >= 1010;
#endif
}

static inline bool is_nvidia_ampere_plus(int device_id) {
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device_id) != hipSuccess) return false;

    // Ampere (RTX 3000-series) has compute capability 8.0+
    // Ada Lovelace (RTX 4000-series) is 8.9
    // Blackwell (RTX 5000-series) is 9.0+
    return (props.major >= 8);
#else
    (void)device_id;
    return false;
#endif
}

enum class ExecMode : uint8_t { Unknown = 0, Monolithic = 1, Split = 2, SplitCooperative = 3 };

static inline ExecMode choose_exec_mode_cached(int device_id) {
    // small fixed cache; HIP max devices is usually small.
    // If you want fully general, use unordered_map in TLS instead.
    thread_local std::vector<ExecMode> mode_by_dev;

    if ((int)mode_by_dev.size() <= device_id) {
        mode_by_dev.resize(device_id + 1, ExecMode::Unknown);
    }

    ExecMode& m = mode_by_dev[device_id];
    if (m != ExecMode::Unknown) return m;

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
    // NVIDIA: Use cooperative split for Ampere+ (RTX 3000-series and newer)
    m = is_nvidia_ampere_plus(device_id) ? ExecMode::SplitCooperative : ExecMode::Monolithic;
#else
    // AMD: Use cooperative split for RDNA+ (gfx1010+)
    m = is_amd_rdna_plus(device_id) ? ExecMode::SplitCooperative : ExecMode::Monolithic;
#endif
    return m;
}

// ============================================================================
// Xelis V3 Split Kernel Execution Strategy
// ============================================================================
inline bool xelis_v3_split_execute(
    const KernelMap& kernels,
    const KernelLaunchContext& ctx
) {
    int dev = 0;
    hipGetDevice(&dev); // current thread device

    const ExecMode mode = choose_exec_mode_cached(dev);

    // Monolithic path for older GPUs
    if (mode == ExecMode::Monolithic) {
        return default_monolithic_execute(kernels, ctx);
    }

    // Determine which Stage 1 and Stage 3 kernels to use
    const char* stage1_name = (mode == ExecMode::SplitCooperative)
        ? "xelis_stage1_cooperative"
        : "xelis_stage1_kernel";

    const char* s3_name = (mode == ExecMode::SplitCooperative)
        ? "xelis_s3_efficient_kernel"
        : "xelis_s3_fused_kernel";

    auto stage1_it = kernels.find(stage1_name);
    auto s3_it = kernels.find(s3_name);

    // Fallback to monolithic if kernels not found
    if (stage1_it == kernels.end() || s3_it == kernels.end()) {
        fprintf(stderr, "[XELIS] Split kernels not found (%s, %s), falling back to monolithic\n",
                stage1_name, s3_name);
        return default_monolithic_execute(kernels, ctx);
    }

    hipFunction_t stage1_kernel = stage1_it->second;
    hipFunction_t s3_kernel = s3_it->second;

    // Stage 1 block size
    // Cooperative Stage 1 is hard-capped at 32 threads
    int stage1_block_size = (mode == ExecMode::SplitCooperative)
        ? 32
        : std::min(ctx.block_size, 32);

    size_t shared_mem = (mode == ExecMode::SplitCooperative)
        ? (32 * 176)  // STAGE1_SHARED_MEM_SIZE
        : 0;

    uint32_t scratch_offset = 0;
    int stage1_num_blocks = (ctx.batch_size + stage1_block_size - 1) / stage1_block_size;

    // Launch Stage 1
    {
        void* args[] = {
            (void*)&ctx.d_input,
            (void*)&ctx.d_scratch,
            (void*)&ctx.nonce_start,
            (void*)&ctx.batch_size,
            (void*)&scratch_offset
        };

        hipError_t err = hipModuleLaunchKernel(
            stage1_kernel,
            stage1_num_blocks, 1, 1,
            stage1_block_size, 1, 1,
            shared_mem, ctx.stream,
            args, nullptr
        );
        if (err != hipSuccess) {
            fprintf(stderr, "[XELIS SPLIT] Stage1 (%s) launch failed: %s (blocks=%d, threads=%d, shared=%zu)\n",
                    stage1_name, hipGetErrorString(err), stage1_num_blocks, stage1_block_size, shared_mem);
            return false;
        }
    }

    // Launch Stage 3 (efficient or standard)
    {
        void* args[] = {
            (void*)&ctx.d_scratch,
            (void*)&ctx.d_outputs,
            (void*)&ctx.nonce_start,
            (void*)&ctx.batch_size,
            (void*)&scratch_offset,
            (void*)&ctx.d_difficulty_target,
            (void*)&ctx.d_solutions
        };

        hipError_t err = hipModuleLaunchKernel(
            s3_kernel,
            ctx.num_blocks, 1, 1,
            ctx.block_size, 1, 1,
            0, ctx.stream,
            args, nullptr
        );
        if (err != hipSuccess) {
            fprintf(stderr, "[XELIS SPLIT] Stage3 (%s) launch failed: %s (blocks=%d, threads=%d)\n",
                    s3_name, hipGetErrorString(err), ctx.num_blocks, ctx.block_size);
            return false;
        }
    }

    return true;
}

// ============================================================================
// Xelis V3 Configuration
// ============================================================================
inline AlgoConfig XELIS_V3_CONFIG = {
    .name = "xelis_v3",
    .source_path = "src/tnn_hip/crypto/xelis-hash/xelis-hash-v3.hip",
#ifdef TNN_XELISHASH
    .source = hip_xelis_v3_source::SRC_TNN_HIP_CRYPTO_XELIS_HASH_XELIS_HASH_V3_HIP_SOURCE.data(),
#else
    .source = {},
#endif

    // Multiple kernels for split execution
    .kernel_names = {
        "xelis_hash_v3_kernel",        // Primary (for tuning, fallback monolithic)
        "xelis_stage1_kernel",          // Stage 1 (standard)
        "xelis_s3_fused_kernel",        // Fused Stage 3 + Finalize (standard)
        "xelis_stage1_cooperative",     // Stage 1 Cooperative (RDNA+/Ampere+)
        "xelis_s3_efficient_kernel"     // Stage 3 Efficient (RDNA+/Ampere+)
    },

    .kernel_name = "",  // Empty - using kernel_names instead
    
#ifdef TNN_XELISHASH
    .rtc_headers = build_rtc_headers(
        hip_embedded::XELIS_SOURCES,
        hip_embedded::COMMON_HEADERS
    ),
#else
    .rtc_headers = {},
#endif
    .template_size = 112,
    .hash_size = 32,
    .nonce_size = 8,
    .scratch_per_hash = 531 * 128 * sizeof(uint64_t),
    .preferred_block_size = 64,
    .algo_id = ALGO_XELISV3,
    .calc_shared_mem = xelis_v3_shared_mem,

    .category = AlgoCategory::MemoryHard,
    .enable_reg_tuning = true,

    .amd_blocks = {32, 1024, 32},
    .nvidia_blocks = {32, 768, 32},
    .target_batch_time_ms = 2500.0,
    .max_batch_time_ms = 3000.0,
    .min_batch_time_ms = 100.0,
    .enable_autotune = true,
    .autotune_warmup = 1,
    .autotune_iterations = 2,
    .memory_reserve_mb = 128.0,
    .memory_usage_factor = 0.9,
    
    // Use split kernel execution strategy
    .execute_fn = xelis_v3_split_execute,
};

// ============================================================================
// Algorithm Registry
// ============================================================================
class AlgoRegistry {
public:
    static AlgoRegistry& instance() {
        static AlgoRegistry inst;
        return inst;
    }
    
    std::unique_ptr<IGPUAlgorithm> create(const std::string& name) {
        if (name == "xelis_v3") {
            return std::make_unique<GPUAlgorithm>(XELIS_V3_CONFIG);
        }
        return nullptr;
    }
    
    std::vector<std::string> list_algorithms() const {
        return {"xelis_v3"};
    }
};