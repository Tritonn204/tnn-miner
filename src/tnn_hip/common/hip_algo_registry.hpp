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
    return 256;  // LDS S-box for AES rounds in stage_3_hybrid_v2
}

static inline int parse_gfx_number(const char* gcnArchName) {
    if (!gcnArchName) return 0;
    const char* p = std::strstr(gcnArchName, "gfx");
    if (!p) return 0;
    p += 3;

    int n = 0;
    while (*p >= '0' && *p <= '9') {
        n = n * 10 + (*p - '0');
        ++p;
    }
    return n;
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
    return (props.major >= 8);
#else
    (void)device_id;
    return false;
#endif
}

// ============================================================================
// Xelis V3 Strategy Definitions
// ============================================================================

// Strategy indices (opaque to the generic tune system)
enum class XelisStrategy : uint8_t {
    Mono     = 0,   // s1+s3_hybrid_v2+b3 monolithic
    Baseline = 1,   // s1+s3_hybrid_v2 fused, b3 separate
    Sep      = 2,   // all 3 separate
    Neo      = 3,   // s1 separate, s3+b3 fused
};

// ============================================================================
// Xelis V3 4-Strategy Execution Dispatch
// ============================================================================

// Helper: choose which stage1 kernel to use based on GPU capabilities
static inline const char* xelis_pick_stage1(int dev) {
    bool cooperative = false;
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
    cooperative = is_nvidia_ampere_plus(dev);
#else
    cooperative = is_amd_rdna_plus(dev);
#endif
    return cooperative ? "xelis_stage1_cooperative" : "xelis_stage1_kernel";
}

// Helper: launch stage1 kernel
static inline bool xelis_launch_stage1(
    const KernelMap& kernels,
    const KernelLaunchContext& ctx,
    int dev)
{
    const char* stage1_name = xelis_pick_stage1(dev);
    auto it = kernels.find(stage1_name);
    if (it == kernels.end()) return false;

    bool cooperative = (std::strcmp(stage1_name, "xelis_stage1_cooperative") == 0);

    int stage1_block_size = cooperative ? 32 : std::min(ctx.block_size, 32);
    size_t shared_mem = cooperative ? (32 * 176) : 0;
    uint32_t scratch_offset = 0;
    int stage1_num_blocks = (ctx.batch_size + stage1_block_size - 1) / stage1_block_size;

    void* args[] = {
        (void*)&ctx.d_input,
        (void*)&ctx.d_scratch,
        (void*)&ctx.nonce_start,
        (void*)&ctx.batch_size,
        (void*)&scratch_offset
    };

    hipError_t err = hipModuleLaunchKernel(
        it->second,
        stage1_num_blocks, 1, 1,
        stage1_block_size, 1, 1,
        shared_mem, ctx.stream,
        args, nullptr
    );
    if (err != hipSuccess) {
        fprintf(stderr, "[XELIS] Stage1 (%s) launch failed: %s\n",
                stage1_name, hipGetErrorString(err));
        return false;
    }
    return true;
}

// Helper: launch blake3 batch kernel
static inline bool xelis_launch_blake3(
    const KernelMap& kernels,
    const KernelLaunchContext& ctx)
{
    auto it = kernels.find("xelis_blake3_batch");
    if (it == kernels.end()) return false;

    uint32_t scratch_offset = 0;
    int blake3_block_size = 256;
    int blake3_num_blocks = (ctx.batch_size + blake3_block_size - 1) / blake3_block_size;

    void* args[] = {
        (void*)&ctx.d_scratch,
        (void*)&ctx.d_outputs,
        (void*)&ctx.batch_size,
        (void*)&scratch_offset,
        (void*)&ctx.d_difficulty_target,
        (void*)&ctx.d_solutions,
        (void*)&ctx.nonce_start
    };

    hipError_t err = hipModuleLaunchKernel(
        it->second,
        blake3_num_blocks, 1, 1,
        blake3_block_size, 1, 1,
        0, ctx.stream,
        args, nullptr
    );
    if (err != hipSuccess) {
        fprintf(stderr, "[XELIS] Blake3 launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

inline bool xelis_v3_execute(
    const KernelMap& kernels,
    const KernelLaunchContext& ctx
) {
    int dev = 0;
    hipGetDevice(&dev);

    const auto strategy = static_cast<XelisStrategy>(ctx.strategy);
    uint32_t scratch_offset = 0;

    switch (strategy) {

    case XelisStrategy::Mono:
        return default_monolithic_execute(kernels, ctx);

    case XelisStrategy::Baseline: {
        // s1+s3 fused, then blake3 separate
        auto it = kernels.find("xelis_s13_noblake_kernel");
        if (it == kernels.end()) return default_monolithic_execute(kernels, ctx);

        void* args[] = {
            (void*)&ctx.d_input,
            (void*)&ctx.d_scratch,
            (void*)&ctx.nonce_start,
            (void*)&ctx.batch_size,
            (void*)&scratch_offset
        };

        hipError_t err = hipModuleLaunchKernel(
            it->second,
            ctx.num_blocks, 1, 1,
            ctx.block_size, 1, 1,
            0, ctx.stream,
            args, nullptr
        );
        if (err != hipSuccess) {
            fprintf(stderr, "[XELIS] s13_noblake launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return xelis_launch_blake3(kernels, ctx);
    }

    case XelisStrategy::Sep: {
        // stage1 separate, s3 separate, blake3 separate
        if (!xelis_launch_stage1(kernels, ctx, dev)) return false;

        auto it = kernels.find("xelis_s3_hybrid_v2_noblake_kernel");
        if (it == kernels.end()) return false;

        void* args[] = {
            (void*)&ctx.d_scratch,
            (void*)&ctx.batch_size,
            (void*)&scratch_offset,
            (void*)&ctx.d_difficulty_target,
            (void*)&ctx.d_solutions,
            (void*)&ctx.nonce_start
        };

        hipError_t err = hipModuleLaunchKernel(
            it->second,
            ctx.num_blocks, 1, 1,
            ctx.block_size, 1, 1,
            0, ctx.stream,
            args, nullptr
        );
        if (err != hipSuccess) {
            fprintf(stderr, "[XELIS] s3_hybrid_v2_noblake launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return xelis_launch_blake3(kernels, ctx);
    }

    case XelisStrategy::Neo: {
        // stage1 separate, s3+b3 fused
        if (!xelis_launch_stage1(kernels, ctx, dev)) return false;

        auto it = kernels.find("xelis_s3b3_hybrid_v2_kernel");
        if (it == kernels.end()) return false;

        void* args[] = {
            (void*)&ctx.d_scratch,
            (void*)&ctx.d_outputs,
            (void*)&ctx.batch_size,
            (void*)&scratch_offset,
            (void*)&ctx.d_difficulty_target,
            (void*)&ctx.d_solutions,
            (void*)&ctx.nonce_start
        };

        hipError_t err = hipModuleLaunchKernel(
            it->second,
            ctx.num_blocks, 1, 1,
            ctx.block_size, 1, 1,
            0, ctx.stream,
            args, nullptr
        );
        if (err != hipSuccess) {
            fprintf(stderr, "[XELIS] s3b3_hybrid_v2 launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    default:
        return default_monolithic_execute(kernels, ctx);
    }
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

    // All kernels used across strategies
    .kernel_names = {
        "xelis_hash_v3_kernel",              // Mono (primary/fallback)
        "xelis_stage1_kernel",               // Sep/Neo stage1
        "xelis_stage1_cooperative",          // Sep/Neo stage1 (RDNA+/Ampere+)
        "xelis_s13_noblake_kernel",          // Baseline (s1+s3 fused)
        "xelis_s3_hybrid_v2_noblake_kernel", // Sep (s3 only)
        "xelis_s3b3_hybrid_v2_kernel",       // Neo (s3+b3 fused)
        "xelis_blake3_batch"                 // Baseline/Sep blake3
    },

    .kernel_name = "",

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
    .scratch_per_hash = (531 * 128 + 1) * sizeof(uint64_t),  // +1 for nonce storage
    .preferred_block_size = 64,
    .algo_id = ALGO_XELISV3,
    .calc_shared_mem = xelis_v3_shared_mem,

    .category = AlgoCategory::MemoryHard,
    .enable_reg_tuning = true,

    .amd_blocks = {32, 1024, 32},
    .nvidia_blocks = {32, 1024, 32},
    .target_batch_time_ms = 1250.0,
    .max_batch_time_ms = 1500.0,
    .min_batch_time_ms = 100.0,
    .enable_autotune = true,
    .autotune_warmup = 1,
    .autotune_iterations = 1,
    .memory_reserve_mb = 128.0,
    .memory_usage_factor = 1.0,

    .execute_fn = xelis_v3_execute,

    // 4 strategies for autotune sweep
    .strategy_variants = {
        (uint8_t)XelisStrategy::Mono,
        (uint8_t)XelisStrategy::Baseline,
        (uint8_t)XelisStrategy::Sep,
        (uint8_t)XelisStrategy::Neo
    },
    .strategy_names = {"Mono", "Baseline", "Sep", "Neo"},
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
