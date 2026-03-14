// gpu_rtc_precompile.cpp
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>

#include <tnn-common.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef TNN_HIP

#include <hip/hip_runtime.h>

// Algo system (manifests + AlgoConfig)
#include "../common/gpu_algo.hpp"
#include "../common/gpu_rtc.hpp"
#include "../common/hip_algo_registry.hpp"

// Xelis manifests + sources (auto-generated)
#include "tnn_hip_common_embedded.hpp"
#ifdef TNN_XELISHASH
#include "xelis_embedded_headers.hpp"
#include "xelis-hash-v3.hip.hpp"
#endif

#endif // TNN_HIP

extern "C" bool precompile_all_kernels()
{
#if !defined(TNN_HIP) || !defined(TNN_XELISHASH)
    printf("[PRECOMPILE] HIP or TNN_XELISHASH not enabled — skipping.\n");
    fflush(stdout);
    return false;
#else

    printf("\n");
    printf("========================================\n");
    printf("[PRECOMPILE] GPU kernel precompile (main thread)\n");
#ifdef _WIN32
    printf("[PRECOMPILE] Thread ID: %lu\n", (unsigned long)GetCurrentThreadId());
#endif
    printf("========================================\n");
    fflush(stdout);

    //----------------------------------------------------------------------
    // STEP 1: Enumerate devices
    //----------------------------------------------------------------------
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount == 0) {
        printf("[PRECOMPILE] No HIP devices found (err=%d)\n", err);
        fflush(stdout);
        return false;
    }

    printf("[PRECOMPILE] Found %d HIP device(s)\n", deviceCount);
    for (int d = 0; d < deviceCount; ++d) {
        hipDeviceProp_t props{};
        (void)hipGetDeviceProperties(&props, d);
        printf("[PRECOMPILE]   Device %d: %s\n", d, props.name);
    }
    fflush(stdout);

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
    // Force context creation early (fixes NV + Windows hazards)
    {
        printf("[PRECOMPILE] Precreating CUDA/HIP context...\n");
        (void)hipSetDevice(0);
        void* p = nullptr;
        if (hipMalloc(&p, 256) == hipSuccess) (void)hipFree(p);
        fflush(stdout);
    }
#endif

    //----------------------------------------------------------------------
    // STEP 2: Decide which algorithm to precompile
    //----------------------------------------------------------------------
    const int algo = miningProfile.coin.miningAlgo;
    printf("[PRECOMPILE] miningAlgo = %d\n", algo);
    fflush(stdout);

    RTCCompiler& rtc = RTCCompiler::instance();
    bool ok = true;

    switch (algo) {

    case ALGO_XELISV2:
    case ALGO_XELISV3:
    {
        printf("[PRECOMPILE] Selected Xelis algorithm — building config...\n");
        fflush(stdout);

        AlgoConfig cfg = XELIS_V3_CONFIG;

        printf("[PRECOMPILE] Source size = %zu bytes\n", cfg.source.size());
        printf("[PRECOMPILE] Header count = %zu\n", cfg.rtc_headers.size());
        fflush(stdout);

        // Register headers once (RTCCompiler::add_header_source is idempotent by include_name)
        for (const auto& header : cfg.rtc_headers) {
            rtc.add_header_source(std::string(header.name), std::string(header.source));
        }

        // Precompile per device (per-props, per-arch, per-maxregs)
        for (int d = 0; d < deviceCount; ++d) {
            hipDeviceProp_t props{};
            (void)hipGetDeviceProperties(&props, d);

            printf("\n");
            printf("------------------------------------------------\n");
            printf("[PRECOMPILE] Device %d: %s\n", d, props.name);
            printf("------------------------------------------------\n");
            fflush(stdout);

            hipError_t setErr = hipSetDevice(d);
            if (setErr != hipSuccess) {
                printf("[PRECOMPILE] ERROR: hipSetDevice(%d) failed: %s\n",
                       d, hipGetErrorString(setErr));
                ok = false;
                continue;
            }

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
            // Ensure context exists on this device too
            {
                void* p = nullptr;
                hipError_t ce = hipMalloc(&p, 256);
                if (ce == hipSuccess) (void)hipFree(p);
                else {
                    printf("[PRECOMPILE] ERROR: context init hipMalloc failed on device %d: %s\n",
                           d, hipGetErrorString(ce));
                    ok = false;
                    continue;
                }
            }
#endif

            // Pick per-backend block-size range => WG hints
#if defined(__HIP_PLATFORM_AMD__)
            const BlockSizeLimits& limits = cfg.amd_blocks;
#else
            const BlockSizeLimits& limits = cfg.nvidia_blocks;
#endif
            const int min_wg = limits.block_min;
            const int max_wg = limits.block_max;

            printf("[PRECOMPILE] Block size limits: min=%d, max=%d, step=%d\n",
                   limits.block_min, limits.block_max, limits.step);
            printf("[PRECOMPILE] WG hints: XELIS_MIN_WG=%d, XELIS_MAX_WG=%d\n",
                   min_wg, max_wg);
            fflush(stdout);

            // Set compiler arch for THIS device so caching separates by architecture
            // (RTCCompiler already knows how to format for NVIDIA/AMD)
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
            {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "sm_%d%d", props.major, props.minor);
                rtc.set_gpu_arch(buf);
                printf("[PRECOMPILE] Using arch=%s\n", buf);
            }
#else
            // AMD HIP provides gcnArchName (e.g. gfx1100)
            rtc.set_gpu_arch(props.gcnArchName);
            printf("[PRECOMPILE] Using arch=%s\n", props.gcnArchName);
#endif
            fflush(stdout);

            // Build options (per device)
            std::vector<std::string> opts;

#if defined(__HIP_PLATFORM_AMD__)
            opts = {"-O3", "-mno-cumode", "-ffast-math"};
            opts.push_back("-DXELIS_MIN_WG=" + std::to_string(min_wg));
            opts.push_back("-DXELIS_MAX_WG=" + std::to_string(max_wg));
#else
            opts = {"--dopt=on", "--use_fast_math"};
#ifdef __linux__
            opts.push_back("--device-int128");
#endif
            opts.push_back("-DXELIS_MIN_WG=" + std::to_string(min_wg));
            opts.push_back("-DXELIS_MAX_WG=" + std::to_string(max_wg));

            // Arch-dependent stage3 register cap (must match gpu_algo_impl.hpp)
            {
                int s3_nreg = (props.major <= 7) ? 56 : 40;
                opts.push_back("-DXELIS_S3_NREG=" + std::to_string(s3_nreg));
            }

            // Per-device module key (NVIDIA modules are bound to a CUDA context)
            opts.push_back("-DDEVICE_ID=" + std::to_string(d));
#endif
            fflush(stdout);

            try {
                std::string primary_kernel = cfg.get_primary_kernel();

                printf("[PRECOMPILE] Compiling module (primary kernel: %s)\n",
                       primary_kernel.c_str());
                fflush(stdout);

                auto compiled = rtc.compile_from_source(
                    std::string(cfg.source),
                    cfg.source_path,
                    primary_kernel,
                    opts
                );

                printf("[PRECOMPILE] JIT compiled.\n");
                printf("[PRECOMPILE]   module = %p\n", (void*)compiled.module);
                printf("[PRECOMPILE]   primary function = %p\n", (void*)compiled.function);
                fflush(stdout);

                // Verify all expected kernels exist in the module
                int loaded_count = 0;
                for (const auto& kname : cfg.get_kernel_names()) {
                    hipFunction_t func = nullptr;
                    hipError_t herr = hipModuleGetFunction(&func, compiled.module, kname.c_str());
                    if (herr == hipSuccess && func) {
                        printf("[PRECOMPILE]   OK: '%s' @ %p\n", kname.c_str(), (void*)func);
                        ++loaded_count;
                    } else {
                        printf("[PRECOMPILE]   MISSING: '%s' : %s\n",
                               kname.c_str(), hipGetErrorString(herr));
                        ok = false;
                    }
                }

                if (loaded_count == 0) {
                    printf("[PRECOMPILE]   ERROR: No kernels loaded from module.\n");
                    ok = false;
                } else {
                    printf("[PRECOMPILE]   Total kernels loaded: %d\n", loaded_count);
                }
                fflush(stdout);
            }
            catch (const std::exception& e) {
                printf("[PRECOMPILE] ERROR: device %d precompile failed: %s\n", d, e.what());
                fflush(stdout);
                ok = false;
            }
        }

        break;
    }

    default:
        printf("[PRECOMPILE] No RTC precompile required for algo %d\n", algo);
        fflush(stdout);
        break;
    }

    printf("\n");
    printf("========================================\n");
    printf("[PRECOMPILE] Result: %s\n", ok ? "SUCCESS" : "FAILURE");
    printf("========================================\n\n");
    fflush(stdout);

    return ok;

#endif // TNN_HIP + TNN_XELISHASH
}
