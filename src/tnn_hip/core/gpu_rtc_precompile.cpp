// gpu_rtc_precompile.cpp
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>

#include <tnn-common.hpp>
#include <tnn_log.hpp>

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
    TNN_LOG_DEBUG("[PRECOMPILE] HIP or TNN_XELISHASH not enabled — skipping.\n");
    return false;
#else

    TNN_LOG_DEBUG("[PRECOMPILE] GPU kernel precompile (main thread)\n");
#ifdef _WIN32
    TNN_LOG_DEBUG("[PRECOMPILE] Thread ID: %lu\n", (unsigned long)GetCurrentThreadId());
#endif

    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount == 0) {
        TNN_LOG_ERROR("[PRECOMPILE] No HIP devices found (err=%d)\n", err);
        return false;
    }

    TNN_LOG_DEBUG("[PRECOMPILE] Found %d HIP device(s)\n", deviceCount);
    for (int d = 0; d < deviceCount; ++d) {
        hipDeviceProp_t props{};
        (void)hipGetDeviceProperties(&props, d);
        TNN_LOG_DEBUG("[PRECOMPILE]   Device %d: %s%s\n", d, props.name,
               shouldUseDevice(d) ? "" : " (skipped)");
    }

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
    {
        TNN_LOG_DEBUG("[PRECOMPILE] Precreating CUDA/HIP context...\n");
        (void)hipSetDevice(0);
        void* p = nullptr;
        if (hipMalloc(&p, 256) == hipSuccess) (void)hipFree(p);
    }
#endif

    const int algo = miningProfile.coin.miningAlgo;
    TNN_LOG_DEBUG("[PRECOMPILE] miningAlgo = %d\n", algo);

    RTCCompiler& rtc = RTCCompiler::instance();
    bool ok = true;

    switch (algo) {

    case ALGO_XELISV2:
    case ALGO_XELISV3:
    {
        TNN_LOG_DEBUG("[PRECOMPILE] Selected Xelis algorithm\n");

        AlgoConfig cfg = XELIS_V3_CONFIG;

        TNN_LOG_DEBUG("[PRECOMPILE] Source size = %zu bytes, headers = %zu\n",
                      cfg.source.size(), cfg.rtc_headers.size());

        for (const auto& header : cfg.rtc_headers) {
            rtc.add_header_source(std::string(header.name), std::string(header.source));
        }

        for (int d = 0; d < deviceCount; ++d) {
            TNN_GPU_GATE(d)
            hipDeviceProp_t props{};
            (void)hipGetDeviceProperties(&props, d);

            TNN_LOG_INFO("[PRECOMPILE] Device %d: %s — compiling kernels...\n", d, props.name);

            hipError_t setErr = hipSetDevice(d);
            if (setErr != hipSuccess) {
                TNN_LOG_ERROR("[PRECOMPILE] hipSetDevice(%d) failed: %s\n",
                              d, hipGetErrorString(setErr));
                ok = false;
                continue;
            }

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
            {
                void* p = nullptr;
                hipError_t ce = hipMalloc(&p, 256);
                if (ce == hipSuccess) (void)hipFree(p);
                else {
                    TNN_LOG_ERROR("[PRECOMPILE] Context init failed on device %d: %s\n",
                                  d, hipGetErrorString(ce));
                    ok = false;
                    continue;
                }
            }
#endif

#if defined(__HIP_PLATFORM_AMD__)
            const BlockSizeLimits& limits = cfg.amd_blocks;
#else
            const BlockSizeLimits& limits = cfg.nvidia_blocks;
#endif
            const int min_wg = limits.block_min;
            const int max_wg = limits.block_max;

            TNN_LOG_DEBUG("[PRECOMPILE] Block limits: min=%d, max=%d, step=%d\n",
                          limits.block_min, limits.block_max, limits.step);

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
            {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "sm_%d%d", props.major, props.minor);
                rtc.set_gpu_arch(buf);
                TNN_LOG_DEBUG("[PRECOMPILE] arch=%s\n", buf);
            }
#else
            rtc.set_gpu_arch(props.gcnArchName);
            TNN_LOG_DEBUG("[PRECOMPILE] arch=%s\n", props.gcnArchName);
#endif

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

            {
                int s3_nreg = (props.major <= 7) ? 56 : 40;
                opts.push_back("-DXELIS_S3_NREG=" + std::to_string(s3_nreg));
            }

            opts.push_back("-DDEVICE_ID=" + std::to_string(d));
#endif

            try {
                std::string primary_kernel = cfg.get_primary_kernel();

                TNN_LOG_DEBUG("[PRECOMPILE] Compiling (primary: %s)\n", primary_kernel.c_str());

                auto compiled = rtc.compile_from_source(
                    std::string(cfg.source),
                    cfg.source_path,
                    primary_kernel,
                    opts
                );

                TNN_LOG_DEBUG("[PRECOMPILE] JIT compiled, module=%p\n", (void*)compiled.module);

                int loaded_count = 0;
                for (const auto& kname : cfg.get_kernel_names()) {
                    hipFunction_t func = nullptr;
                    hipError_t herr = hipModuleGetFunction(&func, compiled.module, kname.c_str());
                    if (herr == hipSuccess && func) {
                        TNN_LOG_DEBUG("[PRECOMPILE]   OK: '%s'\n", kname.c_str());
                        ++loaded_count;
                    } else {
                        TNN_LOG_ERROR("[PRECOMPILE]   MISSING: '%s' : %s\n",
                                      kname.c_str(), hipGetErrorString(herr));
                        ok = false;
                    }
                }

                if (loaded_count == 0) {
                    TNN_LOG_ERROR("[PRECOMPILE] No kernels loaded from module on device %d\n", d);
                    ok = false;
                } else {
                    TNN_LOG_INFO("[PRECOMPILE] Device %d: %d kernel(s) loaded\n", d, loaded_count);
                }
            }
            catch (const std::exception& e) {
                TNN_LOG_ERROR("[PRECOMPILE] Device %d precompile failed: %s\n", d, e.what());
                ok = false;
            }
        }

        break;
    }

    default:
        TNN_LOG_DEBUG("[PRECOMPILE] No RTC precompile required for algo %d\n", algo);
        break;
    }

    TNN_LOG_INFO("[PRECOMPILE] %s\n", ok ? "All kernels compiled" : "FAILED");

    return ok;

#endif // TNN_HIP + TNN_XELISHASH
}
