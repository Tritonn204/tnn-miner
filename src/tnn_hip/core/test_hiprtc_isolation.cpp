#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef TNN_HIP
#include <tnn_hip/common/gpu_compat.hpp>

// Embedded sources / manifests
#include "tnn_hip_common_embedded.hpp"
#ifdef TNN_XELISHASH
#include "xelis_embedded_headers.hpp"
#include "xelis-hash-v3.hip.hpp"
#endif

// RTCCompiler (inline) and manifest helpers used by miner
#include "../common/gpu_rtc.hpp"         // RTCCompiler
#include "../common/gpu_algo.hpp" // RTCHeader, build_rtc_headers, AlgoConfig, etc.

#endif // TNN_HIP

// ============================================================================
// Shared helpers for RTCCompiler path (used by isolation test + precompile)
// ============================================================================

#if defined(TNN_HIP) && defined(TNN_XELISHASH)

// Register all headers in a manifest vector into RTCCompiler.
static void register_rtc_headers_with_compiler(
    RTCCompiler& compiler,
    const std::vector<RTCHeader>& rtc_headers)
{
    printf("[RTC-HEADERS] Registering %zu manifest headers with RTCCompiler...\n",
           rtc_headers.size());
    fflush(stdout);

    for (const auto& h : rtc_headers) {
        compiler.add_header_source(std::string(h.name), std::string(h.source));
    }
}

// Build the same manifests the miner uses and register them.
static void register_xelis_manifests_once(RTCCompiler& compiler) {
    static bool done = false;
    if (done) return;
    done = true;

    printf("[RTC-HEADERS] Building manifests via build_rtc_headers...\n");
    fflush(stdout);

    // Adjust these manifest names if your generator uses different ones.
    auto rtc_headers = build_rtc_headers(
        hip_embedded::XELIS_SOURCES,
        hip_embedded::COMMON_HEADERS
    );

    register_rtc_headers_with_compiler(compiler, rtc_headers);
}

// Helper that compiles the Xelis v3 kernel via RTCCompiler using the same
// manifest + options pattern as the miner.
static void compile_xelis_v3_via_rtc(const std::vector<std::string>& extra_opts) {
    auto& compiler = RTCCompiler::instance();

    // 1) ensure manifests are registered (only once)
    register_xelis_manifests_once(compiler);

    // 2) embedded kernel source
    const std::string xelis_source =
        std::string(hip_xelis_v3_source::SRC_TNN_HIP_CRYPTO_XELIS_HASH_XELIS_HASH_V3_HIP_SOURCE);

    printf("[RTC-XELIS] Compiling Xelis v3 via RTCCompiler\n");
    printf("[RTC-XELIS]   source size: %zu bytes\n", xelis_source.size());
    printf("[RTC-XELIS]   options:\n");
    for (size_t i = 0; i < extra_opts.size(); ++i) {
        printf("[RTC-XELIS]     [%zu] %s\n", i, extra_opts[i].c_str());
    }
    fflush(stdout);

    auto kernel = compiler.compile_from_source(
        xelis_source,
        "src/tnn_hip/crypto/xelis-hash/xelis-hash-v3.hip",
        "xelis_hash_v3_kernel",
        extra_opts
    );

    printf("[RTC-XELIS] RTCCompiler successfully compiled & loaded Xelis v3\n");
    printf("[RTC-XELIS]   kernel_name: %s\n", kernel.kernel_name.c_str());
    printf("[RTC-XELIS]   module: %p, function: %p\n",
           (void*)kernel.module, (void*)kernel.function);
    fflush(stdout);
}

#endif // TNN_HIP && TNN_XELISHASH

// ============================================================================
// Isolation test
// ============================================================================
extern "C" void test_hiprtc_isolation() {
    printf("\n");
    printf("========================================\n");
    printf("[ISOLATION TEST] Starting HIPRTC isolation test\n");
    printf("[ISOLATION TEST] This runs BEFORE any other GPU initialization\n");
#ifdef _WIN32
    printf("[ISOLATION TEST] Thread ID: %lu\n", (unsigned long)GetCurrentThreadId());
    printf("[ISOLATION TEST] Process ID: %lu\n", (unsigned long)GetCurrentProcessId());
#endif
    printf("========================================\n");
    fflush(stdout);

#ifdef TNN_HIP
    printf("[ISOLATION TEST] Step 1: Checking HIP runtime initialization\n");
    fflush(stdout);

    int deviceCount = 0;
    oroError_t gpu_err = oroGetDeviceCount(&deviceCount);
    printf("[ISOLATION TEST] oroGetDeviceCount returned: %d (error code: %d)\n", deviceCount, gpu_err);
    if (gpu_err != oroSuccess) {
        printf("[ISOLATION TEST] WARNING: HIP runtime not initialized properly\n");
        fflush(stdout);
        return;
    }
    printf("[ISOLATION TEST] Found %d HIP device(s)\n", deviceCount);
    fflush(stdout);

    oroDeviceProp_t props{};
    gpu_err = oroGetDeviceProperties(&props, tnn_get_device(0));
    if (gpu_err != oroSuccess) {
        printf("[ISOLATION TEST] WARNING: Could not get device properties\n");
        fflush(stdout);
        return;
    }
    
    printf("[ISOLATION TEST] Device 0: %s\n", props.name);
    printf("[ISOLATION TEST] Compute capability: %d.%d\n", props.major, props.minor);
    fflush(stdout);

    // Architecture option (for TEST 1 only)
    std::string arch_option;
    if (tnn_is_nvidia_device(0)) {
        char buf[64];
        snprintf(buf, sizeof(buf), "--gpu-architecture=sm_%d%d", props.major, props.minor);
        arch_option = buf;
        printf("[ISOLATION TEST] Platform: NVIDIA (%s)\n", buf);
    } else {
        arch_option = std::string("--gpu-architecture=") + props.gcnArchName;
        printf("[ISOLATION TEST] Platform: AMD (%s)\n", props.gcnArchName);
    }
    fflush(stdout);

#ifdef TNN_XELISHASH
#ifdef WITH_OROCHI
    {
        oroApi loaded = oroLoadedAPI();
        bool has_rtc = (loaded & ORO_API_HIPRTC) || (loaded & ORO_API_CUDARTC);
        if (!has_rtc) {
            printf("[ISOLATION TEST] WARNING: Runtime compiler is not loaded; skipping HIPRTC isolation test\n");
            fflush(stdout);
            return;
        }
    }
#endif

    // ------------------------------------------------------------------------
    // TEST 1: Minimal kernel (raw HIPRTC using arch_option)
    // ------------------------------------------------------------------------
    printf("\n");
    printf("[ISOLATION TEST] === TEST 1: Minimal kernel (raw HIPRTC sanity check) ===\n");
    fflush(stdout);

    static constexpr auto minimal_kernel = R"(
        extern "C"
        __global__ void minimal_test_kernel(float* output, float* input1, float* input2, int size) {
          int i = threadIdx.x + blockIdx.x * blockDim.x;
          if (i < size) {
            output[i] = input1[i] + input2[i];
          }
        }
    )";

    {
        orortcProgram prog;
        orortcResult rtc_result;

        printf("[ISOLATION TEST] Creating minimal program...\n");
        fflush(stdout);

        rtc_result = orortcCreateProgram(&prog, minimal_kernel, "minimal.hip", 0, nullptr, nullptr);
        if (rtc_result != ORORTC_SUCCESS) {
            printf("[ISOLATION TEST] ERROR: Minimal orortcCreateProgram failed: %d\n", rtc_result);
            fflush(stdout);
            return;
        }

        const char* opts[] = { arch_option.c_str() };
        rtc_result = orortcCompileProgram(prog, 1, opts);

        if (rtc_result != ORORTC_SUCCESS) {
            size_t log_size = 0;
            orortcGetProgramLogSize(prog, &log_size);
            if (log_size > 0) {
                std::vector<char> log(log_size);
                orortcGetProgramLog(prog, log.data());
                printf("[ISOLATION TEST] Minimal compile log:\n%s\n", log.data());
            }
            orortcDestroyProgram(&prog);
            printf("[ISOLATION TEST] ERROR: Minimal compilation failed: %d\n", rtc_result);
            fflush(stdout);
            return;
        }

        size_t code_size = 0;
        orortcGetCodeSize(prog, &code_size);
        printf("[ISOLATION TEST] Minimal kernel compiled OK (%zu bytes)\n", code_size);
        orortcDestroyProgram(&prog);
        fflush(stdout);
    }

    // ------------------------------------------------------------------------
    // TEST 2: Full Xelis kernel via RTCCompiler + manifest system
    // ------------------------------------------------------------------------
    printf("\n");
    printf("[ISOLATION TEST] === TEST 2: Full Xelis kernel via RTCCompiler + manifests ===\n");
    fflush(stdout);

    // IMPORTANT: do NOT include arch_option here.
    // Let RTCCompiler use its internally detected gpu_arch_.
    std::vector<std::string> xelis_extra_opts = {
        "--dopt=on",
        "--use_fast_math"
        #ifdef __linux__
        ,"--device-int128"
        #endif
    };

    try {
        compile_xelis_v3_via_rtc(xelis_extra_opts);
    } catch (const std::exception& e) {
        printf("[ISOLATION TEST] ERROR: RTCCompiler Xelis compile failed: %s\n", e.what());
        fflush(stdout);
        return;
    }

    printf("\n");
    printf("========================================\n");
    printf("[ISOLATION TEST] SUCCESS via RTCCompiler + manifests!\n");
    printf("[ISOLATION TEST] Full Xelis kernel compiled on MAIN THREAD using the\n");
    printf("[ISOLATION TEST] same manifest + RTCCompiler path used in the miner.\n");
    printf("========================================\n");
    printf("\n");
    fflush(stdout);

#else
    printf("[ISOLATION TEST] TNN_XELISHASH not defined, skipping Xelis test\n");
    fflush(stdout);
#endif // TNN_XELISHASH

#else
    printf("[ISOLATION TEST] TNN_HIP not defined, skipping test\n");
    fflush(stdout);
#endif // TNN_HIP
}
