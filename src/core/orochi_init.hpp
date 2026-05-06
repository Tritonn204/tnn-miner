#pragma once

#include <Orochi/Orochi.h>
#include <filesystem>
#include <string>
#include <cstdio>
#include <tnn_log.hpp>

#ifdef __linux__
#include <unistd.h>
#include <limits.h>
#endif

namespace oro
{

  struct InitPaths
  {
    std::string libsDir;
    std::string hipLib;
    std::string hiprtcLib;
    std::string nvrtcLib;
    std::string nvrtcLibVersioned;
    std::string legacyNvrtcLib;
    std::string legacyNvrtcLibVersioned;

    // HIP paths (bundled + system fallbacks)
    const char *hip[14];
    const char *hiprtc[14];
    const char *nvrtc[32];

    void build()
    {
      // HIP driver
      hip[0] = hipLib.c_str();
      hip[1] = "/opt/rocm/hip/lib/libamdhip64.so.7";
      hip[2] = "/opt/rocm/lib/libamdhip64.so.7";
      hip[3] = "libamdhip64.so.7";
      hip[4] = "/opt/rocm/hip/lib/libamdhip64.so.6";
      hip[5] = "/opt/rocm/lib/libamdhip64.so.6";
      hip[6] = "libamdhip64.so.6";
      hip[7] = "/opt/rocm/hip/lib/libamdhip64.so.5";
      hip[8] = "/opt/rocm/lib/libamdhip64.so.5";
      hip[9] = "libamdhip64.so.5";
      hip[10] = "/opt/rocm/hip/lib/libamdhip64.so";
      hip[11] = "/opt/rocm/lib/libamdhip64.so";
      hip[12] = "libamdhip64.so";
      hip[13] = nullptr;

      // HIP RTC
      hiprtc[0] = hiprtcLib.c_str();
      hiprtc[1] = "/opt/rocm/hip/lib/libhiprtc.so.7";
      hiprtc[2] = "/opt/rocm/lib/libhiprtc.so.7";
      hiprtc[3] = "libhiprtc.so.7";
      hiprtc[4] = "/opt/rocm/hip/lib/libhiprtc.so.6";
      hiprtc[5] = "/opt/rocm/lib/libhiprtc.so.6";
      hiprtc[6] = "libhiprtc.so.6";
      hiprtc[7] = "/opt/rocm/hip/lib/libhiprtc.so.5";
      hiprtc[8] = "/opt/rocm/lib/libhiprtc.so.5";
      hiprtc[9] = "libhiprtc.so.5";
      hiprtc[10] = "/opt/rocm/hip/lib/libhiprtc.so";
      hiprtc[11] = "/opt/rocm/lib/libhiprtc.so";
      hiprtc[12] = "libhiprtc.so";
      hiprtc[13] = nullptr;

      // NVRTC. CUEW treats a custom path list as a replacement for its
      // built-in scan, so keep system locations first and use bundled copies
      // beside the miner as the final fallback. CUDA's libnvrtc loads its
      // builtins from the same directory, so keep libnvrtc-builtins.so* beside
      // whichever libnvrtc.so* is bundled.
      nvrtc[0] = "libnvrtc.so.13";
      nvrtc[1] = "libnvrtc.so.12";
      nvrtc[2] = "libnvrtc.so";
      nvrtc[3] = "/usr/local/cuda/lib64/libnvrtc.so.13";
      nvrtc[4] = "/usr/local/cuda/lib64/libnvrtc.so.12";
      nvrtc[5] = "/usr/local/cuda/lib64/libnvrtc.so";
      nvrtc[6] = "/usr/local/cuda/lib/libnvrtc.so";
      nvrtc[7] = "/usr/lib/x86_64-linux-gnu/libnvrtc.so.13";
      nvrtc[8] = "/usr/lib/x86_64-linux-gnu/libnvrtc.so.12";
      nvrtc[9] = "/usr/lib/x86_64-linux-gnu/libnvrtc.so";
      nvrtc[10] = "/usr/local/cuda-13.0/lib64/libnvrtc.so.13";
      nvrtc[11] = "/usr/local/cuda-13.0/lib64/libnvrtc.so";
      nvrtc[12] = "/usr/local/cuda-13/lib64/libnvrtc.so.13";
      nvrtc[13] = "/usr/local/cuda-13/lib64/libnvrtc.so";
      nvrtc[14] = "/usr/local/cuda-12.9/lib64/libnvrtc.so.12";
      nvrtc[15] = "/usr/local/cuda-12.9/lib64/libnvrtc.so";
      nvrtc[16] = "/usr/local/cuda-12.8/lib64/libnvrtc.so.12";
      nvrtc[17] = "/usr/local/cuda-12.8/lib64/libnvrtc.so";
      nvrtc[18] = "/usr/local/cuda-12.7/lib64/libnvrtc.so.12";
      nvrtc[19] = "/usr/local/cuda-12.7/lib64/libnvrtc.so";
      nvrtc[20] = "/usr/local/cuda-12.6/lib64/libnvrtc.so.12";
      nvrtc[21] = "/usr/local/cuda-12.6/lib64/libnvrtc.so";
      nvrtc[22] = "/usr/local/cuda-12/lib64/libnvrtc.so.12";
      nvrtc[23] = "/usr/local/cuda-12/lib64/libnvrtc.so";
      nvrtc[24] = nvrtcLib.c_str();
      nvrtc[25] = nvrtcLibVersioned.c_str();
      nvrtc[26] = legacyNvrtcLib.c_str();
      nvrtc[27] = legacyNvrtcLibVersioned.c_str();
      nvrtc[28] = nullptr;
    }
  };

  inline std::string getExeDir()
  {
#ifdef __linux__
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len != -1)
    {
      exePath[len] = '\0';
      return std::filesystem::path(exePath).parent_path().string();
    }
#elif defined(_WIN32)
    char exePath[MAX_PATH];
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    return std::filesystem::path(exePath).parent_path().string();
#endif
    return "";
  }

  inline void printLoadedBackends()
  {
    int deviceCount = 0;

    // Check HIP devices
    if (oroGetDeviceCount(&deviceCount) == oroSuccess && deviceCount > 0)
    {
      TNN_LOG_INFO("  [HIP]  Found %d device(s)\n", deviceCount);
      for (int i = 0; i < deviceCount; i++)
      {
        oroDevice dev;
        char name[256];
        if (oroDeviceGet(&dev, i) == oroSuccess &&
            oroDeviceGetName(name, sizeof(name), dev) == oroSuccess)
        {
          TNN_LOG_INFO("         [%d] %s\n", i, name);
        }
      }
    }

    // Note: Orochi abstracts HIP/CUDA, so oroGetDeviceCount returns
    // whichever backend was loaded. If you need both simultaneously,
    // you'd need separate context handling.
  }

  inline int initialize()
  {
    // Build paths
    InitPaths paths;
    const std::string exeDir = getExeDir();
    paths.libsDir = exeDir + "/libs";
    paths.hipLib = paths.libsDir + "/libamdhip64.so.6";
    paths.hiprtcLib = paths.libsDir + "/libhiprtc.so.6";
    paths.nvrtcLib = paths.libsDir + "/libnvrtc.so.12";
    paths.nvrtcLibVersioned = paths.libsDir + "/libnvrtc.so";
    paths.legacyNvrtcLib = exeDir + "/nvrtc_libs/libnvrtc.so.12";
    paths.legacyNvrtcLibVersioned = exeDir + "/nvrtc_libs/libnvrtc.so";
    paths.build();

    const char **hipPaths = paths.libsDir.empty() ? nullptr : paths.hip;
    const char **hiprtcPaths = paths.libsDir.empty() ? nullptr : paths.hiprtc;
    const char **nvrtcPaths = paths.libsDir.empty() ? nullptr : paths.nvrtc;

// Try HIP + CUDA
#ifdef _WIN32
    TNN_LOG_DEBUG("  [INIT] Trying HIP + CUDA...\n");
    int err = oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0);
    TNN_LOG_DEBUG("  [INIT] HIP+CUDA result: %d, loadedAPIs: 0x%x\n", err, (int)oroLoadedAPI());
#else
    int err = oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0,
                            hipPaths, hiprtcPaths,
                            nullptr, nullptr, nvrtcPaths);
#endif
    // Fallback to HIP only
    if (err != 0)
    {
      TNN_LOG_DEBUG("  [INIT] Trying HIP only...\n");
      err = oroInitialize(ORO_API_HIP, 0,
                          hipPaths, hiprtcPaths);
      TNN_LOG_DEBUG("  [INIT] HIP-only result: %d, loadedAPIs: 0x%x\n", err, (int)oroLoadedAPI());
    }


    // Fallback to CUDA only
    if (err != 0)
    {
      TNN_LOG_DEBUG("  [INIT] Trying CUDA only...\n");
#ifdef _WIN32
      err = oroInitialize(ORO_API_CUDA, 0);
#else
      err = oroInitialize(ORO_API_CUDA, 0,
                          nullptr, nullptr,
                          nullptr, nullptr, nvrtcPaths);
#endif
      TNN_LOG_DEBUG("  [INIT] CUDA-only result: %d, loadedAPIs: 0x%x\n", err, (int)oroLoadedAPI());
    }

    fflush(stdout);

    if (err == 0)
    {
      oroInit(0);
      printLoadedBackends();

      // Check which API components actually loaded
      oroApi loaded = oroLoadedAPI();
      if (loaded & ORO_API_HIPDRIVER) {
        TNN_LOG_INFO("  [HIP]  HIP driver loaded\n");
        if (loaded & ORO_API_HIPRTC) {
          TNN_LOG_INFO("  [HIP]  HIPRTC loaded (runtime compilation available)\n");
        } else {
          TNN_LOG_ERROR("  [HIP]  WARNING: HIPRTC not loaded! Runtime compilation will fail.\n");
          TNN_LOG_ERROR("  [HIP]  Install the HIP SDK: https://rocm.docs.amd.com\n");
        }
      }
      if (loaded & ORO_API_CUDADRIVER) {
        TNN_LOG_INFO("  [CUDA] CUDA driver loaded\n");
        if (loaded & ORO_API_CUDARTC) {
          TNN_LOG_INFO("  [CUDA] NVRTC loaded (runtime compilation available)\n");
        } else {
          TNN_LOG_ERROR("  [CUDA] WARNING: NVRTC not loaded! Runtime compilation will fail.\n");
          TNN_LOG_ERROR("  [CUDA] Install the CUDA Toolkit: https://developer.nvidia.com/cuda-downloads\n");
        }
      }
      fflush(stdout);
    }

    return err;
  }

} // namespace oro
