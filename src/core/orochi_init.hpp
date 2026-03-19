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

    // HIP paths (bundled + system fallbacks)
    const char *hip[14];
    const char *hiprtc[14];

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
    paths.libsDir = getExeDir() + "/libs";
    paths.hipLib = paths.libsDir + "/libamdhip64.so.6";
    paths.hiprtcLib = paths.libsDir + "/libhiprtc.so.6";
    paths.build();

    const char **hipPaths = paths.libsDir.empty() ? nullptr : paths.hip;
    const char **hiprtcPaths = paths.libsDir.empty() ? nullptr : paths.hiprtc;

// Try HIP + CUDA
#ifdef _WIN32
    int err = oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0);
#else
    int err = oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0,
                            hipPaths, hiprtcPaths,
                            nullptr, nullptr, nullptr);
#endif
    // Fallback to HIP only
    if (err != 0)
    {
      err = oroInitialize(ORO_API_HIP, 0,
                          hipPaths, hiprtcPaths);
    }


    // Fallback to CUDA only
    if (err != 0)
    {
      err = oroInitialize(ORO_API_CUDA, 0);
    }

    fflush(stdout);

    if (err == 0)
    {
      oroInit(0);
      printLoadedBackends();
      fflush(stdout);
    }

    return err;
  }

} // namespace oro