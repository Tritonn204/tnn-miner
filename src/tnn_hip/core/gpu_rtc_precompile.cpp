// gpu_rtc_precompile.cpp
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <map>

#include <tnn-common.hpp>
#include <tnn_log.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef TNN_HIP

#include <tnn_hip/common/gpu_compat.hpp>

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

// Simple thread pool with bounded worker count.
// Workers are joinable — the destructor signals shutdown and joins all
// threads so that no thread touches destroyed mutexes/condvars.
class CompilePool
{
public:
  explicit CompilePool(unsigned max_workers)
      : max_workers_(max_workers > 0 ? max_workers : 1) {}

  ~CompilePool() { shutdown(); }

  void submit(std::function<void()> task)
  {
    std::lock_guard<std::mutex> lock(mu_);
    tasks_.push(std::move(task));
    if (threads_.size() < max_workers_ && threads_.size() < tasks_.size() + active_)
    {
      threads_.emplace_back([this]()
                            { worker_loop(); });
    }
    cv_.notify_one();
  }

  void wait()
  {
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [&]
                  { return tasks_.empty() && active_ == 0; });
  }

  void shutdown()
  {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : threads_)
    {
      if (t.joinable())
        t.join();
    }
    threads_.clear();
  }

private:
  void worker_loop()
  {
    while (true)
    {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]
                 { return !tasks_.empty() || stop_; });
        if (stop_ && tasks_.empty())
        {
          done_cv_.notify_all();
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
        ++active_;
      }
      task();
      {
        std::lock_guard<std::mutex> lock(mu_);
        --active_;
      }
      done_cv_.notify_all();
    }
  }

  unsigned max_workers_;
  unsigned active_ = 0;
  bool stop_ = false;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> threads_;
};

extern "C" bool precompile_all_kernels()
{
#if !defined(TNN_HIP) || !defined(TNN_XELISHASH)
  TNN_LOG_DEBUG("[PRECOMPILE] HIP or TNN_XELISHASH not enabled — skipping.\n");
  return false;
#else

  TNN_LOG_DEBUG_COLOR(BRIGHT_YELLOW, "[PRECOMPILE] GPU kernel precompile (main thread)\n");
#ifdef _WIN32
  TNN_LOG_DEBUG("[PRECOMPILE] Thread ID: %lu\n", (unsigned long)GetCurrentThreadId());
#endif

  int deviceCount = 0;
  oroError_t err = oroGetDeviceCount(&deviceCount);
  if (err != oroSuccess || deviceCount == 0)
  {
    TNN_LOG_ERROR("[PRECOMPILE] No GPU devices found (err=%d)\n", err);
    return false;
  }

  TNN_LOG_DEBUG_COLOR(BRIGHT_YELLOW, "[PRECOMPILE] Found %d GPU device(s)\n", deviceCount);

  // Collect device info (must happen on main thread before spawning workers)
  struct DeviceInfo
  {
    int id;
    oroDeviceProp_t props;
    std::string arch;
    bool use;
  };
  std::vector<DeviceInfo> devices(deviceCount);

  for (int d = 0; d < deviceCount; ++d)
  {
    auto &di = devices[d];
    di.id = d;
    (void)oroGetDeviceProperties(&di.props, tnn_get_device(d));
    di.use = shouldUseDevice(d);

    if (tnn_is_nvidia_device(d))
    {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "sm_%d%d", di.props.major, di.props.minor);
      di.arch = buf;
    }
    else
    {
      di.arch = di.props.gcnArchName;
    }

    TNN_LOG_DEBUG("[PRECOMPILE]   Device %d: %s (arch=%s)%s\n",
                  d, di.props.name, di.arch.c_str(),
                  di.use ? "" : " (skipped)");
  }

  const int algo = miningProfile.coin.miningAlgo;
  TNN_LOG_DEBUG("[PRECOMPILE] miningAlgo = %d\n", algo);

  std::atomic<bool> ok{true};

  switch (algo)
  {

  case ALGO_XELISV2:
  case ALGO_XELISV3:
  {
    TNN_LOG_DEBUG("[PRECOMPILE] Selected Xelis algorithm\n");

    AlgoConfig cfg = XELIS_V3_CONFIG;

    TNN_LOG_DEBUG("[PRECOMPILE] Source size = %zu bytes, headers = %zu\n",
                  cfg.source.size(), cfg.rtc_headers.size());

    RTCCompiler &rtc = RTCCompiler::instance();
    for (const auto &header : cfg.rtc_headers)
    {
      rtc.add_header_source(std::string(header.name), std::string(header.source));
    }

    // Group devices by unique compile key (arch + compile-affecting opts)
    // so we know how many actual compilations are needed
    std::map<std::string, std::vector<int>> arch_groups;
    for (int d = 0; d < deviceCount; ++d)
    {
      if (!devices[d].use)
        continue;
      arch_groups[devices[d].arch].push_back(d);
    }

    unsigned num_unique = (unsigned)arch_groups.size();
    unsigned max_workers = std::min(num_unique,
                                    std::max(1u, std::thread::hardware_concurrency()));

    TNN_LOG_INFO_COLOR(BRIGHT_YELLOW, "[PRECOMPILE] %u unique arch(es), using up to %u compile worker(s)\n",
                       num_unique, max_workers);

    // Capture config values needed by workers
    const std::string source_str(cfg.source);
    const std::string source_path = cfg.source_path;
    const std::string primary_kernel = cfg.get_primary_kernel();
    const auto kernel_names = cfg.get_kernel_names();

    CompilePool pool(max_workers);

    // Collect precompile contexts so we can destroy them after compilation
    // to reclaim NVRTC/driver VRAM before workers start tuning.
    std::mutex ctx_mutex;
    std::vector<oroCtx> precompile_contexts;

    for (auto &[arch, dev_ids] : arch_groups)
    {
      // Submit one task per device. The first device for a given arch
      // does the actual NVRTC compile; subsequent same-arch devices
      // hit the RTCCompiler code_cache and only do a fast module load.
      for (int d : dev_ids)
      {
        pool.submit([&, d, arch]()
                    {
                    auto& di = devices[d];

                    TNN_LOG_INFO_COLOR(BRIGHT_YELLOW, "[PRECOMPILE] Device %d: %s (arch=%s) — compiling kernels...\n",
                                 d, di.props.name, arch.c_str());

                    // Create a per-thread GPU context — this sets the thread-local
                    // s_api so all oro* calls dispatch to the correct backend.
                    oroCtx ctx;
                    oroError_t ctxErr = oroCtxCreate(&ctx, 0, tnn_get_device(d));
                    if (ctxErr != oroSuccess) {
                        TNN_LOG_ERROR("[PRECOMPILE] oroCtxCreate(%d) failed: %s\n",
                                      d, tnn_error_string(ctxErr));
                        ok.store(false);
                        return;
                    }

                    oroError_t setCurrErr = oroCtxSetCurrent(ctx);
                    if (setCurrErr != oroSuccess) {
                        TNN_LOG_ERROR("[PRECOMPILE] oroCtxSetCurrent(%d) failed: %s\n",
                                      d, tnn_error_string(setCurrErr));
                        ok.store(false);
                        oroCtxDestroy(ctx);
                        return;
                    }

                    {
                        std::lock_guard<std::mutex> lock(ctx_mutex);
                        precompile_contexts.push_back(ctx);
                    }

                    // Per-device block size limits and compile options
                    const BlockSizeLimits& limits = tnn_is_amd_device(d)
                        ? cfg.amd_blocks : cfg.nvidia_blocks;
                    const int min_wg = limits.block_min;
                    const int max_wg = limits.block_max;

                    std::vector<std::string> opts;

                    if (tnn_is_amd_device(d)) {
                        fflush(stdout);
                        opts = {"-O3", "-mno-cumode", "-ffast-math"};
                        opts.push_back("--gpu-architecture=" + arch);
                        opts.push_back("-DXELIS_MIN_WG=" + std::to_string(min_wg));
                        opts.push_back("-DXELIS_MAX_WG=" + std::to_string(max_wg));
                    } else {
                        opts = {"--dopt=on", "--use_fast_math"};
                        opts.push_back("--gpu-architecture=" + arch);
#ifdef __linux__
                        opts.push_back("--device-int128");
#endif
                        opts.push_back("-DXELIS_MIN_WG=" + std::to_string(min_wg));
                        opts.push_back("-DXELIS_MAX_WG=" + std::to_string(max_wg));

                        {
                            int s3_nreg = (di.props.major <= 7) ? 56 : 40;
                            opts.push_back("-DXELIS_S3_NREG=" + std::to_string(s3_nreg));
                        }

                        opts.push_back("-DDEVICE_ID=" + std::to_string(d));
                    }

                    try {
                        auto compiled = rtc.compile_from_source(
                            source_str,
                            source_path,
                            primary_kernel,
                            opts,
                            d
                        );

                        TNN_LOG_DEBUG("[PRECOMPILE] Device %d: module=%p\n",
                                      d, (void*)compiled.module);

                        int loaded_count = 0;
                        for (const auto& kname : kernel_names) {
                            oroFunction_t func = nullptr;
                            oroError_t herr = oroModuleGetFunction(
                                &func, compiled.module, kname.c_str());
                            if (herr == oroSuccess && func) {
                                TNN_LOG_DEBUG("[PRECOMPILE]   Device %d OK: '%s'\n",
                                              d, kname.c_str());
                                ++loaded_count;
                            } else {
                                TNN_LOG_ERROR("[PRECOMPILE]   Device %d MISSING: '%s' : %s\n",
                                              d, kname.c_str(), tnn_error_string(herr));
                                ok.store(false);
                            }
                        }

                        if (loaded_count == 0) {
                            TNN_LOG_ERROR("[PRECOMPILE] No kernels loaded on device %d\n", d);
                            ok.store(false);
                        } else {
                            TNN_LOG_INFO_COLOR(BRIGHT_YELLOW, "[PRECOMPILE] Device %d: %d kernel(s) loaded\n",
                                         d, loaded_count);
                        }
                    }
                    catch (const std::exception& e) {
                        TNN_LOG_ERROR("[PRECOMPILE] Device %d precompile failed: %s\n",
                                      d, e.what());
                        ok.store(false);
                    } });
      }
    }

    pool.wait();

    // Reclaim VRAM: clear stale module handles (bound to precompile contexts)
    // then destroy the precompile contexts. Code cache stays intact on the host
    // so workers can reload modules cheaply via oroModuleLoadData.
    rtc.clear_stale_modules();

    for (auto &ctx : precompile_contexts)
    {
      (void)oroCtxDestroy(ctx);
    }
    TNN_LOG_INFO_COLOR(BRIGHT_YELLOW, "[PRECOMPILE] Destroyed %zu precompile context(s) — VRAM reclaimed\n",
                       precompile_contexts.size());

    break;
  }

  default:
    TNN_LOG_DEBUG("[PRECOMPILE] No RTC precompile required for algo %d\n", algo);
    break;
  }

  TNN_LOG_INFO_COLOR(BRIGHT_YELLOW, "[PRECOMPILE] %s\n", ok.load() ? "All kernels compiled" : "FAILED");

  return ok.load();

#endif // TNN_HIP + TNN_XELISHASH
}
