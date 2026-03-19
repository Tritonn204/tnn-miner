#include "gpuOC.hip.h"
#include "devInfo.hpp" // getGPUCount()

#include <tnn_log.hpp>
#include <adl_defines.h>

#include <tnn_hip/common/gpu_compat.hpp>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#define T_LIB HMODULE
#define T_OPEN LoadLibraryA
#define T_SYM(h, s) ((void *)GetProcAddress(h, s))
#define T_CLOSE FreeLibrary
#else
#include <dlfcn.h>
#define T_LIB void *
#define T_OPEN(n) dlopen(n, RTLD_LAZY)
#define T_SYM(h, s) dlsym(h, s)
#define T_CLOSE dlclose
#endif

// ======================================================================
//  Internal state
// ======================================================================
enum class TuneBack
{
  None,
  NVML,
  RSMI,
  ADL
};
static TuneBack g_back = TuneBack::None;
static T_LIB g_lib = nullptr;

// Saved per-device baseline so resetGpuTuning() can restore.
struct Baseline
{
  bool saved = false;
  // NVML
  unsigned nvmlDefPowerMw = 0;
  // RSMI
  uint64_t rsmiPowerCapUw = 0;
  uint64_t rsmiMaxSclkMHz = 0; // highest DPM frequency (MHz) at init
  uint64_t rsmiMaxMclkMHz = 0;
};
static std::vector<Baseline> g_base;

// ======================================================================
//  NVML types & pointers
// ======================================================================
typedef void *nvmlDevice_t;
#define NVML_OK 0

typedef int (*fn_nvmlInit)();
typedef int (*fn_nvmlShutdown)();
typedef int (*fn_nvmlGetHandle)(unsigned, nvmlDevice_t *);
typedef int (*fn_nvmlSetPowerLimit)(nvmlDevice_t, unsigned);
typedef int (*fn_nvmlGetPowerLimit)(nvmlDevice_t, unsigned *);
typedef int (*fn_nvmlGetDefPowerLimit)(nvmlDevice_t, unsigned *);
typedef int (*fn_nvmlGetPowerConstraints)(nvmlDevice_t, unsigned *, unsigned *);
typedef int (*fn_nvmlSetLockedGpu)(nvmlDevice_t, unsigned, unsigned);
typedef int (*fn_nvmlResetLockedGpu)(nvmlDevice_t);
typedef int (*fn_nvmlSetLockedMem)(nvmlDevice_t, unsigned, unsigned);
typedef int (*fn_nvmlResetLockedMem)(nvmlDevice_t);
typedef int (*fn_nvmlSetGpcOffset)(nvmlDevice_t, int);
typedef int (*fn_nvmlSetMemOffset)(nvmlDevice_t, int);

static fn_nvmlInit pN_Init = nullptr;
static fn_nvmlShutdown pN_Shutdown = nullptr;
static fn_nvmlGetHandle pN_GetHandle = nullptr;
static fn_nvmlSetPowerLimit pN_SetPowerLimit = nullptr;
static fn_nvmlGetPowerLimit pN_GetPowerLimit = nullptr;
static fn_nvmlGetDefPowerLimit pN_GetDefPowerLimit = nullptr;
static fn_nvmlGetPowerConstraints pN_GetPowerConstr = nullptr;
static fn_nvmlSetLockedGpu pN_SetLockedGpu = nullptr;
static fn_nvmlResetLockedGpu pN_ResetLockedGpu = nullptr;
static fn_nvmlSetLockedMem pN_SetLockedMem = nullptr;
static fn_nvmlResetLockedMem pN_ResetLockedMem = nullptr;
static fn_nvmlSetGpcOffset pN_SetGpcOff = nullptr; // may be null (old driver)
static fn_nvmlSetMemOffset pN_SetMemOff = nullptr; // may be null

// Helper: get handle, or log error.
static bool nvmlHandle(int dev, nvmlDevice_t &h)
{
  if (!pN_GetHandle)
    return false;
  // Try _v2 variant first (loaded into same pointer, see init)
  if (pN_GetHandle((unsigned)dev, &h) != NVML_OK)
  {
    TNN_LOG_ERROR("NVML: cannot get handle for GPU %d\n", dev);
    return false;
  }
  return true;
}

// ======================================================================
//  RSMI types & pointers
// ======================================================================
// We define our own versions of the rsmi types to avoid requiring the
// ROCm SMI header at compile time.  Only the fields we read are listed;
// extra padding absorbs anything newer ROCm versions might append.

#define RSMI_OK 0
#define RSMI_CLK_TYPE_SYS 0x0 // Core (GFX) clock
#define RSMI_CLK_TYPE_MEM 0x4 // Memory clock
#define RSMI_PERF_LEVEL_AUTO 0
#define RSMI_PERF_LEVEL_MANUAL 3
#define RSMI_FREQ_IND_MAX 1
#define RSMI_MAX_FREQS 33

struct RsmiFreqs
{ // compatible superset of rsmi_frequencies_t
  uint32_t num_supported;
  uint32_t current;
  uint64_t frequency[RSMI_MAX_FREQS]; // Hz
  uint64_t _pad[8];                   // absorb has_deep_sleep + future fields
};

typedef int (*fn_rsmiInit)(uint64_t);
typedef int (*fn_rsmiShutdown)();
typedef int (*fn_rsmiPowerCapSet)(uint32_t, uint32_t, uint64_t);
typedef int (*fn_rsmiPowerCapGet)(uint32_t, uint32_t, uint64_t *);
typedef int (*fn_rsmiPowerCapRange)(uint32_t, uint32_t, uint64_t *, uint64_t *);
typedef int (*fn_rsmiPerfLevelSet)(uint32_t, uint32_t);                     // dev, perf_level enum
typedef int (*fn_rsmiClkRangeSet)(uint32_t, uint64_t, uint64_t, uint32_t);  // dev, min, max, clkType — MHz
typedef int (*fn_rsmiOdClkInfoSet)(uint32_t, uint32_t, uint64_t, uint32_t); // dev, level, MHz, clkType
typedef int (*fn_rsmiClkFreqGet)(uint32_t, uint32_t, RsmiFreqs *);          // dev, clkType, out

static fn_rsmiInit pR_Init = nullptr;
static fn_rsmiShutdown pR_Shutdown = nullptr;
static fn_rsmiPowerCapSet pR_PowerCapSet = nullptr;
static fn_rsmiPowerCapGet pR_PowerCapGet = nullptr;
static fn_rsmiPowerCapRange pR_PowerCapRange = nullptr;
static fn_rsmiPerfLevelSet pR_PerfLvlSet = nullptr;
static fn_rsmiClkRangeSet pR_ClkRangeSet = nullptr;   // ROCm 5.0+; may be null
static fn_rsmiOdClkInfoSet pR_OdClkInfoSet = nullptr; // fallback for older ROCm
static fn_rsmiClkFreqGet pR_ClkFreqGet = nullptr;

// ======================================================================
//  Sym-load helper
// ======================================================================
template <typename T>
static bool loadSym(T &dst, T_LIB lib, const char *primary, const char *fallback = nullptr)
{
  dst = (T)T_SYM(lib, primary);
  if (!dst && fallback)
    dst = (T)T_SYM(lib, fallback);
  return dst != nullptr;
}

// ======================================================================
//  NVML backend init
// ======================================================================
static bool initNvml()
{
#ifdef _WIN32
  g_lib = T_OPEN("nvml.dll");
#else
  g_lib = T_OPEN("libnvidia-ml.so.1");
  if (!g_lib)
    g_lib = T_OPEN("libnvidia-ml.so");
#endif
  if (!g_lib)
    return false;

  // Required symbols -----------------------------------------------
  bool ok = true;
  ok &= loadSym(pN_Init, g_lib, "nvmlInit_v2", "nvmlInit");
  ok &= loadSym(pN_Shutdown, g_lib, "nvmlShutdown");
  ok &= loadSym(pN_GetHandle, g_lib, "nvmlDeviceGetHandleByIndex_v2",
                "nvmlDeviceGetHandleByIndex");
  ok &= loadSym(pN_SetPowerLimit, g_lib, "nvmlDeviceSetPowerManagementLimit");
  ok &= loadSym(pN_GetPowerLimit, g_lib, "nvmlDeviceGetPowerManagementLimit");
  if (!ok)
  {
    T_CLOSE(g_lib);
    g_lib = nullptr;
    return false;
  }

  // Optional symbols (older drivers may lack some) ------------------
  loadSym(pN_GetDefPowerLimit, g_lib, "nvmlDeviceGetPowerManagementDefaultLimit");
  loadSym(pN_GetPowerConstr, g_lib, "nvmlDeviceGetPowerManagementLimitConstraints");
  loadSym(pN_SetLockedGpu, g_lib, "nvmlDeviceSetGpuLockedClocks");
  loadSym(pN_ResetLockedGpu, g_lib, "nvmlDeviceResetGpuLockedClocks");
  loadSym(pN_SetLockedMem, g_lib, "nvmlDeviceSetMemoryLockedClocks");
  loadSym(pN_ResetLockedMem, g_lib, "nvmlDeviceResetMemoryLockedClocks");
  loadSym(pN_SetGpcOff, g_lib, "nvmlDeviceSetGpcClkVfOffset");
  loadSym(pN_SetMemOff, g_lib, "nvmlDeviceSetMemClkVfOffset");

  if (pN_Init() != NVML_OK)
  {
    TNN_LOG_ERROR("NVML tuning: nvmlInit failed\n");
    T_CLOSE(g_lib);
    g_lib = nullptr;
    return false;
  }

  // Save baselines --------------------------------------------------
  int n = getGPUCount();
  g_base.assign(n, Baseline{});
  for (int i = 0; i < n; ++i)
  {
    nvmlDevice_t h;
    if (!nvmlHandle(i, h))
      continue;
    g_base[i].saved = true;
    if (pN_GetDefPowerLimit)
      pN_GetDefPowerLimit(h, &g_base[i].nvmlDefPowerMw);
    else if (pN_GetPowerLimit)
      pN_GetPowerLimit(h, &g_base[i].nvmlDefPowerMw);
  }

  TNN_LOG_DEBUG("NVML tuning: initialized (%d GPUs, lockedClocks=%s, vfOffset=%s)\n",
                n,
                pN_SetLockedGpu ? "yes" : "no",
                pN_SetGpcOff ? "yes" : "no");
  return true;
}

// ======================================================================
//  RSMI backend init
// ======================================================================
static bool initRsmi()
{
#ifdef _WIN32
  return false; // RSMI is Linux-only
#else
  g_lib = T_OPEN("librocm_smi64.so");
  if (!g_lib)
    return false;

  bool ok = true;
  ok &= loadSym(pR_Init, g_lib, "rsmi_init");
  ok &= loadSym(pR_Shutdown, g_lib, "rsmi_shut_down");
  ok &= loadSym(pR_PowerCapSet, g_lib, "rsmi_dev_power_cap_set");
  ok &= loadSym(pR_PowerCapGet, g_lib, "rsmi_dev_power_cap_get");
  if (!ok)
  {
    T_CLOSE(g_lib);
    g_lib = nullptr;
    return false;
  }

  // Optional (version-dependent)
  loadSym(pR_PowerCapRange, g_lib, "rsmi_dev_power_cap_range_get");
  loadSym(pR_PerfLvlSet, g_lib, "rsmi_dev_perf_level_set");
  loadSym(pR_ClkRangeSet, g_lib, "rsmi_dev_clk_range_set");    // ROCm ≥ 5.0
  loadSym(pR_OdClkInfoSet, g_lib, "rsmi_dev_od_clk_info_set"); // fallback
  loadSym(pR_ClkFreqGet, g_lib, "rsmi_dev_gpu_clk_freq_get");

  if (pR_Init(0) != RSMI_OK)
  {
    TNN_LOG_ERROR("RSMI tuning: rsmi_init failed\n");
    T_CLOSE(g_lib);
    g_lib = nullptr;
    return false;
  }

  // Save baselines --------------------------------------------------
  int n = getGPUCount();
  g_base.assign(n, Baseline{});
  for (int i = 0; i < n; ++i)
  {
    Baseline &b = g_base[i];
    if (pR_PowerCapGet)
      if (pR_PowerCapGet(i, 0, &b.rsmiPowerCapUw) == RSMI_OK)
        b.saved = true;

    if (pR_ClkFreqGet)
    {
      RsmiFreqs f{};
      if (pR_ClkFreqGet(i, RSMI_CLK_TYPE_SYS, &f) == RSMI_OK && f.num_supported > 0)
        b.rsmiMaxSclkMHz = f.frequency[f.num_supported - 1] / 1000000ULL;
      if (pR_ClkFreqGet(i, RSMI_CLK_TYPE_MEM, &f) == RSMI_OK && f.num_supported > 0)
        b.rsmiMaxMclkMHz = f.frequency[f.num_supported - 1] / 1000000ULL;
      b.saved = true;
    }
  }

  bool canClock = (pR_ClkRangeSet || pR_OdClkInfoSet);
  TNN_LOG_DEBUG("RSMI tuning: initialized (%d GPUs, clockControl=%s, perfLevel=%s)\n",
                n,
                canClock ? "yes" : "no",
                pR_PerfLvlSet ? "yes" : "no");
  return true;
#endif
}

// ======================================================================
//  NVML apply / reset
// ======================================================================
static bool applyNvml(int dev, const GpuTuneParams &p)
{
  nvmlDevice_t h;
  if (!nvmlHandle(dev, h))
    return false;
  bool allOk = true;

  // ---- Power limit ------------------------------------------------
  if (p.powerLimitW > 0)
  {
    unsigned mw = (unsigned)p.powerLimitW * 1000u;

    if (pN_GetPowerConstr)
    {
      unsigned lo = 0, hi = 0;
      if (pN_GetPowerConstr(h, &lo, &hi) == NVML_OK)
      {
        if (mw < lo || mw > hi)
        {
          TNN_LOG_ERROR("NVML: GPU[%d] %dW outside allowed [%u, %u]W\n",
                        dev, p.powerLimitW, lo / 1000, hi / 1000);
          allOk = false;
          mw = 0;
        }
      }
    }
    if (mw > 0)
    {
      int r = pN_SetPowerLimit(h, mw);
      if (r != NVML_OK)
      {
        TNN_LOG_ERROR("NVML: GPU[%d] SetPowerManagementLimit(%uW) → error %d "
                      "(need root / nvidia-smi persistence?)\n",
                      dev, p.powerLimitW, r);
        allOk = false;
      }
      else
      {
        TNN_LOG_INFO("NVML: GPU[%d] power limit → %dW\n", dev, p.powerLimitW);
      }
    }
  }

  // ---- Core clock lock --------------------------------------------
  if (p.coreClockMHz > 0)
  {
    if (!pN_SetLockedGpu)
    {
      TNN_LOG_ERROR("NVML: GPU[%d] locked clocks not supported (driver too old)\n", dev);
      allOk = false;
    }
    else
    {
      int r = pN_SetLockedGpu(h, (unsigned)p.coreClockMHz, (unsigned)p.coreClockMHz);
      if (r != NVML_OK)
      {
        TNN_LOG_ERROR("NVML: GPU[%d] SetGpuLockedClocks(%d MHz) → error %d\n",
                      dev, p.coreClockMHz, r);
        allOk = false;
      }
      else
      {
        TNN_LOG_INFO("NVML: GPU[%d] core clock locked → %d MHz\n", dev, p.coreClockMHz);
      }
    }
  }

  // ---- Core clock offset ------------------------------------------
  if (p.coreClockOffsetMHz != 0)
  {
    if (!pN_SetGpcOff)
    {
      TNN_LOG_ERROR("NVML: GPU[%d] GPC clock offset not supported (driver too old)\n", dev);
      allOk = false;
    }
    else
    {
      int r = pN_SetGpcOff(h, p.coreClockOffsetMHz);
      if (r != NVML_OK)
      {
        TNN_LOG_ERROR("NVML: GPU[%d] SetGpcClkVfOffset(%+d MHz) → error %d\n",
                      dev, p.coreClockOffsetMHz, r);
        allOk = false;
      }
      else
      {
        TNN_LOG_INFO("NVML: GPU[%d] core clock offset → %+d MHz\n",
                     dev, p.coreClockOffsetMHz);
      }
    }
  }

  // ---- Memory clock offset ----------------------------------------
  if (p.memClockOffsetMHz != 0)
  {
    if (!pN_SetMemOff)
    {
      TNN_LOG_ERROR("NVML: GPU[%d] mem clock offset not supported (driver too old)\n", dev);
      allOk = false;
    }
    else
    {
      int r = pN_SetMemOff(h, p.memClockOffsetMHz);
      if (r != NVML_OK)
      {
        TNN_LOG_ERROR("NVML: GPU[%d] SetMemClkVfOffset(%+d MHz) → error %d\n",
                      dev, p.memClockOffsetMHz, r);
        allOk = false;
      }
      else
      {
        TNN_LOG_INFO("NVML: GPU[%d] memory clock offset → %+d MHz\n",
                     dev, p.memClockOffsetMHz);
      }
    }
  }

  return allOk;
}

static bool resetNvml(int dev)
{
  nvmlDevice_t h;
  if (!nvmlHandle(dev, h))
    return false;
  bool ok = true;

  // Unlock clocks
  if (pN_ResetLockedGpu && pN_ResetLockedGpu(h) != NVML_OK)
    TNN_LOG_ERROR("NVML: GPU[%d] ResetGpuLockedClocks failed\n", dev), ok = false;
  if (pN_ResetLockedMem && pN_ResetLockedMem(h) != NVML_OK)
    TNN_LOG_ERROR("NVML: GPU[%d] ResetMemoryLockedClocks failed\n", dev), ok = false;

  // Zero offsets
  if (pN_SetGpcOff)
    pN_SetGpcOff(h, 0);
  if (pN_SetMemOff)
    pN_SetMemOff(h, 0);

  // Restore original power limit
  if (dev < (int)g_base.size() && g_base[dev].saved && g_base[dev].nvmlDefPowerMw > 0)
    pN_SetPowerLimit(h, g_base[dev].nvmlDefPowerMw);

  return ok;
}

// ======================================================================
//  RSMI apply / reset
// ======================================================================

// Set OD max for a clock domain.
// Uses rsmi_dev_clk_range_set (ROCm ≥ 5) with rsmi_dev_od_clk_info_set as fallback.
// `mhz` is the desired maximum, `clkType` is RSMI_CLK_TYPE_SYS or _MEM.
// Caller must have already set perf level to MANUAL.
static bool rsmiSetOdMax(int dev, uint64_t mhz, uint32_t clkType)
{
  const char *label = (clkType == RSMI_CLK_TYPE_SYS) ? "core" : "mem";

  if (pR_ClkRangeSet)
  {
    // rsmi_dev_clk_range_set(dev, minMHz, maxMHz, clkType)
    // Set min=500 (safe floor) and max=target so the GPU can still
    // idle at low clocks.  Adjust the floor for your workload if needed.
    uint64_t floor = (clkType == RSMI_CLK_TYPE_SYS) ? 500 : 96;
    int r = pR_ClkRangeSet(dev, floor, mhz, clkType);
    if (r == RSMI_OK)
      return true;
    TNN_LOG_DEBUG("RSMI: GPU[%d] clk_range_set(%s, %llu MHz) → %d, trying od_clk_info_set\n",
                  dev, label, (unsigned long long)mhz, r);
  }

  if (pR_OdClkInfoSet)
  {
    // rsmi_dev_od_clk_info_set(dev, RSMI_FREQ_IND_MAX, mhz, clkType)
    int r = pR_OdClkInfoSet(dev, RSMI_FREQ_IND_MAX, mhz, clkType);
    if (r == RSMI_OK)
      return true;
    TNN_LOG_ERROR("RSMI: GPU[%d] od_clk_info_set(%s, %llu MHz) → error %d\n",
                  dev, label, (unsigned long long)mhz, r);
  }

  TNN_LOG_ERROR("RSMI: GPU[%d] no working clock-control API for %s\n", dev, label);
  return false;
}

static bool applyRsmi(int dev, const GpuTuneParams &p)
{
  bool allOk = true;

  // ---- Power limit ------------------------------------------------
  if (p.powerLimitW > 0 && pR_PowerCapSet)
  {
    uint64_t capUw = (uint64_t)p.powerLimitW * 1000000ULL;

    if (pR_PowerCapRange)
    {
      uint64_t maxCap = 0, minCap = 0;
      if (pR_PowerCapRange(dev, 0, &maxCap, &minCap) == RSMI_OK)
      {
        if (capUw < minCap || capUw > maxCap)
        {
          TNN_LOG_ERROR("RSMI: GPU[%d] %dW outside range [%.0f, %.0f]W\n",
                        dev, p.powerLimitW, (double)minCap / 1e6, (double)maxCap / 1e6);
          allOk = false;
          capUw = 0;
        }
      }
    }
    if (capUw > 0)
    {
      int r = pR_PowerCapSet(dev, 0, capUw);
      if (r != RSMI_OK)
      {
        TNN_LOG_ERROR("RSMI: GPU[%d] power_cap_set(%dW) → error %d (need root?)\n",
                      dev, p.powerLimitW, r);
        allOk = false;
      }
      else
      {
        TNN_LOG_INFO("RSMI: GPU[%d] power limit → %dW\n", dev, p.powerLimitW);
      }
    }
  }

  // ---- Clocks: enter MANUAL perf level if needed ------------------
  bool needClock = (p.coreClockMHz > 0 || p.coreClockOffsetMHz != 0 ||
                    p.memClockOffsetMHz != 0);
  if (needClock && pR_PerfLvlSet)
  {
    int r = pR_PerfLvlSet(dev, RSMI_PERF_LEVEL_MANUAL);
    if (r != RSMI_OK)
    {
      TNN_LOG_ERROR("RSMI: GPU[%d] perf_level_set(MANUAL) → error %d\n", dev, r);
      allOk = false;
      needClock = false; // can't proceed without manual mode
    }
  }
  else if (needClock && !pR_PerfLvlSet)
  {
    TNN_LOG_ERROR("RSMI: GPU[%d] rsmi_dev_perf_level_set not available\n", dev);
    allOk = false;
    needClock = false;
  }

  if (!needClock)
    return allOk;

  // ---- Core clock (absolute) --------------------------------------
  if (p.coreClockMHz > 0)
  {
    if (rsmiSetOdMax(dev, (uint64_t)p.coreClockMHz, RSMI_CLK_TYPE_SYS))
      TNN_LOG_INFO("RSMI: GPU[%d] core OD max → %d MHz\n", dev, p.coreClockMHz);
    else
      allOk = false;
  }

  // ---- Core clock offset ------------------------------------------
  if (p.coreClockOffsetMHz != 0 && p.coreClockMHz == 0)
  {
    // offset is relative to the baseline max we saved at init
    uint64_t baseMHz = (dev < (int)g_base.size()) ? g_base[dev].rsmiMaxSclkMHz : 0;
    if (baseMHz == 0)
    {
      TNN_LOG_ERROR("RSMI: GPU[%d] no baseline SCLK for offset calculation\n", dev);
      allOk = false;
    }
    else
    {
      int64_t target = (int64_t)baseMHz + p.coreClockOffsetMHz;
      if (target <= 0)
      {
        TNN_LOG_ERROR("RSMI: GPU[%d] core offset %+d MHz would be ≤ 0\n",
                      dev, p.coreClockOffsetMHz);
        allOk = false;
      }
      else
      {
        if (rsmiSetOdMax(dev, (uint64_t)target, RSMI_CLK_TYPE_SYS))
          TNN_LOG_INFO("RSMI: GPU[%d] core OD max → %lld MHz (base %llu %+d)\n",
                       dev, (long long)target,
                       (unsigned long long)baseMHz, p.coreClockOffsetMHz);
        else
          allOk = false;
      }
    }
  }

  // ---- Memory clock offset ----------------------------------------
  if (p.memClockOffsetMHz != 0)
  {
    uint64_t baseMHz = (dev < (int)g_base.size()) ? g_base[dev].rsmiMaxMclkMHz : 0;
    if (baseMHz == 0)
    {
      TNN_LOG_ERROR("RSMI: GPU[%d] no baseline MCLK for offset calculation\n", dev);
      allOk = false;
    }
    else
    {
      int64_t target = (int64_t)baseMHz + p.memClockOffsetMHz;
      if (target <= 0)
      {
        TNN_LOG_ERROR("RSMI: GPU[%d] mem offset %+d MHz would be ≤ 0\n",
                      dev, p.memClockOffsetMHz);
        allOk = false;
      }
      else
      {
        if (rsmiSetOdMax(dev, (uint64_t)target, RSMI_CLK_TYPE_MEM))
          TNN_LOG_INFO("RSMI: GPU[%d] mem OD max → %lld MHz (base %llu %+d)\n",
                       dev, (long long)target,
                       (unsigned long long)baseMHz, p.memClockOffsetMHz);
        else
          allOk = false;
      }
    }
  }

  return allOk;
}

static bool resetRsmi(int dev)
{
  bool ok = true;

  // Restore AUTO perf level → driver reclaims full DPM table
  if (pR_PerfLvlSet)
  {
    int r = pR_PerfLvlSet(dev, RSMI_PERF_LEVEL_AUTO);
    if (r != RSMI_OK)
    {
      TNN_LOG_ERROR("RSMI: GPU[%d] perf_level_set(AUTO) → error %d\n", dev, r);
      ok = false;
    }
  }

  // Restore saved power cap
  if (dev < (int)g_base.size() && g_base[dev].saved && g_base[dev].rsmiPowerCapUw > 0)
  {
    int r = pR_PowerCapSet(dev, 0, g_base[dev].rsmiPowerCapUw);
    if (r != RSMI_OK)
    {
      TNN_LOG_ERROR("RSMI: GPU[%d] restore power cap → error %d\n", dev, r);
      ok = false;
    }
  }
  return ok;
}

#ifdef _WIN32
// ======================================================================
//  ADL Overdrive8  —  Windows AMD, Navi / RDNA1+
//
//  Piggybacks on devInfo's ADL2 context + HIP→ADL adapter map.
//  >>> Call initPowerMonitoring() BEFORE initGpuTuning() on Windows. <<<
// ======================================================================

#include <adl_defines.h>   // OD8_* setting IDs, ADL_OD8_* cap bits, ADL_OK

extern void* adlGetContext();
extern void* adlGetLibHandle();
extern int   adlGetAdapterIndex(int oroDev);

typedef void* ADL_CONTEXT_HANDLE;

// ---- OD8 structs (inlined from adl_structures.h) --------------------
// Array size is deliberately larger than any current OD8_COUNT so the
// driver can't overflow the CurrentSetting output buffer.
#define TNN_OD8_MAX 96

struct ADLOD8SingleInitSetting {
    int featureID;
    int minValue;
    int maxValue;
    int defaultValue;
};

struct ADLOD8SingleSetSetting {
    int value;
    int requested;   // 1 = apply this slot
    int reset;       // 1 = reset this slot to driver default
};

struct ADLOD8SetSetting {
    int count;
    ADLOD8SingleSetSetting od8SettingTable[TNN_OD8_MAX];
};

struct ADLOD8CurrentSetting {
    int count;
    int Od8SettingTable[TNN_OD8_MAX];
};

// ---- OD8 function pointers ------------------------------------------
typedef int (*fn_OD8_InitX2)(ADL_CONTEXT_HANDLE, int,
                             int* /*caps*/, int* /*nFeat*/,
                             ADLOD8SingleInitSetting** /*driver-alloc*/);
typedef int (*fn_OD8_Set)   (ADL_CONTEXT_HANDLE, int,
                             ADLOD8SetSetting*, ADLOD8CurrentSetting*);

static fn_OD8_InitX2 pA_InitX2 = nullptr;
static fn_OD8_Set    pA_Set    = nullptr;
static T_LIB         g_adlOcLib = nullptr;

// ---- Per-device cached capability table -----------------------------
struct Od8Caps {
    bool    ok    = false;
    int     caps  = 0;           // ADL_OD8_* bitmask
    int     nFeat = 0;           // driver's feature count
    ADLOD8SingleInitSetting init[TNN_OD8_MAX] = {};  // indexed by ADLOD8SettingId
};
static std::vector<Od8Caps> g_od8;

// ---- Init ------------------------------------------------------------
static bool initAdlOc() {
    void* ctx = adlGetContext();
    if (!ctx) {
        TNN_LOG_DEBUG("ADL OC: no ADL2 context (call initPowerMonitoring first)\n");
        return false;
    }

    // Reuse the ADL library handle from devInfo (same DLL, same init context)
    g_adlOcLib = (T_LIB)adlGetLibHandle();
    if (!g_adlOcLib) {
        TNN_LOG_DEBUG("ADL OC: no ADL library handle from devInfo\n");
        return false;
    }

    pA_InitX2 = (fn_OD8_InitX2) T_SYM(g_adlOcLib, "ADL2_Overdrive8_Init_SettingX2_Get");
    pA_Set    = (fn_OD8_Set)    T_SYM(g_adlOcLib, "ADL2_Overdrive8_Setting_Set");

    if (!pA_InitX2 || !pA_Set) {
        TNN_LOG_ERROR("ADL OC: OD8 symbols missing — driver too old for Overdrive8?\n");
        T_CLOSE(g_adlOcLib); g_adlOcLib = nullptr;
        return false;
    }

    int nGpu = getGPUCount();
    g_od8.assign(nGpu, {});
    bool any = false;

    for (int h = 0; h < nGpu; ++h) {
        int adlIdx = adlGetAdapterIndex(h);
        if (adlIdx < 0) continue;

        Od8Caps& c = g_od8[h];
        c.nFeat = OD8_COUNT;  // Must be pre-initialized per ADL SDK
        ADLOD8SingleInitSetting* list = nullptr;

        TNN_LOG_DEBUG("ADL OC: HIP[%d] calling Init_SettingX2_Get(ctx=%p, adlIdx=%d, nFeat=%d)\n", h, ctx, adlIdx, c.nFeat);
        int r = pA_InitX2((ADL_CONTEXT_HANDLE)ctx, adlIdx, &c.caps, &c.nFeat, &list);
        if (r != ADL_OK || !list || c.nFeat <= 0) {
            if (r == -9)
                TNN_LOG_INFO("[GPU OC] ADL: GPU %d (adlIdx=%d) — ADL_ERR_NULL_POINTER (-9). "
                             "Context or adapter index may be invalid.\n", h, adlIdx);
            else
                TNN_LOG_INFO("[GPU OC] ADL: GPU %d (adlIdx=%d) — Init_SettingX2_Get error %d\n", h, adlIdx, r);
            if (list) free(list);   // allocated via the malloc callback devInfo registered
            continue;
        }

        int nCopy = (std::min)(c.nFeat, (int)TNN_OD8_MAX);
        memcpy(c.init, list, nCopy * sizeof(ADLOD8SingleInitSetting));
        free(list);

        c.ok = true;
        any  = true;

        TNN_LOG_DEBUG(
            "ADL OC: HIP[%d] caps=0x%08x feat=%d  "
            "GFX_LIMITS=%d GFX_CURVE=%d UCLK=%d POWER=%d\n",
            h, c.caps, c.nFeat,
            !!(c.caps & ADL_OD8_GFXCLK_LIMITS),
            !!(c.caps & ADL_OD8_GFXCLK_CURVE),
            !!(c.caps & ADL_OD8_UCLK_MAX),
            !!(c.caps & ADL_OD8_POWER_LIMIT));

        // Dump the clock-related ranges so the user sees what's legal
        auto dump = [&](int id, const char* name) {
            if (id < c.nFeat)
                TNN_LOG_DEBUG("  %-20s  [%5d .. %5d]  default=%d\n",
                              name, c.init[id].minValue, c.init[id].maxValue,
                              c.init[id].defaultValue);
        };
        dump(OD8_GFXCLK_FMAX,      "GFXCLK_FMAX");
        dump(OD8_GFXCLK_FMIN,      "GFXCLK_FMIN");
        dump(OD8_GFXCLK_FREQ3,     "GFXCLK_FREQ3");
        dump(OD8_UCLK_FMAX,        "UCLK_FMAX");
        dump(OD8_POWER_PERCENTAGE, "POWER_PERCENTAGE");
    }

    if (!any) { T_CLOSE(g_adlOcLib); g_adlOcLib = nullptr; }
    return any;
}

// ---- Stage helper: clamp to init range, mark requested --------------
static bool od8Stage(ADLOD8SetSetting& ss, const Od8Caps& c,
                     int id, int value, const char* label, int hip)
{
    if (id < 0 || id >= c.nFeat || id >= TNN_OD8_MAX) return false;
    const auto& lim = c.init[id];

    int v = value;
    if (v < lim.minValue || v > lim.maxValue) {
        int cl = (std::max)(lim.minValue, (std::min)(v, lim.maxValue));
        TNN_LOG_ERROR("ADL OC: HIP[%d] %s=%d outside [%d..%d], clamped to %d\n",
                      hip, label, v, lim.minValue, lim.maxValue, cl);
        v = cl;
    }

    ss.od8SettingTable[id] = { v, 1, 0 };
    return true;
}

// ---- Apply -----------------------------------------------------------
static bool applyAdl(int hip, const GpuTuneParams& p) {
    if (hip < 0 || hip >= (int)g_od8.size() || !g_od8[hip].ok) {
        TNN_LOG_ERROR("ADL OC: HIP[%d] OD8 unavailable\n", hip);
        return false;
    }
    const Od8Caps& c = g_od8[hip];
    void* ctx  = adlGetContext();
    int adlIdx = adlGetAdapterIndex(hip);

    ADLOD8SetSetting ss{};
    ss.count = c.nFeat;
    bool ok = true, staged = false;

    // Which GFXCLK slot does this card expose?
    //   RDNA2/3 → GFXCLK_LIMITS → write FMAX
    //   RDNA1   → GFXCLK_CURVE  → write FREQ3 (top curve point)
    int gfxSlot = -1; const char* gfxLbl = nullptr;
    if      (c.caps & ADL_OD8_GFXCLK_LIMITS) { gfxSlot = OD8_GFXCLK_FMAX;  gfxLbl = "GFXCLK_FMAX";  }
    else if (c.caps & ADL_OD8_GFXCLK_CURVE)  { gfxSlot = OD8_GFXCLK_FREQ3; gfxLbl = "GFXCLK_FREQ3"; }

    bool touchClk = (p.coreClockMHz > 0 || p.coreClockOffsetMHz != 0 ||
                     p.memClockOffsetMHz != 0);

    // ── power limit ── OD8 is PERCENTAGE OFFSET, not watts ─────────
    if (p.powerLimitW != 0) {
        if (!(c.caps & ADL_OD8_POWER_LIMIT)) {
            TNN_LOG_ERROR("ADL OC: HIP[%d] POWER_LIMIT cap not present\n", hip);
            ok = false;
        } else {
            static bool warned = false;
            if (!warned) {
                TNN_LOG_INFO("ADL OC: NOTE — on Windows AMD, --gpu-plimit is a "
                             "percentage offset (e.g. -20 = 80%% of stock TDP)\n");
                warned = true;
            }
            staged |= od8Stage(ss, c, OD8_POWER_PERCENTAGE, p.powerLimitW,
                               "POWER_PERCENTAGE", hip);
        }
    }

    // ── disable auto-OC/UV before manual clock writes ──────────────
    if (touchClk) {
        if ((c.caps & ADL_OD8_AUTO_OC_ENGINE) && OD8_AUTO_OC_ENGINE_CONTROL < c.nFeat)
            { ss.od8SettingTable[OD8_AUTO_OC_ENGINE_CONTROL] = {0,1,0}; staged = true; }
        if ((c.caps & ADL_OD8_AUTO_UV_ENGINE) && OD8_AUTO_UV_ENGINE_CONTROL < c.nFeat)
            { ss.od8SettingTable[OD8_AUTO_UV_ENGINE_CONTROL] = {0,1,0}; staged = true; }
        if ((c.caps & ADL_OD8_AUTO_OC_MEMORY) && OD8_AUTO_OC_MEMORY_CONTROL < c.nFeat)
            { ss.od8SettingTable[OD8_AUTO_OC_MEMORY_CONTROL] = {0,1,0}; staged = true; }
    }

    // ── core clock absolute ────────────────────────────────────────
    if (p.coreClockMHz > 0) {
        if (gfxSlot < 0) {
            TNN_LOG_ERROR("ADL OC: HIP[%d] no GFXCLK control path (caps=0x%x)\n", hip, c.caps);
            ok = false;
        } else if (od8Stage(ss, c, gfxSlot, p.coreClockMHz, gfxLbl, hip)) {
            staged = true;
            TNN_LOG_INFO("ADL OC: HIP[%d] %s → %d MHz\n", hip, gfxLbl, p.coreClockMHz);
        } else ok = false;
    }

    // ── core clock offset (only if absolute not given) ─────────────
    if (p.coreClockOffsetMHz != 0 && p.coreClockMHz == 0) {
        if (gfxSlot < 0) {
            TNN_LOG_ERROR("ADL OC: HIP[%d] no GFXCLK control path\n", hip);
            ok = false;
        } else {
            int base   = c.init[gfxSlot].defaultValue;
            int target = base + p.coreClockOffsetMHz;
            if (od8Stage(ss, c, gfxSlot, target, gfxLbl, hip)) {
                staged = true;
                TNN_LOG_INFO("ADL OC: HIP[%d] %s → %d MHz (default %d %+d)\n",
                             hip, gfxLbl, target, base, p.coreClockOffsetMHz);
            } else ok = false;
        }
    }

    // ── memory clock offset ────────────────────────────────────────
    if (p.memClockOffsetMHz != 0) {
        if (!(c.caps & ADL_OD8_UCLK_MAX)) {
            TNN_LOG_ERROR("ADL OC: HIP[%d] UCLK_MAX cap not present\n", hip);
            ok = false;
        } else {
            int base   = c.init[OD8_UCLK_FMAX].defaultValue;
            int target = base + p.memClockOffsetMHz;
            if (od8Stage(ss, c, OD8_UCLK_FMAX, target, "UCLK_FMAX", hip)) {
                staged = true;
                TNN_LOG_INFO("ADL OC: HIP[%d] UCLK_FMAX → %d MHz (default %d %+d)\n",
                             hip, target, base, p.memClockOffsetMHz);
            } else ok = false;
        }
    }

    // ── commit ─────────────────────────────────────────────────────
    if (!staged) return ok;

    ADLOD8CurrentSetting cur{};
    cur.count = c.nFeat;

    int r = pA_Set((ADL_CONTEXT_HANDLE)ctx, adlIdx, &ss, &cur);
    if (r != ADL_OK) {
        TNN_LOG_ERROR("ADL OC: HIP[%d] Overdrive8_Setting_Set → %d\n", hip, r);
        return false;
    }
    return ok;
}

// ---- Reset -----------------------------------------------------------
static bool resetAdl(int hip) {
    if (hip < 0 || hip >= (int)g_od8.size() || !g_od8[hip].ok) return true;
    const Od8Caps& c = g_od8[hip];
    void* ctx  = adlGetContext();
    int adlIdx = adlGetAdapterIndex(hip);

    ADLOD8SetSetting ss{};
    ss.count = c.nFeat;

    // requested=1 + reset=1 on everything we might have touched.
    static const int ids[] = {
        OD8_GFXCLK_FMAX, OD8_GFXCLK_FMIN, OD8_GFXCLK_FREQ3,
        OD8_UCLK_FMAX,   OD8_POWER_PERCENTAGE,
        OD8_AUTO_OC_ENGINE_CONTROL, OD8_AUTO_UV_ENGINE_CONTROL,
        OD8_AUTO_OC_MEMORY_CONTROL,
    };
    for (int id : ids)
        if (id < c.nFeat && id < TNN_OD8_MAX)
            ss.od8SettingTable[id] = { 0, 1, 1 };

    ADLOD8CurrentSetting cur{};
    cur.count = c.nFeat;

    int r = pA_Set((ADL_CONTEXT_HANDLE)ctx, adlIdx, &ss, &cur);
    if (r != ADL_OK)
        TNN_LOG_ERROR("ADL OC: HIP[%d] reset → %d\n", hip, r);
    return r == ADL_OK;
}

static void shutdownAdlOc() {
    g_od8.clear();
    // Don't close g_adlOcLib — it's borrowed from devInfo's power monitoring
    g_adlOcLib = nullptr;
}

#endif // _WIN32

// ======================================================================
//  Public API
// ======================================================================
bool initGpuTuning() {
#ifdef _WIN32
    if (initAdlOc())  { g_back = TuneBack::ADL;  return true; }
#else
    if (initRsmi())   { g_back = TuneBack::RSMI; return true; }
    if (g_lib) { T_CLOSE(g_lib); g_lib = nullptr; }
#endif
    if (initNvml())   { g_back = TuneBack::NVML; return true; }
    TNN_LOG_DEBUG("GPU tuning: no backend available\n");
    return false;
}

bool applyGpuTuning(int dev, const GpuTuneParams& p) {
    if (dev < 0 || dev >= getGPUCount()) return false;
    switch (g_back) {
        case TuneBack::NVML: return applyNvml(dev, p);
        case TuneBack::RSMI: return applyRsmi(dev, p);
#ifdef _WIN32
        case TuneBack::ADL:  return applyAdl(dev, p);
#endif
        default: return false;
    }
}

bool resetGpuTuning(int dev) {
    if (dev < 0 || dev >= getGPUCount()) return false;
    switch (g_back) {
        case TuneBack::NVML: return resetNvml(dev);
        case TuneBack::RSMI: return resetRsmi(dev);
#ifdef _WIN32
        case TuneBack::ADL:  return resetAdl(dev);
#endif
        default: return false;
    }
}

void shutdownGpuTuning() {
    int n = getGPUCount();
    for (int i = 0; i < n; ++i) resetGpuTuning(i);
    switch (g_back) {
        case TuneBack::NVML: if (pN_Shutdown) pN_Shutdown(); break;
        case TuneBack::RSMI: if (pR_Shutdown) pR_Shutdown(); break;
#ifdef _WIN32
        case TuneBack::ADL:  shutdownAdlOc(); break;
#endif
        default: break;
    }
    if (g_lib) { T_CLOSE(g_lib); g_lib = nullptr; }
    g_back = TuneBack::None;
}

std::string getTuneBackendName()
{
  switch (g_back)
  {
  case TuneBack::NVML:
    return "NVML";
  case TuneBack::RSMI:
    return "RSMI";
  case TuneBack::ADL:
    return "ADL";
  default:
    return "None";
  }
}