#include "devInfo.hip.h"
#include "tnn_log.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <cstring>
#include <cstdio>

#ifdef _WIN32
  #include <windows.h>
  #include <adl_defines.h>        // vendored: brings in ADL_PMLOG_SENSORS enum
  #define LIB_HANDLE HMODULE
  #define LIB_OPEN(name) LoadLibraryA(name)
  #define LIB_SYM(h, sym) ((void*)GetProcAddress(h, sym))
  #define LIB_CLOSE(h) FreeLibrary(h)
#else
  #include <dlfcn.h>
  #define LIB_HANDLE void*
  #define LIB_OPEN(name) dlopen(name, RTLD_LAZY)
  #define LIB_SYM(h, sym) dlsym(h, sym)
  #define LIB_CLOSE(h) dlclose(h)
#endif

// ============== HIP Device Info ==============

int getGPUCount() {
    int n;
    hipError_t e = hipGetDeviceCount(&n);
    if (e != hipSuccess) {
        TNN_LOG_ERROR("HIP: hipGetDeviceCount failed: %s\n", hipGetErrorString(e));
        return 0;
    }
    return n;
}

std::string getDeviceName(int device) {
    hipDeviceProp_t p;
    hipError_t e = hipGetDeviceProperties(&p, device);
    if (e != hipSuccess) {
        TNN_LOG_ERROR("HIP: hipGetDeviceProperties(%d) failed: %s\n", device, hipGetErrorString(e));
        return "";
    }
    return std::string(p.name);
}

std::string getPCIBusId(int device) {
    char buf[64];
    hipError_t e = hipDeviceGetPCIBusId(buf, sizeof(buf), device);
    if (e != hipSuccess) {
        TNN_LOG_ERROR("HIP: hipDeviceGetPCIBusId(%d) failed: %s\n", device, hipGetErrorString(e));
        return "";
    }
    return std::string(buf);
}

// ============== Power Monitoring ==============

enum class PowerBackend { None, RSMI, NVML, ADL };
static PowerBackend g_powerBackend   = PowerBackend::None;
static LIB_HANDLE   g_powerLib       = nullptr;
static bool         g_powerAvailable = false;

// --- ROCm SMI (Linux AMD) ---
typedef int (*rsmi_init_fn)(uint64_t);
typedef int (*rsmi_shut_down_fn)();
typedef int (*rsmi_dev_power_ave_get_fn)(uint32_t, uint32_t, uint64_t*);
static rsmi_init_fn              p_rsmi_init              = nullptr;
static rsmi_shut_down_fn         p_rsmi_shut_down         = nullptr;
static rsmi_dev_power_ave_get_fn p_rsmi_dev_power_ave_get = nullptr;

// --- NVML (NVIDIA, both platforms) ---
typedef void* nvmlDevice_t;
typedef int (*nvmlInit_fn)();
typedef int (*nvmlShutdown_fn)();
typedef int (*nvmlDeviceGetHandleByIndex_fn)(unsigned int, nvmlDevice_t*);
typedef int (*nvmlDeviceGetPowerUsage_fn)(nvmlDevice_t, unsigned int*);
static nvmlInit_fn                   p_nvmlInit                   = nullptr;
static nvmlShutdown_fn               p_nvmlShutdown               = nullptr;
static nvmlDeviceGetHandleByIndex_fn p_nvmlDeviceGetHandleByIndex = nullptr;
static nvmlDeviceGetPowerUsage_fn    p_nvmlDeviceGetPowerUsage    = nullptr;

#ifdef _WIN32
// =====================================================================
// ADL (Windows AMD) — streaming PMLog API
// =====================================================================

// adl_defines.h gives us the ADL_PMLOG_SENSORS enum. We still need a few
// structs from adl_structures.h — inline them to avoid pulling the whole SDK.
// NOTE: default packing, no #pragma pack.

typedef struct AdapterInfo {
    int  iSize;
    int  iAdapterIndex;
    char strUDID[256];
    int  iBusNumber;
    int  iDeviceNumber;
    int  iFunctionNumber;
    int  iVendorID;
    char strAdapterName[256];
    char strDisplayName[256];
    int  iPresent;
    int  iExist;
    char strDriverPath[256];
    char strDriverPathExt[256];
    char strPNPString[256];
    int  iOSDisplayIndex;
} AdapterInfo, *LPAdapterInfo;

#ifndef ADL_PMLOG_MAX_SENSORS
#define ADL_PMLOG_MAX_SENSORS 256
#endif

typedef struct ADLPMLogSupportInfo {
    unsigned short usSensors[ADL_PMLOG_MAX_SENSORS];  // enum values, 0-terminated
    int            iReserved[16];
} ADLPMLogSupportInfo;

typedef struct ADLPMLogStartInput {
    unsigned short usSensors[ADL_PMLOG_MAX_SENSORS];  // requested sensors, 0-terminated
    unsigned long  ulSampleRate;                       // ms
    int            iReserved[15];
} ADLPMLogStartInput;

typedef struct ADLPMLogData {
    unsigned int       ulVersion;
    unsigned int       ulActiveSampleRate;
    unsigned long long ulLastUpdated;
    unsigned int       ulValues[ADL_PMLOG_MAX_SENSORS][2];  // [i][0]=enum, [i][1]=value
    unsigned int       ulReserved[256];
} ADLPMLogData;

typedef struct ADLPMLogStartOutput {
    union {
        void*              pLoggingAddress;
        unsigned long long _align;
    };
    int iReserved[14];
} ADLPMLogStartOutput;

typedef void*        ADL_CONTEXT_HANDLE;
typedef unsigned int ADL_D3DKMT_HANDLE;
typedef void* (__stdcall *ADL_MAIN_MALLOC_CALLBACK)(int);

// Legacy ADL (global state) — adapter enumeration
typedef int (*ADL_Main_Control_Create_fn)(ADL_MAIN_MALLOC_CALLBACK, int);
typedef int (*ADL_Main_Control_Destroy_fn)(void);
typedef int (*ADL_Adapter_NumberOfAdapters_Get_fn)(int*);
typedef int (*ADL_Adapter_AdapterInfo_Get_fn)(LPAdapterInfo, int);

// ADL2 (context-based) — PMLog
typedef int (*ADL2_Main_Control_Create_fn)(ADL_MAIN_MALLOC_CALLBACK, int, ADL_CONTEXT_HANDLE*);
typedef int (*ADL2_Main_Control_Destroy_fn)(ADL_CONTEXT_HANDLE);
typedef int (*ADL2_Adapter_PMLog_Support_Get_fn)(ADL_CONTEXT_HANDLE, int, ADLPMLogSupportInfo*);
typedef int (*ADL2_Adapter_PMLog_Start_fn)(ADL_CONTEXT_HANDLE, int, ADLPMLogStartInput*, ADLPMLogStartOutput*, ADL_D3DKMT_HANDLE);
typedef int (*ADL2_Adapter_PMLog_Stop_fn)(ADL_CONTEXT_HANDLE, int, ADL_D3DKMT_HANDLE);
typedef int (*ADL2_Device_PMLog_Device_Create_fn)(ADL_CONTEXT_HANDLE, int, ADL_D3DKMT_HANDLE*);
typedef int (*ADL2_Device_PMLog_Device_Destroy_fn)(ADL_CONTEXT_HANDLE, ADL_D3DKMT_HANDLE);

static ADL_Main_Control_Create_fn           p_ADL_Main_Control_Create           = nullptr;
static ADL_Main_Control_Destroy_fn          p_ADL_Main_Control_Destroy          = nullptr;
static ADL_Adapter_NumberOfAdapters_Get_fn  p_ADL_Adapter_NumberOfAdapters_Get  = nullptr;
static ADL_Adapter_AdapterInfo_Get_fn       p_ADL_Adapter_AdapterInfo_Get       = nullptr;
static ADL2_Main_Control_Create_fn          p_ADL2_Main_Control_Create          = nullptr;
static ADL2_Main_Control_Destroy_fn         p_ADL2_Main_Control_Destroy         = nullptr;
static ADL2_Adapter_PMLog_Support_Get_fn    p_ADL2_Adapter_PMLog_Support_Get    = nullptr;
static ADL2_Adapter_PMLog_Start_fn          p_ADL2_Adapter_PMLog_Start          = nullptr;
static ADL2_Adapter_PMLog_Stop_fn           p_ADL2_Adapter_PMLog_Stop           = nullptr;
static ADL2_Device_PMLog_Device_Create_fn   p_ADL2_Device_PMLog_Device_Create   = nullptr;
static ADL2_Device_PMLog_Device_Destroy_fn  p_ADL2_Device_PMLog_Device_Destroy  = nullptr;

static bool               g_adlLegacyInit = false;
static ADL_CONTEXT_HANDLE g_adl2Context   = nullptr;
static std::vector<int>   g_hipToAdlIndex;

struct PMLogState {
    ADL_D3DKMT_HANDLE hDevice     = 0;
    ADLPMLogData*     pData       = nullptr;   // driver-owned shared buffer
    bool              running     = false;
    int               powerRow    = -1;        // row in ulValues[] holding power
    int               powerEnum   = -1;        // which ADL_PMLOG_* enum that row is
};
static std::vector<PMLogState> g_pmlog;

static void* __stdcall ADL_Alloc(int sz) { return malloc((size_t)sz); }

static bool parsePCI(const std::string& s, int& bus, int& dev, int& fn) {
    unsigned dom, b, d, f;
    if (sscanf(s.c_str(), "%x:%x:%x.%x", &dom, &b, &d, &f) == 4 ||
        sscanf(s.c_str(), "%x:%x.%x",     &b, &d, &f)      == 3) {
        bus = (int)b; dev = (int)d; fn = (int)f;
        return true;
    }
    return false;
}

static const char* pmlogEnumName(int e) {
    switch (e) {
        case ADL_PMLOG_BOARD_POWER: return "BOARD_POWER";
        case ADL_PMLOG_ASIC_POWER:  return "ASIC_POWER";
        case ADL_PMLOG_GFX_POWER:   return "GFX_POWER";
        case ADL_PMLOG_SOC_POWER:   return "SOC_POWER";
        default:                    return "?";
    }
}

static bool startPMLog(int hipIdx) {
    int adlIdx = g_hipToAdlIndex[hipIdx];
    PMLogState& st = g_pmlog[hipIdx];
    if (st.running) return true;

    if (p_ADL2_Device_PMLog_Device_Create(g_adl2Context, adlIdx, &st.hDevice) != ADL_OK) {
        TNN_LOG_ERROR("ADL: PMLog device create failed (HIP %d, ADL %d)\n", hipIdx, adlIdx);
        return false;
    }

    ADLPMLogSupportInfo support{};
    if (p_ADL2_Adapter_PMLog_Support_Get(g_adl2Context, adlIdx, &support) != ADL_OK) {
        TNN_LOG_ERROR("ADL: PMLog support query failed (HIP %d)\n", hipIdx);
        p_ADL2_Device_PMLog_Device_Destroy(g_adl2Context, st.hDevice);
        st.hDevice = 0;
        return false;
    }

    // Build request: only the power sensors this adapter actually advertises.
    // Preference: BOARD > ASIC > GFX > SOC.
    static const int want[] = {
        ADL_PMLOG_BOARD_POWER, ADL_PMLOG_ASIC_POWER,
        ADL_PMLOG_GFX_POWER,   ADL_PMLOG_SOC_POWER,
    };
    ADLPMLogStartInput input{};   // zero-init → list is already 0-terminated
    int n = 0;
    for (int w : want) {
        for (int i = 0; i < ADL_PMLOG_MAX_SENSORS && support.usSensors[i] != ADL_SENSOR_MAXTYPES; ++i) {
            if (support.usSensors[i] == w) { input.usSensors[n++] = (unsigned short)w; break; }
        }
    }
    if (n == 0) {
        TNN_LOG_ERROR("ADL: adapter advertises no power sensors (HIP %d)\n", hipIdx);
        p_ADL2_Device_PMLog_Device_Destroy(g_adl2Context, st.hDevice);
        st.hDevice = 0;
        return false;
    }
    input.ulSampleRate = 100;  // ms

    ADLPMLogStartOutput output{};
    if (p_ADL2_Adapter_PMLog_Start(g_adl2Context, adlIdx, &input, &output, st.hDevice) != ADL_OK) {
        TNN_LOG_ERROR("ADL: PMLog_Start failed (HIP %d)\n", hipIdx);
        p_ADL2_Device_PMLog_Device_Destroy(g_adl2Context, st.hDevice);
        st.hDevice = 0;
        return false;
    }

    st.pData   = (ADLPMLogData*)output.pLoggingAddress;
    st.running = true;

    // Driver populates the shared buffer asynchronously; give it a moment.
    for (int tries = 0; tries < 20 && st.pData->ulValues[0][0] == ADL_SENSOR_MAXTYPES; ++tries)
        Sleep(10);

    // Find the row for our preferred sensor (first in want[] that the driver emits).
    for (int w : want) {
        for (int i = 0; i < ADL_PMLOG_MAX_SENSORS && st.pData->ulValues[i][0] != ADL_SENSOR_MAXTYPES; ++i) {
            if ((int)st.pData->ulValues[i][0] == w) {
                st.powerRow  = i;
                st.powerEnum = w;
                goto found;
            }
        }
    }
found:
    if (st.powerRow < 0) {
        TNN_LOG_ERROR("ADL: PMLog started but no power row appeared (HIP %d)\n", hipIdx);
        return false;
    }

    TNN_LOG_TRACE("ADL: HIP[%d] power = %s (enum %d, row %d)\n",
                 hipIdx, pmlogEnumName(st.powerEnum), st.powerEnum, st.powerRow);
    return true;
}

static void stopPMLog(int hipIdx) {
    PMLogState& st = g_pmlog[hipIdx];
    if (!st.running) return;
    int adlIdx = g_hipToAdlIndex[hipIdx];
    if (p_ADL2_Adapter_PMLog_Stop)          p_ADL2_Adapter_PMLog_Stop(g_adl2Context, adlIdx, st.hDevice);
    if (p_ADL2_Device_PMLog_Device_Destroy) p_ADL2_Device_PMLog_Device_Destroy(g_adl2Context, st.hDevice);
    st = PMLogState{};
}

static bool initADL() {
    g_powerLib = LIB_OPEN("atiadlxx.dll");
    if (!g_powerLib) g_powerLib = LIB_OPEN("atiadlxy.dll");
    if (!g_powerLib) {
        TNN_LOG_DEBUG("ADL: atiadlxx.dll not present (no AMD driver?)\n");
        return false;
    }

    p_ADL_Main_Control_Create           = (ADL_Main_Control_Create_fn)          LIB_SYM(g_powerLib, "ADL_Main_Control_Create");
    p_ADL_Main_Control_Destroy          = (ADL_Main_Control_Destroy_fn)         LIB_SYM(g_powerLib, "ADL_Main_Control_Destroy");
    p_ADL_Adapter_NumberOfAdapters_Get  = (ADL_Adapter_NumberOfAdapters_Get_fn) LIB_SYM(g_powerLib, "ADL_Adapter_NumberOfAdapters_Get");
    p_ADL_Adapter_AdapterInfo_Get       = (ADL_Adapter_AdapterInfo_Get_fn)      LIB_SYM(g_powerLib, "ADL_Adapter_AdapterInfo_Get");
    p_ADL2_Main_Control_Create          = (ADL2_Main_Control_Create_fn)         LIB_SYM(g_powerLib, "ADL2_Main_Control_Create");
    p_ADL2_Main_Control_Destroy         = (ADL2_Main_Control_Destroy_fn)        LIB_SYM(g_powerLib, "ADL2_Main_Control_Destroy");
    p_ADL2_Adapter_PMLog_Support_Get    = (ADL2_Adapter_PMLog_Support_Get_fn)   LIB_SYM(g_powerLib, "ADL2_Adapter_PMLog_Support_Get");
    p_ADL2_Adapter_PMLog_Start          = (ADL2_Adapter_PMLog_Start_fn)         LIB_SYM(g_powerLib, "ADL2_Adapter_PMLog_Start");
    p_ADL2_Adapter_PMLog_Stop           = (ADL2_Adapter_PMLog_Stop_fn)          LIB_SYM(g_powerLib, "ADL2_Adapter_PMLog_Stop");
    p_ADL2_Device_PMLog_Device_Create   = (ADL2_Device_PMLog_Device_Create_fn)  LIB_SYM(g_powerLib, "ADL2_Device_PMLog_Device_Create");
    p_ADL2_Device_PMLog_Device_Destroy  = (ADL2_Device_PMLog_Device_Destroy_fn) LIB_SYM(g_powerLib, "ADL2_Device_PMLog_Device_Destroy");

    if (!p_ADL_Main_Control_Create || !p_ADL_Main_Control_Destroy ||
        !p_ADL_Adapter_NumberOfAdapters_Get || !p_ADL_Adapter_AdapterInfo_Get ||
        !p_ADL2_Main_Control_Create || !p_ADL2_Main_Control_Destroy ||
        !p_ADL2_Adapter_PMLog_Support_Get || !p_ADL2_Adapter_PMLog_Start ||
        !p_ADL2_Adapter_PMLog_Stop || !p_ADL2_Device_PMLog_Device_Create ||
        !p_ADL2_Device_PMLog_Device_Destroy) {
        TNN_LOG_ERROR("ADL: required symbols missing (driver too old for PMLog?)\n");
        LIB_CLOSE(g_powerLib); g_powerLib = nullptr;
        return false;
    }

    if (p_ADL_Main_Control_Create(ADL_Alloc, 1) != ADL_OK) {
        TNN_LOG_ERROR("ADL: ADL_Main_Control_Create failed\n");
        LIB_CLOSE(g_powerLib); g_powerLib = nullptr;
        return false;
    }
    g_adlLegacyInit = true;

    if (p_ADL2_Main_Control_Create(ADL_Alloc, 1, &g_adl2Context) != ADL_OK) {
        TNN_LOG_ERROR("ADL: ADL2_Main_Control_Create failed\n");
        g_adl2Context = nullptr;
        return false;
    }

    int numAdapters = 0;
    if (p_ADL_Adapter_NumberOfAdapters_Get(&numAdapters) != ADL_OK || numAdapters <= 0) {
        TNN_LOG_ERROR("ADL: no adapters enumerated\n");
        return false;
    }
    TNN_LOG_DEBUG("ADL: %d adapter entries\n", numAdapters);

    std::vector<AdapterInfo> info(numAdapters);
    memset(info.data(), 0, sizeof(AdapterInfo) * numAdapters);
    p_ADL_Adapter_AdapterInfo_Get(info.data(), (int)(sizeof(AdapterInfo) * numAdapters));

    int hipCount = getGPUCount();
    g_hipToAdlIndex.assign(hipCount, -1);
    g_pmlog.assign(hipCount, PMLogState{});

    for (int h = 0; h < hipCount; ++h) {
        std::string pci = getPCIBusId(h);
        int hb, hd, hf;
        if (!parsePCI(pci, hb, hd, hf)) {
            TNN_LOG_DEBUG("ADL: HIP[%d] PCI parse failed: \"%s\"\n", h, pci.c_str());
            continue;
        }
        for (int a = 0; a < numAdapters; ++a) {
            if (info[a].iVendorID != 1002) continue;                // AMD vendor ID (decimal)
            if (info[a].iBusNumber != hb || info[a].iDeviceNumber != hd) continue;
            g_hipToAdlIndex[h] = info[a].iAdapterIndex;
            TNN_LOG_DEBUG("ADL: HIP[%d] \"%s\" -> ADL idx %d (bus %d dev %d)\n",
                          h, info[a].strAdapterName, info[a].iAdapterIndex, hb, hd);
            break;
        }
        if (g_hipToAdlIndex[h] < 0)
            TNN_LOG_DEBUG("ADL: HIP[%d] (%s) has no matching ADL adapter\n", h, pci.c_str());
    }

    bool any = false;
    for (int h = 0; h < hipCount; ++h)
        if (g_hipToAdlIndex[h] >= 0 && startPMLog(h))
            any = true;

    return any;
}

static double getADLPowerWatts(int hipDev) {
    if (hipDev < 0 || hipDev >= (int)g_pmlog.size()) return 0.0;
    const PMLogState& st = g_pmlog[hipDev];
    if (!st.running || !st.pData || st.powerRow < 0) return 0.0;
    // Driver writes into pData continuously; just read the latest sample.
    return (double)st.pData->ulValues[st.powerRow][1];
}

static void shutdownADL() {
    for (int h = 0; h < (int)g_pmlog.size(); ++h) stopPMLog(h);
    g_pmlog.clear();
    g_hipToAdlIndex.clear();
    if (g_adl2Context && p_ADL2_Main_Control_Destroy) {
        p_ADL2_Main_Control_Destroy(g_adl2Context);
        g_adl2Context = nullptr;
    }
    if (g_adlLegacyInit && p_ADL_Main_Control_Destroy) {
        p_ADL_Main_Control_Destroy();
        g_adlLegacyInit = false;
    }
}

void* adlGetContext()              { return g_adl2Context; }
void* adlGetLibHandle()            { return (void*)g_powerLib; }
int   adlGetAdapterIndex(int hip)  {
    return (hip >= 0 && hip < (int)g_hipToAdlIndex.size())
           ? g_hipToAdlIndex[hip] : -1;
}

#endif // _WIN32

// ============== Public API ==============

static bool tryNVML() {
#ifdef _WIN32
    g_powerLib = LIB_OPEN("nvml.dll");
#else
    g_powerLib = LIB_OPEN("libnvidia-ml.so.1");
    if (!g_powerLib) g_powerLib = LIB_OPEN("libnvidia-ml.so");
#endif
    if (!g_powerLib) return false;

    p_nvmlInit                   = (nvmlInit_fn)                   LIB_SYM(g_powerLib, "nvmlInit_v2");
    if (!p_nvmlInit) p_nvmlInit  = (nvmlInit_fn)                   LIB_SYM(g_powerLib, "nvmlInit");
    p_nvmlShutdown               = (nvmlShutdown_fn)               LIB_SYM(g_powerLib, "nvmlShutdown");
    p_nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_fn) LIB_SYM(g_powerLib, "nvmlDeviceGetHandleByIndex_v2");
    if (!p_nvmlDeviceGetHandleByIndex)
        p_nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_fn) LIB_SYM(g_powerLib, "nvmlDeviceGetHandleByIndex");
    p_nvmlDeviceGetPowerUsage    = (nvmlDeviceGetPowerUsage_fn)    LIB_SYM(g_powerLib, "nvmlDeviceGetPowerUsage");

    if (p_nvmlInit && p_nvmlShutdown && p_nvmlDeviceGetHandleByIndex && p_nvmlDeviceGetPowerUsage &&
        p_nvmlInit() == 0) {
        g_powerBackend = PowerBackend::NVML;
        g_powerAvailable = true;
        TNN_LOG_DEBUG("NVML: initialized\n");
        return true;
    }
    LIB_CLOSE(g_powerLib); g_powerLib = nullptr;
    return false;
}

bool initPowerMonitoring() {
#ifdef _WIN32
    if (initADL()) {
        g_powerBackend = PowerBackend::ADL;
        g_powerAvailable = true;
        return true;
    }
    // ADL may have partially initialized (loaded dll, created contexts) but
    // returned false. Clean up before trying NVML, else g_powerLib leaks.
    shutdownADL();
    if (g_powerLib) { LIB_CLOSE(g_powerLib); g_powerLib = nullptr; }
#else
    g_powerLib = LIB_OPEN("librocm_smi64.so");
    if (g_powerLib) {
        p_rsmi_init              = (rsmi_init_fn)              LIB_SYM(g_powerLib, "rsmi_init");
        p_rsmi_shut_down         = (rsmi_shut_down_fn)         LIB_SYM(g_powerLib, "rsmi_shut_down");
        p_rsmi_dev_power_ave_get = (rsmi_dev_power_ave_get_fn) LIB_SYM(g_powerLib, "rsmi_dev_power_ave_get");
        if (p_rsmi_init && p_rsmi_shut_down && p_rsmi_dev_power_ave_get && p_rsmi_init(0) == 0) {
            g_powerBackend = PowerBackend::RSMI;
            g_powerAvailable = true;
            TNN_LOG_DEBUG("ROCm SMI: initialized\n");
            return true;
        }
        LIB_CLOSE(g_powerLib); g_powerLib = nullptr;
    }
#endif
    return tryNVML();
}

void shutdownPowerMonitoring() {
    switch (g_powerBackend) {
        case PowerBackend::RSMI: if (p_rsmi_shut_down) p_rsmi_shut_down(); break;
        case PowerBackend::NVML: if (p_nvmlShutdown)   p_nvmlShutdown();   break;
#ifdef _WIN32
        case PowerBackend::ADL:  shutdownADL(); break;
#endif
        default: break;
    }
    if (g_powerLib) { LIB_CLOSE(g_powerLib); g_powerLib = nullptr; }
    g_powerAvailable = false;
    g_powerBackend = PowerBackend::None;
}

double getDevicePowerWatts(int device) {
    if (!g_powerAvailable || device < 0) return 0.0;
    switch (g_powerBackend) {
        case PowerBackend::RSMI: {
            uint64_t uw = 0;
            if (p_rsmi_dev_power_ave_get((uint32_t)device, 0, &uw) == 0) return (double)uw / 1e6;
            break;
        }
        case PowerBackend::NVML: {
            nvmlDevice_t h = nullptr;
            if (p_nvmlDeviceGetHandleByIndex((unsigned)device, &h) == 0) {
                unsigned mw = 0;
                if (p_nvmlDeviceGetPowerUsage(h, &mw) == 0) return (double)mw / 1e3;
            }
            break;
        }
#ifdef _WIN32
        case PowerBackend::ADL: return getADLPowerWatts(device);
#endif
        default: break;
    }
    return 0.0;
}