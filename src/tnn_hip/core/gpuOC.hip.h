// gpuOC.hip.h
#pragma once
#include <string>

struct GpuTuneParams {
    int coreClockOffsetMHz = 0;   // --gpu-coff   (V/F curve offset, MHz; NVML native, AMD emulated)
    int coreClockMHz       = 0;   // --gpu-cclock (lock / OD-max, MHz; 0 = don't touch)
    int memClockOffsetMHz  = 0;   // --gpu-moff   (memory offset, MHz; 0 = don't touch)
    int powerLimitW        = 0;   // --gpu-plimit (watts; 0 = don't touch)
};

/// Call once after HIP init.  Returns true if any tuning backend loaded.
bool initGpuTuning();

/// Apply one or more dials to a single GPU.
/// Fields left at 0 are skipped.  Returns true if every requested setting succeeded.
bool applyGpuTuning(int deviceIndex, const GpuTuneParams& params);

/// Restore saved baseline for one GPU (clocks → default, power → original cap).
bool resetGpuTuning(int deviceIndex);

/// Shutdown the tuning backend and release library handles.
void shutdownGpuTuning();

/// Diagnostic: "NVML", "RSMI", or "None".
std::string getTuneBackendName();