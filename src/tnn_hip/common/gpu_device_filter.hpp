#pragma once

#include <set>

// GPU device filtering (--devices / --gpu-disable)
// Defined in miner.cpp, populated from CLI args.
extern std::set<int> HIP_includeDevices;
extern std::set<int> HIP_excludeDevices;

inline bool shouldUseDevice(int d) {
    if (!HIP_includeDevices.empty())
        return HIP_includeDevices.count(d) > 0;
    return HIP_excludeDevices.count(d) == 0;
}

// Drop-in gate for GPU init loops: place TNN_GPU_GATE(d) at top of loop body
#define TNN_GPU_GATE(d) if (!shouldUseDevice(d)) continue;
