#pragma once

#include <string>

#ifdef _WIN32
// Shared ADL state for the OC layer. Only valid after initPowerMonitoring()
// has succeeded with the ADL backend.
void* adlGetContext();                 // ADL_CONTEXT_HANDLE as void*, or nullptr
int   adlGetAdapterIndex(int hipDev);  // -1 if unmapped
#endif

int getGPUCount();
std::string getDeviceName(int device);
std::string getPCIBusId(int device);

// Power monitoring (runtime-loaded, returns 0 if unavailable)
bool initPowerMonitoring();
void shutdownPowerMonitoring();
double getDevicePowerWatts(int device);  // Returns 0.0 if unavailable