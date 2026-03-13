#pragma once

#include <string>

int getGPUCount();
std::string getDeviceName(int device);
std::string getPCIBusId(int device);

// Power monitoring (runtime-loaded, returns 0 if unavailable)
bool initPowerMonitoring();
void shutdownPowerMonitoring();
double getDevicePowerWatts(int device);  // Returns 0.0 if unavailable