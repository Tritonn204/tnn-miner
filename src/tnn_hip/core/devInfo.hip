#include "devInfo.hip.h"

#include <hip/hip_runtime.h>
#include <iostream>

int getGPUCount() {
  int result;
  hipError_t err = hipGetDeviceCount(&result);

  if (err != hipSuccess) {
    std::cerr << "Failed to get device count: " << hipGetErrorString(err) << std::endl;
    return 0;
  }

  return result;
}

std::string getDeviceName(int device)
{
  hipSetDevice(device);

  hipDeviceProp_t deviceProp;
  hipError_t err = hipGetDeviceProperties(&deviceProp, device);

  if (err != hipSuccess)
  {
    std::cerr << "Error getting device properties: " << hipGetErrorString(err) << std::endl;
    return "";
  }

  return std::string(deviceProp.name);
}

std::string getPCIBusId(int device)
{
  char pciBusId[256];

  hipError_t err = hipDeviceGetPCIBusId(pciBusId, 256, device);

  if (err != hipSuccess)
  {
    std::cerr << "Error getting PCI Bus ID: " << hipGetErrorString(err) << std::endl;
    return "";
  }

  return std::string(pciBusId);
}