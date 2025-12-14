#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#define CHECK_HIP(call)                                                                         \
  do                                                                                            \
  {                                                                                             \
    hipError_t err = call;                                                                      \
    if (err != hipSuccess)                                                                      \
    {                                                                                           \
      fprintf(stderr, "HIP Error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

void print_hex(const char *label, const uint8_t *data, size_t len)
{
  printf("%s: ", label);
  for (size_t i = 0; i < len; i++)
    printf("%02x", data[i]);
  printf("\n");
}

struct DeviceInfo
{
  char name[256];
  size_t total_memory;
  size_t free_memory;
  int compute_units;
  int warp_size;
  size_t shared_mem_per_block;
};

DeviceInfo get_device_info(int device = 0)
{
  DeviceInfo info;
  hipDeviceProp_t props;
  CHECK_HIP(hipGetDeviceProperties(&props, device));
  strncpy(info.name, props.name, 255);
  info.total_memory = props.totalGlobalMem;
  info.compute_units = props.multiProcessorCount;
  info.warp_size = props.warpSize;
  info.shared_mem_per_block = props.sharedMemPerBlock;
  CHECK_HIP(hipMemGetInfo(&info.free_memory, &info.total_memory));
  return info;
}

struct BatchConfig
{
  int block_size;
  int num_blocks;
  uint32_t batch_size;
  size_t scratch_memory_needed;
  size_t shared_mem_per_block;
};
