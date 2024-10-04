#include <tnn_hip/arch_def.h>
#include <unordered_map>
#include <tuple>  // For using std::tuple
#include <string>
#include <cstring>  // For strcmp

// Structure to hold blocks, threads, and batchSize
using ArchDims = std::tuple<size_t, size_t, size_t>;

// Architecture-dependent grid dimensions and shared memory size
#if defined(__HIP_PLATFORM_AMD__)
// AMD platform: Define grid dimensions and shared memory size based on architecture

// VEGA
  // Radeon VII
    archDim(HIP_ASTRIX_gfx906, 3840*1024, 1024);
  // Vega 56/64
    archDim(HIP_ASTRIX_gfx900, 4090*1024, 1024);

// RDNA1
    archDim(HIP_ASTRIX_gfx1010, 2560*1024, 1024);
// RDNA2
  // RX 6800+
    archDim(HIP_ASTRIX_gfx1030, 5120*1024, 1024);
  // RX 6700+
    archDim(HIP_ASTRIX_gfx1031, 2560*1024, 1024);
  // RX 6600+
    archDim(HIP_ASTRIX_gfx1032, 2048*1024, 1024);
  // RX 6300+
    archDim(HIP_ASTRIX_gfx1034, 1024*1024, 1024);

// RDNA 3
  // RX 7900+
    archDim(HIP_ASTRIX_gfx1100, 6144*1024, 1024);
  // RX 7800+
    archDim(HIP_ASTRIX_gfx1101, 3840*1024, 1024);
  // RX 7700+
    archDim(HIP_ASTRIX_gfx1102, 3456*1024, 1024);
  // Default AMD architecture
    archDim(HIP_ASTRIX, 2048*1024, 1024);

#elif defined(__HIP_PLATFORM_NVIDIA__)
// NVIDIA platform: Define grid dimensions and shared memory size based on architecture
    #if __CUDA_ARCH__ >= 900  // Hopper
      archDim(HIP_ASTRIX, 2048, 256);

    #elif __CUDA_ARCH__ >= 800  // Ampere
      archDim(HIP_ASTRIX, 2048, 256);

    #elif __CUDA_ARCH__ >= 700  // Volta
      archDim(HIP_ASTRIX, 2048, 128);

    #elif __CUDA_ARCH__ >= 600  // Pascal
      archDim(HIP_ASTRIX, 2048, 128);

    #else  // Older NVIDIA architectures
      archDim(HIP_ASTRIX, 2048, 128);
    #endif
#else
    #error "Unsupported platform"
#endif

static const ArchDims defaultDims = {HIP_ASTRIX_BLOCKS, HIP_ASTRIX_THREADS, HIP_ASTRIX_BATCH_SIZE};

// Define a lookup table (map) that holds the block/thread/batch sizes for each architecture
#if defined(__HIP_PLATFORM_AMD__)
static inline const std::unordered_map<std::string, ArchDims> archDimsMap = {
  ARCH_DIM_ENTRY(HIP_ASTRIX, gfx900),
  ARCH_DIM_ENTRY(HIP_ASTRIX, gfx906),
  ARCH_DIM_ENTRY(HIP_ASTRIX, gfx1010),
  ARCH_DIM_ENTRY(HIP_ASTRIX, gfx1030),
  // ARCH_DIM_ENTRY(HIP_ASTRIX, gfx1031),
  // ARCH_DIM_ENTRY(HIP_ASTRIX, gfx1032),
  // ARCH_DIM_ENTRY(HIP_ASTRIX, gfx1034),
  ARCH_DIM_ENTRY(HIP_ASTRIX, gfx1100)
};
#elif defined(__HIP_PLATFORM_NVIDIA__)
static inline const std::unordered_map<std::string, ArchDims> archDimsMap = {
}
#endif

// Function to retrieve architecture dimensions
static inline void getArchDims(size_t &blocks, size_t &threads, size_t &batchSize) {
  int device;
  hipGetDevice(&device);

  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);

  const char* archName = props.gcnArchName;

  // Find the architecture in the lookup table
  auto it = archDimsMap.find(std::string(archName));
  const ArchDims& dims = (it != archDimsMap.end()) ? it->second : defaultDims;

  // Unpack the tuple into blocks, threads, and batchSize
  std::tie(blocks, threads, batchSize) = dims;
}