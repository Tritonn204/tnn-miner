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
    archDim(CSHAKE256_KAS_gfx906, 38400, 128);
    archDim(CSHAKE256_HEAVY_gfx906, 38400, 128);
  // Vega 56/64
    archDim(CSHAKE256_KAS_gfx900, 40960, 128);
    archDim(CSHAKE256_HEAVY_gfx900, 40960, 128);

// RDNA2
  // RX 6800+
    archDim(CSHAKE256_KAS_gfx1030, 51200, 128);
    archDim(CSHAKE256_HEAVY_gfx1030, 51200, 128);
  // RX 6700+
    archDim(CSHAKE256_KAS_gfx1031, 25600, 128);
    archDim(CSHAKE256_HEAVY_gfx1031, 25600, 128);
  // RX 6600+
    archDim(CSHAKE256_KAS_gfx1032, 20480, 128);
    archDim(CSHAKE256_HEAVY_gfx1032, 20480, 128);
  // RX 6300+
    archDim(CSHAKE256_KAS_gfx1034, 10240, 128);
    archDim(CSHAKE256_HEAVY_gfx1034, 10240, 128);

// RDNA 3
  // RX 7900+
    archDim(CSHAKE256_KAS_gfx1100, 61440*32, 128);
    archDim(CSHAKE256_HEAVY_gfx1100, 61440*32, 128);
  // RX 7800+
    archDim(CSHAKE256_KAS_gfx1101, 38400, 128);
    archDim(CSHAKE256_HEAVY_gfx1101, 38400, 128);
  // RX 7700+
    archDim(CSHAKE256_KAS_gfx1102, 34560, 128);
    archDim(CSHAKE256_HEAVY_gfx1102, 34560, 128);
  // Default AMD architecture
    archDim(CSHAKE256_KAS, 20480, 128);

#elif defined(__HIP_PLATFORM_NVIDIA__)
// NVIDIA platform: Define grid dimensions and shared memory size based on architecture
  #if defined(__CUDA_ARCH__)
    #if __CUDA_ARCH__ >= 900  // Hopper
      archDim(CSHAKE256_KAS, 2048, 256);

    #elif __CUDA_ARCH__ >= 800  // Ampere
      archDim(CSHAKE256_KAS, 2048, 256);

    #elif __CUDA_ARCH__ >= 700  // Volta
      archDim(CSHAKE256_KAS, 2048, 128);

    #elif __CUDA_ARCH__ >= 600  // Pascal
      archDim(CSHAKE256_KAS, 2048, 128);

    #else  // Older NVIDIA architectures
      archDim(CSHAKE256_KAS, 2048, 128);
    #endif
  #endif
#else
    #error "Unsupported platform"
#endif

ArchDims defaultDims = {CSHAKE256_KAS_BLOCKS, CSHAKE256_KAS_THREADS, CSHAKE256_KAS_BATCH_SIZE};

// Define a lookup table (map) that holds the block/thread/batch sizes for each architecture
static inline const std::unordered_map<std::string, ArchDims> archDimsMap = {
  ARCH_DIM_ENTRY(CSHAKE256_KAS, gfx900),
  ARCH_DIM_ENTRY(CSHAKE256_KAS, gfx906),
  ARCH_DIM_ENTRY(CSHAKE256_KAS, gfx1030),
  ARCH_DIM_ENTRY(CSHAKE256_KAS, gfx1031),
  ARCH_DIM_ENTRY(CSHAKE256_KAS, gfx1032),
  ARCH_DIM_ENTRY(CSHAKE256_KAS, gfx1034),
  ARCH_DIM_ENTRY(CSHAKE256_KAS, gfx1100)
};

static inline const std::unordered_map<std::string, ArchDims> archDimsMap_heavy = {
  ARCH_DIM_ENTRY(CSHAKE256_HEAVY, gfx900),
  ARCH_DIM_ENTRY(CSHAKE256_HEAVY, gfx906),
  ARCH_DIM_ENTRY(CSHAKE256_HEAVY, gfx1030),
  ARCH_DIM_ENTRY(CSHAKE256_HEAVY, gfx1031),
  ARCH_DIM_ENTRY(CSHAKE256_HEAVY, gfx1032),
  ARCH_DIM_ENTRY(CSHAKE256_HEAVY, gfx1034),
  ARCH_DIM_ENTRY(CSHAKE256_HEAVY, gfx1100)
};

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

// Function to retrieve architecture dimensions
static inline void getArchDims_heavy(size_t &blocks, size_t &threads, size_t &batchSize) {
  int device;
  hipGetDevice(&device);

  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);

  const char* archName = props.gcnArchName;

  // Find the architecture in the lookup table
  auto it = archDimsMap_heavy.find(std::string(archName));
  const ArchDims& dims = (it != archDimsMap.end()) ? it->second : defaultDims;

  // Unpack the tuple into blocks, threads, and batchSize
  std::tie(blocks, threads, batchSize) = dims;
}
