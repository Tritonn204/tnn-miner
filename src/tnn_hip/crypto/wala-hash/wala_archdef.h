#include <tnn_hip/arch_def.h>
#include <unordered_map>
#include <tuple>  // For using std::tuple
#include <string>
#include <cstring>  // For strcmp

// Structure to hold blocks, threads, and batchSize
using ArchDims = std::tuple<size_t, size_t, size_t>;

static constexpr double factor = 6;

// Architecture-dependent grid dimensions and shared memory size
#if defined(__HIP_PLATFORM_AMD__)
// AMD platform: Define grid dimensions and shared memory size based on architecture

// VEGA
  // Radeon VII
    archDim(HIP_WALA_gfx906, 3840*4096*factor, 1024);
  // Vega 56/64
    archDim(HIP_WALA_gfx900, 4090*4096*factor, 1024);

// RDNA1
  // RX 5600+
    archDim(HIP_WALA_gfx1010, 2560*4096*factor, 1024);
  // Radeon Pro V520
    archDim(HIP_WALA_gfx1011, 2560*4096*factor, 1024);
  // RX 5500+
    archDim(HIP_WALA_gfx1012, 2560*4096*factor, 1024);
    archDim(HIP_WALA_gfx1013, 1408*4096*factor, 1024);
// RDNA2
  // RX 6800+
    archDim(HIP_WALA_gfx1030, 5120*4096*factor, 1024);
  // RX 6700+
    archDim(HIP_WALA_gfx1031, 2560*4096*factor, 1024);
  // RX 6600+
    archDim(HIP_WALA_gfx1032, 2048*4096*factor, 1024);
  // RX 6300+
    archDim(HIP_WALA_gfx1034, 1024*4096*factor, 1024);

// RDNA 3
  // RX 7900+
    archDim(HIP_WALA_gfx1100, 6114*4096*factor, 1024);
  // RX 7800+
    archDim(HIP_WALA_gfx1101, 3840*4096*factor, 1024);
  // RX 7700+
    archDim(HIP_WALA_gfx1102, 3456*4096*factor, 1024);
  // Default AMD architecture
    archDim(HIP_WALA, 2048*1024, 1024);

#elif defined(__HIP_PLATFORM_NVIDIA__)
  // NVIDIA platform: Define grid dimensions and shared memory size based on architecture

// Pascal
    archDim(HIP_WALA_61, 3584*4096*factor, 1024);

// Volta
    archDim(HIP_WALA_70, 5120*4096*factor, 1024);

// Turing
    archDim(HIP_WALA_75, 4352*4096*factor, 1024);

// Ampere
    archDim(HIP_WALA_86, 10496*4096*factor, 1024);

// Ada Lovelace (RTX 40 series)
    archDim(HIP_WALA_89, 16384*4096*factor, 1024);

// Default nVidia architecture
    archDim(HIP_WALA, 2048*4096*factor, 1024);

#else
    #error "Unsupported platform"
#endif

static const ArchDims defaultDims = {HIP_WALA_BLOCKS, HIP_WALA_THREADS, HIP_WALA_BATCH_SIZE};

// Define a lookup table (map) that holds the block/thread/batch sizes for each architecture
#if defined(__HIP_PLATFORM_AMD__)
static inline const std::unordered_map<std::string, ArchDims> archDimsMap = {
  ARCH_DIM_ENTRY(HIP_WALA, gfx900),   // Vega 56/64
  ARCH_DIM_ENTRY(HIP_WALA, gfx906),   // Radeon VII

  ARCH_DIM_ENTRY(HIP_WALA, gfx1010),  // RX 5600+
  ARCH_DIM_ENTRY(HIP_WALA, gfx1011),  // Radeon Pro V520
  ARCH_DIM_ENTRY(HIP_WALA, gfx1012),  // RX 5500+
  ARCH_DIM_ENTRY(HIP_WALA, gfx1013),  // RX 5300

  ARCH_DIM_ENTRY(HIP_WALA, gfx1030),  // RX 6800+
  ARCH_DIM_ENTRY(HIP_WALA, gfx1031),  // RX 6700+
  ARCH_DIM_ENTRY(HIP_WALA, gfx1032),  // RX 6600+
  ARCH_DIM_ENTRY(HIP_WALA, gfx1034),  // RX 6300+

  ARCH_DIM_ENTRY(HIP_WALA, gfx1100),  // RX 7900+
  ARCH_DIM_ENTRY(HIP_WALA, gfx1101),  // RX 7800+
  ARCH_DIM_ENTRY(HIP_WALA, gfx1102),  // RX 7700+
};
#elif defined(__HIP_PLATFORM_NVIDIA__)
static inline const std::unordered_map<std::string, ArchDims> archDimsMap = {
  ARCH_DIM_ENTRY(HIP_WALA, 61),
  ARCH_DIM_ENTRY(HIP_WALA, 70),
  ARCH_DIM_ENTRY(HIP_WALA, 75),
  ARCH_DIM_ENTRY(HIP_WALA, 86),
  ARCH_DIM_ENTRY(HIP_WALA, 89),
};
#endif

static inline void trimArchName(const char *archName, char *trimmedArchName)
{
  // Find the first occurrence of ':'
  const char *colonPos = strchr(archName, ':');

  // If ':' is found, copy only up to that position
  if (colonPos != nullptr)
  {
    size_t lengthToCopy = colonPos - archName;
    strncpy(trimmedArchName, archName, lengthToCopy);
    trimmedArchName[lengthToCopy] = '\0'; // Null-terminate the string
  }
  else
  {
    // If ':' is not found, copy the entire string
    strcpy(trimmedArchName, archName);
  }
}

// Function to retrieve architecture dimensions
static inline void getArchDims(size_t &blocks, size_t &threads, size_t &batchSize) {
  int device;
  hipGetDevice(&device);

  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);

  #if defined(__HIP_PLATFORM_AMD__)
  const char* archName = props.gcnArchName;
  char trimmedArchName[256];
  trimArchName(archName, trimmedArchName);

  // Find the architecture in the lookup table
  auto it = archDimsMap.find(std::string(trimmedArchName));
  const ArchDims& dims = (it != archDimsMap.end()) ? it->second : defaultDims;

  // int smCount = props.multiProcessorCount * (2*(HIP_ARCH >= 1010));

  // size_t BS = smCount*11*1024;
  // const ArchDims& dims = {smCount*11, 1024, BS};
  #else
  int compute = props.major*10 + props.minor;
  auto it = archDimsMap.find(std::to_string(compute));
  // int smCount = props.multiProcessorCount;

  // size_t BS = smCount*1024*32*11;
  // const ArchDims& dims = {smCount*32*11, 1024, BS};
  const ArchDims& dims = (it != archDimsMap.end()) ? it->second : defaultDims;
  
  #endif

  // Unpack the tuple into blocks, threads, and batchSize
  std::tie(blocks, threads, batchSize) = dims;
}