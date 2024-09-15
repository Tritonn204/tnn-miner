#include <tnn_hip/arch_def.h>

// Architecture-dependent grid dimensions and shared memory size
#if defined(__HIP_PLATFORM_AMD__)
// AMD platform: Define grid dimensions and shared memory size based on architecture
  #if defined(__gfx9__)  // Vega
    #if defined(__gfx906__)  // Radeon VII (Vega 20)
      archDim(CSHAKE256, 4096, 512);
    #elif defined(__gfx900__)  // Vega 56/64 (Vega 10)
      archDim(CSHAKE256, 2048, 256);
    #else
      archDim(CSHAKE256, 2048, 256);
    #endif

  #elif defined(__gfx10__)  // RDNA 1/2
    archDim(CSHAKE256, 2048, 256);

  #elif defined(__gfx11__)  // RDNA3
    archDim(CSHAKE256, 2048, 256);

  #else  // Default AMD architecture
    archDim(CSHAKE256, 1024, 256);
  #endif

#elif defined(__HIP_PLATFORM_NVIDIA__)
// NVIDIA platform: Define grid dimensions and shared memory size based on architecture
  #if defined(__CUDA_ARCH__)
    #if __CUDA_ARCH__ >= 900  // Hopper
      archDim(CSHAKE256, 2048, 256);

    #elif __CUDA_ARCH__ >= 800  // Ampere
      archDim(CSHAKE256, 2048, 256);

    #elif __CUDA_ARCH__ >= 700  // Volta or newer
      archDim(CSHAKE256, 2048, 256);

    #elif __CUDA_ARCH__ >= 600  // Pascal or newer
      archDim(CSHAKE256, 2048, 256);

    #else  // Older NVIDIA architectures
      archDim(CSHAKE256, 2048, 256);
    #endif
  #endif
#else
    #error "Unsupported platform"
#endif