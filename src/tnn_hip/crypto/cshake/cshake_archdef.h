#include <tnn_hip/arch_def.h>

// Architecture-dependent grid dimensions and shared memory size
#if defined(__HIP_PLATFORM_AMD__)
// AMD platform: Define grid dimensions and shared memory size based on architecture
  #if defined(__gfx9__)  // Vega
    #if defined(__gfx906__)  // Radeon VII (Vega 20)
      archDim(CSHAKE256, 38400, 192);
    #elif defined(__gfx900__)  // Vega 56/64 (Vega 10)
      archDim(CSHAKE256, 40960, 192);
    #else
      archDim(CSHAKE256, 2048, 192);
    #endif

  #elif defined(__gfx10__)  // RDNA 1/2
    archDim(CSHAKE256, 2048, 192);


  // RDNA 3
  #elif defined(__gfx1100__)  // 7900 XTX/XT
    archDim(CSHAKE256, 61440, 192);

  #elif defined(__gfx1101__)  // 7800XT
    archDim(CSHAKE256, 38400, 192);

  #elif defined(__gfx1102__)  // 7700XT/7600XT
    archDim(CSHAKE256, 34560, 192);

  #else  // Default AMD architecture
    archDim(CSHAKE256, 10240, 192);
  #endif

#elif defined(__HIP_PLATFORM_NVIDIA__)
// NVIDIA platform: Define grid dimensions and shared memory size based on architecture
  #if defined(__CUDA_ARCH__)
    #if __CUDA_ARCH__ >= 900  // Hopper
      archDim(CSHAKE256, 2048, 256);

    #elif __CUDA_ARCH__ >= 800  // Ampere
      archDim(CSHAKE256, 2048, 256);

    #elif __CUDA_ARCH__ >= 700  // Volta or newer
      archDim(CSHAKE256, 2048, 192);

    #elif __CUDA_ARCH__ >= 600  // Pascal or newer
      archDim(CSHAKE256, 2048, 192);

    #else  // Older NVIDIA architectures
      archDim(CSHAKE256, 2048, 192);
    #endif
  #endif
#else
    #error "Unsupported platform"
#endif