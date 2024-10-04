#include <hip/hip_runtime.h>
#include <tnn_hip/crypto/astrix-hash/astrix_archdef.h>

#include <iostream>

#include "mine_astrix.hip.h"

#define GPU_FOR(d, count) for (d = 0; d < count; d++)

namespace Astrix_HIP_Worker {
  __host__ void newCtx(workerCtx &ctx) {
    hipError_t err = hipGetDeviceCount(&ctx.GPUCount);

    if (err != hipSuccess) {
      std::cerr << "Failed to get device count: " << hipGetErrorString(err) << std::endl;
      return;
    }

    int d = 0;

    GPU_FOR(d, ctx.GPUCount) {
      hipDeviceProp_t deviceProps;
      hipGetDeviceProperties(&deviceProps, d); // Query device properties
      int smCount = deviceProps.multiProcessorCount;

      int numBlocksPerSm;
      Astrix_HIP::getHashBlocksPerSM(&numBlocksPerSm);
      ctx.blocks[d] = numBlocksPerSm*smCount*128;
      ctx.batchSizes[d] = ctx.blocks[d]*Astrix_HIP::THREAD_DIM;

      if (ctx.batchSizes[d] == 0) {
        getArchDims(ctx.blocks[d], ctx.threads[d], ctx.batchSizes[d]);
        printf("Using default GPU config instead\nblocks: %d, batchSize: %d\n", ctx.blocks[d],ctx.batchSizes[d]);
      } else {
        printf("Using GPU config:\nblocks: %d, batchSize: %d\n", ctx.blocks[d],ctx.batchSizes[d]);
      }
    }
  }

  __host__ void ctxMalloc(workerCtx &ctx) {
    int d = 0;
    GPU_FOR(d, ctx.GPUCount) {
      hipSetDevice(d);

      hipMalloc(&ctx.d_nonceCount, sizeof(int)); 
      hipMalloc(&ctx.d_hashBuffer, 32*ctx.batchSizes[d]);
      hipMalloc(&ctx.d_nonceBuffer, sizeof(uint64_t)*Astrix_HIP::MAX_NONCES);

      hipError_t err = hipGetLastError();
      if (err != hipSuccess) {
        std::cerr << "hipMalloc failed on GPU " << d << ": " << hipGetErrorString(err) << std::endl;
        return;
      }
    }
  }

  __host__ void setDevice(int d) {
    hipSetDevice(d);
  }

  __host__ void ctxMatrix(workerCtx &ctx, uint8_t *work, bool devMine) {
    int d = 0;
    GPU_FOR(d, ctx.GPUCount) {
      hipSetDevice(d);
      Astrix_HIP::newMatrix(work, devMine);
    }
  }
}

#undef GPU_FOR