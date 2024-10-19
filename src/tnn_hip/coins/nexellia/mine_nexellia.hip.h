#pragma once

#include <stddef.h>
#include <inttypes.h>

#include <tnn_hip/crypto/nxl-hash/nxl-hash.hip.h>

namespace Nxl_HIP_Worker {
  typedef struct workerCtx {
    int GPUCount;
    size_t blocks[32];
    size_t threads[32];
    size_t batchSizes[32];
    int *d_nonceCount;
    uint8_t *d_hashBuffer;
    uint64_t *d_nonceBuffer;
    uint8_t* cmpDiff;
  } workerCtx;

  void newCtx(workerCtx &ctx);
  void ctxMalloc(workerCtx &ctx);
  void setDevice(int d);
  void ctxMatrix(workerCtx &ctx, uint8_t *work, bool devMine);
}