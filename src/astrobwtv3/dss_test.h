#pragma once

#include <hip/hip_runtime.h>
#include <sha256_hip.h>

using byte = unsigned char;

void DSSTEST(int runs);

__device__ __forceinline__ void SHA256_hip2(SHA256_CTX_HIP *ctx, byte* in, byte* out, int size)
{
  sha256_init_hip(ctx);
  sha256_update_hip(ctx, in, size);
  sha256_final_hip(ctx, out);
}