#include <dss_test.h>
#include <hex.h>
#include <sha256_hip.h>

__global__ void DSSKernel(byte *output, int runs, int offset)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < runs; i += stride)
  {
    byte *input = new byte[sizeof(int)];
    memcpy(input, &i, sizeof(int));
    SHA256_CTX_HIP sha256;
    SHA256_hip2(&sha256, input, &output[(i+offset) * 32], sizeof(int));
    delete[] input;
  }
}

void DSSTEST(int runs)
{
  // int *sArrays = new int[runs * 32];
  // int *sArrays_hip;
  // hipMalloc(&sArrays_hip, runs * 32 * sizeof(int));

  // byte *hashes = new byte[runs * 32];
  // byte *hashes_hip;
  // hipMalloc(&hashes_hip, runs * 32);

  // int kernelBatchSize = 4096;

  int a = 0;

  // for (int i = 0; i < runs; i += kernelBatchSize)
  // {
  //   int runBufferSize = (runs - i > kernelBatchSize) ? kernelBatchSize : runs - i;
  //   int B = (runBufferSize + 256 - 1) / 256;
  //   int T = 256;
  //   DSSKernel<<<B, T>>>(&hashes_hip[i], runBufferSize, i);
  //   hipDeviceSynchronize();
  // }

  // hipMemcpy(hashes, hashes_hip, runs * 32, hipMemcpyDeviceToHost);

  // for (int i = 0; i < runs; i++)
  // {
  //   byte *input = new byte[sizeof(int)];
  //   memcpy(input, &i, sizeof(int));
  //   printf("Input of DSSTEST hash #%d : %s\n", i, hexStr(input, sizeof(int)).c_str());
  //   printf("Result of DSSTEST hash #%d: %s\n", i, hexStr(&hashes[i * 32], 32).c_str());
  //   delete[] input;

  //   // printf("Result of DSSTEST %d: [");
  //   // for (int j = 0; j < 32; j++) {
  //   //   if (j > 0) printf(", ");
  //   //   printf("%d", sArrays[i*32 + j]);
  //   // }
  //   // printf("]\n");
  // }
}