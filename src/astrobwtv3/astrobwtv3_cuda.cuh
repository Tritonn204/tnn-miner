#ifndef astrobwtv3_cuda
#define astrobwtv3_cuda

#include <cuda.h>
#include <cuda_runtime.h>

#include <inttypes.h>

#include "sha256_cuda.cuh"
#include "siphash_cuda.cuh"
#include "salsa20_cuda.cuh"

#include <openssl/rc4.h>

extern "C"
{
#include "rc4_cuda.cuh"
}

#ifdef _WIN32
#include <winsock2.h>
#include <intrin.h>
#else
#include <arpa/inet.h>
#endif

#define D_HTONL(n) (((((unsigned long)(n) & 0xFF)) << 24) |    \
                    ((((unsigned long)(n) & 0xFF00)) << 8) |   \
                    ((((unsigned long)(n) & 0xFF0000)) >> 8) | \
                    ((((unsigned long)(n) & 0xFF000000)) >> 24))

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

const uint32_t MAXX = (256 * 384) - 1; // this is the maximum

class workerData_cuda
{
public:
  // byte step_3[256];
  // unsigned char sData[MAXX + 64];

  // SHA256_CTX_CUDA sha256;
  // byte sha_key[32];

  int32_t *sa;
  byte *sData;
  int *BA;
  int *BB;
  byte *sha_key;
  uint8_t *keystream;
  uint32_t *data_len;

  // rc4_state key;
  // uint64_t tries;
  // uint64_t lhash;
  // uint64_t prev_lhash;
  // uint64_t random_switcher;
  // byte pos1;
  // byte pos2;
  // byte op;
  // byte A;
  // byte t1;
  // byte t2;

  // int32_t sa[MAXX];
  // uint8_t keystream[64];

  // int BA[256];
  // int BB[256 * 256];
};

extern workerData_cuda worker;
extern cudaStream_t *stream;

__global__ void branchedSHATest_kernel(workerData_cuda *w, byte *input, byte *output, int len, int count);
__global__ void branchedSHATest(workerData_cuda *w, byte *input, byte *output, int len, int count);

__global__ void ASTRO_1_kernel(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int device, int offset, int nonce);
__global__ void ASTRO_BRANCH_kernel(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int d, int offset);
__global__ void ASTRO_hybrid_kernel(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int device, int offset);
__global__ void ASTRO_3_kernel(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int device, int offset);

void ASTRO_INIT(int device, byte *work, int batchSize, int offset, int nonce);
void ASTRO_CUDA(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int device, int offset, int nonce);
void ASTRO_1(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int device, int offset);
void ASTRO_2(void **cudaStore, workerData_cuda worker, int batchSize);
void ASTRO_3(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int device, int offset);

void workerMalloc(workerData_cuda &worker, int batchSize);

__device__ void AstroBWTv3_cuda_p1(
    unsigned char *input,
    int inputLen, workerData_cuda worker,
    byte *step3_shared, SHA256_CTX_CUDA &sha256,
    byte *sha_key_shared, rc4_state &key_shared,
    uint64_t &lhash_shared, uint64_t &prev_lhash_shared);

__device__ void AstroBWTv3_cuda_p2(
    unsigned char *inputs, int inputLen,
    unsigned char *outputHashes, workerData_cuda worker,
    byte *step3, uint64_t *tries_arr,
    uint64_t *lhash_arr, uint64_t *prev_lhash_arr,
    uint64_t *random_switcher_arr,
    rc4_state *key_arr, byte *pos1_arr,
    byte *pos2_arr, byte *op_arr,
    byte *A_arr, byte *temp,
    int batchSize, int offset);

__device__ void AstroBWTv3_cuda_p3(unsigned char *outputhash, workerData_cuda worker, SHA256_CTX_CUDA &sha56_shared);

__global__ void AstroBWTv3_cuda(unsigned char *input, int inputLen, unsigned char *outputhash, workerData_cuda worker);
__device__ void branchCompute(
    workerData_cuda worker, byte *step3_shared,
    uint64_t &lhash_shared, uint64_t &prev_lhash_shared,
    rc4_state &key_shared, byte &pos1_shared,
    byte &pos2_shared, byte &op_shared,
    byte *temp,
    int batchSize);

__host__ void branchComputeCPU_cuda(workerData_cuda &worker);

__host__ __device__ __forceinline__ unsigned char
leftRotate8_cuda(unsigned char n, unsigned d)
{ // rotate n by d bits
#if defined(_WIN32)
  return _rotl8(n, d);
#else
  d = d % 8;
  return (n << d) | (n >> (8 - d));
#endif
}

__device__ __forceinline__ void SHA256_cuda(SHA256_CTX_CUDA &ctx, byte *in, byte *out, int size)
{
  sha256_init_cuda(&ctx);
  sha256_update_cuda(&ctx, in, size);
  sha256_final_cuda(&ctx, out);
}

__host__ __device__ __forceinline__ unsigned char reverse8_cuda(unsigned char b)
{
  return (b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
}

#endif