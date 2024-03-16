#pragma once

#include <hip/hip_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>
#include <thrust/copy.h>

#include <inttypes.h>

#include "sha256_hip.h"
#include "siphash_hip.h"
#include "salsa20_hip.h"

#include <2dlookup.h>
#include <3dlookup.h>
#include <iostream>

#ifdef _WIN32
#include <winsock2.h>
#include <intrin.h>
#else
#include <arpa/inet.h>
#endif

__device__ const unsigned char branchedOps_global_hip[] = {
1,3,5,9,11,13,15,17,20,21,23,27,29,30,35,39,40,43,45,47,51,54,58,60,62,64,68,70,72,74,75,80,82,85,91,92,93,94,103,108,109,115,116,117,119,120,123,124,127,132,133,134,136,138,140,142,143,146,148,149,150,154,155,159,161,165,168,169,176,177,178,180,182,184,187,189,190,193,194,195,199,202,203,204,212,214,215,216,219,221,222,223,226,227,230,231,234,236,239,240,241,242,250,253
};

__device__ const int branchedOps_size_hip = 104; // Manually counted size of branchedOps_global
__device__ const int regOps_size_hip = 256-branchedOps_size_hip; // 256 - branchedOps_global.size()

struct rc4_state
{
  int x, y, m[256];
};

#define D_HTONL(n) (((((unsigned long)(n)&0xFF)) << 24) |    \
                    ((((unsigned long)(n)&0xFF00)) << 8) |   \
                    ((((unsigned long)(n)&0xFF0000)) >> 8) | \
                    ((((unsigned long)(n)&0xFF000000)) >> 24))

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort = true)
{
  if (code != hipSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

const uint32_t MAXX = (256 * 384) - 1; // this is the maximum

class workerData_hip
{
public:
  unsigned char sHash[32];
  unsigned char sha_key[32];
  unsigned char sData[MAXX + 64];

  unsigned char counter[64];

  SHA256_CTX_HIP sha256;
  rc4_state key;

  int32_t sa[MAXX];
  
  alignas(32) byte branchedOps[branchedOps_size_hip];
  alignas(32) byte regularOps[regOps_size_hip];

  alignas(32) byte branched_idx[256];
  alignas(32) byte reg_idx[256];

  // Salsa20_cuda salsa20;

  int bucket_A[256];
  int bucket_B[256*256];
  int M;

  unsigned char step_3[256];

  unsigned char *lookup3D;
  uint16_t *lookup2D;

  uint64_t random_switcher;

  uint64_t lhash;
  uint64_t prev_lhash;
  uint64_t tries;

  unsigned char op;
  unsigned char pos1;
  unsigned char pos2;
  unsigned char t1;
  unsigned char t2;

  // Vars for split sais kernels
  int *sais_C, *sais_B, *sais_D, *sais_RA, *sais_b;
  int sais_i, sais_j, sais_m, sais_p, sais_q, sais_t_var, sais_name, sais_pidx = 0, sais_newfs;
  int sais_c0, sais_c1;
  unsigned int sais_flags;

  unsigned char A;
  uint32_t data_len;
};

// Tentatively planning to init these arrays on host, then copy them over to GPU after

__device__ __forceinline__ void initWorker_hip(workerData_hip &worker) {
  // printf("branchedOps size:cl %d", worker.branchedOps.size());
  thrust::copy(thrust::device, branchedOps_global_hip, branchedOps_global_hip + branchedOps_size_hip, worker.branchedOps);

  unsigned char full[256];
  unsigned char diff[256];
  thrust::sequence(thrust::device, full, full + 256, 0);

  // Execution policy is reversed for set_difference atm, strange.
  thrust::set_difference(thrust::host, full, full + 256, branchedOps_global_hip, branchedOps_global_hip + branchedOps_size_hip, diff);
  memcpy(worker.regularOps, diff, 256 - branchedOps_size_hip);
  
  // printf("Branched Ops:\n");
  // for (int i = 0; i < branchedOps_size_hip; i++) {
  //   printf("%02X, ", worker.branchedOps[i]);
  // }
  // printf("\n");
  // printf("Regular Ops:\n");
  // for (int i = 0; i < regOps_size_hip; i++) {
  //   printf("%02X, ", worker.regularOps[i]);
  // }
  // printf("\n");
}

extern workerData_hip *workers;
extern hipStream_t *stream;

__global__ void branchedSHATest_kernel(workerData_hip *w, byte *input, byte *output, int len, int count);
void branchedSHATest(workerData_hip *w, byte *input, byte *output, int len, int count);

__global__ void ASTRO_1_kernel(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int device, int offset);
__global__ void ASTRO_2_kernel(byte *work, byte *output, workerData_hip *workers, int inputLen, int d, int offset);
__global__ void ASTRO_hybrid_kernel(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int device, int offset);
__global__ void ASTRO_3_kernel(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int device, int offset);

__host__ void ASTRO_LOOKUPGEN_HIP(int device, int batchSize, workerData_hip *workers, uint16_t *l2D, byte *l3D);
void ASTRO_INIT_HIP(int device, byte *work, int batchSize, int offset, int nonce);
void ASTRO_HIP(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int device, int offset);
void ASTRO_1(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int device, int offset);
void ASTRO_2(void **cudaStore, workerData_hip *workers, int batchSize);
void ASTRO_3(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int device, int offset);

__device__ void AstroBWTv3_hip_p1(unsigned char *input, int inputLen, workerData_hip *workers, int offset);
__device__ void AstroBWTv3_hip_p2(workerData_hip *workers, byte *s3, int offset);
__device__ void AstroBWTv3_hip_p3(unsigned char *outputHash, workerData_hip *workers, int offset);

__global__ void AstroBWTv3_hip(unsigned char *input, int inputLen, unsigned char *outputhash, workerData_hip &scratch);
__device__ void branchCompute(workerData_hip &worker);
__host__ void branchComputeCPU_hip(workerData_hip &worker);

__device__ void lookupCompute_hip(workerData_hip &worker, byte *s3, int idx);
__device__ void processAfterMarker_hip(workerData_hip& worker, byte *s3);

__host__ __device__ __forceinline__ unsigned char
leftRotate8_hip(unsigned char n, unsigned d)
{ // rotate n by d bits
  d = d % 8;
  return (n << d) | (n >> (8 - d));
}

__device__ __forceinline__ void SHA256_hip(SHA256_CTX_HIP &ctx, byte *in, byte *out, int size)
{
  sha256_init_hip(&ctx);
  sha256_update_hip(&ctx, in, size);
  sha256_final_hip(&ctx, out);
}

__host__ __device__ __forceinline__ unsigned char reverse8_hip(unsigned char b)
{
  return (b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
}
