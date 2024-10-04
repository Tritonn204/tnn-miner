#include <stdint.h>
#include <assert.h>

#include <hip/hip_runtime.h>
#include "astrix_archdef.h"
#include "astrix-hash.hip.h"

#include <tnn_hip/crypto/keccak-tiny.hip.inc>
#include <tnn_hip/crypto/blake3_compact.h>
#include <tnn_hip/crypto/xoshiro256starstar.hip.inc>

#include "test_hip_astrix.h"

typedef union _uint256_t
{
  uint64_t number[4];
  uint8_t hash[32];
} uint256_t;

hipDeviceProp_t deviceProps;

#define MATRIX_SIZE 64
#define HALF_MATRIX_SIZE 32
#define QUARTER_MATRIX_SIZE 16

#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

#define BENCH_RUNS 500

#define LT_U256(X, Y) (X.number[3] != Y.number[3] ? X.number[3] < Y.number[3] : X.number[2] != Y.number[2] ? X.number[2] < Y.number[2] \
                                                                            : X.number[1] != Y.number[1]   ? X.number[1] < Y.number[1] \
                                                                                                           : X.number[0] < Y.number[0])
__align__(32) __constant__ uint8_t matrix[MATRIX_SIZE][MATRIX_SIZE];
__align__(32) __constant__ uint8_t matrix_dev[MATRIX_SIZE][MATRIX_SIZE];
__align__(32) __constant__ uint8_t hash_header[Astrix_HIP::HASH_HEADER_SIZE];
__align__(32) __constant__ uint8_t hash_header_dev[Astrix_HIP::HASH_HEADER_SIZE];
__constant__ uint256_t target;
__align__(32) __constant__ constexpr static uint8_t powP[Plen] = {0x3d, 0xd8, 0xf6, 0xa1, 0x0d, 0xff, 0x3c, 0x11, 0x3c, 0x7e, 0x02, 0xb7, 0x55, 0x88, 0xbf, 0x29, 0xd2, 0x44, 0xfb, 0x0e, 0x72, 0x2e, 0x5f, 0x1e, 0xa0, 0x69, 0x98, 0xf5, 0xa3, 0xa4, 0xa5, 0x1b, 0x65, 0x2d, 0x5e, 0x87, 0xca, 0xaf, 0x2f, 0x7b, 0x46, 0xe2, 0xdc, 0x29, 0xd6, 0x61, 0xef, 0x4a, 0x10, 0x5b, 0x41, 0xad, 0x1e, 0x98, 0x3a, 0x18, 0x9c, 0xc2, 0x9b, 0x78, 0x0c, 0xf6, 0x6b, 0x77, 0x40, 0x31, 0x66, 0x88, 0x33, 0xf1, 0xeb, 0xf8, 0xf0, 0x5f, 0x28, 0x43, 0x3c, 0x1c, 0x65, 0x2e, 0x0a, 0x4a, 0xf1, 0x40, 0x05, 0x07, 0x96, 0x0f, 0x52, 0x91, 0x29, 0x5b, 0x87, 0x67, 0xe3, 0x44, 0x15, 0x37, 0xb1, 0x25, 0xa4, 0xf1, 0x70, 0xec, 0x89, 0xda, 0xe9, 0x82, 0x8f, 0x5d, 0xc8, 0xe6, 0x23, 0xb2, 0xb4, 0x85, 0x1f, 0x60, 0x1a, 0xb2, 0x46, 0x6a, 0xa3, 0x64, 0x90, 0x54, 0x85, 0x34, 0x1a, 0x85, 0x2f, 0x7a, 0x1c, 0xdd, 0x06, 0x0f, 0x42, 0xb1, 0x3b, 0x56, 0x1d, 0x02, 0xa2, 0xc1, 0xe4, 0x68, 0x16, 0x45, 0xe4, 0xe5, 0x1d, 0xba, 0x8d, 0x5f, 0x09, 0x05, 0x41, 0x57, 0x02, 0xd1, 0x4a, 0xcf, 0xce, 0x9b, 0x84, 0x4e, 0xca, 0x89, 0xdb, 0x2e, 0x74, 0xa8, 0x27, 0x94, 0xb0, 0x48, 0x72, 0x52, 0x8b, 0xe7, 0x9c, 0xce, 0xfc, 0xb1, 0xbc, 0xa5, 0xaf, 0x82, 0xcf, 0x29, 0x11, 0x5d, 0x83, 0x43, 0x82, 0x6f, 0x78, 0x7c, 0xb9, 0x02};
__align__(32) __constant__ constexpr static uint8_t heavyP[Plen] = {0x09, 0x85, 0x24, 0xb2, 0x52, 0x4c, 0xd7, 0x3a, 0x16, 0x42, 0x9f, 0x2f, 0x0e, 0x9b, 0x62, 0x79, 0xee, 0xf8, 0xc7, 0x16, 0x48, 0xff, 0x14, 0x7a, 0x98, 0x64, 0x05, 0x80, 0x4c, 0x5f, 0xa7, 0x11, 0xda, 0xce, 0xee, 0x44, 0xdf, 0xe0, 0x20, 0xe7, 0x69, 0x40, 0xf3, 0x14, 0x2e, 0xd8, 0xc7, 0x72, 0xba, 0x35, 0x89, 0x93, 0x2a, 0xff, 0x00, 0xc1, 0x62, 0xc4, 0x0f, 0x25, 0x40, 0x90, 0x21, 0x5e, 0x48, 0x6a, 0xcf, 0x0d, 0xa6, 0xf9, 0x39, 0x80, 0x0c, 0x3d, 0x2a, 0x79, 0x9f, 0xaa, 0xbc, 0xa0, 0x26, 0xa2, 0xa9, 0xd0, 0x5d, 0xc0, 0x31, 0xf4, 0x3f, 0x8c, 0xc1, 0x54, 0xc3, 0x4c, 0x1f, 0xd3, 0x3d, 0xcc, 0x69, 0xa7, 0x01, 0x7d, 0x6b, 0x6c, 0xe4, 0x93, 0x24, 0x56, 0xd3, 0x5b, 0xc6, 0x2e, 0x44, 0xb0, 0xcd, 0x99, 0x3a, 0x4b, 0xf7, 0x4e, 0xb0, 0xf2, 0x34, 0x54, 0x83, 0x86, 0x4c, 0x77, 0x16, 0x94, 0xbc, 0x36, 0xb0, 0x61, 0xe9, 0x07, 0x07, 0xcc, 0x65, 0x77, 0xb1, 0x1d, 0x8f, 0x7e, 0x39, 0x6d, 0xc4, 0xba, 0x80, 0xdb, 0x8f, 0xea, 0x58, 0xca, 0x34, 0x7b, 0xd3, 0xf2, 0x92, 0xb9, 0x57, 0xb9, 0x81, 0x84, 0x04, 0xc5, 0x76, 0xc7, 0x2e, 0xc2, 0x12, 0x51, 0x67, 0x9f, 0xc3, 0x47, 0x0a, 0x0c, 0x29, 0xb5, 0x9d, 0x39, 0xbb, 0x92, 0x15, 0xc6, 0x9f, 0x2f, 0x31, 0xe0, 0x9a, 0x54, 0x35, 0xda, 0xb9, 0x10, 0x7d, 0x32, 0x19, 0x16};
__align__(32) __constant__ constexpr static uint8_t powP_nonce[8] = {powP[72], powP[73], powP[74], powP[75], powP[76], powP[77], powP[78], powP[79]};
__align__(32) __constant__ constexpr static uint8_t powP_tail[8] = {powP[192], powP[193], powP[194], powP[195], powP[196], powP[197], powP[198], powP[199]};

__align__(32) __constant__ static uint8_t shakeState[Plen];
__align__(32) __constant__ static uint8_t shakeState_dev[Plen];

__device__ int deviceArch;

__device__ __forceinline__ unsigned int v_dot4_u32_u8(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int result;
#if (HIP_ARCH >= 1030)
  asm volatile("v_dot4_u32_u8 %0, %1, %2, %3" : "=v"(result) : "v"(a), "v"(b), "v"(c));
#endif
  return result;
}

__device__ __forceinline__ unsigned int v_dot4_u32_u4(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int result;
#if (HIP_ARCH >= 1030)
  __asm__("v_dot8_u32_u4" " %0, %1, %2, %3;": "=v" (result): "r" (a), "r" (b), "v" (c));
#endif
  return result;
}

__device__ __forceinline__ void amul4bit(uint32_t packed_vec1[32], uint32_t packed_vec2[32], uint32_t *ret)
{
  // We assume each 32 bits have four values: A0 B0 C0 D0
  unsigned int res = 0;
#pragma unroll
  for (int i = 0; i < QUARTER_MATRIX_SIZE; i++)
  {
#ifdef __HIP_PLATFORM_AMD__ // Ensure this is only compiled on AMD platforms
#if (HIP_ARCH >= 1030)
    res = v_dot4_u32_u4(packed_vec1[i], packed_vec2[i], res);
#else
    char4 *a4 = (char4 *)packed_vec1;
    char4 *b4 = (char4 *)packed_vec2;
    res += a4[i].x * b4[i].x;
    res += a4[i].y * b4[i].y;
    res += a4[i].z * b4[i].z;
    res += a4[i].w * b4[i].w;
#endif
#else
#if __CUDA_ARCH__ >= 610
    res = __dp4a(packed_vec1[i], packed_vec2[i], res);
#else
    char4 *a4 = (char4 *)packed_vec1;
    char4 *b4 = (char4 *)packed_vec2;
    res += a4[i].x * b4[i].x;
    res += a4[i].y * b4[i].y;
    res += a4[i].z * b4[i].z;
    res += a4[i].w * b4[i].w;
#endif
#endif
  }

  *ret = res;
}

__device__ __forceinline__ void vecXORassign(uint32_t *a, uint32_t *b, uint32_t *c) {
  uint4 vB = make_uint4(b[0], b[1], b[2], b[3]);
  uint4 vC = make_uint4(c[0], c[1], c[2], c[3]);

  uint4 result = vB ^ vC;

  a[0] = result.x;
  a[1] = result.y;
  a[2] = result.z;
  a[3] = result.w;
}

__device__ __forceinline__ void vecAssign(uint32_t *a, uint32_t *b, uint32_t *c) {
  ((uint4*)a)[0] = ((uint4*)b)[0];
}

template <bool isDev>
__launch_bounds__(1024, 10)
__global__ void astrixShake_hip(const uint64_t nonce_mask, const uint64_t nonce_fixed, int kIndex, size_t batchSize, uint8_t *output, uint8_t device = 0) {
  
  alignas(32) uint8_t input[200];
  #pragma unroll
  for (int i = 72; i < 200; i++) {
    input[i] = powP[i];
  }

  blake3_hasher blake3_hasher;
  blake3_hasher_init(&blake3_hasher);

  int nonceId = threadIdx.x + blockIdx.x * blockDim.x;

  int i;
  int j;
  uint64_t nonce;

  nonce = nonceId + kIndex * batchSize;
  nonce = (nonce << 4) | device;
  nonce = (nonce & nonce_mask) | nonce_fixed;

  #pragma unroll
  for (i = 0; i < 72; i++) {
    if constexpr (isDev) {
      input[i] = shakeState_dev[i];
    }
    else {
      input[i] = shakeState[i];
    }
  }

  ((uint64_t *)input)[9] ^= nonce;

  if constexpr (BENCH_RUNS == 1)
    if (nonceId == 0) {
      printf("SHA3 shake input: ");
      for (int i = 0; i < 200; i++) {
        printf("%02x", input[i]);
      }
      printf("\n");
    }

  keccakf(input);

  blake3_hasher_update_aix(&blake3_hasher, input);
  blake3_hasher_finalize_aix(&blake3_hasher, input);

  memset(input + 32, 0, 200-32);

  input[32] = 0x06;
  input[135] = 0x80;

  keccakf(input);

  memcpy(&output[nonceId*32], input, 32);
}

template <bool isDev>
__launch_bounds__(1024, 10)
__global__ void astrixHash_hip(uint64_t *final_nonces, int *nonce_count, uint8_t *dataBuf, uint8_t device = 0)
{
  int globalId = threadIdx.x + blockIdx.x * blockDim.x;
  int i;

  alignas(32) uint8_t input[200] = {0};
  memcpy(input, &dataBuf[globalId*32], 32);

  if constexpr (BENCH_RUNS == 1)
    if (globalId == 0) {
      printf("SHA3 GPU result: ");
      for (int i = 0; i < 32; i++) {
        printf("%02x", input[i]);
      }
      printf("\n");
    }

  uchar4 packed_hash[QUARTER_MATRIX_SIZE];
#pragma unroll
  for (i = 0; i < QUARTER_MATRIX_SIZE; i++)
  {
    packed_hash[i] = make_uchar4(
        (input[2 * i] & 0xF0) >> 4,
        (input[2 * i] & 0x0F),
        (input[2 * i + 1] & 0xF0) >> 4,
        (input[2 * i + 1] & 0x0F));
  }
  uint32_t product1, product2;

#pragma unroll
  for (int rowId = 0; rowId < HALF_MATRIX_SIZE; rowId++)
  {
    if constexpr (isDev) {
      amul4bit((uint32_t *)(matrix_dev[(2 * rowId)]), (uint32_t *)(packed_hash), &product1);
      amul4bit((uint32_t *)(matrix_dev[(2 * rowId + 1)]), (uint32_t *)(packed_hash), &product2);
    } else {
      amul4bit((uint32_t *)(matrix[(2 * rowId)]), (uint32_t *)(packed_hash), &product1);
      amul4bit((uint32_t *)(matrix[(2 * rowId + 1)]), (uint32_t *)(packed_hash), &product2);
    }
    product1 >>= 6;
    product1 &= 0xF0;
    product2 >>= 10;

#ifdef __HIP_PLATFORM_NVCC__
#if __CUDA_ARCH__ < 500 || __CUDA_ARCH__ > 700
    input[rowId] = input[rowId] ^ ((uint8_t)(product1) | (uint8_t)(product2));
#else
    uint32_t lop_temp = input[rowId];
    asm("lop3.b32"
        " %0, %1, %2, %3, 0x56;" : "=r"(lop_temp) : "r"(product1), "r"(product2), "r"(lop_temp));
    input[rowId] = lop_temp;
#endif
#else
    input[rowId] = input[rowId] ^ ((uint8_t)(product1) | (uint8_t)(product2));
#endif
  }

  vecXORassign((uint32_t*)input, (uint32_t*)input, (uint32_t*)heavyP);
  vecXORassign(&((uint32_t*)input)[4], &((uint32_t*)input)[4], &((uint32_t*)heavyP)[4]);

  #pragma unroll
  for (i = 32; i < 200; i++) input[i] = heavyP[i];
  // input[199] = heavyP[199];
  // memcpy(input + 32, heavyP + 32, 200-32);

  keccakf(input);
  if (LT_U256(((uint256_t*)input)[0], target))
  {
    int index = atomicAdd(nonce_count, 1);
    index = index >= Astrix_HIP::MAX_NONCES ? Astrix_HIP::MAX_NONCES - 1 : index;
    final_nonces[index] = globalId;
  }
  
  if constexpr (BENCH_RUNS == 1)
    if (globalId == 0) {
      printf("GPU result: ");
      for (int i = 0; i < 32; i++) {
        printf("%02x", input[i]);
      }
      printf("\n");
    }
}

inline int calculateRank(uint8_t mat[MATRIX_SIZE][MATRIX_SIZE])
{
  double copied[MATRIX_SIZE][MATRIX_SIZE];
  bool rowsSelected[MATRIX_SIZE] = {false};
  const double epsilon = 1e-9;

  for (int i = 0; i < MATRIX_SIZE; i++)
  {
    for (int j = 0; j < MATRIX_SIZE; j++)
    {
      copied[i][j] = static_cast<double>(mat[i][j]);
    }
  }

  int rank = 0;
  for (int i = 0; i < MATRIX_SIZE; i++)
  {
    int j;
    for (j = 0; j < MATRIX_SIZE; j++)
    {
      if (!rowsSelected[j] && std::abs(copied[j][i]) > epsilon)
      {
        break;
      }
    }

    if (j != MATRIX_SIZE)
    {
      rank++;
      rowsSelected[j] = true;
      for (int k = i + 1; k < MATRIX_SIZE; k++)
      {
        copied[j][k] /= copied[j][i];
      }

      for (int k = 0; k < MATRIX_SIZE; k++)
      {
        if (k == j || std::abs(copied[k][i]) <= epsilon)
        {
          continue;
        }

        for (int l = i + 1; l < MATRIX_SIZE; l++)
        {
          copied[k][l] -= copied[j][l] * copied[k][i];
        }
      }
    }
  }

  return rank;
}

namespace Astrix_HIP {
  void getHashBlocksPerSM(int *numBlocksPerSm, bool isDev) {
    hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocksPerSm, astrixHash_hip<false>, Astrix_HIP::THREAD_DIM, 0);
  }

  void absorbPow(uint8_t *work, bool isDev) {
    uint8_t host_powP[Plen];
    hipMemcpyFromSymbol(host_powP, powP, 200, 0, hipMemcpyDeviceToHost);

    for (int i = 0; i < 72; i++) {
      host_powP[i] ^= work[i];
    }

    if (isDev) {
      hipMemcpyToSymbol(shakeState_dev, host_powP, 200);
    } else {
      hipMemcpyToSymbol(shakeState, host_powP, 200);
    }
  }

  template<bool isDev>
  void copyWork(uint8_t *work) {
    if constexpr (isDev) hipMemcpyToSymbol(hash_header_dev, work, 72);
    else hipMemcpyToSymbol(hash_header, work, 72);
  }

  template void copyWork<true>(uint8_t *work);
  template void copyWork<false>(uint8_t *work);

  void copyDiff(uint8_t *diff) {
    hipMemcpyToSymbol(target, diff, 32);
  }

  void astrixHash_wrapper(
    int blocks,
    const uint64_t nonce_mask, 
    const uint64_t nonce_fixed, 
    uint64_t *final_nonces,
    int *nonce_count, 
    uint8_t *dataBuf,
    int kIndex, 
    size_t batchSize, 
    uint8_t device,
    bool isDev
  ) {
    hipMemset(nonce_count, 0, sizeof(int));

    // printf("after memset\n");
    if (isDev) {
      hipLaunchKernelGGL(astrixShake_hip<true>, dim3(blocks), dim3(Astrix_HIP::THREAD_DIM), 0, 0,
                       nonce_mask, nonce_fixed, kIndex, batchSize, dataBuf);

      hipError_t err = hipGetLastError();
      if (err != hipSuccess)
      {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
      }

      hipLaunchKernelGGL(astrixHash_hip<true>, dim3(blocks), dim3(Astrix_HIP::THREAD_DIM), 0, 0,
                        final_nonces, nonce_count, dataBuf);

      err = hipGetLastError();
      if (err != hipSuccess)
      {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
      }
    } else {
      hipLaunchKernelGGL(astrixShake_hip<false>, dim3(blocks), dim3(Astrix_HIP::THREAD_DIM), 0, 0,
                       nonce_mask, nonce_fixed, kIndex, batchSize, dataBuf); 
      
      hipError_t err = hipGetLastError();
      if (err != hipSuccess)
      {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
      }

      hipLaunchKernelGGL(astrixHash_hip<false>, dim3(blocks), dim3(Astrix_HIP::THREAD_DIM), 0, 0,
                        final_nonces, nonce_count, dataBuf);

      err = hipGetLastError();
      if (err != hipSuccess)
      {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
      }
    }

    hipDeviceSynchronize();
  }

  void nonceCounter(int *d_nonce_count, int *h_nonce_count, uint64_t *d_final_nonces, uint64_t *h_nonce_buffer) {
    hipMemcpy(h_nonce_count, d_nonce_count, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_nonce_buffer, d_final_nonces, sizeof(uint64_t)*(*h_nonce_count), hipMemcpyDeviceToHost);
  }

  void newMatrix(uint8_t *in, bool isDev) {
    // Prepare host-side data for the matrix, hash_header, and target
    uint8_t h_matrix[MATRIX_SIZE][MATRIX_SIZE]; // Host-side matrix

    // Fill the matrix
    for (int i = 0; i < MATRIX_SIZE; i++) {
      memset(h_matrix[i], 0, MATRIX_SIZE);
    }

    alignas(64) uint64_t s0 = ((uint64_t*)in)[0];
    uint64_t s1 = ((uint64_t*)in)[1];
    uint64_t s2 = ((uint64_t*)in)[2];
    uint64_t s3 = ((uint64_t*)in)[3];

    Xoshiro256PlusPlusHasher hasher(s0, s1, s2, s3);
                  
    while (calculateRank(h_matrix) != MATRIX_SIZE)
    {
      for (int i = 0; i < MATRIX_SIZE; i++)
      {
        for (int j = 0; j < MATRIX_SIZE; j += QUARTER_MATRIX_SIZE)
        {
          uint64_t value = hasher.next();
          for (int k = 0; k < 16; k++)
          {
            h_matrix[i][j + k] = uint16_t((value >> (4 * k)) & 0x0f);
          }
        }
      }
    }

    if (isDev) hipMemcpyToSymbol(matrix_dev, h_matrix, sizeof(h_matrix));
    else hipMemcpyToSymbol(matrix, h_matrix, sizeof(h_matrix));
  }
}

void benchAstrixHip()
{
  int device;
  hipGetDevice(&device);

  hipDeviceProp_t deviceProps;
  hipGetDeviceProperties(&deviceProps, device); // Query device properties
  int smCount = deviceProps.multiProcessorCount;

  // Get the architecture version (gcnArch) and store it for device code
  // int arch = deviceProps.gcnArch;
  // hipMemcpyToSymbol(HIP_SYMBOL(deviceArch), &arch, sizeof(int), 0, hipMemcpyHostToDevice);

  // Retrieve architecture-specific dimensions
  // getArchDims(blocks, threads, batchSize);

  // Example input data setup
  uint64_t nonce_mask = 0xFFFFFFFFFFFFFFFF;  // Example nonce mask
  uint64_t nonce_fixed = 0x0000000000000000; // Example nonce fixed value

  // Allocate memory for final_nonce on the CPU
  uint64_t h_final_nonces[Astrix_HIP::MAX_NONCES] = {0};

  // Allocate space for final_nonce on the GPU
  uint64_t *d_final_nonces;

  hipMalloc((void **)&d_final_nonces, sizeof(uint64_t) * Astrix_HIP::MAX_NONCES);

  // Timing events
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // Launch the kernel and perform the benchmark
  uint8_t *d_dataBuf;

  int *d_nonce_count;
  int h_nonce_count = 0; // Initialize on the host
  hipMalloc(&d_nonce_count, sizeof(int));

  // Prepare host-side data for the matrix, hash_header, and target
  uint8_t h_matrix[MATRIX_SIZE][MATRIX_SIZE]; // Host-side matrix
  uint8_t h_hash_header[Astrix_HIP::HASH_HEADER_SIZE];    // Host-side hash header
  uint256_t h_target;                         // Host-side target

  for (int i = 0; i < Astrix_HIP::HASH_HEADER_SIZE; i++)
  {
    h_hash_header[i] = i % 256; // Example data
    // h_hash_header[i] = 0; // test 0 input
  }

  for (int i = 0; i < 4; i++)
  {
    h_target.number[i] = 0x000000FFFFFFFFFFULL; // Example target
    // h_target.number[i] = 0; // Example target
  }

  // Fill the matrix
  for (int i = 0; i < MATRIX_SIZE; i++) {
    memset(h_matrix[i], 0, MATRIX_SIZE);
  }

  alignas(64) uint64_t s0 = *(uint64_t*)&h_hash_header[0];
  alignas(64) uint64_t s1 = *(uint64_t*)&h_hash_header[8];
  alignas(64) uint64_t s2 = *(uint64_t*)&h_hash_header[16];
  alignas(64) uint64_t s3 = *(uint64_t*)&h_hash_header[24];

  Xoshiro256PlusPlusHasher hasher(s0, s1, s2, s3);
                
  while (calculateRank(h_matrix) != MATRIX_SIZE)
  {
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
      for (int j = 0; j < MATRIX_SIZE; j += QUARTER_MATRIX_SIZE)
      {
        uint64_t value = hasher.next();
        for (int k = 0; k < 16; k++)
        {
          h_matrix[i][j + k] = uint16_t((value >> (4 * k)) & 0x0f);
        }
      }
    }
  }


  // Transfer the matrix, hash_header, and target data to device constant memory
  hipMemcpyToSymbol(matrix, h_matrix, sizeof(h_matrix));
  hipMemcpyToSymbol(hash_header, h_hash_header, sizeof(h_hash_header));
  hipMemcpyToSymbol(&target, &h_target, sizeof(h_target));

  Astrix_HIP::absorbPow(h_hash_header, false);

  size_t blocks_shake, threads_shake, batchSize_shake;
  size_t blocks, threads, batchSize;

  int numBlocksPerSm;
  hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, astrixHash_hip<false>, Astrix_HIP::THREAD_DIM, 0);
  blocks = numBlocksPerSm*smCount*128;
  batchSize = blocks*Astrix_HIP::THREAD_DIM;

  if (batchSize == 0) {
    // blocks = 1;
    // batchSize = Astrix_HIP::THREAD_DIM;
    getArchDims(blocks, threads, batchSize);
  }

  hipMalloc((void **)&d_dataBuf, sizeof(uint8_t) * batchSize * 32);

  printf("HASH: batchSize: %llu\nsmCount: %llu\n", batchSize, smCount);
  fflush(stdout);fflush(stdout);

  int shakeBlocks = (batchSize + (Astrix_HIP::SHAKE_THREAD_DIM-1)) / Astrix_HIP::SHAKE_THREAD_DIM;

  // Record start event
  hipEventRecord(start, 0);

  for (int i = 0; i < BENCH_RUNS; ++i)
  {
    hipMemset(d_nonce_count, 0, sizeof(int));

    hipLaunchKernelGGL(astrixShake_hip<false>, dim3(blocks), dim3(Astrix_HIP::THREAD_DIM), 0, 0,
                       nonce_mask, nonce_fixed, i, batchSize, d_dataBuf);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
      printf("SHAKE: Kernel launch failed: %s\n", hipGetErrorString(err));
      break;
    }

    hipLaunchKernelGGL(astrixHash_hip<false>, dim3(blocks), dim3(Astrix_HIP::THREAD_DIM), 0, 0,
                       d_final_nonces, d_nonce_count, d_dataBuf);

    // Check for errors after the kernel launch
    err = hipGetLastError();
    if (err != hipSuccess)
    {
      printf("HASH: Kernel launch failed: %s\n", hipGetErrorString(err));
      break;
    }
  }

  // Record stop event
  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);

  // Calculate throughput
  float seconds = milliseconds / 1000.0f;
  size_t totalComputations = batchSize * BENCH_RUNS;
  float runsPerSecond = totalComputations / seconds;

  printf("Total time for %d runs: %.3f ms\n", BENCH_RUNS, milliseconds);
  printf("Runs per second: %.2f\n", runsPerSecond);

  // Copy final nonce result back to the host
  hipMemcpy(&h_final_nonces, d_final_nonces, sizeof(uint64_t), hipMemcpyDeviceToHost);

  // printf("Final nonce: %llu\n", h_final_nonces[0]);

  // Free memory on the GPU and CPU
  hipFree(d_final_nonces);
  hipFree(d_dataBuf);
  hipFree(d_nonce_count);

  // Destroy events
  hipEventDestroy(start);
  hipEventDestroy(stop);

  hipDeviceReset();
}
