#include <cuda.h>
#include <cuda_runtime.h>
#include "powtest.h"
#include "astrobwtv3_cuda.cuh"
#include "sha256_cuda.cuh"

#include <xxhash64_cuda.cuh>
#include <fnv1a_cuda.cuh>
#include <inttypes.h>
#include <iostream>

#include <fnv1a.h>
#include <xxhash64.h>
#include <highwayhash/sip_hash.h>
#include <openssl/rc4.h>

#include <bitset>
#include <libcubwt.cuh>
#include <device_sa.cuh>
#include <hex.h>
#include <lookup.h>
#include <chrono>

#include "divsufsort_def.cuh"
#include "divsufsort_cuda.cuh"
#include "sais_cuda.cuh"

#include <cuda_profiler_api.h>

#define PROFILE_KERNELS 1

#define MB_SIZE 48

using byte = unsigned char;

const int kernelThreads = 256;
const int sharedSize = kernelThreads / 32;

template <typename T>
__host__ __device__ void swap(T &a, T &b)
{
  T c(a);
  a = b;
  b = c;
}

__host__ __device__ byte rotl_cuda(byte x, int n)
{
  return (x << (n % 8)) | (x >> (8 - (n % 8)));
}

void workerMalloc(workerData_cuda &worker, int batchSize)
{
  cudaMalloc((void **)&worker.sha_key, 32 * batchSize);
  cudaMalloc((void **)&worker.keystream, 64 * sizeof(uint8_t) * batchSize);
  cudaMalloc((void **)&worker.BA, 256 * sizeof(int) * batchSize);
  cudaMalloc((void **)&worker.BB, 256 * 256 * sizeof(int) * batchSize);
  cudaMalloc((void **)&worker.sa, MAXX * sizeof(int32_t) * batchSize);
  cudaMalloc((void **)&worker.sData, (MAXX + 64) * sizeof(int) * batchSize);
  cudaMalloc((void **)&worker.data_len, 256 * sizeof(uint32_t) * batchSize);
}

void TestAstroBWTv3_cuda()
{
  workerData_cuda *worker_h = (workerData_cuda *)malloc(sizeof(workerData_cuda));

  printf("after malloc\n");

  void *cudaStore;
  // libcubwt_allocate_device_storage(&cudaStore, MAXX);

  workerData_cuda worker;
  // cudaMalloc((void **)&worker, sizeof(workerData_cuda));
  // cudaMemcpy(worker, worker_h, sizeof(workerData_cuda), cudaMemcpyHostToDevice);
  workerMalloc(worker, 1);

  int i = 0;
  // SHA256 setup
  for (PowTest t : random_pow_tests)
  {
    // if (i > 0) break;
    byte *d_buf;
    cudaMalloc((void **)&d_buf, t.in.size());
    cudaMemset(d_buf, 0, t.in.size());
    cudaMemcpy(d_buf, t.in.c_str(), (int)t.in.size(), cudaMemcpyHostToDevice);
    byte *res = (byte *)malloc(32);
    byte *d_res;
    cudaMalloc((void **)&d_res, 32);

    ASTRO_CUDA(d_buf, d_res, worker, (int)t.in.size(), 1, 0, 0, 0);

    cudaMemcpy(res, d_res, 32, cudaMemcpyDeviceToHost);

    std::string s = hexStr(res, 32);

    if (s.c_str() != t.out)
    {
      printf("CUDA: FAIL. Pow function: pow(%s) = %s want %s\n", t.in.c_str(), s.c_str(), t.out.c_str());
    }
    else
    {
      printf("CUDA: SUCCESS! pow(%s) = %s want %s\n", t.in.c_str(), s.c_str(), t.out.c_str());
    }
    cudaFree(d_res);
    cudaFree(d_buf);
    free(res);
    i++;
  }
}

__global__ void ASTRO_INIT_kernel(int device, byte *work, int batchSize, int offset, int nonce)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  clock_t start = clock();

  if (nonce > 0)
  {
    int i = nonce + offset + index / 4;
    // i = 0;
    if (index < batchSize * 4)
    {
      if (index % 4 == 0)
        work[offset * MB_SIZE + MB_SIZE * index / 4 + MB_SIZE - 5] = i & 0x000000ff;
      else if (index % 4 == 1)
        work[offset * MB_SIZE + MB_SIZE * index / 4 + MB_SIZE - 4] = (i & 0x0000ff00) >> 8;
      else if (index % 4 == 2)
        work[offset * MB_SIZE + MB_SIZE * index / 4 + MB_SIZE - 3] = (i & 0x00ff0000) >> 16;
      else if (index % 4 == 3)
        work[offset * MB_SIZE + MB_SIZE * index / 4 + MB_SIZE - 2] = (i & 0xff000000) >> 24;
    }
  }

  __syncwarp();
  clock_t stop = clock();

  int time = (int)(stop - start);
  if (PROFILE_KERNELS == 1 && index == 0)
    printf("WORKER %d: Nonce copying took %d clock cycles\n", index % 32, time);
}

__host__ void ASTRO_INIT(int device, byte *work, int batchSize, int offset, int nonce)
{
  int B = (batchSize + 256 - 1) / 256;
  int T = 256;
  ASTRO_INIT_kernel<<<B, T>>>(device, work, batchSize, offset, nonce);
}

__global__ void
__launch_bounds__(kernelThreads)
    ASTRO_1_kernel(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int d, int offset, int nonce)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ byte s3_shared[256 * sharedSize];
  __shared__ SHA256_CTX_CUDA sha256_shared[sharedSize];
  __shared__ byte shaKey_shared[32 * sharedSize];
  __shared__ rc4_state key_shared[sharedSize];
  __shared__ uint64_t tries_shared[sharedSize];
  __shared__ uint64_t lhash_shared[sharedSize];
  __shared__ uint64_t prev_lhash_shared[sharedSize];
  __shared__ uint64_t random_switcher_shared[sharedSize];
  __shared__ byte pos1_shared[sharedSize];
  __shared__ byte pos2_shared[sharedSize];
  __shared__ byte op_shared[sharedSize];
  __shared__ byte A_shared[sharedSize];
  __shared__ byte temp[sharedSize * 2];

  clock_t start = clock();
  if (index < batchSize * 32)
  {
    AstroBWTv3_cuda_p1(
        &work[offset * inputLen + (index / 32) * inputLen],
        inputLen, worker, &s3_shared[threadIdx.x / 32 * 256],
        sha256_shared[threadIdx.x / 32], &shaKey_shared[threadIdx.x],
        key_shared[threadIdx.x / 32], lhash_shared[threadIdx.x / 32],
        prev_lhash_shared[threadIdx.x / 32]);
    tries_shared[threadIdx.x / 32] = 0;
  }

  // __syncwarp();
  clock_t stop = clock();

  int time = (int)(stop - start);
  if (PROFILE_KERNELS == 1 && index == 0)
    printf("WORKER %d: AstroBWTv3_cuda_p1() took %d clock cycles\n", index % 32, time);

  unsigned mask = 0xffffffff;

  start = clock();

  if (index < batchSize * 32)
  {
    AstroBWTv3_cuda_p2(
        work, inputLen, output, worker,
        s3_shared, tries_shared, lhash_shared,
        prev_lhash_shared, random_switcher_shared,
        key_shared, pos1_shared,
        pos2_shared, op_shared,
        A_shared, temp,
        batchSize, offset);
  }

  __syncwarp();

  stop = clock();
  time = (int)(stop - start);
  if (PROFILE_KERNELS == 1 && index == 0)
    printf("WORKER %d: AstroBWTv3_cuda_p2() took %d clock cycles\n", index % 32, time);
}

__global__ void ASTRO_2_kernel(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int d, int offset)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef _DSS_PARALLEL
  int wIndex = (index - index % _DSS_THREADS) / _DSS_THREADS;

  if (index < batchSize * _DSS_THREADS)
  {
    divsufsort_cuda(&worker.sData[(offset + wIndex) * (MAXX + 64)], &worker.sa[(offset + wIndex) * MAXX], worker.data_len[offset + wIndex], &worker.BA[(offset + wIndex) * 256], &worker.BB[(offset + wIndex) * 256 * 256]);
    // sais(workers[offset + index].sData, workers[offset + index].sa, workers[offset + index].data_len);
  }
#else
  if (index < batchSize)
  {
    divsufsort_cuda(worker[offset + index].sData, worker[offset + index].sa, worker[offset + index].data_len, worker[offset + index].BA, worker[offset + index].BB);
    // sais(workers[offset + index].sData, workers[offset + index].sa, workers[offset + index].data_len);
  }
#endif
  __syncthreads();
}

void ASTRO_CUDA(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int d, int offset, int nonce)
{
  if (PROFILE_KERNELS == 1)
    printf("\n=========\nNEW ROUND\n=========\n");
  int B = (batchSize * 32 + kernelThreads - 1) / kernelThreads;
  int T = kernelThreads;

  if (PROFILE_KERNELS == 1)
    printf("\nLaunching ASTRO_1_kernel\n\n");
  auto start = std::chrono::high_resolution_clock::now();
  ASTRO_1_kernel<<<B, T>>>(work, output, worker, inputLen, batchSize, d, offset, nonce);
  gpuErrchk(cudaDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  if (PROFILE_KERNELS == 1)
    printf("\nASTRO_1_kernel took %d milliseconds\n", time.count());

#ifndef _DSS_PARALLEL
  B = (batchSize + 512 - 1) / 512;
  T = 512;
#else
  B = (batchSize * _DSS_THREADS + 512 - 1) / 512;
  T = 512;
#endif

  if (PROFILE_KERNELS == 1) printf("\nLaunching ASTRO_2_kernel\n\n");
  start = std::chrono::high_resolution_clock::now();
  ASTRO_2_kernel<<<B, T>>>(work, output, worker, inputLen, batchSize, d, offset);
  gpuErrchk(cudaDeviceSynchronize());
  stop = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  if (PROFILE_KERNELS == 1) printf("\nASTRO_2_kernel took %d milliseconds\n", time.count());

  // // printf("after kernel 2\n");

  // // int B = (batchSize + 32 - 1) / 32;
  // // int T = 32;
  // // gpuErrchk(cudaDeviceSynchronize());

  B = (batchSize + 384 - 1) / 384;
  T = 384;
  ASTRO_3_kernel<<<B, T>>>(work, output, worker, inputLen, batchSize, d, offset);
  gpuErrchk(cudaDeviceSynchronize());
}

void ASTRO_1(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int d, int offset)
{
  int B = (batchSize + 128 - 1) / 128;
  int T = 128;
  ASTRO_1_kernel<<<B, T>>>(work, output, worker, inputLen, batchSize, d, offset, 0);
}

void ASTRO_2(void **cudaStore, workerData_cuda worker, int batchSize)
{
  for (int i = 0; i < batchSize; i++)
  {
    printf("made it into\n");
    // divsufsort_cuda(workers[i].sData, workers[i].sa, workers[i].data_len);
  }
}

__global__ void ASTRO_3_kernel(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int d, int offset)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  __shared__ SHA256_CTX_CUDA sha256_shared[384];

  for (int i = index; i < batchSize; i += stride)
  {
    // workers[offset + i].data_len = 70000;
    // printf("data_len = %d\n", workers[offset + i].data_len);
    // sais(workers[offset + i].sData, workers[offset + i].sa, workers[offset + i].data_len);
    AstroBWTv3_cuda_p3(&output[offset * 32 + i * 32], worker, sha256_shared[threadIdx.x]);
  }
}

void ASTRO_3(byte *work, byte *output, workerData_cuda worker, int inputLen, int batchSize, int d, int offset)
{
  int B = (batchSize + 512 - 1) / 512;
  int T = 512;
  ASTRO_3_kernel<<<B, T>>>(work, output, worker, inputLen, batchSize, d, offset);
}

__device__ void AstroBWTv3_cuda_p1(
    unsigned char *input,
    int inputLen, workerData_cuda worker,
    byte *step3_shared, SHA256_CTX_CUDA &sha256,
    byte *sha_key_shared, rc4_state &key_shared,
    uint64_t &lhash_shared, uint64_t &prev_lhash_shared)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadIdx.x % 32; i < 256; i += 32)
  {
    step3_shared[i] = 0;
  }
  __syncwarp();

  if (threadIdx.x % 32 == 0)
  {
    SHA256_cuda(sha256, input, sha_key_shared, inputLen);

    if (s20_crypt(sha_key_shared, s20_keylen_t::S20_KEYLEN_256, 0, step3_shared, 256, &worker.keystream[(index / 32) * 64]) != S20_SUCCESS)
      printf("salsa20 failure\n");

    rc4_setup(&key_shared, step3_shared, 256);
    rc4_crypt(&key_shared, step3_shared, 256);

    // printf("worker.step_3 post rc4: ");
    // printf(hexStr_cuda(worker.step_3, 256));
    // printf("\n\n\n");

    // // std::cout << "worker.step_3 post rc4: " << hexStr(worker.step_3, 256) << std::endl;

    hash_64_fnv1a_cuda(step3_shared, 256, &lhash_shared);

    prev_lhash_shared = lhash_shared;
  }
  __syncwarp();
}

__device__ void AstroBWTv3_cuda_p2(
    unsigned char *inputs, int inputLen,
    unsigned char *outputHashes, workerData_cuda worker,
    byte *step3, uint64_t *tries_arr,
    uint64_t *lhash_arr, uint64_t *prev_lhash_arr,
    uint64_t *random_switcher_arr,
    rc4_state *key_arr, byte *pos1_arr,
    byte *pos2_arr, byte *op_arr,
    byte *A_arr, byte *temp,
    int batchSize, int offset)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  clock_t start = clock();

  byte *step3_shared;
  // workerData_cuda *worker;
  uint64_t &tries_shared = tries_arr[(threadIdx.x - threadIdx.x % 32) / 32];
  uint64_t &lhash_shared = lhash_arr[(threadIdx.x - threadIdx.x % 32) / 32];
  uint64_t &prev_lhash_shared = prev_lhash_arr[(threadIdx.x - threadIdx.x % 32) / 32];
  uint64_t &random_switcher_shared = random_switcher_arr[(threadIdx.x - threadIdx.x % 32) / 32];
  rc4_state &key_shared = key_arr[(threadIdx.x - threadIdx.x % 32) / 32];
  byte &pos1_shared = pos1_arr[(threadIdx.x - threadIdx.x % 32) / 32];
  byte &pos2_shared = pos2_arr[(threadIdx.x - threadIdx.x % 32) / 32];
  byte &op_shared = op_arr[(threadIdx.x - threadIdx.x % 32) / 32];
  byte &A_shared = A_arr[(threadIdx.x - threadIdx.x % 32) / 32];

  if (index < batchSize * 32)
  {
    // worker = &workers[offset + (index - index % 32) / 32];
    step3_shared = &step3[(threadIdx.x - threadIdx.x % 32) / 32 * 256];
  }

  __syncwarp();
  clock_t stop = clock();
  int time = (int)(stop - start);
  if (PROFILE_KERNELS == 1 && index == 0)
    printf("WORKER %d: p2 init took %d clock cycles\n", index % 32, time);
  // // printf("thread %d, warp/index %d\n", threadIdx.x, (offset + (index - index%32)/32));

  // // Previous version of this invocation for reference
  // AstroBWTv3_cuda_p2(&work[offset * inputLen + (index/32) * inputLen], inputLen, &output[offset * 32 + (index/32) * 32], workers[offset + (index/32)], &s3_shared[threadIdx.x * 256 / 32]);

  while (true)
  {
    __syncwarp();

    start = clock();

    if (index < batchSize * 32 && index % 32 == 0)
    {
      tries_shared++;
      random_switcher_shared = prev_lhash_shared ^ lhash_shared ^ tries_shared;
      // printf("%d random_switcher_shared %d %08jx\n", tries_shared, random_switcher_shared, random_switcher_shared);

      op_shared = static_cast<byte>(random_switcher_shared);

      // printf("worker.op: %d\n", op_shared);

      pos1_shared = static_cast<byte>(random_switcher_shared >> 8);
      pos2_shared = static_cast<byte>(random_switcher_shared >> 16);

      if (pos1_shared > pos2_shared)
      {
        swap(pos1_shared, pos2_shared);
      }

      if (pos2_shared - pos1_shared > 32)
      {
        pos2_shared = pos1_shared + ((pos2_shared - pos1_shared) & 0x1f);
      }
    }

    __syncwarp();
    stop = clock();

    time = (int)(stop - start);
    if (PROFILE_KERNELS == 1 && index == 0 && tries_shared == 200)
      printf("WORKER %d: p2 config took %d clock cycles\n", index % 32, time);

    start = clock();

    if (index < batchSize * 32)
      branchCompute(worker, step3_shared, lhash_shared, prev_lhash_shared, key_shared, pos1_shared, pos2_shared, op_shared, temp, batchSize);

    __syncwarp();
    stop = clock();
    time = (int)(stop - start);
    if (PROFILE_KERNELS == 1 && index == 0 && tries_shared == 200)
      printf("WORKER %d: p2 branchCompute took %d clock cycles\n", index % 32, time);

    start = clock();
    if (index < batchSize * 32 && index % 32 == 0)
    {
      A_shared = (step3_shared[pos1_shared] - step3_shared[pos2_shared]);
      A_shared = (256 + (A_shared % 256)) % 256;

      if (A_shared < 0x10)
      { // 6.25 % probability
        prev_lhash_shared = lhash_shared + prev_lhash_shared;
        lhash_shared = XXHash64_cuda::hash(&step3_shared[0], pos2_shared, 0);
        // printf("A: new (*worker).lhash: %" PRIx64 "\n", lhash_shared);
      }

      if (A_shared < 0x20)
      { // 12.5 % probability
        prev_lhash_shared = lhash_shared + prev_lhash_shared;
        hash_64_fnv1a_cuda(step3_shared, pos2_shared, &lhash_shared);
        // printf("B: new (*worker).lhash: %" PRIx64 "\n", lhash_shared);
      }

      if (A_shared < 0x30)
      { // 18.75 % probability
        // std::copy(step3_shared, step3_shared + pos2_shared, s3);
        prev_lhash_shared = lhash_shared + prev_lhash_shared;

        const uint64_t key2[2] = {tries_shared, prev_lhash_shared};
        siphash_cuda(&step3_shared[0], pos2_shared, key2, (uint8_t *)&lhash_shared, 8);
        // printf("C: new (*worker).lhash: %" PRIx64 "\n", lhash_shared);
      }

      if (A_shared <= 0x40)
      { // 25% probablility
        rc4_crypt(&key_shared, step3_shared, 256);
      }

      step3_shared[255] = step3_shared[255] ^ step3_shared[pos1_shared] ^ step3_shared[pos2_shared];
    }

    __syncwarp();
    stop = clock();
    time = (int)(stop - start);
    if (PROFILE_KERNELS == 1 && index == 0 && tries_shared == 200)
      printf("WORKER %d: p2 op shuffle took %d clock cycles\n", index % 32, time);

    start = clock();
    if (index < batchSize * 32)
    {
      for (int i = threadIdx.x % 32; i < 256; i += 32)
      {
        worker.sData[((index - index % 32) / 32) * (MAXX + 64) + (tries_shared - 1) * 256 + i] = step3_shared[i];
      }
    }

    __syncwarp();
    stop = clock();
    time = (int)(stop - start);
    if (PROFILE_KERNELS == 1 && index == 0 && tries_shared == 200)
      printf("WORKER %d: p2 sData copy took %d clock cycles\n", index % 32, time);

    // std::copy(step3_shared, step3_shared + 256, &(*worker).sData[(tries_shared - 1) * 256]);

    // copy_kernel(&(*worker).data.data()[(tries_shared - 1) * 256], step3_shared, 256);

    // std::cout << hexStr(step3_shared, 256) << std::endl;

    if (index >= batchSize * 32 || tries_shared > 260 + 16 || (step3_shared[255] >= 0xf0 && tries_shared > 260))
    {
      break;
    }
  }

  start = clock();
  if (index < batchSize * 32 && index % 32 == 0)
  {
    worker.data_len[index / 32] = static_cast<uint32_t>((tries_shared - 4) * 256 + (((static_cast<uint64_t>(step3_shared[253]) << 8) | static_cast<uint64_t>(step3_shared[254])) & 0x3ff));
  }
  __syncwarp();
  stop = clock();
  time = (int)(stop - start);
  if (PROFILE_KERNELS == 1 && index == 0)
    printf("WORKER %d: p2 data_len copy took %d clock cycles\n", index % 32, time);
}

__device__ void AstroBWTv3_cuda_p3(unsigned char *outputHash, workerData_cuda worker, SHA256_CTX_CUDA &sha256_shared)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  SHA256_cuda(sha256_shared, (byte *)&worker.sa[index * MAXX], outputHash, worker.data_len[index] * 4);
  // worker.sHash = nHash;
}

__device__ void branchCompute(
    workerData_cuda worker, byte *step3_shared,
    uint64_t &lhash_shared, uint64_t &prev_lhash_shared,
    rc4_state &key_shared, byte &pos1_shared,
    byte &pos2_shared, byte &op_shared,
    byte *temp,
    int batchSize)
{

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  switch (op_shared)
  {
  case 0:
  {
    if (index % 32 == 0)
    {
      for (int i = pos1_shared; i < pos2_shared; i++)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
        step3_shared[i] *= step3_shared[i];                            // *
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        // INSERT_RANDOM_CODE_END
        temp[threadIdx.x / 16] = step3_shared[pos1_shared];
        temp[threadIdx.x / 16 + 1] = step3_shared[pos2_shared];
        step3_shared[pos1_shared] = reverse8_cuda(temp[threadIdx.x / 16 + 1]);
        step3_shared[pos2_shared] = reverse8_cuda(temp[threadIdx.x / 16]);
      }
    }
  }
  break;
  case 1:
  {
    if (index % 32 == 0)
    {
      for (int i = pos1_shared; i < pos2_shared; i++)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
        step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
        step3_shared[i] += step3_shared[i];                            // +
                                                                       // INSERT_RANDOM_CODE_END
      }
    }
  }
  break;
  case 2:
  {
    int i = pos1_shared + threadIdx.x % 32;

    if (i < pos2_shared)
    {
      // INSERT_RANDOM_CODE_START
      step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
      step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
      step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
      step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
                                                                  // INSERT_RANDOM_CODE_END
    }
  }
  break;
  case 3:
  {
    if (index % 32 == 0)
    {
      for (int i = pos1_shared; i < pos2_shared; i++)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
        step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
    }
  }
  break;
  case 4:
  {
    int i = pos1_shared + threadIdx.x % 32;

    if (i < pos2_shared)
    {
      // INSERT_RANDOM_CODE_START
      step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
      step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
      step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
      step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
                                                                     // INSERT_RANDOM_CODE_END
    }
  }
  break;
  case 5:
  {
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {

          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right

          // INSERT_RANDOM_CODE_END
        }
      }
      break;
    case 6:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);            // rotate  bits by 3
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -

        // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 7:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                            // +
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
        step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 8:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
        step3_shared[i] = rotl_cuda(step3_shared[i], 10); // rotate  bits by 5
        // step3_shared[i] = rotl_cuda(step3_shared[i], 5);// rotate  bits by 5
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 9:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 10:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];              // binary NOT operator
        step3_shared[i] *= step3_shared[i];              // *
        step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
        step3_shared[i] *= step3_shared[i];              // *
                                                         // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 11:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 6); // rotate  bits by 1
          // step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 12:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 13:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 14:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] *= step3_shared[i];                         // *
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 15:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 16:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 17:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];    // XOR
          step3_shared[i] *= step3_shared[i];              // *
          step3_shared[i] = rotl_cuda(step3_shared[i], 5); // rotate  bits by 5
          step3_shared[i] = ~step3_shared[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 18:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], 9);  // rotate  bits by 3
                                                          // step3_shared[i] = rotl_cuda(step3_shared[i], 1);             // rotate  bits by 1
                                                          // step3_shared[i] = rotl_cuda(step3_shared[i], 5);         // rotate  bits by 5
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 19:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] += step3_shared[i];                         // +
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 20:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 21:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] += step3_shared[i];                            // +
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 22:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] *= step3_shared[i];                         // *
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 23:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 4); // rotate  bits by 3
          // step3_shared[i] = rotl_cuda(step3_shared[i], 1);                           // rotate  bits by 1
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 24:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 25:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 26:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] *= step3_shared[i];                   // *
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] += step3_shared[i];                   // +
        step3_shared[i] = reverse8_cuda(step3_shared[i]);     // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 27:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 28:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 29:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] *= step3_shared[i];                         // *
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
          step3_shared[i] += step3_shared[i];                         // +
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 30:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 31:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] *= step3_shared[i];                         // *
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 32:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 33:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
        step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
        step3_shared[i] *= step3_shared[i];                            // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 34:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 35:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] += step3_shared[i];              // +
          step3_shared[i] = ~step3_shared[i];              // binary NOT operator
          step3_shared[i] = rotl_cuda(step3_shared[i], 1); // rotate  bits by 1
          step3_shared[i] ^= step3_shared[pos2_shared];    // XOR
                                                           // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 36:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);     // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 37:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
        step3_shared[i] *= step3_shared[i];                            // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 38:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 39:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 40:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 41:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
        step3_shared[i] -= (step3_shared[i] ^ 97);        // XOR and -
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 42:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 4); // rotate  bits by 1
        // step3_shared[i] = rotl_cuda(step3_shared[i], 3);                // rotate  bits by 3
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 43:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] += step3_shared[i];                            // +
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 44:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 45:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 10); // rotate  bits by 5
          // step3_shared[i] = rotl_cuda(step3_shared[i], 5);                       // rotate  bits by 5
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 46:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] += step3_shared[i];                   // +
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);      // rotate  bits by 5
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 47:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 48:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        // step3_shared[i] = ~step3_shared[i];                    // binary NOT operator
        // step3_shared[i] = ~step3_shared[i];                    // binary NOT operator
        step3_shared[i] = rotl_cuda(step3_shared[i], 5); // rotate  bits by 5
                                                         // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 49:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] += step3_shared[i];                   // +
        step3_shared[i] = reverse8_cuda(step3_shared[i]);     // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 50:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
        step3_shared[i] += step3_shared[i];               // +
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 51:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 52:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
        step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 53:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                   // +
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 54:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
                                                            // step3_shared[i] = ~step3_shared[i];    // binary NOT operator
                                                            // step3_shared[i] = ~step3_shared[i];    // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 55:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 56:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 57:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = rotl_cuda(step3_shared[i], 8);               // rotate  bits by 5
        // step3_shared[i] = rotl_cuda(step3_shared[i], 3);                // rotate  bits by 3
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 58:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] += step3_shared[i];                            // +
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 59:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {

        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
        step3_shared[i] *= step3_shared[i];                            // *
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 60:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];    // XOR
          step3_shared[i] = ~step3_shared[i];              // binary NOT operator
          step3_shared[i] *= step3_shared[i];              // *
          step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 61:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], 8);            // rotate  bits by 3
                                                                    // step3_shared[i] = rotl_cuda(step3_shared[i], 5);// rotate  bits by 5
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 62:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
          step3_shared[i] += step3_shared[i];                            // +
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 63:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);      // rotate  bits by 5
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] -= (step3_shared[i] ^ 97);            // XOR and -
        step3_shared[i] += step3_shared[i];                   // +
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 64:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
          step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
          step3_shared[i] *= step3_shared[i];               // *
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 65:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 8); // rotate  bits by 5
        // step3_shared[i] = rotl_cuda(step3_shared[i], 3);             // rotate  bits by 3
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] *= step3_shared[i];               // *
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 66:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 67:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);     // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);      // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 68:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 69:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] *= step3_shared[i];                         // *
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 70:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] *= step3_shared[i];                         // *
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 71:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] *= step3_shared[i];                         // *
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 72:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 73:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = reverse8_cuda(step3_shared[i]);     // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);      // rotate  bits by 5
        step3_shared[i] -= (step3_shared[i] ^ 97);            // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 74:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] *= step3_shared[i];                            // *
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
          step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 75:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] *= step3_shared[i];                            // *
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 76:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 77:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);            // rotate  bits by 3
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 78:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
        step3_shared[i] *= step3_shared[i];                            // *
        step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 79:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] += step3_shared[i];               // +
        step3_shared[i] *= step3_shared[i];               // *
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 80:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] += step3_shared[i];                            // +
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 81:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 82:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared]; // XOR
          // step3_shared[i] = ~step3_shared[i];        // binary NOT operator
          // step3_shared[i] = ~step3_shared[i];        // binary NOT operator
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 83:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);            // rotate  bits by 3
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 84:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] += step3_shared[i];                         // +
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 85:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 86:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
        step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 87:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];               // +
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] += step3_shared[i];               // +
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 88:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 89:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];               // +
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 90:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 6);  // rotate  bits by 5
        // step3_shared[i] = rotl_cuda(step3_shared[i], 1);    // rotate  bits by 1
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 91:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
          step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 92:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 93:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
          step3_shared[i] *= step3_shared[i];                            // *
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] += step3_shared[i];                            // +
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 94:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 95:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
        step3_shared[i] = rotl_cuda(step3_shared[i], 10); // rotate  bits by 5
                                                          // step3_shared[i] = rotl_cuda(step3_shared[i], 5); // rotate  bits by 5
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 96:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);     // rotate  bits by 2
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);     // rotate  bits by 2
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 97:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 98:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 99:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 100:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
        step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 101:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 102:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
        step3_shared[i] -= (step3_shared[i] ^ 97);       // XOR and -
        step3_shared[i] += step3_shared[i];              // +
        step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
                                                         // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 103:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
          step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 104:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]);     // reverse bits
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);      // rotate  bits by 5
        step3_shared[i] += step3_shared[i];                   // +
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 105:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 106:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
        step3_shared[i] *= step3_shared[i];               // *
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 107:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 6);            // rotate  bits by 5
                                                                    // step3_shared[i] = rotl_cuda(step3_shared[i], 1);             // rotate  bits by 1
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 108:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 109:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] *= step3_shared[i];                            // *
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 110:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 111:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] *= step3_shared[i];                         // *
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] *= step3_shared[i];                         // *
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 112:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
        step3_shared[i] = ~step3_shared[i];              // binary NOT operator
        step3_shared[i] = rotl_cuda(step3_shared[i], 5); // rotate  bits by 5
        step3_shared[i] -= (step3_shared[i] ^ 97);       // XOR and -
                                                         // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 113:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 6); // rotate  bits by 5
        // step3_shared[i] = rotl_cuda(step3_shared[i], 1);                           // rotate  bits by 1
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = ~step3_shared[i];                   // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 114:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
        step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 115:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 116:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 117:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 118:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 119:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
          step3_shared[i] = ~step3_shared[i];               // binary NOT operator
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 120:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
          step3_shared[i] *= step3_shared[i];               // *
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
          step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 121:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
        step3_shared[i] *= step3_shared[i];                         // *
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 122:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 123:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] = rotl_cuda(step3_shared[i], 6);               // rotate  bits by 3
                                                                         // step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 124:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
          step3_shared[i] = ~step3_shared[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 125:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 126:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 9); // rotate  bits by 3
        // step3_shared[i] = rotl_cuda(step3_shared[i], 1); // rotate  bits by 1
        // step3_shared[i] = rotl_cuda(step3_shared[i], 5); // rotate  bits by 5
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 127:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] *= step3_shared[i];                            // *
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 128:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 129:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 130:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 131:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] -= (step3_shared[i] ^ 97);            // XOR and -
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] *= step3_shared[i];                   // *
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 132:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 133:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 134:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 135:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 136:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
          step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 137:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
        step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 138:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared]; // XOR
          step3_shared[i] ^= step3_shared[pos2_shared]; // XOR
          step3_shared[i] += step3_shared[i];           // +
          step3_shared[i] -= (step3_shared[i] ^ 97);    // XOR and -
                                                        // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 139:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 8); // rotate  bits by 5
        // step3_shared[i] = rotl_cuda(step3_shared[i], 3);             // rotate  bits by 3
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 140:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 141:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
        step3_shared[i] -= (step3_shared[i] ^ 97);            // XOR and -
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] += step3_shared[i];                   // +
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 142:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 143:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 144:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
        step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 145:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 146:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 147:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] *= step3_shared[i];                         // *
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 148:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 149:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
          step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
          step3_shared[i] -= (step3_shared[i] ^ 97);        // XOR and -
          step3_shared[i] += step3_shared[i];               // +
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 150:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 151:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] *= step3_shared[i];                         // *
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 152:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 153:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 4); // rotate  bits by 1
                                                         // step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
                                                         // step3_shared[i] = ~step3_shared[i];     // binary NOT operator
                                                         // step3_shared[i] = ~step3_shared[i];     // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 154:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);      // rotate  bits by 5
          step3_shared[i] = ~step3_shared[i];                   // binary NOT operator
          step3_shared[i] ^= step3_shared[pos2_shared];         // XOR
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
                                                                // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 155:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] -= (step3_shared[i] ^ 97);            // XOR and -
          step3_shared[i] ^= step3_shared[pos2_shared];         // XOR
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
          step3_shared[i] ^= step3_shared[pos2_shared];         // XOR
                                                                // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 156:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = rotl_cuda(step3_shared[i], 4);            // rotate  bits by 3
                                                                    // step3_shared[i] = rotl_cuda(step3_shared[i], 1);    // rotate  bits by 1
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 157:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 158:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);      // rotate  bits by 3
        step3_shared[i] += step3_shared[i];                   // +
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 159:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 160:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 4);            // rotate  bits by 1
                                                                    // step3_shared[i] = rotl_cuda(step3_shared[i], 3);    // rotate  bits by 3
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 161:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 162:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] -= (step3_shared[i] ^ 97);        // XOR and -
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 163:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 164:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] *= step3_shared[i];                   // *
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] -= (step3_shared[i] ^ 97);            // XOR and -
        step3_shared[i] = ~step3_shared[i];                   // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 165:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
          step3_shared[i] += step3_shared[i];                         // +
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 166:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
        step3_shared[i] += step3_shared[i];               // +
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 167:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        // step3_shared[i] = ~step3_shared[i];        // binary NOT operator
        // step3_shared[i] = ~step3_shared[i];        // binary NOT operator
        step3_shared[i] *= step3_shared[i];                         // *
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 168:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 169:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 170:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] -= (step3_shared[i] ^ 97);        // XOR and -
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] -= (step3_shared[i] ^ 97);        // XOR and -
        step3_shared[i] *= step3_shared[i];               // *
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 171:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);      // rotate  bits by 3
        step3_shared[i] -= (step3_shared[i] ^ 97);            // XOR and -
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = reverse8_cuda(step3_shared[i]);     // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 172:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 173:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] *= step3_shared[i];                         // *
        step3_shared[i] += step3_shared[i];                         // +
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 174:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 175:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
        step3_shared[i] -= (step3_shared[i] ^ 97);       // XOR and -
        step3_shared[i] *= step3_shared[i];              // *
        step3_shared[i] = rotl_cuda(step3_shared[i], 5); // rotate  bits by 5
                                                         // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 176:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];    // XOR
          step3_shared[i] *= step3_shared[i];              // *
          step3_shared[i] ^= step3_shared[pos2_shared];    // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 177:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 178:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] += step3_shared[i];                            // +
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 179:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 180:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 181:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 182:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];    // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], 6); // rotate  bits by 1
          // step3_shared[i] = rotl_cuda(step3_shared[i], 5);         // rotate  bits by 5
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 183:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];        // +
        step3_shared[i] -= (step3_shared[i] ^ 97); // XOR and -
        step3_shared[i] -= (step3_shared[i] ^ 97); // XOR and -
        step3_shared[i] *= step3_shared[i];        // *
                                                   // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 184:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
          step3_shared[i] *= step3_shared[i];                         // *
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 185:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 186:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 187:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];    // XOR
          step3_shared[i] = ~step3_shared[i];              // binary NOT operator
          step3_shared[i] += step3_shared[i];              // +
          step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 188:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 189:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
          step3_shared[i] -= (step3_shared[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 190:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 191:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] += step3_shared[i];                            // +
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 192:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] *= step3_shared[i];                         // *
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 193:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 194:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 195:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);     // rotate  bits by 2
          step3_shared[i] ^= step3_shared[pos2_shared];         // XOR
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
                                                                // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 196:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);            // rotate  bits by 3
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 197:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] *= step3_shared[i];                            // *
        step3_shared[i] *= step3_shared[i];                            // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 198:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 199:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = ~step3_shared[i];           // binary NOT operator
          step3_shared[i] += step3_shared[i];           // +
          step3_shared[i] *= step3_shared[i];           // *
          step3_shared[i] ^= step3_shared[pos2_shared]; // XOR
                                                        // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 200:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 201:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 202:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 203:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);               // rotate  bits by 1
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 204:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 205:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] += step3_shared[i];                         // +
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 206:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
        step3_shared[i] = reverse8_cuda(step3_shared[i]);     // reverse bits
        step3_shared[i] = reverse8_cuda(step3_shared[i]);     // reverse bits
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 207:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 8); // rotate  bits by 5
        // step3_shared[i] = rotl_cuda(step3_shared[i], 3);                           // rotate  bits by 3
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 208:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);            // rotate  bits by 3
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 209:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);      // rotate  bits by 5
        step3_shared[i] = reverse8_cuda(step3_shared[i]);     // reverse bits
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] -= (step3_shared[i] ^ 97);            // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 210:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);               // rotate  bits by 5
        step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 211:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
        step3_shared[i] += step3_shared[i];                            // +
        step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 212:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 213:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);            // rotate  bits by 3
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 214:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
          step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 215:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] *= step3_shared[i];                            // *
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 216:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 217:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
        step3_shared[i] += step3_shared[i];               // +
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);  // rotate  bits by 1
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 218:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] -= (step3_shared[i] ^ 97);        // XOR and -
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 219:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 220:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 221:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
          step3_shared[i] = ~step3_shared[i];               // binary NOT operator
          step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 222:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] *= step3_shared[i];                         // *
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 223:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 224:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 4);  // rotate  bits by 1
        // step3_shared[i] = rotl_cuda(step3_shared[i], 3);             // rotate  bits by 3
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 225:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];                         // binary NOT operator
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);            // rotate  bits by 3
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 226:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
          step3_shared[i] -= (step3_shared[i] ^ 97);        // XOR and -
          step3_shared[i] *= step3_shared[i];               // *
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
                                                            // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 227:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
          step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 228:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];       // ones count bits
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 229:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);               // rotate  bits by 3
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);              // rotate  bits by 2
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 230:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] *= step3_shared[i];                            // *
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 231:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);            // rotate  bits by 3
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
          step3_shared[i] ^= step3_shared[pos2_shared];               // XOR
          step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
                                                                      // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 232:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4); // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 233:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);      // rotate  bits by 3
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 234:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] *= step3_shared[i];                            // *
          step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3);    // shift right
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 235:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] *= step3_shared[i];               // *
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 236:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= step3_shared[pos2_shared];                  // XOR
          step3_shared[i] += step3_shared[i];                            // +
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] -= (step3_shared[i] ^ 97);                     // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 237:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);            // rotate  bits by 3
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 238:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];              // +
        step3_shared[i] += step3_shared[i];              // +
        step3_shared[i] = rotl_cuda(step3_shared[i], 3); // rotate  bits by 3
        step3_shared[i] -= (step3_shared[i] ^ 97);       // XOR and -
                                                         // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 239:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 6); // rotate  bits by 5
          // step3_shared[i] = rotl_cuda(step3_shared[i], 1); // rotate  bits by 1
          step3_shared[i] *= step3_shared[i];                            // *
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 240:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = ~step3_shared[i];                            // binary NOT operator
          step3_shared[i] += step3_shared[i];                            // +
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3);    // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 241:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);     // rotate  bits by 4
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
          step3_shared[i] ^= step3_shared[pos2_shared];         // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
                                                                // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 242:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] += step3_shared[i];           // +
          step3_shared[i] += step3_shared[i];           // +
          step3_shared[i] -= (step3_shared[i] ^ 97);    // XOR and -
          step3_shared[i] ^= step3_shared[pos2_shared]; // XOR
                                                        // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 243:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);      // rotate  bits by 5
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);     // rotate  bits by 2
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);      // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 244:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] = reverse8_cuda(step3_shared[i]); // reverse bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 245:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] -= (step3_shared[i] ^ 97);                  // XOR and -
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);            // rotate  bits by 5
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 246:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                         // +
        step3_shared[i] = rotl_cuda(step3_shared[i], 1);            // rotate  bits by 1
        step3_shared[i] = step3_shared[i] >> (step3_shared[i] & 3); // shift right
        step3_shared[i] += step3_shared[i];                         // +
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 247:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);  // rotate  bits by 5
        step3_shared[i] = ~step3_shared[i];               // binary NOT operator
                                                          // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 248:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = ~step3_shared[i];                   // binary NOT operator
        step3_shared[i] -= (step3_shared[i] ^ 97);            // XOR and -
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 5);      // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 249:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]);              // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
        step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 250:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = step3_shared[i] & step3_shared[pos2_shared]; // AND
          step3_shared[i] = rotl_cuda(step3_shared[i], step3_shared[i]); // rotate  bits by random
          step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]];          // ones count bits
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);              // rotate  bits by 4
                                                                         // INSERT_RANDOM_CODE_END
        }
      }
      __syncwarp();
    }
    break;
    case 251:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] += step3_shared[i];                   // +
        step3_shared[i] ^= (byte)bitTable_d[step3_shared[i]]; // ones count bits
        step3_shared[i] = reverse8_cuda(step3_shared[i]);     // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);     // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 252:
    {
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] = reverse8_cuda(step3_shared[i]);           // reverse bits
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 4);           // rotate  bits by 4
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);           // rotate  bits by 2
        step3_shared[i] = step3_shared[i] << (step3_shared[i] & 3); // shift left
                                                                    // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    case 253:
    {
      if (index % 32 == 0)
      {
        for (int i = pos1_shared; i < pos2_shared; i++)
        {
          // INSERT_RANDOM_CODE_START
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
          step3_shared[i] ^= rotl_cuda(step3_shared[i], 2); // rotate  bits by 2
          step3_shared[i] ^= step3_shared[pos2_shared];     // XOR
          step3_shared[i] = rotl_cuda(step3_shared[i], 3);  // rotate  bits by 3
          // INSERT_RANDOM_CODE_END

          prev_lhash_shared = lhash_shared + prev_lhash_shared;
          lhash_shared = XXHash64_cuda::hash(&step3_shared[0], pos2_shared, 0); // more deviations
        }
      }
      __syncwarp();
    }
    break;
    case 254:
    case 255:
    {
      if (index % 32 == 0)
        rc4_setup(&key_shared, step3_shared, 256);
      __syncwarp();
      // step3_shared = highwayhash.Sum(step3_shared[:], step3_shared[:])
      int i = pos1_shared + threadIdx.x % 32;

      if (i < pos2_shared)
      {
        // INSERT_RANDOM_CODE_START
        step3_shared[i] ^= static_cast<uint8_t>(__popc((int)step3_shared[i])); // ones count bits
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);                       // rotate  bits by 3
        step3_shared[i] ^= rotl_cuda(step3_shared[i], 2);                      // rotate  bits by 2
        step3_shared[i] = rotl_cuda(step3_shared[i], 3);                       // rotate  bits by 3
                                                                               // INSERT_RANDOM_CODE_END
      }
      __syncwarp();
    }
    break;
    default:
      break;
    }
  }
  }
}
