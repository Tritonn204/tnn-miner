#include <hip/hip_runtime.h>

#include "powtest.h"
#include "pow_hip.h"
#include <sha256_hip.h>

#include <xxhash64_hip.h>
#include <fnv1a.h>
#include <fnv1a_hip.h>
#include <inttypes.h>
#include <iostream>

#include "xxhash64.h"
#include <highwayhash/sip_hash.h>
#include <openssl/rc4.h>
#include "sais.h"

#include <bitset>
#include <hex.h>
#include <lookup.h>

__host__ __device__ void rc4_setup(struct rc4_state *s, unsigned char *key, int length)
{
  int i, j, k, *m, a;

  s->x = 0;
  s->y = 0;
  m = s->m;

#pragma unroll 256
  for (i = 0; i < 256; i++)
  {
    m[i] = i;
  }

  j = k = 0;

#pragma unroll 256
  for (i = 0; i < 256; i++)
  {
    a = m[i];
    j = (unsigned char)(j + a + key[k]);
    m[i] = m[j];
    m[j] = a;
    if (++k >= length)
      k = 0;
  }
}

__host__ __device__ void rc4_crypt(struct rc4_state *s, unsigned char *data, int length)
{
  int i, x, y, *m, a, b;

  x = s->x;
  y = s->y;
  m = s->m;

#pragma unroll 256
  for (i = 0; i < 256; i++)
  {
    x = (unsigned char)(x + 1);
    a = m[x];
    y = (unsigned char)(y + a);
    m[x] = b = m[y];
    m[y] = a;
    data[i] ^= m[(unsigned char)(a + b)];
  }

  s->x = x;
  s->y = y;
}

#define MB_SIZE 48

using byte = unsigned char;

template <typename T>
__host__ __device__ void swap(T &a, T &b)
{
  T c(a);
  a = b;
  b = c;
}

__host__ __device__ byte rotl_hip(byte x, int n)
{
  return (x << (n % 8)) | (x >> (8 - (n % 8)));
}

void TestRace_hip()
{
  workerData_hip *worker_h = (workerData_hip *)malloc(sizeof(workerData_hip) * 512);
  workerData_hip *workers;
  hipMalloc((void **)&workers, sizeof(workerData_hip) * 512);
  hipMemcpy(workers, worker_h, sizeof(workerData_hip) * 512, hipMemcpyHostToDevice);

  std::string Input = "asdasn13e190d#v saf";
  byte *inputs = (byte *)malloc(Input.size() * 512);
  byte *inputs_d;

  hipMalloc((void **)&inputs_d, Input.size() * 512);

  for (int i = 0; i < 512; i++)
  {
    memcpy(&inputs[i * Input.size()], Input.c_str(), Input.size());
  }
  hipMemcpy(inputs_d, inputs, Input.size() * 512, hipMemcpyHostToDevice);

  byte outputs[32 * 512];
  byte *outputs_d;
  hipMalloc((void **)&outputs_d, 32 * 512);

  branchedSHATest_kernel<<<1, 1>>>(workers, inputs_d, outputs_d, Input.size(), 1);
  hipDeviceSynchronize();

  printf("\n\nReference above, parallel below\n\n\n");

  branchedSHATest_kernel<<<4, 128>>>(workers, inputs_d, outputs_d, Input.size(), 512);
  hipDeviceSynchronize();
}

void TestAstroBWTv3_hip()
{
  workerData_hip *worker_h = (workerData_hip *)malloc(sizeof(workerData_hip));

  void *hipStore;

  workerData_hip *worker;
  hipMalloc((void **)&worker, sizeof(workerData_hip));
  hipMemcpy(worker, worker_h, sizeof(workerData_hip), hipMemcpyHostToDevice);

  int i = 0;
  // SHA256 setup
  for (PowTest t : random_pow_tests)
  {
    // if (i > 0) break;
    byte *d_buf;
    hipMalloc((void **)&d_buf, t.in.size());
    hipMemset(d_buf, 0, t.in.size());
    hipMemcpy(d_buf, t.in.c_str(), (int)t.in.size(), hipMemcpyHostToDevice);
    byte *res = (byte *)malloc(32);
    byte *d_res;
    hipMalloc((void **)&d_res, 32);

    ASTRO_HIP(d_buf, d_res, worker, (int)t.in.size(), 1, 0, 0);

    // ASTRO_1_kernel<<<1, 1>>>(d_buf, d_res, worker, (int)t.in.size(), 1, 0, 0);
    // cudaDeviceSynchronize();

    // // // cudaMemcpy(worker_h, worker, sizeof(workerData_cuda), cudaMemcpyDeviceToHost);
    // // // libcubwt_sa(cudaStore, worker_h->sData, worker_h->sa, worker_h->data_len);
    // // // cudaDeviceSynchronize();
    // // // cudaMemcpy(worker, worker_h, sizeof(workerData_cuda), cudaMemcpyHostToDevice);

    // ASTRO_3_kernel<<<1, 1>>>(d_buf, d_res, worker, (int)t.in.size(), 1, 0, 0);
    // cudaDeviceSynchronize();

    hipMemcpy(res, d_res, 32, hipMemcpyDeviceToHost);

    std::string s = hexStr(res, 32);

    if (s.c_str() != t.out)
    {
      printf("HIP: FAIL. Pow function: pow(%s) = %s want %s\n", t.in.c_str(), s.c_str(), t.out.c_str());
    }
    else
    {
      printf("HIP: SUCCESS! pow(%s) = %s want %s\n", t.in.c_str(), s.c_str(), t.out.c_str());
    }
    hipFree(d_res);
    hipFree(d_buf);
    free(res);
    i++;
  }
}

__global__ void ASTRO_INIT_kernel(int device, byte *work, int batchSize, int offset, int nonce)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int j = index; j < batchSize; j += stride)
  {
    int i = nonce + offset + j;
    memcpy(&work[offset * MB_SIZE + MB_SIZE * j + MB_SIZE - 5], (byte *)&i, sizeof(i));

    // swap endianness
    swap(
        work[offset * MB_SIZE + MB_SIZE * j + MB_SIZE - 5],
        work[offset * MB_SIZE + MB_SIZE * j + MB_SIZE - 2]);
    swap(
        work[offset * MB_SIZE + MB_SIZE * j + MB_SIZE - 4],
        work[offset * MB_SIZE + MB_SIZE * j + MB_SIZE - 3]);
    // std::cout << i << std::endl;
    // ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Initializing HSA stack.");
    // if (j == 0) printf("\nWork with nonce: %s\n", hexStr_hip(&work[offset*MB_SIZE + MB_SIZE * j], MB_SIZE));
  }
}

__host__ void ASTRO_INIT_HIP(int device, byte *work, int batchSize, int offset, int nonce)
{
  int B = (batchSize + 1024 - 1) / 1024;
  int T = 1024;
  ASTRO_INIT_kernel<<<B, T>>>(device, work, batchSize, offset, nonce);
}

__global__ void ASTRO_1_kernel(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int d, int offset)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < batchSize; i += stride)
  {
    AstroBWTv3_hip_p1(&work[offset * inputLen + i * inputLen], inputLen, &output[offset * 32 + i * 32], workers[offset + i]);
    AstroBWTv3_hip_p2(&work[offset * inputLen + i * inputLen], inputLen, &output[offset * 32 + i * 32], workers[offset + i]);
  }
}

__global__ void branchedSHATest_kernel(workerData_hip *w, byte *input, byte *output, int len, int count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < count)
  {
    memset(w[index].step_3, 0, 256);

    SHA256_hip(w[index].sha256, &input[index * len], &output[index * 32], len);
    if (s20_crypt(w[index].sha_key, s20_keylen_t::S20_KEYLEN_256, 0, w[index].step_3, 256) != S20_SUCCESS)
      printf("salsa20 failure\n");

    rc4_setup(&w[index].key, w[index].step_3, 256);
    rc4_crypt(&w[index].key, w[index].step_3, 256);

    // printf("worker.step_3 post rc4: ");
    // printf(hexStr_cuda(worker.step_3, 256));
    // printf("\n\n\n");

    // printf("worker.step_3 post rc4: %s\n", hexStr_cuda(w[index].step_3, 256));

    // printf("lhash pre fnv: %" PRIx64 "\n", w[index].lhash);
    hash_64_fnv1a_hip(w[index].step_3, 256, &w[index].lhash);
    w[index].prev_lhash = w[index].lhash;

    // printf("lhash result: %" PRIx64 "\n", w[index].lhash);
  }
}

__host__ void branchedSHATest(workerData_hip *w, byte *input, byte *output, int len, int count)
{
  int B = (count + 128 - 1) / 128;
  int T = 128;

  branchedSHATest_kernel<<<B, T>>>(w, input, output, len, count);
}

__global__ void ASTRO_hybrid_kernel(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int d, int offset)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < batchSize; i += stride)
  {
    // AstroBWTv3_cuda_p1(&work[offset * MB_SIZE + i * MB_SIZE], inputLen, &output[offset * 32 + i * 32], workers[offset + i]);
    // AstroBWTv3_cuda_p2(&work[offset * MB_SIZE + i * MB_SIZE], inputLen, &output[offset * 32 + i * 32], workers[offset + i]);
    // divsufsort_cuda(workers[offset + i].sData, workers[offset + i].sa, workers[offset + i].data_len);
    // AstroBWTv3_cuda_p3(&work[offset * MB_SIZE + i * MB_SIZE], inputLen, &output[offset * 32 + i * 32], workers[offset + i]);
  }
}

// __global__ void ASTRO_DSS_kernel_1(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int d, int offset)
// {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;

//   for (int i = index; i < batchSize; i += stride)
//   {
//     if (workers[offset + i].data_len == 1)
//     {
//       workers[offset + i].sa[0] = 0;
//     }
//     else if (workers[offset + i].data_len == 2)
//     {
//       workers[offset + i].M = (workers[offset + i].sData[0] < workers[offset + i].sData[1]);
//       workers[offset + i].sa[workers[offset + i].M ^ 1] = 0, workers[offset + i].sa[workers[offset + i].M] = 1;
//     }
//     // workers[offset + i].data_len = 70000;
//     // A[i] = divsufsort_hip(workers[offset + i].sData, workers[offset + i].sa, workers[offset + i].data_len);
//     // AstroBWTv3_hip_p3(&work[offset * inputLen + i * inputLen], inputLen, &output[offset * 32 + i * 32], workers[offset + i]);
//     else
//     {
//       memset(workers[offset + i].bucket_A, 0, 255);
//       memset(workers[offset + i].bucket_B, 0, 255);
//       workers[offset + i].M = sort_typeBstar_hip(
//           workers[offset + i].sData, workers[offset + i].sa,
//           workers[offset + i].bucket_A, workers[offset + i].bucket_B, workers[offset + i].data_len);
//     }
//   }
// }

// __global__ void ASTRO_DSS_kernel_2(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int d, int offset)
// {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;

//   for (int i = index; i < batchSize; i += stride)
//   {
//     // workers[offset + i].data_len = 70000;
//     // A[i] = divsufsort_hip(workers[offset + i].sData, workers[offset + i].sa, workers[offset + i].data_len);
//     // AstroBWTv3_hip_p3(&work[offset * inputLen + i * inputLen], inputLen, &output[offset * 32 + i * 32], workers[offset + i]);
//     if (workers[offset + i].data_len > 2)
//     {
//       construct_SA_hip(
//           workers[offset + i].sData, workers[offset + i].sa,
//           workers[offset + i].bucket_A, workers[offset + i].bucket_B,
//           workers[offset + i].data_len, workers[offset + i].M);
//     }
//   }
// }

__global__ void ASTRO_3_kernel(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int d, int offset)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < batchSize; i += stride)
  {
    // workers[offset + i].data_len = 70000;
    // divsufsort_hip(workers[offset + i].sData, workers[offset + i].sa, workers[offset + i].data_len);
    // sais_hip(workers[offset + i].sData, workers[offset + i].sa, workers[offset + i].data_len);
    AstroBWTv3_hip_p3(&work[offset * inputLen + i * inputLen], inputLen, &output[offset * 32 + i * 32], workers[offset + i]);
  }
}

__global__ void ASTRO_SAIS_kernel(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int d, int offset)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < batchSize; i += stride)
  {
    sais_hip(workers[offset + i].sData, workers[offset + i].sa, workers[offset + i].data_len);
  }
}

void ASTRO_HIP(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int d, int offset)
{

  int kernelBatchSize = 8912;

  for (int i = 0; i < batchSize; i += kernelBatchSize)
  {
    int runBufferSize = (batchSize - i > kernelBatchSize) ? kernelBatchSize : batchSize - i;
    int B = (runBufferSize + 512 - 1) / 512;
    int T = 512;

    ASTRO_1_kernel<<<B, T>>>(work, output, workers, inputLen, runBufferSize, d, offset + i);
    gpuErrchk(hipDeviceSynchronize());
  }

  kernelBatchSize = 4096;

  // __nanosleep(1000000);

  // for (int i = 0; i < batchSize; i += kernelBatchSize)
  // {
  //   int runBufferSize = (batchSize - i > kernelBatchSize) ? kernelBatchSize : batchSize - i;
  //   int B = (runBufferSize + 128 - 1) / 128;
  //   int T = 128;

  //   ASTRO_SAIS_kernel<<<B, T>>>(work, output, workers, inputLen, runBufferSize, d, offset + i);
  //   gpuErrchk(hipDeviceSynchronize());
  // }

  kernelBatchSize = 8912;

  // __nanosleep(1000000);

  for (int i = 0; i < batchSize; i += kernelBatchSize)
  {
    int runBufferSize = (batchSize - i > kernelBatchSize) ? kernelBatchSize : batchSize - i;
    int B = (runBufferSize + 512 - 1) / 512;
    int T = 512;

    ASTRO_3_kernel<<<B, T>>>(work, output, workers, inputLen, runBufferSize, d, offset + i);
    gpuErrchk(hipDeviceSynchronize());
  }
}

void ASTRO_1(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int d, int offset)
{
  int B = (batchSize + 256 - 1) / 256;
  int T = 256;
  ASTRO_1_kernel<<<B, T>>>(work, output, workers, inputLen, batchSize, d, offset);
}

void ASTRO_2(void **hipStore, workerData_hip *workers, int batchSize)
{
  for (int i = 0; i < batchSize; i++)
  {
    printf("made it into\n");
    // divsufsort_cuda(workers[i].sData, workers[i].sa, workers[i].data_len);
  }
}

void ASTRO_3(byte *work, byte *output, workerData_hip *workers, int inputLen, int batchSize, int d, int offset)
{
  int B = (batchSize + 256 - 1) / 256;
  int T = 256;

  // ASTRO_DSS_kernel<<<B, T>>>(work, output, workers, inputLen, batchSize, d, offset);
  // gpuErrchk(hipDeviceSynchronize());
  // ASTRO_3_kernel<<<B, T>>>(work, output, workers, inputLen, batchSize, d, offset);
  // gpuErrchk(hipDeviceSynchronize());
}

__device__ void AstroBWTv3_hip_p1(unsigned char *input, int inputLen, unsigned char *outputHash, workerData_hip &worker)
{
  memset(worker.step_3, 0, 256);

  SHA256_hip(worker.sha256, input, worker.sha_key, inputLen);

  if (s20_crypt(worker.sha_key, s20_keylen_t::S20_KEYLEN_256, 0, worker.step_3, 256) != S20_SUCCESS)
  {
    // do nothing
  }

  rc4_setup(&worker.key, worker.step_3, 256);
  rc4_crypt(&worker.key, worker.step_3, 256);

  // printf("worker.step_3 post rc4: ");
  // printf(hexStr_cuda(worker.step_3, 256));
  // printf("\n\n\n");

  // // std::cout << "worker.step_3 post rc4: " << hexStr(worker.step_3, 256) << std::endl;

  hash_64_fnv1a_hip(worker.step_3, 256, &worker.lhash);
  worker.prev_lhash = worker.lhash;

  worker.tries = 0;

  // printf(hexStr_cuda(worker.step_3, 256));
  // printf("\n\n");
}

__device__ void AstroBWTv3_hip_p2(unsigned char *input, int inputLen, unsigned char *outputHash, workerData_hip &worker)
{
  while (true)
  {
    worker.tries++;
    worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries;
    // printf("%d worker.random_switcher %d %08jx\n", worker.tries, worker.random_switcher, worker.random_switcher);

    worker.op = static_cast<byte>(worker.random_switcher);

    worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
    worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

    if (worker.pos1 > worker.pos2)
    {
      swap(worker.pos1, worker.pos2);
    }

    if (worker.pos2 - worker.pos1 > 32)
    {
      worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
    }

    branchCompute(worker);

    worker.A = (worker.step_3[worker.pos1] - worker.step_3[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    if (worker.A < 0x10)
    { // 6.25 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64_cuda::hash(&worker.step_3, worker.pos2, 0);
      // printf("new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x20)
    { // 12.5 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      hash_64_fnv1a_hip(worker.step_3, worker.pos2, &worker.lhash);
      // printf("new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x30)
    { // 18.75 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;

      __align__(16)
          const uint64_t key2[2] = {worker.tries, worker.prev_lhash};
      siphash_cuda(worker.step_3, worker.pos2, key2, (uint8_t *)&worker.lhash, 8);
      // printf("new worker.lhash: %" PRIx64 "\n", worker.lhash);
    }

    if (worker.A <= 0x40)
    { // 25% probablility
      rc4_crypt(&worker.key, worker.step_3, 256);
    }

    worker.step_3[255] = worker.step_3[255] ^ worker.step_3[worker.pos1] ^ worker.step_3[worker.pos2];

    memcpy(&worker.sData[(worker.tries - 1) * 256], worker.step_3, 256);
    // std::copy(worker.step_3, worker.step_3 + 256, &worker.sData[(worker.tries - 1) * 256]);

    // copy_kernel(&worker->data.data()[(worker.tries - 1) * 256], worker.step_3, 256);

    // std::cout << hexStr(worker.step_3, 256) << std::endl;

    if (worker.tries > 260 + 16 || (worker.step_3[255] >= 0xf0 && worker.tries > 260))
    {
      break;
    }
  }

  worker.data_len = static_cast<uint32_t>((worker.tries - 4) * 256 + (((static_cast<uint64_t>(worker.step_3[253]) << 8) | static_cast<uint64_t>(worker.step_3[254])) & 0x3ff));
}

__device__ void AstroBWTv3_hip_p3(unsigned char *input, int inputLen, unsigned char *outputHash, workerData_hip &worker)
{
  byte *B = reinterpret_cast<byte *>(worker.sa);
  SHA256_hip(worker.sha256, B, outputHash, worker.data_len * 4);
  // worker.sHash = nHash;
}

__global__ void AstroBWTv3_hip(unsigned char *input, int inputLen, unsigned char *outputHash, workerData_hip &worker)
{
  memset(worker.step_3, 0, 256);

  SHA256_hip(worker.sha256, input, worker.sha_key, inputLen);

  if (s20_crypt(worker.sha_key, s20_keylen_t::S20_KEYLEN_256, 0, worker.step_3, 256) != S20_SUCCESS)
    printf("salsa20 failure\n");

  // // std::cout << "worker.step_3 post XOR: " << hexStr(worker.step_3, 256) << std::endl;

  rc4_setup(&worker.key, worker.step_3, 256);
  rc4_crypt(&worker.key, worker.step_3, 256);

  // printf("worker.step_3 post rc4: ");
  // printf(hexStr_cuda(worker.step_3, 256));
  // printf("\n\n\n");

  // // std::cout << "worker.step_3 post rc4: " << hexStr(worker.step_3, 256) << std::endl;

  hash_64_fnv1a_hip(worker.step_3, 256, &worker.lhash);
  worker.prev_lhash = worker.lhash;

  worker.tries = 0;
}

__device__ void branchCompute(workerData_hip &worker)
{

  switch (worker.op)
  {
  case 0:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
      worker.step_3[i] *= worker.step_3[i];                            // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      // INSERT_RANDOM_CODE_END
      worker.t1 = worker.step_3[worker.pos1];
      worker.t2 = worker.step_3[worker.pos2];
      worker.step_3[worker.pos1] = reverse8_hip(worker.t2);
      worker.step_3[worker.pos2] = reverse8_hip(worker.t1);
    }
    break;
  case 1:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] += worker.step_3[i];                             // +
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 2:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 3:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 4:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 5:
  {
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {

      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right

      // INSERT_RANDOM_CODE_END
    }
  }
  break;
  case 6:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -

      // INSERT_RANDOM_CODE_END
    }
    break;
  case 7:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                            // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 8:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], 10); // rotate  bits by 5
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);// rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 9:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 10:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
      worker.step_3[i] *= worker.step_3[i];             // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
      worker.step_3[i] *= worker.step_3[i];             // *
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 11:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 6); // rotate  bits by 1
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);            // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 12:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 13:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 14:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 15:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 16:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 17:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
      worker.step_3[i] *= worker.step_3[i];             // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
      worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 18:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 9);  // rotate  bits by 3
                                                         // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);             // rotate  bits by 1
                                                         // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);         // rotate  bits by 5
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 19:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] += worker.step_3[i];                          // +
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 20:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 21:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
      worker.step_3[i] += worker.step_3[i];                             // +
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 22:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 23:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 4); // rotate  bits by 3
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);                           // rotate  bits by 1
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 24:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 25:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 26:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];                   // *
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] += worker.step_3[i];                   // +
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);      // reverse bits
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 27:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 28:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 29:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] += worker.step_3[i];                          // +
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 30:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 31:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] *= worker.step_3[i];                          // *
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 32:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 33:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
      worker.step_3[i] *= worker.step_3[i];                            // *
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 34:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 35:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];             // +
      worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1); // rotate  bits by 1
      worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 36:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);      // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 37:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
      worker.step_3[i] *= worker.step_3[i];                            // *
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 38:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 39:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 40:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 41:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
      worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 42:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 4); // rotate  bits by 1
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 43:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] += worker.step_3[i];                             // +
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 44:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 45:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 10); // rotate  bits by 5
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);                       // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 46:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] += worker.step_3[i];                   // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);       // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 47:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 48:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
      // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 49:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] += worker.step_3[i];                   // +
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);      // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 50:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
      worker.step_3[i] += worker.step_3[i];              // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 51:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 52:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 53:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                   // +
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 54:

#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
                                                         // worker.step_3[i] = ~worker.step_3[i];    // binary NOT operator
                                                         // worker.step_3[i] = ~worker.step_3[i];    // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
    }

    break;
  case 55:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 56:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 57:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 8);                // rotate  bits by 5
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 58:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] += worker.step_3[i];                             // +
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 59:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {

      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
      worker.step_3[i] *= worker.step_3[i];                            // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 60:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
      worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
      worker.step_3[i] *= worker.step_3[i];             // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 61:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], 8);              // rotate  bits by 3
                                                                     // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);// rotate  bits by 5
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 62:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
      worker.step_3[i] += worker.step_3[i];                             // +
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 63:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);       // rotate  bits by 5
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] -= (worker.step_3[i] ^ 97);            // XOR and -
      worker.step_3[i] += worker.step_3[i];                   // +
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 64:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] *= worker.step_3[i];              // *
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 65:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 8); // rotate  bits by 5
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);             // rotate  bits by 3
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] *= worker.step_3[i];              // *
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 66:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 67:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);      // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);       // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 68:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 69:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 70:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 71:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 72:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 73:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);      // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);       // rotate  bits by 5
      worker.step_3[i] -= (worker.step_3[i] ^ 97);            // XOR and -
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 74:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];                             // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 75:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];                             // *
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 76:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 77:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 78:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
      worker.step_3[i] *= worker.step_3[i];                            // *
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 79:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] += worker.step_3[i];              // +
      worker.step_3[i] *= worker.step_3[i];              // *
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 80:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] += worker.step_3[i];                             // +
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 81:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 82:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
      // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
      // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 83:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 84:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] += worker.step_3[i];                          // +
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 85:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 86:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 87:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];              // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] += worker.step_3[i];              // +
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 88:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 89:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];              // +
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 90:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 6);  // rotate  bits by 5
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);    // rotate  bits by 1
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 91:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 92:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 93:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
      worker.step_3[i] *= worker.step_3[i];                             // *
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] += worker.step_3[i];                             // +
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 94:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 95:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], 10); // rotate  bits by 5
                                                         // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5); // rotate  bits by 5
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 96:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);      // rotate  bits by 2
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);      // rotate  bits by 2
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 97:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 98:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 99:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 100:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 101:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 102:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
      worker.step_3[i] -= (worker.step_3[i] ^ 97);      // XOR and -
      worker.step_3[i] += worker.step_3[i];             // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 103:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 104:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);      // reverse bits
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);       // rotate  bits by 5
      worker.step_3[i] += worker.step_3[i];                   // +
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 105:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 106:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
      worker.step_3[i] *= worker.step_3[i];              // *
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 107:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 6);              // rotate  bits by 5
                                                                     // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);             // rotate  bits by 1
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 108:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 109:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];                            // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 110:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 111:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 112:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
      worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
      worker.step_3[i] -= (worker.step_3[i] ^ 97);      // XOR and -
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 113:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 6); // rotate  bits by 5
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);                           // rotate  bits by 1
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = ~worker.step_3[i];                   // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 114:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 115:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 116:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 117:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 118:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 119:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 120:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 121:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] *= worker.step_3[i];                          // *
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 122:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 123:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], 6);                 // rotate  bits by 3
                                                                        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3); // rotate  bits by 3
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 124:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 125:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 126:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 9); // rotate  bits by 3
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1); // rotate  bits by 1
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5); // rotate  bits by 5
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 127:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] *= worker.step_3[i];                             // *
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 128:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 129:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 130:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 131:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] -= (worker.step_3[i] ^ 97);            // XOR and -
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] *= worker.step_3[i];                   // *
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 132:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 133:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 134:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 135:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 136:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 137:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 138:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
      worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
      worker.step_3[i] += worker.step_3[i];           // +
      worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
                                                      // INSERT_RANDOM_CODE_END
    }
    break;
  case 139:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 8); // rotate  bits by 5
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);             // rotate  bits by 3
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 140:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 141:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
      worker.step_3[i] -= (worker.step_3[i] ^ 97);            // XOR and -
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] += worker.step_3[i];                   // +
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 142:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 143:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 144:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 145:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 146:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 147:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] *= worker.step_3[i];                          // *
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 148:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 149:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
      worker.step_3[i] += worker.step_3[i];              // +
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 150:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 151:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 152:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 153:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 4); // rotate  bits by 1
                                                        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3); // rotate  bits by 3
                                                        // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
                                                        // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 154:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);       // rotate  bits by 5
      worker.step_3[i] = ~worker.step_3[i];                   // binary NOT operator
      worker.step_3[i] ^= worker.step_3[worker.pos2];         // XOR
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 155:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] -= (worker.step_3[i] ^ 97);            // XOR and -
      worker.step_3[i] ^= worker.step_3[worker.pos2];         // XOR
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] ^= worker.step_3[worker.pos2];         // XOR
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 156:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = rotl_hip(worker.step_3[i], 4);              // rotate  bits by 3
                                                                     // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);    // rotate  bits by 1
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 157:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 158:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);       // rotate  bits by 3
      worker.step_3[i] += worker.step_3[i];                   // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 159:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 160:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 4);              // rotate  bits by 1
                                                                     // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);    // rotate  bits by 3
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 161:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 162:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 163:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 164:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];                   // *
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] -= (worker.step_3[i] ^ 97);            // XOR and -
      worker.step_3[i] = ~worker.step_3[i];                   // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 165:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] += worker.step_3[i];                          // +
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 166:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
      worker.step_3[i] += worker.step_3[i];              // +
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 167:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
      // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 168:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 169:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 170:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
      worker.step_3[i] *= worker.step_3[i];              // *
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 171:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);       // rotate  bits by 3
      worker.step_3[i] -= (worker.step_3[i] ^ 97);            // XOR and -
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);      // reverse bits
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 172:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 173:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] += worker.step_3[i];                          // +
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 174:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 175:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
      worker.step_3[i] -= (worker.step_3[i] ^ 97);      // XOR and -
      worker.step_3[i] *= worker.step_3[i];             // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 176:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
      worker.step_3[i] *= worker.step_3[i];             // *
      worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 177:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 178:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] += worker.step_3[i];                             // +
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 179:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 180:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 181:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 182:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], 6); // rotate  bits by 1
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);         // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 183:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];        // +
      worker.step_3[i] -= (worker.step_3[i] ^ 97); // XOR and -
      worker.step_3[i] -= (worker.step_3[i] ^ 97); // XOR and -
      worker.step_3[i] *= worker.step_3[i];        // *
                                                   // INSERT_RANDOM_CODE_END
    }
    break;
  case 184:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] *= worker.step_3[i];                          // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 185:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 186:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 187:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
      worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
      worker.step_3[i] += worker.step_3[i];             // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 188:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 189:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
      worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 190:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 191:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                            // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 192:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] *= worker.step_3[i];                          // *
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 193:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 194:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 195:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);      // rotate  bits by 2
      worker.step_3[i] ^= worker.step_3[worker.pos2];         // XOR
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 196:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 197:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] *= worker.step_3[i];                            // *
      worker.step_3[i] *= worker.step_3[i];                            // *
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 198:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 199:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];           // binary NOT operator
      worker.step_3[i] += worker.step_3[i];           // +
      worker.step_3[i] *= worker.step_3[i];           // *
      worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                      // INSERT_RANDOM_CODE_END
    }
    break;
  case 200:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 201:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 202:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 203:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 204:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 205:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] += worker.step_3[i];                          // +
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 206:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);      // reverse bits
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);      // reverse bits
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 207:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 8); // rotate  bits by 5
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);                           // rotate  bits by 3
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 208:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 209:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);       // rotate  bits by 5
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);      // reverse bits
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] -= (worker.step_3[i] ^ 97);            // XOR and -
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 210:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
      worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 211:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
      worker.step_3[i] += worker.step_3[i];                            // +
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 212:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 213:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 214:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 215:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] *= worker.step_3[i];                             // *
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 216:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 217:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
      worker.step_3[i] += worker.step_3[i];              // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 218:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 219:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 220:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 221:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 222:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] *= worker.step_3[i];                          // *
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 223:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 224:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 4);  // rotate  bits by 1
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);             // rotate  bits by 3
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 225:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 226:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 227:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 228:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];        // ones count bits
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 229:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 230:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];                             // *
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 231:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 232:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 233:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);       // rotate  bits by 3
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 234:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] *= worker.step_3[i];                             // *
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 235:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] *= worker.step_3[i];              // *
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 236:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
      worker.step_3[i] += worker.step_3[i];                             // +
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 237:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 238:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];             // +
      worker.step_3[i] += worker.step_3[i];             // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
      worker.step_3[i] -= (worker.step_3[i] ^ 97);      // XOR and -
                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 239:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 6); // rotate  bits by 5
      // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1); // rotate  bits by 1
      worker.step_3[i] *= worker.step_3[i];                             // *
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 240:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
      worker.step_3[i] += worker.step_3[i];                             // +
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 241:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);      // rotate  bits by 4
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] ^= worker.step_3[worker.pos2];         // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 242:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];           // +
      worker.step_3[i] += worker.step_3[i];           // +
      worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
      worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                      // INSERT_RANDOM_CODE_END
    }
    break;
  case 243:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);       // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);      // rotate  bits by 2
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);       // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 244:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 245:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 246:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                          // +
      worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
      worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
      worker.step_3[i] += worker.step_3[i];                          // +
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 247:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
      worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                         // INSERT_RANDOM_CODE_END
    }
    break;
  case 248:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = ~worker.step_3[i];                   // binary NOT operator
      worker.step_3[i] -= (worker.step_3[i] ^ 97);            // XOR and -
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 5);       // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 249:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                       // INSERT_RANDOM_CODE_END
    }
    break;
  case 250:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
      worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]];           // ones count bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
                                                                        // INSERT_RANDOM_CODE_END
    }
    break;
  case 251:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] += worker.step_3[i];                   // +
      worker.step_3[i] ^= (byte)bitTable_d[worker.step_3[i]]; // ones count bits
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);      // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);      // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
    }
    break;
  case 252:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
      worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                     // INSERT_RANDOM_CODE_END
    }
    break;
  case 253:
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
      worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
      // INSERT_RANDOM_CODE_END

      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64_cuda::hash(&worker.step_3, worker.pos2, 0); // more deviations
    }
    break;
  case 254:
  case 255:
    rc4_setup(&worker.key, worker.step_3, 256);
// worker.step_3 = highwayhash.Sum(worker.step_3[:], worker.step_3[:])
#pragma unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.step_3[i] ^= static_cast<uint8_t>(__popc((int)worker.step_3[i])); // ones count bits
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                        // rotate  bits by 3
      worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                       // rotate  bits by 2
      worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                        // rotate  bits by 3
                                                                               // INSERT_RANDOM_CODE_END
    }
    break;
  default:
    break;
  }
}

__host__ void branchComputeCPU_hip(workerData_hip &worker)
{
  while (true)
  {
    worker.tries++;
    worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries;
    // printf("%d worker.random_switcher %d %08jx\n", worker.tries, worker.random_switcher, worker.random_switcher);

    worker.op = static_cast<byte>(worker.random_switcher);

    worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
    worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

    if (worker.pos1 > worker.pos2)
    {
      std::swap(worker.pos1, worker.pos2);
    }

    if (worker.pos2 - worker.pos1 > 32)
    {
      worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
    }

    // fmt::printf("op: %d, ", worker.op);
    // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

    switch (worker.op)
    {
    case 0:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] *= worker.step_3[i];                            // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        // INSERT_RANDOM_CODE_END
        worker.t1 = worker.step_3[worker.pos1];
        worker.t2 = worker.step_3[worker.pos2];
        worker.step_3[worker.pos1] = reverse8_hip(worker.t2);
        worker.step_3[worker.pos2] = reverse8_hip(worker.t1);
      }
      break;
    case 1:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 2:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 3:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 4:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 5:
    {
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {

        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right

        // INSERT_RANDOM_CODE_END
      }
    }
    break;
    case 6:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -

        // INSERT_RANDOM_CODE_END
      }
      break;
    case 7:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                            // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 8:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], 10); // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);// rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 9:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 10:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];             // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] *= worker.step_3[i];             // *
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 11:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 6); // rotate  bits by 1
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);            // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 12:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 13:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 14:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 15:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 16:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 17:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
        worker.step_3[i] *= worker.step_3[i];             // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 18:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 9);  // rotate  bits by 3
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);             // rotate  bits by 1
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);         // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 19:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 20:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 21:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 22:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 23:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 4); // rotate  bits by 3
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);                           // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 24:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 25:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 26:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                 // *
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);    // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 27:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 28:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 29:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 30:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 31:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 32:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 33:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
        worker.step_3[i] *= worker.step_3[i];                            // *
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 34:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 35:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];             // +
        worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1); // rotate  bits by 1
        worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 36:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);    // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 37:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
        worker.step_3[i] *= worker.step_3[i];                            // *
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 38:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 39:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 40:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 41:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 42:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 4); // rotate  bits by 1
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 43:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 44:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 45:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 10); // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);                       // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 46:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);     // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 47:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 48:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 49:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);    // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 50:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 51:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 52:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 53:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 54:

#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        // worker.step_3[i] = ~worker.step_3[i];    // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];    // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }

      break;
    case 55:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 56:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 57:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 8);                // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 58:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 59:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] *= worker.step_3[i];                            // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 60:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
        worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];             // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 61:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], 8);              // rotate  bits by 3
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);// rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 62:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 63:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);     // rotate  bits by 5
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] += worker.step_3[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 64:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] *= worker.step_3[i];              // *
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 65:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 8); // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];              // *
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 66:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 67:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);    // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);     // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 68:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 69:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 70:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 71:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 72:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 73:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);    // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);     // rotate  bits by 5
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 74:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 75:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 76:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 77:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 78:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
        worker.step_3[i] *= worker.step_3[i];                            // *
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 79:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] *= worker.step_3[i];              // *
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 80:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 81:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 82:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
        // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 83:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 84:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 85:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 86:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 87:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] += worker.step_3[i];              // +
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 88:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 89:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 90:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 6);  // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 91:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 92:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 93:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 94:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 95:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], 10); // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5); // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 96:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);    // rotate  bits by 2
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);    // rotate  bits by 2
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 97:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 98:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 99:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 100:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 101:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 102:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);      // XOR and -
        worker.step_3[i] += worker.step_3[i];             // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 103:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 104:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);    // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);     // rotate  bits by 5
        worker.step_3[i] += worker.step_3[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 105:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 106:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] *= worker.step_3[i];              // *
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 107:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 6);              // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);             // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 108:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 109:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                            // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 110:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 111:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 112:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] -= (worker.step_3[i] ^ 97);      // XOR and -
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 113:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 6); // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);                           // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 114:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 115:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 116:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 117:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 118:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 119:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 120:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 121:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 122:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 123:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], 6);                 // rotate  bits by 3
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3); // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 124:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 125:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 126:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 9); // rotate  bits by 3
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1); // rotate  bits by 1
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 127:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 128:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 129:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 130:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 131:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] *= worker.step_3[i];                 // *
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 132:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 133:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 134:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 135:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 136:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 137:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 138:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
        worker.step_3[i] += worker.step_3[i];           // +
        worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 139:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 8); // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 140:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 141:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 142:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 143:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 144:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 145:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 146:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 147:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 148:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 149:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
        worker.step_3[i] += worker.step_3[i];              // +
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 150:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 151:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 152:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 153:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 4); // rotate  bits by 1
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3); // rotate  bits by 3
        // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 154:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);     // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 155:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 156:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = rotl_hip(worker.step_3[i], 4);              // rotate  bits by 3
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1);    // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 157:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);   // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                // rotate  bits by 1
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 158:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);     // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 159:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 160:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 4);              // rotate  bits by 1
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);    // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 161:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 162:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 163:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 164:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                 // *
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 165:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 166:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 167:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 168:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 169:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 170:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
        worker.step_3[i] *= worker.step_3[i];              // *
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 171:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);     // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);    // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 172:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 173:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 174:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 175:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);      // XOR and -
        worker.step_3[i] *= worker.step_3[i];             // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 176:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
        worker.step_3[i] *= worker.step_3[i];             // *
        worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5); // rotate  bits by 5
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 177:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 178:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 179:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 180:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 181:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 182:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], 6); // rotate  bits by 1
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 5);         // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 183:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];        // +
        worker.step_3[i] -= (worker.step_3[i] ^ 97); // XOR and -
        worker.step_3[i] -= (worker.step_3[i] ^ 97); // XOR and -
        worker.step_3[i] *= worker.step_3[i];        // *
                                                     // INSERT_RANDOM_CODE_END
      }
      break;
    case 184:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 185:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 186:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 187:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];   // XOR
        worker.step_3[i] = ~worker.step_3[i];             // binary NOT operator
        worker.step_3[i] += worker.step_3[i];             // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 188:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 189:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 190:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                 // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 191:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                            // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);   // shift right
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 192:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 193:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 194:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 195:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);    // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 196:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 197:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] *= worker.step_3[i];                            // *
        worker.step_3[i] *= worker.step_3[i];                            // *
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 198:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 199:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];           // binary NOT operator
        worker.step_3[i] += worker.step_3[i];           // +
        worker.step_3[i] *= worker.step_3[i];           // *
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 200:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 201:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 202:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 203:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);                 // rotate  bits by 1
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 204:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 205:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 206:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);    // reverse bits
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);    // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 207:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 8); // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);                           // rotate  bits by 3
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 208:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 209:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);     // rotate  bits by 5
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);    // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 210:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];                            // binary NOT operator
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 211:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] += worker.step_3[i];                            // +
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 212:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 213:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 214:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 215:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] *= worker.step_3[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 216:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 217:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 218:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 219:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                 // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);                // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 220:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 221:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 222:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 223:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= worker.step_3[worker.pos2];                  // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                     // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 224:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 4);  // rotate  bits by 1
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       //
      }
      break;
    case 225:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 226:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 227:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 228:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 229:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];            // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 230:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 231:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 232:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 233:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);     // rotate  bits by 3
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 234:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 235:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 236:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 237:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);              // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 238:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];             // +
        worker.step_3[i] += worker.step_3[i];             // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);      // XOR and -
                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 239:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 6); // rotate  bits by 5
        // worker.step_3[i] = rotl_cuda(worker.step_3[i], 1); // rotate  bits by 1
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 240:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 241:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);    // rotate  bits by 4
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 242:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];           // +
        worker.step_3[i] += worker.step_3[i];           // +
        worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 243:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);     // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);    // rotate  bits by 2
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);     // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 244:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = reverse8_hip(worker.step_3[i]); // reverse bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 245:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);              // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 246:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = rotl_hip(worker.step_3[i], 1);              // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 247:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 248:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 5);     // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 249:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);               // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                         // INSERT_RANDOM_CODE_END
      }
      break;
    case 250:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = rotl_hip(worker.step_3[i], worker.step_3[i]);  // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);                // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 251:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);    // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);    // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 252:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8_hip(worker.step_3[i]);             // reverse bits
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 4);             // rotate  bits by 4
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);             // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 253:
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);  // rotate  bits by 3
        // INSERT_RANDOM_CODE_END

        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(&worker.step_3, worker.pos2, 0); // more deviations
      }
      break;
    case 254:
    case 255:
      rc4_setup(&worker.key, worker.step_3, 256);
// worker.step_3 = highwayhash.Sum(worker.step_3[:], worker.step_3[:])
#pragma unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= static_cast<uint8_t>(std::bitset<8>(worker.step_3[i]).count()); // ones count bits
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                                   // rotate  bits by 3
        worker.step_3[i] ^= rotl_hip(worker.step_3[i], 2);                                  // rotate  bits by 2
        worker.step_3[i] = rotl_hip(worker.step_3[i], 3);                                   // rotate  bits by 3
                                                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    default:
      break;
    }

    // if (op == 53) {
    //   std::cout << hexStr(worker.step_3, 256) << std::endl << std::endl;
    //   std::cout << hexStr(&worker.step_3[worker.pos1], 1) << std::endl;
    //   std::cout << hexStr(&worker.step_3[worker.pos2], 1) << std::endl;
    // }

    worker.A = (worker.step_3[worker.pos1] - worker.step_3[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    if (worker.A < 0x10)
    { // 6.25 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(&worker.step_3, worker.pos2, 0);
      // printf("new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x20)
    { // 12.5 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = hash_64_fnv1a(worker.step_3, worker.pos2);
      // printf("new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x30)
    { // 18.75 % probability
      // std::copy(worker.step_3, worker.step_3 + worker.pos2, s3);
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      HH_ALIGNAS(16)
      const highwayhash::HH_U64 key2[2] = {worker.tries, worker.prev_lhash};
      worker.lhash = highwayhash::SipHash(key2, (char*)worker.step_3, worker.pos2); // more deviations
      // printf("new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A <= 0x40)
    { // 25% probablility
      rc4_crypt(&worker.key, worker.step_3, 256);
    }

    worker.step_3[255] = worker.step_3[255] ^ worker.step_3[worker.pos1] ^ worker.step_3[worker.pos2];

    memcpy(&worker.sData[(worker.tries - 1) * 256], worker.step_3, 256);
    // std::copy(worker.step_3, worker.step_3 + 256, &worker.sData[(worker.tries - 1) * 256]);

    // memcpy(&worker->data.data()[(worker.tries - 1) * 256], worker.step_3, 256);

    // std::cout << hexStr(worker.step_3, 256) << std::endl;

    if (worker.tries > 260 + 16 || (worker.step_3[255] >= 0xf0 && worker.tries > 260))
    {
      break;
    }
  }
}
