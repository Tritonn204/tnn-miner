#ifndef SHA256_HIP_H
#define SHA256_HIP_H


/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#include <inttypes.h>
#include <hip/hip_runtime.h>

#define checkCudaErrors(x) \
{ \
    hipGetLastError(); \
    x; \
    cudaError_t err = hipGetLastError(); \
    if (err != hipSuccess) \
        printf("GPU: hipError %d (%s)\n", err, hipGetErrorString(err)); \
}
/**************************** DATA TYPES ****************************/



typedef unsigned char BYTE;             // 8-bit byte

struct JOB {
	BYTE * data;
	unsigned long long size;
	BYTE digest[64];
	char fname[128];
};


struct SHA256_CTX_HIP {
	BYTE data[64];
	uint32_t datalen;
	unsigned long long bitlen;
	uint32_t state[8];
};

static const uint32_t host_k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*********************** FUNCTION DECLARATIONS **********************/

__device__ void mycpy12(uint32_t *d, const uint32_t *s);

__device__ void mycpy16(uint32_t *d, const uint32_t *s);

__device__ void mycpy32(uint32_t *d, const uint32_t *s);

__device__ void mycpy44(uint32_t *d, const uint32_t *s);

__device__ void mycpy48(uint32_t *d, const uint32_t *s);

__device__ void mycpy64(uint32_t *d, const uint32_t *s);

__device__ void sha256_transform(SHA256_CTX_HIP *ctx, const BYTE data[]);

__device__ void sha256_init_hip(SHA256_CTX_HIP *ctx);

__device__ void sha256_update_hip(SHA256_CTX_HIP *ctx, const BYTE data[], size_t len);

__device__ void sha256_final_hip(SHA256_CTX_HIP *ctx, BYTE hash[]);

#endif   // SHA256_H