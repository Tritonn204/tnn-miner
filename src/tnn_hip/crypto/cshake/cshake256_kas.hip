#include "common.hip.h"
#include "cshake256.h"

#include <iostream>
#include <random>

#define rotate(x, n, w, wmod) ((((x) >> ((w) - ((n) % (w)))) | ((x) << ((n) % (w)))) & (wmod))


/**
 * Rotate a 64-bit word
 * 
 * @param   x:uint64_t  The value to rotate
 * @param   n:long          Rotation steps, may not be zero
 * @return   :uint64_t  The value rotated
 */
#define rotate64(x, n) ((uint64_t)(((uint64_t)(x) >> (64L - (n))) | ((uint64_t)(x) << (n))))

#define LIST_5_REF X(0) X(1) X(2) X(3) X(4)

/**
 * X-macro-enabled listing of all intergers in [0, 7]
 */
#define LIST_8_REF LIST_5_REF X(5) X(6) X(7)

/**
 * X-macro-enabled listing of all intergers in [0, 24]
 */
#define LIST_25_REF LIST_8_REF X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)\
                X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) X(24)



#define X(N) (N % 5) * 5 + N / 5,
/**
 * The order the lanes should be read when absorbing or squeezing,
 * it transposes the lanes in the sponge
 */
static const long int LANE_TRANSPOSE_MAP[] = { LIST_25_REF };
#undef X

static const uint64_t RC_REF[] = {
	0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
	0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
	0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
	0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
	0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
	0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ static void
hipkeccak_f_round(uint64_t *A, uint64_t rc)
{
	uint64_t B[25];
	uint64_t C[5];
	uint64_t da, db, dc, dd, de;
	uint64_t wmod = ~0LL;
	long int w = 64;

	/* θ step (step 1 of 3). */
#define X(N) C[N] = A[N * 5] ^ A[N * 5 + 1] ^ A[N * 5 + 2] ^ A[N * 5 + 3] ^ A[N * 5 + 4];
	LIST_5_REF
#undef X

	/* θ step (step 2 of 3). */
	da = C[4] ^ rotate(C[1], 1, w, wmod);
	dd = C[2] ^ rotate(C[4], 1, w, wmod);
	db = C[0] ^ rotate(C[2], 1, w, wmod);
	de = C[3] ^ rotate(C[0], 1, w, wmod);
	dc = C[1] ^ rotate(C[3], 1, w, wmod);

	/* ρ and π steps, with last two part of θ. */
#define X(bi, ai, dv, r) B[bi] = rotate(A[ai] ^ dv, r, w, wmod)
	B[0] = A[0] ^ da;   X( 1, 15, dd, 28);  X( 2,  5, db,  1);  X( 3, 20, de, 27);  X( 4, 10, dc, 62);
	X( 5,  6, db, 44);  X( 6, 21, de, 20);  X( 7, 11, dc,  6);  X( 8,  1, da, 36);  X( 9, 16, dd, 55);
	X(10, 12, dc, 43);  X(11,  2, da,  3);  X(12, 17, dd, 25);  X(13,  7, db, 10);  X(14, 22, de, 39);
	X(15, 18, dd, 21);  X(16,  8, db, 45);  X(17, 23, de,  8);  X(18, 13, dc, 15);  X(19,  3, da, 41);
	X(20, 24, de, 14);  X(21, 14, dc, 61);  X(22,  4, da, 18);  X(23, 19, dd, 56);  X(24,  9, db,  2);
#undef X

	/* ξ step. */
#define X(N) A[N] = B[N] ^ ((~(B[(N + 5) % 25])) & B[(N + 10) % 25]);
	LIST_25_REF
#undef X

	/* ι step. */
	A[0] ^= rc;
}

__device__ static void
hipkeccak_f_round64(uint64_t *A, uint64_t rc)
{
	uint64_t B[25];
	uint64_t C[5];
	uint64_t da, db, dc, dd, de;

	/* θ step (step 1 of 3). */
#define X(N) C[N] = A[N * 5] ^ A[N * 5 + 1] ^ A[N * 5 + 2] ^ A[N * 5 + 3] ^ A[N * 5 + 4];
	LIST_5_REF
#undef X

	/* θ step (step 2 of 3). */
	da = C[4] ^ rotate64(C[1], 1);
	dd = C[2] ^ rotate64(C[4], 1);
	db = C[0] ^ rotate64(C[2], 1);
	de = C[3] ^ rotate64(C[0], 1);
	dc = C[1] ^ rotate64(C[3], 1);

	/* ρ and π steps, with last two part of θ. */
#define X(bi, ai, dv, r) B[bi] = rotate64(A[ai] ^ dv, r)
	B[0] = A[0] ^ da;   X( 1, 15, dd, 28);  X( 2,  5, db,  1);  X( 3, 20, de, 27);  X( 4, 10, dc, 62);
	X( 5,  6, db, 44);  X( 6, 21, de, 20);  X( 7, 11, dc,  6);  X( 8,  1, da, 36);  X( 9, 16, dd, 55);
	X(10, 12, dc, 43);  X(11,  2, da,  3);  X(12, 17, dd, 25);  X(13,  7, db, 10);  X(14, 22, de, 39);
	X(15, 18, dd, 21);  X(16,  8, db, 45);  X(17, 23, de,  8);  X(18, 13, dc, 15);  X(19,  3, da, 41);
	X(20, 24, de, 14);  X(21, 14, dc, 61);  X(22,  4, da, 18);  X(23, 19, dd, 56);  X(24,  9, db,  2);
#undef X

	/* ξ step. */
#define X(N) A[N] = B[N] ^ ((~(B[(N + 5) % 25])) & B[(N + 10) % 25]);
	LIST_25_REF
#undef X

	/* ι step. */
	A[0] ^= rc;
}

__device__ static inline void
hipkeccak_f(uint64_t *stateBuffer)
{
  for (uint8_t i = 0; i < 24; i++)
    hipkeccak_f_round64(stateBuffer, (uint64_t)(RC_REF[i]));
}

__device__ static inline uint64_t
hipkeccak_to_lane(const unsigned char *message, size_t msglen,
                  long int rr, long int ww, size_t off)
{
	long int n = (long)((msglen < (size_t)rr ? msglen : (size_t)rr) - off);
	uint64_t rc = 0;
	message += off;
	while (ww--) {
		rc <<= 8;
		rc |= __builtin_expect(ww < n, 1) ? (uint64_t)(unsigned char)(message[ww]) : 0L;
	}
	return rc;
}


/**
 * 64-bit lane version of `hipkeccak_to_lane`
 * 
 * @param   message  The message
 * @param   msglen   The length of the message
 * @param   rr       Bitrate in bytes
 * @param   off      The offset in the message
 * @return           The lane
 */
__device__ static inline uint64_t
hipkeccak_to_lane64(const unsigned char *message, size_t msglen, long int rr, size_t off)
{
	long int n = (long)((msglen < (size_t)rr ? msglen : (size_t)rr) - off);
	uint64_t rc = 0;
	message += off;
#define X(N) if (__builtin_expect(N < n, 1)) rc |= (uint64_t)(unsigned char)(message[N]) << (N * 8);\
             else  return rc;
	LIST_8_REF
#undef X
	return rc;
}

__device__ static inline size_t
hipkeccak_pad10star1(size_t r, unsigned char *msg, size_t msglen, size_t bits)
{
	size_t nrf = msglen - !!bits;
	size_t len = (nrf << 3) | bits;
	size_t ll = len % r;
	unsigned char b = (unsigned char)(bits ? (msg[nrf] | (1 << bits)) : 1);

	if (r - 8 <= ll && ll <= r - 2) {
		msg[nrf] = (unsigned char)(b ^ 0x80);
		msglen = nrf + 1;
	} else {
		len = ++nrf << 3;
		len = (len - (len % r) + (r - 8)) >> 3;
		msglen = len + 1;

		msg[nrf - 1] = b;
		// __builtin_memset(&msg[nrf], 0, (len - nrf) * sizeof(char));
		msg[len] = (unsigned char)0x80;
	}
	return msglen;
}

__device__ static void
hipkeccak_absorption_phase(uint64_t *stateBuffer,
                           const unsigned char *message, size_t len)
{
	constexpr long int rr = 1088 >> 3;
	constexpr long int ww = 64 >> 3;
	long int n = (long)len / rr;
	if (__builtin_expect(ww >= 8, 1)) { /* ww > 8 is impossible, it is just for optimisation possibilities. */
		while (n--) {
#define X(N) stateBuffer[N] ^= hipkeccak_to_lane64(message, len, rr, (size_t)(LANE_TRANSPOSE_MAP[N] * 8));
			LIST_25_REF
#undef X
			hipkeccak_f(stateBuffer);
			message += (size_t)rr;
			len -= (size_t)rr;
		}
	} else {
		while (n--) {
#define X(N) stateBuffer[N] ^= hipkeccak_to_lane(message, len, rr, ww, (size_t)(LANE_TRANSPOSE_MAP[N] * ww));
			LIST_25_REF
#undef X
			hipkeccak_f(stateBuffer);
			message += (size_t)rr;
			len -= (size_t)rr;
		}
	}
}

__device__ static void
hipkeccak_squeezing_phase(uint64_t *stateBuffer, long int rr,
                          long int nn, long int ww, unsigned char *hashsum)
{
	uint64_t v;
	long int ni = rr / ww;
	long int i, j = 0;
	long int k;
  for (i = 0; i < ni && j < nn; i++) {
    v = stateBuffer[LANE_TRANSPOSE_MAP[i]];
    for (k = 0; k++ < ww && j++ < nn; v >>= 8)
      *hashsum++ = (unsigned char)v;
  }
}

__device__ static void
hipkeccak_zerocopy_update(uint64_t *stateBuffer, const uint8_t *msg, size_t msglen)
{
	hipkeccak_absorption_phase(stateBuffer, msg, msglen);
}

__device__ static int hipkeccak_update(uint64_t *stateBuffer, uint8_t *memBuffer, const uint8_t *msg, size_t msglen, size_t &mptr)
{
  size_t len;

  // Copy the new message data to the state message buffer
  memcpy(memBuffer + mptr, msg, msglen * sizeof(char));
  mptr += msglen; // Update the message pointer

  // Calculate how much of the message buffer can be processed
  len = mptr;
  len -= mptr % (size_t)(1088 >> 3); // Align to the rate (r)
  mptr -= len;                           // Reduce the message pointer by the amount we can process

  // Absorb the message into the Keccak state
  hipkeccak_absorption_phase(stateBuffer, memBuffer, len);

  // Move the remaining message data to the front of the buffer
  memcpy(memBuffer, memBuffer+ len, mptr * sizeof(char));

  return 0;
}

__device__ static int
hipkeccak_digest(uint64_t *stateBuffer, uint8_t *memBuffer, const char *suffix, uint8_t *hashsum, size_t &mptr)
{
  uint8_t bits = 0;
	constexpr long int rr = 1088 >> 3;
	long int i;

  memBuffer[mptr] = 0;
  memBuffer[mptr] |= (unsigned char)((*suffix++ & 1) << bits++);
  if (bits == 8) {
    bits = 0;
    memBuffer[++(mptr)] = 0;
  }
  memBuffer[mptr] |= (unsigned char)((*suffix++ & 1) << bits++);
  if (bits == 8) {
    bits = 0;
    memBuffer[++(mptr)] = 0;
  }
	if (bits)
		mptr++;

	mptr = hipkeccak_pad10star1((size_t)1088, memBuffer, mptr, bits);

  // printf("kernel buffer\n");
  // for (int i = 0; i < 137; i++) {
  //   printf("%02X", memBuffer[i]);
  // }
  // printf("\n");

	hipkeccak_absorption_phase(stateBuffer, memBuffer, mptr);

  hipkeccak_squeezing_phase(stateBuffer, rr, (256 + 7) >> 3, 64 >> 3, hashsum);

	return 0;
}

#include "initialize_kas.hip.inc"

__global__ void cSHAKE256_kernel_kas(const uint8_t *msg, const char* custom, uint8_t *digest, size_t output_length) {
  int threadId = threadIdx.x;  // Thread index within the block (x-dimension)
  int blockId = blockIdx.x;    // Block index within the grid (x-dimension)

  if (threadId == 0) printf("block %d used\n", blockId);
  
  // Calculate global thread index (linear index)
  int globalId = blockId * CSHAKE256_THREADS + threadId;

  // Each block handles 80 * CSHAKE256_THREADS bytes of message
  size_t block_msg_len = 80 * CSHAKE256_THREADS;
 
  // Pointer to the start of this block's message data
  const uint8_t* block_msg = &msg[blockId * block_msg_len];

  __shared__ struct hipkeccak_state_optimized state;

  for (size_t i = threadId; i < CSHAKE256_MLEN*blockDim.x; i += blockDim.x) {
    state.M[i] = ((i % CSHAKE256_MLEN) <= 80) ? block_msg[i] : 0;
  }

  size_t mptr = 0;
  for (size_t x = threadId; x < 25*blockDim.x; x += blockDim.x)
    state.S[x] = 0;

  __syncthreads();

  hipkeccak_cshake_initialise(&state.S[threadId*25], &state.M[threadId*CSHAKE256_MLEN], (uint8_t*)custom, 15);
  
  // printf("kernel state\n");
  // for (int i = 0; i < 25; i++) {
  //   printf("%llu\n", state.S[i]);
  // }
  // printf("\n");

  hipkeccak_update(&state.S[threadId*25], &state.M[threadId*CSHAKE256_MLEN], msg, 80, mptr);
  hipkeccak_digest(&state.S[threadId*25], &state.M[threadId*CSHAKE256_MLEN], "00", &digest[32*globalId], mptr);
}

void test_cshake256_hip()
{
    int invalid = 0;

    printf("batchSize = %d, blocks: %d, threads: %d\n", CSHAKE256_BATCH_SIZE, CSHAKE256_BLOCKS, CSHAKE256_THREADS);

    // Example input data
    // printf("original input\n");
    uint8_t input[80*CSHAKE256_BATCH_SIZE]; // Input data
    for (int i = 0; i < 80; i++) {
        // input[i] = rand() % 256;
        input[i] = (i*i) % 256;
        // printf("%02x", input[i]);
    }
    // printf("\n");

    // printf("test input\n");
    for (int i = 1; i < CSHAKE256_BATCH_SIZE; i++) {
      memcpy(&input[i*80], input, 80);
    }

    // for (int i = 0; i < 80; i++) {
    //     printf("%02x", input[i]);
    // }
    // printf("\n");

    // printf("PRE LAUNCH\n");

    size_t inputLen = sizeof(input);

    // Customization string used in both cases
    const char *customString = "ProofOfWorkHash";

    // Output buffers for both implementations (32 bytes = 256 bits)
    uint8_t customOutput[32*CSHAKE256_BATCH_SIZE] = {0}; // For custom implementation

    // Output length (32 bytes = 256 bits)
    size_t outputLen = 32;

    // Customization string as byte array (for custom implementation)
    uint8_t S[] = "ProofOfWorkHash"; // Same as the custom string
    size_t sLen = strlen((char *)S); // Length of the custom string

    // --- GPU Memory Allocation ---
    uint8_t *d_input, *d_customOutput;
    char *d_customString;

    // Allocate memory on the GPU for input, output, and custom string
    hipMalloc((void **)&d_input, inputLen * sizeof(uint8_t));          // Allocate input on GPU
    hipMalloc((void **)&d_customOutput, outputLen * sizeof(uint8_t) * CSHAKE256_BATCH_SIZE);  // Allocate output on GPU
    hipMalloc((void **)&d_customString, sLen * sizeof(char));          // Allocate custom string on GPU

    // --- Copy Data to GPU ---
    hipMemcpy(d_input, input, inputLen * sizeof(uint8_t), hipMemcpyHostToDevice); // Copy input to GPU
    hipMemcpy(d_customString, customString, sLen * sizeof(char), hipMemcpyHostToDevice); // Copy custom string to GPU

    constexpr int BLOCKS = (CSHAKE256_BATCH_SIZE + CSHAKE256_THREADS-1) / CSHAKE256_THREADS;

    // --- Kernel Launch ---
    cSHAKE256_kernel_kas<<<BLOCKS, CSHAKE256_THREADS>>>(d_input, d_customString, d_customOutput, outputLen * 8);
    // Synchronize and check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
    }

    // --- Copy Results Back to Host ---
    hipMemcpy(customOutput, d_customOutput, outputLen * sizeof(uint8_t) * CSHAKE256_BATCH_SIZE, hipMemcpyDeviceToHost); // Copy result back to host

    // --- Print the result ---
    printf("HIP cSHAKE256 output:\n");
    // printf("output at index: %d\n", (CSHAKE256_BATCH_SIZE-1)*outputLen);
    for (size_t i = 0; i < outputLen; i++) {
        printf("%02x", customOutput[i + (CSHAKE256_BATCH_SIZE-1)*outputLen]);
    }
    printf("\n");

    // --- Free GPU Memory ---
    hipFree(d_input);
    hipFree(d_customOutput);
    hipFree(d_customString);
}