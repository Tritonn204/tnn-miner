#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h> 
#include <time.h> 
#include <math.h> 

#include "common.hip.h"

__global__ void cSHAKE256_kernel_kas(const uint8_t *msg, const char* custom, uint8_t *digest, size_t output_length) {
  int threadId = threadIdx.x;  // Thread index within the block (x-dimension)
  int blockId = blockIdx.x;    // Block index within the grid (x-dimension)
  
  // Calculate global thread index (linear index)
  int globalId = blockId * blockDim.x + threadId;

  // Each block handles 80 * CSHAKE256_THREADS bytes of message
  size_t block_msg_len = 80 * CSHAKE256_THREADS;

  // Pointer to the start of this block's message data
  const uint8_t* block_msg = &msg[blockId * block_msg_len];

  __shared__ struct hipkeccak_state_optimized state;

  for (int i = threadId; i < 80*CSHAKE256_THREADS; i += CSHAKE256_THREADS) {
    state.M[i] = block_msg[i];
  }

  size_t mptr = 0;
  for (uint8_t x = threadId; x < 25*CSHAKE256_THREADS; x += CSHAKE256_THREADS)
    state.S[x] = 0;

  // for (int i = 0; i < 80; i++) {
  //   if (threadId + 80 < CSHAKE256_MLEN) state.M[80*i + threadId + 80] = 0;
  // }

  // hipkeccak_cshake_initialise(&state, NULL, 0, 0, NULL,
  //                               (uint8_t*)custom, strlen(custom), 0, NULL);
  
  // hipkeccak_fast_update(&state, msg, msg_len);
  // hipkeccak_fast_digest(&state, NULL, 0, 0, "00", digest);
  // hipkeccak_state_destroy(&state);
}
// int main(){

// 	clock_t start, enc_time, dec_time, end;
//     char *file_name;
	
// 	file_name = "input.txt";
// 	num_messages = 120000;
// 	// num_messages = 400;

//     gpu_init();
    
//     // start = clock();

// 	runBenchmarks(file_name, num_messages);

// 	// end = clock(); 
// 	// printf("time used:%f\n",  (double)(end - start) / CLOCKS_PER_SEC); 
// 	// printf("GPU encryption throughput: %f bytes/second\n",  (double)(num_messages*10) / ((double)(end - start) / CLOCKS_PER_SEC)); 

//     return EXIT_SUCCESS;
// }