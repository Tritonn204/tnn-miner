#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h> 
#include <time.h> 
#include <math.h> 
#include <hip/hip_runtime.h> 

#include "cshake256.h"

void gpu_init();
void runBenchmarks();
char *read_in_messages();
int gcd(int a, int b);


int clock_speed;
int number_multi_processors;
int number_blocks;
int number_threads;
int max_threads_per_mp;


#define MSG_SIZE 16
#define THREADS_PER_BLOCK 512


int num_messages;
const int digest_size = 256;
const int digest_size_bytes = digest_size / 8;
const size_t str_length = 7;	//change for different sizes


// hipEvent_t start, stop;

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

__device__ const char *chars = "123abcABC";
	

__device__ const uint64_t RC[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};



__device__ const int r[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14, 
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

__device__ const int piln[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4, 
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1 
};

__device__ void keccak256(uint64_t *state);



__device__ void generate_message(char *message, uint64_t tid, int *str_len){
	int len = 0;
	const int num_chars = 94;
	char str[21];
	while (tid > 0){
		str[len++] = chars[tid % num_chars];
		tid /= num_chars;
	}
	
	str[len] = '\0';
	memcpy(message, str, len + 1);
	*str_len = len;
}

__device__ size_t encode_length(uint8_t *buffer, size_t rate, size_t value, size_t offset, uint64_t *state) {
    size_t length_in_bytes = 0;
    size_t x = value;

    // Calculate the number of bytes needed to represent the value
    while (x > 0) {
        length_in_bytes++;
        x >>= 8;
    }
    if (length_in_bytes == 0) length_in_bytes = 1;

    // Encode the length as the first byte
    buffer[offset++] = (uint8_t)length_in_bytes;

    // If the buffer reaches the rate, apply the keccak permutation
    if (offset == rate) {
        for (int i = 0; i < (rate / 8); i++) {
            state[i] ^= ((uint64_t*)buffer)[i];  // Absorb the buffer into the state
        }
        keccak256(state);  // Apply Keccak permutation
        offset = 0;
    }

    // Now encode the actual value in big-endian format
    for (int i = length_in_bytes - 1; i >= 0; --i) {
        buffer[offset++] = (uint8_t)(value >> (i * 8));
        if (offset == rate) {
            for (int j = 0; j < (rate / 8); j++) {
                state[j] ^= ((uint64_t*)buffer)[j];  // Absorb the buffer into the state
            }
            keccak256(state);  // Apply Keccak permutation
            offset = 0;
        }
    }

    return offset;  // Return the new offset
}

__device__ size_t absorb_string(const char *custom_str, uint8_t custom_len,uint8_t *buffer, size_t rate, size_t offset, uint64_t *state) {
    // Loop through each byte of the customization string
    for (size_t i = 0; i < custom_len; ++i) {
        buffer[offset++] = (uint8_t)custom_str[i];  // Add each byte to the buffer

        // If the buffer reaches the rate, absorb it into the Keccak state
        if (offset == rate) {
            for (int j = 0; j < (rate / 8); j++) {
                state[j] ^= ((uint64_t*)buffer)[j];  // XOR the buffer with the Keccak state
            }
            keccak256(state);  // Apply the Keccak permutation to the state
            offset = 0;  // Reset the buffer offset
        }
    }

    // Return the current offset in the buffer
    return offset;
}

__device__ void keccak256(uint64_t *state){
    uint64_t temp, C[5];
	int j;
	
    for (int i = 0; i < 24; i++) {
        // Theta
		// for i = 0 to 5 
		//    C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
		C[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
		C[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
		C[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
		C[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
		C[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];
		
		// for i = 0 to 5
		//     temp = C[(i + 4) % 5] ^ ROTL64(C[(i + 1) % 5], 1);
		//     for j = 0 to 25, j += 5
		//          state[j + i] ^= temp;
		temp = C[4] ^ ROTL64(C[1], 1);
		state[0] ^= temp;
		state[5] ^= temp;
		state[10] ^= temp;
		state[15] ^= temp;
		state[20] ^= temp;
		
		temp = C[0] ^ ROTL64(C[2], 1);
		state[1] ^= temp;
		state[6] ^= temp;
		state[11] ^= temp;
		state[16] ^= temp;
		state[21] ^= temp;
		
		temp = C[1] ^ ROTL64(C[3], 1);
		state[2] ^= temp;
		state[7] ^= temp;
		state[12] ^= temp;
		state[17] ^= temp;
		state[22] ^= temp;
		
		temp = C[2] ^ ROTL64(C[4], 1);
		state[3] ^= temp;
		state[8] ^= temp;
		state[13] ^= temp;
		state[18] ^= temp;
		state[23] ^= temp;
		
		temp = C[3] ^ ROTL64(C[0], 1);
		state[4] ^= temp;
		state[9] ^= temp;
		state[14] ^= temp;
		state[19] ^= temp;
		state[24] ^= temp;
		
        // Rho Pi
		// for i = 0 to 24
		//     j = piln[i];
		//     C[0] = state[j];
		//     state[j] = ROTL64(temp, r[i]);
		//     temp = C[0];
		temp = state[1];
		j = piln[0];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[0]);
		temp = C[0];
		
		j = piln[1];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[1]);
		temp = C[0];
		
		j = piln[2];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[2]);
		temp = C[0];
		
		j = piln[3];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[3]);
		temp = C[0];
		
		j = piln[4];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[4]);
		temp = C[0];
		
		j = piln[5];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[5]);
		temp = C[0];
		
		j = piln[6];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[6]);
		temp = C[0];
		
		j = piln[7];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[7]);
		temp = C[0];
		
		j = piln[8];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[8]);
		temp = C[0];
		
		j = piln[9];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[9]);
		temp = C[0];
		
		j = piln[10];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[10]);
		temp = C[0];
		
		j = piln[11];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[11]);
		temp = C[0];
		
		j = piln[12];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[12]);
		temp = C[0];
		
		j = piln[13];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[13]);
		temp = C[0];
		
		j = piln[14];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[14]);
		temp = C[0];
		
		j = piln[15];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[15]);
		temp = C[0];
		
		j = piln[16];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[16]);
		temp = C[0];
		
		j = piln[17];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[17]);
		temp = C[0];
		
		j = piln[18];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[18]);
		temp = C[0];
		
		j = piln[19];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[19]);
		temp = C[0];
		
		j = piln[20];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[20]);
		temp = C[0];
		
		j = piln[21];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[21]);
		temp = C[0];
		
		j = piln[22];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[22]);
		temp = C[0];
		
		j = piln[23];
		C[0] = state[j];
		state[j] = ROTL64(temp, r[23]);
		temp = C[0];

        //  Chi
		// for j = 0 to 25, j += 5
		//     for i = 0 to 5
		//         C[i] = state[j + i];
		//     for i = 0 to 5
		//         state[j + 1] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
		C[0] = state[0];
		C[1] = state[1];
		C[2] = state[2];
		C[3] = state[3];
		C[4] = state[4];
			
		state[0] ^= (~C[1]) & C[2];
		state[1] ^= (~C[2]) & C[3];
		state[2] ^= (~C[3]) & C[4];
		state[3] ^= (~C[4]) & C[0];
		state[4] ^= (~C[0]) & C[1];
		
		C[0] = state[5];
		C[1] = state[6];
		C[2] = state[7];
		C[3] = state[8];
		C[4] = state[9];
			
		state[5] ^= (~C[1]) & C[2];
		state[6] ^= (~C[2]) & C[3];
		state[7] ^= (~C[3]) & C[4];
		state[8] ^= (~C[4]) & C[0];
		state[9] ^= (~C[0]) & C[1];
		
		C[0] = state[10];
		C[1] = state[11];
		C[2] = state[12];
		C[3] = state[13];
		C[4] = state[14];
			
		state[10] ^= (~C[1]) & C[2];
		state[11] ^= (~C[2]) & C[3];
		state[12] ^= (~C[3]) & C[4];
		state[13] ^= (~C[4]) & C[0];
		state[14] ^= (~C[0]) & C[1];

		C[0] = state[15];
		C[1] = state[16];
		C[2] = state[17];
		C[3] = state[18];
		C[4] = state[19];
			
		state[15] ^= (~C[1]) & C[2];
		state[16] ^= (~C[2]) & C[3];
		state[17] ^= (~C[3]) & C[4];
		state[18] ^= (~C[4]) & C[0];
		state[19] ^= (~C[0]) & C[1];
		
		C[0] = state[20];
		C[1] = state[21];
		C[2] = state[22];
		C[3] = state[23];
		C[4] = state[24];
			
		state[20] ^= (~C[1]) & C[2];
		state[21] ^= (~C[2]) & C[3];
		state[22] ^= (~C[3]) & C[4];
		state[23] ^= (~C[4]) & C[0];
		state[24] ^= (~C[0]) & C[1];
		
        //  Iota
        state[0] ^= RC[i];
    }
}


__device__ void cshake256(const char *message, int message_len, const char *custom_str, uint8_t custom_len, unsigned char *output, int output_len) {
    uint64_t state[25];    
    uint8_t buffer[144];     // Buffer for absorbing data
    int rsize = 136;         // SHAKE256 rate is 136 bytes
    int rsize_byte = 17;
    size_t offset = 0;

    memset(state, 0, sizeof(state));

    // Step 1: Encode and absorb the length of the customization string (in bits)
    size_t custom_len_bits = custom_len * 8;
    offset = encode_length(buffer, rsize, custom_len_bits, offset, state);

    // Step 2: Absorb the customization string itself
    offset = absorb_string(custom_str, custom_len, buffer, rsize, offset, state);

    // Step 3: Absorb the input message
    for ( ; message_len >= rsize; message_len -= rsize, message += rsize) {
        for (int i = 0; i < rsize_byte; i++) {
            state[i] ^= ((uint64_t *)message)[i];  // Absorb message block by block
        }
        keccak256(state);  // Apply Keccak permutation
    }

    // Step 4: Absorb last message block and padding
    memcpy(buffer, message, message_len);     // Copy remaining part of the message
    buffer[message_len++] = 1;                // Padding: append '01'
    memset(buffer + message_len, 0, rsize - message_len);
    buffer[rsize - 1] |= 0x80;                // Append '1' in the last byte

    for (int i = 0; i < rsize_byte; i++) {
        state[i] ^= ((uint64_t *)buffer)[i];  // Absorb final block
    }

    keccak256(state);  // Apply final Keccak permutation

    // Step 5: Squeeze output
    memcpy(output, state, output_len);
}

__global__ void cshake256_kernel(const char *message, int message_len, const char *custom_str, uint8_t custom_len, unsigned char *output, int output_len) {
    // Call the cshake256 function you implemented
    cshake256(message, message_len, custom_str, custom_len, output, output_len);
}

void cshake256_single(const char *message, int message_len, const char *custom_str, uint8_t custom_len, unsigned char *output, int output_len) {
  cshake256_kernel<<<1, 1>>>(message, message_len, custom_str, custom_len, output, output_len);
}

__device__ void keccak(const char *message, int message_len, unsigned char *output, int output_len){
    uint64_t state[25];    
    uint8_t temp[144];
    int rsize = 136;
    int rsize_byte = 17;
    
    memset(state, 0, sizeof(state));

    for ( ; message_len >= rsize; message_len -= rsize, message += rsize) {
        for (int i = 0; i < rsize_byte; i++) {
            state[i] ^= ((uint64_t *) message)[i];
		}
        keccak256(state);
    }
    
    // last block and padding
    memcpy(temp, message, message_len);
    temp[message_len++] = 1;
    memset(temp + message_len, 0, rsize - message_len);
    temp[rsize - 1] |= 0x80;

    for (int i = 0; i < rsize_byte; i++) {
        state[i] ^= ((uint64_t *) temp)[i];
	}

    keccak256(state);
    memcpy(output, state, output_len);
}




__global__ void benchmark(const char *messages, unsigned char *output, int num_messages){

	// char buf_t[MSG_SIZE];
	// const int output_len = 32;
	// unsigned long offset = (blockIdx.x * THREADS_PER_BLOCK * MSG_SIZE) + (threadIdx.x * MSG_SIZE);
	// int tid = threadIdx.x + (blockIdx.x * blockDim.x);


	// memcpy(buf_t, &messages[offset], MSG_SIZE);

	// keccak(buf_t, MSG_SIZE, &output[tid * output_len], output_len);

	const int str_len = 6;
	const int output_len = 32;
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int num_threads = blockDim.x * gridDim.x;
	
	for (; tid < num_messages; tid += num_threads)
	{
		keccak(&messages[tid * str_len], str_len, &output[tid * output_len], output_len);
	}
}




void gpu_init(){
    hipDeviceProp_t device_prop;
    int device_count, block_size;

    hipGetDeviceCount(&device_count);
    if (device_count != 1) {
        exit(EXIT_FAILURE);
    }

    if (hipGetDeviceProperties(&device_prop, 0) != hipSuccess) {
        exit(EXIT_FAILURE);
    } 

    number_threads = device_prop.maxThreadsPerBlock;
    number_multi_processors = device_prop.multiProcessorCount;
    max_threads_per_mp = device_prop.maxThreadsPerMultiProcessor;
    block_size = (max_threads_per_mp / gcd(max_threads_per_mp, number_threads));
    number_threads = max_threads_per_mp / block_size;
    number_blocks = block_size * number_multi_processors;
    clock_speed = (int) (device_prop.memoryClockRate * 1000 * 1000);    
}




int gcd(int a, int b) {
    return (a == 0) ? b : gcd(b % a, a);
}




char *read_in_messages(char *file_name){
	FILE *f;
	if(!(f = fopen(file_name, "r")))
    {
        printf("Error opening file %s", file_name);
        exit(1);
    }

	char *messages = (char *) malloc(sizeof(char) * num_messages * str_length);
	if (messages == NULL)
	{
	    perror("Error allocating memory for list of Strings.\n");
        exit(1);
	}
	
	int index = 0;
	char buf[10];
	while(1){
		if (fgets(buf, str_length + 1, f) == NULL)
		    break;
		buf[strlen(buf) - 1] = '\0';
		memcpy(&messages[index], buf, str_length);
		index += str_length - 1;
	}
	
	return messages;
}




void runBenchmarks(char *file_name, int num_messages){

	
	hipEvent_t start,stop;

	float time_h_to_d = 0.0;
	float time_k = 0.0;
	float time_d_to_h = 0.0;

	// float comp_time = 0.0;
	// float d_to_h_time = 0.0;
	// float total_time = 0.0;
    float elapsed_time;
	// int hashes_per_sec;
	
	size_t array_size = sizeof(char) * str_length * num_messages;
	size_t output_size = digest_size_bytes * num_messages;
	
	// Allocate host arrays
    char *h_messages = read_in_messages(file_name);
	unsigned char *h_output = (unsigned char *) malloc(output_size);

	char *d_messages;
	unsigned char *d_output;

	dim3 dimBlock(ceil((double)array_size / (double)(512 * 7)));
  	dim3 dimGrid(512);
	
    // Allocate device arrays
    hipMalloc((void**) &d_messages, array_size); // upload original data to device
	hipMalloc((void**) &d_output, output_size); 
	


	// Recording host-to-device
	hipEventCreate(&start); // create event
	hipEventCreate(&stop);

	hipEventRecord( start,0); // start record 	
    hipMemcpy(d_messages, h_messages, array_size, hipMemcpyHostToDevice); // load message into device
	hipEventRecord( stop,0); // end record
	hipEventSynchronize(start); // waits for an event to complete
	hipEventSynchronize(stop);    //Waits for an event to complete
	hipEventElapsedTime(&time_h_to_d,start,stop); // calculate delta time



	// Recording kernel
	hipEventRecord( start,0); // start record
	benchmark<<<dimBlock, dimGrid>>>(d_messages, d_output, num_messages);
	// benchmark<<<10512, 512>>>(d_messages, d_output, num_messages);
	hipEventRecord( stop,0); // end record
	hipEventSynchronize(start); // waits for an event to complete
	hipEventSynchronize(stop);    //Waits for an event to complete
	hipEventElapsedTime(&time_k,start,stop); // calculate delta time


	// Recording device to host
	hipEventRecord( start,0); // start record
	hipMemcpy(h_output, d_output, array_size, hipMemcpyDeviceToHost); // copy message from device
	hipEventRecord(stop,0); // end record
	hipEventSynchronize(start); // waits for an event to complete
	hipEventSynchronize(stop);    //Waits for an event to complete
	hipEventElapsedTime(&time_d_to_h,start,stop); // calculate delta time
	
	
	// printf
	printf("Time used memcpy from host to device %0.3g seconds\n", time_h_to_d/1000);
	printf("Time used kernel %0.3g seconds\n", time_k/1000);
	printf("Time used memcpy from device to host %0.3g seconds\n", time_d_to_h/1000);
	printf("Total %0.3g seconds\n", (time_h_to_d+time_k+time_d_to_h)/1000);


	// Free arrays from memory
    free(h_messages);
	free(h_output);
    hipFree(d_messages);
	hipFree(d_output);
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