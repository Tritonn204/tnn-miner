#include "xelis-hash.hpp"
#include <stdlib.h>
#include <iostream>
#include "aes.hpp"
#include "chacha20.h"
#include <crc32.h>
#include <BLAKE3/c/blake3.h>
#include <chacha20.h>

#if defined(__x86_64__)
  #include <emmintrin.h>
  #include <immintrin.h>
#elif defined(__aarch64__)
  #include <arm_neon.h>
#endif
#include <numeric>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <array>
#include <cassert>
#include <chrono>

#include <sodium.h>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

#define rl64(x, a) (((x << (a & 63)) | (x >> (64 - (a & 63)))))
#define rr64(x, a) (((x >> (a & 63)) | (x << (64 - (a & 63)))))

#define htonll(x) ((1 == htonl(1)) ? (x) : ((uint64_t)htonl((x) & 0xFFFFFFFF) << 32) | htonl((x) >> 32))
#define ntohll(x) ((1 == ntohl(1)) ? (x) : ((uint64_t)ntohl((x) & 0xFFFFFFFF) << 32) | ntohl((x) >> 32))

const int sign_bit_values_avx512[16][16] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

alignas(32) const int sign_bit_values_avx2[8][8] = {
    {0, 0, 0, 0, 0, 0, 0, -1},
    {0, 0, 0, 0, 0, 0, -1, 0},
    {0, 0, 0, 0, 0, -1, 0, 0},
    {0, 0, 0, 0, -1, 0, 0, 0},
    {0, 0, 0, -1, 0, 0, 0, 0},
    {0, 0, -1, 0, 0, 0, 0, 0},
    {0, -1, 0, 0, 0, 0, 0, 0},
    {-1, 0, 0, 0, 0, 0, 0, 0}};

alignas(16) const int sign_bit_values_sse[4][4] = {
    {0, 0, 0, -1},
    {0, 0, -1, 0},
    {0, -1, 0, 0},
    {-1, 0, 0, 0}};

static inline uint64_t swap_bytes(uint64_t value)
{
  return ((value & 0xFF00000000000000ULL) >> 56) |
         ((value & 0x00FF000000000000ULL) >> 40) |
         ((value & 0x0000FF0000000000ULL) >> 24) |
         ((value & 0x000000FF00000000ULL) >> 8) |
         ((value & 0x00000000FF000000ULL) << 8) |
         ((value & 0x0000000000FF0000ULL) << 24) |
         ((value & 0x000000000000FF00ULL) << 40) |
         ((value & 0x00000000000000FFULL) << 56);
}

static inline void blake3(const uint8_t *input, int len, uint8_t *output) {
        blake3_hasher hasher;
	blake3_hasher_init(&hasher);
	blake3_hasher_update(&hasher, input, len);
	blake3_hasher_finalize(&hasher, output, BLAKE3_OUT_LEN);
}

static inline void aes_round(uint8_t *block, const uint8_t *key)
{
#if defined(__AES__)
  #if defined(__x86_64__)
    __m128i block_m128i = _mm_load_si128((__m128i *)block);
    __m128i key_m128i = _mm_load_si128((__m128i *)key);
    __m128i result = _mm_aesenc_si128(block_m128i, key_m128i);
    _mm_store_si128((__m128i *)block, result);
  #elif defined(__aarch64__)
    uint8x16_t blck = vld1q_u8(block);
    uint8x16_t ky = vld1q_u8(key);
    // This magic sauce is from here: https://blog.michaelbrase.com/2018/06/04/optimizing-x86-aes-intrinsics-on-armv8-a/
    uint8x16_t rslt = vaesmcq_u8(vaeseq_u8(blck, (uint8x16_t){})) ^ ky;
    vst1q_u8(block, rslt);
  #endif
#else
  aes_single_round_no_intrinsics(block, key);
#endif
}

void chacha_encrypt(uint8_t *key, uint8_t *nonce, uint8_t *in, uint8_t *out, size_t bytes, uint32_t rounds)
{
	uint8_t state[48] = {0};
	ChaCha20SetKey(state, key);
	ChaCha20SetNonce(state, nonce);
	ChaCha20EncryptBytes(state, in, out, bytes, rounds);
}

void stage_1(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  const size_t chunk_size = 32;
  const size_t nonce_size = 12;
  const size_t output_size = XELIS_MEMORY_SIZE_V2 * 8;
  const size_t chunks = 4;

  uint8_t *t = reinterpret_cast<uint8_t *>(sp);
  uint8_t key[chunk_size * chunks] = {0};
  uint8_t K2[32] = {0};
  uint8_t buffer[chunk_size*2] = {0};

  memcpy(key, input, input_len);
  blake3(input, input_len, buffer);

  memcpy(buffer + chunk_size, key, chunk_size);
  blake3(buffer, chunk_size*2, K2);
  chacha_encrypt(K2, buffer, NULL, t, output_size / chunks, 8);

  t += output_size / chunks;

  memcpy(buffer, K2, chunk_size);
  memcpy(buffer + chunk_size, key + chunk_size, chunk_size);
  blake3(buffer, chunk_size*2, K2);
  chacha_encrypt(K2, t - nonce_size, NULL, t, output_size / chunks, 8);

  t += output_size / chunks;

  memcpy(buffer, K2, chunk_size);
  memcpy(buffer + chunk_size, key + 2*chunk_size, chunk_size);
  blake3(buffer, chunk_size*2, K2);
  chacha_encrypt(K2, t - nonce_size, NULL, t, output_size / chunks, 8);

  t += output_size / chunks;

  memcpy(buffer, K2, chunk_size);
  memcpy(buffer + chunk_size, key + 3*chunk_size, chunk_size);
  blake3(buffer, chunk_size*2, K2);
  chacha_encrypt(K2, t - nonce_size, NULL, t, output_size / chunks, 8);

  // Crc32 crc32;
  // crc32.input((uint8_t *)sp, XELIS_MEMORY_SIZE_V2 * 8);
  // printf("%lu\n", crc32.result());
  // Crc32 crc32;
  // crc32.input(scratch_pad, 10);
  // std::cout << "Stage 1 scratch pad CRC32: 0x" << std::hex << std::setw(8) << std::setfill('0') << crc32.result() << std::endl;
}

static inline uint64_t isqrt(uint64_t n)
{
  if (n < 2)
  {
    return n;
  }

  uint64_t x = n;
  uint64_t y = (x + 1) >> 1;

  while (y < x)
  {
    x = y;
    y = (x + n / x) >> 1;
  }

  return x;
}

#define COMBINE_UINT64(high, low) (((__uint128_t)(high) << 64) | (low))
static inline __uint128_t combine_uint64(uint64_t high, uint64_t low)
{
	return ((__uint128_t)high << 64) | low;
}

#if defined(__AVX2__)
__attribute__((target("avx2")))
void static inline uint64_to_le_bytes(uint64_t value, uint8_t *bytes) {
    // // Set up the 64-bit value in a 128-bit register
    // __m128i val = _mm_set1_epi64x(value);

    // // Mask to isolate each byte
    // __m128i mask = _mm_set_epi8(
    //     -1, -1, -1, -1, -1, -1, -1, -1,
    //     7, 6, 5, 4, 3, 2, 1, 0
    // );

    // // Shuffle bytes into the correct order for little-endian
    // __m128i shuffled = _mm_shuffle_epi8(_mm_set1_epi64x(value), _mm_set_epi8(
    //     -1, -1, -1, -1, -1, -1, -1, -1,
    //     7, 6, 5, 4, 3, 2, 1, 0
    // ));

    // Store the result
    _mm_storel_epi64((__m128i*)bytes, _mm_shuffle_epi8(_mm_set1_epi64x(value), _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        7, 6, 5, 4, 3, 2, 1, 0
    )));
}

__attribute__((target("avx2")))
uint64_t static inline le_bytes_to_uint64(const uint8_t *bytes) {
    // // Load the bytes into a 128-bit register
    // __m128i input = _mm_loadu_si128((const __m128i*)bytes);

    // // Mask to isolate each byte and order them correctly for little-endian
    // __m128i mask = _mm_set_epi8(
    //     15, 14, 13, 12, 11, 10, 9, 8,
    //     7,  6,  5,  4,  3,  2, 1, 0
    // );

    // // Shuffle bytes into the correct order for 64-bit integer
    // __m128i shuffled = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)bytes), _mm_set_epi8(
    //     15, 14, 13, 12, 11, 10, 9, 8,
    //     7,  6,  5,  4,  3,  2, 1, 0
    // ));

    // Extract the lower 64 bits as a 64-bit integer
    // uint64_t result = _mm_cvtsi128_si64(shuffled);

    return _mm_cvtsi128_si64(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)bytes), _mm_set_epi8(
        15, 14, 13, 12, 11, 10, 9, 8,
        7,  6,  5,  4,  3,  2, 1, 0
    )));
}
#endif

#if defined(__x86_64__)
__attribute__((target("default")))
#endif
void static inline uint64_to_le_bytes(uint64_t value, uint8_t *bytes)
{
	for (int i = 0; i < 8; i++)
	{
		bytes[i] = value & 0xFF;
		value >>= 8;
	}
}

#if defined(__x86_64__)
__attribute__((target("default")))
#endif
uint64_t static inline le_bytes_to_uint64(const uint8_t *bytes)
{
	uint64_t value = 0;
	for (int i = 7; i >= 0; i--)
		value = (value << 8) | bytes[i];
	return value;
}

#if defined(__x86_64__)

void static inline aes_single_round(uint8_t *block, const uint8_t *key)
{
	// Perform single AES encryption round
	__m128i block_vec = _mm_aesenc_si128(_mm_loadu_si128((const __m128i *)block), _mm_loadu_si128((const __m128i *)key));
	_mm_storeu_si128((__m128i *)block, block_vec);
}

static inline uint64_t div128(__uint128_t dividend, __uint128_t divisor) {
  return dividend / divisor;
}

static inline uint64_t Divide128Div64To64(uint64_t high, uint64_t low, uint64_t divisor, uint64_t *remainder)
{
	uint64_t result;
	__asm__("divq %[v]"
			: "=a"(result), "=d"(*remainder) // Output parametrs, =a for rax, =d for rdx, [v] is an
			// alias for divisor, input paramters "a" and "d" for low and high.
			: [v] "r"(divisor), "a"(low), "d"(high));
	return result;
}

static inline uint64_t ROTR(uint64_t x, uint32_t r)
{
	asm("rorq %%cl, %0" : "+r"(x) : "c"(r));
	return x;
}

static inline uint64_t ROTL(uint64_t x, uint32_t r)
{
	asm("rolq %%cl, %0" : "+r"(x) : "c"(r));
	return x;
}

#else // aarch64

static inline uint64_t div128(__uint128_t dividend, __uint128_t divisor) {
    return dividend / divisor;
}

static inline uint64_t Divide128Div64To64(uint64_t high, uint64_t low, uint64_t divisor, uint64_t *remainder)
{
    // Combine high and low into a 128-bit dividend
    __uint128_t dividend = ((__uint128_t)high << 64) | low;

    // Perform division using built-in compiler functions
    *remainder = dividend % divisor;
    return dividend / divisor;
}

static inline uint64_t ROTR(uint64_t x, uint32_t r)
{
    r %= 64;  // Ensure r is within the range [0, 63] for a 64-bit rotate
    return (x >> r) | (x << (64 - r));
}

static inline uint64_t ROTL(uint64_t x, uint32_t r)
{
    r %= 64;  // Ensure r is within the range [0, 63] for a 64-bit rotate
    return (x << r) | (x >> (64 - r));
}

#endif

static inline uint64_t udiv(uint64_t high, uint64_t low, uint64_t divisor)
{
	uint64_t remainder;

	if (high < divisor)
	{
		return Divide128Div64To64(high, low, divisor, &remainder);
	}
	else
	{
		uint64_t qhi = Divide128Div64To64(0, high, divisor, &high);
		return Divide128Div64To64(high, low, divisor, &remainder);
	}
  return low;
}

// __attribute__((noinline))
static inline uint64_t case_0(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return ROTL(c, i * j) ^ b; 
}
// __attribute__((noinline))
static inline uint64_t case_1(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return ROTR(c, i * j) ^ a; 
}
// __attribute__((noinline))
static inline uint64_t case_2(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return a ^ b ^ c; 
}
// __attribute__((noinline))
static inline uint64_t case_3(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return (a + b) * c; 
}
// __attribute__((noinline))
static inline uint64_t case_4(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return (b - c) * a; 
}
// __attribute__((noinline))
static inline uint64_t case_5(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return c - a + b; 
}
// __attribute__((noinline))
static inline uint64_t case_6(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return a - b + c; 
}
// __attribute__((noinline))
static inline uint64_t case_7(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return b * c + a; 
}
// __attribute__((noinline))
static inline uint64_t case_8(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return c * a + b; 
}
// __attribute__((noinline))
static inline uint64_t case_9(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return a * b * c; 
}
// __attribute__((noinline))
static inline uint64_t case_10(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return COMBINE_UINT64(a,b) % (c | 1); 
}
// __attribute__((noinline))
static inline uint64_t case_11(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  __uint128_t t2 = COMBINE_UINT64(ROTL(result, r), a | 2);
  return (t2 > COMBINE_UINT64(b,c)) ? c : COMBINE_UINT64(b,c) % t2;
}
// __attribute__((noinline))
static inline uint64_t case_12(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return udiv(c, a, b | 4); 
}
// __attribute__((noinline))
static inline uint64_t case_13(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  __uint128_t t1 = COMBINE_UINT64(ROTL(result, r), b);
  __uint128_t t2 = COMBINE_UINT64(a, c | 8);
  return (t1 > t2) ? t1 / t2 : a ^ b;
}
// __attribute__((noinline))
static inline uint64_t case_14(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return (COMBINE_UINT64(b,a) * c) >> 64; 
}
// __attribute__((noinline))
static inline uint64_t case_15(uint64_t a, uint64_t b, uint64_t c, int r, uint64_t result, int i, int j) { 
  return (COMBINE_UINT64(a,c) * COMBINE_UINT64(ROTR(result, r), b)) >> 64; 
}

typedef uint64_t (*operation_func)(uint64_t, uint64_t, uint64_t, int, uint64_t, int, int);

operation_func operations[] = {
    case_0, case_1, case_2, case_3, case_4, case_5, case_6, case_7,
    case_8, case_9, case_10, case_11, case_12, case_13, case_14, case_15,
    // Add other functions for cases 10-15
};

void stage_3(uint64_t *scratch_pad, workerData_xelis_v2 &worker)
{
    const uint8_t key[17] = "xelishash-pow-v2";
    uint8_t block[16] = {0};

    uint64_t *mem_buffer_a = scratch_pad;
    uint64_t *mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V2;

    uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V2 - 1];
    uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V2 - 1] >> 32;
    size_t r = 0;


    #pragma unroll 3
    for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V2; ++i) {
        uint64_t mem_a = mem_buffer_a[addr_a % XELIS_BUFFER_SIZE_V2];
        uint64_t mem_b = mem_buffer_b[addr_b % XELIS_BUFFER_SIZE_V2];

        // std::copy(&mem_b, &mem_b + 8, block);
        // std::copy(&mem_a, &mem_a + 8, block + 8);

        uint64_to_le_bytes(mem_b, block);
		    uint64_to_le_bytes(mem_a, block + 8);

        aes_round(block, key);

        uint64_t hash1 = 0, hash2 = 0;
        // hash1 = ((uint64_t*)block)[0]; // simple assignment, slower than SIMD on my CPU
        hash1 = le_bytes_to_uint64(block);
        // std::copy(block, block + 8, &hash1);
        // hash1 = _byteswap_uint64(hash1);
        hash2 = mem_a ^ mem_b;

        addr_a = ~(hash1 ^ hash2);

        // printf("pre result: %llu\n", result);

        for (size_t j = 0; j < XELIS_BUFFER_SIZE_V2; ++j) {
            uint64_t a = mem_buffer_a[(addr_a % XELIS_BUFFER_SIZE_V2)];
            uint64_t b = mem_buffer_b[~ROTR(addr_a, r) % XELIS_BUFFER_SIZE_V2];
            uint64_t c = (r < XELIS_BUFFER_SIZE_V2) ? mem_buffer_a[r] : mem_buffer_b[r - XELIS_BUFFER_SIZE_V2];
            r = (r+1) % XELIS_MEMORY_SIZE_V2;

            // printf("a %llu, b %llu, c %llu, ", a, b, c);
            uint64_t v;
            uint32_t idx = ROTL(addr_a, (uint32_t)c) & 0xF;
            v = operations[idx](a,b,c,r,addr_a,i,j);

            addr_a = ROTL(addr_a ^ v, 1);

            uint64_t t = mem_buffer_a[XELIS_BUFFER_SIZE_V2 - j - 1] ^ addr_a;
            mem_buffer_a[XELIS_BUFFER_SIZE_V2 - j - 1] = t;
            mem_buffer_b[j] ^= ROTR(t, (uint32_t)addr_a);
        }
        // printf("post result: %llu\n", result);
        addr_b = isqrt(addr_a);
    }

  // Crc32 crc32;
  // crc32.input((uint8_t *)scratch_pad, XELIS_MEMORY_SIZE_V2 * 8);
  // printf("%llu\n", crc32.result());
}


// void stage_3(workerData_xelis_v2 &worker)
// {
//     memset(worker.block, 0, 16);

//     uint64_t *mem_buffer_a = worker.scratchPad;
//     uint64_t *mem_buffer_b = worker.scratchPad + XELIS_BUFFER_SIZE_V2;

//     worker.addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V2 - 1];
//     worker.addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V2 - 1] >> 32;
//     size_t r = 0;

//     for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V2; ++i) {
//         worker.mem_a = mem_buffer_a[worker.addr_a % XELIS_BUFFER_SIZE_V2];
//         worker.mem_b = mem_buffer_b[worker.addr_b % XELIS_BUFFER_SIZE_V2];

//         // std::copy(&worker.mem_b, &worker.mem_b + 8, block);
//         // std::copy(&worker.mem_a, &worker.mem_a + 8, block + 8);
//         uint64_to_le_bytes(worker.mem_b, worker.block);
// 		    uint64_to_le_bytes(worker.mem_a, worker.block + 8);

//         aes_round(worker.block, worker.key);

//         uint64_t hash1 = 0, hash2 = 0;
//         hash1 = le_bytes_to_uint64(worker.block);
//         // std::copy(block, block + 8, &hash1);
//         // hash1 = _byteswap_uint64(hash1);
//         hash2 = worker.mem_a ^ worker.mem_b;

//         uint64_t result = ~(hash1 ^ hash2);

//         // printf("pre result: %llu\n", result);

//         for (size_t j = 0; j < XELIS_BUFFER_SIZE_V2; ++j) {
//             worker.a = mem_buffer_a[(result % XELIS_BUFFER_SIZE_V2)];
//             worker.b = mem_buffer_b[~ROTR(result, r) % XELIS_BUFFER_SIZE_V2];
//             uint64_t c = (r < XELIS_BUFFER_SIZE_V2) ? mem_buffer_a[r] : mem_buffer_b[r - XELIS_BUFFER_SIZE_V2];
//             r = (r < XELIS_MEMORY_SIZE_V2 - 1) ? r + 1 : 0;

//             // printf("a %llu, b %llu, c %llu, ", a, b, c);
//             uint64_t v;
//             __uint128_t t1, t2;
//             int scase = ROTL(result, (uint32_t)c) & 0xF;
//             switch (ROTL(result, (uint32_t)c) & 0xF) {
//               case 0: v = ROTL(c, i * j) ^ worker.b; break;
//               case 1: v = ROTR(c, i * j) ^ worker.a; break;
//               case 2:
//                 v = worker.a ^ worker.b ^ c;
//                 break;
//               case 3:
//                 v = ((worker.a + worker.b) * c);
//                 break;
//               case 4:
//                 v = ((worker.b - c) * worker.a);
//                 break;
//               case 5:
//                 v = (c - worker.a + worker.b);
//                 break;
//               case 6:
//                 v = (worker.a - worker.b + c);
//                 break;
//               case 7:
//                 v = (worker.b * c + worker.a);
//                 break;
//               case 8:
//                 v = (c * worker.a + worker.b);
//                 break;
//               case 9:
//                 v = (worker.a * worker.b * c);
//                 break;
//               case 10:
//               {
//                 v = COMBINE_UINT64(worker.a,worker.b) % (c | 1);
//               }
//               break;
//               case 11:
//               {
//                 t2 = COMBINE_UINT64(ROTL(result, r), worker.a | 2);
//                 v = (t2 > COMBINE_UINT64(worker.b,c)) ? c : COMBINE_UINT64(worker.b,c) % t2;
//               }
//               break;
//               case 12: v = udiv(c, worker.a, worker.b | 4); break;
//               case 13:
//               {
//                 t1 = COMBINE_UINT64(ROTL(result, r), worker.b);
//                 t2 = COMBINE_UINT64(worker.a, c | 8);
//                 v = (t1 > t2) ? t1 / t2 : worker.a ^ worker.b;
//               }
//               break;
//               case 14: v = (COMBINE_UINT64(worker.b,worker.a) * c) >> 64; break;
//               case 15: v = (COMBINE_UINT64(worker.a,c) * COMBINE_UINT64(ROTR(result, r), worker.b)) >> 64; break;
//             }

//             result = ROTL(result ^ v, 1);

//             // printf("post result: %llu, case: %d\n", result, scase);

//             uint64_t t = mem_buffer_a[XELIS_BUFFER_SIZE_V2 - j - 1] ^ result;
//             mem_buffer_a[XELIS_BUFFER_SIZE_V2 - j - 1] = t;
//             mem_buffer_b[j] ^= ROTR(t, (uint32_t)result);
//         }
//         // printf("post result: %llu\n", result);
//         worker.addr_a = result;
//         worker.addr_b = isqrt(result);
//     }

//   // Crc32 crc32;
//   // crc32.input((uint8_t *)scratch_pad, XELIS_MEMORY_SIZE_V2 * 8);
//   // printf("%llu\n", crc32.result());
// }

void xelis_hash_v2(byte *input, workerData_xelis_v2 &worker, byte *hashResult)
{
  stage_1(input, worker.scratchPad, 112);
  stage_3(worker.scratchPad, worker);
  blake3((uint8_t*)worker.scratchPad, XELIS_MEMORY_SIZE_V2 * 8, hashResult);
  // memset(hashResult, 0xFF, 32*batchSize); // For testing without node errors
}

void xelis_benchmark_cpu_hash_v2()
{
  const uint32_t ITERATIONS = 2000;
  byte input[112] = {0};
  workerData_xelis_v2 worker;
  workerData_xelis_v2 worker2;
  byte hash_result[XELIS_HASH_SIZE] = {0};

  printf("v2 bench\n");

  auto start = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < ITERATIONS; ++i)
  {
    // input[0] = i & 0xFF;
    // input[1] = (i >> 8) & 0xFF;
    memset(worker.scratchPad, 0, XELIS_MEMORY_SIZE_V2*8);
    xelis_hash_v2(input, worker, hash_result);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time took: " << elapsed.count() << " ms" << std::endl;
  std::cout << "H/s: " << ((ITERATIONS * 1000.0) / elapsed.count()) << std::endl;
  std::cout << "ms per hash: " << (elapsed.count() / ITERATIONS) << std::endl;

  for (int i = 0; i < 32; i++) {
    printf("%02x", hash_result[i]);
  }
  printf("\n");

  // Crc32 crc32;
  // crc32.input(reinterpret_cast<uint8_t*>(worker.scratchPad), 10);
  // std::cout << "Stage 1 scratch pad CRC32: 0x" << std::hex << std::setw(8) << std::setfill('0') << crc32.result() << std::endl;
}

static int char2int(char input)
{
  if (input >= '0' && input <= '9')
    return input - '0';
  if (input >= 'A' && input <= 'F')
    return input - 'A' + 10;
  if (input >= 'a' && input <= 'f')
    return input - 'a' + 10;
  throw std::invalid_argument("Invalid input string");
}

// This function assumes src to be a zero terminated sanitized string with
// an even number of [0-9a-f] characters, and target to be sufficiently large
static void hex2bin(const char *src, char *target)
{
  while (*src && src[1])
  {
    *(target++) = char2int(*src) * 16 + char2int(src[1]);
    src += 2;
  }
}

static const char *testTemplate = "97dff4761917c2692df3be38e72ca7a59c3f55252e2245cc21564ef65fa8ea6f0000018f22fe78f80000000000064202f2a40463ccfcea839c4950a56ee38fa69c7ce2d4ba45d4b060cc63c297fb73b8a09c69661b1690b0a238d096a7ccb3cb204ce5dd604da9bb6c79c4ab00000000";

namespace xelis_tests_v2
{
  using Hash = std::array<byte, XELIS_HASH_SIZE>;

  bool test_input(const char *test_name, byte *input, size_t input_size, const Hash &expected_hash)
  {
    workerData_xelis_v2 worker;
    byte hash_result[XELIS_HASH_SIZE] = {0};

    xelis_hash_v2(input, worker, hash_result);

    if (std::memcmp(hash_result, expected_hash.data(), XELIS_HASH_SIZE) != 0)
    {
      std::cout << "Test '" << test_name << "' failed!" << std::endl;
      std::cout << "Expected hash: ";
      for (size_t i = 0; i < XELIS_HASH_SIZE; ++i)
      {
        printf("%02x ", expected_hash[i]);
      }
      std::cout << std::endl;
      std::cout << "Actual hash:   ";
      for (size_t i = 0; i < XELIS_HASH_SIZE; ++i)
      {
        printf("%02x ", hash_result[i]);
      }
      std::cout << std::endl;
      return false;
    }
    return true;
  }

  bool test_zero_input()
  {
    alignas(32) byte input[112] = {0};
    Hash expected_hash = {
		126, 219, 112, 240, 116, 133, 115,
		144, 39, 40, 164, 105, 30, 158, 45,
		126, 64, 67, 238, 52, 200, 35, 161, 19,
		144, 211, 214, 225, 95, 190, 146, 27};

    return test_input("test_zero_input", input, sizeof(input), expected_hash);
  }

  bool test_xelis_input()
  {
    byte input[] = {
            172, 236, 108, 212, 181, 31, 109, 45, 44, 242, 54, 225, 143, 133,
            89, 44, 179, 108, 39, 191, 32, 116, 229, 33, 63, 130, 33, 120, 185, 89,
            146, 141, 10, 79, 183, 107, 238, 122, 92, 222, 25, 134, 90, 107, 116,
            110, 236, 53, 255, 5, 214, 126, 24, 216, 97, 199, 148, 239, 253, 102,
            199, 184, 232, 253, 158, 145, 86, 187, 112, 81, 78, 70, 80, 110, 33,
            37, 159, 233, 198, 1, 178, 108, 210, 100, 109, 155, 106, 124, 124, 83,
            89, 50, 197, 115, 231, 32, 74, 2, 92, 47, 25, 220, 135, 249, 122,
            172, 220, 137, 143, 234, 68, 188
    };

    Hash expected_hash = {
            199, 114, 154, 28, 4, 164, 196, 178, 117, 17, 148,
            203, 125, 228, 51, 145, 162, 222, 106, 202, 205,
            55, 244, 178, 94, 29, 248, 242, 98, 221, 158, 179
    };
    return test_input("test_xelis_input", input, sizeof(input), expected_hash);
  }
}

int xelis_runTests_v2()
{
  bool all_tests_passed = true;
  all_tests_passed &= xelis_tests_v2::test_zero_input();
  all_tests_passed &= xelis_tests_v2::test_xelis_input();

  if (all_tests_passed)
  {
    std::cout << "XELIS-HASH-V2: All tests passed!" << std::endl;
    return 0;
  }
  else
  {
    std::cout << "XELIS-HASH-V2: Some tests failed!" << std::endl;
    return 1;
  }
}
