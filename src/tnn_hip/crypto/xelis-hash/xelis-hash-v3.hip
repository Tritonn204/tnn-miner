#include "blake3-inline.hip.inc"
// #include "uint128-compat.hip.inc"

// ============================================================================
// Constants
// ============================================================================

constexpr size_t XELIS_MEMORY_SIZE_V3 = 531 * 128;
constexpr size_t XELIS_BUFFER_SIZE_V3 = XELIS_MEMORY_SIZE_V3 / 2;
constexpr uint16_t XELIS_SCRATCHPAD_ITERS_V3 = 2;
constexpr size_t XELIS_HASH_SIZE = 32;
constexpr size_t XELIS_TEMPLATE_SIZE = 112;

constexpr size_t XELIS_CHUNK_SIZE = 32;
constexpr size_t XELIS_OUTPUT_SIZE_V3 = XELIS_MEMORY_SIZE_V3 * 8;
constexpr size_t XELIS_BYTES_PER_CHUNK_V3 = XELIS_OUTPUT_SIZE_V3 / 4;

__constant__ uint64_t XELIS_MURMUR_CONST1 = 0xff51afd7ed558ccdULL;
__constant__ uint64_t XELIS_MURMUR_CONST2 = 0xc4ceb9fe1a85ec53ULL;
__constant__ uint64_t XELIS_GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;
__constant__ uint64_t XELIS_SCATTER_CONST = 0xd2b74407b1ce6e93ULL;
__constant__ uint64_t XELIS_PICK_HALF_BIT = (1ULL << 58);

#include "aes-inline.hip.inc"
constexpr uint8_t AES_KEY[16] = {'x', 'e', 'l', 'i', 's', 'h', 'a', 's', 'h', '-', 'p', 'o', 'w', '-', 'v', '3'};

#ifndef XELIS_MIN_WG
#define XELIS_MIN_WG 64
#endif

#ifndef XELIS_MAX_WG
#define XELIS_MAX_WG 1024
#endif

// ============================================================================
// Basic Device Functions
// ============================================================================

__device__ __forceinline__ uint64_t ROTL(uint64_t x, uint32_t r)
{
  r &= 63;
  return (x << r) | (x >> (64 - r));
}

__device__ __forceinline__ uint64_t ROTR(uint64_t x, uint32_t r)
{
  r &= 63;
  return (x >> r) | (x << (64 - r));
}

__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t r)
{
#if defined(__CUDA_ARCH__)
  // NVIDIA: Use funnel shift for arbitrary rotations (more efficient than shift+or)
  uint32_t result;
  asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(result) : "r"(x), "r"(r));
  return result;
#elif defined(__HIP_PLATFORM_AMD__)
  // AMD: Use v_alignbit_b32 instruction (funnel shift equivalent)
  // alignbit(hi, lo, shift) extracts 32 bits from the 64-bit value {hi, lo} >> shift
  // For left rotation by r: we want bits from position (32-r) in {x, x}
  return __builtin_amdgcn_alignbit(x, x, 32 - r);
#else
  return (x << r) | (x >> (32 - r));
#endif
}

__device__ __forceinline__ uint32_t rotl32_16(uint32_t x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  return __byte_perm(x, x, 0x1032);
#else
  return (x << 16) | (x >> 16);
#endif
}

__device__ __forceinline__ uint32_t rotl32_8(uint32_t x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  return __byte_perm(x, x, 0x2103);
#else
  return (x << 8) | (x >> 24);
#endif
}

__device__ __forceinline__ uint64_t murmurhash3(uint64_t seed)
{
  seed ^= seed >> 55;
  seed *= XELIS_MURMUR_CONST1;
  seed ^= seed >> 32;
  seed *= XELIS_MURMUR_CONST2;
  seed ^= seed >> 15;
  return seed;
}

__device__ __forceinline__ uint64_t map_index(uint64_t x)
{
  x ^= x >> 33;
  x *= XELIS_MURMUR_CONST1;
  return __umul64hi(x, XELIS_BUFFER_SIZE_V3);
}

__device__ __forceinline__ int pick_half(uint64_t v)
{
  return (murmurhash3(v) & XELIS_PICK_HALF_BIT) != 0;
}

__device__ __forceinline__ uint64_t isqrt(uint64_t n)
{
  if (n < 2)
    return n;

#ifdef __HIP_PLATFORM_AMD__

#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)

  double d = sqrt(__ull2double_rn(n));
  uint64_t root = __double2ull_rz(d);

  if ((root + 1) * (root + 1) <= n)
    ++root;
  else if (root * root > n)
    --root;

#else
  float x = __uint2float_rn(n >> 32) * 4294967296.0f + __uint2float_rn(n & 0xFFFFFFFFu);
  float root_f = __builtin_amdgcn_sqrtf(x);
  uint64_t root = __float2ull_rz(root_f);

  root = (root + n / root) >> 1;

  if (root * root > n)
    --root;
  else if ((root + 1) * (root + 1) <= n)
    ++root;
#endif

#elif defined(__CUDA_ARCH__)
  uint32_t hi = static_cast<uint32_t>(n >> 32);
  uint32_t lo = static_cast<uint32_t>(n & 0xFFFFFFFFu);

  float xf = __uint2float_rn(hi) * 4294967296.0f + __uint2float_rn(lo);

  float root_f = sqrtf(xf);
  uint64_t root = static_cast<uint64_t>(root_f);

  if (root == 0)
    root = 1;

  root = (root + n / root) >> 1;

  if (root * root > n)
    --root;
  else if ((root + 1) * (root + 1) <= n)
    ++root;

#else
  uint64_t lo = 1, hi = (n >> 1) < 0xFFFFFFFFULL ? (n >> 1) : 0xFFFFFFFFULL;

  while (lo <= hi)
  {
    uint64_t mid = (lo + hi) >> 1;
    uint64_t sq = mid * mid;

    if (sq == n)
      return mid;
    if (sq < n)
      lo = mid + 1;
    else
      hi = mid - 1;
  }

  uint64_t root = hi;
#endif

  return root;
}

__device__ __forceinline__ uint64_t modular_power(uint64_t base, uint64_t exp, uint64_t mod) {
    mod += (mod == 0);
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            __uint128_t full = (__uint128_t)result * base;
            result = (uint64_t)(full % mod);
        }
        __uint128_t full = (__uint128_t)base * base;
        base = (uint64_t)(full % mod);
        exp >>= 1;
    }
    return result;
}

__device__ __forceinline__ uint32_t le_bytes_to_uint32(const uint8_t *bytes)
{
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
         ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

// ============================================================================
// Optimized 128-bit Division
// ============================================================================

#define COMBINE_UINT64(high, low) (((__uint128_t)(high) << 64) | (low))
#define U128_HI(x) (uint64_t)((x) >> 64)
#define U128_LO(x) (uint64_t)(x)

__device__ __forceinline__ uint32_t estimate_quotient_fp32(__uint128_t dividend, __uint128_t divisor) {
    uint64_t dividend_hi = U128_HI(dividend);
    uint64_t divisor_hi = U128_HI(divisor);
    
    float d_hi = __uint2float_rn((uint32_t)(dividend_hi >> 32)) * 4294967296.0f + 
                 __uint2float_rn((uint32_t)(dividend_hi & 0xFFFFFFFF));
    float v_hi = __uint2float_rn((uint32_t)(divisor_hi >> 32)) * 4294967296.0f + 
                 __uint2float_rn((uint32_t)(divisor_hi & 0xFFFFFFFF));
    
    v_hi = fmaxf(v_hi, 1.0f);
    float q_float = d_hi / v_hi;
    return __float2uint_rz(q_float);
}

__device__ __forceinline__ uint64_t div128_optimized(__uint128_t dividend, __uint128_t divisor) {
    uint32_t q_est = estimate_quotient_fp32(dividend, divisor);
    return q_est;
}

__device__ __forceinline__ uint64_t mod128_optimized(__uint128_t dividend, __uint128_t divisor) {
    if (dividend < divisor) return U128_LO(dividend);
    uint32_t q_est = estimate_quotient_fp32(dividend, divisor);
    
    __uint128_t product = divisor * q_est;
    
    if (dividend < product) {
        q_est--;
        product = divisor * q_est;
    }
    
    __uint128_t remainder = dividend - product;
    
    if (remainder >= divisor) {
        remainder -= divisor;
    }
    
    return U128_LO(remainder);
}

// ============================================================================
// Streaming Store Intrinsics (Platform-Specific)
// ============================================================================

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
// NVIDIA: Use cache-streaming stores
__device__ __forceinline__ void store_streaming_u256(uint64_t* ptr,
                                                     uint64_t v0, uint64_t v1,
                                                     uint64_t v2, uint64_t v3)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "st.global.cs.v2.u64 [%0], {%1, %2};\n\t"
        "st.global.cs.v2.u64 [%3], {%4, %5};\n\t"
        :
        : "l"(ptr),
          "l"(v0), "l"(v1),
          "l"(ptr + 2),
          "l"(v2), "l"(v3)
        : "memory");
#else
    ptr[0] = v0;
    ptr[1] = v1;
    ptr[2] = v2;
    ptr[3] = v3;
#endif
}
#else
// AMD: Use non-temporal stores
__device__ __forceinline__ void store_streaming_u256(uint64_t *ptr,
                                                     uint64_t v0, uint64_t v1,
                                                     uint64_t v2, uint64_t v3)
{
  __builtin_nontemporal_store(v0, ptr);
  __builtin_nontemporal_store(v1, ptr + 1);
  __builtin_nontemporal_store(v2, ptr + 2);
  __builtin_nontemporal_store(v3, ptr + 3);
}
#endif

// ============================================================================
// ChaCha20 Implementation (Platform-Specific)
// ============================================================================

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)

// NVIDIA: Full ASM ChaCha20 with funnel shifts
__device__ void chacha20_block(uint32_t *__restrict__ output,
                               const uint32_t *__restrict__ input)
{
  uint32_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;

  // Load state into registers
  s0 = input[0];
  s1 = input[1];
  s2 = input[2];
  s3 = input[3];
  s4 = input[4];
  s5 = input[5];
  s6 = input[6];
  s7 = input[7];
  s8 = input[8];
  s9 = input[9];
  s10 = input[10];
  s11 = input[11];
  s12 = input[12];
  s13 = input[13];
  s14 = input[14];
  s15 = input[15];

  // Store original values for final addition
  const uint32_t i0 = s0, i1 = s1, i2 = s2, i3 = s3;
  const uint32_t i4 = s4, i5 = s5, i6 = s6, i7 = s7;
  const uint32_t i8 = s8, i9 = s9, i10 = s10, i11 = s11;
  const uint32_t i12 = s12, i13 = s13, i14 = s14, i15 = s15;

#pragma unroll 4
  for (int round = 0; round < 8; round += 2)
  {
    // Column round
    // QR(0, 4, 8, 12)
    s0 += s4;
    s12 ^= s0;
    s12 = __byte_perm(s12, s12, 0x1032);
    s8 += s12;
    s4 ^= s8;
    asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s4));
    s0 += s4;
    s12 ^= s0;
    s12 = __byte_perm(s12, s12, 0x2103);
    s8 += s12;
    s4 ^= s8;
    asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s4));

    // QR(1, 5, 9, 13)
    s1 += s5;
    s13 ^= s1;
    s13 = __byte_perm(s13, s13, 0x1032);
    s9 += s13;
    s5 ^= s9;
    asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s5));
    s1 += s5;
    s13 ^= s1;
    s13 = __byte_perm(s13, s13, 0x2103);
    s9 += s13;
    s5 ^= s9;
    asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s5));

    // QR(2, 6, 10, 14)
    s2 += s6;
    s14 ^= s2;
    s14 = __byte_perm(s14, s14, 0x1032);
    s10 += s14;
    s6 ^= s10;
    asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s6));
    s2 += s6;
    s14 ^= s2;
    s14 = __byte_perm(s14, s14, 0x2103);
    s10 += s14;
    s6 ^= s10;
    asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s6));

    // QR(3, 7, 11, 15)
    s3 += s7;
    s15 ^= s3;
    s15 = __byte_perm(s15, s15, 0x1032);
    s11 += s15;
    s7 ^= s11;
    asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s7));
    s3 += s7;
    s15 ^= s3;
    s15 = __byte_perm(s15, s15, 0x2103);
    s11 += s15;
    s7 ^= s11;
    asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s7));

    // Diagonal round
    // QR(0, 5, 10, 15)
    s0 += s5;
    s15 ^= s0;
    s15 = __byte_perm(s15, s15, 0x1032);
    s10 += s15;
    s5 ^= s10;
    asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s5));
    s0 += s5;
    s15 ^= s0;
    s15 = __byte_perm(s15, s15, 0x2103);
    s10 += s15;
    s5 ^= s10;
    asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s5));

    // QR(1, 6, 11, 12)
    s1 += s6;
    s12 ^= s1;
    s12 = __byte_perm(s12, s12, 0x1032);
    s11 += s12;
    s6 ^= s11;
    asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s6));
    s1 += s6;
    s12 ^= s1;
    s12 = __byte_perm(s12, s12, 0x2103);
    s11 += s12;
    s6 ^= s11;
    asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s6));

    // QR(2, 7, 8, 13)
    s2 += s7;
    s13 ^= s2;
    s13 = __byte_perm(s13, s13, 0x1032);
    s8 += s13;
    s7 ^= s8;
    asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s7));
    s2 += s7;
    s13 ^= s2;
    s13 = __byte_perm(s13, s13, 0x2103);
    s8 += s13;
    s7 ^= s8;
    asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s7));

    // QR(3, 4, 9, 14)
    s3 += s4;
    s14 ^= s3;
    s14 = __byte_perm(s14, s14, 0x1032);
    s9 += s14;
    s4 ^= s9;
    asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s4));
    s3 += s4;
    s14 ^= s3;
    s14 = __byte_perm(s14, s14, 0x2103);
    s9 += s14;
    s4 ^= s9;
    asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s4));
  }

  // Final addition and store
  output[0] = s0 + i0;
  output[1] = s1 + i1;
  output[2] = s2 + i2;
  output[3] = s3 + i3;
  output[4] = s4 + i4;
  output[5] = s5 + i5;
  output[6] = s6 + i6;
  output[7] = s7 + i7;
  output[8] = s8 + i8;
  output[9] = s9 + i9;
  output[10] = s10 + i10;
  output[11] = s11 + i11;
  output[12] = s12 + i12;
  output[13] = s13 + i13;
  output[14] = s14 + i14;
  output[15] = s15 + i15;
}

#else

// AMD: Standard ChaCha20 (compiler handles alignbit optimization)
#define CHACHA_QR(a, b, c, d) \
  a += b;                     \
  d ^= a;                     \
  d = rotl32_16(d);           \
  c += d;                     \
  b ^= c;                     \
  b = rotl32(b, 12);          \
  a += b;                     \
  d ^= a;                     \
  d = rotl32_8(d);            \
  c += d;                     \
  b ^= c;                     \
  b = rotl32(b, 7);

__device__ void chacha20_block(uint32_t *output, const uint32_t *input)
{
  uint32_t x[16];
#pragma unroll
  for (int i = 0; i < 16; i++)
    x[i] = input[i];

#pragma unroll
  for (int i = 0; i < 8; i += 2)
  {
    CHACHA_QR(x[0], x[4], x[8], x[12])
    CHACHA_QR(x[1], x[5], x[9], x[13])
    CHACHA_QR(x[2], x[6], x[10], x[14])
    CHACHA_QR(x[3], x[7], x[11], x[15])
    CHACHA_QR(x[0], x[5], x[10], x[15])
    CHACHA_QR(x[1], x[6], x[11], x[12])
    CHACHA_QR(x[2], x[7], x[8], x[13])
    CHACHA_QR(x[3], x[4], x[9], x[14])
  }

#pragma unroll
  for (int i = 0; i < 16; i++)
    output[i] = x[i] + input[i];
}

#endif

// ============================================================================
// ChaCha20 Init State (Common)
// ============================================================================

__device__ __forceinline__ void chacha20_init_state(uint32_t *state,
                                                    const uint8_t *key,
                                                    const uint8_t *nonce)
{
  state[0] = 0x61707865;
  state[1] = 0x3320646e;
  state[2] = 0x79622d32;
  state[3] = 0x6b206574;

#pragma unroll
  for (int i = 0; i < 8; i++)
    state[4 + i] = le_bytes_to_uint32(key + i * 4);

  state[12] = 0;
  state[13] = le_bytes_to_uint32(nonce);
  state[14] = le_bytes_to_uint32(nonce + 4);
  state[15] = le_bytes_to_uint32(nonce + 8);
}

// ============================================================================
// Stage 1 (Unified with Streaming Stores)
// ============================================================================

__device__ __forceinline__ void stage_1(
    const uint8_t *input,
    uint64_t *my_scratch,
    uint32_t block_size)
{
    uint8_t key[128];
    uint8_t K2[4][32];
    uint8_t nonces[4][12];
    uint8_t buffer[64];

    // Initialize key from input
    #pragma unroll
    for (int i = 0; i < XELIS_TEMPLATE_SIZE; i++) {
        key[i] = input[i];
    }
    #pragma unroll
    for (int i = XELIS_TEMPLATE_SIZE; i < 128; i++) {
        key[i] = 0;
    }

    // First hash to get nonce[0]
    blake3_hash_single_chunk(input, XELIS_TEMPLATE_SIZE, buffer);
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        nonces[0][i] = buffer[i];
    }

    // Derive K2[0]
    #pragma unroll
    for (int i = 0; i < XELIS_CHUNK_SIZE; i++) {
        buffer[XELIS_CHUNK_SIZE + i] = key[i];
    }
    blake3_hash_single_chunk(buffer, XELIS_CHUNK_SIZE * 2, K2[0]);

    // Derive K2[1..3]
    #pragma unroll
    for (int k = 1; k < 4; k++) {
        #pragma unroll
        for (int i = 0; i < XELIS_CHUNK_SIZE; i++) {
            buffer[i] = K2[k-1][i];
            buffer[XELIS_CHUNK_SIZE + i] = key[k * XELIS_CHUNK_SIZE + i];
        }
        blake3_hash_single_chunk(buffer, XELIS_CHUNK_SIZE * 2, K2[k]);
    }

    // Derive nonces[1..3] using ChaCha20
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        uint32_t state[16];
        chacha20_init_state(state, K2[k], nonces[k]);
        state[12] = (XELIS_BYTES_PER_CHUNK_V3 / 64) - 1;

        uint32_t block[16];
        chacha20_block(block, state);

        // Extract nonce from last 12 bytes of block
        uint8_t* block_bytes = (uint8_t*)block;
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            nonces[k + 1][i] = block_bytes[52 + i];
        }
    }

    // Generate scratchpad - all in registers, stream to memory
    const uint32_t blocks_per_chunk = XELIS_BYTES_PER_CHUNK_V3 / 64;

    for (uint32_t chunk = 0; chunk < 4; chunk++) {
        for (uint32_t blk = 0; blk < blocks_per_chunk; blk++) {
            uint32_t state[16];
            chacha20_init_state(state, K2[chunk], nonces[chunk]);
            state[12] = blk;

            uint32_t chacha_out[16];
            chacha20_block(chacha_out, state);

            uint32_t global_block = chunk * blocks_per_chunk + blk;
            uint64_t* dest = my_scratch + global_block * 8;
            uint64_t* out64 = (uint64_t*)chacha_out;

            // Streaming stores - bypass L1 cache
            store_streaming_u256(dest, out64[0], out64[1], out64[2], out64[3]);
            store_streaming_u256(dest + 4, out64[4], out64[5], out64[6], out64[7]);
        }
    }
}

// ============================================================================
// Cooperative Stage 1 (Bandwidth Optimized)
// ============================================================================
// Hard-capped at 32 threads for optimal efficiency
// Uses shared memory coordination for coalesced writes
constexpr int STAGE1_COOPERATIVE_BLOCK_SIZE = 32;
constexpr int STAGE1_SHARED_MEM_SIZE = STAGE1_COOPERATIVE_BLOCK_SIZE * 176; // 32 * (4*32 + 4*12)

__device__ __forceinline__ void stage_1_cooperative(
    const uint8_t *input,
    uint64_t *my_scratch,
    uint8_t *shared_nonces_mem)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t block_size = blockDim.x;

    // Shared memory layout: [K2: 4*32 bytes][nonces: 4*12 bytes] per thread
    uint8_t (*shared_K2)[4][32] = (uint8_t (*)[4][32])shared_nonces_mem;
    uint8_t (*shared_nonces)[4][12] = (uint8_t (*)[4][12])(shared_nonces_mem + block_size * 4 * 32);

    // Phase 1: Each thread computes its own K2 and nonces
    uint8_t key[128];
    uint8_t K2[4][32];
    uint8_t my_nonces[4][12];
    uint8_t buffer[64];

    // Initialize key
    #pragma unroll
    for (int i = 0; i < XELIS_TEMPLATE_SIZE; i++) {
        key[i] = input[i];
    }
    #pragma unroll
    for (int i = XELIS_TEMPLATE_SIZE; i < 128; i++) {
        key[i] = 0;
    }

    // First hash to get nonce[0]
    blake3_hash_single_chunk(input, XELIS_TEMPLATE_SIZE, buffer);
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        my_nonces[0][i] = buffer[i];
    }

    // Derive K2[0]
    #pragma unroll
    for (int i = 0; i < XELIS_CHUNK_SIZE; i++) {
        buffer[XELIS_CHUNK_SIZE + i] = key[i];
    }
    blake3_hash_single_chunk(buffer, XELIS_CHUNK_SIZE * 2, K2[0]);

    // Derive K2[1..3]
    #pragma unroll
    for (int k = 1; k < 4; k++) {
        #pragma unroll
        for (int i = 0; i < XELIS_CHUNK_SIZE; i++) {
            buffer[i] = K2[k-1][i];
            buffer[XELIS_CHUNK_SIZE + i] = key[k * XELIS_CHUNK_SIZE + i];
        }
        blake3_hash_single_chunk(buffer, XELIS_CHUNK_SIZE * 2, K2[k]);
    }

    // Derive nonces[1..3]
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        uint32_t state[16];
        chacha20_init_state(state, K2[k], my_nonces[k]);
        state[12] = (XELIS_BYTES_PER_CHUNK_V3 / 64) - 1;

        uint32_t block[16];
        chacha20_block(block, state);

        uint8_t* block_bytes = (uint8_t*)block;
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            my_nonces[k + 1][i] = block_bytes[52 + i];
        }
    }

    // Store K2 and nonces to shared memory
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            shared_K2[tid][k][i] = K2[k][i];
        }
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            shared_nonces[tid][k][i] = my_nonces[k][i];
        }
    }

    __syncthreads();

    // Phase 2: Cooperative scratchpad generation
    // All threads collaborate on EACH hash's scratchpad
    const uint32_t blocks_per_chunk = XELIS_BYTES_PER_CHUNK_V3 / 64;

    for (uint32_t hash_idx = 0; hash_idx < block_size; hash_idx++) {
        uint64_t *target_scratch = my_scratch - tid * XELIS_MEMORY_SIZE_V3 + hash_idx * XELIS_MEMORY_SIZE_V3;

        for (uint32_t chunk = 0; chunk < 4; chunk++) {
            // Strided loop: thread i generates blocks i, i+32, i+64, ...
            for (uint32_t blk = tid; blk < blocks_per_chunk; blk += block_size) {
                uint32_t state[16];
                chacha20_init_state(state, shared_K2[hash_idx][chunk], shared_nonces[hash_idx][chunk]);
                state[12] = blk;

                uint32_t chacha_out[16];
                chacha20_block(chacha_out, state);

                uint32_t global_block = chunk * blocks_per_chunk + blk;
                uint64_t* dest = target_scratch + global_block * 8;
                uint64_t* out64 = (uint64_t*)chacha_out;

                // Coalesced writes across threads
                store_streaming_u256(dest, out64[0], out64[1], out64[2], out64[3]);
                store_streaming_u256(dest + 4, out64[4], out64[5], out64[6], out64[7]);
            }
        }

        __syncthreads();
    }
}

__device__ __forceinline__ uint64_t execute_op_offset(
    uint32_t offset_op, uint64_t a, uint64_t b, uint64_t c,
    uint32_t r, uint64_t result, uint32_t i, uint32_t j)
{
  uint64_t t0, t1, t2, t3, t4, t5, t6;
  
  __uint128_t w0, w1;

  if (offset_op < 7)
  {
    t0 = a * b;
    t1 = a * c;
    t2 = b * c;

    t3 = t1 + t2;
    t4 = t0 - t1;
    t5 = c - a + b;
    t6 = a - b + c;
    uint64_t r4 = t2 + a;
    uint64_t r5 = t1 + b;
    uint64_t r6 = t0 * c;

    uint64_t m0 = -(uint64_t)(offset_op == 0);
    uint64_t m1 = -(uint64_t)(offset_op == 1);
    uint64_t m2 = -(uint64_t)(offset_op == 2);
    uint64_t m3 = -(uint64_t)(offset_op == 3);
    uint64_t m4 = -(uint64_t)(offset_op == 4);
    uint64_t m5 = -(uint64_t)(offset_op == 5);
    uint64_t m6 = -(uint64_t)(offset_op == 6);

    return (t3 & m0) | (t4 & m1) | (t5 & m2) | (t6 & m3) |
           (r4 & m4) | (r5 & m5) | (r6 & m6);
  }

  switch (offset_op)
  {
  case 7:
  {
    w0 = COMBINE_UINT64(a, b);
    return (uint64_t)(w0 % (c | 1));
  }
  case 8:
  {
    t0 = ROTL(result, r); \
    w0 = COMBINE_UINT64(t0, a | 2); \
    w1 = COMBINE_UINT64(b, c); \
    return (w1 < w0) ? c : mod128_optimized(w1, w0); \
  }
  case 9:
  {
    w0 = COMBINE_UINT64(c, a);
    return (uint64_t)(w0 / (b | 4));
  }
  case 10:
  {
    t0 = ROTL(result, r);
    w0 = COMBINE_UINT64(t0, b);
    w1 = COMBINE_UINT64(a, c | 8);
    return (w0 > w1) ? div128_optimized(w0, w1) : (a ^ b);
  }
  case 11:
  {
    t0 = __umul64hi(a, c);
    t1 = b * c;
    return t0 + t1;
  }
  case 12:
  {
    t0 = __umul64hi(c, b);
    t1 = a * b;
    t2 = ROTR(result, r);
    t3 = c * t2;
    return t0 + t1 + t3;
  }
  case 13:
  {
    t0 = isqrt(b + j);
    t1 = murmurhash3(c ^ result ^ i ^ j) | 1;
    w0 = COMBINE_UINT64(a + i, t0);
    return (uint64_t)(w0 % t1);
  }
  case 14:
  {
    t0 = isqrt(b | 2);
    t1 = isqrt(a + j);
    t2 = (t0 > 0) ? (c + i) % t0 : 0;
    t3 = ROTL(t2, i + j);
    return t3 * t1;
  }
  default:
  {
    t0 = isqrt(a + i);
    t1 = isqrt(c + j);
    t2 = t0 * t1;
    t3 = b + i + j;
    return t2 ^ t3;
  }
  }
}

#if defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_NVIDIA__)
// NVIDIA: Use PTX inline assembly prefetch - CRITICAL for performance
__device__ __forceinline__ void prefetch_l1(const void *ptr)
{
  asm volatile("prefetch.global.L1 [%0];" ::"l"(ptr));
}

__device__ __forceinline__ void prefetch_l2(const void *ptr)
{
  asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

#define XELIS_PREFETCH_ENABLED 1
#define PREFETCH_READ_L1(ptr) prefetch_l1(ptr)
#define PREFETCH_READ_L2(ptr) prefetch_l2(ptr)
#define PREFETCH_WRITE_L2(ptr) prefetch_l2(ptr)

#else
#define XELIS_PREFETCH_ENABLED 0
#define PREFETCH_READ_L1(ptr) ((void)0)
#define PREFETCH_READ_L2(ptr) ((void)0)
#define PREFETCH_WRITE_L2(ptr) ((void)0)

#endif

// ============================================================================
// Stage 3 - Memory-hard scratchpad iteration
// ============================================================================

__device__ void stage_3(uint64_t *my_scratch)
{
  uint64_t *mem_buffer_a = my_scratch;
  uint64_t *mem_buffer_b = my_scratch + XELIS_BUFFER_SIZE_V3;

  uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
  uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;
  uint32_t r = 0;

#pragma unroll
  for (uint32_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; i++)
  {
    uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
    uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];

    uint64_t hash1 = mem_b;
    uint64_t hash2 = mem_a;
    aes_round_u64(&hash1, &hash2, AES_KEY);

    uint64_t result = ~(hash1 ^ hash2);

    for (uint32_t j = 0; j < XELIS_BUFFER_SIZE_V3; j++)
    {
      uint64_t idx_a_read = map_index(result);
      uint64_t a = mem_buffer_a[idx_a_read];

      uint64_t idx_b_read = map_index(a ^ ~ROTR(result, r));

      uint64_t b_val = mem_buffer_b[idx_b_read];
      uint64_t c = my_scratch[r];

      r = (r < XELIS_MEMORY_SIZE_V3 - 1) ? r + 1 : 0;

      uint32_t op_idx = (ROTL(result, (uint32_t)c) + 13) & 0xF;
      uint64_t v = execute_op_offset(op_idx, a, b_val, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      int use_buffer_b = pick_half(v);
      uint64_t idx_t = map_index(idx_seed);

      uint64_t t = (use_buffer_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

      uint64_t idx_a_write = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
      uint64_t idx_b_write = map_index(idx_a_write ^ ~result ^ XELIS_SCATTER_CONST);

#if XELIS_PREFETCH_ENABLED
      // Prefetch write locations
      PREFETCH_WRITE_L2(&mem_buffer_a[idx_a_write]);
      PREFETCH_WRITE_L2(&mem_buffer_b[idx_b_write]);
#endif

      uint64_t old_mem_a = mem_buffer_a[idx_a_write];
      mem_buffer_a[idx_a_write] = t;
      mem_buffer_b[idx_b_write] ^= old_mem_a ^ ROTR(t, i + j);
    }

    addr_a = modular_power(addr_a, addr_b, result);
    addr_b = isqrt(result) * (r + 1) * isqrt(addr_a);
  }
}

// ============================================================================
// Stage 3 Efficient - Optimized prefetch strategy (Split+Eff)
// ============================================================================
// Removes wasteful prefetches, downgrades write prefetches to L2
// - Sequential access handled by HW prefetcher
// - L2 prefetch sufficient for writes (avoids L1 cache churn)
// - Critical dependent-load prefetches preserved

__device__ void stage_3_efficient(uint64_t *my_scratch)
{
  uint64_t *mem_buffer_a = my_scratch;
  uint64_t *mem_buffer_b = my_scratch + XELIS_BUFFER_SIZE_V3;

  uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
  uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;
  uint32_t r = 0;

#pragma unroll
  for (uint32_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; i++)
  {
    uint64_t idx_a0 = map_index(addr_a);

#if XELIS_PREFETCH_ENABLED
    // Critical prefetch: dependent load (addr_b depends on mem_a)
    PREFETCH_READ_L1(&mem_buffer_a[idx_a0]);
#endif

    uint64_t mem_a = mem_buffer_a[idx_a0];
    uint64_t idx_b0 = map_index(mem_a ^ addr_b);

#if XELIS_PREFETCH_ENABLED
    // Critical prefetch: dependent load (random access via hash)
    PREFETCH_READ_L1(&mem_buffer_b[idx_b0]);
#endif

    uint64_t mem_b = mem_buffer_b[idx_b0];

    uint64_t hash1 = mem_b;
    uint64_t hash2 = mem_a;
    aes_round_u64(&hash1, &hash2, AES_KEY);

    uint64_t result = ~(hash1 ^ hash2);

    for (uint32_t j = 0; j < XELIS_BUFFER_SIZE_V3; j++)
    {
      // REMOVED: Loop prefetch for my_scratch[r_pf]
      // Sequential access pattern, HW prefetcher handles it automatically

      uint64_t idx_a_read = map_index(result);
      uint64_t a = mem_buffer_a[idx_a_read];

      uint64_t idx_b_read = map_index(a ^ ~ROTR(result, r));
      uint64_t b_val = mem_buffer_b[idx_b_read];
      uint64_t c = my_scratch[r];

      r = (r < XELIS_MEMORY_SIZE_V3 - 1) ? r + 1 : 0;

      uint32_t op_idx = (ROTL(result, (uint32_t)c) + 13) & 0xF;
      uint64_t v = execute_op_offset(op_idx, a, b_val, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      int use_buffer_b = pick_half(v);
      uint64_t idx_t = map_index(idx_seed);

      uint64_t t = (use_buffer_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

      uint64_t idx_a_write = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
      uint64_t idx_b_write = map_index(idx_a_write ^ ~result ^ XELIS_SCATTER_CONST);

#if XELIS_PREFETCH_ENABLED
      // CHANGED: L1 → L2 for write prefetches (reduces cache churn)
      // Writes evict to L2 anyway, L1 prefetch wastes power
      PREFETCH_WRITE_L2(&mem_buffer_a[idx_a_write]);
      PREFETCH_WRITE_L2(&mem_buffer_b[idx_b_write]);
#endif

      uint64_t old_mem_a = mem_buffer_a[idx_a_write];
      mem_buffer_a[idx_a_write] = t;
      mem_buffer_b[idx_b_write] ^= old_mem_a ^ ROTR(t, i + j);
    }

    addr_a = modular_power(addr_a, addr_b, result);
    addr_b = isqrt(result) * (r + 1) * isqrt(addr_a);
  }
}

// GPU-side difficulty check: Compare hash <= target
__device__ __forceinline__ bool check_difficulty_gpu(const uint8_t *hash, const uint64_t *target)
{
  const uint8_t *target_bytes = (const uint8_t *)target;

  for (int i = 0; i < 32; i++)
  {
    uint8_t hash_byte = hash[i];
    uint8_t target_byte = target_bytes[i];

    if (hash_byte < target_byte)
    {
      return true;
    }
    else if (hash_byte > target_byte)
    {
      return false;
    }
  }

  return true;
}

// ============================================================================
// SPLIT KERNELS (Optimized scheduling)
// ============================================================================

// Stage 1 only kernel - prepares scratchpad (standard)
// #if defined(__HIP_PLATFORM_AMD__)
// __attribute__((amdgpu_flat_work_group_size(XELIS_MIN_WG, XELIS_MAX_WG)))
// #endif
extern "C" __global__ void xelis_stage1_kernel(
    const uint8_t *__restrict__ input_template,
    uint64_t *__restrict__ scratch_memory,
    uint64_t nonce_start,
    uint32_t batch_size,
    uint32_t scratch_offset)
{
  uint32_t local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_tid >= batch_size)
    return;

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;

  uint8_t input[XELIS_TEMPLATE_SIZE];
  memcpy(input, input_template, XELIS_TEMPLATE_SIZE);

  uint64_t nonce = nonce_start + local_tid;
  ((uint64_t *)(input))[5] = nonce;

  stage_1(input, my_scratch, blockDim.x);
}

// Stage 1 Cooperative kernel - bandwidth optimized (hard-capped at 32 threads)
// Uses shared memory coordination for coalesced writes
extern "C" __global__ void xelis_stage1_cooperative(
    const uint8_t *__restrict__ input_template,
    uint64_t *__restrict__ scratch_memory,
    uint64_t nonce_start,
    uint32_t batch_size,
    uint32_t scratch_offset)
{
  // Hard-capped at 32 threads per block
  __shared__ uint8_t shared_mem[STAGE1_SHARED_MEM_SIZE];

  uint32_t local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_tid >= batch_size)
    return;

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;

  uint8_t input[XELIS_TEMPLATE_SIZE];
  memcpy(input, input_template, XELIS_TEMPLATE_SIZE);

  uint64_t nonce = nonce_start + local_tid;
  ((uint64_t *)(input))[5] = nonce;

  stage_1_cooperative(input, my_scratch, shared_mem);
}

// Fused Stage 3 + Blake3 + Difficulty check kernel (standard)
#if defined(__HIP_PLATFORM_AMD__)
__attribute__((amdgpu_flat_work_group_size(XELIS_MIN_WG, XELIS_MAX_WG)))
#endif
extern "C" __global__ void xelis_s3_fused_kernel(
    uint64_t *__restrict__ scratch_memory,
    uint8_t *__restrict__ outputs,
    uint64_t nonce_start,
    uint32_t batch_size,
    uint32_t scratch_offset,
    const uint64_t *__restrict__ difficulty_target,
    uint64_t *__restrict__ solutions)
{
  uint32_t local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_tid >= batch_size)
    return;

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;
  uint8_t *my_output = outputs + global_idx * XELIS_HASH_SIZE;

  // Execute Stage 3
  stage_3(my_scratch);

  // Execute Blake3 finalization
  blake3_hasher_hash((uint8_t *)my_scratch, XELIS_OUTPUT_SIZE_V3, my_output);

  // Check difficulty and record solutions
  if (difficulty_target && solutions && check_difficulty_gpu(my_output, difficulty_target))
  {
    uint64_t nonce = nonce_start + local_tid;
    unsigned long long idx = atomicAdd((unsigned long long *)&solutions[0], 1ULL);

    if (idx < 1024)
    {
      uint64_t *solution_slot = &solutions[1 + idx * 5];
      solution_slot[0] = nonce;
      uint8_t *hash_dest = (uint8_t *)&solution_slot[1];
      memcpy(hash_dest, my_output, 32);
    }
  }
}

// Fused Stage 3 Efficient + Blake3 + Difficulty check kernel (optimized prefetch)
#if defined(__HIP_PLATFORM_AMD__)
__attribute__((amdgpu_flat_work_group_size(XELIS_MIN_WG, XELIS_MAX_WG)))
#endif
extern "C" __global__ void xelis_s3_efficient_kernel(
    uint64_t *__restrict__ scratch_memory,
    uint8_t *__restrict__ outputs,
    uint64_t nonce_start,
    uint32_t batch_size,
    uint32_t scratch_offset,
    const uint64_t *__restrict__ difficulty_target,
    uint64_t *__restrict__ solutions)
{
  uint32_t local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_tid >= batch_size)
    return;

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;
  uint8_t *my_output = outputs + global_idx * XELIS_HASH_SIZE;

  // Execute Stage 3 Efficient (optimized prefetch strategy)
  stage_3_efficient(my_scratch);

  // Execute Blake3 finalization
  blake3_hasher_hash((uint8_t *)my_scratch, XELIS_OUTPUT_SIZE_V3, my_output);

  // Check difficulty and record solutions
  if (difficulty_target && solutions && check_difficulty_gpu(my_output, difficulty_target))
  {
    uint64_t nonce = nonce_start + local_tid;
    unsigned long long idx = atomicAdd((unsigned long long *)&solutions[0], 1ULL);

    if (idx < 1024)
    {
      uint64_t *solution_slot = &solutions[1 + idx * 5];
      solution_slot[0] = nonce;
      uint8_t *hash_dest = (uint8_t *)&solution_slot[1];
      memcpy(hash_dest, my_output, 32);
    }
  }
}

// ============================================================================
// ORIGINAL COMBINED KERNEL (API compatible - legacy)
// ============================================================================

#if defined(__HIP_PLATFORM_AMD__)
__attribute__((amdgpu_flat_work_group_size(XELIS_MIN_WG, XELIS_MAX_WG)))
#endif
extern "C" __global__ void xelis_hash_v3_kernel(
    const uint8_t *__restrict__ input_template,
    uint8_t *__restrict__ outputs,
    uint64_t *__restrict__ scratch_memory,
    uint64_t nonce_start,
    uint32_t batch_size,
    const uint64_t *__restrict__ difficulty_target,
    uint64_t *__restrict__ solutions)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= batch_size)
    return;

  uint64_t *my_scratch = scratch_memory + (size_t)tid * XELIS_MEMORY_SIZE_V3;
  uint8_t *my_output = outputs + tid * XELIS_HASH_SIZE;

  uint8_t input[XELIS_TEMPLATE_SIZE];
  memcpy(input, input_template, XELIS_TEMPLATE_SIZE);

  uint64_t nonce = nonce_start + tid;
  ((uint64_t *)(input))[5] = nonce;

  stage_1(input, my_scratch, blockDim.x);

  stage_3(my_scratch);
  blake3_hasher_hash((uint8_t *)my_scratch, XELIS_OUTPUT_SIZE_V3, my_output);

  if (difficulty_target && solutions && check_difficulty_gpu(my_output, difficulty_target))
  {
    unsigned long long idx = atomicAdd((unsigned long long *)&solutions[0], 1ULL);

    if (idx < 1024)
    {
      uint64_t *solution_slot = &solutions[1 + idx * 5];
      solution_slot[0] = nonce;
      uint8_t *hash_dest = (uint8_t *)&solution_slot[1];
      memcpy(hash_dest, my_output, 32);
    }
  }
}