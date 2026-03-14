#include "blake3-inline.hip.inc"
#include "aes-inline.hip.inc"
// #include "uint128-compat.hip.inc"

// ============================================================================
// Constants
// ============================================================================

constexpr size_t XELIS_MEMORY_SIZE_V3 = 531 * 128;
constexpr size_t XELIS_BUFFER_SIZE_V3 = XELIS_MEMORY_SIZE_V3 / 2;
constexpr uint16_t XELIS_SCRATCHPAD_ITERS_V3 = 2;
constexpr size_t XELIS_HASH_SIZE = 32;
constexpr size_t XELIS_TEMPLATE_SIZE = 112;
constexpr size_t XELIS_NONCE_STORAGE_OFFSET = XELIS_MEMORY_SIZE_V3;  // Extra slot after scratch for nonce

constexpr size_t XELIS_CHUNK_SIZE = 32;
constexpr size_t XELIS_OUTPUT_SIZE_V3 = XELIS_MEMORY_SIZE_V3 * 8;
constexpr size_t XELIS_BYTES_PER_CHUNK_V3 = XELIS_OUTPUT_SIZE_V3 / 4;

__constant__ uint64_t XELIS_MURMUR_CONST1 = 0xff51afd7ed558ccdULL;
__constant__ uint64_t XELIS_MURMUR_CONST2 = 0xc4ceb9fe1a85ec53ULL;
__constant__ uint64_t XELIS_GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;
__constant__ uint64_t XELIS_SCATTER_CONST = 0xd2b74407b1ce6e93ULL;
__constant__ uint64_t XELIS_PICK_HALF_BIT = (1ULL << 58);

__device__ constexpr uint8_t AES_KEY[16] = {'x', 'e', 'l', 'i', 's', 'h', 'a', 's', 'h', '-', 'p', 'o', 'w', '-', 'v', '3'};

// Hot path hint — supported by hipclang, silently ignored by NVRTC
#if defined(__HIP_PLATFORM_AMD__) || defined(__clang__)
#define XELIS_HOT __attribute__((hot))
#else
#define XELIS_HOT
#endif

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
  uint32_t result;
  asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(result) : "r"(x), "r"(r));
  return result;
#elif defined(__HIP_PLATFORM_AMD__)
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

__device__ __forceinline__ uint64_t isqrt(uint64_t a)
{
    if (a < 2) return a;

    uint64_t arg = a;

    // Normalize to even shift
    uint32_t scal = __clzll(a) & ~1;
    a <<= scal;
    uint32_t b = (uint32_t)(a >> 32);

    // rsqrt seed
    float fr = __frsqrt_rn((float)b);
    uint32_t r = (uint32_t)fmaf(1.407374884e14f, fr, -438.0f);

    // sqrt ≈ input × rsqrt
    uint32_t s = __umulhi(r, b);  // 32×32→high32, works on both
    s = s * 2;

    // Newton-Raphson — compiler emits optimal wide multiply for (uint64_t)s * s
    uint64_t rem = a - (uint64_t)s * s;
    r = __umulhi((uint32_t)(rem >> 32) + 1, r);
    s = s + r;

    // Denormalize
    s = s >> (scal >> 1);

    // Correction
    uint64_t sq = (uint64_t)s * s;
    if (sq > arg) { s--; sq = (uint64_t)s * s; }

    sq = (uint64_t)(s + 1) * (s + 1);
    if (sq <= arg) { s++; sq = (uint64_t)(s + 1) * (s + 1); }

    return s;
}

__device__ __forceinline__ uint64_t barrett_reduce(
    uint64_t x_lo, uint64_t x_hi,
    uint64_t mod,
    uint64_t mu_lo, uint64_t mu_hi)
{
    // q = high 128 bits of (x * mu)
    uint64_t p0_hi = __umul64hi(x_lo, mu_lo);
    
    uint64_t p1_lo = x_lo * mu_hi;
    uint64_t p1_hi = __umul64hi(x_lo, mu_hi);
    
    uint64_t p2_lo = x_hi * mu_lo;
    uint64_t p2_hi = __umul64hi(x_hi, mu_lo);
    
    uint64_t p3_lo = x_hi * mu_hi;
    
    // Middle column + carry
    uint64_t mid = p0_hi + p1_lo;
    uint64_t c1 = (mid < p0_hi);
    mid += p2_lo;
    uint64_t c2 = (mid < p2_lo);
    
    uint64_t q = p3_lo + p1_hi + p2_hi + c1 + c2;
    
    // r = x - q*mod (128-bit subtract)
    uint64_t qm_lo = q * mod;
    uint64_t qm_hi = __umul64hi(q, mod);
    
    uint64_t r_lo = x_lo - qm_lo;
    uint64_t borrow = (x_lo < qm_lo);
    uint64_t r_hi = x_hi - qm_hi - borrow;
    
    // Single correction
    if (r_hi | (r_lo >= mod)) {
        r_lo -= mod;
    }
    
    return r_lo;
}

// Forward declarations for hwrecip functions (defined later, needed by Barrett setup)
__device__ __forceinline__ uint64_t div64_hwrecip(uint64_t dividend, uint64_t divisor);
__device__ __forceinline__ uint64_t mod64_hwrecip(uint64_t a, uint64_t d);
__device__ __forceinline__ uint64_t div128by64_hwrecip(uint64_t hi, uint64_t lo, uint64_t d);

__device__ __forceinline__ uint64_t modular_power(
    uint64_t base, uint64_t exp, uint64_t mod)
{
    mod += (mod == 0);
    if (mod == 1) return 0;
    
    base %= mod;
    if (base < 2) return base;
    if (exp == 0) return 1;
    
    // Barrett constant: mu = floor((2^128 - 1) / mod), split into (mu_hi, mu_lo).
    // Two-step 128/64 division using hwrecip — no __uint128_t needed.
    uint64_t mu_hi = div64_hwrecip(0xFFFFFFFFFFFFFFFFULL, mod);
    uint64_t rem   = mod64_hwrecip(0xFFFFFFFFFFFFFFFFULL, mod);
    uint64_t mu_lo = div128by64_hwrecip(rem, 0xFFFFFFFFFFFFFFFFULL, mod);
    
    uint64_t result = 1;
    
    while (exp > 0) {
        if (exp & 1) {
            uint64_t prod_lo = result * base;
            uint64_t prod_hi = __umul64hi(result, base);
            result = barrett_reduce(prod_lo, prod_hi, mod, mu_lo, mu_hi);
        }
        exp >>= 1;
        if (exp > 0) {
            uint64_t prod_lo = base * base;
            uint64_t prod_hi = __umul64hi(base, base);
            base = barrett_reduce(prod_lo, prod_hi, mod, mu_lo, mu_hi);
            if (base == 0) return 0;
        }
    }
    
    return result;
}

__device__ __forceinline__ uint32_t le_bytes_to_uint32(const uint8_t *bytes)
{
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
         ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

// Lean Blake3 implementation for Xelis (531 chunks, fully unrolled)
__device__ __forceinline__ void xelis_blake3_lean(const uint8_t* input, uint8_t* output)
{
    constexpr int NUM_CHUNKS = 531;
    constexpr uint8_t FULL_BLEN = BLAKE3_BLOCK_LEN;

    uint32_t cv_stack[10][8];
    int stack_len = 0;

    uint8_t left_bytes[BLAKE3_OUT_LEN];
    uint8_t right_bytes[BLAKE3_OUT_LEN];
    uint8_t parent_bytes[BLAKE3_OUT_LEN];

    // Leaves + incremental merges
    for (uint64_t chunk_idx = 0; chunk_idx < (uint64_t)NUM_CHUNKS; chunk_idx++) {
        const uint8_t* chunk_ptr = input + (size_t)chunk_idx * (size_t)BLAKE3_CHUNK_LEN;

        uint32_t cv[8];
        blake3_cv_init(cv);

        // 16 full blocks
        blake3_compress_in_place(cv, chunk_ptr, FULL_BLEN, chunk_idx, (uint8_t)CHUNK_START);

        #pragma unroll
        for (int b = 1; b < 15; b++) {
            blake3_compress_in_place(cv, chunk_ptr + b * BLAKE3_BLOCK_LEN, FULL_BLEN, chunk_idx, (uint8_t)0);
        }

        blake3_compress_in_place(cv, chunk_ptr + 15 * BLAKE3_BLOCK_LEN, FULL_BLEN, chunk_idx, (uint8_t)CHUNK_END);

        // Push to stack
        #pragma unroll
        for (int i = 0; i < 8; i++) cv_stack[stack_len][i] = cv[i];
        stack_len++;

        // Merges: ctz(chunk_idx+1), capped to 9
        uint32_t tz = __ffsll(chunk_idx + 1ULL) - 1;
        tz = min(tz, 9u);

        // Fixed merge attempts
        #pragma unroll
        for (int m = 0; m < 10; m++) {
            int do_merge = (m < (int)tz) & (stack_len >= 2);

            int idx_r = max(stack_len - 1, 0);
            int idx_l = max(stack_len - 2, 0);

            uint32_t right_w[8], left_w[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                right_w[i] = cv_stack[idx_r][i];
                left_w[i]  = cv_stack[idx_l][i];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                blake3_store32(left_bytes  + i * 4, left_w[i]);
                blake3_store32(right_bytes + i * 4, right_w[i]);
            }

            blake3_hash_parent(left_bytes, right_bytes, parent_bytes, false);

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint32_t parent_w = blake3_load32(parent_bytes + i * 4);
                cv_stack[idx_l][i] = do_merge ? parent_w : cv_stack[idx_l][i];
            }

            stack_len -= do_merge;
        }
    }

    // Final reduction to 2
    #pragma unroll
    for (int t = 0; t < 10; t++) {
        int do_merge = (stack_len > 2);

        int idx_r = max(stack_len - 1, 0);
        int idx_l = max(stack_len - 2, 0);

        uint32_t right_w[8], left_w[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            right_w[i] = cv_stack[idx_r][i];
            left_w[i]  = cv_stack[idx_l][i];
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            blake3_store32(left_bytes  + i * 4, left_w[i]);
            blake3_store32(right_bytes + i * 4, right_w[i]);
        }

        blake3_hash_parent(left_bytes, right_bytes, parent_bytes, false);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t parent_w = blake3_load32(parent_bytes + i * 4);
            cv_stack[idx_l][i] = do_merge ? parent_w : cv_stack[idx_l][i];
        }

        stack_len -= do_merge;
    }

    // Final ROOT merge
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        blake3_store32(left_bytes  + i * 4, cv_stack[0][i]);
        blake3_store32(right_bytes + i * 4, cv_stack[1][i]);
    }

    blake3_hash_parent(left_bytes, right_bytes, parent_bytes, true);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        blake3_store32(output + i * 4, blake3_load32(parent_bytes + i * 4));
    }
}


// ============================================================================
// Optimized 128-bit Division
// ============================================================================

// COMBINE_UINT64 / U128 macros removed — all 128-bit ops use hwrecip functions

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
// Cache Control Intrinsics (Platform-Specific)
// ============================================================================

#if defined(__HIP_PLATFORM_AMD__)

// Standard load (default caching)
__device__ __forceinline__ uint64_t load_default(const uint64_t *ptr)
{
  return *ptr;
}

// GLC bit - Bypass L2 cache (reduces prefetcher activity)
__device__ __forceinline__ uint64_t load_glc(const uint64_t *ptr)
{
  uint64_t val;
  asm volatile(
      "global_load_dwordx2 %0, %1, off glc\n\t"
      "s_waitcnt vmcnt(0)"
      : "=v"(val)
      : "v"(ptr)
      : "memory");
  return val;
}

// SLC bit - Streaming/non-temporal hint
__device__ __forceinline__ uint64_t load_slc(const uint64_t *ptr)
{
  uint64_t val;
  asm volatile(
      "global_load_dwordx2 %0, %1, off slc\n\t"
      "s_waitcnt vmcnt(0)"
      : "=v"(val)
      : "v"(ptr)
      : "memory");
  return val;
}

// GLC + SLC - Full uncached
__device__ __forceinline__ uint64_t load_uncached(const uint64_t *ptr)
{
  uint64_t val;
  asm volatile(
      "global_load_dwordx2 %0, %1, off glc slc\n\t"
      "s_waitcnt vmcnt(0)"
      : "=v"(val)
      : "v"(ptr)
      : "memory");
  return val;
}

// Uncached store
__device__ __forceinline__ void store_glc_slc(uint64_t *ptr, uint64_t val)
{
  asm volatile(
      "global_store_dwordx2 %0, %1, off glc slc"
      :
      : "v"(ptr), "v"(val)
      : "memory");
}

#else  // NVIDIA

// Bypass L1, cache at L2 only (ld.global.cg)
__device__ __forceinline__ uint64_t load_uncached(const uint64_t *ptr)
{
  uint64_t val;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(val) : "l"(ptr) : "memory");
  return val;
}

// Standard load
__device__ __forceinline__ uint64_t load_default(const uint64_t *ptr)
{
  return *ptr;
}

// Bypass L1 store (st.global.cg)
__device__ __forceinline__ void store_glc_slc(uint64_t *ptr, uint64_t val)
{
  asm volatile("st.global.cg.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");
}

#endif  // Cache control intrinsics

// ============================================================================
// Large-L2 scratch access — cached loads/stores for GPUs with big L2
// ============================================================================
// NVIDIA: A100/CMP170HX (sm_80=40MB), RTX 40xx (sm_89=24-72MB), H100 (sm_90=50MB)
// AMD:    MI300 (gfx940/941/942)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 900)
  #define LARGE_L2 1
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  #define LARGE_L2 1
#else
  #define LARGE_L2 0
#endif

#if LARGE_L2
  __device__ __forceinline__ uint64_t load_scratch(const uint64_t *p) { return *p; }
  __device__ __forceinline__ void store_scratch(uint64_t *p, uint64_t v) { *p = v; }
#else
  __device__ __forceinline__ uint64_t load_scratch(const uint64_t *p) { return load_uncached(p); }
  __device__ __forceinline__ void store_scratch(uint64_t *p, uint64_t v) { store_glc_slc(p, v); }
#endif

// HBM detection — bandwidth-rich GPUs can afford speculative dual-loads
// NVIDIA: A100/CMP170HX (sm_80), H100 (sm_90) — NOT RTX 40xx (GDDR6X)
// AMD:    Vega (gfx900), RVII (gfx906), MI200 (gfx90a), MI300 (gfx940/941/942)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900)
  #define HAS_HBM 1
#elif defined(__gfx900__) || defined(__gfx906__) || defined(__gfx90a__) || \
      defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  #define HAS_HBM 1
#else
  #define HAS_HBM 0
#endif

// ============================================================================
// ChaCha20 Implementation (Platform-Specific)
// ============================================================================

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)

// NVIDIA: Full ASM ChaCha20 with funnel shifts
__device__ void chacha20_block(uint32_t* __restrict__ output, 
                                         const uint32_t* __restrict__ input) {
    uint32_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
    
    // Load state into registers
    s0 = input[0]; s1 = input[1]; s2 = input[2]; s3 = input[3];
    s4 = input[4]; s5 = input[5]; s6 = input[6]; s7 = input[7];
    s8 = input[8]; s9 = input[9]; s10 = input[10]; s11 = input[11];
    s12 = input[12]; s13 = input[13]; s14 = input[14]; s15 = input[15];

    // Store original values for final addition
    const uint32_t i0 = s0, i1 = s1, i2 = s2, i3 = s3;
    const uint32_t i4 = s4, i5 = s5, i6 = s6, i7 = s7;
    const uint32_t i8 = s8, i9 = s9, i10 = s10, i11 = s11;
    const uint32_t i12 = s12, i13 = s13, i14 = s14, i15 = s15;

    #pragma unroll 4
    for (int round = 0; round < 8; round += 2) {
        // Column round
        // QR(0, 4, 8, 12)
        s0 += s4; s12 ^= s0; s12 = __byte_perm(s12, s12, 0x1032);
        s8 += s12; s4 ^= s8; asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s4));
        s0 += s4; s12 ^= s0; s12 = __byte_perm(s12, s12, 0x2103);
        s8 += s12; s4 ^= s8; asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s4));

        // QR(1, 5, 9, 13)
        s1 += s5; s13 ^= s1; s13 = __byte_perm(s13, s13, 0x1032);
        s9 += s13; s5 ^= s9; asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s5));
        s1 += s5; s13 ^= s1; s13 = __byte_perm(s13, s13, 0x2103);
        s9 += s13; s5 ^= s9; asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s5));

        // QR(2, 6, 10, 14)
        s2 += s6; s14 ^= s2; s14 = __byte_perm(s14, s14, 0x1032);
        s10 += s14; s6 ^= s10; asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s6));
        s2 += s6; s14 ^= s2; s14 = __byte_perm(s14, s14, 0x2103);
        s10 += s14; s6 ^= s10; asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s6));

        // QR(3, 7, 11, 15)
        s3 += s7; s15 ^= s3; s15 = __byte_perm(s15, s15, 0x1032);
        s11 += s15; s7 ^= s11; asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s7));
        s3 += s7; s15 ^= s3; s15 = __byte_perm(s15, s15, 0x2103);
        s11 += s15; s7 ^= s11; asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s7));

        // Diagonal round
        // QR(0, 5, 10, 15)
        s0 += s5; s15 ^= s0; s15 = __byte_perm(s15, s15, 0x1032);
        s10 += s15; s5 ^= s10; asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s5));
        s0 += s5; s15 ^= s0; s15 = __byte_perm(s15, s15, 0x2103);
        s10 += s15; s5 ^= s10; asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s5));

        // QR(1, 6, 11, 12)
        s1 += s6; s12 ^= s1; s12 = __byte_perm(s12, s12, 0x1032);
        s11 += s12; s6 ^= s11; asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s6));
        s1 += s6; s12 ^= s1; s12 = __byte_perm(s12, s12, 0x2103);
        s11 += s12; s6 ^= s11; asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s6));

        // QR(2, 7, 8, 13)
        s2 += s7; s13 ^= s2; s13 = __byte_perm(s13, s13, 0x1032);
        s8 += s13; s7 ^= s8; asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s7));
        s2 += s7; s13 ^= s2; s13 = __byte_perm(s13, s13, 0x2103);
        s8 += s13; s7 ^= s8; asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s7));

        // QR(3, 4, 9, 14)
        s3 += s4; s14 ^= s3; s14 = __byte_perm(s14, s14, 0x1032);
        s9 += s14; s4 ^= s9; asm("shf.l.wrap.b32 %0, %0, %0, 12;" : "+r"(s4));
        s3 += s4; s14 ^= s3; s14 = __byte_perm(s14, s14, 0x2103);
        s9 += s14; s4 ^= s9; asm("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(s4));
    }

    // Final addition and store
    output[0] = s0 + i0; output[1] = s1 + i1; output[2] = s2 + i2; output[3] = s3 + i3;
    output[4] = s4 + i4; output[5] = s5 + i5; output[6] = s6 + i6; output[7] = s7 + i7;
    output[8] = s8 + i8; output[9] = s9 + i9; output[10] = s10 + i10; output[11] = s11 + i11;
    output[12] = s12 + i12; output[13] = s13 + i13; output[14] = s14 + i14; output[15] = s15 + i15;
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

// ============================================================================
// Hardware Reciprocal Division Functions
// ============================================================================

__device__ __forceinline__ uint64_t div64_hwrecip(uint64_t dividend, uint64_t divisor)
{
    double recip = __drcp_rn((double)divisor);
    uint64_t q = (uint64_t)((double)dividend * recip);

    q &= (uint64_t)-(dividend >= divisor);

    uint64_t prod = q * divisor;
    uint64_t rem  = dividend - prod;

    uint64_t over = (uint64_t)(prod > dividend);
    q -= over;
    rem += divisor & (uint64_t)-over;

    uint64_t under = (uint64_t)(rem >= divisor);
    q += under;

    return q;
}

__device__ __forceinline__ uint64_t mod64_hwrecip(uint64_t a, uint64_t d)
{
    double recip = __drcp_rn((double)d);
    uint64_t q = (uint64_t)((double)a * recip);

    uint64_t rem = a - q * d;

    uint64_t under = (uint64_t)-(rem > a);
    rem += d & under;

    uint64_t over = (uint64_t)-(rem >= d);
    rem -= d & over;

    return rem;
}

__device__ __forceinline__ uint64_t umul64_hi(uint64_t a, uint64_t b)
{
    return __umul64hi(a, b);
}

// Divide M = (r << 32) | x by normalized d, where:
//   d has top bit set, r < d
// Returns one 32-bit quotient digit and the 64-bit remainder.
__device__ __forceinline__ uint32_t div96by64_q32_hwrecip(
    uint64_t r, uint32_t x, uint64_t d, double recip, uint64_t *rem_out)
{
    // Quotient estimate
    uint64_t q = (uint64_t)(((double)r * 4294967296.0 + (double)x) * recip);
    q = min(q, 0xffffffffull);

    // Compute diff = M - q*d inline
    uint64_t rem = (r << 32) | (uint64_t)x;
    uint64_t p_lo = q * d;

    int64_t hi = (int64_t)(r >> 32)
               - (int64_t)__umul64hi(q, d)
               - (rem < p_lo);
    rem -= p_lo;

    // Over-correction: hi < 0 means q too large by 1
    uint64_t over = (uint64_t)(hi < 0);
    q -= over;
    rem += d & -over;
    hi += over & (rem < d);

    // Under-correction: hi > 0 || rem >= d means q too small
    uint64_t under = (uint64_t)(hi > 0) | (uint64_t)(rem >= d);
    q += under;
    rem -= d & -under;

    *rem_out = rem;
    return (uint32_t)q;
}

// NVIDIA NVRTC on Windows lacks __uint128_t — force hwrecip path
#if defined(__CUDACC_RTC__) && defined(_WIN32)
#undef XELIS_128_NORCP
#endif

#ifdef XELIS_128_NORCP

__device__ __forceinline__ uint64_t div128by64_hwrecip(uint64_t hi, uint64_t lo, uint64_t d)
{
    __uint128_t x = ((__uint128_t)hi << 64) | lo;
    return (uint64_t)(x / d);
}

__device__ __forceinline__ uint64_t mod128by64_hwrecip(uint64_t hi, uint64_t lo, uint64_t d)
{
    __uint128_t x = ((__uint128_t)hi << 64) | lo;
    return (uint64_t)(x % d);
}

#else

__device__ __forceinline__ uint64_t div128by64_hwrecip(uint64_t hi, uint64_t lo, uint64_t d)
{
    uint64_t r = mod64_hwrecip(hi, d);

    int s = __clzll(d);
    uint64_t dn = d << s;
    double recip = __drcp_rn((double)dn);

    uint64_t n_hi = s ? ((r << s) | (lo >> (64 - s))) : r;
    uint64_t n_lo = lo << s;

    uint64_t rem;
    uint32_t q1 = div96by64_q32_hwrecip(n_hi, (uint32_t)(n_lo >> 32), dn, recip, &rem);
    uint32_t q0 = div96by64_q32_hwrecip(rem, (uint32_t)n_lo, dn, recip, &rem);

    return ((uint64_t)q1 << 32) | q0;
}

__device__ __forceinline__ uint64_t mod128by64_hwrecip(uint64_t hi, uint64_t lo, uint64_t d)
{
    uint64_t r = mod64_hwrecip(hi, d);

    int s = __clzll(d);
    uint64_t dn = d << s;
    double recip = __drcp_rn((double)dn);

    uint64_t n_hi = s ? ((r << s) | (lo >> (64 - s))) : r;
    uint64_t n_lo = lo << s;

    uint64_t rem;
    (void)div96by64_q32_hwrecip(n_hi, (uint32_t)(n_lo >> 32), dn, recip, &rem);
    (void)div96by64_q32_hwrecip(rem, (uint32_t)n_lo, dn, recip, &rem);

    return rem >> s;  // denormalize
}

#endif // XELIS_128_NORCP

#define U128_LT(a_hi,a_lo,b_hi,b_lo) (((a_hi) < (b_hi)) || (((a_hi) == (b_hi)) && ((a_lo) < (b_lo))))
#define U128_GT(a_hi,a_lo,b_hi,b_lo) (((a_hi) > (b_hi)) || (((a_hi) == (b_hi)) && ((a_lo) > (b_lo))))

__device__ __forceinline__ uint64_t execute_op_offset(
    uint32_t offset_op, uint64_t a, uint64_t b, uint64_t c,
    uint32_t r, uint64_t result, uint32_t i, uint32_t j)
{
  uint64_t t0, t1, t2;

  if (offset_op < 7) {
    t0 = a * b; t1 = a * c; t2 = b * c;
    uint64_t r0 = t1 + t2, r1 = t0 - t1, r2 = c - a + b;
    uint64_t r3 = a - b + c, r4 = t2 + a, r5 = t1 + b, r6 = t0 * c;

    uint64_t out = 0;
    out |= r0 & -(uint64_t)(offset_op == 0);
    out |= r1 & -(uint64_t)(offset_op == 1);
    out |= r2 & -(uint64_t)(offset_op == 2);
    out |= r3 & -(uint64_t)(offset_op == 3);
    out |= r4 & -(uint64_t)(offset_op == 4);
    out |= r5 & -(uint64_t)(offset_op == 5);
    out |= r6 & -(uint64_t)(offset_op == 6);
    return out;
  }

  switch (offset_op)
  {
  case 7:
  {
    return mod128by64_hwrecip(a, b, (c | 1));
  }
  case 8:
  {
    t0 = ROTL(result, r);
    t1 = a | 2;
    if (U128_LT(b, c, t0, t1))
      return c;
    t2 = (t0 > 0) ? div64_hwrecip(b, t0) : 0;
    return c - t1 * t2;
  }
  case 9:
  {
    return div128by64_hwrecip(c, a, (b | 4));
  }
  case 10:
  {
    t0 = ROTL(result, r);
    return U128_GT(t0, b, a, (c | 8)) ? ((a > 0) ? div64_hwrecip(t0, a) : 0) : (a ^ b);
  }
  case 11:
  {
    return __umul64hi(a, c) + b * c;
  }
  case 12:
  {
    t0 = __umul64hi(c, b);
    t1 = a * b;
    t2 = ROTR(result, r);
    return t0 + t1 + c * t2;
  }
  case 13:
  {
    t0 = isqrt(b + j);
    t1 = murmurhash3(c ^ result ^ i ^ j) | 1;
    return mod128by64_hwrecip(a+i, t0, t1);
  }
  case 14:
  {
    t0 = isqrt(b | 2);
    t1 = isqrt(a + j);
    t2 = (t0 > 0) ? mod64_hwrecip((c + i), t0) : 0;
    return ROTL(t2, i + j) * t1;
  }
  default:
  {
    return (isqrt(a + i) * isqrt(c + j)) ^ (b + i + j);
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


#else
__device__ __forceinline__ void prefetch_l1(const void *ptr) {
    __builtin_prefetch(ptr, 0, 3);
}
__device__ __forceinline__ void prefetch_l2(const void *ptr) {
    __builtin_prefetch(ptr, 0, 1);
}
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
    aes_round_u64_computed(&hash1, &hash2, AES_KEY);

    uint64_t result = ~(hash1 ^ hash2);

#pragma unroll 4
    for (uint32_t j = 0; j < XELIS_BUFFER_SIZE_V3; j++)
    {
      uint64_t idx_a_read = map_index(result);
      uint64_t a = mem_buffer_a[idx_a_read];

      uint64_t rot_res = ~ROTR(result, r);
      uint64_t idx_b_read = map_index(a ^ rot_res);

      uint64_t b_val = mem_buffer_b[idx_b_read];
      uint64_t c = my_scratch[r];

      r++;

      uint32_t op_idx = (ROTL(result, (uint32_t)c) + 13) & 0xF;
      uint64_t v = execute_op_offset(op_idx, a, b_val, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      int use_buffer_b = pick_half(v);
      uint64_t idx_t = map_index(idx_seed);

      uint64_t t = (use_buffer_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

      uint64_t idx_a_write = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);

#if defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_NVIDIA__)
      prefetch_l1(&mem_buffer_a[idx_a_write]);
#endif

      uint64_t idx_b_write = map_index(idx_a_write ^ ~result ^ XELIS_SCATTER_CONST);

#if defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_NVIDIA__)
      prefetch_l1(&mem_buffer_b[idx_b_write]);
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


    uint64_t mem_a = mem_buffer_a[idx_a0];
    uint64_t idx_b0 = map_index(mem_a ^ addr_b);


    uint64_t mem_b = mem_buffer_b[idx_b0];

    uint64_t hash1 = mem_b;
    uint64_t hash2 = mem_a;
    aes_round_u64_computed(&hash1, &hash2, AES_KEY);

    uint64_t result = ~(hash1 ^ hash2);

    for (uint32_t j = 0; j < XELIS_BUFFER_SIZE_V3; j++)
    {
      // REMOVED: Loop prefetch for my_scratch[r_pf]
      // Sequential access pattern, HW prefetcher handles it automatically

      uint64_t idx_a_read = map_index(result);
      uint64_t a = mem_buffer_a[idx_a_read];

      uint64_t rot_res = ~ROTR(result, r);
      uint64_t idx_b_read = map_index(a ^ rot_res);
      uint64_t b_val = mem_buffer_b[idx_b_read];
      uint64_t c = my_scratch[r];

      r++;

      uint32_t op_idx = (ROTL(result, (uint32_t)c) + 13) & 0xF;
      uint64_t v = execute_op_offset(op_idx, a, b_val, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      int use_buffer_b = pick_half(v);
      uint64_t idx_t = map_index(idx_seed);

      uint64_t t = (use_buffer_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

      uint64_t idx_a_write = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);

#if defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_NVIDIA__)
      // L2 write prefetch for mem_buffer_a (issued early to hide latency)
      prefetch_l2(&mem_buffer_a[idx_a_write]);
#endif

      uint64_t idx_b_write = map_index(idx_a_write ^ ~result ^ XELIS_SCATTER_CONST);

#if defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_NVIDIA__)
      // L2 write prefetch for mem_buffer_b
      prefetch_l2(&mem_buffer_b[idx_b_write]);
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
// Stage 3 Hybrid V2 - Uncache random reads, cache sequential, uncached RMW
// ============================================================================

XELIS_HOT __device__ void stage_3_hybrid_v2(uint64_t *my_scratch, const uint8_t *sbox)
{
  uint64_t *__restrict mem_buffer_a = my_scratch;
  uint64_t *__restrict mem_buffer_b = my_scratch + XELIS_BUFFER_SIZE_V3;

  uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
  uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;
  uint32_t r = 0;

  for (uint32_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; i++)
  {
    // Random reads
    uint64_t mem_a = load_scratch(&mem_buffer_a[map_index(addr_a)]);
    uint64_t mem_b = load_scratch(&mem_buffer_b[map_index(mem_a ^ addr_b)]);

    uint64_t hash1 = mem_b;
    uint64_t hash2 = mem_a;
    aes_round_u64_lds(&hash1, &hash2, AES_KEY, sbox);

    uint64_t result = ~(hash1 ^ hash2);

    uint64_t idx_a_read = map_index(result);

    int pending_store = 0;
    uint64_t pending_idx_a_write = 0, pending_idx_b_write = 0;
    uint64_t pending_store_a = 0, pending_store_b = 0;

    uint64_t a_preloaded = load_scratch(&mem_buffer_a[idx_a_read]);

    for (uint32_t j = 0; j < XELIS_BUFFER_SIZE_V3; j++)
    {
      if (pending_store)
      {
        store_scratch(&mem_buffer_b[pending_idx_b_write], pending_store_b);
        store_scratch(&mem_buffer_a[pending_idx_a_write], pending_store_a);
      }

      // RAW forwarding: if deferred store hit the preloaded slot, use store value
      uint64_t a = (pending_store && pending_idx_a_write == idx_a_read)
                   ? pending_store_a
                   : a_preloaded;

#if HAS_HBM
      prefetch_l1(&my_scratch[r + 8]);
#endif
      uint64_t c = my_scratch[r];
      uint64_t rot_res = ~ROTR(result, r);
      r++;
      uint32_t op_idx = (ROTL(result, (uint32_t)c) + 13) & 0xF;

      uint64_t b_val = load_scratch(&mem_buffer_b[map_index(a ^ rot_res)]);

      uint64_t v = execute_op_offset(op_idx, a, b_val, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      // Early load next A into registers (~800cy flight time before j+1)
      idx_a_read = map_index(result);
      a_preloaded = load_scratch(&mem_buffer_a[idx_a_read]);

      uint64_t idx_t = map_index(idx_seed);
#if HAS_HBM
      uint64_t t_a = load_scratch(&mem_buffer_a[idx_t]);
      uint64_t t_b = load_scratch(&mem_buffer_b[idx_t]);
      uint64_t t = (pick_half(v) ? t_b : t_a) ^ result;
#else
      uint64_t t = (pick_half(v) ? load_scratch(&mem_buffer_b[idx_t])
                                 : load_scratch(&mem_buffer_a[idx_t])) ^ result;
#endif

      uint64_t idx_a_write = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
      uint64_t idx_b_write = map_index(idx_a_write ^ ~result ^ XELIS_SCATTER_CONST);

      uint64_t old_mem_a = load_scratch(&mem_buffer_a[idx_a_write]);
      uint64_t old_b     = load_scratch(&mem_buffer_b[idx_b_write]);

      pending_idx_a_write = idx_a_write;
      pending_idx_b_write = idx_b_write;
      pending_store_a = t;
      pending_store_b = old_b ^ old_mem_a ^ ROTR(t, i + j);
      pending_store = 1;
    }

    if (pending_store)
    {
      store_scratch(&mem_buffer_b[pending_idx_b_write], pending_store_b);
      store_scratch(&mem_buffer_a[pending_idx_a_write], pending_store_a);
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
#if defined(__HIP_PLATFORM_AMD__)
extern "C" __attribute__((amdgpu_flat_work_group_size(XELIS_MIN_WG, XELIS_MAX_WG)))
#elif defined(__CUDA_ARCH__)
extern "C" __maxnreg__(96)
#else
extern "C"
#endif
__global__ void xelis_stage1_kernel(
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

  // Nonce segmentation: bits 59-63=device_id, bits 48-58=random, bits 0-47=counter
  // Must use global_idx (not local_tid) since scratch memory is indexed by global_idx
  uint64_t counter = nonce_start & 0xFFFFFFFFFFFFULL;  // Extract counter (bits 0-47)
  uint64_t nonce = (nonce_start & 0xFFFF000000000000ULL) | ((counter + global_idx) & 0xFFFFFFFFFFFFULL);
  ((uint64_t *)(input))[5] = nonce;

  stage_1(input, my_scratch, blockDim.x);
}

// Stage 1 Cooperative kernel - bandwidth optimized (hard-capped at 32 threads)
// Uses shared memory coordination for coalesced writes
extern "C"
#if defined(__CUDA_ARCH__)
__maxnreg__(96)
#endif
__global__ void xelis_stage1_cooperative(
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

  uint64_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;

  uint8_t input[XELIS_TEMPLATE_SIZE];
  memcpy(input, input_template, XELIS_TEMPLATE_SIZE);

  // Nonce segmentation: bits 59-63=device_id, bits 48-58=random, bits 0-47=counter
  // Must use global_idx (not local_tid) since scratch memory is indexed by global_idx
  uint64_t counter = nonce_start & 0xFFFFFFFFFFFFULL;  // Extract counter (bits 0-47)
  uint64_t nonce = (nonce_start & 0xFFFF000000000000ULL) | ((counter + global_idx) & 0xFFFFFFFFFFFFULL);
  ((uint64_t *)(input))[5] = nonce;

  stage_1_cooperative(input, my_scratch, shared_mem);
}

// ============================================================================
// Baseline strategy: Stage1 + Stage3_hybrid_v2 fused, no Blake3
// ============================================================================
#if defined(__HIP_PLATFORM_AMD__)
extern "C" __attribute__((amdgpu_flat_work_group_size(XELIS_MIN_WG, XELIS_MAX_WG)))
#else
extern "C"
#endif
__global__ void xelis_s13_noblake_kernel(
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

  uint64_t counter = nonce_start & 0xFFFFFFFFFFFFULL;
  uint64_t nonce = (nonce_start & 0xFFFF000000000000ULL) | ((counter + global_idx) & 0xFFFFFFFFFFFFULL);
  ((uint64_t *)(input))[5] = nonce;

  stage_1(input, my_scratch, blockDim.x);

  __shared__ uint8_t sbox_lds[256];
  load_sbox_to_lds(sbox_lds);
  stage_3_hybrid_v2(my_scratch, sbox_lds);
}

// ============================================================================
// Sep strategy: Stage3_hybrid_v2 only (no Blake3)
// ============================================================================
extern "C"
#if defined(__CUDA_ARCH__) && defined(XELIS_S3_NREG)
__maxnreg__(XELIS_S3_NREG)
#endif
__global__ void xelis_s3_hybrid_v2_noblake_kernel(
    uint64_t *__restrict__ scratch_memory,
    uint32_t batch_size,
    uint32_t scratch_offset,
    const uint64_t *__restrict__ difficulty_target,
    uint64_t *__restrict__ solutions,
    uint64_t nonce_start)
{
  uint32_t local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_tid >= batch_size)
    return;

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;

  __shared__ uint8_t sbox_lds[256];
  load_sbox_to_lds(sbox_lds);
  stage_3_hybrid_v2(my_scratch, sbox_lds);
}

// ============================================================================
// Neo strategy: Stage3_hybrid_v2 + Blake3 fused
// ============================================================================
#if defined(__HIP_PLATFORM_AMD__)
extern "C" __attribute__((amdgpu_flat_work_group_size(XELIS_MIN_WG, XELIS_MAX_WG)))
#elif defined(__CUDA_ARCH__)
extern "C" __maxnreg__(48)
#else
extern "C"
#endif
__global__ void xelis_s3b3_hybrid_v2_kernel(
    uint64_t *__restrict__ scratch_memory,
    uint8_t *__restrict__ outputs,
    uint32_t batch_size,
    uint32_t scratch_offset,
    const uint64_t *__restrict__ difficulty_target,
    uint64_t *__restrict__ solutions,
    uint64_t nonce_start)
{
  uint32_t local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_tid >= batch_size)
    return;

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;
  uint8_t *my_output = outputs + global_idx * XELIS_HASH_SIZE;

  __shared__ uint8_t sbox_lds[256];
  load_sbox_to_lds(sbox_lds);
  stage_3_hybrid_v2(my_scratch, sbox_lds);
  xelis_blake3_lean((uint8_t *)my_scratch, my_output);

  if (difficulty_target && solutions && check_difficulty_gpu(my_output, difficulty_target))
  {
    uint64_t counter = nonce_start & 0xFFFFFFFFFFFFULL;
    uint64_t nonce = (nonce_start & 0xFFFF000000000000ULL) | ((counter + global_idx) & 0xFFFFFFFFFFFFULL);
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

// Separate Blake3 Kernel - lightweight, can run at high thread count
// Recommended block size: 256-512 threads (low register pressure)
extern "C"
__global__ void xelis_blake3_batch(
    uint64_t *__restrict__ scratch_memory,
    uint8_t *__restrict__ outputs,
    uint32_t batch_size,
    uint32_t scratch_offset,
    const uint64_t *__restrict__ difficulty_target,
    uint64_t *__restrict__ solutions,
    uint64_t nonce_start)
{
  uint32_t local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_tid >= batch_size)
    return;

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;
  uint8_t *my_output = outputs + global_idx * XELIS_HASH_SIZE;

  // Execute Blake3 finalization using lean implementation
  xelis_blake3_lean((uint8_t *)my_scratch, my_output);

  // Check difficulty and record solutions
  if (difficulty_target && solutions && check_difficulty_gpu(my_output, difficulty_target))
  {
    // Recompute the nonce that was used during Stage 1 (same computation as in Stage 1 kernel)
    uint64_t counter = nonce_start & 0xFFFFFFFFFFFFULL;
    uint64_t nonce = (nonce_start & 0xFFFF000000000000ULL) | ((counter + global_idx) & 0xFFFFFFFFFFFFULL);
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
extern "C" __attribute__((amdgpu_flat_work_group_size(XELIS_MIN_WG, XELIS_MAX_WG)))
#else
extern "C"
// __maxnreg__(96)  // disabled — let compiler pick register allocation
#endif
__global__ void xelis_hash_v3_kernel(
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

  // Nonce segmentation: bits 59-63=device_id, bits 48-58=random, bits 0-47=counter
  uint64_t counter = nonce_start & 0xFFFFFFFFFFFFULL;  // Extract counter (bits 0-47)
  uint64_t nonce = (nonce_start & 0xFFFF000000000000ULL) | ((counter + tid) & 0xFFFFFFFFFFFFULL);
  ((uint64_t *)(input))[5] = nonce;

  stage_1(input, my_scratch, blockDim.x);

  __shared__ uint8_t sbox_lds[256];
  load_sbox_to_lds(sbox_lds);
  stage_3_hybrid_v2(my_scratch, sbox_lds);
  xelis_blake3_lean((uint8_t *)my_scratch, my_output);

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