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

// Wavefront/warp size detection — compile-time
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
  #define XELIS_WAVE_SIZE __AMDGCN_WAVEFRONT_SIZE__
#elif defined(__CUDA_ARCH__)
  #define XELIS_WAVE_SIZE 32
#else
  #define XELIS_WAVE_SIZE 32
#endif

// Use hardware FP64 reciprocal division on architectures with fast FP64 (1:2).
// NVIDIA: only HPC dies (GP100, V100, A100, H100) — NOT consumer (sm_86, sm_89).
// AMD: GCN/CDNA (wave64) have 1:2 FP64; RDNA (wave32) have 1:16.
// Cooperative TPB: wave32 → stage1=32, b3=96 | wave64 → stage1=64, b3=64
#define XELIS_STAGE1_COOP_TPB  XELIS_WAVE_SIZE
#define XELIS_B3_COOP_TPB      (XELIS_WAVE_SIZE == 32 ? 96 : 64)

// ============================================================================
// Basic Device Functions
// ============================================================================

__device__ __forceinline__ uint64_t ROTL(uint64_t x, uint32_t r)
{
  r &= 63;
  return (x << r) | (x >> ((-r) & 63));
}

__device__ __forceinline__ uint64_t ROTR(uint64_t x, uint32_t r)
{
  r &= 63;
  return (x >> r) | (x << ((-r) & 63));
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

    // Branchless correction — eliminates warp divergence
    uint64_t sq = (uint64_t)s * s;
    s -= (uint32_t)(sq > arg);
    sq = (uint64_t)(s + 1) * (s + 1);
    s += (uint32_t)(sq <= arg);

    return s;
}

__device__ __forceinline__ void isqrt_pair(uint64_t xa, uint64_t xb,
    uint64_t *__restrict__ ra, uint64_t *__restrict__ rb)
{
    uint64_t xa_arg = xa, xb_arg = xb;
    xa = xa < 2 ? 4 : xa;
    xb = xb < 2 ? 4 : xb;
    uint32_t sa_scal = __clzll(xa) & ~1u, sb_scal = __clzll(xb) & ~1u;
    uint64_t sa_n = xa << sa_scal, sb_n = xb << sb_scal;
    uint32_t sa_hi = (uint32_t)(sa_n >> 32), sb_hi = (uint32_t)(sb_n >> 32);
    float sa_fr = __frsqrt_rn((float)sa_hi), sb_fr = __frsqrt_rn((float)sb_hi);
    uint32_t sa_r = (uint32_t)fmaf(1.407374884e14f, sa_fr, -438.0f);
    uint32_t sb_r = (uint32_t)fmaf(1.407374884e14f, sb_fr, -438.0f);
    uint32_t sa_s = __umulhi(sa_r, sa_hi) * 2, sb_s = __umulhi(sb_r, sb_hi) * 2;
    uint64_t sa_rem = sa_n - (uint64_t)sa_s * sa_s, sb_rem = sb_n - (uint64_t)sb_s * sb_s;
    sa_r = __umulhi((uint32_t)(sa_rem >> 32) + 1, sa_r);
    sb_r = __umulhi((uint32_t)(sb_rem >> 32) + 1, sb_r);
    sa_s += sa_r; sb_s += sb_r;
    sa_s >>= (sa_scal >> 1); sb_s >>= (sb_scal >> 1);
    uint64_t sa_sq = (uint64_t)sa_s * sa_s, sb_sq = (uint64_t)sb_s * sb_s;
    uint32_t sa_high = (uint32_t)(sa_sq > xa_arg), sb_high = (uint32_t)(sb_sq > xb_arg);
    sa_s -= sa_high; sb_s -= sb_high;
    sa_sq -= (2 * (uint64_t)sa_s + 1) & -(uint64_t)sa_high;
    sb_sq -= (2 * (uint64_t)sb_s + 1) & -(uint64_t)sb_high;
    uint64_t sa_nxt = sa_sq + 2 * (uint64_t)sa_s + 1;
    uint64_t sb_nxt = sb_sq + 2 * (uint64_t)sb_s + 1;
    sa_s += (uint32_t)(sa_nxt <= xa_arg);
    sb_s += (uint32_t)(sb_nxt <= xb_arg);
    *ra = (xa_arg < 2) ? xa_arg : (uint64_t)sa_s;
    *rb = (xb_arg < 2) ? xb_arg : (uint64_t)sb_s;
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
// Optimized Blake3 — Tier 1 (u32 loads, branched merge, shared cv_stack)
// ============================================================================

// Compress taking uint32_t block words directly (scratchpad is u64-aligned)
__device__ __forceinline__ void blake3_compress_u32(
    uint32_t cv[8], const uint32_t block_words[16],
    uint8_t block_len, uint64_t counter, uint8_t flags)
{
    uint32_t state[16];
    state[0]=cv[0]; state[1]=cv[1]; state[2]=cv[2]; state[3]=cv[3];
    state[4]=cv[4]; state[5]=cv[5]; state[6]=cv[6]; state[7]=cv[7];
    state[8]=IV_0; state[9]=IV_1; state[10]=IV_2; state[11]=IV_3;
    state[12]=(uint32_t)counter; state[13]=(uint32_t)(counter>>32);
    state[14]=(uint32_t)block_len; state[15]=(uint32_t)flags;
    BLAKE3_ROUND(0); BLAKE3_ROUND(1); BLAKE3_ROUND(2); BLAKE3_ROUND(3);
    BLAKE3_ROUND(4); BLAKE3_ROUND(5); BLAKE3_ROUND(6);
    cv[0]=state[0]^state[8]; cv[1]=state[1]^state[9]; cv[2]=state[2]^state[10]; cv[3]=state[3]^state[11];
    cv[4]=state[4]^state[12]; cv[5]=state[5]^state[13]; cv[6]=state[6]^state[14]; cv[7]=state[7]^state[15];
}

// Parent hash working entirely with uint32_t arrays (no byte round-trip)
__device__ __forceinline__ void blake3_hash_parent_u32(
    const uint32_t left[8], const uint32_t right[8],
    uint32_t output[8], bool is_root)
{
    uint32_t cv[8] = {IV_0, IV_1, IV_2, IV_3, IV_4, IV_5, IV_6, IV_7};
    uint32_t block_words[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) { block_words[i] = left[i]; block_words[i+8] = right[i]; }
    uint32_t state[16];
    state[0]=cv[0]; state[1]=cv[1]; state[2]=cv[2]; state[3]=cv[3];
    state[4]=cv[4]; state[5]=cv[5]; state[6]=cv[6]; state[7]=cv[7];
    state[8]=IV_0; state[9]=IV_1; state[10]=IV_2; state[11]=IV_3;
    state[12]=0; state[13]=0;
    state[14]=BLAKE3_BLOCK_LEN; state[15]=(uint32_t)(PARENT | (is_root ? ROOT : 0));
    BLAKE3_ROUND(0); BLAKE3_ROUND(1); BLAKE3_ROUND(2); BLAKE3_ROUND(3);
    BLAKE3_ROUND(4); BLAKE3_ROUND(5); BLAKE3_ROUND(6);
    #pragma unroll
    for (int i = 0; i < 8; i++) output[i] = state[i] ^ state[i+8];
}

// Optimized lean Blake3: branched merge + shared cv_stack + u32 loads
// cv_stack: caller provides per-thread shared memory slice [10][8]
__device__ __forceinline__ void xelis_blake3_opt(
    const uint8_t* input, uint8_t* output,
    uint32_t (*cv_stack)[8])
{
    constexpr int NUM_CHUNKS = 531;
    int stack_len = 0;

    for (uint64_t chunk_idx = 0; chunk_idx < (uint64_t)NUM_CHUNKS; chunk_idx++) {
        const uint32_t* chunk_w = (const uint32_t*)(input + (size_t)chunk_idx * BLAKE3_CHUNK_LEN);
        uint32_t cv[8] = {IV_0, IV_1, IV_2, IV_3, IV_4, IV_5, IV_6, IV_7};

        blake3_compress_u32(cv, chunk_w, BLAKE3_BLOCK_LEN, chunk_idx, CHUNK_START);
        #pragma unroll
        for (int b = 1; b < 15; b++)
            blake3_compress_u32(cv, chunk_w + b * 16, BLAKE3_BLOCK_LEN, chunk_idx, 0);
        blake3_compress_u32(cv, chunk_w + 15 * 16, BLAKE3_BLOCK_LEN, chunk_idx, CHUNK_END);

        #pragma unroll
        for (int i = 0; i < 8; i++) cv_stack[stack_len][i] = cv[i];
        stack_len++;

        // Branched merge — chunk_idx is warp-uniform, zero divergence cost.
        // Eliminates ~4800 wasted parent compressions vs branchless loop.
        int tz = __ffsll(chunk_idx + 1ULL) - 1;
        while (tz-- > 0 && stack_len >= 2) {
            uint32_t parent[8];
            blake3_hash_parent_u32(cv_stack[stack_len-2], cv_stack[stack_len-1], parent, false);
            stack_len--;
            #pragma unroll
            for (int i = 0; i < 8; i++) cv_stack[stack_len-1][i] = parent[i];
        }
    }

    while (stack_len > 2) {
        uint32_t parent[8];
        blake3_hash_parent_u32(cv_stack[stack_len-2], cv_stack[stack_len-1], parent, false);
        stack_len--;
        #pragma unroll
        for (int i = 0; i < 8; i++) cv_stack[stack_len-1][i] = parent[i];
    }

    // Root merge — write final hash as bytes
    uint32_t root[8];
    blake3_hash_parent_u32(cv_stack[0], cv_stack[1], root, true);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        blake3_store32(output + i * 4, root[i]);
    }
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
          "l"(v2), "l"(v3));
#else
    ptr[0] = v0;
    ptr[1] = v1;
    ptr[2] = v2;
    ptr[3] = v3;
#endif
}
#else
// AMD: Non-temporal stores + compiler fence
__device__ __forceinline__ void store_streaming_u256(uint64_t *ptr,
                                                     uint64_t v0, uint64_t v1,
                                                     uint64_t v2, uint64_t v3)
{
  __builtin_nontemporal_store(v0, ptr);
  __builtin_nontemporal_store(v1, ptr + 1);
  __builtin_nontemporal_store(v2, ptr + 2);
  __builtin_nontemporal_store(v3, ptr + 3);
  asm volatile("" ::: "memory");
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
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(val) : "l"(ptr));
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
  asm volatile("st.global.cg.u64 [%0], %1;" : : "l"(ptr), "l"(val));
}

// Write-through store (st.global.wt) — writes to DRAM + updates L2
__device__ __forceinline__ void store_wt(uint64_t *ptr, uint64_t val)
{
  asm volatile("st.global.wt.u64 [%0], %1;" : : "l"(ptr), "l"(val));
}

#endif  // Cache control intrinsics

// ============================================================================
// Large-L2 scratch access — cached loads/stores for GPUs with big L2
// ============================================================================
// NVIDIA: A100/CMP170HX (sm_80=40MB), RTX 40xx (sm_89=24-72MB), H100 (sm_90=50MB)
// AMD:    MI300 (gfx940/941/942)
// TODO: test whether LARGE_L2 actually helps — may thrash L2 with random scratch access
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 900)
//   #define LARGE_L2 1
// #elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
//   #define LARGE_L2 1
// #else
//   #define LARGE_L2 0
// #endif
#define LARGE_L2 0
#if LARGE_L2
  __device__ __forceinline__ uint64_t load_scratch(const uint64_t *p) { return *p; }
  __device__ __forceinline__ void store_scratch(uint64_t *p, uint64_t v) { *p = v; }
#else
  __device__ __forceinline__ uint64_t load_scratch(const uint64_t *p) { return load_uncached(p); }
  // NVIDIA: st.global.wt (write-through to DRAM+L2) — matches s3opt proven path
  // AMD: store_glc_slc (GLC+SLC) — already write-through equivalent
  #if defined(__HIP_PLATFORM_AMD__)
    __device__ __forceinline__ void store_scratch(uint64_t *p, uint64_t v) { store_glc_slc(p, v); }
  #else
    __device__ __forceinline__ void store_scratch(uint64_t *p, uint64_t v) { store_wt(p, v); }
  #endif
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
// Uses shared memory coordination for coalesced writes
// Block size: XELIS_STAGE1_COOP_TPB (32 for wave32, 64 for wave64)
constexpr int STAGE1_SHARED_MEM_SIZE = XELIS_STAGE1_COOP_TPB * 176; // TPB * (4*32 + 4*12)

// Phase 1: Each thread computes its own K2 and nonces into shared memory.
// All threads must call this (even invalid ones) to keep __syncthreads well-defined.
__device__ __forceinline__ void stage_1_cooperative_phase1(
    const uint8_t *input,
    uint8_t *shared_nonces_mem)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t block_size = blockDim.x;

    // Shared memory layout: [K2: 4*32 bytes][nonces: 4*12 bytes] per thread
    uint8_t (*shared_K2)[4][32] = (uint8_t (*)[4][32])shared_nonces_mem;
    uint8_t (*shared_nonces)[4][12] = (uint8_t (*)[4][12])(shared_nonces_mem + block_size * 4 * 32);

    // Scoped so nvcc can reclaim registers for Phase 2.
    // key[128] eliminated — reads from input[] directly (saves 128B register pressure).
    {
        uint8_t K2[4][32];
        uint8_t my_nonces[4][12];
        uint8_t buffer[64];

        // First hash to get nonce[0]
        blake3_hash_single_chunk(input, XELIS_TEMPLATE_SIZE, buffer);
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            my_nonces[0][i] = buffer[i];
        }

        // Derive K2[0]: hash(blake3_output[0:32] ++ input[0:32])
        #pragma unroll
        for (int i = 0; i < XELIS_CHUNK_SIZE; i++) {
            buffer[XELIS_CHUNK_SIZE + i] = input[i];
        }
        blake3_hash_single_chunk(buffer, XELIS_CHUNK_SIZE * 2, K2[0]);

        // Derive K2[1..3]: read key bytes from input, zero-pad past TEMPLATE_SIZE
        #pragma unroll
        for (int k = 1; k < 4; k++) {
            #pragma unroll
            for (int i = 0; i < XELIS_CHUNK_SIZE; i++) {
                buffer[i] = K2[k-1][i];
                int src_idx = k * XELIS_CHUNK_SIZE + i;
                buffer[XELIS_CHUNK_SIZE + i] = (src_idx < XELIS_TEMPLATE_SIZE) ? input[src_idx] : 0;
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
    } // K2, my_nonces, buffer all dead — nvcc can reclaim registers
}

// Phase 2: Cooperative scratchpad generation.
// All threads collaborate on EACH valid hash's scratchpad.
// block_base_scratch = scratch pointer for hash_idx 0 in this block.
// valid_count = number of valid hashes in this block (may be < blockDim.x).
__device__ __forceinline__ void stage_1_cooperative_phase2(
    uint64_t *block_base_scratch,
    uint8_t *shared_nonces_mem,
    uint32_t valid_count)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t block_size = blockDim.x;

    uint8_t (*shared_K2)[4][32] = (uint8_t (*)[4][32])shared_nonces_mem;
    uint8_t (*shared_nonces)[4][12] = (uint8_t (*)[4][12])(shared_nonces_mem + block_size * 4 * 32);

    const uint32_t blocks_per_chunk = XELIS_BYTES_PER_CHUNK_V3 / 64;

    for (uint32_t hash_idx = 0; hash_idx < valid_count; hash_idx++) {
        uint64_t *target_scratch = block_base_scratch + (size_t)hash_idx * XELIS_MEMORY_SIZE_V3;

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
    }
}

// Combined stage_1_cooperative — for callers that always launch full blocks.
// block_base_scratch = scratch pointer for the first hash in this block.
__device__ __forceinline__ void stage_1_cooperative(
    const uint8_t *input,
    uint64_t *block_base_scratch,
    uint8_t *shared_nonces_mem)
{
    stage_1_cooperative_phase1(input, shared_nonces_mem);
    __syncthreads();
    stage_1_cooperative_phase2(block_base_scratch, shared_nonces_mem, blockDim.x);
}

// ============================================================================
// Hardware Reciprocal Division Functions
// ============================================================================

// ------------------------------------------------------------
// split32 — 32-bit hardware division path
// Handles all divisor sizes correctly including tiny d (isqrt case 14).
// ------------------------------------------------------------

struct divmod_result { uint64_t q; uint64_t r; };

__device__ __forceinline__ divmod_result divmod64_split32(uint64_t a, uint64_t d) {
    if (d <= 0xFFFFFFFFULL) {
        uint32_t d32 = (uint32_t)d;
        uint32_t a_hi = (uint32_t)(a >> 32);
        uint32_t q_hi = a_hi / d32;
        uint32_t r_hi = a_hi - q_hi * d32;
        uint64_t mid = ((uint64_t)r_hi << 32) | (uint32_t)a;
        uint32_t q_lo = (uint32_t)(mid / d32);
        uint64_t rem = mid - (uint64_t)q_lo * d32;
        return { ((uint64_t)q_hi << 32) | q_lo, rem };
    }
    int s = __clzll(d);
    uint64_t dn = d << s;
    uint32_t v = (uint32_t)(dn >> 32);
    uint64_t a_shifted = (s == 0) ? (a >> 32) : (a >> (32 - s));
    uint32_t q = (v == 0xFFFFFFFFu) ? (uint32_t)a_shifted
                                    : (uint32_t)(a_shifted / ((uint64_t)v + 1));
    uint64_t rem = a - (uint64_t)q * d;
    if (rem > a) { q--; rem += d; }
    else if (rem >= d) { q++; rem -= d; }
    return { (uint64_t)q, rem };
}

// Convenience wrappers — zero-cost, compiler elides the unused field
__device__ __forceinline__ uint64_t mod64_split32(uint64_t a, uint64_t d) {
    return divmod64_split32(a, d).r;
}
__device__ __forceinline__ uint64_t div64_split32(uint64_t a, uint64_t d) {
    return divmod64_split32(a, d).q;
}

// ------------------------------------------------------------
// Tier: norm  — normalized-divisor reciprocal (dn ≥ 2^63)
//   (float)dn already captures top-24 bits; skip FP32 NR.
//   One FP64 NR → ~48 bits. Downstream q < 2^32, so error
//   < 2^-16 — the existing ±1 correction in div96by64 is plenty.
// ------------------------------------------------------------

__device__ __forceinline__ double recip_norm(uint64_t dn) {
    double dd = (double)dn;
    double x  = (double)__frcp_rn((float)dn);
    x = x * __fma_rn(-dd, x, 2.0);
    return x;
}

// ------------------------------------------------------------
// div96by64 — used internally by 128-bit split32 division.
// ±1 correction is sufficient with recip_norm.
// ------------------------------------------------------------

__device__ __forceinline__ uint32_t div96by64_q32_frcp(
    uint64_t r, uint32_t x, uint64_t d, double recip, uint64_t *rem_out)
{
    uint64_t q = (uint64_t)(((double)r * 4294967296.0 + (double)x) * recip);
    if (q > 0xffffffffull) q = 0xffffffffull;
    uint64_t rem = (r << 32) | (uint64_t)x;
    uint64_t p_lo = q * d;
    int64_t hi = (int64_t)(r >> 32)
               - (int64_t)__umul64hi(q, d)
               - (rem < p_lo);
    rem -= p_lo;
    uint64_t over = (uint64_t)(hi < 0);
    q   -= over;
    rem += d & -over;
    hi  += over & (rem < d);
    uint64_t under = (uint64_t)(hi > 0) | (uint64_t)(rem >= d);
    q   += under;
    rem -= d & -under;
    *rem_out = rem;
    return (uint32_t)q;
}

// ------------------------------------------------------------
// 128-bit ops: proven outer, norm inner
// ------------------------------------------------------------

__device__ __forceinline__ divmod_result divmod128by64_split32(uint64_t hi, uint64_t lo, uint64_t d) {
    uint64_t r = mod64_split32(hi, d);
    int s = __clzll(d);
    uint64_t dn = d << s;
    double recip = recip_norm(dn);
    uint64_t n_hi = s ? ((r << s) | (lo >> (64 - s))) : r;
    uint64_t n_lo = lo << s;
    uint64_t rem;
    uint32_t q1 = div96by64_q32_frcp(n_hi, (uint32_t)(n_lo >> 32), dn, recip, &rem);
    uint32_t q0 = div96by64_q32_frcp(rem,  (uint32_t)n_lo,          dn, recip, &rem);
    return { ((uint64_t)q1 << 32) | q0, rem >> s };
}

__device__ __forceinline__ uint64_t div128by64_split32(uint64_t hi, uint64_t lo, uint64_t d) {
    return divmod128by64_split32(hi, lo, d).q;
}
__device__ __forceinline__ uint64_t mod128by64_split32(uint64_t hi, uint64_t lo, uint64_t d) {
    return divmod128by64_split32(hi, lo, d).r;
}

// Barrett modular exponentiation — precompute mu once
__device__ __forceinline__ uint64_t modular_power_barrett(uint64_t base, uint64_t exp, uint64_t mod) {
    mod += (mod == 0);
    if (mod == 1) return 0;

    base %= mod;
    if (base < 2) return base;
    if (exp == 0) return 1;

    // Precompute mu = floor((2^128 - 1) / mod) using split32 (exact, CLZ-normalized)
    uint64_t mu_hi = div64_split32(0xFFFFFFFFFFFFFFFFULL, mod);
    uint64_t rem   = mod64_split32(0xFFFFFFFFFFFFFFFFULL, mod);
    uint64_t mu_lo = div128by64_split32(rem, 0xFFFFFFFFFFFFFFFFULL, mod);

    uint64_t result = 1;

    while (exp > 0) {
        if (exp & 1) {
            uint64_t lo = result * base;
            uint64_t hi = __umul64hi(result, base);
            result = barrett_reduce(lo, hi, mod, mu_lo, mu_hi);
        }
        exp >>= 1;
        if (exp > 0) {
            uint64_t lo = base * base;
            uint64_t hi = __umul64hi(base, base);
            base = barrett_reduce(lo, hi, mod, mu_lo, mu_hi);
            if (base == 0) return 0;
        }
    }
    return result;
}

#define U128_LT(a_hi,a_lo,b_hi,b_lo) (((a_hi) < (b_hi)) || (((a_hi) == (b_hi)) && ((a_lo) < (b_lo))))
#define U128_GT(a_hi,a_lo,b_hi,b_lo) (((a_hi) > (b_hi)) || (((a_hi) == (b_hi)) && ((a_lo) > (b_lo))))

// ============================================================================
// ============================================================================
// Newton LUT division — reciprocal seed + unconditional FMA refinement
// LUT is passed as a kernel argument (HIPRTC: no __device__ globals)
// Config (NEWTON_LUT_BITS/N/SHIFT/BASE) from shared header:
// ============================================================================
#include "newton-lut-config.hip.inc"

#if defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_NVIDIA__)
__device__ __forceinline__ double lut_load(const double *lut, int vi) {
    return __ldg(&lut[vi]);
}
#else
__device__ __forceinline__ double lut_load(const double *lut, int vi) {
    return lut[vi];
}
#endif

// mod64 via LUT-seeded reciprocal + unconditional FMA + correction
__device__ __forceinline__ uint64_t s3_mod64_newton(uint64_t a, uint64_t d,
                                                     const double *lut) {
    int s = __clzll(d); uint64_t dn = d << s; uint32_t v = (uint32_t)(dn >> 32);
    uint64_t ash = (s == 0) ? (a >> 32) : (a >> (32 - s));
    uint32_t vi = (v >> NEWTON_LUT_SHIFT) - NEWTON_LUT_BASE;
    double r = lut_load(lut, vi);
    r = r * __fma_rn(-(double)((uint64_t)v + 1), r, 2.0);  // unconditional FMA
    uint32_t q = (uint32_t)((double)ash * r);
    uint64_t rem = a - (uint64_t)q * d;
    if (rem > a) { rem += d; } else if (rem >= d) { rem -= d; }
    return rem;
}

// ============================================================================
// Float-reciprocal division helpers (used by s3_divmod128_newton)
// ============================================================================

__device__ __forceinline__ uint64_t s3_mod64(uint64_t a, uint64_t d) {
    if (d <= 0xFFFFFFFFULL) {
        uint32_t d32 = (uint32_t)d, ah = (uint32_t)(a >> 32);
        uint32_t rh = ah - (ah / d32) * d32;
        uint64_t mid = ((uint64_t)rh << 32) | (uint32_t)a;
        return mid - (uint64_t)((uint32_t)(mid / d32)) * d32;
    }
    int s = __clzll(d); uint64_t dn = d << s; uint32_t v = (uint32_t)(dn >> 32);
    uint64_t ash = (s == 0) ? (a >> 32) : (a >> (32 - s));
    uint32_t q = (v == 0xFFFFFFFFu) ? (uint32_t)ash : (uint32_t)(ash / ((uint64_t)v + 1));
    uint64_t rem = a - (uint64_t)q * d;
    if (rem > a) { rem += d; } else if (rem >= d) { rem -= d; }
    return rem;
}

__device__ __forceinline__ double s3_recip_norm(uint64_t dn) {
    double dd = (double)dn, x = (double)__frcp_rn((float)dn);
    return x * __fma_rn(-dd, x, 2.0);
}

__device__ __forceinline__ uint32_t s3_div96by64(uint64_t r, uint32_t x, uint64_t d,
    double recip, uint64_t *ro) {
    uint64_t q = (uint64_t)(((double)r * 4294967296.0 + (double)x) * recip);
    if (q > 0xffffffffull) q = 0xffffffffull;
    uint64_t rem = (r << 32) | (uint64_t)x, p_lo = q * d;
    int64_t hi = (int64_t)(r >> 32) - (int64_t)__umul64hi(q, d) - (rem < p_lo); rem -= p_lo;
    uint64_t ov = (uint64_t)(hi < 0); q -= ov; rem += d & -ov; hi += ov & (rem < d);
    uint64_t un = (uint64_t)(hi > 0) | (uint64_t)(rem >= d); q += un; rem -= d & -un;
    *ro = rem; return (uint32_t)q;
}

// divmod128 using newton mod64 for the hi reduction, then existing split32 for lo
__device__ __forceinline__ divmod_result s3_divmod128_newton(uint64_t hi, uint64_t lo,
                                                              uint64_t d, const double *lut) {
    uint64_t r = s3_mod64_newton(hi, d, lut);
    int s = __clzll(d); uint64_t dn = d << s; double recip = s3_recip_norm(dn);
    uint64_t n_hi = s ? ((r << s) | (lo >> (64 - s))) : r, n_lo = lo << s, rem;
    uint32_t q1 = s3_div96by64(n_hi, (uint32_t)(n_lo >> 32), dn, recip, &rem);
    uint32_t q0 = s3_div96by64(rem, (uint32_t)n_lo, dn, recip, &rem);
    return { ((uint64_t)q1 << 32) | q0, rem >> s };
}

// ============================================================================
// Split exec_op — branch on cheap vs expensive, newton LUT for divmod128
// ============================================================================

__device__ __forceinline__ uint64_t exec_cheap(
    uint32_t op, uint64_t a, uint64_t b, uint64_t c,
    uint32_t r, uint64_t result) {
    uint64_t m3  = -(uint64_t)(op==3),  m4  = -(uint64_t)(op==4);
    uint64_t m5  = -(uint64_t)(op==5),  m6  = -(uint64_t)(op==6);
    uint64_t m7  = -(uint64_t)(op==7),  m8  = -(uint64_t)(op==8);
    uint64_t m9  = -(uint64_t)(op==9);
    uint64_t m14 = -(uint64_t)(op==14), m15 = -(uint64_t)(op==15);
    uint64_t ab = a * b, ac = a * c, bc = b * c;
    uint64_t cx = (ac & (m3|m8)) | (ab & m4) | (c & m5) | (a & m6) | (bc & m7);
    uint64_t cy = (bc & m3)      | (b & m5)  | (c & m6) | (a & m7) | (b & m8);
    uint64_t cz = (ac & m4)      | (a & m5)  | (b & m6);
    uint64_t cheap = cx + cy - cz;
    cheap |= (ab * c) & m9;
    uint64_t mhi = m14 | m15;
    uint64_t hp = (a & m14) | (c & m15);
    uint64_t hq = (c & m14) | (b & m15);
    uint64_t hn = (bc & m14) | ((ab + c * ROTR(result, r)) & m15);
    cheap |= (__umul64hi(hp, hq) + hn) & mhi;
    return cheap;
}

__device__ __noinline__ uint64_t exec_op_split_newton(
    uint32_t op, uint64_t a, uint64_t b, uint64_t c,
    uint32_t r, uint64_t result, uint32_t i, uint32_t j,
    const double *lut) {

    bool is_expensive = (op <= 2) || (op >= 10 && op <= 13);
    if (!is_expensive) {
        return exec_cheap(op, a, b, c, r, result);
    }

    uint64_t t0 = ROTL(result, r), mur = murmurhash3(c ^ result ^ i ^ j) | 1;
    uint64_t m0  = -(uint64_t)(op==0),  m1  = -(uint64_t)(op==1),  m2  = -(uint64_t)(op==2);
    uint64_t m10 = -(uint64_t)(op==10), m11 = -(uint64_t)(op==11);
    uint64_t m12 = -(uint64_t)(op==12), m13 = -(uint64_t)(op==13);
    uint64_t pA = m0 | m10 | m12, pB = m1 | m11 | m13, pC = m2;

    uint64_t sq0_in = ((b + j) & pA) | ((b | 2) & pB) | ((a + i) & pC);
    uint64_t sq1_in = ((a + j) & pB) | ((c + j) & pC) | (4 & ~(pB | pC));
    uint64_t sq0, sq1;
    isqrt_pair(sq0_in, sq1_in, &sq0, &sq1);

    uint64_t hiA  = ((a + i) & m0) | (a & m10) | (c & m12) | ~pA;
    uint64_t loA  = (sq0 & m0) | (b & m10) | (a & m12);
    uint64_t dvA  = (mur & m0) | ((c | 1) & m10) | ((b | 4) & m12) | ~pA;
    uint64_t numB = ((c + i) & m1) | (b & m11) | (t0 & m13) | ~pB;
    uint64_t denB = (sq0 & m1) | (t0 & m11) | (a & m13) | ~pB;
    denB |= (uint64_t)(denB == 0);

    uint64_t dm_hi  = hiA & pA;
    uint64_t dm_lo  = (loA & pA) | (numB & pB) | (1 & ~(pA | pB));
    uint64_t dm_div = (dvA & pA) | (denB & pB) | (1 & ~(pA | pB));
    dm_div |= (uint64_t)(dm_div == 0);

    divmod_result dm = s3_divmod128_newton(dm_hi, dm_lo, dm_div, lut);

    uint64_t resA = (dm.r & (m0 | m10)) | (dm.q & m12);
    uint64_t res1 = ROTL(dm.r, i + j) * sq1;
    uint64_t t1 = a | 2;
    uint64_t c11 = -(uint64_t)U128_LT(b, c, t0, t1);
    uint64_t r11 = (c & c11) | ((c - t1 * dm.q) & ~c11);
    uint64_t c13 = -(uint64_t)U128_GT(t0, b, a, (c | 8));
    uint64_t r13 = (dm.q & c13) | ((a ^ b) & ~c13);
    uint64_t resB = (res1 & m1) | (r11 & m11) | (r13 & m13);
    uint64_t resC = (sq0 * sq1) ^ (b + i + j);
    return (resA & pA) | (resB & pB) | (resC & pC);
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
// Stage 3 Hybrid V2 - Uncache random reads, cache sequential, uncached RMW
// ============================================================================

XELIS_HOT __device__ void stage_3_hybrid_v2(uint64_t *my_scratch,
                                              const double *newton_lut)
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
#if defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_NVIDIA__)
    aes_round_u64_ldg(&hash1, &hash2, AES_KEY);
#else
    aes_round_u64_tex(&hash1, &hash2, AES_KEY);
#endif

    uint64_t result = ~(hash1 ^ hash2);

#if defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_NVIDIA__)
    // DEFER_FIX2: deferred stores with 3 software bypasses (apre, bv, c)
    // Maximizes ILP by issuing all loads before flushing deferred stores.
    uint64_t iar = map_index(result);
    uint64_t apre = load_scratch(&mem_buffer_a[iar]);

    uint64_t d_iaw = XELIS_BUFFER_SIZE_V3 + 1;
    uint64_t d_ibw = XELIS_BUFFER_SIZE_V3 + 1;
    uint64_t d_t = 0;
    uint32_t d_ij = 0;
    int ds = 0;

    for (uint32_t j = 0; j < XELIS_BUFFER_SIZE_V3; j++)
    {
      uint64_t a = (ds && d_iaw == iar) ? d_t : apre;

      uint64_t rot_res = ~ROTR(result, r);
      uint64_t bv_addr = map_index(a ^ rot_res);

      uint64_t bv = load_scratch(&mem_buffer_b[bv_addr]);
      uint64_t oa = 0, ob = 0, pvb = 0;
      if (ds)
      {
        oa = load_scratch(&mem_buffer_a[d_iaw]);
        ob = load_scratch(&mem_buffer_b[d_ibw]);
      }

      uint64_t c = my_scratch[r];

      if (ds) pvb = ob ^ oa ^ ROTR(d_t, d_ij);

      if (ds)
      {
        if (d_iaw == (uint64_t)r)
          c = d_t;
        else if ((d_ibw + XELIS_BUFFER_SIZE_V3) == (uint64_t)r)
          c = pvb;
      }

      r++;
      uint32_t op_idx = ROTL(result, (uint32_t)c) & 0xF;

      if (ds)
      {
        if (d_ibw == bv_addr) bv = pvb;
        store_scratch(&mem_buffer_b[d_ibw], pvb);
        store_scratch(&mem_buffer_a[d_iaw], d_t);
      }

      uint64_t v = exec_op_split_newton(op_idx, a, bv, c, r, result, i, j, newton_lut);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      uint64_t idx_t = map_index(idx_seed);
      iar = map_index(result);
      apre = load_scratch(&mem_buffer_a[iar]);

      uint64_t t = (pick_half(v) ? load_scratch(&mem_buffer_b[idx_t])
                                 : load_scratch(&mem_buffer_a[idx_t])) ^ result;

      uint64_t iaw = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
      uint64_t ibw = map_index(iaw ^ ~result ^ XELIS_SCATTER_CONST);

      d_iaw = iaw;
      d_ibw = ibw;
      d_t = t;
      d_ij = i + j;
      ds = 1;
    }

    if (ds)
    {
      uint64_t oa = load_scratch(&mem_buffer_a[d_iaw]);
      uint64_t ob = load_scratch(&mem_buffer_b[d_ibw]);
      store_scratch(&mem_buffer_b[d_ibw], ob ^ oa ^ ROTR(d_t, d_ij));
      store_scratch(&mem_buffer_a[d_iaw], d_t);
    }

#else
    // AMD: 1-bypass deferred store (v0.7.7 pattern).
    // LLVM's aggressive reordering makes the 3-bypass variant unreliable.
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

      uint64_t a = (pending_store && pending_idx_a_write == idx_a_read)
                   ? pending_store_a
                   : a_preloaded;

      uint64_t c = my_scratch[r];
      uint64_t rot_res = ~ROTR(result, r);
      r++;
      uint32_t op_idx = ROTL(result, (uint32_t)c) & 0xF;

      uint64_t b_val = load_scratch(&mem_buffer_b[map_index(a ^ rot_res)]);

      uint64_t v = exec_op_split_newton(op_idx, a, b_val, c, r, result, i, j, newton_lut);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      idx_a_read = map_index(result);
      a_preloaded = load_scratch(&mem_buffer_a[idx_a_read]);

      uint64_t idx_t = map_index(idx_seed);
      uint64_t t = (pick_half(v) ? load_scratch(&mem_buffer_b[idx_t])
                                 : load_scratch(&mem_buffer_a[idx_t])) ^ result;

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

#endif // DEFER_FIX2 platform gate

    addr_a = modular_power_barrett(addr_a, addr_b, result);
    uint64_t sq_r, sq_a;
    isqrt_pair(result, addr_a, &sq_r, &sq_a);
    addr_b = sq_r * (r + 1) * sq_a;
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

// Stage 1 Cooperative kernel - bandwidth optimized
// Uses shared memory coordination for coalesced writes
// TPB = XELIS_STAGE1_COOP_TPB (32 for wave32, 64 for wave64)
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
  extern __shared__ uint8_t shared_mem[];
  // Number of valid hashes in this block (may be < blockDim.x in last block)
  __shared__ uint32_t block_valid_count;

  const uint32_t tid = threadIdx.x;
  uint32_t local_tid = blockIdx.x * blockDim.x + tid;
  bool valid = (local_tid < batch_size);

  // First thread computes how many valid hashes this block has
  if (tid == 0) {
    uint32_t block_start = blockIdx.x * blockDim.x;
    uint32_t block_end = block_start + blockDim.x;
    block_valid_count = (batch_size >= block_end) ? blockDim.x
                      : (batch_size > block_start) ? (batch_size - block_start) : 0;
  }

  // Block-base scratch pointer: first hash in this block
  uint64_t block_base_idx = (uint64_t)(blockIdx.x * blockDim.x) + scratch_offset;
  uint64_t *block_base_scratch = scratch_memory + (size_t)block_base_idx * XELIS_MEMORY_SIZE_V3;

  uint8_t input[XELIS_TEMPLATE_SIZE];
  if (valid) {
    uint64_t global_idx = (uint64_t)local_tid + scratch_offset;
    memcpy(input, input_template, XELIS_TEMPLATE_SIZE);
    uint64_t counter = nonce_start & 0xFFFFFFFFFFFFULL;
    uint64_t nonce = (nonce_start & 0xFFFF000000000000ULL) | ((counter + global_idx) & 0xFFFFFFFFFFFFULL);
    ((uint64_t *)(input))[5] = nonce;
  }

  // Phase 1: all threads compute K2/nonces into shared memory.
  // Invalid threads produce garbage but phase 2 won't iterate to their hash_idx.
  stage_1_cooperative_phase1(input, shared_mem);

  __syncthreads();

  // Phase 2: cooperative scratchpad generation — only iterate valid hashes
  uint32_t valid_count = block_valid_count;
  stage_1_cooperative_phase2(block_base_scratch, shared_mem, valid_count);
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
    uint32_t scratch_offset,
    const double *__restrict__ newton_lut)
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

  stage_3_hybrid_v2(my_scratch, newton_lut);
}

// ============================================================================
// Sep strategy: Stage3_hybrid_v2 only (no Blake3)
// ============================================================================
extern "C"
#if defined(__CUDA_ARCH__) && defined(XELIS_S3_NREG)
// __maxnreg__(XELIS_S3_NREG)
#endif
__global__ void xelis_s3_hybrid_v2_noblake_kernel(
    uint64_t *__restrict__ scratch_memory,
    uint32_t batch_size,
    uint32_t scratch_offset,
    const uint64_t *__restrict__ difficulty_target,
    uint64_t *__restrict__ solutions,
    uint64_t nonce_start,
    const double *__restrict__ newton_lut)
{
  uint32_t local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_tid >= batch_size)
    return;

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;

  stage_3_hybrid_v2(my_scratch, newton_lut);
}

// ============================================================================
// Neo strategy: Stage3_hybrid_v2 + Blake3 fused
// ============================================================================
#if defined(__HIP_PLATFORM_AMD__)
extern "C" __attribute__((amdgpu_flat_work_group_size(XELIS_MIN_WG, XELIS_MAX_WG)))
#elif defined(__CUDA_ARCH__)
extern "C" __maxnreg__(64)
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
    uint64_t nonce_start,
    const double *__restrict__ newton_lut)
{
  uint32_t local_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_tid >= batch_size)
    return;

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;
  uint8_t *my_output = outputs + global_idx * XELIS_HASH_SIZE;

  stage_3_hybrid_v2(my_scratch, newton_lut);
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
// Optimized Blake3 batch kernel (Tier 1: u32 loads, branched merge, shared cv_stack)
// Uses dynamic shared memory for cv_stack — launch with smem = tpb * 10 * 8 * 4
// ============================================================================
extern "C"
__global__ void xelis_blake3_opt_batch(
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

  // Per-thread cv_stack in dynamic shared memory
  extern __shared__ uint32_t smem_cv[];
  uint32_t (*my_cv_stack)[8] = (uint32_t (*)[8])&smem_cv[threadIdx.x * 10 * 8];

  uint32_t global_idx = local_tid + scratch_offset;
  uint64_t *my_scratch = scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3;
  uint8_t *my_output = outputs + global_idx * XELIS_HASH_SIZE;

  xelis_blake3_opt((uint8_t *)my_scratch, my_output, my_cv_stack);

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

// ============================================================================
// Blake3 warp-cooperative — CVs in global memory (L1-cached), no smem
//
// Uses global memory for the 266-entry CV merge tree instead of shared memory.
// 8.5KB of CVs per block fits easily in L1 cache (192KB on NVIDIA, 16KB on AMD).
// Removes the smem/LDS bottleneck that limited occupancy to ~5 blocks/SM.
//
// CV workspace is stored at the end of the scratch allocation:
//   cv_base = scratch_memory + batch_size * XELIS_MEMORY_SIZE_V3
// Requires scratch_per_hash to include XELIS_B3_CV_BYTES_PER_HASH extra bytes.
// ============================================================================
constexpr size_t XELIS_B3_CV_WORDS_PER_HASH = 266 * 8;  // 266 CVs × 8 uint32_t
constexpr size_t XELIS_B3_CV_BYTES_PER_HASH = XELIS_B3_CV_WORDS_PER_HASH * sizeof(uint32_t);  // 8512

extern "C"
__global__ void xelis_blake3_warp_coop_batch(
    uint64_t *__restrict__ scratch_memory,
    uint8_t *__restrict__ outputs,
    uint32_t batch_size,
    uint32_t scratch_offset,
    const uint64_t *__restrict__ difficulty_target,
    uint64_t *__restrict__ solutions,
    uint64_t nonce_start)
{
  uint32_t hash_idx = blockIdx.x;
  if (hash_idx >= batch_size)
    return;
  uint32_t lane = threadIdx.x;

  uint32_t global_idx = hash_idx + scratch_offset;
  const uint32_t* scratch = (const uint32_t*)(scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3);
  uint8_t *my_output = outputs + global_idx * XELIS_HASH_SIZE;

  // CV workspace lives after all scratchpad data in the same allocation
  uint32_t *cv_base = (uint32_t*)(scratch_memory + (size_t)batch_size * XELIS_MEMORY_SIZE_V3);
  uint32_t *my_cvs = cv_base + (size_t)hash_idx * XELIS_B3_CV_WORDS_PER_HASH;

  // Phase 1: Paired chunk hashing with level-0 merge — same as warp-coop
  for (uint32_t pair = lane; pair < 265; pair += blockDim.x) {
    uint32_t c_L = pair * 2;
    uint32_t c_R = pair * 2 + 1;
    const uint32_t* chunk_L = scratch + c_L * 256;
    const uint32_t* chunk_R = scratch + c_R * 256;

    uint32_t cv_L[8] = {IV_0, IV_1, IV_2, IV_3, IV_4, IV_5, IV_6, IV_7};
    uint32_t cv_R[8] = {IV_0, IV_1, IV_2, IV_3, IV_4, IV_5, IV_6, IV_7};

    blake3_compress_u32(cv_L, chunk_L, BLAKE3_BLOCK_LEN, c_L, CHUNK_START);
    blake3_compress_u32(cv_R, chunk_R, BLAKE3_BLOCK_LEN, c_R, CHUNK_START);
    #pragma unroll
    for (int b = 1; b < 15; b++) {
      blake3_compress_u32(cv_L, chunk_L + b * 16, BLAKE3_BLOCK_LEN, c_L, 0);
      blake3_compress_u32(cv_R, chunk_R + b * 16, BLAKE3_BLOCK_LEN, c_R, 0);
    }
    blake3_compress_u32(cv_L, chunk_L + 15 * 16, BLAKE3_BLOCK_LEN, c_L, CHUNK_END);
    blake3_compress_u32(cv_R, chunk_R + 15 * 16, BLAKE3_BLOCK_LEN, c_R, CHUNK_END);

    uint32_t merged[8];
    blake3_hash_parent_u32(cv_L, cv_R, merged, false);
    #pragma unroll
    for (int i = 0; i < 8; i++) my_cvs[pair * 8 + i] = merged[i];
  }

  // Straggler: chunk 530 has no pair
  if (lane == 0) {
    const uint32_t* chunk_ptr = scratch + 530 * 256;
    uint32_t cv[8] = {IV_0, IV_1, IV_2, IV_3, IV_4, IV_5, IV_6, IV_7};
    blake3_compress_u32(cv, chunk_ptr, BLAKE3_BLOCK_LEN, 530, CHUNK_START);
    #pragma unroll
    for (int b = 1; b < 15; b++)
      blake3_compress_u32(cv, chunk_ptr + b * 16, BLAKE3_BLOCK_LEN, 530, 0);
    blake3_compress_u32(cv, chunk_ptr + 15 * 16, BLAKE3_BLOCK_LEN, 530, CHUNK_END);
    #pragma unroll
    for (int i = 0; i < 8; i++) my_cvs[265 * 8 + i] = cv[i];
  }
  __syncthreads();

  // Phase 2: Merge tree via global memory (L1-cached within block)
  uint32_t n = 266;
  while (n > 1) {
    uint32_t pairs = n / 2;
    uint32_t odd = n & 1;
    bool is_root = (pairs == 1 && odd == 0);

    for (uint32_t p = lane; p < pairs; p += blockDim.x) {
      uint32_t left[8], right[8], result[8];
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        left[i]  = my_cvs[(p * 2) * 8 + i];
        right[i] = my_cvs[(p * 2 + 1) * 8 + i];
      }
      blake3_hash_parent_u32(left, right, result, is_root);
      #pragma unroll
      for (int i = 0; i < 8; i++) my_cvs[p * 8 + i] = result[i];
    }
    __syncthreads();

    if (odd && lane == 0) {
      #pragma unroll
      for (int j = 0; j < 8; j++)
        my_cvs[pairs * 8 + j] = my_cvs[(n - 1) * 8 + j];
    }
    if (odd) __syncthreads();

    n = pairs + odd;
  }

  // Output — thread 0 writes result and checks difficulty
  if (lane == 0) {
    #pragma unroll
    for (int i = 0; i < 8; i++)
      blake3_store32(my_output + i * 4, my_cvs[i]);

    if (difficulty_target && solutions && check_difficulty_gpu(my_output, difficulty_target))
    {
      uint64_t counter = nonce_start & 0xFFFFFFFFFFFFULL;
      uint64_t nonce = (nonce_start & 0xFFFF000000000000ULL) | ((counter + global_idx) & 0xFFFFFFFFFFFFULL);
      unsigned long long idx = atomicAdd((unsigned long long *)&solutions[0], 1ULL);

      if (idx < 1024)
      {
        uint64_t *solution_slot = &solutions[1 + idx * 5];
        solution_slot[0] = nonce;
        solution_slot[1] = ((uint64_t *)my_output)[0];
        solution_slot[2] = ((uint64_t *)my_output)[1];
        solution_slot[3] = ((uint64_t *)my_output)[2];
        solution_slot[4] = ((uint64_t *)my_output)[3];
      }
    }
  }
}

// ============================================================================
// Smem-based Blake3: 512 TPB, pipelined chunk hashing, smem tree reduction
// ============================================================================
// 1 block per hash, 512 threads. Each thread hashes ~1 chunk (531/512).
// Double-buffered loads: issue next block's ld.cg before computing current.
// CVs stored in shared memory (25600B) — no global CV workspace needed.
// Benchmarked at ~3.8% overhead on CMP 170HX (vs 4.5% for warp-coop).

#define XELIS_B3_SMEM_WORDS (800 * 8)  // 531 CVs + tree reduction temp

__device__ __forceinline__ void blake3_load_block_uncached(
    uint32_t bw[16], const uint32_t* src)
{
  #pragma unroll
  for (int i = 0; i < 16; i += 2) {
    uint64_t v = load_uncached((const uint64_t*)(src + i));
    bw[i] = (uint32_t)v;
    bw[i+1] = (uint32_t)(v >> 32);
  }
}

// Double-buffered chunk hasher: issue next block's loads BEFORE
// computing current block. GPU scoreboarding lets loads complete in
// background during the ~700-cycle blake3 compress.
__device__ __forceinline__ void blake3_hash_chunk_pipelined(
    uint32_t cv[8], const uint32_t* chunk_data, uint64_t chunk_counter)
{
  uint32_t buf[16];
  blake3_load_block_uncached(buf, chunk_data);

  #pragma unroll 1
  for (int b = 0; b < 15; b++) {
    uint32_t next[16];
    blake3_load_block_uncached(next, chunk_data + (b + 1) * 16);
    uint8_t flags = (b == 0) ? CHUNK_START : 0;
    blake3_compress_u32(cv, buf, BLAKE3_BLOCK_LEN, chunk_counter, flags);
    #pragma unroll
    for (int i = 0; i < 16; i++) buf[i] = next[i];
  }
  blake3_compress_u32(cv, buf, BLAKE3_BLOCK_LEN, chunk_counter, CHUNK_END);
}

extern "C"
__global__
void xelis_blake3_smem_batch(
    uint64_t *__restrict__ scratch_memory,
    uint8_t *__restrict__ outputs,
    uint32_t batch_size,
    uint32_t scratch_offset,
    const uint64_t *__restrict__ difficulty_target,
    uint64_t *__restrict__ solutions,
    uint64_t nonce_start)
{
  uint32_t hash_idx = blockIdx.x;
  if (hash_idx >= batch_size)
    return;
  uint32_t lane = threadIdx.x;

  uint32_t global_idx = hash_idx + scratch_offset;
  const uint32_t* scratch = (const uint32_t*)(scratch_memory + (size_t)global_idx * XELIS_MEMORY_SIZE_V3);
  uint8_t *my_output = outputs + global_idx * XELIS_HASH_SIZE;

  __shared__ uint32_t smem[XELIS_B3_SMEM_WORDS];

  // Phase 1: each thread hashes its assigned chunk(s) via pipelined loads
  for (uint32_t c = lane; c < 531; c += blockDim.x) {
    uint32_t cv[8] = {IV_0, IV_1, IV_2, IV_3, IV_4, IV_5, IV_6, IV_7};
    blake3_hash_chunk_pipelined(cv, scratch + c * 256, c);
    #pragma unroll
    for (int i = 0; i < 8; i++) smem[c * 8 + i] = cv[i];
  }
  __syncthreads();

  // Phase 2: tree reduction 531 → 266
  {
    for (uint32_t p = lane; p < 265; p += blockDim.x) {
      uint32_t left[8], right[8], result[8];
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        left[i]  = smem[(p * 2) * 8 + i];
        right[i] = smem[(p * 2 + 1) * 8 + i];
      }
      blake3_hash_parent_u32(left, right, result, false);
      #pragma unroll
      for (int i = 0; i < 8; i++) smem[(531 + p) * 8 + i] = result[i];
    }
    __syncthreads();
    for (uint32_t p = lane; p < 265; p += blockDim.x) {
      #pragma unroll
      for (int i = 0; i < 8; i++) smem[p * 8 + i] = smem[(531 + p) * 8 + i];
    }
    if (lane == 0) {
      #pragma unroll
      for (int j = 0; j < 8; j++) smem[265 * 8 + j] = smem[530 * 8 + j];
    }
    __syncthreads();
  }

  // Phase 3: tree reduction 266 → 1
  uint32_t n = 266;
  while (n > 1) {
    uint32_t pairs = n / 2;
    uint32_t odd = n & 1;
    bool is_root = (pairs == 1 && odd == 0);
    for (uint32_t p = lane; p < pairs; p += blockDim.x) {
      uint32_t left[8], right[8], result[8];
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        left[i]  = smem[(p * 2) * 8 + i];
        right[i] = smem[(p * 2 + 1) * 8 + i];
      }
      blake3_hash_parent_u32(left, right, result, is_root);
      #pragma unroll
      for (int i = 0; i < 8; i++) smem[(n + p) * 8 + i] = result[i];
    }
    __syncthreads();
    for (uint32_t p = lane; p < pairs; p += blockDim.x) {
      #pragma unroll
      for (int i = 0; i < 8; i++) smem[p * 8 + i] = smem[(n + p) * 8 + i];
    }
    if (odd && lane == 0) {
      #pragma unroll
      for (int j = 0; j < 8; j++)
        smem[pairs * 8 + j] = smem[(n - 1) * 8 + j];
    }
    __syncthreads();
    n = pairs + odd;
  }

  // Output — thread 0 writes result and checks difficulty
  if (lane == 0) {
    #pragma unroll
    for (int i = 0; i < 8; i++)
      blake3_store32(my_output + i * 4, smem[i]);

    if (difficulty_target && solutions && check_difficulty_gpu(my_output, difficulty_target))
    {
      uint64_t counter = nonce_start & 0xFFFFFFFFFFFFULL;
      uint64_t nonce = (nonce_start & 0xFFFF000000000000ULL) | ((counter + global_idx) & 0xFFFFFFFFFFFFULL);
      unsigned long long idx = atomicAdd((unsigned long long *)&solutions[0], 1ULL);

      if (idx < 1024)
      {
        uint64_t *solution_slot = &solutions[1 + idx * 5];
        solution_slot[0] = nonce;
        solution_slot[1] = ((uint64_t *)my_output)[0];
        solution_slot[2] = ((uint64_t *)my_output)[1];
        solution_slot[3] = ((uint64_t *)my_output)[2];
        solution_slot[4] = ((uint64_t *)my_output)[3];
      }
    }
  }
}

// ============================================================================
// Mono strategy: Stage1 + Stage3_hybrid_v2 + Blake3 combined
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
    uint64_t *__restrict__ solutions,
    const double *__restrict__ newton_lut)
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

  stage_3_hybrid_v2(my_scratch, newton_lut);
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