#ifndef KAWPOW_HIP
#define KAWPOW_HIP

#include "hiprtc_types.hip.h"

#ifdef __HIP_PLATFORM_AMD__
#include "keccak-amd.hip.inc"
#else
#include "keccak-nvidia.hip.inc"
#endif

#define PROGPOW_PERIOD        3
#define PROGPOW_LANES         16
#define PROGPOW_REGS          32
#define PROGPOW_DAG_LOADS     4
#define PROGPOW_CACHE_BYTES   16384
#define PROGPOW_CACHE_WORDS   (PROGPOW_CACHE_BYTES / 4)  // 4096
#define PROGPOW_CNT_DAG       64
#define PROGPOW_CNT_CACHE     11
#define PROGPOW_CNT_MATH      18
#define PROGPOW_LANES_LOG2    4

#define KAWPOW_MAIN_LOOPS     64

#define FNV_PRIME             0x01000193u
#define FNV_OFFSET_BASIS      0x811c9dc5u

// Coin-specific Keccak padding — injected as constexpr literals by host before RTC.
// Phase 1 (seed):  state[10..24] = KPAD0..KPAD14
// Phase 5 (final): state[16..24] = KPAD0..KPAD8
/* KAWPOW_COIN_PADDING */

// Barrett reduction constants for DAG address modulo — injected by host before RTC.
/* KAWPOW_DAG_CONSTS */


/* KAWPOW_CONST_L1 */
// constant-memory L1 table + load binding
#ifndef KAWPOW_L1_LOAD
#define KAWPOW_L1_LOAD(IDX) l1_cache[(IDX)]
#endif

#define PROGPOW_BPERMUTE_STAGE(MIX0_VAL, IA_RAW, LOOP) do { \
    uint32_t _isrc = (threadIdx.x & ~15u) | ((LOOP) & 15u); \
    (IA_RAW) = PROGPOW_BPERMUTE(_isrc, (MIX0_VAL)); \
} while(0)

#define PROGPOW_DAG_LOAD_STAGE(IA_RAW, DG, LOOP) do { \
    uint32_t _ia = (IA_RAW); \
    uint32_t _iq = __umulhi(_ia, KAWPOW_BARRETT_M) >> KAWPOW_BARRETT_SHIFT; \
    _ia -= _iq * KAWPOW_DAG_NUM_ITEMS_DIV; \
    if (_ia >= KAWPOW_DAG_NUM_ITEMS_DIV) _ia -= KAWPOW_DAG_NUM_ITEMS_DIV; \
    uint64_t _boff = ((uint64_t)_ia << 8) + (((uint64_t)((lane_id ^ (LOOP)) & 15u)) << 4); \
    const uint8_t* _ptr = reinterpret_cast<const uint8_t*>(d_dag) + _boff; \
    (DG) = dag_ptrload_vec4_ptr(_ptr); \
} while(0)

// ===========================================================================
// Static helper macros for software-pipelined DAG loads.
// PROGPOW_ISSUE_DAG computes the next DAG address from mix[0] via Barrett
// reduction and issues a cache-bypass 128-bit load into a dag_vec4.
// Used with PROGPOW_BODY (generated macro injected at **PROGPOW_MACROS**)
// for N-way hash interleaving.
// ===========================================================================

#ifdef __HIP_PLATFORM_AMD__
#define PROGPOW_BPERMUTE(src, val) __builtin_amdgcn_ds_bpermute((src) << 2, (val))
#else
#define PROGPOW_BPERMUTE(src, val) __shfl_sync(0xFFFFFFFF, (val), (src), 16)
#endif

#define KAWPOW_DAG_AUX 0x01

// TODO: Hoist

// GFX11 raw buffer word 3:
//   OOB_SELECT = 3 (disabled)          [29:28]
//   FORMAT     = 22 (32_FLOAT, valid)  [18:12]   <-- required on gfx11
//   TYPE       = 0 (buffer)            [31:30]
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#  define KAWPOW_SRD_FLAGS 0x30016000u
#elif defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
      defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
      defined(__gfx1036__) || \
      defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__)
#  define KAWPOW_SRD_FLAGS 0x30000000u   // gfx10: OOB_SELECT=3 is sufficient
#else
#  define KAWPOW_SRD_FLAGS 0u            // gfx9: no OOB_SELECT field
#endif

struct dag128_u64 { uint64_t lo, hi; };

__device__ __forceinline__ uint32_t dag_word0(const dag128_u64& v) { return (uint32_t)(v.lo);       }
__device__ __forceinline__ uint32_t dag_word1(const dag128_u64& v) { return (uint32_t)(v.lo >> 32); }
__device__ __forceinline__ uint32_t dag_word2(const dag128_u64& v) { return (uint32_t)(v.hi);       }
__device__ __forceinline__ uint32_t dag_word3(const dag128_u64& v) { return (uint32_t)(v.hi >> 32); }

// ---- buffer-intrinsic load, result bit-cast to the good storage type ----
__device__ __forceinline__ dag128_u64
dag_ptrload_plain(const uint8_t* base, uint32_t byte_off)
{
    const uint64_t* p = reinterpret_cast<const uint64_t*>(base + byte_off);
    dag128_u64 out;
    out.lo = p[0];
    out.hi = p[1];
    return out;
}

#ifndef KAWPOW_DAG_VEC_TYPES_DEFINED
#define KAWPOW_DAG_VEC_TYPES_DEFINED
#if defined(__HIP_PLATFORM_AMD__)
typedef uint32_t dag_vec4 __attribute__((ext_vector_type(4)));
typedef uint32_t dag_vec2 __attribute__((ext_vector_type(2)));
#else
typedef uint4 dag_vec4;
typedef uint2 dag_vec2;
#endif
#endif

__device__ __forceinline__ dag128_u64
dag_ptrload_vec4_ptr(const uint8_t* ptr)
{
    const dag_vec4* p = reinterpret_cast<const dag_vec4*>(ptr);
    dag_vec4 t = *p;
#if defined(__HIP_PLATFORM_AMD__)
    return __builtin_bit_cast(dag128_u64, t);
#else
    dag128_u64 out;
    out.lo = (uint64_t)t.x | ((uint64_t)t.y << 32);
    out.hi = (uint64_t)t.z | ((uint64_t)t.w << 32);
    return out;
#endif
}

#define PROGPOW_ISSUE_DAG_FROM(MIX0_VAL, DG, LOOP) do {                              \
    uint32_t _isrc = (threadIdx.x & ~15u) | ((LOOP) & 15u);                          \
    uint32_t _ia   = PROGPOW_BPERMUTE(_isrc, (MIX0_VAL));                            \
    uint32_t _iq   = __umulhi(_ia, KAWPOW_BARRETT_M) >> KAWPOW_BARRETT_SHIFT;        \
    _ia -= _iq * KAWPOW_DAG_NUM_ITEMS_DIV;                                            \
    if (_ia >= KAWPOW_DAG_NUM_ITEMS_DIV) _ia -= KAWPOW_DAG_NUM_ITEMS_DIV;            \
    uint64_t _boff = ((uint64_t)_ia << 8) +                                           \
                     (((uint64_t)((lane_id ^ (LOOP)) & 15u)) << 4);                   \
    const uint8_t* _ptr = reinterpret_cast<const uint8_t*>(d_dag) + _boff;           \
    (DG) = dag_ptrload_vec4_ptr(_ptr);                                                \
} while(0)

#define PROGPOW_ISSUE_DAG(MIX, DG, LOOP) \
    PROGPOW_ISSUE_DAG_FROM((MIX)[0], DG, LOOP)

#define PROGPOW_STORE_COOP(MIX, VALUE, LOOP) do {                                    \
    uint32_t _sglane = lane_id & 15u;                                                \
    uint32_t _isrc   = (threadIdx.x & ~15u) | ((LOOP) & 15u);                        \
    uint32_t _seed   = PROGPOW_BPERMUTE(_isrc, (MIX)[1]);                            \
    uint32_t _q      = __umulhi(_seed, KAWPOW_BARRETT_M) >> KAWPOW_BARRETT_SHIFT;    \
    _seed -= _q * KAWPOW_DAG_NUM_ITEMS_DIV;                                          \
    if (_seed >= KAWPOW_DAG_NUM_ITEMS_DIV) _seed -= KAWPOW_DAG_NUM_ITEMS_DIV;        \
    uint64_t _base   = ((uint64_t)_seed << 8);                                       \
    uint64_t _off    = ((uint64_t)((_sglane ^ (LOOP)) & 15u)) << 4;                  \
    uint8_t* _ptr    = reinterpret_cast<uint8_t*>(d_dag) + _base + _off;             \
    store_glc_slc(reinterpret_cast<uint64_t*>(_ptr), VALUE);                         \
} while(0)

// ProgPoW body macro — injected by host before RTC.
// #define PROGPOW_BODY(MIX, DG) do { ... } while(0)
// MIX = uint32_t[32] mix array name, DG = dag_vec4 pre-loaded DAG data variable
/* PROGPOW_MACROS */

// ============================================================================
// keccak_f800 — AMD-optimized Keccak-f[800] (25 × 32-bit, 22 rounds)
//
// Same techniques as the f1600: in-place ρ/π chain (1 temp, not 25),
// BFI-pattern χ, and last-round pruning for 8-word output.
// ============================================================================

__device__ __forceinline__ uint32_t fnv1a(uint32_t h, uint32_t d)
{
    return (h ^ d) * FNV_PRIME;
}

struct kiss99_state
{
    uint32_t z, w, jsr, jcong;
};

__device__ __forceinline__ uint32_t kiss99(kiss99_state& st)
{
    uint32_t z = st.z, w = st.w, jsr = st.jsr, jcong = st.jcong;
    z     = 36969u * (z & 0xffffu) + (z >> 16);
    w     = 18000u * (w & 0xffffu) + (w >> 16);
    jsr  ^= (jsr << 17);
    jsr  ^= (jsr >> 13);
    jsr  ^= (jsr << 5);
    jcong = 69069u * jcong + 1234567u;
    st.z = z; st.w = w; st.jsr = jsr; st.jcong = jcong;
    return (((z << 16) + w) ^ jcong) + jsr;
}

__device__ __forceinline__ void fill_mix_from_zw(
    uint32_t z, uint32_t w, uint32_t lane_id, uint32_t* __restrict__ mix)
{
    uint32_t jsr   = fnv1a(w, lane_id);
    uint32_t jcong = fnv1a(jsr, lane_id);
    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++) {
        z     = 36969u * (z & 0xffffu) + (z >> 16);
        w     = 18000u * (w & 0xffffu) + (w >> 16);
        jsr  ^= (jsr << 17);
        jsr  ^= (jsr >> 13);
        jsr  ^= (jsr << 5);
        jcong = 69069u * jcong + 1234567u;
        mix[i] = (((z << 16) + w) ^ jcong) + jsr;
    }
}

__device__ __forceinline__ void fill_mix(uint64_t seed, uint32_t lane_id, uint32_t* mix)
{
    uint32_t z     = fnv1a(FNV_OFFSET_BASIS, (uint32_t)seed);
    uint32_t w     = fnv1a(z,                (uint32_t)(seed >> 32));
    uint32_t jsr   = fnv1a(w,                lane_id);
    uint32_t jcong = fnv1a(jsr,              lane_id);

    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++) {
        z     = 36969u * (z & 0xffffu) + (z >> 16);
        w     = 18000u * (w & 0xffffu) + (w >> 16);
        jsr  ^= (jsr << 17);
        jsr  ^= (jsr >> 13);
        jsr  ^= (jsr << 5);
        jcong = 69069u * jcong + 1234567u;
        mix[i] = (((z << 16) + w) ^ jcong) + jsr;
    }
}

__device__ __forceinline__ void fill_mix_2way(
    uint64_t seed0,
    uint64_t seed1,
    uint32_t lane_id,
    uint32_t* mix0,
    uint32_t* mix1)
{
    uint32_t z0     = fnv1a(FNV_OFFSET_BASIS, (uint32_t)seed0);
    uint32_t z1     = fnv1a(FNV_OFFSET_BASIS, (uint32_t)seed1);

    uint32_t w0     = fnv1a(z0, (uint32_t)(seed0 >> 32));
    uint32_t w1     = fnv1a(z1, (uint32_t)(seed1 >> 32));

    uint32_t jsr0   = fnv1a(w0, lane_id);
    uint32_t jsr1   = fnv1a(w1, lane_id);

    uint32_t jcong0 = fnv1a(jsr0, lane_id);
    uint32_t jcong1 = fnv1a(jsr1, lane_id);

    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++) {
        z0     = 36969u * (z0 & 0xffffu) + (z0 >> 16);
        z1     = 36969u * (z1 & 0xffffu) + (z1 >> 16);

        w0     = 18000u * (w0 & 0xffffu) + (w0 >> 16);
        w1     = 18000u * (w1 & 0xffffu) + (w1 >> 16);

        jsr0  ^= (jsr0 << 17);
        jsr1  ^= (jsr1 << 17);

        jsr0  ^= (jsr0 >> 13);
        jsr1  ^= (jsr1 >> 13);

        jsr0  ^= (jsr0 << 5);
        jsr1  ^= (jsr1 << 5);

        jcong0 = 69069u * jcong0 + 1234567u;
        jcong1 = 69069u * jcong1 + 1234567u;

        mix0[i] = (((z0 << 16) + w0) ^ jcong0) + jsr0;
        mix1[i] = (((z1 << 16) + w1) ^ jcong1) + jsr1;
    }
}

__device__ __forceinline__ void fill_mix_4way(
    uint64_t seed0,
    uint64_t seed1,
    uint64_t seed2,
    uint64_t seed3,
    uint32_t lane_id,
    uint32_t* mix0,
    uint32_t* mix1,
    uint32_t* mix2,
    uint32_t* mix3)
{
    uint32_t z0     = fnv1a(FNV_OFFSET_BASIS, (uint32_t)seed0);
    uint32_t z1     = fnv1a(FNV_OFFSET_BASIS, (uint32_t)seed1);
    uint32_t z2     = fnv1a(FNV_OFFSET_BASIS, (uint32_t)seed2);
    uint32_t z3     = fnv1a(FNV_OFFSET_BASIS, (uint32_t)seed3);

    uint32_t w0     = fnv1a(z0, (uint32_t)(seed0 >> 32));
    uint32_t w1     = fnv1a(z1, (uint32_t)(seed1 >> 32));
    uint32_t w2     = fnv1a(z2, (uint32_t)(seed2 >> 32));
    uint32_t w3     = fnv1a(z3, (uint32_t)(seed3 >> 32));

    uint32_t jsr0   = fnv1a(w0, lane_id);
    uint32_t jsr1   = fnv1a(w1, lane_id);
    uint32_t jsr2   = fnv1a(w2, lane_id);
    uint32_t jsr3   = fnv1a(w3, lane_id);

    uint32_t jcong0 = fnv1a(jsr0, lane_id);
    uint32_t jcong1 = fnv1a(jsr1, lane_id);
    uint32_t jcong2 = fnv1a(jsr2, lane_id);
    uint32_t jcong3 = fnv1a(jsr3, lane_id);

    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++) {
        z0     = 36969u * (z0 & 0xffffu) + (z0 >> 16);
        z1     = 36969u * (z1 & 0xffffu) + (z1 >> 16);
        z2     = 36969u * (z2 & 0xffffu) + (z2 >> 16);
        z3     = 36969u * (z3 & 0xffffu) + (z3 >> 16);

        w0     = 18000u * (w0 & 0xffffu) + (w0 >> 16);
        w1     = 18000u * (w1 & 0xffffu) + (w1 >> 16);
        w2     = 18000u * (w2 & 0xffffu) + (w2 >> 16);
        w3     = 18000u * (w3 & 0xffffu) + (w3 >> 16);

        jsr0  ^= (jsr0 << 17);
        jsr1  ^= (jsr1 << 17);
        jsr2  ^= (jsr2 << 17);
        jsr3  ^= (jsr3 << 17);

        jsr0  ^= (jsr0 >> 13);
        jsr1  ^= (jsr1 >> 13);
        jsr2  ^= (jsr2 >> 13);
        jsr3  ^= (jsr3 >> 13);

        jsr0  ^= (jsr0 << 5);
        jsr1  ^= (jsr1 << 5);
        jsr2  ^= (jsr2 << 5);
        jsr3  ^= (jsr3 << 5);

        jcong0 = 69069u * jcong0 + 1234567u;
        jcong1 = 69069u * jcong1 + 1234567u;
        jcong2 = 69069u * jcong2 + 1234567u;
        jcong3 = 69069u * jcong3 + 1234567u;

        mix0[i] = (((z0 << 16) + w0) ^ jcong0) + jsr0;
        mix1[i] = (((z1 << 16) + w1) ^ jcong1) + jsr1;
        mix2[i] = (((z2 << 16) + w2) ^ jcong2) + jsr2;
        mix3[i] = (((z3 << 16) + w3) ^ jcong3) + jsr3;
    }
}

__device__ __forceinline__ uint32_t shfl_lane16(uint32_t val, int src_lane)
{
#ifdef __HIP_PLATFORM_AMD__
    uint32_t base = __lane_id() & ~15u;
    return __builtin_amdgcn_ds_bpermute(((base | src_lane) << 2), val);
#else
    return __shfl_sync(0xFFFFFFFF, val, src_lane, 16);
#endif
}

// ===========================================================================
// dag_vec4 — native C type for 128-bit DAG entries
//
// Replaces HIP uint4 for portability and clean inline ASM constraints.
// Two uint64_t fields map naturally to dwordx2 registers on AMD and
// 64-bit registers on NVIDIA, enabling platform-specific cache-bypass
// loads without fighting vector-type register allocation.
// ===========================================================================

// 128-bit DAG load into dag_vec4.
// AMD:    plain dwordx4 via ext_vector_type(4) (true clang hardware vector)
// NVIDIA: single ld.global.cg.v2.u64 (bypass L1, cache in L2)
// Fallback: plain two-uint64 load
#ifndef KAWPOW_DAG_VEC_TYPES_DEFINED
#define KAWPOW_DAG_VEC_TYPES_DEFINED
#if defined(__HIP_PLATFORM_AMD__)
typedef uint32_t dag_vec4 __attribute__((ext_vector_type(4)));
typedef uint32_t dag_vec2 __attribute__((ext_vector_type(2)));
#else
typedef uint4 dag_vec4;
typedef uint2 dag_vec2;
#endif
#endif

// Non-temporal store — bypasses L2 (GLC on AMD, streaming on NVIDIA).
// Use for write-once data (intermediate buffer) to avoid evicting DAG lines.
__device__ __forceinline__ void store_nt_u32x8(uint32_t* dst, const uint32_t* src)
{
#if defined(__HIP_PLATFORM_AMD__)
    // Two 64-bit non-temporal stores (8 bytes each = 32 bytes total) 
    dag_vec2 v0 = {src[0], src[1]}; 
    dag_vec2 v1 = {src[2], src[3]}; 
    dag_vec2 v2 = {src[4], src[5]}; 
    dag_vec2 v3 = {src[6], src[7]}; 
    __builtin_nontemporal_store(v0, (dag_vec2*)(dst)); 
    __builtin_nontemporal_store(v1, (dag_vec2*)(dst + 2)); 
    __builtin_nontemporal_store(v2, (dag_vec2*)(dst + 4)); 
    __builtin_nontemporal_store(v3, (dag_vec2*)(dst + 6));

#elif defined(__CUDA_ARCH__)
    const uint2* s = reinterpret_cast<const uint2*>(src);
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};" :: "l"(dst),     "r"(s[0].x), "r"(s[0].y));
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};" :: "l"(dst + 2), "r"(s[1].x), "r"(s[1].y));
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};" :: "l"(dst + 4), "r"(s[2].x), "r"(s[2].y));
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};" :: "l"(dst + 6), "r"(s[3].x), "r"(s[3].y));
#else
    for (int i = 0; i < 8; i++) dst[i] = src[i];
#endif
}


// ===========================================================================
// Shared helpers: L1 cache cooperative fill + main loop + mix reduction
// ===========================================================================

__device__ __forceinline__ void cooperative_l1_fill(
    uint32_t* __restrict__ l1_cache,
    const uint32_t* __restrict__ d_l1_cache)
{
    const uint32_t tid = threadIdx.x;
    const uint4* __restrict__ src4 = reinterpret_cast<const uint4*>(d_l1_cache);
    uint4* __restrict__ dst4 = reinterpret_cast<uint4*>(l1_cache);
    constexpr uint32_t CACHE_VEC4 = PROGPOW_CACHE_WORDS / 4;

#pragma unroll
    for (uint32_t i = tid; i < CACHE_VEC4; i += blockDim.x)
        dst4[i] = src4[i];
}

// ---------------------------------------------------------------------------
// 1-way main loop — software-pipelined with loop peeling
// ---------------------------------------------------------------------------
__device__ __forceinline__ void progpow_main_loop(
    uint32_t* mix,
    const uint32_t* __restrict__ l1_cache,
    const uint32_t* __restrict__ d_dag,
    uint32_t lane_id)
{
    dag128_u64 _dg0, _dg1;
    PROGPOW_ISSUE_DAG(mix, _dg0, 0u);

    #pragma unroll 1
    for (uint32_t loop = 0; loop < KAWPOW_MAIN_LOOPS - 2; loop += 2) {
        PROGPOW_BODY_PIPE(mix, _dg0, _dg1, loop + 1);
        PROGPOW_BODY_PIPE(mix, _dg1, _dg0, loop + 2);
    }
    PROGPOW_BODY_PIPE(mix, _dg0, _dg1, KAWPOW_MAIN_LOOPS - 1);
    PROGPOW_BODY(mix, _dg1);
}

// ---------------------------------------------------------------------------
// 2-way main loop — interleave two hashes to hide DAG load latency
// ---------------------------------------------------------------------------
__device__ __forceinline__ void progpow_main_loop_2way(
    uint32_t* m0, uint32_t* m1,
    const uint32_t* __restrict__ l1_cache,
    const uint32_t* __restrict__ d_dag,
    uint32_t lane_id)
{
    dag128_u64 _da0, _da1;
    dag128_u64 _db0, _db1;

    PROGPOW_ISSUE_DAG(m0, _da0, 0u);
    PROGPOW_ISSUE_DAG(m1, _db0, 0u);

    #pragma unroll 1
    for (uint32_t loop = 0; loop < KAWPOW_MAIN_LOOPS - 2; loop += 2) {
        uint32_t _c0, _c1, _m, _m1;
        uint32_t ia0, ia1;
        uint32_t mix0_next0, mix0_next1;

        PROGPOW_BODY_PRECRITICAL(m0, _da0, ia0, mix0_next0, loop + 1u);
        PROGPOW_BODY_PRECRITICAL(m1, _db0, ia1, mix0_next1, loop + 1u);
        PROGPOW_DAG_LOAD_STAGE(ia0, _da1, loop + 1u);
        PROGPOW_DAG_LOAD_STAGE(ia1, _db1, loop + 1u);
        PROGPOW_BODY_PREFILLER(m0);
        PROGPOW_BODY_PREFILLER(m1);
        PROGPOW_BODY_POSTISSUE(m0, _da0, mix0_next0);
        PROGPOW_BODY_POSTISSUE(m1, _db0, mix0_next1);

        PROGPOW_BODY_PRECRITICAL(m0, _da1, ia0, mix0_next0, loop + 2u);
        PROGPOW_BODY_PRECRITICAL(m1, _db1, ia1, mix0_next1, loop + 2u);
        PROGPOW_DAG_LOAD_STAGE(ia0, _da0, loop + 2u);
        PROGPOW_DAG_LOAD_STAGE(ia1, _db0, loop + 2u);
        PROGPOW_BODY_PREFILLER(m0);
        PROGPOW_BODY_PREFILLER(m1);
        PROGPOW_BODY_POSTISSUE(m0, _da1, mix0_next0);
        PROGPOW_BODY_POSTISSUE(m1, _db1, mix0_next1);
    }

    PROGPOW_BODY_PIPE(m0, _da0, _da1, KAWPOW_MAIN_LOOPS - 1u);
    PROGPOW_BODY_PIPE(m1, _db0, _db1, KAWPOW_MAIN_LOOPS - 1u);
    PROGPOW_BODY(m0, _da1);
    PROGPOW_BODY(m1, _db1);
}

// ---------------------------------------------------------------------------
// 4-way main loop — interleave four hashes for maximum latency hiding
// ---------------------------------------------------------------------------
__device__ __forceinline__ void progpow_main_loop_4way(
    uint32_t* m0, uint32_t* m1, uint32_t* m2, uint32_t* m3,
    const uint32_t* __restrict__ l1_cache,
    const uint32_t* __restrict__ d_dag,
    uint32_t lane_id)
{
    dag128_u64 _da0, _da1;
    dag128_u64 _db0, _db1;
    dag128_u64 _dc0, _dc1;
    dag128_u64 _dd0, _dd1;

    PROGPOW_ISSUE_DAG(m0, _da0, 0u);
    PROGPOW_ISSUE_DAG(m1, _db0, 0u);
    PROGPOW_ISSUE_DAG(m2, _dc0, 0u);
    PROGPOW_ISSUE_DAG(m3, _dd0, 0u);

    #pragma unroll 1
    for (uint32_t loop = 0; loop < KAWPOW_MAIN_LOOPS - 2; loop += 2) {
        uint32_t _c0, _c1, _m, _m1;
        uint32_t ia0, ia1, ia2, ia3;
        uint32_t mix0_next0, mix0_next1, mix0_next2, mix0_next3;

        PROGPOW_BODY_PRECRITICAL(m0, _da0, ia0, mix0_next0, loop + 1u);
        PROGPOW_BODY_PRECRITICAL(m1, _db0, ia1, mix0_next1, loop + 1u);
        PROGPOW_BODY_PRECRITICAL(m2, _dc0, ia2, mix0_next2, loop + 1u);
        PROGPOW_BODY_PRECRITICAL(m3, _dd0, ia3, mix0_next3, loop + 1u);
        PROGPOW_DAG_LOAD_STAGE(ia0, _da1, loop + 1u);
        PROGPOW_DAG_LOAD_STAGE(ia1, _db1, loop + 1u);
        PROGPOW_DAG_LOAD_STAGE(ia2, _dc1, loop + 1u);
        PROGPOW_DAG_LOAD_STAGE(ia3, _dd1, loop + 1u);
        PROGPOW_BODY_PREFILLER(m0);
        PROGPOW_BODY_PREFILLER(m1);
        PROGPOW_BODY_PREFILLER(m2);
        PROGPOW_BODY_PREFILLER(m3);
        PROGPOW_BODY_POSTISSUE(m0, _da0, mix0_next0);
        PROGPOW_BODY_POSTISSUE(m1, _db0, mix0_next1);
        PROGPOW_BODY_POSTISSUE(m2, _dc0, mix0_next2);
        PROGPOW_BODY_POSTISSUE(m3, _dd0, mix0_next3);

        PROGPOW_BODY_PRECRITICAL(m0, _da1, ia0, mix0_next0, loop + 2u);
        PROGPOW_BODY_PRECRITICAL(m1, _db1, ia1, mix0_next1, loop + 2u);
        PROGPOW_BODY_PRECRITICAL(m2, _dc1, ia2, mix0_next2, loop + 2u);
        PROGPOW_BODY_PRECRITICAL(m3, _dd1, ia3, mix0_next3, loop + 2u);
        PROGPOW_DAG_LOAD_STAGE(ia0, _da0, loop + 2u);
        PROGPOW_DAG_LOAD_STAGE(ia1, _db0, loop + 2u);
        PROGPOW_DAG_LOAD_STAGE(ia2, _dc0, loop + 2u);
        PROGPOW_DAG_LOAD_STAGE(ia3, _dd0, loop + 2u);
        PROGPOW_BODY_PREFILLER(m0);
        PROGPOW_BODY_PREFILLER(m1);
        PROGPOW_BODY_PREFILLER(m2);
        PROGPOW_BODY_PREFILLER(m3);
        PROGPOW_BODY_POSTISSUE(m0, _da1, mix0_next0);
        PROGPOW_BODY_POSTISSUE(m1, _db1, mix0_next1);
        PROGPOW_BODY_POSTISSUE(m2, _dc1, mix0_next2);
        PROGPOW_BODY_POSTISSUE(m3, _dd1, mix0_next3);
    }

    PROGPOW_BODY_PIPE(m0, _da0, _da1, KAWPOW_MAIN_LOOPS - 1u);
    PROGPOW_BODY_PIPE(m1, _db0, _db1, KAWPOW_MAIN_LOOPS - 1u);
    PROGPOW_BODY_PIPE(m2, _dc0, _dc1, KAWPOW_MAIN_LOOPS - 1u);
    PROGPOW_BODY_PIPE(m3, _dd0, _dd1, KAWPOW_MAIN_LOOPS - 1u);

    PROGPOW_BODY(m0, _da1);
    PROGPOW_BODY(m1, _db1);
    PROGPOW_BODY(m2, _dc1);
    PROGPOW_BODY(m3, _dd1);
}

__device__ __forceinline__ void progpow_reduce(
    const uint32_t* mix, uint32_t* digest)
{
    uint32_t lane_result = FNV_OFFSET_BASIS;
    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        lane_result = fnv1a(lane_result, mix[i]);

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        digest[i] = FNV_OFFSET_BASIS;
        digest[i] = fnv1a(digest[i], shfl_lane16(lane_result, i));
        digest[i] = fnv1a(digest[i], shfl_lane16(lane_result, i + 8));
    }
}

__device__ __forceinline__ void progpow_reduce_2way(
    const uint32_t* mix0,
    const uint32_t* mix1,
    uint32_t* digest0,
    uint32_t* digest1)
{
    uint32_t lane_result0 = FNV_OFFSET_BASIS;
    uint32_t lane_result1 = FNV_OFFSET_BASIS;

    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++) {
        lane_result0 = fnv1a(lane_result0, mix0[i]);
        lane_result1 = fnv1a(lane_result1, mix1[i]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t a0 = shfl_lane16(lane_result0, i);
        uint32_t b0 = shfl_lane16(lane_result0, i + 8);
        uint32_t a1 = shfl_lane16(lane_result1, i);
        uint32_t b1 = shfl_lane16(lane_result1, i + 8);

        uint32_t d0 = FNV_OFFSET_BASIS;
        uint32_t d1 = FNV_OFFSET_BASIS;

        d0 = fnv1a(d0, a0);
        d0 = fnv1a(d0, b0);

        d1 = fnv1a(d1, a1);
        d1 = fnv1a(d1, b1);

        digest0[i] = d0;
        digest1[i] = d1;
    }
}

// ===========================================================================
// Strategy 0 — Monolithic kernel (seed keccak + progpow + final keccak)
// ===========================================================================
__device__ __forceinline__ void kawpow_seed_keccak_lane0(
    const uint32_t* __restrict__ d_header,
    uint64_t nonce,
    uint32_t* __restrict__ seed_state_out)
{
    uint32_t state[25];

    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = d_header[i];

    state[8]  = (uint32_t)(nonce);
    state[9]  = (uint32_t)(nonce >> 32);

    state[10] = KPAD0;  state[11] = KPAD1;  state[12] = KPAD2;
    state[13] = KPAD3;  state[14] = KPAD4;  state[15] = KPAD5;
    state[16] = KPAD6;  state[17] = KPAD7;  state[18] = KPAD8;
    state[19] = KPAD9;  state[20] = KPAD10; state[21] = KPAD11;
    state[22] = KPAD12; state[23] = KPAD13; state[24] = KPAD14;

    keccak_f800<true>(state);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        seed_state_out[i] = state[i];
}

__device__ __forceinline__ void kawpow_final_keccak_lane0(
    uint64_t nonce,
    uint32_t hash_id,
    const uint32_t* __restrict__ seed_state,
    uint32_t digest0,
    uint32_t digest1,
    uint32_t digest2,
    uint32_t digest3,
    uint32_t digest4,
    uint32_t digest5,
    uint32_t digest6,
    uint32_t digest7,
    const uint64_t* __restrict__ d_target,
    uint64_t* __restrict__ d_solutions,
    uint32_t* __restrict__ d_solution_flag,
    uint32_t* __restrict__ d_result_hashes)
{
    uint32_t state[25];

    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = seed_state[i];

    state[8]  = digest0;
    state[9]  = digest1;
    state[10] = digest2;
    state[11] = digest3;
    state[12] = digest4;
    state[13] = digest5;
    state[14] = digest6;
    state[15] = digest7;

    state[16] = KPAD0;  state[17] = KPAD1;  state[18] = KPAD2;
    state[19] = KPAD3;  state[20] = KPAD4;  state[21] = KPAD5;
    state[22] = KPAD6;  state[23] = KPAD7;  state[24] = KPAD8;

    keccak_f800<true>(state);

    if (d_result_hashes != nullptr)
    {
        uint32_t base = hash_id * 16u;
        d_result_hashes[base + 0]  = digest0;
        d_result_hashes[base + 1]  = digest1;
        d_result_hashes[base + 2]  = digest2;
        d_result_hashes[base + 3]  = digest3;
        d_result_hashes[base + 4]  = digest4;
        d_result_hashes[base + 5]  = digest5;
        d_result_hashes[base + 6]  = digest6;
        d_result_hashes[base + 7]  = digest7;
        d_result_hashes[base + 8]  = state[0];
        d_result_hashes[base + 9]  = state[1];
        d_result_hashes[base + 10] = state[2];
        d_result_hashes[base + 11] = state[3];
        d_result_hashes[base + 12] = state[4];
        d_result_hashes[base + 13] = state[5];
        d_result_hashes[base + 14] = state[6];
        d_result_hashes[base + 15] = state[7];
    }

    const uint64_t h0 = ((uint64_t)state[1] << 32) | state[0];
    const uint64_t h1 = ((uint64_t)state[3] << 32) | state[2];
    const uint64_t h2 = ((uint64_t)state[5] << 32) | state[4];
    const uint64_t h3 = ((uint64_t)state[7] << 32) | state[6];

    bool below = false;
    bool equal = true;

    if (equal) {
        uint64_t t = d_target[3];
        if (h3 < t) { below = true; equal = false; }
        else if (h3 > t) { below = false; equal = false; }
    }
    if (equal) {
        uint64_t t = d_target[2];
        if (h2 < t) { below = true; equal = false; }
        else if (h2 > t) { below = false; equal = false; }
    }
    if (equal) {
        uint64_t t = d_target[1];
        if (h1 < t) { below = true; equal = false; }
        else if (h1 > t) { below = false; equal = false; }
    }
    if (equal) {
        uint64_t t = d_target[0];
        if (h0 < t) { below = true; equal = false; }
        else if (h0 > t) { below = false; equal = false; }
    }

    if (below)
    {
        if (d_solution_flag) *d_solution_flag = 1;
        uint32_t idx = (uint32_t)atomicAdd((unsigned long long*)&d_solutions[0], 1ULL);
        if (idx < 1024)
        {
            d_solutions[1 + idx * 5 + 0] = nonce;
            d_solutions[1 + idx * 5 + 1] = ((uint64_t)digest1 << 32) | digest0;
            d_solutions[1 + idx * 5 + 2] = ((uint64_t)digest3 << 32) | digest2;
            d_solutions[1 + idx * 5 + 3] = ((uint64_t)digest5 << 32) | digest4;
            d_solutions[1 + idx * 5 + 4] = ((uint64_t)digest7 << 32) | digest6;
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 6)
void kawpow_hash_kernel(
    const uint32_t* __restrict__ d_header,        // 8 × uint32 = 32 bytes
    const uint32_t* __restrict__ d_dag,            // full DAG (uint32 array)
    uint64_t                     nonce_start,
    uint32_t                     batch_size,
    const uint64_t* __restrict__ d_target,         // 4 × uint64 = 32 bytes (LE)
    uint64_t* __restrict__       d_solutions,      // [0]=count(u64), [1+i*5]=nonce(u64), [+1..4]=mix_hash(4×u64)
    uint32_t* __restrict__       d_solution_flag,  // nullable mapped host flag, written only on solution
    uint32_t                     block_number,
    uint32_t* __restrict__       d_result_hashes,  // nullable: [hash_id*16+0..7]=mix, [+8..15]=final
    const uint32_t* __restrict__ d_l1_cache)       // first 16KB of DAG, uploaded once per epoch
{
    const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id   = global_id & 0xF;
    const uint32_t hash_id   = global_id >> PROGPOW_LANES_LOG2;

    const uint64_t nonce = nonce_start + (uint64_t)hash_id;

    // LDS layout: [l1_cache: 4096 words][seed_state: hashes_per_block * 8 words]
    extern __shared__ uint32_t lds_raw[];
    uint32_t* __restrict__ l1_cache = lds_raw;

    cooperative_l1_fill(l1_cache, d_l1_cache);

    // Lane 0 writes the per-hash seed state while the block finishes L1 fill.
    // No early return — all threads must reach __syncthreads().
    const bool valid = (hash_id < batch_size);

    uint32_t mix[PROGPOW_REGS];

    uint32_t* lds_seed_state = &lds_raw[PROGPOW_CACHE_WORDS];
    uint32_t* seed_state = &lds_seed_state[(threadIdx.x >> PROGPOW_LANES_LOG2) * 8u];

    if (valid && lane_id == 0)
        kawpow_seed_keccak_lane0(d_header, nonce, seed_state);

    __syncthreads();  // L1 fill + seed state writes all visible

    if (!valid) return;

    const uint64_t seed = (uint64_t)seed_state[0] | ((uint64_t)seed_state[1] << 32);
    fill_mix(seed, lane_id, mix);

    // ---- Phase 2–4: main loop + reduction ----
    progpow_main_loop(mix, l1_cache, d_dag, lane_id);

    uint32_t digest[8];
    progpow_reduce(mix, digest);

    if (lane_id == 0)
        kawpow_final_keccak_lane0(
            nonce, hash_id, seed_state,
            digest[0], digest[1], digest[2], digest[3],
            digest[4], digest[5], digest[6], digest[7],
            d_target, d_solutions, d_solution_flag, d_result_hashes);
}

#if defined(__HIP_PLATFORM_AMD__)
  // GFX10 / RDNA1
  #if defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1013__)
    #define KAWPOW_AMD_HAS_DLC 1
  // GFX10.3 / RDNA2
  #elif defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
        defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
        defined(__gfx1036__)
    #define KAWPOW_AMD_HAS_DLC 1
  // GFX11 / RDNA3+
  #elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
        defined(__gfx1103__)
    #define KAWPOW_AMD_HAS_DLC 1
  #else
    // gfx906 / gfx90a / older GCN/CDNA: stay conservative
    #define KAWPOW_AMD_HAS_DLC 0
  #endif
#endif

__device__ __forceinline__ void store_nt_u64(uint64_t* ptr, uint64_t val)
{
#if defined(__HIP_PLATFORM_AMD__)
  #if KAWPOW_AMD_HAS_DLC
    asm volatile(
        "global_store_dwordx2 %0, %1, off glc dlc"
        :
        : "v"(ptr), "v"(val));
  #else
    asm volatile(
        "global_store_dwordx2 %0, %1, off glc"
        :
        : "v"(ptr), "v"(val));
  #endif
#else
    asm volatile(
        "st.global.cg.u64 [%0], %1;"
        :
        : "l"(ptr), "l"(val));
#endif
}

__device__ __forceinline__ void store_nt_u32(uint32_t* ptr, uint32_t val)
{
#if defined(__HIP_PLATFORM_AMD__)
  #if KAWPOW_AMD_HAS_DLC
    asm volatile(
        "global_store_dword %0, %1, off glc dlc"
        :
        : "v"(ptr), "v"(val));
  #else
    asm volatile(
        "global_store_dword %0, %1, off glc"
        :
        : "v"(ptr), "v"(val));
  #endif
#else
    asm volatile(
        "st.global.cg.u32 [%0], %1;"
        :
        : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void store_seed_u64x4(uint32_t* dst, const uint32_t state[8])
{
    const uint64_t* src64 = reinterpret_cast<const uint64_t*>(state);
    uint64_t* dst64 = reinterpret_cast<uint64_t*>(dst);

    store_nt_u64(&dst64[0], src64[0]);
    store_nt_u64(&dst64[1], src64[1]);
    store_nt_u64(&dst64[2], src64[2]);
    store_nt_u64(&dst64[3], src64[3]);
}

__device__ __forceinline__ void kawpow_make_seed_state(
    const uint32_t* __restrict__ d_header,
    uint64_t nonce,
    uint32_t state[25])
{
    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = d_header[i];

    state[8]  = (uint32_t)(nonce);
    state[9]  = (uint32_t)(nonce >> 32);

    state[10] = KPAD0;  state[11] = KPAD1;  state[12] = KPAD2;
    state[13] = KPAD3;  state[14] = KPAD4;  state[15] = KPAD5;
    state[16] = KPAD6;  state[17] = KPAD7;  state[18] = KPAD8;
    state[19] = KPAD9;  state[20] = KPAD10; state[21] = KPAD11;
    state[22] = KPAD12; state[23] = KPAD13; state[24] = KPAD14;

    keccak_f800<true>(state);
}

__device__ __forceinline__ uint32_t kawpow_broadcast_lane16(uint32_t val, int src_lane)
{
#ifdef __HIP_PLATFORM_AMD__
    uint32_t base = __lane_id() & ~15u;
    return PROGPOW_BPERMUTE(base + (uint32_t)src_lane, val);
#else
    return __shfl_sync(0xFFFFFFFF, val, src_lane, 16);
#endif
}

// ===========================================================================
// Strategy 1 — Split kernels
//
// Intermediate buffer layout per hash (16 × uint32 = 64 bytes):
//   [hash_id * 16 +  0.. 7] = seed state[0..7]   (written by seed, read by final)
//   [hash_id * 16 +  8..15] = digest[0..7]        (written by progpow, read by final)
// ===========================================================================

// ---------------------------------------------------------------------------
// Split kernel A: Seed Keccak  (1 thread per hash, no LDS, no DAG)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 12)
void kawpow_seed_kernel(
    const uint32_t* __restrict__ d_header,
    uint64_t                     nonce_start,
    uint32_t                     batch_size,
    uint32_t* __restrict__       d_intermediate)
{
    const uint32_t hash_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hash_id >= batch_size) return;

    const uint64_t nonce = nonce_start + (uint64_t)hash_id;

    uint32_t state[25];

    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = d_header[i];

    state[8]  = (uint32_t)(nonce);
    state[9]  = (uint32_t)(nonce >> 32);

    state[10] = KPAD0;  state[11] = KPAD1;  state[12] = KPAD2;
    state[13] = KPAD3;  state[14] = KPAD4;  state[15] = KPAD5;
    state[16] = KPAD6;  state[17] = KPAD7;  state[18] = KPAD8;
    state[19] = KPAD9;  state[20] = KPAD10; state[21] = KPAD11;
    state[22] = KPAD12; state[23] = KPAD13; state[24] = KPAD14;

    keccak_f800<true>(state);

    // Write seed state to intermediate buffer (non-temporal to avoid evicting DAG from L2)
    uint32_t base = hash_id * 16u;
    store_seed_u64x4(d_intermediate + base, state);
}

// ---------------------------------------------------------------------------
// Split kernel B: ProgPoW main loop  (16 threads per hash, LDS = 16KB L1 only)
// ---------------------------------------------------------------------------
extern "C" __global__
void kawpow_progpow_kernel(
    const uint32_t* __restrict__ d_dag,
    uint32_t                     batch_size,
    const uint32_t* __restrict__ d_l1_cache,
    uint32_t* __restrict__       d_intermediate)
{
    const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id   = global_id & 0xF;
    const uint32_t hash_id   = global_id >> PROGPOW_LANES_LOG2;

    // LDS: L1 cache only — no seed_state needed (final keccak is a separate kernel)
    extern __shared__ uint32_t lds_raw[];
    uint32_t* __restrict__ l1_cache = lds_raw;

    cooperative_l1_fill(l1_cache, d_l1_cache);

    const bool valid = (hash_id < batch_size);
    uint32_t mix[PROGPOW_REGS];
    uint32_t base = 0;

    if (valid) {
        base = hash_id * 16u;
        uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);
        fill_mix(seed, lane_id, mix);
    }

    __syncthreads();

    if (!valid) return;

    progpow_main_loop(mix, l1_cache, d_dag, lane_id);

    uint32_t digest[8];
    progpow_reduce(mix, digest);

    if (lane_id == 0) {
        store_nt_u32x8(d_intermediate + base + 8, digest);
    }
}

// ---------------------------------------------------------------------------
// Split kernel B2: ProgPoW main loop — global L1 (no LDS, 0 shared mem)
// Same as kawpow_progpow_kernel but reads d_l1_cache from global/L1$ directly.
// 16KB fits in hardware L1 on RDNA (16KB scalar + 128KB vector) and NVIDIA.
// ---------------------------------------------------------------------------
extern "C" __global__
void kawpow_progpow_global_kernel(
    const uint32_t* __restrict__ d_dag,
    uint32_t                     batch_size,
    const uint32_t* __restrict__ d_l1_cache,
    uint32_t* __restrict__       d_intermediate)
{
    const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id   = global_id & 0xF;
    const uint32_t hash_id   = global_id >> PROGPOW_LANES_LOG2;

    if (hash_id >= batch_size) return;

    // Read seed from intermediate buffer (state[0..1])
    uint32_t base = hash_id * 16u;
    uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);

    uint32_t mix[PROGPOW_REGS];
    fill_mix(seed, lane_id, mix);

    // Use d_l1_cache directly — no LDS, rely on hardware L1 cache
    progpow_main_loop(mix, d_l1_cache, d_dag, lane_id);

    uint32_t digest[8];
    progpow_reduce(mix, digest);

    // Write digest to intermediate buffer (lane 0 only — all lanes computed same digest)
    if (lane_id == 0) {
        store_nt_u32x8(d_intermediate + base + 8, digest);
    }
}

__device__ __forceinline__ void store_digest_coop_16x8_u32(
    uint32_t* __restrict__ dst0,
    const uint32_t digest0[8],
    uint32_t* __restrict__ dst1,
    const uint32_t digest1[8],
    bool have_h1)
{
    const uint32_t lane_id = (blockIdx.x * blockDim.x + threadIdx.x) & 0xF;

    if (lane_id < 8u) {
        store_nt_u32(&dst0[lane_id], digest0[lane_id]);
    } else if (have_h1) {
        store_nt_u32(&dst1[lane_id - 8u], digest1[lane_id - 8u]);
    }
}

// ---------------------------------------------------------------------------
// Split kernel B-2way: ProgPoW main loop — 2-way hash interleaving
// Each wavefront group processes 2 hashes, interleaving their ProgPoW
// iterations to hide DAG load latency across hashes.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 2)
void kawpow_progpow_kernel_2way(
    const uint32_t* __restrict__ d_dag,
    uint32_t                     batch_size,
    const uint32_t* __restrict__ d_l1_cache,
    uint32_t* __restrict__       d_intermediate)
{
    const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id   = global_id & 0xF;
    const uint32_t group_id  = global_id >> PROGPOW_LANES_LOG2;
    const uint32_t hash_id_0 = group_id * 2u;

    extern __shared__ uint32_t lds_raw[];
    uint32_t* __restrict__ l1_cache = lds_raw;

    cooperative_l1_fill(l1_cache, d_l1_cache);

    const bool valid = (hash_id_0 < batch_size);

    uint32_t m0[PROGPOW_REGS];
    uint32_t m1[PROGPOW_REGS];
    bool have_h1 = false;

    if (valid) {
        uint32_t base0 = hash_id_0 * 16u;
        uint64_t seed0 = (uint64_t)d_intermediate[base0] |
                        ((uint64_t)d_intermediate[base0 + 1] << 32);
        fill_mix(seed0, lane_id, m0);

        have_h1 = (hash_id_0 + 1 < batch_size);
        if (have_h1) {
            uint32_t base1 = (hash_id_0 + 1) * 16u;
            uint64_t seed1 = (uint64_t)d_intermediate[base1] |
                            ((uint64_t)d_intermediate[base1 + 1] << 32);
            fill_mix(seed1, lane_id, m1);
        }
    }

    __syncthreads();

    if (!valid) return;

    if (have_h1) {
        progpow_main_loop_2way(m0, m1, l1_cache, d_dag, lane_id);
    } else {
        progpow_main_loop(m0, l1_cache, d_dag, lane_id);
    }

    {
        uint32_t digest0[8];
        progpow_reduce(m0, digest0);

        uint32_t digest1[8];
        if (have_h1) {
            progpow_reduce(m1, digest1);
        }

        store_digest_coop_16x8_u32(
            d_intermediate + hash_id_0 * 16u + 8,
            digest0,
            d_intermediate + (hash_id_0 + 1) * 16u + 8,
            digest1,
            have_h1);
    }
}

// ---------------------------------------------------------------------------
// Packed split path: seed64 -> digest slot reuse.
// Intermediate footprint is 8 uint32 per hash. Seed writes only state[0..1]
// packed by ProgPoW block for coalesced PP reads; ProgPoW overwrites the same
// per-hash slot with digest[0..7]; final recomputes the seed state.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 12)
void kawpow_seed_kernel_seed64(
    const uint32_t* __restrict__ d_header,
    uint64_t                     nonce_start,
    uint32_t                     batch_size,
    uint32_t* __restrict__       d_seed_digest)
{
    const uint32_t hash_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hash_id >= batch_size) return;

    const uint64_t nonce = nonce_start + (uint64_t)hash_id;

    uint32_t state[25];

    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = d_header[i];

    state[8]  = (uint32_t)(nonce);
    state[9]  = (uint32_t)(nonce >> 32);

    state[10] = KPAD0;  state[11] = KPAD1;  state[12] = KPAD2;
    state[13] = KPAD3;  state[14] = KPAD4;  state[15] = KPAD5;
    state[16] = KPAD6;  state[17] = KPAD7;  state[18] = KPAD8;
    state[19] = KPAD9;  state[20] = KPAD10; state[21] = KPAD11;
    state[22] = KPAD12; state[23] = KPAD13; state[24] = KPAD14;

    keccak_f800<true>(state);

    // Precompute lane-independent fill_mix prefix
    const uint32_t z = fnv1a(FNV_OFFSET_BASIS, state[0]);
    const uint32_t w = fnv1a(z, state[1]);

    // Store z,w at [0..1]; PP will overwrite [0..7] with digest later.
    // Final kernel recomputes seed keccak independently.
    const uint32_t base = hash_id * 8u;
    store_nt_u64(
        reinterpret_cast<uint64_t*>(d_seed_digest + base),
        (uint64_t)z | ((uint64_t)w << 32));
}

extern "C" __global__ __launch_bounds__(256, 2)
void kawpow_progpow_kernel_2way_seed64_digest(
    const uint32_t* __restrict__ d_dag,
    uint32_t                     batch_size,
    const uint32_t* __restrict__ d_l1_cache,
    uint32_t* __restrict__       d_seed_digest)
{
    const uint32_t global_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id    = global_id & 0xF;
    const uint32_t group_id   = global_id >> PROGPOW_LANES_LOG2;
    const uint32_t hash_id_0  = group_id * 2u;

    extern __shared__ uint32_t lds_raw[];
    uint32_t* __restrict__ l1_cache = lds_raw;

    cooperative_l1_fill(l1_cache, d_l1_cache);

    const bool valid   = (hash_id_0 < batch_size);
    const bool have_h1 = valid && (hash_id_0 + 1 < batch_size);

    // --- Read precomputed z,w from seed kernel (1 load per hash, on 1 lane) ---
    uint32_t z0 = 0, w0 = 0;
    uint32_t z1 = 0, w1 = 0;

    if (valid && lane_id == 0) {
        const uint32_t base0 = hash_id_0 * 8u;
        z0 = d_seed_digest[base0];
        w0 = d_seed_digest[base0 + 1];
    }
    if (have_h1 && lane_id == 1) {
        const uint32_t base1 = (hash_id_0 + 1) * 8u;
        z1 = d_seed_digest[base1];
        w1 = d_seed_digest[base1 + 1];
    }

    // --- Broadcast z,w to all 16 lanes in the group ---
    z0 = kawpow_broadcast_lane16(z0, 0);
    w0 = kawpow_broadcast_lane16(w0, 0);
    z1 = kawpow_broadcast_lane16(z1, 1);
    w1 = kawpow_broadcast_lane16(w1, 1);

    // --- Fill mix directly from z,w — no seed-to-fnv1a work on the hot path ---
    uint32_t m0[PROGPOW_REGS];
    uint32_t m1[PROGPOW_REGS];

    fill_mix_from_zw(z0, w0, lane_id, m0);
    if (have_h1) {
        fill_mix_from_zw(z1, w1, lane_id, m1);
    }

    __syncthreads();  // L1 fill visibility

    if (!valid) return;

    // --- Main loop ---
    if (have_h1) {
        progpow_main_loop_2way(m0, m1, l1_cache, d_dag, lane_id);
    } else {
        progpow_main_loop(m0, l1_cache, d_dag, lane_id);
    }

    // --- Reduce + store digest ---
    // Digest overwrites the z,w slot — seed data is consumed, no longer needed.
    {
        uint32_t digest0[8];
        progpow_reduce(m0, digest0);

        uint32_t digest1[8];
        if (have_h1)
            progpow_reduce(m1, digest1);

        store_digest_coop_16x8_u32(
            d_seed_digest + hash_id_0 * 8u,
            digest0,
            d_seed_digest + (hash_id_0 + 1) * 8u,
            digest1,
            have_h1);
    }
}

__device__ __forceinline__ void store_digest_coop_32x8_u32(
    uint32_t* dst0, const uint32_t digest0[8],
    uint32_t* dst1, const uint32_t digest1[8],
    uint32_t* dst2, const uint32_t digest2[8],
    uint32_t* dst3, const uint32_t digest3[8],
    uint32_t num_hashes)
{
    const uint32_t lane_id = threadIdx.x & 31u;

    if (lane_id < 8u) {
        store_nt_u32(&dst0[lane_id], digest0[lane_id]);
    } else if (lane_id < 16u && num_hashes >= 2) {
        const uint32_t l = lane_id - 8u;
        store_nt_u32(&dst1[l], digest1[l]);
    } else if (lane_id < 24u && num_hashes >= 3) {
        const uint32_t l = lane_id - 16u;
        store_nt_u32(&dst2[l], digest2[l]);
    } else if (lane_id < 32u && num_hashes >= 4) {
        const uint32_t l = lane_id - 24u;
        store_nt_u32(&dst3[l], digest3[l]);
    }
}

// ---------------------------------------------------------------------------
// Split kernel B-4way: ProgPoW main loop — 4-way hash interleaving
// Each wavefront group processes 4 hashes for maximum DAG latency hiding.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void kawpow_progpow_kernel_4way(
    const uint32_t* __restrict__ d_dag,
    uint32_t                     batch_size,
    const uint32_t* __restrict__ d_l1_cache,
    uint32_t* __restrict__       d_intermediate)
{
    const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id   = global_id & 0xF;
    const uint32_t group_id  = global_id >> PROGPOW_LANES_LOG2;
    const uint32_t hash_id_0 = group_id * 4u;

    extern __shared__ uint32_t lds_raw[];
    uint32_t* __restrict__ l1_cache = lds_raw;

    cooperative_l1_fill(l1_cache, d_l1_cache);

    const bool valid = (hash_id_0 < batch_size);

    uint32_t m0[PROGPOW_REGS], m1[PROGPOW_REGS], m2[PROGPOW_REGS], m3[PROGPOW_REGS];
    uint32_t num_hashes = 0;

    if (valid) {
        num_hashes = min(4u, batch_size - hash_id_0);

        {
            uint32_t base = hash_id_0 * 16u;
            uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);
            fill_mix(seed, lane_id, m0);
        }
        if (num_hashes >= 2) {
            uint32_t base = (hash_id_0 + 1) * 16u;
            uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);
            fill_mix(seed, lane_id, m1);
        }
        if (num_hashes >= 3) {
            uint32_t base = (hash_id_0 + 2) * 16u;
            uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);
            fill_mix(seed, lane_id, m2);
        }
        if (num_hashes >= 4) {
            uint32_t base = (hash_id_0 + 3) * 16u;
            uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);
            fill_mix(seed, lane_id, m3);
        }
    }

    __syncthreads();

    if (!valid) return;

    // Dispatch to best N-way variant based on remaining hashes
    if (num_hashes >= 4) {
        progpow_main_loop_4way(m0, m1, m2, m3, l1_cache, d_dag, lane_id);
    } else if (num_hashes == 3) {
        // 2-way + 1-way — no 3-way variant to avoid code bloat
        progpow_main_loop_2way(m0, m1, l1_cache, d_dag, lane_id);
        progpow_main_loop(m2, l1_cache, d_dag, lane_id);
    } else if (num_hashes == 2) {
        progpow_main_loop_2way(m0, m1, l1_cache, d_dag, lane_id);
    } else {
        progpow_main_loop(m0, l1_cache, d_dag, lane_id);
    }

    // Reduce and write all hashes
    {
        uint32_t digest0[8];
        progpow_reduce(m0, digest0);

        uint32_t digest1[8];
        if (num_hashes >= 2) {
            progpow_reduce(m1, digest1);
        }

        uint32_t digest2[8];
        if (num_hashes >= 3) {
            progpow_reduce(m2, digest2);
        }

        uint32_t digest3[8];
        if (num_hashes >= 4) {
            progpow_reduce(m3, digest3);
        }

        store_digest_coop_16x8_u32(
            d_intermediate + hash_id_0 * 16u + 8,
            digest0,
            d_intermediate + (hash_id_0 + 1) * 16u + 8,
            digest1,
            num_hashes >= 2);

        if (num_hashes >= 3) {
            store_digest_coop_16x8_u32(
                d_intermediate + (hash_id_0 + 2) * 16u + 8,
                digest2,
                d_intermediate + (hash_id_0 + 3) * 16u + 8,
                digest3,
                num_hashes >= 4);
        }
    }
}

// ---------------------------------------------------------------------------
// Split kernel C: Final Keccak + target check  (1 thread per hash, no LDS)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void kawpow_final_kernel(
    uint64_t                     nonce_start,
    uint32_t                     batch_size,
    const uint64_t* __restrict__ d_target,
    uint64_t* __restrict__       d_solutions,
    uint32_t* __restrict__       d_solution_flag,
    uint32_t* __restrict__       d_result_hashes,
    const uint32_t* __restrict__ d_intermediate)
{
    const uint32_t hash_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hash_id >= batch_size) return;

    const uint64_t nonce = nonce_start + (uint64_t)hash_id;
    const uint32_t base = hash_id * 16u;

    uint32_t state[25];

    // Load 64-byte record as 4 x uint4
    const uint4* rec4 = reinterpret_cast<const uint4*>(d_intermediate + base);
    uint4 v0 = rec4[0];
    uint4 v1 = rec4[1];
    uint4 v2 = rec4[2];
    uint4 v3 = rec4[3];

    // Seed state
    state[0] = v0.x;  state[1] = v0.y;  state[2] = v0.z;  state[3] = v0.w;
    state[4] = v1.x;  state[5] = v1.y;  state[6] = v1.z;  state[7] = v1.w;

    // Digest
    state[8]  = v2.x; state[9]  = v2.y; state[10] = v2.z; state[11] = v2.w;
    state[12] = v3.x; state[13] = v3.y; state[14] = v3.z; state[15] = v3.w;

    // Coin-specific Keccak padding
    state[16] = KPAD0;  state[17] = KPAD1;  state[18] = KPAD2;
    state[19] = KPAD3;  state[20] = KPAD4;  state[21] = KPAD5;
    state[22] = KPAD6;  state[23] = KPAD7;  state[24] = KPAD8;

    keccak_f800<true>(state);

    if (d_result_hashes != nullptr)
    {
        d_result_hashes[base + 0] = v2.x;
        d_result_hashes[base + 1] = v2.y;
        d_result_hashes[base + 2] = v2.z;
        d_result_hashes[base + 3] = v2.w;
        d_result_hashes[base + 4] = v3.x;
        d_result_hashes[base + 5] = v3.y;
        d_result_hashes[base + 6] = v3.z;
        d_result_hashes[base + 7] = v3.w;

        #pragma unroll
        for (int i = 0; i < 8; i++) d_result_hashes[base + 8 + i] = state[i];
    }

    {
        const uint8_t* hash8   = reinterpret_cast<const uint8_t*>(state);
        const uint8_t* target8 = reinterpret_cast<const uint8_t*>(d_target);

        bool below = false;
        bool equal = true;

        #pragma unroll
        for (int i = 31; i >= 0; i--)
        {
            uint8_t h = hash8[31 - i];
            uint8_t t = target8[i];

            if (equal)
            {
                if (h < t) { below = true;  equal = false; }
                else if (h > t) { below = false; equal = false; }
            }
        }

        if (below || equal)
        {
            if (d_solution_flag) *d_solution_flag = 1;
            uint32_t idx = (uint32_t)atomicAdd((unsigned long long*)&d_solutions[0], 1ULL);
            if (idx < 1024)
            {
                d_solutions[1 + idx * 5 + 0] = nonce;

                const uint64_t* mix64 = reinterpret_cast<const uint64_t*>(&state[8]); // wrong source? no, use digest words separately
                (void)mix64;
                d_solutions[1 + idx * 5 + 1] = ((uint64_t)v2.x) | ((uint64_t)v2.y << 32);
                d_solutions[1 + idx * 5 + 2] = ((uint64_t)v2.z) | ((uint64_t)v2.w << 32);
                d_solutions[1 + idx * 5 + 3] = ((uint64_t)v3.x) | ((uint64_t)v3.y << 32);
                d_solutions[1 + idx * 5 + 4] = ((uint64_t)v3.z) | ((uint64_t)v3.w << 32);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256)
void kawpow_final_kernel_reseed_seed64_digest(
    const uint32_t* __restrict__ d_header,
    uint64_t                     nonce_start,
    uint32_t                     batch_size,
    const uint64_t* __restrict__ d_target,
    uint64_t* __restrict__       d_solutions,
    uint32_t* __restrict__       d_solution_flag,
    uint32_t* __restrict__       d_result_hashes,
    const uint32_t* __restrict__ d_seed_digest)
{
    const uint32_t hash_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hash_id >= batch_size) return;

    const uint64_t nonce = nonce_start + (uint64_t)hash_id;
    const uint32_t digest_base = hash_id * 8u;
    const uint32_t result_base = hash_id * 16u;

    uint32_t state[25];
    kawpow_make_seed_state(d_header, nonce, state);

    const uint2* dig2 = reinterpret_cast<const uint2*>(d_seed_digest + digest_base);
    uint2 v0 = dig2[0];
    uint2 v1 = dig2[1];
    uint2 v2 = dig2[2];
    uint2 v3 = dig2[3];

    state[8]  = v0.x; state[9]  = v0.y; state[10] = v1.x; state[11] = v1.y;
    state[12] = v2.x; state[13] = v2.y; state[14] = v3.x; state[15] = v3.y;

    state[16] = KPAD0;  state[17] = KPAD1;  state[18] = KPAD2;
    state[19] = KPAD3;  state[20] = KPAD4;  state[21] = KPAD5;
    state[22] = KPAD6;  state[23] = KPAD7;  state[24] = KPAD8;

    keccak_f800<true>(state);

    if (d_result_hashes != nullptr)
    {
        d_result_hashes[result_base + 0] = v0.x;
        d_result_hashes[result_base + 1] = v0.y;
        d_result_hashes[result_base + 2] = v1.x;
        d_result_hashes[result_base + 3] = v1.y;
        d_result_hashes[result_base + 4] = v2.x;
        d_result_hashes[result_base + 5] = v2.y;
        d_result_hashes[result_base + 6] = v3.x;
        d_result_hashes[result_base + 7] = v3.y;

        #pragma unroll
        for (int i = 0; i < 8; i++) d_result_hashes[result_base + 8 + i] = state[i];
    }

    {
        const uint8_t* hash8   = reinterpret_cast<const uint8_t*>(state);
        const uint8_t* target8 = reinterpret_cast<const uint8_t*>(d_target);

        bool below = false;
        bool equal = true;

        #pragma unroll
        for (int i = 31; i >= 0; i--)
        {
            uint8_t h = hash8[31 - i];
            uint8_t t = target8[i];

            if (equal)
            {
                if (h < t) { below = true;  equal = false; }
                else if (h > t) { below = false; equal = false; }
            }
        }

        if (below || equal)
        {
            if (d_solution_flag) *d_solution_flag = 1;
            uint32_t idx = (uint32_t)atomicAdd((unsigned long long*)&d_solutions[0], 1ULL);
            if (idx < 1024)
            {
                d_solutions[1 + idx * 5 + 0] = nonce;
                d_solutions[1 + idx * 5 + 1] = ((uint64_t)v0.x) | ((uint64_t)v0.y << 32);
                d_solutions[1 + idx * 5 + 2] = ((uint64_t)v1.x) | ((uint64_t)v1.y << 32);
                d_solutions[1 + idx * 5 + 3] = ((uint64_t)v2.x) | ((uint64_t)v2.y << 32);
                d_solutions[1 + idx * 5 + 4] = ((uint64_t)v3.x) | ((uint64_t)v3.y << 32);
            }
        }
    }
}

#endif // KAWPOW_HIP
