#ifndef KAWPOW_HIP
#define KAWPOW_HIP

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

// ProgPoW body macro — injected by host before RTC.
// #define PROGPOW_BODY(MIX, DG) do { ... } while(0)
// MIX = uint32_t[32] mix array name, DG = uint4 pre-loaded DAG data variable
/* PROGPOW_MACROS */

// ============================================================================
// keccak_f800 — AMD-optimized Keccak-f[800] (25 × 32-bit, 22 rounds)
//
// Same techniques as the f1600: in-place ρ/π chain (1 temp, not 25),
// BFI-pattern χ, and last-round pruning for 8-word output.
// ============================================================================

__device__ __constant__ static const uint32_t keccak_f800_rc[22] = {
    0x00000001u, 0x00008082u, 0x0000808au, 0x80008000u,
    0x0000808bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008au, 0x00000088u, 0x80008009u, 0x8000000au,
    0x8000808bu, 0x0000008bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800au, 0x8000000au,
    0x80008081u, 0x00008080u
};

__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n)
{
#ifdef __HIP_PLATFORM_AMD__
    return __builtin_amdgcn_alignbit(x, x, 32u - n);
#else
    return __funnelshift_l(x, x, n);
#endif
}

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n)
{
#ifdef __HIP_PLATFORM_AMD__
    return __builtin_amdgcn_alignbit(x, x, n);
#else
    return __funnelshift_r(x, x, n);
#endif
}

// χ step: a ^ (~b & c)
// AMD:    v_xor + v_bfi_b32 (2 instructions)
// NVIDIA: lop3.b32 0xD2     (1 instruction)
__device__ __forceinline__ uint32_t chi32(uint32_t a, uint32_t b, uint32_t c)
{
#ifdef __HIP_PLATFORM_AMD__
    // v_bfi_b32 dst, src0, src1, src2 → (src0 & src1) | (~src0 & src2)
    // bfi(b, a, a^c) = (b & a) | (~b & (a^c)) = a ^ (~b & c)
    uint32_t result;
    asm volatile("v_bfi_b32 %0, %1, %2, %3"
                 : "=v"(result) : "v"(b), "v"(a), "v"(a ^ c));
    return result;
#else
    uint32_t result;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xD2;"
                 : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
#endif
}

// 5-way XOR for θ column parity
// AMD:    4 v_xor (compiler handles)
// NVIDIA: 2 lop3.b32 0x96 (3-input XOR truth table)
__device__ __forceinline__ uint32_t xor5_32(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e)
{
#ifndef __HIP_PLATFORM_AMD__
    uint32_t result;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, 0x96;\n\t"
        "lop3.b32 %0, %0, %4, %5, 0x96;"
        : "=r"(result) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e));
    return result;
#else
    return a ^ b ^ c ^ d ^ e;
#endif
}

__device__ void keccak_f800(uint32_t* __restrict__ st)
{
    uint32_t s[25], b[5], u;

    #pragma unroll
    for (int i = 0; i < 25; i++) s[i] = st[i];

    // ---- Rounds 0..20 (21 full rounds) ----
    #pragma unroll
    for (int round = 0; round < 21; round++)
    {
        // θ — column parity
        b[0] = xor5_32(s[0], s[5], s[10], s[15], s[20]);
        b[1] = xor5_32(s[1], s[6], s[11], s[16], s[21]);
        b[2] = xor5_32(s[2], s[7], s[12], s[17], s[22]);
        b[3] = xor5_32(s[3], s[8], s[13], s[18], s[23]);
        b[4] = xor5_32(s[4], s[9], s[14], s[19], s[24]);

        u = b[4] ^ rotl32(b[1], 1);
        s[0]  ^= u; s[5]  ^= u; s[10] ^= u; s[15] ^= u; s[20] ^= u;
        u = b[0] ^ rotl32(b[2], 1);
        s[1]  ^= u; s[6]  ^= u; s[11] ^= u; s[16] ^= u; s[21] ^= u;
        u = b[1] ^ rotl32(b[3], 1);
        s[2]  ^= u; s[7]  ^= u; s[12] ^= u; s[17] ^= u; s[22] ^= u;
        u = b[2] ^ rotl32(b[4], 1);
        s[3]  ^= u; s[8]  ^= u; s[13] ^= u; s[18] ^= u; s[23] ^= u;
        u = b[3] ^ rotl32(b[0], 1);
        s[4]  ^= u; s[9]  ^= u; s[14] ^= u; s[19] ^= u; s[24] ^= u;

        // ρ/π in-place chain (1 temp instead of 25)
        u     = s[1];
        s[1]  = rotl32(s[6],  12);
        s[6]  = rotl32(s[9],  20);
        s[9]  = rotl32(s[22], 29);
        s[22] = rotl32(s[14],  7);
        s[14] = rotl32(s[20], 18);
        s[20] = rotl32(s[2],  30);
        s[2]  = rotl32(s[12], 11);
        s[12] = rotl32(s[13], 25);
        s[13] = rotl32(s[19],  8);
        s[19] = rotl32(s[23], 24);
        s[23] = rotl32(s[15],  9);
        s[15] = rotl32(s[4],  27);
        s[4]  = rotl32(s[24], 14);
        s[24] = rotl32(s[21],  2);
        s[21] = rotl32(s[8],  23);
        s[8]  = rotl32(s[16], 13);
        s[16] = rotl32(s[5],   4);
        s[5]  = rotl32(s[3],  28);
        s[3]  = rotl32(s[18], 21);
        s[18] = rotl32(s[17], 15);
        s[17] = rotl32(s[11], 10);
        s[11] = rotl32(s[7],   6);
        s[7]  = rotl32(s[10],  3);
        s[10] = rotl32(u,      1);

        // χ — platform-optimized (LOP3 on NVIDIA, BFI on AMD)
        #pragma unroll
        for (int y = 0; y < 25; y += 5)
        {
            uint32_t v = s[y], t = s[y + 1];
            s[y]     = chi32(s[y],     s[y + 1], s[y + 2]);
            s[y + 1] = chi32(s[y + 1], s[y + 2], s[y + 3]);
            s[y + 2] = chi32(s[y + 2], s[y + 3], s[y + 4]);
            s[y + 3] = chi32(s[y + 3], s[y + 4], v);
            s[y + 4] = chi32(s[y + 4], v,        t);
        }

        // ι
        s[0] ^= keccak_f800_rc[round];
    }

    // ---- Round 21 (last) — pruned for 8-word output ----
    // θ: full column parity (needs all 25 words)
    b[0] = xor5_32(s[0], s[5], s[10], s[15], s[20]);
    b[1] = xor5_32(s[1], s[6], s[11], s[16], s[21]);
    b[2] = xor5_32(s[2], s[7], s[12], s[17], s[22]);
    b[3] = xor5_32(s[3], s[8], s[13], s[18], s[23]);
    b[4] = xor5_32(s[4], s[9], s[14], s[19], s[24]);

    uint32_t d0 = b[4] ^ rotl32(b[1], 1);
    uint32_t d1 = b[0] ^ rotl32(b[2], 1);
    uint32_t d2 = b[1] ^ rotl32(b[3], 1);
    uint32_t d3 = b[2] ^ rotl32(b[4], 1);
    uint32_t d4 = b[3] ^ rotl32(b[0], 1);

    // Fused θ + ρ/π — only the 10 positions feeding output rows 0–1
    uint32_t t0 = s[0]  ^ d0;
    uint32_t t1 = rotl32(s[6]  ^ d1, 12);
    uint32_t t2 = rotl32(s[12] ^ d2, 11);
    uint32_t t3 = rotl32(s[18] ^ d3, 21);
    uint32_t t4 = rotl32(s[24] ^ d4, 14);
    uint32_t t5 = rotl32(s[3]  ^ d3, 28);
    uint32_t t6 = rotl32(s[9]  ^ d4, 20);
    uint32_t t7 = rotl32(s[10] ^ d0,  3);
    uint32_t t8 = rotl32(s[16] ^ d1, 13);
    uint32_t t9 = rotl32(s[22] ^ d2, 29);

    // χ rows 0–1 only (positions 0–7 are the output)
    st[0] = chi32(t0, t1, t2) ^ keccak_f800_rc[21];  // fused ι
    st[1] = chi32(t1, t2, t3);
    st[2] = chi32(t2, t3, t4);
    st[3] = chi32(t3, t4, t0);
    st[4] = chi32(t4, t0, t1);
    st[5] = chi32(t5, t6, t7);
    st[6] = chi32(t6, t7, t8);
    st[7] = chi32(t7, t8, t9);
}

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
// Static helper macros for software-pipelined DAG loads.
// PROGPOW_ISSUE_DAG computes the next DAG address from mix[0] via Barrett
// reduction and issues a real uint4 load.  Used with PROGPOW_BODY (generated
// macro injected at /* PROGPOW_MACROS */) for N-way hash interleaving.
// ===========================================================================

#ifdef __HIP_PLATFORM_AMD__
#define PROGPOW_BPERMUTE(src, val) __builtin_amdgcn_ds_bpermute((src) << 2, (val))
#else
#define PROGPOW_BPERMUTE(src, val) __shfl_sync(0xFFFFFFFF, (val), (src), 16)
#endif

// Issue a DAG load for iteration LOOP into DG.
// Reads MIX[0] via cross-lane shuffle, applies Barrett modular reduction,
// then loads the 128-bit DAG line.
// Requires: d_dag, lane_id in enclosing scope.
#define PROGPOW_ISSUE_DAG(MIX, DG, LOOP) do { \
    uint32_t _isrc = (threadIdx.x & ~15u) | ((LOOP) & 15u); \
    uint32_t _ia = PROGPOW_BPERMUTE(_isrc, (MIX)[0]); \
    uint32_t _iq = __umulhi(_ia, KAWPOW_BARRETT_M) >> KAWPOW_BARRETT_SHIFT; \
    _ia -= _iq * KAWPOW_DAG_NUM_ITEMS_DIV; \
    if (_ia >= KAWPOW_DAG_NUM_ITEMS_DIV) _ia -= KAWPOW_DAG_NUM_ITEMS_DIV; \
    (DG) = ((const uint4*)d_dag)[_ia * 16u + ((lane_id ^ (LOOP)) & 15u)]; \
} while(0)

// GLC variant: same Barrett logic but uses asm volatile GLC load for cache bypass.
__device__ __forceinline__ uint4 load_dag_glc(const void* p)
{
    uint4 r;
#if __gfx11__ || __gfx1100__ || __gfx1101__ || __gfx1102__
    asm volatile("global_load_b128 %0, %1, off glc" : "=v"(r) : "v"(p));
#else
    asm volatile("global_load_dwordx4 %0, %1, off glc" : "=v"(r) : "v"(p));
#endif
    return r;
}

#define PROGPOW_ISSUE_DAG_GLC(MIX, DG, LOOP) do { \
    uint32_t _isrc = (threadIdx.x & ~15u) | ((LOOP) & 15u); \
    uint32_t _ia = PROGPOW_BPERMUTE(_isrc, (MIX)[0]); \
    uint32_t _iq = __umulhi(_ia, KAWPOW_BARRETT_M) >> KAWPOW_BARRETT_SHIFT; \
    _ia -= _iq * KAWPOW_DAG_NUM_ITEMS_DIV; \
    if (_ia >= KAWPOW_DAG_NUM_ITEMS_DIV) _ia -= KAWPOW_DAG_NUM_ITEMS_DIV; \
    (DG) = load_dag_glc((const uint4*)d_dag + _ia * 16u + ((lane_id ^ (LOOP)) & 15u)); \
} while(0)

// ===========================================================================
// Shared helpers: L1 cache cooperative fill + main loop + mix reduction
// ===========================================================================

__device__ __forceinline__ void cooperative_l1_fill(
    uint32_t* __restrict__ l1_cache,
    const uint32_t* __restrict__ d_l1_cache)
{
    const uint32_t tid = threadIdx.x;
    const uint4* __restrict__ src4 = (const uint4*)d_l1_cache;
    uint4* dst4 = (uint4*)l1_cache;
    constexpr uint32_t CACHE_VEC4 = PROGPOW_CACHE_WORDS / 4;
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
    uint4 _dg0, _dg1;
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
    uint4 _da0, _da1;  // ping-pong buffers for hash 0
    uint4 _db0, _db1;  // ping-pong buffers for hash 1
    PROGPOW_ISSUE_DAG(m0, _da0, 0u);
    PROGPOW_ISSUE_DAG(m1, _db0, 0u);

    #pragma unroll 1
    for (uint32_t loop = 0; loop < KAWPOW_MAIN_LOOPS - 2; loop += 2) {
        PROGPOW_BODY_PIPE(m0, _da0, _da1, loop + 1);
        PROGPOW_BODY_PIPE(m1, _db0, _db1, loop + 1);
        PROGPOW_BODY_PIPE(m0, _da1, _da0, loop + 2);
        PROGPOW_BODY_PIPE(m1, _db1, _db0, loop + 2);
    }
    PROGPOW_BODY_PIPE(m0, _da0, _da1, KAWPOW_MAIN_LOOPS - 1);
    PROGPOW_BODY_PIPE(m1, _db0, _db1, KAWPOW_MAIN_LOOPS - 1);
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
    uint4 _da0, _da1;  // ping-pong for hash 0
    uint4 _db0, _db1;  // ping-pong for hash 1
    uint4 _dc0, _dc1;  // ping-pong for hash 2
    uint4 _dd0, _dd1;  // ping-pong for hash 3
    PROGPOW_ISSUE_DAG(m0, _da0, 0u);
    PROGPOW_ISSUE_DAG(m1, _db0, 0u);
    PROGPOW_ISSUE_DAG(m2, _dc0, 0u);
    PROGPOW_ISSUE_DAG(m3, _dd0, 0u);

    #pragma unroll 1
    for (uint32_t loop = 0; loop < KAWPOW_MAIN_LOOPS - 2; loop += 2) {
        PROGPOW_BODY_PIPE(m0, _da0, _da1, loop + 1);
        PROGPOW_BODY_PIPE(m1, _db0, _db1, loop + 1);
        PROGPOW_BODY_PIPE(m2, _dc0, _dc1, loop + 1);
        PROGPOW_BODY_PIPE(m3, _dd0, _dd1, loop + 1);
        PROGPOW_BODY_PIPE(m0, _da1, _da0, loop + 2);
        PROGPOW_BODY_PIPE(m1, _db1, _db0, loop + 2);
        PROGPOW_BODY_PIPE(m2, _dc1, _dc0, loop + 2);
        PROGPOW_BODY_PIPE(m3, _dd1, _dd0, loop + 2);
    }
    PROGPOW_BODY_PIPE(m0, _da0, _da1, KAWPOW_MAIN_LOOPS - 1);
    PROGPOW_BODY_PIPE(m1, _db0, _db1, KAWPOW_MAIN_LOOPS - 1);
    PROGPOW_BODY_PIPE(m2, _dc0, _dc1, KAWPOW_MAIN_LOOPS - 1);
    PROGPOW_BODY_PIPE(m3, _dd0, _dd1, KAWPOW_MAIN_LOOPS - 1);
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

// ===========================================================================
// Strategy 0 — Monolithic kernel (seed keccak + progpow + final keccak)
// ===========================================================================
extern "C" __global__
void kawpow_hash_kernel(
    const uint32_t* __restrict__ d_header,        // 8 × uint32 = 32 bytes
    const uint32_t* __restrict__ d_dag,            // full DAG (uint32 array)
    uint64_t                     nonce_start,
    uint32_t                     batch_size,
    const uint64_t* __restrict__ d_target,         // 4 × uint64 = 32 bytes (LE)
    uint64_t* __restrict__       d_solutions,      // [0]=count(u64), [1+i*5]=nonce(u64), [+1..4]=mix_hash(4×u64)
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

    // Sandwich keccak + fill_mix between L1 fill and barrier to hide LDS write latency.
    // No early return — all threads must reach __syncthreads().
    const bool valid = (hash_id < batch_size);

    uint32_t state[25];
    uint32_t mix[PROGPOW_REGS];
    uint64_t seed = 0;

    uint32_t* lds_seed_state = &lds_raw[PROGPOW_CACHE_WORDS];

    if (valid) {
        // ---- Phase 1: Seed Keccak ----
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

        keccak_f800(state);

        seed = (uint64_t)state[0] | ((uint64_t)state[1] << 32);

        // Save state[0..7] to LDS for phase 5
        {
            uint32_t hwb = threadIdx.x >> PROGPOW_LANES_LOG2;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                lds_seed_state[hwb * 8u + i] = state[i];
        }

        fill_mix(seed, lane_id, mix);
    }

    __syncthreads();  // L1 fill + seed state writes all visible

    if (!valid) return;

    // ---- Phase 2–4: main loop + reduction ----
    progpow_main_loop(mix, l1_cache, d_dag, lane_id);

    uint32_t digest[8];
    progpow_reduce(mix, digest);

    // ---- Phase 5: Final Keccak ----
    {
        uint32_t hwb = threadIdx.x >> PROGPOW_LANES_LOG2;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            state[i] = lds_seed_state[hwb * 8u + i];
    }

    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[8 + i] = digest[i];

    state[16] = KPAD0;  state[17] = KPAD1;  state[18] = KPAD2;
    state[19] = KPAD3;  state[20] = KPAD4;  state[21] = KPAD5;
    state[22] = KPAD6;  state[23] = KPAD7;  state[24] = KPAD8;

    keccak_f800(state);

    // ---- Target comparison (lane 0 only) ----
    if (d_result_hashes != nullptr && lane_id == 0)
    {
        uint32_t base = hash_id * 16u;
        #pragma unroll
        for (int i = 0; i < 8; i++) d_result_hashes[base + i] = digest[i];
        #pragma unroll
        for (int i = 0; i < 8; i++) d_result_hashes[base + 8 + i] = state[i];
    }

    if (lane_id == 0)
    {
        const uint64_t* hash64 = (const uint64_t*)state;

        bool below = false;
        bool equal = true;

        #pragma unroll
        for (int i = 3; i >= 0; i--)
        {
            uint64_t h = hash64[i];
            uint64_t t = d_target[i];
            if (equal)
            {
                if (h < t) { below = true; equal = false; }
                else if (h > t) { below = false; equal = false; }
            }
        }

        if (below)
        {
            uint32_t idx = (uint32_t)atomicAdd((unsigned long long*)&d_solutions[0], 1ULL);
            if (idx < 1024)
            {
                d_solutions[1 + idx * 5 + 0] = nonce;
                // mix_hash = digest[0..7] as 4 × uint64
                const uint64_t* digest64 = (const uint64_t*)digest;
                d_solutions[1 + idx * 5 + 1] = digest64[0];
                d_solutions[1 + idx * 5 + 2] = digest64[1];
                d_solutions[1 + idx * 5 + 3] = digest64[2];
                d_solutions[1 + idx * 5 + 4] = digest64[3];
            }
        }
    }
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
extern "C" __global__ __launch_bounds__(128, 8)
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

    keccak_f800(state);

    // Write seed state to intermediate buffer
    uint32_t base = hash_id * 16u;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        d_intermediate[base + i] = state[i];
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
        #pragma unroll
        for (int i = 0; i < 8; i++)
            d_intermediate[base + 8 + i] = digest[i];
    }
}

// ---------------------------------------------------------------------------
// Split kernel B-GLC: ProgPoW main loop — GLC DAG loads (LDS L1 cache)
// Uses non-volatile inline asm for globally-coherent DAG loads.
// The program body MUST be inlined directly in the loop — calling a separate
// function breaks the compiler's def-use chain for the asm load output.
// ---------------------------------------------------------------------------
extern "C" __global__
void kawpow_progpow_kernel_glc(
    const uint32_t* __restrict__ d_dag,
    uint32_t                     batch_size,
    const uint32_t* __restrict__ d_l1_cache,
    uint32_t* __restrict__       d_intermediate)
{
    const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id   = global_id & 0xF;
    const uint32_t hash_id   = global_id >> PROGPOW_LANES_LOG2;

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

    const uint32_t lane_base = (threadIdx.x & ~15u);

    __syncthreads();

    if (!valid) return;

    #pragma unroll 1
    for (uint32_t loop = 0; loop < KAWPOW_MAIN_LOOPS; loop++) {
        uint32_t src = (lane_base | (loop & 15u));
        uint32_t dag_addr = __builtin_amdgcn_ds_bpermute(src << 2, mix[0]);
        uint32_t q = __umulhi(dag_addr, KAWPOW_BARRETT_M) >> KAWPOW_BARRETT_SHIFT;
        dag_addr  -= q * KAWPOW_DAG_NUM_ITEMS_DIV;
        if (dag_addr >= KAWPOW_DAG_NUM_ITEMS_DIV) dag_addr -= KAWPOW_DAG_NUM_ITEMS_DIV;

/* PROGPOW_PROGRAM_GLC */
    }

    uint32_t digest[8];
    progpow_reduce(mix, digest);

    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            d_intermediate[base + 8 + i] = digest[i];
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
        #pragma unroll
        for (int i = 0; i < 8; i++)
            d_intermediate[base + 8 + i] = digest[i];
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
        {
            uint32_t base = hash_id_0 * 16u;
            uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);
            fill_mix(seed, lane_id, m0);
        }

        have_h1 = (hash_id_0 + 1 < batch_size);
        if (have_h1) {
            uint32_t base = (hash_id_0 + 1) * 16u;
            uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);
            fill_mix(seed, lane_id, m1);
        }
    }

    __syncthreads();

    if (!valid) return;

    if (have_h1) {
        progpow_main_loop_2way(m0, m1, l1_cache, d_dag, lane_id);
    } else {
        progpow_main_loop(m0, l1_cache, d_dag, lane_id);
    }

    // Reduce and write hash 0
    {
        uint32_t digest[8];
        progpow_reduce(m0, digest);
        if (lane_id == 0) {
            uint32_t base = hash_id_0 * 16u;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                d_intermediate[base + 8 + i] = digest[i];
        }
    }

    // Reduce and write hash 1
    if (have_h1) {
        uint32_t digest[8];
        progpow_reduce(m1, digest);
        if (lane_id == 0) {
            uint32_t base = (hash_id_0 + 1) * 16u;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                d_intermediate[base + 8 + i] = digest[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Split kernel B-4way: ProgPoW main loop — 4-way hash interleaving
// Each wavefront group processes 4 hashes for maximum DAG latency hiding.
// ---------------------------------------------------------------------------
extern "C" __global__
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
        uint32_t digest[8];
        progpow_reduce(m0, digest);
        if (lane_id == 0) {
            uint32_t base = hash_id_0 * 16u;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                d_intermediate[base + 8 + i] = digest[i];
        }
    }
    if (num_hashes >= 2) {
        uint32_t digest[8];
        progpow_reduce(m1, digest);
        if (lane_id == 0) {
            uint32_t base = (hash_id_0 + 1) * 16u;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                d_intermediate[base + 8 + i] = digest[i];
        }
    }
    if (num_hashes >= 3) {
        uint32_t digest[8];
        progpow_reduce(m2, digest);
        if (lane_id == 0) {
            uint32_t base = (hash_id_0 + 2) * 16u;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                d_intermediate[base + 8 + i] = digest[i];
        }
    }
    if (num_hashes >= 4) {
        uint32_t digest[8];
        progpow_reduce(m3, digest);
        if (lane_id == 0) {
            uint32_t base = (hash_id_0 + 3) * 16u;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                d_intermediate[base + 8 + i] = digest[i];
        }
    }
}

// ---------------------------------------------------------------------------
// GLC pipelined main loops — use PROGPOW_ISSUE_DAG_GLC + PROGPOW_BODY_PIPE_GLC
// ---------------------------------------------------------------------------

// 1-way GLC ping-pong (fallback for odd trailing hash in 2-way)
__device__ __forceinline__ void progpow_main_loop_glc(
    uint32_t* mix,
    const uint32_t* __restrict__ l1_cache,
    const uint32_t* __restrict__ d_dag,
    uint32_t lane_id)
{
    uint4 _dg0, _dg1;
    PROGPOW_ISSUE_DAG_GLC(mix, _dg0, 0u);

    #pragma unroll 1
    for (uint32_t loop = 0; loop < KAWPOW_MAIN_LOOPS - 2; loop += 2) {
        PROGPOW_BODY_PIPE_GLC(mix, _dg0, _dg1, loop + 1);
        PROGPOW_BODY_PIPE_GLC(mix, _dg1, _dg0, loop + 2);
    }
    PROGPOW_BODY_PIPE_GLC(mix, _dg0, _dg1, KAWPOW_MAIN_LOOPS - 1);
    PROGPOW_BODY(mix, _dg1);
}

// 2-way GLC ping-pong
__device__ __forceinline__ void progpow_main_loop_2way_glc(
    uint32_t* m0, uint32_t* m1,
    const uint32_t* __restrict__ l1_cache,
    const uint32_t* __restrict__ d_dag,
    uint32_t lane_id)
{
    uint4 _da0, _da1;  // ping-pong for hash 0
    uint4 _db0, _db1;  // ping-pong for hash 1
    PROGPOW_ISSUE_DAG_GLC(m0, _da0, 0u);
    PROGPOW_ISSUE_DAG_GLC(m1, _db0, 0u);

    #pragma unroll 1
    for (uint32_t loop = 0; loop < KAWPOW_MAIN_LOOPS - 2; loop += 2) {
        PROGPOW_BODY_PIPE_GLC(m0, _da0, _da1, loop + 1);
        PROGPOW_BODY_PIPE_GLC(m1, _db0, _db1, loop + 1);
        PROGPOW_BODY_PIPE_GLC(m0, _da1, _da0, loop + 2);
        PROGPOW_BODY_PIPE_GLC(m1, _db1, _db0, loop + 2);
    }
    PROGPOW_BODY_PIPE_GLC(m0, _da0, _da1, KAWPOW_MAIN_LOOPS - 1);
    PROGPOW_BODY_PIPE_GLC(m1, _db0, _db1, KAWPOW_MAIN_LOOPS - 1);
    PROGPOW_BODY(m0, _da1);
    PROGPOW_BODY(m1, _db1);
}

// ---------------------------------------------------------------------------
// Split kernel B-2way-GLC: 2-way hash interleaving with GLC DAG loads
// ---------------------------------------------------------------------------
extern "C" __global__
void kawpow_progpow_kernel_2way_glc(
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
        {
            uint32_t base = hash_id_0 * 16u;
            uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);
            fill_mix(seed, lane_id, m0);
        }

        have_h1 = (hash_id_0 + 1 < batch_size);
        if (have_h1) {
            uint32_t base = (hash_id_0 + 1) * 16u;
            uint64_t seed = (uint64_t)d_intermediate[base] | ((uint64_t)d_intermediate[base + 1] << 32);
            fill_mix(seed, lane_id, m1);
        }
    }

    __syncthreads();

    if (!valid) return;

    if (have_h1) {
        progpow_main_loop_2way_glc(m0, m1, l1_cache, d_dag, lane_id);
    } else {
        progpow_main_loop_glc(m0, l1_cache, d_dag, lane_id);
    }

    // Reduce and write hash 0
    {
        uint32_t digest[8];
        progpow_reduce(m0, digest);
        if (lane_id == 0) {
            uint32_t base = hash_id_0 * 16u;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                d_intermediate[base + 8 + i] = digest[i];
        }
    }

    // Reduce and write hash 1
    if (have_h1) {
        uint32_t digest[8];
        progpow_reduce(m1, digest);
        if (lane_id == 0) {
            uint32_t base = (hash_id_0 + 1) * 16u;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                d_intermediate[base + 8 + i] = digest[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Split kernel C: Final Keccak + target check  (1 thread per hash, no LDS)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(128, 8)
void kawpow_final_kernel(
    uint64_t                     nonce_start,
    uint32_t                     batch_size,
    const uint64_t* __restrict__ d_target,
    uint64_t* __restrict__       d_solutions,
    uint32_t* __restrict__       d_result_hashes,
    const uint32_t* __restrict__ d_intermediate)
{
    const uint32_t hash_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hash_id >= batch_size) return;

    const uint64_t nonce = nonce_start + (uint64_t)hash_id;
    uint32_t base = hash_id * 16u;

    uint32_t state[25];

    // Absorb seed state (from seed kernel)
    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = d_intermediate[base + i];

    // Absorb digest (from progpow kernel)
    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[8 + i] = d_intermediate[base + 8 + i];

    // Coin-specific Keccak padding
    state[16] = KPAD0;  state[17] = KPAD1;  state[18] = KPAD2;
    state[19] = KPAD3;  state[20] = KPAD4;  state[21] = KPAD5;
    state[22] = KPAD6;  state[23] = KPAD7;  state[24] = KPAD8;

    keccak_f800(state);

    // Debug output
    if (d_result_hashes != nullptr)
    {
        #pragma unroll
        for (int i = 0; i < 8; i++) d_result_hashes[base + i] = d_intermediate[base + 8 + i];
        #pragma unroll
        for (int i = 0; i < 8; i++) d_result_hashes[base + 8 + i] = state[i];
    }

    // Target comparison
    {
        const uint64_t* hash64 = (const uint64_t*)state;

        bool below = false;
        bool equal = true;

        #pragma unroll
        for (int i = 3; i >= 0; i--)
        {
            uint64_t h = hash64[i];
            uint64_t t = d_target[i];
            if (equal)
            {
                if (h < t) { below = true; equal = false; }
                else if (h > t) { below = false; equal = false; }
            }
        }

        if (below)
        {
            uint32_t idx = (uint32_t)atomicAdd((unsigned long long*)&d_solutions[0], 1ULL);
            if (idx < 1024)
            {
                d_solutions[1 + idx * 5 + 0] = nonce;
                // mix_hash from intermediate buffer [base+8..base+15]
                const uint64_t* mix64 = (const uint64_t*)&d_intermediate[base + 8];
                d_solutions[1 + idx * 5 + 1] = mix64[0];
                d_solutions[1 + idx * 5 + 2] = mix64[1];
                d_solutions[1 + idx * 5 + 3] = mix64[2];
                d_solutions[1 + idx * 5 + 4] = mix64[3];
            }
        }
    }
}

#endif // KAWPOW_HIP