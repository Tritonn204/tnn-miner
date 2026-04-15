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

// Coin-specific Keccak padding — injected by host before RTC compilation.
// Phase 1 (seed):  state[10..24] = kawpow_coin_padding[0..14]
// Phase 5 (final): state[16..24] = kawpow_coin_padding[0..8]
/* KAWPOW_COIN_PADDING */

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
    return (x << n) | (x >> (32u - n));
}

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32u - n));
}

// BFI pattern: (x & ~c) | (a & c)  →  v_bfi_b32 on AMD
__device__ __forceinline__ uint32_t bfi32(uint32_t x, uint32_t a, uint32_t c)
{
    return (x & ~c) | (a & c);
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
        // θ
        b[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        b[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        b[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        b[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        b[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

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

        // χ (BFI pattern — 2 instructions per element on AMD)
        #pragma unroll
        for (int y = 0; y < 25; y += 5)
        {
            uint32_t v = s[y], t = s[y + 1];
            s[y]     = bfi32(s[y]     ^ s[y + 2], s[y],     s[y + 1]);
            s[y + 1] = bfi32(s[y + 1] ^ s[y + 3], s[y + 1], s[y + 2]);
            s[y + 2] = bfi32(s[y + 2] ^ s[y + 4], s[y + 2], s[y + 3]);
            s[y + 3] = bfi32(s[y + 3] ^ v,        s[y + 3], s[y + 4]);
            s[y + 4] = bfi32(s[y + 4] ^ t,        s[y + 4], v);
        }

        // ι
        s[0] ^= keccak_f800_rc[round];
    }

    // ---- Round 21 (last) — pruned for 8-word output ----
    // θ: full column parity (needs all 25 words)
    b[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
    b[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
    b[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
    b[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
    b[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

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
    st[0] = (t0 ^ (~t1 & t2)) ^ keccak_f800_rc[21];  // fused ι
    st[1] = t1 ^ (~t2 & t3);
    st[2] = t2 ^ (~t3 & t4);
    st[3] = t3 ^ (~t4 & t0);
    st[4] = t4 ^ (~t0 & t1);
    st[5] = t5 ^ (~t6 & t7);
    st[6] = t6 ^ (~t7 & t8);
    st[7] = t7 ^ (~t8 & t9);
}

__device__ __forceinline__ uint32_t fnv1a(uint32_t h, uint32_t d)
{
    return (h ^ d) * FNV_PRIME;
}

__device__ __forceinline__ uint32_t progpow_merge(uint32_t a, uint32_t b, uint32_t sel)
{
    const uint32_t x = ((sel >> 16) % 31) + 1;
    switch (sel % 4)
    {
        case 0:  return (a * 33) + b;
        case 1:  return (a ^ b) * 33;
        case 2:  return rotl32(a, x) ^ b;
        default: return rotr32(a, x) ^ b;
    }
}

__device__ __forceinline__ uint32_t progpow_math(uint32_t a, uint32_t b, uint32_t sel)
{
    switch (sel % 11)
    {
        case 0:  return a + b;
        case 1:  return a * b;
        case 2:  return __umulhi(a, b);
        case 3:  return min(a, b);
        case 4:  return rotl32(a, b % 32);
        case 5:  return rotr32(a, b % 32);
        case 6:  return a & b;
        case 7:  return a | b;
        case 8:  return a ^ b;
        case 9:  return __clz(a) + __clz(b);
        default: return __popc(a) + __popc(b);
    }
}

struct kiss99_state
{
    uint32_t z, w, jsr, jcong;
};

__device__ __forceinline__ uint32_t kiss99(kiss99_state& st)
{
    st.z     = 36969u * (st.z & 0xffffu) + (st.z >> 16);
    st.w     = 18000u * (st.w & 0xffffu) + (st.w >> 16);
    uint32_t mwc = (st.z << 16) + st.w;
    st.jsr  ^= (st.jsr << 17);
    st.jsr  ^= (st.jsr >> 13);
    st.jsr  ^= (st.jsr << 5);
    st.jcong = 69069u * st.jcong + 1234567u;
    return (mwc ^ st.jcong) + st.jsr;
}

__device__ void fill_mix(uint64_t seed, uint32_t lane_id, uint32_t* mix)
{
    kiss99_state rng;
    rng.z     = fnv1a(FNV_OFFSET_BASIS, (uint32_t)seed);
    rng.w     = fnv1a(rng.z,            (uint32_t)(seed >> 32));
    rng.jsr   = fnv1a(rng.w,            lane_id);
    rng.jcong = fnv1a(rng.jsr,          lane_id);

    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix[i] = kiss99(rng);
}

__device__ __forceinline__ uint32_t shfl_lane16(uint32_t val, int src_lane)
{
#ifdef __HIP_PLATFORM_AMD__
    // __shfl with width=16: reads from src_lane within the 16-lane sub-group
    return __shfl(val, src_lane, 16);
#else
    // NVIDIA: __shfl_sync with full mask, width 16
    return __shfl_sync(0xFFFFFFFF, val, src_lane, 16);
#endif
}

extern "C" __global__
void kawpow_hash_kernel(
    const uint32_t* __restrict__ d_header,        // 8 × uint32 = 32 bytes
    const uint32_t* __restrict__ d_dag,            // full DAG (uint32 array)
    uint64_t                     nonce_start,
    uint32_t                     batch_size,
    const uint64_t* __restrict__ d_target,         // 4 × uint64 = 32 bytes (LE)
    uint32_t* __restrict__       d_solutions,      // [0]=count, [1..]=nonces
    uint32_t                     block_number,
    uint32_t                     dag_num_items_div, // full_dataset_num_items / 2
    uint32_t* __restrict__       d_result_hashes)  // nullable: [hash_id*16+0..7]=mix, [+8..15]=final
{
    // ---- Thread → lane / hash mapping ----
    const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id   = global_id & 0xF;           // 0..15
    const uint32_t hash_id   = global_id >> PROGPOW_LANES_LOG2;

    const uint64_t nonce = nonce_start + (uint64_t)hash_id;

    // ---- Load 16 KB L1 cache into LDS ----
    // All threads participate BEFORE early exit (syncthreads requires it).
    __shared__ uint32_t l1_cache[PROGPOW_CACHE_WORDS]; // 4096 words = 16 KB

    const uint32_t threads_in_block = blockDim.x;
    static_assert(PROGPOW_CACHE_WORDS % 4 == 0);
    for (uint32_t i = threadIdx.x; i < PROGPOW_CACHE_WORDS / 4; i += threads_in_block)
        ((uint4*)l1_cache)[i] = ((const uint4*)d_dag)[i];
        
    __syncthreads();

    if (hash_id >= batch_size) return;

    // ================================================================
    // Phase 1:  Keccak-f800 seed
    // ================================================================

    uint32_t state[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) state[i] = 0;

    // Absorb header hash (8 words)
    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = d_header[i];

    // Absorb nonce (2 words, little-endian)
    state[8]  = (uint32_t)(nonce);
    state[9]  = (uint32_t)(nonce >> 32);

    // Coin-specific Keccak padding (15 words, injected at RTC compile time)
    #pragma unroll
    for (int i = 10; i < 25; i++)
        state[i] = kawpow_coin_padding[i - 10];

    keccak_f800(state);

    // Seed for fill_mix
    const uint64_t seed = (uint64_t)state[0] | ((uint64_t)state[1] << 32);

    // Save state[0..7] for final hash
    uint32_t seed_state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        seed_state[i] = state[i];

    // ================================================================
    // Phase 2:  Initialize mix registers
    // ================================================================

    uint32_t mix[PROGPOW_REGS];
    fill_mix(seed, lane_id, mix);

    // ================================================================
    // Phase 3:  Main loop (×64)
    // ================================================================

    for (uint32_t loop = 0; loop < KAWPOW_MAIN_LOOPS; loop++)
    {
        // ---- DAG address (broadcast from the designated lane) ----
        uint32_t dag_addr = shfl_lane16(mix[0], loop % PROGPOW_LANES);
        dag_addr %= dag_num_items_div;

        /* PROGPOW_PROGRAM */

    } // end main loop

    // ================================================================
    // Phase 4:  Mix reduction
    // ================================================================

    uint32_t lane_result = FNV_OFFSET_BASIS;
    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        lane_result = fnv1a(lane_result, mix[i]);

    uint32_t digest[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        digest[i] = FNV_OFFSET_BASIS;
        digest[i] = fnv1a(digest[i], shfl_lane16(lane_result, i));
        digest[i] = fnv1a(digest[i], shfl_lane16(lane_result, i + 8));
    }

    // ================================================================
    // Phase 5:  Final Keccak-f800
    // (All lanes compute — divergence-free; only lane 0 checks solution)
    // ================================================================

    #pragma unroll
    for (int i = 0; i < 25; i++) state[i] = 0;

    // Absorb seed state (8 words)
    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = seed_state[i];

    // Absorb mix digest (8 words)
    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[8 + i] = digest[i];

    // Coin-specific Keccak padding (first 9 words of padding array)
    #pragma unroll
    for (int i = 16; i < 25; i++)
        state[i] = kawpow_coin_padding[i - 16];

    keccak_f800(state);

    // ================================================================
    // Target comparison (lane 0 only)
    // ================================================================

    // Debug output: write full mix_hash + final_hash per hash (test/bench mode)
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
            uint32_t idx = atomicAdd(&d_solutions[0], 1u);
            if (idx < 15) // max solutions buffer
            {
                d_solutions[1 + idx * 5 + 0] = (uint32_t)(nonce);
                d_solutions[1 + idx * 5 + 1] = (uint32_t)(nonce >> 32);
                // Also store mix_hash (digest) for stratum submission
                d_solutions[1 + idx * 5 + 2] = digest[0];
                d_solutions[1 + idx * 5 + 3] = digest[4];
                d_solutions[1 + idx * 5 + 4] = hash_id;
            }
        }
    }
}

#endif // KAWPOW_HIP