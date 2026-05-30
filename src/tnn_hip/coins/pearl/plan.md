Fused GPU Kernel Pipeline for Pearl PoUW — gfx1100 (AMD RDNA3)
0. Summary of Current Architecture
Data flow:
CPU: header(76B)+config(52B) ──blake3──▶ job_key(32B)
CPU: job_key || hash_b ──blake3──▶ b_noise_seed(32B)
CPU: b_noise_seed || hash_a ──blake3──▶ a_noise_seed(32B)
CPU: a_noise_seed + seed_label ──blake3──▶ EAL(h×R dense), EAR(k×R sparse)  [BLAKE3 per row]
CPU: b_noise_seed + seed_label ──blake3──▶ EBR(w×R dense), EBL(k×R sparse)
CPU: noise_a = EAL × EAR^T  (h×k),  noise_b = EBR × EBL^T  (w×k)
CPU: ApEA = clip_i8(s_a + noise_a), BpEB = clip_i8(s_b + noise_b)
       ──oroMemcpy──▶ GPU
GPU: C = rocWMMA_GEMM(ApEA, BpEB)       [pearl_gemm_simple.hip]
       ──oroMemcpy──▶ CPU
CPU: jackpot = per-rank-block XOR of C
CPU: jackpot_msg ──blake3_keyed──▶ jackpot_hash
GEMM tile parameters (gfx11, rocwmma_trial_params):
Parameter
WARP_SIZE
ROCWMMA_M/N/K
BLOCKS_M/N
WARP_TILE_M
WARP_TILE_N
WARP_TILE_K
TBLOCK_X/Y
WARPS_M/N
MACRO_TILE_M
MACRO_TILE_N
MACRO_TILE_K
rocWMMA fragment types:
- MmaFragA: fragment<matrix_a, 64, 32, 16, int8_t, row_major>
- MmaFragB: fragment<matrix_b, 64, 32, 16, int8_t, col_major>
- MmaFragAcc: fragment<accumulator, 64, 32, 16, int32_t>
- GRFragA: fragment<matrix_a, 128, 64, 16, int8_t, row_major, CoopScheduler> — macro-tile global read
- LWFragA/B: same data in col_major LDS layout
- LRFragA/B: warp-local LDS reads, also col_major
LDS usage (current): Double-buffered: 2 × (LDS_HEIGHT_A + LDS_HEIGHT_B) × MACRO_TILE_K = 2 × (128+64) × 16 = 6144 int8 = 6KB
Key noise identity: noise_a[i][l] = EAL[i][ear_first[l]] - EAL[i][ear_second[l]]. Only 2 lookups per (i,l), not a full dot product.
Phase 1: Fused Noise Generation → GEMM
1.1 Kernel Launch Config
gridDim  = (ceil_div(M, MACRO_TILE_M), ceil_div(N, MACRO_TILE_N), 1)
         = (ceil_div(M, 128), ceil_div(N, 64), 1)

blockDim = (TBLOCK_X, TBLOCK_Y, 1) = (64, 2, 1)  // 128 threads

sharedMem = LDS_EAL + LDS_EBR + LDS_GEMM_DOUBLE_BUFFER
          = (128 × 32) + (64 × 32) + 2 × (128+64) × 16  bytes
          = 4096     + 2048      + 6144
          = 12288 bytes ≈ 12 KB    (gfx1100 has 64KB LDS → 19% utilization)
1.2 Register vs LDS vs Global Memory Map
Per Thread (Registers ~ 128-160 VGPRs, target 128)
Data
mmaFragAcc (4×2 WMMA blocks)
lrFragA, lrFragB (double-buffered)
grFragA, grFragB (prefetch)
ear_first/ear_second for MACRO_TILE_K=16
ebl_first/ebl_second for MACRO_TILE_K=16
BLAKE3 temp (reused)
Total VGPR budget ~ 256 regs peak (with rocwmma fragments). Need to be careful — rocwMMA on gfx1100 typically uses 96-128 VGPRs per warp for GEMM alone. Adding BLAKE3 on top pushes this. Mitigation: Move BLAKE3 to a separate prep phase (LDS fill, not simultaneous with MMA), allowing register reuse.
Per Block (LDS — 12KB total)
Data
EAL dense (128×32 int8)
EBR dense (64×32 int8)
GEMM double-buffer A tile (128×16)
GEMM double-buffer B tile (64×16)
GEMM double-buffer A' (128×16)
GEMM double-buffer B' (64×16)
BLAKE3 thread scratch
Global Memory
Data
ear_first[k], ear_second[k]
ebl_first[k], ebl_second[k]
s_a_seed[8], s_b_seed[8]
d[M×N] int32
1.3 Key Pseudocode
// Phase 1: Fused noise-gen + GEMM kernel
// grid: (ceil_div(M,128), ceil_div(N,64)), block: (64,2), shmem: 12KB

extern "C" __global__ __launch_bounds__(128)
void pearl_fused_noise_gemm(
    uint32_t m, n, k, R,
    uint8_t const* key_a,      // 32B BLAKE3 key for A-side
    uint8_t const* key_b,      // 32B BLAKE3 key for B-side
    uint8_t const* seed_a,     // 32B seed label for A
    uint8_t const* seed_b,     // 32B seed label for B
    uint32_t const* s_a_seed,  // 8 u32 for deterministic signal A
    uint32_t const* s_b_seed,  // 8 u32 for deterministic signal B
    int32_t const* ear_first,  // k values: sparse col idx 1 per K row
    int32_t const* ear_second, // k values: sparse col idx 2 per K row
    int32_t const* ebl_first,  // k values
    int32_t const* ebl_second, // k values
    int32_t*       d,          // output C: M×N int32 row-major
    uint32_t       ldd)
{
    using namespace rocwmma_trial_params;

    // ---- LDS layout ----
    HIP_DYNAMIC_SHARED(uint8_t*, lds);
    uint8_t* eal_lds = lds;                           // [0..4095]
    uint8_t* ebr_lds = lds + MACRO_TILE_M * R;         // [4096..6143]
    uint8_t* gemm_lds = lds + MACRO_TILE_M * R + MACRO_TILE_N * R; // [6144..12287]

    // ---- Compute block's M and N ranges ----
    uint32_t m_base = blockIdx.x * MACRO_TILE_M;
    uint32_t n_base = blockIdx.y * MACRO_TILE_N;

    // ============== STEP A: Generate EAL dense noise into LDS ==============
    // 128 threads, each generates 32 int8 values (one BLAKE3 output)
    // Produces 128×32 = MACRO_TILE_M × R values
    {
        uint32_t tid = threadIdx.x + threadIdx.y * TBLOCK_X; // 0..127
        uint32_t thread_coord = m_base * R / 32 + tid + 1;   // unique per (block, thread)

        uint8_t block_buf[64] = {};
        // coord_word=0 for dense (data[0..3] = thread_coord)
        block_buf[0] = thread_coord & 0xff;
        block_buf[1] = (thread_coord >> 8) & 0xff;
        block_buf[2] = (thread_coord >> 16) & 0xff;
        block_buf[3] = (thread_coord >> 24) & 0xff;
        for (int i = 0; i < 32; ++i) block_buf[32+i] = seed_a[i];

        uint32_t cv[8];
        for (int i = 0; i < 8; ++i) cv[i] = blake3_load32(key_a + i*4);
        uint8_t flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
        blake3_compress_in_place(cv, block_buf, 64, 0, flags);

        // Clamp 32 bytes → 32 int8 ∈ [-32, 32)
        int row_in_tile = tid / (R / 32);  // tid/4 for R=32 → rows 0..127
        int col_start   = (tid % (R / 32)) * 32; // 0, 8, 16, 24 for R=32, ValsPerThread=8... 
        // Actually ValsPerThread=32, ThreadsPerRow=R/32=1 for R=32.
        // For R=32: each thread handles one full row (32 values).
        // For larger R: more threads per row.
        int8_t* row_ptr = (int8_t*)(eal_lds + row_in_tile * R);
        for (int i = 0; i < 32; ++i) {
            int8_t raw = ((int8_t*)cv)[i];
            row_ptr[col_start + i] = (int8_t)(((raw + 128) % 64) - 32);
        }
    }
    synchronize_workgroup();

    // ============== STEP B: Generate EBR dense noise into LDS ==============
    // 64 rows × 32 cols, first 64 threads generate (others idle)
    {
        uint32_t tid = threadIdx.x + threadIdx.y * TBLOCK_X;
        if (tid < MACRO_TILE_N) {  // MACRO_TILE_N = 64
            uint32_t thread_coord = n_base * R / 32 + tid + 1;
            // ... identical BLAKE3 pattern with key_b, seed_b ...
            // store to ebr_lds[tid * R .. tid * R + 31]
        }
    }
    synchronize_workgroup();

    // ============== STEP C: Deterministic s_a, s_b seeds → registers ==============
    uint32_t sa_seed[8], sb_seed[8];
    for (int i = 0; i < 8; ++i) {
        sa_seed[i] = s_a_seed[i];
        sb_seed[i] = s_b_seed[i];
    }

    // ============== STEP D: rocwMMA GEMM with on-the-fly noise composition ==============
    auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto localWarpOffset = localWarpCoord * make_coord2d(WARP_TILE_M, WARP_TILE_N);
    auto macroTileCoord  = make_coord2d(blockIdx.x, blockIdx.y) * make_coord2d(MACRO_TILE_M, MACRO_TILE_N);
    auto warpTileCoord   = macroTileCoord + localWarpOffset;

    if (get<0>(warpTileCoord + make_coord2d(WARP_TILE_M, WARP_TILE_N)) > m ||
        get<1>(warpTileCoord + make_coord2d(WARP_TILE_N, WARP_TILE_N)) > n)
        return;

    // LDS read/write offsets for GEMM (same as original pearl_gemm_simple)
    constexpr uint32_t ldsWidthGEMM  = MACRO_TILE_K;   // 16
    constexpr uint32_t ldsHtA        = MACRO_TILE_M;   // 128
    constexpr uint32_t ldsHtB        = MACRO_TILE_N;   // 64
    constexpr uint32_t ldsHt         = ldsHtA + ldsHtB;
    constexpr uint32_t ldsldGEMM     = ldsHt;          // col_major
    constexpr uint32_t sizeLdsGEMM   = ldsHt * ldsWidthGEMM; // 192×16 = 3072

    auto* ldsLo = gemm_lds;
    auto* ldsHi = ldsLo + sizeLdsGEMM;

    auto ldsWriteOffsetA = 0u;
    auto ldsWriteOffsetB = ldsldGEMM * ldsHtA; // start after A data in col_major
    auto ldsReadOffsetA  = ldsWriteOffsetA + get<0>(localWarpOffset); // warp's A slice
    auto ldsReadOffsetB  = ldsWriteOffsetB + get<1>(localWarpOffset); // warp's B slice

    MmaFragAcc mmaFragAcc;
    fill_fragment(mmaFragAcc, (ComputeT)0);

    // ---- Main K loop ----
    for (uint32_t k_base = 0; k_base < k; k_base += MACRO_TILE_K) {
        uint32_t k_tile = min(MACRO_TILE_K, k - k_base);

        // Pre-load sparse index pairs for this K tile
        int ear_f[16], ear_s[16], ebl_f[16], ebl_s[16];
        for (uint32_t l = 0; l < k_tile; ++l) {
            uint32_t k_idx = k_base + l;
            ear_f[l] = ear_first[k_idx];
            ear_s[l] = ear_second[k_idx];
            ebl_f[l] = ebl_first[k_idx];
            ebl_s[l] = ebl_second[k_idx];
        }

        // ---- Fill LDS buffer with noised A tile (128 × 16) ----
        // Each thread fills a subset; target is col_major layout
        // A[i][l] = clip_i8(s_a(i, k_base+l) + eal[i][ear_f[l]] - eal[i][ear_s[l]])
        {
            int tid = threadIdx.x + threadIdx.y * TBLOCK_X; // 0..127
            // 128 threads × 16 elements = 2048 elements. Each thread does 16.
            for (uint32_t l = 0; l < k_tile; ++l) {
                int row = tid; // 0..127 covers MACRO_TILE_M exactly
                if (row >= get<0>(macroTileCoord + make_coord2d(MACRO_TILE_M, 0)) - m_base)
                    continue; // boundary check within block's M range, not needed if aligned

                // Deterministic s_a[row][k_base+l]
                uint32_t idx = (m_base + row) * k + (k_base + l);
                int32_t s_val = (int32_t)(int8_t)(((idx * 0x9e3779b9u) & 0xff) ^ sa_seed[idx % 8]);

                // Noise lookup from LDS
                int r0 = ear_f[l], r1 = ear_s[l];
                int32_t noise = (int32_t)eal_lds[row * R + r0] - (int32_t)eal_lds[row * R + r1];

                int32_t sum = s_val + noise;
                int8_t val = (int8_t)max(-128, min(127, sum)); // clip_i8

                // Write to GEMM LDS in col_major: element (row, l) → offset l * ldsHtA + row
                ldsLo[ldsWriteOffsetA + l * ldsHtA + row] = val;
            }
        }

        // ---- Fill LDS buffer with noised B tile (64 × 16) ----
        {
            int tid = threadIdx.x + threadIdx.y * TBLOCK_X;
            // 64 threads handle B, first 64 of 128
            if (tid < MACRO_TILE_N) {
                for (uint32_t l = 0; l < k_tile; ++l) {
                    int col = tid; // 0..63
                    uint32_t idx = (n_base + col) * k + (k_base + l);
                    int32_t s_val = (int32_t)(int8_t)((((idx * 0x9e3779b9u + 0x55) & 0xff) ^ sb_seed[idx % 8]));

                    int r0 = ebl_f[l], r1 = ebl_s[l];
                    int32_t noise = (int32_t)ebr_lds[col * R + r0] - (int32_t)ebr_lds[col * R + r1];

                    int32_t sum = s_val + noise;
                    int8_t val = (int8_t)max(-128, min(127, sum));

                    // col_major: element (col, l) in B perspective
                    // B is (N,K) row-major → LDS col_major with offset l * ldsHtB + col
                    ldsLo[ldsWriteOffsetB + l * ldsHtB + col] = val;
                }
            }
        }
        synchronize_workgroup();

        // ---- rocwMMA: LDS → register fragments → mma_sync ----
        LRFragA lrFragA;
        LRFragB lrFragB;
        load_matrix_sync(lrFragA, ldsLo + ldsReadOffsetA, ldsldGEMM);
        load_matrix_sync(lrFragB, ldsLo + ldsReadOffsetB, ldsldGEMM);
        mma_sync(mmaFragAcc,
                 transformLRFragAToMmaFragA(lrFragA),
                 transformLRFragBToMmaFragB(lrFragB),
                 mmaFragAcc);

        // Swap double buffer for next iteration
        auto* tmp = ldsLo;
        ldsLo = ldsHi;
        ldsHi = tmp;
        synchronize_workgroup();
    }

    // ---- Store output C tile ----
    MmaFragD mmaFragD;
    for (int i = 0; i < mmaFragD.num_elements; ++i)
        mmaFragD.x[i] = (OutputT)mmaFragAcc.x[i];
    store_matrix_sync(d + MmaFragDMap1d::fromMatrixCoord(warpTileCoord, ldd), mmaFragD, ldd);
}
1.4 Noise Tile → rocwMMA Fragment Mapping
A matrix (M×K, row-major):
The kernel generates a MACRO_TILE_M × MACRO_TILE_K (=128×16) chunk of noised A per K iteration.
Level	Shape	Layout
Generated tile	128×16 int8	row-major (in LDS: col_major)
LWFragA (=LDS write)	128×16 int8	col_major in LDS
LRFragA (warp read)	64×16 int8	col_major in LDS → registers
MmaFragA (inner)	64×32×16 int8	row_major
B matrix (N×K, row-major in host → K×N col-major in rocwMMA):
Level	Shape
Generated tile	64×16 int8
LWFragB (=LDS write)	64×16 int8, transposed then col_major in LDS
LRFragB (warp read)	32×16 int8
MmaFragB (inner)	64×32×16 int8
The fusion point: Instead of loading A/B from global memory into GRFragA/B, we fill the LDS buffer directly (LDS[writeOffset + l * ldsHt + row] in col_major). The rest of the pipeline (LDS→LRFrag→transform→mma_sync) is identical to the original pearl_gemm_simple.
1.5 Register Pressure Analysis & Mitigation
Peak register pressure (per warp):
Component
MmaFragAcc (4×2 blocks × 16×16 i32)
LRFragA (64×16 int8)
LRFragB (32×16 int8)
GRFragA equivalent (128×16 int8, only during fill)
GRFragB equivalent (64×16 int8, only during fill)
BLAKE3 temp (state[16] + cv[8] + block_words[16])
Sparse indices (4×16 int)
Address arithmetic, loop counters
Theoretical peak
Mitigation strategies:
1. Phase registers: The BLAKE3 noise generation (Step A, Step B) completes BEFORE the GEMM K-loop. After synchronize_workgroup(), BLAKE3 registers are dead and can be reused by MMA. This alone eliminates 40 VGPRs of overlap.
2. LDS fill → GEMM buffer handoff: The noise composition fills LDS directly (no intermediate GRFragA/B fragments needed). Values go from eal_lds, ear_f/s, sa_seed → LDS buffer. The old GRFragA/B VGPRs (for global read prefetch) are replaced by cheaper index lookups.
3. Eliminate double-buffer prefetch: With on-the-fly noise generation, there is no global memory latency to hide. We can simplify to single-buffered LDS (read current, write next, sync). This reduces LDS from 2× to 1× (6KB → 3KB) and eliminates the GRFragA/B live range overlap. The cost is one extra synchronize_workgroup() per K iteration.
4. Target VGPR count: 128 VGPRs. gfx1100 has 1536 VGPRs per CU / 128 per wave = 12 waves per CU. At 128 VGPRs, 6 waves can run concurrently per CU (1536/(128×2)). This is a good occupancy target.
1.6 Validation Strategy
In test_pearl_hip.cpp, add test_pearl_fused_noise_gemm():
static bool test_pearl_fused_noise_gemm(const char* tag) {
    HarnessFixture fixture = default_fixture(); // m=256 n=128 common_dim=1024 rank=32
    int h = fixture.m, w = fixture.n, k = fixture.config.common_dim, R = fixture.config.rank;

    // 1. Compute reference the two-step way (existing code)
    //    CPU: EAL, EAR, EBL, EBR → noise_a, noise_b → ApEA, BpEB → GEMM_ref
    //    GPU: pearl_gemm_simple(ApEA, BpEB) → C_ref

    // 2. Compute fused way
    //    GPU: Upload ear/ebl sparse pairs (2*k*4 bytes each)
    //    GPU: Launch pearl_fused_noise_gemm(kernel)
    //    GPU: Download C_fused

    // 3. Compare C_ref vs C_fused element-by-element
    //    Expected: exact match (int32), zero tolerance
    //    Because noise composition is deterministic and identical algorithm

    // 4. Also verify against CPU-only reference (compute_host_gemm_ref)
}
Test dimensions:
- Small: M=128, N=64, K=64, R=32 (single block, validates tile logic)
- Default: M=256, N=128, K=1024, R=32 (multi-block, matches fixture)
- Stress: M=1024, N=1024, K=4096, R=32 (benchmark dimensions)
Correctness criteria:
- Element-by-element exact match between fused and two-step approach
- All C elements must match the pure CPU reference
- No edge artifacts at tile boundaries (M%128≠0, N%64≠0 cases)
Phase 2: GPU Blake3
2.1 Approach
The existing blake3-inline.hip.inc provides all necessary __device__ functions:
- blake3_hash_single_chunk(input, input_len, output) — for ≤1024B inputs
- blake3_compress_in_place(cv, block, block_len, counter, flags) — single block compression
- blake3_cv_init(cv) — IV initialization
All three hashing operations are single-chunk (inputs ≤128B):
Operation
compute_job_key
compute_commitment_hash (b_seed)
compute_commitment_hash (a_seed)
compute_jackpot_hash
2.2 Device-Side Functions
// ---- GPU compute_job_key ----
__device__ void gpu_compute_job_key(
    uint8_t const header_bytes[76],   // IncompleteBlockHeader serialized
    uint8_t const config_bytes[52],   // MiningConfiguration serialized
    uint8_t job_key_out[32])
{
    uint8_t input[128];
    for (int i = 0; i < 76; ++i) input[i] = header_bytes[i];
    for (int i = 0; i < 52; ++i) input[76 + i] = config_bytes[i];
    blake3_hash_single_chunk(input, 128, job_key_out);
}

// ---- GPU compute_commitment_hash ----
__device__ void gpu_compute_commitment_hash(
    uint8_t const job_key[32],
    uint8_t const hash_a[32],
    uint8_t const hash_b[32],
    uint8_t b_noise_seed_out[32],
    uint8_t a_noise_seed_out[32])
{
    uint8_t buf[64];

    // b_noise_seed = blake3(job_key || hash_b)
    for (int i = 0; i < 32; ++i) buf[i] = job_key[i];
    for (int i = 0; i < 32; ++i) buf[32 + i] = hash_b[i];
    blake3_hash_single_chunk(buf, 64, b_noise_seed_out);

    // a_noise_seed = blake3(b_noise_seed || hash_a)
    for (int i = 0; i < 32; ++i) buf[i] = b_noise_seed_out[i];
    for (int i = 0; i < 32; ++i) buf[32 + i] = hash_a[i];
    blake3_hash_single_chunk(buf, 64, a_noise_seed_out);
}

// ---- GPU compute_jackpot_hash (keyed) ----
__device__ void gpu_compute_jackpot_hash(
    uint32_t const jackpot_msg[16],   // 64B as 16 uint32
    uint8_t const a_noise_seed[32],   // key
    uint8_t jackpot_hash_out[32])
{
    uint32_t cv[8];
    for (int i = 0; i < 8; ++i)
        cv[i] = blake3_load32(a_noise_seed + i * 4);

    uint8_t block[64];
    for (int i = 0; i < 16; ++i) {
        block[i*4 + 0] = (jackpot_msg[i]) & 0xff;
        block[i*4 + 1] = (jackpot_msg[i] >> 8) & 0xff;
        block[i*4 + 2] = (jackpot_msg[i] >> 16) & 0xff;
        block[i*4 + 3] = (jackpot_msg[i] >> 24) & 0xff;
    }

    uint8_t flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
    blake3_compress_in_place(cv, block, 64, 0, flags);

    for (int i = 0; i < 8; ++i)
        blake3_store32(jackpot_hash_out + i * 4, cv[i]);
}
2.3 Validation
// Compare GPU-computed job_key against CPU blake3
// All 32 bytes must match exactly
// Test with default_fixture() values
// Similarly for commitment_hash and jackpot_hash
Test:
1. Upload header_bytes + config_bytes to GPU constant memory
2. Launch a single-block kernel (1×1) that computes gpu_compute_job_key
3. Download result, compare byte-by-byte against CPU compute_job_key()
4. Repeat for commitment_hash and jackpot_hash
Cost: Each blake3 single-chunk call is ~14 rounds of 8 G-functions = ~112 G-functions per 128B hash. At 3 cycles per G-function on gfx1100, that's 336 cycles. Negligible compared to GEMM.
Phase 3: Full Fused Kernel
3.1 Architecture
Single launch, zero CPU-GPU sync between steps:
┌─────────────────────────────────────────────────────────┐
│                   __global__ pearl_full_fused            │
│                                                         │
│  INPUT (constant/global):                                │
│    header_bytes[76], config_bytes[52]                    │
│    hash_a[32], hash_b[32]                                │
│    seed_a_label[32], seed_b_label[32]                    │
│    s_a_seed[4], s_b_seed[4]                              │
│    ear_first[k], ear_second[k]                           │
│    ebl_first[k], ebl_second[k]                           │
│                                                         │
│  STEP 1 (ALL threadblocks, redundant):                   │
│    blake3(header+config) → job_key[32]                   │
│    blake3(job_key || hash_b) → b_noise_seed[32]          │
│    blake3(b_noise_seed || hash_a) → a_noise_seed[32]     │
│                                                         │
│  STEP 2 (per threadblock, cooperative):                  │
│    Generate EAL in LDS (128×R, BLAKE3 keyed)            │
│    Generate EBR in LDS (64×R, BLAKE3 keyed)             │
│                                                         │
│  STEP 3 (per threadblock, K-loop):                       │
│    For k_base in 0..K step MACRO_TILE_K:                 │
│      Compose A_tile[k_base] = s[a] + EAL[ear] noise      │
│      Compose B_tile[k_base] = s[b] + EBR[ebl] noise      │
│      rocwMMA mma_sync accumulate                          │
│                                                         │
│  STEP 4 (per threadblock, reduction):                    │
│    Per rank-group: XOR all C[i][j] for this block's tile │
│    AtomicXOR into global jackpot_accum[16]               │
│    (one threadblock does final blake3 jackpot hash)      │
│                                                         │
│  OUTPUT (global memory):                                  │
│    C[M×N] int32                                          │
│    job_key[32] (written by block 0)                      │
│    jackpot_hash[32] (written by block 0)                 │
└─────────────────────────────────────────────────────────┘
3.2 Kernel Launch Config
gridDim  = (ceil_div(M, 128), ceil_div(N, 64), 1)
blockDim = (64, 2, 1)  // 128 threads

sharedMem = LDS_EAL + LDS_EBR + LDS_GEMM_BUFFER + LDS_REDUCE_SCRATCH
          = 4096       + 2048       + 3072           + 512
          = 9728 bytes ≈ 9.5 KB

Additional global memory:
  jackpot_accum[16] (uint32, zero-initialized by host)
  out_job_key[32]
  out_jackpot_hash[32]
3.3 Register vs LDS vs Global Memory Map
Component
BLAKE3 working state
job_key, a/b_noise_seed
EAL dense (128×R)
EBR dense (64×R)
GEMM A tile (128×16)
GEMM B tile (64×16)
Reduce scratch
mmaFragAcc
jackpot_partial[16]
jackpot_accum[16]
output C[M×N]
out_job_key, out_hash
3.4 Jackpot Reduction Detail
The jackpot accumulation follows the original compute_jackpot_words algorithm:
For each rank-group r = 0..(K/R - 1):
  1. Compute partial GEMM for K-columns [r*R, (r+1)*R)
  2. XOR all elements of C for this rank-group
  3. jackpot_msg[r % 16] = rotate_left(jackpot_msg[r % 16], 13)
  4. jackpot_msg[r % 16] ^= xored_tile
For the GPU:
// Per threadblock, accumulate partial jackpot as GEMM progresses
uint32_t block_jackpot[16] = {0};

for (uint32_t k_base = 0; k_base < k; k_base += MACRO_TILE_K) {
    // ... GEMM accumulation into mmaFragAcc (int32 accumulator) ...

    // Check if we crossed a rank boundary within this K range
    uint32_t rank_group_start = (k_base / R) * R;
    uint32_t rank_group_end   = min(k, rank_group_start + R);

    if (k_base + MACRO_TILE_K >= rank_group_end) {
        // This K tile completes the current rank group
        // XOR all elements of mmaFragAcc (partial for this warp's tile)
        uint32_t warp_xor = 0;
        for (int i = 0; i < mmaFragAcc.num_elements; ++i)
            warp_xor ^= (uint32_t)mmaFragAcc.x[i];

        // Reduce across warps in block
        uint32_t block_xor = block_reduce_xor_u32_wave_first<128>(warp_xor, lds_scratch);

        // Rotate + XOR into jackpot
        uint32_t tid = (rank_group_end / R - 1) % 16;
        uint32_t rotated = (block_jackpot[tid] << 13) | (block_jackpot[tid] >> 19);
        block_jackpot[tid] = rotated ^ block_xor;

        // Reset accumulator for next rank group
        if (threadIdx.x == 0 && threadIdx.y == 0)
            fill_fragment(mmaFragAcc, (ComputeT)0);
    }
}

// Atomic XOR block_jackpot into global accumulator
for (int i = 0; i < 16; ++i) {
    atomicXor(&global_jackpot[i], block_jackpot[i]);
}
Caveat: The jackpot computation requires per-rank-group partial GEMM results. If MACRO_TILE_K=16 does not align with R=32, the rank boundaries fall mid-tile. The above logic handles this by checking k_base + MACRO_TILE_K >= rank_group_end.
After all threadblocks complete, one designated threadblock (e.g., block (0,0)) computes gpu_compute_jackpot_hash(global_jackpot, a_noise_seed, out_hash).
3.5 Complete Pseudocode
extern "C" __global__ __launch_bounds__(128)
void pearl_full_fused(
    uint32_t m, n, k, R,
    // Serialized header + config (128B total, could be constant memory)
    uint8_t const* header_config,     // 128B: header(76)+config(52)
    uint8_t const* hash_a,            // 32B
    uint8_t const* hash_b,            // 32B
    uint8_t const* seed_a_label,      // 32B
    uint8_t const* seed_b_label,      // 32B
    uint32_t const* s_a_seed,         // 8 u32
    uint32_t const* s_b_seed,         // 8 u32
    int32_t const* ear_first,         // k ints
    int32_t const* ear_second,        // k ints
    int32_t const* ebl_first,         // k ints
    int32_t const* ebl_second,        // k ints
    int32_t*       d,                 // M×N output
    uint32_t       ldd,
    uint32_t*      global_jackpot,    // 16 u32, zero-init
    uint8_t*       out_job_key,       // 32B (only block 0 writes)
    uint8_t*       out_jackpot_hash)  // 32B (only block 0 writes)
{
    HIP_DYNAMIC_SHARED(uint8_t*, lds);

    // ===== STEP 1: BLAKE3 hashing (redundant per block, ~5 μs) =====
    uint8_t job_key[32];
    blake3_hash_single_chunk(header_config, 128, job_key);

    uint8_t b_noise_seed[32], a_noise_seed[32];
    {
        uint8_t buf[64];
        for (int i = 0; i < 32; ++i) buf[i] = job_key[i];
        for (int i = 0; i < 32; ++i) buf[32+i] = hash_b[i];
        blake3_hash_single_chunk(buf, 64, b_noise_seed);

        for (int i = 0; i < 32; ++i) buf[i] = b_noise_seed[i];
        for (int i = 0; i < 32; ++i) buf[32+i] = hash_a[i];
        blake3_hash_single_chunk(buf, 64, a_noise_seed);
    }

    // Block 0 writes job_key (one-time)
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 32; ++i) out_job_key[i] = job_key[i];
    }

    // ===== STEP 2: Generate EAL + EBR dense noise into LDS =====
    // (Same as Phase 1 Steps A-B)
    // EAL: 128×R @ lds[0..4095], key=a_noise_seed, seed=seed_a_label
    // EBR: 64×R  @ lds[4096..6143], key=b_noise_seed, seed=seed_b_label

    // ===== STEP 3: GEMM K-loop with noise composition =====
    // (Same as Phase 1 Step D)
    // PLUS: accumulate partial jackpot per rank-group

    uint32_t block_jackpot[16] = {0};

    for (uint32_t k_base = 0; k_base < k; k_base += MACRO_TILE_K) {
        // Fill noised A/B tiles into LDS
        // ... (Phase 1 code) ...

        // rocwMMA
        // ...

        // Jackpot accumulation on rank boundaries
        // ... (jackpot code above) ...
    }

    // ===== STEP 4: Jackpot reduction & final hash =====
    // Atomic XOR into global (all blocks)
    synchronize_workgroup(); // ensure block result is ready
    if (threadIdx.x < 16 && threadIdx.y == 0) {
        atomicXor(&global_jackpot[threadIdx.x], block_jackpot[threadIdx.x]);
    }

    // Use cooperative groups or a grid sync for the final hash
    // On HIP, use a fence + last-block pattern:
    __threadfence();
    // In practice, launch a separate tiny kernel or use the last block to finish

    // If we can guarantee all blocks have atomically XORed:
    // (requires grid sync via cooperative_groups or a separate launch)
}
Note on grid synchronization: HIP does not natively support cooperative_groups::grid_group::sync() without special launch. The simplest approach: split into two launches: (a) main fused kernel writes C + jackpot_accum, (b) a single-block kernel computes final blake3 from jackpot_accum.
3.6 Validation Strategy
static bool test_pearl_full_fused(const char* tag) {
    HarnessFixture fixture = default_fixture();
    int h = fixture.m, w = fixture.n, k = fixture.config.common_dim, R = fixture.config.rank;

    // --- CPU reference (full pipeline) ---
    Hash256 job_key_cpu    = compute_job_key(fixture.header, fixture.config);
    auto [b_seed, a_seed]  = compute_commitment_hash(job_key_cpu, hash_a, hash_b);
    // ... noise generation, GEMM, jackpot ...
    Hash256 jackpot_hash_cpu = compute_jackpot_hash(jackpot_cpu, a_seed);

    // --- GPU fused ---
    // Upload header_config, hashes, seeds, sparse pairs
    // Launch pearl_full_fused
    // Download: C_gpu, job_key_gpu, jackpot_hash_gpu

    // --- Checks ---
    // 1. job_key_cpu == job_key_gpu (exact 32 bytes)
    // 2. C_cpu == C_gpu (exact, all M×N elements)
    // 3. jackpot_hash_cpu == jackpot_hash_gpu (exact 32 bytes)

    return all_passed;
}
Summary Table
Phase	Kernel
Current	pearl_gemm_simple
Phase 1	pearl_fused_noise_gemm
Phase 2	pearl_blake3_gpu (aux)
Phase 3	pearl_full_fused
Expected performance improvement:
- Phase 1 eliminates: CPU noise generation (BLAKE3 on CPU for O(k×R) rows), noise composition (O(k×h+k×w) CPU), int8 clipping, and two oroMemcpy H2D transfers (A,B matrices of size h×k + w×k int8)
- Phase 2 eliminates: 3 CPU BLAKE3 calls per iteration (~negligible)
- Phase 3 eliminates: One oroMemcpy D2H for job_key (32B), internal sync points
For typical mining dimensions (M=256, N=128, K=1024, R=32):
- A matrix: 256×1024 = 256KB
- B matrix: 128×1024 = 128KB
- Total H2D savings: ~384KB per nonce iteration
- CPU noise generation savings: ~(256+128)×1024 BLAKE3 keyed hashes ≈ O(M×K) operations now on GPU in LDS
The primary bottleneck in the current pipeline is the CPU-side noise generation and the H2D transfers for A and B matrices. Phase 1 directly addresses both by generating noise on-GPU in LDS and keeping the noised matrices entirely on-chip.