    using Schedule = typename Configuration::Scheduling;
    using Pearl = typename Configuration::Checkpoint;
    using Arch = typename Configuration::Arch;

    static_assert(Schedule::tile_m == 128 && Schedule::tile_n == 256 && Schedule::tile_k == 32 &&
                      Schedule::threads == 256,
                  "Only the paired128x256 recipe has been triaged");
    static_assert(Pearl::rank == 128 && !NoD, "Only rank128 full-D specializations are in scope");
    constexpr bool Prefetch = Schedule::prefetch_enabled;
    constexpr bool Local = Arch::local_barrier;
    constexpr bool Double = Schedule::double_buffer;
    constexpr bool Priority = Schedule::priority;
    constexpr bool Paired = Arch::paired;
    constexpr bool Swizzle = Arch::swizzle;
    constexpr bool Half = Arch::half_reads;
    constexpr int NB = Schedule::tile_n / Arch::wave_size;
    constexpr int Rank = Pearl::rank;
    constexpr int MW = Schedule::tile_m / Arch::wave_size;
    constexpr bool Streamed = Schedule::streamed;

    // Shared-memory staging and the fixed lane ownership for this recipe.
    __shared__ __align__(16) unsigned char sa[Schedule::tile_m * Schedule::tile_k];
    constexpr unsigned BSpan = Swizzle ? 2048 : 2304;
    __shared__ __align__(16) unsigned char sb[BSpan * (NB / 2) * (Double ? 2 : 1)];
    unsigned tid = threadIdx.x;
    unsigned lane = tid % Arch::wave_size;
    unsigned wave = tid / Arch::wave_size;
    // Neighboring N tiles share A in L2. WGM is a measured build parameter.
    constexpr unsigned WGM = Schedule::mapping;
    unsigned gm = m / Schedule::tile_m;
    unsigned gn = n / (NB * 32);
    unsigned group = blockIdx.x / (gm * WGM);
    unsigned width = min(WGM, gn - group * WGM);
    unsigned local = blockIdx.x - group * gm * WGM;
    unsigned bm = (local / width) * Schedule::tile_m;
    unsigned bn = (group * WGM + local % width) * (NB * 32);

    const int8_t *ap = a + bm + (tid % (Schedule::tile_m / Arch::vector_bytes)) * 16 +
                       size_t(tid / (Schedule::tile_m / Arch::vector_bytes)) * lda;
    const int8_t *bp = b + size_t(bn + tid / 2) * ldb + (tid % 2) * 16;
    unsigned aw = (tid / (Schedule::tile_m / Arch::vector_bytes)) * Schedule::tile_m +
                  (tid % (Schedule::tile_m / Arch::vector_bytes)) * 16;
    unsigned am = Arch::row(wave, lane);
    unsigned nn = (wave / MW) * 16 + lane % 16;
    // Accumulators are cumulative across rank checkpoints, never reset per rank.
    I8 c[2 * NB] = {};
    uint32_t transcript[16] = {};
    I4 ga{};
    I4 gb[256 / Schedule::tile_m]{};

    auto fetch = [&]() {
        ga = *reinterpret_cast<const I4 *>(ap);
#pragma unroll
        for (int i = 0; i < 256 / Schedule::tile_m; ++i)
            gb[i] = *reinterpret_cast<const I4 *>(bp + size_t(i * Schedule::tile_m) * ldb);
    };
    if constexpr (Prefetch)
        fetch();

    // Stage, prefetch, consume both K=16 operands, then checkpoint in-place.
    for (unsigned kk = 0; kk < k; kk += Schedule::tile_k) {
        unsigned slot = Double ? ((kk / Schedule::tile_k) & 1) : 0;
        auto *as = sa;
        auto *bs = sb + slot * BSpan * (NB / 2);
        if constexpr (!Prefetch)
            fetch();
        *reinterpret_cast<I4 *>(as + aw) = ga;
#pragma unroll
        for (int i = 0; i < 256 / Schedule::tile_m; ++i)
            *reinterpret_cast<I4 *>(bs + b_offset<Arch, Swizzle>(tid / 2 + i * Schedule::tile_m,
                                                                 (tid % 2) * 16)) = gb[i];
        shared_barrier<Local>();
        ap += size_t(Schedule::tile_k) * lda;
        bp += Schedule::tile_k;
        if constexpr (Prefetch && Schedule::prefetch == 0) {
            if (kk + Schedule::tile_k < k)
                fetch();
        }
        unsigned operand_am = am;
        unsigned operand_nn = nn;
        if constexpr (Schedule::shared_address) {
            unsigned fresh;
            asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(fresh));
            unsigned fresh_wave = __builtin_amdgcn_readfirstlane(threadIdx.x) / 32;
            operand_am = (fresh_wave % MW) * 32 + (fresh % 16) * 2;
            operand_nn = (fresh_wave / MW) * 16 + fresh % 16;
        }
        auto x0 = operands<Schedule, 0, NB, Paired, Swizzle, Half>(as, bs, operand_am, operand_nn);
        if constexpr (Streamed) {
            compute<Arch, NB, Priority>(c, x0);
            __builtin_amdgcn_sched_barrier(0);
            auto x1 =
                operands<Schedule, 16, NB, Paired, Swizzle, Half>(as, bs, operand_am, operand_nn);
            compute<Arch, NB, Priority>(c, x1);
        } else {
            auto x1 =
                operands<Schedule, 16, NB, Paired, Swizzle, Half>(as, bs, operand_am, operand_nn);
            if constexpr (Schedule::prefetch == 1)
                if (kk + Schedule::tile_k < k)
                    fetch();
            compute<Arch, NB, Priority>(c, x0);
            compute<Arch, NB, Priority>(c, x1);
        }

        if constexpr (!Raw)
            if ((kk + Schedule::tile_k) % Rank == 0) {
#include "checkpoint.inc"
            }
        if constexpr (Schedule::prefetch == 2)
            if (kk + Schedule::tile_k < k)
                fetch();
        // With two slots the next write uses the other slot. The next start
        // barrier proves all reads complete before this slot is reused.
        if constexpr (!Double)
            shared_barrier<Local>();
    }

    // End accumulator lifetimes before hashing the per-thread transcript.
    if constexpr (!NoD) {
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                unsigned row = bm + am;
                unsigned col = bn + Arch::col(wave, lane, nb, e);
                // Adjacent rows are a vector store, following the studied epilogue.
                using I2 = int __attribute__((ext_vector_type(2)));
                *reinterpret_cast<I2 *>(d + row + size_t(col) * ldd) =
                    I2{c[nb * 2][e], c[nb * 2 + 1][e]};
            }
        }
    }
    if constexpr (!Raw) {
        // Optional D stores precede hashing; GEMM accumulators are now dead.
        uint32_t digest[8];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i)
            digest[i] = key[i];
        blake3_compress_in_place(digest, reinterpret_cast<const uint8_t *>(transcript), 64, 0,
                                 CHUNK_START | CHUNK_END | ROOT | (1 << 4));
        if constexpr (Diagnostic) {
            unsigned row = bm + am, col = bn + (wave / MW) * 16 + lane / 16;
            size_t pos =
                size_t(row / 2) * (n / 64) + (col / 256) * 4 + ((col % 256) / 16) * 2 + col % 2;
            size_t positions = size_t(m) * n / 128;
#pragma unroll
            for (unsigned i = 0; i < 16; ++i)
                diagnostic[size_t(i) * positions + pos] = transcript[i];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
                diagnostic[size_t(i + 16) * positions + pos] = digest[i];
        }
        bool equal = true, lower = false;
#pragma unroll
        for (int i = 7; i >= 0; --i) {
            lower |= equal && digest[i] < target[i];
            equal &= digest[i] == target[i];
        }
        if (lower || equal) {
            unsigned slot = atomicAdd(&results.state->total_hits, 1u);
            if (slot < results.capacity) {
                NativeWinner &winner = results.winners[slot];
                winner.row = bm + am;
                winner.col = bn + (wave / MW) * 16 + lane / 16;
#pragma unroll
                for (unsigned i = 0; i < 8; ++i)
                    winner.digest[i] = digest[i];
            } else
                atomicExch(&results.state->overflow, 1u);
        }
    }
