#pragma once

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__)
#error "Unqualified device architecture: recipe requires gfx1100"
#endif

namespace tnn::hip::iris::gemm::qualified {

// Instruction and ownership mechanics. These are not interchangeable with
// another architecture merely because it also supports integer matrix ops.
struct Gfx1100 {
    static constexpr int wave_size = 32;
    static constexpr int vector_bytes = 16;
    static constexpr bool signed_inputs = true;
    static constexpr bool paired = true;
    static constexpr bool swizzle = false;
    static constexpr bool half_reads = false;
    static constexpr bool local_barrier = true;

    static constexpr unsigned b_offset(unsigned n, unsigned k) {
        unsigned x = n * 32 + k;
        return x + (x / 128) * 16;
    }

    static constexpr unsigned row(unsigned wave, unsigned lane) {
        return (wave % 4) * 32 + (lane % 16) * 2;
    }

    static constexpr unsigned col(unsigned wave, unsigned lane, unsigned nb, unsigned reg) {
        return (wave / 4) * 16 + nb * 32 + 2 * reg + lane / 16;
    }
};

// Scheduling choices for the one qualified tile. Values are compile-time;
// there is deliberately no runtime schedule switch in the launch API.
struct Paired128x256 {
    using Arch = Gfx1100;

    static constexpr int tile_m = 128;
    static constexpr int tile_n = 256;
    static constexpr int tile_k = 32;
    static constexpr int threads = 256;

    // prefetch: 0=before operands, 1=between operands and WMMA, 2=after checkpoint.
    static constexpr int prefetch = 0;
    static constexpr bool shared_address = true;
    static constexpr bool streamed = false;
    static constexpr bool rematerialize = true;
    static constexpr int mapping = 1;
    static constexpr bool prefetch_enabled = true;
    static constexpr bool double_buffer = false;
    static constexpr bool priority = false;
};

// Constants only: checkpoint statements intentionally live in checkpoint.inc.
struct PearlRank128 {
    static constexpr int rank = 128;
    static constexpr int chains = 4;
    static constexpr int update = 0;
};

template <class Architecture = Gfx1100, class Schedule = Paired128x256, class Pearl = PearlRank128>
struct Recipe {
    static_assert(__is_same(Architecture, Gfx1100), "Unqualified architecture");
    static_assert(__is_same(Schedule, Paired128x256), "Unqualified schedule or tile");
    static_assert(__is_same(Pearl, PearlRank128), "Unqualified Pearl policy or rank");
    static_assert(__is_same(typename Schedule::Arch, Architecture),
                  "Architecture/schedule mismatch");

    using Arch = Architecture;
    using Scheduling = Schedule;
    using Checkpoint = Pearl;
};

using Gfx1100Pearl = Recipe<>;

} // namespace tnn::hip::iris::gemm::qualified
