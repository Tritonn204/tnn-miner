#pragma once
#include <cstdint>

namespace tnn::hip::iris::gemm::native128 {

// Exact gfx1100 SourceSwap=1 ownership reconstructed from the selected ISA.
struct Gfx1100Wmma {
    static constexpr unsigned wave_size = 32;
    static constexpr unsigned vector_bytes = 16;

    static constexpr unsigned a_row(unsigned wave, unsigned lane, unsigned fragment) {
        return (wave % 2) * 16 + lane % 16 + fragment * 32;
    }

    static constexpr unsigned b_col(unsigned wave, unsigned lane, unsigned fragment) {
        return (wave / 2) * 64 + (lane % 16) * 4 + fragment;
    }

    static constexpr unsigned output_col(unsigned wave, unsigned lane, unsigned fragment,
                                         unsigned element) {
        return (wave / 2) * 64 + (2 * element + lane / 16) * 4 + fragment;
    }

    static constexpr unsigned b_offset(unsigned col, unsigned k) {
        const unsigned linear = col * 32 + k;
        return linear + (linear / 128) * 16;
    }
};

template <bool Pipelined, unsigned Mapping = 4> struct Vendor128Schedule {
    static constexpr unsigned tile_m = 128;
    static constexpr unsigned tile_n = 128;
    static constexpr unsigned tile_k = 32;
    static constexpr unsigned threads = 128;
    static constexpr unsigned fragments_m = 4;
    static constexpr unsigned fragments_n = 4;
    static constexpr unsigned mapping = Mapping;
    static constexpr bool pipelined = Pipelined;
    static constexpr unsigned a_bytes = 4096;
    static constexpr unsigned b_bytes = 4608;
    static constexpr unsigned bank_stride = 16384;
    static constexpr unsigned lds_bytes = (Pipelined ? bank_stride : 0) + a_bytes + b_bytes;
};

struct GroupedLoads;
template <class Architecture, class Scheduling, class Loading = GroupedLoads>
struct Recipe {
    using Arch = Architecture;
    using Schedule = Scheduling;
    using Loads = Loading;
    static_assert(Scheduling::threads == 128 && Scheduling::tile_m == 128 &&
                  Scheduling::tile_n == 128);
};

using Native128 = Recipe<Gfx1100Wmma, Vendor128Schedule<true>>;

} // namespace tnn::hip::iris::gemm::native128
