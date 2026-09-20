#pragma once
// Opt-in only. Do not route this through DefaultBackend.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__)
#error "Experimental paired backend requires validated gfx1100 mechanics"
#endif
namespace tnn::hip::iris::gemm::experimental {
struct Gfx1100Paired {
    static constexpr int wave_size=32, vector_bytes=16;
    static constexpr bool source_swap=true, signed_inputs=true;
    static constexpr unsigned b_offset(unsigned n,unsigned k) {
        unsigned x=n*32+k;return x+(x/128)*16;
    }
    static constexpr unsigned row(unsigned wave,unsigned lane) {
        return (wave%4)*32+(lane%16)*2;
    }
    static constexpr unsigned col(unsigned wave,unsigned lane,unsigned nb,unsigned reg) {
        return (wave/4)*16+nb*32+2*reg+lane/16;
    }
};
struct Paired128x256Schedule {
    using Arch=Gfx1100Paired;
    static constexpr int tile_m=128,tile_n=256,tile_k=32,threads=256;
    static constexpr int prefetch=0,shared_address=1,streamed=0,rematerialize=1,mapping=1;
};
// A deliberately narrow validated specialization, not a portable default.
static_assert(Paired128x256Schedule::tile_m*2==Paired128x256Schedule::threads);
}
