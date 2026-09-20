#pragma once

namespace iris::hip {

struct generic_arch {
    static constexpr int gfx_major = 0;
    static constexpr int wave_size = 64;
    static constexpr int vector_bytes = 8;
    static constexpr bool has_wmma = false;
    static constexpr bool has_xdl  = false;
    static constexpr bool has_mfma = false;
    static constexpr bool has_rdna3_wmma = false;
    static constexpr bool has_rdna4_wmma = false;
    static constexpr bool waitcnt_has_exp = true;
    static constexpr int waitcnt_exp_mask = 0x07;
    static constexpr int default_block_threads = 128;
};

#if defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
struct arch : generic_arch {
    static constexpr int gfx_major = 11;
    static constexpr int wave_size = 32;
    static constexpr int vector_bytes = 16;
    static constexpr bool has_wmma = true;
    static constexpr bool has_rdna3_wmma = true;
    static constexpr bool has_rdna4_wmma = false;
    static constexpr bool waitcnt_has_exp = true;
    static constexpr int waitcnt_exp_mask = 0x07;
    static constexpr int default_block_threads = 256;
};
#elif defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__)
struct arch : generic_arch {
    static constexpr int gfx_major = 12;
    static constexpr int wave_size = 32;
    static constexpr int vector_bytes = 16;
    static constexpr bool has_wmma = true;
    static constexpr bool has_rdna3_wmma = false;
    static constexpr bool has_rdna4_wmma = true;
    static constexpr bool waitcnt_has_exp = false;
    static constexpr int waitcnt_exp_mask = 0x00;
    static constexpr int default_block_threads = 256;
};
#elif defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)
struct arch : generic_arch {
    static constexpr int gfx_major = 9;
    static constexpr int wave_size = 64;
    static constexpr int vector_bytes = 16;
    static constexpr bool has_xdl  = true;
    static constexpr bool has_mfma = true;
    static constexpr bool has_wmma = false;
    static constexpr bool waitcnt_has_exp = true;
    static constexpr int waitcnt_exp_mask = 0x07;
    static constexpr int default_block_threads = 256;
};
#else
using arch = generic_arch;
#endif

} // namespace iris::hip
