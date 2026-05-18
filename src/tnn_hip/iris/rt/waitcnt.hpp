#pragma once

#include "arch_traits.hpp"
#include "coordinate.hpp"

namespace iris::hip {

struct WaitcntLayoutGfx11 {
    static constexpr int vm_mask = 0x3f;
    static constexpr int lgkm_mask = 0x3f;
    static constexpr int exp_mask = 0x07;
    static constexpr bool has_exp = true;

    IRIS_HOST_DEVICE_INLINE static constexpr int pack_vm(int c) {
        return (c & vm_mask) << 10;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int pack_lgkm(int c) {
        return (c & lgkm_mask) << 4;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int pack_exp(int c) {
        return c & exp_mask;
    }
};

struct WaitcntLayoutGfx12 {
    static constexpr int vm_mask = 0x3f;
    static constexpr int lgkm_mask = 0x3f;
    static constexpr int exp_mask = 0x00;
    static constexpr bool has_exp = false;

    IRIS_HOST_DEVICE_INLINE static constexpr int pack_vm(int c) {
        return (c & vm_mask) << 10;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int pack_lgkm(int c) {
        return (c & lgkm_mask) << 4;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int pack_exp(int) {
        return 0;
    }
};

struct WaitcntLayoutLegacy {
    static constexpr int vm_mask = 0x3f;
    static constexpr int lgkm_mask = 0x0f;
    static constexpr int exp_mask = 0x07;
    static constexpr bool has_exp = true;

    IRIS_HOST_DEVICE_INLINE static constexpr int pack_vm(int c) {
        return ((c & 0x30) << 10) | ((c & 0x0f) << 0);
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int pack_lgkm(int c) {
        return (c & lgkm_mask) << 8;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int pack_exp(int c) {
        return (c & exp_mask) << 4;
    }
};

#if defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
using WaitcntLayout = WaitcntLayoutGfx11;
#elif defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__)
using WaitcntLayout = WaitcntLayoutGfx12;
#else
using WaitcntLayout = WaitcntLayoutLegacy;
#endif

template <int VmCnt, int LgkmCnt, int ExpCnt = 0>
IRIS_DEVICE_INLINE void waitcnt() {
    static_assert((VmCnt & ~WaitcntLayout::vm_mask) == 0, "vmcnt out of range");
    static_assert((LgkmCnt & ~WaitcntLayout::lgkm_mask) == 0, "lgkmcnt out of range");
    static_assert(!WaitcntLayout::has_exp || ((ExpCnt & ~WaitcntLayout::exp_mask) == 0), "expcnt out of range");

#if IRIS_AMDGCN_FRONTEND
    constexpr int packed =
        WaitcntLayout::pack_vm(VmCnt) |
        WaitcntLayout::pack_lgkm(LgkmCnt) |
        WaitcntLayout::pack_exp(ExpCnt);
    asm volatile("s_waitcnt %0" : : "n"(packed) : "memory");
#endif
}

IRIS_DEVICE_INLINE void wait_lgkmcnt0() {
#if defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    waitcnt<WaitcntLayout::vm_mask, 0, WaitcntLayout::exp_mask>();
#elif defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__)
    waitcnt<WaitcntLayout::vm_mask, 0, 0>();
#else
    waitcnt<WaitcntLayout::vm_mask, 0, WaitcntLayout::exp_mask>();
#endif
}

} // namespace iris::hip
