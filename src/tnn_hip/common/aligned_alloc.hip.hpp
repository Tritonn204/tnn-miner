
#pragma once

#include <tnn_hip/common/gpu_compat.hpp>
#include <stdint.h>

struct AlignedDevAlloc {
    uint8_t* raw = nullptr;
    uint8_t* aligned = nullptr;
    size_t raw_size = 0;
};

static inline uintptr_t align_up_uintptr(uintptr_t p, size_t align) {
    return (p + (align - 1)) & ~(uintptr_t)(align - 1);
}

static inline oroError_t oroMallocAligned(AlignedDevAlloc& out, size_t size, size_t align) {
    if (align == 0 || (align & (align - 1)) != 0) return oroErrorInvalidValue;

    out.raw = nullptr;
    out.aligned = nullptr;
    out.raw_size = 0;

    const size_t alloc_size = size + align - 1;
    oroError_t err = oroMalloc((oroDeviceptr*)&out.raw, alloc_size);
    if (err != oroSuccess) return err;

    uintptr_t p = reinterpret_cast<uintptr_t>(out.raw);
    uintptr_t pa = align_up_uintptr(p, align);

    out.aligned = reinterpret_cast<uint8_t*>(pa);
    out.raw_size = alloc_size;
    return oroSuccess;
}

static inline void oroFreeAligned(AlignedDevAlloc& a) {
    if (a.raw) oroFree((oroDeviceptr)a.raw);
    a.raw = nullptr;
    a.aligned = nullptr;
    a.raw_size = 0;
}