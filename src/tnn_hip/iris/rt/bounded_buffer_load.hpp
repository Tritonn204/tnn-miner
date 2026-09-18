#pragma once

#include "coordinate.hpp"

namespace iris::hip {

using BufferDword4 = int __attribute__((ext_vector_type(4)));

#if IRIS_AMDGCN_FRONTEND
// Compiler-visible asynchronous memory operation: dependency waits remain the
// compiler's responsibility. A naked inline-assembly load is not equivalent.
extern "C" __device__ BufferDword4 iris_raw_buffer_load_dword4(
    BufferDword4, int, int, int) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
#endif

// Explicit specialization, not a descriptor ABI promise for other architectures.
struct Gfx1100ByteBuffer {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__)
    static constexpr bool target_supported = false;
#else
    static constexpr bool target_supported = true;
#endif
    static constexpr unsigned control = 0x31004000;
    static constexpr unsigned vector_bytes = 16;

    IRIS_HOST_DEVICE static constexpr bool valid_span(
        unsigned long long address, unsigned long long bytes)
    {
        return bytes >= vector_bytes && bytes <= 0xffffffffull &&
               address < (1ull << 48) && bytes <= (1ull << 48) - address &&
               address % vector_bytes == 0;
    }

    IRIS_HOST_DEVICE static constexpr bool contains(unsigned bytes, unsigned offset)
    {
        return bytes >= vector_bytes && offset <= bytes - vector_bytes &&
               offset % vector_bytes == 0;
    }

    IRIS_HOST_DEVICE static inline BufferDword4 resource(
        const int8_t* pointer, unsigned bytes)
    {
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        return BufferDword4{int(unsigned(address)), int(unsigned(address >> 32)),
                            int(bytes), int(control)};
    }
};

// Caller proves span/offset validity before issuing. No hidden per-load branch,
// scalar offset, cache-policy override, or wrapping of out-of-range addresses.
template <class Descriptor>
struct BoundedBufferLoad128 {
    using Resource = BufferDword4;

    IRIS_DEVICE_INLINE static Resource load(Resource resource, unsigned byte_offset)
    {
#if IRIS_AMDGCN_FRONTEND
        static_assert(Descriptor::target_supported, "Descriptor is not qualified for this GPU target");
        return iris_raw_buffer_load_dword4(resource, int(byte_offset), 0, 0);
#else
        // Host tests exercise descriptors and bounds, never fake GPU execution.
        static_assert(sizeof(Descriptor) == 0, "Buffer loads require AMD device compilation");
#endif
    }
};

} // namespace iris::hip
