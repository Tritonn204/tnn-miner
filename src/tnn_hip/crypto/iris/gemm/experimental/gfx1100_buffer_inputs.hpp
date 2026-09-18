#pragma once

#include "../../../../iris/rt/bounded_buffer_load.hpp"

namespace tnn::hip::iris::gemm::experimental {

// Packed A[M,K], B[K,N], 128x128x32 tile, four wave32s. This input mapping
// belongs to the recipe, not the architecture-neutral buffer primitive.
template <class Load = ::iris::hip::BoundedBufferLoad128<::iris::hip::Gfx1100ByteBuffer>>
struct Gfx1100BufferInputs128 {
    using I4 = ::iris::hip::BufferDword4;
    I4 resource_a, resource_b;
    unsigned offset_a[2], offset_b[2];
    unsigned stride_a;

    IRIS_DEVICE_INLINE static I4 resource(const int8_t* pointer, unsigned bytes)
    {
        return ::iris::hip::Gfx1100ByteBuffer::resource(pointer, bytes);
    }

    IRIS_DEVICE_INLINE Gfx1100BufferInputs128(
        const int8_t* a, const int8_t* b, unsigned bm, unsigned bn,
        unsigned m, unsigned n, unsigned k, unsigned lane, unsigned wave)
        : resource_a(resource(a + bm, m * k - bm)),
          resource_b(resource(b + size_t(bn) * k, (n - bn) * k)), stride_a(m)
    {
#pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            offset_a[i] = (lane % 8) * 16 + (wave * 8 + lane / 8 + i * 4) * m;
            offset_b[i] = (lane % 2) * 16 + (wave * 32 + lane / 2 + i * 16) * k;
        }
    }

    IRIS_DEVICE_INLINE I4 a(unsigned tile, unsigned vector) const
    {
        return Load::load(resource_a, offset_a[vector] + tile * 32 * stride_a);
    }

    IRIS_DEVICE_INLINE I4 b(unsigned tile, unsigned vector) const
    {
        return Load::load(resource_b, offset_b[vector] + tile * 32);
    }
};

} // namespace tnn::hip::iris::gemm::experimental
