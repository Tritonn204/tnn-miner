#pragma once

#include "buffer_view.hpp"
#include "tensor_adaptor.hpp"
#include "tensor_desc.hpp"

namespace iris::hip {

template <typename Buffer, typename Desc>
struct TensorView1D {
    using data_type = typename Buffer::data_type;

    Buffer buffer;
    Desc desc;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i) const {
        return desc.in_bounds(i);
    }

    IRIS_HOST_DEVICE_INLINE int offset(int i) const {
        return desc.offset(i);
    }

    IRIS_HOST_DEVICE_INLINE data_type* ptr_at(int i) const {
        return buffer.ptr_at(offset(i));
    }

    IRIS_DEVICE_INLINE data_type& ref(int i) const {
        return buffer.ref(offset(i));
    }

    IRIS_DEVICE_INLINE data_type load(int i) const {
        return buffer.load(offset(i));
    }

    IRIS_DEVICE_INLINE void store(int i, data_type v) const {
        buffer.store(offset(i), v);
    }
};

template <typename Buffer, typename Desc>
struct TensorView2D {
    using data_type = typename Buffer::data_type;

    Buffer buffer;
    Desc desc;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i, int j) const {
        return desc.in_bounds(i, j);
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(Coord2D c) const {
        return desc.in_bounds(c);
    }

    IRIS_HOST_DEVICE_INLINE int offset(int i, int j) const {
        return desc.offset(i, j);
    }

    IRIS_HOST_DEVICE_INLINE int offset(Coord2D c) const {
        return desc.offset(c);
    }

    IRIS_HOST_DEVICE_INLINE data_type* ptr_at(int i, int j) const {
        return buffer.ptr_at(offset(i, j));
    }

    IRIS_DEVICE_INLINE data_type& ref(int i, int j) const {
        return buffer.ref(offset(i, j));
    }

    IRIS_DEVICE_INLINE data_type load(int i, int j) const {
        return buffer.load(offset(i, j));
    }

    IRIS_DEVICE_INLINE void store(int i, int j, data_type v) const {
        buffer.store(offset(i, j), v);
    }

    IRIS_HOST_DEVICE_INLINE auto subview(int origin_i, int origin_j, int rows, int cols) const {
        return TensorView2D<Buffer, SliceDesc2D<Desc>>{
            buffer,
            make_slice_desc(desc, origin_i, origin_j, rows, cols),
        };
    }

    IRIS_HOST_DEVICE_INLINE auto transposed() const {
        return TensorView2D<Buffer, TransposeDesc2D<Desc>>{
            buffer,
            make_transposed_desc(desc),
        };
    }

    IRIS_HOST_DEVICE_INLINE auto xor_swizzled(int k_pack, int m_lds_layer) const {
        return TensorView2D<Buffer, XorDesc2D<Desc>>{
            buffer,
            make_xor_desc(desc, k_pack, m_lds_layer),
        };
    }
};

template <typename Buffer, typename Desc>
IRIS_HOST_DEVICE_INLINE TensorView1D<Buffer, Desc> make_tensor_view_1d(Buffer buffer, Desc desc) {
    return {buffer, desc};
}

template <typename Buffer, typename Desc>
IRIS_HOST_DEVICE_INLINE TensorView2D<Buffer, Desc> make_tensor_view_2d(Buffer buffer, Desc desc) {
    return {buffer, desc};
}

template <typename T>
IRIS_HOST_DEVICE_INLINE auto make_global_tensor_view_1d(T* ptr, int n) {
    return make_tensor_view_1d(make_global_buffer(ptr, n), make_tensor_desc_1d<T>(n));
}

template <typename T>
IRIS_HOST_DEVICE_INLINE auto make_global_row_major_tensor_view(T* ptr, int rows, int cols, int ld) {
    return make_tensor_view_2d(make_global_buffer(ptr, rows * ld), make_row_major_desc<T>(rows, cols, ld));
}

template <typename T>
IRIS_HOST_DEVICE_INLINE auto make_global_col_major_tensor_view(T* ptr, int rows, int cols, int ld) {
    return make_tensor_view_2d(make_global_buffer(ptr, cols * ld), make_col_major_desc<T>(rows, cols, ld));
}

} // namespace iris::hip
