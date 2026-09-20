#pragma once

#include "tensor_desc.hpp"
#include "transform.hpp"

namespace iris::hip {

template <typename Desc>
struct TransposeDesc2D {
    using data_type = typename Desc::data_type;

    Desc desc;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i, int j) const {
        return i >= 0 && i < extent0() && j >= 0 && j < extent1();
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(Coord2D c) const {
        return in_bounds(c.i, c.j);
    }

    IRIS_HOST_DEVICE_INLINE int offset(int i, int j) const {
        return desc.offset(j, i);
    }

    IRIS_HOST_DEVICE_INLINE int offset(Coord2D c) const {
        return offset(c.i, c.j);
    }

    IRIS_HOST_DEVICE_INLINE int extent0() const {
        return desc.extent1();
    }

    IRIS_HOST_DEVICE_INLINE int extent1() const {
        return desc.extent0();
    }
};

template <typename Desc>
struct SliceDesc2D {
    using data_type = typename Desc::data_type;

    Desc desc;
    int origin_i;
    int origin_j;
    int rows;
    int cols;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i, int j) const {
        return i >= 0 && i < rows && j >= 0 && j < cols;
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(Coord2D c) const {
        return in_bounds(c.i, c.j);
    }

    IRIS_HOST_DEVICE_INLINE int offset(int i, int j) const {
        return desc.offset(i + origin_i, j + origin_j);
    }

    IRIS_HOST_DEVICE_INLINE int offset(Coord2D c) const {
        return offset(c.i, c.j);
    }

    IRIS_HOST_DEVICE_INLINE int extent0() const {
        return rows;
    }

    IRIS_HOST_DEVICE_INLINE int extent1() const {
        return cols;
    }
};

template <typename Desc>
struct XorDesc2D {
    using data_type = typename Desc::data_type;

    Desc desc;
    int k_pack;
    int m_lds_layer;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i, int j) const {
        return desc.in_bounds(i, j);
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(Coord2D c) const {
        return in_bounds(c.i, c.j);
    }

    IRIS_HOST_DEVICE_INLINE int offset(int i, int j) const {
        const int layer = i % m_lds_layer;
        const int swizzled_j = ((j / k_pack) ^ layer) * k_pack + (j % k_pack);
        return desc.offset(i, swizzled_j);
    }

    IRIS_HOST_DEVICE_INLINE int offset(Coord2D c) const {
        return offset(c.i, c.j);
    }

    IRIS_HOST_DEVICE_INLINE int extent0() const {
        return desc.extent0();
    }

    IRIS_HOST_DEVICE_INLINE int extent1() const {
        return desc.extent1();
    }
};

template <typename Desc>
IRIS_HOST_DEVICE_INLINE TransposeDesc2D<Desc> make_transposed_desc(Desc desc) {
    return {desc};
}

template <typename Desc>
IRIS_HOST_DEVICE_INLINE SliceDesc2D<Desc> make_slice_desc(Desc desc, int origin_i, int origin_j, int rows, int cols) {
    return {desc, origin_i, origin_j, rows, cols};
}

template <typename Desc>
IRIS_HOST_DEVICE_INLINE XorDesc2D<Desc> make_xor_desc(Desc desc, int k_pack, int m_lds_layer) {
    return {desc, k_pack, m_lds_layer};
}

} // namespace iris::hip
