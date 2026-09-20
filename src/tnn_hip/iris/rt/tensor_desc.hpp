#pragma once

#include "coordinate.hpp"

namespace iris::hip {

template <typename T>
struct TensorDesc1D {
    int dim0;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i) const {
        return i >= 0 && i < dim0;
    }

    IRIS_HOST_DEVICE_INLINE int offset(int i) const {
        return i;
    }

    IRIS_HOST_DEVICE_INLINE int extent0() const {
        return dim0;
    }

    using data_type = T;
};

template <typename T>
struct TensorDesc2D {
    int dim0;
    int dim1;
    int stride0;
    int stride1;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i, int j) const {
        return i >= 0 && i < dim0 && j >= 0 && j < dim1;
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(Coord2D c) const {
        return in_bounds(c.i, c.j);
    }

    IRIS_HOST_DEVICE_INLINE int offset(int i, int j) const {
        return i * stride0 + j * stride1;
    }

    IRIS_HOST_DEVICE_INLINE int offset(Coord2D c) const {
        return offset(c.i, c.j);
    }

    IRIS_HOST_DEVICE_INLINE int extent0() const {
        return dim0;
    }

    IRIS_HOST_DEVICE_INLINE int extent1() const {
        return dim1;
    }

    using data_type = T;
};

template <typename T>
IRIS_HOST_DEVICE_INLINE TensorDesc1D<T> make_tensor_desc_1d(int n) {
    return {n};
}

template <typename T>
IRIS_HOST_DEVICE_INLINE TensorDesc2D<T> make_tensor_desc_2d(int rows, int cols, int stride0, int stride1) {
    return {rows, cols, stride0, stride1};
}

template <typename T>
IRIS_HOST_DEVICE_INLINE TensorDesc2D<T> make_row_major_desc(int rows, int cols, int ld) {
    return {rows, cols, ld, 1};
}

template <typename T>
IRIS_HOST_DEVICE_INLINE TensorDesc2D<T> make_col_major_desc(int rows, int cols, int ld) {
    return {rows, cols, 1, ld};
}

} // namespace iris::hip
