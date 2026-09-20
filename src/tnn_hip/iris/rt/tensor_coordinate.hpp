#pragma once

#include "tensor_desc.hpp"

namespace iris::hip {

struct TensorCoord1D {
    int i;

    IRIS_HOST_DEVICE_INLINE void move(int di) {
        i += di;
    }

    template <typename Desc>
    IRIS_HOST_DEVICE_INLINE int offset(const Desc& desc) const {
        return desc.offset(i);
    }

    template <typename Desc>
    IRIS_HOST_DEVICE_INLINE bool in_bounds(const Desc& desc) const {
        return desc.in_bounds(i);
    }
};

struct TensorCoord2D {
    int i;
    int j;

    IRIS_HOST_DEVICE_INLINE Coord2D coord() const {
        return {i, j};
    }

    IRIS_HOST_DEVICE_INLINE void move(int di, int dj) {
        i += di;
        j += dj;
    }

    IRIS_HOST_DEVICE_INLINE TensorCoord2D moved(int di, int dj) const {
        return {i + di, j + dj};
    }

    template <typename Desc>
    IRIS_HOST_DEVICE_INLINE int offset(const Desc& desc) const {
        return desc.offset(i, j);
    }

    template <typename Desc>
    IRIS_HOST_DEVICE_INLINE bool in_bounds(const Desc& desc) const {
        return desc.in_bounds(i, j);
    }
};

IRIS_HOST_DEVICE_INLINE TensorCoord1D make_tensor_coord(int i) {
    return {i};
}

IRIS_HOST_DEVICE_INLINE TensorCoord2D make_tensor_coord(int i, int j) {
    return {i, j};
}

} // namespace iris::hip
