#pragma once

#include "coordinate.hpp"

namespace iris::hip {

struct PassThroughTransform1D {
    int length;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i) const {
        return i >= 0 && i < length;
    }

    IRIS_HOST_DEVICE_INLINE int lower(int i) const {
        return i;
    }
};

struct OffsetTransform1D {
    int length;
    int offset;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i) const {
        return i >= 0 && i < length;
    }

    IRIS_HOST_DEVICE_INLINE int lower(int i) const {
        return i + offset;
    }
};

struct PadTransform1D {
    int lower_length;
    int left_pad;
    int right_pad;

    IRIS_HOST_DEVICE_INLINE int upper_length() const {
        return lower_length + left_pad + right_pad;
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i) const {
        const int lower_i = lower(i);
        return lower_i >= 0 && lower_i < lower_length;
    }

    IRIS_HOST_DEVICE_INLINE int lower(int i) const {
        return i - left_pad;
    }
};

struct SliceTransform1D {
    int length;
    int start;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i) const {
        return i >= 0 && i < length;
    }

    IRIS_HOST_DEVICE_INLINE int lower(int i) const {
        return i + start;
    }
};

template <int Inner>
struct Merge2Transform {
    static_assert(Inner > 0, "Inner must be positive");

    IRIS_HOST_DEVICE_INLINE static constexpr int lower(int outer, int inner) {
        return outer * Inner + inner;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Coord2D upper(int lower_i) {
        return {lower_i / Inner, lower_i % Inner};
    }
};

template <int Outer, int Inner>
struct Unmerge2Transform {
    static_assert(Outer > 0, "Outer must be positive");
    static_assert(Inner > 0, "Inner must be positive");

    IRIS_HOST_DEVICE_INLINE static constexpr Coord2D upper(int lower_i) {
        return {lower_i / Inner, lower_i % Inner};
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int lower(int outer, int inner) {
        return outer * Inner + inner;
    }
};

template <int KPack, int MLdsLayer = 2>
struct Xor2Transform {
    static_assert(KPack > 0, "KPack must be positive");
    static_assert(MLdsLayer > 0, "MLdsLayer must be positive");

    IRIS_HOST_DEVICE_INLINE static constexpr Coord2D lower(int row, int col) {
        const int layer = row % MLdsLayer;
        return {row, ((col / KPack) ^ layer) * KPack + (col % KPack)};
    }
};

IRIS_HOST_DEVICE_INLINE PassThroughTransform1D make_pass_through_transform(int length) {
    return {length};
}

IRIS_HOST_DEVICE_INLINE OffsetTransform1D make_offset_transform(int length, int offset) {
    return {length, offset};
}

IRIS_HOST_DEVICE_INLINE PadTransform1D make_pad_transform(int lower_length, int left_pad, int right_pad) {
    return {lower_length, left_pad, right_pad};
}

IRIS_HOST_DEVICE_INLINE SliceTransform1D make_slice_transform(int length, int start) {
    return {length, start};
}

} // namespace iris::hip
