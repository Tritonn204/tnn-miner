#pragma once

#include "coordinate.hpp"

namespace iris::hip {

template <int Rows, int Cols, int VecElems = 1>
struct RowMajorSfc2D {
    static_assert(Rows > 0, "Rows must be positive");
    static_assert(Cols > 0, "Cols must be positive");
    static_assert(VecElems > 0, "VecElems must be positive");
    static_assert(Cols % VecElems == 0, "Cols must be divisible by VecElems");

    static constexpr int rows = Rows;
    static constexpr int cols = Cols;
    static constexpr int vec_elems = VecElems;
    static constexpr int accesses = Rows * (Cols / VecElems);

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access(int idx) {
        constexpr int chunks_per_row = Cols / VecElems;
        return {idx / chunks_per_row, (idx % chunks_per_row) * VecElems, VecElems};
    }
};

template <int Rows, int Cols, int VecElems = 1>
struct ColMajorSfc2D {
    static_assert(Rows > 0, "Rows must be positive");
    static_assert(Cols > 0, "Cols must be positive");
    static_assert(VecElems > 0, "VecElems must be positive");
    static_assert(Rows % VecElems == 0, "Rows must be divisible by VecElems");

    static constexpr int rows = Rows;
    static constexpr int cols = Cols;
    static constexpr int vec_elems = VecElems;
    static constexpr int accesses = Cols * (Rows / VecElems);

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access(int idx) {
        constexpr int chunks_per_col = Rows / VecElems;
        return {(idx % chunks_per_col) * VecElems, idx / chunks_per_col, VecElems};
    }
};

template <int Rows, int Cols, int VecElems = 1>
struct SnakeRowMajorSfc2D {
    static_assert(Rows > 0, "Rows must be positive");
    static_assert(Cols > 0, "Cols must be positive");
    static_assert(VecElems > 0, "VecElems must be positive");
    static_assert(Cols % VecElems == 0, "Cols must be divisible by VecElems");

    static constexpr int rows = Rows;
    static constexpr int cols = Cols;
    static constexpr int vec_elems = VecElems;
    static constexpr int accesses = Rows * (Cols / VecElems);

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access(int idx) {
        constexpr int chunks_per_row = Cols / VecElems;
        const int row = idx / chunks_per_row;
        const int chunk = idx % chunks_per_row;
        const int snake_chunk = (row & 1) ? (chunks_per_row - 1 - chunk) : chunk;
        return {row, snake_chunk * VecElems, VecElems};
    }
};

template <int Bits>
struct Morton2D {
    static_assert(Bits > 0, "Bits must be positive");

    static constexpr int extent = 1 << Bits;
    static constexpr int accesses = extent * extent;

    IRIS_HOST_DEVICE_INLINE static constexpr int encode(int y, int x) {
        int out = 0;
        for (int b = 0; b < Bits; ++b) {
            out |= ((x >> b) & 1) << (2 * b);
            out |= ((y >> b) & 1) << (2 * b + 1);
        }
        return out;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Coord2D decode(int idx) {
        int y = 0;
        int x = 0;
        for (int b = 0; b < Bits; ++b) {
            x |= ((idx >> (2 * b)) & 1) << b;
            y |= ((idx >> (2 * b + 1)) & 1) << b;
        }
        return {y, x};
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access(int idx) {
        const Coord2D c = decode(idx);
        return {c.i, c.j, 1};
    }
};

} // namespace iris::hip
