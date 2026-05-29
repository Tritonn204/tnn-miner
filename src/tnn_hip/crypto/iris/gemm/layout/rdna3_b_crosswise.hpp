#pragma once

#include "../iris_include.hpp"

namespace tnn::hip::iris::gemm::layout {

template <int Rank, int TileRows, int Stride = Rank>
struct Rdna3BPackedColTile {
    static constexpr int rows = Rank;
    static constexpr int cols = TileRows;
    static constexpr int stride = Stride;

    signed char* ptr;

    IRIS_HOST_DEVICE_INLINE int offset(int row, int col) const {
        return col * stride + row;
    }

    IRIS_DEVICE_INLINE signed char load(int row, int col) const {
        return ptr[offset(row, col)];
    }

    IRIS_DEVICE_INLINE void store(int row, int col, signed char v) const {
        ptr[offset(row, col)] = v;
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int row, int col) const {
        return row >= 0 && row < rows && col >= 0 && col < cols;
    }
};

template <int Rank, int TileRows, int Stride = Rank, int HalfRows = 16, int HalfPad = 0>
struct Rdna3BHalfPadColTile {
    static constexpr int rows = Rank;
    static constexpr int cols = TileRows;
    static constexpr int stride = Stride;

    signed char* ptr;

    IRIS_HOST_DEVICE_INLINE int offset(int row, int col) const {
        const int half = row / HalfRows;
        const int local_row = row - half * HalfRows;
        return col * stride + half * (HalfRows + HalfPad) + local_row;
    }

    IRIS_DEVICE_INLINE signed char load(int row, int col) const {
        return ptr[offset(row, col)];
    }

    IRIS_DEVICE_INLINE void store(int row, int col, signed char v) const {
        ptr[offset(row, col)] = v;
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int row, int col) const {
        return row >= 0 && row < rows && col >= 0 && col < cols;
    }
};

template <int Rank, int TileRows, int StepRows = 16>
struct Rdna3BStepMajorTile {
    static constexpr int rows = Rank;
    static constexpr int cols = TileRows;
    static constexpr int step_rows = StepRows;

    signed char* ptr;

    IRIS_HOST_DEVICE_INLINE int offset(int row, int col) const {
        const int step = row / StepRows;
        const int local_row = row - step * StepRows;
        return step * (TileRows * StepRows) + col * StepRows + local_row;
    }

    IRIS_DEVICE_INLINE signed char load(int row, int col) const {
        return ptr[offset(row, col)];
    }

    IRIS_DEVICE_INLINE void store(int row, int col, signed char v) const {
        ptr[offset(row, col)] = v;
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int row, int col) const {
        return row >= 0 && row < rows && col >= 0 && col < cols;
    }
};

template <int Rank, int TileRows>
IRIS_HOST_DEVICE_INLINE constexpr int grouped_b_offset(int group, int row, int col) {
    return group * (Rank * TileRows) + row * TileRows + ::iris::hip::LdsXorSwizzle<Rank, 16>::col(row, col);
}

template <int Rank, int TileRows, int Stride = Rank>
IRIS_HOST_DEVICE_INLINE constexpr int rdna3_crosswise_b_offset(int group, int row, int col) {
    return group * (Stride * TileRows) + col * Stride + row;
}

template <int Rank, int TileRows, int Stride = Rank, int HalfRows = 16, int HalfPad = 0>
IRIS_HOST_DEVICE_INLINE constexpr int rdna3_halfpad_b_offset(int group, int row, int col) {
    const int half = row / HalfRows;
    const int local_row = row - half * HalfRows;
    return group * (Stride * TileRows) + col * Stride + half * (HalfRows + HalfPad) + local_row;
}

template <int Rank, int TileRows, int StepRows = 16>
IRIS_HOST_DEVICE_INLINE constexpr int rdna3_stepmajor_b_offset(int group, int row, int col) {
    const int step = row / StepRows;
    const int local_row = row - step * StepRows;
    return group * (Rank * TileRows) + step * (TileRows * StepRows) + col * StepRows + local_row;
}

template <int Rank, int TileRows>
IRIS_HOST_DEVICE_INLINE constexpr int grouped_b_bank(int group, int row, int col) {
    return ::iris::hip::LdsBankMap<>::bank_from_byte_offset(grouped_b_offset<Rank, TileRows>(group, row, col));
}

template <int Rank, int TileRows, int Stride = Rank>
IRIS_HOST_DEVICE_INLINE constexpr int rdna3_crosswise_b_bank(int group, int row, int col) {
    return ::iris::hip::LdsBankMap<>::bank_from_byte_offset(rdna3_crosswise_b_offset<Rank, TileRows, Stride>(group, row, col));
}

template <int Rank, int TileRows, int Stride = Rank, int HalfRows = 16, int HalfPad = 0>
IRIS_HOST_DEVICE_INLINE constexpr int rdna3_halfpad_b_bank(int group, int row, int col) {
    return ::iris::hip::LdsBankMap<>::bank_from_byte_offset(
        rdna3_halfpad_b_offset<Rank, TileRows, Stride, HalfRows, HalfPad>(group, row, col));
}

template <int Rank, int TileRows, int StepRows = 16>
IRIS_HOST_DEVICE_INLINE constexpr int rdna3_stepmajor_b_bank(int group, int row, int col) {
    return ::iris::hip::LdsBankMap<>::bank_from_byte_offset(
        rdna3_stepmajor_b_offset<Rank, TileRows, StepRows>(group, row, col));
}

template <int Rank, int TileRows, int Stride = Rank, class Storage>
IRIS_DEVICE_INLINE auto make_rdna3_b_crosswise_tile(Storage& storage, int group) {
    return Rdna3BPackedColTile<Rank, TileRows, Stride>{storage + group * (Stride * TileRows)};
}

template <int Rank, int TileRows, int Stride = Rank, int HalfRows = 16, int HalfPad = 0, class Storage>
IRIS_DEVICE_INLINE auto make_rdna3_b_halfpad_tile(Storage& storage, int group) {
    return Rdna3BHalfPadColTile<Rank, TileRows, Stride, HalfRows, HalfPad>{
        storage + group * (Stride * TileRows)};
}

template <int Rank, int TileRows, int StepRows = 16, class Storage>
IRIS_DEVICE_INLINE auto make_rdna3_b_stepmajor_tile(Storage& storage, int group) {
    return Rdna3BStepMajorTile<Rank, TileRows, StepRows>{storage + group * (Rank * TileRows)};
}

} // namespace tnn::hip::iris::gemm::layout
