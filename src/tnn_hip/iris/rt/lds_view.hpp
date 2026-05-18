#pragma once

#include "tensor_desc.hpp"

namespace iris::hip {

IRIS_DEVICE_INLINE void block_sync() {
    __syncthreads();
}

struct LdsNoSwizzle {
    IRIS_HOST_DEVICE_INLINE static constexpr int col(int, int j) {
        return j;
    }
};

template <int KPerBlock, int KPack, int MLdsLayer = 2>
struct LdsXorSwizzle {
    static_assert(KPerBlock > 0, "KPerBlock must be positive");
    static_assert(KPack > 0, "KPack must be positive");
    static_assert(MLdsLayer > 0, "MLdsLayer must be positive");
    static_assert(KPerBlock % KPack == 0, "KPerBlock must be divisible by KPack");

    IRIS_HOST_DEVICE_INLINE static constexpr int col(int row, int j) {
        constexpr int cols_per_pack = KPerBlock / KPack;
        const int layer = row % MLdsLayer;
        return (((j / KPack) ^ layer) % cols_per_pack) * KPack + (j % KPack);
    }
};

template <int BankCount = 32, int BankBytes = 4>
struct LdsBankMap {
    static_assert(BankCount > 0, "BankCount must be positive");
    static_assert(BankBytes > 0, "BankBytes must be positive");

    IRIS_HOST_DEVICE_INLINE static constexpr int bank_from_byte_offset(int byte_offset) {
        return (byte_offset / BankBytes) % BankCount;
    }
};

template <typename T, int Rows, int Cols, int PadCols = 0, typename Swizzle = LdsNoSwizzle>
struct LdsTile2D {
    static_assert(Rows > 0, "Rows must be positive");
    static_assert(Cols > 0, "Cols must be positive");
    static_assert(PadCols >= 0, "PadCols must be non-negative");

    static constexpr int rows = Rows;
    static constexpr int cols = Cols;
    static constexpr int stride = Cols + PadCols;

    T* ptr;

    IRIS_HOST_DEVICE_INLINE int offset(int i, int j) const {
        return i * stride + Swizzle::col(i, j);
    }

    IRIS_DEVICE_INLINE T& ref(int i, int j) const {
        return ptr[offset(i, j)];
    }

    IRIS_DEVICE_INLINE T load(int i, int j) const {
        return ptr[offset(i, j)];
    }

    IRIS_DEVICE_INLINE void store(int i, int j, T v) const {
        ptr[offset(i, j)] = v;
    }

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int i, int j) const {
        return i >= 0 && i < Rows && j >= 0 && j < Cols;
    }

    IRIS_HOST_DEVICE_INLINE TensorDesc2D<T> desc() const {
        return {Rows, Cols, stride, 1};
    }
};

template <typename T, int Rows, int Cols, int PadCols = 0, typename Swizzle = LdsNoSwizzle>
IRIS_DEVICE_INLINE LdsTile2D<T, Rows, Cols, PadCols, Swizzle> make_lds_tile(T* ptr) {
    return {ptr};
}

} // namespace iris::hip
