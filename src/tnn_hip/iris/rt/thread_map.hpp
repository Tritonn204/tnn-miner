#pragma once

#include "arch_traits.hpp"
#include "coordinate.hpp"

namespace iris::hip {

IRIS_DEVICE_INLINE PartitionIndex partition_index() {
    const int thread = static_cast<int>(threadIdx.x);
    return {
        static_cast<int>(blockIdx.x),
        thread,
        thread / arch::wave_size,
        thread % arch::wave_size,
    };
}

template <int Threads, int WaveSize = arch::wave_size>
struct BlockThreadLayout {
    static_assert(Threads > 0, "Threads must be positive");
    static_assert(WaveSize > 0, "WaveSize must be positive");
    static_assert(Threads % WaveSize == 0, "Threads must be a whole number of waves");

    static constexpr int threads = Threads;
    static constexpr int wave_size = WaveSize;
    static constexpr int waves = Threads / WaveSize;

    IRIS_HOST_DEVICE_INLINE static constexpr int wave(int tid) {
        return tid / WaveSize;
    }

    IRIS_HOST_DEVICE_INLINE static constexpr int lane(int tid) {
        return tid % WaveSize;
    }
};

template <int TileRows, int TileCols, int Threads, int VecElems>
struct RowMajorCoalesced {
    static_assert(TileRows > 0, "TileRows must be positive");
    static_assert(TileCols > 0, "TileCols must be positive");
    static_assert(Threads > 0, "Threads must be positive");
    static_assert(VecElems > 0, "VecElems must be positive");
    static_assert(TileCols % VecElems == 0, "TileCols must be divisible by VecElems");

    static constexpr int tile_rows = TileRows;
    static constexpr int tile_cols = TileCols;
    static constexpr int rank = 2;
    static constexpr int threads = Threads;
    static constexpr int vec_elems = VecElems;
    static constexpr int threads_per_row = TileCols / VecElems;
    static constexpr int rows_per_block = Threads / threads_per_row;

    static_assert(Threads % threads_per_row == 0, "Threads must cover a whole number of tile rows");

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access_for_thread(int tid) {
        return {
            tid / threads_per_row,
            (tid % threads_per_row) * VecElems,
            VecElems,
        };
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access_for_partition(PartitionIndex p) {
        return access_for_thread(p.thread);
    }
};

template <int TileRows, int TileCols, int Threads, int VecElems>
struct ColMajorCoalesced {
    static_assert(TileRows > 0, "TileRows must be positive");
    static_assert(TileCols > 0, "TileCols must be positive");
    static_assert(Threads > 0, "Threads must be positive");
    static_assert(VecElems > 0, "VecElems must be positive");
    static_assert(TileRows % VecElems == 0, "TileRows must be divisible by VecElems");

    static constexpr int tile_rows = TileRows;
    static constexpr int tile_cols = TileCols;
    static constexpr int rank = 2;
    static constexpr int threads = Threads;
    static constexpr int vec_elems = VecElems;
    static constexpr int threads_per_col = TileRows / VecElems;
    static constexpr int cols_per_block = Threads / threads_per_col;

    static_assert(Threads % threads_per_col == 0, "Threads must cover a whole number of tile columns");

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access_for_thread(int tid) {
        return {
            (tid % threads_per_col) * VecElems,
            tid / threads_per_col,
            VecElems,
        };
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access_for_partition(PartitionIndex p) {
        return access_for_thread(p.thread);
    }
};

template <int Threads, int VecElems>
struct Striped1D {
    static_assert(Threads > 0, "Threads must be positive");
    static_assert(VecElems > 0, "VecElems must be positive");

    static constexpr int threads = Threads;
    static constexpr int vec_elems = VecElems;

    IRIS_HOST_DEVICE_INLINE static constexpr Access1D access_for_thread(int tid) {
        return {tid * VecElems, VecElems};
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Access1D access_for_partition(PartitionIndex p) {
        return access_for_thread(p.thread);
    }
};

} // namespace iris::hip
