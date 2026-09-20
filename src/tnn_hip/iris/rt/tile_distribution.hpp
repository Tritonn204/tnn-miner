#pragma once

#include "space_filling_curve.hpp"
#include "thread_map.hpp"

namespace iris::hip {

template <typename Map, typename Sfc = RowMajorSfc2D<Map::tile_rows, Map::tile_cols, Map::vec_elems>>
struct TileDistribution2D {
    using map_type = Map;
    using sfc_type = Sfc;

    static constexpr int rank = 2;
    static constexpr int tile_rows = Map::tile_rows;
    static constexpr int tile_cols = Map::tile_cols;
    static constexpr int threads = Map::threads;
    static constexpr int vec_elems = Map::vec_elems;
    static constexpr int values_per_thread = Map::vec_elems;
    static constexpr int logical_accesses = Sfc::accesses;

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D thread_access(int tid) {
        return Map::access_for_thread(tid);
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access_for_thread(int tid) {
        return thread_access(tid);
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D partition_access(PartitionIndex p) {
        return Map::access_for_partition(p);
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D access_for_partition(PartitionIndex p) {
        return partition_access(p);
    }

    IRIS_HOST_DEVICE_INLINE static constexpr Access2D logical_access(int idx) {
        return Sfc::access(idx);
    }
};

template <typename Map>
using RowMajorTileDistribution2D = TileDistribution2D<
    Map,
    RowMajorSfc2D<Map::tile_rows, Map::tile_cols, Map::vec_elems>>;

template <typename Map>
using ColMajorTileDistribution2D = TileDistribution2D<
    Map,
    ColMajorSfc2D<Map::tile_rows, Map::tile_cols, Map::vec_elems>>;

template <typename Map>
using SnakeRowMajorTileDistribution2D = TileDistribution2D<
    Map,
    SnakeRowMajorSfc2D<Map::tile_rows, Map::tile_cols, Map::vec_elems>>;

} // namespace iris::hip
