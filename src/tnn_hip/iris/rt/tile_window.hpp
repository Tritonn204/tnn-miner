#pragma once

#include "copy.hpp"
#include "load_store_traits.hpp"
#include "static_distributed_tensor.hpp"
#include "tensor_coordinate.hpp"
#include "tensor_view.hpp"
#include "tile_distribution.hpp"

namespace iris::hip {

/// 2D tile window into a TensorView.
template <typename View, int TileRows, int TileCols>
struct TileWindow2D {
    using view_type = View;
    static constexpr int tile_rows = TileRows;
    static constexpr int tile_cols = TileCols;
    static constexpr int rank = 2;

    View view;
    TensorCoord2D origin;

    IRIS_DEVICE_INLINE auto load(int ti, int tj) const {
        return view.load(origin.i + ti, origin.j + tj);
    }
    IRIS_DEVICE_INLINE void store(int ti, int tj, decltype(view.load(0,0)) v) const {
        view.store(origin.i + ti, origin.j + tj, v);
    }
    IRIS_DEVICE_INLINE bool in_bounds(int ti, int tj) const {
        return view.in_bounds(origin.i + ti, origin.j + tj);
    }
    IRIS_DEVICE_INLINE void move(int di, int dj) {
        origin.move(di, dj);
    }
};

template <typename View, typename Map>
struct DistributedTileWindow2D {
    using view_type = View;
    using map_type = Map;
    using distribution_type = TileDistribution2D<Map>;
    using data_type = typename View::data_type;
    using traits_type = TileLoadStoreTraits<Map, data_type>;
    static constexpr int tile_rows = Map::tile_rows;
    static constexpr int tile_cols = Map::tile_cols;
    static constexpr int rank = 2;

    View view;
    TensorCoord2D origin;

    IRIS_DEVICE_INLINE Access2D access(int tid) const {
        const Access2D local = Map::access_for_thread(tid);
        return {origin.i + local.row, origin.j + local.col, local.count};
    }

    IRIS_DEVICE_INLINE bool in_bounds(int tid) const {
        const Access2D a = access(tid);
        return view.in_bounds(a.row, a.col);
    }

    template <typename T>
    IRIS_DEVICE_INLINE void store(int tid, const T (&values)[Map::vec_elems]) const {
        store_row_major_vector<Map>(view, origin.i, origin.j, tid, values);
    }

    IRIS_DEVICE_INLINE auto load(int tid) const {
        return load_row_major_vector<Map>(view, origin.i, origin.j, tid);
    }

    template <typename T, typename Distribution>
    IRIS_DEVICE_INLINE void load_into(StaticDistributedTensor<T, Distribution>& tensor, int tid) const {
        const Access2D local = Distribution::thread_access(tid);

        #pragma unroll
        for (int x = 0; x < Distribution::values_per_thread; ++x) {
            const int row = origin.i + local.row;
            const int col = origin.j + local.col + x;
            tensor[x] = view.in_bounds(row, col) ? view.load(row, col) : T{};
        }
    }

    template <typename T, typename Distribution>
    IRIS_DEVICE_INLINE StaticDistributedTensor<T, Distribution> load_tile(int tid) const {
        StaticDistributedTensor<T, Distribution> tensor{};
        load_into(tensor, tid);
        return tensor;
    }

    IRIS_DEVICE_INLINE auto load_tile(int tid) const {
        StaticDistributedTensor<data_type, distribution_type> tensor{};
        load_into(tensor, tid);
        return tensor;
    }

    template <typename T, typename Distribution>
    IRIS_DEVICE_INLINE void store_from(const StaticDistributedTensor<T, Distribution>& tensor, int tid) const {
        const Access2D local = Distribution::thread_access(tid);

        #pragma unroll
        for (int x = 0; x < Distribution::values_per_thread; ++x) {
            const int row = origin.i + local.row;
            const int col = origin.j + local.col + x;
            if (view.in_bounds(row, col)) {
                view.store(row, col, tensor[x]);
            }
        }
    }
};

template <int TileRows, int TileCols, typename View>
IRIS_DEVICE_INLINE TileWindow2D<View, TileRows, TileCols>
make_tile_window(View v, int origin_i, int origin_j) {
    return {v, {origin_i, origin_j}};
}

template <int TileRows, int TileCols, typename View>
IRIS_DEVICE_INLINE TileWindow2D<View, TileRows, TileCols>
make_tile_window(View v, TensorCoord2D origin) {
    return {v, origin};
}

template <typename Map, typename View>
IRIS_DEVICE_INLINE DistributedTileWindow2D<View, Map>
make_distributed_tile_window(View v, int origin_i, int origin_j) {
    return {v, {origin_i, origin_j}};
}

template <typename Map, typename View>
IRIS_DEVICE_INLINE DistributedTileWindow2D<View, Map>
make_distributed_tile_window(View v, TensorCoord2D origin) {
    return {v, origin};
}

} // namespace iris::hip
