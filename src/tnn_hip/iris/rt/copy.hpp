#pragma once

#include "thread_map.hpp"
#include "vec_store.hpp"

namespace iris::hip {

template <typename T, int Count>
struct Fragment {
    static_assert(Count > 0, "Fragment count must be positive");

    T data[Count];

    IRIS_DEVICE_INLINE T& operator[](int i) {
        return data[i];
    }

    IRIS_DEVICE_INLINE const T& operator[](int i) const {
        return data[i];
    }
};

template <typename T>
struct IsInt8Type {
    static constexpr bool value = false;
};

template <>
struct IsInt8Type<int8_t> {
    static constexpr bool value = true;
};

template <typename Map, typename View>
IRIS_DEVICE_INLINE auto load_row_major_vector(const View& view, int origin_i, int origin_j, int tid) {
    using T = decltype(view.load(0, 0));
    Fragment<T, Map::vec_elems> frag{};
    const Access2D access = Map::access_for_thread(tid);

    #pragma unroll
    for (int x = 0; x < Map::vec_elems; ++x) {
        const int row = origin_i + access.row;
        const int col = origin_j + access.col + x;
        frag[x] = view.in_bounds(row, col) ? view.load(row, col) : T{};
    }

    return frag;
}

template <typename Map, typename View, typename T>
IRIS_DEVICE_INLINE void store_row_major_vector(const View& view, int origin_i, int origin_j, int tid, const T (&values)[Map::vec_elems]) {
    const Access2D access = Map::access_for_thread(tid);
    const int row = origin_i + access.row;
    const int col = origin_j + access.col;

    if (!view.in_bounds(row, col)) {
        return;
    }

    if constexpr (Map::vec_elems == 32 && IsInt8Type<T>::value) {
        T* ptr = view.ptr_at(row, col);
        if (view.in_bounds(row, col + Map::vec_elems - 1) && VectorAccess<T, 16>::is_aligned(ptr)) {
            store_int8_vec<32>(ptr, values);
            return;
        }
    }

    {
        #pragma unroll
        for (int x = 0; x < Map::vec_elems; ++x) {
            if (view.in_bounds(row, col + x)) {
                view.store(row, col + x, values[x]);
            }
        }
    }
}

template <typename Map, typename View, typename T>
IRIS_DEVICE_INLINE void store_row_major_vector(const View& view, int origin_i, int origin_j, int tid, const Fragment<T, Map::vec_elems>& frag) {
    T values[Map::vec_elems];
    #pragma unroll
    for (int x = 0; x < Map::vec_elems; ++x) {
        values[x] = frag[x];
    }
    store_row_major_vector<Map>(view, origin_i, origin_j, tid, values);
}

template <typename Map, typename SrcView, typename DstView>
IRIS_DEVICE_INLINE void copy_row_major_vector(const SrcView& src, const DstView& dst, int src_i, int src_j, int dst_i, int dst_j, int tid) {
    auto frag = load_row_major_vector<Map>(src, src_i, src_j, tid);
    const Access2D access = Map::access_for_thread(tid);
    const int row = dst_i + access.row;
    const int col = dst_j + access.col;

    #pragma unroll
    for (int x = 0; x < Map::vec_elems; ++x) {
        if (dst.in_bounds(row, col + x)) {
            dst.store(row, col + x, frag[x]);
        }
    }
}

} // namespace iris::hip
