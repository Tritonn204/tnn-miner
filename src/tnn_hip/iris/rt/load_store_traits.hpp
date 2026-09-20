#pragma once

#include "arch_traits.hpp"
#include "coordinate.hpp"
#include "space_filling_curve.hpp"

namespace iris::hip {

template <typename T, int RequestedVecBytes = arch::vector_bytes, int FastDimElems = RequestedVecBytes / sizeof(T)>
struct LoadStoreTraits {
    static constexpr int requested_vec_bytes = RequestedVecBytes;
    static constexpr int scalar_bytes = sizeof(T);
    static constexpr int max_vec_bytes = RequestedVecBytes < scalar_bytes ? scalar_bytes : RequestedVecBytes;
    static constexpr int max_vec_elems = max_vec_bytes / scalar_bytes;
    static constexpr int vec_elems = FastDimElems < max_vec_elems ? FastDimElems : max_vec_elems;
    static constexpr int vec_bytes = vec_elems * scalar_bytes;
    static constexpr int vector_dim = 1;
    static constexpr bool has_vector_access = vec_elems > 1;
    static constexpr bool requires_tail_guard = true;

    static_assert(vec_bytes % scalar_bytes == 0, "Vector byte width must be a whole number of scalars");

    IRIS_HOST_DEVICE_INLINE static bool is_aligned(const T* ptr) {
        return (reinterpret_cast<uintptr_t>(ptr) & (vec_bytes - 1)) == 0;
    }
};

template <typename Map>
struct MapAccessTraits {
    static constexpr int vec_elems = Map::vec_elems;
    static constexpr int threads = Map::threads;
    static constexpr int num_access = Map::threads;
    static constexpr int vector_dim = 1;
    static constexpr bool has_vector_access = vec_elems > 1;
};

template <typename Map, typename T, int RequestedVecBytes = arch::vector_bytes>
struct TileLoadStoreTraits {
    static_assert(Map::rank == 2, "TileLoadStoreTraits currently expects a 2D map");

    using scalar_type = T;
    using scalar_traits = LoadStoreTraits<T, RequestedVecBytes, Map::vec_elems>;
    using sfc_type = RowMajorSfc2D<Map::tile_rows, Map::tile_cols, Map::vec_elems>;

    static constexpr int vec_elems = scalar_traits::vec_elems;
    static constexpr int vec_bytes = scalar_traits::vec_bytes;
    static constexpr int vector_dim = scalar_traits::vector_dim;
    static constexpr int num_access = sfc_type::accesses;
    static constexpr bool has_vector_access = scalar_traits::has_vector_access;
    static constexpr bool requires_tail_guard = scalar_traits::requires_tail_guard;
};

} // namespace iris::hip
