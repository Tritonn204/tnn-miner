#pragma once

#include "copy.hpp"
#include "tile_distribution.hpp"

namespace iris::hip {

template <typename T, typename Distribution>
struct StaticDistributedTensor {
    using data_type = T;
    using distribution_type = Distribution;
    using fragment_type = Fragment<T, Distribution::values_per_thread>;

    static constexpr int rank = Distribution::rank;
    static constexpr int values_per_thread = Distribution::values_per_thread;
    static constexpr int vec_elems = Distribution::vec_elems;

    fragment_type values;

    IRIS_DEVICE_INLINE void clear(T value = T{}) {
        #pragma unroll
        for (int i = 0; i < values_per_thread; ++i) {
            values[i] = value;
        }
    }

    IRIS_DEVICE_INLINE T& operator[](int i) {
        return values[i];
    }

    IRIS_DEVICE_INLINE const T& operator[](int i) const {
        return values[i];
    }

    IRIS_DEVICE_INLINE T get(int i) const {
        return values[i];
    }

    IRIS_DEVICE_INLINE void set(int i, T value) {
        values[i] = value;
    }

    IRIS_DEVICE_INLINE fragment_type& fragment() {
        return values;
    }

    IRIS_DEVICE_INLINE const fragment_type& fragment() const {
        return values;
    }
};

template <typename T, typename Distribution>
IRIS_DEVICE_INLINE StaticDistributedTensor<T, Distribution> make_static_distributed_tensor() {
    StaticDistributedTensor<T, Distribution> tensor{};
    return tensor;
}

} // namespace iris::hip
