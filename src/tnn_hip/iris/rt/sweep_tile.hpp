#pragma once

#include "space_filling_curve.hpp"
#include "static_distributed_tensor.hpp"
#include "tile_window.hpp"

namespace iris::hip {

template <int V>
struct Int {
    static constexpr int value = V;

    IRIS_HOST_DEVICE_INLINE constexpr operator int() const {
        return V;
    }
};

template <int Begin, int End, int Step = 1>
struct StaticFor {
    static_assert(Step > 0, "Step must be positive");

    template <typename Fn>
    IRIS_HOST_DEVICE_INLINE void operator()(Fn&& fn) const {
        if constexpr (Begin < End) {
            fn(Int<Begin>{});
            StaticFor<Begin + Step, End, Step>{}(fn);
        }
    }
};

template <int Rows, int Cols, typename Fn>
IRIS_HOST_DEVICE_INLINE void sweep_tile_indices(Fn&& fn) {
    StaticFor<0, Rows>{}([&](auto i) {
        StaticFor<0, Cols>{}([&](auto j) {
            fn(Coord2D{i, j});
        });
    });
}

template <typename Sfc, typename Fn>
IRIS_HOST_DEVICE_INLINE void sweep_space_filling_curve(Fn&& fn) {
    StaticFor<0, Sfc::accesses>{}([&](auto idx) {
        fn(Sfc::access(idx));
    });
}

template <typename Window, typename Fn>
IRIS_DEVICE_INLINE void sweep_tile(Window& window, Fn&& fn) {
    sweep_tile_indices<Window::tile_rows, Window::tile_cols>([&](Coord2D local) {
        fn(local, window);
    });
}

template <typename Sfc, typename Window, typename Fn>
IRIS_DEVICE_INLINE void sweep_tile(Window& window, Fn&& fn) {
    sweep_space_filling_curve<Sfc>([&](Access2D access) {
        fn(access, window);
    });
}

template <typename Map, typename Fn>
IRIS_HOST_DEVICE_INLINE void sweep_thread_accesses(Fn&& fn) {
    StaticFor<0, Map::threads>{}([&](auto tid) {
        fn(tid, Map::access_for_thread(tid));
    });
}

template <typename Window, typename Fn>
IRIS_DEVICE_INLINE void sweep_thread_tile(Window& window, int tid, Fn&& fn) {
    const Access2D access = window.access(tid);

    #pragma unroll
    for (int x = 0; x < Window::map_type::vec_elems; ++x) {
        const Coord2D coord{access.row, access.col + x};
        if (window.view.in_bounds(coord)) {
            fn(coord, window);
        }
    }
}

template <typename Sfc, int BlockAccesses, typename Fn>
IRIS_HOST_DEVICE_INLINE void sweep_blocks(Fn&& fn) {
    static_assert(BlockAccesses > 0, "BlockAccesses must be positive");
    StaticFor<0, Sfc::accesses, BlockAccesses>{}([&](auto block_start) {
        constexpr int remaining = Sfc::accesses - block_start.value;
        constexpr int block_count = remaining < BlockAccesses ? remaining : BlockAccesses;
        StaticFor<0, block_count>{}([&](auto block_i) {
            constexpr int idx = block_start.value + block_i.value;
            fn(Int<idx>{}, Sfc::access(idx));
        });
    });
}

template <typename T, int Count, typename Fn>
IRIS_DEVICE_INLINE void sweep_fragment(Fragment<T, Count>& frag, Fn&& fn) {
    StaticFor<0, Count>{}([&](auto i) {
        fn(i, frag[i]);
    });
}

template <typename T, int Count, typename Fn>
IRIS_DEVICE_INLINE void sweep_fragment(const Fragment<T, Count>& frag, Fn&& fn) {
    StaticFor<0, Count>{}([&](auto i) {
        fn(i, frag[i]);
    });
}

template <typename T, typename Distribution, typename Fn>
IRIS_DEVICE_INLINE void sweep_register_tile(StaticDistributedTensor<T, Distribution>& tensor, Fn&& fn) {
    StaticFor<0, Distribution::values_per_thread>{}([&](auto i) {
        fn(i, tensor[i]);
    });
}

template <typename T, typename Distribution, typename Fn>
IRIS_DEVICE_INLINE void sweep_register_tile(const StaticDistributedTensor<T, Distribution>& tensor, Fn&& fn) {
    StaticFor<0, Distribution::values_per_thread>{}([&](auto i) {
        fn(i, tensor[i]);
    });
}

} // namespace iris::hip
