#pragma once

#include "../atom/rdna3_wmma_i8_16x16x16.hpp"
#include "../layout/rdna3_b_crosswise.hpp"
#include "../thread_map/rdna3_b_crosswise_map.hpp"
#include "../collective/dense_serial_mainloop.hpp"
#include "../config/gfx1100_i8_defaults.hpp"
#include "../kernel/gemm_dense_wmma_i8.hpp"

namespace tnn::hip::iris::gemm::arch::gfx1100 {

struct Backend {
    using WmmaI8 = atom::Rdna3WmmaI8_16x16x16<true, true, false>;
    using BThreadMap = thread_map::Rdna3BCrosswiseMap;
    using DenseSerialMainloop = collective::DenseSerialMainloop;
    using DefaultConfig = config::Gfx1100I8Mt64x64x32Default;
    using KernelTag = kernel::GemmDenseWmmaI8Tag;
    static constexpr int kBStride = DefaultConfig::kBStride;
};

template <int Rank, int TileRows, class Storage>
IRIS_DEVICE_INLINE auto make_b_tile(Storage& storage, int group) {
    return layout::make_rdna3_b_crosswise_tile<Rank, TileRows, Backend::kBStride>(storage, group);
}

} // namespace tnn::hip::iris::gemm::arch::gfx1100
