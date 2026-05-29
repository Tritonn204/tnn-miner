#pragma once

namespace tnn::hip::iris::gemm::config {

struct Gfx1100I8Mt64x64x32Default {
    static constexpr int kThreadblockM = 64;
    static constexpr int kThreadblockN = 64;
    static constexpr int kThreadblockK = 32;
    static constexpr int kThreads = 128;
    static constexpr int kStages = 1;
    static constexpr int kGlobalLoadBytes = 16;
    static constexpr int kBStride = 48;
};

using Gfx1100I8Sq64x64Default = Gfx1100I8Mt64x64x32Default;

} // namespace tnn::hip::iris::gemm::config
