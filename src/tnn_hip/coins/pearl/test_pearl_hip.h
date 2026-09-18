#pragma once
#include <cstdint>

namespace tnn::pearl {

int test_pearl_hip(bool experimental_gfx12 = false, unsigned recipe = 0);
int tune_pearl_hip();
int bench_pearl_hip(uint32_t m = 8192, uint32_t n = 8192, uint32_t k = 4096,
                    uint32_t seconds = 5, bool experimental_gfx12 = false,
                    unsigned recipe = 0, unsigned workload = 0);

} // namespace tnn::pearl
