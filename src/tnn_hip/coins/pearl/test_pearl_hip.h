#pragma once
#include <cstdint>

namespace tnn::pearl {

int test_pearl_hip();
int bench_pearl_hip(uint32_t m = 8192, uint32_t n = 8192, uint32_t k = 4096);

} // namespace tnn::pearl
