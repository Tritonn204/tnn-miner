#include <hip/hip_runtime.h>
#include <cstdint>
#include "recipe.hpp"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx90a__)
#error "Experimental MFMA prototype targets gfx90a only"
#endif

namespace tnn::hip::iris::gemm::cdna_experimental {

using I32x4 = int __attribute__((ext_vector_type(4)));
using Instruction = OneWaveSchedule::Instruction;

// Compile-only prototype, not registered with automatic dispatch. Full 16x16
// tiles only; a future checked wrapper must validate shapes and buffer sizes.
extern "C" __global__ __launch_bounds__(OneWaveSchedule::threads) void iris_cdna2_raw(
    const int8_t *a, const int8_t *b, int32_t *d, unsigned k, unsigned lda, unsigned ldb,
    unsigned ldd) {
    const unsigned lane = threadIdx.x;
    const unsigned row_base = blockIdx.x * Instruction::m;
    const unsigned column_base = blockIdx.y * Instruction::n;
    I32x4 accumulator = {};

    for (unsigned base = 0; base < k; base += Instruction::k) {
        uint32_t packed_a = 0;
        uint32_t packed_b = 0;
#pragma unroll
        for (unsigned byte = 0; byte < Instruction::input_bytes; ++byte) {
            const unsigned inner = base + Instruction::input_k(lane, byte);
            const unsigned outer = Instruction::input_outer(lane);
            packed_a |= uint32_t(uint8_t(a[row_base + outer + size_t(inner) * lda])) << (byte * 8);
            packed_b |= uint32_t(uint8_t(b[inner + size_t(column_base + outer) * ldb]))
                        << (byte * 8);
        }
        accumulator = __builtin_amdgcn_mfma_i32_16x16x16i8(__builtin_bit_cast(int, packed_a),
                                                           __builtin_bit_cast(int, packed_b),
                                                           accumulator, 0, 0, 0);
    }

#pragma unroll
    for (unsigned element = 0; element < Instruction::accumulator_elements; ++element) {
        const unsigned row = row_base + Instruction::output_row(lane, element);
        const unsigned column = column_base + Instruction::output_column(lane);
        d[row + size_t(column) * ldd] = accumulator[element];
    }
}

} // namespace tnn::hip::iris::gemm::cdna_experimental
