#pragma once

namespace tnn::hip::iris::gemm::cdna_experimental {

// gfx90a / CDNA2 only. These mappings come from AMD's matrix calculator for
// v_mfma_i32_16x16x16i8 with cbsz=abid=blgp=0.
struct MfmaI8_16x16x16 {
    static constexpr unsigned wave_size = 64;
    static constexpr unsigned m = 16;
    static constexpr unsigned n = 16;
    static constexpr unsigned k = 16;
    static constexpr unsigned input_bytes = 4;
    static constexpr unsigned accumulator_elements = 4;

    static constexpr unsigned input_outer(unsigned lane) {
        return lane % 16;
    }

    static constexpr unsigned input_k(unsigned lane, unsigned byte) {
        return (lane / 16) * 4 + byte;
    }

    static constexpr unsigned output_row(unsigned lane, unsigned element) {
        return (lane / 16) * 4 + element;
    }

    static constexpr unsigned output_column(unsigned lane) {
        return lane % 16;
    }
};

struct OneWaveSchedule {
    using Instruction = MfmaI8_16x16x16;
    static constexpr unsigned threads = Instruction::wave_size;
    static constexpr bool hardware_qualified = false;
};

} // namespace tnn::hip::iris::gemm::cdna_experimental
