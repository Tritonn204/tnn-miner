#pragma once

namespace tnn::hip::iris::gemm::rdna4 {

// RDNA4 wave32 v_wmma_i32_16x16x16_iu8. Register indices are logical
// fragment offsets, not a prescription for physical VGPR placement.
struct I8Atom {
    static constexpr unsigned wave_size = 32;
    static constexpr unsigned input_bytes = 8;
    static constexpr unsigned outputs = 8;

    static constexpr unsigned input_outer(unsigned lane) { return lane % 16; }
    static constexpr unsigned input_k(unsigned lane, unsigned byte) {
        return lane / 16 * 8 + byte;
    }
    static constexpr unsigned output_row(unsigned lane, unsigned element) {
        return lane / 16 * 8 + element;
    }
    static constexpr unsigned output_col(unsigned lane) { return lane % 16; }
};

// Four waves partition a 128-square tile. Interleaved M atoms leave the
// consumer free to reduce rows separated by 32 without crossing wave boundaries.
struct Tile128 {
    static constexpr unsigned threads = 128;
    static constexpr unsigned row(unsigned wave, unsigned atom, unsigned lane, unsigned e) {
        return wave % 2 * 16 + atom * 32 + I8Atom::output_row(lane, e);
    }
    static constexpr unsigned col(unsigned wave, unsigned atom, unsigned lane) {
        return wave / 2 * 64 + atom * 16 + I8Atom::output_col(lane);
    }
};

template<unsigned Step, unsigned Banks, unsigned LoadBytes> struct Recipe {
    static_assert(Step == 32 || Step == 64);
    static_assert(Banks == 1 || Banks == 2);
    static_assert(LoadBytes == 8 || LoadBytes == 16);
    static constexpr unsigned step = Step;
    static constexpr unsigned banks = Banks;
    static constexpr unsigned load_bytes = LoadBytes;
    static constexpr unsigned operand_bytes = 128 * Step;
    static constexpr unsigned lds_bytes = Banks * 2 * operand_bytes;
};

} // namespace tnn::hip::iris::gemm::rdna4
