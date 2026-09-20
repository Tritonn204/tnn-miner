#pragma once

namespace tnn::hip::iris::gemm::cdna {

// CDNA1/2 use four packed bytes per lane (K=16), CDNA3 uses eight
// (K=32). Both atoms own four adjacent output rows in a wave of 64.
template<unsigned Bytes>
struct I8Atom {
    static_assert(Bytes == 4 || Bytes == 8);
    static constexpr unsigned input_bytes = Bytes;
    static constexpr unsigned depth = Bytes * 4;

    static constexpr unsigned input_outer(unsigned lane) { return lane % 16; }
    static constexpr unsigned input_k(unsigned lane, unsigned byte) {
        return lane / 16 * Bytes + byte;
    }
    static constexpr unsigned output_row(unsigned lane, unsigned element) {
        return lane / 16 * 4 + element;
    }
    static constexpr unsigned output_col(unsigned lane) { return lane % 16; }
};

// Two wave64 groups divide N. Every wave spans all 128 rows, with eight
// M atoms and four N atoms. This is an independent CDNA accumulator map.
struct Tile128 {
    static constexpr unsigned row(unsigned, unsigned atom, unsigned lane, unsigned element) {
        return atom * 16 + lane / 16 * 4 + element;
    }
    static constexpr unsigned col(unsigned wave, unsigned atom, unsigned lane) {
        return wave * 64 + atom * 16 + lane % 16;
    }
};

} // namespace tnn::hip::iris::gemm::cdna
