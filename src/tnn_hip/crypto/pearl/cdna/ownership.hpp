#pragma once

namespace tnn::pearl::gpu::cdna {

// Select a 4x32 Pearl ticket from a CDNA wave's native 8x4 MFMA atoms.
// This reduction is a proof concern, not part of the generic Iris GEMM.
struct Ticket {
    static constexpr unsigned atom_row(unsigned parity, unsigned atom) {
        return atom * 2 + parity;
    }
    static constexpr unsigned source_lane(unsigned lane) {
        return lane % 16 / 4 * 16 + lane % 32 / 16 * 4;
    }
    static constexpr unsigned source_element(unsigned lane) { return lane % 4; }
    static constexpr unsigned row(unsigned lane) { return lane / 32 * 16 + lane % 16; }
    static constexpr unsigned col(unsigned lane) { return lane % 32 / 16 * 4; }
};

} // namespace tnn::pearl::gpu::cdna
