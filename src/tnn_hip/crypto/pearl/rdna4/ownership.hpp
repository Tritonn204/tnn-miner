#pragma once

namespace tnn::pearl::gpu::rdna4 {

// Preserve the native_4x32 proof contract independently of WMMA's fragment
// ownership. A ticket consumes four strided rows and eight groups of four cols.
struct Ticket {
    static constexpr unsigned row(unsigned wave, unsigned lane) {
        return wave % 2 * 16 + lane % 16;
    }
    static constexpr unsigned col(unsigned wave, unsigned lane) {
        return wave / 2 * 64 + lane / 16 * 4;
    }
    static constexpr unsigned source_lane(unsigned lane) {
        return (lane % 16) / 8 * 16 + lane / 16 * 4;
    }
    static constexpr unsigned source_element(unsigned lane) { return lane % 8; }
    static constexpr unsigned row_offset(unsigned i) { return i * 32; }
    static constexpr unsigned col_offset(unsigned i) { return i / 4 * 8 + i % 4; }
};

// 128x64 SIMT: each logical wave owns eight rows, all 64 columns.
// Sixteen lanes per wave emit tickets; the remaining lanes still participate
// in every shuffle. Coordinates retain the native_4x32 proof contract.
struct NarrowTicket {
    static constexpr bool active(unsigned lane) { return lane % 16 < 8; }
    static constexpr unsigned row(unsigned wave, unsigned lane) {
        return wave % 2 * 16 + wave / 2 * 4 + (lane % 8) / 4 * 8 + lane % 4;
    }
    static constexpr unsigned col(unsigned, unsigned lane) { return lane / 16 * 4; }
    static constexpr unsigned source_lane(unsigned lane) {
        return (lane % 8) / 4 * 16 + lane / 16 * 4;
    }
    static constexpr unsigned source_element(unsigned lane) { return lane % 4; }
};

} // namespace tnn::pearl::gpu::rdna4
