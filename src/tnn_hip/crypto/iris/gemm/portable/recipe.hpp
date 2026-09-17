#pragma once

#include <cstdint>

namespace tnn::hip::iris::gemm::portable {

// Logical ticket ownership, not hardware-wave ownership. Works with wave32/64.
struct PearlTicketLayout {
    static constexpr unsigned rows = 128;
    static constexpr unsigned columns = 256;
    static constexpr unsigned threads = 256;
    static constexpr unsigned values_per_thread = 128;

    static constexpr unsigned row(unsigned thread) {
        return ((thread / 32) % 4) * 32 + (thread % 16) * 2;
    }

    static constexpr unsigned ticket_column(unsigned thread) {
        return (thread / 128) * 16 + (thread % 32) / 16;
    }

    static constexpr unsigned column(unsigned thread, unsigned element) {
        return ticket_column(thread) + (element / 8) * 32 + (element % 8) * 2;
    }

    static constexpr unsigned ticket_index(unsigned row, unsigned column, unsigned n) {
        return (row / 2) * (n / 64) + (column / 256) * 4 + ((column % 256) / 16) * 2 + column % 2;
    }
};

struct ConservativeSchedule {
    static constexpr unsigned tile_k = 32;
    static constexpr unsigned rank = 128;
    static constexpr unsigned max_tiles_per_launch = 8;
};

struct SimtPearl {
    using Layout = PearlTicketLayout;
    using Schedule = ConservativeSchedule;
};

} // namespace tnn::hip::iris::gemm::portable
