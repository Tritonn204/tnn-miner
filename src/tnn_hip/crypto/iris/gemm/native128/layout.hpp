#pragma once

// Architectural ownership, not a universal Pearl shape restriction.
struct NativeCandidate {
    static constexpr unsigned rows = 4;
    static constexpr unsigned columns = 32;
    static constexpr unsigned lanes = 128;

    static constexpr unsigned row_offset(unsigned i) {
        return i * 32;
    }
    static constexpr unsigned col_offset(unsigned i) {
        return (i / 4) * 8 + i % 4;
    }
};

struct NativeTranscript {
    static constexpr unsigned words = 16;
    static constexpr unsigned allocation_bytes = 25600;

    static constexpr unsigned offset(unsigned word, unsigned thread) {
        return (word < 15 ? 8704 + word * 512 : 25088) + thread * 4;
    }
};

static_assert(NativeCandidate::rows * NativeCandidate::columns == 128);
static_assert(NativeTranscript::offset(14, 127) + 4 == 16384);
static_assert(NativeTranscript::offset(15, 127) + 4 == NativeTranscript::allocation_bytes);
