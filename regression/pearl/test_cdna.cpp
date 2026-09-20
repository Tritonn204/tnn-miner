#include <tnn_hip/crypto/iris/gemm/cdna/contract.hpp>
#include <tnn_hip/crypto/pearl/cdna/ownership.hpp>
#include <tnn_hip/coins/pearl/pearl_validation.hpp>
#include <array>
#include <iostream>

namespace gemm = tnn::hip::iris::gemm::cdna;
using Ticket = tnn::pearl::gpu::cdna::Ticket;
using tnn::pearl::validation::require;

template<unsigned Bytes>
void check_atom() {
    using Atom = gemm::I8Atom<Bytes>;
    std::array<unsigned, 16 * Atom::depth> seen{};
    for (unsigned lane = 0; lane < 64; ++lane)
        for (unsigned byte = 0; byte < Bytes; ++byte) {
            const unsigned row = Atom::input_outer(lane);
            const unsigned k = Atom::input_k(lane, byte);
            require(row < 16 && k < Atom::depth, "CDNA input bounds");
            ++seen[row * Atom::depth + k];
        }
    for (unsigned count : seen) require(count == 1, "CDNA input coverage");
}

bool check_tickets(unsigned pattern, bool mutated) {
    // Scalar matrix values represent cumulative checkpoint accumulators.
    // Keep the oracle purely in matrix coordinates, independent of lane maps.
    std::array<uint32_t, 128 * 128> matrix{};
    for (unsigned row = 0; row < 128; ++row)
        for (unsigned col = 0; col < 128; ++col) {
            uint32_t value = (row * 128 + col + 1) * 0x9e3779b9u;
            value ^= value >> 15; value *= 0x85ebca6bu; value ^= value >> 13;
            matrix[row * 128 + col] = pattern == 0 ? value :
                pattern == 1 ? uint32_t(int(row) - int(col)) :
                uint32_t(row == 17 && col == 31 ? -14351 : 0);
        }

    std::array<unsigned, 128 * 128> covered{};
    for (unsigned wave = 0; wave < 2; ++wave) {
        uint32_t acc[64][8][4][4]{};
        for (unsigned lane = 0; lane < 64; ++lane)
            for (unsigned i = 0; i < 8; ++i)
                for (unsigned j = 0; j < 4; ++j)
                    for (unsigned e = 0; e < 4; ++e) {
                        const unsigned row = gemm::Tile128::row(wave, i, lane, e);
                        const unsigned col = gemm::Tile128::col(wave, j, lane);
                        require(row < 128 && col < 128, "CDNA output bounds");
                        ++covered[row * 128 + col];
                        acc[lane][i][j][e] = matrix[row * 128 + col];
                    }

        uint32_t actual[64]{};
        for (unsigned parity = 0; parity < 2; ++parity)
            for (unsigned e = 0; e < 4; ++e) {
                std::array<uint32_t, 64> values{};
                for (unsigned lane = 0; lane < 64; ++lane)
                    for (unsigned i = 0; i < 4; ++i)
                        for (unsigned j = 0; j < 4; ++j)
                            values[lane] ^= acc[lane][Ticket::atom_row(parity, i)][j][e];
                for (unsigned mask : {1u, 2u, 8u}) {
                    const auto previous = values;
                    for (unsigned lane = 0; lane < 64; ++lane)
                        values[lane] ^= previous[lane ^ mask];
                }
                for (unsigned lane = 0; lane < 64; ++lane)
                    if (parity == lane / 32 && e == Ticket::source_element(lane))
                        actual[lane] = values[Ticket::source_lane(lane) ^ (mutated ? 16u : 0u)];
            }

        for (unsigned lane = 0; lane < 64; ++lane) {
            const unsigned row = lane / 32 * 16 + lane % 16;
            const unsigned col = wave * 64 + lane % 32 / 16 * 4;
            uint32_t expected = 0;
            for (unsigned r = 0; r < 4; ++r)
                for (unsigned c = 0; c < 32; ++c)
                    expected ^= matrix[(row + r * 32) * 128 + col + c / 4 * 8 + c % 4];
            if (actual[lane] != expected) return false;
        }
    }
    for (unsigned count : covered) require(count == 1, "CDNA output coverage");
    return true;
}

template<unsigned Bytes>
void print_mapping() {
    using Atom = gemm::I8Atom<Bytes>;
    for (char matrix : {'A', 'B', 'D'})
        for (unsigned lane = 0; lane < 64; ++lane)
            for (unsigned e = 0; e < (matrix == 'D' ? 4 : Bytes); ++e) {
                unsigned row = Atom::input_outer(lane), col = Atom::input_k(lane, e);
                if (matrix == 'B') std::swap(row, col);
                if (matrix == 'D') { row = Atom::output_row(lane, e); col = Atom::output_col(lane); }
                std::cout << Bytes << ',' << matrix << ',' << lane << ',' << e << ',' << row << ',' << col << '\n';
            }
}

int main(int argc, char**) try {
    if (argc > 1) {
        print_mapping<4>();
        print_mapping<8>();
        return 0;
    }
    check_atom<4>();
    check_atom<8>();
    for (unsigned pattern = 0; pattern < 3; ++pattern)
        require(check_tickets(pattern, false), "CDNA checkpoint ownership mismatch");
    // Select a different row quartet, not a lane within the same XOR group.
    require(!check_tickets(0, true), "Wrong CDNA source lane survived validation");
    std::cout << "CDNA atom coverage and complete Pearl ticket reduction passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
