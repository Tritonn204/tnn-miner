#include <tnn_hip/crypto/iris/gemm/rdna4/contract.hpp>
#include <tnn_hip/crypto/pearl/rdna4/ownership.hpp>
#include <array>
#include <cassert>
#include <cstdint>

#ifdef NDEBUG
#error "RDNA4 correctness assertions must remain enabled in Release builds"
#endif

namespace gemm = tnn::hip::iris::gemm::rdna4;
using Ticket = tnn::pearl::gpu::rdna4::Ticket;

int main() {
    std::array<unsigned, 256> input{}, output{};
    for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned e = 0; e < 8; ++e) {
            ++input[gemm::I8Atom::input_outer(lane) * 16 + gemm::I8Atom::input_k(lane, e)];
            ++output[gemm::I8Atom::output_row(lane, e) * 16 + gemm::I8Atom::output_col(lane)];
        }
    for (unsigned i = 0; i < 256; ++i) assert(input[i] == 1 && output[i] == 1);

    std::array<unsigned, 128 * 128> coverage{};
    for (unsigned wave = 0; wave < 4; ++wave)
        for (unsigned lane = 0; lane < 32; ++lane)
            for (unsigned i = 0; i < 4; ++i)
                for (unsigned j = 0; j < 4; ++j)
                    for (unsigned e = 0; e < 8; ++e)
                        ++coverage[gemm::Tile128::row(wave, i, lane, e) * 128 +
                                   gemm::Tile128::col(wave, j, lane)];
    for (auto count : coverage) assert(count == 1);

    // Feed actual cumulative signed GEMM values into an independent simulation
    // of the wave reduction. Compare with direct ticket-coordinate ownership.
    std::array<int32_t, 128 * 128> matrix{};
    bool lane_mutation_detected = false;
    for (unsigned pattern = 0; pattern < 4; ++pattern) {
        matrix.fill(0);
        for (unsigned checkpoint = 0; checkpoint < 3; ++checkpoint) {
            for (unsigned row = 0; row < 128; ++row)
                for (unsigned col = 0; col < 128; ++col)
                    for (unsigned kk = checkpoint * 128; kk < (checkpoint + 1) * 128; ++kk) {
                        const int a = pattern == 0 ? 0 : pattern == 1 ? -128 :
                                      pattern == 3 ? (row == 17 && kk == 129 ? -127 : 0) :
                                      int((row * 17 + kk * 11) % 256) - 128;
                        const int b = pattern == 0 ? 0 : pattern == 1 ? 127 :
                                      pattern == 3 ? (col == 31 && kk == 129 ? 113 : 0) :
                                      int((col * 13 + kk * 7) % 256) - 128;
                        matrix[row * 128 + col] += a * b;
                    }
            for (unsigned wave = 0; wave < 4; ++wave) {
                std::array<std::array<uint32_t, 8>, 32> values{};
                for (unsigned lane = 0; lane < 32; ++lane)
                    for (unsigned e = 0; e < 8; ++e)
                        for (unsigned i = 0; i < 4; ++i)
                            for (unsigned j = 0; j < 4; ++j)
                                values[lane][e] ^= uint32_t(matrix[
                                    gemm::Tile128::row(wave, i, lane, e) * 128 +
                                    gemm::Tile128::col(wave, j, lane)]);
                for (unsigned mask : {1u, 2u, 8u}) {
                    const auto before = values;
                    for (unsigned lane = 0; lane < 32; ++lane)
                        for (unsigned e = 0; e < 8; ++e)
                            values[lane][e] ^= before[lane ^ mask][e];
                }
                for (unsigned lane = 0; lane < 32; ++lane) {
                    uint32_t expected = 0;
                    for (unsigned r = 0; r < 4; ++r)
                        for (unsigned c = 0; c < 32; ++c)
                            expected ^= uint32_t(matrix[(Ticket::row(wave, lane) + r * 32) * 128 +
                                                       Ticket::col(wave, lane) + c / 4 * 8 + c % 4]);
                    assert(values[Ticket::source_lane(lane)][Ticket::source_element(lane)] == expected);
                    lane_mutation_detected |=
                        values[Ticket::source_lane(lane)][(Ticket::source_element(lane) + 1) % 8] != expected;
                }
            }
        }
    }
    assert(lane_mutation_detected);
    // Signedness mutation in the sparse case must not agree with the oracle.
    assert(int(int8_t(-127)) * 113 != int(uint8_t(-127)) * 113);
    static_assert(gemm::Recipe<32, 1, 8>::lds_bytes == 8192);
    static_assert(gemm::Recipe<64, 2, 16>::lds_bytes + 8192 <= 65536);
}
