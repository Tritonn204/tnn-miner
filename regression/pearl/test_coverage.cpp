#include <cstdint>
#include <tnn_hip/crypto/iris/gemm/qualified/recipe.hpp>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

void check(bool ok) { if (!ok) throw std::runtime_error("Candidate coverage mismatch"); }

// Independent translation of pinned upstream PeriodicPattern::offset_is_valid.
bool valid_offset(unsigned offset, const std::array<std::array<unsigned, 2>, 3>& shape) {
    for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
        offset %= (*it)[0] * (*it)[1];
        if (offset >= (*it)[0]) return false;
    }
    return true;
}

int main() {
    using Arch = tnn::hip::iris::gemm::qualified::Gfx1100;
    const std::array<std::array<unsigned, 2>, 3> rows{{{1, 2}, {2, 1}, {2, 1}}};
    const std::array<std::array<unsigned, 2>, 3> cols{{{2, 8}, {32, 8}, {256, 1}}};
    for (unsigned size : {256u, 8192u}) {
        std::set<uint64_t> expected, actual;
        std::vector<uint8_t> cells(size_t(size) * size);
        for (unsigned r = 0; r + 1 < size; ++r)
            if (valid_offset(r, rows))
                for (unsigned c = 0; c + 238 < size; ++c)
                    if (valid_offset(c, cols)) expected.insert(uint64_t(r) * size + c);

        for (unsigned bm = 0; bm < size; bm += 128)
            for (unsigned bn = 0; bn < size; bn += 256)
                for (unsigned tid = 0; tid < 256; ++tid) {
                    const unsigned wave = tid / 32, lane = tid % 32;
                    const unsigned r = bm + Arch::row(wave, lane);
                    const unsigned c = bn + wave / 4 * 16 + lane / 16;
                    check(actual.insert(uint64_t(r) * size + c).second);
                    for (unsigned a = 0; a < 2; ++a)
                        for (unsigned b = 0; b < 64; ++b) {
                            auto& count = cells[size_t(r + a) * size + c + b / 8 * 32 + b % 8 * 2];
                            check(++count == 1);
                        }
                }
        check(expected == actual);
        check(actual.size() == size_t(size) * size / 128);
        for (auto count : cells) check(count == 1);
        std::cout << "COVERAGE_PASS size=" << size << " candidates=" << actual.size()
                  << " missing=0 duplicates=0 uncovered_outputs=0\n";
    }
    check(!valid_offset(1, rows));
    check(!valid_offset(2, cols));
}
