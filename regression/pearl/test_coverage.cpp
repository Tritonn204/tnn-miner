#include <tnn_hip/crypto/iris/gemm/native128/recipe.hpp>
#include <tnn_hip/crypto/pearl/native128/layout.hpp>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(bool ok) {
    if (!ok) throw std::runtime_error("Native128 candidate coverage mismatch");
}

int main() {
    using Arch = tnn::hip::iris::gemm::native128::Gfx1100Wmma;
    for (auto dims : {std::array{256u, 256u}, std::array{384u, 512u},
                      std::array{6144u, 4096u}, std::array{16384u, 16384u}}) {
        const auto [m, n] = dims;
        std::vector<uint8_t> cells(size_t(m) * n);
        std::vector<uint8_t> candidates(size_t(m) * n / 128);
        for (unsigned bm = 0; bm < m; bm += 128) {
            for (unsigned bn = 0; bn < n; bn += 128) {
                for (unsigned tid = 0; tid < 128; ++tid) {
                    const unsigned wave = tid / 32, lane = tid % 32;
                    const unsigned row = bm + Arch::a_row(wave, lane, 0);
                    const unsigned col = bn + wave / 2 * 64 + lane / 16 * 4;
                    // Independent native 4x32 origin constraints.
                    check(row % 128 < 32 && (col % 64 == 0 || col % 64 == 4));
                    const size_t position = (row / 128 * 32 + row % 128) * (n / 32) +
                                            col / 64 * 2 + col % 64 / 4;
                    check(position < candidates.size() && ++candidates[position] == 1);
                    for (unsigned a = 0; a < NativeCandidate::rows; ++a)
                        for (unsigned b = 0; b < NativeCandidate::columns; ++b) {
                            const auto r = row + NativeCandidate::row_offset(a);
                            const auto c = col + NativeCandidate::col_offset(b);
                            check(r < m && c < n && ++cells[size_t(r) * n + c] == 1);
                        }
                }
            }
        }
        check(std::all_of(cells.begin(), cells.end(), [](auto count) { return count == 1; }));
        check(std::all_of(candidates.begin(), candidates.end(), [](auto count) { return count == 1; }));
        std::cout << "COVERAGE_PASS " << m << 'x' << n << " candidates=" << candidates.size()
                  << " missing=0 duplicates=0 uncovered_outputs=0\n";
    }
}
