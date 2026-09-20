#pragma once

// Host-only offline validation. No fragment-map or GPU scheduling dependencies.
#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

namespace tnn::pearl::validation {

using DigestWords = std::array<uint32_t, 8>;
using Record = std::array<uint32_t, 10>;

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

inline std::vector<uint32_t> normalize_diagnostics(std::span<const uint32_t> input,
                                                  unsigned m, unsigned n, bool plane_major) {
    require(m && n && m % 128 == 0 && n % 128 == 0, "Invalid diagnostic dimensions");
    const size_t positions = size_t(m) * n / 128;
    require(input.size() == positions * 24, "Invalid diagnostic size");
    if (!plane_major) return {input.begin(), input.end()};

    std::vector<uint32_t> output(input.size());
    for (unsigned tile = 0; tile < positions / 128; ++tile)
        for (unsigned tid = 0; tid < 128; ++tid) {
            const unsigned row = tile / (n / 128) * 128 + (tid / 32) % 2 * 16 + tid % 16;
            const unsigned col = tile % (n / 128) * 128 + tid / 64 * 64 + tid % 32 / 16 * 4;
            const size_t source = (row / 128 * 32 + row % 128) * (n / 32) + col / 64 * 2 + col % 64 / 4;
            require(source < positions, "Diagnostic coordinate outside allocation");
            for (unsigned word = 0; word < 24; ++word)
                output[(tile * 128 + tid) * 24 + word] = input[word * positions + source];
        }
    return output;
}

inline bool meets_target(const DigestWords& digest, const DigestWords& target) {
    // Compare the mathematical LE integer, independently of the device's
    // lower/equal accumulation. Equality is accepted by the Pearl contract.
    for (unsigned i = 8; i-- > 0;) {
        if (digest[i] != target[i]) return digest[i] < target[i];
    }
    return true;
}

template<class Winner>
Record record(const Winner& winner) {
    Record result{winner.row, winner.col};
    std::copy_n(winner.digest, 8, result.begin() + 2);
    return result;
}

template<class Winner>
void check_winners(std::span<const Record> expected, std::span<const Winner> actual,
                   uint32_t total_hits, uint32_t overflow, unsigned capacity) {
    require(total_hits == expected.size(), "Winner total mismatch");
    require(overflow == unsigned(expected.size() > capacity), "Winner overflow mismatch");
    require(actual.size() == std::min<size_t>(expected.size(), capacity), "Stored winner count mismatch");

    const std::set<Record> allowed(expected.begin(), expected.end());
    require(allowed.size() == expected.size(), "Duplicate oracle winner");
    std::set<Record> seen;
    std::set<std::pair<uint32_t, uint32_t>> coordinates;
    for (const auto& winner : actual) {
        const auto item = record(winner);
        require(allowed.contains(item), "Unexpected winner or digest");
        require(seen.insert(item).second && coordinates.emplace(item[0], item[1]).second,
                "Duplicate returned winner");
    }
    // During overflow the atomic winner order is unspecified. Any unique
    // subset is valid, but the total hit count must still be exact.
    if (!overflow) require(seen == allowed, "Missing winner");
}

} // namespace tnn::pearl::validation
