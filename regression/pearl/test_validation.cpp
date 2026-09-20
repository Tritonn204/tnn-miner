#include <tnn_hip/coins/pearl/pearl_validation.hpp>
#include <tnn_hip/coins/pearl/pearl_arch.hpp>
#include <bit>
#include <iostream>
#include <string>

namespace v = tnn::pearl::validation;

struct Winner { uint32_t row, col, digest[8]; };

template<class F>
void rejects(F operation, const char* reason) {
    try { operation(); }
    catch (const std::runtime_error& error) {
        v::require(std::string(error.what()) == reason, "Wrong validation failure");
        return;
    }
    throw std::runtime_error("Mutation survived validation");
}

int main(int argc, char**) try {
    if (argc > 1) v::require(false, "Deliberate release-check failure");
    for (const char* arch : {"gfx1100", "gfx1101:xnack-", "gfx1102"})
        v::require(tnn::pearl::rdna3_target(arch) && !tnn::pearl::rdna4_target(arch), "RDNA3 routing");
    for (const char* arch : {"gfx1200", "gfx1201"})
        v::require(tnn::pearl::rdna4_target(arch) && !tnn::pearl::rdna3_target(arch), "RDNA4 routing");
    for (const char* arch : {"gfx1103", "gfx1150", "gfx1151", "gfx1202", "gfx11010"})
        v::require(!tnn::pearl::rdna3_target(arch) && !tnn::pearl::rdna4_target(arch), "Unknown RDNA target admitted");

    for (unsigned n : {256u, 768u}) {
        const unsigned m = 256, positions = m * n / 128;
        std::vector<uint32_t> planes(positions * 24), canonical(planes.size());
        for (unsigned row_tile = 0; row_tile < m / 128; ++row_tile)
            for (unsigned row = 0; row < 32; ++row)
                for (unsigned col_group = 0; col_group < n / 64; ++col_group)
                    for (unsigned half = 0; half < 2; ++half)
                        for (unsigned word = 0; word < 24; ++word) {
                            const unsigned r = row_tile * 128 + row, c = col_group * 64 + half * 4;
                            const unsigned source = (row_tile * 32 + row) * (n / 32) + col_group * 2 + half;
                            planes[word * positions + source] = word * 1000000 + r * 1024 + c;
                        }
        for (unsigned tile = 0; tile < positions / 128; ++tile)
            for (unsigned tid = 0; tid < 128; ++tid)
                for (unsigned word = 0; word < 24; ++word) {
                    const unsigned r = tile / (n / 128) * 128 + tid / 32 % 2 * 16 + tid % 16;
                    const unsigned c = tile % (n / 128) * 128 + tid / 64 * 64 + tid % 32 / 16 * 4;
                    canonical[(tile * 128 + tid) * 24 + word] = word * 1000000 + r * 1024 + c;
                }
        v::require(v::normalize_diagnostics(planes, m, n, true) == canonical, "Plane diagnostic normalization");
        v::require(v::normalize_diagnostics(canonical, m, n, false) == canonical, "Tile diagnostic normalization");
        planes[0] ^= 1;
        v::require(v::normalize_diagnostics(planes, m, n, true) != canonical, "Diagnostic mutation survived");
    }
    for (const char* name : {"gfx908", "gfx90a:xnack-", "gfx942:sramecc+:xnack-"})
        v::require(tnn::pearl::cdna_target(name), "Missing exact CDNA target");
    for (const char* name : {"gfx90", "gfx90abc", "gfx950", "gfx940", "gfx941", "gfx900"})
        v::require(!tnn::pearl::cdna_target(name), "CDNA architecture prefix admitted");

    for (int gfx : {900, 906, 1010, 1011, 1012, 1030, 1031, 1032, 1033, 1034})
        v::require(tnn::pearl::legacy_mining_target(gfx), "Missing legacy target");
    for (int gfx : {902, 904, 909, 1013, 1035, 1036, 1100, 1101, 1201, 908, 942, 950})
        v::require(!tnn::pearl::legacy_mining_target(gfx), "Unqualified legacy target admitted");
    for (int gfx : {900, 1010, 1011, 1012})
        v::require(tnn::pearl::portable_recipe(gfx) == 0, "Unsupported dot4 selected");
    for (int gfx : {906, 1030, 1031, 1032, 1033, 1034})
        v::require(tnn::pearl::portable_recipe(gfx) == 2, "Missing supported dot4 recipe");

    std::vector<Winner> actual{{17, 31, {3}}, {18, 31, {7}}};
    std::vector<v::Record> expected{v::record(actual[0]), v::record(actual[1])};
    auto check = [&](uint32_t total, uint32_t overflow, unsigned capacity) {
        v::check_winners<Winner>(expected, actual, total, overflow, capacity);
    };
    check(2, 0, 2);
    std::reverse(actual.begin(), actual.end());
    check(2, 0, 2);
    rejects([&] { check(1, 0, 2); }, "Winner total mismatch");
    rejects([&] { check(2, 1, 2); }, "Winner overflow mismatch");
    actual[1] = actual[0];
    rejects([&] { check(2, 0, 2); }, "Duplicate returned winner");
    actual[1] = {17, 31, {3}};
    actual[1].digest[0] ^= 1;
    rejects([&] { check(2, 0, 2); }, "Unexpected winner or digest");
    actual.resize(1);
    rejects([&] { check(2, 0, 2); }, "Stored winner count mismatch");
    check(2, 1, 1);
    actual.clear();
    check(2, 1, 0);

    v::DigestWords digest{}, bound{};
    digest[0] = 255; bound[1] = 1;
    v::require(v::meets_target(digest, bound), "LE target order");
    v::require(!v::meets_target(bound, digest), "Target order mutation survived");
    v::require(v::meets_target(bound, bound), "Target equality");
    auto swapped = bound;
    std::reverse(swapped.begin(), swapped.end());
    v::require(v::meets_target(swapped, bound) != v::meets_target(bound, bound), "Endian mutation survived");

    // Independent rotate definition and linear history versus circular slots.
    // Values identify both checkpoint and candidate; no symmetric constants.
    for (unsigned count : {15u, 16u, 17u, 31u, 32u, 33u}) {
        std::array<uint32_t, 16> slots{}, shifted{}, bad_rotation{}, reference{};
        std::vector<uint32_t> history;
        for (unsigned i = 0; i < count; ++i) {
            const uint32_t value = (i + 1) * 0x9e3779b9u;
            history.push_back(value);
            slots[i % 16] = std::rotl(slots[i % 16], 13) ^ value;
            shifted[(i + 1) % 16] = std::rotl(shifted[(i + 1) % 16], 13) ^ value;
            bad_rotation[i % 16] = std::rotl(bad_rotation[i % 16], 12) ^ value;
        }
        for (unsigned slot = 0; slot < 16; ++slot)
            for (unsigned i = slot; i < count; i += 16) {
                auto x = reference[slot];
                reference[slot] = ((x << 13) | (x >> 19)) ^ history[i];
            }
        v::require(slots == reference, "Checkpoint wrap mismatch");
        v::require(shifted != reference, "Checkpoint index mutation survived");
        if (count > 16) v::require(bad_rotation != reference, "Rotation mutation survived");
    }
    std::cout << "Winner-set, overflow, target and checkpoint mutations rejected\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
