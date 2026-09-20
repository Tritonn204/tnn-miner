#include <tnn_hip/crypto/pearl/pearl_target.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>

using boost::multiprecision::cpp_int;
using Target = std::array<uint8_t, 32>;

cpp_int integer(const Target& bytes) {
    cpp_int result = 0;
    for (int i = 31; i >= 0; --i) result = (result << 8) + bytes[i];
    return result;
}

template<class F> void rejects(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}

Target read(const std::string& path) {
    Target result{};
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(result.data()), result.size());
    assert(stream.gcount() == result.size());
    return result;
}

int main(int argc, char** argv) {
    using namespace tnn::pearl;
    for (uint32_t k : {2048u, 4096u, 8192u}) {
        for (uint32_t rank : {128u, 256u}) {
            if (k < rank * 16) continue;
            assert(jackpot_work(2, 64, k, rank) == uint64_t(128) * (k / rank) * 128);
        }
    }
    assert(jackpot_work(2, 64, 4096, 128) == 524288);
    assert(jackpot_work(2, 64, 2048, 128) == 262144);
    assert(jackpot_work(2, 64, 4097, 128) == 524288);
    rejects([] { jackpot_work(0, 64, 4096, 128); });
    rejects([] { jackpot_work(2, 64, 4096, 0); });
    rejects([] { jackpot_work(2, 64, 4096, 64); });
    rejects([] { jackpot_work(2, 64, 2048, 256); });
    rejects([] { jackpot_work(UINT32_MAX, UINT32_MAX, UINT32_MAX, 128); });
    rejects([] { scale_jackpot_target(Target{}, 1); });

    Target one{};
    one[0] = 1;
    rejects([&] { scale_jackpot_target(one, 0); });
    assert(integer(scale_jackpot_target(one, 524288)) == 524288);
    const cpp_int maximum = (cpp_int(1) << 256) - 1;
    std::mt19937_64 random(0x504541524c);
    for (unsigned iteration = 0; iteration < 4000; ++iteration) {
        Target input{};
        for (auto& byte : input) byte = uint8_t(random());
        uint64_t factor = iteration % 2 ? random() : 524288;
        if (iteration % 3 == 0) {
            for (unsigned i = 24; i < 32; ++i) input[i] = 0;
        }
        cpp_int expected = integer(input) * factor;
        if (expected > maximum) rejects([&] { scale_jackpot_target(input, factor); });
        else assert(integer(scale_jackpot_target(input, factor)) == expected);
    }
    Target safe{};
    safe.fill(255);
    safe[31] = safe[30] = 0;
    safe[29] = 31;
    assert(integer(scale_jackpot_target(safe, 524288)) == integer(safe) * 524288);
    ++safe[29];
    rejects([&] { scale_jackpot_target(safe, 524288); });

    // Optional replay of retained live receipts; no network or GPU involved.
    if (argc == 2) {
        unsigned rejected_raw = 0;
        for (unsigned i = 1; i <= 6; ++i) {
            const std::string base = std::string(argv[1]) + "/candidate-" + std::to_string(i);
            const auto raw = read(base + ".raw");
            const auto bound = scale_jackpot_target(raw, jackpot_work(2, 64, 4096, 128));
            const auto digest = read(base + ".digest");
            // Controls/near-boundary probes deliberately used a tighter 2*T
            // study bound; only the high-range probes used the full bound.
            const auto probe_bound = read(base + ".bound");
            assert(integer(probe_bound) <= integer(bound));
            if (i % 3 == 0) assert(probe_bound == bound);
            assert(integer(digest) <= integer(bound));
            rejected_raw += integer(digest) > integer(raw);
        }
        assert(rejected_raw == 4);
        std::cout << "Six live receipts pass; four fail the original unscaled bound.\n";
    }
    std::cout << "Pearl target tests passed.\n";
}
