#include <stratum/pearl-stratum.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <cmath>
#include <stdexcept>

using namespace tnn::pearl::stratum;

void check(bool condition) {
    if (!condition) throw std::runtime_error("Pearl difficulty regression");
}

int main() {
    auto captured = decode_hex<32>(
        "00000051eb851eb851eb851eb851eb851eb851eb851eb851eb851eb851eb851e");
    std::reverse(captured.begin(), captured.end());
    check(std::abs(share_difficulty(captured) - 200.0) < 1e-10);
    std::reverse(captured.begin(), captured.end());
    check(std::abs(share_difficulty(captured) - 200.0) > 1);

    using boost::multiprecision::cpp_int;
    for (unsigned difficulty : {200u, 20'000u, 500'000u}) {
        cpp_int integer_target = (cpp_int(1) << 238) / difficulty;
        std::array<uint8_t, 32> target{};
        for (auto& byte : target) {
            byte = static_cast<uint8_t>(integer_target & 255);
            integer_target >>= 8;
        }
        check(std::abs(share_difficulty(target) / difficulty - 1) < 1e-12);
    }

    std::array<uint8_t, 32> fractional{};
    fractional[29] = 128; // 2^239, hence difficulty 0.5.
    check(share_difficulty(fractional) == 0.5);
    bool rejected = false;
    try { (void)share_difficulty({}); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected);
    check(authorization_password(true, "d=200") == "x");
    check(authorization_password(false, "d=200") == "d=200");
    std::cout << "PEARL_DIFFICULTY_PASS\n";
}
