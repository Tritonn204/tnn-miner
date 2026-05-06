// Standalone test: prints generated ProgPoW program code
// Compile: g++ -std=c++17 -I src/tnn_hip/crypto/kawpow -I src/crypto/kawpow test_proggen.cpp -o test_proggen

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

#include "kawpow_proggen.hpp"

int main(int argc, char** argv) {
    int block = 0;
    if (argc > 1) block = atoi(argv[1]);

    printf("=== ProgPoW program for block %d (period %d) ===\n\n", block, block / 3);

    std::string prog = kawpow_proggen::generate_program(block);
    printf("%s\n", prog.c_str());

    // Line-based fusion analysis
    size_t pos = 0;
    int fused = 0, unfused = 0;
    while (pos < prog.size()) {
        size_t eol = prog.find('\n', pos);
        if (eol == std::string::npos) eol = prog.size();
        std::string line = prog.substr(pos, eol - pos);

        if (line.find("_m = ") != std::string::npos) unfused++;
        if (line.find("lop3.b32") != std::string::npos) fused++;
        if (line.find("* 33u) +") != std::string::npos) fused++; // v_add3
        // v_xor3: 3-way xor (two ^ mix[ on same line, no lop3)
        {
            auto p1 = line.find("^ mix[");
            if (p1 != std::string::npos) {
                auto p2 = line.find("^ mix[", p1 + 1);
                if (p2 != std::string::npos && line.find("lop3") == std::string::npos)
                    fused++;
            }
        }

        pos = eol + 1;
    }
    printf("\n--- Stats: %d fused, %d unfused (of 18 math ops) ---\n", fused, unfused);

    return 0;
}
