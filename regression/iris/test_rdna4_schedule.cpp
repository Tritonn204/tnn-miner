// Abstract workgroup event model, not GPU execution or ISA race verification.
#include <array>
#include <stdexcept>
#include <iostream>

struct Bank { unsigned generation = 0, retired = 15; bool published = false; };

void simulate(unsigned banks, unsigned steps, unsigned fault) {
    std::array<Bank, 2> storage{};
    for (unsigned step = 0; step < steps; ++step) {
        auto& bank = storage[step % banks];
        if (fault == 2 && step == banks) bank.retired &= ~8u;
        if (bank.retired != 15) throw std::runtime_error("overwrite before retirement");
        bank.generation = step;
        bank.published = !(fault == 1 && step == 0);
        bank.retired = 0;
        // Each wave may arrive in a different order. All must see publication.
        for (unsigned index = 0; index < 4; ++index) {
            const unsigned wave = (index + step) % 4;
            if (!bank.published || bank.generation != step)
                throw std::runtime_error("read before publication");
            bank.retired |= 1u << wave;
        }
    }
}

int main() try {
    for (unsigned banks : {1u, 2u}) {
        for (unsigned steps : {0u, 1u, 2u, 3u, 15u, 16u, 17u, 33u, 257u})
            simulate(banks, steps, 0);
        for (unsigned fault : {1u, 2u}) {
            bool caught = false;
            try { simulate(banks, 5, fault); }
            catch (const std::runtime_error&) { caught = true; }
            if (!caught) throw std::runtime_error("schedule mutation survived");
        }
    }
    std::cout << "Publication/retirement model passed; both mutations rejected\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
