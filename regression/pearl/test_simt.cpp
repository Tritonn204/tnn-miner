// Host execution of the production SIMT traversal and LDS staging. This tests
// ownership/barrier logic, not AMD instruction semantics or GPU performance.
#include <array>
#include <barrier>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <set>
#include <utility>
#include <vector>

struct Index { unsigned x = 0; };
thread_local Index threadIdx;
Index blockIdx;
std::barrier sync_point(128);
void host_barrier() { sync_point.arrive_and_wait(); }

int host_dot(int a, int b, int sum, bool) {
    for (unsigned byte = 0; byte < 4; ++byte)
        sum += int(int8_t(unsigned(a) >> (byte * 8))) *
               int(int8_t(unsigned(b) >> (byte * 8)));
    return sum;
}

#define __device__
#define __forceinline__ inline
#define __syncthreads host_barrier
#define __builtin_amdgcn_sdot4 host_dot
#include <tnn_hip/crypto/iris/gemm/simt/kernel.hpp>
#include <tnn_hip/crypto/pearl/rdna4/ownership.hpp>

namespace simt = tnn::hip::iris::gemm::simt;

template<class R> struct Output {
    std::vector<int>& values;
    unsigned m;

    void checkpoint(typename R::Accumulator&, unsigned, unsigned, unsigned) {}

    void finish(typename R::Accumulator& accum, unsigned bm, unsigned bn,
                unsigned lane, unsigned group) {
        for (unsigned i = 0; i < 4; ++i)
            for (unsigned j = 0; j < 4; ++j)
                for (unsigned e = 0; e < R::elements; ++e)
                    values[simt::Tile<R>::row(group, i, lane, e) + bm +
                           size_t(simt::Tile<R>::col(group, j, lane) + bn) * m] = accum[i][j][e];
    }
};

template<class R>
void check() {
    constexpr unsigned m = 256, n = 128, k = 256;
    alignas(16) std::array<signed char, m * k> a;
    alignas(16) std::array<signed char, n * k> b;
    alignas(16) std::array<unsigned char, R::lds_bytes> shared;
    for (size_t i = 0; i < a.size(); ++i) a[i] = int(i * 17 % 255) - 128;
    for (size_t i = 0; i < b.size(); ++i) b[i] = int(i * 13 % 255) - 128;
    std::vector<int> actual(m * n, 0x55555555);

    for (blockIdx.x = 0; blockIdx.x < m / 128 * (n / R::tile_n); ++blockIdx.x) {
        std::vector<std::thread> workers;
        for (unsigned tid = 0; tid < 128; ++tid)
            workers.emplace_back([&, tid] {
                threadIdx.x = tid;
                Output<R> output{actual, m};
                simt::run<R>(a.data(), b.data(), m, n, k, shared.data(), output);
            });
        for (auto& worker : workers) worker.join();
    }

    for (unsigned row = 0; row < m; ++row)
        for (unsigned col = 0; col < n; ++col) {
            int expected = 0;
            for (unsigned depth = 0; depth < k; ++depth)
                expected += int(a[row + depth * m]) * int(b[col * k + depth]);
            assert(actual[row + col * m] == expected);
        }
}

int main() {
    using R = simt::Recipe<64, 2, 16, true, 64>;
    using Ticket = tnn::pearl::gpu::rdna4::NarrowTicket;
    std::set<std::pair<unsigned, unsigned>> tickets;
    for (unsigned bn : {0u, 64u})
        for (unsigned group = 0; group < 4; ++group)
            for (unsigned lane = 0; lane < 32; ++lane) {
                if (!Ticket::active(lane)) continue;
                const unsigned row = Ticket::row(group, lane), col = Ticket::col(group, lane);
                assert(tickets.emplace(row, bn + col).second);
                std::set<std::pair<unsigned, unsigned>> actual, expected;
                for (unsigned i = 0; i < 4; ++i)
                    for (unsigned j = 0; j < 4; ++j)
                        for (unsigned partner : {0u, 1u, 2u, 3u, 8u, 9u, 10u, 11u}) {
                            const unsigned source = Ticket::source_lane(lane) ^ partner;
                            assert(actual.emplace(simt::Tile<R>::row(group, i, source, Ticket::source_element(lane)),
                                                  simt::Tile<R>::col(group, j, source)).second);
                        }
                for (unsigned r = 0; r < 4; ++r)
                    for (unsigned c = 0; c < 32; ++c)
                        expected.emplace(row + r * 32, col + c / 4 * 8 + c % 4);
                assert(actual == expected);
            }
    assert(tickets.size() == 128);
    for (unsigned row = 0; row < 32; ++row)
        for (unsigned col : {0u, 4u, 64u, 68u}) assert(tickets.contains({row, col}));
    check<simt::Recipe<32, 1, 1, false>>();
    check<simt::Recipe<32, 1, 8, false>>();
    check<simt::Recipe<64, 1, 16, false>>();
    check<simt::Recipe<64, 2, 16, false>>();
    check<simt::Recipe<32, 1, 1, true>>();
    check<simt::Recipe<32, 1, 8, true>>();
    check<simt::Recipe<64, 1, 16, true>>();
    check<simt::Recipe<64, 2, 16, true>>();
    check<simt::Recipe<64, 1, 16, false, 64>>();
    check<simt::Recipe<64, 2, 16, false, 64>>();
    check<simt::Recipe<64, 1, 16, true, 64>>();
    check<simt::Recipe<64, 2, 16, true, 64>>();
}
