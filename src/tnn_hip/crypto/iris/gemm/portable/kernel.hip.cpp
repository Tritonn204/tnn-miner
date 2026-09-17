#include "../dispatch.hpp"
#include "recipe.hpp"
#include "../../../blake3-inline.hip.inc"

namespace tnn::hip::iris::gemm {
namespace {

template <Mode Operation, class Recipe = portable::SimtPearl>
__global__ __launch_bounds__(Recipe::Layout::threads) void simt_pearl(Problem p,
                                                                      unsigned first_tile) {
    using Layout = typename Recipe::Layout;
    using Schedule = typename Recipe::Schedule;

    const unsigned thread = threadIdx.x;
    const unsigned tile = first_tile + blockIdx.x;
    const unsigned row_base = (tile % (p.m / Layout::rows)) * Layout::rows;
    const unsigned column_base = (tile / (p.m / Layout::rows)) * Layout::columns;
    const unsigned row = Layout::row(thread);

    __shared__ int8_t shared_a[Layout::rows * Schedule::tile_k];
    __shared__ int8_t shared_b[Layout::columns * Schedule::tile_k];
    int32_t accumulators[2][64] = {};
    uint32_t transcript[16] = {};

    for (unsigned k_base = 0; k_base < p.k; k_base += Schedule::tile_k) {
        for (unsigned index = thread; index < Layout::rows * Schedule::tile_k;
             index += Layout::threads) {
            shared_a[index] = p.a[row_base + index % Layout::rows +
                                  size_t(k_base + index / Layout::rows) * p.lda];
        }
        for (unsigned index = thread; index < Layout::columns * Schedule::tile_k;
             index += Layout::threads) {
            shared_b[index] = p.b[k_base + index % Schedule::tile_k +
                                  size_t(column_base + index / Schedule::tile_k) * p.ldb];
        }
        __syncthreads();

        // One logical owner computes all 128 values in one Pearl ticket.
        // There are no wave shuffles or matrix instructions in this path.
        for (unsigned inner_k = 0; inner_k < Schedule::tile_k; ++inner_k) {
            const int a0 = shared_a[row + inner_k * Layout::rows];
            const int a1 = shared_a[row + 1 + inner_k * Layout::rows];
#pragma unroll
            for (unsigned element = 0; element < 64; ++element) {
                const int b =
                    shared_b[Layout::column(thread, element) * Schedule::tile_k + inner_k];
                accumulators[0][element] += a0 * b;
                accumulators[1][element] += a1 * b;
            }
        }

        if constexpr (Operation != Mode::Raw) {
            if ((k_base + Schedule::tile_k) % Schedule::rank == 0) {
                uint32_t reduction = 0;
#pragma unroll
                for (unsigned element = 0; element < 64; ++element) {
                    reduction ^= uint32_t(accumulators[0][element]);
                    reduction ^= uint32_t(accumulators[1][element]);
                }
                const unsigned word = ((k_base + Schedule::tile_k) / Schedule::rank - 1) % 16;
#pragma unroll
                for (unsigned index = 0; index < 16; ++index) {
                    if (index == word) {
                        transcript[index] =
                            (transcript[index] << 13 | transcript[index] >> 19) ^ reduction;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned element = 0; element < 64; ++element) {
        const unsigned column = column_base + Layout::column(thread, element);
        p.d[row_base + row + size_t(column) * p.ldd] = accumulators[0][element];
        p.d[row_base + row + 1 + size_t(column) * p.ldd] = accumulators[1][element];
    }

    if constexpr (Operation != Mode::Raw) {
        uint32_t digest[8];
#pragma unroll
        for (unsigned index = 0; index < 8; ++index) {
            digest[index] = p.key[index];
        }
        blake3_compress_in_place(digest, reinterpret_cast<const uint8_t *>(transcript), 64, 0,
                                 CHUNK_START | CHUNK_END | ROOT | (1 << 4));
        const unsigned ticket_row = row_base + row;
        const unsigned ticket_column = column_base + Layout::ticket_column(thread);

        if constexpr (Operation == Mode::Diagnostic) {
            const size_t index = Layout::ticket_index(ticket_row, ticket_column, p.n);
            const size_t tickets = size_t(p.m) * p.n / 128;
#pragma unroll
            for (unsigned word = 0; word < 16; ++word) {
                p.diagnostic[word * tickets + index] = transcript[word];
            }
#pragma unroll
            for (unsigned word = 0; word < 8; ++word) {
                p.diagnostic[(word + 16) * tickets + index] = digest[word];
            }
        }

        bool equal = true;
        bool lower = false;
#pragma unroll
        for (int word = 7; word >= 0; --word) {
            lower |= equal && digest[word] < p.target[word];
            equal &= digest[word] == p.target[word];
        }
        if (lower || equal) {
            const unsigned slot = atomicAdd(&p.results.state->total_hits, 1u);
            if (slot < p.results.capacity) {
                auto &winner = p.results.winners[slot];
                winner.row = ticket_row;
                winner.col = ticket_column;
#pragma unroll
                for (unsigned word = 0; word < 8; ++word) {
                    winner.digest[word] = digest[word];
                }
            } else {
                atomicExch(&p.results.state->overflow, 1u);
            }
        }
    }
}

} // namespace

hipError_t launch_portable_range(const Problem &p, Mode mode, hipStream_t stream,
                                 unsigned first_tile, unsigned tile_count) {
    using Layout = portable::SimtPearl::Layout;
    using Schedule = portable::SimtPearl::Schedule;
    if (!valid_problem(p, mode)) {
        return hipErrorInvalidValue;
    }
    const unsigned total = (p.m / Layout::rows) * (p.n / Layout::columns);
    if (tile_count == 0 || tile_count > Schedule::max_tiles_per_launch || first_tile >= total ||
        tile_count > total - first_tile) {
        return hipErrorInvalidValue;
    }

#define RUN(operation)                                                                             \
    hipLaunchKernelGGL((simt_pearl<operation>), dim3(tile_count), dim3(Layout::threads), 0,        \
                       stream, p, first_tile)
    switch (mode) {
    case Mode::Raw:
        RUN(Mode::Raw);
        break;
    case Mode::Fused:
        RUN(Mode::Fused);
        break;
    case Mode::Diagnostic:
        RUN(Mode::Diagnostic);
        break;
    }
#undef RUN
    return hipGetLastError();
}

} // namespace tnn::hip::iris::gemm
