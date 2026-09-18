#pragma once

#include "coordinate.hpp"

namespace iris::hip {

// Each stage owns both operands. Stride may exceed the occupied byte span.
template <unsigned ABytes, unsigned BBytes, unsigned Stages = 1,
          unsigned Alignment = 16,
          unsigned StageStride = ((ABytes + Alignment - 1) / Alignment * Alignment +
                                  BBytes + Alignment - 1) / Alignment * Alignment>
struct LdsStages {
    static_assert(ABytes > 0 && BBytes > 0);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0);

    static constexpr unsigned stage_count = Stages;
    static constexpr unsigned alignment = Alignment;
    static constexpr unsigned b_offset = (ABytes + Alignment - 1) / Alignment * Alignment;
    static constexpr unsigned stage_span = b_offset + BBytes;
    static constexpr unsigned stage_stride = StageStride;
    static constexpr unsigned allocation_bytes = (Stages - 1) * StageStride + stage_span;

    static_assert(StageStride >= stage_span && StageStride % Alignment == 0);

    IRIS_HOST_DEVICE static constexpr unsigned offset(unsigned iteration)
    {
        return (iteration % Stages) * StageStride;
    }
};

struct LinearK {
    static constexpr bool reorders_k = false;

    IRIS_HOST_DEVICE static constexpr unsigned tile(unsigned iteration, unsigned, unsigned)
    {
        return iteration;
    }
};

// Seed is the mapped workgroup coordinate, not necessarily the launch block ID.
// Nonzero tile_count and whole K tiles are required by the caller.
template <unsigned Positions = 32, unsigned StrideShift = 2>
struct StaggeredK {
    static_assert(Positions > 0 && (Positions & (Positions - 1)) == 0);
    static_assert(StrideShift < 31);
    static constexpr bool reorders_k = true;

    IRIS_HOST_DEVICE static constexpr unsigned effective_positions(unsigned tile_count)
    {
        unsigned count = Positions;
        while (count > 1 && count > (tile_count >> StrideShift))
            count /= 2;
        return count;
    }

    IRIS_HOST_DEVICE static constexpr unsigned tile(unsigned iteration, unsigned tile_count,
                                                   unsigned seed)
    {
        const unsigned start = (seed & (effective_positions(tile_count) - 1)) << StrideShift;
        // Avoid overflow in iteration + start. Inputs are in [0, tile_count).
        return iteration >= tile_count - start ? iteration - (tile_count - start)
                                               : iteration + start;
    }
};

template <class Traversal, bool OrderSensitiveCheckpoints>
struct KTraversalContract {
    static_assert(!OrderSensitiveCheckpoints || !Traversal::reorders_k,
                  "K staggering is not qualified for order-sensitive checkpoints");
    using Policy = Traversal;
};

} // namespace iris::hip
