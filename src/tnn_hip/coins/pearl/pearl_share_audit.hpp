#pragma once

#include <atomic>
#include <cstdint>

namespace tnn::pearl {

// Host-only accounting at batch/share boundaries, never in the GPU hot loop.
struct ShareAudit {
    std::atomic<uint64_t> discovered{0};
    std::atomic<uint64_t> proofs_built{0};
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> accepted{0};
    std::atomic<uint64_t> rejected{0};
    std::atomic<uint64_t> stale{0};
    std::atomic<uint64_t> cancelled{0};
    std::atomic<uint64_t> ambiguous{0};
    std::atomic<uint64_t> failed{0};
};

inline ShareAudit share_audit;

} // namespace tnn::pearl
