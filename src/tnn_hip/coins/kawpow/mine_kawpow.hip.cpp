#ifdef TNN_KAWPOW

#include "../../common/gpu_compat.hpp"
#include "../../common/gpu_miner.hpp"
#include "../../common/hip_algo_registry.hpp"
#include <algo_definitions.h>
#include <tnn_log.hpp>

// ============================================================================
// KawPow GPU mining entry point (stub)
// ============================================================================

void mineKawPow_hip(int tid)
{
    TNN_LOG_INFO("[KawPow] GPU mining not yet implemented\n");
    (void)tid;
}

#endif // TNN_KAWPOW
