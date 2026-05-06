#include "../coins/miners.hpp"

#if defined(TNN_XELISHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
#include "common/mine_cpu_unified.hpp"
#include "algos/xelis_cpu.hpp"

// Wrapper function for Xelis V2/V3 unified mining
void mineXelis_unified(int tid) {
    // Determine which version to use based on mining profile
    std::string algo_name;

    if (miningProfile.coin.miningAlgo == ALGO_XELISV3) {
        algo_name = "xelis_v3";
    } else {
        algo_name = "xelis_v2";
    }

    // Call unified CPU miner
    mineCPU_unified(tid, algo_name);
}
#endif
