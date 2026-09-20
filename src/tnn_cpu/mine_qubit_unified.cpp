#include "../coins/miners.hpp"

#if defined(TNN_QHASH) && (!defined(TNN_HIP) || defined(WITH_OROCHI))
#include "common/mine_cpu_unified.hpp"
#include "algos/qhash_cpu.hpp"

void mineQubit_unified(int tid) {
    mineCPU_unified(tid, "qhash");
}
#endif
