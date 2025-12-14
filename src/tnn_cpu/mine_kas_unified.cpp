#include "../coins/miners.hpp"

#ifndef TNN_HIP
#include "common/mine_cpu_unified.hpp"

#ifdef TNN_ASTRIXHASH
#include "algos/astrix_cpu.hpp"
#endif

#ifdef TNN_NXLHASH
#include "algos/nexellia_cpu.hpp"
#endif

#ifdef TNN_WALAHASH
#include "algos/waglayla_cpu.hpp"
#endif

#ifdef TNN_HOOHASH
#include "algos/hoosat_cpu.hpp"
#endif

#ifdef TNN_ASTRIXHASH
// Wrapper function for Astrix unified mining
void mineAstrix_unified(int tid) {
    mineCPU_unified(tid, "astrix");
}
#endif

#ifdef TNN_NXLHASH
// Wrapper function for Nexellia unified mining
void mineNexellia_unified(int tid) {
    mineCPU_unified(tid, "nexellia");
}
#endif

#ifdef TNN_WALAHASH
// Wrapper function for Waglayla unified mining
void mineWaglayla_unified(int tid) {
    mineCPU_unified(tid, "waglayla");
}
#endif

#ifdef TNN_HOOHASH
// Wrapper function for Hoosat unified mining
void mineHoosat_unified(int tid) {
    mineCPU_unified(tid, "hoosat");
}
#endif

#endif // !TNN_HIP
