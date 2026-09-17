#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
namespace tnn::hip::iris::gemm::experimental {
struct NativeWinner { uint32_t row,col,digest[8]; };
struct NativeState { uint32_t total_hits,overflow; };
struct NativeResults { NativeWinner* winners; NativeState* state; uint32_t capacity; };
static_assert(sizeof(NativeWinner)==40 && sizeof(NativeState)==8);
inline bool valid_inputs(const int8_t* a,const int8_t* b,int32_t* d,
    uint32_t m,uint32_t n,uint32_t k,uint32_t lda,uint32_t ldb,uint32_t ldd,
    unsigned rank,const uint32_t* key,const uint32_t* target,NativeResults results,uint32_t* diagnostic) {
    auto aligned=[](const void* p,unsigned align){return p && reinterpret_cast<uintptr_t>(p)%align==0;};
    return !(rank!=128||m<256||n<256||m>8192||n>8192||m%128||n%256||
       (k!=2048&&k!=4096)||lda<m||ldb<k||ldd<m||lda>m+16||ldb>k+16||ldd>m+16||
       lda%16||ldb%16||ldd%16||!aligned(a,16)||!aligned(b,16)||!aligned(d,16)||
       !aligned(key,4)||!aligned(target,4)||!aligned(results.state,4)||
       (results.capacity&&!aligned(results.winners,4))||results.capacity>size_t(m)*n/128||
       (diagnostic&&!aligned(diagnostic,4)));
}
// Same input/result layout and validation contract as the pinned core.
// Caller resets state. Full D is mandatory; force_full bypasses diagnostics.
hipError_t launch_paired(const int8_t*,const int8_t*,int32_t*,uint32_t,uint32_t,uint32_t,
    uint32_t,uint32_t,uint32_t,unsigned,bool,const uint32_t*,const uint32_t*,
    NativeResults,uint32_t*,hipStream_t,bool raw=false,bool force_full=false);
}
