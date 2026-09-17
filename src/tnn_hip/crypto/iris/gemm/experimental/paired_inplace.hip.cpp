// Derived from frozen gemm_next/tune.hip.cpp; see gemm_next/core/provenance.json.
// Selected in-place checkpoint variant. Existing Iris defaults are not changed.
#include "api.hpp"
#include "gfx1100_paired_ops.hpp"
#include "../../pearl/register_checkpoint.hpp"
#include "../../../blake3-inline.hip.inc"
namespace tnn::hip::iris::gemm::experimental {
namespace {
template<bool Diagnostic, bool Raw, bool NoD, class Schedule=Paired128x256Schedule, class Pearl=::tnn::hip::iris::pearl::RegisterCheckpoint128>
__global__ __launch_bounds__(Schedule::tile_m*2) void tune_kernel(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b,
    int32_t* __restrict__ d, unsigned m, unsigned n, unsigned k,
    unsigned lda, unsigned ldb, unsigned ldd, const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ target, NativeResults results, uint32_t* diagnostic) {
    static_assert(Schedule::tile_m==128 && Schedule::tile_n==256 && Schedule::tile_k==32 && Schedule::threads==256,
                  "Only the paired128x256 recipe has been triaged");
    static_assert(Pearl::rank==128 && !NoD,"Only rank128 full-D specializations are in scope");
    constexpr bool Prefetch=true, Local=true, Double=false, Priority=false, Paired=true, Swizzle=false, Half=false;
    constexpr int NB=8, Rank=Pearl::rank, Threads=Schedule::tile_m*2, MW=Schedule::tile_m/32;
    constexpr bool Streamed=Schedule::streamed;
    __shared__ __align__(16) unsigned char sa[Schedule::tile_m*32];
    constexpr unsigned BSpan=Swizzle?2048:2304;
    __shared__ __align__(16) unsigned char sb[BSpan*(NB/2)*(Double?2:1)];
    unsigned tid = threadIdx.x, lane=tid%32, wave=tid/32;
    // Neighboring N tiles share A in L2. WGM is a measured build parameter.
    constexpr unsigned WGM=Schedule::mapping;
    unsigned gm=m/Schedule::tile_m, gn=n/(NB*32), group=blockIdx.x/(gm*WGM);
    unsigned width=min(WGM,gn-group*WGM), local=blockIdx.x-group*gm*WGM;
    unsigned bm=(local/width)*Schedule::tile_m, bn=(group*WGM+local%width)*(NB*32);
    const int8_t* ap=a+bm+(tid%(Schedule::tile_m/16))*16+size_t(tid/(Schedule::tile_m/16))*lda;
    const int8_t* bp=b+size_t(bn+tid/2)*ldb+(tid%2)*16;
    unsigned aw=(tid/(Schedule::tile_m/16))*Schedule::tile_m+(tid%(Schedule::tile_m/16))*16;
    unsigned bw=b_offset<Swizzle>(tid/2,(tid%2)*16);
    unsigned am=(wave%MW)*32+(lane%16)*2;
    unsigned nn=(wave/MW)*16+lane%16;
    I8 c[2*NB] = {};
    uint32_t transcript[16]={};
    I4 ga{},gb[256/Schedule::tile_m]{};
    auto fetch=[&](){
        ga=*reinterpret_cast<const I4*>(ap);
        #pragma unroll
        for(int i=0;i<256/Schedule::tile_m;++i) gb[i]=*reinterpret_cast<const I4*>(bp+size_t(i*Schedule::tile_m)*ldb);
    };
    if constexpr(Prefetch) fetch();
    for(unsigned kk=0; kk<k; kk+=32) {
        unsigned slot=Double?((kk/32)&1):0;
        auto* as=sa;
        auto* bs=sb+slot*BSpan*(NB/2);
        if constexpr(!Prefetch) fetch();
        *reinterpret_cast<I4*>(as+aw)=ga;
        #pragma unroll
        for(int i=0;i<256/Schedule::tile_m;++i) *reinterpret_cast<I4*>(bs+b_offset<Swizzle>(tid/2+i*Schedule::tile_m,(tid%2)*16))=gb[i];
        shared_barrier<Local>();
        ap+=size_t(32)*lda; bp+=32;
        if constexpr(Prefetch && Schedule::prefetch==0) {
            if(kk+32<k) fetch();
        }
        unsigned operand_am=am, operand_nn=nn;
        if constexpr(Schedule::shared_address) {
            unsigned fresh;
            asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(fresh));
            unsigned fresh_wave=__builtin_amdgcn_readfirstlane(threadIdx.x)/32;
            operand_am=(fresh_wave%MW)*32+(fresh%16)*2;
            operand_nn=(fresh_wave/MW)*16+fresh%16;
        }
        auto x0=operands<Schedule,0,NB,Paired,Swizzle,Half>(as,bs,operand_am,operand_nn);
        if constexpr(Streamed) {
            compute<NB,Priority>(c,x0);
            __builtin_amdgcn_sched_barrier(0);
            auto x1=operands<Schedule,16,NB,Paired,Swizzle,Half>(as,bs,operand_am,operand_nn);
            compute<NB,Priority>(c,x1);
        } else {
            auto x1=operands<Schedule,16,NB,Paired,Swizzle,Half>(as,bs,operand_am,operand_nn);
            if constexpr(Schedule::prefetch==1) if(kk+32<k) fetch();
            compute<NB,Priority>(c,x0); compute<NB,Priority>(c,x1);
        }

        if constexpr(!Raw) if((kk+32)%Rank==0)
            {
#include "../../pearl/register_checkpoint_body.inc"
        }
        if constexpr(Schedule::prefetch==2) if(kk+32<k) fetch();
        // With two slots the next write uses the other slot. The next start
        // barrier proves all reads complete before this slot is reused.
        if constexpr(!Double) shared_barrier<Local>();
    }
    if constexpr(!NoD) {
    #pragma unroll
    for(int nb=0;nb<NB;++nb) {
        #pragma unroll
        for(int e=0;e<8;++e) {
            unsigned row=bm+am;
            unsigned col=bn+(wave/MW)*16+nb*32+2*e+lane/16;
            // Adjacent rows are a vector store, following the studied epilogue.
            using I2=int __attribute__((ext_vector_type(2)));
            *reinterpret_cast<I2*>(d+row+size_t(col)*ldd)=I2{c[nb*2][e],c[nb*2+1][e]};
        }
    }

    }
    if constexpr(!Raw) {
    // Optional D stores precede hashing; GEMM accumulators are now dead.
    uint32_t digest[8];
    #pragma unroll
    for(unsigned i=0;i<8;++i) digest[i]=key[i];
    blake3_compress_in_place(digest,reinterpret_cast<const uint8_t*>(transcript),64,0,
                            CHUNK_START|CHUNK_END|ROOT|(1<<4));
    if constexpr(Diagnostic) {
        unsigned row=bm+am, col=bn+(wave/MW)*16+lane/16;
        size_t pos=size_t(row/2)*(n/64)+(col/256)*4+((col%256)/16)*2+col%2;
        size_t positions=size_t(m)*n/128;
        #pragma unroll
        for(unsigned i=0;i<16;++i) diagnostic[size_t(i)*positions+pos]=transcript[i];
        #pragma unroll
        for(unsigned i=0;i<8;++i) diagnostic[size_t(i+16)*positions+pos]=digest[i];
    }
    bool equal=true, lower=false;
    #pragma unroll
    for(int i=7;i>=0;--i) {
        lower |= equal && digest[i]<target[i];
        equal &= digest[i]==target[i];
    }
    if(lower||equal) {
        unsigned slot=atomicAdd(&results.state->total_hits,1u);
        if(slot<results.capacity) {
            NativeWinner& winner=results.winners[slot];
            winner.row=bm+am;
            winner.col=bn+(wave/MW)*16+lane/16;
            #pragma unroll
            for(unsigned i=0;i<8;++i) winner.digest[i]=digest[i];
        } else atomicExch(&results.state->overflow,1u);
    }
    }
}

} // anonymous namespace


using Schedule=Paired128x256Schedule;
hipError_t launch_paired_inplace(const int8_t* a,const int8_t* b,int32_t* d,
    uint32_t m,uint32_t n,uint32_t k,uint32_t lda,uint32_t ldb,uint32_t ldd,
    unsigned rank,bool streamed,const uint32_t* key,const uint32_t* target,
    NativeResults results,uint32_t* diagnostic,hipStream_t stream,bool raw,bool force_full) {
    if(!valid_inputs(a,b,d,m,n,k,lda,ldb,ldd,rank,key,target,results,diagnostic)) return hipErrorInvalidValue;
    dim3 grid(size_t(m/Schedule::tile_m)*(n/256));
#define RUN(D,R,N) hipLaunchKernelGGL((tune_kernel<D,R,N>),grid,dim3(Schedule::tile_m*2),0,stream,a,b,d,m,n,k,lda,ldb,ldd,key,target,results,diagnostic)
    if(raw){RUN(false,true,false);}else if(force_full){RUN(false,false,false);}else if(diagnostic){RUN(true,false,false);}else{RUN(false,false,false);}
#undef RUN
    return hipGetLastError();
}
} // namespace tnn::hip::iris::gemm::experimental
