"""Exercise the actual production reduction bodies, without allocating a DAG.

Default: host shuffle model for wave32/wave64 and partial N-way batches.
--gpu: additionally execute a tiny HIP kernel on gfx1100 (explicit opt-in).
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true')
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
source = (root / 'src/tnn_hip/crypto/kawpow/kawpow.hip').read_text(encoding='utf-8')
start = source.index('__device__ __forceinline__ void progpow_reduce_distributed(')
end = source.index('// ===========================================================================\n// Strategy 0', start)
bodies = source[start:end]
assert '__lane_id' not in bodies
assert source.count('progpow_reduce_full(mix, digest, lane_id);') == 3
assert source.count('progpow_reduce_distributed_2way(m0, m1, digest0, digest1, lane_id);') == 2

prefix = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#define FNV_OFFSET_BASIS 0x811c9dc5u
#define PROGPOW_REGS 32
#ifdef GPU_TEST
#include <hip/hip_runtime.h>
#define PROGPOW_SHUFFLE_XOR8(v) __builtin_amdgcn_ds_swizzle(v, 0x201f)
#define PROGPOW_BPERMUTE(s,v) __builtin_amdgcn_ds_bpermute((s)<<2,v)
#else
#define __device__
#define __forceinline__ inline
struct { unsigned x; } threadIdx;
unsigned reference_lane[2][64], reference_digest[2][4][8];
unsigned simulated_xor(unsigned value) {
    unsigned lane = threadIdx.x;
    unsigned which = value == reference_lane[0][lane] ? 0 : 1;
    assert(value == reference_lane[which][lane]);
    return reference_lane[which][lane ^ 8];
}
unsigned simulated_broadcast(unsigned lane, unsigned value) {
    const unsigned own = threadIdx.x;
    assert(value == (own % 16 < 8 ? reference_digest[0][own / 16][own % 16] : 0));
    return reference_digest[0][lane / 16][lane % 16];
}
#define PROGPOW_SHUFFLE_XOR8(v) simulated_xor(v)
#define PROGPOW_BPERMUTE(s,v) simulated_broadcast(s,v)
#endif
__device__ __forceinline__ unsigned fnv1a(unsigned a, unsigned b) { return (a ^ b) * 0x01000193u; }
'''
suffix = r'''
unsigned input(unsigned hash, unsigned lane, unsigned reg) {
    unsigned x = 1234567u + hash * 977u + lane * 13331u + reg * 719u;
    x ^= x >> 16; x *= 0x7feb352du; return x ^ (x >> 15);
}
unsigned fold(unsigned hash, unsigned lane) {
    unsigned x = FNV_OFFSET_BASIS;
    for (unsigned r=0;r<32;++r) x = (x ^ input(hash,lane,r)) * 0x01000193u;
    return x;
}
unsigned expected(unsigned hash, unsigned word) {
    return (((FNV_OFFSET_BASIS ^ fold(hash,word)) * 0x01000193u)
            ^ fold(hash,word+8)) * 0x01000193u;
}
#ifdef GPU_TEST
__global__ void check_kernel(const unsigned* inputs, unsigned* output, unsigned count, unsigned ways) {
    unsigned lane = threadIdx.x % 16, group = threadIdx.x / 16;
    unsigned first = group * ways;
    if (first >= count) return;
    unsigned mix[4][32], digest[4][8];
    unsigned valid = min(ways, count-first);
    for(unsigned h=0;h<valid;++h) for(unsigned r=0;r<32;++r)
        mix[h][r]=inputs[((first+h)*16+lane)*32+r];
    if(valid>=2) progpow_reduce_distributed_2way(mix[0],mix[1],digest[0],digest[1],lane);
    else progpow_reduce_distributed(mix[0],digest[0],lane);
    for(unsigned h=2;h<valid;++h) progpow_reduce_distributed(mix[h],digest[h],lane);
    if(lane<8) for(unsigned h=0;h<valid;++h) output[(first+h)*16+lane]=digest[h][lane];
    for(unsigned h=0;h<valid;++h) {
        progpow_reduce_full(mix[h],digest[h],lane);
        if(lane==0) for(unsigned r=0;r<8;++r) output[(first+h)*16+8+r]=digest[h][r];
    }
}
void hip_check(hipError_t e) { if(e != hipSuccess) { fprintf(stderr,"%s\n",hipGetErrorString(e)); std::exit(1); } }
#endif
int main() {
#ifdef GPU_TEST
    hipDeviceProp_t prop{}; hip_check(hipGetDeviceProperties(&prop,0));
    assert(std::string(prop.gcnArchName).find("gfx1100") == 0);
    unsigned *di=nullptr,*do_=nullptr;
    hip_check(hipMalloc(&di,16*16*32*sizeof(unsigned)));
    hip_check(hipMalloc(&do_,16*16*sizeof(unsigned)));
    for(unsigned ways : {1u,2u,4u}) for(unsigned count=1;count<=16;++count) {
        unsigned in[16*16*32], out[16*16];
        for(unsigned h=0;h<16;++h) for(unsigned l=0;l<16;++l) for(unsigned r=0;r<32;++r)
            in[(h*16+l)*32+r]=input(h,l,r);
        hip_check(hipMemcpy(di,in,sizeof(in),hipMemcpyHostToDevice));
        hip_check(hipMemset(do_,0xa5,sizeof(out)));
        check_kernel<<<1,256>>>(di,do_,count,ways);
        hip_check(hipGetLastError()); hip_check(hipDeviceSynchronize());
        hip_check(hipMemcpy(out,do_,sizeof(out),hipMemcpyDeviceToHost));
        for(unsigned h=0;h<count;++h) for(unsigned r=0;r<16;++r) assert(out[h*16+r]==expected(h,r%8));
        for(unsigned i=count*16;i<256;++i) assert(out[i]==0xa5a5a5a5u);
    }
    hip_check(hipFree(di)); hip_check(hipFree(do_));
    puts("GPU PASS: 48 tiny launches; full/distributed/2way; partial batches; output guards");
#else
    for(unsigned wave : {32u,64u}) for(unsigned group=0;group<wave/16;++group)
        for(unsigned h=0;h<2;++h) for(unsigned lane=0;lane<16;++lane) {
            reference_lane[h][group*16+lane]=fold(group*2+h,lane);
            if(lane<8) reference_digest[h][group][lane]=expected(group*2+h,lane);
        }
    for(unsigned wave : {32u,64u}) for(unsigned active=1;active<=wave/16;++active)
        for(unsigned lane=0;lane<active*16;++lane) {
            threadIdx.x=lane;
            unsigned mix[2][32], digest[2][8];
            for(unsigned h=0;h<2;++h) for(unsigned r=0;r<32;++r) mix[h][r]=input(lane/16*2+h,lane%16,r);
            for (auto& words : digest) std::fill(words, words+8, 0xa5a5a5a5u);
            progpow_reduce_distributed_2way(mix[0],mix[1],digest[0],digest[1],lane%16);
            for(unsigned h=0;h<2;++h) for(unsigned r=0;r<8;++r)
                assert(digest[h][r] == (lane%16==r ? expected(lane/16*2+h,r) : 0xa5a5a5a5u));
            progpow_reduce_full(mix[0],digest[0],lane%16);
            for(unsigned r=0;r<8;++r) assert(digest[0][r]==expected(lane/16*2,r));
        }
    puts("CPU PASS: actual production reducers; wave32/wave64; partial groups; ownership guards");
#endif
}
'''
with tempfile.TemporaryDirectory(prefix='tnn-kawpow-reduction-') as folder:
    folder = Path(folder)
    cpp, exe = folder/'test.cpp', folder/'test.exe'
    cpp.write_text('#include <string>\n' + prefix + bodies + suffix, encoding='utf-8')
    command = ['clang++', '-std=c++20', '-O2', str(cpp), '-o', str(exe)]
    if args.gpu:
        command = ['C:/Program Files/AMD/ROCm/6.4/bin/hipcc.exe', '-DGPU_TEST',
                   '--offload-arch=gfx1100', '-std=c++20', '-O2', str(cpp), '-o', str(exe)]
    subprocess.run(command, check=True, timeout=60)
    subprocess.run([str(exe)], check=True, timeout=20)
