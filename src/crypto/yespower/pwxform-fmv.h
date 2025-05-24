// pwxform-fmv.hpp
#ifndef PWXFORM_FMV_HPP
#define PWXFORM_FMV_HPP

#ifdef __x86_64__
#include <immintrin.h>
#endif

// Configuration constants
const size_t PWXsimple = 2;
const size_t PWXgather = 4;
const size_t PWXbytes = PWXgather * PWXsimple * 8;

// Context structure
typedef struct PwxformContext {
    uint8_t *S0, *S1, *S2;
    size_t w;
    uint32_t Sbytes;
    uint64_t smask2;  // Precomputed mask
    int version;      // 0 for v0.5, 1 for v1.0
    int rounds;       // 3 or 6
} PwxformContext;

#ifdef __x86_64__
// SSE2 implementation
void pwxform(uint32_t* B, PwxformContext* ctx) {
    __m128i* X = (__m128i*)B;
    uint8_t* S0 = ctx->S0;
    uint8_t* S1 = ctx->S1;
    uint64_t smask2 = ctx->smask2;
    
    #define PWXFORM_ROUND_ASM(Xi) { \
        __m128i H; \
        __asm__( \
            "movd %0, %%rax\n\t" \
            "pshufd $0xb1, %0, %1\n\t" \
            "andq %2, %%rax\n\t" \
            "pmuludq %1, %0\n\t" \
            "movl %%eax, %%ecx\n\t" \
            "shrq $0x20, %%rax\n\t" \
            "paddq (%3,%%rcx), %0\n\t" \
            "pxor (%4,%%rax), %0\n\t" \
            : "+x" (Xi), "=x" (H) \
            : "r" (smask2), "r" (S0), "r" (S1) \
            : "cc", "rax", "rcx"); \
    }
    
    // Main pwxform logic using optimized assembly
    for (int round = 0; round < ctx->rounds; round++) {
        // Unroll for performance
        PWXFORM_ROUND_ASM(X[0])
        PWXFORM_ROUND_ASM(X[1])
        PWXFORM_ROUND_ASM(X[2])
        PWXFORM_ROUND_ASM(X[3])
        
        // Handle S-box updates for v1.0
        if (ctx->version == 1) {
            // Implementation details...
        }
    }
    #undef PWXFORM_ROUND_ASM
}
#else // __x86_64__

void pwxform(uint32_t* B, PwxformContext* ctx) {
    uint64_t* X = (uint64_t*)B;
    uint8_t* S0 = ctx->S0;
    uint8_t* S1 = ctx->S1;
    size_t w = ctx->w;
    
    for (int round = 0; round < ctx->rounds; round++) {
        for (size_t j = 0; j < PWXgather; j++) {
            uint64_t x0 = X[j * PWXsimple];
            uint64_t x1 = X[j * PWXsimple + 1];
            
            for (size_t k = 0; k < PWXsimple; k++) {
                uint64_t x = X[j * PWXsimple + k];
                
                // Extract indices
                uint32_t lo = (uint32_t)x & (uint32_t)ctx->smask2;
                uint32_t hi = (x >> 32) & (uint32_t)(ctx->smask2 >> 32);
                
                // S-box lookups
                uint64_t* p0 = (uint64_t*)(S0 + lo);
                uint64_t* p1 = (uint64_t*)(S1 + hi);
                
                // Multiply-add-xor
                x = ((x >> 32) * (uint32_t)x) + p0[0];
                x ^= p1[0];
                
                X[j * PWXsimple + k] = x;
            }
            
            // Version 1.0: Update S-boxes
            if (ctx->version == 1 && (round == 0 || j < PWXgather / 2)) {
                if (j & 1) {
                    for (size_t k = 0; k < PWXsimple; k++) {
                        ((uint64_t*)(S1 + w))[k] = X[j * PWXsimple + k];
                    }
                } else {
                    for (size_t k = 0; k < PWXsimple; k++) {
                        ((uint64_t*)(S0 + w))[k] = X[j * PWXsimple + k];
                    }
                }
            }
        }
    }
    
    // Version 1.0: Rotate S-boxes
    if (ctx->version == 1) {
        w += 16;
        ctx->w = w & (ctx->Sbytes / 3 - 1);
        uint8_t* tmp = ctx->S2;
        ctx->S2 = ctx->S1;
        ctx->S1 = ctx->S0;
        ctx->S0 = tmp;
    }
}
#endif

#endif // PWXFORM_FMV_HPP