#pragma once

#define TNN_TARGETS /* __attribute__ ((target_clones ("arch=x86-64-v4", "arch=x86-64-v3", "arch=x86-64-v2", "arch=x86-64", "arch=alderlake", "default"))) */

#if !defined(__x86_64__) && !defined(_M_X64)
    // For non-x86 platforms, map _mm_prefetch to __builtin_prefetch
    #define _MM_HINT_T0  1
    #define _MM_HINT_T1  2
    #define _MM_HINT_T2  3
    #define _MM_HINT_NTA 0
    
    #define _mm_prefetch(p, h) \
        __builtin_prefetch((p), ((h) == _MM_HINT_T0) ? 0 : 0, \
                          ((h) == _MM_HINT_NTA) ? 0 : \
                          ((h) == _MM_HINT_T0) ? 3 : \
                          ((h) == _MM_HINT_T1) ? 2 : 1)
#endif