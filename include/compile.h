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

#define TNN_TARGET_CLONE(NAME, RET_TYPE, ARGS, BODY, ...) \
    EXPAND_TARGET_CLONE(NAME, RET_TYPE, ARGS, BODY, __VA_ARGS__)

#define EXPAND_TARGET_CLONE(NAME, RET_TYPE, ARGS, BODY, ...) \
    FOR_EACH_TARGET_CLONE(NAME, RET_TYPE, ARGS, BODY, __VA_ARGS__)

#define FOR_EACH_TARGET_CLONE(NAME, RET_TYPE, ARGS, BODY, ...) \
    GET_CLONE_MACRO(__VA_ARGS__, \
        CLONE_12, CLONE_11, CLONE_10, CLONE_9, CLONE_8, \
        CLONE_7, CLONE_6, CLONE_5, CLONE_4, CLONE_3, CLONE_2, CLONE_1) \
        (NAME, RET_TYPE, ARGS, BODY, __VA_ARGS__)

#define GET_CLONE_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,NAME,...) NAME

#define CLONE_1(NAME, RET_TYPE, ARGS, BODY, T1) \
    __attribute__((target(T1))) RET_TYPE NAME ARGS BODY

#define CLONE_2(NAME, RET_TYPE, ARGS, BODY, T1, T2) \
    CLONE_1(NAME, RET_TYPE, ARGS, BODY, T1) \
    __attribute__((target(T2))) RET_TYPE NAME ARGS BODY

#define CLONE_3(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3) \
    CLONE_2(NAME, RET_TYPE, ARGS, BODY, T1, T2) \
    __attribute__((target(T3))) RET_TYPE NAME ARGS BODY

#define CLONE_4(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4) \
    CLONE_3(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3) \
    __attribute__((target(T4))) RET_TYPE NAME ARGS BODY

#define CLONE_5(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5) \
    CLONE_4(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4) \
    __attribute__((target(T5))) RET_TYPE NAME ARGS BODY

#define CLONE_6(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6) \
    CLONE_5(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5) \
    __attribute__((target(T6))) RET_TYPE NAME ARGS BODY

#define CLONE_7(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7) \
    CLONE_6(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6) \
    __attribute__((target(T7))) RET_TYPE NAME ARGS BODY

#define CLONE_8(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7, T8) \
    CLONE_7(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7) \
    __attribute__((target(T8))) RET_TYPE NAME ARGS BODY

#define CLONE_9(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7, T8, T9) \
    CLONE_8(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7, T8) \
    __attribute__((target(T9))) RET_TYPE NAME ARGS BODY

#define CLONE_10(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) \
    CLONE_9(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7, T8, T9) \
    __attribute__((target(T10))) RET_TYPE NAME ARGS BODY

#define CLONE_11(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) \
    CLONE_10(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) \
    __attribute__((target(T11))) RET_TYPE NAME ARGS BODY

#define CLONE_12(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12) \
    CLONE_11(NAME, RET_TYPE, ARGS, BODY, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) \
    __attribute__((target(T12))) RET_TYPE NAME ARGS BODY

#define TNN_FEATURES_ZNVER1  "sse4.2,popcnt,avx,avx2,bmi,bmi2,fma"
#define TNN_FEATURES_ZNVER4  "sse4.2,popcnt,avx,avx2,bmi,bmi2,fma,avx512f,avx512dq,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg"
#define TNN_FEATURES_ZNVER5 \
  "avx512f,avx512dq,avx512bw,avx512vl,avx512vbmi,avx512vbmi2," \
  "avx512vnni,avx512bitalg,avx512fp16,avx512ifma," \
  "sse4.2,popcnt,avx,avx2,bmi,bmi2,fma"

#define TNN_TARGETS_X86_AVX2    "avx2", TNN_FEATURES_ZNVER1
#define TNN_TARGETS_X86_AVX512  "avx512f", TNN_FEATURES_ZNVER4, TNN_FEATURES_ZNVER5
#define TNN_TARGETS_X86_AVX512BW  "avx512f,avx512bw", TNN_FEATURES_ZNVER4, TNN_FEATURES_ZNVER5