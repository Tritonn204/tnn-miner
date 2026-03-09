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

// Base clone macros (no section placement)
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

// ============================================================================
// Pair-based section clone: TNN_SECTION_CLONE(name, ret, args, body, pairs...)
//
// Each pair is (section_attr, target_string).  Every clone gets its OWN section.
//
// Usage:
//   TNN_SECTION_CLONE(stage_3, static void, (uint64_t *sp, workerData &w), { ... },
//       TNN_SECTION_XELIS_AES, "aes,sse4.1",
//       TNN_SECTION_COLD,      "default")
//
// Expands to:
//   __attribute__((target("aes,sse4.1"))) __attribute__((section(".text$a_xelis_aes")))
//       static void stage_3(uint64_t *sp, workerData &w) { ... }
//   __attribute__((target("default"))) __attribute__((section(".text$z_cold")))
//       static void stage_3(uint64_t *sp, workerData &w) { ... }
// ============================================================================

#define TNN_SECTION_CLONE(NAME, RET_TYPE, ARGS, BODY, ...) \
    EXPAND_SPCLONE(NAME, RET_TYPE, ARGS, BODY, __VA_ARGS__)

// Count variadic args by 2s (each pair = section + target = 2 args)
#define EXPAND_SPCLONE(NAME, RET_TYPE, ARGS, BODY, ...) \
    GET_SPCLONE_MACRO(__VA_ARGS__, \
        SPCLONE_6, _sp11, SPCLONE_5, _sp9, \
        SPCLONE_4, _sp7, SPCLONE_3, _sp5, \
        SPCLONE_2, _sp3, SPCLONE_1, _sp1) \
        (NAME, RET_TYPE, ARGS, BODY, __VA_ARGS__)

#define GET_SPCLONE_MACRO( \
    _1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12, NAME, ...) NAME

// 1 pair (2 args: S1, T1)
#define SPCLONE_1(NAME, RET_TYPE, ARGS, BODY, S1, T1) \
    __attribute__((target(T1))) S1 RET_TYPE NAME ARGS BODY

// 2 pairs (4 args)
#define SPCLONE_2(NAME, RET_TYPE, ARGS, BODY, S1, T1, S2, T2) \
    SPCLONE_1(NAME, RET_TYPE, ARGS, BODY, S1, T1) \
    __attribute__((target(T2))) S2 RET_TYPE NAME ARGS BODY

// 3 pairs (6 args)
#define SPCLONE_3(NAME, RET_TYPE, ARGS, BODY, S1, T1, S2, T2, S3, T3) \
    SPCLONE_2(NAME, RET_TYPE, ARGS, BODY, S1, T1, S2, T2) \
    __attribute__((target(T3))) S3 RET_TYPE NAME ARGS BODY

// 4 pairs (8 args)
#define SPCLONE_4(NAME, RET_TYPE, ARGS, BODY, S1, T1, S2, T2, S3, T3, S4, T4) \
    SPCLONE_3(NAME, RET_TYPE, ARGS, BODY, S1, T1, S2, T2, S3, T3) \
    __attribute__((target(T4))) S4 RET_TYPE NAME ARGS BODY

// 5 pairs (10 args)
#define SPCLONE_5(NAME, RET_TYPE, ARGS, BODY, S1, T1, S2, T2, S3, T3, S4, T4, S5, T5) \
    SPCLONE_4(NAME, RET_TYPE, ARGS, BODY, S1, T1, S2, T2, S3, T3, S4, T4) \
    __attribute__((target(T5))) S5 RET_TYPE NAME ARGS BODY

// 6 pairs (12 args)
#define SPCLONE_6(NAME, RET_TYPE, ARGS, BODY, S1, T1, S2, T2, S3, T3, S4, T4, S5, T5, S6, T6) \
    SPCLONE_5(NAME, RET_TYPE, ARGS, BODY, S1, T1, S2, T2, S3, T3, S4, T4, S5, T5) \
    __attribute__((target(T6))) S6 RET_TYPE NAME ARGS BODY

// No-section placeholder for clones that should stay in default .text
#define TNN_SECTION_DEFAULT /* no section attribute */

// ============================================================================
// Section placement for code layout optimization
//
// TNN_SECTION(name) builds the platform-correct section attribute.
// Names sort alphabetically to control layout order.
//
// PE/COFF (Windows): .text$<name>  — sorted by $ suffix automatically
// ELF (Linux):       .text.<name>  — sorted by --sort-section=name (lld)
// ============================================================================

#ifdef _WIN32
  #define TNN_SECTION(name) __attribute__((section(".text$" name)))
#else
  #define TNN_SECTION(name) __attribute__((section(".text." name)))
#endif

// General-purpose tiers. Algorithm-specific tiers belong in their own files.
#define TNN_SECTION_HOT   TNN_SECTION("b_hot")
#define TNN_SECTION_COLD  TNN_SECTION("z_cold")

#define TNN_FEATURES_ZNVER1  "sse4.2,popcnt,avx,avx2,bmi,bmi2,fma"
#define TNN_FEATURES_ZNVER4  "sse4.2,popcnt,avx,avx2,bmi,bmi2,fma,avx512f,avx512dq,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg"
#define TNN_FEATURES_ZNVER5 \
    "avx512f,avx512dq,avx512bw,avx512vl,avx512vbmi,avx512vbmi2," \
    "avx512vnni,avx512bitalg,avx512ifma," \
    "sse4.2,popcnt,avx,avx2,bmi,bmi2,fma"

#define TNN_TARGETS_X86_AVX2    "avx2", TNN_FEATURES_ZNVER1
#define TNN_TARGETS_X86_AVX512  "avx512f", TNN_FEATURES_ZNVER4, TNN_FEATURES_ZNVER5
#define TNN_TARGETS_X86_AVX512BW  "avx512f,avx512bw", TNN_FEATURES_ZNVER4, TNN_FEATURES_ZNVER5