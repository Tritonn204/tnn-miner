/*
 * sha_detect.h - Minimal SHA-NI detection for mining applications
 */

#ifndef SHA_DETECT_H
#define SHA_DETECT_H

#include <stdint.h>

#ifdef _MSC_VER
  #include <intrin.h>
  #define SHA_DETECT_WINDOWS 1
#elif defined(__MINGW32__) || defined(__MINGW64__)
  #include <intrin.h>
  #include <immintrin.h>
  #define SHA_DETECT_WINDOWS 1
  #define SHA_DETECT_MINGW 1
#else
  #include <cpuid.h>
  #include <immintrin.h>
  #include <signal.h>
  #include <setjmp.h>
  #include <string.h>
#endif

#ifndef SHA_DETECT_WINDOWS
// Global variables for signal handling (only on POSIX)
static jmp_buf sha_detect_jmp_env;
static volatile int sha_detect_instruction_failed = 0;

// Signal handler (only compiled on POSIX)
static void sha_detect_sigill_handler(int sig) {
    (void)sig; // Suppress unused parameter warning
    sha_detect_instruction_failed = 1;
    longjmp(sha_detect_jmp_env, 1);
}
#endif

// Cross-platform CPUID wrapper
static inline void sha_detect_cpuid(uint32_t leaf, uint32_t subleaf, 
                                   uint32_t *eax, uint32_t *ebx, 
                                   uint32_t *ecx, uint32_t *edx) {
#if defined(_MSC_VER)
    int regs[4];
    if (subleaf == 0) {
        __cpuid(regs, leaf);
    } else {
        __cpuidex(regs, leaf, subleaf);
    }
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#elif defined(SHA_DETECT_MINGW)
    int regs[4];
    if (subleaf == 0) {
        __cpuid(regs, leaf);
    } else {
        __cpuidex(regs, leaf, subleaf);
    }
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#else
    if (subleaf == 0) {
        __cpuid(leaf, *eax, *ebx, *ecx, *edx);
    } else {
        __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
    }
#endif
}

/**
 * Detects if SHA-NI instructions are supported and usable
 * 
 * @return 1 if SHA-NI is supported and usable, 0 otherwise
 */

static inline int has_sha_ni_support(void) {
#ifdef __x86_64__
    // Variables for CPUID results
    uint32_t eax, ebx, ecx, edx;
    int has_cpuid_support = 0;

    // Check CPUID leaf 7 support
    sha_detect_cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    
    if (eax < 7) {
        return 0; // CPUID leaf 7 not supported
    }

    // Get SHA-NI support bit (bit 29 of EBX from leaf 7, subleaf 0)
    sha_detect_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    has_cpuid_support = (ebx & (1U << 29)) != 0;

    if (!has_cpuid_support) {
        return 0; // CPUID says SHA-NI not supported
    }

    // Verify actual SHA-NI instruction execution
#ifdef SHA_DETECT_WINDOWS
    // On Windows/MinGW, just trust CPUID since exception handling is complex
    return has_cpuid_support;
#else
    // On POSIX systems, try executing SHA-NI with signal protection
    struct sigaction old_action, new_action;
    memset(&new_action, 0, sizeof(new_action));
    new_action.sa_handler = sha_detect_sigill_handler;
    sigemptyset(&new_action.sa_mask);
    new_action.sa_flags = 0;
    
    // Install signal handler
    if (sigaction(SIGILL, &new_action, &old_action) != 0) {
        return has_cpuid_support; // Can't test, assume CPUID is right
    }
    
    sha_detect_instruction_failed = 0;
    
    if (setjmp(sha_detect_jmp_env) == 0) {
        // Try executing a SHA-NI instruction
        __m128i state0 = _mm_setzero_si128();
        __m128i state1 = _mm_setzero_si128();
        __m128i msg = _mm_setzero_si128();
        
        // This causes SIGILL if SHA-NI not available
        __m128i result = _mm_sha256rnds2_epu32(state1, state0, msg);
        
        // Prevent optimization
        volatile uint32_t dummy = _mm_extract_epi32(result, 0);
        (void)dummy;
    }
    
    // Restore original signal handler
    sigaction(SIGILL, &old_action, NULL);
    
    return has_cpuid_support && !sha_detect_instruction_failed;
#endif
#else
    return 0;
#endif
}

/**
 * Cached SHA-NI support detection
 * 
 * @return 1 if SHA-NI is supported and usable, 0 otherwise
 */
static inline int has_sha_ni_support_cached(void) {
    static int checked = 0;
    static int result = 0;
    
    if (!checked) {
        result = has_sha_ni_support();
        checked = 1;
    }
    
    return result;
}

#endif /* SHA_DETECT_H */