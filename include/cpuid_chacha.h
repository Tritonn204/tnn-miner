#ifndef CPUID_CHACHA_H
#define CPUID_CHACHA_H

#define LOCAL_PREFIX3(a,b) a##_##b
#define LOCAL_PREFIX2(a,b) LOCAL_PREFIX3(a,b)
#define LOCAL_PREFIX(n) LOCAL_PREFIX2(chacha,n)
#define PROJECT_NAME chacha

#if defined(__cplusplus)
extern "C" {
#endif

enum cpuid_flags_generic_t {
	CPUID_GENERIC = (0)
};

#if defined(__x86_64__)
enum cpuid_flags_x86_t {
	CPUID_X86       = (1 <<  0),
	CPUID_MMX       = (1 <<  1),
	CPUID_SSE       = (1 <<  2),
	CPUID_SSE2      = (1 <<  3),
	CPUID_SSE3      = (1 <<  4),
	CPUID_SSSE3     = (1 <<  5),
	CPUID_SSE4_1    = (1 <<  6),
	CPUID_SSE4_2    = (1 <<  7),
	CPUID_AVX       = (1 <<  8),
	CPUID_XOP       = (1 <<  9),
	CPUID_AVX2      = (1 << 10),
	CPUID_AVX512    = (1 << 11),

	CPUID_RDTSC     = (1 << 25),
	CPUID_RDRAND    = (1 << 26),
	CPUID_POPCNT    = (1 << 27),
	CPUID_FMA4      = (1 << 28),
	CPUID_FMA3      = (1 << 29),
	CPUID_PCLMULQDQ = (1 << 30),
	CPUID_AES       = (1 << 31)
};
#else
enum cpuid_flags_arm_t {
	CPUID_ARM       = (1 <<  0),
	CPUID_ARMv6     = (1 <<  1),
	CPUID_ARMv7     = (1 <<  2),
	CPUID_ARMv8     = (1 <<  3),

	CPUID_ASIMD     = (1 << 18),
	CPUID_TLS       = (1 << 19),
	CPUID_AES       = (1 << 20),
	CPUID_PMULL     = (1 << 21),
	CPUID_SHA1      = (1 << 22),
	CPUID_SHA2      = (1 << 23),
	CPUID_CRC32     = (1 << 24),
	CPUID_IWMMXT    = (1 << 25),
	CPUID_IDIVT     = (1 << 26),
	CPUID_IDIVA     = (1 << 27),
	CPUID_VFP3D16   = (1 << 28),
	CPUID_VFP3      = (1 << 29),
	CPUID_VFP4      = (1 << 30),
	CPUID_NEON      = (1 << 31)
};
#endif

unsigned long LOCAL_PREFIX(cpuid)(void);

/* runtime dispatching based on current cpu */
typedef struct cpu_specific_impl_t {
	unsigned long cpu_flags;
	const char *desc;
	/* additional information, pointers to methods, etc... */
} cpu_specific_impl_t;

typedef int (*impl_test)(const void *impl);

const void *LOCAL_PREFIX(cpu_select)(const void *impls, size_t impl_size, impl_test test_fn);

#if defined(__cplusplus)
}
#endif

#endif /* CPUID_CHACHA_H */