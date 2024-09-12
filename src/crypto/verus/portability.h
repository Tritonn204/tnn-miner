#ifndef PORT_H
#define PORT_H
#include <immintrin.h>
#include <x86intrin.h>

typedef __m128i u128;

#ifdef __APPLE__
#include <sys/types.h>
#endif// APPLE
#ifdef _WIN32
#ifdef _MSC_VER
#define bit_AVX  (1 << 28)  // AVX is bit 28 in ECX (for cpuid leaf 1)
#define bit_AES  (1 << 25)  // AES is bit 25 in ECX (for cpuid leaf 1)
#define bit_PCLMUL (1 << 1)  // PCLMUL is bit 1 in ECX (for cpuid leaf 1)
#endif
#include <intrin.h>
#else
#include <cpuid.h>
#endif // !WIN32

#include "verus_clhash.h"


#ifdef __cplusplus
extern "C"
{
#include "haraka.h"
}
#else
#include "haraka.h"
#endif
#endif /* PORT_H */