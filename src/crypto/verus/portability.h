#ifndef PORT_H
#define PORT_H
#include <immintrin.h>
#include <x86intrin.h>

typedef __m128i u128;

#ifdef __APPLE__
#include <sys/types.h>
#endif// APPLE
#ifdef _WIN32
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