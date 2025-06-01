#pragma once
#include <stdint.h>

#ifdef _MSC_VER
#include <x86intrin.h>
#endif

#define Salsa20StateSizeBytes 64
#define Salsa20BlockSizeBytes 64

#ifdef __cplusplus
extern "C" {
#endif

void Salsa20SetState(uint8_t* state, const uint8_t* input);
void Salsa20GetState(const uint8_t* state, uint8_t* output);

#if defined(__x86_64__)
void Salsa20Transform(uint8_t* state, int rounds);
void Salsa20Transform4(uint8_t* state0, uint8_t* state1, uint8_t* state2, uint8_t* state3, int rounds);
#endif

// Integration functions for yespower
void salsa20_core(uint8_t* state, int rounds);
void salsa20_core_4way(uint8_t* state0, uint8_t* state1, uint8_t* state2, uint8_t* state3, int rounds);

#ifdef __cplusplus
}
#endif