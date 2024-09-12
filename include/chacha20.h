
#pragma once
#include <stdint.h>

#ifdef _MSC_VER
#include <x86intrin.h>
#endif

#define ChaCha20StateSizeBytes 48;
#define ChaCha20KeySizeByte 32
#define ChaCha20NonceSizeByte 12
#define ChaCha20CounterSizeByte 4

#ifdef __cplusplus
extern "C" {
#endif

void ChaCha20SetKey(uint8_t * state, const uint8_t *Key);
void ChaCha20SetNonce(uint8_t * state, const uint8_t *Nonce);
void ChaCha20SetCtr(uint8_t * state, const uint8_t *Ctr);

#if defined(__x86_64__)
  __attribute__((target("sse2"))) void ChaCha20EncryptBytes(uint8_t * state, uint8_t * In, uint8_t * Out, const uint64_t Size, int rounds); //if In=nullptr - just fill Out
  __attribute__((target("avx2"))) void ChaCha20EncryptBytes(uint8_t * state, uint8_t * In, uint8_t * Out, const uint64_t Size, int rounds); //if In=nullptr - just fill Out
  __attribute__((target("avx512f,avx512dq,avx512bw"))) void ChaCha20EncryptBytes(uint8_t * state, uint8_t * In, uint8_t * Out, const uint64_t Size, int rounds); //if In=nullptr - just fill Out
#endif
// We do this to avoid needing Clang 16+ when building for AArch
#if defined(__x86_64__)
__attribute__((target("default")))
#endif
void ChaCha20EncryptBytes(uint8_t * state, uint8_t * In, uint8_t * Out, const uint64_t Size, int rounds); //if In=nullptr - just fill Out

void ChaCha20IncrementNonce(uint8_t * state);

inline void ChaCha20AddCounter(uint8_t* ChaCha, const uint32_t value_to_add)
{
	uint32_t* State32bits = (uint32_t*)ChaCha;
	State32bits[8]+=value_to_add;
}

#ifdef __cplusplus
}
#endif