#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef _MSC_VER
#include <x86intrin.h>
#endif

#include "compile.h"

#define ChaCha20StateSizeBytes 48;
#define ChaCha20KeySizeByte 32
#define ChaCha20NonceSizeByte 12
#define ChaCha20CounterSizeByte 4

void ChaCha20SetKey(uint8_t * state, const uint8_t *Key);
void ChaCha20SetNonce(uint8_t * state, const uint8_t *Nonce);
void ChaCha20SetCtr(uint8_t * state, const uint8_t *Ctr);

#define TNN_TARGETS_X86_CHACHA512 \
    "avx512f,avx512dq,avx512bw", TNN_FEATURES_ZNVER4, TNN_FEATURES_ZNVER5

#include "simd/simd_utils.h"

#if defined(__x86_64__)

TNN_TARGET_CLONE(
    ChaCha20EncryptBytes,
    void,
    (uint8_t *state, uint8_t *In, uint8_t *Out, uint64_t Size, int rounds),
    ;,
    "sse2", "avx2", TNN_TARGETS_X86_CHACHA512
)

// Temporal - default callsite
TNN_TARGET_CLONE(
    ChaCha20EncryptXelis,
    void,
    (
        const uint8_t keys[4][32],
        const uint8_t nonces[4][12],
        uint8_t* outputs[4],
        size_t bytes_per_stream,
        int rounds),
    ;,
    "ssse3", "avx2"
)

// Non-temporal - explicit callsite opt-in
TNN_TARGET_CLONE(
    ChaCha20EncryptXelisNT,
    void,
    (
        const uint8_t keys[4][32],
        const uint8_t nonces[4][12],
        uint8_t* outputs[4],
        size_t bytes_per_stream,
        int rounds),
    ;,
    "ssse3", "avx2"
)

#endif // __x86_64__

#if defined(__x86_64__)
__attribute__((target("default")))
#endif
void ChaCha20EncryptBytes(
    uint8_t *state, uint8_t *In, uint8_t *Out,
    const uint64_t Size, int rounds);

#if defined(__x86_64__)
__attribute__((target("default")))
#endif
void ChaCha20EncryptXelis(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    const size_t bytes_per_stream,
    int rounds);

#if defined(__x86_64__)
__attribute__((target("default")))
#endif
void ChaCha20EncryptXelisNT(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    const size_t bytes_per_stream,
    int rounds);

void ChaCha20IncrementNonce(uint8_t *state);

static inline void ChaCha20AddCounter(uint8_t* ChaCha, const uint32_t value_to_add)
{
    uint32_t* State32bits = (uint32_t*)ChaCha;
    State32bits[8] += value_to_add;
}

// ============================================================================
// CALLSITE HELPER - lets callers use policy as template param
// while FMV dispatch still works correctly underneath
// ============================================================================
template<StorePolicy POLICY = StorePolicy::TEMPORAL>
static inline void ChaCha20EncryptXelisPoly(
    const uint8_t keys[4][32],
    const uint8_t nonces[4][12],
    uint8_t* outputs[4],
    size_t bytes_per_stream,
    int rounds)
{
    if constexpr (POLICY == StorePolicy::NON_TEMPORAL) {
        ChaCha20EncryptXelisNT(keys, nonces, outputs, bytes_per_stream, rounds);
    } else {
        ChaCha20EncryptXelis(keys, nonces, outputs, bytes_per_stream, rounds);
    }
}