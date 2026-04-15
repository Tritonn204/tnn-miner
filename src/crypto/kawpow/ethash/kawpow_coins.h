#pragma once
// ============================================================================
// kawpow_coins.h — Per-coin Keccak padding for KawPow variants
//
// The KawPow (ProgPoW 0.9.4) Keccak permutation uses coin-specific padding:
//   Phase 1 (seed):  state[10..24] = padding[0..14]   (15 words)
//   Phase 5 (final): state[16..24] = padding[0..8]    ( 9 words)
//
// Each coin that forks KawPow defines its own 15-word padding array.
// This header provides the padding definitions so both GPU (RTC-injected)
// and CPU (reference library) code share a single source of truth.
// ============================================================================

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 15 uint32 words — covers both Phase 1 (all 15) and Phase 5 (first 9)
typedef struct {
    uint32_t words[15];
} kawpow_coin_padding_t;

// --- Ravencoin (RVN) ---
// "rAVENCOINKAWPOW" — note lowercase 'r' matches cpp-kawpow reference
static const kawpow_coin_padding_t KAWPOW_PADDING_RVN = {{
    0x00000072u, // r
    0x00000041u, // A
    0x00000056u, // V
    0x00000045u, // E
    0x0000004Eu, // N
    0x00000043u, // C
    0x0000004Fu, // O
    0x00000049u, // I
    0x0000004Eu, // N
    0x0000004Bu, // K
    0x00000041u, // A
    0x00000057u, // W
    0x00000050u, // P
    0x0000004Fu, // O
    0x00000057u, // W
}};

// --- Add more coins here ---
// static const kawpow_coin_padding_t KAWPOW_PADDING_NEOXA = {{ ... }};
// static const kawpow_coin_padding_t KAWPOW_PADDING_MEOW  = {{ ... }};

#ifdef __cplusplus
}
#endif
