#pragma once
#include <stdint.h>
#include <string.h>

#include <crypto/tiny-keccak/tiny-keccak.h>
#include <crypto/keccak4/keccak4.h>
#include "compile.h"

inline void keccak1600(const uint8_t *input, size_t inlen, uint8_t *output)
{
  uint8_t state[200] = {0};

  memcpy(state, input, inlen);

  state[inlen] ^= 0x06;
  state[135] ^= 0x80;

  keccakf(state);

  memcpy(output, state, 200);
}

inline void keccakf(uint64_t *state, int rounds)
{
  ::keccakf((void *)state);
}

#ifdef __x86_64__
#ifndef TNN_LEGACY_AMD64
TNN_TARGET_CLONE(
  keccak1600_4way,
  inline void,
  (const uint8_t *input0, const uint8_t *input1,
                            const uint8_t *input2, const uint8_t *input3,
                            uint8_t *output0, uint8_t *output1,
                            uint8_t *output2, uint8_t *output3),
  {
    KeccakP1600times4_states states;
    KeccakP1600times4_InitializeAll(&states);

    KeccakP1600times4_AddBytes(&states, 0, input0, 0, 64);
    KeccakP1600times4_AddBytes(&states, 1, input1, 0, 64);
    KeccakP1600times4_AddBytes(&states, 2, input2, 0, 64);
    KeccakP1600times4_AddBytes(&states, 3, input3, 0, 64);

    uint8_t pad1 = 0x06;
    uint8_t pad2 = 0x80;
    for (int i = 0; i < 4; i++)
    {
      KeccakP1600times4_AddBytes(&states, i, &pad1, 64, 1);
      KeccakP1600times4_AddBytes(&states, i, &pad2, 135, 1);
    }

    KeccakP1600times4_PermuteAll_24rounds(&states);

    KeccakP1600times4_ExtractBytes(&states, 0, output0, 0, 200);
    KeccakP1600times4_ExtractBytes(&states, 1, output1, 0, 200);
    KeccakP1600times4_ExtractBytes(&states, 2, output2, 0, 200);
    KeccakP1600times4_ExtractBytes(&states, 3, output3, 0, 200);
  },
  TNN_TARGETS_X86_AVX2              
)

#endif
__attribute__((target("default")))
#endif
inline void keccakf_4way(uint64_t *state0, uint64_t *state1,
                         uint64_t *state2, uint64_t *state3)
{
  KeccakP1600times4_states states;

  uint64_t *s = (uint64_t *)states.A;
  for (int i = 0; i < 25; i++)
  {
    s[i * 4 + 0] = state0[i];
    s[i * 4 + 1] = state1[i];
    s[i * 4 + 2] = state2[i];
    s[i * 4 + 3] = state3[i];
  }

  KeccakP1600times4_PermuteAll_24rounds(&states);

  for (int i = 0; i < 25; i++)
  {
    state0[i] = s[i * 4 + 0];
    state1[i] = s[i * 4 + 1];
    state2[i] = s[i * 4 + 2];
    state3[i] = s[i * 4 + 3];
  }
}
#endif