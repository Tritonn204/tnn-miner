#include "sha3.h"
#include <libkeccak/libkeccak.h>

void sha3_256(uint8_t *in, int len, uint8_t *output)
{
  struct libkeccak_spec spec;
  libkeccak_spec_sha3(&spec, 256);

  struct libkeccak_state state;
  libkeccak_state_initialise(&state, &spec); // 256-bit for SHA3-256

  libkeccak_fast_update(&state, in, len); // Absorb input data

  libkeccak_fast_digest(&state, NULL, 0, 0, LIBKECCAK_SHA3_SUFFIX, output); // Produce the hash

  libkeccak_state_destroy(&state);
}