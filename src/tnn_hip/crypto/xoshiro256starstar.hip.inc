/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include <stdint.h>

/* This is xoshiro256** 1.0, one of our all-purpose, rock-solid
   generators. It has excellent (sub-ns) speed, a state (256 bits) that is
   large enough for any parallel application, and it passes all tests we
   are aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

class Xoshiro256PlusPlusHasher
{
private:
  uint64_t s[4];

public:
  Xoshiro256PlusPlusHasher(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3)
  {
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
  }

  uint64_t next()
  {
    uint64_t tmp = s[0] + s[3];
    const uint64_t result = (tmp << 23 | tmp >> 41) + s[0];

    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> 19);

    return result;
  }
};