#include "common.hip.h"

// __device__ static size_t
// encode_left(struct hipkeccak_state_optimized *state, uint8_t *buf, size_t byterate, size_t value, size_t off)
// {
//     size_t x, n, j, i = off;

//     for (x = value, n = 0; x; x >>= 8)
//         n += 1;
//     if (!n)
//         n = 1;
//     buf[i++] = static_cast<uint8_t>(n);  // Explicit cast to uint8_t in C++
//     if (i == byterate) {
//         libkeccak_zerocopy_update_ref(state, buf, byterate);
//         i = 0;
//     }

//     for (j = 0; j < n;) {
//         buf[i++] = static_cast<uint8_t>(value >> ((n - ++j) << 3));  // Explicit cast to uint8_t
//         if (i == byterate) {
//             libkeccak_zerocopy_update_ref(state, buf, byterate);
//             i = 0;
//         }
//     }

//     return i;
// }

// __device__ static size_t
// encode_left_shifted(struct hipkeccak_state_optimized *state, uint8_t *buf, size_t byterate, size_t value, size_t off, size_t bitoff)
// {
//     size_t x, n, j, i = off;
//     uint16_t v;

//     for (x = value, n = 0; x; x >>= 8)
//         n += 1;
//     if (!n)
//         n = 1;
//     v = static_cast<uint16_t>((n & 255UL) << bitoff);  // Explicit cast to uint16_t
//     buf[i++] |= static_cast<uint8_t>(v);               // Explicit cast to uint8_t
//     if (i == byterate) {
//         libkeccak_zerocopy_update_ref(state, buf, byterate);
//         i = 0;
//     }
//     buf[i] = static_cast<uint8_t>(n >> 8);  // Explicit cast

//     for (j = 0; j < n;) {
//         v = static_cast<uint16_t>(((value >> ((n - ++j) << 3)) & 255UL) << bitoff);  // Explicit cast to uint16_t
//         buf[i++] |= static_cast<uint8_t>(v);                                          // Explicit cast to uint8_t
//         if (i == byterate) {
//             libkeccak_zerocopy_update_ref(state, buf, byterate);
//             i = 0;
//         }
//         buf[i] = static_cast<uint8_t>(v >> 8);  // Explicit cast
//     }

//     return i;
// }


__device__ void hipkeccak_cshake_initialise_kas(struct hipkeccak_state_optimized *state, // Remove restrict
                                 uint8_t *s_text, size_t s_len)
{
  int threadId = threadIdx.x;  // Thread index within the block (x-dimension)
  int blockId = blockIdx.x;    // Block index within the grid (x-dimension)

  size_t off = 0, bitoff = 0;
  size_t byterate = static_cast<size_t>(1088) >> 3;

  if (!0 && !s_len && !0 && !0)
    return;

  uint32_t mIndex = threadId*CSHAKE256_MLEN;

  // // Function calls with adjusted parameters
  // off = encode_left(state, static_cast<uint8_t *>(&state->M[mIndex]), byterate, byterate, off); // Explicit cast for state->M
  // off = encode_left(state, static_cast<uint8_t *>(&state->M[mIndex]), byterate, 0, off);
  // off = feed_text(state, static_cast<uint8_t *>(&state->M[mIndex]), "", 0, 0, "", off, byterate, &bitoff);

  // off = encode_left(state, static_cast<uint8_t *>(&state->M[mIndex]), byterate, (s_len << 3), off);
  // off = feed_text(state, static_cast<uint8_t *>(&state->M[mIndex]), s_text, s_len, 0, "", off, byterate, &bitoff);

  // if (off)
  // {
  //   memset(&state->M[mIndex + off], 0, CSHAKE256_MLEN - off);
  //   hipkeccak_zerocopy_update_ref(state, static_cast<uint8_t *>(&state->M[mIndex]), CSHAKE256_MLEN);
  // }

  // printf("final off = %llu\n", off);
}