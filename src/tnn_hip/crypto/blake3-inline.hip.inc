#pragma once

#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_BUF_CAP 384
#define BLAKE3_BUF_LEN 326

#ifdef __HIP_PLATFORM_AMD__
#define ssize_t size_t
#endif

#define IV_0 0x6A09E667UL
#define IV_1 0xBB67AE85UL
#define IV_2 0x3C6EF372UL
#define IV_3 0xA54FF53AUL
#define IV_4 0x510E527FUL
#define IV_5 0x9B05688CUL
#define IV_6 0x1F83D9ABUL
#define IV_7 0x5BE0CD19UL

#define CHUNK_START (1 << 0)
#define CHUNK_END (1 << 1)
#define ROOT (1 << 3)

#define REF_Z00 0
#define REF_Z01 1
#define REF_Z02 2
#define REF_Z03 3
#define REF_Z04 4
#define REF_Z05 5
#define REF_Z06 6
#define REF_Z07 7
#define REF_Z08 8
#define REF_Z09 9
#define REF_Z0A 10
#define REF_Z0B 11
#define REF_Z0C 12
#define REF_Z0D 13
#define REF_Z0E 14
#define REF_Z0F 15
#define REF_Z10 2
#define REF_Z11 6
#define REF_Z12 3
#define REF_Z13 10
#define REF_Z14 7
#define REF_Z15 0
#define REF_Z16 4
#define REF_Z17 13
#define REF_Z18 1
#define REF_Z19 11
#define REF_Z1A 12
#define REF_Z1B 5
#define REF_Z1C 9
#define REF_Z1D 14
#define REF_Z1E 15
#define REF_Z1F 8
#define REF_Z20 3
#define REF_Z21 4
#define REF_Z22 10
#define REF_Z23 12
#define REF_Z24 13
#define REF_Z25 2
#define REF_Z26 7
#define REF_Z27 14
#define REF_Z28 6
#define REF_Z29 5
#define REF_Z2A 9
#define REF_Z2B 0
#define REF_Z2C 11
#define REF_Z2D 15
#define REF_Z2E 8
#define REF_Z2F 1
#define REF_Z30 10
#define REF_Z31 7
#define REF_Z32 12
#define REF_Z33 9
#define REF_Z34 14
#define REF_Z35 3
#define REF_Z36 13
#define REF_Z37 15
#define REF_Z38 4
#define REF_Z39 0
#define REF_Z3A 11
#define REF_Z3B 2
#define REF_Z3C 5
#define REF_Z3D 8
#define REF_Z3E 1
#define REF_Z3F 6
#define REF_Z40 12
#define REF_Z41 13
#define REF_Z42 9
#define REF_Z43 11
#define REF_Z44 15
#define REF_Z45 10
#define REF_Z46 14
#define REF_Z47 8
#define REF_Z48 7
#define REF_Z49 2
#define REF_Z4A 5
#define REF_Z4B 3
#define REF_Z4C 0
#define REF_Z4D 1
#define REF_Z4E 6
#define REF_Z4F 4
#define REF_Z50 9
#define REF_Z51 14
#define REF_Z52 11
#define REF_Z53 5
#define REF_Z54 8
#define REF_Z55 12
#define REF_Z56 15
#define REF_Z57 1
#define REF_Z58 13
#define REF_Z59 3
#define REF_Z5A 0
#define REF_Z5B 10
#define REF_Z5C 2
#define REF_Z5D 6
#define REF_Z5E 4
#define REF_Z5F 7
#define REF_Z60 11
#define REF_Z61 15
#define REF_Z62 5
#define REF_Z63 0
#define REF_Z64 1
#define REF_Z65 9
#define REF_Z66 8
#define REF_Z67 6
#define REF_Z68 14
#define REF_Z69 10
#define REF_Z6A 2
#define REF_Z6B 12
#define REF_Z6C 3
#define REF_Z6D 4
#define REF_Z6E 7
#define REF_Z6F 13

__forceinline__ __device__ void cv_state_init(uint32_t *cv)
{
  cv[0] = IV_0;
  cv[1] = IV_1;
  cv[2] = IV_2;
  cv[3] = IV_3;
  cv[4] = IV_4;
  cv[5] = IV_5;
  cv[6] = IV_6;
  cv[7] = IV_7;
}

__forceinline__ __device__ void blake3_compress_in_place(uint32_t cv[8],
                                                         const uint8_t block[BLAKE3_BLOCK_LEN],
                                                         uint8_t block_len,
                                                         uint8_t flags);

__forceinline__ __device__ void chunk_state_update_single(uint32_t cv[8], uint8_t *input, const size_t initial_len)
{
  memset(input + initial_len, 0, 64 - initial_len);
  blake3_compress_in_place(cv, input, initial_len, CHUNK_START | CHUNK_END | ROOT);
}

__forceinline__ __device__ void chunk_state_update_80(uint32_t cv[8], uint8_t *input)
{
  memset(input + 80, 0, 128 - 80);
  blake3_compress_in_place(cv, input, 64, CHUNK_START);
  blake3_compress_in_place(cv, input + 80, 16, CHUNK_END | ROOT);
}

__forceinline__ __device__ void chunk_state_update(uint32_t cv[8], uint8_t *input, const size_t initial_len)
{
  ssize_t input_len = initial_len;

  while (input_len > 0)
  {
    ssize_t take = input_len >= BLAKE3_BLOCK_LEN ? BLAKE3_BLOCK_LEN : input_len;

    uint8_t maybe_start_flag = input_len == initial_len ? CHUNK_START : 0;
    input_len -= take;
    uint8_t maybe_end_flag = 0;
    if (input_len == 0)
    {
      maybe_end_flag = CHUNK_END | ROOT;
      memset(input + take, 0, BLAKE3_BLOCK_LEN - take);
    }

    blake3_compress_in_place(cv, input, take, maybe_start_flag | maybe_end_flag);
    input += take;
  }
}

__forceinline__ __device__ static uint32_t rotr32(const uint32_t w, const uint32_t c)
{
  return (w >> c) | (w << (32 - c));
}

__device__ static __forceinline__ uint32_t ROR16(const uint32_t a, const uint8_t r) {
  return __byte_perm(a, a, 0x5432);
}

__device__ static __forceinline__ uint32_t ROR8(const uint32_t a, const uint8_t r) {
  return __byte_perm(a, a, 0x4321);
}

#define REF_G(a, b, c, d, x, y)                 \
  do                                            \
  {                                             \
    state[a] = state[a] + state[b] + x;         \
    state[d] = ROR16(state[d] ^ state[a], 16); \
    state[c] = state[c] + state[d];             \
    state[b] = rotr32(state[b] ^ state[c], 12); \
    state[a] = state[a] + state[b] + y;         \
    state[d] = ROR8(state[d] ^ state[a], 8);  \
    state[c] = state[c] + state[d];             \
    state[b] = rotr32(state[b] ^ state[c], 7);  \
  } while (0)

#define REF_Mx(r, i) (block_words[REF_Z##r##i])

#define ROUND_S(r)                                         \
  do                                                       \
  {                                                        \
    REF_G(0x0, 0x4, 0x8, 0xC, REF_Mx(r, 0), REF_Mx(r, 1)); \
    REF_G(0x1, 0x5, 0x9, 0xD, REF_Mx(r, 2), REF_Mx(r, 3)); \
    REF_G(0x2, 0x6, 0xA, 0xE, REF_Mx(r, 4), REF_Mx(r, 5)); \
    REF_G(0x3, 0x7, 0xB, 0xF, REF_Mx(r, 6), REF_Mx(r, 7)); \
    REF_G(0x0, 0x5, 0xA, 0xF, REF_Mx(r, 8), REF_Mx(r, 9)); \
    REF_G(0x1, 0x6, 0xB, 0xC, REF_Mx(r, A), REF_Mx(r, B)); \
    REF_G(0x2, 0x7, 0x8, 0xD, REF_Mx(r, C), REF_Mx(r, D)); \
    REF_G(0x3, 0x4, 0x9, 0xE, REF_Mx(r, E), REF_Mx(r, F)); \
  } while (0)

__forceinline__ __device__ void compress_pre(uint32_t state[16], const uint32_t cv[8],
                                             const uint8_t block[BLAKE3_BLOCK_LEN],
                                             uint8_t block_len, uint8_t flags)
{
  uint32_t *block_words = (uint32_t *)block;

  state[0] = cv[0];
  state[1] = cv[1];
  state[2] = cv[2];
  state[3] = cv[3];
  state[4] = cv[4];
  state[5] = cv[5];
  state[6] = cv[6];
  state[7] = cv[7];
  state[8] = IV_0;
  state[9] = IV_1;
  state[10] = IV_2;
  state[11] = IV_3;
  state[12] = 0;
  state[13] = 0;
  state[14] = (uint32_t)block_len;
  state[15] = (uint32_t)flags;

  ROUND_S(0);
  ROUND_S(1);
  ROUND_S(2);
  ROUND_S(3);
  ROUND_S(4);
  ROUND_S(5);
  ROUND_S(6);
}

__forceinline__ __device__ void blake3_compress_in_place(uint32_t cv[8],
                                                         const uint8_t block[BLAKE3_BLOCK_LEN],
                                                         uint8_t block_len,
                                                         uint8_t flags)
{
  uint32_t state[16];
  compress_pre(state, cv, block, block_len, flags);
  cv[0] = state[0] ^ state[8];
  cv[1] = state[1] ^ state[9];
  cv[2] = state[2] ^ state[10];
  cv[3] = state[3] ^ state[11];
  cv[4] = state[4] ^ state[12];
  cv[5] = state[5] ^ state[13];
  cv[6] = state[6] ^ state[14];
  cv[7] = state[7] ^ state[15];

  // printf("== final state: ");
  // for (int i = 0; i < 16; i++) {
  //   printf("%d, ", state[i]);
  // }
  // printf("\n");
  // printf("== final cv: ");
  // for (int i = 0; i < 16; i++) {
  //   printf("%d, ", cv[i]);
  // }
  // printf("\n\n");
}

typedef struct
{
  uint8_t buf[BLAKE3_BUF_CAP];

  uint32_t cv[8];

  uint8_t hash[64]; // 64 bytes needed as hash will used as block words as well

  uint8_t target[32];
  uint32_t from_group;
  uint32_t to_group;

  uint32_t hash_count;
  int found_good_hash;
} blake3_hasher;

__forceinline__ __device__ void blake3_inline_hash_single(uint8_t *input, const size_t input_len, uint8_t *out)
{
  __align__(4) uint32_t cv[8];
  cv_state_init(cv);
  chunk_state_update_single(cv, input, input_len);
#pragma unroll
  for (int i = 0; i < 8; i++)
  {
    ((uint32_t *)out)[i] = cv[i];
  }
}

__forceinline__ __device__ void blake3_inline_hash_80(uint8_t *input, uint8_t *out)
{
  __align__(4) uint32_t cv[8];
  cv_state_init(cv);
  chunk_state_update_80(cv, input);
#pragma unroll
  for (int i = 0; i < 8; i++)
  {
    ((uint32_t *)out)[i] = cv[i];
  }
}

__forceinline__ __device__ void blake3_inline_hash(uint8_t *input, const size_t input_len, uint8_t *out)
{
  __align__(4) uint32_t cv[8];
  cv_state_init(cv);
  chunk_state_update(cv, input, input_len);
#pragma unroll
  for (int i = 0; i < 8; i++)
  {
    ((uint32_t *)out)[i] = cv[i];
  }
}