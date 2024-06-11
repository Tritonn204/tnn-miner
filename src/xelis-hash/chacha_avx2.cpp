#include "chacha20.hpp"
#include <inttypes.h>
#include <immintrin.h>
#include <array>

using namespace chacha20;

template <typename F>
__attribute__((target("avx2")))
inline void inner(uint32_t *state, F &&f)
{
  const __m128i *state_ptr = reinterpret_cast<const __m128i *>(state);
  __m256i v[3] = {
      _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr + 0)),
      _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr + 1)),
      _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr + 2))};
  __m256i c = _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr + 3));
  c = _mm256_add_epi32(c, _mm256_set_epi32(0, 0, 0, 1, 0, 0, 0, 0));
  __m256i ctr[CHACHA_N];
  for (int i = 0; i < CHACHA_N; ++i)
  {
    ctr[i] = c;
    c = _mm256_add_epi32(c, _mm256_set_epi32(0, 0, 0, 2, 0, 0, 0, 2));
  }
  Backend_AVX2 backend = {
      {v[0], v[1], v[2]},
      {ctr[0], ctr[1]},
      0};

  f(backend);

  state[12] = static_cast<uint32_t>(_mm256_extract_epi32(backend.ctr[0], 0));
}

// Helper function: add-xor-rotate
__attribute__((target("avx2")))
inline void add_xor_rot(std::array<std::array<__m256i, 4>, CHACHA_N> vs)
{
  const __m256i rol16_mask = _mm256_set_epi64x(
      0x0d0c0f0e09080b0a,
      0x0504070601000302,
      0x0d0c0f0e09080b0a,
      0x0504070601000302);
  const __m256i rol8_mask = _mm256_set_epi64x(
      0x0e0d0c0f0a09080b,
      0x0605040702010003,
      0x0e0d0c0f0a09080b,
      0x0605040702010003);

  for (int i = 0; i < CHACHA_N; ++i)
  {
    __m256i &a = vs[i][0];
    __m256i &b = vs[i][1];
    __m256i &c = vs[i][2];
    __m256i &d = vs[i][3];

    a = _mm256_add_epi32(a, b);
    d = _mm256_xor_si256(d, a);
    d = _mm256_shuffle_epi8(d, rol16_mask);

    c = _mm256_add_epi32(c, d);
    b = _mm256_xor_si256(b, c);
    b = _mm256_xor_si256(_mm256_slli_epi32(b, 12), _mm256_srli_epi32(b, 20));

    a = _mm256_add_epi32(a, b);
    d = _mm256_xor_si256(d, a);
    d = _mm256_shuffle_epi8(d, rol8_mask);

    c = _mm256_add_epi32(c, d);
    b = _mm256_xor_si256(b, c);
    b = _mm256_xor_si256(_mm256_slli_epi32(b, 7), _mm256_srli_epi32(b, 25));
  }
}

__attribute__((target("avx2")))
__attribute__((always_inline)) inline void rows_to_cols(std::array<std::array<__m256i, 4>, CHACHA_N> vs)
{
  for (int i = 0; i < CHACHA_N; ++i)
  {
    vs[i][2] = _mm256_shuffle_epi32(vs[i][2], _MM_SHUFFLE(0, 3, 2, 1));
    vs[i][3] = _mm256_shuffle_epi32(vs[i][3], _MM_SHUFFLE(1, 0, 3, 2));
    vs[i][0] = _mm256_shuffle_epi32(vs[i][0], _MM_SHUFFLE(2, 1, 0, 3));
  }
}

// Helper function: cols_to_rows
__attribute__((target("avx2")))
__attribute__((always_inline)) inline void cols_to_rows(std::array<std::array<__m256i, 4>, CHACHA_N> vs)
{
  for (int i = 0; i < CHACHA_N; ++i)
  {
    vs[i][2] = _mm256_shuffle_epi32(vs[i][2], _MM_SHUFFLE(2, 1, 0, 3));
    vs[i][3] = _mm256_shuffle_epi32(vs[i][3], _MM_SHUFFLE(1, 0, 3, 2));
    vs[i][0] = _mm256_shuffle_epi32(vs[i][0], _MM_SHUFFLE(0, 3, 2, 1));
  }
}

// Helper function: double quarter round
__attribute__((target("avx2")))
inline void double_quarter_round(std::array<std::array<__m256i, 4>, CHACHA_N> v)
{
  add_xor_rot(v);
  rows_to_cols(v);
  add_xor_rot(v);
  cols_to_rows(v);
}

// Helper function: perform R rounds on the state
template <byte R>
__attribute__((target("avx2")))
inline std::array<std::array<__m256i, 4>, CHACHA_N> rounds(__m256i *v, __m256i *ctr)
{
  std::array<std::array<__m256i, 4>, CHACHA_N> vs;
  for (int i = 0; i < CHACHA_N; i++)
  {
    vs[i] = {v[0], v[1], v[2], ctr[i]};
  }

  for (int i = 0; i < R; i += 2)
  {
    double_quarter_round(vs);
  }

  for (int i = 0; i < CHACHA_N; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      vs[i][j] = _mm256_add_epi32(vs[i][j], v[j]);
    }
    vs[i][3] = _mm256_add_epi32(vs[i][3], ctr[i]);
  }

  return vs;
}

// Generate a single keystream block
template <unsigned int R>
__attribute__((target("avx2")))
inline void gen_ks_block(Backend_AVX2 &self, byte *block)
{
  auto res = rounds<R>(self.v, self.ctr);
  for (int i = 0; i < CHACHA_N; ++i)
  {
    self.ctr[i] = _mm256_add_epi32(self.ctr[i], _mm256_set_epi32(0, 0, 0, 1, 0, 0, 0, 1));
  }

  __m128i *res0 = reinterpret_cast<__m128i *>(&res[0]);
  __m128i *block_ptr = reinterpret_cast<__m128i *>(block);
  for (int i = 0; i < 4; ++i)
  {
    _mm_storeu_si128(block_ptr + i, res0[2 * i]);
  }
}

// Generate keystream blocks in parallel
template <unsigned int R>
__attribute__((target("avx2")))
inline void gen_par_ks_blocks(Backend_AVX2 &self, byte *blocks)
{
  auto vs = rounds<R>(self.v, self.ctr);

  const int pb = PAR_BLOCKS;
  for (int i = 0; i < CHACHA_N; ++i)
  {
    self.ctr[i] = _mm256_add_epi32(self.ctr[i], _mm256_set_epi32(0, 0, 0, pb, 0, 0, 0, pb));
  }

  __m128i *block_ptr = reinterpret_cast<__m128i *>(blocks);
  for (const auto &v : vs)
  {
    __m128i *t = reinterpret_cast<__m128i *>(&v);
    for (int i = 0; i < 4; ++i)
    {
      _mm_storeu_si128(block_ptr + i, t[2 * i]);
      _mm_storeu_si128(block_ptr + 4 + i, t[2 * i + 1]);
    }
    block_ptr += 8;
  }
}