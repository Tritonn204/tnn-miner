#pragma once

#include <stddef.h>
#include <inttypes.h>
#include <array>
#include <functional>
#include <immintrin.h>

#define CHACHA_STATE_WORDS 16
#define PAR_BLOCKS 4
#define CHACHA_N PAR_BLOCKS / 2

namespace chacha20
{
  using byte = unsigned char;

  constexpr uint32_t CONSTANTS[4] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
  constexpr size_t STATE_WORDS = 16;

  using Block = std::array<uint8_t, 64>;

  // Function pointer or std::function for the callback
  template <typename F>
  using StreamClosure = std::function<void(F &&f)>;

  static inline uint32_t pack4(const uint8_t *a)
  {
    return uint32_t(a[0] << 0 * 8) |
           uint32_t(a[1] << 1 * 8) |
           uint32_t(a[2] << 2 * 8) |
           uint32_t(a[3] << 3 * 8);
  }

  typedef struct Backend_AVX2
  {
    __m256i v[3];
    __m256i ctr[CHACHA_N];
    unsigned int R;
  } Backend_AVX2;

  template <unsigned int R_>
  void rounds();

  template <typename F>
  void inner(uint32_t *state, F &&f);

  template <unsigned int R>
  void gen_ks_block(Backend_AVX2 &self, byte *block);
  template <unsigned int R>
  void gen_par_ks_blocks(Backend_AVX2 &self, byte *blocks);

  template <unsigned int R>
  class ChaChaCore
  {
  public:
    ChaChaCore(uint8_t *key, byte *iv)
    {
      state[0] = CONSTANTS[0];
      state[1] = CONSTANTS[1];
      state[2] = CONSTANTS[2];
      state[3] = CONSTANTS[3];

      state[4] = pack4(key + 0 * 4);
      state[5] = pack4(key + 1 * 4);
      state[6] = pack4(key + 2 * 4);
      state[7] = pack4(key + 3 * 4);
      state[8] = pack4(key + 4 * 4);
      state[9] = pack4(key + 5 * 4);
      state[10] = pack4(key + 6 * 4);
      state[11] = pack4(key + 7 * 4);
      // 32-bit counter
      state[12] = 0;
      // 96-bit nonce
      state[13] = pack4(iv + 0 * 4);
      state[14] = pack4(iv + 1 * 4);
      state[15] = pack4(iv + 2 * 4);
    }

    __attribute__((target("avx2")))
    void process_block(Block &block)
    {
      inner<decltype([this](Backend_AVX2& backend) {})>(state, [this, &block](Backend_AVX2 &backend)
               { gen_ks_block<R>(backend, block.data()); });
    }

    __attribute__((target("avx2")))
    void process_blocks(Block *blocks, size_t count)
    {
      inner<decltype([this](Backend_AVX2& backend) {})>(state, [this, blocks, count](Backend_AVX2 &backend)
               { gen_par_ks_blocks<R>(backend, reinterpret_cast<uint8_t *>(blocks)); });
    }

    uint32_t get_block_pos() const
    {
      return state[12];
    }
    void seek(uint32_t pos)
    {
      state[12] = pos;
    }

    __attribute__((target("avx2")))
    void apply_keystream(uint8_t *data, size_t len)
    {
      size_t i = 0;
      while (i + PAR_BLOCKS * 64 <= len)
      {
        std::array<Block, PAR_BLOCKS> blocks;
        process_blocks(blocks.data(), PAR_BLOCKS);

        for (size_t j = 0; j < PAR_BLOCKS; ++j)
        {
          __m256i *data_ptr = reinterpret_cast<__m256i *>(data + i + j * 64);
          __m256i *block_ptr = reinterpret_cast<__m256i *>(blocks[j].data());

          data_ptr[0] = _mm256_xor_si256(data_ptr[0], block_ptr[0]);
          data_ptr[1] = _mm256_xor_si256(data_ptr[1], block_ptr[1]);
        }

        i += PAR_BLOCKS * 64;
      }

      while (i + 64 <= len)
      {
        Block block;
        process_block(block);

        __m256i *data_ptr = reinterpret_cast<__m256i *>(data + i);
        __m256i *block_ptr = reinterpret_cast<__m256i *>(block.data());

        data_ptr[0] = _mm256_xor_si256(data_ptr[0], block_ptr[0]);
        data_ptr[1] = _mm256_xor_si256(data_ptr[1], block_ptr[1]);

        i += 64;
      }

      if (i < len)
      {
        Block block;
        process_block(block);

        for (size_t j = 0; j < len - i; ++j)
        {
          data[i + j] ^= block[j];
        }
      }
    }

  private:
    uint32_t state[STATE_WORDS];
  };

  using ChaCha20 = ChaChaCore<10>;
}