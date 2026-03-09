//@ spec xelis_v3_fmv
//@ section xelis

//@ tier aes aes,sse4.1
//@ tier avx2 avx2
//@ tier avx512 avx512f,avx512dq,avx512bw
//@ tier fallback default

//@ common
#include "xelis-hash/xelis-hash-v3-internal.hpp"
//@ end

// =========================================================================
// AVX512 tier -- single-stream ChaCha
// =========================================================================
//@ begin avx512

TNN_TARGET_CLONE(
    stage_1,
    void,
    (const uint8_t *input, uint64_t *sp, size_t input_len),
    {
      constexpr size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3;
      uint8_t *t = reinterpret_cast<uint8_t *>(sp);
      uint8_t K2_values[4][32];
      uint8_t nonces[4][12];

      stage_1_derive(input, input_len, K2_values, nonces);

      for (int i = 0; i < 4; i++)
      {
        uint8_t state[48] = {0};
        ChaCha20SetKey(state, K2_values[i]);
        ChaCha20SetNonce(state, nonces[i]);
        ChaCha20EncryptBytes(state, NULL, t + i * bytes_per_chunk, bytes_per_chunk, 8);
      }
    },
    TNN_TARGETS_X86_CHACHA512)

__attribute__((target("avx512f,avx512dq,avx512bw"))) void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY
}

__attribute__((target("avx512f,avx512dq,avx512bw"))) void stage_3_modpow(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  stage_3(scratch_pad, worker);
}

__attribute__((target("avx512f,avx512dq,avx512bw"))) void stage_3_modpow_switch(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  stage_3(scratch_pad, worker);
}

//@ end

// =========================================================================
// AVX2 tier -- 4-stream ChaCha (ssse3 + avx2 shared body)
// =========================================================================
//@ begin avx2

TNN_TARGET_CLONE(
    stage_1,
    void,
    (const uint8_t *input, uint64_t *sp, size_t input_len),
    {
      constexpr size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3;
      uint8_t *t = reinterpret_cast<uint8_t *>(sp);
      uint8_t K2_values[4][32];
      uint8_t nonces[4][12];

      stage_1_derive(input, input_len, K2_values, nonces);

      byte *outputs[4];
      for (int i = 0; i < 4; i++)
      {
        outputs[i] = t + i * bytes_per_chunk;
      }
      ChaCha20EncryptXelis(K2_values, nonces, outputs, bytes_per_chunk, 8);
    },
    "ssse3", "avx2")

__attribute__((target("avx2"))) void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY
}

__attribute__((target("avx2"))) void stage_3_modpow(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  stage_3(scratch_pad, worker);
}

__attribute__((target("avx2"))) void stage_3_modpow_switch(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  stage_3(scratch_pad, worker);
}

//@ end

// =========================================================================
// AES tier
// =========================================================================
//@ begin aes

__attribute__((target("aes,sse4.1"))) void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  constexpr uint8_t key[17] = "xelishash-pow-v3";
  __m128i key_vec = _mm_loadu_si128((const __m128i *)key);

  uint64_t *__restrict mem_buffer_a = scratch_pad;
  uint64_t *__restrict mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V3;

  uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
  uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;
  size_t r = 0;

  for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; ++i)
  {
    prefetch_L2(&mem_buffer_a[map_index(addr_a)]);
    uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
    prefetch_L2(&mem_buffer_b[map_index(mem_a ^ addr_b)]);
    uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];

    __m128i block_vec = _mm_set_epi64x(mem_a, mem_b);
    block_vec = _mm_aesenc_si128(block_vec, key_vec);
    uint64_t hash1 = _mm_extract_epi64(block_vec, 0);
    uint64_t hash2 = _mm_extract_epi64(block_vec, 1);
    uint64_t result = ~(hash1 ^ hash2);

    for (size_t j = 0; j < XELIS_BUFFER_SIZE_V3; ++j)
    {
      uint64_t rot_res = ~ROTR(result, r);
      uint64_t a = mem_buffer_a[map_index(result)];
      uint64_t b = mem_buffer_b[map_index(a ^ rot_res)];
      uint64_t c = scratch_pad[r];
      r++;

      uint32_t op_raw = ROTL(result, (uint32_t)c);
      uint64_t v = execute_operation(op_raw, a, b, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      int use_b = pick_half(v);
      uint64_t idx_t = map_index(idx_seed);
      uint64_t t = (use_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

      uint64_t idx_a = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
      uint64_t idx_b = map_index(idx_a ^ ~result ^ XELIS_SCATTER_CONST);

      uint64_t mem_a_tmp = mem_buffer_a[idx_a];
      mem_buffer_a[idx_a] = t;
      mem_buffer_b[idx_b] ^= mem_a_tmp ^ ROTR(t, i + j);
    }

    uint64_t addr_a_next = modular_power(addr_a, addr_b, result);
    uint64_t addr_b_next = isqrt(result) * (r + 1) * isqrt(addr_a_next);
    prefetch_L2(&mem_buffer_a[map_index(addr_a_next)]);
    addr_a = addr_a_next;
    addr_b = addr_b_next;
  }
}

// Modpow variant templated on dispatch strategy
template <OpDispatch D = OpDispatch::Goto>
__attribute__((target("aes,sse4.1"))) void stage_3_modpow_impl(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  constexpr uint8_t key[17] = "xelishash-pow-v3";
  __m128i key_vec = _mm_loadu_si128((const __m128i *)key);

  uint64_t *__restrict mem_buffer_a = scratch_pad;
  uint64_t *__restrict mem_buffer_b = scratch_pad + XELIS_BUFFER_SIZE_V3;

  uint64_t addr_a = mem_buffer_b[XELIS_BUFFER_SIZE_V3 - 1];
  uint64_t addr_b = mem_buffer_a[XELIS_BUFFER_SIZE_V3 - 1] >> 32;
  size_t r = 0;

  for (size_t i = 0; i < XELIS_SCRATCHPAD_ITERS_V3; ++i)
  {
    prefetch_L2(&mem_buffer_a[map_index(addr_a)]);
    uint64_t mem_a = mem_buffer_a[map_index(addr_a)];
    prefetch_L2(&mem_buffer_b[map_index(mem_a ^ addr_b)]);
    uint64_t mem_b = mem_buffer_b[map_index(mem_a ^ addr_b)];

    __m128i block_vec = _mm_set_epi64x(mem_a, mem_b);
    block_vec = _mm_aesenc_si128(block_vec, key_vec);
    uint64_t hash1 = _mm_extract_epi64(block_vec, 0);
    uint64_t hash2 = _mm_extract_epi64(block_vec, 1);
    uint64_t result = ~(hash1 ^ hash2);

    for (size_t j = 0; j < XELIS_BUFFER_SIZE_V3; ++j)
    {
      uint64_t rot_res = ~ROTR(result, r);
      uint64_t a = mem_buffer_a[map_index(result)];
      uint64_t b = mem_buffer_b[map_index(a ^ rot_res)];
      uint64_t c = scratch_pad[r];
      r++;

      uint32_t op_raw = ROTL(result, (uint32_t)c);
      uint64_t v = execute_operation<D>(op_raw, a, b, c, r, result, i, j);

      uint64_t idx_seed = v ^ result;
      result = ROTL(idx_seed, r);

      int use_b = pick_half(v);
      uint64_t idx_t = map_index(idx_seed);
      uint64_t t = (use_b ? mem_buffer_b[idx_t] : mem_buffer_a[idx_t]) ^ result;

      uint64_t idx_a = map_index(t ^ result ^ XELIS_GOLDEN_RATIO);
      uint64_t idx_b = map_index(idx_a ^ ~result ^ XELIS_SCATTER_CONST);

      uint64_t mem_a_tmp = mem_buffer_a[idx_a];
      mem_buffer_a[idx_a] = t;
      mem_buffer_b[idx_b] ^= mem_a_tmp ^ ROTR(t, i + j);
    }

    uint64_t addr_a_next = modular_power_fast(addr_a, addr_b, result);
    uint64_t addr_b_next = isqrt(result) * (r + 1) * isqrt(addr_a_next);
    prefetch_L2(&mem_buffer_a[map_index(addr_a_next)]);
    addr_a = addr_a_next;
    addr_b = addr_b_next;
  }
}

__attribute__((target("aes,sse4.1"))) void stage_3_modpow(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  stage_3_modpow_impl<OpDispatch::Goto>(scratch_pad, worker);
}

__attribute__((target("aes,sse4.1"))) void stage_3_modpow_switch(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  stage_3_modpow_impl<OpDispatch::Switch>(scratch_pad, worker);
}

//@ end

// =========================================================================
// Fallback tier
// =========================================================================
//@ begin fallback

__attribute__((target("default"))) void stage_1(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  constexpr size_t bytes_per_chunk = XELIS_BYTES_PER_CHUNK_V3;
  uint8_t *t = reinterpret_cast<uint8_t *>(sp);
  uint8_t K2_values[4][32];
  uint8_t nonces[4][12];

  stage_1_derive(input, input_len, K2_values, nonces);

  byte *outputs[4];
  for (int i = 0; i < 4; i++)
  {
    outputs[i] = t + i * bytes_per_chunk;
  }
  ChaCha20EncryptXelis(K2_values, nonces, outputs, bytes_per_chunk, 8);
}

__attribute__((target("default"))) void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY
}

__attribute__((target("default"))) void stage_3_modpow(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  stage_3(scratch_pad, worker);
}

__attribute__((target("default"))) void stage_3_modpow_switch(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  stage_3(scratch_pad, worker);
}

//@ end
