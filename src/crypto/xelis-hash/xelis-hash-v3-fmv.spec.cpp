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
// AVX512 tier -- inline 4-stream ChaCha with native rotates (temporal + NT)
// =========================================================================
//@ begin avx512

__attribute__((target("avx512f,avx512dq,avx512bw"))) void stage_1(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_INLINE(ChaCha20EncryptXelis_avx512_inline, StorePolicy::TEMPORAL)
}

__attribute__((target("avx512f,avx512dq,avx512bw"))) void stage_1_nt(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_INLINE(ChaCha20EncryptXelis_avx512_inline, StorePolicy::NON_TEMPORAL)
}

__attribute__((target("avx512f,avx512dq,avx512bw"))) void stage_1_serial(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_SERIAL_BODY
}

__attribute__((target("avx512f,avx512dq,avx512bw"))) void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY
}

__attribute__((target("avx512f,avx512dq,avx512bw,aes"))) void stage_3_hw(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_HW_AES_BODY
}

__attribute__((target("avx512f,avx512dq,avx512bw"))) void stage_3_l2(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY_PF(prefetch_L2)
}

__attribute__((target("avx512f,avx512dq,avx512bw,aes"))) void stage_3_hw_l2(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_HW_AES_BODY_PF(prefetch_L2)
}

__attribute__((target("avx512f,avx512dq,avx512bw"))) void stage_3_sw(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY_SWITCH
}

__attribute__((target("avx512f,avx512dq,avx512bw,aes"))) void stage_3_hw_sw(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_HW_AES_BODY_SWITCH
}

//@ end

// =========================================================================
// AVX2 tier -- inline 4-stream ChaCha (temporal + NT)
// =========================================================================
//@ begin avx2

__attribute__((target("avx2"))) void stage_1(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_INLINE(ChaCha20EncryptXelis_avx2_inline, StorePolicy::TEMPORAL)
}

__attribute__((target("avx2"))) void stage_1_nt(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_INLINE(ChaCha20EncryptXelis_avx2_inline, StorePolicy::NON_TEMPORAL)
}

__attribute__((target("avx2"))) void stage_1_serial(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_SERIAL_BODY
}

__attribute__((target("avx2"))) void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY
}

__attribute__((target("avx2,aes"))) void stage_3_hw(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_HW_AES_BODY
}

__attribute__((target("avx2"))) void stage_3_l2(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY_PF(prefetch_L2)
}

__attribute__((target("avx2,aes"))) void stage_3_hw_l2(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_HW_AES_BODY_PF(prefetch_L2)
}

__attribute__((target("avx2"))) void stage_3_sw(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY_SWITCH
}

__attribute__((target("avx2,aes"))) void stage_3_hw_sw(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_HW_AES_BODY_SWITCH
}

//@ end

// =========================================================================
// AES tier -- inline 4-stream SSSE3 ChaCha (temporal + NT)
// =========================================================================
//@ begin aes

__attribute__((target("aes,sse4.1"))) void stage_1(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_INLINE(ChaCha20EncryptXelis_ssse3_inline, StorePolicy::TEMPORAL)
}

__attribute__((target("aes,sse4.1"))) void stage_1_nt(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_INLINE(ChaCha20EncryptXelis_ssse3_inline, StorePolicy::NON_TEMPORAL)
}

__attribute__((target("aes,sse4.1"))) void stage_1_serial(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_SERIAL_BODY
}

__attribute__((target("aes,sse4.1"))) void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_HW_AES_BODY
}

__attribute__((target("aes,sse4.1"))) void stage_3_l2(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_HW_AES_BODY_PF(prefetch_L2)
}

__attribute__((target("aes,sse4.1"))) void stage_3_sw(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_HW_AES_BODY_SWITCH
}

//@ end

// =========================================================================
// Fallback tier -- FMV dispatch ChaCha (no inline, no guaranteed SIMD)
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

__attribute__((target("default"))) void stage_1_nt(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  stage_1(input, sp, input_len);
}

__attribute__((target("default"))) void stage_1_serial(const uint8_t *input, uint64_t *sp, size_t input_len)
{
  XELIS_STAGE1_SERIAL_BODY
}

__attribute__((target("default"))) void stage_3(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY
}

__attribute__((target("default"))) void stage_3_l2(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY_PF(prefetch_L2)
}

__attribute__((target("default"))) void stage_3_sw(uint64_t *scratch_pad, workerData_xelis_v3 &worker)
{
  XELIS_STAGE3_SOFT_AES_BODY_SWITCH
}

//@ end
