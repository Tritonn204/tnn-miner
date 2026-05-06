//@ spec wolf_compute
//@ section astrobwt
//@ guard __x86_64__

//@ tier avx512bitalg avx512f,avx512bw,avx512vl,avx512bitalg,avx512vbmi,bmi2
//@ tier avx512 avx512f,avx512bw,avx512vl,bmi2
//@ tier avx2 avx2
//@ tier fallback default

//@ common
#include "astrobwtv3/wolfbranching_internal.hpp"
//@ end

// =========================================================================
// AVX-512 BITALG+VBMI tier -- native vpopcntb + kmask blend
// =========================================================================
//@ begin avx512bitalg

/* Override popcnt to use native AVX-512 BITALG instruction */
#undef  WOLF_POPCNT256_EPI8
#define WOLF_POPCNT256_EPI8(x) _mm256_popcnt_epi8(x)

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bitalg,avx512vbmi,bmi2")))
void copyChunkData(workerData &worker, int start, int end) {
  for (int i = start; i + 63 < end; i += 64) {
    __m512i prev_data = _mm512_loadu_si512((__m512i*)&worker.prev_chunk[i]);
    _mm512_storeu_si512((__m512i*)&worker.chunk[i], prev_data);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bitalg,avx512vbmi,bmi2")))
void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker)
{
  uint32_t Opcode = CodeLUT_16[op];
  __m256i data = _mm256_loadu_si256((__m256i *)&in[pos1]);
  __m256i old = data;

  WOLF_BRANCH_SIMD(data, in[pos2], Opcode);

  /* AVX-512 kmask blend: much cheaper than genMask_avx2 + blendv */
  __mmask32 k = _bzhi_u32(0xFFFFFFFFu, (uint32_t)(pos2 - pos1));
  data = _mm256_mask_blend_epi8(k, old, data);

  _mm256_storeu_si256((__m256i *)&out[pos1], data);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bitalg,avx512vbmi,bmi2")))
void wolfCompute(workerData &worker, bool isTest, int wIndex)
{
  WOLF_COMPUTE_BODY(wolfPermute, copyChunkData)
}

/* Restore default popcnt for subsequent tiers if re-included */
#undef  WOLF_POPCNT256_EPI8
#define WOLF_POPCNT256_EPI8(x) popcnt256_epi8(x)

//@ end

// =========================================================================
// AVX-512 tier -- kmask blend, LUT popcnt
// =========================================================================
//@ begin avx512

__attribute__((target("avx512f,avx512bw,avx512vl,bmi2")))
void copyChunkData(workerData &worker, int start, int end) {
  for (int i = start; i + 63 < end; i += 64) {
    __m512i prev_data = _mm512_loadu_si512((__m512i*)&worker.prev_chunk[i]);
    _mm512_storeu_si512((__m512i*)&worker.chunk[i], prev_data);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,bmi2")))
void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker)
{
  uint32_t Opcode = CodeLUT_16[op];
  __m256i data = _mm256_loadu_si256((__m256i *)&in[pos1]);
  __m256i old = data;

  WOLF_BRANCH_SIMD(data, in[pos2], Opcode);

  /* AVX-512 kmask blend */
  __mmask32 k = _bzhi_u32(0xFFFFFFFFu, (uint32_t)(pos2 - pos1));
  data = _mm256_mask_blend_epi8(k, old, data);

  _mm256_storeu_si256((__m256i *)&out[pos1], data);
}

__attribute__((target("avx512f,avx512bw,avx512vl,bmi2")))
void wolfCompute(workerData &worker, bool isTest, int wIndex)
{
  WOLF_COMPUTE_BODY(wolfPermute, copyChunkData)
}

//@ end

// =========================================================================
// AVX2 tier -- genMask_avx2 + blendv (current production path)
// =========================================================================
//@ begin avx2

__attribute__((target("avx2")))
void copyChunkData(workerData &worker, int start, int end) {
  for (int i = start; i < end; i += 32) {
    __m256i prev_data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[i]);
    _mm256_storeu_si256((__m256i*)&worker.chunk[i], prev_data);
  }
}

__attribute__((target("avx2")))
void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker)
{
  uint32_t Opcode = CodeLUT_16[op];
  __m256i data = _mm256_loadu_si256((__m256i *)&in[pos1]);
  __m256i old = data;

  WOLF_BRANCH_SIMD(data, in[pos2], Opcode);

  data = _mm256_blendv_epi8(old, data, genMask_avx2(pos2 - pos1));
  _mm256_storeu_si256((__m256i *)&out[pos1], data);
}

__attribute__((target("avx2")))
void wolfCompute(workerData &worker, bool isTest, int wIndex)
{
  WOLF_COMPUTE_BODY(wolfPermute, copyChunkData)
}

//@ end

// =========================================================================
// Fallback tier -- scalar wolfBranch loop
// =========================================================================
//@ begin fallback

__attribute__((target("default")))
void copyChunkData(workerData &worker, int start, int end) {
  std::copy_n(&worker.prev_chunk[start], end - start, &worker.chunk[start]);
}

__attribute__((target("default")))
void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker)
{
  uint32_t Opcode = CodeLUT[op];

  for (int i = pos1; i < pos2; ++i)
  {
    out[i] = wolfBranch(in[i], in[pos2], Opcode);
  }
}

__attribute__((target("default")))
void wolfCompute(workerData &worker, bool isTest, int wIndex)
{
  WOLF_COMPUTE_BODY(wolfPermute, copyChunkData)
}

//@ end
