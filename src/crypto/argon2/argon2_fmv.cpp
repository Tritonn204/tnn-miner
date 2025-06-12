#include "argon2_core.h"
#include "argon2.h"

#ifdef __x86_64__
__attribute__((target("default")))
#endif
void argon2_slice_fmv_dispatch(const argon2_instance_t* instance, argon2_position_t position) {
  randomx_argon2_fill_segment_ref(instance, position);
}

#ifdef __x86_64__
__attribute__((target("default")))
#endif
void argon2_finalize_fmv_dispatch(const argon2_instance_t* instance, uint8_t* out, size_t outlen) {
  argon2_finalize_ref(instance, out, outlen);
}

#ifdef __x86_64__
__attribute__((target("ssse3")))
void argon2_slice_fmv_dispatch(const argon2_instance_t* instance, argon2_position_t position) {
  randomx_argon2_impl_ssse3()(instance, position);
}

__attribute__((target("ssse3")))
void argon2_finalize_fmv_dispatch(const argon2_instance_t* instance, uint8_t* out, size_t outlen) {
  argon2_finalize_ssse3(instance, out, outlen);
}

#ifndef TNN_LEGACY_AMD64
__attribute__((target("avx2")))
void argon2_slice_fmv_dispatch(const argon2_instance_t* instance, argon2_position_t position) {
  argon2_fill_segment_avx2(instance, position);
}

__attribute__((target("avx2")))
void argon2_finalize_fmv_dispatch(const argon2_instance_t* instance, uint8_t* out, size_t outlen) {
  argon2_finalize_avx2(instance, out, outlen);
}

__attribute__((target("avx512f,avx512bw")))
void argon2_slice_fmv_dispatch(const argon2_instance_t* instance, argon2_position_t position) {
  randomx_argon2_impl_avx512()(instance, position);
}

__attribute__((target("avx512f,avx512bw")))
void argon2_finalize_fmv_dispatch(const argon2_instance_t* instance, uint8_t* out, size_t outlen) {
  argon2_finalize_avx512(instance, out, outlen);
}
#endif
#endif

void argon2_slice_fmv(const argon2_instance_t* instance, argon2_position_t position) {
  argon2_slice_fmv_dispatch(instance, position);
}

void argon2_finalize_fmv(const argon2_instance_t* instance, uint8_t* out, size_t outlen) {
  argon2_finalize_fmv_dispatch(instance, out, outlen);
}