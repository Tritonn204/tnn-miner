#include "argon2_core.h"
#include "argon2.h"

#ifdef __x86_64__
__attribute__((target("default")))
#endif
void argon2_fmv_dispatch(const argon2_instance_t* instance, argon2_position_t position) {
  randomx_argon2_fill_segment_ref(instance, position);
}

#ifdef __x86_64__
__attribute__((target("ssse3")))
void argon2_fmv_dispatch(const argon2_instance_t* instance, argon2_position_t position) {
  randomx_argon2_impl_ssse3(instance, position);
}

#ifdef __AVX2__ // TODO make #ifndef TNN_LEGACY_AMD64
__attribute__((target("avx2")))
void argon2_fmv_dispatch(const argon2_instance_t* instance, argon2_position_t position) {
  randomx_argon2_impl_avx2(instance, position);
}

__attribute__((target("avx512f,avx512bw")))
void argon2_fmv_dispatch(const argon2_instance_t* instance, argon2_position_t position) {
  randomx_argon2_impl_avx512(instance, position);
}
#endif
#endif