#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <malloc.h>

#include "terminal.h"
#include <BLAKE3/c/blake3.h>
#include <crypto/tiny-keccak/tiny-keccak.h>
#include "argon2/argon2.h"
#include "argon2/argon2_core.h"

namespace RinHash {
  typedef struct rin_context_holder{
    blake3_hasher blake;
    argon2_context argon;
    uint8_t blake3_out[32];
    uint8_t argon2_out[32];
  } rin_context_holder;

  thread_local rin_context_holder* rin_ctx;

  #ifdef _WIN32
      #include <malloc.h>
      #define aligned_malloc _aligned_malloc
      #define aligned_free   _aligned_free
  #else
  void* aligned_malloc(size_t size, size_t alignment) {
      void* ptr = NULL;
      if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
      return ptr;
  }

  void aligned_free(void* ptr) {
      free(ptr);
  }
  #endif

  static inline void sha3_256_rin(uint8_t *in, uint8_t *scratch) {
    scratch[32] = 0x06;
    scratch[135] = 0x80;

    keccakf(scratch);
  }

  void hash(void* state, const void* input)
  {
    if (rin_ctx == NULL) {
      rin_ctx = (rin_context_holder*) aligned_malloc(sizeof(rin_context_holder), 64);
      if (!rin_ctx) {
        setcolor(RED);
        fprintf(stderr, "Failed to allocate rin_ctx\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);

        memset(state, 0, 32);
        return;
      }
      const char* salt_str = "RinCoinSalt";

      argon2_context context = {0};
      (*rin_ctx).argon = context;

      (*rin_ctx).argon.outlen = 32;
      (*rin_ctx).argon.pwd = (*rin_ctx).blake3_out;
      (*rin_ctx).argon.out = (*rin_ctx).argon2_out;
      (*rin_ctx).argon.pwdlen = 32;
      (*rin_ctx).argon.salt = (uint8_t*)salt_str;
      (*rin_ctx).argon.saltlen = strlen(salt_str);
      (*rin_ctx).argon.t_cost = 2;
      (*rin_ctx).argon.m_cost = 64;
      (*rin_ctx).argon.lanes = 1;
      (*rin_ctx).argon.threads = 1;
      (*rin_ctx).argon.version = ARGON2_VERSION_13;
      (*rin_ctx).argon.allocate_cbk = NULL;
      (*rin_ctx).argon.free_cbk = NULL;
      (*rin_ctx).argon.flags = ARGON2_DEFAULT_FLAGS;
    }

    blake3_hasher_init(&rin_ctx->blake);
    blake3_hasher_update(&rin_ctx->blake, input, 80);
    blake3_hasher_finalize(&rin_ctx->blake, (*rin_ctx).blake3_out, 32);

    argon2_instance_t instance;
    argon2_context* ctx = &rin_ctx->argon;

    if (randomx_argon2_initialize(&instance, ctx) != ARGON2_OK) {
      setcolor(RED);
      fprintf(stderr, "Argon2 initialize failed!\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      memset(state, 0, 32);
      return;
    }

    #pragma unroll 2
    for (uint32_t pass = 0; pass < 2; ++pass) {
      #pragma unroll 4
      for (uint8_t slice = 0; slice < 4; ++slice) {
        argon2_position_t pos = {
          .pass = pass,
          .lane = 0,
          .slice = slice,
          .index = 0
        };
        argon2_fmv_dispatch(&instance, pos);
      }
    }
    
    sha3_256_rin((*rin_ctx).blake3_out, (uint8_t *)instance.memory[0].v);
    memcpy(state, (*rin_ctx).blake3_out, 32);
  }

  // int scanhash_rinhash(struct work *work, uint32_t max_nonce,
  //     uint64_t *hashes_done, struct thr_info *mythr)
  // {
  //     uint32_t *pdata = work->data;
  //     uint32_t *ptarget = work->target;
  //     uint32_t n = pdata[19] - 1;
  //     const uint32_t first_nonce = pdata[19];
  //     int thr_id = mythr->id;
  //     uint8_t hash[32];

  //     do {
  //         n++;
  //         pdata[19] = n;

  //         rinhash(hash, pdata);
  //         uint32_t hash32[8];

  //         // 安全に変換（リトルエンディアン）
  //         for (int i = 0; i < 8; i++) {
  //             hash32[i] = ((uint32_t)hash[i*4 + 0]) |
  //                         ((uint32_t)hash[i*4 + 1] << 8) |
  //                         ((uint32_t)hash[i*4 + 2] << 16) |
  //                         ((uint32_t)hash[i*4 + 3] << 24);
  // }
  //         if (fulltest(hash32, ptarget)) {
  //             submit_solution(work, hash, mythr);
  //             break;
  //         }
  //     } while (n < max_nonce && !work_restart[thr_id].restart);

  //     pdata[19] = n;
  //     *hashes_done = n - first_nonce + 1;
  //     return 0;
  // }


  // // Register algorithm
  // bool register_rin_algo( algo_gate_t* gate )
  // {
  //   gate->scanhash = (void*)&scanhash_rinhash;
  //   gate->hash = (void*)&rinhash;
  //   gate->optimizations = SSE2_OPT | AVX2_OPT | AVX512_OPT;
  //   gate->build_stratum_request = (void*)&std_be_build_stratum_request;
  //   return true;
  // }
}