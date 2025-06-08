#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <malloc.h>

#include "rinhash.h"

#include "terminal.h"
#include <crypto/tiny-keccak/tiny-keccak.h>
#include "argon2/argon2.h"
#include "argon2/argon2_core.h"
#include <openssl/evp.h>

static inline int blake2b_openssl(const void* input, size_t input_len, void* output, size_t output_len) {
    if (output_len > 64) {
        return 0; // OpenSSL BLAKE2b only supports up to 64 bytes output
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx == NULL) return 0;

    const EVP_MD* md = EVP_blake2b512();
    if (md == NULL) {
        EVP_MD_CTX_free(ctx);
        return 0;
    }

    if (!EVP_DigestInit_ex(ctx, md, NULL)) {
        EVP_MD_CTX_free(ctx);
        return 0;
    }

    // Note: OpenSSL's BLAKE2b doesn't allow setting output length directly in the legacy API.
    // If needed, truncate manually.
    if (!EVP_DigestUpdate(ctx, input, input_len)) {
        EVP_MD_CTX_free(ctx);
        return 0;
    }

    uint8_t full_digest[64];
    unsigned int len = 0;

    if (!EVP_DigestFinal_ex(ctx, full_digest, &len)) {
        EVP_MD_CTX_free(ctx);
        return 0;
    }

    EVP_MD_CTX_free(ctx);
    memcpy(output, full_digest, output_len);
    return 1;
}

namespace RinHash {
  typedef struct rin_context_holder{
    block* memory;
    argon2_context argon;
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

  static inline void sha3_256_rin(uint8_t *scratch) {
    scratch[32] = 0x06;
    scratch[135] = 0x80;

    keccakf(scratch);
  }

  void hash(void* state, const void* input, const blake3_hasher* prehashedPrefix)
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

      // Allocate memory for Argon2
      rin_ctx->memory = (block*) aligned_malloc(64 * sizeof(block), 64);  // m_cost = 64
      if (!rin_ctx->memory) {
        setcolor(RED);
        fprintf(stderr, "Failed to allocate Argon2 memory\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        memset(state, 0, 32);
        return;
      }

      // Set up Argon2 context
      argon2_context context = {0};
      rin_ctx->argon = context;

      rin_ctx->argon.outlen = 32;
      rin_ctx->argon.pwdlen = 32;
      rin_ctx->argon.salt = (uint8_t*)salt_str;
      rin_ctx->argon.saltlen = strlen(salt_str);
      rin_ctx->argon.t_cost = 2;
      rin_ctx->argon.m_cost = 64;
      rin_ctx->argon.lanes = 1;
      rin_ctx->argon.threads = 1;
      rin_ctx->argon.version = ARGON2_VERSION_13;
      rin_ctx->argon.allocate_cbk = NULL;
      rin_ctx->argon.free_cbk = NULL;
      rin_ctx->argon.flags = ARGON2_DEFAULT_FLAGS;
    }

    uint8_t blake3_out[32];
    blake3_hasher blake;
    memcpy(&blake, prehashedPrefix, sizeof(blake3_hasher));
    blake3_hasher_update(&blake, (const uint8_t*)input + 64, 16);
    blake3_hasher_finalize(&blake, (uint8_t*)blake3_out, 32);

    rin_ctx->argon.pwd = blake3_out;

    argon2_instance_t instance;
    argon2_context* ctx = &rin_ctx->argon;

    // Attach memory to instance
    instance.memory = rin_ctx->memory;
    instance.memory_blocks = 64; // must match m_cost
    instance.lanes = 1;
    instance.threads = 1;
    instance.version = ctx->version;
    instance.passes = ctx->t_cost;
    instance.segment_length = 64 / 4;
    instance.lane_length = 64;
    instance.type = Argon2_d;
    instance.print_internals = 0;
    instance.context_ptr = ctx;
    instance.impl = NULL;

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
        argon2_slice_fmv(&instance, pos);
      }
    }

    uint8_t scratch[200] = {0};
    // blake2b_openssl(instance.memory[instance.lane_length - 1].v, ARGON2_BLOCK_SIZE, scratch, 32);
    argon2_finalize_fmv(&instance, scratch, 32);
    sha3_256_rin(scratch);

    memcpy(state, scratch, 32);
  }
}