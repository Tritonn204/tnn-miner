#include <string.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__APPLE__) || defined(__MACH__)
  #include <stdlib.h>
#else
  #include <malloc.h>
#endif

#include "rinhash.h"

#include "terminal.h"
#include <crypto/tiny-keccak/tiny-keccak.h>
#include "argon2/argon2.h"
#include "argon2/argon2_core.h"
#include "blake2/blake2.h"
#include <openssl/evp.h>
#include "hex.h"

#define CHECK_EQUAL(mem1, mem2, len, label) \
  do { \
    if (memcmp(mem1, mem2, len) != 0) { \
      fprintf(stderr, "%s mismatch!\n", label); \
      print_hex("Expected", mem2, len); \
      print_hex("Actual", mem1, len); \
      fflush(stdout); \
    } \
  } while (0)

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
    uint8_t scratch[200];
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

  inline void print_hex(const char* label, const void* data, size_t len) {
    printf("%s: ", label);
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < len; ++i)
      printf("%02x", bytes[i]);
    printf("\n");
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
      rin_ctx->memory = (block*) aligned_malloc(64 * ARGON2_BLOCK_SIZE, 64);  // m_cost = 64
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

    randomx_argon2_initialize(&instance, &(rin_ctx->argon));

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

    const uint32_t last_block_offset = instance.lane_length - 1;
    const block* last_block = &(instance.memory[last_block_offset]);

    blake2b_long(rin_ctx->scratch, 32, (uint8_t*)last_block->v, ARGON2_BLOCK_SIZE);

    memset(rin_ctx->scratch + 32, 0, 200-32);
    sha3_256_rin(rin_ctx->scratch);

    memcpy(state, rin_ctx->scratch, 32);
  }

  void test() {
    const char* input_hex =
      "00000020a14ae141585f60301ed2dafab2a65045d5d12afa0b232233a0065fa7"
      "01000000a710d54064e2a2728c2f435a8a7593dacd261d9418d1b69dd0be6ecd9"
      "91c78efa413456838dd011d00000000";

    const uint8_t expected_final[32] = {
      0xa5, 0x7b, 0x72, 0x38, 0xb9, 0xab, 0x81, 0x8a,
      0x25, 0xcc, 0xcd, 0xb5, 0x68, 0x34, 0x07, 0x20,
      0x76, 0x37, 0xbe, 0x79, 0xcc, 0x5c, 0x58, 0x1d,
      0x8b, 0x16, 0x72, 0xb6, 0xda, 0x89, 0x72, 0x92
    };

    uint8_t input[80];
    hexstrToBytes(input_hex, input);
    print_hex("HASH INPUT", input, 80);

    uint8_t state[32];
    blake3_hasher prehashed;
    blake3_hasher_init(&prehashed);
    blake3_hasher_update(&prehashed, input, 64);  // Prehash only the first 64 bytes

    hash(state, input, &prehashed);  // Call your actual hash pipeline
    memset(state, 0, 32);
    hash(state, input, &prehashed);

    print_hex("FINAL Output", state, 32);
    CHECK_EQUAL(state, expected_final, 32, "Final SHA3-256 Output");
  }
}

#undef CHECK_EQUAL