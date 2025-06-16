#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <cinttypes>

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

  uint64_t simple_xor64(const uint8_t* data, size_t len) {
    uint64_t acc = 0;
    for (size_t i = 0; i < len; i += 8) {
      uint64_t v;
      memcpy(&v, data + i, sizeof(v));
      acc ^= v;
    }
    return acc;
  }

  int test() {
    const char* input_hex =
      "00000020a14ae141585f60301ed2dafab2a65045d5d12afa0b232233a0065fa7"
      "01000000a710d54064e2a2728c2f435a8a7593dacd261d9418d1b69dd0be6ecd9"
      "91c78efa413456838dd011d00000000";

    const uint8_t expect_step1[64] = {
        0x6a,0x24,0x2e,0x9b,0x10,0x53,0x10,0x2f,
        0xf8,0x22,0x16,0x7b,0x5d,0x46,0x81,0x4a,
        0xad,0x3d,0xda,0x70,0x38,0x8c,0x9f,0x08,
        0x77,0xb6,0xb4,0xf0,0xfc,0xd7,0xea,0x07,
        0xe9,0x10,0x88,0xb2,0xaa,0xb2,0x9d,0x00,
        0xdf,0xa5,0x0e,0x2f,0x5b,0xc5,0x58,0x1e,
        0xad,0xcf,0xff,0xc4,0xa1,0x02,0x4e,0x16,
        0x54,0x94,0x4c,0x24,0x76,0xc0,0x26,0x8b
    };
    const uint8_t expect_step2[32] = {
        0x59,0x26,0xbf,0x5b,0x91,0x8f,0x95,0x59,
        0x24,0x70,0x64,0x00,0xdd,0xa5,0x36,0x0a,
        0xa1,0x23,0x65,0xbb,0x57,0xcf,0xf5,0xa4,
        0x2d,0x68,0x67,0x16,0xa7,0x45,0x44,0x23
    };
    const uint8_t expect_step5[32] = {
        0x87,0xba,0x90,0xf2,0xa5,0x3a,0x91,0xfd,
        0x60,0x03,0x71,0x1f,0x1c,0xbe,0x1d,0x53,
        0x15,0x00,0xa5,0x77,0x10,0xee,0x06,0x3e,
        0x9c,0x78,0x9f,0x1b,0x3e,0xac,0xb6,0x0d
    };
    const uint8_t expect_step6[32] = {
        0xa5,0x7b,0x72,0x38,0xb9,0xab,0x81,0x8a,
        0x25,0xcc,0xcd,0xb5,0x68,0x34,0x07,0x20,
        0x76,0x37,0xbe,0x79,0xcc,0x5c,0x58,0x1d,
        0x8b,0x16,0x72,0xb6,0xda,0x89,0x72,0x92
    };

    uint8_t input[80];
    hexstrToBytes(input_hex, input);
    print_hex("INPUT", input, 80);

    // STEP 1: prehash
    blake3_hasher blake;
    blake3_hasher_init(&blake);
    blake3_hasher_update(&blake, input, 64);
    uint8_t step1[64];
    blake3_hasher_finalize(&blake, step1, 64);
    print_hex("STEP1: BLAKE3(64)", step1, 64);
    if (memcmp(step1, expect_step1, 64) != 0) {
        fprintf(stderr, "Error: STEP1 mismatch!\n");
        return 0;
    }

    // STEP 2: extend +16
    blake3_hasher_update(&blake, input + 64, 16);
    uint8_t step2[32];
    blake3_hasher_finalize(&blake, step2, 32);
    print_hex("STEP2: BLAKE3(+16)", step2, 32);
    if (memcmp(step2, expect_step2, 32) != 0) {
        fprintf(stderr, "Error: STEP2 mismatch!\n");
        return 0;
    }

    if (rin_ctx == NULL) {
      rin_ctx = (rin_context_holder*) aligned_malloc(sizeof(rin_context_holder), 64);
      if (!rin_ctx) {
        setcolor(RED);
        fprintf(stderr, "Failed to allocate rin_ctx\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        return 1;
      }

      const char* salt_str = "RinCoinSalt";

      // Allocate memory for Argon2
      rin_ctx->memory = (block*) aligned_malloc(64 * ARGON2_BLOCK_SIZE, 64);  // m_cost = 64
      if (!rin_ctx->memory) {
        setcolor(RED);
        fprintf(stderr, "Failed to allocate Argon2 memory\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        return 1;
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
    // Step 3: Argon2 initialize
    rin_ctx->argon.pwd = step2;

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

    randomx_argon2_initialize(&instance, &rin_ctx->argon);

    // Step 4: Argon2 rounds (2 passes × 4 slices)
    for (uint32_t pass = 0; pass < 2; ++pass) {
      for (uint32_t slice = 0; slice < 4; ++slice) {
        argon2_position_t pos = { pass, 0, (uint8_t)slice, 0 };
        argon2_slice_fmv(&instance, pos);
      }
      // Optionally print a checkpoint block
      uint64_t* checkpoint = (uint64_t*) instance.memory[instance.lane_length / 2].v;
      uint8_t* cp = (uint8_t*) checkpoint;
    }

    const block* last_block = &instance.memory[instance.lane_length - 1];
    const uint8_t* blk_data = (const uint8_t*)last_block->v;
    const size_t blk_size = ARGON2_BLOCK_SIZE;

    uint64_t checksum = simple_xor64(blk_data, blk_size);
    printf("Last block XOR64 checksum: 0x%016" PRIx64 "\n", checksum);

    const uint64_t expect_checksum = 0x60e250323f152f72ULL;
    if (checksum != expect_checksum) {
      setcolor(RED);
      fprintf(stderr, "ERROR: Last-block checksum mismatch!\n");
      fflush(stderr);
      setcolor(BRIGHT_WHITE);
      return 1;
    }

    // Step 5: BLAKE2b on last block
    uint8_t step5[32];
    blake2b_long(step5, 32, (uint8_t*)last_block->v, ARGON2_BLOCK_SIZE);
    print_hex("STEP5: BLAKE2b(last block)", step5, 32);
    if (memcmp(step5, expect_step5, 32) != 0) {
      setcolor(RED);
      fprintf(stderr, "Error: STEP5 mismatch!\n");
      fflush(stderr);
      setcolor(BRIGHT_WHITE);
      return 1;
    }

    // Step 6: SHA3-256
    memcpy(rin_ctx->scratch, step5, 32);
    memset(rin_ctx->scratch + 32, 0, 200 - 32);
    sha3_256_rin(rin_ctx->scratch);
    uint8_t step6[32];
    memcpy(step6, rin_ctx->scratch, 32);
    print_hex("STEP6: SHA3-256", step6, 32);
    if (memcmp(step6, expect_step6, 32) != 0) {
      setcolor(RED);
      fprintf(stderr, "Error: STEP6 mismatch!\n");
      fflush(stderr);
      setcolor(BRIGHT_WHITE);
      return 1;
    }

    printf("✅ All steps match expected output.\n");
    return 0;
  }
}

#undef CHECK_EQUAL