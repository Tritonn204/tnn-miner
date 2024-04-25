#pragma once

#include <openssl/evp.h>
#include <openssl/aes.h>

typedef unsigned char byte;

inline byte* keccak256(const unsigned char* input, size_t input_len) {
  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  if (ctx == NULL) {
      return NULL;
  }

  if (EVP_DigestInit_ex(ctx, EVP_sha3_256(), NULL) != 1) {
      EVP_MD_CTX_free(ctx);
      return NULL;
  }

  if (EVP_DigestUpdate(ctx, input, input_len) != 1) {
      EVP_MD_CTX_free(ctx);
      return NULL;
  }

  byte hash[EVP_MAX_MD_SIZE];
  unsigned int hash_len;
  if (EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
      EVP_MD_CTX_free(ctx);
      return NULL;
  }

  EVP_MD_CTX_free(ctx);

  return hash;
}

inline void aes_round(uint8_t* block, const uint8_t* key) {
  AES_KEY aes_key;
  AES_set_encrypt_key(key, 128, &aes_key);
  uint8_t temp[16];
  AES_encrypt(block, temp, &aes_key);
  std::copy(temp, temp+16, block);
}