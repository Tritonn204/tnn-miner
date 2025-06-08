#pragma once

#include <BLAKE3/c/blake3.h>

namespace RinHash {
  inline bool checkHash(uint8_t *hash_bytes, uint32_t *targetWords) {
      uint32_t *hash_words = (uint32_t*)hash_bytes;
      
      for (int i = 7; i >= 0; i--) {
          if (hash_words[i] > targetWords[i])
              return false;
          if (hash_words[i] < targetWords[i])
              return true;
      }
      
      return true;
  }

  void hash(void* state, const void* input, const blake3_hasher* prehashedPrefix);
}