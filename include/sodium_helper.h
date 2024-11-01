#pragma once
#include <sodium.h>

#define libsodium_checkInit                                      \
  if (sodium_init() < 0)                                         \
  {                                                              \
    throw std::runtime_error("Failed to initialize libsodium."); \
  }