#pragma once

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#define htobe64(x) OSSwapHostToBigInt64(x)
#define be64toh(x) OSSwapBigToHostInt64(x)
#elif defined(__linux__)
#include <endian.h>
#elif defined(_WIN32)
#include <winsock2.h>
#define htobe64(x) htonll(x)
#define be64toh(x) ntohll(x)
#else
// Fallback for other platforms
uint64_t htobe64(uint64_t value)
{
  return (((value & 0x00000000000000FFULL) << 56) |
          ((value & 0x000000000000FF00ULL) << 40) |
          ((value & 0x0000000000FF0000ULL) << 24) |
          ((value & 0x00000000FF000000ULL) << 8) |
          ((value & 0x000000FF00000000ULL) >> 8) |
          ((value & 0x0000FF0000000000ULL) >> 24) |
          ((value & 0x00FF000000000000ULL) >> 40) |
          ((value & 0xFF00000000000000ULL) >> 56));
}
uint64_t be64toh(uint64_t value)
{
  return htobe64(value);
}
#endif