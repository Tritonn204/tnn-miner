#ifndef ENDIAN
#define ENDIAN

inline bool littleEndian()
{
  int n = 1;
  // little endian if true
  if(*(char *)&n == 1) {
    return true;
  } 
  return false;
}

#include <cstdint>

inline uint32_t htobe32_portable(uint32_t host_32bits) {
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return ((host_32bits & 0x000000FF) << 24) |
               ((host_32bits & 0x0000FF00) << 8)  |
               ((host_32bits & 0x00FF0000) >> 8)  |
               ((host_32bits & 0xFF000000) >> 24);
    #else
        return host_32bits;  // Already big-endian
    #endif
}

inline uint32_t be32toh_portable(uint32_t big_endian_32bits) {
    return htobe32_portable(big_endian_32bits);  // Same operation
}

inline uint32_t le32dec(const uint8_t* data) {
    return ((uint32_t)data[0]) |
           ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) |
           ((uint32_t)data[3] << 24);
}

inline uint32_t be32dec(const uint8_t* data) {
    return ((uint32_t)data[3]) |
           ((uint32_t)data[2] << 8) |
           ((uint32_t)data[1] << 16) |
           ((uint32_t)data[0] << 24);
}

inline void le32enc(uint8_t* p, uint32_t x) {
    p[0] = x & 0xff;
    p[1] = (x >> 8) & 0xff;
    p[2] = (x >> 16) & 0xff;
    p[3] = (x >> 24) & 0xff;
}

inline void be32enc(uint8_t* p, uint32_t x) {
    p[3] = x & 0xff;
    p[2] = (x >> 8) & 0xff;
    p[1] = (x >> 16) & 0xff;
    p[0] = (x >> 24) & 0xff;
}

#endif