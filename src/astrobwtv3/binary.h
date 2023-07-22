#ifndef BINARY
#define BINARY

#include <iostream>
#include <vector>
#include <stdint.h>

inline const int MaxVarintLen16 = 3, MaxVarintLen32 = 5, MaxVarintLen64 = 10;

inline int PutUvarint(unsigned char* buf, uint64_t x) {
    int i = 0;
    while (x >= 0x80) {
        buf[i] = static_cast<unsigned char>(x) | 0x80;
        x >>= 7;
        i++;
    }
    buf[i] = static_cast<unsigned char>(x);
    return i + 1;
}

inline int PutVarint(unsigned char* buf, int64_t x) {
    uint64_t ux = static_cast<uint64_t>(x) << 1;
    if (x < 0) {
        ux = ~ux;
    }
    return PutUvarint(buf, ux);
}

#endif