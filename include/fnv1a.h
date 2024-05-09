#pragma once
#include <stdint.h>

//fnv1a 32 and 64 bit hash functions
// key is the data to hash, len is the size of the data (or how much of it to hash against)
// code license: public domain or equivalent
// post: https://notes.underscorediscovery.com/constexpr-fnv1a/

inline uint32_t hash_32_fnv1a(const void* key, const uint32_t len) {

    const char* data = (char*)key;
    uint32_t hash = 0x811c9dc5;
    uint32_t prime = 0x1000193;

    for(int i = 0; i < len; ++i) {
        uint8_t value = data[i];
        hash = hash ^ value;
        hash *= prime;
    }

    return hash;

} //hash_32_fnv1a

inline uint64_t hash_64_fnv1a_256(const void* key) {
    const char* data = (char*)key;
    uint64_t hash = 0xcbf29ce484222325;
    uint64_t prime = 0x100000001b3;
    
    #pragma clang loop unroll_count(256)
    for(int i = 0; i < 256; ++i) {
        uint8_t value = data[i];
        hash = hash ^ value;
        hash *= prime;
    }
    
    return hash;

}

inline uint64_t hash_64_fnv1a(const void* key, const uint64_t len) {
    const char* data = (char*)key;
    uint64_t hash = 0xcbf29ce484222325;
    uint64_t prime = 0x100000001b3;
    
    for(int i = 0; i < len; ++i) {
        uint8_t value = data[i];
        hash = hash ^ value;
        hash *= prime;
    }
    
    return hash;

} //hash_64_fnv1a