#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

// BLAKE-256 constants
static const uint32_t blake256_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// BLAKE-512 constants
static const uint64_t blake512_IV[8] = {
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

// Sigma permutation for message loading
static const uint8_t sigma[10][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 }
};

// Constants for BLAKE
static const uint32_t blake256_c[16] = {
    0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
    0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
    0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
    0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
};

static const uint64_t blake512_c[16] = {
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
    0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
    0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL,
    0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL,
    0x9216D5D98979FB1BULL, 0xD1310BA698DFB5ACULL,
    0x2FFD72DBD01ADFB7ULL, 0xB8E1AFED6A267E96ULL,
    0xBA7C9045F12C7F99ULL, 0x24A19947B3916CF7ULL,
    0x0801F2E2858EFC16ULL, 0x636920D871574E69ULL
};

// Rotation functions
static inline __m256i rotr32_avx2(__m256i x, int c) {
    return _mm256_or_si256(_mm256_srli_epi32(x, c), 
                          _mm256_slli_epi32(x, 32 - c));
}

static inline uint32_t rotr32(uint32_t x, int c) {
    return (x >> c) | (x << (32 - c));
}

// BLAKE-256 G function - scalar reference
static inline void G_256_scalar(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
    uint32_t m1, uint32_t m2
) {
    a = a + b + m1;
    d = rotr32(d ^ a, 16);
    c = c + d;
    b = rotr32(b ^ c, 12);
    a = a + b + m2;
    d = rotr32(d ^ a, 8);
    c = c + d;
    b = rotr32(b ^ c, 7);
}

// BLAKE-256 compression - scalar reference
static inline void blake256_compress_scalar(uint32_t* h, const uint32_t* m, uint64_t t) {
    uint32_t v[16];
    
    // Initialize working variables
    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
    }
    for (int i = 0; i < 8; i++) {
        v[i + 8] = blake256_c[i];
    }
    
    v[12] ^= (uint32_t)t;
    v[13] ^= (uint32_t)(t >> 32);
    
    // 14 rounds
    for (int r = 0; r < 14; r++) {
        int round = (r < 10) ? r : r - 10;
        
        // Column step
        G_256_scalar(v[0], v[4], v[8],  v[12], m[sigma[round][0]] ^ blake256_c[sigma[round][1]], 
                                                m[sigma[round][2]] ^ blake256_c[sigma[round][3]]);
        G_256_scalar(v[1], v[5], v[9],  v[13], m[sigma[round][4]] ^ blake256_c[sigma[round][5]], 
                                                m[sigma[round][6]] ^ blake256_c[sigma[round][7]]);
        G_256_scalar(v[2], v[6], v[10], v[14], m[sigma[round][8]] ^ blake256_c[sigma[round][9]], 
                                                m[sigma[round][10]] ^ blake256_c[sigma[round][11]]);
        G_256_scalar(v[3], v[7], v[11], v[15], m[sigma[round][12]] ^ blake256_c[sigma[round][13]], 
                                                m[sigma[round][14]] ^ blake256_c[sigma[round][15]]);
        
        // Diagonal step
        G_256_scalar(v[0], v[5], v[10], v[15], m[sigma[round][0]] ^ blake256_c[sigma[round][1]], 
                                                m[sigma[round][2]] ^ blake256_c[sigma[round][3]]);
        G_256_scalar(v[1], v[6], v[11], v[12], m[sigma[round][4]] ^ blake256_c[sigma[round][5]], 
                                                m[sigma[round][6]] ^ blake256_c[sigma[round][7]]);
        G_256_scalar(v[2], v[7], v[8],  v[13], m[sigma[round][8]] ^ blake256_c[sigma[round][9]], 
                                                m[sigma[round][10]] ^ blake256_c[sigma[round][11]]);
        G_256_scalar(v[3], v[4], v[9],  v[14], m[sigma[round][12]] ^ blake256_c[sigma[round][13]], 
                                                m[sigma[round][14]] ^ blake256_c[sigma[round][15]]);
    }
    
    // Finalize
    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

// BLAKE-256 AVX2 compression - single hash using vector width
static inline void blake256_compress_avx2(uint32_t* h, const uint32_t* m, uint64_t t) {
    // For testing inline assembly from the article
    // This is a placeholder - you can insert the assembly here
    
    // For now, fall back to scalar
    blake256_compress_scalar(h, m, t);
}

// Full BLAKE-256 hash function - scalar
static inline void blake256_scalar(const void* input, size_t len, void* output) {
    uint32_t h[8];
    uint32_t m[16];
    uint8_t buffer[64];
    uint64_t t = 0;
    
    // Initialize
    memcpy(h, blake256_IV, 32);
    
    // Process full blocks
    const uint8_t* in = (const uint8_t*)input;
    while (len >= 64) {
        // Convert to words
        for (int i = 0; i < 16; i++) {
            m[i] = ((uint32_t)in[i*4 + 0] << 24) |
                   ((uint32_t)in[i*4 + 1] << 16) |
                   ((uint32_t)in[i*4 + 2] << 8) |
                   ((uint32_t)in[i*4 + 3]);
        }
        
        t += 512;
        blake256_compress_scalar(h, m, t);
        in += 64;
        len -= 64;
    }
    
    // Final block with padding
    memset(buffer, 0, 64);
    memcpy(buffer, in, len);
    buffer[len] = 0x80;
    
    if (len >= 55) {
        // Convert to words
        for (int i = 0; i < 16; i++) {
            m[i] = ((uint32_t)buffer[i*4 + 0] << 24) |
                   ((uint32_t)buffer[i*4 + 1] << 16) |
                   ((uint32_t)buffer[i*4 + 2] << 8) |
                   ((uint32_t)buffer[i*4 + 3]);
        }
        t += (len + 1) * 8;
        blake256_compress_scalar(h, m, t);
        
        memset(buffer, 0, 64);
        t = 0; // For final block
    } else {
        t += (len + 1) * 8;
    }
    
    // Length in final block
    buffer[55] |= 1; // Final block flag
    uint64_t msglen = t + (len < 55 ? 0 : 512);
    for (int i = 0; i < 8; i++) {
        buffer[56 + i] = (msglen >> (56 - i*8)) & 0xFF;
    }
    
    // Convert final block to words
    for (int i = 0; i < 16; i++) {
        m[i] = ((uint32_t)buffer[i*4 + 0] << 24) |
               ((uint32_t)buffer[i*4 + 1] << 16) |
               ((uint32_t)buffer[i*4 + 2] << 8) |
               ((uint32_t)buffer[i*4 + 3]);
    }
    
    // Final compression
    blake256_compress_scalar(h, m, t);
    
    // Output
    uint8_t* out = (uint8_t*)output;
    for (int i = 0; i < 8; i++) {
        out[i*4 + 0] = (h[i] >> 24) & 0xFF;
        out[i*4 + 1] = (h[i] >> 16) & 0xFF;
        out[i*4 + 2] = (h[i] >> 8) & 0xFF;
        out[i*4 + 3] = h[i] & 0xFF;
    }
}

// AVX2 version - placeholder for now
static inline void blake256_avx2(const void* input, size_t len, void* output) {
    // You can implement the AVX2 version here
    // For now, use scalar
    blake256_scalar(input, len, output);
}

// CryptoNight compatible wrapper
static inline void do_blake_hash_scalar(const void* input, void* output) {
    blake256_scalar(input, 200, output);
}

static inline void do_blake_hash_avx2(const void* input, void* output) {
    blake256_avx2(input, 200, output);
}