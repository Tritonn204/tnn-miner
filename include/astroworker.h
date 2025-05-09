#ifndef astroworker
#define astroworker

#include <bitset>
#include <stdint.h>
#include <vector>

#include "Salsa20.h"

#include <openssl/sha.h>
#include <openssl/rc4.h>

#define DERO_BATCH 1
#define ASTRO_CHUNK_SIZE 256
#define ASTRO_MAX_CHUNKS 277
#define ASTRO_MAX_LENGTH ((ASTRO_CHUNK_SIZE * ASTRO_MAX_CHUNKS) - 1) // this is the maximum
#define ASTRO_SCRATCH_SIZE ((ASTRO_MAX_LENGTH + 64))
#define MAX_RUN_LEN 48
#define MAX_RUNS_PER_BUCKET 128

#define MAX_DELTA_SIZE 33

const int branchedOps_size = 104; // Manually counted size of branchedOps_global
const int regOps_size = 256-branchedOps_size; // 256 - branchedOps_global.size()

//--------------------------------------------------------//

typedef unsigned char byte;
typedef struct templateMarker {
  uint8_t p1;
  uint8_t p2;
  uint16_t keySpotA;
  uint16_t keySpotB;
  uint16_t posData;

  uint16_t baseIdx; // can just << 8 to get true index
  uint32_t deltaOffset;
  uint8_t deltaCount;
} templateMarker;

const uint8_t iota8_g[256] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};

class workerData
{
public:
  // For aarch64
  byte aarchFixup[256];
  byte opt[256];

  int lucky = 0;

  SHA256_CTX sha256;
  ucstk::Salsa20 salsa20;
  RC4_KEY key[DERO_BATCH];

  byte salsaInput[256] = {0};
  byte op;

  byte A;
  uint32_t data_len;

  byte *chunk;
  byte *prev_chunk;

  byte maskTable_bytes[32*33];
  byte padding[64];

  bool isSame = false;

  int bA[256];
  int bB[256*256];

  byte sHash[32];
  byte sha_key[32];
  byte sha_key2[32];
  byte step_3[256];

  uint8_t _baseChunks[ASTRO_MAX_CHUNKS * ASTRO_CHUNK_SIZE + 31];
  uint8_t chunkDeltas[ASTRO_MAX_CHUNKS * ASTRO_MAX_CHUNKS * MAX_DELTA_SIZE + 1];

  byte sData[ASTRO_SCRATCH_SIZE*DERO_BATCH];

  byte pos1;
  byte pos2;
  byte t1;
  byte t2;

  uint64_t random_switcher;
  uint64_t lhash;
  uint64_t prev_lhash;
  uint16_t tries[DERO_BATCH];

  uint32_t sa_prelim[ASTRO_MAX_CHUNKS*256+1];
  int32_t sa[ASTRO_MAX_CHUNKS*256+1];
  templateMarker astroTemplate[ASTRO_MAX_CHUNKS];
  int templateIdx = 0;

  std::bitset<554> isBSlice;
  uint8_t iota8[256] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};
  uint8_t stampKeys[554];
  uint8_t stampTemplates[ASTRO_MAX_CHUNKS];

  uint16_t buckets_d[256][256];
  uint32_t bHeads[256][256];
  uint32_t bHeadIdx[256][256];
  
  uint8_t* baseChunks() {
    uintptr_t addr = (uintptr_t)_baseChunks;
    return (uint8_t*)((addr + 31) & ~31);
  }
  
  const uint8_t* baseChunks() const {
    uintptr_t addr = (uintptr_t)_baseChunks;
    return (const uint8_t*)((addr + 31) & ~31);
  }

  friend std::ostream& operator<<(std::ostream& os, const workerData& wd);
};

#endif