#ifndef astroworker
#define astroworker

#include <bitset>
#include <stdint.h>
#include <vector>

#include "Salsa20.h"

#include <openssl/sha.h>
#include <openssl/rc4.h>

#define DERO_BATCH 1
#define MAX_LENGTH ((256 * 277) - 1) // this is the maximum
#define ASTRO_SCRATCH_SIZE ((MAX_LENGTH + 64))

const int branchedOps_size = 104; // Manually counted size of branchedOps_global
const int regOps_size = 256-branchedOps_size; // 256 - branchedOps_global.size()

//--------------------------------------------------------//

typedef unsigned char byte;

typedef struct templateMarker {
  uint8_t p1;
  uint8_t p2;
  uint16_t posData;
} templateMarker;

class workerData
{
public:
  // For aarch64
  byte aarchFixup[256];
  byte opt[256];
  byte step_3[256];
  byte simpleLookup[regOps_size*(256*256)];
  byte lookup3D[branchedOps_size*256*256];
  uint16_t lookup2D[regOps_size*(256*256)];
  std::bitset<256> clippedBytes[regOps_size];
  std::bitset<256> unchangedBytes[regOps_size];
  std::bitset<256> isBranched;

  byte branchedOps[branchedOps_size*2];
  byte regularOps[regOps_size*2];

  byte branched_idx[256];
  byte reg_idx[256];

  int lucky = 0;

  SHA256_CTX sha256;
  ucstk::Salsa20 salsa20;
  RC4_KEY key[DERO_BATCH];

  // std::vector<std::tuple<int,int,int>> repeats;

  byte salsaInput[256] = {0};
  byte op;

  byte A;
  uint32_t data_len;

  byte *chunk;
  byte *prev_chunk;

  byte maskTable_bytes[32*33];
  byte padding[32];

  bool isSame = false;

  byte sHash[32];
  byte sha_key[32];
  byte sha_key2[32];
  byte sData[ASTRO_SCRATCH_SIZE*DERO_BATCH];

  byte pos1;
  byte pos2;
  byte t1;
  byte t2;

  uint64_t random_switcher;
  uint64_t lhash;
  uint64_t prev_lhash;
  uint16_t tries[DERO_BATCH];

  int bA[256];
  int bB[256*256];
  uint32_t sa_prelim[277*256+1];
  int32_t sa[277*256+1];
  templateMarker astroTemplate[277];
  int templateIdx = 0;

  int keysA[277*277];
  int keysB[277*277];
  // int stampStarts[554] = {0};
  // int modifiedBytes[MODBUFFER] = {0};
  uint8_t stampTemplates[277];
  uint8_t sBuckets[256];
  uint16_t buckets_d[256][256];
  int bucketSort[256];
  // std::bitset<277*256> isIn;
  // std::bitset<277*256> isKey;

  uint8_t headData_d[256][256][5];

  std::vector<byte> opsA;
  std::vector<byte> opsB;

  friend std::ostream& operator<<(std::ostream& os, const workerData& wd);
};

#endif