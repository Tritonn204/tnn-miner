#include "astrix-hash.h"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <hex.h>
#include <string.h>

#include <BLAKE3/c/blake3.h>
#include <crypto/tiny-keccak/tiny-keccak.h>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace AstrixHash
{
  constexpr int HALF_MATRIX_SIZE = 64/2;
  constexpr int QUARTER_MATRIX_SIZE = 64/4;

  const char *pwHashDomain = "ProofOfWorkHash";
  const char *heavyHashDomain = "HeavyHash";

  const uint256_t trueMax("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
  const uint256_t maxTarget("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
  const uint256_t minHash = (uint256_t(1) << 256) / maxTarget;
  const uint256_t bigGig(1e9);

  static const uint8_t powP[Plen] = {0x3d, 0xd8, 0xf6, 0xa1, 0x0d, 0xff, 0x3c, 0x11, 0x3c, 0x7e, 0x02, 0xb7, 0x55, 0x88, 0xbf, 0x29, 0xd2, 0x44, 0xfb, 0x0e, 0x72, 0x2e, 0x5f, 0x1e, 0xa0, 0x69, 0x98, 0xf5, 0xa3, 0xa4, 0xa5, 0x1b, 0x65, 0x2d, 0x5e, 0x87, 0xca, 0xaf, 0x2f, 0x7b, 0x46, 0xe2, 0xdc, 0x29, 0xd6, 0x61, 0xef, 0x4a, 0x10, 0x5b, 0x41, 0xad, 0x1e, 0x98, 0x3a, 0x18, 0x9c, 0xc2, 0x9b, 0x78, 0x0c, 0xf6, 0x6b, 0x77, 0x40, 0x31, 0x66, 0x88, 0x33, 0xf1, 0xeb, 0xf8, 0xf0, 0x5f, 0x28, 0x43, 0x3c, 0x1c, 0x65, 0x2e, 0x0a, 0x4a, 0xf1, 0x40, 0x05, 0x07, 0x96, 0x0f, 0x52, 0x91, 0x29, 0x5b, 0x87, 0x67, 0xe3, 0x44, 0x15, 0x37, 0xb1, 0x25, 0xa4, 0xf1, 0x70, 0xec, 0x89, 0xda, 0xe9, 0x82, 0x8f, 0x5d, 0xc8, 0xe6, 0x23, 0xb2, 0xb4, 0x85, 0x1f, 0x60, 0x1a, 0xb2, 0x46, 0x6a, 0xa3, 0x64, 0x90, 0x54, 0x85, 0x34, 0x1a, 0x85, 0x2f, 0x7a, 0x1c, 0xdd, 0x06, 0x0f, 0x42, 0xb1, 0x3b, 0x56, 0x1d, 0x02, 0xa2, 0xc1, 0xe4, 0x68, 0x16, 0x45, 0xe4, 0xe5, 0x1d, 0xba, 0x8d, 0x5f, 0x09, 0x05, 0x41, 0x57, 0x02, 0xd1, 0x4a, 0xcf, 0xce, 0x9b, 0x84, 0x4e, 0xca, 0x89, 0xdb, 0x2e, 0x74, 0xa8, 0x27, 0x94, 0xb0, 0x48, 0x72, 0x52, 0x8b, 0xe7, 0x9c, 0xce, 0xfc, 0xb1, 0xbc, 0xa5, 0xaf, 0x82, 0xcf, 0x29, 0x11, 0x5d, 0x83, 0x43, 0x82, 0x6f, 0x78, 0x7c, 0xb9, 0x02};
  static const uint8_t heavyP[Plen] = {0x09, 0x85, 0x24, 0xb2, 0x52, 0x4c, 0xd7, 0x3a, 0x16, 0x42, 0x9f, 0x2f, 0x0e, 0x9b, 0x62, 0x79, 0xee, 0xf8, 0xc7, 0x16, 0x48, 0xff, 0x14, 0x7a, 0x98, 0x64, 0x05, 0x80, 0x4c, 0x5f, 0xa7, 0x11, 0xda, 0xce, 0xee, 0x44, 0xdf, 0xe0, 0x20, 0xe7, 0x69, 0x40, 0xf3, 0x14, 0x2e, 0xd8, 0xc7, 0x72, 0xba, 0x35, 0x89, 0x93, 0x2a, 0xff, 0x00, 0xc1, 0x62, 0xc4, 0x0f, 0x25, 0x40, 0x90, 0x21, 0x5e, 0x48, 0x6a, 0xcf, 0x0d, 0xa6, 0xf9, 0x39, 0x80, 0x0c, 0x3d, 0x2a, 0x79, 0x9f, 0xaa, 0xbc, 0xa0, 0x26, 0xa2, 0xa9, 0xd0, 0x5d, 0xc0, 0x31, 0xf4, 0x3f, 0x8c, 0xc1, 0x54, 0xc3, 0x4c, 0x1f, 0xd3, 0x3d, 0xcc, 0x69, 0xa7, 0x01, 0x7d, 0x6b, 0x6c, 0xe4, 0x93, 0x24, 0x56, 0xd3, 0x5b, 0xc6, 0x2e, 0x44, 0xb0, 0xcd, 0x99, 0x3a, 0x4b, 0xf7, 0x4e, 0xb0, 0xf2, 0x34, 0x54, 0x83, 0x86, 0x4c, 0x77, 0x16, 0x94, 0xbc, 0x36, 0xb0, 0x61, 0xe9, 0x07, 0x07, 0xcc, 0x65, 0x77, 0xb1, 0x1d, 0x8f, 0x7e, 0x39, 0x6d, 0xc4, 0xba, 0x80, 0xdb, 0x8f, 0xea, 0x58, 0xca, 0x34, 0x7b, 0xd3, 0xf2, 0x92, 0xb9, 0x57, 0xb9, 0x81, 0x84, 0x04, 0xc5, 0x76, 0xc7, 0x2e, 0xc2, 0x12, 0x51, 0x67, 0x9f, 0xc3, 0x47, 0x0a, 0x0c, 0x29, 0xb5, 0x9d, 0x39, 0xbb, 0x92, 0x15, 0xc6, 0x9f, 0x2f, 0x31, 0xe0, 0x9a, 0x54, 0x35, 0xda, 0xb9, 0x10, 0x7d, 0x32, 0x19, 0x16};

  #define make_uchar4(a,b,c,d) \
    (((uint32_t)(a << 24) | (uint32_t)(b << 16) | (uint16_t)(c << 8) | d))

  uint256_t diffToTarget(double diff) {
    cpp_dec_float_50 target;
    target = cpp_dec_float_50(maxTarget) / cpp_dec_float_50(diff);

    return uint256_t(target);
  }

  uint256_t diffToHash(double diff) {
    cpp_dec_float_50 hv = cpp_dec_float_50(diff) * cpp_dec_float_50(minHash);
    cpp_dec_float_50 target = hv / cpp_dec_float_50(bigGig);

    return uint256_t(target);
  }

  static inline void heavyHash(byte *scratch, const matrix &mat, byte *out)
  {
    uint8_t v[64];
    
    __builtin_prefetch(v + 32, 1, 3);
    for (int i = 0; i < QUARTER_MATRIX_SIZE; i++)
    {
      v[i * 4] = uint16_t(scratch[i * 2] >> 4);
      v[i * 4 + 1] = uint16_t(scratch[i * 2] & 0x0f);
      v[i * 4 + 2] = uint16_t(scratch[i * 2 + 1] >> 4);
      v[i * 4 + 3] = uint16_t(scratch[i * 2 + 1] & 0x0f);
    }

    // build the product array
    for (int i = 0; i < QUARTER_MATRIX_SIZE; i++)
    {
      uint16_t sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
      __builtin_prefetch(mat[4*(i+1)], 0, 3);
      __builtin_prefetch(mat[4*(i+1)+1], 0, 3);
      __builtin_prefetch(mat[4*(i+1)+2], 0, 3);
      __builtin_prefetch(mat[4*(i+1)+3], 0, 3);

      for (int j = 0; j < 64; j++)
      {
        sum1 += mat[4 * i][j] * v[j];
        sum2 += mat[4 * i + 1][j] * v[j];
        sum3 += mat[4 * i + 2][j] * v[j];
        sum4 += mat[4 * i + 3][j] * v[j];
      }
      
      scratch[i * 2] ^= ((sum1 >> 10) << 4) | (sum2 >> 10);
      scratch[i * 2 + 1] ^= ((sum3 >> 10) << 4) | (sum4 >> 10);
    }

    // calculate the digest
    ((uint64_t*)scratch)[0] ^= ((uint64_t *)heavyP)[0];
    ((uint64_t*)scratch)[1] ^= ((uint64_t *)heavyP)[1];
    ((uint64_t*)scratch)[2] ^= ((uint64_t *)heavyP)[2];
    ((uint64_t*)scratch)[3] ^= ((uint64_t *)heavyP)[3];

    for (int i = 4; i < 25; i++) ((uint64_t *)scratch)[i] = ((uint64_t *)heavyP)[i];

    keccakf(scratch);
  }

  static inline void sha3_256_astrix(byte *in, byte *scratch) {
    scratch[32] = 0x06;
    scratch[135] = 0x80;

    keccakf(scratch);
  }

  static inline void blake3(const uint8_t *input, const int len, uint8_t *output) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, input, len);
    blake3_hasher_finalize(&hasher, output, BLAKE3_OUT_LEN);
  }

  void hash(worker &worker, byte *in, int len, byte *out)
  {
    alignas(4) uint8_t scratchData[200] = {0};

    for (int i=0; i<10; i++) ((uint64_t *)scratchData)[i] = ((uint64_t *)powP)[i] ^ ((uint64_t *)in)[i];
    for (int i = 10; i < 25; i++) ((uint64_t *)scratchData)[i] = ((uint64_t *)powP)[i];

    // printf("Cshake Keccak In:\n%s\n", hexStr(scratchData, 200).c_str());

    keccakf(scratchData);

    // printf("Cshake hash:\n%s\n", hexStr(scratchData, 32).c_str());

    blake3(scratchData, 32, scratchData);

    for (int i = 4; i < 25; i++) ((uint64_t *)scratchData)[i] = 0;

    sha3_256_astrix(scratchData, scratchData);

    // printf("SHA3 hash:\n%s\n", hexStr(scratchData, 32).c_str());

    heavyHash(scratchData, worker.matBuffer, out);
    memcpy(worker.scratchData, scratchData, 32);
  }

  void testWithInput(const char* input, byte *out) {
  //   byte *in = new byte[80];

  //   hexstrToBytes(std::string(input), in);

  //   worker w;
  //   alignas(32) workerData *aw = (workerData*)malloc_huge_pages(sizeof(workerData));
  //   initWorker(*aw);
  //   lookupGen(*aw, nullptr, nullptr);
  //   initWolfLUT();
  //   w.astroWorker = aw;
  //   newMatrix(in, w.matBuffer, w);

  //   hash(w, in, 80, out);
  //   free(aw);
  //   free(in);
  // }

  // void testWithInput(byte* in, byte *out) {
  //   worker w;
  //   alignas(32) workerData *aw = (workerData*)malloc_huge_pages(sizeof(workerData));
  //   initWorker(*aw);
  //   lookupGen(*aw, nullptr, nullptr);
  //   initWolfLUT();
  //   w.astroWorker = aw;
  //   newMatrix(in, w.matBuffer, w);

  //   hash(w, in, 80, out);
  //   free(aw);
  }

  int test() {
    const char* input = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f40414243444546470000000000000000";
    const char* expected = "b76caa07801864da3426861a13c3c58da7ac2ba02183a1860ecbee6149b92518";

    byte in[80];
    memset(in, 0, 80);

    byte out[32];
    memset(out, 0, 32);

    hexstrToBytes(std::string(input), in);
    AstrixHash::worker w;
    
    newMatrix(in, w.matBuffer, w);
    hash(w, in, 80, out);

    printf("CPU Astrix Result:\n%s\nWant:\n%s\n", hexStr(w.scratchData, 32).c_str(), expected);
    return strcmp(hexStr(w.scratchData, 32).c_str(), expected) == 0;
  }

  // int test() {
    // const char* input = "cb75500eabb7db4d2df02372d033c64085a9cec8f29d66a84ed9a8e38d2be6fcab42ea598f01000000000000000000000000000000000000000000000000000000000000000000000300000103000000";
    // byte *in = new byte[80];
    // byte out[32];

    // hexstrToBytes(std::string(input), in);

    // alignas(64) AstrixHash::worker *w = (AstrixHash::worker *)malloc_huge_pages(sizeof(AstrixHash::worker));
    // alignas(64) workerData *aw = (workerData *)malloc_huge_pages(sizeof(workerData));
    // initWorker(*aw);
    // lookupGen(*aw, nullptr, nullptr);
    // initWolfLUT();
    // w->astroWorker = aw;

    // newMatrix(in, w->matBuffer, *w);

    // hash(*w, in, 80, out);

    // // std::reverse(out, out+32);

    // int pieces_failed = 0;
    // const char *pow_expected = "ae63221b94390528bd5a092be6247f7173099978bf6b150031c034ed22b37cea";
    // printf("POW hash: %s\n", hexStr(w->sha3Hash, 32).c_str());
    // printf("WANT    : %s\n\n", pow_expected);

    // const char *bwt_expected = "271bd27bf393fc8854e4ada0f255cef19c0e86c9b7245088bdafc01318172dc5";
    // printf("BWT hash: %s\n", hexStr(w->astrobwtv3Hash, 32).c_str());
    // printf("WANT    : %s\n\n", bwt_expected);

    // const char *heavy_expected = "0b68c38a0d359b9ef74fecfae4b2b0a0ea026fdcee22c1d48bcc824f32050ef5";
    // printf("Heavy hash: %s\n", hexStr(out, 32).c_str());
    // printf("WANT      : %s\n\n", heavy_expected);

    // if(memcmp(hexStr(w->sha3Hash, 32).c_str(), pow_expected, 32) != 0) {
    //   pieces_failed += 1;
    // }
    // if(memcmp(hexStr(w->astrobwtv3Hash, 32).c_str(), bwt_expected, 32) != 0) {
    //   pieces_failed += 1;
    // }
    // if(memcmp(hexStr(out, 32).c_str(), heavy_expected, 32) != 0) {
    //   pieces_failed += 1;
    // }
    // return pieces_failed;
  // }
}