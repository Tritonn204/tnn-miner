#include "hoohash.h"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <hex.h>
#include <string.h>
#include <cmath>

#include <BLAKE3/c/blake3.h>

extern "C"
{
#include <crypto/skein/skein.h>
}

namespace HooHash
{
  constexpr int HALF_MATRIX_SIZE = 64 / 2;
  constexpr int QUARTER_MATRIX_SIZE = 64 / 4;

  constexpr double PI = 3.14159265358979323846;
  
#define make_uchar4(a, b, c, d) \
  (((uint32_t)(a << 24) | (uint32_t)(b << 16) | (uint16_t)(c << 8) | d))

  float MediumComplexNonLinear(float x)
  {
    return exp(sin(x) + cos(x));
  }

  float IntermediateComplexNonLinear(float x)
  {
    if (x == PI / 2 || x == 3 * PI / 2)
    {
      return 0; // Avoid singularity
    }
    return sin(x) * cos(x) * tan(x);
  }

  float HighComplexNonLinear(float x)
  {
    return exp(x) * log(x + 1);
  }

  float ComplexNonLinear(float x)
  {
    float transformFactor = fmod(x, 4) / 4;
    if (x < 1)
    {
      if (transformFactor < 0.25)
      {
        return MediumComplexNonLinear(x + (1 + transformFactor));
      }
      else if (transformFactor < 0.5)
      {
        return MediumComplexNonLinear(x - (1 + transformFactor));
      }
      else if (transformFactor < 0.75)
      {
        return MediumComplexNonLinear(x * (1 + transformFactor));
      }
      else
      {
        return MediumComplexNonLinear(x / (1 + transformFactor));
      }
    }
    else if (x < 10)
    {
      if (transformFactor < 0.25)
      {
        return IntermediateComplexNonLinear(x + (1 + transformFactor));
      }
      else if (transformFactor < 0.5)
      {
        return IntermediateComplexNonLinear(x - (1 + transformFactor));
      }
      else if (transformFactor < 0.75)
      {
        return IntermediateComplexNonLinear(x * (1 + transformFactor));
      }
      else
      {
        return IntermediateComplexNonLinear(x / (1 + transformFactor));
      }
    }
    else
    {
      if (transformFactor < 0.25)
      {
        return HighComplexNonLinear(x + (1 + transformFactor));
      }
      else if (transformFactor < 0.5)
      {
        return HighComplexNonLinear(x - (1 + transformFactor));
      }
      else if (transformFactor < 0.75)
      {
        return HighComplexNonLinear(x * (1 + transformFactor));
      }
      else
      {
        return HighComplexNonLinear(x / (1 + transformFactor));
      }
    }
  }
  
  static inline void blake3(const uint8_t *input, const int len, uint8_t *output)
  {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, input, len);
    blake3_hasher_finalize(&hasher, output, BLAKE3_OUT_LEN);
  }

  static inline void HoohashMatrixMultiplication(uint16_t mat[64][64], const uint8_t *hashBytes, uint8_t *output)
  {
    float vector[64] = {0};
    float product[64] = {0};
    uint8_t res[32] = {0};

    // Populate the vector with floating-point values
    __builtin_prefetch(vector + 32, 1, 3);
    for (int i = 0; i < QUARTER_MATRIX_SIZE; i++)
    {
      vector[i * 4] = uint16_t(hashBytes[i * 2] >> 4);
      vector[i * 4 + 1] = uint16_t(hashBytes[i * 2] & 0x0f);
      vector[i * 4 + 2] = uint16_t(hashBytes[i * 2 + 1] >> 4);
      vector[i * 4 + 3] = uint16_t(hashBytes[i * 2 + 1] & 0x0f);
    }

    // printf("Vector: ");
    // for (int i = 0; i < 64; i++)
    // {
    //   printf("%f, ", vector[i]);
    // }
    // printf("\n");

    // Matrix-vector multiplication with floating point operations
    for (int i = 0; i < QUARTER_MATRIX_SIZE; i++)
    {
      float forComplex1;
      float forComplex2;
      float forComplex3;
      float forComplex4;
      for (int j = 0; j < 64; j++)
      {
        forComplex1 = (float)mat[i*4][j] * vector[j];
        forComplex2 = (float)mat[i*4 + 1][j] * vector[j];
        forComplex3 = (float)mat[i*4 + 2][j] * vector[j];
        forComplex4 = (float)mat[i*4 + 3][j] * vector[j];

        while (forComplex1 > 16)
        {
          forComplex1 = forComplex1 * 0.1;
        }
        while (forComplex2 > 16)
        {
          forComplex2 = forComplex2 * 0.1;
        }
        while (forComplex3 > 16)
        {
          forComplex3 = forComplex3 * 0.1;
        }
        while (forComplex4 > 16)
        {
          forComplex4 = forComplex4 * 0.1;
        }

        // Transform Matrix values with complex non-linear equations and sum into product.
        product[i*4] += ComplexNonLinear(forComplex1);
        product[i*4 + 1] += ComplexNonLinear(forComplex2);
        product[i*4 + 2] += ComplexNonLinear(forComplex3);
        product[i*4 + 3] += ComplexNonLinear(forComplex4);
      }
    }
    // printf("Product: ");
    // for (int i = 0; i < 64; i++)
    // {
    //   printf("%f ", product[i]);
    // }
    // printf("\n");

    // Convert product back to uint16 and then to byte array
    // printf("Hi/Low: ");
    for (int i = 0; i < 32; i++)
    {
      uint64_t high = product[2 * i] * 0.00000001;
      uint64_t low = product[2 * i + 1] * 0.00000001;
      // printf("%d - %d, ", high, low);
      // Combine high and low into a single byte
      uint8_t combined = (high ^ low) & 0xFF;
      res[i] = hashBytes[i] ^ combined;
    }
    // printf("\n");
    // printf("Res: ");
    // for (int i = 0; i < 32; i++)
    // {
    //   printf("%d,", res[i]);
    // }
    // printf("\n");

    // Hash again using BLAKE3
    blake3(res, 32, output);
  }

  // static inline void skein(const uint8_t *input, uint8_t *output) {
  //   SkeinCtx_t ctx;
  //   skeinCtxPrepare(&ctx, Skein256);
  //   skeinInit(&ctx, 256);

  //   skeinUpdate(&ctx, input, 32);
  //   skeinFinal(&ctx, output);
  // }

  void hash(worker &worker, byte *in, int len, byte *out)
  {
    alignas(4) uint8_t scratchData[200] = {0};
    // printf("Cshake Keccak In:\n%s\n", hexStr(scratchData, 200).c_str());
    blake3(in, 80, scratchData);

    // printf("blake3 hash:\n%s\n", hexStr(scratchData, 32).c_str());
    // printf("Skein hash:\n%s\n", hexStr(scratchData, 32).c_str());

    HoohashMatrixMultiplication(worker.matBuffer, scratchData, scratchData);
    memcpy(worker.scratchData, scratchData, 32);
  }

  void testWithInput(const char *input, byte *out)
  {
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

  int test()
  {
    const char *input = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f40414243444546470000000000000000";
    const char *expected = "277423cee51d16aa1b38cf2d6999c60f8f82ca84e74a432500bd000515562e29";

    byte in[80];
    memset(in, 0, 80);

    byte out[32];
    memset(out, 0, 32);

    hexstrToBytes(std::string(input), in);
    HooHash::worker w; 

    newMatrix(in, w.matBuffer, w);
    hash(w, in, 80, out);

    printf("CPU Hoo Result:\n%s\nWant:\n%s\n", hexStr(w.scratchData, 32).c_str(), expected);
    return strcmp(hexStr(w.scratchData, 32).c_str(), expected) == 0;
  }

  // int test() {
  // const char* input = "cb75500eabb7db4d2df02372d033c64085a9cec8f29d66a84ed9a8e38d2be6fcab42ea598f01000000000000000000000000000000000000000000000000000000000000000000000300000103000000";
  // byte *in = new byte[80];
  // byte out[32];

  // hexstrToBytes(std::string(input), in);

  // alignas(64) HooHash::worker *w = (HooHash::worker *)malloc_huge_pages(sizeof(HooHash::worker));
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