#pragma once

#include <inttypes.h>
#include <array>
#include <num.h>
#include <bitset>

// NOTES

// Input structure
// PRE_POW_HASH || TIME || 32 zero byte padding || NONCE
// 0-31            32-39   40-71                   72-79

using byte = unsigned char;

namespace AstrixHash
{
  const int INPUT_SIZE = 80;

  const byte matSize = 64;
  using matrix = uint16_t[matSize][matSize];
  const double epsilon = 1e-9;

  typedef struct worker
  {
    matrix matBuffer;
    matrix mat;
    double copied[matSize][matSize];
    std::bitset<matSize> rowsSelected;
    byte sha3Hash[32];
  } worker;

  class Xoshiro256PlusPlusHasher
  {
  private:
    uint64_t s[4];

  public:
    Xoshiro256PlusPlusHasher(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3)
    {
      s[0] = s0;
      s[1] = s1;
      s[2] = s2;
      s[3] = s3;
    }

    uint64_t next()
    {
      uint64_t tmp = s[0] + s[3];
      const uint64_t result = (tmp << 23 | tmp >> 41) + s[0];

      const uint64_t t = s[1] << 17;

      s[2] ^= s[0];
      s[3] ^= s[1];
      s[1] ^= s[2];
      s[0] ^= s[3];

      s[2] ^= t;
      s[3] = (s[3] << 45) | (s[3] >> 19);

      return result;
    }
  };

  inline int calculateRank(const matrix mat, worker &W)
  {
    for (int i = 0; i < matSize; i++)
    {
      for (int j = 0; j < matSize; j++)
      {
        W.copied[i][j] = static_cast<double>(mat[i][j]);
      }
    }

    int rank = 0;
    W.rowsSelected.reset();
    for (int i = 0; i < matSize; i++)
    {
      int j;
      for (j = 0; j < matSize; j++)
      {
        if (!W.rowsSelected[j] && std::abs(W.copied[j][i]) > epsilon)
        {
          break;
        }
      }

      if (j != matSize)
      {
        rank++;
        W.rowsSelected.set(j);
        for (int k = i + 1; k < matSize; k++)
        {
          W.copied[j][k] /= W.copied[j][i];
        }

        for (int k = 0; k < matSize; k++)
        {
          if (k == j || std::abs(W.copied[k][i]) <= epsilon)
          {
            continue;
          }

          for (int l = i + 1; l < matSize; l++)
          {
            W.copied[k][l] -= W.copied[j][l] * W.copied[k][i];
          }
        }
      }
    }

    return rank;
  }

  inline void newMatrix(byte *hash, matrix out, worker &W)
  {
    // for (int i = 0; i < matSize; i++) {
    //   memset(out[i], 0, matSize);
    // }
    memset(out, 0, matSize*matSize);

    alignas(64) uint64_t s0 = *(uint64_t*)&hash[0];
    alignas(64) uint64_t s1 = *(uint64_t*)&hash[8];
    alignas(64) uint64_t s2 = *(uint64_t*)&hash[16];
    alignas(64) uint64_t s3 = *(uint64_t*)&hash[24];
                  
    Xoshiro256PlusPlusHasher hasher(s0, s1, s2, s3);

    while (calculateRank(out, W) != matSize)
    {
      for (int i = 0; i < matSize; i++)
      {
        for (int j = 0; j < matSize; j += matSize/4)
        {
          uint64_t value = hasher.next();
          for (int k = 0; k < 16; k++)
          {
            out[i][j + k] = uint16_t((value >> (4 * k)) & 0x0f);
          }
        }
      }
    }
  }

  void testWithInput(const char* input, byte *out);
  void testWithInput(byte* in, byte *out);
  void genPrePowHash(byte *in, worker &worker);
  void hash(worker &worker, byte *in, int len, byte *out);
  int test();

  bool checkPow(Num in, Num diff);
  Num diffToTarget(double diff);
  Num diffToHash(double diff);
  int hipCompare();
}