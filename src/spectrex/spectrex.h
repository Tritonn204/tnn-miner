#pragma once

#include <inttypes.h>
#include <astrobwtv3.h>

// NOTES

// Input structure
// PRE_POW_HASH || TIME || 32 zero byte padding || NONCE
// 0-31            32-39   40-71                   72-79

using byte = unsigned char;

namespace SpectreX
{
  const int INPUT_SIZE = 80;

  const byte matSize = 64;
  using matrix = uint16_t[matSize][matSize];
  const double epsilon = 1e-9;

  typedef struct worker
  {
    matrix mat;
    byte prePowHash[32];
    byte sha3Hash[32];
    byte astrobwtv3Hash[32];
    workerData *astroWorker;
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
      const uint64_t result = s[0] + s[3];

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

  inline int calculateRank(const matrix mat)
  {
    std::array<std::array<double, matSize>, matSize> copied;
    for (int i = 0; i < matSize; i++)
    {
      for (int j = 0; j < matSize; j++)
      {
        copied[i][j] = static_cast<double>(mat[i][j]);
      }
    }

    int rank = 0;
    std::array<bool, matSize> rowsSelected{};
    for (int i = 0; i < matSize; i++)
    {
      int j;
      for (j = 0; j < matSize; j++)
      {
        if (!rowsSelected[j] && std::abs(copied[j][i]) > epsilon)
        {
          break;
        }
      }

      if (j != matSize)
      {
        rank++;
        rowsSelected[j] = true;
        for (int k = i + 1; k < matSize; k++)
        {
          copied[j][k] /= copied[j][i];
        }

        for (int k = 0; k < matSize; k++)
        {
          if (k == j || std::abs(copied[k][i]) <= epsilon)
          {
            continue;
          }

          for (int l = i + 1; l < matSize; l++)
          {
            copied[k][l] -= copied[j][l] * copied[k][i];
          }
        }
      }
    }

    return rank;
  }

  inline void newMatrix(byte *hash, matrix out)
  {
    for(int i = 0; i < matSize; i++) {
      memset(out[i], 0, matSize);
    }

    uint64_t s0 = *(uint64_t*)&hash[0];
    uint64_t s1 = *(uint64_t*)&hash[8];
    uint64_t s2 = *(uint64_t*)&hash[16];
    uint64_t s3 = *(uint64_t*)&hash[24];
                  
    Xoshiro256PlusPlusHasher hasher(s0, s1, s2, s3);

    while (calculateRank(out) != matSize)
    {
      for (int i = 0; i < matSize; i++)
      {
        for (int j = 0; j < matSize; j += matSize/4)
        {
          uint64_t value = hasher.next();
          for (int k = 0; k < 64; k++)
          {
            out[i][j + k] = uint16_t((value >> (4 * k)) & 0x0f);
          }
        }
      }
    }
  }

  void genPrePowHash(byte *in, worker &worker);
  void hash(worker &worker, byte *in, int len, byte *out);
  void test();
}

void mineSpectre(int tid);