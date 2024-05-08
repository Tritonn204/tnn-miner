#include "spectrex.h"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <hex.h>

extern "C"{
#include "cshake.h"
}

namespace SpectreX
{
  const char *pwHashDomain = "ProofOfWorkHash";
  const char *heavyHashDomain = "HeavyHash";

  void heavyHash(byte *hash, matrix &mat, byte *out)
  {
    std::array<uint16_t, 64> v{}, p{};
    for (int i = 0; i < matSize / 2; i++)
    {
      v[i * 2] = uint16_t(hash[i] >> 4);
      v[i * 2 + 1] = uint16_t(hash[i] & 0x0f);
    }

    // build the product array
    for (int i = 0; i < 64; i++)
    {
      uint16_t s = 0;
      for (int j = 0; j < 64; j++)
      {
        s += mat[i][j] * v[j];
      }
      p[i] = s >> 10;
    }

    // calculate the digest
    for (size_t i = 0; i < 32; i++)
    {
      out[i] = hash[i] ^ (static_cast<uint8_t>(p[i * 2] << 4) | static_cast<uint8_t>(p[i * 2 + 1]));
    }

    // hash the digest a final time, reverse bytes

    cshake256_nil_function_name(out, 32, "HeavyHash", out, 32*8);
  }

  void hash(worker &worker, byte *in, int len, byte *out)
  {
    // cshake256("ProofOfWorkHash", in, len, worker.sha3Hash, 32);
    cshake256_nil_function_name(in, 80, "ProofOfWorkHash", worker.sha3Hash, 32*8);
    AstroBWTv3(worker.sha3Hash, 64, worker.astrobwtv3Hash, *worker.astroWorker, false);
    heavyHash(worker.astrobwtv3Hash, worker.mat, out);
  }

  void test() {
    const char* input = "d63cad780f8bad8b6e6ba9d24dacb6eb6e7da1290e06e942943c8516787d1da96332bb538f0100000000000000000000000000000000000000000000000000000000000000000000aab7d5ff52ec2f8a";
    byte *in = new byte[80];
    byte out[32];

    hexstrToBytes(std::string(input), in);

    worker w;
    workerData *aw = (workerData*)malloc(sizeof(workerData));
    w.astroWorker = aw;
    newMatrix(in, w.mat);

    cshake256_nil_function_name(in, 80, "ProofOfWorkHash", w.sha3Hash, 32*8);
    AstroBWTv3(w.sha3Hash, 32, w.astrobwtv3Hash, *w.astroWorker, false);
    heavyHash(w.astrobwtv3Hash, w.mat, out);

    printf("POW hash: %s\n", hexStr(w.sha3Hash, 32).c_str());
    printf("BWT hash: %s\n", hexStr(w.astrobwtv3Hash, 32).c_str());
    printf("Heavy hash: %s\n", hexStr(out, 32).c_str());
  }
}