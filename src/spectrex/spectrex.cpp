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
    std::reverse(out, out+32);
  }

  void hash(worker &worker, byte *in, int len, byte *out)
  {
    // cshake256("ProofOfWorkHash", in, len, worker.sha3Hash, 32);
    newMatrix(in, worker.mat);
    cshake256_nil_function_name(in, len, "ProofOfWorkHash", worker.sha3Hash, 32*8);
    AstroBWTv3(worker.sha3Hash, 32, worker.astrobwtv3Hash, *worker.astroWorker, false);
    heavyHash(worker.astrobwtv3Hash, worker.mat, out);
  }

  void testWithInput(const char* input, byte *out) {
    byte *in = new byte[80];

    hexstrToBytes(std::string(input), in);

    worker w;
    workerData *aw = (workerData*)malloc(sizeof(workerData));
    w.astroWorker = aw;
    newMatrix(in, w.mat);

    hash(w, in, 80, out);
    free(aw);
    free(in);
  }

  void testWithInput(byte* in, byte *out) {
    worker w;
    workerData *aw = (workerData*)malloc(sizeof(workerData));
    w.astroWorker = aw;
    newMatrix(in, w.mat);

    hash(w, in, 80, out);
    free(aw);
  }

  void test() {
    const char* input = "cb75500eabb7db4d2df02372d033c64085a9cec8f29d66a84ed9a8e38d2be6fcab42ea598f01000000000000000000000000000000000000000000000000000000000000000000000300000103000000";
    byte *in = new byte[80];
    byte out[32];

    hexstrToBytes(std::string(input), in);

    worker w;
    workerData *aw = (workerData*)malloc(sizeof(workerData));
    w.astroWorker = aw;
    newMatrix(in, w.mat);

    hash(w, in, 80, out);

    printf("POW hash: %s\n", hexStr(w.sha3Hash, 32).c_str());
    printf("WANT    : 91e7bb140b1da77f7c8f1978547f5785d26528293f34fc7fe34f3de8dafe4831\n\n");

    printf("BWT hash: %s\n", hexStr(w.astrobwtv3Hash, 32).c_str());
    printf("WANT    : aee657f5887c509482a3186ca6a789f1fd545995badb5a6bc8d139e15dc177ab\n\n");

    printf("Heavy hash: %s\n", hexStr(out, 32).c_str());
    printf("WANT      : b68c38a0d359b9ef74fecfae4b2b0a0ea026fdcee22c1d48bcc824f32050ef5\n\n");
  }
}