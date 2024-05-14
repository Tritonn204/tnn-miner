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

  const Num trueMax("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
  const Num maxTarget("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
  const Num minHash = (Num(1) << 256) / maxTarget;
  const Num bigGig(1e3);

  bool checkPow(Num in, Num diff) {
    if (trueMax >> (diff.to_double()) < in) return false; 
    return true;
  }

  Num diffToTarget(double diff) {
    // Create a Num object representing the difficulty

    // Calculate the target by dividing maxTarget by difficulty
    Num target = Num::div(maxTarget, diff);

    return target;
  }

  Num diffToHash(double diff) {
    // Create a Num object representing the difficulty

    // Calculate the target by dividing maxTarget by difficulty
    Num hv = Num::mul(diff, minHash);
    Num target = Num::div(hv, bigGig);

    return target;
  }

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

    // std::reverse(out, out+32);

    printf("POW hash: %s\n", hexStr(w.sha3Hash, 32).c_str());
    printf("WANT    : ae63221b94390528bd5a092be6247f7173099978bf6b150031c034ed22b37cea\n\n");

    printf("BWT hash: %s\n", hexStr(w.astrobwtv3Hash, 32).c_str());
    printf("WANT    : 271bd27bf393fc8854e4ada0f255cef19c0e86c9b7245088bdafc01318172dc5\n\n");

    printf("Heavy hash: %s\n", hexStr(out, 32).c_str());
    printf("WANT      : 0b68c38a0d359b9ef74fecfae4b2b0a0ea026fdcee22c1d48bcc824f32050ef5\n\n");
  }
}