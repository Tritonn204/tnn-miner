#include "spectrex.h"
#include <openssl/evp.h>
#include <openssl/sha.h>

void sha3_512(const byte *input, int inputLen, byte *output)
{
  uint32_t digest_length = SHA512_DIGEST_LENGTH;
  const EVP_MD *algorithm = EVP_sha3_512();
  EVP_MD_CTX *context = EVP_MD_CTX_new();
  EVP_DigestInit_ex(context, algorithm, nullptr);
  EVP_DigestUpdate(context, input, inputLen);
  EVP_DigestFinal_ex(context, output, &digest_length);
  EVP_MD_CTX_destroy(context);
}

void heavyHash(matrix &mat, byte *out)
{
  // initialize the matrix

  Matrix mat = newMatrix(s0, s1, s2, s3);

  // build the header
  std::vector<uint8_t> header(32 + 8 + 32 + 8);
  std::copy(hash.begin(), hash.begin() + 32, header.begin());
  std::memcpy(header.data() + 32, &timestamp, sizeof(timestamp));
  std::memcpy(header.data() + 72, &nonce, sizeof(nonce));
  header = CShake256(header, {'P', 'r', 'o', 'o', 'f', 'O', 'f', 'W', 'o', 'r', 'k', 'H', 'a', 's', 'h'}, 32);

  // initialize the vector and product arrays
  std::array<uint16_t, size> v{}, p{};
  for (int i = 0; i < size / 2; i++)
  {
    v[i * 2] = header[i] >> 4;
    v[i * 2 + 1] = header[i] & 0x0f;
  }

  // build the product array
  for (int i = 0; i < size; i++)
  {
    uint16_t s = 0;
    for (int j = 0; j < size; j++)
    {
      s += mat[i][j] * v[j];
    }
    p[i] = s >> 10;
  }

  // calculate the digest
  std::vector<uint8_t> digest(32);
  for (size_t i = 0; i < digest.size(); i++)
  {
    digest[i] = header[i] ^ (static_cast<uint8_t>(p[i * 2] << 4) | static_cast<uint8_t>(p[i * 2 + 1]));
  }

  // hash the digest a final time, reverse bytes
  digest = CShake256(digest, {'H', 'e', 'a', 'v', 'y', 'H', 'a', 's', 'h'}, 32);
  std::reverse(digest.begin(), digest.end());

  return digest;
}

namespace SpectreX
{
  void initState(worker &worker, byte *preHash)
  {
    newMatrix(preHash, worker.mat);
  }

  void hash(worker &worker, byte *in, int len, byte *out)
  {
    sha3_512(in, len, worker.sha3Hash);
    AstroBWTv3(worker.sha3Hash, 64, out, *worker.astroWorker, false);
  }
}