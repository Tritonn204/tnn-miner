#include "spectrex.h"
#include <openssl/evp.h>
#include <openssl/sha.h>

void blake2b(const byte *input, int inputLen, byte *output, const byte *key, int keyLen)
{
  uint32_t digest_length = SHA256_DIGEST_LENGTH;
  const EVP_MD *algorithm = EVP_blake2b512();
  EVP_MD_CTX *context = EVP_MD_CTX_new();
  EVP_DigestInit_ex(context, algorithm, nullptr);
  EVP_PKEY_CTX *pctx = EVP_MD_CTX_pkey_ctx(context);
  EVP_PKEY_CTX_ctrl(pctx, -1, EVP_PKEY_OP_KEYGEN, EVP_PKEY_CTRL_SET_MAC_KEY, keyLen, (void *)key);
  EVP_DigestUpdate(context, input, inputLen);
  EVP_DigestFinal_ex(context, output, &digest_length);
  EVP_MD_CTX_destroy(context);
}

void cshake256(const char *function_name, const unsigned char *message, size_t message_len, unsigned char *output, size_t output_len) {
    EVP_MD_CTX *mdctx;
    unsigned char cshake_input[1 + 1 + strlen(function_name) + 1 + message_len];
    size_t cshake_input_len = 0;

    cshake_input[cshake_input_len++] = 0x01;  // cSHAKE indicator byte
    cshake_input[cshake_input_len++] = 0x00;  // No salt

    // Append function name
    memcpy(cshake_input + cshake_input_len, function_name, strlen(function_name));
    cshake_input_len += strlen(function_name);
    cshake_input[cshake_input_len++] = 0x00;

    // Append message
    memcpy(cshake_input + cshake_input_len, message, message_len);
    cshake_input_len += message_len;

    // Initialize SHAKE256 context
    mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_shake256(), NULL);

    // Update with cSHAKE input
    EVP_DigestUpdate(mdctx, cshake_input, cshake_input_len);

    // Extract the desired output length
    EVP_DigestFinalXOF(mdctx, output, output_len);

    EVP_MD_CTX_free(mdctx);
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
    cshake256("HeavyHash", out, 32, out, 32);
  }

  void genPrePowHash(byte *in, worker &worker) {
    blake2b(in, 32, worker.prePowHash, (byte*)"BlockHash", 9);
    memcpy(worker.prePowHash, in, 32);
  }

  void hash(worker &worker, byte *in, int len, byte *out)
  {
    cshake256("ProofOfWorkHash", in, len, worker.sha3Hash, 32);
    AstroBWTv3(worker.sha3Hash, 64, worker.astrobwtv3Hash, *worker.astroWorker, false);
    heavyHash(worker.astrobwtv3Hash, worker.mat, out);
  }

  void test() {
    const char* input = "d63cad780f8bad8b6e6ba9d24dacb6eb6e7da1290e06e942943c8516787d1da96332bb538f0100000000000000000000000000000000000000000000000000000000000000000000aab7d5ff52ec2f8a";
    
  }
}