#ifndef astrobwtv3
#define astrobwtv3

#include <iostream>
#include <fmt/format.h>

#include <iomanip>
#include <sstream>
#include <string>

#include <unordered_map>

#include <random>
#include <chrono>
#include <math.h>
#include <Salsa20.h>

#include <openssl/sha.h>
#include <openssl/rc4.h>

#ifdef _WIN32
#include <winsock2.h>
#endif

#ifdef linux
#include <arpa/inet.h>
#endif

#ifndef POW_CONST
#define POW_CONST

const uint32_t MAX_LENGTH = (256 * 384) - 1; // this is the maximum

#endif

class workerData
{
public:
  unsigned char sHash[32];
  unsigned char sha_key[32];
  unsigned char sData[MAX_LENGTH + 64];

  unsigned char counter[64];

  ucstk::Salsa20 salsa20;
  RC4_KEY key;

  int32_t sa[MAX_LENGTH];
  unsigned char sa_bytes[MAX_LENGTH * 4];
  
  unsigned char step_3[256];
  char s3[256];
  uint64_t random_switcher;

  uint64_t lhash;
  uint64_t prev_lhash;
  uint64_t tries;

  unsigned char op;
  unsigned char pos1;
  unsigned char pos2;
  unsigned char t1;
  unsigned char t2;

  unsigned char A;
  uint32_t data_len;
};

inline unsigned char
leftRotate8(unsigned char n, unsigned d)
{ // rotate n by d bits
  d = d % 8;
  return (n << d) | (n >> (8 - d));
}

inline unsigned char reverse8(unsigned char b)
{
  return (b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
}

inline unsigned char countSetBits(unsigned char n)
{
  unsigned char count = 0;
  while (n)
  {
    count += n & 1;
    n >>= 1;
  }
  return count;
}

inline unsigned char signByte(unsigned char A)
{
  A = (A + (A % 256)) % 256;
  return A;
}

template <std::size_t N>
inline void generateInitVector(std::uint8_t (&iv_buff)[N])
{
  using random_bytes_engine = std::independent_bits_engine<std::default_random_engine,
                                                           CHAR_BIT, unsigned short>;
  random_bytes_engine rbe;

  std::generate(std::begin(iv_buff), std::end(iv_buff), rbe);
}

inline void hashSHA256(const char *input, unsigned char *digest, unsigned long inputSize)
{
  SHA256_CTX sha256;
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, input, inputSize);
  SHA256_Final(digest, &sha256);
};

inline void hashSHA256(const unsigned char *input, unsigned char *digest, unsigned long inputSize)
{
  SHA256_CTX sha256;
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, input, inputSize);
  SHA256_Final(digest, &sha256);
};

void AstroBWTv3(unsigned char *input, int inputLen, unsigned char *outputhash, workerData &scratch);

#endif