#include <endian.hpp>
#include <inttypes.h>

#define FMT_HEADER_ONLY

#include <fmt/format.h>
#include <fmt/printf.h>

#include <bitset>
#include <iostream>

#include <fnv1a.h>
#include <xxhash64.h>
#include "pow.h"
#include "powtest.h"

#include <unordered_map>
#include <array>
#include <algorithm>

#include <random>
#include <chrono>

#include <Salsa20.h>

#include <highwayhash/sip_hash.h>
#include <functional>

#include <divsufsort.h>

#include <hex.h>
#include <openssl/rc4.h>

#include <fstream>

#include <bit>

using byte = unsigned char;

int ops[256];

// int main(int argc, char **argv)
// {
//   TestAstroBWTv3();
//   TestAstroBWTv3repeattest();
//   return 0;
// }

int archonsort(unsigned char *T, int *SA, int n);

void TestAstroBWTv3()
{
  std::srand(1);
  int n = -1;
  workerData *worker = new workerData;
  for (PowTest t : random_pow_tests)
  {
    byte *buf = new byte[t.in.size()];
    memcpy(buf, t.in.c_str(), t.in.size());
    byte res[32];
    AstroBWTv3(buf, (int)t.in.size(), res, *worker);
    std::string s = hexStr(res, 32);
    if (s.c_str() != t.out)
    {
      printf("FAIL. Pow function: pow(%s) = %s want %s\n", t.in.c_str(), s.c_str(), t.out.c_str());
    }
    else
    {
      printf("SUCCESS! pow(%s) = %s want %s\n", t.in.c_str(), s.c_str(), t.out.c_str());
    }
    delete[] buf;
  }
}

void TestAstroBWTv3repeattest()
{
  workerData *worker = new workerData;
  byte *data = new byte[48];
  byte random_data[48];

  std::string c("419ebb000000001bbdc9bf2200000000635d6e4e24829b4249fe0e67878ad4350000000043f53e5436cf610000086b00");
  hexstr_to_bytes(c, data);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::array<byte, 48> buf;

  for (int i = 0; i < 1024; i++)
  {
    std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                  { return dist(gen); });
    std::memcpy(random_data, buf.data(), buf.size());

    // std::cout << hexStr(data, 48) << std::endl;
    // std::cout << hexStr(random_data, 48) << std::endl;

    if (i % 2 == 0)
    {
      byte res[32];
      AstroBWTv3(data, 48, res, *worker);

      // hexStr(res, 64);
      std::string s = hexStr(res, 32);
      if (s != "c392762a462fd991ace791bfe858c338c10c23c555796b50f665b636cb8c8440")
      {
        printf("%d test failed hash %s\n", i, s.c_str());
      }
    }
    else
    {
      byte res[32];
      AstroBWTv3(buf.data(), 48, res, *worker);
    }
  }
  std::cout << "Repeated test over" << std::endl;
}

void AstroBWTv3(byte *input, int inputLen, byte *outputhash, workerData &worker)
{
  // auto recoverFunc = [&outputhash](void *r)
  // {
  //   std::random_device rd;
  //   std::mt19937 gen(rd());
  //   std::uniform_int_distribution<uint8_t> dist(0, 255);
  //   std::array<uint8_t, 16> buf;
  //   std::generate(buf.begin(), buf.end(), [&dist, &gen]()
  //                 { return dist(gen); });
  //   std::memcpy(outputhash, buf.data(), buf.size());
  //   std::cout << "exception occured, returning random hash" << std::endl;
  // };
  // std::function<void(void *)> recover = recoverFunc;

  try
  {
    std::fill_n(worker.step_3, 256, 0);

    hashSHA256(worker.sha256, input, worker.sha_key, inputLen);

    // std::cout << "first sha256 of data: " << hexStr(worker.sha_key, 32) << std::endl;

    // std::cout << "worker.step_3 pre XOR: " << hexStr(worker.step_3, 256) << std::endl;
    worker.salsa20 = (worker.sha_key);
    worker.salsa20.setIv(worker.counter);
    worker.salsa20.processBytes(worker.step_3, worker.step_3, 256);

    // std::cout << "worker.step_3 post XOR: " << hexStr(worker.step_3, 256) << std::endl;

    RC4_set_key(&worker.key, 256, worker.step_3);
    RC4(&worker.key, 256, worker.step_3, worker.step_3);

    // std::cout << "worker.step_3 post rc4: " << hexStr(worker.step_3, 256) << std::endl;

    worker.lhash = hash_64_fnv1a(worker.step_3, 256);
    worker.prev_lhash = worker.lhash;

    // printf("first worker.lhash: %08jx\n", worker.lhash);

    worker.tries = 0;

    while (true)
    {
      worker.tries++;
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries;
      // printf("%d worker.random_switcher %d %08jx\n", worker.tries, worker.random_switcher, worker.random_switcher);

      worker.op = static_cast<byte>(worker.random_switcher);

      worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
      worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

      if (worker.pos1 > worker.pos2)
      {
        std::swap(worker.pos1, worker.pos2);
      }

      if (worker.pos2 - worker.pos1 > 32)
      {
        worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
      }

      // fmt::printf("op: %d\n\n", op);
      // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

      switch (worker.op)
      {
      case 0:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          // INSERT_RANDOM_CODE_END
          worker.t1 = worker.step_3[worker.pos1];
          worker.t2 = worker.step_3[worker.pos2];
          worker.step_3[worker.pos1] = reverse8(worker.t2);
          worker.step_3[worker.pos2] = reverse8(worker.t1);
        }
        break;
      case 1:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] += worker.step_3[i];                             // +
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 2:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 3:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 4:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 5:
      {
        for (int i = worker.pos1; i < worker.pos2; i++)
        {

          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right

          // INSERT_RANDOM_CODE_END
        }
      }
      break;
      case 6:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -

          // INSERT_RANDOM_CODE_END
        }
        break;
      case 7:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                             // +
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 8:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
          worker.step_3[i] = std::rotl(worker.step_3[i], 10); // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 5);// rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 9:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 10:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
          worker.step_3[i] *= worker.step_3[i];              // *
          worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
          worker.step_3[i] *= worker.step_3[i];              // *
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 11:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 1
          // worker.step_3[i] = std::rotl(worker.step_3[i], 5);            // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 12:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] *= worker.step_3[i];               // *
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 13:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 14:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 15:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 16:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                  // rotate  bits by 4
          worker.step_3[i] *= worker.step_3[i];              // *
          worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
          worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 17:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];      // XOR
          worker.step_3[i] *= worker.step_3[i];                // *
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          worker.step_3[i] = ~worker.step_3[i];                // binary NOT operator
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 18:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                    // rotate  bits by 4
          worker.step_3[i] = leftRotate8(worker.step_3[i], 9); // rotate  bits by 3
          // worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
          // worker.step_3[i] = std::rotl(worker.step_3[i], 5);         // rotate  bits by 5
          // INSERT_RANDOM_CODE_END
        }
        break;
      case 19:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] += worker.step_3[i];                          // +
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 20:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 21:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] += worker.step_3[i];                             // +
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 22:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 23:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 3
          // worker.step_3[i] = std::rotl(worker.step_3[i], 1);                           // rotate  bits by 1
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 24:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                  // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 25:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 26:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i];                        // *
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] += worker.step_3[i];                        // +
          worker.step_3[i] = reverse8(worker.step_3[i]);               // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 27:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                  // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 28:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 29:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] += worker.step_3[i];                          // +
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 30:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 31:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] *= worker.step_3[i];                          // *
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 32:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 33:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                              // rotate  bits by 4
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
          worker.step_3[i] *= worker.step_3[i];          // *
                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 34:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 35:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];              // +
          worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
          worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
          worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 36:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);          // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 37:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] *= worker.step_3[i];                             // *
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 38:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 39:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 40:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 41:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          worker.step_3[i] -= (worker.step_3[i] ^ 97);         // XOR and -
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);   // rotate  bits by 3
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 42:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 1
          // worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 43:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] += worker.step_3[i];                             // +
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 44:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 45:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 10); // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 5);                       // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 46:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] += worker.step_3[i];                        // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);           // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 47:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 48:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
          // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 49:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] += worker.step_3[i];                        // +
          worker.step_3[i] = reverse8(worker.step_3[i]);               // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 50:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]);     // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
          worker.step_3[i] += worker.step_3[i];              // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 51:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                    // rotate  bits by 4
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 52:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 53:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                        // +
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 54:

        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]);  // reverse bits
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          // worker.step_3[i] = ~worker.step_3[i];    // binary NOT operator
          // worker.step_3[i] = ~worker.step_3[i];    // binary NOT operator
          // INSERT_RANDOM_CODE_END
        }

        break;
      case 55:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                  // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 56:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] *= worker.step_3[i];               // *
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 57:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = leftRotate8(worker.step_3[i], 8);              // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 58:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] += worker.step_3[i];                             // +
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 59:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 60:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
          worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
          worker.step_3[i] *= worker.step_3[i];              // *
          worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 61:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = leftRotate8(worker.step_3[i], 8);           // rotate  bits by 3
          // worker.step_3[i] = std::rotl(worker.step_3[i], 5);// rotate  bits by 5
          // INSERT_RANDOM_CODE_END
        }
        break;
      case 62:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] += worker.step_3[i];                             // +
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 63:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);           // rotate  bits by 5
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                 // XOR and -
          worker.step_3[i] += worker.step_3[i];                        // +
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 64:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          worker.step_3[i] = reverse8(worker.step_3[i]);  // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                     // rotate  bits by 4
          worker.step_3[i] *= worker.step_3[i]; // *
                                                // INSERT_RANDOM_CODE_END
        }
        break;
      case 65:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = leftRotate8(worker.step_3[i], 8); // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] *= worker.step_3[i];               // *
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 66:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                  // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 67:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);          // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);           // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 68:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                               // rotate  bits by 4
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                          // INSERT_RANDOM_CODE_END
        }
        break;
      case 69:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 70:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 71:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 72:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 73:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = reverse8(worker.step_3[i]);               // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);           // rotate  bits by 5
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                 // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 74:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 75:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 76:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 77:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 78:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 79:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                   // rotate  bits by 4
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] += worker.step_3[i];               // +
          worker.step_3[i] *= worker.step_3[i];               // *
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 80:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] += worker.step_3[i];                             // +
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 81:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                                 // rotate  bits by 4
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 82:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
          // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 83:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 84:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] += worker.step_3[i];                          // +
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 85:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 86:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                                 // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                     // rotate  bits by 4
          worker.step_3[i] = ~worker.step_3[i]; // binary NOT operator
                                                // INSERT_RANDOM_CODE_END
        }
        break;
      case 87:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];              // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                     // rotate  bits by 4
          worker.step_3[i] += worker.step_3[i]; // +
                                                // INSERT_RANDOM_CODE_END
        }
        break;
      case 88:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
          worker.step_3[i] *= worker.step_3[i];               // *
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 89:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];               // +
          worker.step_3[i] *= worker.step_3[i];               // *
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 90:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]);     // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 91:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                              // rotate  bits by 4
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 92:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 93:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] += worker.step_3[i];                             // +
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 94:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 95:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
          worker.step_3[i] = std::rotl(worker.step_3[i], 10); // rotate  bits by 5
          // worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          // INSERT_RANDOM_CODE_END
        }
        break;
      case 96:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);          // rotate  bits by 2
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);          // rotate  bits by 2
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 97:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 98:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 99:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 100:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 101:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 102:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
          worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
          worker.step_3[i] += worker.step_3[i];              // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 103:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 104:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]);               // reverse bits
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);           // rotate  bits by 5
          worker.step_3[i] += worker.step_3[i];                        // +
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 105:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 106:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                  // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
          worker.step_3[i] *= worker.step_3[i];              // *
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 107:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 6);             // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
          // INSERT_RANDOM_CODE_END
        }
        break;
      case 108:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 109:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 110:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 111:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 112:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);   // rotate  bits by 3
          worker.step_3[i] = ~worker.step_3[i];                // binary NOT operator
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          worker.step_3[i] -= (worker.step_3[i] ^ 97);         // XOR and -
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 113:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 1);                           // rotate  bits by 1
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = ~worker.step_3[i];                        // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 114:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 115:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 116:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 117:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 118:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 119:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
          worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 120:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] *= worker.step_3[i];               // *
          worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
          worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 121:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
          worker.step_3[i] *= worker.step_3[i];                          // *
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 122:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                                 // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 123:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] = std::rotl(worker.step_3[i], 6);                // rotate  bits by 3
          // worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
          // INSERT_RANDOM_CODE_END
        }
        break;
      case 124:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 125:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 126:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = leftRotate8(worker.step_3[i], 9); // rotate  bits by 3
          // worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
          // worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 127:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 128:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 129:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 130:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 131:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                 // XOR and -
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] *= worker.step_3[i];                        // *
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 132:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 133:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 134:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i]; // binary NOT operator
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                                 // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 135:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 136:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 137:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 138:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          worker.step_3[i] += worker.step_3[i];           // +
          worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
                                                          // INSERT_RANDOM_CODE_END
        }
        break;
      case 139:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = leftRotate8(worker.step_3[i], 8); // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 140:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);   // rotate  bits by 1
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);  // rotate  bits by 2
          worker.step_3[i] ^= worker.step_3[worker.pos2];      // XOR
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 141:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                 // XOR and -
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] += worker.step_3[i];                        // +
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 142:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 143:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 144:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 145:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                   // rotate  bits by 4
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 146:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 147:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                     // rotate  bits by 4
          worker.step_3[i] *= worker.step_3[i]; // *
                                                // INSERT_RANDOM_CODE_END
        }
        break;
      case 148:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 149:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          worker.step_3[i] = reverse8(worker.step_3[i]);
          ;                                            // reverse bits
          worker.step_3[i] -= (worker.step_3[i] ^ 97); // XOR and -
          worker.step_3[i] += worker.step_3[i];        // +
                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 150:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 151:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 152:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 153:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 1
          // worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
          // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
          // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
          // INSERT_RANDOM_CODE_END
        }
        break;
      case 154:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);           // rotate  bits by 5
          worker.step_3[i] = ~worker.step_3[i];                        // binary NOT operator
          worker.step_3[i] ^= worker.step_3[worker.pos2];              // XOR
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 155:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                 // XOR and -
          worker.step_3[i] ^= worker.step_3[worker.pos2];              // XOR
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] ^= worker.step_3[worker.pos2];              // XOR
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 156:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 3
          // worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
          // INSERT_RANDOM_CODE_END
        }
        break;
      case 157:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 158:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);           // rotate  bits by 3
          worker.step_3[i] += worker.step_3[i];                        // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 159:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 160:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 1
          // worker.step_3[i] = std::rotl(worker.step_3[i], 3);    // rotate  bits by 3
          // INSERT_RANDOM_CODE_END
        }
        break;
      case 161:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 162:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i];               // *
          worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] -= (worker.step_3[i] ^ 97);        // XOR and -
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 163:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                  // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 164:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i];                        // *
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                 // XOR and -
          worker.step_3[i] = ~worker.step_3[i];                        // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 165:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] += worker.step_3[i];                          // +
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 166:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
          worker.step_3[i] += worker.step_3[i];               // +
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 167:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
          // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 168:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 169:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                                 // rotate  bits by 4
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 170:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] -= (worker.step_3[i] ^ 97);   // XOR and -
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
          worker.step_3[i] -= (worker.step_3[i] ^ 97);   // XOR and -
          worker.step_3[i] *= worker.step_3[i];          // *
                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 171:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);           // rotate  bits by 3
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                 // XOR and -
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = reverse8(worker.step_3[i]);               // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 172:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 173:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] += worker.step_3[i];                          // +
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 174:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 175:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);   // rotate  bits by 3
          worker.step_3[i] -= (worker.step_3[i] ^ 97);         // XOR and -
          worker.step_3[i] *= worker.step_3[i];                // *
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 176:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];      // XOR
          worker.step_3[i] *= worker.step_3[i];                // *
          worker.step_3[i] ^= worker.step_3[worker.pos2];      // XOR
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 177:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 178:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] += worker.step_3[i];                             // +
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 179:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 180:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                               // rotate  bits by 4
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
                                                          // INSERT_RANDOM_CODE_END
        }
        break;
      case 181:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 182:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 1
          // worker.step_3[i] = std::rotl(worker.step_3[i], 5);         // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 183:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];        // +
          worker.step_3[i] -= (worker.step_3[i] ^ 97); // XOR and -
          worker.step_3[i] -= (worker.step_3[i] ^ 97); // XOR and -
          worker.step_3[i] *= worker.step_3[i];        // *
                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 184:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] *= worker.step_3[i];                          // *
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 185:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i]; // binary NOT operator
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 186:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 187:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
          worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
          worker.step_3[i] += worker.step_3[i];              // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 188:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                            // rotate  bits by 4
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 189:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                               // rotate  bits by 4
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
          worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
                                                          // INSERT_RANDOM_CODE_END
        }
        break;
      case 190:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 191:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                             // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 192:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] *= worker.step_3[i];                          // *
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 193:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 194:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 195:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);          // rotate  bits by 2
          worker.step_3[i] ^= worker.step_3[worker.pos2];              // XOR
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 196:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 197:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                                 // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] *= worker.step_3[i];                             // *
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 198:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 199:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];           // binary NOT operator
          worker.step_3[i] += worker.step_3[i];           // +
          worker.step_3[i] *= worker.step_3[i];           // *
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                          // INSERT_RANDOM_CODE_END
        }
        break;
      case 200:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 201:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                     // rotate  bits by 4
          worker.step_3[i] = ~worker.step_3[i]; // binary NOT operator
                                                // INSERT_RANDOM_CODE_END
        }
        break;
      case 202:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 203:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 204:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 205:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] += worker.step_3[i];                          // +
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 206:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                            // rotate  bits by 4
          worker.step_3[i] = reverse8(worker.step_3[i]);               // reverse bits
          worker.step_3[i] = reverse8(worker.step_3[i]);               // reverse bits
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 207:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = leftRotate8(worker.step_3[i], 8); // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 3);                           // rotate  bits by 3
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 208:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 209:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);           // rotate  bits by 5
          worker.step_3[i] = reverse8(worker.step_3[i]);               // reverse bits
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                 // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 210:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 211:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                                 // rotate  bits by 4
          worker.step_3[i] += worker.step_3[i];                             // +
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 212:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 213:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 214:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 215:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] *= worker.step_3[i];                             // *
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 216:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 217:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          worker.step_3[i] += worker.step_3[i];                // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);   // rotate  bits by 1
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 218:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
          worker.step_3[i] = ~worker.step_3[i];          // binary NOT operator
          worker.step_3[i] *= worker.step_3[i];          // *
          worker.step_3[i] -= (worker.step_3[i] ^ 97);   // XOR and -
                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 219:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                                 // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 220:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 221:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          worker.step_3[i] ^= worker.step_3[worker.pos2];      // XOR
          worker.step_3[i] = ~worker.step_3[i];                // binary NOT operator
          worker.step_3[i] = reverse8(worker.step_3[i]);       // reverse bits
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 222:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] *= worker.step_3[i];                          // *
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 223:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 224:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 1
          // worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 225:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 226:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]);  // reverse bits
          worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
          worker.step_3[i] *= worker.step_3[i];           // *
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                          // INSERT_RANDOM_CODE_END
        }
        break;
      case 227:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 228:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];   // ones count bits
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 229:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 230:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 231:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
          worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 232:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] *= worker.step_3[i]; // *
          worker.step_3[i] *= worker.step_3[i]; // *
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                    // rotate  bits by 4
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 233:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);           // rotate  bits by 3
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 234:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 235:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] *= worker.step_3[i];               // *
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
          worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      case 236:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
          worker.step_3[i] += worker.step_3[i];                             // +
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 237:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 238:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];              // +
          worker.step_3[i] += worker.step_3[i];              // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
          worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                             // INSERT_RANDOM_CODE_END
        }
        break;
      case 239:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 5
          // worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
          worker.step_3[i] *= worker.step_3[i];                             // *
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 240:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
          worker.step_3[i] += worker.step_3[i];                             // +
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 241:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                            // rotate  bits by 4
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] ^= worker.step_3[worker.pos2];              // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 242:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];           // +
          worker.step_3[i] += worker.step_3[i];           // +
          worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
          worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                          // INSERT_RANDOM_CODE_END
        }
        break;
      case 243:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);           // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);          // rotate  bits by 2
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);           // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 244:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                // binary NOT operator
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);  // rotate  bits by 2
          worker.step_3[i] = reverse8(worker.step_3[i]);       // reverse bits
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 245:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 246:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                          // +
          worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
          worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
          worker.step_3[i] += worker.step_3[i];                          // +
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 247:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);  // rotate  bits by 2
          worker.step_3[i] = leftRotate8(worker.step_3[i], 5); // rotate  bits by 5
          worker.step_3[i] = ~worker.step_3[i];                // binary NOT operator
                                                               // INSERT_RANDOM_CODE_END
        }
        break;
      case 248:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = ~worker.step_3[i];                        // binary NOT operator
          worker.step_3[i] -= (worker.step_3[i] ^ 97);                 // XOR and -
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 5);           // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 249:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                                 // rotate  bits by 4
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
        }
        break;
      case 250:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
          worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]];      // ones count bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ; // rotate  bits by 4
            // INSERT_RANDOM_CODE_END
        }
        break;
      case 251:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] += worker.step_3[i];                        // +
          worker.step_3[i] ^= (byte)worker.bitTable[worker.step_3[i]]; // ones count bits
          worker.step_3[i] = reverse8(worker.step_3[i]);               // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);          // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
        }
        break;
      case 252:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);
          ;                                                              // rotate  bits by 4
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
          worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                         // INSERT_RANDOM_CODE_END
        }
        break;
      case 253:
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
          worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
          // INSERT_RANDOM_CODE_END

          worker.prev_lhash = worker.lhash + worker.prev_lhash;
          worker.lhash = XXHash64::hash(&worker.step_3, worker.pos2, 0); // more deviations
        }
        break;
      case 254:
      case 255:
        RC4_set_key(&worker.key, 256, worker.step_3);
        // worker.step_3 = highwayhash.Sum(worker.step_3[:], worker.step_3[:])
        for (int i = worker.pos1; i < worker.pos2; i++)
        {
          // INSERT_RANDOM_CODE_START
          worker.step_3[i] ^= static_cast<uint8_t>(std::bitset<8>(worker.step_3[i]).count()); // ones count bits
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                                  // rotate  bits by 3
          worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);                                 // rotate  bits by 2
          worker.step_3[i] = std::rotl(worker.step_3[i], 3);                                  // rotate  bits by 3
                                                                                              // INSERT_RANDOM_CODE_END
        }
        break;
      default:
        break;
      }

      // if (op == 53) {
      //   std::cout << hexStr(worker.step_3, 256) << std::endl << std::endl;
      //   std::cout << hexStr(&worker.step_3[worker.pos1], 1) << std::endl;
      //   std::cout << hexStr(&worker.step_3[worker.pos2], 1) << std::endl;
      // }

      worker.A = (worker.step_3[worker.pos1] - worker.step_3[worker.pos2]);
      worker.A = (256 + (worker.A % 256)) % 256;

      if (worker.A < 0x10)
      { // 6.25 % probability
        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(&worker.step_3, worker.pos2, 0);
        // printf("new worker.lhash: %08jx\n", worker.lhash);
      }

      if (worker.A < 0x20)
      { // 12.5 % probability
        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = hash_64_fnv1a(worker.step_3, worker.pos2);
        // printf("new worker.lhash: %08jx\n", worker.lhash);
      }

      if (worker.A < 0x30)
      { // 18.75 % probability
        memcpy(worker.s3, worker.step_3, worker.pos2);
        // std::copy(worker.step_3, worker.step_3 + worker.pos2, s3);
        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        HH_ALIGNAS(16)
        const highwayhash::HH_U64 key2[2] = {worker.tries, worker.prev_lhash};
        worker.lhash = highwayhash::SipHash(key2, worker.s3, worker.pos2); // more deviations
        // printf("new worker.lhash: %08jx\n", worker.lhash);
      }

      if (worker.A <= 0x40)
      { // 25% probablility
        RC4(&worker.key, 256, worker.step_3, worker.step_3);
      }

      worker.step_3[255] = worker.step_3[255] ^ worker.step_3[worker.pos1] ^ worker.step_3[worker.pos2];

      memcpy(&worker.sData[(worker.tries - 1) * 256], worker.step_3, 256);
      // std::copy(worker.step_3, worker.step_3 + 256, &worker.sData[(worker.tries - 1) * 256]);

      // memcpy(&worker->data.data()[(worker.tries - 1) * 256], worker.step_3, 256);

      // std::cout << hexStr(worker.step_3, 256) << std::endl;

      if (worker.tries > 260 + 16 || (worker.step_3[255] >= 0xf0 && worker.tries > 260))
      {
        break;
      }
    }

    worker.data_len = static_cast<uint32_t>((worker.tries - 4) * 256 + (((static_cast<uint64_t>(worker.step_3[253]) << 8) | static_cast<uint64_t>(worker.step_3[254])) & 0x3ff));


    divsufsort(worker.sData, worker.sa, worker.data_len);

    // byte T[worker.data_len + 1];
    // T[0] = 0;
    // T[worker.data_len] = 0;
    // memcpy(T, worker.sData, worker.data_len);
    // int SA2[worker.data_len + 1];
    // int err = gsaca(T, SA2, worker.data_len + 1);
    // libsais_ctx(worker.sais, worker.sData, worker.sa, worker.data_len, MAX_LENGTH-worker.data_len, NULL);
    // Archon A(worker.data_len);
    // A.saca(worker.sData, worker.sa2, worker.data_len);

    // printf("Validating...");
    // const bool rez = A.validate(worker.sData, worker.sa2, worker.data_len);
    // printf("%s\n", rez?"OK":"Fail");

    // std::cout << worker.sData[0] << std::endl;
    // std::cout << err << std::endl;

    // int D = 0;
    // int F = 0;
    // bool set1 = false, set2 = false;
    // for (int H = 0; H < MAX_LENGTH; H++) {
    //   if (H < 15) printf("DSS: %d, ARC: %d\n", worker.sa[H], worker.sa2[H]);
    //   if (worker.sa[H] != 0) {
    //     D++;
    //   }
    //   if (worker.sa2[H] != 0) {
    //     F++;
    //   }
    // }

    // printf("count dss: %d, count arc: %d\n", D, F);
    // printf("DSS: %d, ARC: %d\n", worker.sa[0], SA2[0]);

    // std::string_view S(hexStr(worker.sData, worker.data_len));
    // auto SA2 = psais::suffix_array<uint32_t>(S);
    // std::vector<byte> I;
    // worker.enCompute();

    // the current implementation relies on these sentinels!

    // unsigned int *SAH = (unsigned int *) malloc(MAX_LENGTH * sizeof(unsigned int));
    // byte *D = new byte[worker.data_len+2];
    // memcpy(&D[1], worker.sData, worker.data_len);
    // D[0] = D[worker.data_len + 1] = 0;

    // // gsaca_ds1(T, SA1, n);
    // // gsaca_ds2(T, SA2, n);
    // // gsaca_ds3(T, SA3, n);
    // gsaca_dsh(D, SAH, worker.data_len+2);

    // std::cout << worker.sa[0] << " : " << SAH[0] << std::endl;

    // worker.enCompute();

    // // fgsaca<uint32_t>((const uint8_t*)D, worker.sa, worker.data_len+2, 256);
    // gsaca_dsh(D, worker.sa, worker.data_len+2);
    // delete [] D;
    // free(SAH);

    // printf("divsufsort result at index 5: %d\ngsaca result at index 5: %d\n\n", worker.sa[5], SA3[5]);

    // for (unsigned int i = 0; i < n; ++i)
    //   std::cout << SA1[i] << " ";
    // std::cout << std::endl;

    // for (unsigned int i = 0; i < n; ++i)
    //   std::cout << SA2[i] << " ";
    // std::cout << std::endl;

    // for (unsigned int i = 0; i < n; ++i)
    //   std::cout << SA3[i] << " ";
    // std::cout << std::endl;

    // free(SA1);
    // free(SA2);
    // free(SA3);

    // byte *nHash;

    if (littleEndian())
    {
      byte *B = reinterpret_cast<byte *>(worker.sa);
      hashSHA256(worker.sha256, B, worker.sHash, worker.data_len * 4);
      // worker.sHash = nHash;
    }
    else
    {
      byte *s = new byte[MAX_LENGTH * 4];
      for (int i = 0; i < worker.data_len; i++)
      {
        s[i << 1] = htonl(worker.sa[i]);
      }
      hashSHA256(worker.sha256, s, worker.sHash, worker.data_len * 4);
      // worker.sHash = nHash;
      delete[] s;
    }
    memcpy(outputhash, worker.sHash, 32);
  }
  catch (const std::exception &ex)
  {
    // recover(outputhash);
    std::cerr << ex.what() << std::endl;
  }
}
