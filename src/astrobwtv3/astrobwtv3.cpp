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
#include "archon.h"

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
#include <libcubwt.cuh>
#include <device_sa.cuh>
#include <lookup.h>
// #include "dc3.h"

#include "archon.h"

using byte = unsigned char;

int ops[256];

// int main(int argc, char **argv)
// {
//   TestAstroBWTv3();
//   TestAstroBWTv3repeattest();
//   return 0;
// }

void TestAstroBWTv3()
{
  std::srand(1);
  int n = -1;
  workerData *worker = new workerData;

  int i = 0;
  for (PowTest t : random_pow_tests)
  {
    // if (i > 0)
    //   break;
    byte *buf = new byte[t.in.size()];
    memcpy(buf, t.in.c_str(), t.in.size());
    byte res[32];
    AstroBWTv3(buf, (int)t.in.size(), res, *worker, false);
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
    i++;
  }

  byte *data = new byte[48];
  byte *data2 = new byte[48];

  std::string c("7199110000261dfb0b02712100000000c09a113bf2050b1e55c79d15116bd94e00000000a9041027027fa800000314bb");
  std::string c2("7199110000261dfb0b02712100000000c09a113bf2050b1e55c79d15116bd94e00000000a9041027027fa800002388bb");
  hexstr_to_bytes(c, data);
  hexstr_to_bytes(c2, data2);

  printf("A: %s, B: %s\n", hexStr(data, 48).c_str(), hexStr(data2, 48).c_str());

  byte res[32];
  AstroBWTv3(data, 48, res, *worker, false);
  std::string s = hexStr(res, 32);
  if (s.c_str() != "cf6b34413d3836d07739464dbc43611bb1f31ac7d3098d1a6a873c2e687a023a")
  {
    printf("Work A FAIL. Pow function: pow(%s) = %s want %s\n", c.c_str(), s.c_str(), "cf6b34413d3836d07739464dbc43611bb1f31ac7d3098d1a6a873c2e687a023a");
  }
  else
  {
    printf("Work A SUCCESS! pow(%s) = %s want %s\n", c.c_str(), s.c_str(), "cf6b34413d3836d07739464dbc43611bb1f31ac7d3098d1a6a873c2e687a023a");
  }

  AstroBWTv3(data2, 48, res, *worker, false);
  s = hexStr(res, 32);
  if (s.c_str() != "cf6b34413d3836d07739464dbc43611bb1f31ac7d3098d1a6a873c2e687a023a")
  {
    printf("Work B FAIL. Pow function: pow(%s) = %s want %s\n", c2.c_str(), s.c_str(), "cf6b34413d3836d07739464dbc43611bb1f31ac7d3098d1a6a873c2e687a023a");
  }
  else
  {
    printf("Work B SUCCESS! pow(%s) = %s want %s\n", c2.c_str(), s.c_str(), "cf6b34413d3836d07739464dbc43611bb1f31ac7d3098d1a6a873c2e687a023a");
  }
  // for (int i = 0; i < 1024; i++)
  // {
  //   std::generate(buf.begin(), buf.end(), [&dist, &gen]()
  //                 { return dist(gen); });
  //   std::memcpy(random_data, buf.data(), buf.size());

  //   // std::cout << hexStr(data, 48) << std::endl;
  //   // std::cout << hexStr(random_data, 48) << std::endl;

  //   if (i % 2 == 0)
  //   {
  //     byte res[32];
  //     AstroBWTv3(data, 48, res, *worker, false);

  //     // hexStr(res, 64);
  //     std::string s = hexStr(res, 32);
  //     if (s != "c392762a462fd991ace791bfe858c338c10c23c555796b50f665b636cb8c8440")
  //     {
  //       printf("%d test failed hash %s\n", i, s.c_str());
  //     }
  //   }
  //   else
  //   {
  //     byte res[32];
  //     AstroBWTv3(buf.data(), 48, res, *worker, false);
  //   }
  // }
  // std::cout << "Repeated test over" << std::endl;
  // libcubwt_free_device_storage(storage);
  // cudaFree(storage);
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
      AstroBWTv3(data, 48, res, *worker, false);

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
      AstroBWTv3(buf.data(), 48, res, *worker, false);
    }
  }
  std::cout << "Repeated test over" << std::endl;
}

void AstroBWTv3(byte *input, int inputLen, byte *outputhash, workerData &worker, bool gpuMine)
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

    // printf("sha256 raw bytes: \n");

    // for(int i = 0; i < 32; i++) {
    //   printf("%d,", worker.sha_key[i]);
    // }

    // printf("\n\n");

    // std::cout << "worker.step_3 pre XOR: " << hexStr(worker.step_3, 256) << std::endl;
    worker.salsa20 = (worker.sha_key);
    worker.salsa20.setIv(worker.counter);
    worker.salsa20.processBytes(worker.step_3, worker.step_3, 256);

    // std::cout << "worker.step_3 post XOR: " << hexStr(worker.step_3, 256) << std::endl;

    // printf("pre setkey: \n");

    // for (int i = 0; i < 256; i++) {
    //   printf("%d, ", worker.key.data[i]);
    // }

    // printf("\n\n");

    RC4_set_key(&worker.key, 256, worker.step_3);

    // printf("post setkey: \n");

    // for (int i = 0; i < 256; i++) {
    //   printf("%d, ", worker.key.data[i]);
    // }

    // printf("\n x: %d, y: %d\n", worker.key.x, worker.key.y);

    RC4(&worker.key, 256, worker.step_3, worker.step_3);

    // std::cout << "worker.step_3 post rc4: " << hexStr(worker.step_3, 256) << std::endl;

    worker.lhash = hash_64_fnv1a(worker.step_3, 256);
    worker.prev_lhash = worker.lhash;

    worker.tries = 0;

    // printf(hexStr(worker.step_3, 256).c_str());
    // printf("\n\n");

    branchComputeCPU(worker);

    worker.data_len = static_cast<uint32_t>((worker.tries - 4) * 256 + (((static_cast<uint64_t>(worker.step_3[253]) << 8) | static_cast<uint64_t>(worker.step_3[254])) & 0x3ff));

    if (gpuMine)
      return;

    // printf("data length: %d\n", worker.data_len);
    divsufsort(worker.sData, worker.sa, worker.data_len);
    // archonsort(worker.sData, worker.sa, worker.data_len);
    // worker.sa = Radix(worker.sData, worker.data_len).build();
    // memcpy(worker.sa, radixSA, worker.data_len);
    // libcubwt_sa(worker.GPUData, worker.sData, worker.sa, worker.data_len);

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

void branchComputeCPU(workerData &worker)
{
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

    // fmt::printf("op: %d, ", worker.op);
    // fmt::printf("worker.pos1: %d, worker.pos2: %d\n", worker.pos1, worker.pos2);

    switch (worker.op)
    {
    case 0:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 3:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {

        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right

        // INSERT_RANDOM_CODE_END
      }
    }
    break;
    case 6:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 8:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 10:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 17:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 18:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 9);  // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);         // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 19:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 4); // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);                           // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 24:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 25:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 26:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                 // *
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 27:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 28:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 31:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] *= worker.step_3[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 34:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 37:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 39:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 41:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] -= (worker.step_3[i] ^ 97);        // XOR and -
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 42:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 4); // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 43:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 45:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 10); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);                       // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 46:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 47:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 49:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 50:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 52:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 53:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 54:

#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 56:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 8);                // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 58:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 8);             // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);// rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 62:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] += worker.step_3[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 64:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] *= worker.step_3[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 65:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 8); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 66:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 67:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 68:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 69:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 71:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 73:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 74:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 76:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 78:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];               // +
        worker.step_3[i] *= worker.step_3[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 80:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 82:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 87:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];               // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] += worker.step_3[i];               // +
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 88:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 92:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 93:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 10); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 96:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 97:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 98:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 99:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 100:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 101:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 102:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] += worker.step_3[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 105:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] *= worker.step_3[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 107:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 113:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);                           // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 114:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 117:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 122:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 123:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 9); // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 127:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 130:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 131:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] *= worker.step_3[i];                 // *
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 132:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 135:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 8); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 140:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 141:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 142:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 146:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 147:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 148:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
        worker.step_3[i] = reverse8(worker.step_3[i]);  // reverse bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
        worker.step_3[i] += worker.step_3[i];           // +
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 150:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 4); // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 154:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 155:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 156:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = std::rotl(worker.step_3[i], 4);             // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 157:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);    // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 159:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 4);             // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);    // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 161:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 164:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                 // *
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 165:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 166:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 170:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);    // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 172:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 173:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 175:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 176:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 177:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 178:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 181:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);         // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 183:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 186:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 187:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 189:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] -= (worker.step_3[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 190:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 196:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] *= worker.step_3[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 198:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 201:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 202:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 206:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 207:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 8); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);                           // rotate  bits by 3
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 208:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 210:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 212:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] += worker.step_3[i];               // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 218:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 220:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] = reverse8(worker.step_3[i]);     // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 222:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 4);  // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       //
      }
      break;
    case 225:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 229:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 230:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 233:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);    // rotate  bits by 3
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 234:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 242:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 244:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 245:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 248:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 249:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 250:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 251:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 252:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 253:
#pragma GCC unroll 32
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
#pragma GCC unroll 32
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
}

void finishBatch(workerData &worker)
{
  for (int i = 0; i < worker.saResults.size(); i++)
  {
    if (littleEndian())
    {
      byte *B = reinterpret_cast<byte *>(worker.saResults[i].data());
      hashSHA256(worker.sha256, B, worker.sHash, worker.saInputs[i].size() * 4);
      // worker.sHash = nHash;
    }
    else
    {
      byte *s = new byte[MAX_LENGTH * 4];
      for (int j = 0; j < worker.data_len; j++)
      {
        s[j << 1] = htonl(worker.saResults[i][j]);
      }
      hashSHA256(worker.sha256, s, worker.sHash, worker.saInputs[i].size() * 4);
      // worker.sHash = nHash;
      delete[] s;
    }
    memcpy(worker.outputHashes[i].data(), worker.sHash, 32);
  }
}