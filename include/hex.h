#pragma once

#include <iostream>
#include <string>
#include <stdint.h>
#include <iomanip>
#include <sstream>
#include <string>

// #include <cuda.h>
// #include <cuda_runtime.h>

// __host__ __forceinline__ CUDA hints
inline std::string hexStr(const unsigned char *data, int len)
{
  static const char characters[] = "0123456789abcdef";
  std::string result(len * 2, ' ');
  for (int i = 0; i < len; i++)
  {
    result[2 * i] = characters[(unsigned int)data[i] >> 4];
    result[2 * i + 1] = characters[(unsigned int)data[i] & 0x0F];
  }
  return result;
}

// __host__ __device__ __forceinline__ CUDA hints
// char* hexStr_cuda(const unsigned char *data, int len)
// {
//   static const char characters[] = "0123456789abcdef";
//   char *result = (char*)malloc(len*2 + 1);
//   memset(result, ' ', len*2);
//   result[len*2] = '\0';
//   for (int i = 0; i < len; i++)
//   {
//     result[2 * i] = characters[(unsigned int)data[i] >> 4];
//     result[2 * i + 1] = characters[(unsigned int)data[i] & 0x0F];
//   }
//   return result;
// }

// __host__ __device__ __forceinline__ CUDA hints
inline void hexstr_to_bytes(std::string s, unsigned char *&b)
{
  for (unsigned int i = 0; i < s.length(); i += 2)
  {
    std::string byteString = s.substr(i, 2);
    uint8_t byte = (uint8_t)strtol(byteString.c_str(), NULL, 16);
    b[i / 2] = byte;
  }
}