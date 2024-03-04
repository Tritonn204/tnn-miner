#ifndef HEX_H
#define HEX_H

#include <iostream>
#include <string>
#include <stdint.h>
#include <iomanip>
#include <sstream>
#include <string>

inline std::string hexStr(const unsigned char *data, int len)
{
  std::stringstream ss;
  ss << std::hex;

  for (int i(0); i < len; ++i)
    ss << std::setw(2) << std::setfill('0') << (int)data[i];

  return ss.str();
}

inline void hexstr_to_bytes(std::string s, unsigned char *&b)
{
  for (unsigned int i = 0; i < s.length(); i += 2)
  {
    std::string byteString = s.substr(i, 2);
    uint8_t byte = (uint8_t)strtol(byteString.c_str(), NULL, 16);
    b[i/2] = byte;
  }
}

#endif