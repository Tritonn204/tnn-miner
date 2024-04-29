#pragma once

#include <inttypes.h>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include <xelis-hash.hpp>

namespace Xatum
{
  typedef struct HandshakePacket
  {
    uint32_t packet_type;
    std::string addr;
    std::string work;
    std::string agent;
    std::vector<std::string> algos;

    std::string serialize() const
    {
      std::ostringstream oss;
      oss << packet_type;
      oss << addr;
      oss << work;
      oss << agent;
      for (const auto &algo : algos)
      {
        oss << algo;
      }
      return oss.str();
    }
  } HandshakePacket;

  typedef struct JobPacket
  {
    std::string algo;
    uint64_t diff;
    uint8_t blob[XELIS_TEMPLATE_SIZE];

    std::string serialize() const
    {
      std::ostringstream oss;
      oss << algo;
      oss << diff;
      oss << std::hex;
      for (size_t i = 0; i < XELIS_TEMPLATE_SIZE; ++i)
      {
        oss << std::setw(2) << std::setfill('0') << static_cast<int>(blob[i]);
      }
      return oss.str();
    }
  } JobPacket;

  typedef struct SubmitPacket
  {
    uint8_t data[XELIS_TEMPLATE_SIZE];
    uint8_t hash[32];

    std::string serialize() const
    {
      std::ostringstream oss;
      oss << std::hex;
      for (size_t i = 0; i < XELIS_TEMPLATE_SIZE; ++i)
      {
        oss << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);
      }
      for (size_t i = 0; i < 32; ++i)
      {
        oss << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
      }
      return oss.str();
    }
  } SubmitPacket;
}