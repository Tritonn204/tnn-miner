#pragma once

#include <inttypes.h>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace Xatum
{
  std::string handshake = "shake~";
  std::string print = "print~";
  std::string newJob = "job~";
  std::string submission = "submit~";
  std::string success = "success~";
  std::string pingPacket = "ping~{}\n";
  std::string pongPacket = "pong~{}\n";

  std::string accepted = "share accepted";
  std::string stale = "invalid extra nonce";

  uint64_t lastReceivedJobTime = 0;
  uint64_t jobTimeout = 90;

  const int ERROR_MSG = 3;
  const int WARN_MSG = 2;
  const int INFO_MSG = 1;
  const int VERBOSE_MSG = 0;

  int logLevel = 0;

  typedef struct packet
  {
    std::string command;
    json data;
  } packet;

  packet parsePacket(const std::string &str, const std::string &delimiter)
  {
    size_t delimiterPos = str.find(delimiter);
    if (delimiterPos != std::string::npos)
    {
      size_t halfPos = delimiterPos + delimiter.length();
      if (halfPos < str.length())
      {
        return {str.substr(0, halfPos), json::parse(str.substr(halfPos))};
      }
    }
    return {str, json({})};
  }
}