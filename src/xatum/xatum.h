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
  static std::string handshake = "shake~";
  static std::string print = "print~";
  static std::string newJob = "job~";
  static std::string submission = "submit~";
  static std::string success = "success~";
  static std::string pingPacket = "ping~{}\n";
  static std::string pongPacket = "pong~{}\n";

  static std::string accepted_msg = "share accepted";
  static std::string stale_msg = "invalid extra nonce";

  static uint64_t lastReceivedJobTime = 0;
  static uint64_t jobTimeout = 90;

  const int ERROR_MSG = 3;
  const int WARN_MSG = 2;
  const int INFO_MSG = 1;
  const int VERBOSE_MSG = 0;

  static int logLevel = 0;

  typedef struct packet
  {
    std::string command;
    json data;
  } packet;

  static packet parsePacket(const std::string &str, const std::string &delimiter)
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