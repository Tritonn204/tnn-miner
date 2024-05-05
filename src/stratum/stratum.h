#pragma once

#include <boost/json.hpp>
#include <string>

using byte = unsigned char;

int handleXStratumPacket(boost::json::object packet, bool isDev);
int handleXStratumResponse(boost::json::object packet, bool isDev);

namespace XelisStratum {
  using bJson = boost::json::object;

  const byte STRATUM_DEBUG = 3;
  const byte STRATUM_ERROR = 2;
  const byte STRATUM_WARN = 1;
  const byte STRATUM_INFO = 0;

  int logLevel = 2;

  bJson stratumCall({
    {"id", 0},
    {"method", ""},
    {"params", bJson({})}
  });

  typedef struct method{
    byte id;
    std::string method;
  } method;

  // Client calls
  method subscribe = {
    .id = 1,
    .method = "mining.subscribe"
  };

  method authorize = {
    .id = 2,
    .method = "mining.authorize"
  };

  method submit = {
    .id = 4,
    .method = "mining.submit"
  };

  method reportHashrate = {
    .id = 4,
    .method = "mining.hashrate"
  };

  // Server calls
  std::string s_notify = "mining.notify";
  std::string s_setDifficulty = "mining.set_difficulty";
  std::string s_setExtraNonce = "mining.set_extranonce";
  std::string s_ping = "mining.ping";
  std::string s_print = "mining.print";

  std::string k1ping = "ping~{}\n";
  std::string k1pong = "pong~{}\n";

  std::string c_pong = R"({"id":4,"method":"mining.pong"})";

  // Server responses IDs
  const byte s_subscribeResult = 1;
  const byte s_authorizeResult = 2;
  const byte s_submitResult = 4;
}