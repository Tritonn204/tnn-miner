#pragma once

#include <boost/json.hpp>
#include <string>

using byte = unsigned char;

int handleXStratumPacket(boost::json::object packet, bool isDev);
int handleXStratumResponse(boost::json::object packet, bool isDev);

int handleSpectreStratumPacket(boost::json::object packet, bool isDev);
int handleSpectreStratumResponse(boost::json::object packet, bool isDev);

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
    int id;
    std::string method;
  } method;

  // Client calls
  const method subscribe = {
    .id = 1,
    .method = "mining.subscribe"
  };

  const method authorize = {
    .id = 2,
    .method = "mining.authorize"
  };

  const method submit = {
    .id = 7,
    .method = "mining.submit"
  };

  const method reportHashrate = {
    .id = 5,
    .method = "mining.hashrate"
  };

  const method pong = {
    .id = 100,
    .method = "mining.pong"
  };

  // Server calls
  std::string s_notify = "mining.notify";
  std::string s_setDifficulty = "mining.set_difficulty";
  std::string s_setExtraNonce = "mining.set_extranonce";
  std::string s_ping = "mining.ping";
  std::string s_print = "mining.print";

  std::string k1ping = "ping~{}\n";
  std::string k1pong = "pong~{}\n";

  // Server responses IDs
  const byte subscribeID = 1;
  const byte authorizeID = 2;
  const byte submitID = 7;

  uint64_t lastReceivedJobTime = 0;
  int jobTimeout = 90;
}

namespace SpectreStratum {
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
    int id;
    std::string method;
  } method;

  // Client calls
  const method subscribe = {
    .id = 1,
    .method = "mining.subscribe"
  };

  const method authorize = {
    .id = 2,
    .method = "mining.authorize"
  };

  const method submit = {
    .id = 7,
    .method = "mining.submit"
  };

  const method reportHashrate = {
    .id = 5,
    .method = "mining.hashrate"
  };

  const method pong = {
    .id = 100,
    .method = "mining.pong"
  };

  // Server calls
  std::string s_notify = "mining.notify";
  std::string s_setDifficulty = "mining.set_difficulty";
  std::string s_setExtraNonce = "mining.set_extranonce";
  std::string s_ping = "mining.ping";
  std::string s_print = "mining.print";

  std::string k1ping = "ping~{}\n";
  std::string k1pong = "pong~{}\n";

  // Server responses IDs
  const byte subscribeID = 1;
  const byte authorizeID = 2;
  const byte submitID = 7;

  uint64_t lastReceivedJobTime = 0;
  uint64_t lastShareSubmissionTime = 0;

  const int shareSubmitTimeout = 70;

  int jobTimeout = 30;
}