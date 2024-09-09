#pragma once

#include <boost/json.hpp>
#include <string>

using byte = unsigned char;

namespace rx0Stratum {
  using bJson = boost::json::object;

  const byte STRATUM_DEBUG = 3;
  const byte STRATUM_ERROR = 2;
  const byte STRATUM_WARN = 1;
  const byte STRATUM_INFO = 0;

  static int logLevel = 2;

  static bJson stratumCall({
    {"id", 0},
    {"method", ""},
    {"params", bJson({})}
  });

  typedef struct method{
    int id;
    std::string method;
  } method;

  // Client calls
  const method login = {
    .id = 1,
    .method = "login"
  };

  const method authorize = {
    .id = 1,
    .method = "authorize"
  };

  const method submit = {
    .id = 7,
    .method = "submit"
  };

  const method reportHashrate = {
    .id = 5,
    .method = "hashrate"
  };

  const method pong = {
    .id = 100,
    .method = "pong"
  };

  // Server calls
  static std::string s_job = "job";
  static std::string s_ping = "ping";
  static std::string s_print = "print";

  // Server responses IDs
  const byte loginID = 1;
  const byte authorizeID = 2;
  const byte submitID = 7;

  static uint64_t lastReceivedJobTime = 0;
  static int jobTimeout = 1200;
}

namespace XelisStratum {
  using bJson = boost::json::object;

  const byte STRATUM_DEBUG = 3;
  const byte STRATUM_ERROR = 2;
  const byte STRATUM_WARN = 1;
  const byte STRATUM_INFO = 0;

  static int logLevel = 2;

  static bJson stratumCall({
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
  static std::string s_notify = "mining.notify";
  static std::string s_setDifficulty = "mining.set_difficulty";
  static std::string s_setExtraNonce = "mining.set_extranonce";
  static std::string s_ping = "mining.ping";
  static std::string s_print = "mining.print";

  static std::string k1ping = "ping~{}\n";
  static std::string k1pong = "pong~{}\n";

  // Server responses IDs
  const byte subscribeID = 1;
  const byte authorizeID = 2;
  const byte submitID = 7;

  static uint64_t lastReceivedJobTime = 0;
  static int jobTimeout = 90;
}

namespace SpectreStratum {
  using bJson = boost::json::object;

  const byte STRATUM_DEBUG = 3;
  const byte STRATUM_ERROR = 2;
  const byte STRATUM_WARN = 1;
  const byte STRATUM_INFO = 0;

  static int logLevel = 2;

  typedef struct jobCache{
    uint64_t header[4];
  } jobCache;

  static bJson stratumCall({
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
  static std::string s_notify = "mining.notify";
  static std::string s_setDifficulty = "mining.set_difficulty";
  static std::string s_setExtraNonce = "set_extranonce";
  static std::string s_ping = "mining.ping";
  static std::string s_print = "mining.print";

  static std::string k1ping = "ping~{}\n";
  static std::string k1pong = "pong~{}\n";

  // Server responses IDs
  const byte subscribeID = 1;
  const byte authorizeID = 2;
  const byte submitID = 7;

  static uint64_t lastReceivedJobTime = 0;
  static uint64_t lastShareSubmissionTime = 0;

  const int shareSubmitTimeout = 70;

  static int jobTimeout = 30;
}

namespace AstrixStratum {
  using bJson = boost::json::object;

  const byte STRATUM_DEBUG = 3;
  const byte STRATUM_ERROR = 2;
  const byte STRATUM_WARN = 1;
  const byte STRATUM_INFO = 0;

  static int logLevel = 2;

  typedef struct jobCache{
    uint64_t header[4];
  } jobCache;

  static bJson stratumCall({
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
  static std::string s_notify = "mining.notify";
  static std::string s_setDifficulty = "mining.set_difficulty";
  static std::string s_setExtraNonce = "set_extranonce";
  static std::string s_ping = "mining.ping";
  static std::string s_print = "mining.print";

  static std::string k1ping = "ping~{}\n";
  static std::string k1pong = "pong~{}\n";

  // Server responses IDs
  const byte subscribeID = 1;
  const byte authorizeID = 2;
  const byte submitID = 7;

  static uint64_t lastReceivedJobTime = 0;
  static uint64_t lastShareSubmissionTime = 0;

  const int shareSubmitTimeout = 70;

  static int jobTimeout = 30;
}