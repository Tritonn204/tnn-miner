#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace KasStratum {
  using bJson = boost::json::object;

  constexpr byte STRATUM_DEBUG = 3;
  constexpr byte STRATUM_ERROR = 2;
  constexpr byte STRATUM_WARN = 1;
  constexpr byte STRATUM_INFO = 0;

  using uint256_t = boost::multiprecision::uint256_t;
  using cpp_dec_float_50 = boost::multiprecision::cpp_dec_float_50;

  const uint256_t trueMax("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
  const uint256_t maxTarget("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
  const uint256_t minHash = (uint256_t(1) << 256) / maxTarget;
  const uint256_t bigGig(1e9);

  inline uint256_t diffToTarget(double diff) {
    cpp_dec_float_50 target;
    target = cpp_dec_float_50(maxTarget) / cpp_dec_float_50(diff);

    return uint256_t(target);
  }

  inline uint256_t diffToHash(double diff) {
    cpp_dec_float_50 hv = cpp_dec_float_50(diff) * cpp_dec_float_50(minHash);
    cpp_dec_float_50 target = hv / cpp_dec_float_50(bigGig);

    return uint256_t(target);
  }

  constexpr int INPUT_SIZE = 80;

  static int logLevel = 2;

  typedef struct jobCache{
    uint64_t header[4];
    uint64_t ts;
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