#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <endian.hpp>

namespace BTCStratum {
  using bJson = boost::json::object;

  const byte STRATUM_DEBUG = 3;
  const byte STRATUM_ERROR = 2;
  const byte STRATUM_WARN = 1;
  const byte STRATUM_INFO = 0;

  static int logLevel = 2;

  typedef struct jobCache {
    std::string jobId;
    std::vector<uint8_t> prevHash;
    std::vector<uint8_t> coinbase;
    std::vector<std::vector<uint8_t>> merkleTree;
    uint32_t version;
    uint32_t nBits;
    uint32_t nTime;
    bool cleanJobs;
    std::vector<uint8_t> extraNonce1;
    int extraNonce1Size;
    int extraNonce2Size;
    uint64_t extraNonce2;
    double difficulty;
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
  static std::string s_ping = "mining.ping";

  // Server responses IDs
  const byte subscribeID = 1;
  const byte authorizeID = 2;
  const byte submitID = 7;

  static uint64_t lastReceivedJobTime = 0;
  static uint64_t lastShareSubmissionTime = 0;

  const int shareSubmitTimeout = 70;
  static int jobTimeout = 120;

  inline void formatShare(boost::json::object& share,
                      const boost::json::value& jobData,
                      const std::string& worker,
                      uint32_t nonce,
                      uint32_t nTime,
                      uint32_t extraNonce2) {
    
    share = BTCStratum::stratumCall;
    share["id"] = BTCStratum::submit.id;
    share["method"] = BTCStratum::submit.method;

    // Format nonce in LE hex (4 bytes)
    uint8_t nonceBytes[4];
    le32enc(nonceBytes, nonce);
    std::string nonceHex = hexStr(nonceBytes, 4);

    // Format nTime in LE hex (4 bytes)
    uint8_t timeBytes[4];
    le32enc(timeBytes, nTime);
    std::string nTimeHex = hexStr(timeBytes, 4);
    
    // ExtraNonce2 formatting
    std::stringstream ss;
    ss << std::hex << std::setfill('0') 
      << std::setw(jobData.at("extraNonce2Size").get_uint64() * 2) 
      << extraNonce2;
    std::string extraNonce2Hex = ss.str();

    share["params"] = boost::json::array{
      worker,
      jobData.at("jobId").as_string().c_str(),
      extraNonce2Hex,
      nTimeHex,
      nonceHex
    };
  }

  inline void diffToWords(double difficulty, uint32_t target[8]) {
    // First clear the target array
    memset(target, 0, 32);

    if (difficulty == 0) {
      memset(target, 0xff, 32);
      return;
    }

    // Find first word position where diff <= 1.0
    int k;
    double diff = difficulty;
    for (k = 6; k > 0 && diff > 1.0; k--) {
      diff /= 4294967296.0;
    }

    // Calculate mantissa
    uint64_t m = 4294901760.0 / diff;
    
    if (m == 0 && k == 6) {
      memset(target, 0xff, 32);
    } else {
      target[k] = (uint32_t)m;
      target[k + 1] = (uint32_t)(m >> 32);
    }
  }
}