#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace KawPowStratum {
  using bJson = boost::json::object;

  static int logLevel = 2;

  typedef struct jobCache {
    std::string jobId;
    std::string headerHash;    // 32-byte header hash (hex)
    std::string seedHash;      // 32-byte seed hash for epoch (hex)
    std::string target;        // 64-char hex target
    bool cleanJobs;
    int64_t height;            // block height
    std::string bits;          // compact difficulty bits (hex)
    std::vector<uint8_t> extraNonce1;
    int extraNonce1Size;
    double difficulty;
  } jobCache;

  static bJson stratumCall({
    {"id", 0},
    {"method", ""},
    {"params", bJson({})}
  });

  typedef struct method {
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

  // Server calls
  static std::string s_notify = "mining.notify";
  static std::string s_setTarget = "mining.set_target";
  static std::string s_ping = "mining.ping";

  const method pong = {
    .id = 100,
    .method = "mining.pong"
  };

  // Server response IDs
  const uint8_t subscribeID = 1;
  const uint8_t authorizeID = 2;
  const uint8_t submitID = 7;

  static uint64_t lastReceivedJobTime = 0;
  static uint64_t lastShareSubmissionTime = 0;

  const int shareSubmitTimeout = 70;
  static int jobTimeout = 120;
}
