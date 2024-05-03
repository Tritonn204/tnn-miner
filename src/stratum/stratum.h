#pragma once

#include <boost/json.hpp>
#include <string>

using byte = unsigned char;

namespace XelisStratum{
  using bJson = boost::json::object;
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

  method pong = {
    .id = 4,
    .method = "mining.pong"
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

  // Server responses IDs
  byte s_subscribeResult = 1;
  byte s_authorizeResult = 2;
  byte s_submitResult = 4;
}