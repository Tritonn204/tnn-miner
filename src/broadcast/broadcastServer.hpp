#pragma once

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/json.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace beast = boost::beast;
namespace http = beast::http;
namespace json_b = boost::json;
using tcp = boost::asio::ip::tcp;

namespace BroadcastServer {
  extern int broadcastPort;
  extern std::vector<int64_t> *rate30sec_ptr;
  extern uint64_t startTime;
  extern int *accepted_ptr;
  extern int *rejected_ptr;
  extern const char* version_b;

  void serverThread(std::vector<int64_t> *HR30, int *accepted, int *rejected, const char *version);
};