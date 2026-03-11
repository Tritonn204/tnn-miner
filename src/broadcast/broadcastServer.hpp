#pragma once

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/json.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

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
  extern const char* algo_b;
  extern const char* version_b;

  // Per-GPU data (set by caller, nullptr if CPU-only)
  extern int gpu_count;
  extern std::vector<std::vector<int64_t>> *gpu_rates1min_ptr;
  extern std::string *gpu_names_ptr;      // array[32]
  extern std::string *gpu_pcie_ids_ptr;   // array[32]

  void serverThread(std::vector<int64_t> *HR30, int *accepted, int *rejected, const char *algo, const char *version, int rinterval);
};
