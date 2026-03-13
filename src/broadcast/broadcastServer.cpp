#include "broadcastServer.hpp"
#include "tnn-common.hpp"
#include <chrono>
#include <numeric>

namespace BroadcastServer
{
#ifndef TNN_BROADCAST_PORT
#define TNN_BROADCAST_PORT 8989
#endif
  bool mmposEnabled = false;
  int broadcastPort = TNN_BROADCAST_PORT;
  std::vector<int64_t> *rate30sec_ptr;
  uint64_t startTime = 0;
  int *accepted_ptr;
  int *rejected_ptr;
  int interval;
  const char *algo_b;
  const char *version_b;

  // Per-GPU data
  int gpu_count = 0;
  std::vector<std::vector<int64_t>> *gpu_rates1min_ptr = nullptr;
  std::string *gpu_names_ptr = nullptr;
  std::string *gpu_pcie_ids_ptr = nullptr;

  void handleRequest(http::request<http::string_body> &req, http::response<http::string_body> &res)
  {
    if (req.method() == http::verb::get && req.target() == "/stats")
    {

      // Create a JSON object with some sample data
      json_b::object jsonData;

      jsonData["hashrate"] = std::accumulate((*rate30sec_ptr).begin(), (*rate30sec_ptr).end(), 0LL) / (*rate30sec_ptr).size();
      jsonData["accepted"] = *accepted_ptr;
      jsonData["rejected"] = *rejected_ptr;

      // Calculate the uptime using std::chrono
      auto currentTime = std::chrono::steady_clock::now();
      auto startTimePoint = std::chrono::steady_clock::time_point(std::chrono::seconds(startTime));
      auto uptime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTimePoint).count();
      jsonData["uptime"] = uptime;

      jsonData["algo"] = algo_b;
      jsonData["version"] = version_b;

      // Per-GPU stats
      if (gpu_count > 0 && gpu_rates1min_ptr) {
        json_b::array gpus;
        for (int i = 0; i < gpu_count; i++) {
          json_b::object gpu;
          gpu["id"] = i;

          // Compute average hashrate from 1-min rolling window (in H/s)
          auto& rates = (*gpu_rates1min_ptr)[i];
          double hr = 0.0;
          if (!rates.empty()) {
            hr = (double)std::accumulate(rates.begin(), rates.end(), 0LL) / (double)rates.size();
          }
          gpu["hashrate"] = hr;

          if (gpu_names_ptr) {
            gpu["name"] = gpu_names_ptr[i].c_str();
          }
          if (gpu_pcie_ids_ptr) {
            gpu["pcie_id"] = gpu_pcie_ids_ptr[i].c_str();
          }

          gpus.push_back(gpu);
        }
        jsonData["gpus"] = gpus;
      }

      // Set the response headers and body
      res.version(11);
      res.set(http::field::content_type, "application/json");
      res.body() = json_b::serialize(jsonData);
      res.prepare_payload();
    }
    else if (mmposEnabled && req.method() == http::verb::get && req.target() == "/mmpos")
    {
      json_b::object jsonData;

      // busid + hash arrays
      json_b::array busid_arr;
      json_b::array hash_arr;

      if (gpu_count > 0 && gpu_rates1min_ptr) {
        for (int i = 0; i < gpu_count; i++) {
          // PCI bus ID as decimal, or fall back to device index
          if (gpu_pcie_ids_ptr && !gpu_pcie_ids_ptr[i].empty()) {
            // hipDeviceGetPCIBusId returns "0000:0a:00.0" (domain:bus:dev.fn)
            // mmpOS expects the bus number as decimal integer
            const std::string &pcie = gpu_pcie_ids_ptr[i];
            unsigned int domain = 0, busNum = 0;
            if (std::sscanf(pcie.c_str(), "%x:%x:", &domain, &busNum) == 2) {
              busid_arr.push_back(static_cast<int64_t>(busNum));
            } else if (std::sscanf(pcie.c_str(), "%x:", &busNum) == 1) {
              busid_arr.push_back(static_cast<int64_t>(busNum));
            } else {
              busid_arr.push_back(static_cast<int64_t>(i));
            }
          } else {
            busid_arr.push_back(static_cast<int64_t>(i));
          }

          auto& rates = (*gpu_rates1min_ptr)[i];
          double hr = 0.0;
          if (!rates.empty())
            hr = (double)std::accumulate(rates.begin(), rates.end(), 0LL) / (double)rates.size();
          hash_arr.push_back(hr);
        }
      } else {
        // CPU miner
        busid_arr.push_back("cpu");
        double hr = 0.0;
        if (rate30sec_ptr && !rate30sec_ptr->empty())
          hr = (double)std::accumulate(rate30sec_ptr->begin(), rate30sec_ptr->end(), 0LL) / (double)rate30sec_ptr->size();
        hash_arr.push_back(hr);
      }

      jsonData["busid"] = busid_arr;
      jsonData["hash"] = hash_arr;
      jsonData["units"] = "hs";

      // air: [accepted, invalid, rejected]
      json_b::array air;
      air.push_back(*accepted_ptr);
      air.push_back(0);
      air.push_back(*rejected_ptr);
      jsonData["air"] = air;

      // Per-device shares: { "busid": [accepted, rejected, invalid], ... }
      json_b::object shares_obj;
      if (gpu_count > 0 && gpu_pcie_ids_ptr) {
        for (int i = 0; i < gpu_count; i++) {
          std::string key;
          const std::string &pcie = gpu_pcie_ids_ptr[i];
          unsigned int domain = 0, busNum = 0;
          if (!pcie.empty() && std::sscanf(pcie.c_str(), "%x:%x:", &domain, &busNum) == 2) {
            key = std::to_string(busNum);
          } else if (!pcie.empty() && std::sscanf(pcie.c_str(), "%x:", &busNum) == 1) {
            key = std::to_string(busNum);
          } else {
            key = std::to_string(i);
          }
          json_b::array dev_shares;
          dev_shares.push_back(deviceAccepted[i].load(std::memory_order_relaxed));
          dev_shares.push_back(deviceRejected[i].load(std::memory_order_relaxed));
          dev_shares.push_back(0); // invalid
          shares_obj[key] = dev_shares;
        }
      } else {
        // CPU miner
        json_b::array cpu_shares;
        cpu_shares.push_back(deviceAccepted[DEVICE_SHARE_CPU].load(std::memory_order_relaxed));
        cpu_shares.push_back(deviceRejected[DEVICE_SHARE_CPU].load(std::memory_order_relaxed));
        cpu_shares.push_back(0);
        shares_obj["cpu"] = cpu_shares;
      }
      jsonData["shares"] = shares_obj;

      jsonData["miner_name"] = "tnn-miner";
      jsonData["miner_version"] = version_b;

      res.version(11);
      res.set(http::field::content_type, "application/json");
      res.body() = json_b::serialize(jsonData);
      res.prepare_payload();
    }
    else
    {
      // Handle other routes or return an error response
      res.result(http::status::not_found);
      res.set(http::field::content_type, "text/plain");
      res.body() = "Not Found";
      res.prepare_payload();
    }
  }

  void handleConnection(tcp::socket socket)
  {
    try
    {
      while (true)
      {
        http::request<http::string_body> req;
        beast::flat_buffer buffer;

        try
        {
          http::read(socket, buffer, req);
        }
        catch (const boost::system::system_error &e)
        {
          if (e.code() == boost::asio::error::eof)
          {
            // Client closed the connection, break the loop
            break;
          }
          throw;
        }

        http::response<http::string_body> res;
        handleRequest(req, res);

        http::write(socket, res);
      }
    }
    catch (const std::exception &e)
    {
      // Unnecessary bloat if the server still works
      // std::cerr << "Error handling connection: " << e.what() << std::endl;
    }
  }

  void serverThread(std::vector<int64_t> *HR30, int *accepted, int *rejected, const char *algo, const char *version, int rinterval)
  {
    interval = rinterval;
    startTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    boost::asio::io_context ioContext;
    tcp::acceptor acceptor(ioContext, tcp::endpoint(tcp::v4(), broadcastPort));

    rate30sec_ptr = HR30;
    accepted_ptr = accepted;
    rejected_ptr = rejected;

    algo_b = algo;
    version_b = version;

    while (true)
    {
      tcp::socket socket(ioContext);
      acceptor.accept(socket);

      std::thread(handleConnection, std::move(socket)).detach();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}
