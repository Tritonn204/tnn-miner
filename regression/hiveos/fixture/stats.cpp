#include <fstream>
#include <iterator>
#include <boost/json/src.hpp>
#include <tnn-common.hpp>
#include <broadcast/broadcastServer.hpp>

namespace BroadcastServer {
void handleRequest(http::request<http::string_body>&, http::response<http::string_body>&);
}

// Reads input state, calls the actual production HTTP handler and emits its
// response. No API-test mode is compiled into the miner.
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) return 2;
    try {
        std::ifstream file(argv[1]);
        std::string text{std::istreambuf_iterator<char>(file), {}};
        auto input = boost::json::parse(text).as_object();
        using namespace BroadcastServer;
        cpu_mining = input.at("cpu").as_bool();
        miningProfile.coin.miningAlgo = input.at("pearl").as_bool() ? ALGO_PEARL_POUW : ALGO_XELISV3;
        std::vector<int64_t> cpu;
        for (auto& n : input.at("cpu_rates").as_array()) cpu.push_back(n.to_number<int64_t>());
        std::vector<std::vector<int64_t>> rates;
        std::vector<std::string> names, buses;
        for (auto& value : input.at("gpus").as_array()) {
            auto& gpu = value.as_object();
            rates.emplace_back();
            for (auto& n : gpu.at("rates").as_array()) rates.back().push_back(n.to_number<int64_t>());
            names.emplace_back("Fixture GPU");
            buses.emplace_back(gpu.at("pci").as_string());
        }
        for (auto& n : input.at("include").as_array()) HIP_includeDevices.insert(n.to_number<int>());
        for (auto& n : input.at("exclude").as_array()) HIP_excludeDevices.insert(n.to_number<int>());
        deviceAccepted[DEVICE_SHARE_CPU] = 3;
        deviceRejected[DEVICE_SHARE_CPU] = 1;
        for (size_t i = 0; i < rates.size(); ++i) {
            deviceAccepted[i] = 10 + int(i);
            deviceRejected[i] = 20 + int(i);
        }
        int accepted = 7, rejected = 2;
        rate30sec_ptr = &cpu;
        accepted_ptr = &accepted;
        rejected_ptr = &rejected;
        gpu_count = int(rates.size());
        gpu_rates1min_ptr = &rates;
        gpu_names_ptr = names.data();
        gpu_pcie_ids_ptr = buses.data();
        startTime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() - 120;
        algo_b = algoName(miningProfile.coin.miningAlgo);
        version_b = "fixture-v1";
        mmposEnabled = !input.contains("mmpos_enabled") || input.at("mmpos_enabled").as_bool();
        http::request<http::string_body> request{http::verb::get, argc == 3 ? argv[2] : "/stats", 11};
        http::response<http::string_body> response;
        handleRequest(request, response);
        if (response.result() != http::status::ok) return 3;
        std::cout << response.body() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
