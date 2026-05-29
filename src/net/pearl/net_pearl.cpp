#include "net.hpp"

#include <base64.hpp>
#include <tnn_log.hpp>

#include <boost/asio.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/json.hpp>

#include <chrono>
#include <regex>
#include <sstream>
#include <string>
#include <thread>

namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

namespace {

struct PearlMiningInfo {
  std::string incomplete_header_b64;
  std::string target_decimal;
};

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"':  out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:   out.push_back(c); break;
    }
  }
  return out;
}

std::string make_request_line(int id, const std::string& method, const std::string& params_json = "{}") {
  std::ostringstream oss;
  oss << "{\"jsonrpc\":\"2.0\",\"method\":\"" << json_escape(method)
      << "\",\"id\":" << id << ",\"params\":" << params_json << "}\n";
  return oss.str();
}

bool parse_error_message(const std::string& line, std::string& message) {
  static const std::regex kMessageRe("\"message\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch m;
  if (std::regex_search(line, m, kMessageRe) && m.size() >= 2) {
    message = m[1].str();
    return true;
  }
  return false;
}

bool parse_mining_info_line(const std::string& line, PearlMiningInfo& out, std::string& err) {
  if (line.find("\"error\"") != std::string::npos && line.find("\"error\":null") == std::string::npos) {
    std::string message;
    if (parse_error_message(line, message)) {
      err = message;
    } else {
      err = "gateway returned an error";
    }
    return false;
  }

  static const std::regex kHeaderRe("\"incomplete_header_bytes\"\\s*:\\s*\"([^\"]+)\"");
  static const std::regex kTargetRe("\"target\"\\s*:\\s*([0-9]+)");
  std::smatch m;

  if (!std::regex_search(line, m, kHeaderRe) || m.size() < 2) {
    err = "missing incomplete_header_bytes in getMiningInfo response";
    return false;
  }
  out.incomplete_header_b64 = m[1].str();

  if (!std::regex_search(line, m, kTargetRe) || m.size() < 2) {
    err = "missing target in getMiningInfo response";
    return false;
  }
  out.target_decimal = m[1].str();
  return true;
}

bool rpc_readline(tcp::socket& socket, net::streambuf& buffer, net::yield_context yield, std::string& line) {
  boost::system::error_code ec;
  const std::size_t n = net::async_read_until(socket, buffer, '\n', yield[ec]);
  if (ec) {
    fail(ec, "pearl-readline");
    return false;
  }

  std::istream is(&buffer);
  std::getline(is, line);
  return true;
}

bool rpc_writeline(tcp::socket& socket, const std::string& msg, net::yield_context yield) {
  boost::system::error_code ec;
  net::async_write(socket, net::buffer(msg), yield[ec]);
  if (ec) {
    fail(ec, "pearl-writeline");
    return false;
  }
  return true;
}

void store_pearl_job(const PearlMiningInfo& info, bool isDev) {
  std::scoped_lock<std::mutex> lockGuard(mutex);

  boost::json::object obj;
  obj["incomplete_header_bytes"] = info.incomplete_header_b64;
  obj["target_decimal"] = info.target_decimal;

  if (isDev) {
    devJob = obj;
    devBlob = info.incomplete_header_b64;
    devHeight++;
  } else {
    job = obj;
    currentBlob = info.incomplete_header_b64;
    ourHeight++;
    jobCounter++;
  }
}

void log_pearl_job(const PearlMiningInfo& info, bool isDev) {
  const char* tag = isDev ? "DEV" : "MAIN";
  const std::size_t target_digits = info.target_decimal.size();
  const std::size_t header_b64_len = info.incomplete_header_b64.size();
  setcolor(isDev ? CYAN : BRIGHT_YELLOW);
  printf("[PEARL %s] getMiningInfo header_b64=%zu target_digits=%zu\n",
         tag, header_b64_len, target_digits);
  fflush(stdout);
  setcolor(BRIGHT_WHITE);
}

} // namespace

namespace tnn::pearl {

void pearl_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    net::yield_context yield,
    bool isDev)
{
  (void)wallet;
  (void)worker;

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  tcp::socket socket(ioc);
  boost::system::error_code ec;
  socket.async_connect(endpoint, yield[ec]);
  if (ec) {
    fail(ec, "connect-pearl");
    return;
  }

  net::streambuf buffer;
  int request_id = 1;
  std::string last_header_b64;
  bool *connected = isDev ? &devConnected : &isConnected;
  bool *submitPtr = isDev ? &submittingDev : &submitting;
  *connected = true;

  while (!ABORT_MINER) {
    const std::string request = make_request_line(request_id++, "getMiningInfo");
    if (!rpc_writeline(socket, request, yield)) {
      break;
    }

    std::string line;
    if (!rpc_readline(socket, buffer, yield, line)) {
      break;
    }

    PearlMiningInfo info;
    std::string err;
    if (!parse_mining_info_line(line, info, err)) {
      fail("pearl-getMiningInfo", err.c_str());
      break;
    }

    if (info.incomplete_header_b64 != last_header_b64) {
      last_header_b64 = info.incomplete_header_b64;
      store_pearl_job(info, isDev);
      log_pearl_job(info, isDev);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  setForDisconnectedNoAbort(connected, submitPtr, &data_ready, &cv);
  boost::system::error_code ignored;
  socket.shutdown(tcp::socket::shutdown_both, ignored);
  socket.close(ignored);
}

int pearl_rpc_test(
    std::string host,
    std::string const &port)
{
  try {
    net::io_context ioc;
    tcp::resolver resolver(ioc);
    auto results = resolver.resolve(host, port);
    tcp::socket socket(ioc);
    net::connect(socket, results);

    const std::string request = make_request_line(1, "getMiningInfo");
    net::write(socket, net::buffer(request));

    net::streambuf buffer;
    boost::asio::read_until(socket, buffer, '\n');
    std::istream is(&buffer);
    std::string line;
    std::getline(is, line);

    PearlMiningInfo info;
    std::string err;
    if (!parse_mining_info_line(line, info, err)) {
      TNN_LOG_ERROR("[PEARL-RPC-TEST] getMiningInfo failed: %s\n", err.c_str());
      return 1;
    }

    std::string header_bytes;
    try {
      header_bytes = base64::from_base64(info.incomplete_header_b64);
    } catch (const std::exception& e) {
      TNN_LOG_ERROR("[PEARL-RPC-TEST] invalid header base64: %s\n", e.what());
      return 1;
    }

    TNN_LOG_INFO("[PEARL-RPC-TEST] host=%s port=%s\n", host.c_str(), port.c_str());
    TNN_LOG_INFO("[PEARL-RPC-TEST] incomplete_header_bytes_b64_len=%zu decoded_len=%zu target_digits=%zu\n",
                 info.incomplete_header_b64.size(), header_bytes.size(), info.target_decimal.size());
    return 0;
  } catch (const std::exception& e) {
    TNN_LOG_ERROR("[PEARL-RPC-TEST] %s\n", e.what());
    return 1;
  }
}

} // namespace tnn::pearl
