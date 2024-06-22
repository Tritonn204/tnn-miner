#include "broadcastServer.hpp"
#include <chrono>

namespace BroadcastServer
{
  int broadcastPort = 8989;
  std::vector<int64_t> *rate30sec_ptr;
  uint64_t startTime = 0;
  int *accepted_ptr;
  int *rejected_ptr;
  int interval;
  const char *version_b;

  void handleRequest(http::request<http::string_body> &req, http::response<http::string_body> &res)
  {
    if (req.method() == http::verb::get && req.target() == "/stats")
    {

      // Create a JSON object with some sample data
      json_b::object jsonData;

      jsonData["hashrate"] = std::accumulate((*rate30sec_ptr).begin(), (*rate30sec_ptr).end(), 0LL) / (*rate30sec_ptr).size() / interval;
      jsonData["accepted"] = *accepted_ptr;
      jsonData["rejected"] = *rejected_ptr;

      // Calculate the uptime using std::chrono
      auto currentTime = std::chrono::steady_clock::now();
      auto startTimePoint = std::chrono::steady_clock::time_point(std::chrono::seconds(startTime));
      auto uptime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTimePoint).count();
      jsonData["uptime"] = uptime;

      jsonData["version"] = version_b;

      // Set the response headers and body
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

  void serverThread(std::vector<int64_t> *HR30, int *accepted, int *rejected, const char *version, int rinterval)
  {
    interval = rinterval;
    startTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    boost::asio::io_context ioContext;
    tcp::acceptor acceptor(ioContext, tcp::endpoint(tcp::v4(), broadcastPort));

    rate30sec_ptr = HR30;
    accepted_ptr = accepted;
    rejected_ptr = rejected;

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