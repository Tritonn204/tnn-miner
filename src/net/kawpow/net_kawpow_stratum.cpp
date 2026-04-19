#include "../net.hpp"
#include <hex.h>
#include <tnn_log.hpp>

#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/json.hpp>

#include <queue>
#include <stratum/kawpow-stratum.h>

namespace beast = boost::beast;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

// ============================================================================
// Thread-safe write queue (matches Xelis pattern)
// Submit thread enqueues messages; actual I/O happens on the strand.
// ============================================================================

template <class Stream>
struct StratumWriteQueue {
  Stream& stream;
  net::strand<net::io_context::executor_type> strand;

  std::queue<std::string> q;
  std::mutex qmtx;
  std::atomic<bool> writeInProgress{false};
  std::atomic<bool> abort{false};

  explicit StratumWriteQueue(Stream& s, net::io_context& ioc)
      : stream(s), strand(net::make_strand(ioc)) {}

  void enqueue(std::string msg) {
    {
      std::lock_guard<std::mutex> lock(qmtx);
      q.push(std::move(msg));
    }
    process();
  }

  void process() {
    bool expected = false;
    if (!writeInProgress.compare_exchange_strong(expected, true)) {
      return;
    }

    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        {
          std::lock_guard<std::mutex> lock(qmtx);
          if (q.empty()) {
            writeInProgress.store(false);
            return;
          }
          msg = std::move(q.front());
          q.pop();
        }

        beast::error_code wec;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
        boost::asio::write(stream, boost::asio::buffer(msg), wec);
        if (wec) {
          abort.store(true);
          writeInProgress.store(false);
          return;
        }
      }
      writeInProgress.store(false);
    });
  }
};

// RAII guard: ensures a thread is always joined before destruction.
struct ThreadGuard {
  std::thread& t;
  std::atomic<bool>& abort;
  std::condition_variable& cv;

  ThreadGuard(std::thread& t_, std::atomic<bool>& a_, std::condition_variable& cv_)
      : t(t_), abort(a_), cv(cv_) {}

  ~ThreadGuard() {
    abort.store(true);
    cv.notify_all();
    if (t.joinable()) t.join();
  }

  ThreadGuard(const ThreadGuard&) = delete;
  ThreadGuard& operator=(const ThreadGuard&) = delete;
};

// ============================================================================
// KawPow stratum packet handlers
// ============================================================================

int handleKawPowStratumPacket(boost::json::object packet, KawPowStratum::jobCache *cache, bool isDev)
{
  std::string M = packet["method"].as_string().c_str();

  if (M.compare(KawPowStratum::s_notify) == 0)
  {
    std::scoped_lock<std::mutex> lockGuard(mutex);
    boost::json::value *J = isDev ? &devJob : &job;
    int64_t *h = isDev ? &devHeight : &ourHeight;

    auto params = packet["params"].as_array();

    // KawPow notify: [jobId, header_hash, seed_hash, target, clean_jobs, height, bits]
    cache->jobId = params[0].as_string().c_str();
    cache->headerHash = params[1].as_string().c_str();
    cache->seedHash = params[2].as_string().c_str();
    cache->target = params[3].as_string().c_str();
    cache->cleanJobs = params[4].as_bool();
    cache->height = params[5].to_number<int64_t>();
    if (params.size() > 6)
      cache->bits = params[6].as_string().c_str();

    // Store into job JSON for the mining loop
    (*J).as_object()["jobId"] = cache->jobId;
    (*J).as_object()["header_hash"] = cache->headerHash;
    (*J).as_object()["seed_hash"] = cache->seedHash;
    (*J).as_object()["target"] = cache->target;
    (*J).as_object()["height"] = cache->height;
    if (params.size() > 6)
      (*J).as_object()["bits"] = cache->bits;

    KawPowStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count();

    bool *C = isDev ? &devConnected : &isConnected;
    if (!beQuiet)
    {
      setcolor(CYAN);
      printf("\n");
      if (isDev)
        printf("DEV | ");
      printf("KawPow Stratum: new job (height %lld, epoch %lld)\n",
             (long long)cache->height, (long long)(cache->height / 7500));
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    if (!*C)
    {
      if (!isDev)
      {
        setcolor(BRIGHT_YELLOW);
        printf("Mining at: %s to wallet %s\n", miningProfile.host.c_str(), miningProfile.wallet.c_str());
        fflush(stdout);
        setcolor(CYAN);
        printf("Dev fee: %.2f%% of your total hashrate\n", devFee);
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
      else
      {
        setcolor(CYAN);
        printf("Connected to dev node\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }

    *C = true;
    (*h)++;
    jobCounter++;
  }
  else if (M.compare(KawPowStratum::s_setTarget) == 0)
  {
    // mining.set_target: [target_hex_64chars]
    auto params = packet["params"].as_array();
    std::string targetHex = params[0].as_string().c_str();

    // Convert 64-char hex target to difficulty
    // Target is a 256-bit number; difficulty = max_target / target
    // For now, store raw target and let the mining loop use it directly
    double *d = isDev ? &doubleDiffDev : &doubleDiff;

    // Parse target as difficulty: diff = 2^256 / target
    // Simple approach: count leading zeros for approximate difficulty
    boost::multiprecision::uint256_t tgt("0x" + targetHex);
    if (tgt > 0) {
      boost::multiprecision::uint256_t max256 = (boost::multiprecision::uint256_t(1) << 256) - 1;
      boost::multiprecision::uint256_t diff256 = max256 / tgt;
      *d = diff256.convert_to<double>();
    } else {
      *d = 1.0;
    }

    cache->difficulty = *d;

    if (!beQuiet)
    {
      setcolor(CYAN);
      if (isDev)
        printf("DEV | ");
      printf("Target set: %s (diff ~%.4f)\n", targetHex.substr(0, 16).c_str(), *d);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    jobCounter++;
  }
  else if (M.compare(KawPowStratum::s_ping) == 0)
  {
    return 1; // Signal to send pong
  }
  else
  {
    std::cout << "KawPow Stratum: unrecognized packet: " << boost::json::serialize(packet).c_str() << std::endl;
  }

  return 0;
}

int handleKawPowStratumResponse(boost::json::object packet, KawPowStratum::jobCache *cache, bool isDev)
{
  if (!packet.contains("id"))
    return 0;
  int64_t id = packet["id"].to_number<int64_t>();

  switch (id)
  {
  case KawPowStratum::subscribeID:
  {
    if (!packet["result"].is_null())
    {
      auto result = packet["result"].as_array();
      // KawPow subscribe response: [null, extraNonce1_hex]
      if (result.size() >= 2 && !result[1].is_null())
      {
        std::string xnonce1_hex = result[1].as_string().c_str();
        cache->extraNonce1Size = xnonce1_hex.length() / 2;
        cache->extraNonce1.resize(cache->extraNonce1Size);
        hexstrToBytes(xnonce1_hex, cache->extraNonce1.data());
      }
      return 0;
    }
    else
    {
      std::string errorMsg = "subscribe failed";
      if (packet.contains("error") && !packet["error"].is_null()) {
        auto &ev = packet["error"];
        if (ev.is_string())
          errorMsg = std::string(ev.get_string());
        else if (ev.is_array()) {
          auto &ea = ev.as_array();
          if (ea.size() > 1 && ea[1].is_string())
            errorMsg = std::string(ea[1].as_string());
          else if (ea.size() > 0 && ea[0].is_string())
            errorMsg = std::string(ea[0].as_string());
        } else if (ev.is_object() && ev.as_object().contains("message"))
          errorMsg = std::string(ev.as_object()["message"].as_string());
      }
      TNN_LOG_ERROR("%sKawPow Stratum ERROR: %s\n", isDev ? "DEV | " : "", errorMsg.c_str());
      return -1;
    }
  }
  break;

  case KawPowStratum::authorizeID:
  {
    // Authorize response: {result: true/false, error: ...}
    bool authorized = false;
    if (!packet["result"].is_null()) {
      if (packet["result"].is_bool())
        authorized = packet["result"].get_bool();
      else
        authorized = true; // non-null result = OK
    }
    if (!authorized) {
      std::string errorMsg = "authorization failed";
      if (packet.contains("error") && !packet["error"].is_null()) {
        auto &ev = packet["error"];
        if (ev.is_string())
          errorMsg = std::string(ev.get_string());
        else if (ev.is_array()) {
          auto &ea = ev.as_array();
          if (ea.size() > 1 && ea[1].is_string())
            errorMsg = std::string(ea[1].as_string());
          else if (ea.size() > 0 && ea[0].is_string())
            errorMsg = std::string(ea[0].as_string());
        } else if (ev.is_object() && ev.as_object().contains("message"))
          errorMsg = std::string(ev.as_object()["message"].as_string());
      }
      TNN_LOG_ERROR("%sKawPow Stratum: %s\n", isDev ? "DEV | " : "", errorMsg.c_str());
    }
    return 0;
  }
  break;

  default:
  {
    if (!SubmitTracker::isSubmitId(id)) break;
    int submitDevice = submitTracker.resolve(id);
    if (!packet["result"].is_null() && packet["result"].get_bool())
    {
      if (!isDev)
        accepted++;
      if (!isDev) recordDeviceShare(submitDevice, true);
      printf("\n");
      if (isDev) { setcolor(CYAN); printf("DEV | "); }
      printf("KawPow Stratum: share accepted\n");
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    else
    {
      if (!isDev)
        rejected++;
      if (!isDev) recordDeviceShare(submitDevice, false);

      std::string errorMsg = "Unknown error";
      if (packet.contains("error") && !packet["error"].is_null())
      {
        auto &ev = packet["error"];
        if (ev.is_string())
          errorMsg = std::string(ev.get_string());
        else if (ev.is_array()) {
          auto &ea = ev.as_array();
          if (ea.size() > 1 && ea[1].is_string())
            errorMsg = std::string(ea[1].as_string());
          else if (ea.size() > 0 && ea[0].is_string())
            errorMsg = std::string(ea[0].as_string());
        } else if (ev.is_object() && ev.as_object().contains("message"))
          errorMsg = std::string(ev.as_object()["message"].as_string());
      }
      printf("\n");
      if (isDev) { setcolor(CYAN); printf("DEV | "); }
      else setcolor(RED);
      printf("KawPow Stratum: share rejected: %s\n", errorMsg.c_str());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    break;
  }
  }
  return 0;
}

// ============================================================================
// Helpers (match Xelis pattern)
// ============================================================================

template <class Stream>
static inline void kp_send_json(Stream& stream, net::yield_context yield, beast::error_code& ec,
                                const boost::json::object& obj)
{
  std::string msg = boost::json::serialize(obj);
  msg.push_back('\n');
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
}

template <class Stream>
static void kp_dispatch_lines(
    Stream& stream,
    net::yield_context yield,
    beast::error_code& ec,
    std::string& packetBuffer,
    KawPowStratum::jobCache* jobCache,
    bool isDev,
    const char* tag)
{
  while (true) {
    auto pos = packetBuffer.find('\n');
    if (pos == std::string::npos) break;

    std::string line = packetBuffer.substr(0, pos);
    packetBuffer.erase(0, pos + 1);

    // strip CR
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line.empty()) continue;

    TNN_LOG_DEBUG("%s recv: %s\n", tag, line.c_str());

    try {
      auto rpc = boost::json::parse(line).as_object();
      if (rpc.contains("method")) {
        int result = handleKawPowStratumPacket(rpc, jobCache, isDev);
        if (result == 1) {
          boost::json::object pong({{"id", rpc["id"].get_uint64()},
                                    {"method", KawPowStratum::pong.method}});
          kp_send_json(stream, yield, ec, pong);
          if (ec) return;
        }
      } else {
        handleKawPowStratumResponse(rpc, jobCache, isDev);
      }
    } catch (const std::exception& e) {
      setcolor(RED);
      printf("Parse error: %s\nPacket: %s\n", e.what(), line.c_str());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
  }
}

template <class Stream>
static bool kp_read_and_dispatch(
    Stream& stream,
    net::yield_context yield,
    beast::error_code& ec,
    std::string& packetBuffer,
    boost::asio::streambuf& readbuf,
    KawPowStratum::jobCache* jobCache,
    bool isDev,
    bool* connectedFlag,
    bool* submittingFlag,
    std::atomic<bool>* abortFlag,
    const char* tag)
{
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));

  std::size_t n = boost::asio::async_read_until(stream, readbuf, "\n", yield[ec]);

  // Even on error/EOF, async_read_until may have transferred data — process it first
  if (n > 0) {
    std::string chunk(
        boost::asio::buffers_begin(readbuf.data()),
        boost::asio::buffers_begin(readbuf.data()) + n);
    readbuf.consume(n);
    packetBuffer += chunk;
  }

  if (packetBuffer.size() > 1024 * 1024) {
    setcolor(RED);
    printf("\nPacket buffer overflow, disconnecting\n");
    setcolor(BRIGHT_WHITE);
    fflush(stdout);
    setForDisconnected(connectedFlag, submittingFlag, abortFlag, &data_ready, &cv);
    return false;
  }

  // Dispatch all complete lines we have
  kp_dispatch_lines(stream, yield, ec, packetBuffer, jobCache, isDev, tag);

  // Now check if the read itself failed
  if (ec) {
    TNN_LOG_DEBUG("%s async_read error: %s\n", tag, ec.message().c_str());
    return false;
  }

  return true;
}

template <class Stream>
static bool kp_wait_for_response(
    Stream& stream,
    net::yield_context yield,
    beast::error_code& ec,
    std::string& packetBuffer,
    boost::asio::streambuf& readbuf,
    KawPowStratum::jobCache* jobCache,
    bool isDev,
    bool* connectedFlag,
    bool* submittingFlag,
    std::atomic<bool>* abortFlag,
    int64_t wantedId,
    const char* tag)
{
  while (!ABORT_MINER && !abortFlag->load()) {
    // Read more data if we don't have a complete line
    while (packetBuffer.find('\n') == std::string::npos) {
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      std::size_t n = boost::asio::async_read_until(stream, readbuf, "\n", yield[ec]);

      // Even on error/EOF, transfer any data we got
      if (n > 0) {
        std::string chunk(
            boost::asio::buffers_begin(readbuf.data()),
            boost::asio::buffers_begin(readbuf.data()) + n);
        readbuf.consume(n);
        packetBuffer += chunk;
      }

      if (ec) {
        TNN_LOG_DEBUG("%s wait_for_response read error: %s\n", tag, ec.message().c_str());
        // Process any complete lines we got before the error
        break;
      }

      if (packetBuffer.size() > 1024 * 1024) {
        setForDisconnected(connectedFlag, submittingFlag, abortFlag, &data_ready, &cv);
        return false;
      }
    }

    // Process all complete lines we have
    bool foundWanted = false;
    while (true) {
      auto pos = packetBuffer.find('\n');
      if (pos == std::string::npos) break;

      std::string line = packetBuffer.substr(0, pos);
      packetBuffer.erase(0, pos + 1);

      while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
      if (line.empty()) continue;

      TNN_LOG_DEBUG("%s recv: %s\n", tag, line.c_str());

      try {
        auto rpc = boost::json::parse(line).as_object();

        bool isWanted = false;
        if (!rpc.contains("method") && rpc.contains("id")) {
          int64_t id = rpc["id"].to_number<int64_t>();
          if (id == wantedId) isWanted = true;
        }

        if (rpc.contains("method")) {
          int result = handleKawPowStratumPacket(rpc, jobCache, isDev);
          if (result == 1) {
            boost::json::object pong({{"id", rpc["id"].get_uint64()},
                                      {"method", KawPowStratum::pong.method}});
            kp_send_json(stream, yield, ec, pong);
            if (ec) return false;
          }
        } else {
          handleKawPowStratumResponse(rpc, jobCache, isDev);
        }

        if (isWanted) { foundWanted = true; break; }
      } catch (const std::exception& e) {
        setcolor(RED);
        printf("Parse error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }

    if (foundWanted) return true;

    // If we had an error and didn't find what we wanted, give up
    if (ec) return false;
  }
  return false;
}

// ============================================================================
// KawPow stratum session (SSL)
// ============================================================================

void kawpow_stratum_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  KawPowStratum::lastReceivedJobTime = 0;
  beast::error_code ec;

  const char* tag = isDev ? "[KP-SSL-DEV]" : "[KP-SSL]";
  TNN_LOG_DEBUG("%s ssl session start: host=%s port=%s\n", tag, host.c_str(), port.c_str());

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);

  auto strand = net::make_strand(ioc);
  boost::beast::tcp_stream stream(strand);

  TNN_LOG_DEBUG("%s connecting...\n", tag);
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) {
    TNN_LOG_DEBUG("%s connect failed: %s\n", tag, ec.message().c_str());
    return fail(ec, "connect");
  }
  TNN_LOG_DEBUG("%s connected\n", tag);

  std::string packetBuffer;
  boost::asio::streambuf readbuf;

  bool *C = isDev ? &devConnected : &isConnected;
  bool *B = isDev ? &submittingDev : &submitting;

  std::atomic<bool> abort{false};

  std::string minerName = "tnn-miner/" + std::string(versionString);
  KawPowStratum::jobCache jobCache;

  StratumWriteQueue<decltype(stream)> writeq(stream, ioc);

  // Subscribe
  {
    boost::json::object packet = KawPowStratum::stratumCall;
    packet["id"] = KawPowStratum::subscribe.id;
    packet["method"] = KawPowStratum::subscribe.method;
    packet["params"] = boost::json::array({minerName});

    TNN_LOG_DEBUG("%s sending subscribe\n", tag);
    kp_send_json(stream, yield, ec, packet);
    if (ec) return fail(ec, "KawPow subscribe");

    TNN_LOG_DEBUG("%s waiting for subscribe response\n", tag);
    if (!kp_wait_for_response(stream, yield, ec, packetBuffer, readbuf, &jobCache,
                              isDev, C, B, &abort, KawPowStratum::subscribeID, tag)) {
      TNN_LOG_DEBUG("%s subscribe response failed\n", tag);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      return;
    }
    TNN_LOG_DEBUG("%s subscribe OK\n", tag);
  }

  // Authorize
  {
    boost::json::object packet = KawPowStratum::stratumCall;
    packet["id"] = KawPowStratum::authorize.id;
    packet["method"] = KawPowStratum::authorize.method;
    if (isDev) {
      packet["params"] = boost::json::array({devWallet + "." + worker + "-" + tnnTargetArch + "-tnn-dev", "x"});
    } else {
      packet["params"] = boost::json::array({wallet + "." + worker, stratumPassword});
    }

    TNN_LOG_DEBUG("%s sending authorize\n", tag);
    kp_send_json(stream, yield, ec, packet);
    if (ec) return fail(ec, "KawPow authorize");

    TNN_LOG_DEBUG("%s waiting for authorize response\n", tag);
    if (!kp_wait_for_response(stream, yield, ec, packetBuffer, readbuf, &jobCache,
                              isDev, C, B, &abort, KawPowStratum::authorizeID, tag)) {
      TNN_LOG_DEBUG("%s authorize response failed\n", tag);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      return;
    }
    TNN_LOG_DEBUG("%s authorize OK\n", tag);
  }

  KawPowStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();

  // Submit thread
  TNN_LOG_DEBUG("%s launching submit thread\n", tag);
  std::thread subThread([&]() {
    while (!abort.load()) {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&] { return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;

      try {
        boost::json::object& S = isDev ? devShare : share;
        hoist_rpc_id(S);
        std::string msg = boost::json::serialize(S) + "\n";
        writeq.enqueue(std::move(msg));
        if (!isDev) KawPowStratum::lastShareSubmissionTime =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
      } catch (const std::exception& e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        abort.store(true);
        break;
      }

      *B = false;
      data_ready = false;
      std::this_thread::yield();
    }
  });

  // RAII guard: ensures subThread is always joined, even if an exception unwinds the stack
  ThreadGuard threadGuard(subThread, abort, cv);

  // Main read loop
  while (!ABORT_MINER && !abort.load() && !writeq.abort.load()) {
    try {
      if (KawPowStratum::lastReceivedJobTime > 0 &&
          (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count() -
           KawPowStratum::lastReceivedJobTime) > KawPowStratum::jobTimeout) {
        TNN_LOG_DEBUG("%s timeout\n", tag);
        setcolor(RED);
        printf("timeout\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      if (!kp_read_and_dispatch(stream, yield, ec, packetBuffer, readbuf, &jobCache,
                                isDev, C, B, &abort, tag)) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        fail(ec, "async_read");
        break;
      }
    } catch (const std::exception& e) {
      TNN_LOG_DEBUG("%s main loop exception: %s\n", tag, e.what());
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      break;
    }

    std::this_thread::yield();
    if (ABORT_MINER) {
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      ioc.stop();
    }
  }

  // Shutdown (ThreadGuard destructor handles abort + join)
  writeq.abort.store(true);
  TNN_LOG_DEBUG("%s session closed\n", tag);
  stream.close();
}

// ============================================================================
// KawPow stratum session (no SSL)
// ============================================================================

void kawpow_stratum_session_nossl(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  KawPowStratum::lastReceivedJobTime = 0;
  beast::error_code ec;

  const char* tag = isDev ? "[KP-DEV]" : "[KP]";
  TNN_LOG_DEBUG("%s nossl session start: host=%s port=%s wallet=%s\n", tag, host.c_str(), port.c_str(), wallet.c_str());

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);

  auto strand = net::make_strand(ioc);
  boost::beast::tcp_stream stream(strand);

  TNN_LOG_DEBUG("%s connecting...\n", tag);
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) {
    TNN_LOG_DEBUG("%s connect failed: %s\n", tag, ec.message().c_str());
    return fail(ec, "connect");
  }
  TNN_LOG_DEBUG("%s connected\n", tag);

  std::string packetBuffer;
  boost::asio::streambuf readbuf;

  bool *C = isDev ? &devConnected : &isConnected;
  bool *B = isDev ? &submittingDev : &submitting;

  std::atomic<bool> abort{false};

  std::string minerName = "tnn-miner/" + std::string(versionString);
  KawPowStratum::jobCache jobCache;

  StratumWriteQueue<decltype(stream)> writeq(stream, ioc);

  // Subscribe
  {
    boost::json::object packet = KawPowStratum::stratumCall;
    packet["id"] = KawPowStratum::subscribe.id;
    packet["method"] = KawPowStratum::subscribe.method;
    packet["params"] = boost::json::array({minerName});

    TNN_LOG_DEBUG("%s sending subscribe\n", tag);
    kp_send_json(stream, yield, ec, packet);
    if (ec) return fail(ec, "KawPow subscribe");

    TNN_LOG_DEBUG("%s waiting for subscribe response\n", tag);
    if (!kp_wait_for_response(stream, yield, ec, packetBuffer, readbuf, &jobCache,
                              isDev, C, B, &abort, KawPowStratum::subscribeID, tag)) {
      TNN_LOG_DEBUG("%s subscribe response failed\n", tag);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      return;
    }
    TNN_LOG_DEBUG("%s subscribe OK\n", tag);
  }

  // Authorize
  {
    boost::json::object packet = KawPowStratum::stratumCall;
    packet["id"] = KawPowStratum::authorize.id;
    packet["method"] = KawPowStratum::authorize.method;
    if (isDev) {
      packet["params"] = boost::json::array({devWallet + "." + worker + "-" + tnnTargetArch, "x"});
    } else {
      packet["params"] = boost::json::array({wallet + "." + worker, stratumPassword});
    }

    TNN_LOG_DEBUG("%s sending authorize\n", tag);
    kp_send_json(stream, yield, ec, packet);
    if (ec) return fail(ec, "KawPow authorize");

    TNN_LOG_DEBUG("%s waiting for authorize response\n", tag);
    if (!kp_wait_for_response(stream, yield, ec, packetBuffer, readbuf, &jobCache,
                              isDev, C, B, &abort, KawPowStratum::authorizeID, tag)) {
      TNN_LOG_DEBUG("%s authorize response failed\n", tag);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      return;
    }
    TNN_LOG_DEBUG("%s authorize OK\n", tag);
  }

  KawPowStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();

  // Submit thread
  TNN_LOG_DEBUG("%s launching submit thread\n", tag);
  std::thread subThread([&]() {
    while (!abort.load()) {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&] { return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;

      try {
        boost::json::object& S = isDev ? devShare : share;
        hoist_rpc_id(S);
        std::string msg = boost::json::serialize(S) + "\n";
        writeq.enqueue(std::move(msg));
        if (!isDev) KawPowStratum::lastShareSubmissionTime =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
      } catch (const std::exception& e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        abort.store(true);
        break;
      }

      *B = false;
      data_ready = false;
      std::this_thread::yield();
    }
  });

  // RAII guard: ensures subThread is always joined, even if an exception unwinds the stack
  ThreadGuard threadGuard(subThread, abort, cv);

  // Main read loop
  while (!ABORT_MINER && !abort.load() && !writeq.abort.load()) {
    try {
      if (KawPowStratum::lastReceivedJobTime > 0 &&
          (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count() -
           KawPowStratum::lastReceivedJobTime) > KawPowStratum::jobTimeout) {
        TNN_LOG_DEBUG("%s timeout\n", tag);
        setcolor(RED);
        printf("timeout\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      if (!kp_read_and_dispatch(stream, yield, ec, packetBuffer, readbuf, &jobCache,
                                isDev, C, B, &abort, tag)) {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        fail(ec, "async_read");
        break;
      }
    } catch (const std::exception& e) {
      TNN_LOG_DEBUG("%s main loop exception: %s\n", tag, e.what());
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      break;
    }

    std::this_thread::yield();
    if (ABORT_MINER) {
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      ioc.stop();
    }
  }

  // Shutdown (ThreadGuard destructor handles abort + join)
  writeq.abort.store(true);
  TNN_LOG_DEBUG("%s session closed\n", tag);
  beast::get_lowest_layer(stream).close();
}
