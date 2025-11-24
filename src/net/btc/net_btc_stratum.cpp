#include "../net.hpp"
#include <hex.h>

#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/json.hpp>

#include <openssl/sha.h>

#include <endian.hpp>
#include <stratum/stratum.h>

#include <atomic>
#include <queue>

namespace beast = boost::beast;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

std::vector<uint8_t> sha256d(const std::vector<uint8_t> &input)
{
  std::vector<uint8_t> hash(32);
  std::vector<uint8_t> temp(32);

  SHA256_CTX sha256;
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, input.data(), input.size());
  SHA256_Final(temp.data(), &sha256);

  SHA256_Init(&sha256);
  SHA256_Update(&sha256, temp.data(), temp.size());
  SHA256_Final(hash.data(), &sha256);

  return hash;
}

std::vector<uint8_t> calculateMerkleRoot(const BTCStratum::jobCache &cache)
{
  std::vector<uint8_t> merkleRoot = sha256d(cache.coinbase);

  for (const auto &branchBin : cache.merkleTree)
  {
    std::vector<uint8_t> combined;
    combined.insert(combined.end(), merkleRoot.begin(), merkleRoot.end());
    combined.insert(combined.end(), branchBin.begin(), branchBin.end());

    merkleRoot = sha256d(combined);
  }

  return merkleRoot;
}

std::string buildBlockHeader(const BTCStratum::jobCache &cache)
{
  std::vector<uint8_t> merkleRoot = calculateMerkleRoot(cache);

  unsigned char blockHeader[80];
  memset(blockHeader, 0, 80);

  switch (current_algo_config.header_endian)
  {
  case ENDIAN_LITTLE:
    le32enc(blockHeader + 0, cache.version);
    memcpy(blockHeader + 4, cache.prevHash.data(), 32);
    memcpy(blockHeader + 36, merkleRoot.data(), 32);
    le32enc(blockHeader + 68, cache.nTime);
    le32enc(blockHeader + 72, cache.nBits);
    break;

  case ENDIAN_SWAP_32:
    le32enc(blockHeader + 0, cache.version);

    if (current_algo_config.swap_prev_hash)
    {
      for (int i = 0; i < 8; i++)
      {
        be32enc(blockHeader + 4 + i * 4, ((uint32_t *)cache.prevHash.data())[i]);
      }
    }
    else
    {
      memcpy(blockHeader + 4, cache.prevHash.data(), 32);
    }

    if (current_algo_config.swap_merkle_root)
    {
      for (int i = 0; i < 8; i++)
      {
        be32enc(blockHeader + 36 + i * 4, ((uint32_t *)merkleRoot.data())[i]);
      }
    }
    else
    {
      memcpy(blockHeader + 36, merkleRoot.data(), 32);
    }

    le32enc(blockHeader + 68, cache.nTime);
    le32enc(blockHeader + 72, cache.nBits);
    break;

  case ENDIAN_SWAP_32_BE:
    be32enc(blockHeader + 0, cache.version);

    if (current_algo_config.swap_prev_hash)
    {
      for (int i = 0; i < 8; i++)
      {
        be32enc(blockHeader + 4 + i * 4, ((uint32_t *)cache.prevHash.data())[i]);
      }
    }
    else
    {
      memcpy(blockHeader + 4, cache.prevHash.data(), 32);
    }

    if (current_algo_config.swap_merkle_root)
    {
      for (int i = 0; i < 8; i++)
      {
        be32enc(blockHeader + 36 + i * 4, ((uint32_t *)merkleRoot.data())[i]);
      }
    }
    else
    {
      memcpy(blockHeader + 36, merkleRoot.data(), 32);
    }

    be32enc(blockHeader + 68, cache.nTime);
    be32enc(blockHeader + 72, cache.nBits);
    break;

  case ENDIAN_BIG:
    be32enc(blockHeader + 0, cache.version);
    memcpy(blockHeader + 4, cache.prevHash.data(), 32);
    memcpy(blockHeader + 36, merkleRoot.data(), 32);
    be32enc(blockHeader + 68, cache.nTime);
    be32enc(blockHeader + 72, cache.nBits);
    break;

  case ENDIAN_MIXED:
    break;
  }

  return hexStr(blockHeader, 80);
}

int handleBTCStratumPacket(boost::json::object packet, BTCStratum::jobCache *cache, bool isDev)
{
  std::string M = std::string(packet["method"].as_string());

  if (M == BTCStratum::s_notify)
  {
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    boost::json::value *J = isDev ? &devJob : &job;
    int64_t *h = isDev ? &devHeight : &ourHeight;

    auto params = packet["params"].as_array();

    cache->jobId = std::string(params[0].as_string());

    std::string prevHashHex = std::string(params[1].as_string());
    cache->prevHash.resize(32);
    hexstrToBytes(prevHashHex, cache->prevHash.data());

    std::string coinb1Hex = std::string(params[2].as_string());
    std::string coinb2Hex = std::string(params[3].as_string());
    size_t coinb1_size = coinb1Hex.length() / 2;
    size_t coinb2_size = coinb2Hex.length() / 2;

    size_t total_coinbase_size = coinb1_size +
                                 cache->extraNonce1.size() +
                                 cache->extraNonce2Size +
                                 coinb2_size;
    cache->coinbase.resize(total_coinbase_size);

    uint8_t *pCoinbase = cache->coinbase.data();
    hexstrToBytes(coinb1Hex, pCoinbase);
    pCoinbase += coinb1_size;

    memcpy(pCoinbase, cache->extraNonce1.data(), cache->extraNonce1.size());
    pCoinbase += cache->extraNonce1.size();

    memset(pCoinbase, 0, cache->extraNonce2Size);
    pCoinbase += cache->extraNonce2Size;

    hexstrToBytes(coinb2Hex, pCoinbase);

    auto merkleArray = params[4].as_array();
    cache->merkleTree.clear();
    cache->merkleTree.reserve(merkleArray.size());
    for (const auto &branch : merkleArray)
    {
      std::string branchHex = std::string(branch.as_string());
      std::vector<uint8_t> branchBin(32);
      hexstrToBytes(branchHex, branchBin.data());
      cache->merkleTree.push_back(std::move(branchBin));
    }

    std::string versionHex = std::string(params[5].as_string());
    uint8_t versionBin[4];
    hexstrToBytes(versionHex, versionBin);
    cache->version = le32dec(versionBin);

    std::string nBitsHex = std::string(params[6].as_string());
    uint8_t nBitsBin[4];
    hexstrToBytes(nBitsHex, nBitsBin);
    cache->nBits = le32dec(nBitsBin);

    std::string nTimeHex = std::string(params[7].as_string());
    uint8_t nTimeBin[4];
    hexstrToBytes(nTimeHex, nTimeBin);
    cache->nTime = le32dec(nTimeBin);

    cache->cleanJobs = params[8].as_bool();

    std::string blockTemplate = buildBlockHeader(*cache);

    (*J).as_object()["jobId"] = cache->jobId;
    (*J).as_object()["template"] = blockTemplate;
    (*J).as_object()["extraNonce2"] = cache->extraNonce2;
    (*J).as_object()["extraNonce2Size"] = cache->extraNonce2Size;
    (*J).as_object()["nTime"] = uint32ArrayToHex(&cache->nTime, 1);

    BTCStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count();

    bool *C = isDev ? &devConnected : &isConnected;
    if (!beQuiet)
    {
      setcolor(CYAN);
      printf("\n");
      if (isDev) printf("DEV | ");
      printf("Stratum: new job received\n");
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
  else if (M == BTCStratum::s_setDifficulty)
  {
    double *d = isDev ? &doubleDiffDev : &doubleDiff;
    (*d) = packet["params"].as_array()[0].get_double();
    if ((*d) < 0.00000000001)
      (*d) = packet["params"].as_array()[0].get_uint64();

    cache->difficulty = *d;

    if (!beQuiet)
    {
      setcolor(CYAN);
      if (isDev) printf("DEV | ");
      printf("Difficulty set to: %.8f\n", *d);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    jobCounter++;
  }
  else if (M == BTCStratum::s_ping)
  {
    return 1;
  }
  else
  {
    std::string packetStr = boost::json::serialize(packet);
    std::cout << "Stratum: unrecognized packet: " << packetStr << std::endl;
  }

  return 0;
}

int handleBTCStratumResponse(boost::json::object packet, BTCStratum::jobCache *cache, bool isDev)
{
  if (!packet.contains("id")) return 0;
  
  int64_t id = packet["id"].to_number<int64_t>();

  switch (id)
  {
  case BTCStratum::subscribeID:
  {
    if (!packet["result"].is_null())
    {
      auto result = packet["result"].as_array();
      if (result.size() >= 2)
      {
        std::string xnonce1_hex = std::string(result[1].as_string());
        cache->extraNonce1Size = xnonce1_hex.length() / 2;
        cache->extraNonce1.resize(cache->extraNonce1Size);
        hexstrToBytes(xnonce1_hex, cache->extraNonce1.data());
        cache->extraNonce2Size = result[2].get_int64();
      }
      return 0;
    }
    else
    {
      std::string errorMsg = "Unknown error";
      if (packet.contains("error") && packet["error"].is_string()) {
        errorMsg = std::string(packet["error"].get_string());
      }
      setcolor(RED);
      printf("\n");
      if (isDev)
      {
        setcolor(CYAN);
        printf("DEV | ");
      }
      printf("Stratum ERROR: %s\n", errorMsg.c_str());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return -1;
    }
  }
  break;

  case BTCStratum::submitID:
  {
    printf("\n");
    if (isDev)
    {
      setcolor(CYAN);
      printf("DEV | ");
    }
    if (!packet["result"].is_null() && packet["result"].get_bool())
    {
      if (!isDev) accepted++;
      std::cout << "Stratum: share accepted" << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    else
    {
      if (!isDev) rejected++;
      if (!isDev) setcolor(RED);

      std::string errorMsg = "Unknown error";
      if (packet.contains("error"))
      {
        if (packet["error"].is_array()) {
          errorMsg = std::string(packet["error"].as_array()[1].as_string());
        } else if (packet["error"].is_object() && packet["error"].as_object().contains("message")) {
          errorMsg = std::string(packet["error"].as_object()["message"].as_string());
        } else if (packet["error"].is_string()) {
          errorMsg = std::string(packet["error"].get_string());
        }
      }
      std::cout << "Stratum: share rejected: " << errorMsg << std::endl;

      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    break;
  }
  }
  return 0;
}

void btc_stratum_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  beast::error_code ec;

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  auto strand = net::make_strand(ioc);
  boost::beast::ssl_stream<boost::beast::tcp_stream> stream(strand, ctx);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-btc-ssl");

  if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
  {
    throw beast::system_error{
        static_cast<int>(::ERR_get_error()),
        boost::asio::error::get_ssl_category()};
  }

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  stream.async_handshake(ssl::stream_base::client, yield[ec]);
  if (ec) return fail(ec, "handshake-btc-ssl");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  BTCStratum::jobCache jobCache;
  std::string packetBuffer;

  // Subscribe
  boost::json::object packet = BTCStratum::stratumCall;
  packet["id"] = BTCStratum::subscribe.id;
  packet["method"] = BTCStratum::subscribe.method;
  packet["params"] = boost::json::array({minerName});
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum subscribe");
  }

  // Authorize
  packet = BTCStratum::stratumCall;
  packet["id"] = BTCStratum::authorize.id;
  packet["method"] = BTCStratum::authorize.method;
  if (isDev) {
    packet["params"] = boost::json::array({devWallet + "." + worker + "-" + tnnTargetArch + "-tnn-dev"});
  } else {
    packet["params"] = boost::json::array({wallet + "." + worker, stratumPassword});
  }
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum authorize");
  }

  BTCStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  std::queue<std::string> submitQueue;
  boost::mutex submitMutex;
  std::atomic<bool> abort{false};

  auto process_write_queue = [&]() {
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          if (submitQueue.empty()) return;
          msg = std::move(submitQueue.front());
          submitQueue.pop();
        }
        
        beast::error_code wec;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
        boost::asio::write(stream, boost::asio::buffer(msg), wec);
        
        if (wec) {
          printf("error on write: %s\n", wec.message().c_str());
          fflush(stdout);
          abort = true;
          return;
        }
        
        if (!isDev) {
          BTCStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
        }
      }
    });
  };

  bool submitThreadRunning = true;

  boost::thread subThread([&]() {
    while (!abort.load()) {
      boost::unique_lock<boost::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&] { return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;

      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          submitQueue.push(std::move(msg));
        }
        process_write_queue();
        
      } catch (const std::exception &e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        break;
      }
      *B = false;
      data_ready = false;
      boost::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  while (!ABORT_MINER && !abort.load())
  {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try
    {
      if (BTCStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() - 
          BTCStratum::lastReceivedJobTime > BTCStratum::jobTimeout)
      {
        setcolor(RED);
        printf("\nStratum session timed out\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t n = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec)
      {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      if (n > 0)
      {
        std::string newData(
            boost::asio::buffers_begin(response.data()),
            boost::asio::buffers_begin(response.data()) + n
        );
        response.consume(n);
        packetBuffer += newData;

        if (packetBuffer.size() > 1024 * 1024)
        {
          setcolor(RED);
          printf("\nPacket buffer overflow, disconnecting\n");
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
          setForDisconnected(C, B, &abort, &data_ready, &cv);
          break;
        }

        size_t pos;
        while ((pos = packetBuffer.find('\n')) != std::string::npos)
        {
          std::string line = packetBuffer.substr(0, pos);
          packetBuffer.erase(0, pos + 1);

          if (line.empty()) continue;

          try
          {
            boost::json::object sRPC = boost::json::parse(line).as_object();
            if (sRPC.contains("method"))
            {
              int result = handleBTCStratumPacket(sRPC, &jobCache, isDev);
              if (result == 1)
              {
                boost::json::object pong = {
                    {"id", sRPC["id"].get_uint64()},
                    {"method", BTCStratum::pong.method}
                };
                std::string pongPacket = boost::json::serialize(pong) + "\n";
                boost::asio::async_write(stream, boost::asio::buffer(pongPacket), yield[ec]);
                if (ec) {
                  setForDisconnected(C, B, &abort, &data_ready, &cv);
                  break;
                }
              }
            }
            else
            {
              handleBTCStratumResponse(sRPC, &jobCache, isDev);
            }
          }
          catch (const std::exception &e)
          {
            setcolor(RED);
            printf("\nParse error: %s\n", e.what());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      printf("\nSession exception: %s\n", e.what());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }

    boost::this_thread::yield();
  }

  abort = true;
  cv.notify_all();
  
  if (submitThreadRunning) {
    subThread.interrupt();
    if (subThread.joinable()) {
      subThread.join();
    }
  }

  beast::error_code shutdown_ec;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(5));
  stream.async_shutdown(yield[shutdown_ec]);
  beast::get_lowest_layer(stream).close();
}

void btc_stratum_session_nossl(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    std::string const &worker,
    net::io_context &ioc,
    ssl::context &ctx,
    net::yield_context yield,
    bool isDev)
{
  beast::error_code ec;

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  
  auto strand = net::make_strand(ioc);
  boost::beast::tcp_stream stream(strand);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec) return fail(ec, "connect-btc-nossl");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  BTCStratum::jobCache jobCache;
  std::string packetBuffer;

  // Subscribe
  boost::json::object packet = BTCStratum::stratumCall;
  packet["id"] = BTCStratum::subscribe.id;
  packet["method"] = BTCStratum::subscribe.method;
  packet["params"] = boost::json::array({minerName});
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum subscribe");
  }

  // Authorize
  packet = BTCStratum::stratumCall;
  packet["id"] = BTCStratum::authorize.id;
  packet["method"] = BTCStratum::authorize.method;
  if (isDev) {
    packet["params"] = boost::json::array({devWallet + "." + worker + "-" + tnnTargetArch});
  } else {
    packet["params"] = boost::json::array({wallet + "." + worker, stratumPassword});
  }
  {
    std::string msg = boost::json::serialize(packet) + "\n";
    beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
    boost::asio::async_write(stream, boost::asio::buffer(msg), yield[ec]);
    if (ec) return fail(ec, "Stratum authorize");
  }

  BTCStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  std::queue<std::string> submitQueue;
  boost::mutex submitMutex;
  std::atomic<bool> abort{false};

  auto process_write_queue = [&]() {
    net::post(strand, [&]() {
      while (!abort.load()) {
        std::string msg;
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          if (submitQueue.empty()) return;
          msg = std::move(submitQueue.front());
          submitQueue.pop();
        }
        
        beast::error_code wec;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
        boost::asio::write(stream, boost::asio::buffer(msg), wec);
        
        if (wec) {
          printf("error on write: %s\n", wec.message().c_str());
          fflush(stdout);
          abort = true;
          return;
        }
        
        if (!isDev) {
          BTCStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
        }
      }
    });
  };

  bool submitThreadRunning = true;

  boost::thread subThread([&]() {
    while (!abort.load()) {
      boost::unique_lock<boost::mutex> lock(mutex);
      bool *B = isDev ? &submittingDev : &submitting;
      cv.wait(lock, [&] { return (data_ready && (*B)) || abort.load(); });
      if (abort.load()) break;

      try {
        boost::json::object &S = isDev ? devShare : share;
        std::string msg = boost::json::serialize(S) + "\n";
        
        {
          boost::lock_guard<boost::mutex> qlock(submitMutex);
          submitQueue.push(std::move(msg));
        }
        process_write_queue();
        
      } catch (const std::exception &e) {
        setcolor(RED);
        printf("\nSubmit thread error: %s\n", e.what());
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        break;
      }
      *B = false;
      data_ready = false;
      boost::this_thread::yield();
    }
    submitThreadRunning = false;
  });

  while (!ABORT_MINER && !abort.load())
  {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try
    {
      if (BTCStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() - 
          BTCStratum::lastReceivedJobTime > BTCStratum::jobTimeout)
      {
        setcolor(RED);
        printf("\nStratum session timed out\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));
      size_t n = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec)
      {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        break;
      }

      if (n > 0)
      {
        std::string newData(
            boost::asio::buffers_begin(response.data()),
            boost::asio::buffers_begin(response.data()) + n
        );
        response.consume(n);
        packetBuffer += newData;

        if (packetBuffer.size() > 1024 * 1024)
        {
          setcolor(RED);
          printf("\nPacket buffer overflow, disconnecting\n");
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
          setForDisconnected(C, B, &abort, &data_ready, &cv);
          break;
        }

        size_t pos;
        while ((pos = packetBuffer.find('\n')) != std::string::npos)
        {
          std::string line = packetBuffer.substr(0, pos);
          packetBuffer.erase(0, pos + 1);

          if (line.empty()) continue;

          try
          {
            boost::json::object sRPC = boost::json::parse(line).as_object();
            if (sRPC.contains("method"))
            {
              int result = handleBTCStratumPacket(sRPC, &jobCache, isDev);
              if (result == 1)
              {
                boost::json::object pong = {
                    {"id", sRPC["id"].get_uint64()},
                    {"method", BTCStratum::pong.method}
                };
                std::string pongPacket = boost::json::serialize(pong) + "\n";
                boost::asio::async_write(stream, boost::asio::buffer(pongPacket), yield[ec]);
                if (ec) {
                  setForDisconnected(C, B, &abort, &data_ready, &cv);
                  break;
                }
              }
            }
            else
            {
              handleBTCStratumResponse(sRPC, &jobCache, isDev);
            }
          }
          catch (const std::exception &e)
          {
            setcolor(RED);
            printf("\nParse error: %s\n", e.what());
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      setcolor(RED);
      printf("\nSession exception: %s\n", e.what());
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      break;
    }

    boost::this_thread::yield();
  }

  abort = true;
  cv.notify_all();
  
  if (submitThreadRunning) {
    subThread.interrupt();
    if (subThread.joinable()) {
      subThread.join();
    }
  }

  beast::error_code close_ec;
  stream.close();
}