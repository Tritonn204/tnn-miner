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

namespace beast = boost::beast;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

// SHA256d (double SHA256) helper
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
    // Little-endian format
    le32enc(blockHeader + 0, cache.version);
    memcpy(blockHeader + 4, cache.prevHash.data(), 32);
    memcpy(blockHeader + 36, merkleRoot.data(), 32);
    le32enc(blockHeader + 68, cache.nTime);
    le32enc(blockHeader + 72, cache.nBits);
    break;

  case ENDIAN_SWAP_32:
    // Bitcoin-style: swap 32-bit chunks
    le32enc(blockHeader + 0, cache.version);

    // Handle prevHash swapping if configured
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

    // Handle merkleRoot swapping if configured
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
    // Bitcoin-style: swap 32-bit chunks
    be32enc(blockHeader + 0, cache.version);

    // Handle prevHash swapping if configured
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

    // Handle merkleRoot swapping if configured
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
    // Pure big-endian (rare, but included for completeness)
    be32enc(blockHeader + 0, cache.version);
    memcpy(blockHeader + 4, cache.prevHash.data(), 32);
    memcpy(blockHeader + 36, merkleRoot.data(), 32);
    be32enc(blockHeader + 68, cache.nTime);
    be32enc(blockHeader + 72, cache.nBits);
    break;

  case ENDIAN_MIXED:
    // Reserved for future custom per-field handling
    // Could implement algorithm-specific logic here
    break;
  }

  return hexStr(blockHeader, 80);
}

int handleBTCStratumPacket(boost::json::object packet, BTCStratum::jobCache *cache, bool isDev)
{
  std::string M = packet["method"].as_string().c_str();

  if (M.compare(BTCStratum::s_notify) == 0)
  {
    // printf("%s%s\n", isDev ? "DEV: " : "USER: ", boost::json::serialize(packet).c_str());
    std::scoped_lock<boost::mutex> lockGuard(mutex);
    boost::json::value *J = isDev ? &devJob : &job;
    int64_t *h = isDev ? &devHeight : &ourHeight;

    auto params = packet["params"].as_array();

    // Job ID can stay as string
    cache->jobId = params[0].as_string().c_str();

    // Convert prevHash (32 bytes)
    std::string prevHashHex = params[1].as_string().c_str();
    cache->prevHash.resize(32);
    hexstrToBytes(prevHashHex, cache->prevHash.data());

    // Handle coinbase parts
    std::string coinb1Hex = params[2].as_string().c_str();
    std::string coinb2Hex = params[3].as_string().c_str();
    size_t coinb1_size = coinb1Hex.length() / 2;
    size_t coinb2_size = coinb2Hex.length() / 2;

    // Calculate total coinbase size and resize buffer
    size_t total_coinbase_size = coinb1_size +
                                 cache->extraNonce1.size() +
                                 cache->extraNonce2Size +
                                 coinb2_size;
    cache->coinbase.resize(total_coinbase_size);

    // Assemble complete coinbase
    uint8_t *pCoinbase = cache->coinbase.data();
    hexstrToBytes(coinb1Hex, pCoinbase);
    pCoinbase += coinb1_size;

    // Copy extraNonce1 (should already be binary from subscription)
    memcpy(pCoinbase, cache->extraNonce1.data(), cache->extraNonce1.size());
    pCoinbase += cache->extraNonce1.size();

    // Copy extraNonce2 is left as is (will be filled during mining)
    memset(pCoinbase, 0, cache->extraNonce2Size);
    pCoinbase += cache->extraNonce2Size;

    // Add coinbase2
    hexstrToBytes(coinb2Hex, pCoinbase);

    // Convert merkle branches to binary
    auto merkleArray = params[4].as_array();
    cache->merkleTree.clear();
    cache->merkleTree.reserve(merkleArray.size());
    for (const auto &branch : merkleArray)
    {
      std::string branchHex = branch.as_string().c_str();
      std::vector<uint8_t> branchBin(32); // Each merkle branch is 32 bytes
      hexstrToBytes(branchHex, branchBin.data());
      cache->merkleTree.push_back(std::move(branchBin));
    }

    // Convert version (4 bytes)
    std::string versionHex = params[5].as_string().c_str();
    uint8_t versionBin[4];
    hexstrToBytes(versionHex, versionBin);
    cache->version = le32dec(versionBin);

    // Convert nBits (4 bytes)
    std::string nBitsHex = params[6].as_string().c_str();
    uint8_t nBitsBin[4];
    hexstrToBytes(nBitsHex, nBitsBin);
    cache->nBits = le32dec(nBitsBin);

    // Convert nTime (4 bytes)
    std::string nTimeHex = params[7].as_string().c_str();
    uint8_t nTimeBin[4];
    hexstrToBytes(nTimeHex, nTimeBin);
    cache->nTime = le32dec(nTimeBin);

    cache->cleanJobs = params[8].as_bool();

    // Build block template (you'll need to modify this to use binary data)
    std::string blockTemplate = buildBlockHeader(*cache);

    // Update job info
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
      if (isDev)
        printf("DEV | ");
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
  else if (M.compare(BTCStratum::s_setDifficulty) == 0)
  {
    double *d = isDev ? &doubleDiffDev : &doubleDiff;
    (*d) = packet["params"].as_array()[0].get_double();
    if ((*d) < 0.00000000001)
      (*d) = packet["params"].as_array()[0].get_uint64();

    cache->difficulty = *d;

    if (!beQuiet)
    {
      setcolor(CYAN);
      if (isDev)
        printf("DEV | ");
      printf("Difficulty set to: %.8f\n", *d);
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }

    jobCounter++;
  }
  else if (M.compare(BTCStratum::s_ping) == 0)
  {
    // Ping will be handled in main loop
    return 1; // Signal to send pong
  }
  else
  {
    std::cout << "Stratum: unrecognized packet: " << boost::json::serialize(packet).c_str() << std::endl;
  }

  return 0;
}

int handleBTCStratumResponse(boost::json::object packet, BTCStratum::jobCache *cache, bool isDev)
{
  if (!packet.contains("id"))
    return 0;
  int64_t id = packet["id"].to_number<int64_t>();

  switch (id)
  {
  case BTCStratum::subscribeID:
  {
    // printf("%s%s\n", (isDev ? "DEV: " : ""), boost::json::serialize(packet).c_str());
    if (!packet["result"].is_null())
    {
      auto result = packet["result"].as_array();
      // Extract subscription details and extranonce1
      if (result.size() >= 2)
      {
        std::string xnonce1_hex = result[1].as_string().c_str();
        cache->extraNonce1Size = xnonce1_hex.length() / 2;
        cache->extraNonce1.resize(cache->extraNonce1Size);
        hexstrToBytes(xnonce1_hex, cache->extraNonce1.data());
        cache->extraNonce2Size = result[2].get_int64();
      }
      return 0;
    }
    else
    {
      const char *errorMsg = packet["error"].get_string().c_str();
      setcolor(RED);
      printf("\n");
      if (isDev)
      {
        setcolor(CYAN);
        printf("DEV | ");
      }
      printf("Stratum ERROR: %s\n", errorMsg);
      fflush(stdout);
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
      if (!isDev)
        accepted++;
      std::cout << "Stratum: share accepted" << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
    }
    else
    {
      if (!isDev)
        rejected++;
      if (!isDev)
        setcolor(RED);

      std::string errorMsg = "Unknown error";
      if (packet.contains("error") && packet["error"].is_array())
      {
        errorMsg = packet["error"].as_array()[1].as_string().c_str();
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
  boost::system::error_code jsonEc;

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  boost::beast::tcp_stream stream(ioc);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec)
    return fail(ec, "connect");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  BTCStratum::jobCache jobCache;

  // Subscribe to Stratum
  boost::json::object packet = BTCStratum::stratumCall;
  packet["id"] = BTCStratum::subscribe.id;
  packet["method"] = BTCStratum::subscribe.method;
  packet["params"] = boost::json::array({minerName});
  std::string subscription = boost::json::serialize(packet) + "\n";

  size_t trans;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::async_write(stream, boost::asio::buffer(subscription), yield[ec]);
  if (ec)
    return fail(ec, "Stratum subscribe");

  // Handle subscribe response
  boost::asio::streambuf subRes;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::read_until(stream, subRes, "\n");

  std::string subResString = beast::buffers_to_string(subRes.data());
  subRes.consume(trans);

  boost::json::object subRPC = boost::json::parse(subResString.c_str()).as_object();
  handleBTCStratumResponse(subRPC, &jobCache, isDev);

  // Authorize worker
  packet = BTCStratum::stratumCall;
  packet["id"] = BTCStratum::authorize.id;
  packet["method"] = BTCStratum::authorize.method;
  packet["params"] = boost::json::array({wallet + "." + worker});
  if (isDev)
  {
    packet["params"] = boost::json::array({devWallet + "." + worker + "-" + tnnTargetArch});
  }

  std::string authorization = boost::json::serialize(packet) + "\n";

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  boost::asio::async_write(stream, boost::asio::buffer(authorization), yield[ec]);
  if (ec)
    return fail(ec, "Stratum authorize");

  BTCStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count();

  // Persistent packet buffer for handling split packets
  std::string packetBuffer;

  bool submitThread = false;
  bool abort = false;

  // Submit thread
  boost::thread subThread([&]()
                          {
        submitThread = true;
        while(!abort) {
            boost::unique_lock<boost::mutex> lock(mutex);
            bool *B = isDev ? &submittingDev : &submitting;
            cv.wait(lock, [&]{ return (data_ready && (*B)) || abort; });
            if (abort) break;
            
            try {
                boost::json::object *S = &share;
                if (isDev) S = &devShare;

                std::string msg = boost::json::serialize((*S)) + "\n";
                beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(1));
                boost::asio::async_write(stream, boost::asio::buffer(msg), [&](const boost::system::error_code& error, std::size_t bytes_transferred) {
                    if (error) {
                        printf("error on write: %s\n", error.message().c_str());
                        fflush(stdout);
                        abort = true;
                    }
                    if (!isDev) BTCStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                });
                (*B) = false;
                data_ready = false;
            } catch (const std::exception &e) {
                setcolor(RED);
                printf("\nSubmit thread error: %s\n", e.what());
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                break;
            }
            boost::this_thread::yield();
        }
        submitThread = false; });

  // Main message loop with optimal packet handling
  while (!ABORT_MINER)
  {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try
    {
      // Timeout check
      if (BTCStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - BTCStratum::lastReceivedJobTime > BTCStratum::jobTimeout)
      {
        setcolor(RED);
        printf("timeout\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);

        for (;;)
        {
          if (!submitThread)
            break;
          boost::this_thread::yield();
        }
        stream.close();
        return fail(ec, "Stratum session timed out");
      }

      // Read incoming data
      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));

      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec)
      {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        for (;;)
        {
          if (!submitThread)
            break;
          boost::this_thread::yield();
        }
        stream.close();
        return fail(ec, "async_read");
      }

      if (trans > 0)
      {
        std::string newData = beast::buffers_to_string(response.data());
        response.consume(trans);

        // Add new data to persistent buffer
        packetBuffer += newData;

        // Process all complete packets (lines ending with \n)
        size_t pos = 0;
        while ((pos = packetBuffer.find('\n')) != std::string::npos)
        {
          std::string completePacket = packetBuffer.substr(0, pos);
          packetBuffer.erase(0, pos + 1); // Remove processed packet including \n

          if (!completePacket.empty())
          {
            try
            {
              boost::json::object sRPC = boost::json::parse(completePacket.c_str()).as_object();
              if (sRPC.contains("method"))
              {
                int result = handleBTCStratumPacket(sRPC, &jobCache, isDev);
                if (result == 1)
                { // Ping response needed
                  boost::json::object pong({{"id", sRPC["id"].get_uint64()},
                                            {"method", BTCStratum::pong.method}});
                  std::string pongPacket = boost::json::serialize(pong) + "\n";
                  boost::asio::async_write(stream, boost::asio::buffer(pongPacket), yield[ec]);
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
              printf("Parse error: %s\nPacket: %s\n", e.what(), completePacket.c_str());
              fflush(stdout);
              setcolor(BRIGHT_WHITE);
            }
          }
        }

        // Prevent buffer from growing indefinitely
        if (packetBuffer.length() > 65536)
        {
          setcolor(RED);
          printf("Packet buffer overflow, clearing\n");
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
          packetBuffer.clear();
        }
      }
    }
    catch (const std::exception &e)
    {
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      for (;;)
      {
        if (!submitThread)
          break;
        boost::this_thread::yield();
      }
      stream.close();
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return;
    }

    boost::this_thread::yield();
    if (ABORT_MINER)
    {
      bool *connPtr = isDev ? &devConnected : &isConnected;
      bool *submitPtr = isDev ? &submittingDev : &submitting;
      setForDisconnected(connPtr, submitPtr, &abort, &data_ready, &cv);
      ioc.stop();
    }
  }

  cv.notify_all();
  subThread.interrupt();
  subThread.join();
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
  boost::system::error_code jsonEc;

  auto endpoint = resolve_host(wsMutex, ioc, yield, host, port);
  boost::beast::tcp_stream stream(ioc);

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  beast::get_lowest_layer(stream).async_connect(endpoint, yield[ec]);
  if (ec)
    return fail(ec, "connect");

  std::string minerName = "tnn-miner/" + std::string(versionString);
  BTCStratum::jobCache jobCache;

  // Subscribe to Stratum
  boost::json::object packet = BTCStratum::stratumCall;
  packet["id"] = BTCStratum::subscribe.id;
  packet["method"] = BTCStratum::subscribe.method;
  packet["params"] = boost::json::array({minerName});
  std::string subscription = boost::json::serialize(packet) + "\n";

  size_t trans;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::async_write(stream, boost::asio::buffer(subscription), yield[ec]);
  if (ec)
    return fail(ec, "Stratum subscribe");

  // Handle subscribe response
  boost::asio::streambuf subRes;
  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::read_until(stream, subRes, "\n");

  std::string subResString = beast::buffers_to_string(subRes.data());
  subRes.consume(trans);

  boost::json::object subRPC = boost::json::parse(subResString.c_str()).as_object();
  handleBTCStratumResponse(subRPC, &jobCache, isDev);

  // Authorize worker
  packet = BTCStratum::stratumCall;
  packet["id"] = BTCStratum::authorize.id;
  packet["method"] = BTCStratum::authorize.method;
  packet["params"] = boost::json::array({wallet + "." + worker});
  if (isDev)
  {
    packet["params"] = boost::json::array({devWallet + "." + worker + "-" + tnnTargetArch});
  }

  std::string authorization = boost::json::serialize(packet) + "\n";

  beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
  trans = boost::asio::async_write(stream, boost::asio::buffer(authorization), yield[ec]);
  if (ec)
    return fail(ec, "Stratum authorize");

  BTCStratum::lastReceivedJobTime = std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count();

  // Persistent packet buffer for handling split packets
  std::string packetBuffer;

  bool submitThread = false;
  bool abort = false;

  // Submit thread
  boost::thread subThread([&]()
                          {
        submitThread = true;
        while(!abort) {
            boost::unique_lock<boost::mutex> lock(mutex);
            bool *B = isDev ? &submittingDev : &submitting;
            cv.wait(lock, [&]{ return (data_ready && (*B)) || abort; });
            if (abort) break;
            
            try {
                boost::json::object *S = &share;
                if (isDev) S = &devShare;

                std::string msg = boost::json::serialize((*S)) + "\n";
                beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(1));
                boost::asio::async_write(stream, boost::asio::buffer(msg), [&](const boost::system::error_code& error, std::size_t bytes_transferred) {
                    if (error) {
                        printf("error on write: %s\n", error.message().c_str());
                        fflush(stdout);
                        abort = true;
                    }
                    if (!isDev) BTCStratum::lastShareSubmissionTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                });
                (*B) = false;
                data_ready = false;
            } catch (const std::exception &e) {
                setcolor(RED);
                printf("\nSubmit thread error: %s\n", e.what());
                fflush(stdout);
                setcolor(BRIGHT_WHITE);
                break;
            }
            boost::this_thread::yield();
        }
        submitThread = false; });

  // Main message loop with optimal packet handling
  while (!ABORT_MINER)
  {
    bool *C = isDev ? &devConnected : &isConnected;
    bool *B = isDev ? &submittingDev : &submitting;

    try
    {
      // Timeout check
      if (BTCStratum::lastReceivedJobTime > 0 &&
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - BTCStratum::lastReceivedJobTime > BTCStratum::jobTimeout)
      {
        setcolor(RED);
        printf("timeout\n");
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        setForDisconnected(C, B, &abort, &data_ready, &cv);

        for (;;)
        {
          if (!submitThread)
            break;
          boost::this_thread::yield();
        }
        stream.close();
        return fail(ec, "Stratum session timed out");
      }

      // Read incoming data
      boost::asio::streambuf response;
      beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(60));

      trans = boost::asio::async_read_until(stream, response, "\n", yield[ec]);
      if (ec)
      {
        setForDisconnected(C, B, &abort, &data_ready, &cv);
        for (;;)
        {
          if (!submitThread)
            break;
          boost::this_thread::yield();
        }
        stream.close();
        return fail(ec, "async_read");
      }

      if (trans > 0)
      {
        std::string newData = beast::buffers_to_string(response.data());
        response.consume(trans);

        // Add new data to persistent buffer
        packetBuffer += newData;

        // Process all complete packets (lines ending with \n)
        size_t pos = 0;
        while ((pos = packetBuffer.find('\n')) != std::string::npos)
        {
          std::string completePacket = packetBuffer.substr(0, pos);
          packetBuffer.erase(0, pos + 1); // Remove processed packet including \n

          if (!completePacket.empty())
          {
            try
            {
              boost::json::object sRPC = boost::json::parse(completePacket.c_str()).as_object();
              if (sRPC.contains("method"))
              {
                int result = handleBTCStratumPacket(sRPC, &jobCache, isDev);
                if (result == 1)
                { // Ping response needed
                  boost::json::object pong({{"id", sRPC["id"].get_uint64()},
                                            {"method", BTCStratum::pong.method}});
                  std::string pongPacket = boost::json::serialize(pong) + "\n";
                  boost::asio::async_write(stream, boost::asio::buffer(pongPacket), yield[ec]);
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
              printf("Parse error: %s\nPacket: %s\n", e.what(), completePacket.c_str());
              fflush(stdout);
              setcolor(BRIGHT_WHITE);
            }
          }
        }

        // Prevent buffer from growing indefinitely
        if (packetBuffer.length() > 65536)
        {
          setcolor(RED);
          printf("Packet buffer overflow, clearing\n");
          fflush(stdout);
          setcolor(BRIGHT_WHITE);
          packetBuffer.clear();
        }
      }
    }
    catch (const std::exception &e)
    {
      setForDisconnected(C, B, &abort, &data_ready, &cv);
      for (;;)
      {
        if (!submitThread)
          break;
        boost::this_thread::yield();
      }
      stream.close();
      setcolor(RED);
      std::cerr << e.what() << std::endl;
      fflush(stdout);
      setcolor(BRIGHT_WHITE);
      return;
    }

    boost::this_thread::yield();
    if (ABORT_MINER)
    {
      bool *connPtr = isDev ? &devConnected : &isConnected;
      bool *submitPtr = isDev ? &submittingDev : &submitting;
      setForDisconnected(connPtr, submitPtr, &abort, &data_ready, &cv);
      ioc.stop();
    }
  }

  cv.notify_all();
  subThread.interrupt();
  subThread.join();
}