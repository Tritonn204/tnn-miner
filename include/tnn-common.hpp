#pragma once

#ifndef tnncommon_hpp
#define tnncommon_hpp

#include <stdint.h>
#include <vector>
#include <string>
#include <random>

#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <boost/json.hpp>

#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <boost/tokenizer.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <num.h>

#include "algo_definitions.h"

#define XSTR(x) STR(x)
#define STR(x) #x

extern const char *tnnTargetArch;

#define CMP_LT_U256(X, Y) (X[3] != Y[3] ? X[3] < Y[3] : X[2] != Y[2] ? X[2] < Y[2] \
                                                                            : X[1] != Y[1]   ? X[1] < Y[1] \
                                                                                                           : X[0] < Y[0])

extern bool ABORT_MINER;

#define CHECK_CLOSE if (ABORT_MINER) return;
#define CHECK_CLOSE_RET(s) if (ABORT_MINER) return s;

extern double latest_hashrate;

static const char* devWorkerName = "tnn-dev";                                      

extern bool devTurn;
extern bool lockThreads;
extern int threads;

extern std::string workerName;
extern std::string workerNameFromWallet;
extern bool useLookupMine;

extern bool gpuMine;
extern std::string devWallet;

extern std::map<int, int> threadToPhysicalCore;  // tid -> physical core ID
extern std::mutex threadMapMutex;

typedef struct {
  int coinId;
  std::string devHost;
  std::string devPort;
  std::string devWallet;
  std::string devTestWallet;
} TnnDevMinerInfo;

const TnnDevMinerInfo devInfo[COIN_COUNT] = {
  // Coin         DevHost                        DevPort    DevWallet
  {COIN_DERO,     "dero-node-sk.mysrv.cloud",    "10300",
#if defined(__x86_64__)
  "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7",  // Tritonn
#else
  "dero1qyxrwhew9vkwr9m8sz2ndvqc9zpsjey680le8htzxqevyxn6kwfxqqgemj2x6",  // Dirker
#endif
  "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7"
  },
  {COIN_XELIS,    "stratum+ssl://usw.vipor.net", "5177",    "xel:xz9574c80c4xegnvurazpmxhw5dlg2n0g9qm60uwgt75uqyx3pcsqzzra9m"},
  //{COIN_SPECTRE,  "51.81.211.69",  "5555",
  //{COIN_SPECTRE,  "localhost",  "5555",
  {COIN_SPECTRE,  "stratum+tcp://spectre.cedric-crispin.com",  "4364",
  //{COIN_SPECTRE,  "spr.tw-pool.com", "14001",
  //{COIN_SPECTRE,  "spr.mining.st-ips.de", "4364",
  //{COIN_SPECTRE,  "eu.spectre-network.nevermine.io",  "55555",
  //{COIN_SPECTRE,  "pool.tazmining.ch", "7750",
#if defined(__x86_64__)
  "spectre:qr5l7q4s6mrfs9r7n0l090nhxrjdkxwacyxgk8lt2wt57ka6xr0ucvr0cmgnf",  // Tritonn
#else
  "spectre:qqty6rrlsxwzcwdx7ge60256cw7r2adu7c8nqtsqxjmkt2c83h3kss3uqeay0",  // Dirker
#endif
  "spectredev:qqhh8ul66g7t6aj5ggzl473cpan25tv6yjm0cl4hffprgtqfvmyaq8q28m4z8"
  },
  {COIN_RX0,      "stratum+tcp://monerohash.com",              "2222",    "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  {COIN_XMR,      "stratum+tcp://monerohash.com",              "2222",    "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  {COIN_SAL,      "stratum+tcp://monerohash.com",              "2222",    "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  {COIN_ZEPH,     "stratum+tcp://monerohash.com",              "2222",    "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  {COIN_VERUS,    "",                            "",        "", ""},
  {COIN_AIX,      "na.mining4people.com",        "3394",    "astrix:qz2mzpga6qv9uvnpueau7gs29vgu3ynj80xmd2dmja2kelzh6cssymsk3shjx", "astrix:qz2mzpga6qv9uvnpueau7gs29vgu3ynj80xmd2dmja2kelzh6cssymsk3shjx"},
  {COIN_NXL,      "178.16.131.178",              "5555",    "nexellia:qqq3lwqrnh6alujup2me8gkedvp4w4d8zkjxdzmlrzpju2npdvvmctwl649xr", "nexellia:qqq3lwqrnh6alujup2me8gkedvp4w4d8zkjxdzmlrzpju2npdvvmctwl649xr"},
  {COIN_HTN,      "na.mining4people.com",        "3390",    "hoosat:qr03chtq640d6p9r5p95kw4t4txcrt9x2cyfjf5w6wpfqwugs35yy472wq6hu", "hoosat:qr03chtq640d6p9r5p95kw4t4txcrt9x2cyfjf5w6wpfqwugs35yy472wq6hu"},
  {COIN_WALA,     "stratum+tcp://us-west.sumohash.com",     "4022",    "waglayla:qr6h2tqwx8ad57nkte9kvcd9cqyjfgk30gznnza9jte7qzfa6gu0xy5n3evj5", "waglayla:qr6h2tqwx8ad57nkte9kvcd9cqyjfgk30gznnza9jte7qzfa6gu0xy5n3evj5"},
  {COIN_SHAI,     "shaicoin.viporlab.net",       "3333",    "sh1qvee0lejv22n7s43q3asw4uzap8d9t32k95cznj", "sh1qvee0lejv22n7s43q3asw4uzap8d9t32k95cznj"},
  {COIN_YESPOWER, "stratum+ssl://stratum-eu.rplant.xyz", "17149", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"}, // Default to ADVC
  {COIN_ADVC,     "stratum+ssl://stratum-eu.rplant.xyz", "17149", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_TARI,     "stratum+tcp://monerohash.com",              "2222",    "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  //{COIN_COUNT 16  
};

typedef struct {
    int coinId;
    int miningAlgo;
    std::string coinSymbol;
    std::string coinPrettyName;
} Coin;

const Coin unknownCoin = {COIN_UNKNOWN, ALGO_UNSUPPORTED, "unknown", "unknown"};
const Coin coins[COIN_COUNT] = {
  // Coin         Aglo              Symbol   Name
  {COIN_DERO,     ALGO_ASTROBWTV3,  "DERO",  "Dero"},
  {COIN_XELIS,    ALGO_XELISV2,     "XEL",   "Xelis"},
  {COIN_SPECTRE,  ALGO_SPECTRE_X,   "SPR",   "Spectre"},
  {COIN_RX0,      ALGO_RX0,         "RX0",   "RandomX"},
  {COIN_XMR,      ALGO_RX0,         "XMR",   "Monero"},
  {COIN_SAL,      ALGO_RX0,         "SAL",   "Salvium"},
  {COIN_ZEPH,     ALGO_RX0,         "ZEPH",  "Zephyr"},
  {COIN_VERUS,    ALGO_VERUS,       "VRSC",  "Verus"},
  {COIN_AIX,      ALGO_ASTRIX_HASH, "AIX",   "Astrix"},
  {COIN_NXL,      ALGO_NXL_HASH,    "NXL",   "Nexellia"},
  {COIN_HTN,      ALGO_HOOHASH,     "HTN",   "Hoosat"},
  {COIN_WALA,     ALGO_WALA_HASH,   "WALA",  "Waglayla"},
  {COIN_SHAI,     ALGO_SHAI_HIVE,   "SHAI",  "Shai"},
  {COIN_YESPOWER, ALGO_YESPOWER, "YESPOWER", "Yespower (Generic)"},
  {COIN_ADVC, ALGO_YESPOWER, "ADVC", "AdventureCoin"},
  {COIN_TARI,     ALGO_RX0,         "XTM",   "Tari"},
  //{COIN_COUNT 16 
};

class MiningProfile {
  public:
    MiningProfile() {
      coin = unknownCoin;
    };
    ~MiningProfile() {
      //printf("Goodbye, miner\n");
      //fflush(stdout);
    }
    Coin coin;
    bool isDev;
    int protocol;
    std::string host;
    std::string port;
    std::string wallet;
    std::string workerName;
    std::string transportLayer;
    bool useStratum = false;
    bool doShutdown;

    void setDev(bool testnet) {
      //this->transportLayer = "";
      this->isDev = true;
      this->setPoolAddress(devInfo[this->coin.coinId].devHost + ":" + devInfo[this->coin.coinId].devPort);
      this->wallet = testnet ? devInfo[this->coin.coinId].devTestWallet : devInfo[this->coin.coinId].devWallet;
      devWallet = this->wallet;
    }
    void setPoolAddress(std::string hst) {
      this->host = hst;
      boost::char_separator<char> sep(":");
      boost::tokenizer<boost::char_separator<char>> tok(hst, sep);
      std::vector<std::string> tokens;
      std::copy(tok.begin(), tok.end(), std::back_inserter<std::vector<std::string> >(tokens));
      if(tokens.size() == 2) {
        this->host = tokens[0];
        try
        {
          // given host:port
          const int i{std::stoi(tokens[1])};
          this->port = tokens[1];
        }
        catch (...)
        {
          printf("catch: protocol:host\n");
          // protocol:host
          this->transportLayer = tokens[0];
          this->host = tokens[1];
        }
      } else if(tokens.size() == 3) {
        this->transportLayer = tokens[0];  // wss, stratum+tcp, stratum+ssl, et al
        this->host = tokens[1];
        this->port = tokens[2];
      }
      boost::replace_all(this->host, "/", "");
      if (this->transportLayer.size() > 0) {
        if (this->transportLayer.find("stratum") != std::string::npos) this->useStratum = true;
        if (this->transportLayer.find("xatum") != std::string::npos) this->protocol = PROTO_XELIS_XATUM;
      }
      printf("%s %s %s\n", this->transportLayer.c_str(), this->host.c_str(), this->port.c_str());
    }
};

extern MiningProfile miningProfile;
extern MiningProfile devMiningProfile;

extern Num oneLsh256;      
extern Num maxU256;

extern boost::multiprecision::uint256_t bigDiff;
extern boost::multiprecision::uint256_t bigDiff_dev;

// Dev fee config
// Dev fee is a % of hashrate
extern int batchSize;
extern double minFee;
extern double devFee;

extern int jobCounter;
extern int reportCounter;
extern int reportInterval;

extern int blockCounter;
extern int miniBlockCounter;
extern int rejected;
extern int accepted;
//static int firstRejected;

//extern uint64_t hashrate;
extern int64_t ourHeight;
extern int64_t devHeight;

extern int nonceLen;
extern int nonceLenDev;

extern int64_t difficulty;
extern int64_t difficultyDev;

extern uint64_t nonce0;
extern uint64_t nonce0_dev;

extern double doubleDiff;
extern double doubleDiffDev;

extern int HIP_deviceCount;
extern std::string HIP_names[32];
extern std::string HIP_pcieID[32];
extern uint64_t HIP_kIndex[32];
extern uint64_t HIP_kIndex_dev[32];
extern std::vector<std::atomic<uint64_t>> HIP_counters;
extern std::vector<std::vector<int64_t>> HIP_rates5min;
extern std::vector<std::vector<int64_t>> HIP_rates1min;
extern std::vector<std::vector<int64_t>> HIP_rates30sec;

extern std::vector<int64_t> rate5min;
extern std::vector<int64_t> rate1min;
extern std::vector<int64_t> rate30sec;

extern std::atomic<int64_t> counter;
extern std::atomic<int64_t> benchCounter;

extern bool isConnected;
extern bool devConnected;

extern bool beQuiet;

extern boost::asio::io_context my_context;
// Construct a timer without setting an expiry time.
extern boost::asio::steady_timer update_timer;
extern std::chrono::time_point<std::chrono::steady_clock> g_start_time;
extern int mine_time;
extern boost::asio::steady_timer mine_duration_timer;

inline std::string cpp_int_toHex(boost::multiprecision::cpp_int in) {
  std::ostringstream oss;
  oss << std::hex << in;
  std::string hex_string = oss.str();

  return hex_string;
}

inline void cpp_int_to_byte_array(const boost::multiprecision::uint256_t &num, uint8_t *out) {  
  for (size_t i = 0; i < 32; ++i) {
    out[i] = static_cast<uint8_t>(num >> (i * 8) & 0xFF);
  }
}

inline void cpp_int_to_be_byte_array(const boost::multiprecision::uint256_t &num, uint8_t *out) {  
  for (size_t i = 0; i < 32; ++i) {
    out[i] = static_cast<uint8_t>(num >> ((32 - i - 1) * 8) & 0xFF);
  }
}

inline int randomSleepTimeMs(int low=9000, int high=11000) {
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> distr(low, high); // define the range
  return distr(gen);
}

#endif