#pragma once

#ifndef tnncommon_hpp
#define tnncommon_hpp

#include <stdint.h>
#include <vector>
#include <string>
#include <random>
#include <map>

#include <boost/program_options.hpp>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/spawn.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ip/host_name.hpp>

#include <boost/json.hpp>

#include <boost/atomic.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/atomic.hpp>
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
extern std::string stratumPassword;
extern bool useLookupMine;

extern bool gpuMine;
extern bool g_powerMonAvail;
extern bool g_ocAvail;
extern std::string devWallet;

extern std::map<int, int> threadToPhysicalCore;
extern std::mutex threadMapMutex;

typedef struct {
  int coinId;
  std::string devHost;
  std::string devPort;
  std::string devWallet;
  std::string devTestWallet;
} TnnDevMinerInfo;

const TnnDevMinerInfo devInfo[COIN_COUNT] = {
  {COIN_DERO,     "dero-node-sk.mysrv.cloud",    "10300",
#if defined(__x86_64__)
  "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7",
#else
  "dero1qyxrwhew9vkwr9m8sz2ndvqc9zpsjey680le8htzxqevyxn6kwfxqqgemj2x6",
#endif
  "dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7"
  },
  {COIN_XELIS,    "stratum+ssl://usw.vipor.net", "5177",    "xel:xz9574c80c4xegnvurazpmxhw5dlg2n0g9qm60uwgt75uqyx3pcsqzzra9m" },
  {COIN_SPECTRE,  "stratum+tcp://spectre.cedric-crispin.com",  "4364",
#if defined(__x86_64__)
  "spectre:qr5l7q4s6mrfs9r7n0l090nhxrjdkxwacyxgk8lt2wt57ka6xr0ucvr0cmgnf",
#else
  "spectre:qqty6rrlsxwzcwdx7ge60256cw7r2adu7c8nqtsqxjmkt2c83h3kss3uqeay0",
#endif
  "spectredev:qqhh8ul66g7t6aj5ggzl473cpan25tv6yjm0cl4hffprgtqfvmyaq8q28m4z8"
  },
  {COIN_RX0,      "stratum+ssl://monerohash.com",               "9999",     "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  {COIN_XMR,      "stratum+ssl://monerohash.com",               "9999",     "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  {COIN_SAL,      "stratum+ssl://monerohash.com",               "9999",     "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  {COIN_ZEPH,     "stratum+ssl://monerohash.com",               "9999",     "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  {COIN_VERUS,    "",                                           "",         "", ""},
  {COIN_AIX,      "na.mining4people.com",                       "3394",     "astrix:qz2mzpga6qv9uvnpueau7gs29vgu3ynj80xmd2dmja2kelzh6cssymsk3shjx", "astrix:qz2mzpga6qv9uvnpueau7gs29vgu3ynj80xmd2dmja2kelzh6cssymsk3shjx"},
  {COIN_NXL,      "178.16.131.178",                             "5555",     "nexellia:qqq3lwqrnh6alujup2me8gkedvp4w4d8zkjxdzmlrzpju2npdvvmctwl649xr", "nexellia:qqq3lwqrnh6alujup2me8gkedvp4w4d8zkjxdzmlrzpju2npdvvmctwl649xr"},
  {COIN_HTN,      "na.mining4people.com",                       "3390",     "hoosat:qr03chtq640d6p9r5p95kw4t4txcrt9x2cyfjf5w6wpfqwugs35yy472wq6hu", "hoosat:qr03chtq640d6p9r5p95kw4t4txcrt9x2cyfjf5w6wpfqwugs35yy472wq6hu"},
  {COIN_WALA,     "stratum+tcp://us-west.sumohash.com",         "4022",     "waglayla:qr6h2tqwx8ad57nkte9kvcd9cqyjfgk30gznnza9jte7qzfa6gu0xy5n3evj5", "waglayla:qr6h2tqwx8ad57nkte9kvcd9cqyjfgk30gznnza9jte7qzfa6gu0xy5n3evj5"},
  {COIN_SHAI,     "shaicoin.viporlab.net",                      "3333",     "sh1qvee0lejv22n7s43q3asw4uzap8d9t32k95cznj", "sh1qvee0lejv22n7s43q3asw4uzap8d9t32k95cznj"},
  {COIN_YESPOWER, "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_ADVC,     "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_TARI,     "stratum+ssl://monerohash.com",               "9999",     "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe", "49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe"},
  {COIN_RIN,      "stratum+ssl://stratum-eu.rplant.xyz",        "17148",    "rin1qzcg5vpdypje7f9ql8v4ttwdmmcxr8j64pxfwrv", "rin1qzcg5vpdypje7f9ql8v4ttwdmmcxr8j64pxfwrv"},
  {COIN_TIDE,     "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_YPR16,    "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_YCR16,    "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_YCR8,     "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_MGPC,     "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_URX,      "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_LTNCG,    "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_YSC,      "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_EQPAY,    "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
  {COIN_YCR32,    "stratum+ssl://stratum-eu.rplant.xyz",        "17149",    "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj", "AaM7AxuyWyPKRMGC8wZxub2rXYUiinZDwj"},
};

typedef struct {
    int coinId;
    int miningAlgo;
    std::string coinSymbol;
    std::string coinPrettyName;
} Coin;

const Coin unknownCoin = {COIN_UNKNOWN, ALGO_UNSUPPORTED, "unknown", "unknown"};
const Coin coins[COIN_COUNT] = {
  {COIN_DERO,     ALGO_ASTROBWTV3,  "DERO",     "Dero"},
  {COIN_XELIS,    ALGO_XELISV3,     "XEL",      "Xelis"},
  {COIN_SPECTRE,  ALGO_SPECTRE_X,   "SPR",      "Spectre"},
  {COIN_RX0,      ALGO_RX0,         "RX0",      "RandomX"},
  {COIN_XMR,      ALGO_RX0,         "XMR",      "Monero"},
  {COIN_SAL,      ALGO_RX0,         "SAL",      "Salvium"},
  {COIN_ZEPH,     ALGO_RX0,         "ZEPH",     "Zephyr"},
  {COIN_VERUS,    ALGO_VERUS,       "VRSC",     "Verus"},
  {COIN_AIX,      ALGO_ASTRIX_HASH, "AIX",      "Astrix"},
  {COIN_NXL,      ALGO_NXL_HASH,    "NXL",      "Nexellia"},
  {COIN_HTN,      ALGO_HOOHASH,     "HTN",      "Hoosat"},
  {COIN_WALA,     ALGO_WALA_HASH,   "WALA",     "Waglayla"},
  {COIN_SHAI,     ALGO_SHAI_HIVE,   "SHAI",     "Shai"},
  {COIN_YESPOWER, ALGO_YESPOWER,    "YESPOWER", "Yespower (Generic)"},
  {COIN_ADVC,     ALGO_YESPOWER,    "ADVC",     "AdventureCoin"},
  {COIN_TARI,     ALGO_RX0,         "XTM",      "Tari"},
  {COIN_RIN,      ALGO_RINHASH,     "RIN",      "RinCoin"},
  {COIN_TIDE,     ALGO_YESPOWER,    "TDC",      "Tidecoin (YespowerTIDE)"},
  {COIN_YPR16,    ALGO_YESPOWER,    "YTN",      "Yenten (YespowerR16)"},
  {COIN_YCR16,    ALGO_YESPOWER,    "GOLD",     "Goldcash (YescryptR16)"},
  {COIN_YCR8,     ALGO_YESPOWER,    "MTBC",     "MTBC (YescryptR8)"},
  {COIN_MGPC,     ALGO_YESPOWER,    "MGPC",     "Magpiecoin (YespowerMGPC)"},
  {COIN_URX,      ALGO_YESPOWER,    "URX",      "UraniumX (YespowerURX)"},
  {COIN_LTNCG,    ALGO_YESPOWER,    "CRNC",     "Crane (YespowerLTNCG)"},
  {COIN_YSC,      ALGO_YESPOWER,    "YSC",      "Yescrypt"},
  {COIN_EQPAY,    ALGO_YESPOWER,    "EQPAY",    "EqpayCoin (YespowerEQPAY)"},
  {COIN_YCR32,    ALGO_YESPOWER,    "LPEPE",    "LuckyPepe (YescryptR32)"},
};

// ============================================================================
// Algorithm Versioning Support
// ============================================================================

struct AlgoVersion {
  int algo;
  std::string displayName;  // For pretty printing, e.g., "v3"
};

struct VersionedAlgo {
  std::string coinSymbol;
  int defaultAlgo;
  std::map<std::string, AlgoVersion> versions;  // Key: "V1", "1", etc.
};

// Define versioned algorithms - add new entries here as needed
inline const std::vector<VersionedAlgo>& getVersionedAlgos() {
  static const std::vector<VersionedAlgo> versionedAlgos = {
    {
      "XEL",
      ALGO_XELISV3,  // Default when --xel is used without version
      {
        // {"V1", {ALGO_XELISV1, "v1"}},
        {"V2", {ALGO_XELISV2, "v2"}},
        {"V3", {ALGO_XELISV3, "v3"}},
        // {"1",  {ALGO_XELISV1, "v1"}},
        {"2",  {ALGO_XELISV2, "v2"}},
        {"3",  {ALGO_XELISV3, "v3"}},
      }
    },
  };
  return versionedAlgos;
}

struct CoinParseResult {
  std::string baseSymbol;
  int algoOverride;           // -1 if no version specified
  std::string versionDisplay; // e.g., "v3" for display purposes
  bool isVersioned;           // True if this was a versioned coin match
};

// Parse coin symbol with optional version suffix
// Supports: XEL, XEL-V3, XEL=V3, XELV3, XEL-3, XEL=3, XEL3
inline CoinParseResult parseCoinWithVersion(const std::string& input) {
  CoinParseResult result = {"", -1, "", false};
  
  std::string upperInput = input;
  std::transform(upperInput.begin(), upperInput.end(), upperInput.begin(), ::toupper);
  
  const auto& versionedAlgos = getVersionedAlgos();
  
  for (const auto& va : versionedAlgos) {
    // Exact match (base symbol only) - use default
    if (upperInput == va.coinSymbol) {
      result.baseSymbol = va.coinSymbol;
      result.algoOverride = va.defaultAlgo;
      result.isVersioned = true;
      
      // Find display name for default
      for (const auto& [key, ver] : va.versions) {
        if (ver.algo == va.defaultAlgo && key.length() == 2) {  // Prefer "V3" over "3"
          result.versionDisplay = ver.displayName;
          break;
        }
      }
      return result;
    }
    
    // Try different separator patterns: -, =, or none
    const std::vector<std::string> separators = {"-", "=", ""};
    
    for (const auto& sep : separators) {
      std::string prefix = va.coinSymbol + sep;
      
      if (upperInput.length() > prefix.length() &&
          upperInput.substr(0, prefix.length()) == prefix) {
        
        std::string versionPart = upperInput.substr(prefix.length());
        
        auto it = va.versions.find(versionPart);
        if (it != va.versions.end()) {
          result.baseSymbol = va.coinSymbol;
          result.algoOverride = it->second.algo;
          result.versionDisplay = it->second.displayName;
          result.isVersioned = true;
          return result;
        }
      }
    }
  }
  
  // Not a versioned coin - return base symbol for regular lookup
  result.baseSymbol = upperInput;
  result.isVersioned = false;
  return result;
}

// Find coin by symbol (case-insensitive)
inline const Coin* findCoinBySymbol(const std::string& symbol) {
  std::string upperSymbol = symbol;
  std::transform(upperSymbol.begin(), upperSymbol.end(), upperSymbol.begin(), ::toupper);
  
  for (int i = 0; i < COIN_COUNT; i++) {
    std::string coinSymbol = coins[i].coinSymbol;
    std::transform(coinSymbol.begin(), coinSymbol.end(), coinSymbol.begin(), ::toupper);
    if (coinSymbol == upperSymbol) {
      return &coins[i];
    }
  }
  return nullptr;
}

// Get display string for algorithm versions available for a coin
inline std::string getVersionsHelpString(const std::string& symbol) {
  std::string upperSymbol = symbol;
  std::transform(upperSymbol.begin(), upperSymbol.end(), upperSymbol.begin(), ::toupper);
  
  const auto& versionedAlgos = getVersionedAlgos();
  for (const auto& va : versionedAlgos) {
    if (va.coinSymbol == upperSymbol) {
      std::string help = "Available versions: ";
      std::set<std::string> displayed;
      for (const auto& [key, ver] : va.versions) {
        if (key.length() == 2 && displayed.find(ver.displayName) == displayed.end()) {
          if (displayed.size() > 0) help += ", ";
          help += ver.displayName;
          if (ver.algo == va.defaultAlgo) help += " (default)";
          displayed.insert(ver.displayName);
        }
      }
      return help;
    }
  }
  return "";
}

// ============================================================================
// MiningProfile class
// ============================================================================

class MiningProfile {
  public:
    MiningProfile() {
      coin = unknownCoin;
    };
    ~MiningProfile() {}
    
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
        try {
          const int i{std::stoi(tokens[1])};
          this->port = tokens[1];
        }
        catch (...) {
          printf("catch: protocol:host\n");
          this->transportLayer = tokens[0];
          this->host = tokens[1];
        }
      } else if(tokens.size() == 3) {
        this->transportLayer = tokens[0];
        this->host = tokens[1];
        this->port = tokens[2];
      }
      boost::replace_all(this->host, "/", "");
      if (this->transportLayer.size() > 0) {
        if (this->transportLayer.find("stratum") != std::string::npos) this->useStratum = true;
        if (this->transportLayer.find("xatum") != std::string::npos) this->protocol = PROTO_XELIS_XATUM;
      }
      this->setProtocol();
      printf("%s %s %s\n", this->transportLayer.c_str(), this->host.c_str(), this->port.c_str());
    }

    void setProtocol() {
      if (this->useStratum) {
        switch (this->coin.miningAlgo) {
          // case ALGO_XELISV1:
          case ALGO_XELISV2:
          case ALGO_XELISV3:
            this->protocol = PROTO_XELIS_STRATUM;
            break;
          case ALGO_SPECTRE_X:
            this->protocol = PROTO_SPECTRE_STRATUM;
            break;
          case ALGO_RX0:
            this->protocol = PROTO_RX0_STRATUM;
            break;
          case ALGO_YESPOWER:
            this->protocol = PROTO_BTC_STRATUM;
            break;
          case ALGO_RINHASH:
            this->protocol = PROTO_BTC_STRATUM;
            break;
        }
      } else {
        switch (this->coin.miningAlgo) {
          // case ALGO_XELISV1:
          case ALGO_XELISV2:
          case ALGO_XELISV3:
            this->protocol = PROTO_XELIS_SOLO;
            break;            
          case ALGO_RX0:
            this->protocol = PROTO_RX0_SOLO;
            break;
        }
      }
    }
    
    // Set coin with optional version override
    void setCoin(const Coin& c, int algoOverride = -1) {
      this->coin = c;
      if (algoOverride != -1) {
        this->coin.miningAlgo = algoOverride;
      }
    }
};

extern MiningProfile miningProfile;
extern MiningProfile devMiningProfile;

extern Num oneLsh256;      
extern Num maxU256;

extern boost::multiprecision::uint256_t bigDiff;
extern boost::multiprecision::uint256_t bigDiff_dev;

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
extern std::vector<std::atomic<uint64_t>> HIP_kIndex;
extern std::vector<std::atomic<uint64_t>> HIP_kIndex_dev;
extern std::vector<std::atomic<uint64_t>> HIP_counters;
extern std::vector<std::atomic<uint64_t>> HIP_counters;
extern std::vector<std::vector<int64_t>> HIP_rates5min;
extern std::vector<std::vector<int64_t>> HIP_rates1min;
extern std::vector<std::vector<int64_t>> HIP_rates30sec;
// Per-device share counters.  [0..31] = GPU index, [32] = CPU.
// Updated by net response handlers via recordDeviceShare().
constexpr int DEVICE_SHARE_CPU = 32;
extern std::atomic<int> deviceAccepted[33];
extern std::atomic<int> deviceRejected[33];

inline void recordDeviceShare(int device, bool accepted) {
    int idx = (device < 0) ? DEVICE_SHARE_CPU : device;
    if (idx > DEVICE_SHARE_CPU) idx = DEVICE_SHARE_CPU;
    if (accepted)
        deviceAccepted[idx].fetch_add(1, std::memory_order_relaxed);
    else
        deviceRejected[idx].fetch_add(1, std::memory_order_relaxed);
}

extern std::vector<int64_t> rate5min;
extern std::vector<int64_t> rate1min;
extern std::vector<int64_t> rate30sec;

extern std::atomic<int64_t> counter;
extern std::atomic<int64_t> benchCounter;

extern bool isConnected;
extern bool devConnected;

extern bool beQuiet;

extern boost::asio::io_context my_context;
extern boost::asio::steady_timer update_timer;
extern boost::asio::steady_timer mine_duration_timer;
extern std::chrono::time_point<std::chrono::steady_clock> g_start_time;
extern int mine_time;

inline std::string cpp_int_toHex(boost::multiprecision::cpp_int in) {
  std::ostringstream oss;
  oss << std::hex << in;
  return oss.str();
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
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(low, high);
  return distr(gen);
}

#endif