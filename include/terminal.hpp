#pragma once

#include <string>
#include <algorithm>
#include <boost/program_options.hpp>

#if defined(__linux__)
  #include <sys/ioctl.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

namespace po = boost::program_options;  // from <boost/program_options.hpp>

// macro tricks so we can use a string to set TNN_VERSION
#define XSTR(x) STR(x)
#define STR(x) #x

#ifdef _WIN32
#define RUN_EXTENSION ".exe"
#define SCRIPT_EXTENSION ".bat"
#else
#define RUN_EXTENSION ""
#define SCRIPT_EXTENSION ".sh"
#endif

static const char *versionString = XSTR(TNN_VERSION);
static const char *consoleLine = " TNN-MINER ";
static const char *targetArch = XSTR(CPU_ARCHTARGET);
static const char *TNN = R"(
  
                                                            YB&&@5
                                              :GBB7         &@@@@P
                          .:^~7J5J.           !@@@@@Y.      #@@@@P
            ..:~!?Y5GB#&@@@@@@@@@@@5.         ~@@@@@@@B^    #@@@@G
  ~7J5GB#&&@@@@@@@@@@@@@@@@@@@@@@@@@@P.       ~@@@@@@@@@&7  #@@@@G
  @@@@@@@@@@@@@@@@@@@&#BG5J7!^5@@@@@@@@G:     ~@@@@@&@@@@@@Y&@@@@B
  @@@@@&#BGPY?~G@@@@&         7@@@@@@@@@@B^   ^@@@@@:.5@@@@@@@@@@B
  ::.          7@@@@&         7@@@@@&@@@@@@#~ :@@@@@:   ?&@@@@@@@B
               7@@@@@         !@@@@& ^B@@@@@@#5@@@@@^     ~#@@@@@#
               7@@@@@         !@@@@@   :B@@@@@@@@@@@^       :G@@@#
               !@@@@@         ~@@@@@.    .P@@@@@@@@@~         .Y@&
               !@@@@@         ~@@@@@.      .5@@@@@@@~            ^
               ~@@@@@.        ^@@@@@.        .J@@@@@~             
               ~@@@@@.        ^@@@@@.           ?&@@!             
               ~@@@@@.        ^@@@@@.             !&!             
               ^@@@@@.        :@@@@@.                             
               ^@@@@@.        :@@@@@:                             
               ^@@@@@.         ^G@@@:      ██ ██ █ █   █ ████ ████                
               :@@@@@:           .J&:      █████ █ ██  █ █    █  █           
               :@@@@@:                     █ █ █ █ ███ █ ███  ████                 
               :@@@&Y                      █ █ █ █ █ ███ █    █ █        
               .&Y:                        █   █ █ █  ██ ████ █ ██    

)";  

static const char *DERO = R"(
                                                                  
                              @                                 
                         @@       @@                            
                     @@               @@                        
                 @                         @                    
             @                                 @@               
        @@                    @                    @@           
    @                    @@       @@                    @       
    @                @@       .       @@                @       
    @            @@       ..     ..       @@            @       
    @          @      .               .      @          @       
    @          @   .       @@@@@@@       .   @          @       
    @          @   .    @@@@@@@@@@@@@    .   @          @       
    @          @   .    @@@@@@@@@@@@@    .   @          @       
    @          @   .    @@@@@@@@@@@@@    .   @          @       
    @          @   .     @@@@@@@@@@@     .   @          @       
    @          @    ..     @@@@@@@     ..    @          @       
    @          @@        .@@@@@@@@@.        @@          @       
    @              @@    @@@@@@@@@@@    @@              @       
    @                  @@@@@@@@@@@@@@@                  @       
       @                    @@@@@                    @@         
           @@                                    @              
               @@                           @@                  
                   @@                   @@                      
                        @@         @@                           
                              @                               
                                                                
)";

static const char* coinPrompt = "Please enter the symbol for the coin you'd like to mine (i.e. DERO, XEL)";
static const char* daemonPrompt = "Please enter your mining deamon/host address: ";
static const char* portPrompt = "Please enter your mining port: ";
static const char* walletPrompt = "Please enter your wallet address for mining rewards: ";
static const char* threadPrompt = "Please provide the desired amount of mining threads: ";

static const char* inputIntro = "Please provide your mining settings (leave fields blank to use defaults)";

static int colorPreTable[] = {
    0,   0,   0,   0,   0,   0,   0,   0,     1,   1,   1,   1,   1,   1,   1,   1
};
static int colorTable[] = {
    30,  31,  32,  33,  34,  35,  36,  37,    90,  91,  92,  93,  94,  95,  96,  97
};

#define BLACK           0
#define RED             1
#define GREEN           2
#define YELLOW          3
#define BLUE            4
#define MAGENTA         5
#define CYAN            6
#define WHITE           7
#define BRIGHT_BLACK    8
#define BRIGHT_RED      9
#define BRIGHT_GREEN    10
#define BRIGHT_YELLOW   11
#define BRIGHT_BLUE     12
#define BRIGHT_MAGENTA  13
#define BRIGHT_CYAN     14
#define BRIGHT_WHITE    15

#if defined(_WIN32)
static WORD winColorTable[] = {
    0,                                                                      // BLACK
    FOREGROUND_RED,                                                         // RED
    FOREGROUND_GREEN,                                                       // GREEN
    FOREGROUND_RED | FOREGROUND_GREEN,                                      // YELLOW
    FOREGROUND_BLUE,                                                        // BLUE
    FOREGROUND_RED | FOREGROUND_BLUE,                                       // MAGENTA
    FOREGROUND_GREEN | FOREGROUND_BLUE,                                     // CYAN
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,                    // WHITE
    FOREGROUND_INTENSITY,                                                   // BRIGHT_BLACK
    FOREGROUND_RED | FOREGROUND_INTENSITY,                                  // BRIGHT_RED
    FOREGROUND_GREEN | FOREGROUND_INTENSITY,                                // BRIGHT_GREEN
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY,               // BRIGHT_YELLOW
    FOREGROUND_BLUE | FOREGROUND_INTENSITY,                                 // BRIGHT_BLUE
    FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY,                // BRIGHT_MAGENTA
    FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY,              // BRIGHT_CYAN
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY  // BRIGHT_WHITE
};

inline void setcolor(int color)
{
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), winColorTable[color]);
}
#else
inline void setcolor(int color)
{
    printf("\e[%d;%dm", colorPreTable[color], colorTable[color]);
}
#endif

inline po::options_description get_prog_opts()
{
  int col_width = 80;
  #if defined(__linux__)
    try {
      struct winsize w;
      ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
      col_width=w.ws_col;
    }
    catch (...)
    {
      //std::cout << "ws error\n";
    }
  #endif

  po::options_description general("General", col_width);
  general.add_options()
    ("help", "Produce help message")
    ("broadcast", "Creates an http server to query miner stats")
    ("mmpos", "Enable mmpOS-compatible /mmpos API endpoint (implies --broadcast)")
    ("testnet", "Adjusts in-house parameters to mine on testnets")
    ("daemon-address", po::value<std::string>(), "Node/pool URL or IP address to mine to") // todo: parse out port and/or wss:// or ws://
    ("port", po::value<int>(), "The port used to connect to the node")
    ("wallet", po::value<std::string>(), "Wallet address for receiving mining rewards")
    ("threads", po::value<int>(), "The amount of mining threads to create, default is 1")
    ("dev-fee", po::value<double>(), "Your desired dev fee percentage, default is 2.5, minimum is 1")
    ("report-interval", po::value<int>(), "Your desired status update interval in seconds")
    ("no-lock", "Disables CPU affinity / CPU core binding")
    ("priority", po::value<std::string>(), "<normal|above|high> Set mining thread priority (default: normal). WARNING: 'high' may reduce system responsiveness on Windows")
    ("no-msr", "Disable MSR optimization")
    ("ignore-wallet", "Disables wallet validation, for specific uses with pool mining")
    ("no-cpu", "Disable CPU mining (GPU only)")
    ("no-gpu", "Disable GPU mining (CPU only)")
  ;

  po::options_description stratum("Stratum", col_width);
  stratum.add_options()
    ("stratum", "Required for Stratum pools if not using 'stratum+tcp://' or 'stratum+ssl://' in the daemon url")
    ("password", po::value<std::string>(), "Sets the Stratum password")
    ("worker-name", po::value<std::string>(), "Sets the worker name for this instance when mining on Pools or Bridges")
  ;

  po::options_description coins("Coin Selection", col_width);
  coins.add_options()
    ("<symbol>", "Mine the coin corresponding to <symbol>. "
                "For versioned algorithms, append version: --xel-v3, --xel=v2, etc. "
                "Supported versioned coins: XEL (v1/v2/v3, default v3)")
    ("randomx", "For mining RandomX coins")
    ("yespower", po::value<std::string>(), "Mine with custom yespower parameters (format: N=2048,R=32,pers=string,ver=1.0). ver is optional, defaults to 1.0 (use 0.5 for yescrypt)")
  ;

  po::options_description dero("Dero", col_width);
  dero.add_options()
    ("dero-benchmark", po::value<int>(), "Runs a mining benchmark for <arg> seconds (adheres to -t threads option)")
  ;

  po::options_description xelis("Xelis", col_width);
  xelis.add_options()
    ("xatum", "Required for mining to Xatum pools on Xelis")
    ("bench-xelis", "Run a benchmark of xelis-hash with 1 thread")
    ("xelis-simd", po::value<std::string>(), "<avx512|avx2|aes|none> Override stage 1 SIMD dispatch level")
  ;

  po::options_description randomX("RandomX", col_width);
  randomX.add_options()
    ("rx-hugepages", "Use huge pages for RandomX")
    ("test-randomx", "Run Tevador's reference RandomX tests")
  ;

  po::options_description testing("Testing", col_width);
  testing.add_options()
    ("test-dero", "Runs a set of tests to verify AstrobwtV3 is working (1 test expected to fail)")
    ("test-spectre", "Run detailed diagnostics for SpectreX")
    ("test-xelis", "Run the xelis-hash tests from the official source code")
    ("hip-test-xelis", "Run stage-by-stage validation of Xelis v3 GPU kernels")
    ("test-yespower", "Run yespower known-vector tests for all builtin variants")
    ("test-astrix", "Run a basic astrix-hash validation test")
    ("test-hoosat", "Run a basic hoohash validation test")
    ("test-nexellia", "Run a basic nxl-hash validation test")
    ("test-waglayla", "Run a basic wala-hash validation test")
    ("test-shai", "Run a basic shai-hive validation test")
    ("test-rin", "Run a basic rinhash validation test")
  ;

  po::options_description advanced("Advanced", col_width);
  advanced.add_options()
    ("tune-warmup", po::value<int>()->default_value(1), "Number of seconds to warmup the CPU before starting the AstroBWTv3 tuning")
    ("tune-duration", po::value<int>()->default_value(2), "Number of seconds to tune *each* AstroBWTv3 algorithm. There will 3 or 4 algorithms depending on supported CPU features")
    ("no-tune", po::value<std::string>(), "<branch|lookup|avx2|wolf|aarch64> Use the specified AstroBWTv3 algorithm and skip tuning")
    ("mine-time", po::value<int>()->default_value(0), "Mine for a given number of seconds and then exit")
    ("gpu-retune", po::value<std::string>()->implicit_value(""), "Re-run GPU autotune (optional: comma-separated device indices, e.g. 0,2)")
    ("devices", po::value<std::string>(), "Comma-separated list of GPU indices to mine on (e.g. 0,1,3)")
    ("gpu-disable", po::value<std::string>(), "Comma-separated list of GPU indices to exclude (e.g. 2,5)")
  ;

  po::options_description debug("DEBUG", col_width);
  debug.add_options()
    ("op", po::value<int>(), "Sets which branch op to benchmark (0-255), benchmark will be skipped if unspecified")
    ("len", po::value<int>(), "Sets length of the processed chunk in said benchmark (default 15)")
    ("sabench", "Runs a benchmark for divsufsort on snapshot files in the 'tests' directory")
    ("quiet", "Do not print TNN banner or stratum job messages")
    ("log-level", po::value<std::string>(), "<off|info|debug|trace> Set log verbosity (default: info)")
  ;

  po::options_description gpuOC("GPU Overclocking (requires root/CAP_SYS_ADMIN)", col_width);
  gpuOC.add_options()
    ("gpu-plimit<N>", "Power limit in watts (on Windows AMD/ADL: % offset from stock TDP). "
                      "Single value = all GPUs; comma list = per-device L-to-R; "
                      "append index for one GPU: --gpu-plimit0 250")
    ("gpu-cclock<N>", "Lock core clock to absolute MHz")
    ("gpu-coff<N>",   "Core clock offset, MHz (+/-). NVML native; AMD = offset from stock max")
    ("gpu-moff<N>",   "Memory clock offset, MHz (+/-)")
  ;

  general.add(stratum);
  general.add(coins);
  #ifdef TNN_HIP
  general.add(gpuOC);
  #endif
  general.add(dero);
  general.add(xelis);
  general.add(randomX);
  general.add(testing);
  general.add(advanced);
  general.add(debug);
  return general;
}

inline int get_prog_style()
{
    int style = (po::command_line_style::unix_style | po::command_line_style::allow_long_disguise);
    return style;
}