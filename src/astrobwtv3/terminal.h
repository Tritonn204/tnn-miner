#pragma once

#include <vector>
#include <string>
#include <algorithm>

namespace po = boost::program_options;  // from <boost/program_options.hpp>

const char *consoleLine = " TNN-MINER v0.2.0 ";
const char *TNN = R"(
  
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

#define TNN_DAEMON 0
#define TNN_PORT 2
#define TNN_WALLET 4
#define TNN_THREADS 6
#define TNN_FEE 8
#define TNN_HELP 10
#define TNN_TEST 12
#define TNN_BENCHMARK 13
#define TNN_NO_LOCK 14
#define TNN_GPUMINE 15
#define TNN_BATCHSIZE 16
#define TNN_SABENCH 18
#define TNN_OP 19
#define TNN_TLEN 20

std::vector<std::string> options = {
    "-daemon-address",
    "-d",
    "-port",
    "-p",
    "-wallet",
    "-w",
    "-threads",
    "-t",
    "-dev-fee",
    "-f",
    "-help",
    "-h",
    "-test",
    "-benchmark",
    "-no-lock",
    "-gpu",
    "-batch-size",
    "-b",
    "-sabench",
    "-o",
    "-l"
};

const char* usage = R"(
OPTIONS
    -gpu:
        Mine with GPU instead of CPU
    -daemon-address, -d: 
        Dero node/pool URL or IP address to mine to
    -port, -p: 
        The port used to connect to the Dero node
    -wallet, -w: 
        Wallet address for receiving mining rewards
    -threads, -t: (optional) 
        The amount of mining threads to create, default is 1
    -dev-fee, -f: (optional) 
        Your desired dev fee percentage, default is 2.5, minimum is 1
    -no-lock: (optional) 
        Disables CPU affinity / CPU core binding
    -help, -h: (must be first arg)
        Shows help
    -batch-size, -b: (GPU Setting)
        Sets batch size used for GPU mining
    -sabench: (must be first arg)
        Runs a benchmark for divsufsort on snapshot files in the 'tests' directory
DEBUG
    -test: (must be first arg)
        Runs a set of tests to verify AstrobwtV3 is working (1 test expected to fail)
        Params: (optional)
          -o <num> : Sets which branch op to benchmark (0-255), benchmark will be skipped if unspecified
          -l <num> : Sets length of the processed chunk in said benchmark (default 15) 
    -benchmark <A> <B>:
        Runs a mining benchmark for <B> seconds with <A> threads for hashrate testing
        You may insert the -no-lock flag after <A> and <B> if desired. 
)";

const char* daemonPrompt = "Please enter your mining deamon/host address: ";
const char* portPrompt = "Please enter your mining port: ";
const char* walletPrompt = "Please enter your Dero wallet address for mining rewards: ";
const char* threadPrompt = "Please provide the desired amount of mining threads: ";

const char* inputIntro = "Please provide your mining settings (leave fields blank to use defaults)";

int colorPreTable[] = {
  0,0,0,0,91,
  0,0,0,0,1,
  0,0,0,0,1,
  0
};
int colorTable[] = {
  0,0,0,36,91,
  0,0,0,0,91,
  0,0,0,0,93,
  37
};

#define CYAN 3
#define RED 4
#define BRIGHT_YELLOW 14
#define BRIGHT_WHITE 15

#if defined(_WIN32)
inline void setcolor(WORD color)
{
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),color);
    return;
}
#else
inline void setcolor(int color)
{
    printf("\e[%d;%dm", colorPreTable[color], colorTable[color]);
}
#endif

inline po::options_description get_prog_opts()
{
  //po::command_line_parser parser = po::command_line_parser(argc, argv);

  po::options_description general("General");
  general.add_options()
    ("help", "produce help message")
    ("daemon-address", po::value<std::string>(), "Dero node/pool URL or IP address to mine to") // todo: parse out port and/or wss:// or ws://
    ("port", po::value<int>(), "The port used to connect to the Dero node")
    ("wallet", po::value<std::string>(), "Wallet address for receiving mining rewards")
    ("threads", po::value<int>(), "The amount of mining threads to create, default is 1")
    ("dev-fee", po::value<double>(), "Your desired dev fee percentage, default is 2.5, minimum is 1")
    ("no-lock", po::value<bool>(), "Disables CPU affinity / CPU core binding")
    ("batch-size", po::value<int>(), "(GPU Setting) Sets batch size used for GPU mining")
    ("simd", "Mine with SIMD instead of regular C++")
    ("compression", po::value<double>(), "set compression level")
  ;

  po::options_description debug("DEBUG");
  debug.add_options()
    ("test", "Runs a set of tests to verify AstrobwtV3 is working (1 test expected to fail)")
    ("op", po::value<int>(), "Sets which branch op to benchmark (0-255), benchmark will be skipped if unspecified")
    ("len", po::value<int>(), "Sets length of the processed chunk in said benchmark (default 15)")
    ("sabench", "Runs a benchmark for divsufsort on snapshot files in the 'tests' directory")
    ("benchmark", po::value<int>(), "Runs a mining benchmark for <arg> seconds (adheres to -t threads option)")
    ("verify", "Verifies SIMD produces identical results to C++. Adheres to -op and -len options")
  ;

  return general.add(debug);
}

inline int get_prog_style()
{
    int style = (po::command_line_style::unix_style | po::command_line_style::allow_long_disguise);
    return style;
}