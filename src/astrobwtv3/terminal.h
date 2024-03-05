#pragma once

#include <vector>
#include <string>
#include <algorithm>

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
        Runs a set of tests to verify AstrobwtV3 is working
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