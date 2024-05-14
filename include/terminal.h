#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <boost/program_options.hpp>

#if defined(__linux__)
  #include <sys/ioctl.h>
#endif

namespace po = boost::program_options;  // from <boost/program_options.hpp>

const char *versionString = "0.3.4";
const char *consoleLine = " TNN-MINER ";
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

const char *DERO = R"(
                                                                  
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
    @              @@     @@@@@@@@@     @@              @       
    @                  @@ @@@@@@@@@ @@                  @       
       @                    @@@@@                    @@         
           @@                                    @              
               @@                           @@                  
                   @@                   @@                      
                        @@         @@                           
                              @                               
                                                                
)";

const char* coinPrompt = "Please enter the symbol for the coin you'd like to mine (i.e. DERO, XEL)";
const char* daemonPrompt = "Please enter your mining deamon/host address: ";
const char* portPrompt = "Please enter your mining port: ";
const char* walletPrompt = "Please enter your wallet address for mining rewards: ";
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
    ("help", "produce help message")
    ("dero", "Will mine Dero")
    ("xelis", "Will mine Xelis")
    ("spectre", "Will mine Spectre")
    ("stratum", "Required for mining to Stratum pools")
    ("broadcast", "Creates an http server to query miner stats")
    ("testnet", "Adjusts in-house parameters to mine on testnets")
    ("daemon-address", po::value<std::string>(), "Node/pool URL or IP address to mine to") // todo: parse out port and/or wss:// or ws://
    ("port", po::value<int>(), "The port used to connect to the node")
    ("wallet", po::value<std::string>(), "Wallet address for receiving mining rewards")
    ("threads", po::value<int>(), "The amount of mining threads to create, default is 1")
    ("dev-fee", po::value<double>(), "Your desired dev fee percentage, default is 2.5, minimum is 1")
    ("no-lock", "Disables CPU affinity / CPU core binding")
    // ("gpu", "Mine with GPU instead of CPU")
    // ("batch-size", po::value<int>(), "(GPU Setting) Sets batch size used for GPU mining")
  ;

  po::options_description dero("Dero", col_width);
  dero.add_options()
    ("lookup", "Mine with lookup tables instead of computation")
    ("dero-benchmark", po::value<int>(), "Runs a mining benchmark for <arg> seconds (adheres to -t threads option)")
  ;

  po::options_description debug("DEBUG", col_width);
  debug.add_options()
    ("dero-test", "Runs a set of tests to verify AstrobwtV3 is working (1 test expected to fail)")
    ("op", po::value<int>(), "Sets which branch op to benchmark (0-255), benchmark will be skipped if unspecified")
    ("len", po::value<int>(), "Sets length of the processed chunk in said benchmark (default 15)")
    ("sabench", "Runs a benchmark for divsufsort on snapshot files in the 'tests' directory")
  ;

  po::options_description xelis("Xelis", col_width);
  xelis.add_options()
    ("xatum", "Required for mining to Xatum pools on Xelis")
    ("xelis-bench", "Run a benchmark of xelis-hash with 1 thread")
    ("xelis-test", "Run the xelis-hash tests from the official source code")
    ("worker-name", po::value<std::string>(), "Sets the worker name for this instance when mining Xelis")
  ;

  po::options_description spectre("Spectre", col_width);
  spectre.add_options()
    ("spectre-test", "Run detailed diagnostics for SpectreX")
  ;

  dero.add(debug);
  general.add(dero);
  general.add(xelis);
  general.add(spectre);
  return general;
}

inline int get_prog_style()
{
    int style = (po::command_line_style::unix_style | po::command_line_style::allow_long_disguise);
    return style;
}