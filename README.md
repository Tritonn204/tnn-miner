# tnn-miner
# An open-source Astrobwtv3 miner

**Dependencies are as follows:**
  - OpenSSL v3.0.2 (static libs)
  - Boost v1.8.2 (b2 with link=static)
  - UDNS (UNIX only. sudo apt-get install libudns-dev)
  - FMT (header only)

**Building the easy way**
Use the prereqs.sh scripts (one-time only)
```
./scripts/prereqs.sh
```
Then build!
```
./scripts/build.sh
```

**For Ubuntu 24.04:**
Install development dependencies for Ubuntu 22.04 below, but also install the Boost dev libraries
```
sudo apt install libboost1.83-all-dev
```

**For Ubuntu 22.04:**
Install development dependencies
```
sudo apt install git wget build-essential cmake clang libssl-dev libudns-dev libfmt-dev libc++-dev lld 
# Checkout tnn-miner got from github
git clone https://github.com/Tritonn204/tnn-miner.git
cd tnn-miner
mkdir build
cd build
cmake ..
make -j $(nproc)
```

### This repo can be built from source via cmake once the libraries above are installed on your system
```
git clone https://github.com/Tritonn204/tnn-miner.git
cd tnn-miner
mkdir build
cd build
cmake ..
make
```
### MinGW will work, just swap "make" with "mingw32-make".

Do note that CMakeLists.txt will need to be altered if your libraries are installed at neither **C:/mingw64** nor the **root dir** of this project on Windows.

# USAGE
This miner can be activated from the command line with the following parameters. Simply adjust the syntax for use with your shell or terminal of choice!
```
General:
  --help                 produce help message
  --daemon-address arg   Dero node/pool URL or IP address to mine to
  --port arg             The port used to connect to the Dero node
  --wallet arg           Wallet address for receiving mining rewards
  --threads arg          The amount of mining threads to create, default is 1
  --dev-fee arg          Your desired dev fee percentage, default is 2.5, minimum is 1
  --no-lock              Disables CPU affinity / CPU core binding
  --lookup               Mine with lookup instead of regular C++

DEBUG:
  --test                 Runs a set of tests to verify AstrobwtV3 is working (1 test expected to fail)
  --op arg               Sets which branch op to benchmark (0-255), benchmark will be skipped if unspecified
  --len arg              Sets length of the processed chunk in said benchmark (default 15)
  --sabench              Runs a benchmark for divsufsort on snapshot files in the 'tests' directory
  --benchmark arg        Runs a mining benchmark for <arg> seconds (adheres to -t threads option) 
```
### If the miner is run without any args, a CLI wizard will simply ask you to provide the required options one at a time.

If you intend to build from source without dev fees, please consider a one-time donation to the Dero address **_tritonn_** (Dero Name Service). 

Dev fees allow me to invest more time into maintaining, updating, and improving tnn-miner.

