# tnn-miner
# An open-source Dero/Spectre/Xelis miner

**Dependencies:**
  - OpenSSL v3.0.2 or higher (static libs)
  - Boost v1.8.2 or higher (b2 with link=static)
  - UDNS (UNIX only)
  - Sodium

### Building the easy way!
Use the prereqs.sh scripts (one-time only)
```
./scripts/prereqs.sh
```
Then build!
```
./scripts/build.sh
```
### For both Ubuntu 22.04 (Jammy) and 24.04 (Noble):
Install development dependencies
```
sudo apt install git wget build-essential cmake clang libssl-dev libudns-dev libc++-dev lld libsodium-dev
```
### *Only* for Ubuntu 24.04 (Noble):
Install Boost development dependency, too
```
sudo apt install libboost1.83-all-dev
```

### Windows: Building the easy way!
Use the prereqs.bat scripts (one-time only)
This will download mingw and a few other libraries like Boost and OpenSSL
```
.\scripts\prereqs.bat
```
Then build! Notice this is a powershell script
```
powershell -File .\scripts\build.ps1
```

### Clone the source and then build!
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

### Support

If you have trouble compiling, please either file a GitLab issue or find @Tritton or @dirker on Discord: https://discord.gg/xeVtduYteK

## USAGE
This miner can be activated from the command line with the following parameters. Simply adjust the syntax for use with your shell or terminal of choice!
```
General:
  --help                    Produce help message
  --dero                    Will mine Dero
  --xelis                   Will mine Xelis
  --spectre                 Will mine Spectre
  --stratum                 Required for mining to Stratum pools
  --broadcast               Creates an http server to query miner stats
  --testnet                 Adjusts in-house parameters to mine on testnets
  --daemon-address arg      Node/pool URL or IP address to mine to
  --port arg                The port used to connect to the node
  --wallet arg              Wallet address for receiving mining rewards
  --threads arg             The amount of mining threads to create, default is
                            1
  --dev-fee arg             Your desired dev fee percentage, default is 2.5,
                            minimum is 1
  --report-interval arg     Your desired status update interval in seconds
  --no-lock                 Disables CPU affinity / CPU core binding
  --ignore-wallet           Disables wallet validation, for specific uses with
                            pool mining

Dero:
  --lookup                  Mine with lookup tables instead of computation
  --dero-benchmark arg      Runs a mining benchmark for <arg> seconds (adheres
                            to -t threads option)

Spectre:
  --test-spectre            Run detailed diagnostics for SpectreX

Xelis:
  --xatum                   Required for mining to Xatum pools on Xelis
  --xelis-bench             Run a benchmark of xelis-hash with 1 thread
  --test-xelis              Run the xelis-hash tests from the official source
                            code
  --worker-name arg         Sets the worker name for this instance when mining
                            Xelis

Advanced:
  --tune-warmup arg (=1)    Number of seconds to warmup the CPU before starting
                            the AstroBWTv3 tuning
  --tune-duration arg (=2)  Number of seconds to tune *each* AstroBWTv3
                            algorithm. There will 3 or 4 algorithms depending
                            on supported CPU features
  --no-tune arg             <branch|lookup|avx2|wolf|aarch64> Use the specified
                            AstroBWTv3 algorithm and skip tuning

DEBUG:
  --test-dero               Runs a set of tests to verify AstrobwtV3 is working
                            (1 test expected to fail)
  --op arg                  Sets which branch op to benchmark (0-255),
                            benchmark will be skipped if unspecified
  --len arg                 Sets length of the processed chunk in said
                            benchmark (default 15)
  --sabench                 Runs a benchmark for divsufsort on snapshot files
                            in the 'tests' directory
  --quiet                   Do not print TNN banner
```
### If the miner is run without any args, a CLI wizard will simply ask you to provide the required options one at a time.

If you intend to build from source without dev fees, please consider a one-time donation to the Dero address **_tritonn_** (Dero Name Service). 

Dev fees allow me to invest more time into maintaining, updating, and improving tnn-miner.

