# tnn-miner
# An optimized, open-source Cryptocurrency

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

If you have trouble compiling, please either file a GitLab issue or find @Tritonn or @dirker on Discord: https://discord.gg/xeVtduYteK

## USAGE
This miner can be activated from the command line with the following parameters. Simply adjust the syntax for use with your shell or terminal of choice!
```
General:
  --help                    Produce help message
  --broadcast               Creates an http server to query miner stats
  --mmpos                   Enable mmpOS-compatible /mmpos API endpoint
                            (implies --broadcast)
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
  --priority arg            <normal|above|high> Set mining thread priority
                            (default: normal). WARNING: 'high' may reduce
                            system responsiveness on Windows
  --no-msr                  Disable MSR optimization
  --ignore-wallet           Disables wallet validation, for specific uses with
                            pool mining

Stratum:
  --stratum                 Required for Stratum pools if not using
                            'stratum+tcp://' or 'stratum+ssl://' in the daemon
                            url
  --password arg            Sets the Stratum password
  --worker-name arg         Sets the worker name for this instance when mining
                            on Pools or Bridges

Coin Selection:
  --<symbol>                Mine the coin corresponding to <symbol>. For
                            versioned algorithms, append version: --xel-v3,
                            --xel=v2, etc. Supported versioned coins: XEL
                            (v1/v2/v3, default v3)
  --randomx                 For mining RandomX coins
  --yespower arg            Mine with custom yespower parameters (format:
                            N=2048,R=32,pers=string,ver=1.0). ver is optional,
                            defaults to 1.0 (use 0.5 for yescrypt)

Dero:
  --dero-benchmark arg      Runs a mining benchmark for <arg> seconds (adheres
                            to -t threads option)

Xelis:
  --xatum                   Required for mining to Xatum pools on Xelis
  --bench-xelis             Run a benchmark of xelis-hash with 1 thread
  --xelis-simd arg          <avx512|avx2|aes|none> Override stage 1 SIMD
                            dispatch level

RandomX:
  --rx-hugepages            Use huge pages for RandomX
  --test-randomx            Run Tevador's reference RandomX tests

Testing:
  --test-dero               Runs a set of tests to verify AstrobwtV3 is working
                            (1 test expected to fail)
  --test-spectre            Run detailed diagnostics for SpectreX
  --test-xelis              Run the xelis-hash tests from the official source
                            code
  --hip-test-xelis          Run stage-by-stage validation of Xelis v3 GPU
                            kernels
  --test-yespower           Run yespower known-vector tests for all builtin
                            variants
  --test-astrix             Run a basic astrix-hash validation test
  --test-hoosat             Run a basic hoohash validation test
  --test-nexellia           Run a basic nxl-hash validation test
  --test-waglayla           Run a basic wala-hash validation test
  --test-shai               Run a basic shai-hive validation test
  --test-rin                Run a basic rinhash validation test

Advanced:
  --tune-warmup arg (=1)    Number of seconds to warmup the CPU before starting
                            the AstroBWTv3 tuning
  --tune-duration arg (=2)  Number of seconds to tune *each* AstroBWTv3
                            algorithm. There will 3 or 4 algorithms depending
                            on supported CPU features
  --no-tune arg             <branch|lookup|avx2|wolf|aarch64> Use the specified
                            AstroBWTv3 algorithm and skip tuning
  --mine-time arg (=0)      Mine for a given number of seconds and then exit
  --gpu-retune              Delete GPU autotune cache and re-run tuning from
                            scratch

DEBUG:
  --op arg                  Sets which branch op to benchmark (0-255),
                            benchmark will be skipped if unspecified
  --len arg                 Sets length of the processed chunk in said
                            benchmark (default 15)
  --sabench                 Runs a benchmark for divsufsort on snapshot files
                            in the 'tests' directory
  --quiet                   Do not print TNN banner or stratum job messages
  --log-level arg           <off|info|debug|trace> Set log verbosity (default:
                            info)
```
### If the miner is run without any args, a CLI wizard will simply ask you to provide the required options one at a time.

## mmpos (mmpOS) Setup

tnn-miner ships with custom miner wrappers for [mmpOS](https://app.mmpos.eu/). Three variants are provided:

| Variant | Binary | Broadcast Port | Use Case |
|---------|--------|---------------|----------|
| `hiveos+mmpos` | `tnn-miner` | 8989 | CPU mining |
| `hiveos+mmpos-amd` | `tnn-miner-rocm` | 8990 | AMD GPU mining |
| `hiveos+mmpos-nvidia` | `tnn-miner-cuda` | 8991 | NVIDIA GPU mining |

### Installation

CPU, NVIDIA, and AMD each require their own wrapper — make sure to use the one matching your hardware.

1. Download the appropriate wrapper archive from the [releases page](https://git.tritonn.dev/Tritonn/tnn-miner/-/releases)
2. Upload it as a custom miner in mmpOS

### Flight Sheet Configuration

- **Pool**: `pool_host:port` or `stratum+tcp://pool_host:port`
- **Wallet**: Your wallet address
- **Password**: Optional, defaults to `x`
- **Algorithm**: The wrapper defaults to Xelis (`--XEL`). To mine a different coin, add the coin flag in **extra arguments**, e.g. `--XMR`

### Features

- Native `/mmpos` HTTP endpoint with per-device hashrate, PCI bus IDs, and per-device share tracking (accepted/rejected)
- Per-GPU stats: hashrate, PCI bus mapping, share counts
- Stats served directly from the miner process (no log parsing)

If you intend to build from source without dev fees, please consider a one-time donation:

- **Dero**: `dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7` (or Dero Name Service: **_tritonn_**)
- **Monero**: `49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe`
- **Xelis**: `xel:xz9574c80c4xegnvurazpmxhw5dlg2n0g9qm60uwgt75uqyx3pcsqzzra9m`

Dev fees allow me to invest more time into maintaining, updating, and improving tnn-miner.

