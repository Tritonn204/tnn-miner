# tnn-miner

An optimized, open-source cryptocurrency miner with CPU mining, unified Orochi GPU mining, pool/solo connectivity, Stratum support, Xatum support for Xelis, mmpOS-compatible monitoring, and built-in diagnostics/benchmarks.

The current GPU path uses Orochi for runtime dispatch, so one build can discover and use supported AMD or NVIDIA devices when the required vendor runtime libraries are present. CPU and GPU mining can be enabled together or independently with `--no-cpu` and `--no-gpu`.

## Dependencies

- CMake and Ninja
- Clang/LLVM
- OpenSSL v3.0.2 or higher
- Boost v1.82 or higher
- libsodium
- UDNS on Unix-like systems
- Visual Studio C++ tools on Windows for MSVC/clang-cl build paths
- AMD HIP/ROCm runtime or NVIDIA CUDA runtime for GPU mining, depending on hardware

## Building

### Linux

Use the prerequisite helper once:

```sh
./scripts/prereqs.sh
```

Then build:

```sh
./scripts/build.sh
```

For Ubuntu 22.04 and 24.04, the core development packages are:

```sh
sudo apt install git wget build-essential cmake clang libssl-dev libudns-dev libc++-dev lld libsodium-dev
```

Ubuntu 24.04 also needs Boost development packages:

```sh
sudo apt install libboost1.83-all-dev
```

### Windows

Use the prerequisite helper once. This downloads MinGW and common third-party libraries such as Boost and OpenSSL:

```powershell
.\scripts\prereqs.bat
```

CPU-only build:

```powershell
powershell -File .\scripts\build.ps1
```

Build all standard Windows targets with an explicit build label:

```powershell
powershell -File .\scripts\build_all.ps1 <build-label>
```

Build only the unified Orochi GPU target:

```powershell
powershell -File .\scripts\build_all.ps1 <build-label> orochi
```

Typical Orochi output path on Windows:

```text
hip-build\win32\orochi\bin\tnn-miner.exe
```

### Manual CMake

```sh
git clone https://gitlab.com/Tritonn204/tnn-miner.git
cd tnn-miner
mkdir build
cd build
cmake ..
make
```

With MinGW, use `mingw32-make` instead of `make`.

On Windows, `CMakeLists.txt` expects dependencies in either `C:/mingw64` or the root of this project unless you provide alternate paths.

## Usage

Run the miner with a coin flag, daemon/pool address, port, wallet, and desired CPU thread count:

```sh
tnn-miner --xel --daemon-address stratum+ssl://pool.example.com --port 5177 --wallet xel:your_wallet --threads 8
```

GPU-only mining:

```sh
tnn-miner --xel --daemon-address stratum+ssl://pool.example.com --port 5177 --wallet xel:your_wallet --no-cpu
```

When mining with CPU and GPU together, set `--threads` 1-2 lower than your normal CPU-only thread count. Leaving some CPU headroom for the GPU driver and submission work usually avoids reducing total hashrate.

CPU-only mining:

```sh
tnn-miner --dero --daemon-address node.example.com --port 10100 --wallet dero1... --threads 8 --no-gpu
```

Use a Stratum URL directly, or add `--stratum` when the URL does not include `stratum+tcp://` or `stratum+ssl://`:

```sh
tnn-miner --rvn --daemon-address stratum+tcp://rvn.example.com --port 6060 --wallet R... --no-cpu
```

Generic KawPow pool mining:

```sh
tnn-miner --kawpow --daemon-address stratum+tcp://pool.example.com --port 6060 --wallet wallet.worker --password x --no-cpu
```

Generic RandomX pool mining:

```sh
tnn-miner --rx0 --daemon-address stratum+ssl://pool.example.com --port 9999 --wallet wallet.worker --password x --threads 8 --no-gpu
```

If the miner is run without arguments, it starts an interactive CLI wizard and asks for required options one at a time.

## Supported Coin Symbols

Coin selection uses `--<symbol>`. Symbols are case-insensitive in practice, but examples use lowercase CLI flags.

| Symbol | Coin / Mode | Algorithm |
|--------|-------------|-----------|
| `--dero` | Dero | AstroBWTv3 |
| `--xel`, `--xel-v2`, `--xel-v3`, `--xel=v2`, `--xel=v3` | Xelis | XelisHash |
| `--spr` | Spectre | SpectreX |
| `--rx0` | Generic RandomX | RandomX |
| `--xmr` | Monero | RandomX |
| `--sal` | Salvium | RandomX |
| `--zeph` | Zephyr | RandomX |
| `--xtm` | Tari | RandomX |
| `--vrsc` | Verus | VerusHash |
| `--aix` | Astrix | astrix-hash |
| `--nxl` | Nexellia | nxl-hash |
| `--htn` | Hoosat | hoohash |
| `--wala` | Waglayla | wala-hash |
| `--shai` | Shai | shai-hive |
| `--rin` | RinCoin | rinhash |
| `--yespower` | Generic yespower parameters | yespower / yescrypt |
| `--advc` | AdventureCoin | yespower |
| `--tdc` | Tidecoin | YespowerTIDE |
| `--ytn` | Yenten | YespowerR16 |
| `--gold` | Goldcash | YescryptR16 |
| `--mtbc` | MTBC | YescryptR8 |
| `--mgpc` | Magpiecoin | YespowerMGPC |
| `--urx` | UraniumX | YespowerURX |
| `--crnc` | Crane | YespowerLTNCG |
| `--ysc` | Yescrypt | yescrypt |
| `--eqpay` | EqpayCoin | YespowerEQPAY |
| `--lpepe` | LuckyPepe | YescryptR32 |
| `--kawpow` | Generic KawPow | KawPow |
| `--rvn` | Ravencoin | KawPow |
| `--quai` | Quai Network | KawPow |

For custom yespower settings:

```sh
tnn-miner --yespower N=2048,R=32,pers=example,ver=1.0 --daemon-address stratum+tcp://pool.example.com --port 1234 --wallet wallet
```

Use `ver=0.5` for yescrypt-style parameters.

## Monitoring

`--broadcast` starts the built-in HTTP stats server.

`--mmpos` enables the mmpOS-compatible `/mmpos` endpoint and implies `--broadcast`.

The miner reports per-device GPU stats when GPU mining is enabled, including hashrate and share counters. The mmpOS endpoint is served directly by the miner process and does not rely on log parsing.

## Current CLI Options

The following option list is based on the current `--help` output. The startup banner, detected devices, runtime libraries, and build label are intentionally omitted.

```text
General:
  --help                      Produce help message
  --broadcast                 Creates an http server to query miner stats
  --mmpos                     Enable mmpOS-compatible /mmpos API endpoint
                              (implies --broadcast)
  --testnet                   Adjusts in-house parameters to mine on testnets
  --daemon-address arg        Node/pool URL or IP address to mine to
  --port arg                  The port used to connect to the node
  --wallet arg                Wallet address for receiving mining rewards
  --threads arg               The amount of mining threads to create, default
                              is 1
  --dev-fee arg               Your desired dev fee percentage, default is 2.5,
                              minimum is 1
  --report-interval arg       Your desired status update interval in seconds
  --no-lock                   Disables CPU affinity / CPU core binding
  --priority arg              <normal|above|high> Set mining thread priority
                              (default: normal). WARNING: 'high' may reduce
                              system responsiveness on Windows
  --no-msr                    Disable MSR optimization
  --ignore-wallet             Disables wallet validation, for specific uses
                              with pool mining
  --no-cpu                    Disable CPU mining (GPU only)
  --no-gpu                    Disable GPU mining (CPU only)

Stratum:
  --stratum                   Required for Stratum pools if not using
                              'stratum+tcp://' or 'stratum+ssl://' in the
                              daemon url
  --password arg              Sets the Stratum password
  --worker-name arg           Sets the worker name for this instance when
                              mining on Pools or Bridges

Coin Selection:
  --<symbol>                  Mine the coin corresponding to <symbol>. For
                              versioned algorithms, append version: --xel-v3,
                              --xel=v2, etc. Supported versioned coins: XEL
                              (v1/v2/v3, default v3)
  --randomx                   For mining RandomX coins
  --yespower arg              Mine with custom yespower parameters (format:
                              N=2048,R=32,pers=string,ver=1.0). ver is
                              optional, defaults to 1.0 (use 0.5 for yescrypt)

GPU Overclocking (requires root/CAP_SYS_ADMIN):
  --gpu-plimit<N>             Power limit in watts (on Windows AMD/ADL: %
                              offset from stock TDP). Single value = all GPUs;
                              comma list = per-device L-to-R; append index for
                              one GPU: --gpu-plimit0 250
  --gpu-cclock<N>             Lock core clock to absolute MHz
  --gpu-coff<N>               Core clock offset, MHz (+/-). NVML native; AMD =
                              offset from stock max
  --gpu-moff<N>               Memory clock offset, MHz (+/-)

Dero:
  --dero-benchmark arg        Runs a mining benchmark for <arg> seconds
                              (adheres to -t threads option)

Xelis:
  --xatum                     Required for mining to Xatum pools on Xelis
  --bench-xelis               Run a benchmark of xelis-hash with 1 thread
  --xelis-simd arg            <avx512|avx2|aes|none> Override stage 1 SIMD
                              dispatch level

RandomX:
  --rx-hugepages              Use huge pages for RandomX
  --test-randomx              Run Tevador's reference RandomX tests

Testing:
  --test-dero                 Runs a set of tests to verify AstrobwtV3 is
                              working (1 test expected to fail)
  --test-spectre              Run detailed diagnostics for SpectreX
  --test-xelis                Run the xelis-hash tests from the official source
                              code
  --hip-test-xelis            Run stage-by-stage validation of Xelis v3 GPU
                              kernels
  --hip-test-kawpow           Run KawPow CPU reference + GPU kernel validation
  --bench-kawpow [=arg(=-1)]  Run KawPow GPU benchmark (optional: block height
                              for DAG epoch)
  --dump-kawpow arg           Dump generated ProgPoW program body for a given
                              block number
  --test-yespower             Run yespower known-vector tests for all builtin
                              variants
  --test-astrix               Run a basic astrix-hash validation test
  --test-hoosat               Run a basic hoohash validation test
  --test-nexellia             Run a basic nxl-hash validation test
  --test-waglayla             Run a basic wala-hash validation test
  --test-shai                 Run a basic shai-hive validation test
  --test-rin                  Run a basic rinhash validation test

Advanced:
  --tune-warmup arg (=1)      Number of seconds to warmup the CPU before
                              starting the AstroBWTv3 tuning
  --tune-duration arg (=2)    Number of seconds to tune *each* AstroBWTv3
                              algorithm. There will 3 or 4 algorithms depending
                              on supported CPU features
  --no-tune arg               <branch|lookup|avx2|wolf|aarch64> Use the
                              specified AstroBWTv3 algorithm and skip tuning
  --mine-time arg (=0)        Mine for a given number of seconds and then exit
  --gpu-retune arg            Re-run GPU autotune (optional: comma-separated
                              device indices, e.g. 0,2)
  --devices arg               Comma-separated list of GPU indices to mine on
                              (e.g. 0,1,3)
  --gpu-disable arg           Comma-separated list of GPU indices to exclude
                              (e.g. 2,5)

DEBUG:
  --op arg                    Sets which branch op to benchmark (0-255),
                              benchmark will be skipped if unspecified
  --len arg                   Sets length of the processed chunk in said
                              benchmark (default 15)
  --sabench                   Runs a benchmark for divsufsort on snapshot files
                              in the 'tests' directory
  --quiet                     Do not print TNN banner or stratum job messages
  --log-level arg             <off|info|debug|trace> Set log verbosity
                              (default: info)
```

## mmpOS Setup

tnn-miner ships with a custom miner wrapper for [mmpOS](https://app.mmpos.eu/). Current Linux releases use the unified Orochi wrapper, so the same package can run CPU mining, AMD GPU mining, NVIDIA GPU mining, or mixed CPU/GPU mining depending on your flags and installed drivers.

| Variant | Binary | Broadcast Port | Use Case |
|---------|--------|---------------|----------|
| `hiveos+mmpos` | `tnn-miner` | 8989 | Unified CPU, AMD GPU, NVIDIA GPU |

### Installation

1. Download the `hiveos_mmpos` wrapper archive from the [releases page](https://gitlab.com/Tritonn204/tnn-miner/-/releases).
2. Upload it as a custom miner in mmpOS.

### Flight Sheet Configuration

- **Pool**: `pool_host:port` or `stratum+tcp://pool_host:port`
- **Wallet**: your wallet address
- **Password**: optional, defaults to `x`
- **Algorithm**: the wrapper defaults to Xelis (`--XEL`) if no algorithm is provided. To mine a different coin, add the coin flag in extra arguments.

### Features

- Native `/mmpos` HTTP endpoint
- Per-device hashrate reporting
- PCI bus ID reporting for GPUs
- Per-device accepted/rejected share tracking
- Stats served directly from the miner process
- Bundled GPU runtime libraries for supported Linux AMD/NVIDIA deployments

## Support

If you have trouble compiling, file a GitLab issue or find `@Tritonn` or `@dirker` on Discord:

https://discord.gg/xeVtduYteK

## Donations

If you build from source without dev fees, please consider a one-time donation:

- **Dero**: `dero1qy5ewgqk8cw8drjhrcr0lpdcm26edqcwdwjke4x67m08nwd2hw4wjqqp6y2n7` or Dero Name Service: `_tritonn_`
- **Monero**: `49FCeAUYsPHYV3QLSKzQEpTgmKjHGYMzv2LMs4K7hprWK5FZNS31puWTsSxZo1rQTtVDw9Bi4YhRJYNyMc66zBuMMUhYJqe`
- **Xelis**: `xel:xz9574c80c4xegnvurazpmxhw5dlg2n0g9qm60uwgt75uqyx3pcsqzzra9m`

Dev fees help fund maintenance, updates, and continued tnn-miner development.
