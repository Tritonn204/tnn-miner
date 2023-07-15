# tnn-miner
# An open-source Astrobwtv3 miner

Dependencies are as follows:
  - OpenSSL v3.0.2
  - Boost v1.8.2 (b2 with link=static)
  - GMP (with C++ support enabled)
  - divsufsort (with build_shared turned off)
  - FMT (header only)

## This repo can be built from source via cmake once these libraries are installed on your system
```
git clone https://github.com/Tritonn204/tnn-miner.git
cd tnn-miner
mkdir build
cd build
cmake ..
make
```
MinGW will work, just swap "make" with "mingw32-make".
