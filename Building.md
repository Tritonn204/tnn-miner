**For Ubuntu 24.04:**
Install development dependencies
```
sudo apt install git wget build-essential cmake clang libssl-dev libudns-dev libfmt-dev libc++-dev libboost1.83-all-dev lld
# Checkout tnn-miner got from github
git clone https://github.com/Tritonn204/tnn-miner.git
cd tnn-miner
mkdir build
cd build
cmake ..
make -j $(nproc)
```

**For Ubuntu 22.04:**
Install development dependencies
```
sudo apt install git wget build-essential cmake clang libssl-dev libudns-dev libfmt-dev libc++-dev 
# Checkout tnn-miner got from github
git clone https://github.com/Tritonn204/tnn-miner.git
cd tnn-miner
```

Download and compile Boost 1.82.  This is a one-time thing.
-- Newer versions of Boost exist, but there's some issue linking program_options
```
wget https://github.com/boostorg/boost/releases/download/boost-1.82.0/boost-1.82.0.tar.gz
tar -xf boost-1.82.0.tar.gz
cd boost-1.82.0/
./bootstrap.sh --with-toolset=clang 
./b2 clean
./b2 toolset=clang cxxflags=-std=c++20 -stdlib=libc++ linkflags=-stdlib=libc++ link=static
```
Proceed with Tnn-miner build
```
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