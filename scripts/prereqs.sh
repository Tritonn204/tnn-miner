#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

echo $SCRIPT_DIR
echo $ROOT_DIR
#exit 0

if [[ -f /etc/lsb-release ]]; then
  source /etc/lsb-release
  sudo apt install git wget build-essential cmake clang libssl-dev libudns-dev libfmt-dev libc++-dev 
  if [[ "$DISTRIB_CODENAME" == "jammy" ]]; then
    cd "${ROOT_DIR}"
    wget https://github.com/boostorg/boost/releases/download/boost-1.82.0/boost-1.82.0.tar.gz
    tar -xf boost-1.82.0.tar.gz
    cd boost-1.82.0/
    ./bootstrap.sh --with-toolset=clang 
    ./b2 clean
    ./b2 toolset=clang cxxflags=-std=c++20 -stdlib=libc++ linkflags=-stdlib=libc++ link=static

  elif [[ "$DISTRIB_CODENAME" == "noble" ]]; then
    sudo apt install libboost1.83-all-dev lld
  fi
fi
