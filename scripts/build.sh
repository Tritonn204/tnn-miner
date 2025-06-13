#!/bin/bash

ret=0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

VER_SETTING=
if [[ "$1" != "" ]]; then
  VER_SETTING=-DTNN_VERSION=$1
  echo $VER_SETTING
fi

mkdir build18
pushd build18
  cmake -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 $VER_SETTING ..
  #ret=$?
  #if [[ "$ret" != "0" ]]; then
  #  rm -rf ./*
  #  cmake $VER_SETTING ..
  #fi
  cmake --build . -j $(nproc)
  ret=$?
popd

exit $ret
