#!/bin/bash

ret=0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

VER_SETTING=
if [[ "$1" != "" ]]; then
  VER_SETTING=-DTNN_VERSION=$1
  echo $VER_SETTING
fi
 
mkdir build
pushd build
  cmake $VER_SETTING ..
  ret=$?
  if [[ "$ret" != "0" ]]; then
    rm -rf ./*
    cmake $VER_SETTING ..
  fi
  make -j $(nproc)
  ret=$?
popd

exit $ret
