#!/bin/bash

ret=0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

mkdir build
pushd build
  cmake ..
  ret=$?
  if [[ "$ret" != "0" ]]; then
    rm -rf ./*
    cmake ..
  fi
  make -j $(nproc)
  ret=$?
popd

exit $ret
