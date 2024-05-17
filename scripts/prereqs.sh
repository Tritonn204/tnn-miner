#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

if [[ -f /etc/lsb-release ]]; then
  source /etc/lsb-release
  sudo apt install git wget build-essential cmake clang libssl-dev libudns-dev libfmt-dev libc++-dev lld
  if [[ "$DISTRIB_CODENAME" == "noble" ]]; then
    sudo apt install libboost1.83-all-dev
  fi
fi
