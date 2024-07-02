#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

SUDO=
me=$(whoami)
if [[ "$me" != "root" ]]; then
  SUDO=sudo
fi
if [[ -f /etc/os-release ]]; then
  source /etc/os-release
  $SUDO apt update
  $SUDO apt install -y git wget build-essential cmake clang libssl-dev libudns-dev libc++-dev lld libsodium-dev
  if [[ "$VERSION_CODENAME" == "noble" ]]; then
    $SUDO apt install -y libboost1.83-all-dev
  #elif [[ "$VERSION_CODENAME" == "bookworm" ]]; then
  #  $SUDO apt install -y libboost1.81-all-dev
  fi
fi
