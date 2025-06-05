#!/bin/bash

tgt=amd64
#PACKAGE_VERSION=0.0.0
#CMAKE_ARGS="-DWITH_HIP=OFF"
valid_targets=("amd64" "arm64" "amd" "nvidia")

if [[ "$1" != "" ]]; then
  tgt=$1
fi
chk="\<${tgt}\>"
if [[ "${valid_targets[*]}" =~ $chk ]]; then
  echo "Targetting ${tgt}"
else
  echo "Invalid target specified: ${tgt}"
  exit 1
fi

if [[ "$2" != "" ]]; then
  PACKAGE_VERSION=$2
elif [[ "$PACKAGE_VERSION" == "" ]]; then
  PACKAGE_VERSION=0.0.0
fi

# Use what is provided *or* default if CMAKE_ARGS is not set as an ENV var
if [[ "$3" != "" ]]; then
  CMAKE_ARGS="$3"
elif [[ "$CMAKE_ARGS" == "" ]]; then
  CMAKE_ARGS="-DWITH_HIP=OFF"
fi
if [[ "$TARGZ_FILE" == "" ]]; then
  TARGZ_FILE="miner-$PACKAGE_VERSION-$tgt.tgz"
fi

if [[ "$tgt" == "amd64" || "$tgt" == "arm64" ]]; then
  docker buildx build --platform=linux/${tgt} --build-arg BUILDER_BASE=${tgt} --build-arg PACKAGE_VERSION=${PACKAGE_VERSION} --build-arg TARGZ_FILE=${TARGZ_FILE} --build-arg CMAKE_ARGS=${CMAKE_ARGS} -f docker/Dockerfile.ubu . --output ./
elif [[ "$tgt" == "amd" ]]; then
  docker buildx build --platform=linux/amd64 --build-arg BUILDER_BASE=rocm --build-arg PACKAGE_VERSION=${PACKAGE_VERSION} --build-arg CMAKE_ARGS="-DWITH_HIP=ON -DHIP_PLATFORM=amd -DCMAKE_HIP_PLATFORM=amd" -f docker/Dockerfile.ubu . --output ./
elif [[ "$tgt" == "nvidia" ]]; then
  docker buildx build --platform=linux/amd64 --build-arg BUILDER_BASE=rocm --build-arg PACKAGE_VERSION=${PACKAGE_VERSION} --build-arg CMAKE_ARGS="-DWITH_HIP=ON -DHIP_PLATFORM=nvidia -DCMAKE_HIP_PLATFORM=nvidia" -f docker/Dockerfile.ubu . --output ./
fi
