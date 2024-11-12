#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

declare -a image_versions=()

if [[ -d ${ROOT_DIR}/build ]]; then
  rm -rf ${ROOT_DIR}/build
fi

if [[ "$1" == "smoke" ]]; then
  #image_version=("ubuntu:24.04" "debian:12")
  image_version=("ubuntu:24.04")
else
  if [[ "$1" == "ubu" || "$1" == "" ]]; then
    declare -a ubuntu_vers=("24.04" "22.04" "rolling")
    for ver in "${ubuntu_vers[@]}"
    do
      image_version+=("ubuntu:${ver}")
    done
  fi
  if [[ "$1" == "deb" || "$1" == "" ]]; then
    declare -a debian_vers=("12" "testing") #"11" - unsupported cpu arch
    for ver in "${debian_vers[@]}"
    do
      image_version+=("debian:${ver}")
    done
  fi
  if [[ "$1" =~ ":" ]]; then
    image_version+=("$1")
  fi
fi

if [[ ! -d ${ROOT_DIR}/_deps ]]; then
  mkdir ${ROOT_DIR}/_deps
fi
for ver in "${image_version[@]}"
do
  echo Building for $ver
  docker pull $ver
  #docker buildx build --platform=linux/amd64,linux/arm64 --build-arg IMAGE_VERSION=$ver -f ${ROOT_DIR}/docker/Dockerfile.ubu .
  targz_filename_orig=regress-$ver.tgz
  targz_filename=${targz_filename_orig//:/-}  # Replace : with -

  docker buildx build --platform=linux/amd64 --build-arg IMAGE_VERSION=$ver --build-arg TARGZ_FILE=$targz_filename --build-arg CMAKE_ARGS=-DWITH_HIP=OFF --build-arg REGRESS=1 -f ${ROOT_DIR}/docker/Dockerfile.ubu . --output ./
  if [[ "$?" == "0" ]]; then
    echo Success!
  else
    echo "Failed to build for container ${ver}"
  fi
  #docker run --env IMAGE_VERSION=$ver --mount type=bind,source="${pwd}/.cache/docker",target=/src/.cache/ -f ${ROOT_DIR}/docker/Dockerfile.ubu .
done

