#!/bin/bash

do_push="--load "
tag_base=
if [[ "$1" != "" ]]; then
  tag_base=$1
  do_push="--push "
fi

#echo docker buildx build ${do_push} --target ubuntu24-base --tag ${tag_base}tnn-miner-base:latest --platform=linux/arm64,linux/amd64 -f Dockerfile.base . --builder container-builder
docker buildx build ${do_push} --target ubuntu24-base --tag ${tag_base}tnn-builder-amd64:latest --platform=linux/amd64 -f Dockerfile.base . --builder container-builder

docker buildx build ${do_push} --target ubuntu24-base --tag ${tag_base}tnn-builder-arm64:latest --platform=linux/arm64 -f Dockerfile.base . --builder container-builder

docker buildx build ${do_push} --target ubuntu24-rocm --tag ${tag_base}tnn-builder-rocm:latest --platform=linux/amd64 -f Dockerfile.base . --builder container-builder
