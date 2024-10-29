#!/bin/bash

docker buildx build --push --target ubuntu24-base --tag dirkerdero/tnn-miner-base:latest --platform=linux/arm64,linux/amd64 -f Dockerfile.base . --builder container-builder

docker buildx build --push --target ubuntu24-rocm --tag dirkerdero/tnn-miner-base:rocm-62 --platform=linux/amd64 -f Dockerfile.base . --builder container-builder
#docker buildx build --load --target ubuntu24-rocm --tag dirkerdero/tnn-miner-base:rocm-62 --platform=linux/amd64 -f Dockerfile.base . --builder container-builder
