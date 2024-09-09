#!/bin/bash

docker buildx build --push --tag dirkerdero/tnn-miner-base:latest --platform=linux/arm64,linux/amd64 -f Dockerfile.base . --builder container-builder
