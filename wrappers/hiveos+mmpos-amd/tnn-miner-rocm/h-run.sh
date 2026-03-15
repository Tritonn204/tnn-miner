#!/usr/bin/env bash

# shellcheck source=h-manifest.conf
source h-manifest.conf

CUSTOM_LOG_BASEDIR=$(dirname "$CUSTOM_LOG_BASENAME")
[[ ! -d "$CUSTOM_LOG_BASEDIR" ]] && mkdir -p "$CUSTOM_LOG_BASEDIR"

if [[ -z $CUSTOM_CONFIG_FILENAME ]]; then
	echo -e "The config file is not defined"
fi

MINER_DIR="/hive/miners/custom/$MINER_NAME"
export LD_LIBRARY_PATH="$MINER_DIR/hip_libs:$LD_LIBRARY_PATH:/hive/lib"

# Persist HIPRTC compiled kernel cache next to the miner
export TNN_HIP_CACHE_PATH="$MINER_DIR/.hip_cache"
mkdir -p "$TNN_HIP_CACHE_PATH"

CUSTOM_USER_CONFIG=$(< "$CUSTOM_CONFIG_FILENAME")

echo "args: $CUSTOM_USER_CONFIG"

MINER=$MINER_NAME

# Remove the -arch argument and its value
CLEAN=$(echo "$CUSTOM_USER_CONFIG" | sed -E 's/-arch [^ ]+ //')
echo "args are now: $CLEAN"
echo "We are using miner: $MINER"
date +%s > "/tmp/miner_start_time"
/hive/miners/custom/"$MINER"/"$MINER" -v 2>&1 | grep 'Miner version:' | awk '{print $3}' > /tmp/.tnn-miner-version
# shellcheck disable=SC2086 # Intentional word splitting: CLEAN contains multiple args
/hive/miners/custom/"$MINER"/"$MINER" $CLEAN --broadcast 2>&1 | tee -a "${CUSTOM_LOG_BASENAME}.log"
echo "Miner has exited"
