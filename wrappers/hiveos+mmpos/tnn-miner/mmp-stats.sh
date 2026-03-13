#!/usr/bin/env bash
DEVICE_COUNT=$1
LOG_FILE=$2

MINER_API_PORT=8989
stats_json=$(curl --silent --insecure --header 'Accept: application/json' http://127.0.0.1:${MINER_API_PORT}/mmpos)

if [[ $? -ne 0 || -z $stats_json ]]; then
    echo -e "Miner API connection failed"
else
    echo $stats_json
fi
