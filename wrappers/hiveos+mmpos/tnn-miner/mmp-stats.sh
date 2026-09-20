#!/usr/bin/env bash
# DEVICE_COUNT and LOG_FILE are agent arguments. Device selection comes from
# the miner; do not pad/truncate its arrays to the agent's inventory count.
set -euo pipefail

if ! stats_json=$(curl --fail --silent --show-error --connect-timeout 1 --max-time 3 \
    --header 'Accept: application/json' 'http://127.0.0.1:8989/mmpos'); then
    echo 'Miner API connection failed' >&2
    exit 1
fi

if ! jq -e '
    . as $s | (.hash | length) as $count |
    type == "object" and
    (.busid | type == "array") and (.hash | type == "array") and
    ((.busid | length) == $count) and
    all(.busid[]; . == "cpu" or type == "number") and
    all(.hash[]; type == "number" and . >= 0) and
    (.units == "hs") and
    (.air | type == "array" and length == 3 and all(.[]; type == "number" and . >= 0)) and
    (.miner_name | type == "string") and (.miner_version | type == "string") and
    all($s.shares.accepted, $s.shares.rejected, $s.shares.invalid;
        type == "array" and length == $count and all(.[]; type == "number" and . >= 0))
' <<< "$stats_json" >/dev/null 2>&1; then
    echo 'Miner API returned invalid mmpOS statistics' >&2
    exit 1
fi
printf '%s\n' "$stats_json"
