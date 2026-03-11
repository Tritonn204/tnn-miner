#!/usr/bin/env bash
GPU_COUNT=$1
LOG_FILE=$2
cd `dirname $0`
[ -r mmp-external.conf ] && . mmp-external.conf

get_miner_stats() {
    DATA=$(curl -s http://localhost:8990/stats)
    stats=

    # Per-GPU hashrates
    local gpu_count=$(jq '.gpus | length // 0' <<< "$DATA")
    local hash_arr=()
    local busid_arr=()

    if [[ "$gpu_count" -gt 0 ]]; then
        for (( i=0; i < gpu_count; i++ )); do
            local gpu_hr=$(jq ".gpus[$i].hashrate // 0" <<< "$DATA")
            local gpu_khs=$(echo "scale=2; $gpu_hr / 1000" | bc)
            hash_arr+=($gpu_khs)
            local bid=$(jq -r ".gpus[$i].pcie_id // \"gpu$i\"" <<< "$DATA")
            busid_arr+=("$bid")
        done
    else
        local hash=$(jq '.hashrate // 0' <<< "$DATA")
        local hash=$(echo "scale=2; $hash / 1000" | bc)
        hash_arr+=($hash)
        busid_arr+=("gpu0")
    fi

    # A/R shares by pool
    local acc=$(jq '.accepted // 0' <<< "$DATA")
    local rej=$(jq '.rejected // 0' <<< "$DATA")

    stats=$(jq -nc \
            --argjson hash "$(printf '%s\n' "${hash_arr[@]}" | jq -cs '.')" \
            --argjson busid "$(printf '%s\n' "${busid_arr[@]}" | jq -Rcs 'split("\n") | map(select(. != ""))')" \
            --arg units "khs" \
            --arg ac "$acc" --arg inv "0" --arg rj "$rej" \
            --arg miner_version "$EXTERNAL_VERSION" \
            --arg miner_name "$EXTERNAL_NAME" \
        '{$busid, $hash, $units, air: [$ac, $inv, $rj], miner_name: $miner_name, miner_version: $miner_version}')
    echo $stats
}
get_miner_stats
