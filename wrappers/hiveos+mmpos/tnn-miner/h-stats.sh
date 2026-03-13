#!/usr/bin/env bash
source /hive/miners/custom/tnn-miner/h-manifest.sh

uptime=$(get_miner_uptime)
[[ $uptime -lt 60 ]] && head -n 50 $log_name > $log_head_name

cpu_temp=$(cpu-temp)
[[ -z "$cpu_temp" ]] && cpu_temp=0

DATA=$(curl -s http://localhost:8989/stats)
[[ -z "$DATA" ]] && echo "No stats from miner API" && return

total_hr=$(jq '.hashrate // 0' <<< "$DATA")
khs=$(echo "scale=2; $total_hr / 1000" | bc)
ac=$(jq '.accepted // 0' <<< "$DATA")
rj=$(jq '.rejected // 0' <<< "$DATA")
uptime=$(jq '.uptime // 0' <<< "$DATA")
ver=$(jq -r '.version // "unknown"' <<< "$DATA")
algo=$(jq -r '.algo // "UNKNOWN"' <<< "$DATA")

stats=$(jq -nc \
        --argjson khs "$khs" \
        --arg hs_units "hs" \
        --argjson hs "[$total_hr]" \
        --argjson temp "[$cpu_temp]" \
        --argjson fan "[0]" \
        --arg uptime "$uptime" \
        --arg ver "$ver" \
        --argjson ac "$ac" --argjson rj "$rj" \
        --arg algo "$algo" \
        '{$khs, $hs_units, $hs, $temp, $fan, $uptime, $ver, ar: [$ac, $rj], $algo}')

echo khs: $khs
echo stats: $stats
echo ----------
