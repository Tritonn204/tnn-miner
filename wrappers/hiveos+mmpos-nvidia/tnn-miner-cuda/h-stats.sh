#!/usr/bin/env bash
# shellcheck source=/dev/null
source /hive/miners/custom/tnn-miner-cuda/h-manifest.sh

uptime=$(get_miner_uptime)
# shellcheck disable=SC2154 # log_name and log_head_name are set by HiveOS
[[ $uptime -lt 60 ]] && head -n 50 "$log_name" > "$log_head_name"

DATA=$(curl -s http://localhost:8991/stats)
[[ -z "$DATA" ]] && echo "No stats from miner API" && return

gpu_count=$(jq '.gpus | length // 0' <<< "$DATA")
hs=()
bus_numbers=()

if [[ "$gpu_count" -gt 0 ]]; then
  for (( i=0; i < gpu_count; i++ )); do
    gpu_hr=$(jq ".gpus[$i].hashrate // 0" <<< "$DATA")
    hs+=("$gpu_hr")

    # Parse PCI bus ID (e.g. "0000:0a:00.0") to decimal bus number
    pcie_id=$(jq -r ".gpus[$i].pcie_id // \"\"" <<< "$DATA")
    if [[ -n "$pcie_id" ]]; then
      bus_hex=$(echo "$pcie_id" | sed -n 's/.*:\([0-9a-fA-F]\{2\}\):.*/\1/p')
      if [[ -n "$bus_hex" ]]; then
        bus_numbers+=("$((16#$bus_hex))")
      else
        bus_numbers+=("$i")
      fi
    else
      bus_numbers+=("$i")
    fi
  done
else
  total_hr=$(jq '.hashrate // 0' <<< "$DATA")
  hs+=("$total_hr")
fi

# Total hashrate in khs for the $khs variable HiveOS expects
total_hs=0
for h in "${hs[@]}"; do
  total_hs=$(echo "$total_hs + $h" | bc)
done
khs=$(echo "scale=2; $total_hs / 1000" | bc)

ac=$(jq '.accepted // 0' <<< "$DATA")
rj=$(jq '.rejected // 0' <<< "$DATA")
uptime=$(jq '.uptime // 0' <<< "$DATA")
ver=$(jq -r '.version // "unknown"' <<< "$DATA")
algo=$(jq -r '.algo // "UNKNOWN"' <<< "$DATA")

# GPU temps and fans from HiveOS gpu-stats
temp='[]'
fan='[]'
if [[ -f /run/hive/gpu-stats.json ]]; then
  temp=$(jq -c '[.temp[]? // 0]' /run/hive/gpu-stats.json 2>/dev/null || echo '[]')
  fan=$(jq -c '[.fan[]? // 0]' /run/hive/gpu-stats.json 2>/dev/null || echo '[]')
fi

stats=$(jq -nc \
        --argjson khs "$khs" \
        --arg hs_units "hs" \
        --argjson hs "$(printf '%s\n' "${hs[@]}" | jq -cs '.')" \
        --argjson temp "$temp" \
        --argjson fan "$fan" \
        --arg uptime "$uptime" \
        --arg ver "$ver" \
        --argjson ac "$ac" --argjson rj "$rj" \
        --arg algo "$algo" \
        --argjson bus_numbers "$(printf '%s\n' "${bus_numbers[@]}" | jq -cs '.')" \
        '{$khs, $hs_units, $hs, $temp, $fan, $uptime, $ver, ar: [$ac, $rj], $algo, $bus_numbers}')

echo "khs: $khs"
echo "stats: $stats"
echo ----------
