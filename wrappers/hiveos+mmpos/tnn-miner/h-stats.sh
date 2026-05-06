#!/usr/bin/env bash
# shellcheck disable=SC1091 # h-manifest.conf is provided at runtime by HiveOS
source /hive/miners/custom/tnn-miner/h-manifest.sh

uptime=$(get_miner_uptime)
# shellcheck disable=SC2154 # log_name and log_head_name are set by HiveOS
[[ $uptime -lt 60 ]] && head -n 50 "$log_name" > "$log_head_name"

DATA=$(curl -s http://localhost:8989/stats)
[[ -z "$DATA" ]] && echo "No stats from miner API" && return

gpu_count=$(jq '.gpus | length // 0' <<< "$DATA")
cpu_hr=$(jq '.cpu_hashrate // 0' <<< "$DATA")
has_cpu=$(echo "$cpu_hr > 0.001" | bc -l 2>/dev/null || echo 0)

hs=()
bus_numbers=()
temps=()
fans=()

# CPU entry first (bus_number=null, fan=null)
if [[ "$has_cpu" -eq 1 ]]; then
  hs+=("$cpu_hr")
  bus_numbers+=("null")
  cpu_temp=$(cpu-temp 2>/dev/null)
  [[ -z "$cpu_temp" ]] && cpu_temp=0
  temps+=("$cpu_temp")
  fans+=("null")
fi

# GPU entries
if [[ "$gpu_count" -gt 0 ]]; then
  # GPU temps and fans from HiveOS gpu-stats
  gpu_temps='[]'
  gpu_fans='[]'
  if [[ -f /run/hive/gpu-stats.json ]]; then
    gpu_temps=$(jq -c '[.temp[]? // 0]' /run/hive/gpu-stats.json 2>/dev/null || echo '[]')
    gpu_fans=$(jq -c '[.fan[]? // 0]' /run/hive/gpu-stats.json 2>/dev/null || echo '[]')
  fi

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

    # Per-GPU temp and fan from system gpu-stats (indexed by position)
    t=$(echo "$gpu_temps" | jq ".[$i] // 0" 2>/dev/null || echo 0)
    f=$(echo "$gpu_fans" | jq ".[$i] // 0" 2>/dev/null || echo 0)
    temps+=("$t")
    fans+=("$f")
  done
fi

# Fallback: no GPU and no CPU hashrate — use aggregate
if [[ "${#hs[@]}" -eq 0 ]]; then
  total_hr=$(jq '.hashrate // 0' <<< "$DATA")
  hs+=("$total_hr")
  bus_numbers+=("null")
  cpu_temp=$(cpu-temp 2>/dev/null)
  [[ -z "$cpu_temp" ]] && cpu_temp=0
  temps+=("$cpu_temp")
  fans+=("null")
fi

# Total hashrate in khs
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

# Build JSON arrays — null values need special handling
hs_json=$(printf '%s\n' "${hs[@]}" | jq -cs '.')
temp_json=$(printf '%s\n' "${temps[@]}" | jq -cs '.')
fan_json=$(printf '%s\n' "${fans[@]}" | jq -cs '.')
bus_json=$(printf '%s\n' "${bus_numbers[@]}" | jq -cs '.')

stats=$(jq -nc \
        --argjson khs "$khs" \
        --arg hs_units "hs" \
        --argjson hs "$hs_json" \
        --argjson temp "$temp_json" \
        --argjson fan "$fan_json" \
        --arg uptime "$uptime" \
        --arg ver "$ver" \
        --argjson ac "$ac" --argjson rj "$rj" \
        --arg algo "$algo" \
        --argjson bus_numbers "$bus_json" \
        '{$khs, $hs_units, $hs, $temp, $fan, $uptime, $ver, ar: [$ac, $rj], $algo, $bus_numbers}')

echo "khs: $khs"
echo "stats: $stats"
echo ----------
